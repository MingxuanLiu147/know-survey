"""Know-Surgery FastAPI application and production frontend host."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import yaml
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .catalog import (
    ROOT,
    CatalogError,
    RepositoryCatalog,
    display_command,
)
from .jobs import FINAL_STATES, JobError, JobManager
from .execution import composition_identity
from .planner import PlannerError, create_plan, provider_status
from .results import ResultStore
from .schemas import (
    AgentRequest,
    CompareRequest,
    ComposeRequest,
    PlanRequest,
    RegisterUploadRequest,
    RunRequest,
)
from .uploads import UploadError, UploadStore


API_PREFIX = "/api/v1"
catalog = RepositoryCatalog(ROOT)
jobs = JobManager(ROOT)
results = ResultStore(ROOT)
uploads = UploadStore(ROOT)
app = FastAPI(
    title="Know-Surgery Workbench API",
    version="0.2.0",
    description="Auditable local orchestration for model knowledge surgery.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:7860",
        "http://localhost:7860",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent_sessions: dict[str, Any] = {}
_agent_tasks: dict[str, asyncio.Task] = {}


@app.post(f"{API_PREFIX}/agent/sessions")
async def start_agent(request: AgentRequest) -> dict[str, Any]:
    from know_surgery.agent import AgentSession

    available = {row["id"]: row["configured"] for row in provider_status()}
    if not available.get(request.provider):
        raise HTTPException(
            422, "Provider is not configured on the server. / 服务端尚未配置模型服务。"
        )
    if any(not task.done() for task in _agent_tasks.values()):
        raise HTTPException(
            409, "An autonomous session is already running. / 已有自主任务正在运行。"
        )
    configuration = request.configuration
    composed = None
    if configuration is not None:
        composed = await asyncio.to_thread(
            _checked_compose, configuration, request.preview_hash
        )
        if any(not task.done() for task in _agent_tasks.values()):
            raise HTTPException(
                409,
                "An autonomous session is already running. / 已有自主任务正在运行。",
            )

    def execute_configuration(session):
        # Parameters are captured from the caller's configuration, never from
        # a model-generated tool argument or a newly inferred plan.
        run_request = RunRequest.model_validate(
            {
                **configuration.model_dump(),
                "request_id": f"agent-{session.id}",
                "preview_hash": composed["preview_hash"],
            }
        )
        if session.cancelled.is_set() or time.monotonic() >= session.deadline:
            raise TimeoutError("Session cancelled or time budget exhausted.")
        record = _start_configuration(
            run_request,
            abort=lambda: session.cancelled.is_set()
            or time.monotonic() >= session.deadline,
        )
        session.record_configuration_run(record)
        stopping = False
        while record["status"] not in FINAL_STATES:
            if not stopping and (
                session.cancelled.is_set() or time.monotonic() >= session.deadline
            ):
                try:
                    jobs.stop(record["id"])
                except JobError:
                    pass  # The process may have finished between the two checks.
                stopping = True
            time.sleep(0.2)
            record = jobs.get(record["id"])
        session.record_configuration_run(record)
        return {
            "status": "passed" if record["status"] == "succeeded" else "failed",
            "returncode": record["exit_code"],
            "run_id": record["id"],
            "result_id": record["result_id"],
            "command": record["command"],
            "observation": record,
        }

    session = AgentSession(
        ROOT,
        provider=request.provider,
        modality=request.modality,
        allow_execute=request.execute,
        cuda=request.cuda,
        max_steps=request.max_steps,
        max_cases=request.max_cases,
        timeout=request.timeout,
        configuration=configuration.model_dump() if configuration is not None else None,
        configuration_preview=composed,
        configuration_runner=execute_configuration
        if configuration is not None
        else None,
    )
    _agent_sessions[session.id] = session
    task = asyncio.create_task(asyncio.to_thread(session.run, request.intent))
    _agent_tasks[session.id] = task

    def finish(_):
        _agent_tasks.pop(session.id, None)
        _agent_sessions.pop(session.id, None)

    task.add_done_callback(finish)
    return {"session_id": session.id, "status": "pending"}


@app.get(f"{API_PREFIX}/agent/sessions/{{session_id}}")
async def agent_status(session_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise HTTPException(404, "Unknown session. / 未找到任务。")
    path = ROOT / "artifacts/agent" / session_id / "trace.json"
    if not path.is_file():
        raise HTTPException(404, "Unknown session. / 未找到任务。")
    trace = json.loads(path.read_text())
    if trace["status"] in {"pending", "running"} and session_id not in _agent_sessions:
        trace["status"] = "interrupted"
        trace["error"] = (
            "Server restarted; inspect artifacts before retrying. / 服务已重启，请先检查运行产物。"
        )
    return trace


@app.post(f"{API_PREFIX}/agent/sessions/{{session_id}}/stop")
async def stop_agent(session_id: str) -> dict[str, Any]:
    session = _agent_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "No active session. / 没有运行中的任务。")
    session.cancelled.set()
    return {"session_id": session_id, "status": "cancelling"}


_gpu_cache: tuple[float, list[dict[str, Any]], str | None] = (0.0, [], None)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpus() -> tuple[list[dict[str, Any]], str | None]:
    global _gpu_cache
    now = time.monotonic()
    if now - _gpu_cache[0] < 3:
        return _gpu_cache[1], _gpu_cache[2]
    argv = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _gpu_cache = (now, [], str(exc))
        return [], str(exc)
    if process.returncode != 0:
        error = process.stderr.strip() or "nvidia-smi failed"
        _gpu_cache = (now, [], error)
        return [], error
    rows = []
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        index, name, total, used, free, utilization = parts
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
                "available": int(free) >= 8192 and int(utilization) < 20,
            }
        )
    _gpu_cache = (now, rows, None)
    return rows, None


def _git_revision() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def _skills() -> list[dict[str, Any]]:
    root = ROOT / ".claude" / "skills"
    categories = {
        "setup-env": "foundation",
        "inspect-repo": "foundation",
        "plan-surgery": "foundation",
        "add-model": "extension",
        "add-dataset": "extension",
        "normalize-knowledge": "data",
        "build-cases": "data",
        "add-multimodal-case": "multimodal",
        "run-inject": "execution",
        "run-unlearn": "execution",
        "run-edit": "execution",
        "smoke-test": "validation",
        "batch-sweep": "execution",
        "aggregate-results": "analysis",
        "compare-methods": "analysis",
        "generate-report": "reporting",
        "reproduce-run": "reproducibility",
        "debug-ui": "ui",
        "release-checklist": "release",
    }
    rows = []
    for path in sorted(root.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        if text.startswith("---"):
            _, frontmatter, _ = text.split("---", 2)
            value = yaml.safe_load(frontmatter)
            if isinstance(value, dict):
                metadata = value
        name = str(metadata.get("name") or path.parent.name)
        rows.append(
            {
                "id": name,
                "description": metadata.get("description", ""),
                "category": categories.get(name, "other"),
                "status": "configured",
                "has_runtime": (path.parent / "scripts" / "run.py").exists(),
            }
        )
    return rows


def _response_examples() -> list[dict[str, Any]]:
    path = ROOT / "demos" / "response_comparisons.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    operations = {"unlearning", "inject", "edit"}
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("operation") not in operations:
            continue
        sources = row.get("source_artifacts", [])
        if not isinstance(sources, list):
            sources = []
        result.append(
            {
                **row,
                "source_artifacts": [str(item) for item in sources],
                "sources_available": bool(sources)
                and all((ROOT / str(item)).is_file() for item in sources),
            }
        )
    return result


def _compose(request: ComposeRequest) -> dict[str, Any]:
    try:
        argv = catalog.build_argv(request, preview=True)
        command = catalog.command_preview(request)
    except CatalogError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        process = subprocess.run(
            argv,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="Hydra composition timed out.",
        ) from exc
    result = {
        "valid": process.returncode == 0,
        "returncode": process.returncode,
        "command": command,
        "preview_command": display_command(argv),
        "resolved_config": process.stdout[-100_000:],
        "stderr": process.stderr[-30_000:],
        "request": request.model_dump(),
    }
    if result["valid"]:
        try:
            result.update(composition_identity(result["resolved_config"], ROOT))
        except (OSError, ValueError, TypeError) as exc:
            raise HTTPException(
                422, f"Cannot inspect the composed configuration: {exc}"
            ) from exc
    return result


def _checked_compose(
    request: ComposeRequest, preview_hash: str | None = None
) -> dict[str, Any]:
    result = _compose(request)
    if not result["valid"]:
        raise HTTPException(
            409,
            {
                "message": "Hydra composition failed; run was not started.",
                "stderr": result["stderr"],
            },
        )
    if preview_hash is not None and result["preview_hash"] != preview_hash:
        raise HTTPException(
            409,
            "preview_stale: configuration or checked local inputs changed; preview again.",
        )
    return result


def _start_configuration(request: RunRequest, *, abort=None) -> dict[str, Any]:
    """Start GUI and configured Agent jobs through the same validated path."""
    payload = request.model_dump()
    try:
        existing = jobs.existing_request(payload)
        if existing is not None:
            return existing
        compose_request = ComposeRequest.model_validate(
            request.model_dump(exclude={"gpu", "request_id", "preview_hash"})
        )
        composed = _checked_compose(compose_request, request.preview_hash)
        if abort is not None and abort():
            raise TimeoutError("Session cancelled or time budget exhausted.")
        return jobs.start(
            argv=catalog.build_argv(compose_request),
            operation=request.operation,
            task_name=request.task_name,
            gpu=request.gpu,
            compose=composed,
            request=payload,
        )
    except JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, Any]:
    gpu_rows, gpu_error = _gpus()
    frontend_dist = ROOT / "webui" / "frontend" / "dist"
    return {
        "status": "ready",
        "service": "Know-Surgery Workbench",
        "api_version": app.version,
        "time": _utc(),
        "python": sys.version.split()[0],
        "revision": _git_revision(),
        "gpus": gpu_rows,
        "gpu_error": gpu_error,
        "skills": len(_skills()),
        "frontend_built": (frontend_dist / "index.html").is_file(),
        "operations": ["unlearning", "inject", "edit"],
    }


@app.get(f"{API_PREFIX}/catalog")
def get_catalog() -> dict[str, Any]:
    return catalog.snapshot()


@app.get(f"{API_PREFIX}/skills")
def get_skills() -> dict[str, Any]:
    return {"skills": _skills(), "count": len(_skills())}


@app.get(f"{API_PREFIX}/response-examples")
def get_response_examples() -> dict[str, Any]:
    rows = _response_examples()
    return {"examples": rows, "count": len(rows)}


@app.get(f"{API_PREFIX}/providers")
def get_providers() -> dict[str, Any]:
    return {"providers": provider_status()}


@app.post(f"{API_PREFIX}/plans")
async def plan(request: PlanRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(create_plan, request, catalog)
    except PlannerError as exc:
        status = 409 if "not configured" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post(f"{API_PREFIX}/compose")
async def compose(request: ComposeRequest) -> dict[str, Any]:
    return await asyncio.to_thread(_compose, request)


@app.get(f"{API_PREFIX}/runs")
def list_runs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    return {"runs": jobs.list(limit=limit)}


@app.post(f"{API_PREFIX}/runs", status_code=202)
async def start_run(request: RunRequest) -> dict[str, Any]:
    return await asyncio.to_thread(_start_configuration, request)


@app.get(f"{API_PREFIX}/runs/{{run_id}}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return jobs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@app.get(f"{API_PREFIX}/runs/{{run_id}}/event-list")
def get_run_events(
    run_id: str,
    after: int = Query(default=-1, ge=-1),
) -> dict[str, Any]:
    try:
        return {"events": jobs.events(run_id, after=after)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


async def _event_stream(
    run_id: str,
    after: int,
) -> AsyncIterator[str]:
    cursor = after
    while True:
        try:
            events = jobs.events(run_id, after=cursor)
            record = jobs.get(run_id)
        except KeyError:
            yield 'event: error\ndata: {"message":"Run not found"}\n\n'
            return
        for event in events:
            cursor = int(event["index"])
            yield (
                f"id: {cursor}\n"
                f"event: {event['type']}\n"
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            )
        if record["status"] in FINAL_STATES and not events:
            return
        await asyncio.sleep(0.35)


@app.get(f"{API_PREFIX}/runs/{{run_id}}/events")
def stream_run_events(
    run_id: str,
    after: int = Query(default=-1, ge=-1),
) -> StreamingResponse:
    try:
        jobs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    return StreamingResponse(
        _event_stream(run_id, after),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(f"{API_PREFIX}/runs/{{run_id}}/stop")
def stop_run(run_id: str) -> dict[str, Any]:
    try:
        return jobs.stop(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except JobError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get(f"{API_PREFIX}/results")
def list_results(limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
    return {"results": results.list(limit=limit)}


@app.get(f"{API_PREFIX}/results/{{result_id}}")
def get_result(result_id: str) -> Any:
    try:
        return results.get(result_id)
    except KeyError as exc:
        run = jobs.for_result(result_id)
        if run is not None:
            if run["status"] not in FINAL_STATES:
                return JSONResponse(
                    status_code=202,
                    content={
                        "kind": "pending",
                        "result_id": result_id,
                        "run_id": run["id"],
                        "retry_after_seconds": 2,
                    },
                )
            return {
                "kind": "unavailable",
                "result_id": result_id,
                "run_id": run["id"],
                "run_status": run["status"],
                "reason": run.get("error")
                or "No result artifacts were produced. / 本次运行未产生结果文件。",
            }
        raise HTTPException(status_code=404, detail="Result not found.") from exc


@app.get(f"{API_PREFIX}/results/{{result_id}}/cases")
def get_result_cases(
    result_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    metric_id: str | None = None,
) -> dict[str, Any]:
    try:
        return results.cases(result_id, offset=offset, limit=limit, metric_id=metric_id)
    except KeyError as exc:
        raise HTTPException(404, "Result not found.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get(f"{API_PREFIX}/results/{{result_id}}/metadata")
def get_result_metadata(result_id: str) -> dict[str, Any]:
    try:
        return results.metadata(result_id)
    except KeyError as exc:
        raise HTTPException(404, "Result not found.") from exc


@app.post(f"{API_PREFIX}/results/compare")
def compare_results(request: CompareRequest) -> dict[str, Any]:
    try:
        return results.compare(request.result_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Result not found.") from exc


@app.get(f"{API_PREFIX}/demos")
def get_demos() -> dict[str, Any]:
    return {"demos": results.demos()}


@app.get(f"{API_PREFIX}/results/{{result_id}}/report")
def result_report(result_id: str) -> PlainTextResponse:
    try:
        detail = results.get(result_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Result not found.") from exc
    lines = [
        f"# Know-Surgery Report: {detail['name']}",
        "",
        f"- Operation: {detail['operation']}",
        f"- Evidence: {detail['evidence_level']}",
        f"- Artifact: {detail['path']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Direction | Source |",
        "|---|---:|---|---|",
    ]
    for metric in detail["metrics"]:
        value = "N/A" if metric["value"] is None else metric["value"]
        lines.append(
            f"| {metric['id']} | {value} | {metric['direction']} | "
            f"{metric['source']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            detail.get("command") or "Command artifact unavailable.",
            "",
            "## Limitations",
            "",
            "- The report preserves the evidence level of the source artifacts.",
            "- Missing metrics are reported as N/A and are not replaced with zero.",
        ]
    )
    return PlainTextResponse(
        "\n".join(str(item) for item in lines) + "\n",
        media_type="text/markdown",
        headers={
            "Content-Disposition": (
                f'attachment; filename="know-surgery-{result_id}.md"'
            )
        },
    )


@app.post(f"{API_PREFIX}/uploads/{{operation}}/preview")
async def preview_upload(
    operation: str,
    file: UploadFile = File(...),
    retain_file: UploadFile | None = File(default=None),
    mapping: str | None = Form(default=None),
    knowledge_type: str = Form(default="fact"),
    format_type: str = Form(default="custom"),
    modality: str = Form(default="text"),
) -> dict[str, Any]:
    if operation not in {"unlearning", "inject", "edit"}:
        raise HTTPException(
            status_code=404,
            detail="Unsupported custom upload operation.",
        )
    data = await file.read(50 * 1024 * 1024 + 1)
    retain_data = await retain_file.read(50 * 1024 * 1024 + 1) if retain_file else None
    try:
        return uploads.preview(
            data=data,
            filename=file.filename or "upload.jsonl",
            mapping=mapping,
            knowledge_type=knowledge_type,
            format_type=format_type,
            modality=modality,
            operation=operation,
            retain_data=retain_data,
            retain_filename=retain_file.filename if retain_file else None,
        )
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(f"{API_PREFIX}/uploads/{{operation}}/register")
def register_upload(
    operation: str,
    request: RegisterUploadRequest,
) -> dict[str, Any]:
    if operation not in {"unlearning", "inject", "edit"}:
        raise HTTPException(
            status_code=404,
            detail="Unsupported custom upload operation.",
        )
    try:
        return uploads.register(request, catalog, operation=operation)
    except UploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


FRONTEND_DIST = ROOT / "webui" / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=503,
            detail="Frontend is not built. Run npm run build in webui/frontend.",
        )
    return FileResponse(index)


@app.get("/{path:path}", include_in_schema=False)
def frontend_fallback(path: str) -> FileResponse:
    candidate = (FRONTEND_DIST / path).resolve()
    try:
        candidate.relative_to(FRONTEND_DIST.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not found.") from exc
    if candidate.is_file():
        return FileResponse(candidate)
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Not found.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        "webui.backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
