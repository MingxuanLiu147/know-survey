"""Shell-free local process manager with persisted traces and streamable events."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalog import ROOT, display_command


FINAL_STATES = {"succeeded", "failed", "stopped", "interrupted"}
MODE_DIR = {"unlearning": "unlearn", "inject": "inject", "edit": "edit"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobError(RuntimeError):
    """Raised for invalid or conflicting local job operations."""


class JobManager:
    """Manage local surgery processes without exposing arbitrary shell access."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.state_root = self.root / "artifacts" / "webui" / "runs"
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        for path in self.state_root.glob("*/run.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("status") in {"queued", "running", "stopping"}:
                record["status"] = "interrupted"
                record["finished_at"] = _utc()
                record["error"] = "Backend restarted while the process was active."
                self._write_record(record)
            run_id = str(record.get("id") or path.parent.name)
            self._jobs[run_id] = record
            self._events[run_id] = self._load_events(path.parent)

    @staticmethod
    def _load_events(run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "events.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows[-10_000:]

    def _run_dir(self, run_id: str) -> Path:
        return self.state_root / run_id

    def _write_record(self, record: dict[str, Any]) -> None:
        run_dir = self._run_dir(record["id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / "run.json"
        temporary = run_dir / "run.json.tmp"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def _emit(
        self,
        run_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            events = self._events.setdefault(run_id, [])
            event = {
                "index": len(events),
                "timestamp": _utc(),
                "type": event_type,
                **payload,
            }
            events.append(event)
            run_dir = self._run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            return event

    def _gpu_busy(self, gpu: int | None) -> bool:
        if gpu is None:
            return False
        return any(
            job.get("gpu") == gpu and job.get("status") not in FINAL_STATES
            for job in self._jobs.values()
        )

    def start(
        self,
        *,
        argv: list[str],
        operation: str,
        task_name: str,
        gpu: int | None,
        compose: dict[str, Any],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            if self._gpu_busy(gpu):
                raise JobError(f"GPU {gpu} already has an active WebUI job.")
            run_id = uuid.uuid4().hex[:16]
            output_dir = self.root / "saves" / MODE_DIR[operation] / task_name
            record = {
                "id": run_id,
                "status": "queued",
                "operation": operation,
                "task_name": task_name,
                "gpu": gpu,
                "created_at": _utc(),
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "error": None,
                "argv": argv,
                "command": display_command(argv),
                "output_dir": output_dir.relative_to(self.root).as_posix(),
                "compose": compose,
                "request": request,
                "evidence_level": "configured",
            }
            self._jobs[run_id] = record
            self._events[run_id] = []
            self._write_record(record)
            self._emit(
                run_id,
                "status",
                {"status": "queued", "message": "Command accepted."},
            )
            thread = threading.Thread(
                target=self._execute,
                args=(run_id,),
                daemon=True,
                name=f"know-surgery-{run_id}",
            )
            thread.start()
            return dict(record)

    def _execute(self, run_id: str) -> None:
        with self._lock:
            record = self._jobs[run_id]
            record["status"] = "running"
            record["started_at"] = _utc()
            self._write_record(record)
        self._emit(
            run_id,
            "status",
            {"status": "running", "message": "Local process started."},
        )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        if record["gpu"] is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(record["gpu"])
        try:
            process = subprocess.Popen(
                record["argv"],
                cwd=self.root,
                env=environment,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                self._processes[run_id] = process
            assert process.stdout is not None
            for line in process.stdout:
                self._emit(
                    run_id,
                    "log",
                    {"stream": "combined", "message": line.rstrip("\n")},
                )
            exit_code = process.wait()
            with self._lock:
                record = self._jobs[run_id]
                if record["status"] == "stopping":
                    record["status"] = "stopped"
                else:
                    record["status"] = "succeeded" if exit_code == 0 else "failed"
                record["exit_code"] = exit_code
                record["finished_at"] = _utc()
                record["evidence_level"] = self._evidence(record)
                self._write_record(record)
            self._emit(
                run_id,
                "status",
                {
                    "status": record["status"],
                    "exit_code": exit_code,
                    "evidence_level": record["evidence_level"],
                },
            )
        except Exception as exc:
            with self._lock:
                record = self._jobs[run_id]
                record["status"] = "failed"
                record["finished_at"] = _utc()
                record["error"] = str(exc)
                self._write_record(record)
            self._emit(
                run_id,
                "status",
                {"status": "failed", "message": str(exc)},
            )
        finally:
            with self._lock:
                self._processes.pop(run_id, None)

    def _evidence(self, record: dict[str, Any]) -> str:
        output = self.root / record["output_dir"]
        if record.get("status") != "succeeded" or not output.is_dir():
            return "configured"
        required = (
            output / "command.txt",
            output / "resolved_config.yaml",
            output / "input_manifest.json",
            output / "run_trace.json",
        )
        has_metrics = (
            bool(list(output.glob("*_SUMMARY.json")))
            or (output / "unlearning_metrics.json").is_file()
        )
        if all(path.is_file() for path in required):
            if record.get("request", {}).get("smoke", True):
                return "verified_smoke"
            if has_metrics:
                return "local_validation"
        return "local_validation"

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            process = self._processes.get(run_id)
            if process is None or record["status"] not in {"running", "queued"}:
                raise JobError("Run is not currently stoppable.")
            record["status"] = "stopping"
            self._write_record(record)
        self._emit(
            run_id,
            "status",
            {"status": "stopping", "message": "Interrupt requested."},
        )
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        threading.Thread(
            target=self._force_stop,
            args=(run_id, process),
            daemon=True,
        ).start()
        return self.get(run_id)

    def _force_stop(
        self,
        run_id: str,
        process: subprocess.Popen[str],
    ) -> None:
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                self._emit(
                    run_id,
                    "status",
                    {
                        "status": "stopping",
                        "message": "Grace period expired; process killed.",
                    },
                )
            except ProcessLookupError:
                pass

    def get(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            self._refresh_evidence(record)
            return {**record, "event_count": len(self._events.get(run_id, []))}

    def _refresh_evidence(self, record: dict[str, Any]) -> None:
        if record.get("status") != "succeeded":
            return
        evidence = self._evidence(record)
        if evidence != record.get("evidence_level"):
            record["evidence_level"] = evidence
            self._write_record(record)

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            values = sorted(
                self._jobs.values(),
                key=lambda item: item.get("created_at") or "",
                reverse=True,
            )
            for record in values[:limit]:
                self._refresh_evidence(record)
            return [
                {**item, "event_count": len(self._events.get(item["id"], []))}
                for item in values[:limit]
            ]

    def events(
        self,
        run_id: str,
        *,
        after: int = -1,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if run_id not in self._jobs:
                raise KeyError(run_id)
            return [
                dict(event)
                for event in self._events.get(run_id, [])
                if int(event["index"]) > after
            ]

    def is_final(self, run_id: str) -> bool:
        return self.get(run_id)["status"] in FINAL_STATES
