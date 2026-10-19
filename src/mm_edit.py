"""Multimodal knowledge editing entrypoint.

This sidecar mirrors the text edit path in ``src/train.py`` but loads
image-text models through ``model.multimodal.get_mm_model`` and attaches the
processor to MM editors.
"""

from __future__ import annotations

import json
import hashlib
import logging
import platform
import shlex
import sys
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import transformers

from data import get_data
from evals import get_evaluators
from evals.mm_edit import MMEditLocalityEvaluator
from model.multimodal import get_mm_model
from trainer import load_trainer
from trainer.edit.pipeline import (
    build_edit_requests,
    execute_edit_requests,
    save_edit_artifacts,
    to_jsonable,
)
from trainer.utils import seed_everything

logger = logging.getLogger(__name__)


_MM_CASE_METRICS = (
    "mm_reliability",
    "mm_generalization",
    "text_rephrase_acc",
    "image_rephrase_acc",
    "mm_locality",
    "text_locality_acc",
    "multimodal_locality_acc",
    "multimodal_locality_acc_top10",
    "mm_portability",
    "text_portability_acc",
    "multimodal_portability_acc",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_path(path: Path) -> str:
    """Prefer repository-relative paths in shareable manifests."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _hf_revision_from_local_dir(path: Path) -> str | None:
    metadata = path / ".cache/huggingface/download/config.json.metadata"
    if not metadata.is_file():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines and lines[0].strip() else None


def _dataset_revision(data_path: Path) -> str | None:
    """Read a consistent Hugging Face revision near a materialized data file."""
    for parent in (data_path.parent, *data_path.parents):
        cache = parent / ".official/.cache/huggingface/download"
        if not cache.is_dir():
            continue
        revisions = {
            lines[0].strip()
            for metadata in cache.glob("*.metadata")
            if (lines := metadata.read_text(encoding="utf-8").splitlines())
            and lines[0].strip()
        }
        if len(revisions) == 1:
            return revisions.pop()
    return None


def _collect_configured_data_files(value: Any) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []

    def visit(node: Any, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                visit(child, str(child_key))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child, key)
            return
        if key != "data_path" or not isinstance(node, str):
            return
        path = Path(node)
        path = path if path.is_absolute() else Path.cwd() / path
        if path.is_file():
            files.append(
                {
                    "path": _artifact_path(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "revision": _dataset_revision(path),
                }
            )

    visit(value)
    return files


def _collect_preview_images(value: Any) -> list[dict[str, Any]]:
    images: dict[str, dict[str, Any]] = {}

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if node.get("type") == "PIL.Image" and node.get("path"):
                path = Path(str(node["path"]))
                if path.is_file():
                    artifact_path = _artifact_path(path)
                    images[artifact_path] = {
                        "path": artifact_path,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                        "image_size": node.get("size"),
                        "mode": node.get("mode"),
                    }
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return [images[key] for key in sorted(images)]


def _save_standard_trace_files(output_path: Path, manifest: Mapping[str, Any]) -> None:
    """Materialize the shared Skill artifact contract from an MM run manifest."""
    command = [str(item) for item in manifest.get("command", [])]
    (output_path / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    resolved = manifest.get("resolved_config", {})
    model_cfg = resolved.get("model", {}) if isinstance(resolved, Mapping) else {}
    model_path_raw = (
        model_cfg.get("model_path") if isinstance(model_cfg, Mapping) else None
    )
    model_path = Path(str(model_path_raw)) if model_path_raw else None
    model_revision = (
        _hf_revision_from_local_dir(model_path)
        if model_path is not None and model_path.is_dir()
        else None
    )

    environment = dict(manifest.get("environment", {}))
    environment["model_revision"] = model_revision
    (output_path / "environment.json").write_text(
        json.dumps(to_jsonable(environment), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    preview_path = output_path / "edit_requests_preview.json"
    preview = (
        json.loads(preview_path.read_text(encoding="utf-8"))
        if preview_path.is_file()
        else []
    )
    data_cfg = resolved.get("data", {}) if isinstance(resolved, Mapping) else {}
    data_files = _collect_configured_data_files(data_cfg)
    dataset_ids = []
    if isinstance(data_cfg, Mapping) and isinstance(data_cfg.get("edit"), Mapping):
        dataset_ids = sorted(str(key) for key in data_cfg["edit"])
    edit_type = resolved.get("edit_type") if isinstance(resolved, Mapping) else None
    expected_cases = (
        resolved.get("benchmark_expected_cases")
        if isinstance(resolved, Mapping)
        else None
    )
    complete_benchmark = bool(
        manifest.get("edit_success")
        and edit_type == "independent"
        and isinstance(expected_cases, int)
        and expected_cases > 0
        and len(preview) == expected_cases
    )
    input_manifest = {
        "dataset_ids": dataset_ids,
        "dataset_revision": next(
            (item["revision"] for item in data_files if item.get("revision")), None
        ),
        "request_count": len(preview),
        "complete_benchmark": complete_benchmark,
        "benchmark_expected_cases": expected_cases,
        "request_preview": "edit_requests_preview.json"
        if preview_path.is_file()
        else None,
        "request_preview_sha256": _sha256_file(preview_path)
        if preview_path.is_file()
        else None,
        "data_files": data_files,
        "image_files": _collect_preview_images(preview),
    }
    (output_path / "input_manifest.json").write_text(
        json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    edit_count = int(manifest.get("edit_success", False) and len(preview))
    model_id = None
    if isinstance(model_cfg, Mapping):
        checkpoint = model_cfg.get("checkpoint_contract", {})
        if isinstance(checkpoint, Mapping):
            model_id = checkpoint.get("source_model_id")
        model_id = model_id or model_cfg.get("model_family") or model_path_raw
    runtime = manifest.get("runtime", {})
    evaluator_cfg = resolved.get("eval", {}) if isinstance(resolved, Mapping) else {}
    trainer_cfg = resolved.get("trainer", {}) if isinstance(resolved, Mapping) else {}
    evidence_level = (
        "configured"
        if edit_count == 0
        else "verified_result"
        if complete_benchmark
        else "verified_smoke"
        if edit_count <= 2
        else "local_validation"
    )
    run_trace = {
        "status": manifest.get("status"),
        "operation": resolved.get("mode") if isinstance(resolved, Mapping) else None,
        "method": trainer_cfg.get("handler")
        if isinstance(trainer_cfg, Mapping)
        else None,
        "model_id": model_id,
        "model_revision": model_revision,
        "dataset_id": dataset_ids[0] if len(dataset_ids) == 1 else dataset_ids,
        "dataset_revision": input_manifest["dataset_revision"],
        "scope": (
            "one-official-record"
            if edit_count == 1
            else {
                "edited_records": edit_count,
                "benchmark_expected_cases": expected_cases,
                "complete_benchmark": complete_benchmark,
            }
        ),
        "protocol": {
            "name": "mm-edit-rglp-teacher-forced-token-accuracy-v3",
            "edit_type": resolved.get("edit_type")
            if isinstance(resolved, Mapping)
            else None,
            "evaluators": sorted(evaluator_cfg)
            if isinstance(evaluator_cfg, Mapping)
            else [],
            "locality": {
                "text_primary_top_k": 1,
                "image_primary_top_k": 1,
                "image_vlkeb_ike_compatible_top_k": 10,
                "top_k_comparison": "ordered_rankwise_equality",
            },
        },
        "evidence_level": evidence_level,
        "runtime_seconds": runtime.get("elapsed_seconds")
        if isinstance(runtime, Mapping)
        else None,
        "peak_cuda_memory_bytes": runtime.get("peak_cuda_memory_bytes")
        if isinstance(runtime, Mapping)
        else None,
        "outputs": sorted(
            path.name for path in output_path.iterdir() if path.is_file()
        ),
    }
    (output_path / "run_trace.json").write_text(
        json.dumps(to_jsonable(run_trace), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_runtime_device(cfg: DictConfig) -> str:
    device = str(cfg.get("device", "cuda"))
    require_cuda = bool(cfg.get("require_cuda", device.startswith("cuda")))
    cuda_available = torch.cuda.is_available()

    logger.info(
        "Runtime device check: device=%s, require_cuda=%s, cuda_available=%s",
        device,
        require_cuda,
        cuda_available,
    )

    if device.startswith("cuda") and not cuda_available:
        hint = (
            "CUDA is not available for mm_edit. "
            "Override with `device=cpu require_cuda=false` for a CPU smoke run."
        )
        if require_cuda:
            raise RuntimeError(hint)
        logger.warning("%s", hint)
        return "cpu"

    return device


def _runtime_edit_data(edit_requests) -> list[dict[str, Any]]:
    """Expose request fields to evaluators while retaining live PIL images."""
    runtime_data = []
    for request in edit_requests:
        if is_dataclass(request):
            runtime_data.append(
                {field.name: getattr(request, field.name) for field in fields(request)}
            )
        elif isinstance(request, Mapping):
            runtime_data.append(dict(request))
        else:
            raise TypeError(f"Unsupported edit request type: {type(request)}")
    return runtime_data


def _summarize_eval(eval_results: Mapping[str, Any]) -> dict[str, Any]:
    primary_keys = (
        "mm_reliability",
        "mm_generalization",
        "mm_locality",
        "mm_portability",
        "reliability",
        "generalization",
        "locality",
        "portability",
    )
    summary = {}
    for evaluator_name, result in eval_results.items():
        if not isinstance(result, Mapping):
            summary[evaluator_name] = result
            continue
        for key in primary_keys:
            if key in result:
                summary[evaluator_name] = result[key]
                break
    return summary


def _flatten_mm_case_metrics(eval_results: Mapping[str, Any]) -> dict[str, Any]:
    """Extract paper-facing numeric metrics without replacing missing values."""
    summary = {key: None for key in _MM_CASE_METRICS}
    for result in eval_results.values():
        if not isinstance(result, Mapping):
            continue
        for key in _MM_CASE_METRICS:
            value = result.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                summary[key] = float(value)
            elif key in result and value is None:
                summary[key] = None
        portability = result.get("portability_by_key")
        if isinstance(portability, Mapping):
            for key in ("text_portability_acc", "multimodal_portability_acc"):
                value = portability.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    summary[key] = float(value)
    return summary


def _aggregate_mm_case_metrics(
    case_summaries: list[Mapping[str, Any]],
) -> dict[str, Any]:
    metrics = {}
    for key in _MM_CASE_METRICS:
        values = [
            float(summary[key])
            for summary in case_summaries
            if isinstance(summary.get(key), (int, float))
            and not isinstance(summary.get(key), bool)
        ]
        metrics[key] = {
            "mean": sum(values) / len(values) if values else None,
            "available_cases": len(values),
            "missing_cases": len(case_summaries) - len(values),
        }

    return {
        "case_count": len(case_summaries),
        "metrics": metrics,
        "paper_columns": {
            "Reliability": metrics["mm_reliability"],
            "T-Generality": metrics["text_rephrase_acc"],
            "I-Generality": metrics["image_rephrase_acc"],
            "T-Locality": metrics["text_locality_acc"],
            "I-Locality": metrics["multimodal_locality_acc"],
            "Portability": metrics["mm_portability"],
        },
        "paper_protocol_columns": {
            "I-Locality (uniform top-1)": metrics["multimodal_locality_acc"],
            "I-Locality (VLKEB IKE ordered top-10)": metrics[
                "multimodal_locality_acc_top10"
            ],
        },
    }


def _manifest_eval_results(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    metrics = aggregate.get("metrics", {})
    evaluator_fields = {
        "mm_edit_metrics.mm_reliability": "mm_reliability",
        "mm_edit_metrics.mm_generalization": "mm_generalization",
        "mm_edit_metrics.mm_locality": "mm_locality",
        "mm_edit_metrics.mm_portability": "mm_portability",
    }
    return {
        evaluator: {field: metrics.get(field, {}).get("mean")}
        for evaluator, field in evaluator_fields.items()
    }


def _capture_pristine_locality_baseline(*, model, processor, request):
    """Capture locality from the base model, excluding an IKE/SERAC runtime."""
    runtime_present = hasattr(model, "_edit_runtime")
    runtime = getattr(model, "_edit_runtime", None)
    if runtime_present:
        delattr(model, "_edit_runtime")
    try:
        return MMEditLocalityEvaluator({}).capture_baseline(
            model=model,
            processor=processor,
            edit_data=_runtime_edit_data([request]),
        )
    finally:
        if runtime_present:
            model._edit_runtime = runtime


def _requires_mm_locality_baseline(eval_cfgs: Any) -> bool:
    """Detect locality evaluators at any Hydra nesting depth."""
    if isinstance(eval_cfgs, Mapping):
        for name, config in eval_cfgs.items():
            if str(name).endswith("mm_locality"):
                return True
            if isinstance(config, Mapping) and config.get("handler") == (
                "MMEditLocalityEvaluator"
            ):
                return True
            if _requires_mm_locality_baseline(config):
                return True
    elif isinstance(eval_cfgs, (list, tuple)):
        return any(_requires_mm_locality_baseline(item) for item in eval_cfgs)
    return False


def _prepare_independent_root(cfg: DictConfig, requests, output_dir: str) -> str:
    """Persist enough identity before a long run to make interruption auditable."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    (output_path / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    (output_path / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    _write_json_atomic(output_path / "edit_requests_preview.json", requests)
    identity = _sha256_json(resolved)
    _write_json_atomic(
        output_path / "independent_progress.json",
        {
            "status": "running",
            "run_identity": identity,
            "requested_count": len(requests),
            "completed_cases": 0,
        },
    )
    return identity


def _run_mm_independent_edit_benchmark(
    *,
    trainer,
    model,
    processor,
    edit_requests,
    eval_cfgs,
    output_dir: str,
    run_identity: str,
    resume: bool,
    progress_interval: int,
) -> dict[str, Any]:
    """Edit, evaluate, and restore every multimodal case independently."""
    output_path = Path(output_dir)
    cases_root = output_path / "independent_cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    capture_state = getattr(trainer, "capture_mm_independent_state", None)
    restore_state = getattr(trainer, "restore_mm_independent_state", None)
    if not callable(capture_state) or not callable(restore_state):
        raise TypeError(
            f"{trainer.__class__.__name__} lacks the MM independent reset contract"
        )
    base_state = capture_state()
    case_records = []
    case_summaries = []

    for index, request in enumerate(edit_requests):
        restore_state(base_state)
        case_dir = cases_root / f"{index:05d}"
        record_path = case_dir / "case_record.json"
        case_identity = _sha256_json(
            {"run_identity": run_identity, "case_index": index, "request": request}
        )

        if record_path.is_file() and resume:
            cached = json.loads(record_path.read_text(encoding="utf-8"))
            if cached.get("case_identity") != case_identity:
                raise RuntimeError(
                    f"Independent case {index} has a mismatched resume identity; "
                    "use a new output directory instead of mixing runs"
                )
            if (
                cached.get("success") is True
                and (case_dir / "mm_edit_inline_eval.json").is_file()
                and (case_dir / "mm_edit_inline_SUMMARY.json").is_file()
            ):
                case_records.append(cached)
                case_summaries.append(cached.get("summary", {}))
                continue

        locality_baseline = (
            _capture_pristine_locality_baseline(
                model=model, processor=processor, request=request
            )
            if _requires_mm_locality_baseline(eval_cfgs)
            else None
        )
        case_started = time.perf_counter()
        try:
            edit_result = trainer.edit(request)
            case_success = bool(
                edit_result.get("success", True)
                if isinstance(edit_result, Mapping)
                else True
            )
            save_edit_artifacts(
                case_dir,
                [request],
                {
                    "method": trainer.__class__.__name__,
                    "edit_type": "independent",
                    "requested_count": 1,
                    "result": edit_result,
                    "success": case_success,
                },
            )
            if not case_success:
                raise RuntimeError(f"Independent MM edit {index} reported failure")

            eval_results = _run_mm_edit_evaluation(
                eval_cfgs=eval_cfgs,
                model=model,
                processor=processor,
                edit_requests=[request],
                output_dir=str(case_dir),
                locality_baseline=locality_baseline,
            )
            summary = _flatten_mm_case_metrics(eval_results)
            record = {
                "case_index": index,
                "case_identity": case_identity,
                "success": True,
                "elapsed_seconds": time.perf_counter() - case_started,
                "edit_result": edit_result,
                "summary": summary,
                "output_dir": str(case_dir),
            }
            _write_json_atomic(record_path, record)
        except Exception as exc:
            record = {
                "case_index": index,
                "case_identity": case_identity,
                "success": False,
                "elapsed_seconds": time.perf_counter() - case_started,
                "error": f"{type(exc).__name__}: {exc}",
                "output_dir": str(case_dir),
            }
            _write_json_atomic(record_path, record)
            _write_json_atomic(
                output_path / "independent_progress.json",
                {
                    "status": "failed",
                    "run_identity": run_identity,
                    "requested_count": len(edit_requests),
                    "completed_cases": len(case_records),
                    "failed_case": index,
                    "error": record["error"],
                },
            )
            raise
        finally:
            restore_state(base_state)

        case_records.append(record)
        case_summaries.append(record["summary"])
        completed = index + 1
        if completed % max(progress_interval, 1) == 0 or completed == len(
            edit_requests
        ):
            _write_json_atomic(
                output_path / "independent_progress.json",
                {
                    "status": "running",
                    "run_identity": run_identity,
                    "requested_count": len(edit_requests),
                    "completed_cases": completed,
                },
            )
            logger.info(
                "Independent MM edit progress: %d/%d",
                completed,
                len(edit_requests),
            )

    aggregate = _aggregate_mm_case_metrics(case_summaries)
    payload = {
        "method": trainer.__class__.__name__,
        "edit_type": "independent",
        "requested_count": len(edit_requests),
        "successful_cases": sum(record["success"] for record in case_records),
        "cases": case_records,
        "aggregate": aggregate,
    }
    payload["success"] = payload["successful_cases"] == payload["requested_count"]
    _write_json_atomic(output_path / "independent_results.json", payload)
    _write_json_atomic(output_path / "independent_SUMMARY.json", aggregate)

    flat_summary = {
        f"mm_edit_metrics.{key}": value["mean"]
        for key, value in aggregate["metrics"].items()
    }
    _write_json_atomic(output_path / "mm_edit_inline_SUMMARY.json", flat_summary)
    _write_json_atomic(output_path / "mm_edit_inline_eval.json", aggregate)
    _write_json_atomic(
        output_path / "independent_progress.json",
        {
            "status": "completed",
            "run_identity": run_identity,
            "requested_count": len(edit_requests),
            "completed_cases": len(case_records),
            "successful_cases": payload["successful_cases"],
        },
    )
    return payload


def _run_mm_edit_evaluation(
    *,
    eval_cfgs,
    model,
    processor,
    edit_requests,
    output_dir: str,
    locality_baseline=None,
) -> dict[str, Any]:
    if not eval_cfgs:
        logger.info("No MM edit evaluators configured; skipping evaluation.")
        return {}

    evaluators = get_evaluators(eval_cfgs=eval_cfgs)
    edit_data = _runtime_edit_data(edit_requests)

    eval_results = {}
    for evaluator_name, evaluator in evaluators.items():
        eval_results[evaluator_name] = evaluator.evaluate(
            model=model,
            processor=processor,
            edit_data=edit_data,
            locality_baseline=locality_baseline,
            output_dir=output_dir,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "mm_edit_inline_eval.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(eval_results), f, ensure_ascii=False, indent=2)
    with (output_path / "mm_edit_inline_SUMMARY.json").open("w", encoding="utf-8") as f:
        json.dump(_summarize_eval(eval_results), f, ensure_ascii=False, indent=2)

    logger.info("Saved MM edit evaluation to %s", output_path)
    return eval_results


def _save_run_manifest(
    *,
    cfg: DictConfig,
    device: str,
    output_dir: str,
    edit_requests,
    edit_summary: Mapping[str, Any],
    eval_results: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    """Save the resolved command, environment, timing, and GPU footprint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Input identity is required even when the optional edit summary is disabled.
    _write_json_atomic(output_path / "edit_requests_preview.json", edit_requests)
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    (output_path / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    peak_memory = None
    gpu_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
        peak_memory = torch.cuda.max_memory_allocated(torch_device)
        gpu_name = torch.cuda.get_device_name(torch_device)

    manifest = {
        "status": "success",
        "task_name": cfg.task_name,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "gpu_name": gpu_name,
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_bytes": peak_memory,
        },
        "resolved_config": resolved_config,
        "edit_success": bool(edit_summary.get("success")),
        "evaluation": _summarize_eval(eval_results),
    }
    (output_path / "run_manifest.json").write_text(
        json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _save_standard_trace_files(output_path, manifest)


@hydra.main(version_base=None, config_path="../configs", config_name="mm_edit.yaml")
def main(cfg: DictConfig):
    started_at = time.perf_counter()
    seed_everything(cfg.trainer.args.seed)
    device = _resolve_runtime_device(cfg)

    model, processor = get_mm_model(cfg.model)
    if device.startswith("cuda"):
        model = model.to(device)
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    tokenizer = processor.tokenizer
    data = get_data(
        cfg.data,
        mode=cfg.get("mode", "edit"),
        tokenizer=tokenizer,
        template_args={},
    )

    trainer, trainer_args = load_trainer(
        trainer_cfg=cfg.trainer,
        model=model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=None,
        evaluators=None,
        template_args={},
    )
    if hasattr(trainer, "init_mm"):
        trainer.init_mm(processor)

    edit_type = str(cfg.get("edit_type", "single")).lower()
    max_edits = cfg.get("max_edits", None)
    if edit_type == "single" and (max_edits is None or max_edits > 1):
        max_edits = 1
    edit_requests = build_edit_requests(data.get("edit", None), max_edits=max_edits)

    if edit_type == "independent":
        if not trainer_args.do_eval or not cfg.get("eval", None):
            raise ValueError(
                "edit_type=independent requires trainer.args.do_eval=true and MM evaluators"
            )
        run_identity = _prepare_independent_root(
            cfg, edit_requests, trainer_args.output_dir
        )
        try:
            edit_summary = _run_mm_independent_edit_benchmark(
                trainer=trainer,
                model=model,
                processor=processor,
                edit_requests=edit_requests,
                eval_cfgs=cfg.get("eval", None),
                output_dir=trainer_args.output_dir,
                run_identity=run_identity,
                resume=bool(cfg.get("resume_independent", True)),
                progress_interval=int(cfg.get("progress_interval", 10)),
            )
        except Exception as exc:
            save_edit_artifacts(
                trainer_args.output_dir,
                edit_requests,
                {
                    "method": trainer.__class__.__name__,
                    "edit_type": "independent",
                    "requested_count": len(edit_requests),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

        if cfg.get("save_edit_summary", True):
            save_edit_artifacts(trainer_args.output_dir, edit_requests, edit_summary)
        if not edit_summary["success"]:
            raise RuntimeError(
                "Independent multimodal edit benchmark contains failed cases"
            )
        if cfg.get("save_edited_model", False):
            logger.warning(
                "edit_type=independent restores the base checkpoint after every case; "
                "no edited model is saved"
            )
        _save_run_manifest(
            cfg=cfg,
            device=device,
            output_dir=trainer_args.output_dir,
            edit_requests=edit_requests,
            edit_summary=edit_summary,
            eval_results=_manifest_eval_results(edit_summary["aggregate"]),
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return

    # The single-edit executor applies only the first request. Use that same
    # scope for probes, persisted previews, and reported target metrics.
    executed_requests = edit_requests[:1] if edit_type == "single" else edit_requests
    locality_baseline = None
    if trainer_args.do_eval and cfg.get("eval", None):
        locality_baseline = MMEditLocalityEvaluator({}).capture_baseline(
            model=model,
            processor=processor,
            edit_data=_runtime_edit_data(executed_requests),
        )
        logger.info(
            "Cached %d pre-edit locality predictions.",
            locality_baseline["total_samples"] if locality_baseline else 0,
        )

    try:
        edit_summary = execute_edit_requests(
            trainer=trainer,
            requests=executed_requests,
            edit_type=edit_type,
        )
    except Exception as exc:
        save_edit_artifacts(
            trainer_args.output_dir,
            executed_requests,
            {"success": False, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise

    edit_succeeded = bool(edit_summary["success"])
    if cfg.get("save_edit_summary", True) or not edit_succeeded:
        save_edit_artifacts(trainer_args.output_dir, executed_requests, edit_summary)

    if not edit_succeeded:
        raise RuntimeError(
            "Multimodal edit failed; inspect edit_results.json for details."
        )

    if cfg.get("save_edited_model", True):
        trainer.save_model(trainer_args.output_dir)
        if callable(getattr(processor, "save_pretrained", None)):
            processor.save_pretrained(trainer_args.output_dir)

    eval_results = {}
    if trainer_args.do_eval:
        eval_results = _run_mm_edit_evaluation(
            eval_cfgs=cfg.get("eval", None),
            model=model,
            processor=processor,
            edit_requests=executed_requests,
            output_dir=trainer_args.output_dir,
            locality_baseline=locality_baseline,
        )

    _save_run_manifest(
        cfg=cfg,
        device=device,
        output_dir=trainer_args.output_dir,
        edit_requests=executed_requests,
        edit_summary=edit_summary,
        eval_results=eval_results,
        elapsed_seconds=time.perf_counter() - started_at,
    )


if __name__ == "__main__":
    main()
