"""
Unlearning 训练主入口
====================

本脚本是基于 Hydra 配置框架的 unlearning 训练入口。

核心流程：
1. 通过 Hydra 加载和合并配置文件（支持命令行覆盖）
2. 初始化随机种子，确保实验可复现
3. 加载预训练模型和 tokenizer
4. 加载训练数据（forget 集和 retain 集）
5. 初始化 unlearning trainer（如 SimNPO、GradAscent 等）
6. 执行训练和评估

使用示例：
    python src/train.py \\
        --config-name=unlearn.yaml \\
        experiment=unlearn/tofu/default \\
        trainer=SimNPO \\
        forget_split=forget10 \\
        retain_split=retain90
"""

import copy
import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
import transformers
from omegaconf import DictConfig, OmegaConf
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from trainer.edit.pipeline import (
    build_edit_requests,
    execute_edit_requests,
    save_edit_artifacts,
)
from trainer.edit.artifacts import save_edit_artifacts as save_edit_runtime_artifacts
from evals import get_evaluators
from trainer.utils import seed_everything

logger = logging.getLogger(__name__)


def _edit_output_dir(cfg: DictConfig) -> Path:
    return Path(str(cfg.trainer.args.output_dir)).expanduser()


def _start_edit_run_trace(cfg: DictConfig) -> float:
    """Persist the resolved invocation/config before a surgery run starts."""
    output_dir = _edit_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    except Exception:
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=False)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        (output_dir / "resolved_config.yaml").write_text(
            resolved_yaml, encoding="utf-8"
        )
    command = str(cfg.get("invocation_command", "python src/train.py"))
    if rank == 0:
        (output_dir / "command.txt").write_text(f"{command}\n", encoding="utf-8")

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    environment = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": [
            torch.cuda.get_device_name(index) for index in range(cuda_device_count)
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": int(cfg.trainer.args.seed),
        "task_name": str(cfg.get("task_name", "")),
        "operation": str(cfg.get("operation", "edit")),
        "mode": str(cfg.get("mode", "")),
        "edit_type": str(cfg.get("edit_type", "single")),
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }
    environment_name = (
        "environment.json" if rank == 0 else f"environment.rank{rank}.json"
    )
    (output_dir / environment_name).write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if cuda_available:
        for index in range(cuda_device_count):
            torch.cuda.reset_peak_memory_stats(index)
    return time.perf_counter()


def _record_model_placement(output_dir: Path, model) -> None:
    """Record actual model allocation, including each DDP rank's local replica."""
    rank = int(os.environ.get("RANK", "0"))
    name = "environment.json" if rank == 0 else f"environment.rank{rank}.json"
    path = output_dir / name
    environment = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    parameter_numel_by_device = {}
    for parameter in model.parameters():
        device = str(parameter.device)
        parameter_numel_by_device[device] = (
            parameter_numel_by_device.get(device, 0) + parameter.numel()
        )
    environment["model_hf_device_map"] = {
        str(key): str(value)
        for key, value in getattr(model, "hf_device_map", {}).items()
    }
    environment["parameter_numel_by_device"] = parameter_numel_by_device
    path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _record_training_scope(output_dir: Path, data) -> None:
    """Record lengths after case limits, separately from original source sizes."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    dataset = data.get("train")
    if dataset is None:
        return

    def sample_count(value):
        if isinstance(value, Mapping):
            return sum(sample_count(item) for item in value.values())
        return len(value)

    used_samples = {"train": sample_count(dataset)}
    for split in ("forget", "retain"):
        scoped = getattr(dataset, split, None)
        if scoped is not None:
            used_samples[split] = sample_count(scoped)
    path = output_dir / "input_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest["used_samples"] = used_samples
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_edit_input_manifest(
    output_dir: Path,
    edit_data: Any,
    method_args: Mapping[str, Any] | None = None,
) -> None:
    datasets = edit_data.values() if isinstance(edit_data, Mapping) else [edit_data]
    records = []
    for dataset in datasets:
        data_path = getattr(dataset, "data_path", None)
        path = Path(data_path) if data_path is not None else None
        record = {
            "handler": dataset.__class__.__name__,
            "normalized_samples": len(dataset),
            "data_path": str(path) if path is not None else None,
            "sha256": None,
            "provenance_path": None,
        }
        if path is not None and path.is_file():
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            provenance = path.parent / "provenance.json"
            if provenance.is_file():
                record["provenance_path"] = str(provenance)
        records.append(record)
    method_inputs = []
    for key in ("archive", "edit_network_path", "projection_path"):
        raw_path = (method_args or {}).get(key)
        if not raw_path:
            continue
        path = Path(str(raw_path)).expanduser()
        entry = {
            "key": key,
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": None,
            "sha256": None,
        }
        if path.is_file():
            payload = path.read_bytes()
            entry["size_bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
        method_inputs.append(entry)

    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {"datasets": records, "method_inputs": method_inputs},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_unlearning_input_manifest(output_dir: Path, cfg: DictConfig) -> None:
    """Record the concrete local forget/retain inputs used by an unlearning run."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    records: list[dict[str, Any]] = []

    for split, datasets_cfg in data_cfg.items():
        if split == "anchor" or not isinstance(datasets_cfg, Mapping):
            continue
        for dataset_name, dataset_cfg in datasets_cfg.items():
            if not isinstance(dataset_cfg, Mapping):
                continue
            hf_args = dataset_cfg.get("args", {}).get("hf_args", {})
            data_files = hf_args.get("data_files")
            if isinstance(data_files, Mapping):
                paths = list(data_files.values())
            elif isinstance(data_files, (list, tuple)):
                paths = list(data_files)
            elif data_files is None:
                paths = []
            else:
                paths = [data_files]

            record = {
                "split": split,
                "dataset": dataset_name,
                "handler": dataset_cfg.get("handler"),
                "source": hf_args.get("path"),
                "source_config": hf_args.get("name"),
                "source_split": hf_args.get("split"),
                "files": [],
            }
            for raw_path in paths:
                path = Path(str(raw_path)).expanduser().resolve()
                file_record = {
                    "path": str(path),
                    "records": None,
                    "sha256": None,
                    "provenance_path": None,
                }
                if path.is_file():
                    payload = path.read_bytes()
                    file_record["sha256"] = hashlib.sha256(payload).hexdigest()
                    try:
                        decoded = json.loads(payload)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        decoded = None
                    if isinstance(decoded, list):
                        file_record["records"] = len(decoded)
                    provenance = path.parent / "provenance.json"
                    if provenance.is_file():
                        file_record["provenance_path"] = str(provenance.resolve())
                record["files"].append(file_record)
            records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "operation": str(cfg.get("operation", "unlearning")),
                "task_name": str(cfg.get("task_name", "")),
                "datasets": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_injection_input_manifest(output_dir: Path, cfg: DictConfig) -> None:
    """Record concrete train inputs used by a knowledge injection run."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    datasets_cfg = data_cfg.get("train", {}) if isinstance(data_cfg, Mapping) else {}
    records: list[dict[str, Any]] = []

    if isinstance(datasets_cfg, Mapping):
        for dataset_name, dataset_cfg in datasets_cfg.items():
            if not isinstance(dataset_cfg, Mapping):
                continue
            args = dataset_cfg.get("args", {})
            args = args if isinstance(args, Mapping) else {}
            hf_args = args.get("hf_args", {})
            hf_args = hf_args if isinstance(hf_args, Mapping) else {}
            raw_paths: list[Any] = []
            data_path = args.get("data_path")
            if data_path:
                raw_paths.append(data_path)
            data_files = hf_args.get("data_files")
            if isinstance(data_files, Mapping):
                raw_paths.extend(data_files.values())
            elif isinstance(data_files, (list, tuple)):
                raw_paths.extend(data_files)
            elif data_files:
                raw_paths.append(data_files)

            files = []
            for raw_path in raw_paths:
                path = Path(str(raw_path)).expanduser().resolve()
                file_record = {
                    "path": str(path),
                    "records": None,
                    "sha256": None,
                    "provenance_path": None,
                }
                if path.is_file():
                    payload = path.read_bytes()
                    file_record["sha256"] = hashlib.sha256(payload).hexdigest()
                    try:
                        if path.suffix.lower() == ".jsonl":
                            file_record["records"] = sum(
                                bool(line.strip())
                                for line in payload.decode("utf-8").splitlines()
                            )
                        else:
                            decoded = json.loads(payload)
                            if isinstance(decoded, list):
                                file_record["records"] = len(decoded)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass
                    provenance = path.parent / "provenance.json"
                    if provenance.is_file():
                        file_record["provenance_path"] = str(provenance.resolve())
                files.append(file_record)

            records.append(
                {
                    "split": "train",
                    "dataset": dataset_name,
                    "handler": dataset_cfg.get("handler"),
                    "source": hf_args.get("path") or ("local" if data_path else None),
                    "source_config": hf_args.get("name"),
                    "source_split": hf_args.get("split"),
                    "format_type": args.get("format_type"),
                    "files": files,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "operation": str(cfg.get("operation", "inject")),
                "task_name": str(cfg.get("task_name", "")),
                "datasets": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _finish_edit_run_trace(
    output_dir: Path, started_at: float, *, status: str = "success"
) -> None:
    peak_by_device = {}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            peak_by_device[str(index)] = {
                "name": torch.cuda.get_device_name(index),
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(index),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(index),
            }
    trace = {
        "status": status,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started_at,
        "peak_cuda_memory": peak_by_device,
        "artifacts": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }
    rank = int(os.environ.get("RANK", "0"))
    trace["rank"] = rank
    trace["world_size"] = int(os.environ.get("WORLD_SIZE", "1"))
    trace_name = "run_trace.json" if rank == 0 else f"run_trace.rank{rank}.json"
    (output_dir / trace_name).write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _require_successful_edit_execution(
    summary: Mapping[str, Any], output_dir: Path, started_at: float
) -> None:
    """Reject an update failure before model saving or evaluation.

    Editor ``success`` means that update execution completed; it is distinct
    from evaluator efficacy. A failed update must never produce a successful
    run trace or an apparently valid edited checkpoint.
    """

    if summary.get("success") is True:
        return
    _finish_edit_run_trace(output_dir, started_at, status="failed")
    raise RuntimeError(
        "Knowledge edit execution failed; diagnostic artifacts were preserved "
        "and run_trace.status was set to failed. The edited model was not saved "
        "or evaluated."
    )


def _to_jsonable(value: Any) -> Any:
    """Convert edit requests/results into JSON-safe structures."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    try:
        from PIL.Image import Image as PILImage

        if isinstance(value, PILImage):
            return f"<PIL.Image mode={value.mode} size={value.size}>"
    except ImportError:
        pass
    return value


def _summarize_edit_eval(eval_results: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact metric values aligned with EditSuite SUMMARY fields."""
    primary_keys = (
        "reliability",
        "generalization",
        "locality",
        "portability",
        "overall_score",
        "mm_reliability",
        "mm_generalization",
        "mm_locality",
        "bleu4",
        "rouge1",
        "rouge2",
        "rouge_l",
        "token_f1",
        "semantic_similarity",
        "mmlu_pre_accuracy",
        "mmlu_post_accuracy",
        "mmlu_accuracy_delta",
    )
    summary: dict[str, Any] = {}

    # Prefer flattened suite metrics when present (single EditSuiteEvaluator call).
    suite_blob = eval_results.get("edit") or eval_results.get("EditSuiteEvaluator")
    if isinstance(suite_blob, Mapping):
        for key in primary_keys:
            if key in suite_blob:
                summary[key] = suite_blob[key]
        # Nested per-metric dicts from suite logs
        for metric_name, metric_result in suite_blob.items():
            if metric_name in summary:
                continue
            if isinstance(metric_result, Mapping):
                for key in primary_keys:
                    if key in metric_result:
                        summary[metric_name] = metric_result[key]
                        break
            elif metric_name in primary_keys:
                summary[metric_name] = metric_result
        if summary:
            return summary

    for evaluator_name, result in eval_results.items():
        if not isinstance(result, Mapping):
            summary[evaluator_name] = result
            continue
        for key in primary_keys:
            if key in result:
                summary[key if key in primary_keys else evaluator_name] = result[key]
                if key == evaluator_name or key in primary_keys:
                    summary.setdefault(key, result[key])
                break
        else:
            summary[evaluator_name] = result
    return summary


def _snapshot_original_model(model):
    """CPU state-dict snapshot used to restore a pre-edit reference model."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _materialize_original_model(model, state_snapshot):
    """Clone ``model`` architecture and load the pre-edit snapshot."""
    # IKE/SERAC attach a non-parametric runtime to the edited model.  Detach it
    # while cloning so the locality reference remains genuinely pre-edit and
    # deepcopy does not traverse the trainer -> model cycle.
    runtime = getattr(model, "_edit_runtime", None)
    if runtime is not None:
        delattr(model, "_edit_runtime")
    try:
        original = copy.deepcopy(model)
    finally:
        if runtime is not None:
            model._edit_runtime = runtime
    original.load_state_dict(state_snapshot, strict=True)
    original.eval()
    for p in original.parameters():
        p.requires_grad = False
    return original


def _clone_original_model(model, device: str | torch.device = "cpu"):
    """Clone the pristine pre-edit model onto an explicit evaluation device.

    GRACE and WISE replace a projection with an adapter at edit time.  Cloning
    the model afterwards and loading a plain pre-edit state dict therefore
    produces incompatible keys. The default CPU placement avoids holding a
    second full model on the editing GPU; multi-GPU runs may explicitly place
    the reference on another visible device to avoid prohibitively slow CPU
    locality evaluation.
    """
    reference_device = torch.device(device)
    if reference_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "edit_reference_device requests CUDA but CUDA is unavailable"
            )
        if reference_device.index is None:
            raise ValueError(
                "edit_reference_device must name an explicit CUDA index, "
                "for example cuda:1"
            )
        if reference_device.index >= torch.cuda.device_count():
            raise ValueError(
                f"edit_reference_device={reference_device} is not visible; "
                f"visible CUDA device count is {torch.cuda.device_count()}"
            )

    original = copy.deepcopy(model)
    if getattr(model, "hf_device_map", None) is not None or any(
        hasattr(module, "_hf_hook") for module in model.modules()
    ):
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(original, recurse=True)
        for module in original.modules():
            if hasattr(module, "hf_device_map"):
                delattr(module, "hf_device_map")
        # Accelerate also wraps movement methods in functions closing over the
        # source model. Deepcopy preserves those closures; restore class methods
        # on the copy before moving it, leaving the live edited model untouched.
        for name in ("to", "cuda", "cpu", "half", "float", "bfloat16"):
            method = original.__dict__.get(name)
            if method is not None and hasattr(method, "__wrapped__"):
                delattr(original, name)
    original.to(reference_device)
    original.eval()
    for parameter in original.parameters():
        parameter.requires_grad = False
    return original


def _clone_original_model_on_cpu(model):
    """Backward-compatible CPU reference helper used by unit tests."""
    return _clone_original_model(model, "cpu")


def _restore_independent_model(trainer, model, state_snapshot) -> None:
    """Restore both model weights and editor-installed module structure."""
    reset_layer = getattr(trainer, "reset_layer", None)
    if callable(reset_layer):
        reset_layer()
    model.load_state_dict(state_snapshot, strict=True)
    _reset_independent_editor_state(trainer, model)


def _run_inline_edit_evaluation(
    *,
    eval_cfgs,
    model,
    tokenizer,
    template_args,
    edit_requests,
    output_dir,
    original_model=None,
) -> dict[str, Any]:
    """Run edit evaluation via EditSuiteEvaluator (single call) when configured.

    Matches ``eval.py`` behaviour: one suite handler orchestrates reliability /
    generalization / locality / portability. Locality requires ``original_model``.
    """
    if not eval_cfgs:
        logger.info("No edit evaluators configured; skipping inline edit evaluation.")
        return {}

    if original_model is None:
        logger.warning(
            "Inline edit evaluation running without original_model; "
            "locality will be marked unavailable (not defaulted to 1.0)."
        )

    edit_data = [_to_jsonable(request) for request in edit_requests]
    eval_results: dict[str, Any] = {}

    # Detect EditSuiteEvaluator as the default edit node (configs/eval/edit.yaml).
    edit_node = None
    if isinstance(eval_cfgs, Mapping) or OmegaConf.is_config(eval_cfgs):
        edit_node = eval_cfgs.get("edit", None)
        if edit_node is None and eval_cfgs.get("handler", None) == "EditSuiteEvaluator":
            edit_node = eval_cfgs

    suite_handler = None
    if edit_node is not None:
        suite_handler = edit_node.get("handler", None)

    if suite_handler == "EditSuiteEvaluator":
        from evals.edit import EditSuiteEvaluator

        suite = EditSuiteEvaluator(edit_node)
        suite_logs = suite.evaluate(
            model=model,
            tokenizer=tokenizer,
            edit_data=edit_data,
            original_model=original_model,
            output_dir=output_dir,
            overwrite=True,
        )
        # Persist both raw suite logs and a compact summarized view.
        eval_results["edit"] = suite_logs
        summarized = suite.summarize(suite_logs)
        eval_results["edit"].update(
            {k: v for k, v in summarized.items() if k not in suite_logs}
        )
        # Also attach top-level primary keys for SUMMARY convenience
        for key, value in summarized.items():
            eval_results["edit"].setdefault(key, value)
    else:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )
        for evaluator_name, evaluator in evaluators.items():
            eval_results[evaluator_name] = evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                edit_data=edit_data,
                original_model=original_model,
                output_dir=output_dir,
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "edit_inline_eval.json").open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(eval_results), f, ensure_ascii=False, indent=2)
    compact_summary = _summarize_edit_eval(eval_results)
    with (output_path / "edit_inline_SUMMARY.json").open("w", encoding="utf-8") as f:
        json.dump(compact_summary, f, ensure_ascii=False, indent=2)

    logger.info("Saved inline edit evaluation to %s", output_path)
    return compact_summary


def _prepare_edit_evaluation_requests(trainer, requests):
    """Apply an editor's public evaluation-preprocessing contract when present.

    Instruction-conditioned editors must evaluate the same wrapped prompts used
    for editing.  Most editors expose no hook and retain the canonical requests.
    """

    prepare = getattr(trainer, "prepare_evaluation_requests", None)
    if not callable(prepare):
        return list(requests)
    prepared = list(prepare(list(requests)))
    if len(prepared) != len(requests):
        raise ValueError(
            "Editor evaluation preprocessing changed the number of edit cases"
        )
    return prepared


def _edit_history_checkpoints(cfg, request_count: int) -> set[int]:
    """Validate explicit sequential evaluation steps before changing weights."""
    raw_steps = cfg.get("edit_history_eval_steps", [])
    if not raw_steps:
        return set()
    if str(cfg.get("edit_type", "single")).lower() != "sequential":
        raise ValueError("edit_history_eval_steps requires edit_type=sequential")
    if not cfg.trainer.args.get("do_eval", False) or not cfg.get("eval"):
        raise ValueError("edit_history_eval_steps requires configured evaluation")
    if any(
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 1
        or step > request_count
        for step in raw_steps
    ):
        raise ValueError(
            "edit_history_eval_steps must contain integer steps in [1, request_count]"
        )
    if len(set(raw_steps)) != len(raw_steps):
        raise ValueError("edit_history_eval_steps must not contain duplicate steps")
    return set(raw_steps)


def _make_edit_history_callback(
    *,
    checkpoints,
    trainer,
    tokenizer,
    template_args,
    eval_cfgs,
    output_dir,
    original_model,
):
    """Evaluate successful prefixes and preserve their per-example artifacts.

    Checkpoints use the same suite as final evaluation and the pristine model
    for locality. Request indices identify stable cohorts across growing runs.
    """
    records = []
    history_dir = Path(output_dir) / "history"

    def evaluate_prefix(step, requests):
        if step not in checkpoints:
            return
        checkpoint_dir = history_dir / f"step_{step:06d}"
        training_modes = [
            (module, module.training) for module in trainer.model.modules()
        ]
        try:
            summary = _run_inline_edit_evaluation(
                eval_cfgs=eval_cfgs,
                model=trainer.model,
                tokenizer=tokenizer,
                template_args=template_args,
                edit_requests=_prepare_edit_evaluation_requests(trainer, requests),
                output_dir=checkpoint_dir,
                original_model=original_model,
            )
        finally:
            for module, training in training_modes:
                module.training = training
        records.append(
            {
                "step": step,
                "evaluated_request_indices": list(range(len(requests))),
                "evaluation_scope": "all_successful_edits_so_far",
                "summary": summary,
                "artifacts_dir": checkpoint_dir.name,
            }
        )
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "reference": "pristine_model",
                    "requested_checkpoints": sorted(checkpoints),
                    "checkpoints": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return evaluate_prefix


def _reset_independent_editor_state(trainer, model) -> None:
    """Clear state that must not leak across independent benchmark cases."""
    if hasattr(model, "_edit_runtime"):
        delattr(model, "_edit_runtime")

    reset_independent_state = getattr(trainer, "reset_independent_state", None)
    if callable(reset_independent_state):
        reset_independent_state()
    else:
        # Backward-compatible public reset hooks used by existing editors.
        # These must run before the generic cache cleanup below: IKE and SERAC
        # keep several parallel collections whose lengths must stay aligned.
        for hook_name in (
            "clear_knowledge_store",
            "clear_edit_memory",
            "reset_edit_memory",
        ):
            hook = getattr(trainer, hook_name, None)
            if callable(hook):
                hook()

    history = getattr(trainer, "edit_history", None)
    if hasattr(history, "clear"):
        history.clear()
    for attribute in (
        "_sequential_key_cache",
        "memory",
        "edit_memory",
        "archive",
    ):
        value = getattr(trainer, attribute, None)
        if hasattr(value, "clear"):
            value.clear()


def _aggregate_independent_summaries(
    case_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted({key for summary in case_summaries for key in summary})
    aggregate: dict[str, Any] = {
        "case_count": len(case_summaries),
        "metrics": {},
    }
    for key in keys:
        values = [
            float(summary[key])
            for summary in case_summaries
            if isinstance(summary.get(key), (int, float))
            and summary.get(key) is not None
        ]
        aggregate["metrics"][key] = {
            "mean": sum(values) / len(values) if values else None,
            "available_cases": len(values),
            "missing_cases": len(case_summaries) - len(values),
        }
    return aggregate


def _run_independent_edit_benchmark(
    *,
    trainer,
    model,
    tokenizer,
    template_args,
    requests,
    eval_cfgs,
    output_dir,
    base_snapshot,
) -> dict[str, Any]:
    """Edit/evaluate/restore each case, matching benchmark paper semantics."""
    output_path = Path(output_dir)
    cases_root = output_path / "independent_cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    original_model = _materialize_original_model(model, base_snapshot)
    case_records = []
    case_summaries = []

    try:
        for index, request in enumerate(requests):
            _restore_independent_model(trainer, model, base_snapshot)
            edit_result = trainer.edit(request)
            case_dir = cases_root / f"{index:05d}"
            save_edit_artifacts(
                case_dir,
                [request],
                {
                    "method": trainer.__class__.__name__,
                    "edit_type": "independent",
                    "requested_count": 1,
                    "result": edit_result,
                    "success": bool(
                        edit_result.get("success", True)
                        if isinstance(edit_result, Mapping)
                        else True
                    ),
                },
            )
            case_success = bool(
                edit_result.get("success", True)
                if isinstance(edit_result, Mapping)
                else True
            )
            summary = {}
            if case_success:
                summary = _run_inline_edit_evaluation(
                    eval_cfgs=eval_cfgs,
                    model=model,
                    tokenizer=tokenizer,
                    template_args=template_args,
                    edit_requests=_prepare_edit_evaluation_requests(trainer, [request]),
                    output_dir=case_dir,
                    original_model=original_model,
                )
            case_summaries.append(summary)
            case_records.append(
                {
                    "case_index": index,
                    "success": case_success,
                    "edit_result": _to_jsonable(edit_result),
                    "summary": summary,
                    "output_dir": str(case_dir),
                }
            )
    finally:
        _restore_independent_model(trainer, model, base_snapshot)
        del original_model

    aggregate = _aggregate_independent_summaries(case_summaries)
    payload = {
        "method": trainer.__class__.__name__,
        "edit_type": "independent",
        "requested_count": len(requests),
        "successful_cases": sum(record["success"] for record in case_records),
        "cases": case_records,
        "aggregate": aggregate,
    }
    payload["success"] = payload["successful_cases"] == payload["requested_count"]
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "independent_results.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_to_jsonable(payload), handle, ensure_ascii=False, indent=2)
    with (output_path / "independent_SUMMARY.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    return payload


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """训练主函数

    通过 Hydra 装饰器自动加载配置文件，并支持命令行参数覆盖。

    Args:
        cfg (DictConfig): Hydra 配置对象，包含以下主要部分：
            - model: 模型配置（模型路径、注意力实现等）
            - data: 数据配置（forget/retain 数据集、split 等）
            - trainer: 训练器配置（算法类型、超参数等）
            - eval: 评估配置（评估指标、评估频率等）
            - mode: 训练模式（"train" 或 "unlearn"）

    配置加载顺序：
        1. 加载 config_name 指定的基础配置（如 unlearn.yaml）
        2. 应用 experiment 配置覆盖（如 experiment/unlearn/tofu/default.yaml）
        3. 应用命令行参数覆盖（如 trainer=SimNPO）
    """

    # ==================== 步骤1: 设置随机种子 ====================
    # 确保实验可复现（固定模型初始化、数据打乱、dropout 等随机性）
    seed_everything(cfg.trainer.args.seed)

    # ==================== 步骤2: 加载模型配置 ====================
    # mode 决定数据加载方式：
    #   - "train"/"finetune"/"inject"/"edit": 直接按 split 返回数据
    #   - "unlearn": 遗忘训练，将 forget/retain 合并为 ForgetRetainDataset
    #   - "edit": 知识编辑，按 data.edit 加载原始编辑数据
    mode = cfg.get("mode", "train")
    run_started_at = _start_edit_run_trace(cfg)

    model_cfg = cfg.model
    # template_args 包含对话模板相关配置（用于格式化输入输出）
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."

    # 加载预训练模型和 tokenizer
    # 返回值：
    #   - model: HuggingFace 预训练模型（如 LLaMA、Mistral 等）
    #   - tokenizer: 对应的 tokenizer
    model, tokenizer = get_model(model_cfg)

    # Build the locality reference while the model still has its pristine
    # architecture.  Adapter editors (GRACE/WISE) replace modules during
    # ``edit`` and cannot be strict-loaded from a plain pre-edit state dict.
    original_model = None
    requested_edit_type = str(cfg.get("edit_type", "single")).lower()
    if (
        mode == "edit"
        and requested_edit_type != "independent"
        and bool(cfg.trainer.args.get("do_eval", False))
        and cfg.get("eval", None)
    ):
        reference_device = str(cfg.get("edit_reference_device", "cpu"))
        original_model = _clone_original_model(model, reference_device)
        logger.info(
            "Cloned pristine model to %s for locality evaluation.",
            reference_device,
        )

    # ==================== 步骤3: 加载数据集 ====================
    data_cfg = cfg.data
    # get_data 根据 mode 返回不同格式的数据：
    #   - mode="train"/"finetune"/"inject"/"edit": {"train": Dataset, "eval": Dataset, ...}
    #   - mode="unlearn": {"train": ForgetRetainDataset, "eval": Dataset}
    #     其中 ForgetRetainDataset 每次返回 {"forget": sample, "retain": sample}
    data_args = {"tokenizer": tokenizer, "template_args": template_args}
    if cfg.get("max_train_samples") is not None:
        data_args["max_samples"] = cfg.max_train_samples
    data = get_data(data_cfg, mode=mode, **data_args)
    if mode == "edit":
        method_args = OmegaConf.to_container(
            cfg.trainer.get("method_args", {}), resolve=True
        )
        _write_edit_input_manifest(
            _edit_output_dir(cfg),
            data.get("edit", None),
            method_args=method_args,
        )
    elif mode == "unlearn" and int(os.environ.get("RANK", "0")) == 0:
        _write_unlearning_input_manifest(_edit_output_dir(cfg), cfg)
    elif mode == "inject" and int(os.environ.get("RANK", "0")) == 0:
        _write_injection_input_manifest(_edit_output_dir(cfg), cfg)
    _record_training_scope(_edit_output_dir(cfg), data)

    # ==================== 步骤4: 加载数据整理器 (Collator) ====================
    # Collator 负责将多个样本整理成一个 batch
    # 主要功能：padding、截断、生成 attention_mask 等
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # ==================== 步骤5: 初始化 Trainer ====================
    trainer_cfg = cfg.trainer
    # trainer_cfg 包含：
    #   - handler: Trainer 类名（如 "SimNPO"、"GradAscent"）
    #   - args: TrainingArguments（学习率、batch size、训练轮数等）
    #   - method_args: 算法特定参数（如 SimNPO 的 delta、beta、gamma 等）
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # ==================== 步骤6: 加载评估器 ====================
    # 评估器用于在训练过程中或训练后评估模型性能
    # 常见评估指标：ROUGE、概率差异、模型效用等
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs and mode != "edit":
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    # ==================== 步骤7: 实例化 Trainer ====================
    # load_trainer 根据 trainer_cfg.handler 从注册表中选择对应的 Trainer 类
    # 例如：handler="SimNPO" → 实例化 SimNPO 类
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),  # ForgetRetainDataset（unlearn模式）
        eval_dataset=data.get("eval", None),  # 评估数据集
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,  # 评估器列表
        template_args=template_args,
    )
    _record_model_placement(_edit_output_dir(cfg), trainer.model)

    if mode == "edit":
        edit_requests = build_edit_requests(
            data.get("edit", None), max_edits=cfg.get("max_edits", None)
        )
        edit_type = cfg.get("edit_type", "single").lower()
        executed_requests = (
            edit_requests[:1] if edit_type == "single" else edit_requests
        )
        history_checkpoints = _edit_history_checkpoints(cfg, len(executed_requests))

        # Snapshot pre-edit weights for locality (EditSuite / eval.py parity).
        pre_edit_snapshot = None
        if edit_type == "independent":
            pre_edit_snapshot = _snapshot_original_model(model)
            logger.info("Captured pre-edit model snapshot for locality evaluation.")

        if edit_type == "independent":
            if not trainer_args.do_eval or not eval_cfgs:
                raise ValueError(
                    "edit_type=independent requires do_eval=true and an edit evaluator"
                )
            independent_summary = _run_independent_edit_benchmark(
                trainer=trainer,
                model=model,
                tokenizer=tokenizer,
                template_args=template_args,
                requests=executed_requests,
                eval_cfgs=eval_cfgs,
                output_dir=trainer_args.output_dir,
                base_snapshot=pre_edit_snapshot,
            )
            if cfg.get("save_edit_summary", True):
                save_edit_artifacts(
                    trainer_args.output_dir,
                    executed_requests,
                    independent_summary,
                )
            _require_successful_edit_execution(
                independent_summary,
                _edit_output_dir(cfg),
                run_started_at,
            )
            if cfg.get("save_edited_model", True):
                logger.warning(
                    "edit_type=independent restores the base checkpoint after each "
                    "case; no edited model is saved."
                )
            _finish_edit_run_trace(
                _edit_output_dir(cfg), run_started_at, status="success"
            )
            return

        history_callback = None
        if history_checkpoints:
            history_callback = _make_edit_history_callback(
                checkpoints=history_checkpoints,
                trainer=trainer,
                tokenizer=tokenizer,
                template_args=template_args,
                eval_cfgs=eval_cfgs,
                output_dir=trainer_args.output_dir,
                original_model=original_model,
            )
        try:
            edit_summary = execute_edit_requests(
                trainer=trainer,
                requests=executed_requests,
                edit_type=edit_type,
                on_step=history_callback,
            )
        except Exception:
            _finish_edit_run_trace(
                _edit_output_dir(cfg), run_started_at, status="failed"
            )
            raise

        if cfg.get("save_edit_summary", True):
            save_edit_artifacts(
                trainer_args.output_dir, executed_requests, edit_summary
            )

        _require_successful_edit_execution(
            edit_summary,
            _edit_output_dir(cfg),
            run_started_at,
        )

        if cfg.get("save_edited_model", True):
            trainer.save_model(trainer_args.output_dir)
            if trainer.is_world_process_zero():
                save_edit_runtime_artifacts(
                    trainer,
                    trainer_args.output_dir,
                    method_args=OmegaConf.to_container(
                        cfg.trainer.get("method_args", {}), resolve=True
                    ),
                )

        if trainer_args.do_eval:
            if pre_edit_snapshot is not None:
                original_model = _materialize_original_model(model, pre_edit_snapshot)
                del pre_edit_snapshot
            _run_inline_edit_evaluation(
                eval_cfgs=eval_cfgs,
                model=model,
                tokenizer=tokenizer,
                template_args=template_args,
                edit_requests=_prepare_edit_evaluation_requests(
                    trainer, executed_requests
                ),
                output_dir=trainer_args.output_dir,
                original_model=original_model,
            )
            if original_model is not None:
                del original_model
        _finish_edit_run_trace(_edit_output_dir(cfg), run_started_at, status="success")
        return

    # ==================== 步骤8: 执行训练 ====================
    if trainer_args.do_train:
        # 开始训练循环
        # 训练过程中会：
        #   1. 从 ForgetRetainDataset 采样 batch
        #   2. 调用 trainer.compute_loss() 计算损失
        #   3. 反向传播更新模型参数
        #   4. 定期评估和保存模型
        trainer.train()

        # 保存训练状态（optimizer、scheduler、随机数生成器状态等）
        trainer.save_state()

        # 保存最终模型权重
        trainer.save_model(trainer_args.output_dir)

    # ==================== 步骤9: 执行评估 ====================
    if trainer_args.do_eval:
        # 在评估集上评估模型性能
        # 返回评估指标（如 loss、ROUGE、准确率等）
        trainer.evaluate(metric_key_prefix="eval")

    _finish_edit_run_trace(_edit_output_dir(cfg), run_started_at, status="success")


if __name__ == "__main__":
    main()
