"""Evidence-aware parsing for existing and newly produced run artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .catalog import ROOT


METRICS: dict[str, dict[str, Any]] = {
    "reliability": {
        "label": {"en": "Reliability", "zh": "编辑可靠性"},
        "direction": "higher",
        "category": "efficacy",
        "unit": "ratio",
    },
    "generalization": {
        "label": {"en": "Generalization", "zh": "改写泛化"},
        "direction": "higher",
        "category": "efficacy",
        "unit": "ratio",
    },
    "locality": {
        "label": {"en": "Locality", "zh": "局部性"},
        "direction": "higher",
        "category": "side_effect",
        "unit": "ratio",
    },
    "portability": {
        "label": {"en": "Portability", "zh": "迁移性"},
        "direction": "higher",
        "category": "side_effect",
        "unit": "ratio",
    },
    "task_accuracy": {
        "label": {"en": "Task accuracy", "zh": "任务准确率"},
        "direction": "higher",
        "category": "efficacy",
        "unit": "ratio",
    },
    "knowledge_retention": {
        "label": {"en": "Knowledge retention", "zh": "知识保持"},
        "direction": "higher",
        "category": "side_effect",
        "unit": "ratio",
    },
    "current_score": {
        "label": {"en": "Current benchmark accuracy", "zh": "当前基准绝对准确率"},
        "direction": "higher",
        "category": "side_effect",
        "unit": "ratio",
    },
    "original_score": {
        "label": {"en": "Original benchmark accuracy", "zh": "原始基准绝对准确率"},
        "direction": "context",
        "category": "diagnostic",
        "unit": "ratio",
    },
    "forget_nll_change": {
        "label": {"en": "Forget NLL change", "zh": "遗忘 NLL 变化"},
        "direction": "higher",
        "category": "efficacy",
        "unit": "nll",
    },
    "retain_nll_change": {
        "label": {"en": "Retain NLL change", "zh": "保留 NLL 变化"},
        "direction": "closer_to_zero",
        "category": "side_effect",
        "unit": "nll",
    },
    "base_forget_nll": {
        "label": {"en": "Base forget NLL", "zh": "原始遗忘集 NLL"},
        "direction": "context",
        "category": "diagnostic",
        "unit": "nll",
    },
    "edited_forget_nll": {
        "label": {"en": "Post forget NLL", "zh": "遗忘后 NLL"},
        "direction": "context",
        "category": "diagnostic",
        "unit": "nll",
    },
    "base_retain_nll": {
        "label": {"en": "Base retain NLL", "zh": "原始保留集 NLL"},
        "direction": "context",
        "category": "diagnostic",
        "unit": "nll",
    },
    "edited_retain_nll": {
        "label": {"en": "Post retain NLL", "zh": "遗忘后保留集 NLL"},
        "direction": "context",
        "category": "diagnostic",
        "unit": "nll",
    },
    "base_target_nll": {
        "label": {"en": "Base target NLL", "zh": "原始目标 NLL"},
        "direction": "lower",
        "category": "diagnostic",
        "unit": "nll",
    },
    "post_target_nll": {
        "label": {"en": "Post target NLL", "zh": "注入后目标 NLL"},
        "direction": "lower",
        "category": "efficacy",
        "unit": "nll",
    },
    "target_nll_delta": {
        "label": {"en": "Target NLL delta", "zh": "目标 NLL 变化"},
        "direction": "lower",
        "category": "efficacy",
        "unit": "nll",
    },
    "runtime_seconds": {
        "label": {"en": "Runtime", "zh": "运行时间"},
        "direction": "lower",
        "category": "efficiency",
        "unit": "seconds",
    },
    "peak_memory_gib": {
        "label": {"en": "Peak GPU memory", "zh": "峰值显存"},
        "direction": "lower",
        "category": "efficiency",
        "unit": "GiB",
    },
    "overall_score": {
        "label": {"en": "Overall score", "zh": "综合得分"},
        "direction": "higher",
        "category": "efficacy",
        "unit": "score",
    },
}

CURATED_DEMOS: list[dict[str, Any]] = []

COMPARISON_BASIS = (
    "Compares recorded inputs, model revision, record counts, evaluation settings, "
    "run limits, environment, metric protocols and scoring semantics. "
    "Unmatched results remain side by side without a ranking."
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 12 * 1024 * 1024:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _result_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]


def _operation(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) < 2:
        return "unknown"
    lane = parts[1]
    return {
        "unlearn": "unlearning",
        "unlearning": "unlearning",
        "inject": "inject",
        "edit": "edit",
        "eval": "evaluation",
    }.get(lane, "unknown")


def _candidate_dirs(root: Path) -> list[Path]:
    saves = root / "saves"
    if not saves.exists():
        return []
    directories: set[Path] = set()
    patterns = (
        "run_trace.json",
        "*_SUMMARY.json",
        "unlearning_metrics.json",
        "target_nll.json",
        "edit_results.json",
        "independent_progress.json",
    )
    for pattern in patterns:
        for path in saves.rglob(pattern):
            directory = path.parent
            checkpoint = directory.parent
            if (
                directory.name == "evals"
                and checkpoint.name.startswith("checkpoint-")
                and checkpoint.name.removeprefix("checkpoint-").isdigit()
                and any(
                    (checkpoint.parent / name).is_file()
                    for name in (
                        "trainer_state.json",
                        "run_trace.json",
                        "resolved_config.yaml",
                    )
                )
            ):
                directory = checkpoint.parent
            directories.add(directory)
    return sorted(
        directories,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _history_link(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Link a saved evaluation to its parent only through a matching history index."""
    history_dir = run_dir.parent
    if (
        history_dir.name != "history"
        or run_dir.resolve().parent != history_dir.resolve()
    ):
        return None
    index_path = history_dir / "index.json"
    if index_path.resolve().parent != history_dir.resolve():
        return None
    index = _read_json(index_path)
    records = index.get("checkpoints")
    if index.get("schema_version") != 1 or not isinstance(records, list):
        return None
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("artifacts_dir") == run_dir.name
    ]
    if len(matches) != 1:
        return None
    record = matches[0]
    step, indices = record.get("step"), record.get("evaluated_request_indices")
    if (
        type(step) is not int
        or step <= 0
        or not isinstance(indices, list)
        or not indices
        or len(indices) > step
        or any(
            type(value) is not int or value < 0 or value >= step for value in indices
        )
        or len(set(indices)) != len(indices)
        or not isinstance(record.get("evaluation_scope"), str)
        or not record["evaluation_scope"]
    ):
        return None
    summary_path = run_dir / "edit_SUMMARY.json"
    if summary_path.resolve().parent != run_dir.resolve():
        return None
    summary = _read_json(summary_path)
    if not summary or summary != record.get("summary"):
        return None
    return history_dir.parent, {
        "step": step,
        "reference": index.get("reference"),
        "evaluation_scope": record["evaluation_scope"],
        "evaluated_request_indices": indices,
        "parent_run": history_dir.parent.name,
        "index_source": "history/index.json",
    }


def _resolved_metadata(run_dir: Path) -> dict[str, Any]:
    config_name = "resolved_config.yaml"
    config = _read_yaml(run_dir / "resolved_config.yaml")
    if not config:
        config_name = ".hydra/config.yaml"
        config = _read_yaml(run_dir / ".hydra" / "config.yaml")
    history_link = _history_link(run_dir)
    source_dir = run_dir
    if history_link and not config:
        source_dir = history_link[0]
        config_name = "resolved_config.yaml"
        config = _read_yaml(source_dir / "resolved_config.yaml")
        if not config:
            config_name = ".hydra/config.yaml"
            config = _read_yaml(source_dir / ".hydra" / "config.yaml")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    model_args = (
        model.get("model_args") if isinstance(model.get("model_args"), dict) else {}
    )
    trainer = config.get("trainer") if isinstance(config.get("trainer"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    comparison = _comparison_metadata(source_dir, config)
    if history_link:
        history = history_link[1]
        comparison.update(
            {
                "scope": {
                    key: history[key]
                    for key in (
                        "step",
                        "reference",
                        "evaluation_scope",
                        "evaluated_request_indices",
                    )
                },
                "request_count": len(history["evaluated_request_indices"]),
                "budget": {
                    "edits_applied": history["step"],
                    **(comparison.get("budget") or {}),
                },
                # A completed parent does not turn a prefix into its final benchmark.
                "complete_benchmark": _read_json(run_dir / "input_manifest.json").get(
                    "complete_benchmark"
                ),
            }
        )
    return {
        "model": model_args.get("pretrained_model_name_or_path")
        or model.get("name")
        or config.get("model_name"),
        "trainer": trainer.get("_target_")
        or trainer.get("handler")
        or trainer.get("method")
        or trainer.get("name"),
        "dataset": sorted(
            f"{role}/{name}"
            for role, datasets in data.items()
            if isinstance(datasets, dict)
            for name in datasets
        )
        or None,
        "task_name": run_dir.name
        if history_link
        else config.get("task_name") or run_dir.name,
        "seed": (
            trainer.get("args", {}).get("seed")
            if isinstance(trainer.get("args"), dict)
            else None
        ),
        "comparison": comparison,
        **(
            {
                "history": history_link[1],
                "metadata_source": (
                    ("../../" if source_dir != run_dir else "") + config_name
                )
                if config
                else None,
            }
            if history_link
            else {}
        ),
    }


def _fingerprint(value: Any) -> str | None:
    if not value:
        return None
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    # An unresolved configuration cannot identify the inputs actually used.
    if "${" in encoded or "???" in encoded:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _configured_files(value: Any) -> list[str]:
    """Local data inputs explicitly named in a data or evaluator configuration."""
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            if key in {
                "data_path",
                "edit_data_path",
                "eval_data_path",
                "benchmark_data_path",
            }:
                if isinstance(item, str) and item:
                    paths.append(item)
            elif key == "data_files":
                entries = (
                    item.values()
                    if isinstance(item, dict)
                    else item
                    if isinstance(item, list)
                    else [item]
                )
                paths.extend(
                    entry for entry in entries if isinstance(entry, str) and entry
                )
            else:
                paths.extend(_configured_files(item))
        return paths
    if isinstance(value, list):
        return [path for item in value for path in _configured_files(item)]
    return []


def _data_identity(
    manifest: dict[str, Any], data: dict[str, Any], evaluation: Any
) -> dict[str, Any] | None:
    """Use recorded input hashes, never the current contents of a mutable source."""
    files: list[dict[str, Any]] = []
    for group in ("data_files", "image_files", "evaluation_files"):
        records = manifest.get(group, []) or []
        if not isinstance(records, list):
            return None
        for item in records:
            if not isinstance(item, dict):
                return None
            files.append(
                {"role": group, "path": item.get("path"), "sha256": item.get("sha256")}
            )
    datasets = manifest.get("datasets", []) or []
    if not isinstance(datasets, list):
        return None
    for dataset in datasets:
        if not isinstance(dataset, dict):
            return None
        role = dataset.get("split") or dataset.get("handler") or dataset.get("dataset")
        items = dataset.get("files")
        if items is None:
            items = [
                {"path": dataset.get("data_path"), "sha256": dataset.get("sha256")}
            ]
        # A source with no recorded hash must not disappear from the check.
        if not isinstance(items, list) or not items:
            return None
        for item in items:
            if not isinstance(item, dict):
                return None
            files.append(
                {"role": role, "path": item.get("path"), "sha256": item.get("sha256")}
            )
    if (
        not files
        or not (manifest.get("data_files") or manifest.get("datasets"))
        or not _fingerprint(data)
    ):
        return None
    for item in files:
        digest = item["sha256"]
        if (
            not item["role"]
            or not item["path"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            return None
        item["sha256"] = digest.lower()
    recorded_paths = {Path(item["path"]).as_posix() for item in files}
    for raw in _configured_files(data) + _configured_files(evaluation):
        path = Path(raw).as_posix()
        if not any(
            recorded == path
            or (not Path(path).is_absolute() and recorded.endswith("/" + path))
            for recorded in recorded_paths
        ):
            return None
    identity = {
        "configuration_sha256": _fingerprint(data),
        "files": sorted(files, key=lambda item: json.dumps(item, sort_keys=True)),
    }
    if "request_preview_sha256" in manifest:
        digest = manifest["request_preview_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            return None
        identity["request_preview_sha256"] = digest.lower()
    return identity


def _evaluation_settings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _evaluation_settings(item)
            for key, item in value.items()
            if key not in {"output_dir", "overwrite"}
        }
    if isinstance(value, list):
        return [_evaluation_settings(item) for item in value]
    return value


def _recorded_sha256(value: Any) -> str | None:
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    ):
        return value.lower()
    return None


def _unlearning_settings(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Recover explicit settings from the saved target-NLL evaluator format."""
    boundary = payload.get("boundary_contract")
    if not isinstance(boundary, dict) or not all(
        isinstance(boundary.get(key), str) and boundary[key]
        for key in (
            "version",
            "prompt_encoding",
            "target_encoding",
            "concatenation",
            "labels",
            "truncation",
        )
    ):
        return None
    if type(payload.get("max_length")) is not int or payload["max_length"] <= 0:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    recorded = {}
    for role in ("forget", "retain"):
        item = data.get(role)
        if not isinstance(item, dict):
            return None
        digest = _recorded_sha256(item.get("sha256"))
        count = item.get("records")
        if digest is None or type(count) is not int or count <= 0:
            return None
        recorded[role] = {"sha256": digest, "records": count}
    return {
        "boundary_contract": boundary,
        "max_length": payload["max_length"],
        "data": recorded,
    }


def _comparison_metadata(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Align with Skill comparison fields, adding data-role, evaluator and budget checks.

    These are checks of recorded conditions, not proof of method fidelity or of
    benchmark validity. Older runs without these records remain exploratory.
    """
    trace = _read_json(run_dir / "run_trace.json")
    manifest = _read_json(run_dir / "input_manifest.json")
    environment = _read_json(run_dir / "environment.json")
    unlearning = _read_json(run_dir / "unlearning_metrics.json")
    saved_evaluation = _unlearning_settings(unlearning)
    model_identity = unlearning.get("base_model_identity")
    boundary = unlearning.get("boundary_contract")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    model_args = (
        model.get("model_args") if isinstance(model.get("model_args"), dict) else {}
    )
    trainer = config.get("trainer") if isinstance(config.get("trainer"), dict) else {}
    args = trainer.get("args") if isinstance(trainer.get("args"), dict) else {}
    method_args = (
        trainer.get("method_args")
        if isinstance(trainer.get("method_args"), dict)
        else {}
    )
    data = config.get("data") or {}
    budget = trace.get("budget")
    if not isinstance(budget, dict) or not budget:
        limits = {
            key: args[key]
            for key in ("max_steps", "num_train_epochs")
            if type(args.get(key)) in (int, float) and args[key] > 0
        }
        if (
            type(method_args.get("num_steps")) in (int, float)
            and method_args["num_steps"] > 0
        ):
            limits["method_num_steps"] = method_args["num_steps"]
        budget = (
            {
                **limits,
                **{
                    key: args[key]
                    for key in (
                        "per_device_train_batch_size",
                        "gradient_accumulation_steps",
                    )
                    if key in args
                },
            }
            if limits
            else None
        )
    scope = trace.get("scope") or manifest.get("used_samples")
    count = manifest.get("request_count")
    if count is None and isinstance(manifest.get("used_samples"), dict):
        count = manifest["used_samples"].get("train")
    if not scope and type(count) is int and count > 0:
        scope = {"request_count": count}
    gpu = environment.get("gpu_name") or environment.get("cuda_devices")
    if not gpu and environment.get("cuda_available") is False:
        gpu = "cpu"
    return {
        "model": trace.get("model_id")
        or model_args.get("pretrained_model_name_or_path")
        or model.get("model_path"),
        "model_revision": trace.get("model_revision")
        or environment.get("model_revision")
        or model_args.get("revision")
        or (
            _recorded_sha256(model_identity.get("identity_sha256"))
            if isinstance(model_identity, dict)
            else None
        ),
        "dataset": _data_identity(manifest, data, config.get("eval"))
        if isinstance(data, dict)
        else None,
        "scope": scope,
        "request_count": count if type(count) is int and count > 0 else None,
        "protocol": trace.get("protocol")
        or (boundary.get("version") if isinstance(boundary, dict) else None),
        "evaluation": _fingerprint(
            {
                "configured": _evaluation_settings(config.get("eval")),
                "saved": saved_evaluation,
            }
            if saved_evaluation
            else _evaluation_settings(config.get("eval"))
        ),
        "budget": budget,
        "seed": args.get("seed"),
        "environment": {
            "device": gpu,
            "torch": environment.get("torch"),
            "transformers": environment.get("transformers"),
        },
        "complete_benchmark": manifest.get("complete_benchmark"),
    }


def _checkpoint_eval_dir(run_dir: Path) -> Path | None:
    """Select only evaluations matching the saved final Trainer state.

    Directory names or modification times alone cannot distinguish an interim
    checkpoint from the model saved at the run root.
    """
    step = _read_json(run_dir / "trainer_state.json").get("global_step")
    if type(step) is not int or step < 0:
        return None
    path = run_dir / f"checkpoint-{step}" / "evals"
    return (
        path
        if path.is_dir() and path.resolve().is_relative_to(run_dir.resolve())
        else None
    )


def _summary_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("*_SUMMARY.json"))
    checkpoint_eval = _checkpoint_eval_dir(run_dir)
    if checkpoint_eval is not None:
        paths.extend(sorted(checkpoint_eval.glob("*_SUMMARY.json")))
    return [path for path in paths if path.resolve().is_relative_to(run_dir.resolve())]


METRIC_CONTEXT_FIELDS = (
    "protocol_version",
    "scoring",
    "measurement_kind",
    "denominator",
    "numerator",
    "unavailable_reason",
)


def _metric_identity(source: str, key: str, context: dict[str, Any]) -> dict[str, str]:
    path = Path(source)
    evaluator = path.stem.removesuffix("_SUMMARY").removesuffix("_EVAL")
    location = path.parent.as_posix() if path.parent != Path(".") else "run"
    phase = context.get("phase")
    if not isinstance(phase, str) or not phase:
        phase = location
    identity_phase = phase if phase == location else f"{location}/{phase}"
    return {
        "id": f"{evaluator}:{identity_phase}:{key}",
        "metric_name": key,
        "evaluator": evaluator,
        "phase": phase,
    }


def _metric_context(
    key: str, summary: dict[str, Any], detailed: dict[str, Any]
) -> dict[str, Any]:
    """Recover metadata from the matching evaluator, never another summary file."""
    context = {**detailed, **summary}
    leaf = detailed.get(key)
    if not isinstance(leaf, dict):
        group = "comprehensive" if key == "overall_score" else "long_form"
        candidate = detailed.get(group)
        leaf = candidate if isinstance(candidate, dict) and key in candidate else {}
    context.update(leaf)
    structured = summary.get(key)
    if isinstance(structured, dict):
        context.update(structured)
    result = {field: context.get(field) for field in METRIC_CONTEXT_FIELDS}
    result["phase"] = context.get("phase")
    reason = context.get(f"{key}_unavailable_reason")
    if reason is None and key == "overall_score":
        reason = context.get("overall_unavailable_reason")
    if reason is not None:
        result["unavailable_reason"] = reason
    elif detailed.get("protocol_version") == "inject-v2" and key not in {
        "knowledge_retention",
        "task_accuracy",
        "perplexity",
    }:
        result["unavailable_reason"] = None
    examples = context.get("per_example", context.get("examples", []))
    first = (
        next((entry for entry in examples if isinstance(entry, dict)), {})
        if isinstance(examples, list)
        else {}
    )
    if result["measurement_kind"] is None:
        if "prediction_text" in first or "post_prediction_text" in first:
            result["measurement_kind"] = "teacher_forced_token_prediction"
        elif "generated" in first or "prediction" in first:
            result["measurement_kind"] = "generation"
        elif "value_by_index" in context:
            result["measurement_kind"] = "per_example_score"
    if (
        key == "locality"
        and not result["scoring"]
        and isinstance(examples, list)
        and examples
    ):

        def records_token_agreement(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            before = item.get("pre_prediction_token_ids")
            after = item.get("post_prediction_token_ids")
            if (
                not isinstance(before, list)
                or not before
                or not isinstance(after, list)
            ):
                return False
            if any(type(token) is not int for token in before + after):
                return False
            matches = sum(left == right for left, right in zip(before, after))
            score = item.get("score")
            return (
                type(item.get("matching_tokens")) is int
                and item["matching_tokens"] == matches
                and type(item.get("total_pre_edit_tokens")) is int
                and item["total_pre_edit_tokens"] == len(before)
                and type(score) in (int, float)
                and math.isclose(
                    score, matches / len(before), rel_tol=1e-6, abs_tol=1e-8
                )
            )

        if all(records_token_agreement(item) for item in examples):
            result["scoring"] = "pre_post_token_agreement"
    if result["protocol_version"] == "inject-v2":
        counts = {
            "task_accuracy": ("correct", "total"),
            "current_score": ("current_correct", "current_total"),
            "original_score": ("original_correct", "original_total"),
            "knowledge_retention": ("current_score", "original_score"),
        }
        if key in counts:
            numerator, denominator = counts[key]
            if result["numerator"] is None:
                result["numerator"] = context.get(numerator)
            if result["denominator"] is None:
                result["denominator"] = context.get(denominator)
            if result["measurement_kind"] is None:
                result["measurement_kind"] = (
                    "score_ratio"
                    if key == "knowledge_retention"
                    else "generation_accuracy"
                )
    return {
        field: None if isinstance(value, float) and not math.isfinite(value) else value
        for field, value in result.items()
    }


def _unlearning_metric_context(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Read schema-v2 NLL semantics only when its saved sums reproduce the mean."""
    context = {**payload, "measurement_kind": "target_token_nll"}
    settings = _unlearning_settings(payload)
    if payload.get("schema_version") != 2 or settings is None:
        return context
    split = "retain" if "retain" in key else "forget"
    values = {}
    for phase in ("base", "edited"):
        groups = payload.get(phase)
        entry = groups.get(split) if isinstance(groups, dict) else None
        if not isinstance(entry, dict):
            return context
        total, tokens, mean = (
            entry.get("target_nll_sum"),
            entry.get("target_tokens"),
            entry.get("mean_target_nll"),
        )
        if (
            type(total) not in (int, float)
            or not math.isfinite(total)
            or type(tokens) is not int
            or tokens <= 0
            or type(mean) not in (int, float)
            or not math.isfinite(mean)
            or not math.isclose(total / tokens, mean, rel_tol=1e-6, abs_tol=1e-8)
        ):
            return context
        values[phase] = entry
    change = key.endswith("_change")
    phase = "base" if key.startswith("base_") else "edited"
    expected = (
        values["edited"]["mean_target_nll"] - values["base"]["mean_target_nll"]
        if change
        else values[phase]["mean_target_nll"]
    )
    recorded = payload.get(key)
    if type(recorded) not in (int, float) or not math.isclose(
        recorded, expected, rel_tol=1e-6, abs_tol=1e-8
    ):
        return context
    if not context.get("protocol_version"):
        context["protocol_version"] = settings["boundary_contract"]["version"]
    if not context.get("scoring"):
        context["scoring"] = (
            "edited minus base target-token-weighted mean NLL"
            if change
            else "target_nll_sum / target_tokens"
        )
    if not change:
        context.setdefault("numerator", values[phase]["target_nll_sum"])
        context.setdefault("denominator", values[phase]["target_tokens"])
    return context


def _scalar_metrics(run_dir: Path) -> list[dict[str, Any]]:
    rows = []

    def append(key, value, source, context=None):
        context = context or {}
        if type(value) not in (int, float) or not math.isfinite(value):
            value = None
        definition = dict(
            METRICS.get(
                key,
                {
                    "label": {"en": key.replace("_", " ").title(), "zh": key},
                    "direction": "context",
                    "category": "diagnostic",
                    "unit": "value",
                },
            )
        )
        if (
            key == "knowledge_retention"
            and context.get("protocol_version") == "inject-v2"
        ):
            definition["unit"] = "multiplier"
        rows.append(
            {
                **_metric_identity(source, key, context),
                "value": value,
                "source": source,
                **definition,
                **{field: context.get(field) for field in METRIC_CONTEXT_FIELDS},
            }
        )

    for path in _summary_paths(run_dir):
        summary = _read_json(path)
        detail_path = path.with_name(path.name.replace("_SUMMARY.json", "_EVAL.json"))
        detailed = (
            _read_json(detail_path)
            if detail_path.resolve().is_relative_to(run_dir.resolve())
            else {}
        )
        for key, item in summary.items():
            if (
                key in METRIC_CONTEXT_FIELDS
                or key in {"phase", "evaluator"}
                or key.endswith("_unavailable_reason")
            ):
                continue
            if isinstance(item, dict):
                if "value" not in item and "agg_value" not in item:
                    continue
                value = item.get("value", item.get("agg_value"))
            else:
                value = item
            if type(value) in (int, float) or value is None:
                append(
                    key,
                    value,
                    path.relative_to(run_dir).as_posix(),
                    _metric_context(key, summary, detailed),
                )
    unlearning = _read_json(run_dir / "unlearning_metrics.json")
    for key in (
        "forget_nll_change",
        "retain_nll_change",
        "base_forget_nll",
        "edited_forget_nll",
        "base_retain_nll",
        "edited_retain_nll",
    ):
        if key in unlearning:
            append(
                key,
                unlearning[key],
                "unlearning_metrics.json",
                _unlearning_metric_context(unlearning, key),
            )
    target_nll = _read_json(run_dir / "target_nll.json")
    if target_nll:
        for key, value in (
            (
                "base_target_nll",
                target_nll.get("base", {}).get("weighted_target_token_nll"),
            ),
            (
                "post_target_nll",
                target_nll.get("post", {}).get("weighted_target_token_nll"),
            ),
            ("target_nll_delta", target_nll.get("delta_post_minus_base")),
        ):
            append(
                key,
                value,
                "target_nll.json",
                {**target_nll, "measurement_kind": "target_token_nll"},
            )
    trace = _read_json(run_dir / "run_trace.json")
    if trace.get("runtime_seconds") is not None:
        append("runtime_seconds", trace["runtime_seconds"], "run_trace.json")
    peak = trace.get("peak_cuda_memory")
    if isinstance(peak, dict):
        allocated = [
            value.get("max_memory_allocated_bytes")
            for value in peak.values()
            if isinstance(value, dict)
            and type(value.get("max_memory_allocated_bytes")) in (int, float)
            and math.isfinite(value["max_memory_allocated_bytes"])
        ]
        if allocated:
            append("peak_memory_gib", max(allocated) / (1024**3), "run_trace.json")
    order = {"efficacy": 0, "side_effect": 1, "efficiency": 2, "diagnostic": 3}
    return sorted(rows, key=lambda row: (order.get(row["category"], 9), row["id"]))


def _has_evaluation(metrics: list[dict[str, Any]]) -> bool:
    """Runtime, counters, unavailable values and pre-update diagnostics are not evaluation."""
    counters = {
        "count",
        "skipped",
        "protocol_version",
        "num_samples",
        "num_cases",
        "request_count",
        "requested_count",
        "successful_cases",
        "num_examples",
        "runtime_seconds",
        "peak_memory_gib",
        "total",
        "correct",
        "current_total",
        "current_correct",
        "base_total",
        "base_correct",
        "total_tokens",
        "target_tokens",
        "prompt_tokens",
        "input_tokens",
        "output_tokens",
    }
    post_metrics = {
        "post_target_nll",
        "target_nll_delta",
        "forget_nll_change",
        "retain_nll_change",
        "edited_forget_nll",
        "edited_retain_nll",
    }
    return any(
        type(metric.get("value")) in (int, float)
        and math.isfinite(metric["value"])
        and metric["category"] != "efficiency"
        and metric.get("metric_name", metric["id"]).rsplit(".", 1)[-1] not in counters
        and not metric.get("metric_name", metric["id"])
        .rsplit(".", 1)[-1]
        .startswith(("num_", "base_", "baseline_", "before_", "pre_", "original_"))
        and not metric.get("metric_name", metric["id"])
        .rsplit(".", 1)[-1]
        .endswith(("_count", "_tokens", "_total", "_correct"))
        and (
            metric["source"].endswith("_SUMMARY.json")
            or metric.get("metric_name", metric["id"]) in post_metrics
        )
        for metric in metrics
    )


def _run_status(run_dir: Path) -> str:
    """Explicit failure artifacts override a stale or contradictory success trace."""
    trace = _read_json(run_dir / "run_trace.json")
    for name in (
        "edit_results.json",
        "independent_results.json",
        "independent_progress.json",
        "run_manifest.json",
    ):
        artifact = _read_json(run_dir / name)
        if (
            artifact.get("success") is False
            or artifact.get("edit_success") is False
            or artifact.get("status") in {"failed", "error", "interrupted", "stopped"}
        ):
            return "failed"
    return trace.get("status") or (
        "completed" if _history_link(run_dir) else "configured"
    )


def run_evidence(run_dir: Path, metrics: list[dict[str, Any]] | None = None) -> str:
    """Classify saved evidence consistently for job and result views."""
    trace = _read_json(run_dir / "run_trace.json")
    metrics = _scalar_metrics(run_dir) if metrics is None else metrics
    if _run_status(run_dir) not in {"success", "completed"} or not _has_evaluation(
        metrics
    ):
        return "configured"
    unlearning = _read_json(run_dir / "unlearning_metrics.json")
    explicit = trace.get("evidence_level") or unlearning.get("status")
    if explicit in {"verified_smoke", "verified_result"}:
        # Reuse the Skill runtime's nonempty artifacts, successful edit and
        # positive sample-count checks. This import does not load models.
        from scripts.skill_runtime import _has_smoke_evidence

        if _has_smoke_evidence(run_dir):
            if explicit == "verified_smoke":
                return explicit
            if (
                _read_json(run_dir / "input_manifest.json").get("complete_benchmark")
                is True
            ):
                return explicit
    return "local_validation"


def _timeline(run_dir: Path, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status = _run_status(run_dir)
    execution_success = status in {"success", "completed"}
    history_link = _history_link(run_dir)
    metadata_dir = history_link[0] if history_link else run_dir
    return [
        {
            "id": "input",
            "label": {"en": "Input manifest", "zh": "输入清单"},
            "status": "completed"
            if (metadata_dir / "input_manifest.json").is_file()
            else "missing",
        },
        {
            "id": "compose",
            "label": {"en": "Resolved config", "zh": "配置解析"},
            "status": "completed"
            if (metadata_dir / "resolved_config.yaml").is_file()
            or (metadata_dir / ".hydra" / "config.yaml").is_file()
            else "missing",
        },
        {
            "id": "execute",
            "label": {"en": "Model surgery", "zh": "模型手术"},
            "status": "completed"
            if execution_success
            else "failed"
            if status in {"failed", "error", "interrupted", "stopped"}
            else "unknown",
        },
        {
            "id": "evaluate",
            "label": {"en": "Evaluation", "zh": "评估"},
            "status": "completed"
            if execution_success and _has_evaluation(metrics)
            else "missing",
        },
    ]


class ResultStore:
    """Read-only result index rooted at repository saves."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)

    def _map(self) -> dict[str, Path]:
        values = {}
        for run_dir in _candidate_dirs(self.root):
            relative = _relative(run_dir, self.root)
            values[_result_id(relative)] = run_dir
        return values

    def detail_for_path(self, run_dir: Path) -> dict[str, Any]:
        relative = _relative(run_dir, self.root)
        metrics = _scalar_metrics(run_dir)
        metadata = _resolved_metadata(run_dir)
        history = metadata.get("history")
        name = run_dir.name
        if history:
            method = metadata.get("trainer") or history["parent_run"]
            name = f"{str(method).removesuffix('Editor')} · step {history['step']}"
        trace = _read_json(run_dir / "run_trace.json")
        unlearning = _read_json(run_dir / "unlearning_metrics.json")
        files = sorted(
            _relative(path, run_dir)
            for path in run_dir.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".txt", ".log"}
        )
        checkpoint_eval = _checkpoint_eval_dir(run_dir)
        if checkpoint_eval is not None:
            files.extend(
                path.relative_to(run_dir).as_posix()
                for path in sorted(checkpoint_eval.glob("*.json"))
                if path.is_file()
            )
        warnings = []
        if not _has_evaluation(metrics):
            warnings.append("This run has no completed evaluation metrics.")
        if any(
            _has_evaluation([metric]) and not metric.get("protocol_version")
            for metric in metrics
        ):
            warnings.append("Some metrics do not record protocol or scoring metadata.")
        return {
            "id": _result_id(relative),
            "name": name,
            "path": relative,
            "operation": _operation(relative),
            "evidence_level": run_evidence(run_dir, metrics),
            "status": _run_status(run_dir),
            "updated_at": run_dir.stat().st_mtime,
            "metadata": {
                **metadata,
                "scope": history["evaluation_scope"]
                if history
                else unlearning.get("scope"),
                "finished_at": trace.get("finished_at_utc"),
            },
            "metrics": metrics,
            "comparison_basis": COMPARISON_BASIS,
            "warnings": warnings,
            "timeline": _timeline(run_dir, metrics),
            "files": files,
            "command": (
                (run_dir / "command.txt").read_text(encoding="utf-8").strip()
                if (run_dir / "command.txt").is_file()
                else None
            ),
        }

    def list(self, *, limit: int = 200) -> list[dict[str, Any]]:
        rows = []
        for run_dir in _candidate_dirs(self.root)[:limit]:
            detail = self.detail_for_path(run_dir)
            rows.append(
                {
                    key: detail[key]
                    for key in (
                        "id",
                        "name",
                        "path",
                        "operation",
                        "evidence_level",
                        "status",
                        "updated_at",
                        "metadata",
                        "metrics",
                        "comparison_basis",
                        "warnings",
                    )
                }
            )
        return rows

    def get(self, result_id: str) -> dict[str, Any]:
        run_dir = self._map().get(result_id)
        if run_dir is None:
            raise KeyError(result_id)
        return self.detail_for_path(run_dir)

    def path(self, result_id: str) -> Path:
        run_dir = self._map().get(result_id)
        if run_dir is None:
            raise KeyError(result_id)
        return run_dir

    def metadata(self, result_id: str) -> dict[str, Any]:
        """Return recorded result and metric context without loading example text."""
        result = self.get(result_id)
        return {
            "result_id": result_id,
            **{
                key: result[key]
                for key in (
                    "operation",
                    "status",
                    "evidence_level",
                    "metadata",
                    "metrics",
                    "comparison_basis",
                    "warnings",
                )
            },
        }

    def cases(
        self,
        result_id: str,
        *,
        offset: int = 0,
        limit: int = 50,
        metric_id: str | None = None,
    ) -> dict[str, Any]:
        """Page through saved evaluator examples; never use targets as responses."""
        if (
            type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError(
                "offset must be nonnegative and limit must be between 1 and 100"
            )
        from .cases import read_cases

        run_dir = self.path(result_id)
        rows, warnings = read_cases(
            run_dir,
            _scalar_metrics(run_dir),
            checkpoint_eval=_checkpoint_eval_dir(run_dir),
        )
        if metric_id is not None:
            rows = [row for row in rows if row["metric_id"] == metric_id]
        return {
            "result_id": result_id,
            "offset": offset,
            "limit": limit,
            "total": len(rows),
            "has_more": offset + limit < len(rows),
            "cases": rows[offset : offset + limit],
            "warnings": warnings,
        }

    def demos(self) -> list[dict[str, Any]]:
        rows = []
        for demo in CURATED_DEMOS:
            path = self.root / demo["path"]
            if not path.is_dir():
                continue
            detail = self.detail_for_path(path)
            related = demo.get("related")
            if related and (self.root / related).is_dir():
                related_detail = self.detail_for_path(self.root / related)
                existing = {item["id"] for item in detail["metrics"]}
                detail["metrics"].extend(
                    item
                    for item in related_detail["metrics"]
                    if item["id"] not in existing
                )
            rows.append({**demo, "result": detail})
        return rows

    def compare(self, result_ids: list[str]) -> dict[str, Any]:
        results = [self.get(result_id) for result_id in result_ids]
        fairness_fields = (
            "operation",
            "model",
            "model_revision",
            "dataset",
            "scope",
            "request_count",
            "protocol",
            "evaluation",
            "budget",
            "seed",
            "environment",
            "complete_benchmark",
            "evidence_level",
        )
        fairness = {}
        for field in fairness_fields:
            if field in {"operation", "evidence_level"}:
                values = [result.get(field) for result in results]
            else:
                values = [
                    result["metadata"]["comparison"].get(field) for result in results
                ]
            fairness[field] = {
                "values": values,
                "known": all(_known(value) for value in values),
                "equal": len({json.dumps(value, sort_keys=True) for value in values})
                == 1,
            }
        metric_sets = [
            {metric["id"] for metric in result["metrics"]} for result in results
        ]
        common = sorted(set.intersection(*metric_sets)) if metric_sets else []
        quality_sets = [
            {metric["id"] for metric in result["metrics"] if _has_evaluation([metric])}
            for result in results
        ]
        common_quality = set.intersection(*quality_sets) if quality_sets else set()
        metric_alignment = {}
        for metric_id in sorted(common_quality):
            contexts = [
                {
                    field: metric.get(field)
                    for field in (
                        "protocol_version",
                        "scoring",
                        "measurement_kind",
                        "unit",
                    )
                }
                for result in results
                for metric in result["metrics"]
                if metric["id"] == metric_id
            ]
            metric_alignment[metric_id] = {
                "values": contexts,
                "known": all(_known(context) for context in contexts),
                "equal": len(
                    {json.dumps(context, sort_keys=True) for context in contexts}
                )
                == 1,
            }
        comparable = (
            len(set(result_ids)) >= 2
            and bool(common_quality)
            and all(
                result["status"] in {"success", "completed"}
                and _has_evaluation(result["metrics"])
                for result in results
            )
            and all(item["known"] and item["equal"] for item in fairness.values())
            and all(
                item["known"] and item["equal"] for item in metric_alignment.values()
            )
        )
        return {
            "comparable": comparable,
            "fairness": fairness,
            "metric_alignment": metric_alignment,
            "warnings": []
            if comparable
            else [
                "Check that both runs have completed evaluation, share quality metrics, and use matching comparison settings.",
                *(
                    ["Metric protocols or scoring semantics are missing or different."]
                    if any(
                        not item["known"] or not item["equal"]
                        for item in metric_alignment.values()
                    )
                    else []
                ),
            ],
            "comparison_basis": COMPARISON_BASIS,
            "metric_ids": sorted(set.union(*metric_sets)) if metric_sets else [],
            "common_metric_ids": common,
            "results": results,
        }


def _known(value: Any) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return False
    if isinstance(value, dict):
        return all(_known(item) for item in value.values())
    if isinstance(value, list):
        return all(_known(item) for item in value)
    if isinstance(value, str) and ("${" in value or "???" in value):
        return False
    return not isinstance(value, float) or math.isfinite(value)
