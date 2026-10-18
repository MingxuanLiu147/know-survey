"""Evidence-aware parsing for existing and newly produced run artifacts."""

from __future__ import annotations

import hashlib
import json
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
    )
    for pattern in patterns:
        directories.update(path.parent for path in saves.rglob(pattern))
    return sorted(
        directories,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _resolved_metadata(run_dir: Path) -> dict[str, Any]:
    config = _read_yaml(run_dir / "resolved_config.yaml")
    if not config:
        config = _read_yaml(run_dir / ".hydra" / "config.yaml")
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    model_args = (
        model.get("model_args") if isinstance(model.get("model_args"), dict) else {}
    )
    trainer = config.get("trainer") if isinstance(config.get("trainer"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    return {
        "model": model_args.get("pretrained_model_name_or_path")
        or model.get("name")
        or config.get("model_name"),
        "trainer": trainer.get("_target_")
        or trainer.get("method")
        or trainer.get("name"),
        "dataset": sorted(data.keys()) if data else None,
        "task_name": config.get("task_name") or run_dir.name,
        "seed": (
            trainer.get("args", {}).get("seed")
            if isinstance(trainer.get("args"), dict)
            else None
        ),
    }


def _scalar_metrics(run_dir: Path) -> list[dict[str, Any]]:
    values: dict[str, tuple[Any, str]] = {}
    for path in sorted(run_dir.glob("*_SUMMARY.json")):
        for key, value in _read_json(path).items():
            if isinstance(value, (int, float)) or value is None:
                values[key] = (value, path.name)
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
            values[key] = (unlearning.get(key), "unlearning_metrics.json")
    target_nll = _read_json(run_dir / "target_nll.json")
    if target_nll:
        values["base_target_nll"] = (
            target_nll.get("base", {}).get("weighted_target_token_nll"),
            "target_nll.json",
        )
        values["post_target_nll"] = (
            target_nll.get("post", {}).get("weighted_target_token_nll"),
            "target_nll.json",
        )
        values["target_nll_delta"] = (
            target_nll.get("delta_post_minus_base"),
            "target_nll.json",
        )
    trace = _read_json(run_dir / "run_trace.json")
    if trace.get("runtime_seconds") is not None:
        values["runtime_seconds"] = (
            trace["runtime_seconds"],
            "run_trace.json",
        )
    peak = trace.get("peak_cuda_memory")
    if isinstance(peak, dict):
        allocated = [
            value.get("max_memory_allocated_bytes")
            for value in peak.values()
            if isinstance(value, dict)
            and isinstance(value.get("max_memory_allocated_bytes"), (int, float))
        ]
        if allocated:
            values["peak_memory_gib"] = (
                max(allocated) / (1024**3),
                "run_trace.json",
            )
    rows = []
    for key, (value, source) in values.items():
        definition = METRICS.get(
            key,
            {
                "label": {"en": key.replace("_", " ").title(), "zh": key},
                "direction": "context",
                "category": "diagnostic",
                "unit": "value",
            },
        )
        rows.append({"id": key, "value": value, "source": source, **definition})
    category_order = {
        "efficacy": 0,
        "side_effect": 1,
        "efficiency": 2,
        "diagnostic": 3,
    }
    return sorted(
        rows,
        key=lambda row: (category_order.get(row["category"], 9), row["id"]),
    )


def _evidence(run_dir: Path, metrics: list[dict[str, Any]]) -> str:
    unlearning = _read_json(run_dir / "unlearning_metrics.json")
    explicit = unlearning.get("status")
    if explicit in {
        "planned",
        "configured",
        "local_validation",
        "verified_smoke",
        "verified_result",
    }:
        return explicit
    has_contract = all(
        (run_dir / name).is_file()
        for name in ("run_trace.json", "resolved_config.yaml", "input_manifest.json")
    )
    if metrics and has_contract and "smoke" in run_dir.name.lower():
        return "verified_smoke"
    if metrics and (run_dir / "run_trace.json").is_file():
        return "local_validation"
    if metrics:
        return "local_validation"
    return "configured"


def _timeline(run_dir: Path, metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace = _read_json(run_dir / "run_trace.json")
    execution_success = trace.get("status") == "success"
    return [
        {
            "id": "input",
            "label": {"en": "Input manifest", "zh": "输入清单"},
            "status": "completed"
            if (run_dir / "input_manifest.json").is_file()
            else "missing",
        },
        {
            "id": "compose",
            "label": {"en": "Resolved config", "zh": "配置解析"},
            "status": "completed"
            if (run_dir / "resolved_config.yaml").is_file()
            or (run_dir / ".hydra" / "config.yaml").is_file()
            else "missing",
        },
        {
            "id": "execute",
            "label": {"en": "Model surgery", "zh": "模型手术"},
            "status": "completed"
            if execution_success
            else "failed"
            if trace
            else "unknown",
        },
        {
            "id": "evaluate",
            "label": {"en": "Evaluation", "zh": "评估"},
            "status": "completed" if metrics else "missing",
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
        trace = _read_json(run_dir / "run_trace.json")
        unlearning = _read_json(run_dir / "unlearning_metrics.json")
        files = sorted(
            _relative(path, run_dir)
            for path in run_dir.iterdir()
            if path.is_file() and path.suffix in {".json", ".yaml", ".txt", ".log"}
        )
        return {
            "id": _result_id(relative),
            "name": run_dir.name,
            "path": relative,
            "operation": _operation(relative),
            "evidence_level": _evidence(run_dir, metrics),
            "status": trace.get("status") or ("evaluated" if metrics else "configured"),
            "updated_at": run_dir.stat().st_mtime,
            "metadata": {
                **_resolved_metadata(run_dir),
                "scope": unlearning.get("scope"),
                "finished_at": trace.get("finished_at_utc"),
            },
            "metrics": metrics,
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
        fairness_fields = ("operation", "model", "dataset", "scope", "evidence_level")
        fairness = {}
        for field in fairness_fields:
            if field in {"operation", "evidence_level"}:
                values = [result.get(field) for result in results]
            else:
                values = [result["metadata"].get(field) for result in results]
            fairness[field] = {
                "values": values,
                "known": all(value not in (None, "", []) for value in values),
                "equal": len({json.dumps(value, sort_keys=True) for value in values})
                == 1,
            }
        comparable = all(item["known"] and item["equal"] for item in fairness.values())
        metric_sets = [
            {metric["id"] for metric in result["metrics"]} for result in results
        ]
        common = sorted(set.intersection(*metric_sets)) if metric_sets else []
        return {
            "comparable": comparable,
            "fairness": fairness,
            "warnings": []
            if comparable
            else [
                "Exploratory view only: one or more fairness fields are missing or differ."
            ],
            "metric_ids": common,
            "results": results,
        }
