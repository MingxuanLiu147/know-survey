#!/usr/bin/env python3
"""Deterministic thin runtime shared by the repository Agent Skills."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
OPERATIONS = {"unlearning", "inject", "edit"}
EVIDENCE_LEVELS = {
    "planned",
    "configured",
    "local_validation",
    "verified_smoke",
    "verified_result",
}
KNOWN_METHODS = (
    "GradAscent",
    "GradDiff",
    "NPO",
    "SimNPO",
    "RMU",
    "PDU",
    "LoRA",
    "DoRA",
    "AdaLoRA",
    "LoReFT",
    "FT",
    "ROME",
    "MEMIT",
    "MEND",
    "GRACE",
    "WISE",
    "IKE",
    "SERAC",
    "AlphaEdit",
    "UNKE",
    "AnyEdit",
)

LONG_TEXT_MARKERS = (
    "long text",
    "long-form",
    "long form",
    "unstructured",
    "free text",
    "长文本",
    "非结构化",
    "自由文本",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(
    argv: list[str], *, timeout: int, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = _utc()
    try:
        proc = subprocess.run(
            argv,
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except OSError as exc:
        proc = subprocess.CompletedProcess(argv, 127, stdout="", stderr=str(exc))
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        proc = subprocess.CompletedProcess(
            argv, 124, stdout=stdout, stderr=f"{stderr}\nTimed out after {timeout}s."
        )
    return {
        "started_at": started,
        "finished_at": _utc(),
        "argv": argv,
        "command": shlex.join(argv),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "passed" if proc.returncode == 0 else "failed",
    }


def _operation_from_intent(text: str) -> str:
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "rome",
            "memit",
            "counterfact",
            "replace",
            "update",
            "edit",
            "修改",
            "替换",
            "编辑",
        )
    ):
        return "edit"
    if any(
        token in lowered
        for token in (
            "unlearn",
            "forget",
            "remove knowledge",
            "delete knowledge",
            "遗忘",
            "删除知识",
        )
    ):
        return "unlearning"
    if any(
        token in lowered
        for token in (
            "inject",
            "fine-tune",
            "finetune",
            "add knowledge",
            "注入",
            "新增知识",
        )
    ):
        return "inject"
    raise ValueError(
        "Cannot determine operation; specify --operation unlearning|inject|edit"
    )


def _model_from_intent(text: str) -> str:
    lowered = text.lower().replace("_", "-")
    if "deepseek" in lowered and "7b" in lowered:
        return "DeepSeek-R1-Distill-Qwen-7B"
    if re.search(r"(?:qwen(?:2\.5)?|qwen25).{0,12}7b", lowered):
        return "Qwen2.5-7B-Instruct"
    if re.search(r"llama[ -]?2.{0,12}7b", lowered):
        return "Llama-2-7b-chat-hf" if "chat" in lowered else "Llama-2-7b-hf"
    if "gemma" in lowered and "7b" in lowered:
        return "gemma-7b-it"
    if "zephyr" in lowered and "7b" in lowered:
        return "zephyr-7b-beta"
    if "qwen2.5-1.5b" in lowered or "qwen25-1.5b" in lowered:
        return "Qwen2.5-1.5B-Instruct"
    if "qwen2.5-3b" in lowered or "qwen25-3b" in lowered:
        return "Qwen2.5-3B-Instruct"
    if "llama-3.2-1b" in lowered:
        return "Llama-3.2-1B-Instruct"
    if "gpt2" in lowered or "gpt-2" in lowered:
        return "gpt2"
    if re.search(r"\b7b\b", lowered):
        return "Qwen2.5-7B-Instruct"
    return "gpt2"


def _is_long_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LONG_TEXT_MARKERS)


def _dataset_from_intent(text: str, operation: str) -> tuple[str, str]:
    lowered = text.lower()
    if operation == "edit" and "akew" in lowered:
        return "AKEW_counterfact_unstructured_edit", "free_text"
    if operation == "edit" and "leme" in lowered:
        return "LEME_counterfact_edit", "free_text"
    if operation == "edit" and (
        "unke" in lowered or "anyedit" in lowered or _is_long_text(text)
    ):
        return "UnKE_edit", "free_text"
    if "counterfact" in lowered:
        return "CounterFact_edit", "fact"
    if "tofu" in lowered:
        return "TOFU_local_smoke_forget", "fact"
    if "ruletaker" in lowered or "rule" in lowered or "规则" in lowered:
        return "RuleTaker_train_inject", "rule"
    if "wikihow" in lowered or "procedure" in lowered or "过程" in lowered:
        return "WikiHow_train_inject", "procedural"
    defaults = {
        "unlearning": ("TOFU_local_smoke_forget", "fact"),
        "inject": ("Alpaca_inject", "free_text"),
        "edit": ("CounterFact_edit", "fact"),
    }
    return defaults[operation]


def _methods_from_intent(text: str, operation: str) -> list[str]:
    lowered = text.lower()
    found = sorted(
        (
            method
            for method in KNOWN_METHODS
            if re.search(
                r"(?<![a-z])" + re.escape(method.lower()) + r"(?![a-z])", lowered
            )
        ),
        key=lambda method: lowered.index(method.lower()),
    )
    if found:
        return found
    if operation == "inject" and any(
        marker in lowered
        for marker in (
            "representation fine-tuning",
            "representation finetuning",
            "representation tuning",
            "representational fine-tuning",
            "表示微调",
            "表征微调",
        )
    ):
        return ["LoReFT"]
    if _is_long_text(text):
        if operation == "inject":
            return ["LoReFT"]
        if operation == "edit":
            return [
                method
                for method in ("UNKE", "AnyEdit")
                if (ROOT / "configs/trainer/edit" / f"{method}.yaml").is_file()
            ]
    return {"unlearning": ["NPO"], "inject": ["LoRA"], "edit": ["FT"]}[operation]


def _num_cases(text: str) -> int:
    match = re.search(
        r"\b(\d+)\s*(?:edits?|cases?|samples?|records?|facts?|条)\b",
        text,
        flags=re.I,
    )
    if match is None:
        match = re.search(
            r"\b(?:edit|inject|unlearn|forget|update|replace)\s+(\d+)\b",
            text,
            flags=re.I,
        )
    return max(1, int(match.group(1))) if match else 1


def _task_name(plan: dict[str, Any], method: str, *, prefix: str = "skill") -> str:
    explicit = plan.get("overrides", {}).get("task_name")
    if explicit:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", str(explicit)):
            raise ValueError("Invalid task_name in plan overrides.")
        return f"{explicit}_{method.lower()}"
    raw = f"{prefix}_{plan['operation']}_{method}_{plan['num_cases']}".lower()
    return re.sub(r"[^a-z0-9_-]+", "_", raw).strip("_")


def _experiments(
    model: str, dataset: str, methods: Iterable[str], operation: str | None = None
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for method in methods:
        experiment = None
        if (
            dataset == "TOFU_local_smoke_forget"
            and model == "gpt2"
            and method == "GradAscent"
        ):
            experiment = "unlearn/tofu/gpt2_quickstart"
        elif dataset == "CounterFact_edit" and model == "Qwen2.5-1.5B-Instruct":
            if method == "ROME":
                experiment = "edit/counterfact/qwen25_1_5b_rome_100"
            elif method == "MEMIT":
                experiment = "edit/counterfact/qwen25_1_5b_memit_100"
            elif method == "FT":
                experiment = "edit/model_coverage/qwen25_1_5b_ft_counterfact"
        elif dataset == "CounterFact_edit" and model == "gpt2":
            candidates = {
                "FT": "edit/model_coverage/gpt2_ft_counterfact",
                "ROME": "edit/counterfact/gpt2_rome_real",
                "MEMIT": "edit/counterfact/gpt2_memit_real",
                "IKE": "edit/model_coverage/gpt2_ike_counterfact",
                "SERAC": "edit/model_coverage/gpt2_serac_counterfact",
            }
            experiment = candidates.get(method)
        elif dataset == "Alpaca_inject" and model == "Qwen2.5-7B-Instruct":
            candidates = {
                "LoRA": "inject/alpaca/default",
                "DoRA": "inject/alpaca/dora",
                "AdaLoRA": "inject/alpaca/adalora",
                "LoReFT": "inject/alpaca/qwen25_7b_loreft",
            }
            experiment = candidates.get(method)
        elif dataset == "UnKE_edit" and model == "Qwen2.5-7B-Instruct":
            candidates = {
                "UNKE": "edit/unke/unke",
                "AnyEdit": "edit/unke/anyedit",
                "ROME": "edit/unke/rome",
                "MEMIT": "edit/unke/memit",
                "AlphaEdit": "edit/unke/alphaedit",
            }
            experiment = candidates.get(method)
        elif (
            dataset == "AKEW_counterfact_unstructured_edit"
            and model == "Qwen2.5-7B-Instruct"
            and method == "AnyEdit"
        ):
            experiment = "edit/akew/anyedit"
        elif (
            dataset == "LEME_counterfact_edit"
            and model == "Qwen2.5-7B-Instruct"
            and method == "MEMIT"
        ):
            experiment = "edit/leme/memit"
        if experiment is None:
            registry = ROOT / "configs/extensions/recipes.yaml"
            rows = (
                yaml.safe_load(registry.read_text()).get("recipes", [])
                if registry.is_file()
                else []
            )
            for row in rows:
                if (operation is None or row.get("operation") == operation) and (
                    Path(row.get("model", "")).stem == model
                    and Path(row.get("method", "")).stem == method
                    and any(
                        Path(value).stem == dataset
                        for value in row.get("datasets", {}).values()
                    )
                ):
                    experiment = row["experiment"]
                    break
        result[method] = experiment
    return result


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    operation = args.operation or _operation_from_intent(args.intent)
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")
    model = args.model_id or _model_from_intent(args.intent)
    dataset, knowledge_type = _dataset_from_intent(args.intent, operation)
    dataset = args.dataset_id or dataset
    methods = (
        [method.strip() for method in args.methods.split(",")]
        if args.methods
        else _methods_from_intent(args.intent, operation)
    )
    num_cases = (
        args.num_cases if args.num_cases is not None else _num_cases(args.intent)
    )
    if num_cases < 1:
        raise ValueError("num_cases must be positive")
    evaluators = {
        "unlearning": ["forget_efficacy", "retention", "privacy", "utility"],
        "inject": ["task_accuracy", "knowledge_retention"],
        "edit": ["reliability", "generalization", "locality", "portability"],
    }[operation]
    warnings = []
    if args.model_id is None and model == "gpt2" and "gpt" not in args.intent.lower():
        warnings.append(
            "No model was explicit; selected the local GPT-2 quick-start baseline."
        )
    lowered_intent = args.intent.lower()
    if (
        model == "Qwen2.5-7B-Instruct"
        and re.search(r"\b7b\b", lowered_intent)
        and not any(
            family in lowered_intent
            for family in ("qwen", "llama", "deepseek", "gemma", "zephyr")
        )
    ):
        warnings.append(
            "Only the 7B size was explicit; selected Qwen2.5-7B-Instruct because registered long-text recipes target it."
        )
    experiments = _experiments(model, dataset, methods, operation)
    if args.dataset_id:
        registry = ROOT / "configs/extensions/recipes.yaml"
        rows = (
            yaml.safe_load(registry.read_text()).get("recipes", [])
            if registry.is_file()
            else []
        )
        kinds = {
            kind
            for row in rows
            if any(
                Path(value).stem == dataset
                for value in row.get("datasets", {}).values()
            )
            for kind in row.get("knowledge_types", [])
        }
        if len(kinds) == 1:
            knowledge_type = kinds.pop()
    for method, experiment in experiments.items():
        if experiment is None:
            warnings.append(
                f"{method} has no curated exact recipe for this model/dataset; manual compose is required."
            )
    if model == "Qwen2.5-7B-Instruct":
        warnings.append(
            "The selected Qwen2.5-7B route is configured, not benchmark-verified; verify the local model path and GPU memory, then run a bounded validation first."
        )
    if operation == "inject" and "LoReFT" in methods:
        warnings.append(
            "LoReFT requires pyreft and the ReFT data collator; the registered recipe composes these dependencies but still needs a real-model validation run."
        )
    if operation == "edit" and "UNKE" in methods:
        warnings.append(
            "UNKE requires preservation data; this research-only implementation is excluded from public releases pending source permission."
        )
    if operation == "edit" and "AnyEdit" in methods:
        warnings.append(
            "AnyEdit MEMIT/AlphaEdit backends require method-specific statistics when enabled; UnKE_ARE is a separate research-only backend."
        )
    sequence = ["plan-surgery", "inspect-repo"]
    if re.search(
        r"\b(?:new|add|onboard|register)\b.{0,32}\bdataset\b", args.intent, flags=re.I
    ):
        sequence.append("add-dataset")
    sequence.extend(["normalize-knowledge", "build-cases", "smoke-test"])
    sequence.extend(
        {"unlearning": ["run-unlearn"], "inject": ["run-inject"], "edit": ["run-edit"]}[
            operation
        ]
    )
    if len(methods) > 1:
        sequence.extend(["aggregate-results", "compare-methods"])
    sequence.extend(["generate-report", "reproduce-run"])
    payload = {
        "schema_version": "1.0",
        "intent": args.intent,
        "operation": operation,
        "knowledge_type": knowledge_type,
        "model_id": model,
        "dataset_id": dataset,
        "methods": methods,
        "num_cases": num_cases,
        "evaluators": evaluators,
        "experiments": experiments,
        "overrides": {},
        "steps": [
            {
                "skill": name,
                "status": "completed" if name == "plan-surgery" else "ready",
                "reason": "typed intent and registry-driven workflow",
            }
            for name in sequence
        ],
        "warnings": warnings,
        "requires_confirmation": True,
    }
    for kind, ids in (
        ("model", [model]),
        ("data/datasets", [dataset]),
        (
            "trainer",
            [
                (
                    "edit/"
                    if operation == "edit"
                    else "inject/"
                    if operation == "inject"
                    else ""
                )
                + m
                for m in methods
            ],
        ),
    ):
        for component in ids:
            config = ROOT / "configs" / kind / (component + ".yaml")
            if not config.is_file() or not config.resolve().is_relative_to(
                (ROOT / "configs").resolve()
            ):
                raise ValueError(f"Unregistered {kind}: {component}")
    if getattr(args, "cases", None):
        _bind_plan_cases(payload, _path(args.cases), _path(args.output))
    return payload


def cmd_plan(args: argparse.Namespace) -> int:
    output = _write_json(_path(args.output), build_plan(args))
    print(output)
    return 0


def cmd_environment(args: argparse.Namespace) -> int:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "accelerate",
        "hydra",
        "peft",
        "gradio",
        "fastapi",
    ):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "installed")
        except Exception as exc:  # pragma: no cover - environment dependent
            packages[name] = f"unavailable: {exc}"
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ],
        timeout=20,
    )
    cache_paths = getattr(args, "model_cache", [])
    if not cache_paths:
        cache_home = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        hf_home = Path(os.getenv("HF_HOME", str(cache_home / "huggingface")))
        cache_paths = [os.getenv("HF_HUB_CACHE", str(hf_home / "hub"))]
    payload = {
        "generated_at": _utc(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "gpu_query": {k: gpu[k] for k in ("status", "returncode", "stdout", "stderr")},
        "model_caches": [
            {"path": str(path), "exists": path.is_dir()}
            for path in (Path(raw).expanduser() for raw in cache_paths)
        ],
        "evidence_level": "local_validation",
    }
    output = _write_json(_path(args.output), payload)
    print(output)
    return 0


def _yaml_count(path: Path) -> int:
    return (
        sum(1 for item in path.rglob("*.yaml") if item.is_file())
        if path.exists()
        else 0
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    recipes_file = ROOT / "configs/extensions/recipes.yaml"
    recipes = (
        yaml.safe_load(recipes_file.read_text(encoding="utf-8"))
        if recipes_file.exists()
        else {"recipes": []}
    )
    inventory = {
        "generated_at": _utc(),
        "entrypoint": "src/surgery.py",
        "operations": sorted(OPERATIONS),
        "counts": {
            "models": _yaml_count(ROOT / "configs/model"),
            "trainers": _yaml_count(ROOT / "configs/trainer"),
            "datasets": _yaml_count(ROOT / "configs/data/datasets"),
            "evaluations": _yaml_count(ROOT / "configs/eval"),
            "recipes": len(recipes.get("recipes", [])),
            "skills": len(
                [
                    p
                    for p in (ROOT / ".claude/skills").iterdir()
                    if (p / "SKILL.md").is_file()
                ]
            ),
        },
        "recipes": [
            {
                k: item.get(k)
                for k in (
                    "id",
                    "operation",
                    "experiment",
                    "model",
                    "method",
                    "datasets",
                    "evaluation",
                    "knowledge_types",
                )
            }
            for item in recipes.get("recipes", [])
        ],
        "registry": {
            area: sorted(
                path.relative_to(ROOT / "configs" / folder).with_suffix("").as_posix()
                for path in (ROOT / "configs" / folder).rglob("*.yaml")
            )
            for area, folder in {
                "model_ids": "model",
                "dataset_ids": "data/datasets",
                "trainer_ids": "trainer",
            }.items()
        },
        "evidence_level": "configured",
        "limitations": ["Counts and config presence are not benchmark evidence."],
    }
    output = _write_json(_path(args.output), inventory)
    print(output)
    return 0


def _records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _read_json(path)
        return payload if isinstance(payload, list) else [payload]
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".txt":
        chunks = [
            chunk.strip()
            for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8"))
            if chunk.strip()
        ]
        return [{"text": chunk} for chunk in chunks]
    raise ValueError("Supported input formats: .json, .jsonl, .csv, .txt")


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def cmd_normalize(args: argparse.Namespace) -> int:
    from know_surgery.data_contract import normalize_record

    operation = args.operation
    source = _path(args.input)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    accepted, rejected = [], []
    for index, record in enumerate(_records(source)):
        try:
            normalized = normalize_record(
                record,
                operation,
                knowledge_type=args.knowledge_type,
                index=index,
                role=getattr(args, "role", None),
            )
            normalized["import_provenance"] = {
                "filename": source.name,
                "sha256": source_hash,
                "row": index,
            }
            accepted.append(normalized)
        except (ValueError, TypeError) as exc:
            rejected.append({"index": index, "error": str(exc), "record": record})
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in accepted),
        encoding="utf-8",
    )
    _write_json(output.with_suffix(".rejected.json"), rejected)
    _write_json(
        output.with_suffix(".manifest.json"),
        {
            "operation": operation,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "source": source.name,
            "source_sha256": source_hash,
            "semantic_chunking": False,
        },
    )
    print(output)
    return 0 if accepted and not rejected else 2


def cmd_build_cases(args: argparse.Namespace) -> int:
    from know_surgery.data_contract import build_cases

    source = _path(args.input)
    cases, manifest = build_cases(_records(source), args.operation)
    output = _path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cases),
        encoding="utf-8",
    )
    manifest.update(
        operation=args.operation,
        source=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        created_at=_utc(),
    )
    if args.operation == "unlearning":
        for role in ("forget", "retain"):
            path = output.with_name(output.stem + f".{role}.jsonl")
            path.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n"
                    for item in cases
                    if item["dataset_role"] == role
                ),
                encoding="utf-8",
            )
    _write_json(output.with_suffix(".manifest.json"), manifest)
    print(output)
    return 0


def _bind_plan_cases(plan: dict[str, Any], source: Path, plan_path: Path) -> None:
    """Bind validated custom cases to explicit generic adapters and immutable inputs."""
    from know_surgery.data_contract import build_cases

    cases, manifest = build_cases(_records(source), plan["operation"])
    if plan["operation"] == "edit" and set(plan["methods"]) & {
        "ROME",
        "MEMIT",
        "AlphaEdit",
        "AnyEdit",
    }:
        if any(
            not isinstance(case.get("subject"), str) or not case["subject"].strip()
            for case in cases
        ):
            raise ValueError(
                "The selected rank editor requires an explicit subject for every case."
            )
    roles = (
        ("forget", "retain")
        if plan["operation"] == "unlearning"
        else ("edit",)
        if plan["operation"] == "edit"
        else ("train",)
    )
    inputs = {}
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    for role in roles:
        selected = [case for case in cases if case["dataset_role"] == role]
        if len(selected) < plan["num_cases"]:
            raise ValueError(
                f"Requested {plan['num_cases']} cases but {role} contains {len(selected)}."
            )
        selected = selected[: plan["num_cases"]]
        path = plan_path.with_name(plan_path.stem + f".{role}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in selected)
        )
        inputs[role] = {
            "path": str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(selected),
            "source_sha256": source_hash,
        }
    kinds = {case["knowledge_type"] for case in cases}
    if len(kinds) == 1:
        plan["knowledge_type"] = kinds.pop()
    plan["inputs"] = inputs
    plan["warnings"].append(
        "Custom cases use canonical adapters; original dataset_id records the selected logical dataset. "
        f"Missing probes: {', '.join(manifest['missing_probes']) or 'none'}."
    )


def _plan_command(plan: dict[str, Any], method: str, *, preview: bool) -> list[str]:
    operation = plan["operation"]
    if operation not in OPERATIONS or method not in plan["methods"]:
        raise ValueError("Operation/method is outside the reviewed plan.")
    trainer = (
        "inject/" if operation == "inject" else "edit/" if operation == "edit" else ""
    ) + method
    for kind, value in (
        ("model", plan["model_id"]),
        ("trainer", trainer),
        ("data/datasets", plan["dataset_id"]),
    ):
        path = ROOT / "configs" / kind / f"{value}.yaml"
        if not path.is_file() or not path.resolve().is_relative_to(
            (ROOT / "configs").resolve()
        ):
            raise ValueError(f"Unregistered {kind}: {value}")
    command = [
        sys.executable,
        "src/surgery.py",
        "--config-name=surgery.yaml",
        f"operation={operation}",
    ]
    experiment = plan.get("experiments", {}).get(method)
    if experiment:
        command.append(f"experiment={experiment}")
    command.extend([f"model={plan['model_id']}", f"trainer={trainer}"])
    inputs = plan.get("inputs")
    if inputs:
        roles = (
            {"forget", "retain"}
            if operation == "unlearning"
            else {"edit"}
            if operation == "edit"
            else {"train"}
        )
        if set(inputs) != roles:
            raise ValueError("Bound input roles do not match the operation.")
        for role, binding in inputs.items():
            path = _path(binding["path"])
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
            ):
                raise ValueError(f"Bound {role} cases changed since planning.")
            if (
                len(_records(path)) != binding["records"]
                or binding["records"] < plan["num_cases"]
            ):
                raise ValueError(f"Bound {role} sample count does not match the plan.")
            dataset = "Canonical_inject" if role == "train" else f"Canonical_{role}"
            command.append(f"data/datasets@data.{role}={dataset}")
            field = "data_path" if role == "edit" else "hf_args.data_files"
            command.append(
                f"data.{role}.{dataset}.args.{field}={json.dumps(str(path), ensure_ascii=False)}"
            )
    elif operation in {"edit", "inject"}:
        role = "edit" if operation == "edit" else "train"
        command.append(f"data/datasets@data.{role}={plan['dataset_id']}")
    elif not experiment:
        if not plan.get("overrides", {}).get("data/datasets@data.retain"):
            raise ValueError(
                "Unlearning without an exact recipe requires an explicit retain dataset override."
            )
        command.append(f"data/datasets@data.forget={plan['dataset_id']}")
    command.append(f"task_name={_task_name(plan, method)}")
    if operation == "edit":
        command.append(f"max_edits={plan['num_cases']}")
    else:
        command.append(f"++max_train_samples={plan['num_cases']}")
    for key, value in plan.get("overrides", {}).items():
        if key == "task_name":
            continue
        if inputs and key.lstrip("+").startswith("data"):
            raise ValueError("Data overrides conflict with bound input cases.")
        if not re.fullmatch(r"[+]{0,2}[A-Za-z_][A-Za-z0-9_./@-]*", key):
            raise ValueError(f"Invalid override key: {key}")
        if key.lstrip("+") in {
            "operation",
            "model",
            "trainer",
            "experiment",
            "max_edits",
            "max_train_samples",
        }:
            raise ValueError(f"Override conflicts with typed plan: {key}")
        if isinstance(value, str) and (
            "${" in value or any(ord(c) < 32 for c in value)
        ):
            raise ValueError(f"Unsafe override value for {key}")
        encoded = (
            json.dumps(value, ensure_ascii=False)
            if not isinstance(value, str)
            else value
        )
        command.append(f"{key}={encoded}")
    if preview:
        command.extend(["--cfg", "job"])
    return command


def cmd_run_operation(args: argparse.Namespace) -> int:
    plan = _read_json(_path(args.plan))
    if plan.get("operation") != args.operation:
        raise ValueError(
            f"Plan operation {plan.get('operation')} does not match {args.operation}"
        )
    methods = [args.method] if args.method else list(plan["methods"])
    results = []
    for method in methods:
        command = _plan_command(plan, method, preview=not args.execute)
        result = _run(
            command,
            timeout=args.timeout,
            env={"CUDA_VISIBLE_DEVICES": args.cuda} if args.cuda else None,
        )
        result["method"] = method
        result["evidence_level"] = (
            "local_validation"
            if args.execute and result["returncode"] == 0
            else "configured"
            if result["returncode"] == 0
            else "planned"
        )
        results.append(result)
    output = _write_json(
        _path(args.output),
        {
            "operation": args.operation,
            "execute": args.execute,
            "results": results,
            "created_at": _utc(),
        },
    )
    print(output)
    return 0 if all(item["returncode"] == 0 for item in results) else 1


def _run_artifact_candidates(
    operation: str, task_name: str, *, output_dir: str | None = None
) -> list[Path]:
    if output_dir is not None:
        explicit = _path(output_dir)
        return [explicit] if explicit.is_dir() else []
    roots = [ROOT / "saves" / operation]
    if operation == "unlearning":
        roots.append(ROOT / "saves" / "unlearn")
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.rglob(task_name) if path.is_dir())
    return sorted(set(candidates))


def _has_smoke_evidence(
    run_dir: Path, *, started_at: str | None = None, expected_count: int | None = None
) -> bool:
    required = [
        run_dir / name
        for name in (
            "resolved_config.yaml",
            "run_trace.json",
            "command.txt",
            "environment.json",
            "input_manifest.json",
        )
    ]
    metrics = list(run_dir.glob("*_SUMMARY.json"))
    metrics.extend(
        path for path in (run_dir / "unlearning_metrics.json",) if path.is_file()
    )
    edit_result_path = next(
        (
            run_dir / name
            for name in ("edit_results.json", "independent_results.json")
            if (run_dir / name).is_file()
        ),
        None,
    )
    if edit_result_path is not None:
        required.append(edit_result_path)
    if not metrics or not all(
        path.is_file() and path.stat().st_size for path in required
    ):
        return False
    if started_at:
        started = datetime.fromisoformat(started_at).timestamp()
        if any(path.stat().st_mtime < started - 1 for path in [*required, *metrics]):
            return False
    try:
        trace = _read_json(run_dir / "run_trace.json")
        manifest = _read_json(run_dir / "input_manifest.json")
        count = manifest.get(
            "request_count", manifest.get("records", manifest.get("num_examples"))
        )
        if count is None:
            count = trace.get("scope", {}).get("edited_records")
        if count is None:
            count = manifest.get("consumed_records")
        if count is None:
            used = manifest.get("used_samples", {})
            count = (
                sum(value for value in used.values() if isinstance(value, int)) or None
            )
        if edit_result_path is not None:
            edit = _read_json(edit_result_path)
            requested = edit.get("requested_count")
            if edit.get("success") is not True or type(requested) is not int:
                return False
            if "successful_cases" in edit:
                committed = edit["successful_cases"]
            else:
                steps = edit.get("step_results", [edit.get("result", {})])
                if not steps or any(
                    step.get("success") is not True
                    or type(step.get("edited_count")) is not int
                    for step in steps
                ):
                    return False
                committed = sum(step["edited_count"] for step in steps)
            if type(committed) is not int or committed != requested:
                return False
            if count is not None and count != committed:
                return False
            count = committed
        return (
            trace.get("status") in {"success", "completed"}
            and type(count) is int
            and count > 0
            and (expected_count is None or count == expected_count)
            and all(
                isinstance(_read_json(path), dict) and _read_json(path)
                for path in metrics
            )
        )
    except (ValueError, TypeError, AttributeError):
        return False


def _evaluate_unlearning_smoke(
    run_dir: Path, *, num_cases: int, timeout: int, cuda: str | None
) -> dict[str, Any]:
    """Evaluate real bounded forget/retain NLL when a training-only recipe is used."""
    from omegaconf import OmegaConf
    from know_surgery.data_contract import normalize_record

    # Resolve only the nodes used below. Hydra's runtime resolver is no longer
    # registered in this standalone process; unrelated paths.work_dir must not
    # make an otherwise reproducible local evaluation fail.
    cfg = OmegaConf.load(run_dir / "resolved_config.yaml")
    inputs = {}
    for role in ("forget", "retain"):
        datasets = cfg["data"][role]
        if len(datasets) != 1:
            raise ValueError(
                "Bounded smoke evaluation needs one explicit dataset per forget/retain role."
            )
        dataset_args = next(iter(datasets.values()))["args"]
        files = dataset_args.get("hf_args", {}).get("data_files")
        if not isinstance(files, str) or not _path(files).is_file():
            raise ValueError(
                "Bounded unlearning evaluation requires local forget/retain files; supply --cases."
            )
        rows = _records(_path(files))[:num_cases]
        normalized = []
        for index, row in enumerate(rows):
            adapted = {
                **row,
                "prompt": row[dataset_args.get("question_key", "question")],
                "target_old": row[dataset_args.get("answer_key", "answer")],
            }
            case = normalize_record(adapted, "unlearning", index=index, role=role)
            normalized.append(
                {**case, "question": case["prompt"], "answer": case["target_old"]}
            )
        inputs[role] = _write_json(run_dir / f"smoke_{role}_inputs.json", normalized)
    model_ref = str(cfg["model"]["model_args"]["pretrained_model_name_or_path"])
    if not Path(model_ref).is_dir():
        from huggingface_hub import snapshot_download

        model_ref = snapshot_download(repo_id=model_ref, local_files_only=True)
    command = [
        sys.executable,
        "scripts/evaluate_unlearning_local.py",
        "--base-model",
        model_ref,
        "--edited-model",
        str(run_dir),
        "--forget-data",
        str(inputs["forget"]),
        "--retain-data",
        str(inputs["retain"]),
        "--output",
        str(run_dir / "unlearning_metrics.json"),
        "--max-length",
        "128",
    ]
    return _run(
        command, timeout=timeout, env={"CUDA_VISIBLE_DEVICES": cuda} if cuda else None
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    source = _read_json(_path(args.plan))
    plan = dict(source)
    plan["num_cases"] = min(2, int(source["num_cases"]))
    methods = [args.method] if args.method else list(plan["methods"])
    results = []
    for method in methods:
        command = _plan_command(plan, method, preview=not args.execute)
        if args.execute and plan["operation"] in {"unlearning", "inject"}:
            command.append("++trainer.args.max_steps=1")
        result = _run(
            command,
            timeout=args.timeout,
            env={"CUDA_VISIBLE_DEVICES": args.cuda} if args.cuda else None,
        )
        output_dir = next(
            (
                value
                for key, value in plan.get("overrides", {}).items()
                if key.lstrip("+") == "trainer.args.output_dir"
            ),
            None,
        )
        run_dirs = _run_artifact_candidates(
            plan["operation"], _task_name(plan, method), output_dir=output_dir
        )
        if (
            args.execute
            and result["returncode"] == 0
            and plan["operation"] == "unlearning"
        ):
            for run_dir in run_dirs:
                if (
                    not list(run_dir.glob("*_SUMMARY.json"))
                    and not (run_dir / "unlearning_metrics.json").is_file()
                ):
                    try:
                        evaluation = _evaluate_unlearning_smoke(
                            run_dir,
                            num_cases=plan["num_cases"],
                            timeout=args.timeout,
                            cuda=args.cuda,
                        )
                    except Exception as exc:
                        evaluation = {
                            "returncode": 2,
                            "status": "failed",
                            "error": str(exc),
                        }
                    result["evaluation"] = evaluation
                    if evaluation["returncode"] != 0:
                        result.update(
                            returncode=evaluation["returncode"], status="failed"
                        )
        verified = result["returncode"] == 0 and any(
            _has_smoke_evidence(
                path,
                started_at=result["started_at"],
                expected_count=plan["num_cases"]
                if plan["operation"] == "edit"
                else None,
            )
            for path in run_dirs
        )
        result.update(
            {
                "method": method,
                "num_cases": plan["num_cases"],
                "run_dirs": [str(path) for path in run_dirs],
                "evidence_level": (
                    "verified_smoke"
                    if args.execute and verified
                    else "local_validation"
                    if args.execute and result["returncode"] == 0
                    else "configured"
                    if result["returncode"] == 0
                    else "planned"
                ),
            }
        )
        results.append(result)
    output = _write_json(
        _path(args.output),
        {
            "operation": plan["operation"],
            "execute": args.execute,
            "bounded_num_cases": plan["num_cases"],
            "results": results,
            "created_at": _utc(),
        },
    )
    print(output)
    return 0 if all(item["returncode"] == 0 for item in results) else 1


def _summary_payload(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    metrics: dict[str, Any] = {}
    metric_details: dict[str, Any] = {}
    paper_columns: dict[str, Any] = {}
    paper_protocol_columns: dict[str, Any] = {}
    summary_files = []
    for path in sorted(run_dir.glob("*_SUMMARY.json")):
        payload = _read_json(path)
        summary_files.append(str(path))
        if isinstance(payload, dict):
            nested_metrics = payload.get("metrics")
            if isinstance(nested_metrics, dict):
                for key, detail in nested_metrics.items():
                    if isinstance(detail, dict):
                        metrics[key] = detail.get("mean")
                        metric_details[key] = detail
            if isinstance(payload.get("paper_columns"), dict):
                paper_columns = payload["paper_columns"]
            if isinstance(payload.get("paper_protocol_columns"), dict):
                paper_protocol_columns = payload["paper_protocol_columns"]
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key.rsplit(".", 1)[-1]] = value
                elif value is None:
                    metrics.setdefault(key.rsplit(".", 1)[-1], None)
    local = run_dir / "unlearning_metrics.json"
    explicit_evidence = None
    if local.exists():
        payload = _read_json(local)
        explicit_evidence = payload.get("status")
        for key in (
            "forget_nll_change",
            "retain_nll_change",
            "base_forget_nll",
            "edited_forget_nll",
            "base_retain_nll",
            "edited_retain_nll",
        ):
            if key in payload:
                metrics[key] = payload[key]
    target_nll = run_dir / "target_nll.json"
    if target_nll.exists():
        payload = _read_json(target_nll)
        metrics.update(
            {
                "base_target_nll": payload.get("base", {}).get(
                    "weighted_target_token_nll"
                ),
                "post_target_nll": payload.get("post", {}).get(
                    "weighted_target_token_nll"
                ),
                "target_nll_delta": payload.get("delta_post_minus_base"),
            }
        )
    trace = (
        _read_json(run_dir / "run_trace.json")
        if (run_dir / "run_trace.json").exists()
        else {}
    )
    if trace.get("runtime_seconds") is not None:
        metrics["runtime_seconds"] = trace["runtime_seconds"]
    core = all(
        (run_dir / filename).is_file()
        for filename in (
            "command.txt",
            "resolved_config.yaml",
            "input_manifest.json",
            "run_trace.json",
        )
    )
    trace_evidence = trace.get("evidence_level")
    evidence = (
        trace_evidence
        if trace_evidence in EVIDENCE_LEVELS
        else explicit_evidence
        if explicit_evidence in EVIDENCE_LEVELS
        else "verified_smoke"
        if core and trace.get("status") == "success"
        else "local_validation"
        if metrics
        else "configured"
    )
    manifest = (
        _read_json(run_dir / "input_manifest.json")
        if (run_dir / "input_manifest.json").exists()
        else {}
    )
    environment = (
        _read_json(run_dir / "environment.json")
        if (run_dir / "environment.json").exists()
        else {}
    )
    resolved = (
        yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if (run_dir / "resolved_config.yaml").exists()
        else {}
    )
    trainer_args = (
        resolved.get("trainer", {}).get("args", {})
        if isinstance(resolved, dict)
        else {}
    )
    data_hashes = sorted(
        str(item.get("sha256"))
        for item in manifest.get("data_files", [])
        if isinstance(item, dict) and item.get("sha256")
    )
    fairness = {
        "operation": trace.get("operation"),
        "model_id": trace.get("model_id"),
        "model_revision": trace.get("model_revision"),
        "dataset_id": trace.get("dataset_id"),
        "dataset_revision": trace.get("dataset_revision"),
        "data_hashes": data_hashes,
        "scope": trace.get("scope"),
        "protocol": trace.get("protocol"),
        "evidence_level": evidence,
        "complete_benchmark": manifest.get("complete_benchmark"),
        "request_count": manifest.get("request_count"),
        "seed": trainer_args.get("seed") if isinstance(trainer_args, dict) else None,
        "gpu_name": environment.get("gpu_name")
        if isinstance(environment, dict)
        else None,
    }
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "metric_details": metric_details,
        "paper_columns": paper_columns,
        "paper_protocol_columns": paper_protocol_columns,
        "summary_files": summary_files,
        "trace": trace,
        "input_manifest": manifest,
        "environment": environment,
        "fairness": fairness,
        "evidence_level": evidence,
    }


def cmd_aggregate(args: argparse.Namespace) -> int:
    runs = [_summary_payload(_path(path)) for path in args.run_dirs]
    output = _write_json(
        _path(args.output),
        {"generated_at": _utc(), "runs": runs, "missing_metrics_are_null": True},
    )
    print(output)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    runs = [_summary_payload(_path(path)) for path in args.run_dirs]
    keys = sorted(set().union(*(run["metrics"].keys() for run in runs)))
    rows = [
        {
            "run_dir": run["run_dir"],
            "method": run["trace"].get("method"),
            "evidence_level": run["evidence_level"],
            "metric_details": run["metric_details"],
            **{key: run["metrics"].get(key) for key in keys},
        }
        for run in runs
    ]
    fairness_fields = (
        "operation",
        "model_id",
        "model_revision",
        "dataset_id",
        "dataset_revision",
        "data_hashes",
        "scope",
        "protocol",
        "evidence_level",
        "complete_benchmark",
        "request_count",
        "seed",
        "gpu_name",
    )
    fairness: dict[str, dict[str, Any]] = {}
    for field in fairness_fields:
        values = [run["fairness"].get(field) for run in runs]
        fairness[field] = {
            "values": values,
            "known": all(value not in (None, "") for value in values),
            "equal": len({json.dumps(value, sort_keys=True) for value in values}) == 1,
        }
    comparable = len(runs) >= 2 and all(
        item["known"] and item["equal"] for item in fairness.values()
    )
    output = _write_json(
        _path(args.output),
        {
            "generated_at": _utc(),
            "comparable": comparable,
            "fairness": fairness,
            "metric_keys": keys,
            "rows": rows,
            "limitations": (
                []
                if comparable
                else [
                    "Runs are not ranked because one or more fairness fields are missing or differ."
                ]
            ),
        },
    )
    print(output)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    payload = _read_json(_path(args.input))
    json_output = _write_json(_path(args.json_output), payload)
    lines = [
        "# Know-Surgery Auditable Report",
        "",
        f"Generated: {_utc()}",
        "",
        "## Evidence",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Limitations",
        "",
        "- This report preserves the evidence level of its inputs.",
        "- Missing metrics are not replaced with zero.",
    ]
    md = _path(args.markdown_output)
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md)
    print(json_output)
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    run_dir = _path(args.run_dir)
    command_file = run_dir / "command.txt"
    if not command_file.exists():
        raise FileNotFoundError(command_file)
    command = command_file.read_text(encoding="utf-8").strip()
    if not command:
        raise ValueError("The run's command.txt is empty")
    payload = {
        "run_dir": str(run_dir),
        "command": command,
        "resolved_config": str(run_dir / "resolved_config.yaml")
        if (run_dir / "resolved_config.yaml").exists()
        else None,
        "environment": str(run_dir / "environment.json")
        if (run_dir / "environment.json").exists()
        else None,
        "checked_at": _utc(),
    }
    output = _write_json(_path(args.output), payload)
    print(output)
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    path = _path(args.path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a mapping")
    output = _write_json(
        _path(args.output),
        {
            "path": str(path),
            "kind": args.kind,
            "keys": sorted(payload),
            "status": "configured",
            "checked_at": _utc(),
        },
    )
    print(output)
    return 0


def cmd_validate_mm(args: argparse.Namespace) -> int:
    from PIL import Image

    records = _records(_path(args.input))
    errors = []
    if not records:
        errors.append({"error": "no records"})
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            errors.append({"index": index, "error": "record must be an object"})
            continue
        images = item.get("images") or (
            [item.get("image")] if item.get("image") else []
        )
        if isinstance(images, str):
            images = [images]
        if not isinstance(images, list):
            errors.append({"index": index, "error": "images must be a path or list"})
            continue
        if not images:
            errors.append({"index": index, "error": "missing image"})
        for raw in images:
            if not isinstance(raw, str) or not raw.strip():
                errors.append(
                    {"index": index, "error": "image path must be nonempty text"}
                )
                continue
            path = _path(raw)
            if not path.is_file():
                errors.append({"index": index, "error": f"image not found: {raw}"})
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                errors.append(
                    {
                        "index": index,
                        "error": f"unreadable image: {raw} ({type(exc).__name__})",
                    }
                )
    output = _write_json(
        _path(args.output),
        {
            "records": len(records),
            "errors": errors,
            "status": "configured" if not errors else "blocked",
        },
    )
    print(output)
    return 0 if not errors else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    plan = _read_json(_path(args.plan))
    seeds = list(dict.fromkeys(int(value) for value in args.seeds.split(",")))
    runs = []
    for method in plan["methods"]:
        for seed in seeds:
            command = _plan_command(plan, method, preview=False) + [
                f"trainer.args.seed={seed}"
            ]
            command = [
                f"{arg}_seed{seed}" if arg.startswith("task_name=") else arg
                for arg in command
            ]
            command_text = shlex.join(command)
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "argv": command,
                    "command": command_text,
                    "run_hash": hashlib.sha256(
                        command_text.encode("utf-8")
                    ).hexdigest()[:16],
                    "status": "pending",
                }
            )
    output = _write_json(_path(args.output), {"generated_at": _utc(), "runs": runs})
    print(output)
    return 0


def cmd_debug_ui(args: argparse.Namespace) -> int:
    import urllib.request

    try:
        with urllib.request.urlopen(
            args.url.rstrip("/") + "/api/v1/health", timeout=args.timeout
        ) as response:
            body = response.read().decode("utf-8")
            status = "passed" if response.status == 200 else "failed"
    except Exception as exc:
        body = str(exc)
        status = "failed"
    output = _write_json(
        _path(args.output),
        {"url": args.url, "status": status, "response": body, "checked_at": _utc()},
    )
    print(output)
    return 0 if status == "passed" else 1


def cmd_release(args: argparse.Namespace) -> int:
    """Check a usable source checkout without requiring private validation records."""
    skill_root = ROOT / ".claude/skills"
    skills = (
        sorted(
            path.name for path in skill_root.iterdir() if (path / "SKILL.md").is_file()
        )
        if skill_root.is_dir()
        else []
    )
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "yaml",
            "hydra",
            "torch",
            "transformers",
            "datasets",
            "accelerate",
            "peft",
        )
    }
    checks = {
        "python_311_or_newer": sys.version_info >= (3, 11),
        "runtime_dependencies": all(dependencies.values()),
        "skill_count_19": len(skills) == 19,
        "skill_wrappers": bool(skills)
        and all((skill_root / name / "scripts/run.py").is_file() for name in skills),
        "skill_runtime": (ROOT / "scripts/skill_runtime.py").is_file(),
        "plan_schema": (
            skill_root / "_shared/references/surgery-plan.schema.json"
        ).is_file(),
        "unified_entrypoint": (ROOT / "src/surgery.py").is_file(),
        "surgery_config": (ROOT / "configs/surgery.yaml").is_file(),
        "model_configs": bool(_yaml_count(ROOT / "configs/model")),
        "dataset_configs": bool(_yaml_count(ROOT / "configs/data/datasets")),
        "trainer_configs": bool(_yaml_count(ROOT / "configs/trainer")),
        "license": (ROOT / "LICENSE").is_file(),
        "third_party_notices": (ROOT / "THIRD_PARTY_NOTICES.md").is_file(),
    }
    ready = all(checks.values())
    output = _write_json(
        _path(args.output),
        {
            "checked_at": _utc(),
            "checks": checks,
            "dependencies": dependencies,
            "skills": skills,
            "optional_components": {
                "codex_skill_discovery": (ROOT / ".agents/skills").is_dir(),
                "webui_backend": (ROOT / "webui/backend/app.py").is_file(),
                "webui_frontend": (ROOT / "webui/frontend/package.json").is_file(),
            },
            "status": "passed" if ready else "incomplete",
            "release_ready": ready,
            "scope": "source structure and installed dependency discovery",
            "evidence_level": "local_validation",
            "limitations": [
                "Dependency discovery does not prove import, device or model compatibility.",
                "This check does not execute models, certify licenses, or publish a package.",
                "Preview the intended job and run a bounded model check before full execution.",
            ],
        },
    )
    print(output)
    return 0 if ready else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    env = subs.add_parser("environment")
    env.add_argument("--output", default="artifacts/setup-env/environment_report.json")
    env.add_argument(
        "--model-cache",
        action="append",
        default=[],
        help="Local model/cache path to inspect; repeat as needed.",
    )
    env.set_defaults(func=cmd_environment)

    inspect = subs.add_parser("inspect")
    inspect.add_argument(
        "--output", default="artifacts/inspect-repo/repo_inventory.json"
    )
    inspect.set_defaults(func=cmd_inspect)

    plan = subs.add_parser("plan")
    plan.add_argument("--intent", required=True)
    plan.add_argument("--operation", choices=sorted(OPERATIONS))
    plan.add_argument("--model-id")
    plan.add_argument("--dataset-id")
    plan.add_argument("--methods")
    plan.add_argument("--num-cases", type=int)
    plan.add_argument(
        "--cases",
        help="Validated custom cases; bind explicit adapters and source hashes.",
    )
    plan.add_argument("--output", default="artifacts/plan-surgery/surgery_plan.json")
    plan.set_defaults(func=cmd_plan)

    normalize = subs.add_parser("normalize")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--operation", required=True, choices=sorted(OPERATIONS))
    normalize.add_argument("--knowledge-type", default=None)
    normalize.add_argument("--role", choices=("forget", "retain", "train", "edit"))
    normalize.add_argument(
        "--output", default="artifacts/normalize-knowledge/normalized.jsonl"
    )
    normalize.set_defaults(func=cmd_normalize)

    cases = subs.add_parser("build-cases")
    cases.add_argument("--input", required=True)
    cases.add_argument("--operation", required=True, choices=sorted(OPERATIONS))
    cases.add_argument("--output", default="artifacts/build-cases/cases.jsonl")
    cases.set_defaults(func=cmd_build_cases)

    validate = subs.add_parser("validate-config")
    validate.add_argument("--path", required=True)
    validate.add_argument("--kind", required=True, choices=["model", "dataset"])
    validate.add_argument("--output", default="artifacts/config_validation.json")
    validate.set_defaults(func=cmd_validate_config)

    mm = subs.add_parser("validate-mm")
    mm.add_argument("--input", required=True)
    mm.add_argument("--output", default="artifacts/add-multimodal-case/validation.json")
    mm.set_defaults(func=cmd_validate_mm)

    run = subs.add_parser("run-operation")
    run.add_argument("--plan", required=True)
    run.add_argument("--operation", required=True, choices=sorted(OPERATIONS))
    run.add_argument("--method")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--cuda")
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--output", default="artifacts/run/trace.json")
    run.set_defaults(func=cmd_run_operation)

    smoke = subs.add_parser("smoke")
    smoke.add_argument("--plan", required=True)
    smoke.add_argument("--method")
    smoke.add_argument("--execute", action="store_true")
    smoke.add_argument("--cuda")
    smoke.add_argument("--timeout", type=int, default=300)
    smoke.add_argument("--output", default="artifacts/smoke-test/trace.json")
    smoke.set_defaults(func=cmd_smoke)

    sweep = subs.add_parser("sweep")
    sweep.add_argument("--plan", required=True)
    sweep.add_argument("--seeds", default="42")
    sweep.add_argument("--output", default="artifacts/batch-sweep/sweep_manifest.json")
    sweep.set_defaults(func=cmd_sweep)

    aggregate = subs.add_parser("aggregate")
    aggregate.add_argument("run_dirs", nargs="+")
    aggregate.add_argument(
        "--output", default="artifacts/aggregate-results/aggregate.json"
    )
    aggregate.set_defaults(func=cmd_aggregate)

    compare = subs.add_parser("compare")
    compare.add_argument("run_dirs", nargs="+")
    compare.add_argument(
        "--output", default="artifacts/compare-methods/comparison.json"
    )
    compare.set_defaults(func=cmd_compare)

    report = subs.add_parser("report")
    report.add_argument("--input", required=True)
    report.add_argument(
        "--markdown-output", default="artifacts/generate-report/report.md"
    )
    report.add_argument(
        "--json-output", default="artifacts/generate-report/report.json"
    )
    report.set_defaults(func=cmd_report)

    reproduce = subs.add_parser("reproduce")
    reproduce.add_argument("--run-dir", required=True)
    reproduce.add_argument(
        "--output", default="artifacts/reproduce-run/reproduction.json"
    )
    reproduce.set_defaults(func=cmd_reproduce)

    debug = subs.add_parser("debug-ui")
    debug.add_argument("--url", default="http://127.0.0.1:7860")
    debug.add_argument("--timeout", type=int, default=10)
    debug.add_argument("--output", default="artifacts/debug-ui/health.json")
    debug.set_defaults(func=cmd_debug_ui)

    release = subs.add_parser("release")
    release.add_argument(
        "--output", default="artifacts/release-checklist/checklist.json"
    )
    release.set_defaults(func=cmd_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
