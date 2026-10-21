"""Evidence-aware local and OpenAI-compatible planning providers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .catalog import ROOT, RepositoryCatalog
from .schemas import PlanRequest


class PlannerError(RuntimeError):
    """Raised when a planner cannot return a registry-valid SurgeryPlan."""


PROVIDERS = {
    "local": {
        "label": "Deterministic local",
        "env": None,
        "base_url": None,
        "model_env": None,
        "default_model": "registry-rules-v1",
    },
    "openai": {
        "label": "OpenAI GPT",
        "env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_AGENT_MODEL",
        "default_model": "gpt-4.1-mini",
    },
    "deepseek": {
        "label": "DeepSeek",
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "model_env": "DEEPSEEK_AGENT_MODEL",
        "default_model": "deepseek-chat",
    },
}


def provider_status() -> list[dict[str, Any]]:
    rows = []
    for provider_id, config in PROVIDERS.items():
        key_name = config.get("env")
        rows.append(
            {
                "id": provider_id,
                "label": config["label"],
                "configured": provider_id == "local" or bool(os.getenv(key_name or "")),
                "key_source": key_name,
                "browser_key_required": False,
            }
        )
    return rows


def _local_plan(request: PlanRequest) -> dict[str, Any]:
    output = ROOT / "artifacts" / "webui" / "plans" / f"{uuid.uuid4().hex}.json"
    argv = [
        sys.executable,
        str(ROOT / "scripts" / "skill_runtime.py"),
        "plan",
        "--intent",
        request.intent,
        "--output",
        str(output),
    ]
    if request.operation:
        argv.extend(["--operation", request.operation])
    if request.model_id:
        argv.extend(["--model-id", request.model_id])
    if request.dataset_id:
        argv.extend(["--dataset-id", request.dataset_id])
    if request.methods:
        argv.extend(["--methods", ",".join(request.methods)])
    if request.num_cases:
        argv.extend(["--num-cases", str(request.num_cases)])
    process = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    if process.returncode != 0 or not output.is_file():
        raise PlannerError(process.stderr.strip() or "Local planner failed.")
    plan = json.loads(output.read_text(encoding="utf-8"))
    plan["plan_id"] = output.stem
    plan["provider"] = "local"
    plan["reasoning"] = _local_reasoning(plan, request.language)
    if request.language == "zh":
        for step in plan.get("steps", []):
            step["reason"] = "根据目标选择对应步骤。"
    plan["warnings"] = [
        _localize_warning(warning, request.language)
        for warning in plan.get("warnings", [])
    ]
    return plan


def _localize_warning(warning: str, language: str = "zh") -> str:
    if language == "en":
        if warning.startswith(
            (
                "The selected Qwen2.5-7B route is configured",
                "Check the local Qwen2.5-7B model path",
            )
        ):
            return "Check the local Qwen2.5-7B model path and available GPU memory before running."
        if warning.startswith("LoReFT requires pyreft"):
            return "LoReFT requires pyreft and the ReFT data collator."
        return warning
    if warning.startswith("No model was explicit;"):
        return "未明确指定模型，已选择 GPT-2。"
    match = re.match(
        r"(.+) has no curated exact recipe for this model/dataset; "
        r"manual compose is required\.",
        warning,
    )
    if match:
        return f"{match.group(1)} 没有适用于该模型与数据集的精选配方，需要手工组合。"
    if warning.startswith("Only the 7B size was explicit;"):
        return (
            "请求只明确了 7B 规模；由于已注册的长文本配方以 "
            "Qwen2.5-7B-Instruct 为目标，规划器选择了该模型。"
        )
    if warning.startswith(
        (
            "The selected Qwen2.5-7B route is configured",
            "Check the local Qwen2.5-7B model path",
        )
    ):
        return "运行前请检查本地 Qwen2.5-7B 模型路径和可用 GPU 显存。"
    if warning.startswith("LoReFT requires pyreft"):
        return "LoReFT 需要 pyreft 与 ReFT 数据整理器。"
    if warning.startswith("UNKE requires preservation data"):
        return "UNKE 需要保留数据，公开包未提供此方法。"
    if warning.startswith("AnyEdit MEMIT/AlphaEdit backends require"):
        return "AnyEdit 的 MEMIT/AlphaEdit 后端在启用统计校正时需要相应统计量。"
    return "规划器发现需要人工检查的兼容性问题。"


def _local_reasoning(
    plan: dict[str, Any], language: str = "en"
) -> list[dict[str, str]]:
    operation = plan["operation"]
    model_was_explicit = "No model was explicit" not in " ".join(
        plan.get("warnings", [])
    )
    size_only_selection = "Only the 7B size was explicit" in " ".join(
        plan.get("warnings", [])
    )
    if language == "zh":
        reasons = {
            "unlearning": "选择了区分遗忘集与保留集并检查模型能力的路线。",
            "inject": "选择了参数高效的知识注入路线，并同时评估知识保持。",
            "edit": "选择了局部知识编辑路线，并评估可靠性与副作用。",
        }
        if size_only_selection:
            model_reason = (
                "请求指定了 7B 规模；根据长文本配方覆盖选择 Qwen2.5-7B-Instruct。"
            )
        else:
            model_reason = (
                "采用请求中明确指定的模型。"
                if model_was_explicit
                else "请求未明确指定模型，因此采用 GPT-2。"
            )
        method_reason = "方法仅限已注册的训练器或编辑器标识。"
        evaluation_reason = "同时评估目标效果和副作用。"
    else:
        reasons = {
            "unlearning": "Selected a forget/retain-aware route with utility checks.",
            "inject": "Selected parameter-efficient injection with retention evaluation.",
            "edit": "Selected localized editing with reliability and side-effect probes.",
        }
        if size_only_selection:
            model_reason = (
                "Uses Qwen2.5-7B-Instruct because the request specified 7B and "
                "the registered long-text recipes target that model."
            )
        else:
            model_reason = (
                "Uses the explicit model request."
                if model_was_explicit
                else "Uses GPT-2 because no model was specified."
            )
        method_reason = (
            "Methods are limited to registered trainer or editor identifiers."
        )
        evaluation_reason = "Target efficacy and side effects are evaluated together."
    return [
        {"field": "operation", "reason": reasons[operation]},
        {"field": "model", "reason": model_reason},
        {"field": "methods", "reason": method_reason},
        {"field": "evaluation", "reason": evaluation_reason},
    ]


def _extract_json(content: str) -> dict[str, Any]:
    content = content.strip()
    fence = chr(96) * 3
    if content.startswith(fence):
        content = re.sub(r"^.{3}(?:json)?\s*", "", content)
        content = re.sub(r"\s*.{3}$", "", content)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"Provider returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PlannerError("Provider response must be a JSON object.")
    return value


def _validate_remote_choice(
    choice: dict[str, Any],
    baseline: dict[str, Any],
    catalog: RepositoryCatalog,
) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    models = {item["id"] for item in snapshot["models"]}
    datasets = {item["id"] for item in snapshot["datasets"]}
    methods_by_operation = {
        operation: {Path(item["id"]).name for item in rows}
        for operation, rows in snapshot["trainers"].items()
    }
    operation = choice.get("operation", baseline["operation"])
    if operation not in methods_by_operation:
        raise PlannerError("Provider selected an unsupported operation.")
    model = choice.get("model_id", baseline["model_id"])
    dataset = choice.get("dataset_id", baseline["dataset_id"])
    methods = choice.get("methods", baseline["methods"])
    if model not in models:
        raise PlannerError(f"Provider selected an unregistered model: {model}")
    if dataset not in datasets:
        raise PlannerError(f"Provider selected an unregistered dataset: {dataset}")
    if (
        not isinstance(methods, list)
        or not methods
        or any(str(method) not in methods_by_operation[operation] for method in methods)
    ):
        raise PlannerError("Provider selected one or more unregistered methods.")
    baseline.update(
        {
            "operation": operation,
            "model_id": model,
            "dataset_id": dataset,
            "methods": [str(item) for item in methods],
            "num_cases": max(
                1,
                min(
                    100_000,
                    int(choice.get("num_cases", baseline["num_cases"])),
                ),
            ),
            "reasoning": choice.get("reasoning") or baseline["reasoning"],
            "warnings": [
                *baseline.get("warnings", []),
                *[
                    str(item)
                    for item in choice.get("warnings", [])
                    if str(item).strip()
                ],
            ],
        }
    )
    # Rebuild every derived field after a provider changes the typed choices.
    rebuilt = _local_plan(
        PlanRequest(
            intent=baseline["intent"],
            operation=operation,
            model_id=model,
            dataset_id=dataset,
            methods=baseline["methods"],
            num_cases=baseline["num_cases"],
        )
    )
    for key in ("experiments", "evaluators", "steps", "knowledge_type"):
        baseline[key] = rebuilt[key]
    baseline["warnings"] = list(
        dict.fromkeys(
            [*rebuilt["warnings"], *[str(item) for item in choice.get("warnings", [])]]
        )
    )
    return baseline


def _remote_plan(
    provider: str,
    request: PlanRequest,
    baseline: dict[str, Any],
    catalog: RepositoryCatalog,
) -> dict[str, Any]:
    config = PROVIDERS[provider]
    api_key = os.getenv(config["env"])
    if not api_key:
        raise PlannerError(
            f"{provider} is not configured on the server; set {config['env']}."
        )
    snapshot = catalog.snapshot()
    compact_catalog = {
        "operations": [item["id"] for item in snapshot["operations"]],
        "models": [
            item["id"] for item in snapshot["models"] if item["modality"] == "text"
        ],
        "datasets": [item["id"] for item in snapshot["datasets"]],
        "methods": {
            key: [Path(item["id"]).name for item in value]
            for key, value in snapshot["trainers"].items()
        },
    }
    system = (
        "You are the Know-Surgery planner. Return JSON only. Select only IDs "
        "listed in the supplied registry. Keep public operation names exactly "
        "unlearning, inject, or edit. Include target efficacy and side-effect "
        "evaluation. Return operation, model_id, dataset_id, methods, num_cases, "
        "reasoning (array of field/reason objects), and warnings."
    )
    system += (
        " Write all explanatory reasons and warnings in Simplified Chinese."
        if request.language == "zh"
        else " Write all explanatory reasons and warnings in English."
    )
    body = {
        "model": os.getenv(config["model_env"], config["default_model"]),
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "intent": request.intent,
                        "registry": compact_catalog,
                        "deterministic_baseline": baseline,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    base_url = os.getenv(
        config.get("base_url_env", ""),
        config["base_url"],
    ).rstrip("/")
    http_request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        raise PlannerError(f"{provider} planning request failed: {exc}") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise PlannerError(f"{provider} returned an unexpected response.") from exc
    choice = _extract_json(content)
    for field in ("operation", "model_id", "dataset_id", "methods", "num_cases"):
        explicit = getattr(request, field)
        if explicit is not None and field in choice and choice[field] != explicit:
            raise PlannerError(f"Provider changed the explicit user selection: {field}")
    plan = _validate_remote_choice(
        choice,
        baseline,
        catalog,
    )
    plan["provider"] = provider
    return plan


def create_plan(
    request: PlanRequest,
    catalog: RepositoryCatalog,
) -> dict[str, Any]:
    baseline = _local_plan(request)
    if request.provider == "local":
        return baseline
    return _remote_plan(request.provider, request, baseline, catalog)
