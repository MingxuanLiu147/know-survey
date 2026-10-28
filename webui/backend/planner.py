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

from src.know_surgery.resources import reference_trainer_for_experiment

from .catalog import ROOT, RepositoryCatalog
from .schemas import ComposeRequest, PlanRequest


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
        "default_model": "deepseek-flash",
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
    if request.modality:
        argv.extend(["--modality", request.modality])
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
    for step in plan.get("steps", []):
        step["reason"] = _step_reason(step["skill"], request.language)
    plan["warnings"] = [
        _localize_warning(warning, request.language)
        for warning in plan.get("warnings", [])
    ]
    return plan


def _step_reason(skill: str, language: str) -> str:
    """Describe suggested workflow steps without implying tool execution."""
    descriptions = {
        "plan-surgery": (
            "根据目标生成可修改的模型、方法、数据和评估配置。",
            "Propose editable model, method, data and evaluation settings.",
        ),
        "inspect-repo": (
            "核对所选模型、方法和数据是否已注册。",
            "Check that the selected model, method and dataset are registered.",
        ),
        "normalize-knowledge": (
            "导入原始数据时统一字段；已规范化的数据可跳过。",
            "Normalize fields when importing raw data; skip for normalized data.",
        ),
        "build-cases": (
            "需要时构造任务样本与评估探针；已有可用样本可跳过。",
            "Build task cases and evaluation probes when needed; reuse ready cases.",
        ),
        "smoke-test": (
            "预览配置并核对输入；需要时检查不超过 2 条样本。",
            "Preview the configuration and inputs; optionally check up to 2 cases.",
        ),
        "run-edit": (
            "执行确认后的编辑配置，并保存输出与评估结果。",
            "Run the reviewed editing configuration and save outputs and metrics.",
        ),
        "run-inject": (
            "执行确认后的注入配置，并保存输出与评估结果。",
            "Run the reviewed injection configuration and save outputs and metrics.",
        ),
        "run-unlearn": (
            "执行确认后的遗忘配置，并检查遗忘集和保留集。",
            "Run the reviewed unlearning configuration and evaluate forget and retain sets.",
        ),
        "aggregate-results": (
            "有多次运行时汇总已保存的指标。",
            "Collect saved metrics when there are multiple runs.",
        ),
        "compare-methods": (
            "需要比较方法时，核对运行条件后比较指标。",
            "Compare methods when requested, after checking their run settings.",
        ),
        "generate-report": (
            "按需将已有运行结果整理成报告。",
            "Optionally turn existing run results into a report.",
        ),
        "reproduce-run": (
            "按需提取已保存运行的命令和配置，供再次运行。",
            "Optionally recover a saved run's command and configuration for reuse.",
        ),
        "add-multimodal-case": (
            "使用图文数据时检查图像路径与样本字段。",
            "Check image paths and case fields when using image-text data.",
        ),
        "add-dataset": (
            "使用新数据集时注册其配置和字段映射。",
            "Register the configuration and field mapping when adding a dataset.",
        ),
    }
    fallback = (
        "按任务需要使用此工作流步骤。",
        "Use this workflow step when the task requires it.",
    )
    return descriptions.get(skill, fallback)[0 if language == "zh" else 1]


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
    if warning.startswith("Multimodal injection requires"):
        return "多模态注入需要本地图文样本；请分别检查目标效果和保留探针。"
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
    if plan.get("modality") == "multimodal":
        model_reason = (
            "采用计划中列出的视觉语言模型。"
            if language == "zh"
            else "Uses the vision-language model listed in the plan."
        )
        evaluation_reason = (
            "运行后核对目标效果和保留探针；缺少探针不代表已完成评估。"
            if language == "zh"
            else "Review target and retention probes after execution; absent probes are not completed evaluation."
        )
        if operation == "inject":
            reasons[operation] = (
                "选择图文样本的 LoRA 注入路线。"
                if language == "zh"
                else "Selected image-text injection with LoRA."
            )
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
    language: str = "en",
) -> dict[str, Any]:
    snapshot = catalog.snapshot()
    modality = choice.get("modality", baseline.get("modality", "text"))
    if modality not in {"text", "multimodal"}:
        raise PlannerError("Provider selected an unsupported modality.")
    models = {item["id"] for item in snapshot["models"] if item["modality"] == modality}
    datasets = {
        item["id"] for item in snapshot["datasets"] if item["modality"] == modality
    }
    methods_by_operation = {
        operation: {
            Path(item["id"]).name for item in rows if item["modality"] == modality
        }
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
            "modality": modality,
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
            language=language,
            operation=operation,
            modality=modality,
            model_id=model,
            dataset_id=dataset,
            methods=baseline["methods"],
            num_cases=baseline["num_cases"],
        )
    )
    for key in ("experiments", "evaluators", "steps", "knowledge_type"):
        baseline[key] = rebuilt[key]
    baseline["reasoning"] = choice.get("reasoning") or rebuilt["reasoning"]
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
            item["id"]
            for item in snapshot["models"]
            if item["modality"] == baseline.get("modality", "text")
        ],
        "datasets": [
            item["id"]
            for item in snapshot["datasets"]
            if item["modality"] == baseline.get("modality", "text")
        ],
        "methods": {
            key: [
                Path(item["id"]).name
                for item in value
                if item["modality"] == baseline.get("modality", "text")
            ]
            for key, value in snapshot["trainers"].items()
        },
    }
    system = (
        "You are the Know-Surgery planner. Return JSON only. Select only IDs "
        "listed in the supplied registry. Keep public operation names exactly "
        "unlearning, inject, or edit. Include target efficacy and side-effect "
        "evaluation. Return operation, modality, model_id, dataset_id, methods, num_cases, "
        "reasoning (array of field/reason objects), and warnings. "
        "Use operation, model, dataset, methods, evaluation and num_cases as reasoning "
        "field keys; keep registry IDs and field keys unchanged. "
        "Describe sample counts and training steps concretely. Do not use labels such as "
        "smoke test, 冒烟测试, 冒烟, 极小试运行 or 不代表正式评测. "
        "The baseline steps are suggested workflow definitions, not a mandatory sequence "
        "or a record of completed tool calls. Raw-data preparation is needed only for "
        "unprepared inputs; reporting and reproduction are optional."
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
    for field in (
        "operation",
        "modality",
        "model_id",
        "dataset_id",
        "methods",
        "num_cases",
    ):
        explicit = getattr(request, field)
        if explicit is not None and field in choice and choice[field] != explicit:
            raise PlannerError(f"Provider changed the explicit user selection: {field}")
    plan = _validate_remote_choice(
        choice,
        baseline,
        catalog,
        language=request.language,
    )
    plan["provider"] = provider
    return plan


def create_plan(
    request: PlanRequest,
    catalog: RepositoryCatalog,
) -> dict[str, Any]:
    baseline = _local_plan(request)
    plan = (
        baseline
        if request.provider == "local"
        else _remote_plan(request.provider, request, baseline, catalog)
    )
    plan["language"] = request.language
    step_match = re.search(
        r"(\d+)\s*(?:(?:个|次)?\s*训练步|(?:training\s+)?steps?\b)",
        request.intent,
        flags=re.I,
    )
    max_steps = int(step_match.group(1)) if step_match else None
    if max_steps is not None and max_steps < 1:
        raise PlannerError("Training steps must be positive. / 训练步数必须大于零。")
    sample_match = re.search(
        r"(\d+)\s*(?:edits?\b|cases?\b|samples?\b|records?\b|facts?\b|条|个\s*(?:样本|记录|事实))",
        request.intent,
        flags=re.I,
    )
    explicit_samples = request.num_cases is not None or sample_match is not None
    if request.num_cases is not None:
        plan["num_cases"] = request.num_cases
    elif sample_match is not None:
        count = int(sample_match.group(1))
        if not 1 <= count <= 100_000:
            raise PlannerError("Sample count must be between 1 and 100000.")
        plan["num_cases"] = count
    # A step-only request keeps the selected dataset's range. The shared planner's
    # default num_cases=1 is not evidence that the user requested one training row.
    limit_samples = plan["operation"] == "edit" or max_steps is None or explicit_samples
    plan["configuration_limits"] = {
        "max_samples": plan["num_cases"] if limit_samples else None,
        "max_steps": max_steps if plan["operation"] != "edit" else None,
    }
    configurations = {}
    configuration_errors = {}
    for method in plan["methods"]:
        try:
            configuration = configuration_from_plan(
                plan, method, limit_samples=limit_samples, max_steps=max_steps
            )
            catalog.validate_request(configuration)
            configurations[method] = configuration.model_dump()
        except (PlannerError, ValueError, TypeError) as exc:
            configuration_errors[method] = str(exc)
    plan["configurations"] = configurations
    plan["configuration"] = configurations.get(plan["methods"][0])
    plan["configuration_errors"] = configuration_errors
    return plan


def configuration_from_plan(
    plan: dict[str, Any],
    method: str,
    *,
    limit_samples: bool = True,
    max_steps: int | None = None,
) -> ComposeRequest:
    """Translate explicit plan choices without replacing their model or sample budget."""
    if method not in plan["methods"]:
        raise PlannerError("Method is outside the generated plan.")
    operation = plan["operation"]
    modality = plan.get("modality", "text")
    experiment = plan.get("experiments", {}).get(method)
    overrides = dict(plan.get("overrides", {}))
    task_name = (
        overrides.pop("task_name", None) or f"webui_{operation}_{uuid.uuid4().hex[:12]}"
    )
    seed = overrides.pop("trainer.args.seed", overrides.pop("seed", 42))
    datasets = {}
    for key in list(overrides):
        if key.startswith("data/datasets@data."):
            datasets[key.rsplit(".", 1)[-1]] = overrides.pop(key)
    if not (modality == "multimodal" and operation == "unlearning"):
        role = {"edit": "edit", "inject": "train", "unlearning": "forget"}[operation]
        datasets[role] = plan["dataset_id"]
    if operation == "unlearning" and not experiment:
        if modality == "multimodal" or "retain" not in datasets:
            raise PlannerError(
                "Select an experiment with explicit forget and retain data. / 请选择包含遗忘集与保留集的实验配置。"
            )
    if plan.get("inputs"):
        # Do not discard a role-bound input contract in favor of a dataset ID.
        raise PlannerError(
            "Apply the registered upload configuration for bound inputs. / 绑定输入请使用上传注册后返回的配置。"
        )
    limit_key = (
        "max_edits"
        if operation == "edit"
        else "data.max_samples"
        if modality == "multimodal"
        else "++max_train_samples"
    )
    if limit_samples:
        overrides[limit_key] = int(plan["num_cases"])
    if max_steps is not None and operation != "edit":
        overrides["++trainer.args.max_steps"] = max_steps
    trainer = ("" if operation == "unlearning" else f"{operation}/") + method
    trainer = reference_trainer_for_experiment(experiment, trainer)
    evaluation = (
        None
        if experiment or modality == "multimodal" and operation == "inject"
        else "mm_edit"
        if modality == "multimodal"
        else operation
    )
    return ComposeRequest(
        operation=operation,
        modality=modality,
        experiment=experiment,
        model=plan["model_id"],
        trainer=trainer,
        datasets=datasets,
        evaluation=evaluation,
        task_name=str(task_name),
        seed=int(seed),
        smoke=False,
        overrides=overrides,
    )
