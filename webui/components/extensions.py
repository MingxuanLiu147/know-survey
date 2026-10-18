"""Bilingual, export-only Gradio wizard for custom text extensions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import gradio as gr
import yaml

try:  # Imported as ``webui.components.extensions``.
    from ..utils.extensions import (
        KNOWLEDGE_TYPES,
        ExtensionInputError,
        build_algorithm_extension_draft,
        build_dataset_extension_draft,
        build_model_extension_draft,
        catalog_markdown,
        default_field_mapping,
        export_extension_zip,
        installed_extension_ids,
        parse_data_file,
        refresh_extension_catalog,
        render_diagnostics,
        separate_catalog_records,
        build_canonical_preview,
    )
except ImportError:  # ``webui`` was added directly to sys.path by app.py.
    from utils.extensions import (
        KNOWLEDGE_TYPES,
        ExtensionInputError,
        build_algorithm_extension_draft,
        build_dataset_extension_draft,
        build_model_extension_draft,
        catalog_markdown,
        default_field_mapping,
        export_extension_zip,
        installed_extension_ids,
        parse_data_file,
        refresh_extension_catalog,
        render_diagnostics,
        separate_catalog_records,
        build_canonical_preview,
    )


def _upload_value(value: Any) -> Any:
    """Normalize Gradio File/FileData values without opening anything here."""

    if isinstance(value, Mapping):
        return value.get("path") or value.get("name")
    return value


def _json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ExtensionInputError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ExtensionInputError(f"{label} must be a JSON object.")
    return parsed


def _draft_outputs(draft) -> tuple[str, str, str, str, str]:
    archive = export_extension_zip(draft)
    return (
        render_diagnostics(draft.diagnostics),
        yaml.safe_dump(dict(draft.manifest), sort_keys=False, allow_unicode=True),
        "\n".join(f"- {path}" for path in sorted(draft.files)),
        draft.equivalent_cli,
        archive,
    )


def _error_outputs(exc: Exception) -> tuple[str, str, str, str, None]:
    return (
        f"- **ERROR · input_validation** — {exc}",
        "",
        "",
        "",
        None,
    )


def _recipe_defaults(operation: str) -> tuple[str, str, str, str]:
    if operation == "unlearning":
        return (
            "configs/model/Qwen2.5-3B-Instruct.yaml",
            "configs/trainer/GradAscent.yaml",
            json.dumps(
                {
                    "forget": "configs/data/datasets/TOFU_QA_forget.yaml",
                    "retain": "configs/data/datasets/TOFU_QA_retain.yaml",
                },
                indent=2,
            ),
            "configs/eval/tofu.yaml",
        )
    if operation == "inject":
        return (
            "configs/model/Qwen2.5-3B-Instruct.yaml",
            "configs/trainer/inject/LoRA.yaml",
            json.dumps({"train": "configs/data/datasets/Alpaca_inject.yaml"}, indent=2),
            "configs/eval/inject.yaml",
        )
    return (
        "configs/model/Qwen2.5-3B-Instruct.yaml",
        "configs/trainer/edit/FT.yaml",
        json.dumps({"edit": "configs/data/datasets/CounterFact_edit.yaml"}, indent=2),
        "configs/eval/edit.yaml",
    )


def _model_recipe_updates(operation: str):
    _, method, datasets, evaluation = _recipe_defaults(operation)
    return (
        gr.update(value=method),
        gr.update(value=datasets),
        gr.update(value=evaluation),
    )


def _dataset_form_updates(operation: str, format_type: str):
    model, method, _, evaluation = _recipe_defaults(operation)
    mapping = default_field_mapping(operation, format_type)
    return (
        gr.update(value=json.dumps(mapping, indent=2)),
        gr.update(visible=operation == "unlearning"),
        gr.update(value=model),
        gr.update(value=method),
        gr.update(value=evaluation),
    )


def _algorithm_form_updates(operation: str):
    model, _, datasets, evaluation = _recipe_defaults(operation)
    base = {
        "unlearning": "UnlearnTrainer",
        "inject": "InjectTrainer",
        "edit": "EditTrainer",
    }[operation]
    return (
        gr.update(value=base),
        gr.update(value=model),
        gr.update(value=datasets),
        gr.update(value=evaluation),
    )


def preview_dataset_callback(
    operation: str,
    knowledge_type: str,
    format_type: str,
    primary_upload: Any,
    retain_upload: Any,
    mapping_json: str,
):
    """Parse and normalize uploaded samples; return only the first three per role."""

    try:
        primary = _upload_value(primary_upload)
        retain = _upload_value(retain_upload)
        if primary is None:
            raise ExtensionInputError("Choose a JSON, JSONL, or CSV sample file.")
        records = parse_data_file(primary)
        retain_records = parse_data_file(retain) if retain is not None else None
        preview = build_canonical_preview(
            operation,
            records,
            mapping_json,
            knowledge_type=knowledge_type,
            format_type=format_type,
            retain_records=retain_records,
        )
        counts = f"Parsed {len(records)} primary record(s)" + (
            f" and {len(retain_records)} retain record(s)."
            if retain_records is not None
            else "."
        )
        return (
            preview,
            f"- **INFO · preview** — {counts} Showing at most three per role.",
        )
    except (ExtensionInputError, OSError, ValueError) as exc:
        return {}, f"- **ERROR · data_validation** — {exc}"


def export_model_callback(
    project_root: str,
    extension_id: str,
    operation: str,
    model_reference: str,
    tokenizer_reference: str,
    family: str,
    dtype: str,
    attention: str,
    chat_template: str,
    knowledge_types: list[str],
    method_config: str,
    dataset_configs_json: str,
    evaluation_config: str,
):
    """Review and export one configuration-only model extension."""

    try:
        draft = build_model_extension_draft(
            extension_id=extension_id,
            operation=operation,
            model_reference=model_reference,
            tokenizer_reference=tokenizer_reference or None,
            family=family,
            dtype=dtype,
            attention=attention,
            chat_template=chat_template or None,
            knowledge_types=knowledge_types,
            method_config=method_config,
            dataset_configs=_json_object(
                dataset_configs_json, label="Recipe dataset references"
            ),
            evaluation_config=evaluation_config,
            existing_ids=installed_extension_ids(project_root),
        )
        return _draft_outputs(draft)
    except (ExtensionInputError, OSError, ValueError) as exc:
        return _error_outputs(exc)


def export_dataset_callback(
    project_root: str,
    extension_id: str,
    operation: str,
    knowledge_type: str,
    format_type: str,
    primary_upload: Any,
    retain_upload: Any,
    mapping_json: str,
    family: str,
    model_config: str,
    method_config: str,
    evaluation_config: str,
):
    """Review and export one normalized dataset extension."""

    try:
        primary = _upload_value(primary_upload)
        retain = _upload_value(retain_upload)
        if primary is None:
            raise ExtensionInputError("Choose a JSON, JSONL, or CSV sample file.")
        draft = build_dataset_extension_draft(
            extension_id=extension_id,
            operation=operation,
            source=primary,
            retain_source=retain,
            mapping=mapping_json,
            knowledge_type=knowledge_type,
            format_type=format_type,
            family=family,
            model_config=model_config,
            method_config=method_config,
            evaluation_config=evaluation_config,
            existing_ids=installed_extension_ids(project_root),
        )
        return _draft_outputs(draft)
    except (ExtensionInputError, OSError, ValueError) as exc:
        return _error_outputs(exc)


def export_algorithm_callback(
    project_root: str,
    extension_id: str,
    operation: str,
    base_class: str,
    handler: str,
    parameter_schema_json: str,
    model_families: str,
    knowledge_types: list[str],
    architecture_specific: bool,
    model_config: str,
    dataset_configs_json: str,
    evaluation_config: str,
    requirements_json: str,
):
    """Review/export a generated algorithm scaffold; never import or run it."""

    try:
        expected_base = {
            "unlearning": "UnlearnTrainer",
            "inject": "InjectTrainer",
            "edit": "EditTrainer",
        }[operation]
        if base_class != expected_base:
            raise ExtensionInputError(
                f"{operation} algorithms must use {expected_base}; got {base_class}."
            )
        requirements = _json_object(requirements_json, label="Requirements")
        unknown = set(requirements) - {
            "python_packages",
            "files",
            "checkpoints",
            "statistics",
        }
        if unknown:
            raise ExtensionInputError(
                f"Unknown requirement groups: {', '.join(sorted(unknown))}."
            )
        draft = build_algorithm_extension_draft(
            extension_id=extension_id,
            operation=operation,
            handler=handler,
            parameter_schema=parameter_schema_json,
            model_families=model_families,
            knowledge_types=knowledge_types,
            architecture_specific=architecture_specific,
            model_config=model_config,
            dataset_configs=_json_object(
                dataset_configs_json, label="Recipe dataset references"
            ),
            evaluation_config=evaluation_config,
            python_packages=requirements.get("python_packages", ()),
            required_files=requirements.get("files", ()),
            checkpoints=requirements.get("checkpoints", ()),
            statistics=requirements.get("statistics", ()),
            existing_ids=installed_extension_ids(project_root),
        )
        return _draft_outputs(draft)
    except (ExtensionInputError, KeyError, OSError, ValueError) as exc:
        return _error_outputs(exc)


def refresh_catalog_callback(project_root: str) -> str:
    """Refresh registry state and preserve verified/experimental isolation."""

    return catalog_markdown(refresh_extension_catalog(project_root))


def _review_outputs(prefix: str) -> dict[str, Any]:
    """Create a consistent, accessible Review & Export output group."""

    outputs: dict[str, Any] = {}
    gr.Markdown("### Review & Export / 审查与导出")
    outputs["diagnostics"] = gr.Markdown(
        "_Complete the form, then review diagnostics. / 完成表单后检查诊断。_"
    )
    with gr.Row():
        outputs["manifest"] = gr.Textbox(
            label="Manifest preview / 清单预览",
            lines=18,
            interactive=False,
            elem_id=f"{prefix}-manifest",
        )
        outputs["files"] = gr.Textbox(
            label="Generated files / 生成文件",
            lines=18,
            interactive=False,
            elem_id=f"{prefix}-files",
        )
    outputs["cli"] = gr.Textbox(
        label="Equivalent compose CLI / 等价配置预览命令",
        lines=9,
        interactive=False,
        elem_id=f"{prefix}-cli",
    )
    outputs["download"] = gr.File(
        label="Temporary ZIP export / 临时 ZIP 导出", interactive=False
    )
    return outputs


def create_custom_extensions_panel(
    project_root: str | Path,
    runner: Any | None = None,
    catalog_records: Any | None = None,
) -> dict[str, Any]:
    """Create and bind the export-only custom extensions Gradio area.

    ``runner`` is accepted to make the shared-command ownership explicit; all
    commands shown by the pure builders already use ``CommandRunner`` and this
    panel never calls ``runner.run``.
    """

    del runner
    root = str(Path(project_root).resolve())
    components: dict[str, Any] = {}

    with gr.Accordion(
        "Custom Extensions / 自定义扩展", open=False, elem_id="custom-extensions"
    ):
        gr.Markdown(
            """
> **Export-only safety boundary / 仅导出安全边界**
>
> This wizard accepts **text-only** samples up to **50 MiB**. It writes no repository
> files, installs no package, imports/executes no uploaded or generated Python, and
> downloads no model. Review the deterministic ZIP and install it manually.
>
> 本向导仅处理不超过 **50 MiB** 的文本样例；不会写入仓库、安装依赖、导入或执行代码，也不会下载模型。
"""
        )

        with gr.Tabs():
            with gr.Tab("Model / 模型"):
                with gr.Row():
                    model_id = gr.Textbox(
                        label="Extension id / 扩展 ID", placeholder="my-qwen-model"
                    )
                    model_operation = gr.Radio(
                        choices=["inject", "edit", "unlearning"],
                        value="inject",
                        label="Bound operation / 绑定操作",
                    )
                    model_family = gr.Textbox(
                        value="qwen2", label="Model family / 模型家族"
                    )
                with gr.Row():
                    model_reference = gr.Textbox(
                        label="HF id or local path / HF ID 或本地路径",
                        placeholder="organization/model-or-/absolute/local/path",
                    )
                    tokenizer_reference = gr.Textbox(
                        label="Tokenizer id/path (optional) / Tokenizer（可选）"
                    )
                with gr.Row():
                    model_dtype = gr.Dropdown(
                        choices=["bfloat16", "float16", "float32"],
                        value="bfloat16",
                        label="dtype",
                    )
                    model_attention = gr.Dropdown(
                        choices=["sdpa", "eager", "flash_attention_2"],
                        value="sdpa",
                        label="Attention implementation / 注意力实现",
                    )
                model_chat_template = gr.Textbox(
                    label="Chat template (optional) / 对话模板（可选）", lines=4
                )
                model_knowledge = gr.CheckboxGroup(
                    choices=list(KNOWLEDGE_TYPES),
                    value=["fact"],
                    label="Knowledge types / 知识类型",
                )
                with gr.Accordion(
                    "Explicit recipe binding / 显式 Recipe 绑定", open=False
                ):
                    _, default_method, default_datasets, default_eval = (
                        _recipe_defaults("inject")
                    )
                    model_method = gr.Textbox(
                        value=default_method, label="Method config / 算法配置"
                    )
                    model_datasets = gr.Textbox(
                        value=default_datasets,
                        label="Dataset-role config JSON / 数据角色配置 JSON",
                        lines=4,
                    )
                    model_evaluation = gr.Textbox(
                        value=default_eval, label="Evaluation config / 评测配置"
                    )
                model_export = gr.Button(
                    "Review and export model ZIP / 审查并导出模型 ZIP",
                    variant="primary",
                )
                model_outputs = _review_outputs("extension-model")

            with gr.Tab("Dataset / 数据集"):
                with gr.Row():
                    dataset_id = gr.Textbox(
                        label="Extension id / 扩展 ID", placeholder="my-domain-data"
                    )
                    dataset_operation = gr.Radio(
                        choices=["inject", "edit", "unlearning"],
                        value="inject",
                        label="Operation / 操作",
                    )
                    dataset_knowledge = gr.Dropdown(
                        choices=list(KNOWLEDGE_TYPES),
                        value="fact",
                        label="Knowledge type / 知识类型",
                    )
                    dataset_format = gr.Dropdown(
                        choices=["alpaca", "sharegpt", "custom"],
                        value="alpaca",
                        label="Input schema / 输入模式",
                    )
                dataset_primary = gr.File(
                    label="Train, edit, or forget sample / 训练、编辑或遗忘样例",
                    file_types=[".json", ".jsonl", ".csv"],
                    type="filepath",
                )
                with gr.Group(visible=False) as dataset_retain_group:
                    gr.Markdown("**Retain role required / 必须提供保留集**")
                    dataset_retain = gr.File(
                        label="Retain QA sample / 保留 QA 样例",
                        file_types=[".json", ".jsonl", ".csv"],
                        type="filepath",
                    )
                dataset_mapping = gr.Textbox(
                    value=json.dumps(
                        default_field_mapping("inject", "alpaca"), indent=2
                    ),
                    label="Canonical → source field mapping / 规范字段 → 源字段映射",
                    lines=8,
                )
                with gr.Row():
                    dataset_preview_btn = gr.Button(
                        "Preview first 3 / 预览前三条", variant="secondary"
                    )
                    dataset_preview_status = gr.Markdown()
                dataset_preview = gr.JSON(
                    label="Canonical preview / 规范化预览", value={}
                )
                with gr.Accordion(
                    "Explicit recipe binding / 显式 Recipe 绑定", open=False
                ):
                    default_model, default_method, _, default_eval = _recipe_defaults(
                        "inject"
                    )
                    dataset_family = gr.Textbox(
                        value="qwen2", label="Supported model family / 支持的模型家族"
                    )
                    dataset_model = gr.Textbox(
                        value=default_model, label="Model config / 模型配置"
                    )
                    dataset_method = gr.Textbox(
                        value=default_method, label="Method config / 算法配置"
                    )
                    dataset_evaluation = gr.Textbox(
                        value=default_eval, label="Evaluation config / 评测配置"
                    )
                dataset_export = gr.Button(
                    "Review and export dataset ZIP / 审查并导出数据集 ZIP",
                    variant="primary",
                )
                dataset_outputs = _review_outputs("extension-dataset")

            with gr.Tab("Algorithm / 算法"):
                gr.Markdown(
                    "**Planned scaffold only / 仅生成 Planned 脚手架。** "
                    "The exported class raises `NotImplementedError` and is not registered."
                )
                with gr.Row():
                    algorithm_id = gr.Textbox(
                        label="Extension id / 扩展 ID", placeholder="my-method"
                    )
                    algorithm_operation = gr.Radio(
                        choices=["inject", "edit", "unlearning"],
                        value="inject",
                        label="Operation / 操作",
                    )
                    algorithm_base = gr.Dropdown(
                        choices=["UnlearnTrainer", "InjectTrainer", "EditTrainer"],
                        value="InjectTrainer",
                        label="Required base class / 必需基类",
                    )
                    algorithm_handler = gr.Textbox(
                        label="Handler class / Handler 类",
                        placeholder="MyMethodTrainer",
                    )
                algorithm_schema = gr.Textbox(
                    value='{"rank": {"type": "int", "default": 8}}',
                    label="Parameter schema JSON / 参数模式 JSON",
                    lines=6,
                )
                with gr.Row():
                    algorithm_families = gr.Textbox(
                        value="qwen2", label="Supported families / 支持的模型家族"
                    )
                    algorithm_knowledge = gr.CheckboxGroup(
                        choices=list(KNOWLEDGE_TYPES),
                        value=["fact"],
                        label="Knowledge types / 知识类型",
                    )
                    algorithm_architecture = gr.Checkbox(
                        value=False,
                        label="Architecture-specific / 架构相关（生成模型 hparams）",
                    )
                with gr.Accordion(
                    "Explicit recipe binding / 显式 Recipe 绑定", open=False
                ):
                    default_model, _, default_datasets, default_eval = _recipe_defaults(
                        "inject"
                    )
                    algorithm_model = gr.Textbox(
                        value=default_model, label="Model config / 模型配置"
                    )
                    algorithm_datasets = gr.Textbox(
                        value=default_datasets,
                        label="Dataset-role config JSON / 数据角色配置 JSON",
                        lines=4,
                    )
                    algorithm_evaluation = gr.Textbox(
                        value=default_eval, label="Evaluation config / 评测配置"
                    )
                    algorithm_requirements = gr.Textbox(
                        value=json.dumps(
                            {
                                "python_packages": [],
                                "files": [],
                                "checkpoints": [],
                                "statistics": [],
                            },
                            indent=2,
                        ),
                        label="Requirements JSON / 依赖 JSON",
                        lines=7,
                    )
                algorithm_export = gr.Button(
                    "Export planned scaffold / 导出 Planned 脚手架", variant="primary"
                )
                algorithm_outputs = _review_outputs("extension-algorithm")

            with gr.Tab("Catalog / 组件目录"):
                gr.Markdown(
                    "Only `smoke_verified`/`benchmark_verified` text extensions may appear "
                    "as default runnable. `configured` and `planned` stay Experimental. / "
                    "只有有真实验证证据的文本扩展可进入默认列表。"
                )
                catalog_refresh = gr.Button(
                    "Refresh Components / 刷新组件", variant="secondary"
                )
                initial_catalog = (
                    catalog_markdown(separate_catalog_records(catalog_records))
                    if catalog_records is not None
                    else refresh_catalog_callback(root)
                )
                catalog_output = gr.Markdown(initial_catalog)

        model_operation.change(
            fn=_model_recipe_updates,
            inputs=[model_operation],
            outputs=[model_method, model_datasets, model_evaluation],
        )
        model_export.click(
            fn=lambda *args: export_model_callback(root, *args),
            inputs=[
                model_id,
                model_operation,
                model_reference,
                tokenizer_reference,
                model_family,
                model_dtype,
                model_attention,
                model_chat_template,
                model_knowledge,
                model_method,
                model_datasets,
                model_evaluation,
            ],
            outputs=list(model_outputs.values()),
        )

        for trigger in (dataset_operation, dataset_format):
            trigger.change(
                fn=_dataset_form_updates,
                inputs=[dataset_operation, dataset_format],
                outputs=[
                    dataset_mapping,
                    dataset_retain_group,
                    dataset_model,
                    dataset_method,
                    dataset_evaluation,
                ],
            )
        dataset_preview_btn.click(
            fn=preview_dataset_callback,
            inputs=[
                dataset_operation,
                dataset_knowledge,
                dataset_format,
                dataset_primary,
                dataset_retain,
                dataset_mapping,
            ],
            outputs=[dataset_preview, dataset_preview_status],
        )
        dataset_export.click(
            fn=lambda *args: export_dataset_callback(root, *args),
            inputs=[
                dataset_id,
                dataset_operation,
                dataset_knowledge,
                dataset_format,
                dataset_primary,
                dataset_retain,
                dataset_mapping,
                dataset_family,
                dataset_model,
                dataset_method,
                dataset_evaluation,
            ],
            outputs=list(dataset_outputs.values()),
        )

        algorithm_operation.change(
            fn=_algorithm_form_updates,
            inputs=[algorithm_operation],
            outputs=[
                algorithm_base,
                algorithm_model,
                algorithm_datasets,
                algorithm_evaluation,
            ],
        )
        algorithm_export.click(
            fn=lambda *args: export_algorithm_callback(root, *args),
            inputs=[
                algorithm_id,
                algorithm_operation,
                algorithm_base,
                algorithm_handler,
                algorithm_schema,
                algorithm_families,
                algorithm_knowledge,
                algorithm_architecture,
                algorithm_model,
                algorithm_datasets,
                algorithm_evaluation,
                algorithm_requirements,
            ],
            outputs=list(algorithm_outputs.values()),
        )
        catalog_refresh.click(
            fn=lambda: refresh_catalog_callback(root),
            inputs=[],
            outputs=[catalog_output],
        )

        components.update(
            {
                "catalog_refresh": catalog_refresh,
                "catalog_output": catalog_output,
                "model_download": model_outputs["download"],
                "dataset_download": dataset_outputs["download"],
                "algorithm_download": algorithm_outputs["download"],
            }
        )
    return components
