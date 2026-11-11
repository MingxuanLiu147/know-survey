"""Shared, offline configuration review before model surgery.

Defaults are candidates, not architecture adaptation. A review digest records
the exact parameters and local inputs a user checked. No weights are loaded,
downloaded, edited or trained while producing this report.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

from omegaconf import OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException

from know_surgery.edit_data import (
    EDITING_DATA_DEFAULTS,
    default_edit_data_path,
    file_sha256,
    read_records,
    resolve_data_path,
    validate_scope_sidecar,
)
from know_surgery.zsre_audit import audit_dataset

LAYER_KEYS = ("layers", "edit_layers", "layer", "layer_id")
LAYERLESS_EDITORS = {"IKEEditor", "SERACEditor"}
AUXILIARY_MODEL_KEYS = (
    "sentence_model_name",
    "scope_model_name_or_path",
    "counterfactual_model_name_or_path",
)
MODULE_KEYS = (
    "rewrite_module_tmp",
    "layer_module_tmp",
    "mlp_module_tmp",
    "inner_params",
    "ln_f_module",
    "lm_head_module",
)
ASSET_KEYS = (
    "archive",
    "demonstration_path",
    "memory_path",
    "projection_path",
    "scope_training_path",
    "edit_network_path",
    "preservation_data_path",
    "mom2_data_path",
)
FILE_KEYS = {
    "data_path",
    "data_files",
    "edit_data_path",
    "eval_data_path",
    "benchmark_data_path",
    "train_path",
    "retain_path",
    "forget_path",
    *ASSET_KEYS,
}


class PreflightError(ValueError):
    """The configuration or its user review cannot be used for execution."""


def _plain_config(cfg, root: Path) -> dict:
    config = (
        OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        if OmegaConf.is_config(cfg)
        else OmegaConf.create(cfg)
    )
    # Hydra runtime metadata is not an experimental parameter.
    with open_dict(config):
        if "paths" in config:
            config.paths.work_dir = str(root)
    return OmegaConf.to_container(config, resolve=True)


def _path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _identity(path: Path) -> dict:
    stat = path.stat()
    result = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    if stat.st_size <= 64 * 1024 * 1024:
        result["sha256"] = file_sha256(path)
    else:
        result["identity_method"] = (
            "size_mtime_ctime; full asset compatibility checked by loader"
        )
    return result


def _local_model(source: str, root: Path, revision=None) -> Path | None:
    candidate = _path(source, root)
    if candidate.is_dir():
        return candidate
    if source.startswith(("/", ".", "~")):
        return None
    for base in os.environ.get("KNOW_SURGERY_MODEL_ROOTS", "").split(os.pathsep):
        if base:
            for suffix in (source, source.rsplit("/", 1)[-1]):
                candidate = _path(base, root) / suffix
                if (candidate / "config.json").is_file():
                    return candidate.resolve()
    if not source.startswith(("/", ".", "~")):
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(source, "config.json", revision=revision)
            if isinstance(cached, str):
                return Path(cached).parent.resolve()
        except (ImportError, OSError, ValueError):
            pass
    return None


def prepare_config(cfg, *, root=None):
    """Resolve configured local data/model roots in memory; never rewrite recipes."""
    root = Path(root or Path.cwd()).resolve()
    with open_dict(cfg):
        if (
            "paths" in cfg
            and OmegaConf.to_container(cfg.paths, resolve=False).get("work_dir")
            == "${hydra:runtime.cwd}"
        ):
            cfg.paths.work_dir = str(root)
        roots = OmegaConf.select(cfg, "paths.data_roots", default=None)
        for group in cfg.get("data", {}).values():
            if not OmegaConf.is_dict(group):
                continue
            for node in group.values():
                if (
                    not OmegaConf.is_dict(node)
                    or node.get("handler") not in EDITING_DATA_DEFAULTS
                ):
                    continue
                args = node.setdefault("args", {})
                path = resolve_data_path(
                    args.get("data_path"),
                    default_edit_data_path(node.handler, args),
                    root=root,
                    data_roots=args.get("data_roots", roots),
                )
                if path is not None:
                    args.data_path = str(path)
        for field in ("model.model_args", "model.tokenizer_args"):
            args = OmegaConf.select(cfg, field)
            if args is None:
                continue
            source = args.get("pretrained_model_name_or_path")
            if source:
                local = _local_model(str(source), root, args.get("revision"))
                if local:
                    args.pretrained_model_name_or_path = str(local)
        params = OmegaConf.select(cfg, "trainer.method_args", default={})
        for key in AUXILIARY_MODEL_KEYS:
            source = params.get(key)
            if source:
                local = _local_model(str(source), root)
                if local:
                    params[key] = str(local)
    return cfg


def inspect_configuration(cfg, *, root=None) -> dict:
    """Return effective parameters, input identities and actionable review issues."""
    root = Path(root or Path.cwd()).resolve()
    errors, warnings, parameters, inputs, datasets = [], [], [], {}, []
    auxiliary_models = []

    def issue(code, key, en, zh, *, warning=False):
        (warnings if warning else errors).append(
            {"code": code, "key": key, "en": en, "zh": zh}
        )

    try:
        config = _plain_config(cfg, root)
    except (ValueError, TypeError, KeyError, OmegaConfBaseException) as exc:
        raise PreflightError(
            f"Cannot resolve configuration / 无法展开配置: {exc}"
        ) from exc
    trainer = config.get("trainer") or {}
    method = trainer.get("handler", "")
    params = trainer.get("method_args") or {}
    model_args = (config.get("model") or {}).get("model_args") or {}
    archive_auxiliaries = method == "SERACEditor" and bool(params.get("archive"))
    source = str(model_args.get("pretrained_model_name_or_path") or "")
    model_dir = (
        _local_model(source, root, model_args.get("revision")) if source else None
    )
    metadata = {}
    if model_dir and (model_dir / "config.json").is_file():
        try:
            metadata = json.loads(
                (model_dir / "config.json").read_text(encoding="utf-8")
            )
            if not isinstance(metadata, dict):
                raise ValueError("config.json must contain an object")
        except (OSError, ValueError) as exc:
            issue(
                "model_metadata_invalid",
                "model",
                str(exc),
                "模型配置读取失败：" + str(exc),
            )
            metadata = {}
        # The text tower supplies editing layers for supported multimodal models.
        metadata = metadata.get("text_config", metadata)
        for path in sorted(model_dir.iterdir()):
            if path.is_file() and path.suffix in {
                ".json",
                ".safetensors",
                ".bin",
                ".model",
                ".txt",
                ".jinja",
            }:
                inputs[str(path)] = _identity(path)
    # Tokenizers/processors can live outside the model directory. Their files
    # affect token boundaries and must invalidate a previous parameter review.
    for component in ("tokenizer_args", "processor_args"):
        args = (config.get("model") or {}).get(component) or {}
        component_source = args.get("pretrained_model_name_or_path")
        local = (
            _local_model(str(component_source), root, args.get("revision"))
            if component_source
            else None
        )
        if local and local != model_dir:
            for path in sorted(local.iterdir()):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".model",
                    ".txt",
                    ".jinja",
                }:
                    inputs[str(path)] = _identity(path)
    for key in AUXILIARY_MODEL_KEYS:
        value = params.get(key)
        if not value:
            continue
        if archive_auxiliaries and key in {
            "scope_model_name_or_path",
            "counterfactual_model_name_or_path",
        }:
            continue  # A self-contained archive supplies both trained models.
        # Lexical IKE does not load its inherited sentence encoder setting.
        if (
            key == "sentence_model_name"
            and params.get("retrieval_backend") == "token_overlap"
        ):
            continue
        local = _local_model(str(value), root)
        auxiliary_models.append(
            {
                "key": "trainer.method_args." + key,
                "source": str(value),
                "local_path": str(local) if local else None,
            }
        )
        if local is None:
            explicit_path = str(value).startswith(("/", ".", "~"))
            issue(
                "auxiliary_model_missing"
                if explicit_path
                else "auxiliary_model_unresolved",
                "trainer.method_args." + key,
                "Auxiliary model is unavailable locally. Select a complete local model before execution; no retrieval fallback is allowed.",
                "本地尚未找到辅助模型，请在运行前选择完整模型；不能自动退回其他检索方式。",
                warning=not explicit_path,
            )
        else:
            # SentenceTransformer pooling/tokenizer settings can be nested.
            # Replacing either these settings or weights invalidates review.
            for path in sorted(local.rglob("*")):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".safetensors",
                    ".bin",
                    ".model",
                    ".txt",
                    ".jinja",
                }:
                    inputs[str(path)] = _identity(path)
    depth = metadata.get(
        "num_hidden_layers", metadata.get("n_layer", metadata.get("num_layers"))
    )
    model_type = metadata.get("model_type")
    uses_layers = method not in LAYERLESS_EDITORS
    needs_layers = uses_layers and any(
        params.get(key) is not None for key in LAYER_KEYS
    )
    if type(depth) is not int or depth < 1:
        depth = None
        issue(
            "model_metadata_unavailable",
            "model.model_args.pretrained_model_name_or_path",
            "Model depth is unavailable locally. Select a local checkpoint/config before reviewing layers.",
            "尚未读取到模型总层数，请选择本地模型并核对层数。",
            warning=not needs_layers
            and config.get("operation", config.get("mode")) != "edit",
        )
    layers = []
    for key in LAYER_KEYS:
        if not uses_layers or params.get(key) is None:
            continue
        values = params[key] if isinstance(params[key], list) else [params[key]]
        layers.extend(values)
        if (
            not values
            or any(
                type(value) is not int
                or value < 0
                or (depth is not None and value >= depth)
                for value in values
            )
            or len({str(v) for v in values}) != len(values)
        ):
            issue(
                "invalid_layers",
                "trainer.method_args." + key,
                f"Choose distinct zero-based layer indices within 0..{depth - 1 if depth else '?' }.",
                f"请填写互不重复的层号，编号从 0 开始，上限为 {depth - 1 if depth else '待核实'}。",
            )
    if method == "ROMEEditor" and len(layers) != 1:
        issue(
            "rome_single_layer",
            "trainer.method_args.layers",
            "ROME requires exactly one edit layer.",
            "ROME 必须指定一个编辑层。",
        )
    intermediate = params.get("v_loss_mode") == "intermediate_logits" or (
        method == "AnyEditEditor"
        and params.get("implementation_variant") == "official_reference"
    )
    loss_layer = params.get("v_loss_layer")
    if intermediate and (
        type(loss_layer) is not int
        or loss_layer < 0
        or (depth is not None and loss_layer >= depth)
        or any(type(layer) is int and loss_layer < layer for layer in layers)
    ):
        issue(
            "invalid_loss_layer",
            "trainer.method_args.v_loss_layer",
            "The intermediate loss layer must exist and be at or after the edit layers.",
            "中间损失读取层必须存在，且不能早于编辑层。",
        )
    command = shlex.split(str(config.get("invocation_command", "")))
    explicit = {arg.lstrip("+").split("=", 1)[0] for arg in command if "=" in arg}
    review_keys = (
        *LAYER_KEYS,
        *MODULE_KEYS,
        "v_loss_layer",
        "v_loss_mode",
        "implementation_variant",
        "strategy",
        "ft_lr",
        "num_steps",
        "v_lr",
        "v_num_grad_steps",
        "edit_lr",
        "n_iter",
        "mom2_adjustment",
        "mom2_dataset",
        "mom2_dataset_config",
        "mom2_n_samples",
        "mom2_max_length",
        "mom2_update_weight",
        "nullspace_threshold",
        "l2",
        "window_size",
        "overlap",
        "retrieval_backend",
        "k",
        *ASSET_KEYS,
    )
    for key in dict.fromkeys((*review_keys, *params)):
        if key in params:
            full_key = "trainer.method_args." + key
            parameters.append(
                {
                    "key": full_key,
                    "value": params[key],
                    "source": "explicit_override"
                    if full_key in explicit
                    else "recipe_default",
                    "active": (key not in LAYER_KEYS or uses_layers)
                    and (key != "v_loss_layer" or intermediate)
                    and not (
                        archive_auxiliaries
                        and key
                        in {
                            "scope_model_name_or_path",
                            "counterfactual_model_name_or_path",
                        }
                    ),
                }
            )
    for key in (
        "learning_rate",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "per_device_eval_batch_size",
        "max_steps",
        "seed",
        "bf16",
    ):
        if key in trainer.get("args", {}):
            parameters.append(
                {
                    "key": "trainer.args." + key,
                    "value": trainer["args"][key],
                    "source": "configuration",
                    "active": True,
                }
            )
    if "edit_reference_device" in config:
        parameters.append(
            {
                "key": "edit_reference_device",
                "value": config["edit_reference_device"],
                "source": "explicit_override"
                if "edit_reference_device" in explicit
                else "configuration",
                "active": bool(trainer.get("args", {}).get("do_eval", False)),
            }
        )
    # Known module families can reject an inherited architecture mapping before
    # weights are loaded. Actual module existence is also checked after loading.
    module_values = []
    for key in MODULE_KEYS:
        value = params.get(key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str) and item != "auto":
                module_values.append((key, item))
    for key, value in module_values:
        mismatch = (
            model_type == "gpt2" and value.startswith(("model.layers", "model.norm"))
        ) or (
            model_type in {"qwen2", "llama", "mistral"}
            and value.startswith(("transformer.h", "transformer.ln_f"))
        )
        if mismatch:
            issue(
                "module_family_mismatch",
                "trainer.method_args." + key,
                "This module path belongs to another model family; edit the mapping.",
                "该模块路径属于其他模型结构，请修改模块映射。",
            )
        for number in re.findall(r"(?:layers|h)\.(\d+)\.", value):
            if depth is not None and int(number) >= depth:
                issue(
                    "module_layer_out_of_range",
                    "trainer.method_args." + key,
                    "The module contains an out-of-range layer index.",
                    "模块路径中的层号超过当前模型范围。",
                )
            elif key == "inner_params" and layers and int(number) not in layers:
                issue(
                    "module_layer_mismatch",
                    "trainer.method_args." + key,
                    "The fixed module layer differs from layers; update both together.",
                    "模块路径中的层号与 layers 不一致，请一起修改。",
                )

    def inspect_files(value, location=(), selected=False, remote=False):
        if isinstance(value, dict):
            for key, item in value.items():
                item_remote = remote
                if key == "hf_args" and isinstance(item, dict):
                    source = str(item.get("path") or "")
                    item_remote = (
                        bool(source)
                        and source
                        not in {
                            "json",
                            "csv",
                            "parquet",
                            "text",
                            "arrow",
                            "imagefolder",
                            "audiofolder",
                            "webdataset",
                        }
                        and not source.startswith(("/", ".", "~"))
                        and not _path(source, root).exists()
                    )
                inspect_files(
                    item,
                    (*location, str(key)),
                    selected or key in FILE_KEYS,
                    item_remote,
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect_files(item, (*location, str(index)), selected, remote)
        elif selected and isinstance(value, str) and value and "://" not in value:
            if remote:
                issue(
                    "remote_input",
                    ".".join(location),
                    "Remote dataset files are identified by configuration only; pin a revision or use a local snapshot for a reproducible benchmark.",
                    "远程数据文件仅按配置识别，正式基准请固定版本或选择本地快照。",
                    warning=True,
                )
                return
            path = _path(value, root)
            key = ".".join(location)
            if glob.has_magic(str(path)):
                matches = [
                    Path(match)
                    for match in sorted(glob.glob(str(path)))
                    if Path(match).is_file()
                ]
                if not matches:
                    issue(
                        "input_missing",
                        key,
                        f"No files match {path}",
                        f"没有文件匹配：{path}",
                    )
                for match in matches:
                    inputs[str(match)] = _identity(match)
                return
            if not path.exists():
                if location[-1] == "projection_path":
                    issue(
                        "projection_to_build",
                        key,
                        "Projection will be computed from the reviewed statistics corpus; budget this work before execution.",
                        "投影矩阵将由已检查的统计语料计算，请将这部分计入运行预算。",
                        warning=True,
                    )
                else:
                    issue(
                        "input_missing",
                        key,
                        f"Local input is missing: {path}. Select an existing file.",
                        f"本地输入不存在：{path}。请指定已有文件。",
                    )
            elif path.is_file():
                inputs[str(path)] = _identity(path)
            else:
                for child in sorted(path.rglob("*")):
                    if child.is_file() and child.suffix in {
                        ".json",
                        ".pt",
                        ".bin",
                        ".safetensors",
                        ".model",
                        ".arrow",
                        ".parquet",
                        ".jsonl",
                    }:
                        inputs[str(child)] = _identity(child)

    inspect_files(config.get("data", {}), ("data",))
    inspect_files(config.get("eval", {}), ("eval",))
    inspect_files(params, ("trainer", "method_args"))
    for role, group in config.get("data", {}).items():
        if not isinstance(group, dict):
            continue
        for name, node in group.items():
            if not isinstance(node, dict) or not node.get("handler"):
                continue
            args = node.get("args") or {}
            value = args.get("data_path")
            if not value or not _path(value, root).is_file():
                continue
            key = f"data.{role}.{name}.args.data_path"
            path = _path(value, root)
            if path.suffix not in {".json", ".jsonl"}:
                continue
            try:
                if node["handler"] == "ZSREDataset":
                    audit = audit_dataset(
                        path,
                        args.get("expected_num_samples"),
                        args.get("expected_sha256"),
                    )
                    for message in audit["errors"]:
                        issue(
                            "dataset_contract",
                            key,
                            message,
                            "数据不符合当前实验约定：" + message,
                        )
                    for message in audit["warnings"]:
                        issue(
                            "dataset_coverage",
                            key,
                            message,
                            "请核对数据覆盖情况：" + message,
                            warning=True,
                        )
                    datasets.append(
                        {
                            k: v
                            for k, v in audit.items()
                            if k not in {"case_identities", "errors", "warnings"}
                        }
                        | {
                            "key": key,
                            "split": args.get("split"),
                            "count_kind": "edit_cases",
                        }
                    )
                    if (
                        method == "WISEMergeEditor"
                        and not all(
                            row.get("scope_training_negatives")
                            for row in read_records(path)
                        )
                        and not args.get("scope_training_path")
                    ):
                        issue(
                            "wise_negatives_missing",
                            key,
                            "WISE needs disjoint scope-training negatives. Supply a reviewed scope_training_path sidecar.",
                            "WISE 缺少独立训练负例，请提供并检查 scope_training_path 辅助文件。",
                        )
                    if args.get("scope_training_path"):
                        validate_scope_sidecar(
                            read_records(path),
                            read_records(_path(args["scope_training_path"], root)),
                        )
                else:
                    datasets.append(
                        {
                            "key": key,
                            "path": str(path),
                            "count": len(read_records(path)),
                            "count_kind": "source_records",
                            "sha256": file_sha256(path),
                            "split": args.get("split"),
                        }
                    )
            except (OSError, ValueError, TypeError) as exc:
                issue("dataset_unreadable", key, str(exc), "数据解析失败：" + str(exc))
    if method == "SERACEditor" and not params.get("archive"):
        issue(
            "serac_archive_required",
            "trainer.method_args.archive",
            "Select a trained, compatible SERAC auxiliary archive.",
            "请选择经过训练且与当前模型匹配的 SERAC 辅助模型。",
        )
    if (
        method in {"MENDEditor", "MENDFactorEditor"}
        and params.get("require_pretrained", True)
        and not params.get("edit_network_path")
    ):
        issue(
            "mend_network_required",
            "trainer.method_args.edit_network_path",
            "Select a trained MEND network compatible with the base model and edited matrices.",
            "请选择与当前基础模型、编辑层和矩阵匹配的已训练 MEND 网络。",
        )
    if method == "MENDFactorEditor" and params.get("edit_network_path"):
        archive = _path(str(params["edit_network_path"]), root)
        receipt = archive.with_name(archive.name + ".metadata.json")
        if receipt.is_file():
            inputs[str(receipt)] = _identity(receipt)
            try:
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if not isinstance(saved, dict):
                    raise ValueError("MEND metadata must be a JSON object")
                training = saved["training_metadata"]
                if not isinstance(training, dict):
                    raise ValueError("MEND training metadata must be an object")
                validation = training["validation"]
                if not isinstance(validation, dict):
                    raise ValueError("MEND validation metadata must be an object")
                contract = training.get("training_contract")
                if not isinstance(contract, dict):
                    raise ValueError("MEND training contract must be an object")
                if (
                    saved.get("schema_version") != 1
                    or saved.get("format") != "know_surgery_factor_mend_v1"
                    or saved.get("implementation") != "factor_mend"
                    or training.get("purpose") != "training"
                    or training.get("completed") is not True
                    or type(training.get("optimizer_steps")) is not int
                    or training["optimizer_steps"] < 1
                    or training["optimizer_steps"] != training.get("requested_steps")
                    or training.get("requested_steps") != contract.get("max_steps")
                    or type(validation.get("sample_count")) is not int
                    or validation["sample_count"] < 1
                    or training.get("initial_network_sha256")
                    == training.get("trained_network_sha256")
                    or training.get("initial_parameters_sha256")
                    == training.get("trained_parameters_sha256")
                ):
                    raise ValueError(
                        "MEND metadata lacks completed training and validation"
                    )
                saved_config = saved["config"]
                if not isinstance(saved_config, dict):
                    raise ValueError("MEND saved config must be an object")
                for key in (
                    "layers",
                    "inner_params",
                    "rank",
                    "n_hidden",
                    "normalize",
                    "shared",
                    "edit_lr",
                    "cedit",
                    "cloc",
                    "max_length",
                    "max_locality_samples",
                ):
                    if saved_config[key] != params.get(key):
                        issue(
                            "mend_network_parameter_mismatch",
                            "trainer.method_args." + key,
                            f"MEND network was trained with {key}={saved_config[key]!r}; select matching settings and review again.",
                            f"MEND 网络训练时 {key}={saved_config[key]!r}，请使用匹配参数并重新检查。",
                        )
                if (
                    saved_config.get("outer_objective")
                    != "first_rephrase_nll_and_attention_mask_locality_kl"
                    or saved_config.get("factor_dtype") != "float32"
                ):
                    raise ValueError(
                        "MEND objective or factor precision is incompatible"
                    )
                if archive.is_file():
                    actual = inputs[str(archive)]
                    if saved.get("checkpoint_size_bytes") != actual["size"] or (
                        actual.get("sha256")
                        and saved.get("checkpoint_sha256") != actual["sha256"]
                    ):
                        raise ValueError(
                            "MEND metadata does not match the checkpoint file"
                        )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issue(
                    "mend_network_metadata_invalid",
                    "trainer.method_args.edit_network_path",
                    str(exc),
                    "MEND 网络说明无效：" + str(exc),
                )
        elif archive.is_file():
            issue(
                "mend_network_metadata_required",
                "trainer.method_args.edit_network_path",
                "Factor MEND requires the .metadata.json saved with the trained network; compact MEND and training-state checkpoints cannot be substituted.",
                "需要训练时生成的 .metadata.json；简化版 MEND 或训练状态检查点不能替代该网络。",
            )

    if config.get("operation", config.get("mode")) == "edit":
        checkpoint_every = config.get("edit_checkpoint_every", 1)
        stop_after = config.get("edit_stop_after_cases")
        if type(checkpoint_every) is not int or checkpoint_every < 1:
            issue(
                "invalid_checkpoint_interval",
                "edit_checkpoint_every",
                "Checkpoint interval must be a positive integer.",
                "检查点间隔必须为正整数。",
            )
        if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
            issue(
                "invalid_stop_after",
                "edit_stop_after_cases",
                "A bounded run needs a positive case count.",
                "分段运行的样本数必须为正整数。",
            )
        resume = config.get("edit_resume_from")
        if resume or stop_after is not None:
            if config.get("edit_type") != "independent":
                issue(
                    "resume_requires_independent",
                    "edit_type",
                    "Case-boundary resume and bounded pauses require independent editing.",
                    "逐例恢复和分段运行仅适用于 independent 编辑。",
                )
        if resume:
            path = _path(str(resume), root)
            path = path / "independent_checkpoint.json" if path.is_dir() else path
            if path.is_file():
                inputs[str(path)] = _identity(path)
            else:
                issue(
                    "resume_checkpoint_missing",
                    "edit_resume_from",
                    f"No independent-edit checkpoint exists at {path}.",
                    f"未找到独立编辑检查点：{path}。",
                )
    if method == "SERACEditor" and params.get("archive"):
        archive = _path(str(params["archive"]), root)
        receipt = archive.with_name(archive.name + ".metadata.json")
        if receipt.is_file():
            inputs[str(receipt)] = _identity(receipt)
            try:
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                saved_config = saved["config"]
                for key in (
                    "scope_threshold",
                    "distance_heads",
                    "distance_scale",
                    "max_length",
                    "soft_weighting",
                ):
                    if saved_config[key] != params.get(key):
                        issue(
                            "serac_archive_parameter_mismatch",
                            "trainer.method_args." + key,
                            f"SERAC archive metadata specifies {saved_config[key]!r}; select matching parameters and review again. The archive loader also validates the embedded values.",
                            f"SERAC 归档参数为 {saved_config[key]!r}，请选择匹配参数后重新检查；加载时还会核验归档内实际值。",
                        )
                training = saved["training_metadata"]
                if not isinstance(training, dict):
                    raise ValueError("SERAC training metadata must be a JSON object")
                if (
                    saved.get("state_version") != 3
                    or saved.get("variant") != "supervised_biencoder_causal_v1"
                    or training.get("status") != "trained"
                    or type(training.get("optimizer_steps")) is not int
                    or training["optimizer_steps"] < 1
                ):
                    raise ValueError(
                        "SERAC archive metadata does not describe a trained v3 archive"
                    )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                issue(
                    "serac_archive_metadata_invalid",
                    "trainer.method_args.archive",
                    str(exc),
                    "SERAC 归档说明无效：" + str(exc),
                )
        elif archive.is_file():
            issue(
                "serac_archive_metadata_unavailable",
                "trainer.method_args.archive",
                "No lightweight archive metadata is available. Compatibility and embedded parameters must pass validation when the archive is loaded.",
                "归档没有轻量参数说明，加载时必须核验兼容性与归档内参数。",
                warning=True,
            )
    if (
        method == "IKEEditor"
        and params.get("protocol") == "paper_copy_update_retain"
        and not params.get("demonstration_path")
    ):
        issue(
            "ike_demonstrations_required",
            "trainer.method_args.demonstration_path",
            "Select an independent IKE demonstration corpus.",
            "请选择与测试集隔离的 IKE 示例库。",
        )
    if params.get("mom2_adjustment") is False:
        issue(
            "adapted_covariance",
            "trainer.method_args.mom2_adjustment",
            "Covariance adjustment is disabled; this is an adapted update rule.",
            "已关闭协方差调整，本次使用适配后的更新规则。",
            warning=True,
        )
    if (
        method in {"ROMEEditor", "MEMITEditor", "AlphaEditEditor", "AnyEditEditor"}
        and (params.get("mom2_adjustment") or method == "AlphaEditEditor")
        and not params.get("mom2_data_path")
    ):
        issue(
            "statistics_corpus_unbound",
            "trainer.method_args.mom2_data_path",
            "Statistics corpus is selected by dataset/config name only. Bind a local corpus for offline reproducibility and budget its actual sample count and sequence length.",
            "统计语料目前仅按名称/版本选择，请绑定本地语料，并核对实际样本数、序列长度和统计计算预算。",
            warning=True,
        )
    if (
        method in {"ROMEEditor", "MEMITEditor", "AlphaEditEditor"}
        and (params.get("mom2_adjustment") or method == "AlphaEditEditor")
        and params.get("mom2_max_length") is None
        and metadata.get("max_position_embeddings", 0) > 4096
    ):
        issue(
            "statistics_context_budget",
            "trainer.method_args.mom2_max_length",
            "Statistics inherit the model context limit; the editing dataset's max_length does not cap this work. Review mom2_max_length and mom2_batch_tokens before computing statistics.",
            "统计阶段沿用模型上下文上限，不受编辑数据 max_length 限制；请另行检查 mom2_max_length、mom2_batch_tokens 和显存预算。",
            warning=True,
        )
    if params.get("retrieval_backend") == "token_overlap":
        issue(
            "lexical_retrieval",
            "trainer.method_args.retrieval_backend",
            "This recipe uses lexical retrieval, not semantic demonstration retrieval.",
            "此配置使用词项检索，请核对是否符合实验协议。",
            warning=True,
        )

    statistics_attention = params.get("mom2_attention_policy", "model_default")
    statistics_placement = params.get("mom2_placement_policy", "model_default")
    if statistics_attention not in {"model_default", "qwen2_eager_query_chunked"}:
        issue(
            "statistics_attention_policy_invalid",
            "trainer.method_args.mom2_attention_policy",
            "Select a supported statistics attention policy.",
            "请选择已支持的统计注意力实现。",
        )
    if statistics_placement not in {"model_default", "qwen2_prefix"}:
        issue(
            "statistics_placement_policy_invalid",
            "trainer.method_args.mom2_placement_policy",
            "Select a supported statistics placement policy.",
            "请选择已支持的统计模型放置方式。",
        )
    if (
        statistics_attention == "qwen2_eager_query_chunked"
        or statistics_placement == "qwen2_prefix"
    ):
        if model_type != "qwen2":
            issue(
                "statistics_model_family_mismatch",
                "trainer.method_args.mom2_attention_policy",
                "This statistics policy requires Qwen2. Review the policy and layer mapping when changing models.",
                "当前统计实现仅支持 Qwen2。更换模型时，请重新检查统计方式及层号。",
            )
    if statistics_attention == "qwen2_eager_query_chunked":
        chunk_size = params.get("mom2_query_chunk_size", 256)
        if type(chunk_size) is not int or chunk_size <= 0:
            issue(
                "statistics_query_chunk_invalid",
                "trainer.method_args.mom2_query_chunk_size",
                "Statistics query chunk size must be a positive integer.",
                "统计查询分块大小必须为正整数。",
            )
        if (
            config.get("model", {}).get("model_args", {}).get("attn_implementation")
            != "eager"
        ):
            issue(
                "statistics_eager_required",
                "model.model_args.attn_implementation",
                "The query-chunked statistics implementation requires eager model attention.",
                "当前分块统计实现要求模型使用 eager 注意力。",
            )
        issue(
            "statistics_numerical_variant",
            "trainer.method_args.mom2_attention_policy",
            "Query chunking is an explicit numerical variant. Its caches are separate from unchunked eager statistics.",
            "查询分块属于明确的数值实现变体；其缓存与未分块 eager 统计分别保存。",
            warning=True,
        )

    def semantic_config(value):
        if isinstance(value, dict):
            return {
                key: semantic_config(item)
                for key, item in value.items()
                if key
                not in {
                    "invocation_command",
                    "task_name",
                    "hydra",
                    "output_dir",
                    "work_dir",
                    "logging_dir",
                    "reviewed_config_hash",
                }
            }
        if isinstance(value, list):
            return [semantic_config(item) for item in value]
        return value

    semantic = semantic_config(config)
    identity = {"schema_version": "1.0", "configuration": semantic, "inputs": inputs}
    digest = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "valid": not errors,
        "review_hash": digest,
        "model": {
            "source": source,
            "local_path": str(model_dir) if model_dir else None,
            "model_type": model_type,
            "num_hidden_layers": depth,
        },
        "method": method,
        "operation": config.get("operation", config.get("mode")),
        "parameters": parameters,
        "datasets": datasets,
        "auxiliary_models": auxiliary_models,
        "inputs": inputs,
        "errors": errors,
        "warnings": warnings,
        "requires_confirmation": True,
        "module_check": "checked against actual model modules before editing",
    }


def require_review(report: dict, reviewed_hash: str | None) -> None:
    """Reject invalid or unreviewed parameters; confirmation cannot bypass errors."""
    if not report["valid"]:
        raise PreflightError(
            "Preflight failed / 运行前检查未通过:\n"
            + "\n".join(
                f"{issue['key']}: {issue['en']} / {issue['zh']}"
                for issue in report["errors"]
            )
        )
    if reviewed_hash != report["review_hash"]:
        raise PreflightError(
            "Review the actual model, method, layers and data with --preflight, then pass --reviewed-config <review_hash>. Changing inputs invalidates the review. / 请先预检、检查并修改层数等参数，再明确确认当前配置；参数或输入变化后须重新确认。"
        )


def validate_model_modules(cfg, model) -> None:
    """Verify configured module paths on the loaded model before any update."""
    params = cfg.trainer.get("method_args", {})
    layers = params.get("layers", [])
    for key in MODULE_KEYS:
        value = params.get(key)
        if not isinstance(value, str) or not value or value == "auto":
            continue
        paths = [value.format(layer) for layer in layers] if "{}" in value else [value]
        for path in paths:
            try:
                model.get_submodule(path)
            except (AttributeError, KeyError) as exc:
                raise PreflightError(
                    f"Configured module {key}={path} does not exist in this model. / 当前模型不存在该模块，请修改配置。"
                ) from exc
