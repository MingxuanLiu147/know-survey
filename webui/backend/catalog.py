"""Repository-backed capability catalog and command composition."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .schemas import ComposeRequest


ROOT = Path(__file__).resolve().parents[2]
OPERATION_DIR = {"unlearning": "unlearn", "inject": "inject", "edit": "edit"}
DATASET_ROLES = {
    "unlearning": ("forget", "retain"),
    "inject": ("train",),
    "edit": ("edit",),
}
EVALUATION_SUPPORT = {
    "edit": ("text", ("edit",)),
    "edit_benchmark": ("text", ("edit",)),
    "inject": ("text", ("inject",)),
    "tofu": ("text", ("unlearning",)),
    "muse": ("text", ("unlearning",)),
    "lm_eval": ("text", ("unlearning", "inject", "edit")),
    "clear": ("multimodal", ("unlearning",)),
    "mllmu": ("multimodal", ("unlearning",)),
    "mm_edit": ("multimodal", ("edit",)),
    "mm_edit_main": ("multimodal", ("edit",)),
    "mm_edit_portability_only": ("multimodal", ("edit",)),
}
OVERRIDE_KEY = re.compile(r"^\+{0,2}[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class CatalogError(ValueError):
    """Raised when a UI request references an unsupported repository ID."""


def _yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _component_name(reference: str | None, group: str) -> str | None:
    if not reference:
        return None
    path = PurePosixPath(reference)
    prefix = PurePosixPath("configs") / group
    try:
        relative = path.relative_to(prefix)
    except ValueError:
        return None
    return str(relative.with_suffix(""))


def display_command(argv: list[str]) -> str:
    """Render a public command while the process uses the active interpreter."""

    return shlex.join(["python", *argv[1:]])


def _declared_unlearning_role(name: str) -> str | None:
    """Recognize explicit forget/retain names without classifying generic data."""
    roles = set(re.findall(r"(?:^|[_-])(forget|retain)(?=$|[_-]|\d)", name.lower()))
    return next(iter(roles)) if len(roles) == 1 else None


class RepositoryCatalog:
    """Read-only view over Hydra configs and curated extension recipes."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.configs = self.root / "configs"

    def _scan(self, group: str, *, recursive: bool = True) -> list[str]:
        base = self.configs / group
        if not base.exists():
            return []
        iterator = base.rglob("*.yaml") if recursive else base.glob("*.yaml")
        return sorted(
            path.relative_to(base).with_suffix("").as_posix() for path in iterator
        )

    def models(self) -> list[dict[str, Any]]:
        rows = []
        for name in self._scan("model", recursive=False):
            config = _yaml(self.configs / "model" / f"{name}.yaml")
            modality = (
                "text"
                if config.get("model_args") and config.get("tokenizer_args")
                else "multimodal"
            )
            model_args = (
                config.get("model_args")
                if isinstance(config.get("model_args"), dict)
                else {}
            )
            source = model_args.get("pretrained_model_name_or_path")
            rows.append(
                {
                    "id": name,
                    "modality": modality,
                    "source": source,
                    "local": bool(source and str(source).startswith(("/", "."))),
                    "status": "configured",
                }
            )
        return rows

    def trainers(self) -> dict[str, list[dict[str, str]]]:
        from src.mm_registry import MM_UNLEARNING_ALGORITHMS

        rows: dict[str, list[dict[str, str]]] = {key: [] for key in OPERATION_DIR}
        for operation, group in (
            ("unlearning", "trainer"),
            ("inject", "trainer/inject"),
            ("edit", "trainer/edit"),
        ):
            for name in self._scan(group, recursive=False):
                if name in {"finetune", "base_inject", "base_editor"}:
                    continue
                config = _yaml(self.configs / group / f"{name}.yaml")
                handler = str(config.get("handler", name))
                multimodal = name.startswith("MM")
                if operation == "unlearning" and multimodal:
                    registration = MM_UNLEARNING_ALGORITHMS.get(handler)
                    if not registration or not registration.runnable:
                        continue
                rows[operation].append(
                    {
                        "id": name
                        if operation == "unlearning"
                        else f"{operation}/{name}",
                        "status": "configured",
                        "modality": "multimodal" if multimodal else "text",
                    }
                )
        return rows

    def datasets(self) -> list[dict[str, Any]]:
        rows = []
        for name in self._scan("data/datasets", recursive=False):
            config = _yaml(self.configs / "data" / "datasets" / f"{name}.yaml")
            handlers = [
                value.get("handler")
                for value in config.values()
                if isinstance(value, dict) and value.get("handler")
            ]
            lowered = name.lower()
            operations = []
            if "forget" in lowered or "retain" in lowered or "unlearn" in lowered:
                operations.append("unlearning")
            if "inject" in lowered or "train" in lowered or "alpaca" in lowered:
                operations.append("inject")
            if "edit" in lowered or any(
                "Editing" in str(item) or str(item).startswith(("MMEdit", "MMKE"))
                for item in handlers
            ):
                operations.append("edit")
            rows.append(
                {
                    "id": name,
                    "unlearning_role": _declared_unlearning_role(name),
                    "handlers": handlers,
                    "modality": "multimodal"
                    if (
                        config.get("benchmark")
                        or name.startswith("Canonical_MM_")
                        or any(
                            str(item).startswith(("MM", "VLKEB")) for item in handlers
                        )
                    )
                    else "text",
                    "operations": operations or ["unlearning", "inject", "edit"],
                    "status": "configured",
                }
            )
        return rows

    def evaluation_support(self) -> list[dict[str, Any]]:
        """Expose built-in evaluator scopes while leaving extension IDs available."""
        return [
            {"id": name, "modality": modality, "operations": list(operations)}
            for name in self._scan("eval", recursive=False)
            if name in EVALUATION_SUPPORT
            for modality, operations in [EVALUATION_SUPPORT[name]]
        ]

    def experiments(self) -> list[dict[str, str]]:
        rows = []
        base = self.configs / "experiment"
        paths = sorted(base.rglob("*.yaml")) if base.exists() else []
        for path in paths:
            relative = path.relative_to(base).with_suffix("").as_posix()
            operation = relative.split("/", 1)[0]
            public = "unlearning" if operation == "unlearn" else operation
            if public in OPERATION_DIR:
                config = _yaml(path)
                model = next(
                    (
                        value
                        for item in config.get("defaults", [])
                        if isinstance(item, dict)
                        for key, value in item.items()
                        if key in {"model", "/model", "override /model"}
                    ),
                    None,
                )
                model_cfg = (
                    _yaml(self.configs / "model" / f"{model}.yaml") if model else {}
                )
                modality = config.get("modality") or (
                    "multimodal"
                    if "multimodal" in path.parts or model_cfg.get("model_family")
                    else "text"
                )
                rows.append(
                    {
                        "id": relative,
                        "operation": public,
                        "modality": modality,
                        "status": "configured",
                    }
                )
        return rows

    def recipes(self) -> list[dict[str, Any]]:
        payload = _yaml(self.configs / "extensions" / "recipes.yaml")
        rows = []
        for raw in payload.get("recipes", []):
            if not isinstance(raw, dict):
                continue
            experiment = raw.get("experiment")
            references = [
                raw.get("model"),
                raw.get("method"),
                raw.get("evaluation"),
                *(raw.get("datasets") or {}).values(),
            ]
            files_exist = all(
                isinstance(item, str) and (self.root / item).is_file()
                for item in references
                if item is not None
            )
            experiment_exists = bool(
                experiment
                and (self.configs / "experiment" / f"{experiment}.yaml").is_file()
            )
            rows.append(
                {
                    "id": raw.get("id"),
                    "operation": raw.get("operation"),
                    "modality": raw.get("modality", "text"),
                    "experiment": experiment,
                    "model": _component_name(raw.get("model"), "model"),
                    "trainer": _component_name(raw.get("method"), "trainer"),
                    "datasets": {
                        role: _component_name(value, "data/datasets")
                        for role, value in (raw.get("datasets") or {}).items()
                    },
                    "evaluation": _component_name(raw.get("evaluation"), "eval"),
                    "model_family": raw.get("model_family"),
                    "knowledge_types": raw.get("knowledge_types") or [],
                    "status": (
                        "configured" if files_exist and experiment_exists else "planned"
                    ),
                    "default_runnable": bool(files_exist and experiment_exists),
                }
            )
        return rows

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "operations": [
                {
                    "id": "unlearning",
                    "label": {"en": "Unlearning", "zh": "知识遗忘"},
                    "accent": "coral",
                    "metrics": [
                        "forget efficacy",
                        "retention",
                        "utility",
                        "privacy",
                    ],
                },
                {
                    "id": "inject",
                    "label": {"en": "Injection", "zh": "知识注入"},
                    "accent": "teal",
                    "metrics": ["target efficacy", "retention", "runtime"],
                },
                {
                    "id": "edit",
                    "label": {"en": "Editing", "zh": "知识编辑"},
                    "accent": "gold",
                    "metrics": [
                        "reliability",
                        "generalization",
                        "locality",
                        "portability",
                    ],
                },
            ],
            "models": self.models(),
            "trainers": self.trainers(),
            "datasets": self.datasets(),
            "evaluations": self._scan("eval", recursive=False),
            "evaluation_support": self.evaluation_support(),
            "experiments": self.experiments(),
            "recipes": self.recipes(),
        }

    def recipe(self, recipe_id: str) -> dict[str, Any]:
        for recipe in self.recipes():
            if recipe["id"] == recipe_id:
                return recipe
        raise CatalogError(f"Unknown recipe id: {recipe_id}")

    @staticmethod
    def _require(value: str | None, allowed: set[str], label: str) -> str:
        if not value or value not in allowed:
            raise CatalogError(f"Unknown or missing {label}: {value!r}")
        return value

    def validate_request(self, request: ComposeRequest) -> None:
        snapshot = self.snapshot()
        if request.recipe_id:
            recipe = self.recipe(request.recipe_id)
            if recipe["modality"] != request.modality:
                raise CatalogError(
                    "Recipe modality does not match the request. / 配方模态与请求不一致。"
                )
            if recipe["operation"] != request.operation:
                raise CatalogError("Recipe operation does not match the request.")
            if recipe["status"] != "configured":
                raise CatalogError("Recipe is not currently composable.")
        elif request.experiment:
            experiments = {
                item["id"]: (item["operation"], item["modality"])
                for item in snapshot["experiments"]
            }
            if experiments.get(request.experiment) != (
                request.operation,
                request.modality,
            ):
                raise CatalogError(
                    "Experiment is unknown or belongs to another operation."
                )
        # A model group cannot override an experiment's inline checkpoint path.
        # Refuse an ambiguous cross-model change instead of displaying one model
        # while loading another. Matching-family fine-tuned checkpoints remain valid.
        base_experiment = (
            self.recipe(request.recipe_id)["experiment"]
            if request.recipe_id
            else request.experiment
        )
        if base_experiment and request.model:
            base_config = _yaml(self.configs / "experiment" / f"{base_experiment}.yaml")
            inline_model = base_config.get("model")
            if isinstance(inline_model, dict) and inline_model:
                base_model = next(
                    (
                        value
                        for item in base_config.get("defaults", [])
                        if isinstance(item, dict)
                        for key, value in item.items()
                        if key.replace("override ", "").strip("/") == "model"
                    ),
                    None,
                )
                if base_model and base_model != request.model:
                    raise CatalogError(
                        "This experiment includes model-specific settings. Choose a matching experiment "
                        "or manual composition to change models. / 该实验包含模型专用设置，请选择匹配实验或手动配置。"
                    )
        has_base = bool(request.recipe_id or request.experiment)
        models = {
            item["id"]
            for item in snapshot["models"]
            if item["modality"] == request.modality
        }
        trainers = {
            item["id"]
            for item in snapshot["trainers"][request.operation]
            if item["modality"] == request.modality
        }
        if request.model is not None or not has_base:
            self._require(request.model, models, "model")
        if request.trainer is not None or not has_base:
            self._require(request.trainer, trainers, "trainer")
        if (
            request.modality == "multimodal"
            and request.operation == "unlearning"
            and not has_base
        ):
            raise CatalogError(
                "Use a multimodal benchmark experiment for unlearning. / 多模态遗忘请选择对应基准实验。"
            )
        if request.modality == "multimodal" and request.operation == "inject":
            if request.evaluation is not None:
                raise CatalogError(
                    "Multimodal injection uses its configured probes, not a text evaluation suite. / 多模态注入使用配置中的探针。"
                )
        elif request.evaluation is not None or not has_base:
            evaluations = set(snapshot["evaluations"])
            self._require(request.evaluation, evaluations, "evaluation")
            support = EVALUATION_SUPPORT.get(request.evaluation)
            if support and (
                request.modality != support[0] or request.operation not in support[1]
            ):
                raise CatalogError(
                    f"Evaluation {request.evaluation!r} does not support "
                    f"{request.modality}/{request.operation}; choose a matching evaluation. / "
                    "该评测与当前操作或模态不匹配，请选择对应评测。"
                )
        datasets = {
            item["id"]
            for item in snapshot["datasets"]
            if item["modality"] == request.modality
            and request.operation in item["operations"]
        }
        required_roles = set(DATASET_ROLES[request.operation])
        if not set(request.datasets).issubset(required_roles) or (
            not has_base and set(request.datasets) != required_roles
        ):
            raise CatalogError(
                f"{request.operation} requires dataset roles: {', '.join(sorted(required_roles))}"
            )
        for role, value in request.datasets.items():
            self._require(value, datasets, "dataset")
            declared_role = _declared_unlearning_role(value)
            if request.operation == "unlearning" and declared_role not in (None, role):
                raise CatalogError(
                    f"Dataset {value!r} is declared for {declared_role}, not {role}. / "
                    "遗忘集与保留集的用途不匹配，请检查数据集选择。"
                )
        if request.modality == "multimodal" and request.operation != "edit":
            replacements = {
                "trainer.args.learning_rate": "trainer.args.lr",
                "trainer.args.num_train_epochs": "trainer.args.num_epochs",
                "trainer.args.per_device_train_batch_size": "data.batch_size",
            }
            for key in request.overrides:
                if key.lstrip("+") in replacements:
                    raise CatalogError(
                        f"For multimodal training use {replacements[key.lstrip('+')]}. / 多模态训练请使用该字段。"
                    )
        self._validate_overrides(request.overrides, smoke=request.smoke)

    def validate_composed(self, request: ComposeRequest, resolved_config: str) -> None:
        """Reject known runtime prerequisites after experiment defaults are merged.

        These checks do not load models or certify arbitrary model/method pairs.
        They prevent known mismatches from passing a configuration-only preview.
        """
        config = yaml.safe_load(resolved_config)
        if not isinstance(config, dict):
            raise CatalogError("Expected a composed configuration. / 未得到完整配置。")
        self._validate_local_model_weights(config.get("model") or {})
        data = config.get("data") or {}
        if request.operation == "unlearning" and request.modality == "text":
            scopes = {}
            for role in ("forget", "retain"):
                scope = data.get(role)
                if not isinstance(scope, dict) or not scope:
                    raise CatalogError(
                        f"Unlearning requires explicit {role} data. / 遗忘操作需要分别配置遗忘集与保留集。"
                    )
                # Configuration keys may differ while the same dataset is loaded.
                scopes[role] = [
                    node.get("args", {})
                    for node in scope.values()
                    if isinstance(node, dict) and node.get("handler")
                ]
                for name in scope:
                    declared = _declared_unlearning_role(name)
                    if declared not in (None, role):
                        raise CatalogError(
                            f"Composed {role} data contains {name!r}, declared for {declared}. / "
                            "实验配置中的遗忘集与保留集用途不匹配。"
                        )
            if scopes["forget"] and scopes["forget"] == scopes["retain"]:
                raise CatalogError(
                    "Forget and retain resolve to identical dataset inputs; choose separate scopes. / "
                    "遗忘集与保留集指向相同输入，请分别选择不同的数据范围。"
                )
        if request.modality != "text":
            return
        trainer = config.get("trainer") or {}
        method = trainer.get("method_args") or {}
        handler = trainer.get("handler")
        model = request.model or ""
        source = str(
            (config.get("model") or {})
            .get("model_args", {})
            .get("pretrained_model_name_or_path", "")
        )
        # Prefer the actual source if it explicitly identifies a registered family.
        family_hint = source.rsplit("/", 1)[-1].lower()
        if not family_hint.startswith(("gpt2", "qwen", "llama")):
            family_hint = model.lower()
        if request.operation == "inject" and family_hint.startswith("gpt2"):
            targets = method.get("target_modules")
            foreign_projections = {
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            }
            if (
                handler in {"LoRATrainer", "AdaLoRATrainer", "DoRATrainer"}
                and isinstance(targets, list)
                and targets
                and all(
                    isinstance(target, str)
                    and target.rsplit(".", 1)[-1] in foreign_projections
                    for target in targets
                )
            ):
                raise CatalogError(
                    "GPT-2 has no q_proj/v_proj-style modules. Select a GPT-2 experiment "
                    "or set trainer.method_args.target_modules to its actual modules (for example [c_attn]). / "
                    "GPT-2 不包含所选投影层，请使用 GPT-2 实验配置或指定实际模块（如 [c_attn]）。"
                )
        if request.operation != "edit":
            return
        checkpoint_field = None
        if handler == "MENDEditor" and method.get("require_pretrained", True):
            checkpoint_field = "edit_network_path"
        elif handler == "SERACEditor":
            checkpoint_field = "archive"
        if checkpoint_field:
            value = method.get(checkpoint_field)
            if not isinstance(value, str) or not value.strip() or value == "???":
                raise CatalogError(
                    f"{handler} requires trainer.method_args.{checkpoint_field}; "
                    "provide its trained auxiliary checkpoint before execution. / "
                    "该方法需要已训练的辅助权重，请先填写对应 checkpoint 路径。"
                )
            if (
                "${" not in value
                and not (self.root / Path(value).expanduser()).is_file()
            ):
                raise CatalogError(
                    f"trainer.method_args.{checkpoint_field} is not an existing local file. / "
                    "辅助权重文件不存在，请检查 checkpoint 路径。"
                )
        if handler not in {
            "ROMEEditor",
            "MEMITEditor",
            "AlphaEditEditor",
            "AnyEditEditor",
        }:
            return
        expected_prefix = (
            "transformer.h."
            if family_hint.startswith("gpt2")
            else "model.layers."
            if family_hint.startswith(("qwen", "llama"))
            else None
        )
        for key in ("rewrite_module_tmp", "layer_module_tmp"):
            value = method.get(key)
            if (
                expected_prefix
                and isinstance(value, str)
                and not value.startswith(expected_prefix)
            ):
                raise CatalogError(
                    f"{key}={value!r} does not match the selected model family; "
                    "choose a model-specific experiment or supply the correct layer mapping. / "
                    "编辑层映射与所选模型不匹配，请使用对应模型的实验配置或修改层映射。"
                )

    def _validate_local_model_weights(self, model: dict[str, Any]) -> None:
        """Check explicit local Transformers sources without downloading weights."""
        handler = model.get("model_handler", "AutoModelForCausalLM")
        args = model.get("model_args") or model
        if handler not in {
            "AutoModelForCausalLM",
            "ProbedLlamaForCausalLM",
        } or args.get("trust_remote_code"):
            return
        source = args.get("pretrained_model_name_or_path", model.get("model_path"))
        if (
            not isinstance(source, str)
            or "${" in source
            or args.get("state_dict") is not None
        ):
            return
        path = self.root / Path(source).expanduser()
        if not source.startswith(("/", "./", "../", "~/")) and not path.exists():
            return  # Hugging Face IDs are allowed; this check never fetches them.
        if args.get("from_tf") and Path(str(path) + ".index").is_file():
            return  # TensorFlow checkpoints may be addressed by their prefix.
        if path.is_file():
            return  # Explicit supported conversion/checkpoint files reach the loader.
        path = path / str(args.get("subfolder") or "")
        if not path.is_dir():
            raise CatalogError(
                "Local model directory does not exist. / 本地模型目录不存在，请检查模型路径。"
            )
        names = {
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        }
        variant = args.get("variant")
        if variant:
            names.update(
                {f"model.{variant}.safetensors", f"pytorch_model.{variant}.bin"}
            )
        if args.get("gguf_file"):
            names.add(str(args["gguf_file"]))
        if (path / "adapter_config.json").is_file():
            names.update({"adapter_model.safetensors", "adapter_model.bin"})
        if any(
            (path / name).is_file() and (path / name).stat().st_size for name in names
        ):
            return
        for pattern in ("model*.index*.json", "pytorch_model*.index*.json"):
            for index in path.glob(pattern):
                try:
                    shards = set(
                        json.loads(index.read_text()).get("weight_map", {}).values()
                    )
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
                if shards and all(
                    isinstance(shard, str)
                    and (path / shard).is_file()
                    and (path / shard).stat().st_size
                    for shard in shards
                ):
                    return
        raise CatalogError(
            "Local model directory has no complete model weights or adapter weights; "
            "config/tokenizer files alone cannot load a model. Select a complete checkpoint. / "
            "本地模型目录缺少完整权重，只有配置或分词器文件无法加载模型，请选择已完整保存的模型。"
        )

    @staticmethod
    def _validate_overrides(overrides: dict[str, Any], *, smoke: bool = False) -> None:
        if len(overrides) > 64:
            raise CatalogError("Too many Hydra overrides.")
        forbidden = ("hydra.", "+hydra.", "++hydra.")
        interpolation = "$" + "{"
        owned_fields = {
            "operation",
            "modality",
            "mode",
            "entrypoint",
            "defaults",
            "hydra",
            "task_name",
            "model",
            "model.model_handler",
            "trainer",
            "trainer.handler",
            "trainer._target_",
            "experiment",
            "eval",
            "data",
            "trainer.args.seed",
            "paths",
            "paths.root_dir",
            "paths.output_dir",
            "output_dir",
            "trainer.args.output_dir",
        }
        for key, value in overrides.items():
            if not OVERRIDE_KEY.fullmatch(key) or key.startswith(forbidden):
                raise CatalogError(f"Unsafe override key: {key!r}")
            normalized = key.lstrip("+")
            if normalized == "data.max_samples" and (
                type(value) is not int or value < 1
            ):
                raise CatalogError(
                    "data.max_samples must be a positive integer. / 样本数必须为正整数。"
                )
            if normalized in owned_fields or normalized.endswith(".output_dir"):
                raise CatalogError(
                    f"Override {key!r} conflicts with a WebUI task field. "
                    "Use the corresponding form field; output paths follow task_name."
                )
            parts = normalized.split(".")
            if (
                normalized == "trainer.args"
                or (
                    parts[0] == "data"
                    and normalized != "data.max_samples"
                    and (len(parts) <= 3 or parts[-1] == "handler")
                )
                or (parts[0] == "eval" and len(parts) == 2)
            ):
                raise CatalogError(
                    f"Override {key!r} replaces a typed configuration group. "
                    "Use individual method parameters or the dataset selection fields."
                )
            if smoke and normalized in {"max_edits", "trainer.args.max_steps"}:
                raise CatalogError(
                    f"Override {key!r} conflicts with the selected run limits. "
                    "Change the run limits before setting a larger value."
                )
            rendered = (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else str(value)
            )
            if len(rendered) > 4096 or any(
                token in rendered for token in ("\x00", "\n", "\r", interpolation)
            ):
                raise CatalogError(f"Unsafe override value for {key!r}")

    def build_argv(
        self, request: ComposeRequest, *, preview: bool = False
    ) -> list[str]:
        self.validate_request(request)
        argv = [
            sys.executable,
            "src/surgery.py",
            "--config-name=surgery.yaml",
            f"operation={request.operation}",
            f"modality={request.modality}",
        ]
        if request.recipe_id:
            argv.append(f"experiment={self.recipe(request.recipe_id)['experiment']}")
        elif request.experiment:
            argv.append(f"experiment={request.experiment}")
        # Explicit typed selections override a recipe's defaults in every entrypoint.
        if request.model is not None:
            argv.append(f"model={request.model}")
        if request.trainer is not None:
            argv.append(f"trainer={request.trainer}")
        if request.evaluation is not None:
            argv.append(f"eval={request.evaluation}")
        for role, dataset in request.datasets.items():
            if request.modality == "multimodal" and request.operation == "inject":
                argv.append(f"data/datasets@data={dataset}")
            else:
                argv.append(f"data/datasets@data.{role}={dataset}")
        task_prefix = (
            "++"
            if request.modality == "multimodal" and request.operation == "unlearning"
            else ""
        )
        argv.append(f"{task_prefix}task_name={request.task_name}")
        if request.modality == "multimodal" and request.operation == "unlearning":
            argv.append(f"++output_dir=saves/unlearning/{request.task_name}")
        argv.append(f"++trainer.args.seed={request.seed}")
        if request.smoke:
            if request.operation == "edit":
                argv.append("max_edits=1")
            else:
                argv.append("++trainer.args.max_steps=1")
        for key, value in request.overrides.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                rendered = str(value)
            argv.append(f"{key}={rendered}")
        if preview:
            argv.extend(["--cfg", "job"])
        return argv

    def command_preview(self, request: ComposeRequest) -> str:
        return display_command(self.build_argv(request))
