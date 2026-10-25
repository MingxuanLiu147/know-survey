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
        else:
            models = {
                item["id"]
                for item in snapshot["models"]
                if item["modality"] == request.modality
            }
            self._require(request.model, models, "model")
            trainers = {
                item["id"]
                for item in snapshot["trainers"][request.operation]
                if item["modality"] == request.modality
            }
            self._require(request.trainer, trainers, "trainer")
            if request.modality == "multimodal" and request.operation == "unlearning":
                raise CatalogError(
                    "Use a multimodal benchmark experiment for unlearning. / 多模态遗忘请选择对应基准实验。"
                )
            if request.modality == "multimodal" and request.operation == "inject":
                if request.evaluation is not None:
                    raise CatalogError(
                        "Multimodal injection uses its configured probes, not a text evaluation suite. / 多模态注入使用配置中的探针。"
                    )
            else:
                evaluations = set(snapshot["evaluations"])
                if request.modality == "multimodal":
                    evaluations = {
                        name for name in evaluations if name.startswith("mm_edit")
                    }
                self._require(request.evaluation, evaluations, "evaluation")
            datasets = {
                item["id"]
                for item in snapshot["datasets"]
                if item["modality"] == request.modality
                and request.operation in item["operations"]
            }
            required_roles = DATASET_ROLES[request.operation]
            if set(request.datasets) != set(required_roles):
                raise CatalogError(
                    f"{request.operation} requires dataset roles: "
                    f"{', '.join(required_roles)}"
                )
            for value in request.datasets.values():
                self._require(value, datasets, "dataset")
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
            if normalized in owned_fields or normalized.endswith(".output_dir"):
                raise CatalogError(
                    f"Override {key!r} conflicts with a WebUI task field. "
                    "Use the corresponding form field; output paths follow task_name."
                )
            parts = normalized.split(".")
            if (
                normalized == "trainer.args"
                or (parts[0] == "data" and (len(parts) <= 3 or parts[-1] == "handler"))
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
        else:
            argv.extend(
                [
                    f"model={request.model}",
                    f"trainer={request.trainer}",
                ]
            )
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
