"""Auditable text-edit model/method compatibility registry.

The registry is deliberately stricter than a family wildcard.  It reads
``configs/extensions/edit_model_matrix.yaml`` and resolves one exact model and
method pair to one of three evidence levels:

``registered``
    The model and handler are catalogued.  This is not an execution claim.
``configured``
    An exact Hydra experiment exists and must compose for the pair.
``verified``
    The configured pair additionally has a current, auditable
    ``operation=edit`` evidence directory.

No model weights are loaded by this module.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = "1.0"
EDIT_OPERATION = "edit"
EDIT_COMPATIBILITY_STATUSES = ("registered", "configured", "verified")
_STATUS_RANK = {
    status: index for index, status in enumerate(EDIT_COMPATIBILITY_STATUSES)
}
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_VERIFIED_ARTIFACTS = frozenset(
    {
        "command.txt",
        "environment.json",
        "input_manifest.json",
        "resolved_config.yaml",
        "edit_results.json",
        "run_trace.json",
    }
)


class EditRegistryError(ValueError):
    """Raised when the text-edit compatibility registry is inconsistent."""


@dataclass(frozen=True)
class EditMethodRegistration:
    """Static registration for one text editing handler."""

    name: str
    config: str
    compatibility_class: str
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class EditMethodCompatibility:
    """Evidence level for one exact model/method pair."""

    method: str
    status: str
    experiment: str | None
    evidence: str | None
    mapping_profile: str | None
    notes: str
    declared_status: str | None = None

    def meets(self, minimum_status: str) -> bool:
        """Return whether this pair meets the requested evidence level."""

        if minimum_status not in _STATUS_RANK:
            raise EditRegistryError(
                f"unknown edit compatibility status `{minimum_status}`"
            )
        return _STATUS_RANK[self.status] >= _STATUS_RANK[minimum_status]


@dataclass(frozen=True)
class EditModelRegistration:
    """One exact text model and its method-specific compatibility entries."""

    key: str
    display_name: str
    aliases: tuple[str, ...]
    config: str
    model_family: str
    parameter_scale_billion: float
    source_model_id: str
    notes: str
    methods: Mapping[str, EditMethodCompatibility]


@dataclass(frozen=True)
class EditModelRegistry:
    """Parsed, validated registry facade."""

    methods: Mapping[str, EditMethodRegistration]
    models: Mapping[str, EditModelRegistration]
    aliases: Mapping[str, str]
    project_root: Path

    def get_model(self, name: str) -> EditModelRegistration:
        """Resolve an exact model key/name/alias without loading weights."""

        key = self.aliases.get(_normalize(name))
        if key is None:
            available = ", ".join(sorted(self.models))
            raise KeyError(f"unknown edit model `{name}`; available: {available}")
        return self.models[key]

    def get_compatibility(
        self,
        model_name: str,
        method_name: str,
        *,
        minimum_status: str = "configured",
    ) -> EditMethodCompatibility:
        """Resolve and gate an exact model/method pair.

        The default requires an exact composed recipe.  Callers that only want
        to inspect catalogued pairs must explicitly request
        ``minimum_status="registered"``.
        """

        model = self.get_model(model_name)
        method = _resolve_method(self.methods, method_name)
        compatibility = model.methods.get(method)
        if compatibility is None:
            raise ValueError(
                f"{model.display_name} does not catalogue {method}; "
                "a global handler registration is not model compatibility"
            )
        if minimum_status == "verified" and compatibility.declared_status == "verified":
            # Only this explicitly requested pair needs local run artifacts.
            # A normal catalog/config lookup must work in a clean installation.
            _validate_evidence(
                self.project_root,
                compatibility.evidence,
                source_model_id=model.source_model_id,
                method=method,
            )
            compatibility = replace(compatibility, status="verified")
        if not compatibility.meets(minimum_status):
            raise ValueError(
                f"{model.display_name} + {method} is only "
                f"{compatibility.status}; required {minimum_status}. "
                f"{compatibility.notes}"
            )
        return compatibility


def _normalize(value: str) -> str:
    return _NORMALIZE_RE.sub("", str(value).strip().lower())


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is None:
        from know_surgery.resources import resource_root

        return resource_root()
    return Path(project_root).expanduser().resolve()


def _registry_path(project_root: Path) -> Path:
    return project_root / "configs" / "extensions" / "edit_model_matrix.yaml"


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EditRegistryError(f"{where} must be a mapping")
    return dict(value)


def _strict_keys(
    value: Mapping[str, Any], *, required: set[str], where: str
) -> dict[str, Any]:
    mapping = dict(value)
    missing = required - set(mapping)
    unknown = set(mapping) - required
    if missing:
        raise EditRegistryError(f"{where} is missing fields: {sorted(missing)}")
    if unknown:
        raise EditRegistryError(f"{where} has unknown fields: {sorted(unknown)}")
    return mapping


def _string(value: Any, where: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EditRegistryError(f"{where} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise EditRegistryError(f"{where} must be a list")
    result = tuple(_string(item, f"{where}[]") for item in value)
    if len(set(result)) != len(result):
        raise EditRegistryError(f"{where} contains duplicates")
    return result  # type: ignore[return-value]


def _relative_path(value: Any, where: str, *, prefix: str) -> str:
    raw = _string(value, where)
    assert raw is not None
    if "\\" in raw:
        raise EditRegistryError(f"{where} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise EditRegistryError(f"{where} must be a safe repository-relative path")
    normalized = path.as_posix()
    if not normalized.startswith(prefix.rstrip("/") + "/"):
        raise EditRegistryError(f"{where} must be below {prefix}/")
    return normalized


def _yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise EditRegistryError(f"missing registry reference: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise EditRegistryError(f"invalid YAML at {path}: {error}") from error
    return _mapping(payload, str(path))


def _primary_handler(config: Mapping[str, Any]) -> str | None:
    handler = config.get("handler")
    return handler.strip() if isinstance(handler, str) and handler.strip() else None


def _validate_evidence(
    root: Path,
    evidence: str,
    *,
    source_model_id: str,
    method: str,
) -> None:
    directory = root / evidence
    missing = sorted(
        name for name in _VERIFIED_ARTIFACTS if not (directory / name).is_file()
    )
    if missing:
        raise EditRegistryError(
            f"verified evidence {evidence} is missing artifacts: {missing}"
        )

    run_trace = json.loads((directory / "run_trace.json").read_text(encoding="utf-8"))
    if run_trace.get("status") != "success":
        raise EditRegistryError(f"verified evidence {evidence} has no successful trace")

    environment = json.loads(
        (directory / "environment.json").read_text(encoding="utf-8")
    )
    if environment.get("operation") != EDIT_OPERATION:
        raise EditRegistryError(
            f"verified evidence {evidence} must record operation=edit"
        )

    command = (directory / "command.txt").read_text(encoding="utf-8")
    if "operation=edit" not in command.split():
        raise EditRegistryError(
            f"verified evidence {evidence} command must contain operation=edit"
        )

    results = json.loads((directory / "edit_results.json").read_text(encoding="utf-8"))
    result_success = results.get("success") is True
    if results.get("edit_type") == "independent":
        requested = results.get("requested_count")
        successful = results.get("successful_cases")
        cases = results.get("cases")
        result_success = (
            isinstance(requested, int)
            and requested > 0
            and successful == requested
            and isinstance(cases, list)
            and len(cases) == requested
            and all(case.get("success") is True for case in cases)
        )
    if results.get("method") != method or not result_success:
        raise EditRegistryError(
            f"verified evidence {evidence} does not contain successful {method} results"
        )

    manifest = json.loads(
        (directory / "input_manifest.json").read_text(encoding="utf-8")
    )
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise EditRegistryError(f"verified evidence {evidence} has no input datasets")
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise EditRegistryError(f"verified evidence {evidence} has malformed data")
        data_path = Path(str(dataset.get("data_path", ""))).expanduser()
        if not data_path.is_absolute():
            data_path = root / data_path
        if not data_path.is_file():
            raise EditRegistryError(
                f"verified evidence {evidence} dataset path is missing: {data_path}"
            )
        actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if dataset.get("sha256") != actual_hash:
            raise EditRegistryError(
                f"verified evidence {evidence} dataset hash does not match disk"
            )

    resolved = _yaml(directory / "resolved_config.yaml")
    resolved_model = _mapping(resolved.get("model", {}), "resolved model")
    model_args = _mapping(
        resolved_model.get("model_args", {}), "resolved model.model_args"
    )
    tokenizer_args = _mapping(
        resolved_model.get("tokenizer_args", {}), "resolved model.tokenizer_args"
    )
    model_path = model_args.get("pretrained_model_name_or_path")
    tokenizer_path = tokenizer_args.get("pretrained_model_name_or_path")
    if model_path != tokenizer_path or not _matches_registered_source(
        model_path, source_model_id
    ):
        raise EditRegistryError(
            f"verified evidence {evidence} model checkpoint does not match registry"
        )
    trainer = _mapping(resolved.get("trainer", {}), "resolved trainer")
    if trainer.get("handler") != method:
        raise EditRegistryError(
            f"verified evidence {evidence} trainer does not match {method}"
        )
    method_args = _mapping(trainer.get("method_args", {}), "resolved method args")
    method_inputs = manifest.get("method_inputs", [])
    if not isinstance(method_inputs, list):
        raise EditRegistryError(
            f"verified evidence {evidence} method_inputs must be a list"
        )
    indexed_inputs: dict[str, tuple[Path, str]] = {}
    for item in method_inputs:
        if not isinstance(item, Mapping):
            raise EditRegistryError(
                f"verified evidence {evidence} has malformed method input"
            )
        key = item.get("key")
        raw_path = item.get("path")
        if not isinstance(key, str) or not isinstance(raw_path, str):
            raise EditRegistryError(
                f"verified evidence {evidence} has incomplete method input"
            )
        input_path = Path(raw_path).expanduser()
        if not input_path.is_absolute():
            input_path = root / input_path
        if not input_path.is_file():
            raise EditRegistryError(
                f"verified evidence {evidence} method input is missing: {input_path}"
            )
        actual_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if item.get("sha256") != actual_hash:
            raise EditRegistryError(
                f"verified evidence {evidence} method-input hash does not match"
            )
        if method_args.get(key) != raw_path:
            raise EditRegistryError(
                f"verified evidence {evidence} method input does not match config"
            )
        indexed_inputs[key] = (input_path, actual_hash)

    checkpoint_keys = {
        "MENDEditor": "edit_network_path",
        "MALMENEditor": "archive",
        "InstructEditEditor": "edit_network_path",
    }
    checkpoint_key = checkpoint_keys.get(method)
    if checkpoint_key is not None:
        checkpoint = indexed_inputs.get(checkpoint_key)
        if checkpoint is None:
            raise EditRegistryError(
                f"verified {method} evidence requires a hashed {checkpoint_key}"
            )
        checkpoint_path, checkpoint_hash = checkpoint
        training_evidence_path = checkpoint_path.with_suffix(
            checkpoint_path.suffix + ".training.json"
        )
        if not training_evidence_path.is_file():
            raise EditRegistryError(
                f"verified {method} checkpoint has no training evidence"
            )
        training_evidence = json.loads(
            training_evidence_path.read_text(encoding="utf-8")
        )
        training_checkpoint = training_evidence.get("checkpoint", {})
        training_environment = training_evidence.get("environment", {})
        if (
            training_evidence.get("status") != "success"
            or training_checkpoint.get("sha256") != checkpoint_hash
            or training_evidence.get("training", {}).get("seed")
            != environment.get("seed")
            or training_environment.get("python_executable")
            != environment.get("python_executable")
        ):
            raise EditRegistryError(
                f"verified {method} training evidence does not match inference"
            )


def _matches_registered_source(value: Any, source_model_id: str) -> bool:
    """Match a canonical Hub ID or its exact Hugging Face snapshot path."""

    if value == source_model_id:
        return True
    if not isinstance(value, str):
        return False
    path = Path(value).expanduser()
    marker = f"models--{source_model_id.replace('/', '--')}"
    try:
        marker_index = path.parts.index(marker)
    except ValueError:
        return False
    suffix = path.parts[marker_index + 1 :]
    if len(suffix) != 2 or suffix[0] != "snapshots" or not suffix[1]:
        return False
    # Reject arbitrary paths that merely imitate the Hub cache layout.
    return path.is_dir() and (path / "config.json").is_file()


def load_edit_model_registry(
    project_root: str | Path | None = None,
    *,
    validate_evidence: bool = False,
) -> EditModelRegistry:
    """Load the compatibility catalog without requiring private run outputs.

    Declared verified entries are returned as configured until their evidence
    is checked. Pass ``validate_evidence=True`` to audit every declared run,
    or request one pair with ``minimum_status="verified"`` for a strict audit.
    Neither path silently accepts missing or invalid evidence.
    """

    root = _project_root(project_root)
    raw = _strict_keys(
        _yaml(_registry_path(root)),
        required={
            "schema_version",
            "operation",
            "status_order",
            "compatibility_templates",
            "methods",
            "models",
        },
        where="edit model matrix",
    )
    if str(raw["schema_version"]) != SCHEMA_VERSION:
        raise EditRegistryError("edit model matrix schema_version mismatch")
    if raw["operation"] != EDIT_OPERATION:
        raise EditRegistryError("edit model matrix operation must be edit")
    if tuple(raw["status_order"]) != EDIT_COMPATIBILITY_STATUSES:
        raise EditRegistryError(
            f"status_order must be {list(EDIT_COMPATIBILITY_STATUSES)}"
        )
    # Templates are parsed by YAML before validation; keep the section explicit
    # so status semantics remain visible to non-Python consumers.
    _mapping(raw["compatibility_templates"], "compatibility_templates")

    # Reuse the repository's existing static handler registry. The matrix is
    # an exact-pair evidence layer, not a second way to register Python code.
    extension_registry = _yaml(root / "configs" / "extensions" / "registry.yaml")
    handler_groups = _mapping(
        extension_registry.get("handlers"), "extension registry handlers"
    )
    algorithm_handlers = _mapping(
        handler_groups.get("algorithm"), "extension registry algorithm handlers"
    )

    methods: dict[str, EditMethodRegistration] = {}
    for name, method_raw in _mapping(raw["methods"], "methods").items():
        name = _string(name, "method name")
        assert name is not None
        method = _strict_keys(
            _mapping(method_raw, f"methods.{name}"),
            required={"config", "compatibility_class", "requirements"},
            where=f"methods.{name}",
        )
        config = _relative_path(
            method["config"], f"methods.{name}.config", prefix="configs/trainer/edit"
        )
        config_payload = _yaml(root / config)
        if _primary_handler(config_payload) != name:
            raise EditRegistryError(
                f"{config} handler does not match registered method {name}"
            )
        static_metadata = _mapping(
            algorithm_handlers.get(name), f"extension registry algorithm {name}"
        )
        if EDIT_OPERATION not in static_metadata.get("operations", []):
            raise EditRegistryError(
                f"existing static registry does not allow {name} for operation=edit"
            )
        methods[name] = EditMethodRegistration(
            name=name,
            config=config,
            compatibility_class=_string(
                method["compatibility_class"],
                f"methods.{name}.compatibility_class",
            ),  # type: ignore[arg-type]
            requirements=_string_tuple(
                method["requirements"], f"methods.{name}.requirements"
            ),
        )

    models: dict[str, EditModelRegistration] = {}
    aliases: dict[str, str] = {}
    for key, model_raw in _mapping(raw["models"], "models").items():
        key = _string(key, "model key")
        assert key is not None
        model = _strict_keys(
            _mapping(model_raw, f"models.{key}"),
            required={
                "display_name",
                "aliases",
                "config",
                "model_family",
                "parameter_scale_billion",
                "source_model_id",
                "notes",
                "methods",
            },
            where=f"models.{key}",
        )
        config = _relative_path(
            model["config"], f"models.{key}.config", prefix="configs/model"
        )
        config_payload = _yaml(root / config)
        source_model_id = _string(
            model["source_model_id"], f"models.{key}.source_model_id"
        )
        assert source_model_id is not None
        model_family = _string(model["model_family"], f"models.{key}.model_family")
        assert model_family is not None
        model_args = _mapping(config_payload.get("model_args"), f"{config}.model_args")
        tokenizer_args = _mapping(
            config_payload.get("tokenizer_args"), f"{config}.tokenizer_args"
        )
        for where, args in (("model", model_args), ("tokenizer", tokenizer_args)):
            if args.get("pretrained_model_name_or_path") != source_model_id:
                raise EditRegistryError(
                    f"{config} {where} checkpoint does not match {source_model_id}"
                )

        compatibilities: dict[str, EditMethodCompatibility] = {}
        for method_name, compatibility_raw in _mapping(
            model["methods"], f"models.{key}.methods"
        ).items():
            if method_name not in methods:
                raise EditRegistryError(
                    f"models.{key} references unknown method {method_name}"
                )
            compatibility = _strict_keys(
                _mapping(
                    compatibility_raw,
                    f"models.{key}.methods.{method_name}",
                ),
                required={
                    "status",
                    "experiment",
                    "evidence",
                    "mapping_profile",
                    "notes",
                },
                where=f"models.{key}.methods.{method_name}",
            )
            status = _string(
                compatibility["status"],
                f"models.{key}.methods.{method_name}.status",
            )
            assert status is not None
            if status not in _STATUS_RANK:
                raise EditRegistryError(
                    f"models.{key}.methods.{method_name} has invalid status {status}"
                )
            experiment = _string(
                compatibility["experiment"],
                f"models.{key}.methods.{method_name}.experiment",
                nullable=True,
            )
            evidence = _string(
                compatibility["evidence"],
                f"models.{key}.methods.{method_name}.evidence",
                nullable=True,
            )
            mapping_profile = _string(
                compatibility["mapping_profile"],
                f"models.{key}.methods.{method_name}.mapping_profile",
                nullable=True,
            )
            notes = _string(
                compatibility["notes"],
                f"models.{key}.methods.{method_name}.notes",
            )
            assert notes is not None

            if status == "registered":
                if experiment is not None or evidence is not None:
                    raise EditRegistryError(
                        f"registered pair {key}/{method_name} cannot claim recipe/evidence"
                    )
            else:
                if experiment is None:
                    raise EditRegistryError(
                        f"{status} pair {key}/{method_name} requires an experiment"
                    )
                experiment_path = root / "configs" / "experiment" / f"{experiment}.yaml"
                if not experiment_path.is_file():
                    raise EditRegistryError(
                        f"{status} pair {key}/{method_name} references missing "
                        f"experiment {experiment}"
                    )
                if (
                    methods[method_name].compatibility_class
                    in {"layer_mapped", "backend_dependent"}
                    and mapping_profile is None
                ):
                    raise EditRegistryError(
                        f"configured {key}/{method_name} requires a mapping_profile"
                    )
                method_families = _mapping(
                    algorithm_handlers[method_name],
                    f"extension registry algorithm {method_name}",
                ).get("model_families", [])
                if "*" not in method_families and model_family not in method_families:
                    raise EditRegistryError(
                        f"existing static registry does not configure {method_name} "
                        f"for model family {model_family}"
                    )
                if status == "configured" and evidence is not None:
                    raise EditRegistryError(
                        f"configured pair {key}/{method_name} cannot attach evidence"
                    )
                if status == "verified":
                    if evidence is None:
                        raise EditRegistryError(
                            f"verified pair {key}/{method_name} requires evidence"
                        )
                    evidence = _relative_path(
                        evidence,
                        f"models.{key}.methods.{method_name}.evidence",
                        prefix="saves/edit",
                    )
                    if validate_evidence:
                        _validate_evidence(
                            root,
                            evidence,
                            source_model_id=source_model_id,
                            method=method_name,
                        )

            compatibilities[method_name] = EditMethodCompatibility(
                method=method_name,
                status=(
                    "configured"
                    if status == "verified" and not validate_evidence
                    else status
                ),
                experiment=experiment,
                evidence=evidence,
                mapping_profile=mapping_profile,
                notes=notes,
                declared_status=status,
            )

        display_name = _string(model["display_name"], f"models.{key}.display_name")
        model_aliases = _string_tuple(model["aliases"], f"models.{key}.aliases")
        assert display_name is not None
        try:
            scale = float(model["parameter_scale_billion"])
        except (TypeError, ValueError) as error:
            raise EditRegistryError(
                f"models.{key}.parameter_scale_billion must be numeric"
            ) from error
        if scale <= 0:
            raise EditRegistryError(
                f"models.{key}.parameter_scale_billion must be positive"
            )
        registration = EditModelRegistration(
            key=key,
            display_name=display_name,
            aliases=model_aliases,
            config=config,
            model_family=model_family,
            parameter_scale_billion=scale,
            source_model_id=source_model_id,
            notes=_string(model["notes"], f"models.{key}.notes"),  # type: ignore[arg-type]
            methods=compatibilities,
        )
        models[key] = registration
        config_alias = config[len("configs/model/") :]
        if config_alias.endswith(".yaml"):
            config_alias = config_alias[: -len(".yaml")]
        for alias in (key, display_name, config_alias, *model_aliases):
            normalized = _normalize(alias)
            previous = aliases.get(normalized)
            if previous is not None and previous != key:
                raise EditRegistryError(
                    f"model alias `{alias}` is ambiguous between {previous} and {key}"
                )
            aliases[normalized] = key

    return EditModelRegistry(
        methods=methods, models=models, aliases=aliases, project_root=root
    )


def _resolve_method(
    methods: Mapping[str, EditMethodRegistration], method_name: str
) -> str:
    normalized = _normalize(method_name)
    matches = [
        name
        for name in methods
        if normalized
        in {
            _normalize(name),
            _normalize(name[: -len("Editor")] if name.endswith("Editor") else name),
            _normalize(Path(methods[name].config).stem),
        }
    ]
    if len(matches) != 1:
        available = ", ".join(sorted(methods))
        raise KeyError(f"unknown edit method `{method_name}`; available: {available}")
    return matches[0]


def get_edit_model_registration(
    name: str, project_root: str | Path | None = None
) -> EditModelRegistration:
    """Resolve a text edit model registration without loading it."""

    return load_edit_model_registry(project_root).get_model(name)


def get_edit_model_method_compatibility(
    model_name: str,
    method_name: str,
    *,
    minimum_status: str = "configured",
    project_root: str | Path | None = None,
) -> EditMethodCompatibility:
    """Resolve and evidence-gate one exact text edit combination."""

    return load_edit_model_registry(project_root).get_compatibility(
        model_name, method_name, minimum_status=minimum_status
    )


def assert_composed_edit_compatibility(
    model_name: str,
    method_name: str,
    cfg: Any,
    *,
    project_root: str | Path | None = None,
) -> EditMethodCompatibility:
    """Check that a composed Hydra config matches its registered exact pair."""

    registry = load_edit_model_registry(project_root)
    model = registry.get_model(model_name)
    compatibility = registry.get_compatibility(
        model_name, method_name, minimum_status="configured"
    )
    if str(cfg.get("operation")) != EDIT_OPERATION:
        raise EditRegistryError("composed compatibility gate requires operation=edit")
    if cfg.model.model_args.pretrained_model_name_or_path != model.source_model_id:
        raise EditRegistryError(
            f"composed model does not match registry source for {model.display_name}"
        )
    if cfg.trainer.handler != compatibility.method:
        raise EditRegistryError(
            f"composed trainer {cfg.trainer.handler} does not match {compatibility.method}"
        )
    return compatibility


__all__ = [
    "EDIT_COMPATIBILITY_STATUSES",
    "EditMethodCompatibility",
    "EditMethodRegistration",
    "EditModelRegistration",
    "EditModelRegistry",
    "EditRegistryError",
    "assert_composed_edit_compatibility",
    "get_edit_model_method_compatibility",
    "get_edit_model_registration",
    "load_edit_model_registry",
]
