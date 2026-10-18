"""Safe, auditable registry for Know-Surgery text extensions.

The extension boundary is intentionally conservative:

* uploaded bundles are parsed as data and their Python is never imported;
* installed handlers must be named in ``configs/extensions/registry.yaml``;
* public operations remain ``unlearning``, ``inject`` and ``edit``;
* verification status is derived from checks/evidence, never from a manifest.

The module is importable both as ``src.extension_registry`` from the repository
root and as ``extension_registry`` when ``src`` is on ``PYTHONPATH``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

import yaml


SCHEMA_VERSION = "1.0"
PUBLIC_OPERATIONS = ("unlearning", "inject", "edit")
EXTENSION_KINDS = ("model", "dataset", "algorithm")
TEXT_MODALITY = "text"
VALID_STATUSES = (
    "planned",
    "configured",
    "smoke_verified",
    "benchmark_verified",
    "disabled",
)
KNOWLEDGE_TYPES = ("fact", "event", "procedural", "rule", "free_text", "concept")
DATASET_ROLES = ("train", "edit", "forget", "retain", "eval")
DATA_FORMATS = ("json", "jsonl", "csv", "huggingface")
EDIT_MODES = ("single", "batch", "sequential", "independent")

REQUIRED_ROLES: dict[str, frozenset[str]] = {
    "unlearning": frozenset(("forget", "retain")),
    "inject": frozenset(("train",)),
    "edit": frozenset(("edit",)),
}

MAX_BUNDLE_BYTES = 50 * 1024 * 1024
MAX_BUNDLE_FILES = 1_000
ALLOWED_BUNDLE_ROOTS = frozenset(
    ("extension.yaml", "README.md", "configs", "data", "src", "tests")
)

_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_HANDLER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PACKAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]*$")
_OVERRIDE_KEY_RE = re.compile(r"^[+~]?[A-Za-z_][A-Za-z0-9_.@/+-]*$")
_COMPOSE_LOCK = threading.RLock()


class ExtensionSchemaError(ValueError):
    """Raised when a manifest or registry violates the public schema."""


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtensionSchemaError(f"{where} must be a mapping")
    return dict(value)


def _strict_keys(
    value: Mapping[str, Any], *, required: set[str], where: str
) -> dict[str, Any]:
    mapping = dict(value)
    unknown = set(mapping) - required
    missing = required - set(mapping)
    if unknown:
        promoted = unknown & {
            "status",
            "verified",
            "verification",
            "evidence",
            "smoke_verified",
            "benchmark_verified",
        }
        if promoted:
            raise ExtensionSchemaError(
                f"{where} cannot declare verification state: {sorted(promoted)}"
            )
        raise ExtensionSchemaError(f"unknown {where} fields: {sorted(unknown)}")
    if missing:
        raise ExtensionSchemaError(f"missing {where} fields: {sorted(missing)}")
    return mapping


def _string(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ExtensionSchemaError(f"{where} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ExtensionSchemaError(f"{where} cannot be empty")
    if any(character in result for character in ("\x00", "\r", "\n")):
        raise ExtensionSchemaError(f"{where} contains a forbidden control character")
    return result


def _string_tuple(
    value: Any,
    where: str,
    *,
    allowed: Sequence[str] | None = None,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ExtensionSchemaError(f"{where} must be a list")
    result = tuple(_string(item, f"{where}[]") for item in value)
    if not allow_empty and not result:
        raise ExtensionSchemaError(f"{where} cannot be empty")
    if len(result) != len(set(result)):
        raise ExtensionSchemaError(f"{where} cannot contain duplicates")
    if allowed is not None:
        unknown = set(result) - set(allowed)
        if unknown:
            raise ExtensionSchemaError(
                f"{where} contains unsupported values: {sorted(unknown)}"
            )
    return result


def _safe_relative_path(
    value: Any,
    where: str,
    *,
    prefix: str | None = None,
    yaml_only: bool = False,
) -> str:
    raw = _string(value, where)
    if "\\" in raw:
        raise ExtensionSchemaError(f"{where} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise ExtensionSchemaError(f"{where} must be a safe repository-relative path")
    if ":" in path.parts[0]:
        raise ExtensionSchemaError(f"{where} cannot contain a drive prefix")
    normalized = path.as_posix()
    if prefix is not None and not normalized.startswith(prefix.rstrip("/") + "/"):
        raise ExtensionSchemaError(f"{where} must be below {prefix}/")
    if yaml_only and path.suffix not in {".yaml", ".yml"}:
        raise ExtensionSchemaError(f"{where} must reference a YAML file")
    return normalized


@dataclass(frozen=True)
class ExtensionSupports:
    """Compatibility claims supplied by an extension author."""

    model_families: tuple[str, ...]
    knowledge_types: tuple[str, ...]
    dataset_roles: tuple[str, ...]
    data_formats: tuple[str, ...]
    edit_modes: tuple[str, ...]
    architecture_specific: bool

    @classmethod
    def from_mapping(cls, value: Any) -> "ExtensionSupports":
        mapping = _strict_keys(
            _require_mapping(value, "supports"),
            required={
                "model_families",
                "knowledge_types",
                "dataset_roles",
                "data_formats",
                "edit_modes",
                "architecture_specific",
            },
            where="supports",
        )
        architecture_specific = mapping["architecture_specific"]
        if not isinstance(architecture_specific, bool):
            raise ExtensionSchemaError("supports.architecture_specific must be boolean")
        return cls(
            model_families=_string_tuple(
                mapping["model_families"],
                "supports.model_families",
                allow_empty=False,
            ),
            knowledge_types=_string_tuple(
                mapping["knowledge_types"],
                "supports.knowledge_types",
                allowed=KNOWLEDGE_TYPES,
                allow_empty=False,
            ),
            dataset_roles=_string_tuple(
                mapping["dataset_roles"],
                "supports.dataset_roles",
                allowed=DATASET_ROLES,
                allow_empty=False,
            ),
            data_formats=_string_tuple(
                mapping["data_formats"],
                "supports.data_formats",
                allowed=DATA_FORMATS,
            ),
            edit_modes=_string_tuple(
                mapping["edit_modes"],
                "supports.edit_modes",
                allowed=EDIT_MODES,
            ),
            architecture_specific=architecture_specific,
        )


@dataclass(frozen=True)
class ExtensionRequirements:
    """Packages and local artifacts required before an extension can run."""

    python_packages: tuple[str, ...]
    files: tuple[str, ...]
    checkpoints: tuple[str, ...]
    statistics: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "ExtensionRequirements":
        mapping = _strict_keys(
            _require_mapping(value, "requires"),
            required={"python_packages", "files", "checkpoints", "statistics"},
            where="requires",
        )
        packages = _string_tuple(mapping["python_packages"], "requires.python_packages")
        invalid_packages = [
            package for package in packages if not _PACKAGE_RE.fullmatch(package)
        ]
        if invalid_packages:
            raise ExtensionSchemaError(
                f"requires.python_packages contains invalid names: {invalid_packages}"
            )

        def paths(key: str) -> tuple[str, ...]:
            raw_paths = _string_tuple(mapping[key], f"requires.{key}")
            return tuple(
                _safe_relative_path(path, f"requires.{key}[]") for path in raw_paths
            )

        return cls(
            python_packages=packages,
            files=paths("files"),
            checkpoints=paths("checkpoints"),
            statistics=paths("statistics"),
        )


@dataclass(frozen=True)
class SmokeRecipe:
    """A shell-free surgery invocation and its expected run artifacts."""

    argv: tuple[str, ...]
    expected_artifacts: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "SmokeRecipe":
        mapping = _strict_keys(
            _require_mapping(value, "recipe.smoke"),
            required={"argv", "expected_artifacts"},
            where="recipe.smoke",
        )
        argv = _string_tuple(mapping["argv"], "recipe.smoke.argv", allow_empty=False)
        expected = _string_tuple(
            mapping["expected_artifacts"],
            "recipe.smoke.expected_artifacts",
            allow_empty=False,
        )
        safe_expected = tuple(
            _safe_relative_path(item, "recipe.smoke.expected_artifacts[]")
            for item in expected
        )
        return cls(argv=argv, expected_artifacts=safe_expected)


@dataclass(frozen=True)
class ExtensionRecipe:
    """One explicit, operation-scoped component combination."""

    id: str
    operation: str
    model: str
    method: str
    datasets: dict[str, str]
    evaluation: str
    model_family: str
    knowledge_types: tuple[str, ...]
    model_hparams: str | None
    smoke: SmokeRecipe

    @classmethod
    def from_mapping(cls, value: Any) -> "ExtensionRecipe":
        mapping = _strict_keys(
            _require_mapping(value, "recipe"),
            required={
                "id",
                "operation",
                "model",
                "method",
                "datasets",
                "evaluation",
                "model_family",
                "knowledge_types",
                "model_hparams",
                "smoke",
            },
            where="recipe",
        )
        recipe_id = _string(mapping["id"], "recipe.id")
        if not _ID_RE.fullmatch(recipe_id):
            raise ExtensionSchemaError("recipe.id must be a lowercase slug")
        operation = _string(mapping["operation"], "recipe.operation")
        if operation not in PUBLIC_OPERATIONS:
            raise ExtensionSchemaError(
                "recipe.operation must be one of: unlearning, inject, edit"
            )
        raw_datasets = _require_mapping(mapping["datasets"], "recipe.datasets")
        if not raw_datasets:
            raise ExtensionSchemaError("recipe.datasets cannot be empty")
        unknown_roles = set(raw_datasets) - set(DATASET_ROLES)
        if unknown_roles:
            raise ExtensionSchemaError(
                f"recipe.datasets contains unknown roles: {sorted(unknown_roles)}"
            )
        datasets = {
            role: _safe_relative_path(
                config_path,
                f"recipe.datasets.{role}",
                prefix="configs",
                yaml_only=True,
            )
            for role, config_path in raw_datasets.items()
        }
        model_hparams = mapping["model_hparams"]
        if model_hparams is not None:
            model_hparams = _safe_relative_path(
                model_hparams,
                "recipe.model_hparams",
                prefix="configs",
                yaml_only=True,
            )
        return cls(
            id=recipe_id,
            operation=operation,
            model=_safe_relative_path(
                mapping["model"], "recipe.model", prefix="configs", yaml_only=True
            ),
            method=_safe_relative_path(
                mapping["method"], "recipe.method", prefix="configs", yaml_only=True
            ),
            datasets=datasets,
            evaluation=_safe_relative_path(
                mapping["evaluation"],
                "recipe.evaluation",
                prefix="configs",
                yaml_only=True,
            ),
            model_family=_string(mapping["model_family"], "recipe.model_family"),
            knowledge_types=_string_tuple(
                mapping["knowledge_types"],
                "recipe.knowledge_types",
                allowed=KNOWLEDGE_TYPES,
                allow_empty=False,
            ),
            model_hparams=model_hparams,
            smoke=SmokeRecipe.from_mapping(mapping["smoke"]),
        )


@dataclass(frozen=True)
class ExtensionManifest:
    """Strict public schema for one custom text extension."""

    schema_version: str
    id: str
    kind: Literal["model", "dataset", "algorithm"]
    operation: Literal["unlearning", "inject", "edit"]
    modality: Literal["text"]
    handler: str
    config_path: str
    supports: ExtensionSupports
    requires: ExtensionRequirements
    recipe: ExtensionRecipe

    @classmethod
    def from_mapping(cls, value: Any) -> "ExtensionManifest":
        mapping = _strict_keys(
            _require_mapping(value, "manifest"),
            required={
                "schema_version",
                "id",
                "kind",
                "operation",
                "modality",
                "handler",
                "config_path",
                "supports",
                "requires",
                "recipe",
            },
            where="manifest",
        )
        schema_version = _string(mapping["schema_version"], "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ExtensionSchemaError(
                f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        extension_id = _string(mapping["id"], "id")
        if not _ID_RE.fullmatch(extension_id):
            raise ExtensionSchemaError(
                "id must be a lowercase slug using letters, digits, '-' or '_'"
            )
        kind = _string(mapping["kind"], "kind")
        if kind not in EXTENSION_KINDS:
            raise ExtensionSchemaError("kind must be one of: model, dataset, algorithm")
        operation = _string(mapping["operation"], "operation")
        if operation not in PUBLIC_OPERATIONS:
            raise ExtensionSchemaError(
                "operation must be one of: unlearning, inject, edit"
            )
        modality = _string(mapping["modality"], "modality")
        if modality != TEXT_MODALITY:
            raise ExtensionSchemaError(
                "custom extension v1 supports only modality: text"
            )
        handler = _string(mapping["handler"], "handler")
        if not _HANDLER_RE.fullmatch(handler):
            raise ExtensionSchemaError("handler must be a Python class name")
        config_path = _safe_relative_path(
            mapping["config_path"], "config_path", prefix="configs", yaml_only=True
        )
        supports = ExtensionSupports.from_mapping(mapping["supports"])
        requires = ExtensionRequirements.from_mapping(mapping["requires"])
        recipe = ExtensionRecipe.from_mapping(mapping["recipe"])
        if recipe.operation != operation:
            raise ExtensionSchemaError(
                "recipe.operation must match the manifest operation"
            )
        if recipe.model_family not in supports.model_families:
            raise ExtensionSchemaError(
                "recipe.model_family must be declared in supports.model_families"
            )
        if not set(recipe.knowledge_types).issubset(supports.knowledge_types):
            raise ExtensionSchemaError(
                "recipe.knowledge_types must be a subset of supports.knowledge_types"
            )
        required_roles = REQUIRED_ROLES[operation]
        if not required_roles.issubset(recipe.datasets):
            missing = sorted(required_roles - set(recipe.datasets))
            raise ExtensionSchemaError(
                f"recipe.datasets is missing required {operation} roles: {missing}"
            )
        if not required_roles.issubset(supports.dataset_roles):
            missing = sorted(required_roles - set(supports.dataset_roles))
            raise ExtensionSchemaError(
                f"supports.dataset_roles is missing required {operation} roles: {missing}"
            )
        if set(recipe.datasets) - set(supports.dataset_roles):
            raise ExtensionSchemaError(
                "every recipe dataset role must be declared in supports.dataset_roles"
            )
        if kind == "dataset" and not supports.data_formats:
            raise ExtensionSchemaError("dataset extensions must declare a data format")
        if operation == "edit" and not supports.edit_modes:
            raise ExtensionSchemaError("edit extensions must declare an edit mode")
        return cls(
            schema_version=schema_version,
            id=extension_id,
            kind=kind,  # type: ignore[arg-type]
            operation=operation,  # type: ignore[arg-type]
            modality=modality,  # type: ignore[arg-type]
            handler=handler,
            config_path=config_path,
            supports=supports,
            requires=requires,
            recipe=recipe,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe representation of the manifest."""

        return asdict(self)

    @property
    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ValidationResult:
    """Computed validation state for one bundle or installed extension."""

    id: str | None
    status: str
    reasons: tuple[str, ...]
    manifest: ExtensionManifest | None
    source: str
    experiment: str | None = None
    evidence_path: str | None = None
    command: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.status != "disabled"

    @property
    def default_runnable(self) -> bool:
        return self.status in {"smoke_verified", "benchmark_verified"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "reasons": list(self.reasons),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "source": self.source,
            "experiment": self.experiment,
            "evidence_path": self.evidence_path,
            "command": list(self.command),
            "valid": self.valid,
            "default_runnable": self.default_runnable,
        }


@dataclass(frozen=True)
class CatalogRecord:
    """UI-facing catalog record for a curated recipe or extension."""

    id: str
    status: str
    reasons: tuple[str, ...]
    manifest: ExtensionManifest | None
    source: str
    experiment: str | None
    default_runnable: bool
    operation: str
    modality: str = TEXT_MODALITY

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "reasons": list(self.reasons),
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "source": self.source,
            "experiment": self.experiment,
            "default_runnable": self.default_runnable,
            "operation": self.operation,
            "modality": self.modality,
        }


@dataclass(frozen=True)
class _BundleView:
    source: Path
    files: dict[str, bytes]
    directories: frozenset[str]

    def has_file(self, path: str) -> bool:
        return path in self.files

    def has_tree(self, root: str) -> bool:
        prefix = root.rstrip("/") + "/"
        return root in self.directories or any(
            path.startswith(prefix) for path in self.files
        )

    def read(self, path: str) -> bytes:
        return self.files[path]


def _normalize_archive_member(name: str) -> str:
    if "\\" in name:
        raise ExtensionSchemaError(f"archive member {name!r} must use POSIX separators")
    if name.startswith("/") or "\x00" in name:
        raise ExtensionSchemaError(f"unsafe archive member: {name!r}")
    path = PurePosixPath(name.rstrip("/"))
    if not path.parts or ".." in path.parts or "." in path.parts:
        raise ExtensionSchemaError(f"unsafe archive member: {name!r}")
    if ":" in path.parts[0]:
        raise ExtensionSchemaError(f"unsafe archive member: {name!r}")
    return path.as_posix()


def _strip_single_bundle_root(
    files: dict[str, bytes], directories: set[str]
) -> tuple[dict[str, bytes], set[str]]:
    if "extension.yaml" in files:
        return files, directories
    roots = {PurePosixPath(path).parts[0] for path in (*files.keys(), *directories)}
    if len(roots) != 1:
        return files, directories
    root = next(iter(roots))
    prefix = root + "/"
    stripped_files = {
        path[len(prefix) :]: payload
        for path, payload in files.items()
        if path.startswith(prefix)
    }
    stripped_directories = {
        path[len(prefix) :]
        for path in directories
        if path.startswith(prefix) and path != root
    }
    return stripped_files, stripped_directories


def _check_bundle_roots(files: Mapping[str, bytes], directories: set[str]) -> None:
    roots = {
        PurePosixPath(path).parts[0] for path in (*files.keys(), *directories) if path
    }
    unknown = roots - ALLOWED_BUNDLE_ROOTS
    if unknown:
        raise ExtensionSchemaError(
            f"bundle contains unsupported top-level entries: {sorted(unknown)}"
        )


def _read_zip_bundle(path: Path) -> _BundleView:
    if path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ExtensionSchemaError("bundle exceeds the 50 MB upload limit")
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    folded: set[str] = set()
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_BUNDLE_FILES:
            raise ExtensionSchemaError("bundle contains too many files")
        for info in infos:
            normalized = _normalize_archive_member(info.filename)
            casefolded = normalized.casefold()
            if casefolded in folded:
                raise ExtensionSchemaError(
                    f"duplicate or case-colliding archive member: {normalized}"
                )
            folded.add(casefolded)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ExtensionSchemaError(
                    f"archive symlinks are not allowed: {normalized}"
                )
            if info.is_dir():
                directories.add(normalized)
                continue
            total += info.file_size
            if info.file_size > MAX_BUNDLE_BYTES or total > MAX_BUNDLE_BYTES:
                raise ExtensionSchemaError(
                    "uncompressed bundle exceeds the 50 MB limit"
                )
            files[normalized] = archive.read(info)
    files, directories = _strip_single_bundle_root(files, directories)
    _check_bundle_roots(files, directories)
    return _BundleView(path, files, frozenset(directories))


def _read_directory_bundle(path: Path) -> _BundleView:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    folded: set[str] = set()
    total = 0
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            raise ExtensionSchemaError(
                f"bundle symlinks are not allowed: {entry.relative_to(path)}"
            )
        relative = entry.relative_to(path).as_posix()
        normalized = _normalize_archive_member(relative)
        casefolded = normalized.casefold()
        if casefolded in folded:
            raise ExtensionSchemaError(
                f"duplicate or case-colliding bundle path: {normalized}"
            )
        folded.add(casefolded)
        if entry.is_dir():
            directories.add(normalized)
            continue
        if not entry.is_file():
            raise ExtensionSchemaError(f"unsupported bundle entry: {normalized}")
        total += entry.stat().st_size
        if total > MAX_BUNDLE_BYTES:
            raise ExtensionSchemaError("bundle exceeds the 50 MB upload limit")
        if len(files) >= MAX_BUNDLE_FILES:
            raise ExtensionSchemaError("bundle contains too many files")
        files[normalized] = entry.read_bytes()
    files, directories = _strip_single_bundle_root(files, directories)
    _check_bundle_roots(files, directories)
    return _BundleView(path, files, frozenset(directories))


def _read_bundle(path: str | Path) -> _BundleView:
    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.exists():
        raise ExtensionSchemaError(f"bundle does not exist: {bundle_path}")
    if bundle_path.is_dir():
        return _read_directory_bundle(bundle_path)
    if not zipfile.is_zipfile(bundle_path):
        raise ExtensionSchemaError("bundle must be a directory or ZIP archive")
    return _read_zip_bundle(bundle_path)


def _safe_yaml(payload: bytes, where: str) -> Any:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ExtensionSchemaError(f"{where} must be UTF-8") from error
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ExtensionSchemaError(f"invalid YAML in {where}: {error}") from error


def _project_root(project_root: str | Path | None) -> Path:
    if project_root is None:
        from know_surgery.resources import resource_root

        return resource_root()
    return Path(project_root).expanduser().resolve()


def _safe_project_path(project_root: Path, relative: str) -> Path:
    safe = _safe_relative_path(relative, "repository path")
    candidate = (project_root / safe).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as error:
        raise ExtensionSchemaError(
            f"path escapes the project root: {relative}"
        ) from error
    return candidate


def _load_yaml_ref(
    relative: str, *, project_root: Path, bundle: _BundleView | None
) -> Any:
    if bundle is not None and bundle.has_file(relative):
        return _safe_yaml(bundle.read(relative), relative)
    path = _safe_project_path(project_root, relative)
    if not path.is_file():
        raise ExtensionSchemaError(f"referenced file does not exist: {relative}")
    return _safe_yaml(path.read_bytes(), relative)


def _ref_exists(
    relative: str, *, project_root: Path, bundle: _BundleView | None
) -> bool:
    if bundle is not None and bundle.has_file(relative):
        return True
    return _safe_project_path(project_root, relative).is_file()


def _handler_names(config: Any) -> set[str]:
    """Collect handler values without interpreting or instantiating YAML."""

    found: set[str] = set()
    if isinstance(config, Mapping):
        for key, value in config.items():
            if key == "handler" and isinstance(value, str):
                found.add(value)
            elif isinstance(value, (Mapping, list)):
                found.update(_handler_names(value))
    elif isinstance(config, list):
        for value in config:
            found.update(_handler_names(value))
    return found


def _primary_handler(kind: str, config: Any) -> str | None:
    if not isinstance(config, Mapping):
        return None
    if kind == "model":
        value = config.get("model_handler", "AutoModelForCausalLM")
        return value if isinstance(value, str) else None
    if kind == "algorithm":
        value = config.get("handler")
        return value if isinstance(value, str) else None
    handlers = _handler_names(config)
    return next(iter(handlers)) if len(handlers) == 1 else None


def _experiment_from_argv(argv: Sequence[str]) -> str:
    experiments = [
        token.split("=", 1)[1]
        for token in argv
        if token.lstrip("+").startswith("experiment=")
    ]
    if len(experiments) != 1:
        raise ExtensionSchemaError(
            "recipe.smoke.argv must contain exactly one experiment=<group/name> override"
        )
    experiment = experiments[0]
    if (
        not _EXPERIMENT_RE.fullmatch(experiment)
        or ".." in PurePosixPath(experiment).parts
    ):
        raise ExtensionSchemaError("recipe smoke experiment is not a safe Hydra group")
    return experiment


def _validate_smoke_argv(argv: Sequence[str], operation: str) -> str:
    if len(argv) < 5 or tuple(argv[:4]) != (
        "python",
        "src/surgery.py",
        "--config-name=surgery.yaml",
        f"operation={operation}",
    ):
        raise ExtensionSchemaError(
            "recipe.smoke.argv must start with: python src/surgery.py "
            f"--config-name=surgery.yaml operation={operation}"
        )
    for token in argv[4:]:
        if token.startswith("-"):
            raise ExtensionSchemaError(
                f"unsupported command option in recipe.smoke.argv: {token}"
            )
        if "=" not in token:
            raise ExtensionSchemaError(
                f"recipe.smoke.argv accepts only Hydra key=value overrides: {token}"
            )
        key, value = token.split("=", 1)
        if not _OVERRIDE_KEY_RE.fullmatch(key):
            raise ExtensionSchemaError(f"unsafe Hydra override key: {key}")
        if "${" in value or "\x00" in value or "\n" in value or "\r" in value:
            raise ExtensionSchemaError(f"unsafe Hydra override value for {key}")
        if key.lstrip("+") == "operation" and value != operation:
            raise ExtensionSchemaError(
                "smoke operation override conflicts with manifest"
            )
        if key.lstrip("+") == "modality" and value != TEXT_MODALITY:
            raise ExtensionSchemaError("extension smoke must use modality=text")
        if key.startswith("hydra.") or key.startswith("paths.root_dir"):
            raise ExtensionSchemaError(f"smoke may not override {key}")
    if operation == "edit" and "max_edits=1" not in argv:
        raise ExtensionSchemaError("edit smoke must include max_edits=1")
    if operation in {"inject", "unlearning"} and not any(
        token in {"trainer.args.max_steps=1", "+trainer.args.max_steps=1"}
        for token in argv
    ):
        raise ExtensionSchemaError(
            f"{operation} smoke must include trainer.args.max_steps=1"
        )
    return _experiment_from_argv(argv)


def _registry_file(project_root: Path) -> Path:
    return project_root / "configs" / "extensions" / "registry.yaml"


def _recipes_file(project_root: Path) -> Path:
    return project_root / "configs" / "extensions" / "recipes.yaml"


def _curated_recipe_ids(project_root: Path) -> set[str]:
    path = _recipes_file(project_root)
    if not path.is_file():
        return set()
    payload = _safe_yaml(path.read_bytes(), str(path))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("recipes"), list):
        return set()
    return {
        item["id"]
        for item in payload["recipes"]
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }


def _load_static_registry(project_root: Path) -> dict[str, Any]:
    path = _registry_file(project_root)
    if not path.is_file():
        raise ExtensionSchemaError(f"missing static extension registry: {path}")
    data = _safe_yaml(path.read_bytes(), str(path))
    mapping = _strict_keys(
        _require_mapping(data, "extension registry"),
        required={"schema_version", "installed", "handlers"},
        where="extension registry",
    )
    if str(mapping["schema_version"]) != SCHEMA_VERSION:
        raise ExtensionSchemaError("extension registry schema_version mismatch")
    if not isinstance(mapping["installed"], list):
        raise ExtensionSchemaError("extension registry installed must be a list")
    handlers = _require_mapping(mapping["handlers"], "extension registry handlers")
    if set(handlers) != set(EXTENSION_KINDS):
        raise ExtensionSchemaError(
            "extension registry handlers must contain model, dataset and algorithm"
        )
    return mapping


def _handler_metadata(
    registry: Mapping[str, Any], kind: str, name: str
) -> dict[str, Any] | None:
    handlers = registry.get("handlers", {})
    by_kind = handlers.get(kind, {}) if isinstance(handlers, Mapping) else {}
    value = by_kind.get(name) if isinstance(by_kind, Mapping) else None
    return dict(value) if isinstance(value, Mapping) else None


def _metadata_allows(metadata: Mapping[str, Any], key: str, value: str) -> bool:
    allowed = metadata.get(key, [])
    return isinstance(allowed, list) and ("*" in allowed or value in allowed)


def _validate_handler_metadata(
    metadata: Mapping[str, Any], *, operation: str, model_family: str
) -> list[str]:
    errors: list[str] = []
    if not _metadata_allows(metadata, "operations", operation):
        errors.append(f"handler does not support operation={operation}")
    if not _metadata_allows(metadata, "model_families", model_family):
        errors.append(f"handler does not support model family {model_family}")
    return errors


def _validate_refs_and_compatibility(
    manifest: ExtensionManifest,
    *,
    project_root: Path,
    bundle: _BundleView | None,
    registry: Mapping[str, Any],
) -> tuple[list[str], bool]:
    """Return (errors, unknown_algorithm_scaffold)."""

    errors: list[str] = []
    unknown_scaffold = False
    refs = {
        "config_path": manifest.config_path,
        "recipe.model": manifest.recipe.model,
        "recipe.method": manifest.recipe.method,
        "recipe.evaluation": manifest.recipe.evaluation,
        **{
            f"recipe.datasets.{role}": path
            for role, path in manifest.recipe.datasets.items()
        },
    }
    if manifest.recipe.model_hparams:
        refs["recipe.model_hparams"] = manifest.recipe.model_hparams
    loaded: dict[str, Any] = {}
    for label, relative in refs.items():
        try:
            loaded[label] = _load_yaml_ref(
                relative, project_root=project_root, bundle=bundle
            )
        except ExtensionSchemaError as error:
            errors.append(str(error))

    if "config_path" in loaded:
        primary = _primary_handler(manifest.kind, loaded["config_path"])
        if primary is None:
            errors.append(
                f"{manifest.config_path} does not expose one unambiguous {manifest.kind} handler"
            )
        elif primary != manifest.handler:
            errors.append(
                f"manifest handler {manifest.handler} does not match config handler {primary}"
            )
        if manifest.kind == "model":
            config = loaded["config_path"]
            if not isinstance(config, Mapping) or not isinstance(
                config.get("model_args"), Mapping
            ):
                errors.append("model config must contain model_args")
            elif not config["model_args"].get("pretrained_model_name_or_path"):
                errors.append(
                    "model config must set model_args.pretrained_model_name_or_path"
                )
            if not isinstance(config, Mapping) or not isinstance(
                config.get("tokenizer_args"), Mapping
            ):
                errors.append("model config must contain tokenizer_args")

    manifest_metadata = _handler_metadata(registry, manifest.kind, manifest.handler)
    if manifest_metadata is None:
        if (
            manifest.kind == "algorithm"
            and bundle is not None
            and bundle.has_tree("src")
            and bundle.has_tree("tests")
        ):
            unknown_scaffold = True
        else:
            errors.append(
                f"unknown {manifest.kind} handler {manifest.handler}; "
                "install and add it to configs/extensions/registry.yaml"
            )
    else:
        errors.extend(
            _validate_handler_metadata(
                manifest_metadata,
                operation=manifest.operation,
                model_family=manifest.recipe.model_family,
            )
        )

    model_config = loaded.get("recipe.model")
    model_handler = _primary_handler("model", model_config)
    if model_handler:
        metadata = _handler_metadata(registry, "model", model_handler)
        if metadata is None:
            errors.append(f"recipe references unknown model handler {model_handler}")
        else:
            errors.extend(
                _validate_handler_metadata(
                    metadata,
                    operation=manifest.operation,
                    model_family=manifest.recipe.model_family,
                )
            )

    method_config = loaded.get("recipe.method")
    method_handler = _primary_handler("algorithm", method_config)
    method_metadata = None
    if method_handler:
        method_metadata = _handler_metadata(registry, "algorithm", method_handler)
        if method_metadata is None:
            if not (unknown_scaffold and method_handler == manifest.handler):
                errors.append(
                    f"recipe references unknown algorithm handler {method_handler}"
                )
        else:
            errors.extend(
                _validate_handler_metadata(
                    method_metadata,
                    operation=manifest.operation,
                    model_family=manifest.recipe.model_family,
                )
            )

    architecture_specific = manifest.supports.architecture_specific or bool(
        method_metadata and method_metadata.get("architecture_specific", False)
    )
    if architecture_specific and manifest.recipe.model_hparams is None:
        errors.append("architecture-specific methods require recipe.model_hparams")
    if manifest.recipe.model_hparams and "recipe.model_hparams" in loaded:
        hparams = loaded["recipe.model_hparams"]
        if not isinstance(hparams, Mapping):
            errors.append("recipe.model_hparams must contain a mapping")
        else:
            family = hparams.get("model_family")
            if family != manifest.recipe.model_family:
                errors.append(
                    "model hparams model_family does not match recipe.model_family"
                )
            if (
                method_handler
                and hparams.get("handler", method_handler) != method_handler
            ):
                errors.append("model hparams handler does not match recipe method")
    if method_metadata and method_metadata.get("requires_checkpoints", False):
        if not manifest.requires.checkpoints:
            errors.append(
                f"algorithm handler {method_handler} requires requires.checkpoints"
            )
    if method_metadata and method_metadata.get("requires_statistics", False):
        if not manifest.requires.statistics:
            errors.append(
                f"algorithm handler {method_handler} requires requires.statistics"
            )

    for role, relative in manifest.recipe.datasets.items():
        config = loaded.get(f"recipe.datasets.{role}")
        if config is None:
            continue
        handlers = _handler_names(config)
        if not handlers:
            errors.append(f"dataset config {relative} has no handler")
            continue
        for handler in handlers:
            metadata = _handler_metadata(registry, "dataset", handler)
            if metadata is None:
                if not (
                    manifest.kind == "dataset"
                    and handler == manifest.handler
                    and bundle is not None
                ):
                    errors.append(
                        f"recipe references unknown dataset handler {handler}"
                    )
                continue
            if not _metadata_allows(metadata, "operations", manifest.operation):
                errors.append(
                    f"dataset handler {handler} does not support {manifest.operation}"
                )
            if not _metadata_allows(metadata, "dataset_roles", role):
                errors.append(f"dataset handler {handler} cannot fill role {role}")
            supported_knowledge = metadata.get("knowledge_types", [])
            if (
                not isinstance(supported_knowledge, list)
                or "*" not in supported_knowledge
                and not set(manifest.recipe.knowledge_types).issubset(
                    supported_knowledge
                )
            ):
                errors.append(
                    f"dataset handler {handler} does not support recipe knowledge types"
                )

    for requirement_type in ("files", "checkpoints", "statistics"):
        for relative in getattr(manifest.requires, requirement_type):
            if not _ref_exists(relative, project_root=project_root, bundle=bundle):
                errors.append(f"missing required {requirement_type[:-1]}: {relative}")
    for package in manifest.requires.python_packages:
        module_name = package.replace("-", "_")
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        if not available:
            errors.append(f"missing required Python package: {package}")

    return errors, unknown_scaffold


def _bundle_structure_errors(
    bundle: _BundleView, manifest: ExtensionManifest
) -> list[str]:
    errors: list[str] = []
    if not bundle.has_file("README.md"):
        errors.append("bundle is missing README.md")
    if not bundle.has_tree("configs"):
        errors.append("bundle is missing configs/")
    if not bundle.has_tree("data/sample"):
        errors.append("bundle is missing data/sample/")
    if not bundle.has_file(manifest.config_path):
        errors.append(
            f"bundle must include its primary config_path: {manifest.config_path}"
        )
    return errors


def _compose_overrides(argv: Sequence[str]) -> list[str]:
    overrides: list[str] = []
    for token in argv[4:]:
        key = token.lstrip("+").split("=", 1)[0]
        if key in {"operation", "modality"}:
            continue
        overrides.append(token)
    return overrides


def _materialize_config_overlay(
    project_root: Path, bundle: _BundleView | None, destination: Path
) -> None:
    source = project_root / "configs"
    if not source.is_dir():
        raise ExtensionSchemaError(f"missing project configs directory: {source}")
    shutil.copytree(source, destination)
    if bundle is None:
        return
    for relative, payload in bundle.files.items():
        if not relative.startswith("configs/"):
            continue
        # Every YAML file is parsed before Hydra sees it; this rejects Python tags.
        if relative.endswith((".yaml", ".yml")):
            _safe_yaml(payload, relative)
        target = destination / relative.removeprefix("configs/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _compose_recipe(
    manifest: ExtensionManifest,
    *,
    project_root: Path,
    bundle: _BundleView | None,
) -> list[str]:
    errors: list[str] = []
    experiment = _validate_smoke_argv(manifest.recipe.smoke.argv, manifest.operation)
    experiment_ref = f"configs/experiment/{experiment}.yaml"
    if not _ref_exists(experiment_ref, project_root=project_root, bundle=bundle):
        return [f"smoke experiment does not exist: {experiment_ref}"]
    base_name = {
        "unlearning": "unlearn.yaml",
        "inject": "inject.yaml",
        "edit": "edit.yaml",
    }[manifest.operation]
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        with tempfile.TemporaryDirectory(prefix="know-surgery-extension-") as temp:
            config_dir = Path(temp) / "configs"
            _materialize_config_overlay(project_root, bundle, config_dir)
            with _COMPOSE_LOCK:
                GlobalHydra.instance().clear()
                try:
                    with initialize_config_dir(
                        config_dir=str(config_dir),
                        job_name="validate_extension",
                        version_base=None,
                    ):
                        cfg = compose(
                            config_name=base_name,
                            overrides=_compose_overrides(manifest.recipe.smoke.argv),
                        )
                finally:
                    GlobalHydra.instance().clear()
            trainer_handler = str(cfg.trainer.get("handler", ""))
            method_config = _load_yaml_ref(
                manifest.recipe.method, project_root=project_root, bundle=bundle
            )
            expected_handler = _primary_handler("algorithm", method_config)
            if expected_handler and trainer_handler != expected_handler:
                errors.append(
                    "composed trainer handler does not match recipe.method "
                    f"({trainer_handler} != {expected_handler})"
                )
            data = OmegaConf.to_container(cfg.data, resolve=False)
            if isinstance(data, Mapping):
                for role, relative in manifest.recipe.datasets.items():
                    role_node = data.get(role)
                    if (
                        isinstance(role_node, Mapping)
                        and len(role_node) == 1
                        and role in role_node
                    ):
                        role_node = role_node[role]
                    actual = _handler_names(role_node)
                    expected = _handler_names(
                        _load_yaml_ref(
                            relative, project_root=project_root, bundle=bundle
                        )
                    )
                    if expected and not expected.issubset(actual):
                        errors.append(
                            f"composed data role {role} does not match recipe dataset"
                        )
            else:
                errors.append("composed data config is not a mapping")
    except Exception as error:  # Hydra exposes several configuration exception types.
        errors.append(f"Hydra compose failed: {type(error).__name__}: {error}")
    return errors


def _installed_manifest_paths(
    project_root: Path, registry: Mapping[str, Any]
) -> tuple[list[tuple[Path, ExtensionManifest]], list[str]]:
    manifests: list[tuple[Path, ExtensionManifest]] = []
    errors: list[str] = []
    for index, raw in enumerate(registry.get("installed", [])):
        try:
            relative = _safe_relative_path(raw, f"installed[{index}]")
            path = _safe_project_path(project_root, relative)
            if not path.is_file():
                raise ExtensionSchemaError(
                    f"installed manifest does not exist: {relative}"
                )
            manifest = ExtensionManifest.from_mapping(
                _safe_yaml(path.read_bytes(), relative)
            )
            manifests.append((path, manifest))
        except ExtensionSchemaError as error:
            errors.append(str(error))
    counts: dict[str, int] = {}
    for _, manifest in manifests:
        counts[manifest.id] = counts.get(manifest.id, 0) + 1
    duplicates = sorted(
        extension_id for extension_id, count in counts.items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate installed extension ids: {duplicates}")
    return manifests, errors


def _import_registered_handler(
    project_root: Path,
    metadata: Mapping[str, Any],
    handler: str,
) -> str | None:
    module_name = metadata.get("registry_module")
    attribute = metadata.get("registry_attribute")
    if not isinstance(module_name, str) or not isinstance(attribute, str):
        return (
            f"static handler registration for {handler} lacks registry import metadata"
        )
    if module_name not in {"model", "data", "trainer"}:
        return f"static handler registry module is not allowlisted: {module_name}"
    src_path = str(project_root / "src")
    inserted = src_path not in sys.path
    if "TRITON_CACHE_DIR" not in os.environ:
        safe_cache = Path(tempfile.gettempdir()) / "know-surgery-triton-cache"
        safe_cache.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(safe_cache)
    if inserted:
        sys.path.insert(0, src_path)
    try:
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        expected_source = (project_root / "src").resolve()
        if not isinstance(module_file, str):
            return f"registered module {module_name} has no auditable source file"
        try:
            Path(module_file).resolve().relative_to(expected_source)
        except ValueError:
            return (
                f"registered module {module_name} was imported outside "
                f"{expected_source}"
            )
        registered = getattr(module, attribute, None)
        if not isinstance(registered, Mapping) or handler not in registered:
            return f"handler {handler} is not present in {module_name}.{attribute}"
    except Exception as error:
        return f"failed to import registered handler {handler}: {type(error).__name__}: {error}"
    finally:
        if inserted:
            try:
                sys.path.remove(src_path)
            except ValueError:
                pass
    return None


def _installed_import_errors(
    manifest: ExtensionManifest,
    *,
    project_root: Path,
    registry: Mapping[str, Any],
) -> list[str]:
    checks: set[tuple[str, str]] = {(manifest.kind, manifest.handler)}
    for kind, ref in (
        ("model", manifest.recipe.model),
        ("algorithm", manifest.recipe.method),
    ):
        config = _load_yaml_ref(ref, project_root=project_root, bundle=None)
        handler = _primary_handler(kind, config)
        if handler:
            checks.add((kind, handler))
    for ref in manifest.recipe.datasets.values():
        config = _load_yaml_ref(ref, project_root=project_root, bundle=None)
        checks.update(("dataset", handler) for handler in _handler_names(config))

    errors: list[str] = []
    for kind, handler in sorted(checks):
        metadata = _handler_metadata(registry, kind, handler)
        if metadata is None:
            errors.append(f"handler {handler} is not statically registered")
            continue
        error = _import_registered_handler(project_root, metadata, handler)
        if error:
            errors.append(error)
    return errors


def _evidence_dir(project_root: Path, extension_id: str) -> Path:
    return project_root / "saves" / "extension_validation" / extension_id


def _read_evidence(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _computed_evidence_status(
    manifest: ExtensionManifest, project_root: Path
) -> tuple[str, str | None, list[str]]:
    evidence_dir = _evidence_dir(project_root, manifest.id)
    benchmark_path = evidence_dir / "benchmark.json"
    benchmark = _read_evidence(benchmark_path)
    if benchmark:
        summary = benchmark.get("summary_path")
        valid_summary = False
        if isinstance(summary, str):
            try:
                summary_path = _safe_project_path(project_root, summary)
                valid_summary = _read_evidence(summary_path) is not None
            except ExtensionSchemaError:
                valid_summary = False
        runs = benchmark.get("runs")
        valid_runs = isinstance(runs, list) and bool(runs)
        if valid_runs:
            for run in runs:
                if not isinstance(run, Mapping) or not isinstance(run.get("seed"), int):
                    valid_runs = False
                    break
                for key in (
                    "summary_path",
                    "resolved_config_path",
                    "command_path",
                    "input_manifest_path",
                    "run_trace_path",
                ):
                    relative = run.get(key)
                    if not isinstance(relative, str):
                        valid_runs = False
                        break
                    try:
                        artifact = _safe_project_path(project_root, relative)
                    except ExtensionSchemaError:
                        valid_runs = False
                        break
                    if not artifact.is_file():
                        valid_runs = False
                        break
                    if key in {"summary_path", "input_manifest_path", "run_trace_path"}:
                        payload = _read_evidence(artifact)
                        if payload is None:
                            valid_runs = False
                            break
                        if (
                            key == "run_trace_path"
                            and payload.get("status") != "success"
                        ):
                            valid_runs = False
                            break
                if not valid_runs:
                    break
        if (
            benchmark.get("schema_version") == SCHEMA_VERSION
            and benchmark.get("extension_id") == manifest.id
            and benchmark.get("manifest_sha256") == manifest.digest
            and benchmark.get("status") == "success"
            and isinstance(benchmark.get("run_count"), int)
            and benchmark["run_count"] > 0
            and benchmark["run_count"] == (len(runs) if isinstance(runs, list) else -1)
            and valid_summary
            and valid_runs
        ):
            return "benchmark_verified", str(benchmark_path), []
    smoke_path = evidence_dir / "validation.json"
    smoke = _read_evidence(smoke_path)
    if smoke:
        try:
            expected_command, run_dir = _smoke_command(manifest, project_root)
        except ExtensionSchemaError:
            expected_command, run_dir = (), evidence_dir / "run"
        expected_artifacts = list(manifest.recipe.smoke.expected_artifacts)
        artifacts_exist = all(
            (run_dir / relative).is_file() for relative in expected_artifacts
        )
        if (
            smoke.get("schema_version") == SCHEMA_VERSION
            and smoke.get("extension_id") == manifest.id
            and smoke.get("manifest_sha256") == manifest.digest
            and smoke.get("status") == "success"
            and smoke.get("command") == list(expected_command)
            and smoke.get("returncode") == 0
            and smoke.get("expected_artifacts") == expected_artifacts
            and smoke.get("missing_artifacts") == []
            and artifacts_exist
        ):
            return "smoke_verified", str(smoke_path), []
    reasons: list[str] = []
    if benchmark_path.exists():
        reasons.append("benchmark evidence is invalid or stale")
    if smoke_path.exists():
        reasons.append("smoke evidence is invalid, failed, or stale")
    return "configured", None, reasons


def _smoke_command(
    manifest: ExtensionManifest, project_root: Path
) -> tuple[tuple[str, ...], Path]:
    _validate_smoke_argv(manifest.recipe.smoke.argv, manifest.operation)
    run_dir = _evidence_dir(project_root, manifest.id) / "run"
    argv = list(manifest.recipe.smoke.argv)
    forbidden_output_keys = {
        "trainer.args.output_dir",
        "paths.output_dir",
        "hydra.run.dir",
    }
    for token in argv:
        key = token.lstrip("+").split("=", 1)[0]
        if key in forbidden_output_keys:
            raise ExtensionSchemaError(
                "smoke output directories are controlled by the validator"
            )
    argv[0] = sys.executable
    argv.append(f"trainer.args.output_dir={run_dir.as_posix()}")
    save_key = "save_edited_model" if manifest.operation == "edit" else "save_model"
    save_overrides = [
        token for token in argv if token.lstrip("+").split("=", 1)[0] == save_key
    ]
    if save_overrides:
        if any(token.split("=", 1)[1].lower() != "false" for token in save_overrides):
            raise ExtensionSchemaError(f"installed smoke requires {save_key}=false")
    else:
        argv.append(
            "save_edited_model=false"
            if manifest.operation == "edit"
            else "+save_model=false"
        )
    return tuple(argv), run_dir


def _run_installed_smoke(
    manifest: ExtensionManifest, project_root: Path
) -> tuple[bool, tuple[str, ...], Path, list[str]]:
    command, run_dir = _smoke_command(manifest, project_root)
    evidence_dir = _evidence_dir(project_root, manifest.id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = datetime.now(timezone.utc)
    completed = subprocess.run(
        list(command),
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60 * 60,
        check=False,
    )
    missing: list[str] = []
    for relative in manifest.recipe.smoke.expected_artifacts:
        artifact = (run_dir / relative).resolve()
        try:
            artifact.relative_to(run_dir.resolve())
        except ValueError:
            missing.append(relative)
            continue
        if not artifact.is_file():
            missing.append(relative)
    success = completed.returncode == 0 and not missing
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "extension_id": manifest.id,
        "manifest_sha256": manifest.digest,
        "status": "success" if success else "failed",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "returncode": completed.returncode,
        "expected_artifacts": list(manifest.recipe.smoke.expected_artifacts),
        "missing_artifacts": missing,
        "stdout_tail": completed.stdout[-8_000:],
        "stderr_tail": completed.stderr[-8_000:],
    }
    evidence_path = evidence_dir / "validation.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    reasons: list[str] = []
    if completed.returncode != 0:
        reasons.append(f"smoke command exited with code {completed.returncode}")
    if missing:
        reasons.append(f"smoke is missing expected artifacts: {missing}")
    return success, command, evidence_path, reasons


class ExtensionRegistry:
    """Registry facade used by the CLI and WebUI refresh path."""

    def __init__(self, project_root: str | Path | None = None):
        self.project_root = _project_root(project_root)

    def validate_bundle(self, path: str | Path) -> ValidationResult:
        manifest: ExtensionManifest | None = None
        try:
            bundle = _read_bundle(path)
            if not bundle.has_file("extension.yaml"):
                raise ExtensionSchemaError("bundle is missing extension.yaml")
            manifest = ExtensionManifest.from_mapping(
                _safe_yaml(bundle.read("extension.yaml"), "extension.yaml")
            )
            registry = _load_static_registry(self.project_root)
            installed, installed_errors = _installed_manifest_paths(
                self.project_root, registry
            )
            errors = list(installed_errors)
            reserved_ids = {item.id for _, item in installed} | _curated_recipe_ids(
                self.project_root
            )
            if manifest.id in reserved_ids:
                errors.append(f"duplicate extension id: {manifest.id}")
            errors.extend(_bundle_structure_errors(bundle, manifest))
            compatibility_errors, unknown_scaffold = _validate_refs_and_compatibility(
                manifest,
                project_root=self.project_root,
                bundle=bundle,
                registry=registry,
            )
            errors.extend(compatibility_errors)
            experiment = _validate_smoke_argv(
                manifest.recipe.smoke.argv, manifest.operation
            )
            if not errors:
                errors.extend(
                    _compose_recipe(
                        manifest, project_root=self.project_root, bundle=bundle
                    )
                )
            if errors:
                return ValidationResult(
                    manifest.id,
                    "disabled",
                    tuple(dict.fromkeys(errors)),
                    manifest,
                    "bundle",
                    experiment=experiment,
                )
            if unknown_scaffold:
                return ValidationResult(
                    manifest.id,
                    "planned",
                    ("algorithm scaffold is not installed or statically registered",),
                    manifest,
                    "bundle",
                    experiment=experiment,
                )
            return ValidationResult(
                manifest.id,
                "configured",
                ("schema, static handler compatibility and Hydra compose passed",),
                manifest,
                "bundle",
                experiment=experiment,
            )
        except (
            ExtensionSchemaError,
            OSError,
            RuntimeError,
            RecursionError,
            zipfile.BadZipFile,
            zipfile.LargeZipFile,
        ) as error:
            return ValidationResult(
                manifest.id if manifest else None,
                "disabled",
                (str(error),),
                manifest,
                "bundle",
            )

    def validate_installed(
        self, extension_id: str, *, smoke: bool = False
    ) -> ValidationResult:
        try:
            requested_id = _string(extension_id, "extension id")
            registry = _load_static_registry(self.project_root)
            installed, errors = _installed_manifest_paths(self.project_root, registry)
            matches = [
                (path, item) for path, item in installed if item.id == requested_id
            ]
            if errors:
                return ValidationResult(
                    requested_id,
                    "disabled",
                    tuple(errors),
                    matches[0][1] if matches else None,
                    "installed",
                )
            if not matches:
                return ValidationResult(
                    requested_id,
                    "disabled",
                    (f"extension is not statically installed: {requested_id}",),
                    None,
                    "installed",
                )
            manifest = matches[0][1]
            if manifest.id in _curated_recipe_ids(self.project_root):
                return ValidationResult(
                    manifest.id,
                    "disabled",
                    (
                        f"duplicate extension id reserved by a curated recipe: {manifest.id}",
                    ),
                    manifest,
                    "installed",
                )
            experiment = _validate_smoke_argv(
                manifest.recipe.smoke.argv, manifest.operation
            )
            compatibility_errors, unknown = _validate_refs_and_compatibility(
                manifest,
                project_root=self.project_root,
                bundle=None,
                registry=registry,
            )
            if unknown:
                compatibility_errors.append(
                    "installed extensions cannot use an unregistered handler"
                )
            compatibility_errors.extend(
                _compose_recipe(manifest, project_root=self.project_root, bundle=None)
            )
            if not compatibility_errors:
                compatibility_errors.extend(
                    _installed_import_errors(
                        manifest,
                        project_root=self.project_root,
                        registry=registry,
                    )
                )
            if compatibility_errors:
                return ValidationResult(
                    manifest.id,
                    "disabled",
                    tuple(dict.fromkeys(compatibility_errors)),
                    manifest,
                    "installed",
                    experiment=experiment,
                )
            command: tuple[str, ...] = ()
            evidence_path: str | None = None
            smoke_reasons: list[str] = []
            if smoke:
                succeeded, command, path, smoke_reasons = _run_installed_smoke(
                    manifest, self.project_root
                )
                evidence_path = str(path)
                if succeeded:
                    return ValidationResult(
                        manifest.id,
                        "smoke_verified",
                        ("real installed smoke completed and evidence was recorded",),
                        manifest,
                        "installed",
                        experiment=experiment,
                        evidence_path=evidence_path,
                        command=command,
                    )
            status, existing_evidence, evidence_reasons = _computed_evidence_status(
                manifest, self.project_root
            )
            if evidence_path is None:
                evidence_path = existing_evidence
            reasons = smoke_reasons + evidence_reasons
            if not reasons:
                reasons.append(
                    "static checks passed; run --smoke to create verification evidence"
                )
            return ValidationResult(
                manifest.id,
                status,
                tuple(reasons),
                manifest,
                "installed",
                experiment=experiment,
                evidence_path=evidence_path,
                command=command,
            )
        except (ExtensionSchemaError, OSError) as error:
            return ValidationResult(
                extension_id,
                "disabled",
                (str(error),),
                None,
                "installed",
            )

    def _curated_records(self) -> list[CatalogRecord]:
        path = _recipes_file(self.project_root)
        if not path.is_file():
            return [
                CatalogRecord(
                    "curated-recipes",
                    "disabled",
                    (f"missing curated recipe registry: {path}",),
                    None,
                    "built_in_curated",
                    None,
                    False,
                    "edit",
                )
            ]
        try:
            payload = _safe_yaml(path.read_bytes(), str(path))
            mapping = _strict_keys(
                _require_mapping(payload, "curated recipes"),
                required={"schema_version", "recipes"},
                where="curated recipes",
            )
            if str(mapping["schema_version"]) != SCHEMA_VERSION:
                raise ExtensionSchemaError("curated recipe schema_version mismatch")
            recipes = mapping["recipes"]
            if not isinstance(recipes, list):
                raise ExtensionSchemaError("curated recipes must be a list")
        except ExtensionSchemaError as error:
            return [
                CatalogRecord(
                    "curated-recipes",
                    "disabled",
                    (str(error),),
                    None,
                    "built_in_curated",
                    None,
                    False,
                    "edit",
                )
            ]

        records: list[CatalogRecord] = []
        seen: set[str] = set()
        for index, raw in enumerate(recipes):
            try:
                recipe = _strict_keys(
                    _require_mapping(raw, f"recipes[{index}]"),
                    required={
                        "id",
                        "operation",
                        "modality",
                        "experiment",
                        "model",
                        "method",
                        "datasets",
                        "evaluation",
                        "model_family",
                        "knowledge_types",
                        "model_hparams",
                        "argv",
                    },
                    where=f"recipes[{index}]",
                )
                recipe_id = _string(recipe["id"], f"recipes[{index}].id")
                if recipe_id in seen:
                    raise ExtensionSchemaError(
                        f"duplicate curated recipe id: {recipe_id}"
                    )
                seen.add(recipe_id)
                operation = _string(recipe["operation"], f"recipes[{index}].operation")
                if operation not in PUBLIC_OPERATIONS:
                    raise ExtensionSchemaError("curated recipe has invalid operation")
                if recipe["modality"] != TEXT_MODALITY:
                    raise ExtensionSchemaError("curated recipe must use modality=text")
                experiment = _string(
                    recipe["experiment"], f"recipes[{index}].experiment"
                )
                argv = _string_tuple(recipe["argv"], f"recipes[{index}].argv")
                if _validate_smoke_argv(argv, operation) != experiment:
                    raise ExtensionSchemaError(
                        "curated recipe experiment does not match argv"
                    )
                refs = [
                    _safe_relative_path(
                        recipe["model"],
                        "recipe model",
                        prefix="configs",
                        yaml_only=True,
                    ),
                    _safe_relative_path(
                        recipe["method"],
                        "recipe method",
                        prefix="configs",
                        yaml_only=True,
                    ),
                    _safe_relative_path(
                        recipe["evaluation"],
                        "recipe evaluation",
                        prefix="configs",
                        yaml_only=True,
                    ),
                ]
                datasets = _require_mapping(recipe["datasets"], "recipe datasets")
                required_roles = REQUIRED_ROLES[operation]
                if not required_roles.issubset(datasets):
                    raise ExtensionSchemaError(
                        f"curated recipe lacks roles: {sorted(required_roles - set(datasets))}"
                    )
                refs.extend(
                    _safe_relative_path(
                        value, "recipe dataset", prefix="configs", yaml_only=True
                    )
                    for value in datasets.values()
                )
                if recipe["model_hparams"] is not None:
                    refs.append(
                        _safe_relative_path(
                            recipe["model_hparams"],
                            "recipe model_hparams",
                            prefix="configs",
                            yaml_only=True,
                        )
                    )
                missing = [
                    ref
                    for ref in refs
                    if not _safe_project_path(self.project_root, ref).is_file()
                ]
                experiment_ref = f"configs/experiment/{experiment}.yaml"
                if not _safe_project_path(self.project_root, experiment_ref).is_file():
                    missing.append(experiment_ref)
                if missing:
                    raise ExtensionSchemaError(
                        f"curated recipe references missing configs: {missing}"
                    )
                # Compose through a synthetic manifest so the same gate is used.
                synthetic = ExtensionManifest.from_mapping(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "id": recipe_id,
                        "kind": "model",
                        "operation": operation,
                        "modality": TEXT_MODALITY,
                        "handler": "AutoModelForCausalLM",
                        "config_path": refs[0],
                        "supports": {
                            "model_families": [recipe["model_family"]],
                            "knowledge_types": recipe["knowledge_types"],
                            "dataset_roles": list(datasets),
                            "data_formats": [],
                            "edit_modes": ["single"] if operation == "edit" else [],
                            "architecture_specific": False,
                        },
                        "requires": {
                            "python_packages": [],
                            "files": [],
                            "checkpoints": [],
                            "statistics": [],
                        },
                        "recipe": {
                            "id": recipe_id,
                            "operation": operation,
                            "model": refs[0],
                            "method": refs[1],
                            "datasets": dict(datasets),
                            "evaluation": refs[2],
                            "model_family": recipe["model_family"],
                            "knowledge_types": recipe["knowledge_types"],
                            "model_hparams": recipe["model_hparams"],
                            "smoke": {
                                "argv": list(argv),
                                "expected_artifacts": ["run_trace.json"],
                            },
                        },
                    }
                )
                registry = _load_static_registry(self.project_root)
                compatibility_errors, unknown_scaffold = (
                    _validate_refs_and_compatibility(
                        synthetic,
                        project_root=self.project_root,
                        bundle=None,
                        registry=registry,
                    )
                )
                if unknown_scaffold:
                    compatibility_errors.append(
                        "curated recipe cannot use an unregistered scaffold"
                    )
                compose_errors = compatibility_errors + _compose_recipe(
                    synthetic, project_root=self.project_root, bundle=None
                )
                if compose_errors:
                    raise ExtensionSchemaError("; ".join(compose_errors))
                records.append(
                    CatalogRecord(
                        recipe_id,
                        "configured",
                        ("built-in curated recipe; no benchmark result is implied",),
                        synthetic,
                        "built_in_curated",
                        experiment,
                        True,
                        operation,
                    )
                )
            except ExtensionSchemaError as error:
                fallback_id = (
                    raw.get("id", f"curated-{index}")
                    if isinstance(raw, Mapping)
                    else f"curated-{index}"
                )
                fallback_operation = (
                    raw.get("operation", "edit") if isinstance(raw, Mapping) else "edit"
                )
                records.append(
                    CatalogRecord(
                        str(fallback_id),
                        "disabled",
                        (str(error),),
                        None,
                        "built_in_curated",
                        None,
                        False,
                        str(fallback_operation),
                    )
                )
        return records

    def discover(self) -> tuple[CatalogRecord, ...]:
        records = self._curated_records()
        try:
            registry = _load_static_registry(self.project_root)
            installed, errors = _installed_manifest_paths(self.project_root, registry)
        except ExtensionSchemaError as error:
            records.append(
                CatalogRecord(
                    "extension-registry",
                    "disabled",
                    (str(error),),
                    None,
                    "installed",
                    None,
                    False,
                    "edit",
                )
            )
            return tuple(records)
        if errors:
            records.append(
                CatalogRecord(
                    "extension-registry",
                    "disabled",
                    tuple(errors),
                    None,
                    "installed",
                    None,
                    False,
                    "edit",
                )
            )
            return tuple(records)
        for _, manifest in installed:
            result = self.validate_installed(manifest.id)
            records.append(
                CatalogRecord(
                    manifest.id,
                    result.status,
                    result.reasons,
                    manifest,
                    "installed",
                    result.experiment,
                    result.default_runnable,
                    manifest.operation,
                )
            )
        return tuple(records)


def validate_bundle(
    path: str | Path, project_root: str | Path | None = None
) -> ValidationResult:
    """Validate an uninstalled ZIP/directory without importing its Python."""

    return ExtensionRegistry(project_root).validate_bundle(path)


def validate_installed(
    extension_id: str,
    project_root: str | Path | None = None,
    smoke: bool = False,
) -> ValidationResult:
    """Validate a statically installed extension and optionally run its smoke."""

    return ExtensionRegistry(project_root).validate_installed(extension_id, smoke=smoke)


def discover_extensions(
    project_root: str | Path | None = None,
) -> tuple[CatalogRecord, ...]:
    """Return curated recipes plus statically installed extension records."""

    return ExtensionRegistry(project_root).discover()


__all__ = [
    "CatalogRecord",
    "DATASET_ROLES",
    "ExtensionManifest",
    "ExtensionRegistry",
    "ExtensionRequirements",
    "ExtensionSchemaError",
    "ExtensionSupports",
    "KNOWLEDGE_TYPES",
    "PUBLIC_OPERATIONS",
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "ValidationResult",
    "discover_extensions",
    "validate_bundle",
    "validate_installed",
]
