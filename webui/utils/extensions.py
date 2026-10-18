"""Safe, testable helpers for the text-only custom extension wizard.

The WebUI is deliberately an exporter, not a plugin installer.  This module
only parses small text datasets, builds reviewable configuration/scaffold
files, and writes deterministic ZIP archives to the system temporary
directory.  It never imports generated/uploaded Python, downloads a model, or
executes a command.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import os
import re
import shlex
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import yaml

try:  # Imported as ``webui.utils.extensions``.
    from .runner import CommandRunner
except ImportError:  # Imported after adding ``webui`` to sys.path.
    from utils.runner import CommandRunner


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_CSV_COLUMNS = 256
SCHEMA_VERSION = "1.0"

OPERATIONS = ("unlearning", "inject", "edit")
KINDS = ("model", "dataset", "algorithm")
KNOWLEDGE_TYPES = ("fact", "event", "procedural", "rule", "free_text", "concept")
DATA_FORMATS = ("json", "jsonl", "csv", "huggingface")
EDIT_MODES = ("single", "batch", "sequential", "independent")
DATASET_ROLES = {
    "unlearning": ("forget", "retain"),
    "inject": ("train",),
    "edit": ("edit",),
}

_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_HANDLER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,79}$")
_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_MODEL_REF_RE = re.compile(r"^[A-Za-z0-9_./+@:-]+$")
_PARAMETER_TYPES = {"bool", "float", "int", "list", "string"}

_CANONICAL_FIELDS = {
    "inject": {
        "id",
        "prompt",
        "input",
        "target_new",
        "context",
        "source",
        "timestamp",
        "conditions",
        "steps",
        "rephrase",
        "generalization",
        "locality",
        "portability",
        "conversations",
    },
    "edit": {
        "id",
        "prompt",
        "subject",
        "target_new",
        "target_old",
        "context",
        "source",
        "timestamp",
        "conditions",
        "rephrase",
        "generalization",
        "locality",
        "portability",
    },
    "unlearning": {
        "id",
        "prompt",
        "target_old",
        "context",
        "source",
        "timestamp",
        "rephrase",
        "generalization",
        "locality",
    },
}
for _fields in _CANONICAL_FIELDS.values():
    _fields.update(
        {
            "images",
            "image",
            "steps",
            "conditions",
            "provenance",
            "source",
            "scope_training_prompts",
            "scope_training_negatives",
        }
    )


class ExtensionInputError(ValueError):
    """Raised when an extension form or uploaded sample is unsafe/invalid."""


@dataclass(frozen=True)
class Diagnostic:
    """One bilingual review diagnostic."""

    level: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class ExtensionDraft:
    """An immutable-in-practice, reviewable extension bundle draft."""

    manifest: Mapping[str, Any]
    files: Mapping[str, bytes]
    diagnostics: tuple[Diagnostic, ...]
    equivalent_cli: str

    @property
    def extension_id(self) -> str:
        return str(self.manifest["id"])

    @property
    def has_errors(self) -> bool:
        return any(item.level == "error" for item in self.diagnostics)


@dataclass(frozen=True)
class CatalogSections:
    """Text-only extension catalog split by execution trust."""

    default: tuple[dict[str, Any], ...]
    experimental: tuple[dict[str, Any], ...]
    disabled: tuple[dict[str, Any], ...]
    diagnostics: tuple[str, ...] = ()


def validate_identifier(value: str, *, field: str = "extension id") -> str:
    """Validate a stable lowercase extension identifier."""

    value = str(value or "").strip()
    if not _ID_RE.fullmatch(value) or len(value) > 64:
        raise ExtensionInputError(
            f"Invalid {field}: {value!r}. Use 1-64 lowercase letters, digits, '_' or '-', "
            "starting with a letter. / 标识符须以小写字母开头。"
        )
    return value


def validate_handler_name(value: str) -> str:
    """Validate a Python class-shaped name without importing it."""

    value = str(value or "").strip()
    if not _HANDLER_RE.fullmatch(value) or value.startswith("_"):
        raise ExtensionInputError(
            "Handler must be a plain class name (letters, digits, underscore) and "
            "must not be a module path. / Handler 必须是普通类名。"
        )
    return value


def validate_relative_path(value: str, *, prefix: str | None = None) -> str:
    """Validate a portable bundle/config path and reject traversal."""

    value = str(value or "").strip()
    if not value or "\x00" in value or "\\" in value:
        raise ExtensionInputError(
            "Path is empty or non-POSIX. / 路径为空或不是 POSIX 路径。"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ExtensionInputError(f"Unsafe relative path: {value!r}. / 拒绝路径穿越。")
    normalized = path.as_posix()
    if prefix and not normalized.startswith(prefix.rstrip("/") + "/"):
        raise ExtensionInputError(
            f"Path {normalized!r} must be under {prefix!r}. / 路径不在允许目录内。"
        )
    return normalized


def validate_config_reference(value: str) -> str:
    """Validate a manifest reference to a YAML file below ``configs/``."""

    value = validate_relative_path(value, prefix="configs")
    if not value.endswith((".yaml", ".yml")):
        raise ExtensionInputError("Config references must end in .yaml or .yml.")
    return value


def validate_model_reference(value: str) -> str:
    """Validate an HF identifier or local model path without touching it."""

    value = str(value or "").strip()
    if not value or len(value) > 512 or any(ord(char) < 32 for char in value):
        raise ExtensionInputError(
            "Model reference is empty or contains control characters."
        )
    if "://" in value or not _MODEL_REF_RE.fullmatch(value):
        raise ExtensionInputError(
            "Use a Hugging Face id or local path, not a URL or shell expression. / "
            "请输入 Hugging Face ID 或本地路径。"
        )
    parts = PurePosixPath(value).parts
    if any(part in (".", "..") for part in parts):
        raise ExtensionInputError("Model reference contains path traversal.")
    return value


def validate_upload_size(source: Any, *, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Return upload size and reject payloads over the 50 MiB gate."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if isinstance(source, (bytes, bytearray, memoryview)):
        size = len(source)
    elif isinstance(source, (str, os.PathLike)):
        size = Path(source).stat().st_size
    elif hasattr(source, "name") and isinstance(source.name, (str, os.PathLike)):
        candidate = Path(source.name)
        if candidate.exists():
            size = candidate.stat().st_size
        else:
            size = _file_object_size(source)
    elif hasattr(source, "read"):
        size = _file_object_size(source)
    else:
        raise ExtensionInputError("Unsupported upload object. / 不支持的上传对象。")
    if size > max_bytes:
        raise ExtensionInputError(
            f"Upload is {size} bytes; limit is {max_bytes} bytes (50 MiB). / 上传文件超过 50 MB。"
        )
    return size


def _file_object_size(source: BinaryIO) -> int:
    try:
        position = source.tell()
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(position)
        return int(size)
    except (AttributeError, OSError):
        data = source.read(MAX_UPLOAD_BYTES + 1)
        return len(data.encode("utf-8") if isinstance(data, str) else data)


def _read_upload(source: Any) -> bytes:
    validate_upload_size(source)
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    if hasattr(source, "name") and Path(str(source.name)).exists():
        return Path(str(source.name)).read_bytes()
    if hasattr(source, "seek"):
        source.seek(0)
    data = source.read(MAX_UPLOAD_BYTES + 1)
    if isinstance(data, str):
        data = data.encode("utf-8")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtensionInputError("Upload exceeds 50 MiB.")
    return bytes(data)


def _source_filename(source: Any, filename: str | None) -> str:
    if filename:
        return Path(filename).name
    if isinstance(source, (str, os.PathLike)):
        return Path(source).name
    name = getattr(source, "name", "")
    return Path(str(name)).name


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExtensionInputError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def parse_data_file(
    source: Any,
    *,
    filename: str | None = None,
    data_format: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a bounded UTF-8 JSON, JSONL, or CSV upload into object records."""

    raw = _read_upload(source)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtensionInputError(
            "Uploaded data must be UTF-8 encoded. / 文件必须为 UTF-8。"
        ) from exc

    inferred = Path(_source_filename(source, filename)).suffix.lower().lstrip(".")
    fmt = str(data_format or inferred).lower().lstrip(".")
    if fmt not in {"json", "jsonl", "csv"}:
        raise ExtensionInputError("Only JSON, JSONL, and CSV uploads are accepted.")

    if fmt == "json":
        try:
            payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ExtensionInputError(f"Invalid JSON: {exc}") from exc
        if isinstance(payload, dict):
            for wrapper in ("data", "records", "examples"):
                if isinstance(payload.get(wrapper), list):
                    payload = payload[wrapper]
                    break
            else:
                payload = [payload]
        records = payload
    elif fmt == "jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(
                    json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
                )
            except (json.JSONDecodeError, RecursionError) as exc:
                raise ExtensionInputError(
                    f"Invalid JSONL line {line_number}: {exc}"
                ) from exc
    else:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        headers = reader.fieldnames or []
        if not headers or any(not str(name).strip() for name in headers):
            raise ExtensionInputError("CSV must have a non-empty header row.")
        if len(headers) != len(set(headers)):
            raise ExtensionInputError("CSV contains duplicate column names.")
        if len(headers) > MAX_CSV_COLUMNS:
            raise ExtensionInputError(f"CSV has more than {MAX_CSV_COLUMNS} columns.")
        records = [dict(row) for row in reader]

    if not isinstance(records, list) or not records:
        raise ExtensionInputError("Dataset must contain at least one record.")
    if len(records) > MAX_RECORDS:
        raise ExtensionInputError(f"Dataset has more than {MAX_RECORDS} records.")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ExtensionInputError(f"Record {index} is not an object.")
        for key in record:
            if not isinstance(key, str) or not key.strip():
                raise ExtensionInputError(
                    f"Record {index} has a non-string/empty field name."
                )
    return records


parse_uploaded_data = parse_data_file


def default_field_mapping(
    operation: str, format_type: str = "custom"
) -> dict[str, str]:
    """Return an editable canonical-to-source field mapping preset."""

    operation = _validate_operation(operation)
    fmt = str(format_type or "custom").lower()
    if operation == "inject" and fmt == "sharegpt":
        return {"id": "id", "conversations": "conversations"}
    if operation == "inject":
        return {
            "id": "id",
            "prompt": "instruction",
            "input": "input",
            "target_new": "output",
        }
    if operation == "edit":
        return {
            "id": "id",
            "prompt": "prompt",
            "subject": "subject",
            "target_new": "target_new",
            "target_old": "target_old",
            "rephrase": "rephrase_prompts",
            "locality": "locality_inputs",
            "portability": "portability_inputs",
        }
    return {"id": "id", "prompt": "question", "target_old": "answer"}


def parse_field_mapping(
    operation: str, mapping: str | Mapping[str, str]
) -> dict[str, str]:
    """Validate canonical field mappings, including safe dotted source paths."""

    operation = _validate_operation(operation)
    if isinstance(mapping, str):
        try:
            mapping = json.loads(mapping)
        except json.JSONDecodeError as exc:
            raise ExtensionInputError(
                f"Field mapping is not valid JSON: {exc}"
            ) from exc
    if not isinstance(mapping, Mapping) or not mapping:
        raise ExtensionInputError("Field mapping must be a non-empty JSON object.")
    unknown = set(mapping) - _CANONICAL_FIELDS[operation]
    if unknown:
        raise ExtensionInputError(
            f"Unknown canonical fields: {', '.join(sorted(unknown))}."
        )
    result: dict[str, str] = {}
    for canonical, source in mapping.items():
        if not isinstance(source, str) or not source.strip():
            raise ExtensionInputError(
                f"Mapping for {canonical!r} must be a source field path."
            )
        source = source.strip()
        if any(part in ("", ".", "..") for part in source.split(".")):
            raise ExtensionInputError(f"Unsafe field path: {source!r}.")
        if any(char in source for char in ("[", "]", "\\", "\x00")):
            raise ExtensionInputError(f"Unsupported field path: {source!r}.")
        result[str(canonical)] = source
    return result


def _resolve_field(record: Any, path: str) -> Any:
    if isinstance(record, Mapping) and path in record:
        return record[path]
    current = record
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            if not part.isdigit() or int(part) >= len(current):
                return None
            current = current[int(part)]
        else:
            return None
    return current


def _text(value: Any, *, field: str, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ExtensionInputError(f"Missing required field {field!r}.")
        return None
    if isinstance(value, (dict, list, tuple, set)):
        raise ExtensionInputError(f"Field {field!r} must be scalar text.")
    result = str(value).strip()
    if required and not result:
        raise ExtensionInputError(f"Field {field!r} is empty.")
    return result or None


def _structured(value: Any, *, field: str) -> Any:
    if value is None or value == "":
        return None
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ExtensionInputError(f"Field {field!r} is not JSON-compatible.") from exc
    return value


def _mapped(record: Mapping[str, Any], mapping: Mapping[str, str], key: str) -> Any:
    path = mapping.get(key)
    return None if path is None else _resolve_field(record, path)


def _normalize_sharegpt(
    record: Mapping[str, Any], mapping: Mapping[str, str]
) -> tuple[str, str, str | None]:
    conversations = _mapped(record, mapping, "conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ExtensionInputError("ShareGPT 'conversations' must be a non-empty list.")
    normalized: list[tuple[str, str]] = []
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, Mapping):
            raise ExtensionInputError(
                f"Conversation turn {turn_index} is not an object."
            )
        role = str(turn.get("from", turn.get("role", ""))).strip().lower()
        content = _text(
            turn.get("value", turn.get("content", turn.get("text"))),
            field=f"conversations.{turn_index}.content",
            required=role in {"human", "user", "gpt", "assistant", "system"},
        )
        if role in {"human", "user"}:
            normalized.append(("user", content or ""))
        elif role in {"gpt", "assistant"}:
            normalized.append(("assistant", content or ""))
        elif role == "system":
            normalized.append(("system", content or ""))
    if not normalized or normalized[-1][0] != "assistant":
        raise ExtensionInputError(
            "ShareGPT sample must end with an assistant response."
        )
    users = [content for role, content in normalized[:-1] if role == "user"]
    if not users:
        raise ExtensionInputError("ShareGPT sample has no user prompt.")
    context_lines = [f"{role}: {content}" for role, content in normalized[:-2]]
    return users[-1], normalized[-1][1], "\n".join(context_lines) or None


def canonicalize_records(
    operation: str,
    records: Sequence[Mapping[str, Any]],
    mapping: str | Mapping[str, str],
    *,
    knowledge_type: str = "fact",
    format_type: str = "custom",
    role: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize all records to the operation-aware public data contract."""

    operation = _validate_operation(operation)
    knowledge_type = _validate_knowledge_types((knowledge_type,))[0]
    mapping = parse_field_mapping(operation, mapping)
    if role is not None and role not in DATASET_ROLES[operation]:
        raise ExtensionInputError(f"Role {role!r} is invalid for {operation}.")
    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            sample_id = _text(_mapped(record, mapping, "id"), field="id")
            sample: dict[str, Any] = {
                "id": sample_id or f"sample-{index:06d}",
                "knowledge_type": knowledge_type,
                "operation": operation,
            }
            if operation == "inject" and str(format_type).lower() == "sharegpt":
                prompt, target_new, context = _normalize_sharegpt(record, mapping)
                sample.update({"prompt": prompt, "target_new": target_new})
                if context:
                    sample["context"] = context
            else:
                sample["prompt"] = _text(
                    _mapped(record, mapping, "prompt"), field="prompt", required=True
                )
                if operation in {"inject", "edit"}:
                    sample["target_new"] = _text(
                        _mapped(record, mapping, "target_new"),
                        field="target_new",
                        required=True,
                    )
                if operation == "edit":
                    sample["subject"] = _text(
                        _mapped(record, mapping, "subject"),
                        field="subject",
                        required=True,
                    )
                if operation == "unlearning":
                    sample["target_old"] = _text(
                        _mapped(record, mapping, "target_old"),
                        field="target_old",
                        required=True,
                    )

            scalar_optional = ("input", "target_old", "context", "timestamp")
            for key in scalar_optional:
                if key in sample or key not in mapping:
                    continue
                value = _text(_mapped(record, mapping, key), field=key)
                if value is not None:
                    sample[key] = value
            structured_optional = (
                "source",
                "provenance",
                "images",
                "image",
                "scope_training_prompts",
                "scope_training_negatives",
                "conditions",
                "steps",
                "rephrase",
                "generalization",
                "locality",
                "portability",
            )
            for key in structured_optional:
                if key not in mapping:
                    continue
                value = _structured(_mapped(record, mapping, key), field=key)
                if value is not None:
                    sample[key] = value
            from know_surgery.data_contract import PRESERVED_FIELDS, normalize_record

            for key in PRESERVED_FIELDS:
                if key not in sample and key in record:
                    sample[key] = _structured(record[key], field=key)
            try:
                sample = normalize_record(
                    sample,
                    operation,
                    knowledge_type=knowledge_type,
                    index=index,
                    role=role,
                )
            except ValueError as exc:
                raise ExtensionInputError(str(exc)) from exc
            result.append(sample)
        except ExtensionInputError as exc:
            raise ExtensionInputError(f"Record {index}: {exc}") from exc
    if not result:
        raise ExtensionInputError("No canonical records were produced.")
    return result


def build_canonical_preview(
    operation: str,
    records: Sequence[Mapping[str, Any]],
    mapping: str | Mapping[str, str],
    *,
    knowledge_type: str = "fact",
    format_type: str = "custom",
    retain_records: Sequence[Mapping[str, Any]] | None = None,
    retain_mapping: str | Mapping[str, str] | None = None,
) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
    """Validate all records and return at most three canonical examples per role."""

    operation = _validate_operation(operation)
    if operation == "unlearning":
        if retain_records is None:
            raise ExtensionInputError(
                "Unlearning requires both forget and retain files."
            )
        forget = canonicalize_records(
            operation,
            records,
            mapping,
            knowledge_type=knowledge_type,
            format_type=format_type,
            role="forget",
        )
        retain = canonicalize_records(
            operation,
            retain_records,
            retain_mapping or mapping,
            knowledge_type=knowledge_type,
            format_type=format_type,
            role="retain",
        )
        return {"forget": forget[:3], "retain": retain[:3]}
    normalized = canonicalize_records(
        operation,
        records,
        mapping,
        knowledge_type=knowledge_type,
        format_type=format_type,
        role=DATASET_ROLES[operation][0],
    )
    return normalized[:3]


def preview_uploaded_dataset(
    operation: str,
    source: Any,
    mapping: str | Mapping[str, str],
    *,
    knowledge_type: str = "fact",
    format_type: str = "custom",
    filename: str | None = None,
    retain_source: Any | None = None,
    retain_filename: str | None = None,
) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
    """Parse upload(s) and produce a canonical first-three preview."""

    records = parse_data_file(source, filename=filename)
    retain = (
        parse_data_file(retain_source, filename=retain_filename)
        if retain_source is not None
        else None
    )
    return build_canonical_preview(
        operation,
        records,
        mapping,
        knowledge_type=knowledge_type,
        format_type=format_type,
        retain_records=retain,
    )


def _validate_operation(operation: str) -> str:
    operation = str(operation or "").strip().lower()
    if operation not in OPERATIONS:
        raise ExtensionInputError(f"Operation must be one of: {', '.join(OPERATIONS)}.")
    return operation


def _validate_family(value: str) -> str:
    value = str(value or "").strip().lower()
    if not _FAMILY_RE.fullmatch(value):
        raise ExtensionInputError(f"Invalid model family: {value!r}.")
    return value


def _list_value(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = re.split(r"[,\n]", value)
    else:
        parts = [str(item) for item in value]
    return tuple(dict.fromkeys(part.strip() for part in parts if part.strip()))


def _validate_knowledge_types(values: str | Iterable[str]) -> tuple[str, ...]:
    result = _list_value(values)
    if not result:
        raise ExtensionInputError("Select at least one knowledge type.")
    unknown = set(result) - set(KNOWLEDGE_TYPES)
    if unknown:
        raise ExtensionInputError(
            f"Unknown knowledge types: {', '.join(sorted(unknown))}."
        )
    return result


def _validate_families(values: str | Iterable[str]) -> tuple[str, ...]:
    result = tuple(_validate_family(value) for value in _list_value(values))
    if not result:
        raise ExtensionInputError("Select at least one supported model family.")
    return tuple(dict.fromkeys(result))


def parse_parameter_schema(
    value: str | Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Validate the limited parameter-schema language used by scaffolds."""

    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ExtensionInputError(
                f"Parameter schema is not valid JSON: {exc}"
            ) from exc
    if not isinstance(value, Mapping):
        raise ExtensionInputError("Parameter schema must be a JSON object.")
    result: dict[str, dict[str, Any]] = {}
    for name, definition in value.items():
        if not _HANDLER_RE.fullmatch(str(name)) or str(name).startswith("_"):
            raise ExtensionInputError(f"Invalid parameter name: {name!r}.")
        if isinstance(definition, str):
            definition = {"type": definition}
        if not isinstance(definition, Mapping) or set(definition) - {
            "type",
            "default",
            "description",
        }:
            raise ExtensionInputError(f"Invalid schema for parameter {name!r}.")
        parameter_type = str(definition.get("type", "")).lower()
        if parameter_type not in _PARAMETER_TYPES:
            raise ExtensionInputError(
                f"Parameter {name!r} type must be one of {sorted(_PARAMETER_TYPES)}."
            )
        normalized = {"type": parameter_type}
        if "default" in definition:
            _validate_parameter_default(name, parameter_type, definition["default"])
            normalized["default"] = definition["default"]
        if definition.get("description"):
            normalized["description"] = str(definition["description"])[:500]
        result[str(name)] = normalized
    return result


def _validate_parameter_default(name: str, parameter_type: str, value: Any) -> None:
    expected = {
        "bool": bool,
        "float": (int, float),
        "int": int,
        "list": list,
        "string": str,
    }[parameter_type]
    if isinstance(value, bool) and parameter_type in {"float", "int"}:
        valid = False
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise ExtensionInputError(
            f"Default for {name!r} does not match declared type {parameter_type!r}."
        )


def _default_recipe_refs(operation: str) -> dict[str, Any]:
    operation = _validate_operation(operation)
    if operation == "unlearning":
        return {
            "model": "configs/model/Qwen2.5-3B-Instruct.yaml",
            "method": "configs/trainer/GradAscent.yaml",
            "datasets": {
                "forget": "configs/data/datasets/TOFU_QA_forget.yaml",
                "retain": "configs/data/datasets/TOFU_QA_retain.yaml",
            },
            "evaluation": "configs/eval/tofu.yaml",
        }
    if operation == "inject":
        return {
            "model": "configs/model/Qwen2.5-3B-Instruct.yaml",
            "method": "configs/trainer/inject/LoRA.yaml",
            "datasets": {"train": "configs/data/datasets/Alpaca_inject.yaml"},
            "evaluation": "configs/eval/inject.yaml",
        }
    return {
        "model": "configs/model/Qwen2.5-3B-Instruct.yaml",
        "method": "configs/trainer/edit/FT.yaml",
        "datasets": {"edit": "configs/data/datasets/CounterFact_edit.yaml"},
        "evaluation": "configs/eval/edit.yaml",
    }


def _recipe_refs(
    operation: str,
    *,
    model: str | None = None,
    method: str | None = None,
    datasets: Mapping[str, str] | None = None,
    evaluation: str | None = None,
) -> dict[str, Any]:
    refs = _default_recipe_refs(operation)
    if model:
        refs["model"] = model
    if method:
        refs["method"] = method
    if datasets:
        refs["datasets"] = dict(datasets)
    if evaluation:
        refs["evaluation"] = evaluation
    refs["model"] = validate_config_reference(refs["model"])
    refs["method"] = validate_config_reference(refs["method"])
    refs["evaluation"] = validate_config_reference(refs["evaluation"])
    roles = DATASET_ROLES[operation]
    if set(refs["datasets"]) != set(roles):
        raise ExtensionInputError(
            f"{operation} recipe needs exactly dataset roles {roles}; got {tuple(refs['datasets'])}."
        )
    refs["datasets"] = {
        role: validate_config_reference(refs["datasets"][role]) for role in roles
    }
    return refs


def _hydra_name(reference: str, group: str) -> str:
    prefix = f"configs/{group}/"
    reference = validate_config_reference(reference)
    if not reference.startswith(prefix):
        raise ExtensionInputError(
            f"Expected a {group!r} config reference, got {reference!r}."
        )
    return reference[len(prefix) :].rsplit(".", 1)[0]


def _operation_dir(operation: str) -> str:
    return "unlearn" if operation == "unlearning" else operation


def _experiment_mapping(
    operation: str, refs: Mapping[str, Any], *, task_name: str
) -> dict[str, Any]:
    defaults: list[Any] = [
        {"override /model": _hydra_name(refs["model"], "model")},
        {"override /trainer": _hydra_name(refs["method"], "trainer")},
        {"override /data": _operation_dir(operation)},
    ]
    for role in DATASET_ROLES[operation]:
        defaults.append(
            {
                f"override /data/datasets@data.{role}": _hydra_name(
                    refs["datasets"][role], "data/datasets"
                )
            }
        )
    defaults.append({"override /eval": _hydra_name(refs["evaluation"], "eval")})
    config: dict[str, Any] = {"defaults": defaults, "task_name": task_name}
    if operation == "edit":
        config.update(
            {"edit_type": "single", "max_edits": 1, "save_edit_summary": True}
        )
    return config


def build_equivalent_cli(
    operation: str,
    *,
    experiment: str,
    task_name: str,
    seed: int = 42,
    model: str | None = None,
    method: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    include_cfg_preview: bool = True,
) -> tuple[str, list[str]]:
    """Build preview and smoke argv through the same shell-free runner builder."""

    operation = _validate_operation(operation)
    command_overrides = {"trainer.args.seed": int(seed)}
    if operation == "edit":
        command_overrides["max_edits"] = 1
    else:
        # ``max_steps`` is absent from the strict base trainer schema, so Hydra
        # requires an additive override for the explicit one-step smoke gate.
        command_overrides["+trainer.args.max_steps"] = 1
    command_overrides.update(dict(overrides or {}))
    runner = CommandRunner(Path.cwd())
    command = runner.build_command(
        mode=operation,
        model=model,
        trainer=method,
        experiment=experiment,
        task_name=task_name,
        overrides=command_overrides,
    )
    argv = shlex.split(command.replace("\\\n", " "))
    preview = command
    if include_cfg_preview:
        preview = f"{command} \\\n  --cfg job"
    return preview, argv


def _base_manifest(
    *,
    extension_id: str,
    kind: str,
    operation: str,
    handler: str,
    config_path: str,
    model_families: Sequence[str],
    knowledge_types: Sequence[str],
    data_formats: Sequence[str],
    architecture_specific: bool,
    refs: Mapping[str, Any],
    model_hparams: str | None,
    requires: Mapping[str, Sequence[str]] | None,
    experiment_name: str,
    smoke_argv: Sequence[str],
) -> dict[str, Any]:
    requirements = requires or {}

    def requirement_paths(key: str) -> list[str]:
        return [
            validate_relative_path(item) for item in _list_value(requirements.get(key))
        ]

    packages = list(_list_value(requirements.get("python_packages")))
    invalid_packages = [
        package
        for package in packages
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", package)
    ]
    if invalid_packages:
        raise ExtensionInputError(f"Invalid Python package names: {invalid_packages}.")
    return {
        "schema_version": SCHEMA_VERSION,
        "id": extension_id,
        "kind": kind,
        "operation": operation,
        "modality": "text",
        "handler": handler,
        "config_path": validate_config_reference(config_path),
        "supports": {
            "model_families": list(model_families),
            "knowledge_types": list(knowledge_types),
            "dataset_roles": list(DATASET_ROLES[operation]),
            "data_formats": list(dict.fromkeys(data_formats)),
            "edit_modes": ["single", "batch"] if operation == "edit" else [],
            "architecture_specific": bool(architecture_specific),
        },
        "requires": {
            "python_packages": packages,
            "files": requirement_paths("files"),
            "checkpoints": requirement_paths("checkpoints"),
            "statistics": requirement_paths("statistics"),
        },
        "recipe": {
            "id": f"{extension_id}-recipe",
            "operation": operation,
            "model": refs["model"],
            "method": refs["method"],
            "datasets": dict(refs["datasets"]),
            "evaluation": refs["evaluation"],
            "model_family": model_families[0],
            "knowledge_types": list(knowledge_types),
            "model_hparams": model_hparams,
            "smoke": {
                "argv": list(smoke_argv),
                # Shared training/edit runtime writes this auditable trace for
                # every public operation; the registry evidence gate requires
                # at least one real artifact.
                "expected_artifacts": ["run_trace.json"],
            },
        },
    }


def _yaml_bytes(value: Mapping[str, Any], *, header: str | None = None) -> bytes:
    text = yaml.safe_dump(
        dict(value), sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    if header:
        text = header.rstrip() + "\n" + text
    return text.encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n"
            for record in records
        )
    ).encode("utf-8")


def _readme(extension_id: str, kind: str, operation: str, *, planned: bool) -> bytes:
    state = (
        "This algorithm is a **planned, non-runnable scaffold** until a developer "
        "implements, statically registers, and validates it."
        if planned
        else "Static validation can establish only `configured`; it is not runtime evidence."
    )
    return f"""# {extension_id}

Know-Surgery text `{kind}` extension for `{operation}`.

{state}

## Safety and installation

The WebUI only exported this reviewable archive. It did not install dependencies,
import Python from the archive, download model weights, or write into the repository.
Review every file, then manually copy approved files into a clean checkout.

```bash
python scripts/validate_extension.py {extension_id}.zip
# After manual installation and static registry entry:
python scripts/validate_extension.py --installed {extension_id}
python scripts/validate_extension.py --installed {extension_id} --smoke
```

Only the final command may execute a one-example run and write auditable evidence to
`saves/extension_validation/{extension_id}/validation.json`.
""".encode("utf-8")


def _diagnose_manifest(
    manifest: Mapping[str, Any],
    *,
    existing_ids: Iterable[str] = (),
    planned: bool = False,
) -> tuple[Diagnostic, ...]:
    diagnostics: list[Diagnostic] = []
    if manifest["id"] in set(existing_ids):
        diagnostics.append(
            Diagnostic(
                "error",
                "duplicate_id",
                "Extension id already exists. / 扩展 ID 已存在。",
            )
        )
    forbidden = {"status", "verified", "evidence"}.intersection(manifest)
    if forbidden:
        diagnostics.append(
            Diagnostic(
                "error", "forged_status", f"Forbidden claim fields: {sorted(forbidden)}"
            )
        )
    if manifest.get("modality") != "text" or str(
        manifest.get("handler", "")
    ).startswith("MM"):
        diagnostics.append(
            Diagnostic(
                "error",
                "text_only",
                "V1 wizard accepts text extensions only. / v1 仅支持文本。",
            )
        )
    if planned:
        diagnostics.append(
            Diagnostic(
                "warning",
                "planned_scaffold",
                "Algorithm source is a planned, non-runnable scaffold. / 算法仅为不可运行脚手架。",
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                "info",
                "configured_only",
                "Export/static checks cannot grant smoke_verified status. / 静态检查不能晋级为已验证。",
            )
        )
    diagnostics.append(
        Diagnostic(
            "info",
            "no_execution",
            "No code was imported or executed; no model was downloaded. / 未导入执行代码，未下载模型。",
        )
    )
    return tuple(diagnostics)


def _finalize_draft(
    manifest: dict[str, Any],
    files: dict[str, bytes],
    equivalent_cli: str,
    *,
    existing_ids: Iterable[str],
    planned: bool,
) -> ExtensionDraft:
    # Use the repository-owned strict schema before anything is exportable.
    # This import is trusted project code; generated/uploaded Python is never
    # imported by this path.
    try:
        manifest_type = getattr(
            importlib.import_module("src.extension_registry"), "ExtensionManifest"
        )
        manifest_type.from_mapping(manifest)
    except ImportError:
        # Source distributions that have not yet installed the registry still
        # receive the local structural checks below.
        pass
    except (TypeError, ValueError) as exc:
        raise ExtensionInputError(f"Manifest schema validation failed: {exc}") from exc
    diagnostics = _diagnose_manifest(
        manifest, existing_ids=existing_ids, planned=planned
    )
    files = dict(files)
    files["extension.yaml"] = _yaml_bytes(manifest)
    for path in files:
        validate_relative_path(path)
    return ExtensionDraft(
        manifest=MappingProxyType(manifest),
        files=MappingProxyType(dict(sorted(files.items()))),
        diagnostics=diagnostics,
        equivalent_cli=equivalent_cli,
    )


def build_model_extension_draft(
    *,
    extension_id: str,
    operation: str,
    model_reference: str,
    family: str,
    tokenizer_reference: str | None = None,
    dtype: str = "bfloat16",
    attention: str = "sdpa",
    chat_template: str | None = None,
    knowledge_types: str | Iterable[str] = ("fact",),
    method_config: str | None = None,
    dataset_configs: Mapping[str, str] | None = None,
    evaluation_config: str | None = None,
    seed: int = 42,
    existing_ids: Iterable[str] = (),
) -> ExtensionDraft:
    """Build a configuration-only CausalLM model extension draft."""

    extension_id = validate_identifier(extension_id)
    operation = _validate_operation(operation)
    family = _validate_family(family)
    model_reference = validate_model_reference(model_reference)
    tokenizer_reference = validate_model_reference(
        tokenizer_reference or model_reference
    )
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ExtensionInputError("dtype must be float32, float16, or bfloat16.")
    if attention not in {"eager", "sdpa", "flash_attention_2"}:
        raise ExtensionInputError("Unsupported attention implementation.")
    if chat_template is not None and len(chat_template.encode("utf-8")) > 64 * 1024:
        raise ExtensionInputError("Chat template exceeds 64 KiB.")
    knowledge = _validate_knowledge_types(knowledge_types)
    config_path = f"configs/model/{extension_id}.yaml"
    refs = _recipe_refs(
        operation,
        model=config_path,
        method=method_config,
        datasets=dataset_configs,
        evaluation=evaluation_config,
    )
    experiment_name = f"{_operation_dir(operation)}/{extension_id}_recipe"
    preview, argv = build_equivalent_cli(
        operation,
        experiment=experiment_name,
        task_name=f"extension_{extension_id}_smoke",
        seed=seed,
    )
    manifest = _base_manifest(
        extension_id=extension_id,
        kind="model",
        operation=operation,
        handler="AutoModelForCausalLM",
        config_path=config_path,
        model_families=(family,),
        knowledge_types=knowledge,
        data_formats=(),
        architecture_specific=False,
        refs=refs,
        model_hparams=None,
        requires=None,
        experiment_name=experiment_name,
        smoke_argv=argv,
    )
    tokenizer_args: dict[str, Any] = {
        "pretrained_model_name_or_path": tokenizer_reference,
        "padding_side": "right",
    }
    if chat_template:
        tokenizer_args["chat_template"] = chat_template
    model_config = {
        "model_handler": "AutoModelForCausalLM",
        "model_args": {
            "pretrained_model_name_or_path": model_reference,
            "torch_dtype": dtype,
            "attn_implementation": attention,
        },
        "tokenizer_args": tokenizer_args,
    }
    experiment_path = f"configs/experiment/{experiment_name}.yaml"
    files = {
        config_path: _yaml_bytes(model_config),
        experiment_path: _yaml_bytes(
            _experiment_mapping(
                operation, refs, task_name=f"extension_{extension_id}_smoke"
            ),
            header="# @package _global_\n# Static preview first; smoke requires explicit user execution.",
        ),
        "data/sample/README.md": (
            b"Place a reviewed one-record text sample here before smoke validation.\n"
        ),
        "README.md": _readme(extension_id, "model", operation, planned=False),
    }
    return _finalize_draft(
        manifest, files, preview, existing_ids=existing_ids, planned=False
    )


def _input_format(source: Any, filename: str | None = None) -> str:
    fmt = Path(_source_filename(source, filename)).suffix.lower().lstrip(".")
    if fmt not in {"json", "jsonl", "csv"}:
        raise ExtensionInputError("Dataset upload must be .json, .jsonl, or .csv.")
    return fmt


def build_dataset_extension_draft(
    *,
    extension_id: str,
    operation: str,
    source: Any,
    mapping: str | Mapping[str, str],
    knowledge_type: str,
    format_type: str = "custom",
    retain_source: Any | None = None,
    family: str = "qwen2",
    model_config: str | None = None,
    method_config: str | None = None,
    evaluation_config: str | None = None,
    seed: int = 42,
    existing_ids: Iterable[str] = (),
) -> ExtensionDraft:
    """Build a normalized dataset extension, including forget/retain pairing."""

    extension_id = validate_identifier(extension_id)
    operation = _validate_operation(operation)
    family = _validate_family(family)
    knowledge = _validate_knowledge_types((knowledge_type,))
    source_format = _input_format(source)
    records = parse_data_file(source)
    normalized: dict[str, list[dict[str, Any]]] = {}
    if operation == "unlearning":
        if retain_source is None:
            raise ExtensionInputError("Unlearning requires a retain upload.")
        normalized["forget"] = canonicalize_records(
            operation,
            records,
            mapping,
            knowledge_type=knowledge_type,
            format_type=format_type,
            role="forget",
        )
        normalized["retain"] = canonicalize_records(
            operation,
            parse_data_file(retain_source),
            mapping,
            knowledge_type=knowledge_type,
            format_type=format_type,
            role="retain",
        )
    else:
        role = DATASET_ROLES[operation][0]
        normalized[role] = canonicalize_records(
            operation,
            records,
            mapping,
            knowledge_type=knowledge_type,
            format_type=format_type,
            role=role,
        )

    config_paths = {
        role: f"configs/data/datasets/{extension_id}_{role}.yaml" for role in normalized
    }
    refs = _recipe_refs(
        operation,
        model=model_config,
        method=method_config,
        datasets=config_paths,
        evaluation=evaluation_config,
    )
    experiment_name = f"{_operation_dir(operation)}/{extension_id}_recipe"
    preview, argv = build_equivalent_cli(
        operation,
        experiment=experiment_name,
        task_name=f"extension_{extension_id}_smoke",
        seed=seed,
    )
    handler = {
        "unlearning": "QADataset",
        # Uploaded Alpaca/ShareGPT/custom records are normalized to the tested
        # canonical Alpaca keys before export.
        "inject": "AlpacaDataset",
        "edit": "EditingDataset",
    }[operation]
    manifest = _base_manifest(
        extension_id=extension_id,
        kind="dataset",
        operation=operation,
        handler=handler,
        config_path=config_paths[DATASET_ROLES[operation][0]],
        model_families=(family,),
        knowledge_types=knowledge,
        data_formats=tuple(dict.fromkeys((source_format, "jsonl"))),
        architecture_specific=False,
        refs=refs,
        model_hparams=None,
        requires=None,
        experiment_name=experiment_name,
        smoke_argv=argv,
    )
    files: dict[str, bytes] = {
        "README.md": _readme(extension_id, "dataset", operation, planned=False),
    }
    for role, samples in normalized.items():
        sample_path = f"data/sample/{role}.jsonl"
        files[sample_path] = _jsonl_bytes(samples)
        if operation == "unlearning":
            dataset_args = {
                "hf_args": {
                    "path": "json",
                    "data_files": sample_path,
                    "split": "train",
                },
                "question_key": "prompt",
                "answer_key": "target_old",
                "max_length": 512,
            }
        elif operation == "inject":
            dataset_args = {
                "data_path": sample_path,
                "format_type": "alpaca",
                "instruction_key": "prompt",
                "input_key": "input",
                "output_key": "target_new",
                "max_length": 2048,
            }
        else:
            dataset_args = {
                "data_path": sample_path,
                "prompt_key": "prompt",
                "subject_key": "subject",
                "target_new_key": "target_new",
                "target_old_key": "target_old",
                "max_length": 512,
            }
        files[config_paths[role]] = _yaml_bytes(
            {f"{extension_id}_{role}": {"handler": handler, "args": dataset_args}}
        )
    files[f"configs/experiment/{experiment_name}.yaml"] = _yaml_bytes(
        _experiment_mapping(
            operation, refs, task_name=f"extension_{extension_id}_smoke"
        ),
        header="# @package _global_\n# Static preview first; smoke requires explicit user execution.",
    )
    return _finalize_draft(
        manifest, files, preview, existing_ids=existing_ids, planned=False
    )


def _algorithm_source(operation: str, handler: str) -> bytes:
    if operation == "unlearning":
        source = f'''"""Generated planned scaffold; review and implement before registration."""

from trainer.unlearn.base import UnlearnTrainer


class {handler}(UnlearnTrainer):
    """TODO: implement and test the unlearning objective."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        raise NotImplementedError("Implement compute_loss() before registration")
'''
    elif operation == "inject":
        source = f'''"""Generated planned scaffold; review and implement before registration."""

from trainer.inject.base import InjectTrainer


class {handler}(InjectTrainer):
    """TODO: construct and attach a reviewed PEFT adapter."""

    def setup_peft(self, model, parameter_schema):
        # Build a peft config and call get_peft_model only after dependency review.
        raise NotImplementedError("Implement PEFT setup before registration")
'''
    else:
        source = f'''"""Generated planned scaffold; review and implement before registration."""

from trainer.edit.base import EditRequest, EditTrainer


class {handler}(EditTrainer):
    """TODO: implement a text EditRequest update with side-effect evaluation."""

    def edit(self, requests: EditRequest | list[EditRequest], **kwargs):
        raise NotImplementedError("Implement edit(EditRequest) before registration")
'''
    return source.encode("utf-8")


def build_algorithm_extension_draft(
    *,
    extension_id: str,
    operation: str,
    handler: str,
    parameter_schema: str | Mapping[str, Any] | None,
    model_families: str | Iterable[str],
    knowledge_types: str | Iterable[str] = ("fact",),
    architecture_specific: bool = False,
    model_config: str | None = None,
    dataset_configs: Mapping[str, str] | None = None,
    evaluation_config: str | None = None,
    python_packages: str | Iterable[str] = (),
    required_files: str | Iterable[str] = (),
    checkpoints: str | Iterable[str] = (),
    statistics: str | Iterable[str] = (),
    seed: int = 42,
    existing_ids: Iterable[str] = (),
) -> ExtensionDraft:
    """Build a non-runnable algorithm source/config scaffold."""

    extension_id = validate_identifier(extension_id)
    operation = _validate_operation(operation)
    handler = validate_handler_name(handler)
    families = _validate_families(model_families)
    knowledge = _validate_knowledge_types(knowledge_types)
    schema = parse_parameter_schema(parameter_schema)
    operation_dir = _operation_dir(operation)
    trainer_subdir = "" if operation == "unlearning" else f"{operation_dir}/"
    config_path = f"configs/trainer/{trainer_subdir}{extension_id}.yaml"
    refs = _recipe_refs(
        operation,
        model=model_config,
        method=config_path,
        datasets=dataset_configs,
        evaluation=evaluation_config,
    )
    model_hparams: str | None = None
    files: dict[str, bytes] = {}
    if architecture_specific:
        model_hparams = f"configs/hparams/{extension_id}/{families[0]}.yaml"
        files[model_hparams] = _yaml_bytes(
            {
                "model_family": families[0],
                "method": extension_id,
                "parameters": {
                    name: definition.get("default")
                    for name, definition in schema.items()
                    if "default" in definition
                },
            },
            header="# TODO: verify every architecture-specific module/layer mapping.",
        )
    experiment_name = f"{operation_dir}/{extension_id}_recipe"
    preview, argv = build_equivalent_cli(
        operation,
        experiment=experiment_name,
        task_name=f"extension_{extension_id}_smoke",
        seed=seed,
    )
    manifest = _base_manifest(
        extension_id=extension_id,
        kind="algorithm",
        operation=operation,
        handler=handler,
        config_path=config_path,
        model_families=families,
        knowledge_types=knowledge,
        data_formats=(),
        architecture_specific=architecture_specific,
        refs=refs,
        model_hparams=model_hparams,
        requires={
            "python_packages": _list_value(python_packages),
            "files": _list_value(required_files),
            "checkpoints": _list_value(checkpoints),
            "statistics": _list_value(statistics),
        },
        experiment_name=experiment_name,
        smoke_argv=argv,
    )
    defaults = {
        "unlearning": ["finetune"],
        "inject": ["inject/base_inject@_here_", "_self_"],
        "edit": ["edit/base_editor@_here_", "_self_"],
    }[operation]
    method_args = {
        name: definition["default"]
        for name, definition in schema.items()
        if "default" in definition
    }
    module_id = extension_id.replace("-", "_")
    files.update(
        {
            config_path: _yaml_bytes(
                {"defaults": defaults, "handler": handler, "method_args": method_args},
                header="# PLANNED: generated scaffold, not a runnable implementation.",
            ),
            f"src/trainer/{operation_dir}/{module_id}.py": _algorithm_source(
                operation, handler
            ),
            "data/sample/README.md": (
                b"Add a reviewed one-record text sample before smoke validation.\n"
            ),
            f"configs/experiment/{experiment_name}.yaml": _yaml_bytes(
                _experiment_mapping(
                    operation, refs, task_name=f"extension_{extension_id}_smoke"
                ),
                header="# @package _global_\n# PLANNED: compose-only until implementation and registration.",
            ),
            "README.md": _readme(extension_id, "algorithm", operation, planned=True),
        }
    )
    return _finalize_draft(
        manifest, files, preview, existing_ids=existing_ids, planned=True
    )


def create_extension_draft(kind: str, **kwargs: Any) -> ExtensionDraft:
    """Generic dispatcher used by non-Gradio clients."""

    kind = str(kind or "").strip().lower()
    builders = {
        "model": build_model_extension_draft,
        "dataset": build_dataset_extension_draft,
        "algorithm": build_algorithm_extension_draft,
    }
    if kind not in builders:
        raise ExtensionInputError(f"Extension kind must be one of: {', '.join(KINDS)}.")
    return builders[kind](**kwargs)


def export_extension_zip(draft: ExtensionDraft) -> str:
    """Write a deterministic bundle below the system temp directory only."""

    if draft.has_errors:
        raise ExtensionInputError("Resolve error diagnostics before export.")
    extension_id = validate_identifier(draft.extension_id)
    with tempfile.NamedTemporaryFile(
        mode="w+b", prefix=f"{extension_id}-", suffix=".zip", delete=False
    ) as output:
        with zipfile.ZipFile(
            output, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for relative_path, content in sorted(draft.files.items()):
                relative_path = validate_relative_path(relative_path)
                info = zipfile.ZipInfo(
                    filename=f"{extension_id}/{relative_path}",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, bytes(content))
        path = output.name
    if not Path(path).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        Path(path).unlink(missing_ok=True)
        raise RuntimeError("Extension archive escaped the system temporary directory.")
    return path


build_extension_zip = export_extension_zip


def render_diagnostics(diagnostics: Sequence[Diagnostic]) -> str:
    """Render diagnostics as compact bilingual Markdown."""

    icon = {"error": "ERROR", "warning": "WARNING", "info": "INFO"}
    return "\n".join(
        f"- **{icon.get(item.level, item.level.upper())} · {item.code}** — {item.message}"
        for item in diagnostics
    )


def review_extension(draft: ExtensionDraft) -> dict[str, Any]:
    """Return JSON-serializable review data for the UI/API."""

    return {
        "manifest": dict(draft.manifest),
        "files": sorted(draft.files),
        "diagnostics": [item.to_dict() for item in draft.diagnostics],
        "equivalent_cli": draft.equivalent_cli,
        "exportable": not draft.has_errors,
    }


def _record_mapping(record: Any) -> dict[str, Any]:
    manifest = getattr(record, "manifest", None)
    if manifest is not None and hasattr(manifest, "to_dict"):
        manifest = manifest.to_dict()
    elif manifest is not None and not isinstance(manifest, Mapping):
        manifest = vars(manifest)
    manifest = dict(manifest or {})
    reasons = tuple(getattr(record, "reasons", ()) or ())
    return {
        "id": str(getattr(record, "id", manifest.get("id", ""))),
        "status": str(getattr(record, "status", "disabled")).lower(),
        "reasons": reasons,
        "source": str(getattr(record, "source", "")),
        "default_runnable": bool(getattr(record, "default_runnable", False)),
        "kind": manifest.get("kind", ""),
        "operation": manifest.get("operation", ""),
        "handler": manifest.get("handler", ""),
        "modality": manifest.get("modality", ""),
    }


def separate_catalog_records(records: Iterable[Any]) -> CatalogSections:
    """Keep verified runnable extensions out of Experimental/Disabled lists."""

    default: list[dict[str, Any]] = []
    experimental: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    for raw in records:
        record = _record_mapping(raw)
        if record["source"] == "built_in_curated" and record["default_runnable"]:
            default.append(record)
        elif record["modality"] != "text" or str(record["handler"]).startswith("MM"):
            record["reasons"] = tuple(record["reasons"]) + (
                "text wizard excludes multimodal components",
            )
            disabled.append(record)
        elif (
            record["status"] in {"smoke_verified", "benchmark_verified"}
            and record["default_runnable"]
        ):
            default.append(record)
        elif record["status"] in {"planned", "configured"}:
            experimental.append(record)
        else:
            disabled.append(record)

    def sort_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
        return (str(item["operation"]), str(item["kind"]), str(item["id"]))

    return CatalogSections(
        default=tuple(sorted(default, key=sort_key)),
        experimental=tuple(sorted(experimental, key=sort_key)),
        disabled=tuple(sorted(disabled, key=sort_key)),
    )


def refresh_extension_catalog(
    project_root: str | Path | None = None,
) -> CatalogSections:
    """Refresh installed extension evidence through the optional strict registry."""

    try:
        registry = importlib.import_module("src.extension_registry")
        discover = getattr(registry, "discover_extensions")
        return separate_catalog_records(discover(project_root=project_root))
    except (ImportError, AttributeError) as exc:
        return CatalogSections(
            default=(),
            experimental=(),
            disabled=(),
            diagnostics=(f"Extension registry is unavailable: {exc}",),
        )
    except (
        Exception
    ) as exc:  # A broken installed extension must remain visible as a diagnostic.
        return CatalogSections(
            default=(),
            experimental=(),
            disabled=(),
            diagnostics=(f"Extension discovery failed safely: {exc}",),
        )


def catalog_markdown(sections: CatalogSections) -> str:
    """Render catalog trust boundaries for the Gradio refresh panel."""

    def lines(title: str, records: Sequence[Mapping[str, Any]]) -> list[str]:
        result = [f"### {title}"]
        if not records:
            return result + ["_None / 无_\n"]
        for record in records:
            reasons = "; ".join(record.get("reasons", ())) or "—"
            result.append(
                f"- `{record['id']}` · {record['kind']} · {record['operation']} · "
                f"**{record['status']}** — {reasons}"
            )
        result.append("")
        return result

    output = lines("Default runnable / 默认可运行", sections.default)
    output += lines("Experimental / 实验性", sections.experimental)
    output += lines("Disabled / 已禁用", sections.disabled)
    if sections.diagnostics:
        output += ["### Diagnostics / 诊断"] + [
            f"- {item}" for item in sections.diagnostics
        ]
    return "\n".join(output)


def installed_extension_ids(project_root: str | Path | None = None) -> tuple[str, ...]:
    """Return ids from every catalog section for duplicate prevention."""

    sections = refresh_extension_catalog(project_root)
    return tuple(
        sorted(
            {
                record["id"]
                for group in (
                    sections.default,
                    sections.experimental,
                    sections.disabled,
                )
                for record in group
            }
        )
    )
