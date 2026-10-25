"""Bounded data import and operation-aware normalization for WebUI uploads.

This module validates JSON, JSONL and CSV inputs without importing uploaded code
or loading models. The canonical records reuse the public data contract.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


MAX_RECORDS = 100_000


MAX_CSV_COLUMNS = 256


OPERATIONS = ("unlearning", "inject", "edit")


KNOWLEDGE_TYPES = (
    "fact",
    "event",
    "procedural",
    "rule",
    "free_text",
    "concept",
    "multimodal",
)


DATASET_ROLES = {
    "unlearning": ("forget", "retain"),
    "inject": ("train",),
    "edit": ("edit",),
}


_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


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


def validate_identifier(value: str, *, field: str = "extension id") -> str:
    """Validate a stable lowercase extension identifier."""

    value = str(value or "").strip()
    if not _ID_RE.fullmatch(value) or len(value) > 64:
        raise ExtensionInputError(
            f"Invalid {field}: {value!r}. Use 1-64 lowercase letters, digits, '_' or '-', "
            "starting with a letter. / 标识符须以小写字母开头。"
        )
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


def _validate_operation(operation: str) -> str:
    operation = str(operation or "").strip().lower()
    if operation not in OPERATIONS:
        raise ExtensionInputError(f"Operation must be one of: {', '.join(OPERATIONS)}.")
    return operation


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
