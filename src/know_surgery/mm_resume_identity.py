"""Content identities for independent multimodal editing resumes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.Image import Image as PILImage


RESUME_IDENTITY_VERSION = 2
_IMAGE_FIELDS = {"image", "image_rephrase", "images"}


def _content_value(value: Any, *, image_field: bool = False) -> Any:
    """Include pixels from the live images actually consumed by the editor."""
    if image_field and isinstance(value, (str, Path)):
        # Only declared image fields are paths; prompts/answers may themselves
        # look like filenames and must remain ordinary text.
        with Image.open(value) as image:
            return {"path": str(value), "image": _content_value(image)}
    if isinstance(value, PILImage):
        # RGBA also captures palette/transparency differences for images supplied
        # directly by callers. Dataset loaders normally already provide RGB.
        return {
            "type": "PIL.Image",
            "mode": value.mode,
            "size": list(value.size),
            "pixels_sha256": hashlib.sha256(
                value.convert("RGBA").tobytes()
            ).hexdigest(),
        }
    if is_dataclass(value):
        return {
            field.name: _content_value(
                getattr(value, field.name), image_field=field.name in _IMAGE_FIELDS
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _content_value(item, image_field=key in _IMAGE_FIELDS)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_content_value(item, image_field=image_field) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_mm_resume_identity(
    resolved_config: Mapping[str, Any],
    requests,
    *,
    checkpoint_identity: Mapping[str, Any],
    processor_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a run to its config, loaded checkpoint/processor, and image contents.

    The caller computes the checkpoint identity once before any case is edited;
    individual case keys only depend on this manifest's digest and their index.
    """
    identity = {
        "schema_version": RESUME_IDENTITY_VERSION,
        "configuration_sha256": _digest(resolved_config),
        "checkpoint": dict(checkpoint_identity),
        "processor_sha256": _digest(processor_identity),
        "request_sha256s": [_digest(_content_value(request)) for request in requests],
    }
    return {**identity, "run_identity": _digest(identity)}


def validate_mm_resume_identity(saved: Any, current: Mapping[str, Any]) -> None:
    """Reject legacy or changed inputs before overwriting existing run metadata."""
    if (
        not isinstance(saved, Mapping)
        or saved.get("schema_version") != RESUME_IDENTITY_VERSION
    ):
        raise RuntimeError(
            "Independent MM resume lacks a supported content identity; use a new "
            "output directory. / 旧续跑记录缺少受支持的内容身份，请使用新的输出目录。"
        )
    if saved != current:
        raise RuntimeError(
            "Independent MM resume inputs changed (configuration, checkpoint, "
            "processor or requests); use a new output directory. / 续跑输入已变化，"
            "请使用新的输出目录，避免混用旧指标。"
        )
