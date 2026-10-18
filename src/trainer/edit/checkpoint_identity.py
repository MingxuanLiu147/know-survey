"""Deterministic, bounded-memory base-model identity for learned editors.

Compact learned editors are trained against one exact set of base-model
weights. Architecture/shape checks alone are insufficient: two random
initialisations have identical shapes but require different editor networks.

The weight digest below is intentionally streamed one bounded tensor slice at
a time. It never materialises a CPU copy of the full model (or even a full
large parameter), which keeps the identity gate usable for sharded LLMs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import nn

_IDENTITY_FORMAT = "know_surgery_base_model_identity_v2"
_STATE_HASH_ALGORITHM = "sha256_persistent_state_stream_v2"
_DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
_VOLATILE_CONFIG_KEYS = {
    "_commit_hash",
    "_name_or_path",
    "_attn_implementation",
    "_attn_implementation_autoset",
    "_attn_implementation_internal",
    "attn_implementation",
    "name_or_path",
    "transformers_version",
    # save_pretrained writes these loading hints; actual tensor dtypes are
    # already included in the persistent-state digest.
    "torch_dtype",
    "dtype",
    "architectures",
}


def _qualified_type(value: Any) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _normalise_config_value(value: Any) -> Any:
    """Convert model config values to a stable, JSON-serialisable form."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, type):
        return {"__type__": _qualified_type(value)}
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "Model config tensors are unsupported in checkpoint identity; "
            "convert the value to a deterministic JSON scalar or list"
        )
    if callable(value):
        raise TypeError(
            "Model config callables are unsupported in checkpoint identity; "
            "store a stable qualified name instead"
        )
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, (torch.dtype, torch.device, Path)):
        return str(value)
    if isinstance(value, Enum):
        return _normalise_config_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_config_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalised = [_normalise_config_value(item) for item in value]
        return sorted(normalised, key=_canonical_json)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise_config_value(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if converted is value:
            raise TypeError(
                "Model config to_dict() returned itself and cannot be normalised "
                f"deterministically: {_qualified_type(value)}"
            )
        return _normalise_config_value(converted)
    raise TypeError(
        "Model config contains a value that cannot be normalised for checkpoint "
        f"identity: {_qualified_type(value)}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def model_config_identity(model: nn.Module) -> dict[str, str]:
    """Return class identity and a canonical SHA256 of model.config."""

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(
            "Learned-editor checkpoint identity requires the base model to expose "
            "a config object"
        )
    to_dict = getattr(config, "to_dict", None)
    raw_config = to_dict() if callable(to_dict) else config
    canonical = _canonical_json(
        {
            "config_class": _qualified_type(config),
            "config": _normalise_config_value(raw_config),
        }
    )
    return {
        "model_class": _qualified_type(model),
        "config_class": _qualified_type(config),
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _iter_bounded_tensor_slices(
    tensor: torch.Tensor, max_elements: int
) -> Iterator[torch.Tensor]:
    """Yield slices in logical row-major order without a full contiguous copy."""

    if tensor.numel() <= max_elements or tensor.ndim == 0:
        yield tensor
        return
    if tensor.shape[0] == 0:
        return

    trailing_elements = tensor[0].numel()
    if trailing_elements <= max_elements:
        step = max(1, max_elements // max(trailing_elements, 1))
        for start in range(0, tensor.shape[0], step):
            yield tensor.narrow(0, start, min(step, tensor.shape[0] - start))
        return

    for index in range(tensor.shape[0]):
        yield from _iter_bounded_tensor_slices(tensor.select(0, index), max_elements)


def _update_framed(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def _iter_persistent_state(
    model: nn.Module,
) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Yield state-dict-equivalent parameters and persistent buffers."""

    try:
        parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        parameters = model.named_parameters()
    for name, parameter in parameters:
        yield "parameter", name, parameter

    try:
        modules = model.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        modules = model.named_modules()
    for module_prefix, module in modules:
        non_persistent = module._non_persistent_buffers_set
        for buffer_name, buffer in module._buffers.items():
            if buffer is None or buffer_name in non_persistent:
                continue
            full_name = (
                f"{module_prefix}.{buffer_name}" if module_prefix else buffer_name
            )
            yield "buffer", full_name, buffer


def stream_model_persistent_state_sha256(
    model: nn.Module, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES
) -> dict[str, Any]:
    """Hash parameters and persistent buffers with bounded CPU staging memory."""

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    hasher = hashlib.sha256()
    state_count = 0
    parameter_count = 0
    buffer_count = 0
    state_numel = 0
    with torch.no_grad():
        for kind, name, state_tensor in sorted(
            _iter_persistent_state(model), key=lambda item: (item[1], item[0])
        ):
            tensor = state_tensor.detach()
            if tensor.device.type == "meta":
                raise ValueError(
                    "Cannot compute learned-editor checkpoint identity while base "
                    f"{kind} {name!r} is on the meta device"
                )
            if tensor.layout != torch.strided:
                raise ValueError(
                    "Cannot compute learned-editor checkpoint identity for "
                    f"non-strided {kind} {name!r} ({tensor.layout})"
                )

            state_count += 1
            parameter_count += int(kind == "parameter")
            buffer_count += int(kind == "buffer")
            state_numel += tensor.numel()
            metadata = {
                "kind": kind,
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
            _update_framed(hasher, _canonical_json(metadata))
            max_elements = max(1, chunk_bytes // max(tensor.element_size(), 1))
            for tensor_slice in _iter_bounded_tensor_slices(tensor, max_elements):
                cpu_slice = tensor_slice.to(
                    device="cpu",
                    copy=True,
                    memory_format=torch.contiguous_format,
                )
                cpu_flat = torch.empty(
                    cpu_slice.numel(), device="cpu", dtype=cpu_slice.dtype
                )
                cpu_flat.copy_(cpu_slice.reshape(-1))
                raw_bytes = cpu_flat.view(torch.uint8).numpy()
                hasher.update(memoryview(raw_bytes))

    return {
        "state_sha256": hasher.hexdigest(),
        "state_count": state_count,
        "parameter_count": parameter_count,
        "buffer_count": buffer_count,
        "state_numel": state_numel,
    }


def build_base_model_identity(model: nn.Module) -> dict[str, Any]:
    """Build the complete identity persisted in learned-editor checkpoints."""

    return {
        "format": _IDENTITY_FORMAT,
        **model_config_identity(model),
        "state_hash_algorithm": _STATE_HASH_ALGORITHM,
        **stream_model_persistent_state_sha256(model),
    }


def validate_base_model_identity(
    model: nn.Module,
    identity: Any,
    *,
    checkpoint_label: str,
) -> None:
    """Reject checkpoints trained for a different config or exact weight set."""

    if not isinstance(identity, Mapping):
        raise ValueError(
            f"{checkpoint_label} checkpoint is missing required base_model_identity; "
            "legacy checkpoints without exact base-model identity are incompatible "
            "and must be retrained"
        )
    if identity.get("format") != _IDENTITY_FORMAT:
        raise ValueError(
            f"{checkpoint_label} checkpoint has unsupported base-model identity "
            f"format {identity.get('format')!r}; expected {_IDENTITY_FORMAT!r}"
        )
    if identity.get("state_hash_algorithm") != _STATE_HASH_ALGORITHM:
        raise ValueError(
            f"{checkpoint_label} checkpoint has unsupported persistent-state hash "
            f"algorithm {identity.get('state_hash_algorithm')!r}"
        )

    current_config = model_config_identity(model)
    config_fields = ("model_class", "config_class", "config_sha256")
    if any(identity.get(field) != current_config[field] for field in config_fields):
        raise ValueError(
            f"{checkpoint_label} checkpoint base-model config identity does not "
            "match the current model; load the exact training config and weights"
        )

    current_state = stream_model_persistent_state_sha256(model)
    state_fields = (
        "state_sha256",
        "state_count",
        "parameter_count",
        "buffer_count",
        "state_numel",
    )
    if any(identity.get(field) != current_state[field] for field in state_fields):
        raise ValueError(
            f"{checkpoint_label} checkpoint base-model persistent-state identity "
            "does not match the current model; learned editors cannot be transferred "
            "across different initialisations or modified weights/buffers"
        )
