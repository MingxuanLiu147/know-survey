"""Portable identities for reusable editing statistics and projections."""

from __future__ import annotations

import hashlib
import json

from trainer.edit.checkpoint_identity import stream_model_persistent_state_sha256


def identity_digest(value) -> str:
    """Hash a JSON-compatible identity without depending on a local pathname."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def tokenizer_identity(tokenizer) -> dict:
    """Identify vocabulary, tokenization implementation and special-token policy."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_config = json.loads(backend.to_str()) if backend is not None else None
    if backend_config is not None:
        # These are changed by individual encode calls, not tokenizer identity.
        backend_config.pop("padding", None)
        backend_config.pop("truncation", None)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "backend": backend_config,
        "vocab": get_vocab() if callable(get_vocab) else None,
        "special_tokens": dict(getattr(tokenizer, "special_tokens_map", {})),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }


def model_cache_identity(model) -> dict:
    """Bind a cache to exact parameters/buffers and computation config."""
    config = model.config
    raw = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    raw = {
        key: value
        for key, value in raw.items()
        if key
        not in {
            "_name_or_path",
            "transformers_version",
            "_commit_hash",
            "torch_dtype",
            "dtype",
            "architectures",
        }
    }
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "config": json.loads(json.dumps(raw, default=str)),
        **stream_model_persistent_state_sha256(model),
    }
