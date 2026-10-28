"""MIT MEMIT input second moments, rebuilt for bounded local statistics.

Adapted from ``rome/layer_stats.py`` in kmeng01/memit at
80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b, copyright (c) 2022 Kevin Meng.
See LICENSES/MEMIT.txt and THIRD_PARTY_NOTICES.md. The native
v3 cache, sequential row sampling and resource/state checks are project
adaptations; upstream precomputed statistics are not silently reused.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
import logging
import os
from pathlib import Path
import random
import tempfile

import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from trainer.edit.utils import nethook
from trainer.edit.utils.cache_identity import (
    identity_digest,
    model_cache_identity,
    tokenizer_identity,
)
from trainer.edit.utils.compute_z import _frozen_model, _hidden_state, _input_device

logger = logging.getLogger(__name__)
CACHE_SCHEMA = 3


def _atomic_save(payload: Mapping, destination: Path) -> None:
    """Replace a cache only after a complete torch save in the same directory."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(filename)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_max_length(config) -> int:
    for key in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "seq_length",
    ):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    return 2048


def _load_statistics_split(dataset_name: str, dataset_config: str):
    """Read prepared common corpora before consulting the Hub.

    The datasets cache builder preserves the original Arrow fingerprint and
    avoids both a Hub metadata request and process-wide offline settings.
    Missing caches retain the normal CLI download behavior; unreadable or
    damaged caches must remain visible to the caller.
    """
    if dataset_name in {"wikipedia", "wikitext"}:
        try:
            from datasets.packaged_modules.cache.cache import Cache
        except ImportError:
            pass  # Older datasets versions may not provide this cache reader.
        else:
            try:
                builder = Cache(
                    repo_id=dataset_name,
                    dataset_name=dataset_name,
                    config_name=dataset_config,
                    version="auto",
                    hash="auto",
                )
            except ValueError as exc:
                # Cache's exact-config lookup uses ValueError for a miss. Do
                # not hide JSONDecodeError or other corrupt-cache failures.
                if not str(exc).startswith(f"Couldn't find cache for {dataset_name}"):
                    raise
            else:
                return builder.as_dataset(split="train")
    return load_dataset(dataset_name, dataset_config)["train"]


def _validated_stats(payload, *, source):
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"Invalid layer statistics from {source}: expected a mapping"
        )
    matrix, count = payload.get("mom2"), payload.get("count")
    if (
        not isinstance(matrix, torch.Tensor)
        or matrix.ndim != 2
        or not matrix.numel()
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise RuntimeError(
            f"Invalid layer statistics from {source}: mom2 must be a non-empty square tensor"
        )
    if not torch.isfinite(matrix).all():
        raise RuntimeError(
            f"Invalid layer statistics from {source}: mom2 contains non-finite values"
        )
    if (
        not isinstance(count, torch.Tensor)
        or count.numel() != 1
        or count.dtype not in (torch.int32, torch.int64)
        or int(count.item()) <= 0
    ):
        raise RuntimeError(
            f"Invalid layer statistics from {source}: count must be a positive integer scalar tensor"
        )
    return {"mom2": matrix.detach().cpu(), "count": count.detach().cpu()}


def _text_batches(split, *, limit, key, size):
    rows = iter(split) if limit is None else islice(iter(split), limit)
    while chunk := list(islice(rows, size)):
        texts = [
            row[key]
            for row in chunk
            if isinstance(row, Mapping)
            and isinstance(row.get(key), str)
            and row[key].strip()
        ]
        if texts:
            yield texts


def _upstream_text_batches(
    split, tokenizer, *, limit, key, max_length, token_budget, seed
):
    """Match MEMIT's seeded subset and length_collation within groups of 100."""
    indices = list(range(len(split)))
    if limit is not None:
        random.Random(seed).shuffle(indices)
        indices = indices[:limit]
    for offset in range(0, len(indices), 100):
        rows = []
        for index in indices[offset : offset + 100]:
            text = split[index][key]
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            if tokens:
                rows.append((len(tokens), text))
        rows.sort(key=lambda item: -item[0])
        batch, width = [], 0
        for length, text in rows:
            if batch and width * (len(batch) + 1) > token_budget:
                yield batch
                batch, width = [], 0
            if not batch:
                width = length
            batch.append(text)
        if batch:
            yield batch


def layer_stats(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_name: str,
    stats_dir: str | Path,
    ds_name: str = "wikipedia",
    to_collect: list[str] | None = None,
    *,
    sample_size: int | None = 100_000,
    batch_tokens: int | None = None,
    precision: str = "float32",
    force_recompute: bool = False,
    dataset_config: str | None = None,
    sample_policy: str = "sequential_rows",
    sample_seed: int = 1,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Estimate E[key keyᵀ] over valid tokens and persist an identity-bound cache.

    Only ``mom2`` is exposed by this compatibility API. Native sampling consumes
    rows sequentially; ``upstream_random`` uses the authors' seeded subset and
    length collation. Batching respects a padded-token upper bound. Empty inputs
    fail without creating an artificial zero covariance. ``max_length`` can
    impose a method-specific sequence cap independently of the batch budget;
    omitting it preserves the original context limit and cache identity.
    """
    selected = list(to_collect or ["mom2"])
    if selected != ["mom2"]:
        raise ValueError("This layer_stats adapter supports only to_collect=['mom2']")
    if sample_size is not None and (
        not isinstance(sample_size, int) or sample_size < 0
    ):
        raise ValueError("sample_size must be non-negative or None")
    if precision not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValueError(f"Unsupported statistics precision: {precision}")
    if sample_policy not in {"sequential_rows", "upstream_random"}:
        raise ValueError(f"Unsupported statistics sample_policy: {sample_policy}")
    max_length_override = max_length
    if max_length_override is not None and (
        isinstance(max_length_override, bool)
        or not isinstance(max_length_override, int)
        or max_length_override <= 0
    ):
        raise ValueError("max_length must be a positive integer or None")
    max_length = _resolve_max_length(model.config)
    if max_length_override is not None:
        max_length = min(max_length, max_length_override)
    token_budget = batch_tokens if batch_tokens is not None else max_length * 3
    if max_length <= 0 or not isinstance(token_budget, int) or token_budget <= 0:
        raise ValueError("Model context and batch_tokens must be positive integers")
    max_length = min(max_length, token_budget)
    batch_size = min(8, max(1, token_budget // max_length))
    nethook.get_module(model, layer_name)
    dataset_config = dataset_config or {
        "wikitext": "wikitext-103-raw-v1",
        "wikipedia": "20220301.en",
    }.get(ds_name, ds_name)
    split = _load_statistics_split(ds_name, dataset_config)
    columns = getattr(split, "column_names", None) or ["text"]
    text_key = "text" if "text" in columns else columns[0]
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "implementation": "memit_mit_native_second_moment_v3",
        "upstream_revision": "80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b",
        "model": model_cache_identity(model),
        "tokenizer": tokenizer_identity(tokenizer),
        "layer_name": layer_name,
        "dataset": ds_name,
        "dataset_config": dataset_config,
        "dataset_fingerprint": getattr(split, "_fingerprint", None),
        "text_key": text_key,
        "sample_size": sample_size,
        "sample_policy": sample_policy,
        "max_length": max_length,
        "batch_size": batch_size,
        "batch_tokens": token_budget,
        "padding_policy": "right",
        "precision": precision,
        "to_collect": selected,
        "model_mode": "eval",
    }
    if sample_policy == "upstream_random":
        metadata.update(
            implementation="memit_mit_reference_second_moment_v1",
            sample_seed=sample_seed,
            batch_size=100,
            collation="length_descending_token_budget",
        )
    if max_length_override is not None:
        metadata["max_length_override"] = max_length_override
    path = Path(stats_dir) / f"v{CACHE_SCHEMA}" / f"{identity_digest(metadata)}.pt"
    if (
        path.exists()
        and not force_recompute
        and metadata["dataset_fingerprint"] is not None
    ):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, Mapping) or payload.get("metadata") != metadata:
            raise ValueError(f"Layer statistics cache identity mismatch: {path}")
        return _validated_stats(payload, source=path)

    total = None
    count = 0
    dtype = getattr(torch, precision)
    input_device = _input_device(model)
    with _frozen_model(model), torch.no_grad():
        batches = (
            _upstream_text_batches(
                split,
                tokenizer,
                limit=sample_size,
                key=text_key,
                max_length=max_length,
                token_budget=token_budget,
                seed=sample_seed,
            )
            if sample_policy == "upstream_random"
            else _text_batches(split, limit=sample_size, key=text_key, size=batch_size)
        )
        for texts in batches:
            encoded = tokenizer(
                texts,
                padding=True,
                # Per-call policy preserves the tokenizer's generation setting
                # while keeping absolute positions equal to single-row forwards.
                padding_side="right",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {
                key: value.to(input_device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            mask = batch.get("attention_mask")
            if not isinstance(mask, torch.Tensor):
                raise RuntimeError("Tokenizer output has no attention_mask tensor")
            with nethook.Trace(
                model, layer_name, retain_input=True, retain_output=False, stop=True
            ) as captured:
                model(**batch)
            features = _hidden_state(captured.input)
            if features.ndim != 3 or tuple(features.shape[:2]) != tuple(mask.shape):
                raise RuntimeError(
                    f"Layer {layer_name} produced no valid batch-first input activations"
                )
            values = features[mask.to(features.device).bool()].to(dtype=dtype)
            if not values.shape[0]:
                continue
            contribution = values.T @ values
            total = (
                contribution if total is None else total + contribution.to(total.device)
            )
            count += values.shape[0]
    if total is None or count == 0:
        raise RuntimeError(
            f"No valid token activations collected for layer {layer_name}; statistics cache was not written"
        )
    result = _validated_stats(
        {"mom2": total / count, "count": torch.tensor(count, dtype=torch.int64)},
        source=f"computed layer {layer_name}",
    )
    _atomic_save({**result, "metadata": metadata}, path)
    logger.info("Stored layer input second moment: %s (%d tokens)", layer_name, count)
    return result
