"""MIT MEMIT input second moments, rebuilt for bounded local statistics.

Adapted from ``rome/layer_stats.py`` in kmeng01/memit at
80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b, copyright (c) 2022 Kevin Meng.
See LICENSES/MEMIT.txt and THIRD_PARTY_NOTICES.md. The native
v3 cache, sequential row sampling and resource/state checks are project
adaptations; upstream precomputed statistics are not silently reused.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
from itertools import islice
import logging
import fcntl
import os
from pathlib import Path
import random
import tempfile
import time

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from trainer.edit.utils import nethook
from trainer.edit.utils.cache_identity import (
    identity_digest,
    model_cache_identity,
    tokenizer_identity,
)
from trainer.edit.utils.compute_z import _hidden_state
from trainer.edit.utils.statistics_context import (
    statistics_execution_context,
    statistics_execution_identity,
)

logger = logging.getLogger(__name__)
CACHE_SCHEMA = 3
EXECUTION_CACHE_SCHEMA = 4


class StatisticsInterrupted(RuntimeError):
    """A deliberate bounded interruption left a resumable partial checkpoint."""


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
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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


@lru_cache(maxsize=64)
def _statistics_file_digest(path: str, stat_identity: tuple) -> str:
    """Hash once per unchanged file; stat identity invalidates replaced inputs."""
    del stat_identity
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def statistics_data_identity(data_path: str | Path | None) -> dict | None:
    """Identify explicit local corpus contents independently of mount paths."""
    if data_path is None:
        return None
    path = Path(data_path).expanduser().resolve(strict=True)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    manifest = []
    for item in files:
        if not item.is_file():
            continue
        stat = item.stat()
        digest = _statistics_file_digest(
            str(item),
            (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
        )
        manifest.append(
            {
                "file": str(item.relative_to(path)) if path.is_dir() else "corpus",
                "size": stat.st_size,
                "sha256": digest,
            }
        )
    if not manifest:
        raise ValueError(f"Local statistics corpus is empty: {path}")
    return {
        "format": "saved_dataset" if path.is_dir() else path.suffix.lower(),
        "files": manifest,
    }


def _load_statistics_split(
    dataset_name: str, dataset_config: str, data_path: str | Path | None = None
):
    """Read prepared common corpora before consulting the Hub.

    The datasets cache builder preserves the original Arrow fingerprint and
    avoids both a Hub metadata request and process-wide offline settings.
    Missing caches retain the normal CLI download behavior; unreadable or
    damaged caches must remain visible to the caller.
    """
    if data_path is not None:
        path = Path(data_path).expanduser().resolve(strict=True)
        if path.is_dir():
            split = load_from_disk(str(path))
            if isinstance(split, DatasetDict):
                if "train" not in split:
                    raise ValueError("Local statistics DatasetDict has no train split")
                split = split["train"]
            return split
        loaders = {
            ".arrow": Dataset.from_file,
            ".parquet": Dataset.from_parquet,
            ".json": Dataset.from_json,
            ".jsonl": Dataset.from_json,
        }
        if path.suffix.lower() not in loaders:
            raise ValueError(
                "Local statistics corpus must be Arrow, Parquet, JSON(L), or a saved dataset directory"
            )
        return loaders[path.suffix.lower()](str(path))
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


def _sample_state(split, limit, policy, seed):
    indices = list(range(len(split)))
    generator = random.Random(seed)
    if policy == "upstream_random" and limit is not None:
        generator.shuffle(indices)
    return (indices if limit is None else indices[:limit]), generator.getstate()


def _sample_indices(split, limit, policy, seed):
    return _sample_state(split, limit, policy, seed)[0]


def _indexed_text_batches(
    split,
    tokenizer,
    *,
    indices,
    key,
    max_length,
    token_budget,
    batch_size,
    sample_policy,
):
    """Yield exact source row IDs alongside the reviewed padded-token batches."""
    group_size = 100 if sample_policy == "upstream_random" else batch_size
    for offset in range(0, len(indices), group_size):
        rows = []
        for index in indices[offset : offset + group_size]:
            row = split[index]
            if sample_policy == "upstream_random":
                text = row[key]
                length = len(
                    tokenizer.encode(text, truncation=True, max_length=max_length)
                )
                if length:
                    rows.append((index, text, length))
            elif (
                isinstance(row, Mapping)
                and isinstance(row.get(key), str)
                and row[key].strip()
            ):
                rows.append((index, row[key], 0))
        if sample_policy == "upstream_random":
            rows.sort(key=lambda item: -item[2])
        batch, width = [], 0
        for row in rows:
            if (
                batch
                and sample_policy == "upstream_random"
                and width * (len(batch) + 1) > token_budget
            ):
                yield [r[0] for r in batch], [r[1] for r in batch]
                batch, width = [], 0
            if not batch:
                width = row[2]
            batch.append(row)
        if batch:
            yield [r[0] for r in batch], [r[1] for r in batch]


def _upstream_text_batches(
    split, tokenizer, *, limit, key, max_length, token_budget, seed
):
    """Match MEMIT's seeded subset and length_collation within groups of 100."""
    indices = _sample_indices(split, limit, "upstream_random", seed)
    for _, texts in _indexed_text_batches(
        split,
        tokenizer,
        indices=indices,
        key=key,
        max_length=max_length,
        token_budget=token_budget,
        batch_size=100,
        sample_policy="upstream_random",
    ):
        yield texts


def _tensor_sha256(tensor):
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def _validated_partial(
    payload, *, metadata, sample_order, sampler_rng_state, dimension, dtype, source
):
    if not isinstance(payload, Mapping) or payload.get("status") != "in_progress":
        raise ValueError(f"Invalid statistics partial checkpoint status: {source}")
    if (
        payload.get("metadata") != metadata
        or payload.get("sample_order") != sample_order
    ):
        raise ValueError(
            f"Statistics partial checkpoint source/protocol/sample identity mismatch: {source}"
        )
    cursor, count, total = (
        payload.get("completed_batches"),
        payload.get("count"),
        payload.get("sum"),
    )
    if (
        isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor <= 0
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or not isinstance(total, torch.Tensor)
        or total.shape != (dimension, dimension)
        or total.dtype != dtype
        or not torch.isfinite(total).all()
        or not isinstance(payload.get("batch_order_sha256"), str)
        or len(payload["batch_order_sha256"]) != 64
    ):
        raise ValueError(
            f"Invalid statistics partial checkpoint accumulator/progress: {source}"
        )
    if payload.get("sum_sha256") != _tensor_sha256(total):
        raise ValueError(
            f"Statistics partial checkpoint accumulator checksum mismatch: {source}"
        )
    if payload.get("sampler_rng_state") != sampler_rng_state:
        raise ValueError(
            f"Statistics partial checkpoint sampler RNG mismatch: {source}"
        )
    return cursor, count, total


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
    cache_only: bool = False,
    dataset_config: str | None = None,
    data_path: str | Path | None = None,
    sample_policy: str = "sequential_rows",
    sample_seed: int = 1,
    max_length: int | None = None,
    attention_policy: str = "model_default",
    query_chunk_size: int = 256,
    placement_policy: str = "model_default",
    execution_device: str | torch.device | None = None,
    execution_backend: str = "cuda",
    checkpoint_every_batches: int = 100,
    resume: bool = True,
    stop_after_batches: int | None = None,
    progress_callback=None,
) -> dict[str, torch.Tensor]:
    """Estimate E[key keyᵀ] over valid tokens and persist an identity-bound cache.

    Only ``mom2`` is exposed by this compatibility API. Native sampling consumes
    rows sequentially; ``upstream_random`` uses the authors' seeded subset and
    length collation. Batching respects a padded-token upper bound. Empty inputs
    fail without creating an artificial zero covariance. ``max_length`` can
    impose a method-specific sequence cap independently of the batch budget;
    omitting it preserves the original context limit and cache identity.
    """
    execution = statistics_execution_identity(
        model,
        attention_policy=attention_policy,
        query_chunk_size=query_chunk_size,
        placement_policy=placement_policy,
        execution_backend=execution_backend,
    )
    if (
        isinstance(checkpoint_every_batches, bool)
        or not isinstance(checkpoint_every_batches, int)
        or checkpoint_every_batches <= 0
    ):
        raise ValueError("checkpoint_every_batches must be a positive integer")
    if stop_after_batches is not None and (
        isinstance(stop_after_batches, bool)
        or not isinstance(stop_after_batches, int)
        or stop_after_batches <= 0
    ):
        raise ValueError("stop_after_batches must be a positive integer or None")
    selected = list(to_collect or ["mom2"])
    if selected != ["mom2"]:
        raise ValueError("This layer_stats adapter supports only to_collect=['mom2']")
    if sample_size is not None and (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size <= 0
    ):
        raise ValueError("sample_size must be positive or None")
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
    local_source = statistics_data_identity(data_path)
    split = _load_statistics_split(ds_name, dataset_config, data_path)
    if sample_size is not None and len(split) < sample_size:
        raise ValueError(
            f"Statistics corpus has {len(split)} rows, fewer than requested sample_size={sample_size}; "
            "provide the complete corpus or explicitly review a smaller statistics budget"
        )
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
    if local_source is not None:
        metadata["local_source"] = local_source
    explicit_execution = (
        attention_policy != "model_default" or placement_policy != "model_default"
    )
    if explicit_execution:
        metadata.update(
            schema_version=EXECUTION_CACHE_SCHEMA, statistics_execution=execution
        )
    path = (
        Path(stats_dir)
        / f"v{metadata['schema_version']}"
        / f"{identity_digest(metadata)}.pt"
    )
    partial_path = path.with_suffix(".partial.pt")

    def cached():
        if (
            path.exists()
            and not force_recompute
            and metadata["dataset_fingerprint"] is not None
        ):
            payload = torch.load(path, weights_only=True, map_location="cpu")
            if (
                not isinstance(payload, Mapping)
                or payload.get("metadata") != metadata
                or (explicit_execution and payload.get("status") != "complete")
            ):
                raise ValueError(
                    f"Layer statistics cache identity/status mismatch: {path}"
                )
            result = _validated_stats(payload, source=path)
            if progress_callback is not None:
                progress_callback(
                    {
                        "status": "complete",
                        "cache_hit": True,
                        "phase": "cache_load",
                        "cache_path": str(path),
                        "count": int(result["count"]),
                        "completed_batches": payload.get("completed_batches", 0),
                    }
                )
            return result
        return None

    hit = cached()
    if hit is not None:
        return hit
    if cache_only:
        raise FileNotFoundError(
            f"Required complete layer statistics cache is missing: {path}; prepare it before editing"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another worker owns this exact layer statistics identity: {path}"
            ) from exc
        hit = cached()
        if hit is not None:
            return hit
        sample_order, sampler_rng_state = _sample_state(
            split, sample_size, sample_policy, sample_seed
        )
        dtype = getattr(torch, precision)
        target_module = nethook.get_module(model, layer_name)
        from transformers.pytorch_utils import Conv1D

        dimension = (
            target_module.weight.shape[0]
            if isinstance(target_module, Conv1D)
            else target_module.weight.shape[1]
        )
        total, count, completed = None, 0, 0
        resume_batches, expected_chain = 0, None
        if partial_path.exists() and resume and not force_recompute:
            partial = torch.load(partial_path, weights_only=True, map_location="cpu")
            resume_batches, count, total = _validated_partial(
                partial,
                metadata=metadata,
                sample_order=sample_order,
                sampler_rng_state=sampler_rng_state,
                dimension=dimension,
                dtype=dtype,
                source=partial_path,
            )
            expected_chain = partial["batch_order_sha256"]
        elif partial_path.exists() and not force_recompute:
            raise ValueError(
                f"Existing partial statistics require resume=True or explicit force_recompute: {partial_path}"
            )
        chain = hashlib.sha256()
        replay_count = 0

        def checkpoint():
            if total is None or completed <= 0:
                return
            cpu_sum = total.detach().cpu()
            _atomic_save(
                {
                    "status": "in_progress",
                    "metadata": metadata,
                    "sample_order": sample_order,
                    "sampler_rng_state": sampler_rng_state,
                    "completed_batches": completed,
                    "batch_order_sha256": chain.hexdigest(),
                    "count": count,
                    "sum": cpu_sum,
                    "sum_sha256": _tensor_sha256(cpu_sum),
                },
                partial_path,
            )

        with statistics_execution_context(
            model,
            layer_name,
            attention_policy=attention_policy,
            query_chunk_size=query_chunk_size,
            placement_policy=placement_policy,
            execution_device=execution_device,
            execution_backend=execution_backend,
        ) as input_device:
            batches = _indexed_text_batches(
                split,
                tokenizer,
                indices=sample_order,
                key=text_key,
                max_length=max_length,
                token_budget=token_budget,
                batch_size=batch_size,
                sample_policy=sample_policy,
            )
            try:
                for batch_index, (row_indices, texts) in enumerate(batches, 1):
                    next_chain = chain.copy()
                    next_chain.update(identity_digest(row_indices).encode())
                    if batch_index <= resume_batches:
                        chain = next_chain
                        replay_encoded = tokenizer(
                            texts,
                            padding=True,
                            padding_side="right",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt",
                        )
                        replay_count += int(replay_encoded["attention_mask"].sum())
                        if batch_index == resume_batches:
                            if (
                                chain.hexdigest() != expected_chain
                                or replay_count != count
                            ):
                                raise ValueError(
                                    "Statistics partial checkpoint batch order mismatch"
                                )
                            completed = resume_batches
                            total = total.to(input_device)
                        continue
                    batch_began = time.perf_counter()
                    encoded = tokenizer(
                        texts,
                        padding=True,
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
                        raise RuntimeError(
                            "Tokenizer output has no attention_mask tensor"
                        )
                    with nethook.Trace(
                        model,
                        layer_name,
                        retain_input=True,
                        retain_output=False,
                        stop=True,
                    ) as captured:
                        model(**batch, output_attentions=False)
                    features = _hidden_state(captured.input)
                    if features.ndim != 3 or tuple(features.shape[:2]) != tuple(
                        mask.shape
                    ):
                        raise RuntimeError(
                            f"Layer {layer_name} produced no valid batch-first input activations"
                        )
                    values = features[mask.to(features.device).bool()].to(dtype=dtype)
                    if values.shape[0]:
                        contribution = values.T @ values
                        if not torch.isfinite(contribution).all():
                            raise RuntimeError(
                                "Non-finite statistics contribution; checkpoint remains at last complete batch"
                            )
                        updated = (
                            contribution
                            if total is None
                            else total + contribution.to(total.device)
                        )
                        if not torch.isfinite(updated).all():
                            raise RuntimeError(
                                "Non-finite statistics sum; checkpoint remains at last complete batch"
                            )
                        total = updated
                        count += values.shape[0]
                    chain = next_chain
                    completed = batch_index
                    batch_compute_seconds = time.perf_counter() - batch_began
                    if completed % checkpoint_every_batches == 0:
                        checkpoint()
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "status": "in_progress",
                                "completed_batches": completed,
                                "count": count,
                                "batch_tokens": int(values.shape[0]),
                                "batch_compute_seconds": batch_compute_seconds,
                                "cache_path": str(path),
                                "partial_path": str(partial_path),
                            }
                        )
                    del encoded, batch, mask, captured, features, values
                    if total is not None:
                        contribution = updated = None
                    if (
                        stop_after_batches is not None
                        and completed >= stop_after_batches
                    ):
                        raise StatisticsInterrupted(
                            f"Stopped after {completed} completed batches; resume from {partial_path}"
                        )
                if completed < resume_batches:
                    raise ValueError(
                        "Statistics partial checkpoint cursor exceeds the reconstructed batch sequence"
                    )
            except BaseException:
                # Chain, sum and cursor commit only after a whole valid batch.
                if completed > resume_batches:
                    checkpoint()
                raise
        if total is None or count == 0:
            raise RuntimeError(
                f"No valid token activations collected for layer {layer_name}; statistics cache was not written"
            )
        result = _validated_stats(
            {"mom2": total / count, "count": torch.tensor(count, dtype=torch.int64)},
            source=f"computed layer {layer_name}",
        )
        _atomic_save(
            {
                **result,
                "status": "complete",
                "metadata": metadata,
                "completed_batches": completed,
                "batch_order_sha256": chain.hexdigest(),
                "sample_order_sha256": identity_digest(sample_order),
            },
            path,
        )
        partial_path.unlink(missing_ok=True)
        logger.info(
            "Stored layer input second moment: %s (%d tokens)", layer_name, count
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "status": "complete",
                    "completed_batches": completed,
                    "count": count,
                    "cache_path": str(path),
                }
            )
        return result
