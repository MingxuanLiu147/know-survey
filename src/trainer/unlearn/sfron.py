"""SFR-on saliency-mask generation and validation for MMUnlearner.

The implementation follows MMUnlearner Eq. (4)--(7): approximate the
diagonal Fisher information with squared gradients and keep parameters whose
forget/preserve saliency ratio is at least ``threshold``.  Mask artifacts are
self-describing so a mask cannot silently be reused with a different model,
checkpoint, benchmark, or parameter layout.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

MASK_FORMAT_VERSION = 1
MASK_ARTIFACT_KIND = "mmunlearner_sfron_mask"
FISHER_ARTIFACT_KIND = "mmunlearner_sfron_fisher"
UPSTREAM_COMMIT = "fa77c66b3053fe48480f60711a425d7c45d71400"

# The official trainer drops mask entries outside linear/MLP/attention
# projections before applying the mask. Selecting the same entries while
# computing Fisher avoids retaining unusable tensors in memory.
DEFAULT_PARAMETER_NAME_TOKENS = ("proj", "fc", "linear", "mlp", "qkv")


def _torch_load(path: str | os.PathLike[str]) -> Any:
    """Load tensor-only artifacts safely across supported PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before ``weights_only`` was exposed.
        return torch.load(path, map_location="cpu")


def normalize_checkpoint_id(checkpoint_id: str | os.PathLike[str] | None) -> str | None:
    """Normalize a local checkpoint path while preserving Hugging Face IDs."""

    if checkpoint_id is None:
        return None
    value = str(checkpoint_id)
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value.rstrip("/")


def normalize_model_family(model_family: str | None) -> str | None:
    """Normalize spelling variants such as ``qwen2-vl`` and ``qwen2vl``."""

    if model_family is None:
        return None
    return "".join(char for char in str(model_family).lower() if char.isalnum())


def _has_parameter_prefix(model, prefix: str) -> bool:
    prefix = prefix.rstrip(".")
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for name, _ in model.named_parameters()
    )


def resolve_mm_module_selectors(
    model, model_family: str
) -> tuple[list[str], list[str]]:
    """Resolve official vision/language prefixes for LLaVA and Qwen2-VL."""

    family = normalize_model_family(model_family)
    if family == "llava":
        vision_candidates = [
            ["model.vision_tower", "model.multi_modal_projector"],
            ["vision_tower", "multi_modal_projector"],
        ]
        language_candidates = [["model.language_model"], ["language_model"]]
    elif family in {"qwen2vl", "qwen25vl"}:
        vision_candidates = [["model.visual"], ["visual"]]
        language_candidates = [["model.language_model"], ["language_model"]]
        # Older Qwen2-VL releases exposed ``visual`` separately and used
        # ``model`` exclusively for the language backbone.
        if _has_parameter_prefix(model, "visual") and not _has_parameter_prefix(
            model, "model.visual"
        ):
            language_candidates.append(["model"])
    else:
        raise ValueError(
            f"Unsupported MMUnlearner model family for module mapping: {model_family}"
        )

    def choose(candidates: list[list[str]], scope: str) -> list[str]:
        for candidate in candidates:
            if all(_has_parameter_prefix(model, prefix) for prefix in candidate):
                return candidate
        raise ValueError(
            f"Cannot resolve {scope} parameter prefixes for {model_family}; "
            f"tried {candidates}"
        )

    return choose(vision_candidates, "vision"), choose(language_candidates, "language")


def model_identity(
    model,
    *,
    model_family: str | None = None,
    checkpoint_id: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return inexpensive architecture/checkpoint identity metadata."""

    config = getattr(model, "config", None)
    configured_path = getattr(config, "_name_or_path", None)
    return {
        "model_class": type(model).__name__,
        "model_type": getattr(config, "model_type", None),
        "model_family": normalize_model_family(model_family),
        "checkpoint_id": normalize_checkpoint_id(checkpoint_id or configured_path),
    }


def _matches_module(name: str, selectors: Sequence[str]) -> bool:
    return any(
        name == selector or name.startswith(f"{selector}.") for selector in selectors
    )


def _matches_parameter_kind(name: str, name_tokens: Sequence[str]) -> bool:
    return not name_tokens or any(token in name for token in name_tokens)


def select_mask_parameters(
    model,
    module_selectors: Sequence[str],
    *,
    parameter_name_tokens: Sequence[str] = DEFAULT_PARAMETER_NAME_TOKENS,
) -> dict[str, torch.nn.Parameter]:
    """Select trainable parameters covered by an MMUnlearner module mask."""

    selectors = tuple(str(selector).rstrip(".") for selector in module_selectors)
    tokens = tuple(str(token) for token in parameter_name_tokens)
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and _matches_module(name, selectors)
        and _matches_parameter_kind(name, tokens)
    }
    if not selected:
        sample = [name for name, _ in list(model.named_parameters())[:12]]
        raise ValueError(
            "SFR-on module selectors matched no trainable parameters: "
            f"selectors={list(selectors)}, name_tokens={list(tokens)}, "
            f"sample_model_parameters={sample}"
        )
    return selected


def parameter_shapes(parameters: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    """Serialize a parameter/tensor mapping's exact shape contract."""

    return {name: list(value.shape) for name, value in parameters.items()}


def calc_sparsity(tensor: torch.Tensor) -> tuple[float, int, int]:
    """Return zero ratio, element count, and zero count."""

    total = tensor.numel()
    if total == 0:
        raise ValueError("Cannot compute sparsity of an empty tensor")
    num_zero = total - torch.count_nonzero(tensor).item()
    return num_zero / total, total, num_zero


def _fisher_cache_signature(
    model,
    selected: Mapping[str, torch.Tensor],
    *,
    tag: str,
    modules: Sequence[str],
    parameter_name_tokens: Sequence[str],
    max_batches: int | None,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(identity),
        "tag": tag,
        "module_selectors": list(modules),
        "parameter_name_tokens": list(parameter_name_tokens),
        "max_batches": max_batches,
        "parameter_shapes": parameter_shapes(selected),
        "model_training_class": type(model).__name__,
    }


def _load_matching_fisher_cache(
    cache: Path, expected_signature: Mapping[str, Any]
) -> dict[str, torch.Tensor] | None:
    if not cache.is_file():
        return None
    payload = _torch_load(cache)
    if not isinstance(payload, Mapping):
        logger.warning("Ignoring malformed Fisher cache at %s", cache)
        return None
    if payload.get("kind") != FISHER_ARTIFACT_KIND:
        logger.warning("Ignoring legacy/untyped Fisher cache at %s", cache)
        return None
    if payload.get("metadata") != dict(expected_signature):
        logger.warning("Ignoring incompatible Fisher cache at %s", cache)
        return None
    values = payload.get("values")
    if not isinstance(values, Mapping):
        logger.warning("Ignoring Fisher cache without tensor values at %s", cache)
        return None
    return dict(values)


def compute_fisher(
    model,
    loader: DataLoader,
    modules: Sequence[str],
    save_path: str | os.PathLike[str],
    tag: str = "forget",
    max_batches: int | None = None,
    *,
    parameter_name_tokens: Sequence[str] = DEFAULT_PARAMETER_NAME_TOKENS,
    identity: Mapping[str, Any] | None = None,
    reuse_cache: bool = True,
) -> dict[str, torch.Tensor]:
    """Approximate diagonal Fisher information with mean squared gradients."""

    if len(loader) == 0:
        raise ValueError(f"Cannot compute {tag} Fisher from an empty loader")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")

    selected = select_mask_parameters(
        model,
        modules,
        parameter_name_tokens=parameter_name_tokens,
    )
    identity = dict(identity or model_identity(model))
    signature = _fisher_cache_signature(
        model,
        selected,
        tag=tag,
        modules=modules,
        parameter_name_tokens=parameter_name_tokens,
        max_batches=max_batches,
        identity=identity,
    )
    signature["loader_batches"] = len(loader)
    signature["loader_examples"] = len(loader.dataset)

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{max_batches}b" if max_batches is not None else ""
    cache = output_dir / f"{tag}{suffix}_fisher.pt"
    if reuse_cache:
        cached = _load_matching_fisher_cache(cache, signature)
        if cached is not None:
            logger.info("Loaded compatible %s Fisher cache from %s", tag, cache)
            return cached

    total_batches = min(len(loader), max_batches or len(loader))
    gradients: dict[str, torch.Tensor | None] = dict.fromkeys(selected)
    model.train()
    device = next(model.parameters()).device

    logger.info(
        "Computing %s Fisher over %d batches and %d parameters (%s)",
        tag,
        total_batches,
        len(selected),
        list(modules),
    )
    observed_batches = 0
    for step, batch in enumerate(
        tqdm(loader, total=total_batches, desc=f"{tag} fisher")
    ):
        if step >= total_batches:
            break
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        # The sign is immaterial after squaring and matches the official code.
        loss = -outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {tag} Fisher loss at batch {step}")
        loss.backward()
        observed_batches += 1

        with torch.no_grad():
            for name, parameter in selected.items():
                if parameter.grad is None:
                    continue
                contribution = parameter.grad.detach().cpu().square()
                if gradients[name] is None:
                    gradients[name] = contribution
                else:
                    gradients[name].add_(contribution)

    model.zero_grad(set_to_none=True)
    if observed_batches != total_batches:
        raise RuntimeError(
            f"{tag} Fisher consumed {observed_batches}/{total_batches} expected batches"
        )

    missing_gradients = [name for name, value in gradients.items() if value is None]
    if missing_gradients:
        raise RuntimeError(
            f"{tag} Fisher observed no gradient for {len(missing_gradients)} selected "
            f"parameters; first entries: {missing_gradients[:8]}"
        )

    fisher = {
        name: value.div_(observed_batches)
        for name, value in gradients.items()
        if value is not None
    }
    torch.save(
        {
            "format_version": MASK_FORMAT_VERSION,
            "kind": FISHER_ARTIFACT_KIND,
            "metadata": signature,
            "values": fisher,
        },
        cache,
    )
    logger.info("Saved %s Fisher cache to %s", tag, cache)
    return fisher


def build_mask_artifact(
    mask: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any],
    forget_gradients: Mapping[str, torch.Tensor] | None = None,
    preserve_gradients: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Build the official-compatible three-key payload plus strict metadata."""

    serialized_mask = {
        name: value.to(dtype=torch.bool, device="cpu") for name, value in mask.items()
    }
    artifact: dict[str, Any] = {
        "format_version": MASK_FORMAT_VERSION,
        "kind": MASK_ARTIFACT_KIND,
        "metadata": dict(metadata),
        "weight": serialized_mask,
    }
    if forget_gradients is not None:
        artifact["forget_grad"] = dict(forget_gradients)
    if preserve_gradients is not None:
        artifact["preserve_grad"] = dict(preserve_gradients)
    return artifact


def generate_saliency_mask(
    model,
    forget_loader: DataLoader,
    preserve_loader: DataLoader,
    modules: Sequence[str],
    threshold: float = 1.0,
    save_path: str | os.PathLike[str] = "saves/mm_mask",
    max_batches: int | None = None,
    *,
    parameter_name_tokens: Sequence[str] = DEFAULT_PARAMETER_NAME_TOKENS,
    metadata: Mapping[str, Any] | None = None,
    reuse_cache: bool = True,
    return_artifact: bool = False,
) -> dict[str, torch.Tensor] | dict[str, Any]:
    """Generate and persist one vision/language saliency-mask artifact."""

    if threshold <= 0:
        raise ValueError("MMUnlearner saliency threshold must be positive")

    artifact_metadata = dict(metadata or {})
    identity = model_identity(
        model,
        model_family=artifact_metadata.get("model_family"),
        checkpoint_id=artifact_metadata.get("checkpoint_id"),
    )
    fisher_identity = {
        **identity,
        "benchmark": artifact_metadata.get("benchmark"),
        "forget_ratio": artifact_metadata.get("forget_ratio"),
        "mask_scope": artifact_metadata.get("mask_scope"),
    }
    artifact_metadata.update(identity)
    artifact_metadata.update(
        {
            "benchmark": artifact_metadata.get("benchmark"),
            "mask_scope": artifact_metadata.get("mask_scope"),
            "threshold": float(threshold),
            "module_selectors": list(modules),
            "parameter_name_tokens": list(parameter_name_tokens),
            "max_batches": max_batches,
            "upstream_repository": "https://github.com/Z1zs/MMUnlearner",
            "upstream_commit": UPSTREAM_COMMIT,
        }
    )

    forget_fisher = compute_fisher(
        model,
        forget_loader,
        modules,
        save_path,
        "forget",
        max_batches=max_batches,
        parameter_name_tokens=parameter_name_tokens,
        identity=fisher_identity,
        reuse_cache=reuse_cache,
    )
    preserve_fisher = compute_fisher(
        model,
        preserve_loader,
        modules,
        save_path,
        "preserve",
        max_batches=max_batches,
        parameter_name_tokens=parameter_name_tokens,
        identity=fisher_identity,
        reuse_cache=reuse_cache,
    )

    if forget_fisher.keys() != preserve_fisher.keys():
        missing_preserve = sorted(forget_fisher.keys() - preserve_fisher.keys())
        missing_forget = sorted(preserve_fisher.keys() - forget_fisher.keys())
        raise RuntimeError(
            "Forget/preserve Fisher parameter keys differ: "
            f"missing_preserve={missing_preserve[:8]}, missing_forget={missing_forget[:8]}"
        )

    mask: dict[str, torch.Tensor] = {}
    total_count = 0
    zero_count = 0
    for name, forget_value in forget_fisher.items():
        preserve_value = preserve_fisher[name]
        if forget_value.shape != preserve_value.shape:
            raise ValueError(
                f"Fisher shape mismatch for {name}: "
                f"forget={tuple(forget_value.shape)}, preserve={tuple(preserve_value.shape)}"
            )
        saliency = (forget_value + 1e-15) / (preserve_value + 1e-15)
        weight = saliency >= threshold
        _, total, zeros = calc_sparsity(weight)
        total_count += total
        zero_count += zeros
        mask[name] = weight

    if not mask:
        raise RuntimeError("MMUnlearner generated an empty saliency mask")
    if not any(torch.count_nonzero(value).item() for value in mask.values()):
        raise RuntimeError(
            "MMUnlearner saliency mask selects zero parameter elements; "
            "inspect the data, threshold, and Fisher caches"
        )

    artifact_metadata["parameter_shapes"] = parameter_shapes(mask)
    artifact_metadata["selected_parameter_count"] = len(mask)
    artifact_metadata["selected_element_count"] = total_count - zero_count
    artifact_metadata["total_element_count"] = total_count
    artifact_metadata["sparsity"] = zero_count / total_count
    artifact = build_mask_artifact(
        mask,
        metadata=artifact_metadata,
        forget_gradients=forget_fisher,
        preserve_gradients=preserve_fisher,
    )

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_file = output_dir / "mask.pt"
    torch.save(artifact, mask_file)
    logger.info(
        "Saved %s mask to %s (parameters=%d, sparsity=%.2f%%)",
        artifact_metadata.get("mask_scope") or "saliency",
        mask_file,
        len(mask),
        artifact_metadata["sparsity"] * 100,
    )
    return artifact if return_artifact else mask


def merge_mask_artifacts(
    *artifacts: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge non-overlapping vision/language artifacts into the ``both`` mask."""

    weight: dict[str, torch.Tensor] = {}
    forget_gradients: dict[str, torch.Tensor] = {}
    preserve_gradients: dict[str, torch.Tensor] = {}
    for artifact in artifacts:
        component = artifact.get("weight")
        if not isinstance(component, Mapping):
            raise TypeError("Cannot merge a mask artifact without a weight mapping")
        overlap = weight.keys() & component.keys()
        if overlap:
            raise ValueError(f"Mask components overlap: {sorted(overlap)[:8]}")
        weight.update(component)
        forget_gradients.update(artifact.get("forget_grad", {}))
        preserve_gradients.update(artifact.get("preserve_grad", {}))

    merged_metadata = dict(metadata)
    merged_metadata["parameter_shapes"] = parameter_shapes(weight)
    merged_metadata["selected_parameter_count"] = len(weight)
    total_elements = sum(value.numel() for value in weight.values())
    selected_elements = sum(
        torch.count_nonzero(value).item() for value in weight.values()
    )
    merged_metadata["selected_element_count"] = selected_elements
    merged_metadata["total_element_count"] = total_elements
    merged_metadata["sparsity"] = 1 - selected_elements / total_elements
    return build_mask_artifact(
        weight,
        metadata=merged_metadata,
        forget_gradients=forget_gradients,
        preserve_gradients=preserve_gradients,
    )


def load_and_validate_saliency_mask(
    path: str | os.PathLike[str],
    model,
    *,
    expected_benchmark: str | None = None,
    expected_model_family: str | None = None,
    expected_checkpoint_id: str | os.PathLike[str] | None = None,
    allow_legacy_mask: bool = False,
    min_mask_coverage: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a mask and enforce key, shape, family, checkpoint, and scope coverage."""

    if not 0 < min_mask_coverage <= 1:
        raise ValueError("min_mask_coverage must be in (0, 1]")
    mask_path = Path(path)
    if not mask_path.is_file():
        raise FileNotFoundError(f"MMUnlearner mask not found: {mask_path}")

    payload = _torch_load(mask_path)
    if not isinstance(payload, Mapping):
        raise TypeError("MMUnlearner mask artifact must be a mapping")
    weight = payload.get("weight", payload)
    if not isinstance(weight, Mapping) or not weight:
        raise ValueError("MMUnlearner mask contains no weight tensors")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        if not allow_legacy_mask:
            raise ValueError(
                "Legacy MMUnlearner mask has no compatibility metadata. Regenerate it "
                "with scripts/run_mm_generate_mask.py or set allow_legacy_mask=true "
                "for key/shape-only migration checks."
            )
        metadata = {}
    else:
        metadata = dict(metadata)
        if payload.get("kind") != MASK_ARTIFACT_KIND:
            raise ValueError(
                f"Unexpected MMUnlearner artifact kind: {payload.get('kind')}"
            )
        if payload.get("format_version") != MASK_FORMAT_VERSION:
            raise ValueError(
                "Unsupported MMUnlearner mask format version: "
                f"{payload.get('format_version')}"
            )

    expected_identity = model_identity(
        model,
        model_family=expected_model_family,
        checkpoint_id=expected_checkpoint_id,
    )
    compatibility_fields = {
        "benchmark": (
            str(metadata.get("benchmark")).lower()
            if metadata.get("benchmark")
            else None,
            str(expected_benchmark).lower() if expected_benchmark else None,
        ),
        "model_family": (
            normalize_model_family(metadata.get("model_family")),
            expected_identity["model_family"],
        ),
        "checkpoint_id": (
            normalize_checkpoint_id(metadata.get("checkpoint_id")),
            expected_identity["checkpoint_id"],
        ),
        "model_class": (metadata.get("model_class"), expected_identity["model_class"]),
        "model_type": (metadata.get("model_type"), expected_identity["model_type"]),
    }
    if metadata:
        mismatches = {
            field: {"mask": actual, "runtime": expected}
            for field, (actual, expected) in compatibility_fields.items()
            if expected is not None and actual != expected
        }
        if mismatches:
            raise ValueError(
                f"MMUnlearner mask/model compatibility mismatch: {mismatches}"
            )

    model_parameters = dict(model.named_parameters())
    validated: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    shape_mismatches: list[str] = []
    invalid_tensors: list[str] = []
    for name, value in weight.items():
        if name not in model_parameters:
            unexpected_keys.append(str(name))
            continue
        if not torch.is_tensor(value) or value.dtype != torch.bool:
            invalid_tensors.append(str(name))
            continue
        if value.shape != model_parameters[name].shape:
            shape_mismatches.append(
                f"{name}: mask={tuple(value.shape)}, model={tuple(model_parameters[name].shape)}"
            )
            continue
        validated[str(name)] = value.cpu()

    if unexpected_keys or shape_mismatches or invalid_tensors:
        raise ValueError(
            "Invalid MMUnlearner mask tensors: "
            f"unexpected_keys={unexpected_keys[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}, "
            f"non_boolean={invalid_tensors[:8]}"
        )

    expected_keys: set[str]
    selectors = metadata.get("module_selectors") if metadata else None
    name_tokens = metadata.get("parameter_name_tokens") if metadata else None
    if selectors:
        expected_keys = set(
            select_mask_parameters(
                model,
                selectors,
                parameter_name_tokens=name_tokens or DEFAULT_PARAMETER_NAME_TOKENS,
            )
        )
    else:
        expected_keys = set(validated)

    covered_keys = expected_keys & validated.keys()
    coverage = len(covered_keys) / len(expected_keys) if expected_keys else 0.0
    missing_keys = sorted(expected_keys - validated.keys())
    keys_outside_scope = sorted(validated.keys() - expected_keys)
    if coverage < min_mask_coverage or keys_outside_scope:
        raise ValueError(
            "MMUnlearner mask scope coverage failed: "
            f"coverage={coverage:.6f}, required={min_mask_coverage:.6f}, "
            f"missing={missing_keys[:8]}, outside_scope={keys_outside_scope[:8]}"
        )
    if not any(torch.count_nonzero(value).item() for value in validated.values()):
        raise ValueError("MMUnlearner mask has no active parameter elements")

    report = {
        "path": str(mask_path.resolve()),
        "mask_scope": metadata.get("mask_scope") if metadata else "legacy",
        "benchmark": metadata.get("benchmark") if metadata else expected_benchmark,
        "model_family": metadata.get("model_family")
        if metadata
        else expected_model_family,
        "checkpoint_id": metadata.get("checkpoint_id") if metadata else None,
        "parameter_count": len(validated),
        "expected_parameter_count": len(expected_keys),
        "coverage": coverage,
        "missing_parameter_count": len(missing_keys),
        "legacy": not bool(metadata),
    }
    return validated, report
