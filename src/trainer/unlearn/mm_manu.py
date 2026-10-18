"""Materialized MANU neuron editing for multimodal unlearning.

This module reimplements the ``MANUPruner`` baseline distributed with the
MIP-Editor repository.  MANU contrasts the input activations of language-model
MLP ``down_proj`` modules for the same examples with and without images,
computes separate importance scores on forget and retain data, then attenuates
neurons whose forget/retain importance ratio is largest.

The upstream implementation installs forward pre-hooks.  Those hooks are
process-local and disappear when a checkpoint is saved.  Here the same linear
operation is materialized into the selected ``down_proj`` weight columns:

``W @ (s * x) == (W with column j multiplied by s) @ x``.

Only language-model MLPs are considered; vision-tower MLPs are excluded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


SOURCE_ADJUSTMENTS = (
    "paired batches are independently tokenized for text and image-text forwards",
    "benchmark adapters use entity-anchored text_question views when available",
    "population variance (unbiased=False) keeps one-sample smoke scores finite",
    "right-padding-aware last-token selection replaces outputs[:, -1, :] on padding",
    "absolute-sum mean denominator avoids signed activation cancellation",
    "selected input scaling is materialized into down_proj weight columns",
    "positive alpha selects at least one neuron in every requested selection scope",
)

_VISION_PATH_PARTS = {
    "vision_model",
    "vision_tower",
    "visual",
    "multi_modal_projector",
    "mm_projector",
}

MANU_SOURCE_REPOSITORY = "https://github.com/PreckLi/MIP-Editor"
MANU_SOURCE_COMMIT = "db8e6d2350c5e188f58a6bd38d92d2cebcb037cd"


@dataclass
class _ActivationMoments:
    """Streaming sufficient statistics for one layer and one modality."""

    count: int = 0
    total: torch.Tensor | None = None
    squared_total: torch.Tensor | None = None
    above_tau: torch.Tensor | None = None

    def update(self, values: torch.Tensor, tau: float) -> None:
        if values.ndim != 2:
            raise ValueError(
                "MANU activation observations must be [samples, neurons]; "
                f"received shape={tuple(values.shape)}"
            )
        values = values.detach().to(device="cpu", dtype=torch.float64)
        if values.numel() == 0:
            return
        if not torch.isfinite(values).all():
            raise ValueError("MANU encountered non-finite down_proj activations")
        batch_total = values.sum(dim=0)
        batch_squared_total = values.square().sum(dim=0)
        batch_above_tau = (values > tau).sum(dim=0, dtype=torch.float64)
        if self.total is None:
            self.total = batch_total
            self.squared_total = batch_squared_total
            self.above_tau = batch_above_tau
        else:
            if self.total.shape != batch_total.shape:
                raise ValueError("MANU observed a changing neuron width for one layer")
            self.total += batch_total
            self.squared_total += batch_squared_total
            self.above_tau += batch_above_tau
        self.count += int(values.shape[0])

    def finalized(self) -> dict[str, torch.Tensor | int]:
        if (
            self.count <= 0
            or self.total is None
            or self.squared_total is None
            or self.above_tau is None
        ):
            raise ValueError("MANU cannot finalize an empty activation stream")
        mean = self.total / self.count
        # Population variance is the appropriate streaming equivalent of
        # torch.var(..., unbiased=False), and unlike the upstream default it is
        # defined for a one-record smoke test.
        variance = (self.squared_total / self.count - mean.square()).clamp_min(0)
        return {
            "count": self.count,
            "mean": mean,
            "variance": variance,
            "squared_total": self.squared_total,
            "above_tau": self.above_tau,
        }


def discover_language_down_projections(model) -> dict[str, torch.nn.Module]:
    """Return language MLP ``down_proj`` modules in deterministic name order."""

    modules: dict[str, torch.nn.Module] = {}
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) < 3 or parts[-2:] != ["mlp", "down_proj"]:
            continue
        if "layers" not in parts or _VISION_PATH_PARTS.intersection(parts):
            continue
        weight = getattr(module, "weight", None)
        if not torch.is_tensor(weight) or weight.ndim != 2:
            continue
        modules[name] = module
    if not modules:
        raise ValueError(
            "MANU found no language-model layers.*.mlp.down_proj modules; "
            "check the model architecture before registering it as compatible"
        )
    return dict(sorted(modules.items()))


def _last_token_observations(
    inputs: tuple[Any, ...], attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    if not inputs or not torch.is_tensor(inputs[0]):
        raise TypeError("MANU down_proj pre-hook expected a tensor as its first input")
    activations = inputs[0]
    if activations.ndim == 3:
        if attention_mask is None:
            return activations[:, -1, :]
        if attention_mask.ndim != 2 or attention_mask.shape[0] != activations.shape[0]:
            raise ValueError(
                "MMMANU attention mask is incompatible with down_proj activations"
            )
        # Preserve the released method's last-token statistic while avoiding
        # right-padding activations.  ``padding_width`` remains valid when
        # LLaVA expands image placeholders because expansion happens before the
        # right padding, so subtracting it from the actual activation length
        # still locates the last non-padding position.
        padding_width = attention_mask.shape[1] - attention_mask.long().sum(dim=1)
        indices = activations.shape[1] - padding_width.to(activations.device) - 1
        if bool((indices < 0).any() or (indices >= activations.shape[1]).any()):
            raise ValueError("MMMANU could not locate valid last-token activations")
        rows = torch.arange(activations.shape[0], device=activations.device)
        return activations[rows, indices]
    if activations.ndim == 2:
        # Some fused/model-parallel implementations flatten batch and sequence.
        # Treat the final row as one observation rather than silently changing
        # the score to an all-token statistic.
        return activations[-1:, :]
    raise ValueError(
        "MANU expects down_proj inputs shaped [batch, sequence, neurons] or "
        f"[tokens, neurons], received {tuple(activations.shape)}"
    )


def _move_tensor_batch(batch: Mapping[str, Any], device: torch.device | None) -> dict:
    if device is None:
        return dict(batch)
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _sha256_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    # NumPy cannot represent torch.bfloat16, but a uint8 view preserves the
    # exact serialized bytes for all floating dtypes used by the model matrix.
    raw_bytes = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def _validate_plain_projection(module: torch.nn.Module, name: str) -> torch.Tensor:
    """Return a mutable weight tensor when column scaling is exactly valid."""

    # The reference MANU path calls merge_and_unload() first.  Scaling only a
    # PEFT base weight would not scale adapter branches and therefore would not
    # be equivalent to the released forward pre-hook.
    if hasattr(module, "base_layer") or hasattr(module, "lora_A"):
        raise TypeError(
            f"MANU target {name} is adapter-wrapped; merge adapters before "
            "materializing neuron scaling"
        )
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError(f"MANU target {name} does not expose a 2-D weight tensor")
    if not weight.is_floating_point():
        raise TypeError(
            f"MANU target {name} uses non-floating/quantized weights; materialized "
            "column scaling is unsupported"
        )
    return weight


@torch.no_grad()
def materialize_selected_neurons(
    model,
    selected: Mapping[str, torch.Tensor | list[int]],
    *,
    decay: float,
) -> list[dict[str, Any]]:
    """Materialize MANU's input pre-hook scaling into projection columns.

    Args:
        model: Multimodal model containing language ``down_proj`` modules.
        selected: Mapping from module path to selected input-neuron indices.
        decay: Upstream MANU decay; selected coordinates are scaled by
            ``1 - decay``.
    """

    if not 0 <= decay <= 1:
        raise ValueError(f"MANU decay must lie in [0, 1], got {decay}")
    scale = 1.0 - float(decay)
    records: list[dict[str, Any]] = []
    available = discover_language_down_projections(model)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise KeyError(f"MANU selection references unknown modules: {unknown}")

    for name in sorted(selected):
        module = available[name]
        weight = _validate_plain_projection(module, name)
        indices = torch.as_tensor(selected[name], dtype=torch.long).flatten().cpu()
        if indices.numel() == 0:
            continue
        indices = torch.unique(indices, sorted=True)
        if int(indices.min()) < 0 or int(indices.max()) >= weight.shape[1]:
            raise IndexError(
                f"MANU selected an invalid neuron for {name}: input width={weight.shape[1]}"
            )
        device_indices = indices.to(weight.device)
        before = weight.index_select(1, device_indices).detach().cpu()
        weight[:, device_indices] *= scale
        after = weight.index_select(1, device_indices).detach().cpu()
        records.append(
            {
                "module": name,
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype).removeprefix("torch."),
                "selected_count": int(indices.numel()),
                "selected_indices": indices.tolist(),
                "scale": scale,
                "selected_columns_sha256_before": _sha256_tensor(before),
                "selected_columns_sha256_after": _sha256_tensor(after),
            }
        )
    return records


class MMMANU:
    """One-shot MANU trainer compatible with :mod:`src.mm_train`.

    ``forget_loader`` and ``retain_loader`` must yield paired modality batches
    with ``{"text": model_batch, "multimodal": model_batch}``.  The opt-in
    collate path in :mod:`data.multimodal` produces this contract.
    """

    SELECTION_SCOPES = ("global", "per_layer")
    DENOMINATOR_MODES = ("absolute_sum", "source_signed_sum")

    def __init__(
        self,
        model,
        processor,
        forget_loader,
        retain_loader=None,
        *,
        lr: float = 5e-4,
        num_epochs: int = 1,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        save_dir: str = "./saves/mm_unlearn",
        device: str | torch.device | None = None,
        alpha: float = 0.1,
        decay: float = 0.1,
        tau: float = 0.1,
        epsilon: float = 1e-6,
        selection_scope: str = "global",
        denominator_mode: str = "absolute_sum",
        materialize: bool = True,
        benchmark: str | None = None,
        checkpoint_id: str | None = None,
    ):
        if retain_loader is None:
            raise ValueError("MMMANU requires a retain loader")
        if not 0 < alpha <= 1:
            raise ValueError(f"MMMANU alpha must lie in (0, 1], got {alpha}")
        if not 0 <= decay <= 1:
            raise ValueError(f"MMMANU decay must lie in [0, 1], got {decay}")
        if epsilon <= 0:
            raise ValueError("MMMANU epsilon must be positive")
        selection_scope = str(selection_scope).lower()
        if selection_scope not in self.SELECTION_SCOPES:
            raise ValueError(
                f"MMMANU selection_scope must be one of {self.SELECTION_SCOPES}"
            )
        denominator_mode = str(denominator_mode).lower()
        if denominator_mode not in self.DENOMINATOR_MODES:
            raise ValueError(
                f"MMMANU denominator_mode must be one of {self.DENOMINATOR_MODES}"
            )
        if int(num_epochs) != 1:
            raise ValueError(
                "MMMANU is a one-shot neuron editor and requires num_epochs=1"
            )
        if int(gradient_accumulation_steps) != 1:
            raise ValueError(
                "MMMANU does not optimize gradients; gradient_accumulation_steps "
                "must remain 1"
            )

        self.model = model
        self.processor = processor
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.save_dir = str(save_dir)
        self.device = self._resolve_device(device)
        self.alpha = float(alpha)
        self.decay = float(decay)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.selection_scope = selection_scope
        self.denominator_mode = denominator_mode
        self.materialize = bool(materialize)
        if self.materialize:
            for module_name, module in discover_language_down_projections(
                model
            ).items():
                _validate_plain_projection(module, module_name)
        self.benchmark = benchmark or self._infer_benchmark(forget_loader)
        model_config = getattr(model, "config", None)
        configured_checkpoint = getattr(model_config, "_name_or_path", None)
        self.checkpoint_id = str(
            checkpoint_id or configured_checkpoint or type(model).__name__
        )
        self.runtime_args = {
            "lr": float(lr),
            "num_epochs": int(num_epochs),
            "max_grad_norm": float(max_grad_norm),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "warmup_steps": int(warmup_steps),
        }
        self._has_run = False
        self._last_manifest: dict[str, Any] | None = None

    @staticmethod
    def _infer_benchmark(loader) -> str | None:
        dataset = getattr(loader, "dataset", None)
        while dataset is not None and hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        dataset_name = type(dataset).__name__.lower() if dataset is not None else ""
        if "mllmu" in dataset_name:
            return "mllmu"
        if "clear" in dataset_name:
            return "clear"
        return None

    def _resolve_device(
        self, requested: str | torch.device | None
    ) -> torch.device | None:
        if requested is not None:
            return torch.device(requested)
        try:
            parameter = next(self.model.parameters())
        except StopIteration:
            return None
        return None if parameter.device.type == "meta" else parameter.device

    @staticmethod
    def _paired_branches(batch: Any) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "MMMANU requires paired modality batches; set data.paired_modalities=true"
            )
        text_batch = batch.get("text")
        multimodal_batch = batch.get("multimodal", batch.get("multi"))
        if not isinstance(text_batch, Mapping) or not isinstance(
            multimodal_batch, Mapping
        ):
            raise TypeError(
                "MMMANU batch must contain mapping-valued 'text' and 'multimodal' keys"
            )
        text_ids = text_batch.get("input_ids")
        multimodal_ids = multimodal_batch.get("input_ids")
        if torch.is_tensor(text_ids) and torch.is_tensor(multimodal_ids):
            if text_ids.shape[0] != multimodal_ids.shape[0]:
                raise ValueError("MMMANU paired branches have different batch sizes")
        return text_batch, multimodal_batch

    def _importance_from_moments(
        self,
        moments: Mapping[str, Mapping[str, _ActivationMoments]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
        scores: dict[str, torch.Tensor] = {}
        counts: dict[str, dict[str, int]] = {}
        text_layers = set(moments["text"])
        multimodal_layers = set(moments["multimodal"])
        if text_layers != multimodal_layers:
            raise RuntimeError(
                "MMMANU did not observe identical layers for text and multimodal forwards"
            )
        for layer in sorted(text_layers):
            text = moments["text"][layer].finalized()
            multimodal = moments["multimodal"][layer].finalized()
            if text["count"] != multimodal["count"]:
                raise RuntimeError(
                    f"MMMANU layer {layer} has unpaired observations: "
                    f"text={text['count']} multimodal={multimodal['count']}"
                )
            text_mean = text["mean"]
            multimodal_mean = multimodal["mean"]
            if self.denominator_mode == "absolute_sum":
                mean_denominator = (
                    multimodal_mean.abs() + text_mean.abs() + self.epsilon
                )
            else:
                mean_denominator = multimodal_mean + text_mean + self.epsilon
            mean_component = (multimodal_mean - text_mean).abs() / mean_denominator

            frequency_component = (
                multimodal["above_tau"] - text["above_tau"]
            ).abs() / (multimodal["above_tau"] + text["above_tau"] + self.epsilon)
            variance_component = (multimodal["variance"] + text["variance"]).sqrt()
            squared_difference = multimodal["squared_total"] - text["squared_total"]
            squared_sum = multimodal["squared_total"] + text["squared_total"]
            rms_component = (
                squared_difference.abs() / (squared_sum + self.epsilon)
            ).sqrt()
            score = (
                mean_component
                + frequency_component
                + variance_component
                + rms_component
            )
            if not torch.isfinite(score).all():
                raise ValueError(f"MMMANU produced non-finite importance for {layer}")
            scores[layer] = score.to(dtype=torch.float32, device="cpu")
            counts[layer] = {
                "text": int(text["count"]),
                "multimodal": int(multimodal["count"]),
            }
        return scores, counts

    def compute_importance(
        self, data_loader, *, description: str = "MANU importance"
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, int]]]:
        """Compute released MANU's four-component modality importance score."""

        if len(data_loader) == 0:
            raise ValueError(f"{description} loader is empty")
        modules = discover_language_down_projections(self.model)
        moments: dict[str, dict[str, _ActivationMoments]] = {
            "text": {name: _ActivationMoments() for name in modules},
            "multimodal": {name: _ActivationMoments() for name in modules},
        }
        capture_modality: list[str | None] = [None]
        capture_attention_mask: list[torch.Tensor | None] = [None]
        handles = []

        def make_hook(layer_name: str):
            def capture(_module, inputs):
                modality = capture_modality[0]
                if modality is None:
                    raise RuntimeError("MMMANU activation hook fired outside a forward")
                observations = _last_token_observations(
                    inputs, capture_attention_mask[0]
                )
                moments[modality][layer_name].update(observations, self.tau)

            return capture

        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(make_hook(name)))

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for paired_batch in tqdm(data_loader, desc=description):
                    text_batch, multimodal_batch = self._paired_branches(paired_batch)
                    capture_modality[0] = "text"
                    moved_text_batch = _move_tensor_batch(text_batch, self.device)
                    capture_attention_mask[0] = moved_text_batch.get("attention_mask")
                    self.model(**moved_text_batch)
                    capture_modality[0] = "multimodal"
                    moved_multimodal_batch = _move_tensor_batch(
                        multimodal_batch, self.device
                    )
                    capture_attention_mask[0] = moved_multimodal_batch.get(
                        "attention_mask"
                    )
                    self.model(**moved_multimodal_batch)
                    capture_modality[0] = None
                    capture_attention_mask[0] = None
        finally:
            capture_modality[0] = None
            capture_attention_mask[0] = None
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

        return self._importance_from_moments(moments)

    def select_neurons(
        self,
        forget_scores: Mapping[str, torch.Tensor],
        retain_scores: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Rank forget/retain ratios and select globally or per language layer."""

        if set(forget_scores) != set(retain_scores):
            raise ValueError("MMMANU forget and retain score layers do not match")
        ratios: dict[str, torch.Tensor] = {}
        for layer in sorted(forget_scores):
            forget = forget_scores[layer].to(dtype=torch.float64, device="cpu")
            retain = retain_scores[layer].to(dtype=torch.float64, device="cpu")
            if forget.shape != retain.shape or forget.ndim != 1:
                raise ValueError(f"MMMANU score shape mismatch for {layer}")
            ratio = forget / (retain + self.epsilon)
            if not torch.isfinite(ratio).all():
                raise ValueError(f"MMMANU produced a non-finite ratio for {layer}")
            ratios[layer] = ratio.to(torch.float32)

        selected = {layer: torch.empty(0, dtype=torch.long) for layer in sorted(ratios)}
        if self.selection_scope == "per_layer":
            for layer, ratio in ratios.items():
                k = max(1, int(math.floor(ratio.numel() * self.alpha)))
                ranking = torch.argsort(ratio, descending=True, stable=True)
                selected[layer] = ranking[:k].sort().values
            return selected, ratios

        offsets: list[tuple[str, int, int]] = []
        flattened = []
        start = 0
        for layer, ratio in ratios.items():
            end = start + ratio.numel()
            offsets.append((layer, start, end))
            flattened.append(ratio)
            start = end
        if not flattened:
            raise ValueError("MMMANU has no neuron scores to select")
        all_ratios = torch.cat(flattened)
        k = max(1, int(math.floor(all_ratios.numel() * self.alpha)))
        global_indices = torch.argsort(all_ratios, descending=True, stable=True)[:k]
        for layer, layer_start, layer_end in offsets:
            local = (
                global_indices[
                    (global_indices >= layer_start) & (global_indices < layer_end)
                ]
                - layer_start
            )
            selected[layer] = local.sort().values
        return selected, ratios

    def _write_artifacts(
        self,
        *,
        forget_scores: Mapping[str, torch.Tensor],
        retain_scores: Mapping[str, torch.Tensor],
        ratios: Mapping[str, torch.Tensor],
        selected: Mapping[str, torch.Tensor],
        forget_counts: Mapping[str, Mapping[str, int]],
        retain_counts: Mapping[str, Mapping[str, int]],
        materialization: list[dict[str, Any]],
    ) -> dict[str, Any]:
        output_dir = Path(self.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        score_path = output_dir / "manu_importance.pt"
        torch.save(
            {
                "forget": dict(forget_scores),
                "retain": dict(retain_scores),
                "forget_retain_ratio": dict(ratios),
                "selected": {key: value.cpu() for key, value in selected.items()},
            },
            score_path,
        )
        selected_count = sum(int(value.numel()) for value in selected.values())
        total_neurons = sum(int(value.numel()) for value in ratios.values())
        touched_modules = sum(int(value.numel() > 0) for value in selected.values())
        manifest = {
            "schema_version": 1,
            "algorithm": "MMMANU",
            "source_method": "MANU baseline from the MIP-Editor repository",
            "source_repository": MANU_SOURCE_REPOSITORY,
            "source_commit": MANU_SOURCE_COMMIT,
            "benchmark": self.benchmark,
            "model_class": type(self.model).__name__,
            "checkpoint_id": self.checkpoint_id,
            "status": "materialized" if self.materialize else "selection_only",
            "selection_scope": self.selection_scope,
            "denominator_mode": self.denominator_mode,
            "alpha": self.alpha,
            "decay": self.decay,
            "scale": 1.0 - self.decay,
            "tau": self.tau,
            "epsilon": self.epsilon,
            "materialize": self.materialize,
            "module_count": len(ratios),
            "touched_module_count": touched_modules,
            "total_neuron_count": total_neurons,
            "selected_neuron_count": selected_count,
            "forget_observation_counts": dict(forget_counts),
            "retain_observation_counts": dict(retain_counts),
            "selected_by_module": {
                layer: value.tolist() for layer, value in selected.items()
            },
            "materialization": materialization,
            "importance_artifact": score_path.name,
            "importance_artifact_sha256": hashlib.sha256(
                score_path.read_bytes()
            ).hexdigest(),
            "source_adjustments": list(SOURCE_ADJUSTMENTS),
            "standard_trainer_args": self.runtime_args,
        }
        manifest_path = output_dir / "manu_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._last_manifest = manifest
        return manifest

    def train(self) -> dict[str, Any]:
        """Compute forget/retain importance and perform one persistent edit."""

        if self._has_run:
            raise RuntimeError("MMMANU.train() is one-shot and has already run")
        forget_scores, forget_counts = self.compute_importance(
            self.forget_loader, description="MMMANU forget importance"
        )
        retain_scores, retain_counts = self.compute_importance(
            self.retain_loader, description="MMMANU retain importance"
        )
        selected, ratios = self.select_neurons(forget_scores, retain_scores)
        materialization = []
        if self.materialize:
            materialization = materialize_selected_neurons(
                self.model, selected, decay=self.decay
            )
        manifest = self._write_artifacts(
            forget_scores=forget_scores,
            retain_scores=retain_scores,
            ratios=ratios,
            selected=selected,
            forget_counts=forget_counts,
            retain_counts=retain_counts,
            materialization=materialization,
        )
        self._has_run = True
        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        return {
            "epochs": 1,
            "optimizer_steps": 0,
            "module_count": manifest["module_count"],
            "touched_module_count": manifest["touched_module_count"],
            "total_neuron_count": manifest["total_neuron_count"],
            "selected_neuron_count": manifest["selected_neuron_count"],
            "selection_scope": self.selection_scope,
            "benchmark": self.benchmark,
            "checkpoint_id": self.checkpoint_id,
            "materialized": self.materialize,
            "manifest_path": os.path.join(self.save_dir, "manu_manifest.json"),
            "importance_path": os.path.join(self.save_dir, "manu_importance.pt"),
            "source_adjustments": list(SOURCE_ADJUSTMENTS),
            "peak_memory_mb": peak_memory_mb,
        }

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the already-materialized model through the standard runtime API."""

        if not self._has_run:
            raise RuntimeError("Run MMMANU.train() before saving the model")
        save_path = Path(output_dir or self.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        save_pretrained = getattr(self.model, "save_pretrained", None)
        if save_pretrained is None:
            torch.save(self.model.state_dict(), save_path / "pytorch_model.bin")
        else:
            save_pretrained(save_path)
        processor_save = getattr(self.processor, "save_pretrained", None)
        if processor_save is not None:
            processor_save(save_path)
        logger.info("Saved MMMANU materialized model to %s", save_path)
