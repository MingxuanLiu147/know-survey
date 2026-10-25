"""
MEMIT knowledge editor
======================

Mass-Editing Memory in a Transformer via shared ``compute_z`` / ``compute_ks``
and multi-layer least-squares updates.

Paper: Mass-Editing Memory in a Transformer
https://arxiv.org/abs/2210.07229

The native profile retains the repository objective and update scale. The
explicit ``official_reference`` profile uses the authors' intermediate readout,
grouped keys and unscaled FP64 covariance solve. This is not a vendored,
line-by-line EasyEdit fork or a claim of benchmark reproduction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.utils.compute_z import (
    compute_z,
    get_layer_outputs_at_words,
    get_module_input_output_at_words,
)
from trainer.edit.utils.layer_stats import layer_stats
from trainer.edit.utils.nethook import get_parameter
from trainer.edit.utils.reference import (
    REFERENCE_VARIANT,
    ensure_reference_contexts,
    reference_stats_kwargs,
    validate_reference_profile,
)

logger = logging.getLogger(__name__)


def _match_weight_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Update matrix shape {matrix.shape} does not match weight shape {shape}"
    )


class MEMITEditor(EditTrainer):
    """MEMIT editor: multi-layer closed-form updates driven by ``compute_z``."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        v_lr: float = 0.5,
        v_num_grad_steps: int = 20,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 4.0,
        v_loss_layer: int = 31,
        v_loss_mode: str = "final_logits",
        kl_direction: str = "base_to_edited",
        context_templates: Optional[List[str]] = None,
        implementation_variant: str = "native_adapted",
        kl_factor: float = 0.0625,
        edit_weight: float = 0.5,
        mom2_update_weight: float = 4000.0,
        fact_token: str = "subject_last",
        mom2_adjustment: bool = True,
        mom2_dataset: str = "wikipedia",
        mom2_n_samples: int = 100_000,
        mom2_dtype: str = "float32",
        stats_dir: str = "data/stats",
        covariance_ridge: float = 1e-5,
        rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj",
        layer_module_tmp: str = "model.layers.{}",
        lm_head_module: str = "lm_head",
        ln_f_module: str = "model.norm",
        mom2_dataset_config: str | None = None,
        mom2_sample_seed: int = 1,
        mom2_batch_tokens: int | None = None,
        *args,
        **kwargs,
    ):
        layers = [4, 5, 6, 7, 8] if layers is None else layers
        super().__init__(layers=layers, *args, **kwargs)

        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_weight_decay = v_weight_decay
        self.clamp_norm_factor = clamp_norm_factor
        self.v_loss_layer = v_loss_layer
        self.implementation_variant = implementation_variant
        self.v_loss_mode = v_loss_mode
        self.kl_direction = kl_direction
        self.context_templates = (
            context_templates
            if implementation_variant == REFERENCE_VARIANT
            else list(context_templates or ["{}"])
        )
        self.kl_factor = kl_factor
        self.edit_weight = edit_weight
        self.mom2_update_weight = mom2_update_weight
        self.fact_token = fact_token
        self.mom2_adjustment = mom2_adjustment
        self.mom2_dataset = mom2_dataset
        self.mom2_n_samples = mom2_n_samples
        self.mom2_dtype = mom2_dtype
        self.stats_dir = stats_dir
        self.covariance_ridge = covariance_ridge
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.lm_head_module = lm_head_module
        self.ln_f_module = ln_f_module
        self.mom2_dataset_config = mom2_dataset_config or (
            "20200501.en"
            if implementation_variant == REFERENCE_VARIANT
            and mom2_dataset == "wikipedia"
            else None
        )
        self.mom2_sample_seed = mom2_sample_seed
        self.mom2_batch_tokens = mom2_batch_tokens
        validate_reference_profile(self, "MEMIT")
        self._cov_cache: Dict[str, torch.Tensor] = {}

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "layers_edited": [],
        }

        transaction_state: Optional[Dict[str, Any]] = None
        try:
            ensure_reference_contexts(self, "MEMIT")
            layer_indices = self._resolve_layer_indices(self.layers)
            results["layers_edited"] = layer_indices
            transaction_state = self._snapshot_transaction_state(layer_indices)
            self._apply_memit_edit(
                requests,
                layer_indices,
                weights_copy=transaction_state["weights"],
            )
            results["edited_count"] = len(requests)
        except Exception as e:
            if transaction_state is not None:
                self._restore_transaction_state(transaction_state)
            logger.error("MEMIT edit failed: %s", e, exc_info=True)
            results["success"] = False
            results["edited_count"] = 0
            results["error"] = str(e)

        return results

    def _snapshot_transaction_state(self, layer_indices: List[int]) -> Dict[str, Any]:
        """Capture the state mutated by one public MEMIT edit call."""

        weights = {
            f"{self.rewrite_module_tmp.format(layer)}.weight": get_parameter(
                self.model, f"{self.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in layer_indices
        }
        return {
            "weights": {
                name: weight.detach().clone() for name, weight in weights.items()
            },
            # Covariance entries are immutable after insertion, so a shallow
            # mapping snapshot avoids duplicating large statistics tensors.
            "cov_cache": dict(self._cov_cache),
            "edit_history": list(self.edit_history),
        }

    def _restore_transaction_state(self, state: Dict[str, Any]) -> None:
        """Restore model weights and editor state after a failed MEMIT call."""

        with torch.no_grad():
            for name, value in state["weights"].items():
                parameter = get_parameter(self.model, name)
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )
        self._cov_cache = state["cov_cache"]
        self.edit_history = state["edit_history"]

    def _apply_memit_edit(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
        *,
        weights_copy: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        model = self.model
        tok = self.tokenizer
        weights = {
            f"{self.rewrite_module_tmp.format(layer)}.weight": get_parameter(
                model, f"{self.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in layer_indices
        }
        if weights_copy is None:
            weights_copy = {k: v.detach().clone() for k, v in weights.items()}

        if self.mom2_adjustment:
            # All C0 matrices must describe the same unmodified checkpoint.
            for layer_idx in layer_indices:
                self._get_covariance(self.rewrite_module_tmp.format(layer_idx))

        z_layer = layer_indices[-1]
        req_dicts = [
            {
                "prompt": request.prompt_template or request.prompt,
                "subject": request.subject,
                "target_new": request.target_new,
            }
            for request in requests
        ]

        z_list = []
        for rd in req_dicts:
            z = compute_z(
                model,
                tok,
                rd,
                z_layer,
                layer_module_tmp=self.layer_module_tmp,
                lm_head_module=self.lm_head_module,
                ln_f_module=self.ln_f_module,
                v_lr=self.v_lr,
                v_num_grad_steps=self.v_num_grad_steps,
                v_weight_decay=self.v_weight_decay,
                v_loss_layer=self.v_loss_layer,
                v_loss_mode=self.v_loss_mode,
                kl_direction=self.kl_direction,
                context_templates=self.context_templates,
                clamp_norm_factor=self.clamp_norm_factor,
                kl_factor=self.kl_factor,
                fact_token=self.fact_token,
                optimization_convention=(
                    "upstream"
                    if self.implementation_variant == REFERENCE_VARIANT
                    else "native"
                ),
            )
            z_list.append(z)
        zs = torch.stack(z_list, dim=0)

        prompts = [r.prompt_template or r.prompt for r in requests]
        subjects = [r.subject for r in requests]

        for i, layer_idx in enumerate(layer_indices):
            rewrite_name = self.rewrite_module_tmp.format(layer_idx)

            layer_keys, _ = get_module_input_output_at_words(
                model,
                tok,
                prompts,
                subjects,
                rewrite_name,
                context_templates=self.context_templates,
                fact_token=self.fact_token,
            )
            cur_zs = get_layer_outputs_at_words(
                model,
                tok,
                prompts,
                subjects,
                z_layer,
                layer_module_tmp=self.layer_module_tmp,
                fact_token=self.fact_token,
            )
            if zs.device != cur_zs.device:
                cur_zs = cur_zs.to(zs.device)
            targets = zs - cur_zs

            dtype = (
                torch.float64
                if self.implementation_variant == REFERENCE_VARIANT
                else torch.float32
            )
            layer_ks = layer_keys.T.to(dtype)  # (input_dim, batch)
            resid = targets.to(device=layer_ks.device, dtype=dtype) / max(
                len(layer_indices) - i, 1
            )
            adjusted_keys = self._solve_adjusted_keys(layer_ks, rewrite_name)
            # MEMIT: ΔW = R (C0 + K K^T)^-1 K transposed.
            upd_matrix = resid.T @ adjusted_keys.T
            upd_matrix.mul_(self.edit_weight)

            weight_name = f"{rewrite_name}.weight"
            upd_matrix = _match_weight_shape(upd_matrix, weights[weight_name].shape)

            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.to(
                    device=weights[weight_name].device,
                    dtype=weights[weight_name].dtype,
                )

            del layer_ks, adjusted_keys, cur_zs, targets, upd_matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "MEMIT applied across layers %s: %d requests",
            layer_indices,
            len(requests),
        )

    def _solve_adjusted_keys(
        self, keys: torch.Tensor, rewrite_name: str
    ) -> torch.Tensor:
        """Return ``(lambda*C + K K^T)^-1 K`` without a cubic identity solve."""
        if not self.mom2_adjustment:
            scale = max(float(self.mom2_update_weight), self.covariance_ridge)
            gram = keys.T @ keys
            system = gram + scale * torch.eye(
                gram.shape[0], device=gram.device, dtype=gram.dtype
            )
            return keys @ torch.linalg.solve(
                system,
                torch.eye(system.shape[0], device=system.device, dtype=system.dtype),
            )

        covariance = self._get_covariance(rewrite_name).to(
            device=keys.device, dtype=keys.dtype
        )
        system = self.mom2_update_weight * covariance + keys @ keys.T
        system.diagonal().add_(self.covariance_ridge)
        return torch.linalg.solve(system, keys)

    def _get_covariance(self, rewrite_name: str) -> torch.Tensor:
        """Return fixed preservation statistics for this rewrite module."""
        cache_key = rewrite_name
        if cache_key not in self._cov_cache:
            stats = layer_stats(
                self.model,
                self.tokenizer,
                rewrite_name,
                self.stats_dir,
                ds_name=self.mom2_dataset,
                sample_size=self.mom2_n_samples,
                precision=self.mom2_dtype,
                **reference_stats_kwargs(self),
            )
            covariance = stats.get("mom2")
            if covariance is None:
                raise RuntimeError(f"No mom2 covariance produced for {rewrite_name}")
            self._cov_cache[cache_key] = covariance.float().cpu()

        return self._cov_cache[cache_key]
