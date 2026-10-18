"""AlphaEdit: null-space constrained locate-then-edit updates.

This implementation follows AlphaEdit (Fang et al., ICLR 2025) and the
authors' EasyEdit release:

1. collect the non-central second moment of preserved rewrite keys;
2. obtain its null space with an SVD and form ``P = U_null U_null.T``;
3. optimise target values at the final rewrite layer;
4. solve the projected multi-layer least-squares system while retaining the
   key Gram matrix of earlier edits for sequential editing.

Only decoder-only text models are supported here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.rank_state import RankEditStateMixin
from trainer.edit.utils.compute_z import (
    compute_z,
    get_layer_outputs_at_words,
    get_module_input_output_at_words,
)
from trainer.edit.utils.layer_stats import layer_stats
from trainer.edit.utils.low_rank import solve_projected_ridge_low_rank
from trainer.edit.utils.nethook import get_parameter
from trainer.edit.utils.reference import (
    REFERENCE_VARIANT,
    ensure_reference_contexts,
    reference_stats_kwargs,
    validate_reference_profile,
)

logger = logging.getLogger(__name__)


def _match_weight_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Handle Conv1D-style transposed weights used by GPT-2."""
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Update matrix shape {tuple(matrix.shape)} does not match weight "
        f"shape {tuple(shape)}"
    )


class AlphaEditEditor(RankEditStateMixin, EditTrainer):
    """Apply AlphaEdit's SVD null-space projection to MEMIT-style updates."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        v_lr: float = 5e-1,
        v_num_grad_steps: int = 20,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 4.0,
        v_loss_layer: int = 31,
        v_loss_mode: str = "final_logits",
        kl_direction: str = "base_to_edited",
        context_templates: Optional[List[str]] = None,
        implementation_variant: str = "native_adapted",
        kl_factor: float = 0.0625,
        nullspace_threshold: float = 1e-2,
        l2: float = 10.0,
        fact_token: str = "subject_last",
        rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj",
        layer_module_tmp: str = "model.layers.{}",
        lm_head_module: str = "lm_head",
        ln_f_module: str = "model.norm",
        mom2_dataset: str = "wikipedia",
        mom2_n_samples: int = 100_000,
        mom2_dtype: str = "float32",
        stats_dir: str = "data/stats",
        projection_path: Optional[str] = None,
        save_projection: bool = True,
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
        self.nullspace_threshold = nullspace_threshold
        self.l2 = l2
        self.fact_token = fact_token
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.lm_head_module = lm_head_module
        self.ln_f_module = ln_f_module
        self.mom2_dataset = mom2_dataset
        self.mom2_n_samples = mom2_n_samples
        self.mom2_dtype = mom2_dtype
        self.stats_dir = stats_dir
        self.projection_path = Path(projection_path) if projection_path else None
        self.save_projection = save_projection
        self.mom2_dataset_config = mom2_dataset_config or (
            "20200501.en"
            if implementation_variant == REFERENCE_VARIANT
            and mom2_dataset == "wikipedia"
            else None
        )
        self.mom2_sample_seed = mom2_sample_seed
        self.mom2_batch_tokens = mom2_batch_tokens
        validate_reference_profile(self, "AlphaEdit")

        self._cov_cache: Dict[str, torch.Tensor] = {}
        self._projection_cache: Dict[int, torch.Tensor] = {}
        # K_p K_p^T from edits completed by this editor instance.
        self._sequential_key_cache: Dict[int, torch.Tensor] = {}

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """Edit one batch and retain its keys for later sequential edits."""
        del kwargs
        if isinstance(requests, EditRequest):
            requests = [requests]
        if not requests:
            return {
                "success": True,
                "edited_count": 0,
                "metrics": {},
                "layers_edited": [],
            }

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "layers_edited": [],
        }
        transaction_state: Optional[Dict[str, Any]] = None
        try:
            ensure_reference_contexts(self, "AlphaEdit")
            layer_indices = self._resolve_layer_indices(self.layers)
            results["layers_edited"] = layer_indices
            transaction_state = self._snapshot_transaction_state(layer_indices)
            projections = self._compute_nullspace_projections(layer_indices)
            self._apply_alphaedit(
                requests,
                layer_indices,
                projections,
                weights_copy=transaction_state["weights"],
            )
            results["edited_count"] = len(requests)
        except Exception as exc:
            if transaction_state is not None:
                self._restore_transaction_state(transaction_state)
            logger.error("AlphaEdit edit failed: %s", exc, exc_info=True)
            results["success"] = False
            results["edited_count"] = 0
            results["error"] = str(exc)
        return results

    def _snapshot_transaction_state(self, layer_indices: List[int]) -> Dict[str, Any]:
        """Capture all request-mutable AlphaEdit state."""

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
            "cov_cache": dict(self._cov_cache),
            "projection_cache": dict(self._projection_cache),
            "sequential_key_cache": dict(self._sequential_key_cache),
            "edit_history": list(self.edit_history),
        }

    def _restore_transaction_state(self, state: Dict[str, Any]) -> None:
        """Restore model weights and caches after a failed AlphaEdit call."""

        with torch.no_grad():
            for name, value in state["weights"].items():
                parameter = get_parameter(self.model, name)
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )
        self._cov_cache = state["cov_cache"]
        self._projection_cache = state["projection_cache"]
        self._sequential_key_cache = state["sequential_key_cache"]
        self.edit_history = state["edit_history"]

    def reset_edit_memory(self) -> None:
        """Forget keys from earlier edits without discarding the fixed P matrix."""
        self._sequential_key_cache.clear()

    def _get_covariance(self, layer_name: str) -> torch.Tensor:
        cache_key = layer_name
        if cache_key not in self._cov_cache:
            stats = layer_stats(
                self.model,
                self.tokenizer,
                layer_name,
                self.stats_dir,
                ds_name=self.mom2_dataset,
                to_collect=["mom2"],
                sample_size=self.mom2_n_samples,
                precision=self.mom2_dtype,
                **reference_stats_kwargs(self),
            )
            covariance = stats.get("mom2")
            if covariance is None:
                raise RuntimeError(f"No mom2 covariance produced for {layer_name}")
            self._cov_cache[cache_key] = covariance.float().cpu()
        return self._cov_cache[cache_key]

    def _load_projection_file(self) -> None:
        self._load_rank_projections()

    def _save_projection_file(self) -> None:
        self._save_rank_projections()

    def _compute_nullspace_projections(
        self, layer_indices: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Compute Eq. 8--9 from AlphaEdit with an actual SVD.

        Singular vectors whose singular values are below the configured
        threshold span the null space.  The projection is fixed for all edits
        and may be persisted because it is independent of edit requests.
        """
        self._load_projection_file()
        computed = False
        for layer_idx in layer_indices:
            if layer_idx in self._projection_cache:
                continue
            layer_name = self.rewrite_module_tmp.format(layer_idx)
            projection_device = get_parameter(self.model, f"{layer_name}.weight").device
            covariance = self._get_covariance(layer_name).to(projection_device)
            eigenvectors, singular_values, _ = torch.linalg.svd(
                covariance, full_matrices=False
            )
            null_columns = singular_values < self.nullspace_threshold
            null_basis = eigenvectors[:, null_columns]
            projection = null_basis @ null_basis.T
            self._projection_cache[layer_idx] = projection.float().cpu()
            computed = True
            logger.info(
                "AlphaEdit layer=%d nullity=%d/%d threshold=%g",
                layer_idx,
                int(null_columns.sum().item()),
                covariance.shape[0],
                self.nullspace_threshold,
            )
            del covariance, eigenvectors, singular_values, null_basis, projection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if computed:
            self._save_projection_file()
        return {layer: self._projection_cache[layer] for layer in layer_indices}

    def _apply_alphaedit(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
        projections: Dict[int, torch.Tensor],
        *,
        weights_copy: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        model, tok = self.model, self.tokenizer
        weights = {
            f"{self.rewrite_module_tmp.format(layer)}.weight": get_parameter(
                model, f"{self.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in layer_indices
        }
        if weights_copy is None:
            weights_copy = {
                name: weight.detach().clone() for name, weight in weights.items()
            }

        z_layer = layer_indices[-1]
        target_values = []
        for request in requests:
            target_values.append(
                compute_z(
                    model,
                    tok,
                    {
                        "prompt": request.prompt,
                        "subject": request.subject,
                        "target_new": request.target_new,
                    },
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
            )
        desired_zs = torch.stack(target_values, dim=0)
        prompts = [request.prompt for request in requests]
        subjects = [request.subject for request in requests]

        for position, layer_idx in enumerate(layer_indices):
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
            current_zs = get_layer_outputs_at_words(
                model,
                tok,
                prompts,
                subjects,
                z_layer,
                layer_module_tmp=self.layer_module_tmp,
                fact_token=self.fact_token,
            )
            residual = desired_zs.to(current_zs.device) - current_zs
            residual = residual.float() / max(len(layer_indices) - position, 1)

            keys = layer_keys.T.float()
            device = keys.device
            projection = projections[layer_idx].to(device)
            prior_gram = self._sequential_key_cache.get(layer_idx)
            if prior_gram is None:
                # Independent/batch edits start with no prior key Gram.  The
                # dense system is ridge*I + (P K)K^T, so Woodbury gives the
                # exact same solution through a request-count-sized system.
                # This avoids an O(input_dim^3) solve for every zsRE case.
                projected_keys = projection @ keys
                rhs = projected_keys @ residual.to(device)
                update = solve_projected_ridge_low_rank(
                    projected_keys,
                    keys,
                    rhs,
                    self.l2,
                )
            else:
                prior_gram = prior_gram.to(device)
                system = projection @ (keys @ keys.T + prior_gram)
                system.diagonal().add_(self.l2)
                rhs = projection @ keys @ residual.to(device)
                update = torch.linalg.solve(system, rhs)

            weight_name = f"{rewrite_name}.weight"
            update = _match_weight_shape(update, weights[weight_name].shape)
            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + update.to(
                    device=weights[weight_name].device,
                    dtype=weights[weight_name].dtype,
                )

        # The keys produced by the edited model become K_p in the next call,
        # matching the authors' sequential-edit cache.
        for layer_idx in layer_indices:
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
            keys = layer_keys.T.float().cpu()
            gram = keys @ keys.T
            if layer_idx in self._sequential_key_cache:
                gram = self._sequential_key_cache[layer_idx] + gram
            self._sequential_key_cache[layer_idx] = gram

        logger.info(
            "AlphaEdit applied across layers %s: %d requests",
            layer_indices,
            len(requests),
        )


__all__ = ["AlphaEditEditor"]
