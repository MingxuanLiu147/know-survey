"""
ROME knowledge editor
=====================

Rank-One Model Editing via shared ``compute_z`` + single-layer rank-one update.

Paper: Locating and Editing Factual Associations in GPT
https://arxiv.org/abs/2202.05262

The native profile retains the repository objective. The explicit
``official_reference`` profile uses the authors' MLP intervention, KL direction,
context generation and inverse-moment rank-one constraint. Neither profile is
a vendored, line-by-line EasyEdit fork or a claim of benchmark reproduction.
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


class ROMEEditor(EditTrainer):
    """ROME editor: single-layer rank-one MLP update driven by ``compute_z``."""

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
        mlp_module_tmp: str = "model.layers.{}.mlp",
        context_template_length_params: Optional[List[List[int]]] = None,
        mom2_dataset_config: str | None = None,
        mom2_sample_seed: int = 1,
        mom2_batch_tokens: int | None = None,
        *args,
        **kwargs,
    ):
        layers = [5] if layers is None else layers
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
        self.mlp_module_tmp = mlp_module_tmp
        self.context_template_length_params = context_template_length_params or [
            [5, 10],
            [10, 10],
        ]
        self.mom2_dataset_config = mom2_dataset_config or (
            "20200501.en"
            if implementation_variant == REFERENCE_VARIANT
            and mom2_dataset == "wikipedia"
            else None
        )
        self.mom2_sample_seed = mom2_sample_seed
        self.mom2_batch_tokens = mom2_batch_tokens
        validate_reference_profile(self, "ROME")
        self._inv_cov_cache: Dict[str, torch.Tensor] = {}

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

        weight = None
        original = None
        original_cache = dict(self._inv_cov_cache)
        try:
            ensure_reference_contexts(self, "ROME")
            layer_indices = self._resolve_layer_indices(self.layers)
            if len(layer_indices) != 1:
                raise ValueError("ROME requires exactly one rewrite layer")
            layer_idx = layer_indices[0]
            results["layers_edited"] = [layer_idx]
            weight = get_parameter(
                self.model, f"{self.rewrite_module_tmp.format(layer_idx)}.weight"
            )
            original = weight.detach().clone()
            for request in requests:
                self._apply_rome_edit(request, layer_idx)
            results["edited_count"] = len(requests)
        except Exception as exc:
            if original is not None:
                with torch.no_grad():
                    weight.copy_(original)
            self._inv_cov_cache = original_cache
            logger.error("ROME edit call rolled back: %s", exc, exc_info=True)
            results["success"] = False
            results["edited_count"] = 0
            results["error"] = str(exc)

        return results

    def _apply_rome_edit(self, request: EditRequest, layer_idx: int) -> None:
        model, tok = self.model, self.tokenizer
        rewrite_name = self.rewrite_module_tmp.format(layer_idx)
        weight = get_parameter(model, f"{rewrite_name}.weight")
        req_dict = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
        }
        z = compute_z(
            model,
            tok,
            req_dict,
            layer_idx,
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
            intervention_module_tmp=(
                self.mlp_module_tmp
                if self.implementation_variant == REFERENCE_VARIANT
                else None
            ),
            optimization_convention=(
                "upstream"
                if self.implementation_variant == REFERENCE_VARIANT
                else "native"
            ),
        )

        prompts, subjects = [request.prompt], [request.subject]
        keys, _ = get_module_input_output_at_words(
            model,
            tok,
            prompts,
            subjects,
            rewrite_name,
            context_templates=self.context_templates,
            fact_token=self.fact_token,
        )
        if self.implementation_variant == REFERENCE_VARIANT:
            current_z = get_module_input_output_at_words(
                model,
                tok,
                prompts,
                subjects,
                rewrite_name,
                fact_token=self.fact_token,
            )[1][0]
        else:
            current_z = get_layer_outputs_at_words(
                model,
                tok,
                prompts,
                subjects,
                layer_idx,
                layer_module_tmp=self.layer_module_tmp,
                fact_token=self.fact_token,
            )[0]

        key = keys[0].to(device=weight.device, dtype=torch.float32)
        residual = (z.to(current_z.device) - current_z).to(
            device=weight.device, dtype=torch.float32
        )
        left = self._whiten_key(key, rewrite_name)
        # Contexts determine the left vector, but the rank-one write must
        # satisfy the original request's key: delta @ bare_key == residual.
        # See ROME compute_v at 0874014cd9837e4365f3e6f3c71400ef11509e04.
        if self.context_templates == ["{}"]:
            bare_key = key
        else:
            bare_keys, _ = get_module_input_output_at_words(
                model,
                tok,
                prompts,
                subjects,
                rewrite_name,
                context_templates=["{}"],
                fact_token=self.fact_token,
            )
            bare_key = bare_keys[0].to(device=weight.device, dtype=torch.float32)
        denominator = torch.dot(bare_key, left)
        tolerance = torch.finfo(denominator.dtype).eps * bare_key.norm() * left.norm()
        if not torch.isfinite(denominator) or denominator.abs() <= tolerance:
            raise ValueError("ROME rank-one denominator is non-finite or near zero")
        # A negative inner product is valid; clamping it to a positive epsilon
        # would reverse and greatly amplify the requested update.
        right = residual / denominator

        with torch.no_grad():
            delta = torch.outer(right, left)
            if not torch.isfinite(delta).all():
                raise ValueError("ROME rank-one update contains non-finite values")
            if delta.shape != weight.shape:
                if delta.T.shape == weight.shape:
                    delta = delta.T
                else:
                    raise ValueError(
                        f"ROME update {tuple(delta.shape)} does not match "
                        f"{rewrite_name}.weight {tuple(weight.shape)}"
                    )
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype))

        logger.info(
            "ROME edit applied at layer %d: %s -> %s (residual_norm=%.4f)",
            layer_idx,
            request.subject,
            request.target_new,
            residual.norm().item(),
        )

    def _whiten_key(self, key: torch.Tensor, rewrite_name: str) -> torch.Tensor:
        """Apply the ROME inverse second-moment correction to a rewrite key."""
        if not self.mom2_adjustment:
            return key / key.norm().clamp(min=1e-12)

        cache_key = f"{self.model.config._name_or_path}:{rewrite_name}"
        if cache_key not in self._inv_cov_cache:
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
            covariance = (
                covariance.clone()
                if self.implementation_variant == REFERENCE_VARIANT
                else covariance.float()
            )
            covariance.diagonal().add_(self.covariance_ridge)
            self._inv_cov_cache[cache_key] = torch.linalg.inv(covariance).float().cpu()

        inverse = self._inv_cov_cache[cache_key].to(key.device)
        whitened = inverse @ key.float()
        return whitened / whitened.norm().clamp(min=1e-12)
