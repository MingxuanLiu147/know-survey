"""AnyEdit autoregressive knowledge editing for long-form targets.

AnyEdit (Jiang et al., ICML 2025) is an autoregressive editing *paradigm*,
not a router over existing editors.  It divides a target into token chunks,
optimises one hidden-state perturbation at each chunk boundary while applying
earlier perturbations, and writes all boundary key/value pairs back together.

The public distribution provides ``MEMIT_ARE`` and ``AlphaEdit_ARE``.
The local research checkout may additionally contain an UnKE backend; that
backend is excluded from public exports while its source permission is unresolved.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.rank_state import RankEditStateMixin
from trainer.edit.utils.compute_z import format_request_prompt
from trainer.edit.utils.layer_stats import layer_stats
from trainer.edit.utils.low_rank import solve_ridge_low_rank
from trainer.edit.utils.reference import reference_stats_kwargs
from trainer.edit.utils.nethook import (
    Trace,
    get_module,
    get_parameter,
    set_requires_grad,
)

logger = logging.getLogger(__name__)
UPSTREAM_REVISION = "057a77f185f7ffb55818f6bd9add37f43bb447e7"

SUPPORTED_STRATEGIES = frozenset(
    {
        "memit",
        "memit_are",
        "alphaedit",
        "alphaedit_are",
        "unke",
        "unke_are",
    }
)


def unke_research_available() -> bool:
    """Whether the optional local research backend files are present.

    Availability only describes files; it does not grant source permission.
    """
    return all(
        find_spec(f"trainer.edit.{name}") is not None
        for name in ("unke_editor", "preservation", "research_unke_backend")
    )


def _hidden_state(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Unsupported traced output type: {type(output)!r}")


def _replace_hidden_state(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    raise TypeError(f"Unsupported traced output type: {type(output)!r}")


def _input_device(model) -> torch.device:
    for module_name in ("model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"):
        try:
            return next(get_module(model, module_name).parameters()).device
        except (LookupError, StopIteration):
            continue
    return next(model.parameters()).device


def _token_ids(tokenizer, text: str, *, add_special_tokens: bool) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _match_weight_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Update matrix shape {tuple(matrix.shape)} does not match weight "
        f"shape {tuple(shape)}"
    )


@contextmanager
def _preserve_requires_grad(model):
    states = [parameter.requires_grad for parameter in model.parameters()]
    try:
        set_requires_grad(False, model)
        yield
    finally:
        for parameter, state in zip(model.parameters(), states):
            parameter.requires_grad_(state)


class AnyEditEditor(RankEditStateMixin, EditTrainer):
    """Autoregressively edit long targets with a MEMIT or AlphaEdit backend."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        implementation_variant: str = "native_adapted",
        strategy: str = "memit",
        window_size: int = 40,
        overlap: int = 0,
        target_prefix: Optional[str] = None,
        v_lr: float = 5e-1,
        v_num_grad_steps: int = 25,
        v_loss_layer: Optional[int] = None,
        ln_f_module: str = "model.norm",
        lm_head_module: str = "lm_head",
        v_weight_decay: float = 1e-3,
        clamp_norm_factor: float = 4.0,
        mom2_update_weight: float = 15_000.0,
        mom2_adjustment: bool = True,
        mom2_dataset: str = "wikipedia",
        mom2_n_samples: int = 100_000,
        mom2_dtype: str = "float32",
        mom2_dataset_config: Optional[str] = None,
        mom2_sample_policy: Optional[str] = None,
        mom2_sample_seed: int = 1,
        mom2_batch_tokens: Optional[int] = None,
        mom2_max_length: Optional[int] = None,
        stats_dir: str = "data/stats",
        covariance_ridge: float = 1e-5,
        nullspace_threshold: float = 2e-2,
        l2: float = 10.0,
        rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj",
        layer_module_tmp: str = "model.layers.{}",
        projection_path: Optional[str] = None,
        ft_lr: float = 2e-4,
        ft_epochs: int = 50,
        edit_loss_weight: float = 1.0,
        preservation_loss_weight: float = 1.0,
        preservation_data_path: Optional[
            str
        ] = "data/edit/preservation/alpaca_data.json",
        preservation_sample_size: int = 20,
        preservation_seed: int = 2024,
        preservation_max_length: int = 256,
        preservation_texts: Optional[List[str]] = None,
        require_preservation: bool = True,
        *args,
        **kwargs,
    ):
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown AnyEdit backend {strategy!r}; supported: "
                f"{sorted(SUPPORTED_STRATEGIES)}"
            )
        if strategy.removesuffix("_are") == "unke" and not unke_research_available():
            raise ValueError(
                "UnKE is excluded from the public release because its historical "
                "source permission is unresolved. Use strategy='memit_are' or "
                "strategy='alphaedit_are'."
            )
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must satisfy 0 <= overlap < window_size")
        if v_num_grad_steps < 1:
            raise ValueError("v_num_grad_steps must be positive")

        layers = [4, 5, 6, 7, 8] if layers is None else layers
        super().__init__(layers=layers, *args, **kwargs)
        if implementation_variant not in {"native_adapted", "official_reference"}:
            raise ValueError("Unknown AnyEdit implementation_variant")
        if (
            implementation_variant == "official_reference"
            and strategy.removesuffix("_are") == "unke"
        ):
            raise ValueError(
                "The official_reference profile covers public MEMIT_ARE/AlphaEdit_ARE only"
            )
        self.implementation_variant = implementation_variant
        self.requested_strategy = strategy
        self.strategy = strategy.removesuffix("_are")
        self.window_size = window_size
        self.overlap = overlap
        self.target_prefix = target_prefix
        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_loss_layer = v_loss_layer
        self.ln_f_module = ln_f_module
        self.lm_head_module = lm_head_module
        self.v_weight_decay = v_weight_decay
        self.clamp_norm_factor = clamp_norm_factor
        self.mom2_update_weight = mom2_update_weight
        self.mom2_adjustment = mom2_adjustment
        self.mom2_dataset = mom2_dataset
        self.mom2_n_samples = mom2_n_samples
        self.mom2_dtype = mom2_dtype
        self.mom2_dataset_config = mom2_dataset_config or (
            "20220301.en" if mom2_dataset == "wikipedia" else None
        )
        self.mom2_sample_policy = mom2_sample_policy or (
            "upstream_random"
            if implementation_variant == "official_reference"
            else "sequential_rows"
        )
        if self.mom2_sample_policy not in {"upstream_random", "sequential_rows"}:
            raise ValueError("Unsupported AnyEdit mom2_sample_policy")
        self.mom2_sample_seed = mom2_sample_seed
        self.mom2_max_length = mom2_max_length
        if (
            implementation_variant == "official_reference"
            and self.mom2_max_length is None
        ):
            model_type = str(getattr(self.model.config, "model_type", ""))
            if "qwen2" in model_type:
                self.mom2_max_length = 4096
            elif "mistral" in model_type:
                self.mom2_max_length = (
                    getattr(self.model.config, "sliding_window", None) or 4096
                )
        self.mom2_batch_tokens = mom2_batch_tokens
        if self.mom2_batch_tokens is None and self.mom2_max_length is not None:
            self.mom2_batch_tokens = self.mom2_max_length * 3
        self.stats_dir = stats_dir
        self.covariance_ridge = covariance_ridge
        self.nullspace_threshold = nullspace_threshold
        self.l2 = l2
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.projection_path = Path(projection_path) if projection_path else None
        self.ft_lr = ft_lr
        self.ft_epochs = ft_epochs
        self.edit_loss_weight = edit_loss_weight
        self.preservation_loss_weight = preservation_loss_weight
        self.preservation_data_path = preservation_data_path
        self.preservation_sample_size = preservation_sample_size
        self.preservation_seed = preservation_seed
        self.preservation_max_length = preservation_max_length
        self.preservation_texts = list(preservation_texts or [])
        self.require_preservation = require_preservation

        self._cov_cache: Dict[str, torch.Tensor] = {}
        self._projection_cache: Dict[int, torch.Tensor] = {}
        self._sequential_key_cache: Dict[int, torch.Tensor] = {}
        self._loaded_preservation_texts: Optional[List[str]] = None
        self._last_backend_metrics: Dict[str, Any] = {}
        self._last_target_metrics: Dict[str, Any] = {}

    def _rank_config(self) -> dict:
        """Reject continuation caches produced by older sequential-only ARE code."""
        return {
            **super()._rank_config(),
            "implementation_variant": self.implementation_variant,
            "anyedit_revision": "joint_are_v1",
            "upstream_revision": UPSTREAM_REVISION,
            "target_prefix": self.target_prefix,
            "anyedit_statistics_revision": "upstream_are_20261016_v1",
            "mom2_dataset_config": self.mom2_dataset_config,
            "mom2_sample_policy": self.mom2_sample_policy,
            "mom2_sample_seed": self.mom2_sample_seed,
            "mom2_batch_tokens": self.mom2_batch_tokens,
            "mom2_max_length": self.mom2_max_length,
        }

    def _projection_metadata(self) -> dict:
        """Bind projections to the actual ARE sampling and truncation adapter."""
        return {
            **super()._projection_metadata(),
            "anyedit_statistics_revision": "upstream_are_20261016_v1",
            "dataset_config": self.mom2_dataset_config,
            "sample_policy": self.mom2_sample_policy,
            "sample_seed": self.mom2_sample_seed,
            "batch_tokens": self.mom2_batch_tokens,
            "max_length": self.mom2_max_length,
        }

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        del kwargs
        if isinstance(requests, EditRequest):
            requests = [requests]
        requests = list(requests)

        layer_indices = self._resolve_layer_indices(self.layers) if requests else []
        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "strategy": f"{self.strategy}+anyedit",
            "requested_strategy": self.requested_strategy,
            "layers_edited": layer_indices,
            "autoregressive_chunks": 0,
            "implementation_variant": self.implementation_variant,
            "upstream_revision": UPSTREAM_REVISION,
            "execution_semantics": (
                "request_sequential_atomic_call"
                if self.strategy == "unke"
                else "joint_batch_atomic_call"
            ),
        }
        if not requests:
            return results

        transaction_state = self._snapshot_transaction_state(layer_indices)
        try:
            self.model.eval()
            if self.strategy != "unke":
                counts, metrics = self._apply_joint_requests(
                    requests,
                    layer_indices,
                    weights_copy=transaction_state["rewrite_weights"],
                )
                results["edited_count"] = len(requests)
                results["autoregressive_chunks"] = sum(counts)
                results["chunks_per_request"] = counts
                results["metrics"] = metrics
            else:
                for request in requests:
                    chunk_count = self._apply_request(request, layer_indices)
                    results["edited_count"] += 1
                    results["autoregressive_chunks"] += chunk_count
                    if self._last_backend_metrics:
                        results["metrics"].setdefault("requests", []).append(
                            self._last_backend_metrics
                        )
        except Exception as exc:
            self._restore_transaction_state(transaction_state)
            logger.error("AnyEdit failed: %s", exc, exc_info=True)
            results["success"] = False
            results["edited_count"] = 0
            results["autoregressive_chunks"] = 0
            results["metrics"] = {}
            results["error"] = str(exc)
        finally:
            for module, training in transaction_state["module_training"]:
                module.training = training
        return results

    def _snapshot_transaction_state(self, layer_indices: List[int]) -> Dict[str, Any]:
        """Capture call-level state for all AnyEdit backends."""

        parameters: Dict[str, torch.Tensor] = {}
        if self.strategy == "unke":
            for layer_idx in layer_indices:
                layer_name = self.layer_module_tmp.format(layer_idx)
                layer = get_module(self.model, layer_name)
                for relative_name, parameter in layer.named_parameters():
                    parameters[f"{layer_name}.{relative_name}"] = parameter
        else:
            for layer_idx in layer_indices:
                name = f"{self.rewrite_module_tmp.format(layer_idx)}.weight"
                parameters[name] = get_parameter(self.model, name)

        parameter_state = {
            name: {
                "parameter": parameter,
                "value": parameter.detach().clone(),
                "requires_grad": parameter.requires_grad,
                "grad": (
                    None if parameter.grad is None else parameter.grad.detach().clone()
                ),
            }
            for name, parameter in parameters.items()
        }
        return {
            "parameters": parameter_state,
            "rewrite_weights": {
                name: state["value"] for name, state in parameter_state.items()
            },
            "cov_cache": dict(self._cov_cache),
            "projection_cache": dict(self._projection_cache),
            "sequential_key_cache": dict(self._sequential_key_cache),
            "loaded_preservation_texts": (
                None
                if self._loaded_preservation_texts is None
                else list(self._loaded_preservation_texts)
            ),
            "last_backend_metrics": self._last_backend_metrics,
            "last_target_metrics": self._last_target_metrics,
            "edit_history": list(self.edit_history),
            "module_training": [
                (module, module.training) for module in self.model.modules()
            ],
        }

    def _restore_transaction_state(self, state: Dict[str, Any]) -> None:
        """Restore weights, gradients, caches, and history after failure."""

        with torch.no_grad():
            for parameter_state in state["parameters"].values():
                parameter = parameter_state["parameter"]
                parameter.copy_(
                    parameter_state["value"].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                )
                parameter.requires_grad_(parameter_state["requires_grad"])
                parameter.grad = parameter_state["grad"]
        self._cov_cache = state["cov_cache"]
        self._projection_cache = state["projection_cache"]
        self._sequential_key_cache = state["sequential_key_cache"]
        self._loaded_preservation_texts = state["loaded_preservation_texts"]
        self._last_backend_metrics = state["last_backend_metrics"]
        self._last_target_metrics = state["last_target_metrics"]
        self.edit_history = state["edit_history"]
        for module, training in state["module_training"]:
            module.training = training

    def _apply_request(
        self,
        request: EditRequest,
        layer_indices: List[int],
        *,
        weights_copy: Optional[Dict[str, torch.Tensor]] = None,
    ) -> int:
        z_layer = layer_indices[-1]
        full_ids, lookup_indices, desired_zs = self._compute_autoregressive_targets(
            request, z_layer
        )
        self._last_backend_metrics = {}
        if self.strategy == "unke":
            self._last_backend_metrics = self._apply_unke_are(
                full_ids, lookup_indices, desired_zs, layer_indices, z_layer
            )
            return len(lookup_indices)

        self._apply_rank_targets(
            [(full_ids, lookup_indices, desired_zs)],
            layer_indices,
            weights_copy=weights_copy,
        )
        return len(lookup_indices)

    def _apply_joint_requests(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
        *,
        weights_copy: Dict[str, torch.Tensor],
    ) -> tuple[List[int], Dict[str, Any]]:
        """Optimize every request on the same model, then solve one joint update.

        Forward passes may be streamed per request to avoid padding and excess
        memory; the concatenated K, K_preserve and Z enter one solve per layer.
        This is a joint edit, not a sequence of per-request weight updates.
        """
        targets, diagnostics = [], []
        for request in requests:
            targets.append(
                self._compute_autoregressive_targets(request, layer_indices[-1])
            )
            diagnostics.append(dict(self._last_target_metrics))
        layers = self._apply_rank_targets(
            targets, layer_indices, weights_copy=weights_copy
        )
        return [len(item[1]) for item in targets], {
            "requests": diagnostics,
            "layers": layers,
        }

    def _apply_rank_targets(
        self,
        targets: List[tuple[List[int], List[int], torch.Tensor]],
        layer_indices: List[int],
        *,
        weights_copy: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply the authors' combined boundary/preservation matrix update."""
        z_layer = layer_indices[-1]
        desired_zs = torch.cat([item[2] for item in targets], dim=0)

        weights = {
            f"{self.rewrite_module_tmp.format(layer)}.weight": get_parameter(
                self.model, f"{self.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in layer_indices
        }
        if weights_copy is None:
            weights_copy = {
                name: weight.detach().clone() for name, weight in weights.items()
            }

        if self.strategy == "alphaedit":
            # Fix every projection before writes change later-layer keys.
            for layer_idx in layer_indices:
                self._get_projection(
                    layer_idx, self.rewrite_module_tmp.format(layer_idx)
                )
            self._save_rank_projections()
        elif self.mom2_adjustment:
            for layer_idx in layer_indices:
                self._get_covariance(self.rewrite_module_tmp.format(layer_idx))

        diagnostics = []
        for position, layer_idx in enumerate(layer_indices):
            rewrite_name = self.rewrite_module_tmp.format(layer_idx)
            keys, preserve_keys, current_zs = self._collect_joint_activations(
                targets, rewrite_name, z_layer
            )

            residual = desired_zs.to(current_zs.device) - current_zs
            residual = residual.float() / max(len(layer_indices) - position, 1)
            if self.strategy == "memit":
                update = self._memit_update(keys, preserve_keys, residual, rewrite_name)
            else:
                update = self._alphaedit_update(
                    layer_idx, keys, preserve_keys, residual, rewrite_name
                )

            weight_name = f"{rewrite_name}.weight"
            update = _match_weight_shape(update, weights[weight_name].shape)
            if not torch.isfinite(update).all():
                raise ValueError(f"Non-finite AnyEdit update at {rewrite_name}")
            with torch.no_grad():
                updated_weight = weights_copy[weight_name] + update.to(
                    device=weights[weight_name].device,
                    dtype=weights[weight_name].dtype,
                )
                if not torch.isfinite(updated_weight).all():
                    raise ValueError(f"AnyEdit weight overflow at {rewrite_name}")
                weights[weight_name].copy_(updated_weight)
            diagnostics.append(
                {
                    "layer": layer_idx,
                    "boundary_keys": keys.shape[1],
                    "preservation_keys": preserve_keys.shape[1],
                    "residual_norm": float(residual.norm().item()),
                    "update_norm": float(update.norm().item()),
                }
            )

        if (
            self.strategy == "alphaedit"
            and self.implementation_variant == "native_adapted"
        ):
            for layer_idx in layer_indices:
                rewrite_name = self.rewrite_module_tmp.format(layer_idx)
                keys, _, _ = self._collect_joint_activations(
                    targets, rewrite_name, z_layer
                )
                keys = keys.cpu()
                gram = keys @ keys.T
                if layer_idx in self._sequential_key_cache:
                    gram = self._sequential_key_cache[layer_idx] + gram
                self._sequential_key_cache[layer_idx] = gram

        logger.info(
            "AnyEdit backend=%s layers=%s chunks=%d target_tokens=%d",
            self.strategy,
            layer_indices,
            sum(len(item[1]) for item in targets),
            sum(len(item[0]) for item in targets),
        )
        return diagnostics

    def _collect_joint_activations(self, targets, rewrite_name, z_layer):
        """Concatenate real token columns; padding is never preservation evidence."""
        keys, preservation, values = [], [], []
        for full_ids, lookup_indices, _ in targets:
            inputs, current = self._collect_full_activations(
                full_ids, lookup_indices, rewrite_name, z_layer
            )
            selected = torch.zeros(
                inputs.shape[1], dtype=torch.bool, device=inputs.device
            )
            selected[lookup_indices] = True
            keys.append(inputs[:, lookup_indices].float())
            preservation.append(inputs[:, ~selected].float())
            values.append(current)
        return (
            torch.cat(keys, dim=1),
            torch.cat(preservation, dim=1),
            torch.cat(values, dim=0),
        )

    def _apply_unke_are(self, *args, **kwargs) -> Dict[str, Any]:
        """Dispatch to the separately retained research-only backend."""
        if not unke_research_available():
            raise ValueError(
                "UnKE is excluded from the public release because its historical "
                "source permission is unresolved. Use AnyEdit MEMIT or AlphaEdit."
            )
        from trainer.edit.research_unke_backend import _apply_unke_are

        return _apply_unke_are(self, *args, **kwargs)

    def _compute_autoregressive_targets(
        self, request: EditRequest, layer: int
    ) -> tuple[List[int], List[int], torch.Tensor]:
        """Optimise one value vector per target chunk (paper Eq. 9)."""
        model, tokenizer = self.model, self.tokenizer
        device = _input_device(model)
        prompt = format_request_prompt(request.prompt, request.subject)
        prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True)
        target_text = str(request.target_new)
        official = self.implementation_variant == "official_reference"
        prefix = self.target_prefix
        if prefix is None:
            prefix = "" if official else " "
        if target_text and prefix and not target_text.startswith(prefix):
            target_text = prefix + target_text
        target_ids = _token_ids(tokenizer, target_text, add_special_tokens=official)
        if (
            official
            and target_ids
            and target_ids[0]
            in {
                tokenizer.bos_token_id,
                tokenizer.unk_token_id,
            }
        ):
            target_ids = target_ids[1:]
        if not prompt_ids or not target_ids:
            raise ValueError("AnyEdit requires non-empty prompt and target tokens")

        model_limit = None
        for name in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(model.config, name, None)
            if value:
                model_limit = int(value)
                break
        full_ids = prompt_ids + target_ids
        if model_limit is not None and len(full_ids) > model_limit:
            raise ValueError(
                f"AnyEdit sequence has {len(full_ids)} tokens but model limit is "
                f"{model_limit}; shorten the sample or use a longer-context model"
            )

        layer_name = self.layer_module_tmp.format(layer)
        loss_layer = max(
            layer, self.v_loss_layer if self.v_loss_layer is not None else layer
        )
        loss_layer_name = self.layer_module_tmp.format(loss_layer)
        if official:
            # The authors optimize a configurable intermediate block through
            # the model's final normalization and vocabulary projection.
            get_module(model, loss_layer_name)
            final_norm = get_module(model, self.ln_f_module)
            lm_head = get_module(model, self.lm_head_module)
        edit_device = next(get_module(model, layer_name).parameters()).device
        hidden_size = getattr(model.config, "hidden_size", None) or getattr(
            model.config, "n_embd"
        )
        prior_deltas: List[tuple[int, torch.Tensor]] = []
        lookup_indices: List[int] = []
        desired_values: List[torch.Tensor] = []
        chunk_metrics: List[Dict[str, Any]] = []
        step_size = self.window_size - self.overlap

        with _preserve_requires_grad(model):
            for start in range(0, len(target_ids), step_size):
                end = min(start + self.window_size, len(target_ids))
                current_targets = target_ids[start:end]
                # Tokens up to the last token of the current chunk are inputs;
                # each target token is predicted by the preceding position.
                input_list = prompt_ids + target_ids[: max(end - 1, 0)]
                input_ids = torch.tensor([input_list], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                lookup_idx = len(prompt_ids) + start - 1
                prediction_positions = torch.arange(
                    lookup_idx,
                    lookup_idx + len(current_targets),
                    device=device,
                )
                target_tensor = torch.tensor(
                    current_targets, dtype=torch.long, device=device
                )

                delta = torch.zeros(hidden_size, device=edit_device, requires_grad=True)
                target_init: Optional[torch.Tensor] = None

                def edit_output_fn(output, layer=None):
                    del layer
                    nonlocal target_init
                    hidden = _hidden_state(output)
                    edited = hidden.clone()
                    if target_init is None:
                        target_init = hidden[0, lookup_idx].detach().clone()
                    for old_idx, old_delta in prior_deltas:
                        if old_idx < edited.shape[1]:
                            edited[0, old_idx] += old_delta.to(
                                device=edited.device, dtype=edited.dtype
                            )
                    edited[0, lookup_idx] += delta.to(
                        device=edited.device, dtype=edited.dtype
                    )
                    return _replace_hidden_state(output, edited)

                optimizer = torch.optim.Adam([delta], lr=self.v_lr)
                losses = []
                updates = 0
                for iteration in range(self.v_num_grad_steps):
                    optimizer.zero_grad(set_to_none=True)
                    with Trace(
                        model,
                        layer_name,
                        retain_output=False,
                        edit_output=edit_output_fn,
                    ), Trace(
                        model,
                        loss_layer_name if official else layer_name,
                        retain_output=official,
                    ) as loss_trace:
                        logits = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        ).logits
                    if official:
                        hidden = _hidden_state(loss_trace.output)
                        norm_device = next(final_norm.parameters()).device
                        head_device = next(lm_head.parameters()).device
                        logits = lm_head(
                            final_norm(hidden.to(norm_device)).to(head_device)
                        )
                    log_probs = torch.log_softmax(logits[0], dim=-1)
                    nll = -log_probs[
                        prediction_positions.to(log_probs.device),
                        target_tensor.to(log_probs.device),
                    ].mean()
                    if target_init is None:
                        raise RuntimeError(
                            f"Could not capture AnyEdit representation at {layer_name}"
                        )
                    norm_penalty = (
                        self.v_weight_decay
                        * delta.norm()
                        / target_init.norm().pow(2).clamp(min=1e-12)
                    )
                    loss = nll + norm_penalty.to(nll.device)
                    if not torch.isfinite(loss):
                        raise ValueError(
                            f"Non-finite AnyEdit target loss at chunk {start}"
                        )
                    losses.append(float(loss.detach().item()))
                    if official and (
                        loss.item() < 1e-2 or iteration == self.v_num_grad_steps - 1
                    ):
                        break
                    loss.backward()
                    optimizer.step()
                    updates += 1
                    with torch.no_grad():
                        max_norm = self.clamp_norm_factor * target_init.norm()
                        if delta.norm() > max_norm:
                            delta.mul_(max_norm / delta.norm().clamp(min=1e-12))
                    if loss.item() < 1e-2:
                        break

                assert target_init is not None
                stored_delta = delta.detach().clone()
                prior_deltas.append((lookup_idx, stored_delta))
                lookup_indices.append(lookup_idx)
                desired_values.append(target_init + stored_delta.to(target_init.dtype))
                chunk_metrics.append(
                    {
                        "start": start,
                        "end": end,
                        "boundary": lookup_idx,
                        "target_tokens": len(current_targets),
                        "optimizer_updates": updates,
                        "losses": losses,
                    }
                )
                # An overlapping window that already reaches the end covers
                # every token. The pinned upstream while-loop can otherwise
                # add an all-overlap tail and supervise a token on itself.
                if end == len(target_ids):
                    break

        self._last_target_metrics = {
            "prompt_tokens": len(prompt_ids),
            "target_tokens": len(target_ids),
            "chunk_boundaries": lookup_indices,
            "chunks": chunk_metrics,
            "loss_layer": loss_layer if official else "final_logits",
        }
        return full_ids, lookup_indices, torch.stack(desired_values, dim=0)

    def _collect_full_activations(
        self,
        full_ids: List[int],
        lookup_indices: List[int],
        rewrite_name: str,
        z_layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = _input_device(self.model)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with (
            torch.no_grad(),
            Trace(
                self.model,
                rewrite_name,
                retain_input=True,
                retain_output=False,
                detach=True,
                clone=True,
            ) as key_trace,
            Trace(
                self.model,
                self.layer_module_tmp.format(z_layer),
                retain_input=False,
                retain_output=True,
                detach=True,
                clone=True,
            ) as z_trace,
        ):
            self.model(input_ids=input_ids, attention_mask=attention_mask)
        layer_input = (
            key_trace.input[0]
            if isinstance(key_trace.input, tuple)
            else key_trace.input
        )
        z_output = _hidden_state(z_trace.output)
        current_zs = torch.stack(
            [z_output[0, index] for index in lookup_indices], dim=0
        )
        # Return (input_dim, sequence) so selected and preserved columns share
        # the same orientation as the official AnyEdit implementation.
        return layer_input[0].T, current_zs

    def _memit_update(
        self,
        keys: torch.Tensor,
        preserve_keys: torch.Tensor,
        residual: torch.Tensor,
        rewrite_name: str,
    ) -> torch.Tensor:
        if not self.mom2_adjustment:
            features = torch.cat((keys, preserve_keys), dim=1)
            adjusted_keys = solve_ridge_low_rank(features, keys, self.covariance_ridge)
            return residual.to(keys.device).T @ adjusted_keys.T

        system = (
            keys @ keys.T
            + preserve_keys @ preserve_keys.T
            + self.mom2_update_weight
            * self._get_covariance(rewrite_name).to(keys.device)
        )
        system.diagonal().add_(self.covariance_ridge)
        adjusted_keys = torch.linalg.solve(system, keys)
        return residual.to(keys.device).T @ adjusted_keys.T

    def _alphaedit_update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        preserve_keys: torch.Tensor,
        residual: torch.Tensor,
        rewrite_name: str,
    ) -> torch.Tensor:
        projection = self._get_projection(layer_idx, rewrite_name).to(keys.device)
        # AlphaEdit_ARE at the pinned revision preserves the other tokens in
        # this batch. It does not include plain AlphaEdit's historical K_p.
        prior = (
            self._sequential_key_cache.get(layer_idx)
            if self.implementation_variant == "native_adapted"
            else None
        )
        if prior is None:
            prior = torch.zeros(
                (keys.shape[0], keys.shape[0]),
                device=keys.device,
                dtype=keys.dtype,
            )
        else:
            prior = prior.to(keys.device)
        system = projection @ (keys @ keys.T + preserve_keys @ preserve_keys.T + prior)
        system.diagonal().add_(self.l2)
        rhs = projection @ keys @ residual.to(keys.device)
        return torch.linalg.solve(system, rhs)

    def _get_covariance(self, rewrite_name: str) -> torch.Tensor:
        cache_key = rewrite_name
        if cache_key not in self._cov_cache:
            stats_kwargs = reference_stats_kwargs(self)
            stats_kwargs.update(
                dataset_config=self.mom2_dataset_config,
                sample_policy=self.mom2_sample_policy,
                sample_seed=self.mom2_sample_seed,
                batch_tokens=self.mom2_batch_tokens,
            )
            if self.mom2_max_length is not None:
                stats_kwargs["max_length"] = self.mom2_max_length
            stats = layer_stats(
                self.model,
                self.tokenizer,
                rewrite_name,
                self.stats_dir,
                ds_name=self.mom2_dataset,
                to_collect=["mom2"],
                sample_size=self.mom2_n_samples,
                precision=self.mom2_dtype,
                **stats_kwargs,
            )
            covariance = stats.get("mom2")
            if covariance is None:
                raise RuntimeError(f"No mom2 covariance produced for {rewrite_name}")
            self._cov_cache[cache_key] = covariance.float().cpu()
        return self._cov_cache[cache_key]

    def _load_projections(self) -> None:
        self._load_rank_projections()

    def _get_projection(self, layer_idx: int, rewrite_name: str) -> torch.Tensor:
        self._load_projections()
        if layer_idx not in self._projection_cache:
            device = get_parameter(self.model, f"{rewrite_name}.weight").device
            covariance = self._get_covariance(rewrite_name).to(device)
            vectors, values, _ = torch.linalg.svd(covariance, full_matrices=False)
            basis = vectors[:, values < self.nullspace_threshold]
            self._projection_cache[layer_idx] = (basis @ basis.T).float().cpu()
        return self._projection_cache[layer_idx]


__all__ = ["AnyEditEditor", "SUPPORTED_STRATEGIES"]
