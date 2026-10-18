"""Validated projection caches and continuation state for rank-based editors."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset

from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)
from trainer.edit.utils.cache_identity import model_cache_identity, tokenizer_identity
from trainer.edit.utils.layer_stats import _atomic_save
from trainer.edit.utils.nethook import get_module


class RankEditStateMixin:
    """Persist caches and history needed to continue editing saved model weights."""

    def _rank_config(self) -> dict:
        names = (
            "layers",
            "rewrite_module_tmp",
            "layer_module_tmp",
            "lm_head_module",
            "ln_f_module",
            "nullspace_threshold",
            "mom2_dataset",
            "mom2_n_samples",
            "mom2_dtype",
            "mom2_adjustment",
            "l2",
            "strategy",
            "window_size",
            "overlap",
            "v_lr",
            "v_num_grad_steps",
            "v_weight_decay",
            "clamp_norm_factor",
            "v_loss_layer",
            "v_loss_mode",
            "kl_factor",
            "kl_direction",
            "context_templates",
            "fact_token",
            "ft_lr",
            "ft_epochs",
            "preservation_loss_weight",
            "edit_loss_weight",
            "preservation_sample_size",
            "preservation_seed",
            "preservation_max_length",
            "require_preservation",
            "mom2_update_weight",
            "covariance_ridge",
        )
        config = {
            "shared_utilities_revision": "mit_memit_20260921_v2",
            **{name: getattr(self, name) for name in names if hasattr(self, name)},
        }
        if getattr(
            self, "implementation_variant", "native_adapted"
        ) == "official_reference" and hasattr(self, "mom2_sample_seed"):
            config.update(
                implementation_variant="official_reference",
                reference_protocol="rank_reference_20261016_v1",
                mom2_dataset_config=getattr(self, "mom2_dataset_config", None),
                mom2_sample_seed=getattr(self, "mom2_sample_seed", 1),
                mom2_batch_tokens=getattr(self, "mom2_batch_tokens", None),
            )
        return config

    def _projection_metadata(self) -> dict:
        if not hasattr(self, "_projection_source_identity"):
            self._projection_source_identity = model_cache_identity(self.model)
        if not hasattr(self, "_projection_dataset_identity"):
            dataset = self.mom2_dataset
            split = load_dataset(
                dataset,
                getattr(self, "mom2_dataset_config", None)
                or {"wikitext": "wikitext-103-raw-v1", "wikipedia": "20220301.en"}.get(
                    dataset, dataset
                ),
            )["train"]
            fingerprint = getattr(split, "_fingerprint", None)
            if fingerprint is None:
                raise ValueError(
                    "Projection caches require a versioned dataset fingerprint"
                )
            self._projection_dataset_identity = fingerprint
        metadata = {
            "schema_version": 3,
            "shared_utilities_revision": "mit_memit_20260921_v2",
            "source_model": self._projection_source_identity,
            "tokenizer": tokenizer_identity(self.tokenizer),
            "layers": list(self.layers),
            "rewrite_module_tmp": self.rewrite_module_tmp,
            "layer_module_tmp": self.layer_module_tmp,
            "threshold": self.nullspace_threshold,
            "dataset": self.mom2_dataset,
            "dataset_fingerprint": self._projection_dataset_identity,
            "sample_size": self.mom2_n_samples,
            "precision": self.mom2_dtype,
        }
        if getattr(
            self, "implementation_variant", "native_adapted"
        ) == "official_reference" and hasattr(self, "mom2_sample_seed"):
            metadata.update(
                implementation_variant="official_reference",
                dataset_config=self.mom2_dataset_config,
                sample_policy="upstream_random",
                sample_seed=self.mom2_sample_seed,
                batch_tokens=self.mom2_batch_tokens,
            )
        return metadata

    def _validate_rank_matrices(self, matrices, label: str) -> dict:
        if not isinstance(matrices, dict):
            raise ValueError(f"{label} must be a layer-to-tensor mapping")
        result = {}
        for layer, matrix in matrices.items():
            if layer not in self.layers:
                raise ValueError(f"{label} contains unconfigured layer {layer}")
            module = get_module(self.model, self.rewrite_module_tmp.format(layer))
            width = getattr(module, "in_features", module.weight.shape[0])
            if (
                not isinstance(matrix, torch.Tensor)
                or tuple(matrix.shape) != (width, width)
                or not torch.isfinite(matrix).all()
                or not torch.allclose(
                    matrix.float(), matrix.T.float(), atol=1e-4, rtol=1e-4
                )
            ):
                raise ValueError(
                    f"Invalid {label} tensor for layer {layer}; expected finite symmetric {width}x{width}"
                )
            result[int(layer)] = matrix.float().cpu()
        return result

    def _load_rank_projections(self) -> None:
        if (
            self._projection_cache
            or not self.projection_path
            or not self.projection_path.exists()
        ):
            return
        payload = torch.load(
            self.projection_path, map_location="cpu", weights_only=True
        )
        if (
            not isinstance(payload, dict)
            or payload.get("metadata") != self._projection_metadata()
        ):
            raise ValueError(
                "Projection cache identity mismatch or legacy schema; recompute for this model/tokenizer/dataset/config"
            )
        projections = self._validate_rank_matrices(
            payload.get("projections"), "projection"
        )
        if set(projections) != set(self.layers):
            raise ValueError("Projection cache must cover every configured layer")
        self._projection_cache = projections

    def _save_rank_projections(self) -> None:
        if self.projection_path and getattr(self, "save_projection", True):
            _atomic_save(
                {
                    "metadata": self._projection_metadata(),
                    "projections": self._projection_cache,
                },
                self.projection_path,
            )

    def save_editor_state(self, path: str | Path) -> None:
        """Save continuation state paired with the current edited HF checkpoint."""
        payload = {
            "schema_version": 1,
            "method": type(self).__name__,
            "implementation_variant": getattr(
                self, "implementation_variant", "native_adapted"
            ),
            "edited_model_identity": build_base_model_identity(self.model),
            "tokenizer": tokenizer_identity(self.tokenizer),
            "config": self._rank_config(),
            "projections": self._projection_cache,
            "sequential_keys": self._sequential_key_cache,
            "covariances": self._cov_cache,
            "history": [asdict(request) for request in self.edit_history],
            "projection_source_identity": getattr(
                self, "_projection_source_identity", None
            ),
            "projection_dataset_identity": getattr(
                self, "_projection_dataset_identity", None
            ),
            "preservation_texts": getattr(self, "_loaded_preservation_texts", None),
        }
        _atomic_save(payload, Path(path))

    def load_editor_state(self, path: str | Path) -> None:
        """Validate the entire state before changing any live editor cache."""
        from trainer.edit.base import EditRequest

        payload = torch.load(path, map_location="cpu", weights_only=True)
        expected_config = self._rank_config()
        reference_contexts = None
        if (
            isinstance(payload, dict)
            and getattr(self, "implementation_variant", "native_adapted")
            == "official_reference"
            and hasattr(self, "context_templates")
            and getattr(self, "context_templates", None) is None
            and isinstance(payload.get("config"), dict)
        ):
            from trainer.edit.utils.compute_z import context_template_groups

            reference_contexts = payload["config"].get("context_templates")
            if reference_contexts is None:
                raise ValueError("Reference state is missing its realized contexts")
            context_template_groups(reference_contexts)
            expected_config["context_templates"] = reference_contexts
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("method") != type(self).__name__
            or payload.get("config") != expected_config
            or payload.get("tokenizer") != tokenizer_identity(self.tokenizer)
        ):
            raise ValueError("Rank editor state method/config/tokenizer mismatch")
        validate_base_model_identity(
            self.model,
            payload.get("edited_model_identity"),
            checkpoint_label="rank editor state",
        )
        projections = self._validate_rank_matrices(
            payload.get("projections"), "projection"
        )
        sequential = self._validate_rank_matrices(
            payload.get("sequential_keys"), "sequential Gram"
        )
        covariances = payload.get("covariances")
        if not isinstance(covariances, dict) or any(
            not isinstance(matrix, torch.Tensor)
            or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not torch.isfinite(matrix).all()
            for matrix in covariances.values()
        ):
            raise ValueError("Invalid saved covariance cache")
        history = [EditRequest(**request) for request in payload["history"]]
        self._projection_cache = projections
        self._sequential_key_cache = sequential
        self._cov_cache = covariances
        self.edit_history = history
        if reference_contexts is not None:
            self.context_templates = reference_contexts
        for name in ("projection_source_identity", "projection_dataset_identity"):
            if payload.get(name) is not None:
                setattr(self, f"_{name}", payload[name])
        if hasattr(self, "_loaded_preservation_texts"):
            self._loaded_preservation_texts = payload["preservation_texts"]
