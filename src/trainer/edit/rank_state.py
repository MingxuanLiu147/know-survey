"""Validated projection caches and continuation state for rank-based editors."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)
from trainer.edit.utils.cache_identity import model_cache_identity, tokenizer_identity
from trainer.edit.utils.layer_stats import (
    _atomic_save,
    _load_statistics_split,
    _resolve_max_length,
    statistics_data_identity,
)
from trainer.edit.utils.nethook import get_module


class RankEditStateMixin:
    """Persist caches and history needed to continue editing saved model weights."""

    def _validate_rank_attention_policy(self, *, source_identity=None):
        """Check computation policy without rehashing edited weights as C0."""
        from trainer.edit.utils.reference import validate_statistics_configuration

        policy = getattr(self.model.config, "_attn_implementation", None)
        source = (
            getattr(self, "_projection_source_identity", None)
            if source_identity is None
            else source_identity
        )
        statistics = getattr(self, "_statistics_configuration", None)
        if (
            source is not None
            and (
                not isinstance(source, dict)
                or "attention_implementation" not in source
                or source["attention_implementation"] != policy
            )
        ) or (
            statistics is not None
            and statistics.get("attention_implementation") != policy
        ):
            raise ValueError(
                "Rank cache attention policy changed or is missing from its source identity; "
                "restore the original policy or prepare a fresh editor/state from the base model"
            )
        validate_statistics_configuration(self, bind=False)
        return policy

    def _rank_config(self) -> dict:
        from trainer.edit.utils.reference import statistics_execution_metadata

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
            "mom2_dataset_config",
            "mom2_sample_policy",
            "mom2_sample_seed",
            "mom2_batch_tokens",
            "mom2_max_length",
            "mom2_attention_policy",
            "mom2_query_chunk_size",
            "mom2_placement_policy",
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
            **statistics_execution_metadata(self),
        }
        if getattr(self, "mom2_data_path", None) is not None:
            config["mom2_local_source"] = statistics_data_identity(self.mom2_data_path)
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
        from trainer.edit.utils.reference import (
            statistics_execution_metadata,
            validate_statistics_configuration,
        )

        self._validate_rank_attention_policy()
        if not hasattr(self, "_projection_source_identity"):
            source_identity = model_cache_identity(self.model)
            if statistics_execution_metadata(self):
                validate_statistics_configuration(self)
            self._projection_source_identity = source_identity
        dataset = self.mom2_dataset
        dataset_config = getattr(self, "mom2_dataset_config", None) or {
            "wikitext": "wikitext-103-raw-v1",
            "wikipedia": "20220301.en",
        }.get(dataset, dataset)
        data_path = getattr(self, "mom2_data_path", None)
        local_source = statistics_data_identity(data_path)
        split = _load_statistics_split(dataset, dataset_config, data_path)
        fingerprint = getattr(split, "_fingerprint", None)
        if fingerprint is None:
            raise ValueError(
                "Projection caches require a versioned dataset fingerprint"
            )
        self._projection_dataset_identity = fingerprint
        model_limit = _resolve_max_length(self.model.config)
        length_override = getattr(self, "mom2_max_length", None)
        max_length = min(model_limit, length_override or model_limit)
        batch_tokens = getattr(self, "mom2_batch_tokens", None) or max_length * 3
        metadata = {
            **statistics_execution_metadata(self),
            "schema_version": 4,
            "shared_utilities_revision": "mit_memit_20260921_v2",
            "source_model": self._projection_source_identity,
            "tokenizer": tokenizer_identity(self.tokenizer),
            "layers": list(self.layers),
            "rewrite_module_tmp": self.rewrite_module_tmp,
            "layer_module_tmp": self.layer_module_tmp,
            "threshold": self.nullspace_threshold,
            "dataset": self.mom2_dataset,
            "dataset_config": dataset_config,
            "dataset_fingerprint": self._projection_dataset_identity,
            "sample_size": self.mom2_n_samples,
            "precision": self.mom2_dtype,
            "sample_policy": getattr(self, "mom2_sample_policy", "sequential_rows"),
            "max_length": min(max_length, batch_tokens),
            "batch_tokens": batch_tokens,
        }
        if local_source is not None:
            metadata["local_source"] = local_source
        if getattr(
            self, "implementation_variant", "native_adapted"
        ) == "official_reference" and hasattr(self, "mom2_sample_seed"):
            metadata.update(
                implementation_variant="official_reference",
                sample_policy="upstream_random",
                sample_seed=self.mom2_sample_seed,
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
        self._validate_rank_attention_policy()
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
        self._validate_rank_attention_policy()
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
        policy = self._validate_rank_attention_policy()
        payload = {
            "schema_version": 1,
            "method": type(self).__name__,
            "attention_implementation": policy,
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

        policy = self._validate_rank_attention_policy()
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
        has_cached_state = any(
            bool(payload.get(name))
            for name in ("projections", "sequential_keys", "covariances")
            if isinstance(payload.get(name), dict)
        )
        if has_cached_state and (
            "attention_implementation" not in payload
            or payload["attention_implementation"] != policy
        ):
            raise ValueError(
                "Rank editor state cache attention policy is missing or differs from the model; "
                "prepare new state from the base model with the intended policy"
            )
        self._validate_rank_attention_policy(
            source_identity=payload.get("projection_source_identity")
        )
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
        from trainer.edit.utils.reference import validate_statistics_configuration

        validate_statistics_configuration(self)
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
