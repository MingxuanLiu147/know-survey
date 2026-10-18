"""Compact, checkpoint-gated MALMEN editor.

This module follows the executable core of the public EasyEdit MALMEN path:

* cache each edited projection's input keys and output-value gradients;
* transform key/value-gradient pairs with a learned residual auxiliary net;
* solve the regularised normal equation for a joint batch update; and
* apply the orientation-correct update to ``Linear`` or GPT-2 ``Conv1D``
  weights.

The implementation is intentionally smaller than the official training stack.
It supports the shared :class:`EditTrainer` request/evaluation contract and a
compact differentiable meta-training objective, but it is **not** archive
compatible with EasyEdit and must not be described as a faithful official
reproduction.  In particular, ``edit`` is always gated on a native checkpoint
that was either loaded or produced by :meth:`train_auxiliary_network`; random
auxiliary weights are never applied as an edit.

Paper: Fast Model Editing at Scale (https://arxiv.org/abs/2304.00740)
Reference implementation inspected during this port:
``EasyEdit/easyeditor/trainer/algs/MALMEN.py`` and
``EasyEdit/easyeditor/trainer/algs/malmen/nets.py``.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)

logger = logging.getLogger(__name__)

_LEGACY_CHECKPOINT_FORMAT = "know_surgery_compact_malmen_v1"
_CHECKPOINT_FORMAT = "know_surgery_compact_malmen_v2"


class _RunningMeanStd(nn.Module):
    """Numerically stable feature normaliser stored in the checkpoint."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("count", torch.zeros((), dtype=torch.float32))
        self.register_buffer("mean", torch.zeros(size, dtype=torch.float32))
        self.register_buffer("m2", torch.zeros(size, dtype=torch.float32))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float()
        if values.ndim != 2 or values.shape[0] == 0:
            return
        batch_count = torch.tensor(
            float(values.shape[0]), device=self.count.device, dtype=self.count.dtype
        )
        batch_mean = values.mean(dim=0).to(self.mean.device)
        batch_m2 = (values.to(self.mean.device) - batch_mean).pow(2).sum(dim=0)
        if float(self.count.item()) == 0.0:
            self.count.copy_(batch_count)
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / total))
        self.m2.add_(batch_m2 + delta.pow(2) * self.count * batch_count / total)
        self.count.copy_(total)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if float(self.count.item()) <= 1.0:
            return values
        variance = self.m2 / (self.count - 1.0)
        return (values - self.mean) / variance.clamp_min(1e-12).sqrt()


class _MALMENBlock(nn.Module):
    """Low-rank residual block used by the public MALMEN auxiliary net."""

    def __init__(self, size: int, rank: int, n_modules: int):
        super().__init__()
        self.left = nn.Parameter(torch.empty(size, rank))
        self.right = nn.Parameter(torch.zeros(rank, size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.scale = nn.Embedding(n_modules, size)
        self.shift = nn.Embedding(n_modules, size)
        nn.init.normal_(self.left, std=0.02)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, values: torch.Tensor, module_index: int) -> torch.Tensor:
        index = torch.tensor(module_index, device=values.device, dtype=torch.long)
        # Match the public MALMEN network's ``x.clamp(0)`` exactly.  Unlike
        # ReLU, PyTorch clamp propagates the boundary gradient at zero, which
        # is required because ``right`` and ``bias`` start at all zeros.
        residual = (values @ self.left @ self.right + self.bias).clamp_min(0.0)
        residual = self.scale(index) * residual + self.shift(index)
        return values + residual


class _MALMENNetwork(nn.Module):
    """Transform cached keys/value gradients for one projection shape."""

    def __init__(
        self,
        key_size: int,
        value_size: int,
        rank: int,
        n_blocks: int,
        n_modules: int,
        edit_lr: float,
        lambda_reg: float,
    ):
        super().__init__()
        feature_size = key_size + value_size
        self.key_size = key_size
        self.value_size = value_size
        self.normalizer = _RunningMeanStd(feature_size)
        self.blocks = nn.ModuleList(
            _MALMENBlock(feature_size, rank, n_modules) for _ in range(n_blocks)
        )
        self.log_edit_lr = nn.Embedding(n_modules, 1)
        self.log_lambda = nn.Embedding(n_modules, 1)
        nn.init.constant_(self.log_edit_lr.weight, math.log(edit_lr))
        nn.init.constant_(self.log_lambda.weight, math.log(lambda_reg))

    def forward(
        self,
        keys: torch.Tensor,
        value_gradients: torch.Tensor,
        module_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat((keys, value_gradients), dim=-1)
        hidden = self.normalizer(hidden)
        for block in self.blocks:
            hidden = block(hidden, module_index)
        return hidden.split((self.key_size, self.value_size), dim=-1)

    def edit_lr(self, module_index: int) -> torch.Tensor:
        index = torch.tensor(
            module_index, device=self.log_edit_lr.weight.device, dtype=torch.long
        )
        return self.log_edit_lr(index).exp()

    def regularizer(self, module_index: int) -> torch.Tensor:
        index = torch.tensor(
            module_index, device=self.log_lambda.weight.device, dtype=torch.long
        )
        return self.log_lambda(index).exp()


class _TargetSpec(NamedTuple):
    parameter_name: str
    module_name: str
    module: nn.Module
    key_size: int
    value_size: int
    network_key: str
    module_index: int


class MALMENEditor(EditTrainer):
    """Compact native MALMEN-style batch editor with a strict checkpoint gate.

    This class implements the paper's key/value-gradient transformation and
    regularised joint solve.  The repository-native meta-training objective is
    compact and intended for representative smokes, not official-score
    reproduction.  EasyEdit/official archives are rejected with a clear error
    because their module layout and state-dict schema differ.
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        archive: Optional[str] = None,
        require_pretrained: bool = True,
        target_modules: Optional[List[str]] = None,
        edit_lr: float = 1e-4,
        meta_lr: float = 1e-4,
        rank: int = 64,
        n_blocks: int = 2,
        batch_size: int = 4,
        n_edits: int = 16,
        lambda_reg: float = 1e-3,
        locality_weight: float = 1.0,
        update_norm_weight: float = 1e-6,
        max_locality_samples: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(layers=layers, *args, **kwargs)
        if edit_lr <= 0 or meta_lr <= 0 or lambda_reg <= 0:
            raise ValueError("MALMEN learning rates and lambda_reg must be positive")
        if rank <= 0 or n_blocks <= 0 or batch_size <= 0 or n_edits <= 0:
            raise ValueError("MALMEN rank, blocks, batch_size, and n_edits must be > 0")

        self.archive = archive
        self.require_pretrained = require_pretrained
        self.target_modules = target_modules or [
            "mlp.down_proj",
            "mlp.c_proj",
            "mlp.dense_4h_to_h",
        ]
        self.edit_lr = edit_lr
        self.meta_lr = meta_lr
        self.rank = rank
        self.n_blocks = n_blocks
        self.batch_size = batch_size
        self.n_edits = n_edits
        self.lambda_reg = lambda_reg
        self.locality_weight = locality_weight
        self.update_norm_weight = update_norm_weight
        self.max_locality_samples = max_locality_samples

        self.auxiliary_networks = nn.ModuleDict()
        self._targets: List[_TargetSpec] = []
        self._initialized = False
        self._pretrained_loaded = False
        if archive:
            self.load_auxiliary_network(archive)

    # ------------------------------------------------------------------
    # Target and auxiliary-network initialisation
    # ------------------------------------------------------------------

    def _module_dimensions(self, module: nn.Module) -> tuple[int, int]:
        if isinstance(module, nn.Linear):
            return int(module.in_features), int(module.out_features)
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError(
                f"MALMEN target {module.__class__.__name__} has no matrix weight"
            )
        # Hugging Face GPT-2 Conv1D stores weights as (input, output).
        if module.__class__.__name__ == "Conv1D":
            return int(weight.shape[0]), int(weight.shape[1])
        # Other projection modules are expected to follow Linear orientation.
        return int(weight.shape[1]), int(weight.shape[0])

    def _resolve_target_specs(self) -> list[_TargetSpec]:
        named_modules = {
            id(module): name for name, module in self.model.named_modules()
        }
        raw_targets: list[tuple[str, str, nn.Module, int, int]] = []
        for layer_index in self._resolve_layer_indices(self.layers):
            layer = self._get_layer_module(self.model, layer_index)
            if layer is None:
                raise ValueError(f"Cannot find transformer layer {layer_index}")
            target = None
            for suffix in self.target_modules:
                try:
                    target = self._get_module_by_name(layer, suffix)
                    break
                except AttributeError:
                    continue
            if target is None:
                raise ValueError(
                    f"Cannot resolve any MALMEN target {self.target_modules} "
                    f"inside transformer layer {layer_index}"
                )
            module_name = named_modules.get(id(target))
            if module_name is None:
                raise ValueError("Resolved MALMEN target is not a named model module")
            key_size, value_size = self._module_dimensions(target)
            raw_targets.append(
                (f"{module_name}.weight", module_name, target, key_size, value_size)
            )

        grouped: dict[tuple[int, int], list[tuple[str, str, nn.Module, int, int]]]
        grouped = defaultdict(list)
        for target in raw_targets:
            grouped[(target[3], target[4])].append(target)

        targets: list[_TargetSpec] = []
        for (key_size, value_size), members in grouped.items():
            network_key = f"k{key_size}_v{value_size}"
            for module_index, member in enumerate(members):
                targets.append(
                    _TargetSpec(
                        parameter_name=member[0],
                        module_name=member[1],
                        module=member[2],
                        key_size=key_size,
                        value_size=value_size,
                        network_key=network_key,
                        module_index=module_index,
                    )
                )
        return targets

    def _make_auxiliary_networks(
        self,
        targets: list[_TargetSpec],
        *,
        rank: int,
        n_blocks: int,
        edit_lr: float,
        lambda_reg: float,
    ) -> nn.ModuleDict:
        grouped: dict[str, list[_TargetSpec]] = defaultdict(list)
        for target in targets:
            grouped[target.network_key].append(target)
        networks = nn.ModuleDict()
        for network_key, members in grouped.items():
            first = members[0]
            networks[network_key] = _MALMENNetwork(
                key_size=first.key_size,
                value_size=first.value_size,
                rank=min(rank, first.key_size + first.value_size),
                n_blocks=n_blocks,
                n_modules=len(members),
                edit_lr=edit_lr,
                lambda_reg=lambda_reg,
            )
        return networks.to(next(self.model.parameters()).device)

    def _init_auxiliary_networks(self, *, force: bool = False) -> None:
        if self._initialized and not force:
            return

        targets = self._resolve_target_specs()
        networks = self._make_auxiliary_networks(
            targets,
            rank=self.rank,
            n_blocks=self.n_blocks,
            edit_lr=self.edit_lr,
            lambda_reg=self.lambda_reg,
        )
        self.auxiliary_networks = networks
        self._targets = targets
        self._initialized = True
        logger.info(
            "Initialised compact MALMEN auxiliary nets for %s",
            [target.parameter_name for target in targets],
        )

    @staticmethod
    def _manifest_for_targets(
        targets: Iterable[_TargetSpec],
    ) -> list[dict[str, Any]]:
        return [
            {
                "parameter_name": target.parameter_name,
                "key_size": target.key_size,
                "value_size": target.value_size,
                "weight_shape": list(target.module.weight.shape),
                "network_key": target.network_key,
                "module_index": target.module_index,
            }
            for target in targets
        ]

    def _target_manifest(self) -> list[dict[str, Any]]:
        self._init_auxiliary_networks()
        return self._manifest_for_targets(self._targets)

    @staticmethod
    def _expected_state_shapes(
        targets: list[_TargetSpec], *, rank: int, n_blocks: int
    ) -> dict[str, tuple[int, ...]]:
        grouped: dict[str, list[_TargetSpec]] = defaultdict(list)
        for target in targets:
            grouped[target.network_key].append(target)
        shapes: dict[str, tuple[int, ...]] = {}
        for network_key, members in grouped.items():
            first = members[0]
            feature_size = first.key_size + first.value_size
            block_rank = min(rank, feature_size)
            prefix = f"{network_key}."
            shapes[f"{prefix}normalizer.count"] = ()
            shapes[f"{prefix}normalizer.mean"] = (feature_size,)
            shapes[f"{prefix}normalizer.m2"] = (feature_size,)
            for block_index in range(n_blocks):
                block = f"{prefix}blocks.{block_index}."
                shapes[f"{block}left"] = (feature_size, block_rank)
                shapes[f"{block}right"] = (block_rank, feature_size)
                shapes[f"{block}bias"] = (feature_size,)
                shapes[f"{block}scale.weight"] = (len(members), feature_size)
                shapes[f"{block}shift.weight"] = (len(members), feature_size)
            shapes[f"{prefix}log_edit_lr.weight"] = (len(members), 1)
            shapes[f"{prefix}log_lambda.weight"] = (len(members), 1)
        return shapes

    # ------------------------------------------------------------------
    # Native checkpoint contract
    # ------------------------------------------------------------------

    def save_auxiliary_network(self, path: Union[str, Path]) -> None:
        if not self._pretrained_loaded:
            raise RuntimeError(
                "Refusing to save an untrained MALMEN auxiliary network. "
                "Run train_auxiliary_network first."
            )
        self._init_auxiliary_networks()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _CHECKPOINT_FORMAT,
            "base_model_identity": build_base_model_identity(self.model),
            "state_dict": self.auxiliary_networks.state_dict(),
            "targets": self._target_manifest(),
            "rank": self.rank,
            "n_blocks": self.n_blocks,
            "edit_lr": self.edit_lr,
            "lambda_reg": self.lambda_reg,
            "model_name_or_path": getattr(self.model.config, "_name_or_path", None),
        }
        torch.save(payload, path)
        self.archive = str(path)
        logger.info("Saved compact MALMEN auxiliary checkpoint to %s", path)

    def load_auxiliary_network(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MALMEN auxiliary checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(
                "Unsupported MALMEN checkpoint. Know-Surgery compact MALMEN is "
                "not compatible with EasyEdit/official archives; pretrain a native "
                "checkpoint with scripts/train_malmen_auxiliary_network.py."
            )
        checkpoint_format = payload.get("format")
        if checkpoint_format == _LEGACY_CHECKPOINT_FORMAT:
            raise ValueError(
                "Legacy compact MALMEN checkpoint is missing required "
                "base_model_identity and is incompatible; retrain the auxiliary "
                "network against the exact base-model weights."
            )
        if checkpoint_format != _CHECKPOINT_FORMAT:
            raise ValueError(
                "Unsupported MALMEN checkpoint. Know-Surgery compact MALMEN is "
                "not compatible with EasyEdit/official archives; pretrain a native "
                "checkpoint with scripts/train_malmen_auxiliary_network.py."
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("MALMEN checkpoint is missing a state_dict mapping")
        required = {
            "rank",
            "n_blocks",
            "edit_lr",
            "lambda_reg",
            "targets",
            "base_model_identity",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                f"MALMEN v2 checkpoint is missing required metadata: {missing}"
            )
        if (
            not isinstance(payload["rank"], int)
            or isinstance(payload["rank"], bool)
            or not isinstance(payload["n_blocks"], int)
            or isinstance(payload["n_blocks"], bool)
        ):
            raise ValueError("MALMEN v2 rank/n_blocks must contain exact integers")
        rank = payload["rank"]
        n_blocks = payload["n_blocks"]
        if not 1 <= rank <= 16_384 or not 1 <= n_blocks <= 128:
            raise ValueError("MALMEN checkpoint rank/n_blocks are outside safe bounds")
        try:
            edit_lr = float(payload["edit_lr"])
            lambda_reg = float(payload["lambda_reg"])
        except (TypeError, ValueError) as error:
            raise ValueError("MALMEN v2 learning-rate metadata is invalid") from error
        if (
            not math.isfinite(edit_lr)
            or not math.isfinite(lambda_reg)
            or edit_lr <= 0
            or lambda_reg <= 0
        ):
            raise ValueError("MALMEN v2 learning-rate metadata is outside safe bounds")
        if not isinstance(payload["targets"], list):
            raise ValueError("MALMEN v2 checkpoint targets must be a list")

        targets = self._resolve_target_specs()
        current_targets = self._manifest_for_targets(targets)
        if payload["targets"] != current_targets:
            raise ValueError(
                "MALMEN checkpoint target modules/shapes do not match the current "
                "model. Pretrain a checkpoint for this exact model and layer set."
            )
        expected_shapes = self._expected_state_shapes(
            targets, rank=rank, n_blocks=n_blocks
        )
        if set(state_dict) != set(expected_shapes) or any(
            not isinstance(state_dict[name], torch.Tensor)
            or tuple(state_dict[name].shape) != shape
            for name, shape in expected_shapes.items()
        ):
            raise ValueError(
                "MALMEN checkpoint state_dict keys/shapes do not match its metadata"
            )
        if sum(math.prod(shape) for shape in expected_shapes.values()) > 50_000_000:
            raise ValueError("MALMEN auxiliary network exceeds safe size bounds")

        validate_base_model_identity(
            self.model,
            payload["base_model_identity"],
            checkpoint_label="MALMEN",
        )
        auxiliary_networks = self._make_auxiliary_networks(
            targets,
            rank=rank,
            n_blocks=n_blocks,
            edit_lr=edit_lr,
            lambda_reg=lambda_reg,
        )
        auxiliary_networks.load_state_dict(state_dict, strict=True)

        self.rank = rank
        self.n_blocks = n_blocks
        self.edit_lr = edit_lr
        self.lambda_reg = lambda_reg
        self.auxiliary_networks = auxiliary_networks
        self._targets = targets
        self._initialized = True
        self._pretrained_loaded = True
        self.archive = str(path)
        logger.info("Loaded compact MALMEN auxiliary checkpoint from %s", path)

    def _require_ready(self) -> None:
        self._init_auxiliary_networks()
        if not self._pretrained_loaded:
            reason = (
                "MALMEN requires a trained native auxiliary checkpoint before edit. "
                "Run scripts/train_malmen_auxiliary_network.py and set "
                "trainer.method_args.archive. Random auxiliary weights are never "
                "applied."
            )
            if not self.require_pretrained:
                reason += " require_pretrained=false does not bypass this safety gate."
            raise RuntimeError(reason)

    # ------------------------------------------------------------------
    # Public edit path
    # ------------------------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        del kwargs
        if isinstance(requests, EditRequest):
            requests = [requests]
        requests = list(requests)
        if not requests:
            raise ValueError("MALMEN edit requires at least one request")

        transaction_state = self._snapshot_transaction_state()
        try:
            self._require_ready()

            edited_count = 0
            # MALMEN jointly solves up to n_edits requests.  Larger workloads are
            # explicitly chunked rather than silently truncating the request list.
            for start in range(0, len(requests), self.n_edits):
                batch = requests[start : start + self.n_edits]
                caches = self._cache_requests(batch, update_normalizer=False)
                shifts = self._predict_parameter_shifts(caches)
                self._apply_parameter_shifts(shifts)
                edited_count += len(batch)
        except Exception:
            self._restore_transaction_state(transaction_state)
            raise

        return {
            "success": True,
            "edited_count": edited_count,
            "metrics": {},
            "checkpoint_loaded": True,
            "implementation": "compact_native_malmen",
        }

    def _snapshot_transaction_state(self) -> Dict[str, Any]:
        """Capture model and compact meta-network state for one edit call."""

        parameter_state = {}
        if self._initialized:
            for target in self._targets:
                parameter = target.module.weight
                parameter_state[target.parameter_name] = {
                    "parameter": parameter,
                    "value": parameter.detach().clone(),
                    "requires_grad": parameter.requires_grad,
                    "grad": (
                        None
                        if parameter.grad is None
                        else parameter.grad.detach().clone()
                    ),
                }
        return {
            "parameters": parameter_state,
            "auxiliary_networks": self.auxiliary_networks,
            "auxiliary_state": {
                name: value.detach().clone()
                for name, value in self.auxiliary_networks.state_dict().items()
            },
            "auxiliary_training": [
                (module, module.training)
                for module in self.auxiliary_networks.modules()
            ],
            "targets": list(self._targets),
            "initialized": self._initialized,
            "pretrained_loaded": self._pretrained_loaded,
            "archive": self.archive,
            "edit_history": list(self.edit_history),
            "model_training": self.model.training,
        }

    def _restore_transaction_state(self, state: Dict[str, Any]) -> None:
        """Restore all state if a later MALMEN chunk fails."""

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

        self.auxiliary_networks = state["auxiliary_networks"]
        self.auxiliary_networks.load_state_dict(state["auxiliary_state"], strict=True)
        for module, training in state["auxiliary_training"]:
            module.train(training)
        self._targets = state["targets"]
        self._initialized = state["initialized"]
        self._pretrained_loaded = state["pretrained_loaded"]
        self.archive = state["archive"]
        self.edit_history = state["edit_history"]
        self.model.train(state["model_training"])

    # ------------------------------------------------------------------
    # Key/value-gradient cache and normal-equation solve
    # ------------------------------------------------------------------

    def _safe_max_length(self) -> int:
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", model_limit)
        try:
            tokenizer_limit = int(tokenizer_limit)
        except (TypeError, ValueError, OverflowError):
            tokenizer_limit = model_limit
        if tokenizer_limit <= 0 or tokenizer_limit > 10_000_000:
            tokenizer_limit = model_limit
        return min(model_limit, tokenizer_limit)

    def _build_edit_batch(
        self, request: EditRequest, *, prompt: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        prompt = str(prompt if prompt is not None else request.prompt)
        target = str(request.target_new)
        full_text = f"{prompt} {target}"
        encoded = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._safe_max_length(),
        )
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self._safe_max_length(),
        )["input_ids"]
        batch = {
            key: value.to(self._input_device())
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        labels = batch["input_ids"].clone()
        labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        if not labels.ne(-100).any():
            raise ValueError("MALMEN target produced no supervised tokens")
        batch["labels"] = labels
        return batch

    def _cache_request(
        self, request: EditRequest, *, update_normalizer: bool
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        self._init_auxiliary_networks()
        captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        activations: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        hooks = []

        def make_hook(target: _TargetSpec):
            def hook(_module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(tensor):
                    raise TypeError(
                        f"MALMEN target {target.module_name} returned non-tensor output"
                    )
                activations[target.parameter_name] = (inputs[0], tensor)

            return hook

        for target in self._targets:
            hooks.append(target.module.register_forward_hook(make_hook(target)))

        was_training = self.model.training
        target_requires_grad = [
            target.module.weight.requires_grad for target in self._targets
        ]
        for target in self._targets:
            target.module.weight.requires_grad_(True)
        self.model.eval()
        try:
            outputs = self.model(**self._build_edit_batch(request))
            ordered_outputs = [
                activations[target.parameter_name][1] for target in self._targets
            ]
            output_gradients = torch.autograd.grad(
                outputs.loss,
                ordered_outputs,
                retain_graph=False,
                create_graph=False,
            )
        finally:
            for hook in hooks:
                hook.remove()
            for target, required in zip(self._targets, target_requires_grad):
                target.module.weight.requires_grad_(required)
            self.model.train(was_training)

        for target, output_gradient in zip(self._targets, output_gradients):
            keys = (
                activations[target.parameter_name][0]
                .detach()
                .reshape(-1, target.key_size)
            )
            value_gradients = output_gradient.detach().reshape(-1, target.value_size)
            active = value_gradients.abs().sum(dim=-1) > 0
            keys = keys[active].float()
            value_gradients = value_gradients[active].float()
            if keys.shape[0] == 0:
                raise RuntimeError(
                    f"No active target-token gradients for {target.parameter_name}"
                )
            network = self.auxiliary_networks[target.network_key]
            network_device = next(network.parameters()).device
            keys = keys.to(network_device)
            value_gradients = value_gradients.to(network_device)
            if update_normalizer:
                network.normalizer.update(torch.cat((keys, value_gradients), -1))
            captured[target.parameter_name] = (keys, value_gradients)
        return captured

    def _cache_requests(
        self, requests: Iterable[EditRequest], *, update_normalizer: bool
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        per_target: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(
            list
        )
        for request in requests:
            request_cache = self._cache_request(
                request, update_normalizer=update_normalizer
            )
            for name, pair in request_cache.items():
                per_target[name].append(pair)
        return {
            name: (
                torch.cat([pair[0] for pair in pairs], dim=0),
                torch.cat([pair[1] for pair in pairs], dim=0),
            )
            for name, pairs in per_target.items()
        }

    def _predict_parameter_shifts(
        self, caches: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        shifts: dict[str, torch.Tensor] = {}
        for target in self._targets:
            keys, value_gradients = caches[target.parameter_name]
            network = self.auxiliary_networks[target.network_key]
            pseudo_keys, pseudo_value_gradients = network(
                keys, value_gradients, target.module_index
            )
            coefficients = -network.edit_lr(target.module_index) * (
                keys * pseudo_keys
            ).sum(dim=-1, keepdim=True)
            value_differences = coefficients * pseudo_value_gradients

            # Woodbury/dual form of (K^T K + lambda I)^-1 K^T V.  It is
            # mathematically equivalent to the public implementation's primal
            # normal equation while avoiding a cubic solve in GPT-2's 3072-D
            # c_proj input space for short edit requests.
            regularizer = network.regularizer(target.module_index)
            gram = keys @ keys.T
            identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            solved = torch.linalg.solve(
                gram + regularizer * identity,
                value_differences,
            )
            shift_key_value = keys.T @ solved
            weight = target.module.weight
            if tuple(shift_key_value.shape) == tuple(weight.shape):
                shift = shift_key_value
            elif tuple(shift_key_value.T.shape) == tuple(weight.shape):
                shift = shift_key_value.T
            else:
                raise RuntimeError(
                    f"MALMEN shift {tuple(shift_key_value.shape)} is incompatible "
                    f"with {target.parameter_name} {tuple(weight.shape)}"
                )
            shifts[target.parameter_name] = shift.to(weight.device, weight.dtype)
        return shifts

    def _apply_parameter_shifts(self, shifts: Mapping[str, torch.Tensor]) -> None:
        parameters = dict(self.model.named_parameters())
        with torch.no_grad():
            for name, shift in shifts.items():
                parameters[name].add_(shift)

    # ------------------------------------------------------------------
    # Compact meta-training path
    # ------------------------------------------------------------------

    def train_auxiliary_network(
        self,
        train_requests: List[EditRequest],
        num_epochs: int = 3,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Train the native compact auxiliary network on post-edit losses.

        The base model is not mutated.  The learned normal-equation shifts are
        installed through ``torch.func.functional_call`` and optimized against
        target/rephrase NLL plus optional locality KL and update-norm penalty.
        This is a representative compact objective, not EasyEdit's full custom
        backward/update schedule.
        """

        if not train_requests:
            raise ValueError("MALMEN training requires at least one request")
        self._init_auxiliary_networks()
        batch_size = batch_size or self.batch_size
        optimizer = torch.optim.Adam(
            self.auxiliary_networks.parameters(), lr=lr or self.meta_lr
        )
        history: list[dict[str, float]] = []
        was_training = self.model.training
        parameter_requires_grad = [
            parameter.requires_grad for parameter in self.model.parameters()
        ]
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        try:
            for epoch in range(num_epochs):
                total_loss = 0.0
                total_edit = 0.0
                total_locality = 0.0
                n_batches = 0
                for start in range(0, len(train_requests), batch_size):
                    batch = train_requests[start : start + batch_size]
                    optimizer.zero_grad()
                    caches = self._cache_requests(batch, update_normalizer=True)
                    shifts = self._predict_parameter_shifts(caches)
                    virtual = {
                        name: dict(self.model.named_parameters())[name] + shift
                        for name, shift in shifts.items()
                    }

                    edit_losses = [
                        self._virtual_edit_loss(request, virtual) for request in batch
                    ]
                    locality_losses = [
                        self._virtual_locality_loss(request, virtual)
                        for request in batch
                    ]
                    edit_loss = torch.stack(edit_losses).mean()
                    locality_loss = torch.stack(locality_losses).mean()
                    update_norm = torch.stack(
                        [shift.float().pow(2).mean() for shift in shifts.values()]
                    ).mean()
                    loss = (
                        edit_loss
                        + self.locality_weight * locality_loss
                        + self.update_norm_weight * update_norm
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.auxiliary_networks.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    total_loss += float(loss.detach().item())
                    total_edit += float(edit_loss.detach().item())
                    total_locality += float(locality_loss.detach().item())
                    n_batches += 1

                record = {
                    "epoch": float(epoch + 1),
                    "loss": total_loss / max(n_batches, 1),
                    "edit_nll": total_edit / max(n_batches, 1),
                    "locality_kl": total_locality / max(n_batches, 1),
                }
                history.append(record)
                logger.info(
                    "Compact MALMEN epoch %d/%d loss=%.4f "
                    "edit_nll=%.4f locality_kl=%.4f",
                    epoch + 1,
                    num_epochs,
                    record["loss"],
                    record["edit_nll"],
                    record["locality_kl"],
                )
        finally:
            for parameter, required in zip(
                self.model.parameters(), parameter_requires_grad
            ):
                parameter.requires_grad_(required)
            self.model.train(was_training)

        self._pretrained_loaded = True
        return history

    def _virtual_edit_loss(
        self, request: EditRequest, virtual: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        prompt = (
            request.rephrase_prompts[0] if request.rephrase_prompts else request.prompt
        )
        outputs = functional_call(
            self.model,
            dict(virtual),
            args=(),
            kwargs=self._build_edit_batch(request, prompt=prompt),
            strict=False,
        )
        return outputs.loss

    def _iter_locality_pairs(self, request: EditRequest):
        raw = request.locality_inputs or {}
        groups = raw if isinstance(raw, dict) else {"locality": raw}
        count = 0
        for payload in groups.values():
            if isinstance(payload, dict):
                payload = [payload]
            for item in payload or []:
                if not isinstance(item, dict):
                    continue
                prompt = item.get("prompt")
                target = item.get(
                    "ground_truth", item.get("expected", item.get("answer"))
                )
                if prompt and target:
                    yield str(prompt), str(target)
                    count += 1
                    if count >= self.max_locality_samples:
                        return

    def _virtual_locality_loss(
        self, request: EditRequest, virtual: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        losses = []
        for prompt, target in self._iter_locality_pairs(request):
            locality_request = EditRequest(
                prompt=prompt,
                subject=prompt,
                target_new=target,
            )
            batch = self._build_edit_batch(locality_request)
            labels = batch.pop("labels")
            with torch.no_grad():
                base_logits = self.model(**batch).logits
            edited_logits = functional_call(
                self.model,
                dict(virtual),
                args=(),
                kwargs=batch,
                strict=False,
            ).logits
            mask = labels.ne(-100)
            if mask.any():
                base_log_probs = torch.log_softmax(base_logits[mask], dim=-1)
                edited_log_probs = torch.log_softmax(edited_logits[mask], dim=-1)
                losses.append(
                    F.kl_div(
                        edited_log_probs,
                        base_log_probs,
                        log_target=True,
                        reduction="batchmean",
                    )
                )
        if losses:
            return torch.stack(losses).mean()
        # Differentiable zero tied to the auxiliary-network update.
        return sum(value.sum() for value in virtual.values()) * 0.0

    def _get_layer_module(self, model: nn.Module, layer_idx: int):
        for attr_name in ("model.layers", "transformer.h", "gpt_neox.layers"):
            try:
                return self._get_module_by_name(model, attr_name)[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None
