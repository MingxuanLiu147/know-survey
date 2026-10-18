"""
MEND knowledge editor
=====================

Model Editor Networks with Gradient Decomposition (compact native path).

Paper: Fast Model Editing at Scale
https://arxiv.org/abs/2110.11309

This implementation provides the essential learned-gradient editing path:
1. A dimension-correct low-rank edit network (no reshape bugs)
2. save/load of the edit network checkpoint
3. Meta-training against post-update edit NLL and locality KL

When ``require_pretrained=True`` (default), ``edit()`` raises unless a
checkpoint has been loaded via ``edit_network_path`` or ``load_edit_network``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

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

_LEGACY_CHECKPOINT_FORMAT = "know_surgery_compact_mend_v1"
_CHECKPOINT_FORMAT = "know_surgery_compact_mend_v2"


class _MENDEditNetwork(nn.Module):
    """Maps decomposed gradient factors to a low-rank weight update."""

    def __init__(
        self,
        grad_rank: int,
        n_hidden: int,
        out_features: int,
        in_features: int,
    ):
        super().__init__()
        self.grad_rank = grad_rank
        self.out_features = out_features
        self.in_features = in_features
        self.backbone = nn.Sequential(
            nn.Linear(grad_rank * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.head_u = nn.Linear(n_hidden, out_features)
        self.head_v = nn.Linear(n_hidden, in_features)

    def forward(self, edit_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(edit_input)
        return self.head_u(h), self.head_v(h)


class MENDEditor(EditTrainer):
    """MEND-style editor with dimension-safe low-rank updates."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        edit_lr: float = 1e-4,
        n_hidden: int = 128,
        rank: int = 64,
        edit_network_path: Optional[str] = None,
        require_pretrained: bool = True,
        locality_weight: float = 1.0,
        update_norm_weight: float = 1e-4,
        max_locality_samples: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__(layers=layers, *args, **kwargs)

        self.edit_lr = edit_lr
        self.n_hidden = n_hidden
        self.rank = rank
        self.edit_network_path = edit_network_path
        self.require_pretrained = require_pretrained
        self.locality_weight = locality_weight
        self.update_norm_weight = update_norm_weight
        self.max_locality_samples = max_locality_samples

        self.edit_network: Optional[_MENDEditNetwork] = None
        self._initialized = False
        self._pretrained_loaded = False
        self._target_weight_shape: Optional[torch.Size] = None
        self._target_parameter_name: Optional[str] = None

        if edit_network_path:
            self.load_edit_network(edit_network_path)

    def _resolve_target_proj(self) -> Tuple[int, nn.Module, torch.nn.Parameter]:
        model = self.model
        layer_idx = self._resolve_layer_indices(self.layers)[0]
        layer = self._get_layer_module(model, layer_idx)
        if layer is None:
            raise ValueError(f"Cannot find layer {layer_idx} in model")

        for proj_name in ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"]:
            try:
                proj = self._get_module_by_name(layer, proj_name)
                return layer_idx, proj, proj.weight
            except AttributeError:
                continue
        raise ValueError(f"Cannot find MLP projection in layer {layer_idx}")

    def _init_edit_network(self, force: bool = False) -> None:
        if self._initialized and not force:
            return

        layer_idx, proj, weight = self._resolve_target_proj()
        out_features, in_features = weight.shape
        device = weight.device

        self._target_weight_shape = weight.shape
        self._target_parameter_name = next(
            name
            for name, parameter in self.model.named_parameters()
            if parameter is weight
        )
        self.edit_network = _MENDEditNetwork(
            grad_rank=self.rank,
            n_hidden=self.n_hidden,
            out_features=out_features,
            in_features=in_features,
        ).to(device)

        self._initialized = True
        logger.info(
            "MEND edit network init: layer=%d weight=%s rank=%d n_hidden=%d",
            layer_idx,
            tuple(weight.shape),
            self.rank,
            self.n_hidden,
        )

    def save_edit_network(self, path: str) -> None:
        self._init_edit_network()
        assert self.edit_network is not None
        if not self._pretrained_loaded:
            raise RuntimeError(
                "Refusing to save an untrained MEND edit network. "
                "Run train_edit_network first."
            )
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _CHECKPOINT_FORMAT,
            "editor_type": self._checkpoint_editor_type(),
            "editor_metadata": self._checkpoint_editor_metadata(),
            "base_model_identity": build_base_model_identity(self.model),
            "state_dict": self.edit_network.state_dict(),
            "rank": self.rank,
            "n_hidden": self.n_hidden,
            "out_features": self.edit_network.out_features,
            "in_features": self.edit_network.in_features,
            "edit_lr": self.edit_lr,
            "target_parameter_name": self._target_parameter_name,
            "locality_weight": self.locality_weight,
            "update_norm_weight": self.update_norm_weight,
        }
        torch.save(payload, path_obj)
        self.edit_network_path = str(path_obj)
        logger.info("Saved MEND edit network to %s", path_obj)

    def _checkpoint_editor_type(self) -> str:
        """Identity used to prevent cross-loading incompatible learned editors."""

        return "mend"

    def _checkpoint_editor_metadata(self) -> Dict[str, Any]:
        """Subclass-specific preprocessing contract stored beside the weights."""

        return {}

    def _validate_checkpoint_identity(self, payload: Mapping[str, Any]) -> None:
        checkpoint_format = payload.get("format")
        if checkpoint_format in (None, _LEGACY_CHECKPOINT_FORMAT):
            raise ValueError(
                "Legacy compact MEND checkpoint is missing required "
                "base_model_identity and is incompatible; retrain the edit network "
                "against the exact base-model weights."
            )
        if checkpoint_format != _CHECKPOINT_FORMAT:
            raise ValueError(
                f"Unsupported MEND checkpoint format {checkpoint_format!r}; "
                "EasyEdit/official archives are not compatible with the compact "
                "Know-Surgery edit network."
            )
        if "editor_type" not in payload or not isinstance(
            payload.get("editor_metadata"), Mapping
        ):
            raise ValueError(
                "MEND v2 checkpoint is missing required editor_type/editor_metadata"
            )
        actual_type = payload["editor_type"]
        expected_type = self._checkpoint_editor_type()
        if actual_type != expected_type:
            raise ValueError(
                f"Checkpoint editor_type={actual_type!r} cannot be loaded by "
                f"{expected_type!r}. Train a dedicated checkpoint for this editor."
            )

    def load_edit_network(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"MEND edit network checkpoint not found: {path}")

        payload = torch.load(path_obj, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("MEND checkpoint must contain a metadata mapping")
        self._validate_checkpoint_identity(payload)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("MEND checkpoint is missing a state_dict mapping")
        required = {
            "rank",
            "n_hidden",
            "out_features",
            "in_features",
            "edit_lr",
            "target_parameter_name",
            "locality_weight",
            "update_norm_weight",
            "base_model_identity",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                f"MEND v2 checkpoint is missing required metadata: {missing}"
            )

        integer_fields = ("rank", "n_hidden", "out_features", "in_features")
        if any(
            not isinstance(payload[field], int) or isinstance(payload[field], bool)
            for field in integer_fields
        ):
            raise ValueError("MEND v2 integer metadata must contain exact integers")
        rank = payload["rank"]
        n_hidden = payload["n_hidden"]
        if not 1 <= rank <= 16_384 or not 1 <= n_hidden <= 16_384:
            raise ValueError("MEND checkpoint rank/n_hidden are outside safe bounds")

        float_fields = ("edit_lr", "locality_weight", "update_norm_weight")
        try:
            edit_lr, locality_weight, update_norm_weight = (
                float(payload[field]) for field in float_fields
            )
        except (TypeError, ValueError) as error:
            raise ValueError("MEND v2 learning-rate metadata is invalid") from error
        if (
            not all(
                math.isfinite(value)
                for value in (edit_lr, locality_weight, update_norm_weight)
            )
            or edit_lr <= 0
            or locality_weight < 0
            or update_norm_weight < 0
        ):
            raise ValueError("MEND v2 learning-rate metadata is outside safe bounds")

        _, _, weight = self._resolve_target_proj()
        expected = tuple(int(dimension) for dimension in weight.shape)
        saved = (payload["out_features"], payload["in_features"])
        if saved != expected:
            raise ValueError(
                f"MEND checkpoint weight shape {saved} does not match model "
                f"projection shape {expected}. Retrain for this model."
            )
        target_parameter_name = next(
            name
            for name, parameter in self.model.named_parameters()
            if parameter is weight
        )
        saved_name = payload["target_parameter_name"]
        if not isinstance(saved_name, str) or saved_name != target_parameter_name:
            raise ValueError(
                f"MEND checkpoint targets {saved_name!r}, but the current model "
                f"resolved {target_parameter_name!r}."
            )

        expected_shapes = {
            "backbone.0.weight": (n_hidden, rank * 2),
            "backbone.0.bias": (n_hidden,),
            "backbone.2.weight": (n_hidden, n_hidden),
            "backbone.2.bias": (n_hidden,),
            "head_u.weight": (expected[0], n_hidden),
            "head_u.bias": (expected[0],),
            "head_v.weight": (expected[1], n_hidden),
            "head_v.bias": (expected[1],),
        }
        if set(state_dict) != set(expected_shapes) or any(
            not isinstance(state_dict[name], torch.Tensor)
            or tuple(state_dict[name].shape) != shape
            for name, shape in expected_shapes.items()
        ):
            raise ValueError(
                "MEND checkpoint state_dict keys/shapes do not match its metadata"
            )
        estimated_parameters = sum(
            math.prod(shape) for shape in expected_shapes.values()
        )
        if estimated_parameters > 50_000_000:
            raise ValueError("MEND checkpoint edit network exceeds safe size bounds")

        validate_base_model_identity(
            self.model,
            payload["base_model_identity"],
            checkpoint_label=self._checkpoint_editor_type().upper(),
        )

        edit_network = _MENDEditNetwork(
            grad_rank=rank,
            n_hidden=n_hidden,
            out_features=expected[0],
            in_features=expected[1],
        ).to(weight.device)
        edit_network.load_state_dict(state_dict, strict=True)

        self.rank = rank
        self.n_hidden = n_hidden
        self.edit_lr = edit_lr
        self.locality_weight = locality_weight
        self.update_norm_weight = update_norm_weight
        self.edit_network = edit_network
        self._target_weight_shape = weight.shape
        self._target_parameter_name = target_parameter_name
        self._initialized = True
        self._pretrained_loaded = True
        self.edit_network_path = str(path_obj)
        logger.info("Loaded MEND edit network from %s", path_obj)

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        if self.require_pretrained and not self._pretrained_loaded:
            raise RuntimeError(
                "MENDEditor.require_pretrained=True but no edit network checkpoint "
                "was loaded. Train with scripts/train_mend_edit_network.py and set "
                "trainer.method_args.edit_network_path. Setting "
                "require_pretrained=false is experimental and is never verified "
                "checkpoint evidence."
            )
        self._init_edit_network()

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "checkpoint_loaded": self._pretrained_loaded,
            "implementation": "compact_native_mend",
        }

        for request in requests:
            try:
                self._apply_mend_edit(request)
                results["edited_count"] += 1
            except Exception as e:
                logger.error("MEND edit failed: %s", e, exc_info=True)
                results["success"] = False

        return results

    def _apply_mend_edit(self, request: EditRequest) -> None:
        grad = self._compute_edit_gradient(request)
        u, v = self._decompose_gradient(grad)
        delta_u, delta_v = self._compute_update_from_gradient(u, v)
        self._apply_delta_update(delta_u, delta_v)
        logger.info("MEND edit applied: %s -> %s", request.subject, request.target_new)

    def _compute_edit_gradient(self, request: EditRequest) -> torch.Tensor:
        model = self.model
        _, _, weight = self._resolve_target_proj()
        batch = self._build_edit_batch(request)
        was_training = model.training
        required_grad = weight.requires_grad
        weight.requires_grad_(True)
        model.eval()
        try:
            outputs = model(**batch)
            grad = torch.autograd.grad(outputs.loss, weight, retain_graph=False)[0]
        finally:
            weight.requires_grad_(required_grad)
            model.train(was_training)
        return grad

    def _decompose_gradient(
        self, grad: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._target_weight_shape is not None
        out_f, in_f = self._target_weight_shape
        grad_2d = grad.reshape(out_f, in_f).float()
        rank = min(self.rank, out_f, in_f)
        if rank <= 0:
            device = grad.device
            return (
                torch.zeros(self.rank, device=device),
                torch.zeros(self.rank, device=device),
            )

        U, S, V = torch.svd_lowrank(grad_2d, q=rank)
        u = (U[:, :rank] * S[:rank].sqrt().unsqueeze(0)).mean(dim=0)
        v = (V[:, :rank] * S[:rank].sqrt().unsqueeze(0)).mean(dim=0)
        if rank < self.rank:
            u = F.pad(u, (0, self.rank - rank))
            v = F.pad(v, (0, self.rank - rank))
        return u, v

    def _compute_update_from_gradient(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.edit_network is not None
        device = next(self.edit_network.parameters()).device
        u_padded = F.pad(u, (0, max(0, self.rank - len(u))))[: self.rank].to(device)
        v_padded = F.pad(v, (0, max(0, self.rank - len(v))))[: self.rank].to(device)
        edit_input = torch.cat([u_padded, v_padded]).unsqueeze(0)
        delta_u, delta_v = self.edit_network(edit_input)
        return delta_u.squeeze(0), delta_v.squeeze(0)

    def _apply_delta_update(self, delta_u: torch.Tensor, delta_v: torch.Tensor) -> None:
        _, _, weight = self._resolve_target_proj()
        with torch.no_grad():
            # Linear: (out, in); GPT-2 Conv1D: often (in, out)
            delta_w = torch.outer(delta_u, delta_v).to(weight.dtype).to(weight.device)
            if delta_w.shape != weight.shape:
                if delta_w.T.shape == weight.shape:
                    delta_w = delta_w.T
                else:
                    raise RuntimeError(
                        f"MEND delta shape {tuple(delta_w.shape)} incompatible with "
                        f"weight {tuple(weight.shape)}"
                    )
            weight.add_(delta_w * self.edit_lr)

    def _get_layer_module(self, model: nn.Module, layer_idx: int):
        for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._get_module_by_name(model, attr_name)
                return layers[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None

    def train_edit_network(
        self,
        train_requests: List[EditRequest],
        num_epochs: int = 10,
        batch_size: int = 4,
        lr: Optional[float] = None,
    ) -> List[Dict[str, float]]:
        """Meta-train the gradient transform on post-edit NLL and locality.

        The base model is never mutated.  Each request produces a transformed
        low-rank update, which is installed with ``torch.func.functional_call``;
        gradients then flow from the *post-update* edit/locality losses into the
        edit network.  This is the essential MEND training objective rather
        than the previous zero-update norm proxy.
        """
        if not train_requests:
            raise ValueError("MEND training requires at least one edit request")
        self._init_edit_network()
        assert self.edit_network is not None

        optimizer = torch.optim.Adam(
            self.edit_network.parameters(), lr=lr or self.edit_lr
        )
        model = self.model
        was_training = model.training
        parameter_requires_grad = [
            parameter.requires_grad for parameter in model.parameters()
        ]
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        history: List[Dict[str, float]] = []
        try:
            for epoch in range(num_epochs):
                total_loss = 0.0
                total_edit = 0.0
                total_locality = 0.0
                n_batches = 0
                for i in range(0, len(train_requests), batch_size):
                    batch = train_requests[i : i + batch_size]
                    optimizer.zero_grad()
                    batch_loss = torch.zeros(
                        (), device=next(self.edit_network.parameters()).device
                    )
                    batch_edit = 0.0
                    batch_locality = 0.0

                    for request in batch:
                        grad = self._compute_edit_gradient(request)
                        u, v = self._decompose_gradient(grad)
                        delta_u, delta_v = self._compute_update_from_gradient(u, v)
                        delta_w = torch.outer(delta_u, delta_v)
                        edit_loss = self._virtual_edit_loss(request, delta_w)
                        locality_loss = self._virtual_locality_loss(request, delta_w)
                        norm_loss = delta_w.pow(2).mean()
                        request_loss = (
                            edit_loss
                            + self.locality_weight * locality_loss
                            + self.update_norm_weight * norm_loss
                        )
                        batch_loss = batch_loss + request_loss
                        batch_edit += float(edit_loss.detach().item())
                        batch_locality += float(locality_loss.detach().item())

                    batch_loss = batch_loss / max(len(batch), 1)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.edit_network.parameters(), 1.0)
                    optimizer.step()
                    total_loss += float(batch_loss.item())
                    total_edit += batch_edit / max(len(batch), 1)
                    total_locality += batch_locality / max(len(batch), 1)
                    n_batches += 1

                epoch_record = {
                    "epoch": float(epoch + 1),
                    "loss": total_loss / max(n_batches, 1),
                    "edit_nll": total_edit / max(n_batches, 1),
                    "locality_kl": total_locality / max(n_batches, 1),
                }
                history.append(epoch_record)
                logger.info(
                    "MEND epoch %d/%d loss=%.4f edit_nll=%.4f locality_kl=%.4f",
                    epoch + 1,
                    num_epochs,
                    epoch_record["loss"],
                    epoch_record["edit_nll"],
                    epoch_record["locality_kl"],
                )
        finally:
            for parameter, required in zip(model.parameters(), parameter_requires_grad):
                parameter.requires_grad_(required)
            model.train(was_training)

        self._pretrained_loaded = True
        return history

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

    def _build_edit_batch(self, request: EditRequest) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        prompt = str(request.prompt)
        full_text = f"{prompt} {request.target_new}"
        encoded = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._safe_max_length(),
        )
        prompt_ids = tokenizer(
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
            raise ValueError("MEND target produced no supervised tokens")
        batch["labels"] = labels
        return batch

    def _virtual_parameters(self, delta_w: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._target_parameter_name is None:
            self._init_edit_network()
        assert self._target_parameter_name is not None
        _, _, weight = self._resolve_target_proj()
        update = delta_w
        if update.shape != weight.shape:
            if update.T.shape == weight.shape:
                update = update.T
            else:
                raise RuntimeError(
                    f"MEND virtual update {tuple(update.shape)} is incompatible "
                    f"with target weight {tuple(weight.shape)}"
                )
        return {
            self._target_parameter_name: weight
            + self.edit_lr * update.to(device=weight.device, dtype=weight.dtype)
        }

    def _virtual_edit_loss(
        self, request: EditRequest, delta_w: torch.Tensor
    ) -> torch.Tensor:
        batch = self._build_edit_batch(request)
        outputs = functional_call(
            self.model,
            self._virtual_parameters(delta_w),
            args=(),
            kwargs=batch,
            strict=False,
        )
        return outputs.loss

    def _iter_locality_pairs(self, request: EditRequest):
        raw = request.locality_inputs or {}
        if isinstance(raw, list):
            groups = {"locality": raw}
        elif isinstance(raw, dict):
            groups = raw
        else:
            groups = {}
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
        self, request: EditRequest, delta_w: torch.Tensor
    ) -> torch.Tensor:
        losses = []
        virtual = self._virtual_parameters(delta_w)
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
                virtual,
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
        return delta_w.sum() * 0.0
