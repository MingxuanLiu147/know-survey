"""
FT knowledge editor
===================

Short fine-tuning baseline for knowledge editing: minimise NLL of
``prompt + target_new`` for a small number of gradient steps.

Useful as a simple, trustworthy baseline next to ROME/MEMIT/IKE.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from trainer.edit.base import EditRequest, EditTrainer
from data.utils import tokenize_causal_pair

logger = logging.getLogger(__name__)


class FTEditor(EditTrainer):
    """Prompt+target short fine-tuning editor."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        ft_lr: float = 5e-5,
        num_steps: int = 25,
        weight_decay: float = 0.0,
        edit_layers_only: bool = True,
        max_length: Optional[int] = None,
        *args,
        **kwargs,
    ):
        layers = [4, 5, 6, 7, 8] if layers is None else list(layers)
        super().__init__(layers=layers, *args, **kwargs)
        self.ft_lr = ft_lr
        self.num_steps = num_steps
        self.weight_decay = weight_decay
        self.edit_layers_only = edit_layers_only
        self.max_length = max_length

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

        try:
            layer_indices = self._resolve_layer_indices(self.layers)
            results["layers_edited"] = layer_indices
            self._finetune(requests, layer_indices)
            results["edited_count"] = len(requests)
        except Exception as e:
            logger.error("FT edit failed: %s", e, exc_info=True)
            results["success"] = False
            results["error"] = f"{type(e).__name__}: {e}"

        return results

    def _trainable_parameters(self, layer_indices: List[int]) -> List[nn.Parameter]:
        model = self.model
        if not self.edit_layers_only:
            return [p for p in model.parameters() if p.requires_grad]

        params: List[nn.Parameter] = []
        for layer_idx in layer_indices:
            layer = self._get_layer_module(model, layer_idx)
            if layer is None:
                raise ValueError(f"Cannot find FT layer {layer_idx} in this model.")
            for p in layer.parameters():
                p.requires_grad = True
                params.append(p)
        if not params:
            raise ValueError("FT layer selection contains no trainable parameters.")
        return list(dict.fromkeys(params))

    def _get_layer_module(self, model: nn.Module, layer_idx: int):
        for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._get_module_by_name(model, attr_name)
                return layers[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None

    def _build_batch(self, requests: List[EditRequest]) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizer
        device = self._input_device()
        if not requests:
            raise ValueError("FT requires at least one edit request.")
        rows = []
        for request in requests:
            if not request.target_new or not request.target_new.strip():
                raise ValueError("FT requires a non-empty target answer.")
            separator = "" if request.prompt.endswith((" ", "\n", "\t")) else " "
            prompt_ids, target_ids = tokenize_causal_pair(
                tokenizer, request.prompt, separator + request.target_new
            )
            if not prompt_ids:
                raise ValueError("FT requires at least one prompt token.")
            row = (prompt_ids + target_ids)[: self._safe_max_length()]
            prompt_length = len(prompt_ids)
            if len(row) <= prompt_length:
                raise ValueError(
                    "FT edit has no target tokens after the complete prompt."
                )
            if len(prompt_ids) + len(target_ids) > len(row):
                raise ValueError(
                    "FT complete target exceeds the sequence limit; increase the "
                    "limit or explicitly filter the request instead of truncating it."
                )
            rows.append((row, prompt_length))

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("FT batching requires a pad or EOS token.")
        width = max(len(row) for row, _ in rows)
        input_ids = torch.full(
            (len(rows), width), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for i, (row, prompt_length) in enumerate(rows):
            offset = width - len(row) if tokenizer.padding_side == "left" else 0
            end = offset + len(row)
            input_ids[i, offset:end] = torch.tensor(row, device=device)
            attention_mask[i, offset:end] = 1
            labels[i, offset + prompt_length : end] = input_ids[
                i, offset + prompt_length : end
            ]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": attention_mask.cumsum(dim=-1).sub(1).clamp_min(0),
            "labels": labels,
        }

    def _safe_max_length(self) -> int:
        model_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", model_limit)
        try:
            tokenizer_limit = int(tokenizer_limit)
        except (TypeError, ValueError, OverflowError):
            tokenizer_limit = model_limit
        # Hugging Face uses a ~1e30 sentinel when no tokenizer limit is known.
        if tokenizer_limit <= 0 or tokenizer_limit > 10_000_000:
            tokenizer_limit = model_limit
        limits = [model_limit, tokenizer_limit]
        configured_limit = getattr(self, "max_length", None)
        if configured_limit is not None:
            if (
                isinstance(configured_limit, bool)
                or not isinstance(configured_limit, int)
                or configured_limit < 2
            ):
                raise ValueError("FT max_length must be at least 2 tokens.")
            limits.append(configured_limit)
        return min(limits)

    def _finetune(self, requests: List[EditRequest], layer_indices: List[int]) -> None:
        model = self.model
        if self.num_steps < 1:
            raise ValueError("FT num_steps must be positive.")
        batch = self._build_batch(requests)
        training_modes = [(module, module.training) for module in model.modules()]
        grad_flags = [(p, p.requires_grad) for p in model.parameters()]
        params = []
        original_weights = []
        try:
            model.train()
            if self.edit_layers_only:
                for p in model.parameters():
                    p.requires_grad = False
            params = self._trainable_parameters(layer_indices)
            # Keep rollback copies off the editing GPU. A failed step must not
            # leave partially tuned weights in a single or sequential run.
            original_weights = [p.detach().cpu().clone() for p in params]
            optimizer = torch.optim.AdamW(
                params, lr=self.ft_lr, weight_decay=self.weight_decay
            )
            for step in range(self.num_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**batch).loss
                if not torch.isfinite(loss).all():
                    raise FloatingPointError(f"FT loss is non-finite at step {step}.")
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in params
                ):
                    raise FloatingPointError(
                        f"FT gradient is non-finite at step {step}."
                    )
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in params):
                    raise FloatingPointError(
                        f"FT parameter is non-finite after optimizer step {step}."
                    )
                if step % 5 == 0 or step == self.num_steps - 1:
                    logger.debug(
                        "FT step %d/%d loss=%.4f", step + 1, self.num_steps, loss.item()
                    )
        except Exception:
            with torch.no_grad():
                for parameter, original in zip(params, original_weights):
                    parameter.copy_(original)
            raise
        finally:
            for parameter in params:
                parameter.grad = None
            for parameter, requires_grad in grad_flags:
                parameter.requires_grad_(requires_grad)
            for module, training in training_modes:
                module.training = training
        logger.info(
            "FT edit finished: %d requests, %d steps, layers=%s",
            len(requests),
            self.num_steps,
            layer_indices,
        )
