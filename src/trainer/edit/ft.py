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
        *args,
        **kwargs,
    ):
        layers = layers or [4, 5, 6, 7, 8]
        super().__init__(layers=layers, *args, **kwargs)
        self.ft_lr = ft_lr
        self.num_steps = num_steps
        self.weight_decay = weight_decay
        self.edit_layers_only = edit_layers_only

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

        return results

    def _trainable_parameters(self, layer_indices: List[int]) -> List[nn.Parameter]:
        model = self.model
        if not self.edit_layers_only:
            return [p for p in model.parameters() if p.requires_grad]

        params: List[nn.Parameter] = []
        for layer_idx in layer_indices:
            layer = self._get_layer_module(model, layer_idx)
            if layer is None:
                continue
            for p in layer.parameters():
                p.requires_grad = True
                params.append(p)
        if not params:
            # Fallback: train all parameters that already require grad
            params = [p for p in model.parameters() if p.requires_grad]
        return params

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
        texts = [f"{r.prompt} {r.target_new}" for r in requests]
        prompt_lens = []
        for r in requests:
            prompt_ids = tokenizer(r.prompt, add_special_tokens=True)["input_ids"]
            prompt_lens.append(len(prompt_ids))

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._safe_max_length(),
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            # Only supervise target tokens
            labels[i, : min(plen, labels.shape[1])] = -100
            labels[i, attention_mask[i] == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
        return min(model_limit, tokenizer_limit)

    def _finetune(self, requests: List[EditRequest], layer_indices: List[int]) -> None:
        model = self.model
        was_training = model.training
        model.train()

        # Freeze everything first when editing layers only
        if self.edit_layers_only:
            for p in model.parameters():
                p.requires_grad = False

        params = self._trainable_parameters(layer_indices)
        optimizer = torch.optim.AdamW(
            params, lr=self.ft_lr, weight_decay=self.weight_decay
        )
        batch = self._build_batch(requests)

        for step in range(self.num_steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            if step % 5 == 0 or step == self.num_steps - 1:
                logger.debug(
                    "FT step %d/%d loss=%.4f", step + 1, self.num_steps, loss.item()
                )

        model.train(was_training)
        logger.info(
            "FT edit finished: %d requests, %d steps, layers=%s",
            len(requests),
            self.num_steps,
            layer_indices,
        )
