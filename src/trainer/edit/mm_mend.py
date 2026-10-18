"""
MM-MEND: Multimodal MEND Knowledge Editing
===========================================

Extends MENDEditor with multimodal capabilities.  The gradient
decomposition operates on the language model layers of the MLLM; the
multimodal processor is used to tokenize prompt + image + target so that
the loss gradient captures visual context.

Reference:
  - MEND: https://arxiv.org/abs/2110.11309
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging

import torch

from trainer.edit.base import EditRequest
from trainer.edit.mend import MENDEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMMENDEditor(MENDEditor, MMEditMixin):
    """Multimodal MEND editor.

    Overrides ``_compute_edit_gradient`` to use multimodal tokenization
    when the edit request carries an image.
    """

    def __init__(self, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)

    def _compute_edit_gradient(self, request: EditRequest):
        """Compute the gradient used by the MEND edit network.

        When an image is present, the multimodal forward pass is used so
        that visual information flows into the gradient.
        """
        model = self.model
        device = next(model.parameters()).device

        if request.image is not None and self.processor is not None:
            tokens = self.mm_tokenize(
                request.prompt,
                request.image,
                request.target_new,
                device=device,
            )
            outputs = self.mm_forward(model, tokens)
        else:
            tokenizer = self.tokenizer
            full_text = request.prompt + " " + request.target_new
            prompt_ids = tokenizer(
                request.prompt,
                return_tensors="pt",
            )["input_ids"].to(device)
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

            labels = full_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100

            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                labels=labels,
            )

        loss = outputs.loss
        loss.backward()

        target_layer = self._get_target_layer()
        grad = target_layer.weight.grad
        if grad is None:
            raise RuntimeError(
                "No gradient computed for the target layer. "
                "Ensure the target layer requires grad."
            )

        return grad.detach().clone()

    def _get_target_layer(self):
        """Return the nn.Module whose weight gradient drives the edit."""
        layer_indices = self._resolve_layer_indices()
        layer_idx = layer_indices[0]
        layers = self._get_layers_container(self.model)
        block = layers[layer_idx]

        for name in ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"]:
            parts = name.split(".")
            mod = block
            try:
                for part in parts:
                    mod = getattr(mod, part)
                return mod
            except AttributeError:
                continue
        raise ValueError("Cannot find MLP output projection in the target block")
