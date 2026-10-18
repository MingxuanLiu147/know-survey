"""
MM-GRACE: Multimodal GRACE Knowledge Editing
=============================================

Extends GRACEEditor with multimodal tokenization so the discrete key-value
codebook receives visual information during editing.

Reference:
  - GRACE: https://arxiv.org/abs/2211.11031
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

import torch

from trainer.edit.base import EditRequest
from trainer.edit.grace import GRACEEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMGRACEEditor(GRACEEditor, MMEditMixin):
    """Multimodal GRACE editor.

    Overrides the edit loop to tokenize inputs through the multimodal
    processor so that the model forward pass includes visual tokens.
    """

    def _resolve_target_layer(self, inner_params: str) -> str:
        """Resolve ``auto`` before the text-only GRACE initializer rejects VLMs."""
        if inner_params == "auto":
            return self.configure_mm_target_layer()
        return super()._resolve_target_layer(inner_params)

    def __init__(self, processor=None, *args, **kwargs):
        kwargs.setdefault("inner_params", "auto")
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)
        if self.n_iter < 1:
            raise ValueError("MM-GRACE n_iter must be at least 1")
        self.model.requires_grad_(False)
        self.configure_mm_target_layer()

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Apply image-grounded edits and surface the first invalid request."""
        return self._run_mm_edit_requests(requests, self._apply_grace_edit)

    def _apply_grace_edit(self, request: EditRequest):
        model = self.model
        device = next(model.parameters()).device
        adapter = self._get_adapter()

        tokens = self.mm_tokenize(
            request.prompt,
            request.image,
            request.target_new,
            device=device,
        )

        supervised = (tokens["labels"][0] != -100).nonzero(as_tuple=True)[0]
        if supervised.numel() == 0:
            raise ValueError("MM-GRACE received no supervised answer tokens")
        key_id = max(0, supervised[0].item() - 1)
        adapter.key_id = key_id
        adapter._training_edit = True
        adapter.edit_label = tokens["labels"]
        adapter.edit_id = self._edit_count

        losses: List[float] = []
        try:
            for i in range(self.n_iter):
                adapter.iter = i
                outputs = self.mm_forward(model, tokens)

                if i == 0:
                    optimizer = torch.optim.Adam(
                        self._get_grace_trainable_params(adapter),
                        lr=self.edit_lr,
                    )

                loss = outputs.loss

                if self.val_reg > 0 and adapter.has_keys:
                    loss = loss + self.val_reg * adapter.values.norm()

                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad(set_to_none=True)
                losses.append(loss.item())
        finally:
            adapter._training_edit = False

        self._edit_count += 1

        logger.info(
            "MM-GRACE edit: '%s' -> '%s' (loss=%.4f, codebook=%d)",
            request.prompt,
            request.target_new,
            losses[-1] if losses else float("inf"),
            adapter.codebook_size,
        )
