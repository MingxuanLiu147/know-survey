"""
MM-WISE: Multimodal WISE Knowledge Editing
===========================================

Extends WISEEditor with multimodal tokenization so the adapter receives
visual information during the fine-tuning loop.

Reference:
  - WISE: https://arxiv.org/abs/2405.14768
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from trainer.edit.base import EditRequest
from trainer.edit.wise import WISEEditor, _activation_distance
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMWISEEditor(WISEEditor, MMEditMixin):
    """Multimodal WISE editor.

    Overrides _apply_wise_edit to handle image inputs through the
    multimodal processor while reusing the WISEAdapter machinery.
    """

    def _resolve_target_layer(self, inner_params: str) -> str:
        """Resolve ``auto`` before the text-only WISE initializer rejects VLMs."""
        if inner_params == "auto":
            return self.configure_mm_target_layer()
        return super()._resolve_target_layer(inner_params)

    def __init__(self, processor=None, *args, **kwargs):
        kwargs.setdefault("inner_params", "auto")
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)
        if self.n_iter < 1:
            raise ValueError("MM-WISE n_iter must be at least 1")
        self.model.requires_grad_(False)
        self.configure_mm_target_layer()

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Apply image-grounded edits and surface the first invalid request."""
        return self._run_mm_edit_requests(requests, self._apply_wise_edit)

    def _apply_wise_edit(self, request: EditRequest):
        model = self.model
        device = next(model.parameters()).device
        adapter = self._get_adapter()

        tokens = self.mm_tokenize(
            request.prompt,
            request.image,
            request.target_new,
            device=device,
        )

        self._edit_history.append(
            {
                k: v.detach().cpu()
                for k, v in tokens.items()
                if isinstance(v, torch.Tensor)
            }
        )

        adapter.editing = True
        adapter.set_parameter_tunable()

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.generate_activation_mask()

        optimizer = torch.optim.SGD(
            [adapter.new_weight],
            lr=self.edit_lr,
            weight_decay=1e-5,
        )

        try:
            for i in range(self.n_iter):
                outputs = self.mm_forward(model, tokens)

                logits = outputs.logits
                shift_logits = logits[:1, :-1, :].contiguous()
                shift_labels = tokens["labels"][:1, 1:].contiguous()

                loss_fct = CrossEntropyLoss(reduction="none")
                per_tok = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view_as(shift_labels)
                label_mask = shift_labels.ne(-100)
                if not label_mask.any():
                    raise ValueError("MM-WISE received no supervised answer tokens")
                ft_loss = per_tok[label_mask].mean()

                act_loss = torch.tensor(0.0, device=device)
                orig_out = adapter.original_layer_output
                new_out = adapter.new_weight_layer_output
                if orig_out is not None and new_out is not None:
                    in_s = _activation_distance(orig_out[:1], new_out[:1])
                    if orig_out.shape[0] > 1:
                        out_s = _activation_distance(
                            orig_out[1:],
                            new_out[1:],
                        )
                    else:
                        out_s = torch.tensor(0.0, device=device)
                    act_loss = (
                        F.relu(out_s - in_s + self.gamma)
                        + F.relu(out_s - self.alpha)
                        + F.relu(self.beta - in_s)
                    )

                loss = ft_loss + act_loss

                if i == self.n_iter - 1:
                    adapter.save_editing_activation()

                optimizer.zero_grad()
                loss.backward()
                adapter.mask_new_weight_gradient()
                optimizer.step()

                if self.norm_constraint is not None:
                    with torch.no_grad():
                        orig_w = adapter._original_weight_cpu.to(device)
                        adapter.new_weight.clamp_(
                            orig_w - self.norm_constraint,
                            orig_w + self.norm_constraint,
                        )
        finally:
            adapter.editing = False

        adapter.editing_total_cnt += 1

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.save_weight()

        if adapter.editing_total_cnt % self.merge_freq == 0:
            adapter.merge_weight()

        logger.info(
            "MM-WISE edit: '%s' -> '%s' (ft_loss=%.4f)",
            request.prompt,
            request.target_new,
            ft_loss.item(),
        )
