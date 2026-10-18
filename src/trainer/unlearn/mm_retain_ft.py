"""Retain-only multimodal fine-tuning for reference/oracle checkpoints."""

import torch

from trainer.unlearn.mm_base import MMUnlearnBase


class MMRetainFT(MMUnlearnBase):
    """Train only on the provided loader with standard CE loss.

    The data adapter maps the retain split to the primary loader when
    `data.mode=reference`, so this trainer can reuse the same sidecar entrypoint.
    """

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        return outputs.loss
