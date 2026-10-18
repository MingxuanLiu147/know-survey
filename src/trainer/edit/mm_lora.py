"""Native LoRA editor for image-text knowledge editing."""

from __future__ import annotations

from typing import Any

import torch

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.mm_mixin import MMEditMixin


class MMLoRAEditor(EditTrainer, MMEditMixin):
    """Optimize an already-attached PEFT LoRA adapter on MM edit requests."""

    def __init__(
        self,
        processor=None,
        num_steps: int = 5,
        edit_lr: float = 5e-4,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)
        if num_steps < 1:
            raise ValueError("MM-LoRA num_steps must be at least 1")
        self.num_steps = num_steps
        self.edit_lr = edit_lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

    def edit(
        self, requests: EditRequest | list[EditRequest], **kwargs
    ) -> dict[str, Any]:
        """Update only LoRA parameters; the base multimodal model stays frozen."""
        del kwargs
        requests = requests if isinstance(requests, list) else [requests]
        if not requests:
            raise ValueError("At least one multimodal edit request is required")
        for request in requests:
            self._validate_mm_request(request)

        trainable = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not getattr(self.model, "peft_config", None) or not trainable:
            raise RuntimeError(
                "MM-LoRA requires a PEFT LoRA model. Set model.lora.enabled=true."
            )

        tokens = self.mm_tokenize(
            [request.prompt for request in requests],
            [request.image for request in requests],
            [request.target_new for request in requests],
        )
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.edit_lr,
            weight_decay=self.weight_decay,
        )
        was_training = self.model.training
        final_loss = None
        try:
            self.model.train()
            for _ in range(self.num_steps):
                optimizer.zero_grad(set_to_none=True)
                loss = self.mm_forward(self.model, tokens).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"MM-LoRA produced non-finite loss: {loss}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
                optimizer.step()
                final_loss = float(loss.detach().item())
        finally:
            self.model.train(was_training)

        return {
            "success": True,
            "edited_count": len(requests),
            "metrics": {"final_loss": final_loss},
        }
