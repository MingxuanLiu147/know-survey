"""多模态 unlearning 训练基类（sidecar，不修改现有 trainer 体系）。

提供 Accelerator 驱动的训练循环，子类只需覆写 compute_loss()。
复用上游 MMUnlearner 的训练模式：AdamW + linear scheduler + 梯度裁剪。
"""

import logging
import math
import os
from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_scheduler

logger = logging.getLogger(__name__)


def _enable_non_reentrant_gradient_checkpointing(model) -> bool:
    """Enable checkpointing without requiring gradients on model inputs.

    Recent Transformers versions accept ``gradient_checkpointing_kwargs``.
    Older model implementations expose only the no-argument method, so they
    fall back to their legacy behavior instead of failing configuration checks.
    """

    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        return False
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        enable()
        logger.warning(
            "Model does not accept non-reentrant gradient-checkpointing kwargs; "
            "using its legacy checkpointing implementation"
        )
    return True


class MMUnlearnBase(ABC):
    """多模态 unlearning 训练基类。

    子类需要实现 compute_loss(model, batch) -> loss。
    基类负责：Accelerator 初始化、训练循环、梯度裁剪、模型保存。
    """

    def __init__(
        self,
        model,
        processor,
        forget_loader: DataLoader,
        retain_loader: DataLoader = None,
        *,
        lr: float = 5e-4,
        num_epochs: int = 5,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        max_steps: int = -1,
        save_dir: str = "./saves/mm_unlearn",
        grad_mask: dict = None,
        scheduler_steps_per_epoch: int | None = None,
    ):
        self.processor = processor
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_steps = int(max_steps)
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(
                "max_steps must be -1 (epoch budget) or a positive integer"
            )
        self.save_dir = save_dir
        self.grad_mask = grad_mask
        self.scheduler_steps_per_epoch = (
            int(scheduler_steps_per_epoch)
            if scheduler_steps_per_epoch is not None
            else len(forget_loader)
        )
        if self.scheduler_steps_per_epoch <= 0:
            raise ValueError("scheduler_steps_per_epoch must be positive")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        if _enable_non_reentrant_gradient_checkpointing(model):
            if hasattr(model, "config"):
                model.config.use_cache = False
            logger.info("Gradient checkpointing enabled")

        self.optimizer = AdamW(model.parameters(), lr=lr)
        prepare_args = [model, self.optimizer, forget_loader]
        if retain_loader is not None:
            prepare_args.append(retain_loader)
            (
                self.model,
                self.optimizer,
                self.forget_loader,
                self.retain_loader,
            ) = self.accelerator.prepare(*prepare_args)
        else:
            (
                self.model,
                self.optimizer,
                self.forget_loader,
            ) = self.accelerator.prepare(*prepare_args)

        total_steps = self._optimizer_steps_per_epoch() * num_epochs
        if self.max_steps > 0:
            total_steps = min(total_steps, self.max_steps)
        # Use the prepared loader lengths (per process), and advance exactly once
        # per optimizer update rather than once per microbatch or per GPU.
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _optimizer_steps_per_epoch(self) -> int:
        return math.ceil(len(self.forget_loader) / self.gradient_accumulation_steps)

    def _budget_reached(self, optimizer_steps: int) -> bool:
        return self.max_steps > 0 and optimizer_steps >= self.max_steps

    def _accumulate(self, batch_index: int, batch_count: int):
        """Accumulate against the driving loader, including its final partial batch.

        A cycling retain loader must not decide when a paired update is flushed.
        There is one ordinary optimizer call per microbatch and no extra tail step.
        """
        self.accelerator.sync_gradients = (
            batch_index + 1
        ) % self.gradient_accumulation_steps == 0 or batch_index + 1 == batch_count
        return (
            nullcontext()
            if self.accelerator.sync_gradients
            else self.accelerator.no_sync(self.model)
        )

    def _step_optimizer(self) -> bool:
        self.optimizer.step()
        stepped = (
            self.accelerator.sync_gradients
            and not self.accelerator.optimizer_step_was_skipped
        )
        if stepped:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return stepped

    @abstractmethod
    def compute_loss(self, model, batch) -> torch.Tensor:
        """子类实现具体的 unlearning 损失计算。"""
        ...

    def _apply_grad_mask(self):
        """在 backward 之后、optimizer.step 之前，对梯度应用掩码（用于 MMUnlearner）。"""
        if self.grad_mask is None:
            return
        for name, p in self.model.named_parameters():
            if p.grad is not None and name in self.grad_mask:
                p.grad *= self.grad_mask[name].to(p.grad.device)

    def train(self):
        if len(self.forget_loader) == 0:
            raise ValueError("Multimodal forget loader is empty")

        epoch_losses = []
        optimizer_steps = 0
        last_grad_norm = None
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(
                self.forget_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for step, batch in enumerate(progress):
                with self._accumulate(step, len(self.forget_loader)):
                    loss = self.compute_loss(self.model, batch)
                    self.accelerator.backward(loss)
                    self._apply_grad_mask()
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        last_grad_norm = float(grad_norm)
                    optimizer_steps += int(self._step_optimizer())

                total_loss += loss.detach().item()
                progress.set_postfix(loss=total_loss / (step + 1))
                if self._budget_reached(optimizer_steps):
                    break

            avg_loss = total_loss / (step + 1)
            epoch_losses.append(avg_loss)
            logger.info(
                "Epoch %d/%d - avg loss: %.4f", epoch + 1, self.num_epochs, avg_loss
            )
            if self._budget_reached(optimizer_steps):
                break

        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        return {
            "epochs": len(epoch_losses),
            "max_steps": self.max_steps,
            "optimizer_steps": optimizer_steps,
            "epoch_losses": epoch_losses,
            "last_grad_norm": last_grad_norm,
            "peak_memory_mb": peak_memory_mb,
        }

    def save_model(self, output_dir: str = None):
        save_path = output_dir or self.save_dir
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrapped, PeftModel):
            unwrapped = unwrapped.merge_and_unload()
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            unwrapped.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            logger.info("Model saved to %s", save_path)
