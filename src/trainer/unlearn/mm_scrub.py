"""CLEAR-native multimodal SCRUB prototype.

The implementation follows Appendix B of CLEAR (Dontsov et al., Findings of
ACL 2025, DOI 10.18653/v1/2025.findings-acl.1058): a frozen original model is
the teacher and the trainable model is the student.  The student minimizes

    alpha * KL(teacher || student)_retain
    - KL(teacher || student)_forget
    + gamma * CE_retain.

CLEAR's public code at commit ``bb14c967`` reverses the arguments passed to
``torch.nn.functional.kl_div`` and therefore computes ``KL(student ||
teacher)`` despite the paper's displayed formula.  The paper direction is the
default here.  ``kl_direction=student_to_teacher`` is retained as an explicit,
auditable upstream-code compatibility option.

This module is a prototype until a complete CLEAR run is recorded.  In
particular, the frozen teacher doubles checkpoint memory unless an external
offload strategy is supplied by the runtime.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)

KL_DIRECTIONS = ("teacher_to_student", "student_to_teacher")


def scrub_objective(
    forget_kl: torch.Tensor,
    retain_kl: torch.Tensor,
    retain_task_loss: torch.Tensor,
    *,
    forget_weight: float = 1.0,
    retain_kl_weight: float = 1.0,
    retain_task_weight: float = 1.0,
) -> torch.Tensor:
    """Return CLEAR Appendix-B SCRUB objective ``LR - LF + Ltask``."""

    return (
        retain_kl_weight * retain_kl
        - forget_weight * forget_kl
        + retain_task_weight * retain_task_loss
    )


def token_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    batch: dict,
    *,
    direction: str = "teacher_to_student",
    chunk_size: int = 16,
) -> torch.Tensor:
    """Compute token-mean KL while bounding temporary probability tensors.

    Softmax always spans the full vocabulary.  Chunking only limits how many
    token rows are materialized at once.  Padding is excluded through the
    attention mask (or, when unavailable, the non-ignored label mask).

    Args:
        student_logits: Trainable model logits with shape ``[..., vocab]``.
        teacher_logits: Frozen reference logits with the same shape.
        batch: Encoded multimodal batch containing an optional attention mask.
        direction: ``teacher_to_student`` for the paper formula, or
            ``student_to_teacher`` for CLEAR public-code compatibility.
        chunk_size: Maximum number of token rows per probability calculation.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have identical shapes; got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if direction not in KL_DIRECTIONS:
        raise ValueError(
            f"Unknown SCRUB KL direction={direction!r}; available: {KL_DIRECTIONS}"
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"SCRUB KL chunk size must be positive; got {chunk_size}")

    token_mask = batch.get("attention_mask")
    if token_mask is None:
        labels = batch.get("labels")
        if labels is not None:
            token_mask = labels.ne(-100)
    if token_mask is not None:
        expected_shape = tuple(student_logits.shape[:-1])
        if tuple(token_mask.shape) != expected_shape:
            raise ValueError(
                "SCRUB token mask must match the non-vocabulary logit dimensions; "
                f"got {tuple(token_mask.shape)} and {expected_shape}"
            )
        token_mask = token_mask.to(device=student_logits.device, dtype=torch.bool)
        valid_indices = token_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    else:
        valid_indices = None

    vocab_size = student_logits.shape[-1]
    student_flat = student_logits.reshape(-1, vocab_size)
    teacher_flat = teacher_logits.reshape(-1, vocab_size)
    token_count = (
        int(valid_indices.numel())
        if valid_indices is not None
        else int(student_flat.shape[0])
    )
    if token_count == 0:
        raise ValueError("SCRUB batch has no valid tokens for KL divergence")

    total_kl = None
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        if valid_indices is None:
            student_chunk = student_flat[start:end]
            teacher_chunk = teacher_flat[start:end]
        else:
            indices = valid_indices[start:end]
            student_chunk = student_flat.index_select(0, indices)
            teacher_chunk = teacher_flat.index_select(0, indices)

        student_log_probs = F.log_softmax(student_chunk, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_chunk, dim=-1)
        if direction == "teacher_to_student":
            chunk_kl = F.kl_div(
                student_log_probs,
                teacher_log_probs.exp(),
                reduction="sum",
            )
        else:
            # Explicit form keeps gradients on the student distribution while
            # the teacher remains a frozen reference.
            student_probs = student_log_probs.exp()
            chunk_kl = torch.sum(
                student_probs * (student_log_probs - teacher_log_probs)
            )
        total_kl = chunk_kl if total_kl is None else total_kl + chunk_kl

    return total_kl / token_count


class MMSCRUB(MMUnlearnBase):
    """Teacher-student SCRUB adapter for paired multimodal batches."""

    def __init__(
        self,
        *args,
        teacher_model=None,
        forget_weight: float = 1.0,
        retain_kl_weight: float = 1.0,
        retain_task_weight: float = 1.0,
        kl_direction: str = "teacher_to_student",
        kl_chunk_size: int = 16,
        teacher_device: str | None = None,
        **kwargs,
    ):
        self.forget_weight = float(forget_weight)
        self.retain_kl_weight = float(retain_kl_weight)
        self.retain_task_weight = float(retain_task_weight)
        self.kl_direction = str(kl_direction)
        self.kl_chunk_size = int(kl_chunk_size)
        self.teacher_device = teacher_device
        if self.kl_direction not in KL_DIRECTIONS:
            raise ValueError(
                f"Unknown SCRUB KL direction={self.kl_direction!r}; "
                f"available: {KL_DIRECTIONS}"
            )
        if self.kl_chunk_size <= 0:
            raise ValueError(
                f"MMSCRUB kl_chunk_size must be positive; got {self.kl_chunk_size}"
            )
        if (
            min(
                self.forget_weight,
                self.retain_kl_weight,
                self.retain_task_weight,
            )
            < 0
        ):
            raise ValueError("MMSCRUB objective weights must be non-negative")

        original_model = kwargs.get("model")
        if original_model is None and args:
            original_model = args[0]
        if teacher_model is original_model and teacher_model is not None:
            raise ValueError(
                "MMSCRUB teacher_model must be a separate frozen model, not the "
                "trainable student object"
            )
        if teacher_device and teacher_model is None:
            raise ValueError(
                "MMSCRUB teacher_device requires a separately supplied teacher_model "
                "to avoid deep-copying the student on its GPU"
            )

        super().__init__(*args, **kwargs)
        unwrapped_student = self.accelerator.unwrap_model(self.model)
        teacher = (
            deepcopy(unwrapped_student) if teacher_model is None else teacher_model
        )
        teacher.eval()
        disable_checkpointing = getattr(teacher, "gradient_checkpointing_disable", None)
        if disable_checkpointing is not None:
            disable_checkpointing()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        if teacher_device:
            self.teacher_model = teacher.to(teacher_device)
        else:
            self.teacher_model = self.accelerator.prepare(teacher)
        self.teacher_model.eval()
        logger.info(
            "MMSCRUB initialized with frozen teacher on %s, KL direction=%s, "
            "token chunk size=%d",
            teacher_device or "accelerator_device",
            self.kl_direction,
            self.kl_chunk_size,
        )

    def _teacher_batch(self, batch: dict) -> dict:
        if not self.teacher_device:
            return batch
        return {
            key: value.to(self.teacher_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _kl(self, student_logits, teacher_logits, batch) -> torch.Tensor:
        return token_kl_divergence(
            student_logits,
            teacher_logits,
            batch,
            direction=self.kl_direction,
            chunk_size=self.kl_chunk_size,
        )

    def compute_loss(self, model, batch) -> torch.Tensor:
        """Compute the forget-side term for interface compatibility."""

        student_outputs = model(**batch)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**self._teacher_batch(batch))
        forget_kl = self._kl(
            student_outputs.logits,
            teacher_outputs.logits.to(student_outputs.logits.device),
            batch,
        )
        return -self.forget_weight * forget_kl

    def compute_paired_loss(
        self, model, forget_batch, retain_batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ``LR - LF + Ltask`` for one forget/retain pair."""

        forget_student = model(**forget_batch)
        with torch.no_grad():
            forget_teacher = self.teacher_model(**self._teacher_batch(forget_batch))
        forget_kl = self._kl(
            forget_student.logits,
            forget_teacher.logits.to(forget_student.logits.device),
            forget_batch,
        )

        retain_student = model(**retain_batch)
        with torch.no_grad():
            retain_teacher = self.teacher_model(**self._teacher_batch(retain_batch))
        retain_kl = self._kl(
            retain_student.logits,
            retain_teacher.logits.to(retain_student.logits.device),
            retain_batch,
        )
        retain_task_loss = retain_student.loss
        loss = scrub_objective(
            forget_kl,
            retain_kl,
            retain_task_loss,
            forget_weight=self.forget_weight,
            retain_kl_weight=self.retain_kl_weight,
            retain_task_weight=self.retain_task_weight,
        )
        return loss, forget_kl, retain_kl, retain_task_loss

    def train(self):
        if self.retain_loader is None:
            raise ValueError("MMSCRUB requires retain_loader")
        if len(self.forget_loader) == 0 or len(self.retain_loader) == 0:
            raise ValueError("MMSCRUB requires non-empty forget and retain loaders")

        epoch_losses = []
        epoch_forget_kls = []
        epoch_retain_kls = []
        epoch_retain_task_losses = []
        optimizer_steps = 0
        last_grad_norm = None

        for epoch in range(self.num_epochs):
            self.model.train()
            self.teacher_model.eval()
            retain_iter = iter(self.retain_loader)
            n_iters = len(self.forget_loader)
            total_loss = 0.0
            total_forget_kl = 0.0
            total_retain_kl = 0.0
            total_retain_task = 0.0

            progress = tqdm(
                self.forget_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for step, forget_batch in enumerate(progress):
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)

                with self._accumulate(step, n_iters):
                    loss, forget_kl, retain_kl, retain_task = self.compute_paired_loss(
                        self.model,
                        forget_batch,
                        retain_batch,
                    )
                    self.accelerator.backward(loss)
                    self._apply_grad_mask()
                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        last_grad_norm = float(grad_norm)
                    optimizer_steps += int(self._step_optimizer())

                total_loss += loss.detach().item()
                total_forget_kl += forget_kl.detach().item()
                total_retain_kl += retain_kl.detach().item()
                total_retain_task += retain_task.detach().item()
                progress.set_postfix(
                    loss=total_loss / (step + 1),
                    forget_kl=total_forget_kl / (step + 1),
                    retain_kl=total_retain_kl / (step + 1),
                )
                if self._budget_reached(optimizer_steps):
                    break

            completed_batches = step + 1
            epoch_losses.append(total_loss / completed_batches)
            epoch_forget_kls.append(total_forget_kl / completed_batches)
            epoch_retain_kls.append(total_retain_kl / completed_batches)
            epoch_retain_task_losses.append(total_retain_task / completed_batches)
            logger.info(
                "Epoch %d - loss: %.4f, forget_kl: %.4f, retain_kl: %.4f, "
                "retain_task: %.4f",
                epoch + 1,
                epoch_losses[-1],
                epoch_forget_kls[-1],
                epoch_retain_kls[-1],
                epoch_retain_task_losses[-1],
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
            "epoch_forget_kls": epoch_forget_kls,
            "epoch_retain_kls": epoch_retain_kls,
            "epoch_retain_task_losses": epoch_retain_task_losses,
            "last_grad_norm": last_grad_norm,
            "peak_memory_mb": peak_memory_mb,
            "kl_direction": self.kl_direction,
            "objective_weights": {
                "forget": self.forget_weight,
                "retain_kl": self.retain_kl_weight,
                "retain_task": self.retain_task_weight,
            },
        }
