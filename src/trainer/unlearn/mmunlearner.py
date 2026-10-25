"""MMUnlearner: SFR-on-masked forget ascent plus unmasked retain descent.

The paper's objective is ``-m * L_forget + L_retain``.  The two gradient
components must therefore be back-propagated separately: the saliency mask is
applied only to the forget component, then the full retain gradient is added.
Masking a pre-combined GradDiff gradient would incorrectly mask retention.
"""

from __future__ import annotations

import logging

import torch
from tqdm import tqdm

from trainer.unlearn.mm_grad_diff import MMGradDiff
from trainer.unlearn.sfron import load_and_validate_saliency_mask

logger = logging.getLogger(__name__)


class MMUnlearner(MMGradDiff):
    """Implement MMUnlearner Eq. (3) with an auditable SFR-on mask."""

    def __init__(
        self,
        *args,
        grad_mask_path: str | None = None,
        forget_alpha: float = 1.0,
        expected_benchmark: str | None = None,
        expected_model_family: str | None = None,
        expected_checkpoint_id: str | None = None,
        allow_legacy_mask: bool = False,
        min_mask_coverage: float = 1.0,
        update_schedule: str = "retain_frequency",
        **kwargs,
    ):
        if forget_alpha <= 0:
            raise ValueError("MMUnlearner forget_alpha must be positive")
        if grad_mask_path is None:
            raise ValueError("MMUnlearner requires a generated grad_mask_path")
        self.update_schedule = str(update_schedule).lower()
        if self.update_schedule not in {"retain_frequency", "paired"}:
            raise ValueError(
                "MMUnlearner update_schedule must be retain_frequency or paired"
            )

        gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 1))
        if gradient_accumulation_steps != 1:
            raise ValueError(
                "MMUnlearner currently requires gradient_accumulation_steps=1 so "
                "the forget mask cannot erase retained gradients from an earlier "
                "microbatch."
            )

        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        if model is None:
            raise TypeError("MMUnlearner could not resolve the training model")

        loaded_mask, validation_report = load_and_validate_saliency_mask(
            grad_mask_path,
            model,
            expected_benchmark=expected_benchmark,
            expected_model_family=expected_model_family,
            expected_checkpoint_id=expected_checkpoint_id,
            allow_legacy_mask=allow_legacy_mask,
            min_mask_coverage=min_mask_coverage,
        )
        self.forget_alpha = float(forget_alpha)
        self.mask_validation_report = validation_report
        self._mask_application_totals = {
            "forget_backward_calls": 0,
            "masked_gradient_parameters": 0,
            "zeroed_outside_scope_parameters": 0,
        }
        kwargs["grad_mask"] = loaded_mask
        logger.info("Validated MMUnlearner mask: %s", validation_report)
        parent_schedule = (
            "sequential" if self.update_schedule == "retain_frequency" else "paired"
        )
        super().__init__(*args, schedule_mode=parent_schedule, **kwargs)

    def compute_forget_loss(self, model, batch) -> torch.Tensor:
        """Return the signed, weighted forget-ascent objective."""

        outputs = model(**batch)
        return -self.forget_alpha * outputs.loss

    def _apply_forget_grad_mask(self) -> None:
        """Apply the mask to forget gradients only; absent keys mean zero."""

        unwrapped = self.accelerator.unwrap_model(self.model)
        masked = 0
        zeroed = 0
        for name, parameter in unwrapped.named_parameters():
            if parameter.grad is None:
                continue
            mask = self.grad_mask.get(name)
            if mask is None:
                parameter.grad.zero_()
                zeroed += 1
            else:
                parameter.grad.mul_(mask.to(parameter.grad.device))
                masked += 1
        self._mask_application_totals["forget_backward_calls"] += 1
        self._mask_application_totals["masked_gradient_parameters"] += masked
        self._mask_application_totals["zeroed_outside_scope_parameters"] += zeroed

    def _clip_and_step(self) -> tuple[float | None, bool]:
        grad_norm = None
        if self.accelerator.sync_gradients:
            grad_norm = float(
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            )
        return grad_norm, self._step_optimizer()

    def _train_paired_epoch(self, completed_steps: int = 0):
        total_forget = 0.0
        total_retain = 0.0
        optimizer_steps = 0
        last_grad_norm = None
        retain_iter = iter(self.retain_loader)

        for step, forget_batch in enumerate(self.forget_loader):
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(self.retain_loader)
                retain_batch = next(retain_iter)

            with self._accumulate(step, len(self.forget_loader)):
                forget_objective = self.compute_forget_loss(self.model, forget_batch)
                self.accelerator.backward(forget_objective)
                self._apply_forget_grad_mask()
                retain_objective = self.compute_retain_loss(self.model, retain_batch)
                self.accelerator.backward(retain_objective)
                grad_norm, stepped = self._clip_and_step()
                if grad_norm is not None:
                    last_grad_norm = grad_norm
                optimizer_steps += int(stepped)
            total_forget += forget_objective.detach().item()
            total_retain += retain_objective.detach().item()
            if self._budget_reached(completed_steps + optimizer_steps):
                break

        return {
            "forget_total": total_forget,
            "retain_total": total_retain,
            "forget_steps": step + 1,
            "retain_steps": step + 1,
            "optimizer_steps": optimizer_steps,
            "last_grad_norm": last_grad_norm,
        }

    def _train_retain_frequency_epoch(self, epoch: int, completed_steps: int = 0):
        """Match the official repository's full-retain/frequent-forget schedule."""

        total_forget = 0.0
        total_retain = 0.0
        forget_steps = 0
        retain_steps = 0
        optimizer_steps = 0
        last_grad_norm = None
        forget_iter = iter(self.forget_loader)
        forget_frequency = max(1, len(self.retain_loader) // len(self.forget_loader))
        progress = tqdm(
            self.retain_loader,
            desc=f"Epoch {epoch + 1}/{self.num_epochs} retain-frequency",
            disable=not self.accelerator.is_local_main_process,
        )

        for retain_index, retain_batch in enumerate(progress):
            if retain_index % forget_frequency == 0:
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_batch = None
                if forget_batch is not None:
                    with self._accumulate(forget_steps, len(self.forget_loader)):
                        forget_objective = self.compute_forget_loss(
                            self.model, forget_batch
                        )
                        self.accelerator.backward(forget_objective)
                        self._apply_forget_grad_mask()
                        grad_norm, stepped = self._clip_and_step()
                        if grad_norm is not None:
                            last_grad_norm = grad_norm
                        optimizer_steps += int(stepped)
                    total_forget += forget_objective.detach().item()
                    forget_steps += 1
                    if self._budget_reached(completed_steps + optimizer_steps):
                        break

            with self._accumulate(retain_index, len(self.retain_loader)):
                retain_objective = self.compute_retain_loss(self.model, retain_batch)
                self.accelerator.backward(retain_objective)
                grad_norm, stepped = self._clip_and_step()
                if grad_norm is not None:
                    last_grad_norm = grad_norm
                optimizer_steps += int(stepped)
            total_retain += retain_objective.detach().item()
            retain_steps += 1
            progress.set_postfix(
                f_loss=(total_forget / forget_steps if forget_steps else 0.0),
                r_loss=total_retain / retain_steps,
            )
            if self._budget_reached(completed_steps + optimizer_steps):
                break

        if forget_steps != len(self.forget_loader) and not self._budget_reached(
            completed_steps + optimizer_steps
        ):
            raise RuntimeError(
                "MMUnlearner retain-frequency schedule did not consume the complete "
                f"forget loader: {forget_steps}/{len(self.forget_loader)}"
            )
        return {
            "forget_total": total_forget,
            "retain_total": total_retain,
            "forget_steps": forget_steps,
            "retain_steps": retain_steps,
            "optimizer_steps": optimizer_steps,
            "last_grad_norm": last_grad_norm,
            "forget_frequency": forget_frequency,
        }

    def train(self) -> dict:
        """Optimize masked forget ascent while preserving the complete retain set."""

        if self.retain_loader is None:
            raise ValueError("MMUnlearner requires a retain loader")
        if len(self.forget_loader) == 0 or len(self.retain_loader) == 0:
            raise ValueError("MMUnlearner requires non-empty forget and retain loaders")

        epoch_forget_losses: list[float] = []
        epoch_retain_losses: list[float] = []
        epoch_losses: list[float] = []
        optimizer_steps = 0
        last_grad_norm = None
        epoch_forget_steps = []
        epoch_retain_steps = []
        forget_frequencies = []

        for epoch in range(self.num_epochs):
            self.model.train()
            if self.update_schedule == "retain_frequency":
                stats = self._train_retain_frequency_epoch(epoch, optimizer_steps)
            else:
                stats = self._train_paired_epoch(optimizer_steps)
            optimizer_steps += stats["optimizer_steps"]
            if stats["last_grad_norm"] is not None:
                last_grad_norm = stats["last_grad_norm"]
            avg_forget_loss = (
                stats["forget_total"] / stats["forget_steps"]
                if stats["forget_steps"]
                else None
            )
            avg_retain_loss = (
                stats["retain_total"] / stats["retain_steps"]
                if stats["retain_steps"]
                else None
            )
            avg_loss = (avg_forget_loss or 0.0) + (avg_retain_loss or 0.0)
            epoch_forget_losses.append(avg_forget_loss)
            epoch_retain_losses.append(avg_retain_loss)
            epoch_losses.append(avg_loss)
            epoch_forget_steps.append(stats["forget_steps"])
            epoch_retain_steps.append(stats["retain_steps"])
            forget_frequencies.append(stats.get("forget_frequency"))
            logger.info(
                "Epoch %d - objective: %.4f, masked forget: %s, retain: %s",
                epoch + 1,
                avg_loss,
                avg_forget_loss,
                avg_retain_loss,
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
            "epoch_forget_losses": epoch_forget_losses,
            "epoch_retain_losses": epoch_retain_losses,
            "last_grad_norm": last_grad_norm,
            "peak_memory_mb": peak_memory_mb,
            "mask_validation": self.mask_validation_report,
            "mask_application": dict(self._mask_application_totals),
            "objective": "-mask*forget_alpha*L_forget + L_retain",
            "update_schedule": self.update_schedule,
            "epoch_forget_steps": epoch_forget_steps,
            "epoch_retain_steps": epoch_retain_steps,
            "forget_frequencies": forget_frequencies,
        }
