"""Multimodal gradient-difference baselines.

``paired`` matches CLEAR's mixed forget/retain batches.  ``sequential``
matches the original MLLMU-Bench script, which first visits every forget batch
and then every retain batch in each epoch.  Keeping the schedule explicit
prevents a shared trainer from silently changing a benchmark baseline.
"""

import logging

import torch
from tqdm import tqdm

from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)


class MMGradDiff(MMUnlearnBase):
    """Jointly optimize forget ascent and retain descent on paired batches."""

    SCHEDULE_MODES = ("paired", "sequential", "auto")

    def __init__(
        self,
        model,
        processor,
        forget_loader,
        retain_loader=None,
        *,
        schedule_mode: str = "paired",
        benchmark: str | None = None,
        **kwargs,
    ):
        self.schedule_mode = str(schedule_mode).lower()
        if self.schedule_mode not in self.SCHEDULE_MODES:
            raise ValueError(
                f"Unknown MMGradDiff schedule_mode={schedule_mode!r}; "
                f"available: {self.SCHEDULE_MODES}"
            )
        self.benchmark = str(benchmark).lower() if benchmark is not None else None
        if self.schedule_mode == "auto":
            if self.benchmark not in {"mllmu", "clear"}:
                raise ValueError(
                    "MMGradDiff schedule_mode=auto requires benchmark=mllmu|clear"
                )
            self.schedule_mode = "sequential" if self.benchmark == "mllmu" else "paired"
        scheduler_steps = len(forget_loader)
        if self.schedule_mode == "sequential":
            if retain_loader is None:
                raise ValueError("Sequential MMGradDiff requires retain_loader")
            scheduler_steps += len(retain_loader)
        super().__init__(
            model=model,
            processor=processor,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            scheduler_steps_per_epoch=scheduler_steps,
            **kwargs,
        )

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        return -outputs.loss

    def compute_forget_loss(self, model, batch) -> torch.Tensor:
        """Compute the forget-side objective for alternating training."""
        outputs = model(**batch)
        return -outputs.loss

    def compute_retain_loss(self, model, batch) -> torch.Tensor:
        """Compute the retain-side objective for a paired update."""
        outputs = model(**batch)
        return outputs.loss

    def compute_paired_loss(
        self, model, forget_batch, retain_batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``-L_forget + L_retain`` and its signed components."""

        forget_objective = self.compute_forget_loss(model, forget_batch)
        retain_objective = self.compute_retain_loss(model, retain_batch)
        return (
            forget_objective + retain_objective,
            forget_objective,
            retain_objective,
        )

    def _backward_and_step(self, loss: torch.Tensor) -> tuple[float | None, bool]:
        self.accelerator.backward(loss)
        self._apply_grad_mask()
        grad_norm = None
        if self.accelerator.sync_gradients:
            grad_norm = float(
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            )
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm, bool(self.accelerator.sync_gradients)

    def _train_sequential_epoch(self, epoch: int):
        total_forget = 0.0
        total_retain = 0.0
        forget_steps = 0
        retain_steps = 0
        optimizer_steps = 0
        last_grad_norm = None

        phases = (
            ("forget", self.forget_loader, self.compute_forget_loss),
            ("retain", self.retain_loader, self.compute_retain_loss),
        )
        for phase, loader, objective in phases:
            progress = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs} {phase}",
                disable=not self.accelerator.is_local_main_process,
            )
            for step, batch in enumerate(progress):
                with self.accelerator.accumulate(self.model):
                    loss = objective(self.model, batch)
                    grad_norm, stepped = self._backward_and_step(loss)
                    if grad_norm is not None:
                        last_grad_norm = grad_norm
                    optimizer_steps += int(stepped)
                if phase == "forget":
                    total_forget += loss.detach().item()
                    forget_steps += 1
                    mean_loss = total_forget / forget_steps
                else:
                    total_retain += loss.detach().item()
                    retain_steps += 1
                    mean_loss = total_retain / retain_steps
                progress.set_postfix(loss=mean_loss)

        return {
            "forget_total": total_forget,
            "retain_total": total_retain,
            "forget_steps": forget_steps,
            "retain_steps": retain_steps,
            "optimizer_steps": optimizer_steps,
            "last_grad_norm": last_grad_norm,
        }

    def _train_paired_epoch(self, epoch: int):
        total_forget = 0.0
        total_retain = 0.0
        optimizer_steps = 0
        last_grad_norm = None
        retain_iter = iter(self.retain_loader)
        n_iters = len(self.forget_loader)
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

            with self.accelerator.accumulate(self.model):
                loss, loss_forget, loss_retain = self.compute_paired_loss(
                    self.model, forget_batch, retain_batch
                )
                grad_norm, stepped = self._backward_and_step(loss)
                if grad_norm is not None:
                    last_grad_norm = grad_norm
                optimizer_steps += int(stepped)
            total_forget += loss_forget.detach().item()
            total_retain += loss_retain.detach().item()
            progress.set_postfix(
                loss=(total_forget + total_retain) / (step + 1),
                f_loss=total_forget / (step + 1),
                r_loss=total_retain / (step + 1),
            )

        return {
            "forget_total": total_forget,
            "retain_total": total_retain,
            "forget_steps": n_iters,
            "retain_steps": n_iters,
            "optimizer_steps": optimizer_steps,
            "last_grad_norm": last_grad_norm,
        }

    def train(self):
        if self.retain_loader is None:
            raise ValueError("MMGradDiff requires retain_loader")
        if len(self.forget_loader) == 0 or len(self.retain_loader) == 0:
            raise ValueError("MMGradDiff requires non-empty forget and retain loaders")

        epoch_forget_losses = []
        epoch_retain_losses = []
        epoch_losses = []
        optimizer_steps = 0
        last_grad_norm = None

        epoch_forget_steps = []
        epoch_retain_steps = []
        for epoch in range(self.num_epochs):
            self.model.train()
            if self.schedule_mode == "sequential":
                stats = self._train_sequential_epoch(epoch)
            else:
                stats = self._train_paired_epoch(epoch)
            optimizer_steps += stats["optimizer_steps"]
            if stats["last_grad_norm"] is not None:
                last_grad_norm = stats["last_grad_norm"]
            avg_forget_loss = stats["forget_total"] / stats["forget_steps"]
            avg_retain_loss = stats["retain_total"] / stats["retain_steps"]
            avg_loss = avg_forget_loss + avg_retain_loss
            epoch_forget_losses.append(avg_forget_loss)
            epoch_retain_losses.append(avg_retain_loss)
            epoch_losses.append(avg_loss)
            epoch_forget_steps.append(stats["forget_steps"])
            epoch_retain_steps.append(stats["retain_steps"])
            logger.info(
                "Epoch %d - loss: %.4f, forget_objective: %.4f, retain_loss: %.4f",
                epoch + 1,
                avg_loss,
                avg_forget_loss,
                avg_retain_loss,
            )

        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        return {
            "epochs": self.num_epochs,
            "optimizer_steps": optimizer_steps,
            "epoch_losses": epoch_losses,
            "epoch_forget_losses": epoch_forget_losses,
            "epoch_retain_losses": epoch_retain_losses,
            "last_grad_norm": last_grad_norm,
            "peak_memory_mb": peak_memory_mb,
            "schedule_mode": self.schedule_mode,
            "benchmark_protocol": self.benchmark,
            "epoch_forget_steps": epoch_forget_steps,
            "epoch_retain_steps": epoch_retain_steps,
        }
