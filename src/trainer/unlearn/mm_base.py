"""多模态 unlearning 训练基类（sidecar，不修改现有 trainer 体系）。

提供 Accelerator 驱动的训练循环，子类只需覆写 compute_loss()。
复用上游 MMUnlearner 的训练模式：AdamW + linear scheduler + 梯度裁剪。
"""

import logging
import math
import os
import json
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam, AdamW
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
        max_grad_norm: float | None = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int | str = 0,
        max_steps: int = -1,
        save_dir: str = "./saves/mm_unlearn",
        grad_mask: dict = None,
        scheduler_steps_per_epoch: int | None = None,
        optimizer: str = "adamw",
        weight_decay: float | None = None,
        checkpoint_each_epoch: bool = False,
        resume_from_checkpoint: str | None = None,
        stop_after_epochs: int | None = None,
        checkpoint_identity: dict | None = None,
        gradient_checkpointing: bool = True,
    ):
        self.processor = processor
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        if max_grad_norm is not None and (
            not math.isfinite(max_grad_norm) or max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be positive and finite, or null")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.max_steps = int(max_steps)
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(
                "max_steps must be -1 (epoch budget) or a positive integer"
            )
        self.save_dir = save_dir
        self.grad_mask = grad_mask
        self.checkpoint_each_epoch = bool(checkpoint_each_epoch)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.stop_after_epochs = stop_after_epochs
        self.checkpoint_identity = checkpoint_identity or {}
        self._training_progress = {}
        self._checkpoint_loaded = False
        if stop_after_epochs is not None and (
            int(stop_after_epochs) < 1 or not self.checkpoint_each_epoch
        ):
            raise ValueError(
                "stop_after_epochs requires epoch checkpointing and a positive count"
            )
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

        if gradient_checkpointing and _enable_non_reentrant_gradient_checkpointing(
            model
        ):
            if hasattr(model, "config"):
                model.config.use_cache = False
            logger.info("Gradient checkpointing enabled")

        optimizer_name = str(optimizer).lower()
        optimizer_classes = {"adam": Adam, "adamw": AdamW}
        if optimizer_name not in optimizer_classes:
            raise ValueError(f"Unsupported multimodal optimizer: {optimizer!r}")
        optimizer_kwargs = {"lr": lr}
        if weight_decay is not None:
            optimizer_kwargs["weight_decay"] = float(weight_decay)
        # Preserve each optimizer's defaults when no explicit decay is supplied.
        self.optimizer = optimizer_classes[optimizer_name](
            model.parameters(), **optimizer_kwargs
        )
        self._optimizer_contract = {
            "name": optimizer_name,
            "defaults": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.optimizer.defaults.items()
            },
        }
        if self.checkpoint_each_epoch or self.resume_from_checkpoint:
            if self.accelerator.num_processes != 1 or any(
                loader is not None and loader.num_workers
                for loader in (forget_loader, retain_loader)
            ):
                raise ValueError(
                    "Epoch checkpoints currently require one process and num_workers=0"
                )
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
        self.warmup_steps = (
            self._optimizer_steps_per_epoch()
            if warmup_steps == "steps_per_epoch"
            else int(warmup_steps)
        )
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative or steps_per_epoch")
        # Use the prepared loader lengths (per process), and advance exactly once
        # per optimizer update rather than once per microbatch or per GPU.
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        self.accelerator.register_for_checkpointing(self.lr_scheduler, self)
        # A verified immutable source supplies frozen weights. Store trainable
        # tensors and buffers once, rather than rewriting Vanilla and its frozen
        # reference at every epoch (tens of GB for a small LoRA update).
        if self.checkpoint_identity.get("source_files"):
            self.accelerator.register_save_state_pre_hook(self._save_updated_weights)
            self.accelerator.register_load_state_pre_hook(self._load_updated_weights)

    @staticmethod
    def _updated_state(model):
        names = {name for name, p in model.named_parameters() if p.requires_grad}
        names.update(name for name, _ in model.named_buffers())
        return {
            name: value for name, value in model.state_dict().items() if name in names
        }

    def _save_updated_weights(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            for index, model in enumerate(models):
                torch.save(
                    {
                        name: value.detach().cpu()
                        for name, value in self._updated_state(model).items()
                    },
                    Path(output_dir) / f"updated_weights_{index}.pt",
                )
        weights.clear()

    def _load_updated_weights(self, models, input_dir):
        for index, model in enumerate(models):
            state = torch.load(
                Path(input_dir) / f"updated_weights_{index}.pt",
                map_location="cpu",
                weights_only=True,
            )
            if state.keys() != self._updated_state(model).keys():
                raise ValueError("Checkpoint trainable parameter/buffer scope changed")
            model.load_state_dict(state, strict=False)
        models.clear()

    def _checkpoint_contract(self):
        return {
            "trainer": type(self).__name__,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "lr": self.lr,
            "optimizer": self._optimizer_contract,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "accumulation": self.gradient_accumulation_steps,
            "world_size": self.accelerator.num_processes,
            "loaders": [
                None
                if loader is None
                else {
                    "batches": len(loader),
                    "examples": len(loader.dataset),
                    "batch_size": loader.batch_size,
                }
                for loader in (self.forget_loader, self.retain_loader)
            ],
            "identity": self.checkpoint_identity,
        }

    @staticmethod
    def _loader_objects(loader):
        """Include Accelerate's private sampler generator and iteration counter."""
        objects = {}

        def visit(value, path):
            if value is None or id(value) in {id(x) for x in objects.values()}:
                return
            objects[path] = value
            for name in ("base_dataloader", "batch_sampler", "sampler"):
                child = getattr(value, name, None)
                if child is not value:
                    visit(child, path + "." + name)

        visit(loader, "loader")
        return objects

    def state_dict(self):
        """State at a completed epoch; no partially accumulated gradients exist."""
        loaders = []
        for loader in (self.forget_loader, self.retain_loader):
            state = {}
            for path, obj in self._loader_objects(loader).items():
                item = {}
                for key in ("iteration", "epoch"):
                    if isinstance(getattr(obj, key, None), int):
                        item[key] = getattr(obj, key)
                generator = getattr(obj, "generator", None)
                if isinstance(generator, torch.Generator):
                    item["generator"] = generator.get_state()
                state[path] = item
            loaders.append(state)
        algorithm = {
            name: getattr(self, name)
            for name in (
                "grad_mask",
                "forget_mask",
                "preparation",
                "_mask_application_totals",
            )
            if hasattr(self, name)
        }
        if hasattr(self, "teacher"):
            algorithm["teacher"] = self.teacher.state_dict()
        return {
            "progress": self._training_progress,
            "loaders": loaders,
            "algorithm": algorithm,
        }

    def load_state_dict(self, state):
        self._training_progress = state["progress"]
        for loader, saved in zip(
            (self.forget_loader, self.retain_loader), state["loaders"]
        ):
            objects = self._loader_objects(loader)
            for path, item in saved.items():
                obj = objects[path]
                for key, value in item.items():
                    if key == "generator":
                        generator = getattr(obj, key, None)
                        if generator is None:
                            generator = torch.Generator()
                            setattr(obj, key, generator)
                        generator.set_state(value.cpu())
                    else:
                        setattr(obj, key, value)
        for name, value in state["algorithm"].items():
            if name == "teacher":
                self.teacher.load_state_dict(value)
            else:
                setattr(self, name, value)

    def _restore_training(self):
        if not self._checkpoint_loaded and self.resume_from_checkpoint:
            path = Path(self.resume_from_checkpoint)
            contract = json.loads((path / "COMPLETE.json").read_text())
            if contract != self._checkpoint_contract():
                raise ValueError(
                    "Checkpoint training/data/batch contract does not match this run"
                )
            self.accelerator.load_state(str(path))
            self._checkpoint_loaded = True
        return self._training_progress

    def _finish_epoch(self, epoch, progress):
        """Save an atomic, complete epoch checkpoint before the next shuffle."""
        self._training_progress = {**progress, "next_epoch": epoch + 1}
        if self.checkpoint_each_epoch:
            if any(
                loader is not None and loader.num_workers
                for loader in (self.forget_loader, self.retain_loader)
            ):
                raise ValueError("Exact epoch resume currently requires num_workers=0")
            root = Path(self.save_dir) / "training_state"
            final = root / f"epoch-{epoch + 1:04d}"
            pending = root / f"epoch-{epoch + 1:04d}.partial"
            if final.exists() or pending.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint {final}")
            self.accelerator.save_state(str(pending), safe_serialization=False)
            if self.accelerator.is_main_process:
                (pending / "COMPLETE.json").write_text(
                    json.dumps(self._checkpoint_contract(), indent=2)
                )
                pending.rename(final)
            self.accelerator.wait_for_everyone()
        return (
            self.stop_after_epochs is not None and epoch + 1 >= self.stop_after_epochs
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

    def _clip_grad_norm(self) -> float | None:
        """Clip only synchronized updates; null reproduces unclipped baselines."""
        if not self.accelerator.sync_gradients or self.max_grad_norm is None:
            return None
        return float(
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        )

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

        state = self._restore_training()
        epoch_losses = state.get("epoch_losses", [])
        optimizer_steps = state.get("optimizer_steps", 0)
        last_grad_norm = state.get("last_grad_norm")
        for epoch in range(state.get("next_epoch", 0), self.num_epochs):
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
                    grad_norm = self._clip_grad_norm()
                    if grad_norm is not None:
                        last_grad_norm = grad_norm
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
            if step + 1 == len(self.forget_loader) and self._finish_epoch(
                epoch,
                {
                    "epoch_losses": epoch_losses,
                    "optimizer_steps": optimizer_steps,
                    "last_grad_norm": last_grad_norm,
                },
            ):
                break
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
            "training_complete": len(epoch_losses) == self.num_epochs
            or self._budget_reached(optimizer_steps),
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
