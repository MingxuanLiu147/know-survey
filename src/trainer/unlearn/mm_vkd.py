"""VKD (arXiv:2512.11325v2, equations 3--11), for Qwen2/2.5-VL.

Prune visual MLP neurons, then apply Fisher-masked forget ascent and unmasked
retain descent plus projector-feature distillation. The teacher is a frozen copy
of the incoming Vanilla visual module, made BEFORE pruning. The LLM is frozen.
Text-only QA terms have identically zero visual gradients and can be omitted.

This is a paper-based implementation, not an official-code reproduction. See
docs/multimodal-vkd.md for numerical conventions and unreported hyperparameters.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import time

import torch

from model.vkd import QwenVKDVisual
from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)


def pruning_threshold(ratios, fraction):
    """Calibrate the paper's threshold on training statistics, above ratio one.

    ``fraction`` is a local pruning budget, not a value reported by VKD. Ties
    may reduce the selection; they must never silently exceed that budget.
    """
    values = torch.cat([value.flatten() for value in ratios.values()])
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite VKD neuron importance ratios")
    count = max(1, int(values.numel() * fraction))
    ordered = values.sort(descending=True).values
    threshold = max(float(ordered[count - 1]), 1.0)
    if threshold == 1.0 or int((values >= threshold).sum()) > count:
        threshold = float(
            torch.nextafter(torch.tensor(threshold), torch.tensor(float("inf")))
        )
    if not torch.any(values >= threshold):
        raise ValueError("VKD has no selectively prunable neurons within the budget")
    return threshold


def calibrated_loss_weights(records, retain_fraction=0.5, distill_fraction=0.25):
    """Choose fixed alpha/beta from separate, unweighted training gradients.

    Each calibration batch bounds retain and KD gradient norms by the requested
    fractions of alpha times the masked forget gradient. This is a disclosed
    local parameter calibration, not an additional term in the paper's loss.
    """
    if not records or not (
        retain_fraction > 0
        and distill_fraction > 0
        and retain_fraction + distill_fraction < 1
    ):
        raise ValueError(
            "VKD preservation gradient fractions must sum to less than one"
        )
    for row in records:
        if any(
            not math.isfinite(row[key]) or row[key] < 0
            for key in ("forget_norm", "retain_norm", "distill_norm")
        ):
            raise ValueError("Non-finite VKD calibration gradient")
        if row["forget_norm"] <= 1e-12:
            raise ValueError("VKD masked forget gradient is zero")
    alpha = max(
        1.0,
        max(
            row["retain_norm"] / (retain_fraction * row["forget_norm"])
            for row in records
        ),
    )
    beta = min(
        [1.0]
        + [
            distill_fraction * alpha * row["forget_norm"] / row["distill_norm"]
            for row in records
            if row["distill_norm"] > 1e-12
        ]
    )
    return alpha, beta


class MMVKD(MMUnlearnBase):
    """Single-process visual unlearning with auditable calibration and masks."""

    def __init__(
        self,
        model,
        processor,
        forget_loader,
        retain_loader=None,
        *,
        forget_alpha: float = 1.0,
        distill_beta: float = 1.0,
        prune_threshold: float = 2.0,
        fisher_threshold: float = 1.0,
        vision_layer_fraction: float = 0.5,
        calibration_max_samples: int | None = None,
        epsilon: float = 1e-12,
        visual_fp32: bool = True,
        prune_fraction: float | None = None,
        require_nonzero_pruning: bool = False,
        calibrate_loss_weights: bool = False,
        loss_calibration_batches: int = 8,
        retain_gradient_fraction: float = 0.5,
        distill_gradient_fraction: float = 0.25,
        **kwargs,
    ):
        for name, value in {
            "forget_alpha": forget_alpha,
            "distill_beta": distill_beta,
            "prune_threshold": prune_threshold,
            "fisher_threshold": fisher_threshold,
            "epsilon": epsilon,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"MMVKD {name} must be finite and positive")
        if int(kwargs.get("gradient_accumulation_steps", 1)) != 1:
            raise ValueError("MMVKD requires gradient_accumulation_steps=1")
        if kwargs.get("grad_mask") is not None:
            raise ValueError("MMVKD computes its own Vanilla Fisher mask")
        if retain_loader is None or not len(retain_loader) or not len(forget_loader):
            raise ValueError("MMVKD requires non-empty forget and retain loaders")
        if calibration_max_samples is not None and (
            not isinstance(calibration_max_samples, int) or calibration_max_samples < 1
        ):
            raise ValueError(
                "calibration_max_samples must be a positive integer or null"
            )
        if prune_fraction is not None and not 0 < prune_fraction <= 1:
            raise ValueError("VKD prune_fraction must be in (0, 1]")
        if loss_calibration_batches < 1 or not (
            retain_gradient_fraction > 0
            and distill_gradient_fraction > 0
            and retain_gradient_fraction + distill_gradient_fraction < 1
        ):
            raise ValueError("Invalid VKD gradient calibration budget")
        self.prune_fraction = prune_fraction
        self.require_nonzero_pruning = require_nonzero_pruning
        self.calibrate_loss_weights = calibrate_loss_weights
        self.loss_calibration_batches = loss_calibration_batches
        self.retain_gradient_fraction = retain_gradient_fraction
        self.distill_gradient_fraction = distill_gradient_fraction
        self._vkd_contract = dict(
            forget_alpha=forget_alpha,
            distill_beta=distill_beta,
            prune_threshold=prune_threshold,
            fisher_threshold=fisher_threshold,
            vision_layer_fraction=vision_layer_fraction,
            calibration_max_samples=calibration_max_samples,
            epsilon=epsilon,
            visual_fp32=visual_fp32,
            prune_fraction=prune_fraction,
            require_nonzero_pruning=require_nonzero_pruning,
            calibrate_loss_weights=calibrate_loss_weights,
            loss_calibration_batches=loss_calibration_batches,
            retain_gradient_fraction=retain_gradient_fraction,
            distill_gradient_fraction=distill_gradient_fraction,
        )
        self.layout = QwenVKDVisual(model, vision_layer_fraction)
        model.requires_grad_(False)
        self.layout.visual.requires_grad_(True)
        # Full visual weight updates need sufficient precision at small learning
        # rates. Adam on bf16 parameters has no fp32 master weights by default.
        if visual_fp32:
            self.layout.visual.float()
        self.teacher = deepcopy(self.layout.visual).requires_grad_(False).eval()
        self.forget_alpha = forget_alpha
        self.distill_beta = distill_beta
        self.prune_threshold = prune_threshold
        self.fisher_threshold = fisher_threshold
        self.calibration_max_samples = calibration_max_samples
        self.epsilon = epsilon
        self.preparation = None
        super().__init__(model, processor, forget_loader, retain_loader, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError(
                "MMVKD currently supports one process; calibration is local"
            )
        self.teacher.to(self.accelerator.device)

    def _calibrate(self, loader):
        """Estimate per-example mean absolute activation and empirical Fisher."""
        parameters = self.layout.masked_parameters()
        fisher = {
            name: torch.zeros_like(p, device="cpu", dtype=torch.float32)
            for name, p in parameters.items()
        }
        activations = {
            name: torch.zeros(module.in_features, dtype=torch.float32)
            for name, module in self.layout.projections.items()
        }
        current = {}

        def capture(name):
            def hook(_module, inputs):
                value = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
                current[name] = value.abs().mean(0).cpu()

            return hook

        handles = [
            module.register_forward_pre_hook(capture(name))
            for name, module in self.layout.projections.items()
        ]
        count = 0
        self.model.eval()
        try:
            for batch in loader:
                for example in self.layout.examples(batch):
                    current.clear()
                    loss = self.model(**example).loss
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite VKD calibration loss")
                    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
                    if current.keys() != activations.keys():
                        raise ValueError("Selected VKD neurons were not activated")
                    for name, value in current.items():
                        activations[name].add_(value)
                    for name, gradient in zip(parameters, gradients):
                        fisher[name].add_(gradient.detach().float().cpu().square())
                    count += 1
                    if (
                        self.calibration_max_samples
                        and count >= self.calibration_max_samples
                    ):
                        break
                if (
                    self.calibration_max_samples
                    and count >= self.calibration_max_samples
                ):
                    break
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)
        if count == 0:
            raise ValueError("VKD calibration produced no QA examples")
        for value in (*fisher.values(), *activations.values()):
            value.div_(count)
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite VKD calibration statistic")
        return activations, fisher, count

    def prepare_unlearning(self):
        """Calibrate the incoming Vanilla, save selections, then prune columns."""
        if self.preparation is not None:
            return self.preparation
        started = time.perf_counter()
        forget_a, forget_f, forget_count = self._calibrate(self.forget_loader)
        retain_a, retain_f, retain_count = self._calibrate(self.retain_loader)
        self.forget_mask = {
            name: value / (retain_f[name] + self.epsilon) >= self.fisher_threshold
            for name, value in forget_f.items()
        }
        ratios = {
            name: value / (retain_a[name] + self.epsilon)
            for name, value in forget_a.items()
        }
        threshold = self.prune_threshold
        if self.prune_fraction is not None:
            threshold = pruning_threshold(ratios, self.prune_fraction)
        pruning = {name: value >= threshold for name, value in ratios.items()}
        selected = sum(int(value.sum()) for value in pruning.values())
        if self.require_nonzero_pruning and selected == 0:
            raise ValueError(
                "VKD selected zero neurons; calibrate the pruning threshold before training"
            )
        self.preparation = {
            "forget_calibration_examples": forget_count,
            "retain_calibration_examples": retain_count,
            "selected_mlps": list(self.layout.mlps),
            "pruned_neurons": selected,
            "candidate_neurons": sum(value.numel() for value in pruning.values()),
            "forget_mask_weights": sum(
                int(value.sum()) for value in self.forget_mask.values()
            ),
            "candidate_mask_weights": sum(
                value.numel() for value in self.forget_mask.values()
            ),
            "fisher_estimator": "mean_of_squared_per_example_ce_gradients",
            "text_terms": "zero_visual_gradient_with_frozen_language_model",
            "teacher_source": "incoming_vanilla_visual_module_before_pruning",
            "visual_parameter_dtype": str(next(self.layout.visual.parameters()).dtype),
            "forward_mixed_precision": self.accelerator.mixed_precision,
            "feature_loss_reduction": "mean_tokens_sum_feature_dimensions",
            "prune_threshold": threshold,
            "requested_prune_threshold": self.prune_threshold,
            "prune_fraction_budget": self.prune_fraction,
            "activation_ratio_quantiles": torch.quantile(
                torch.cat(list(ratios.values())),
                torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]),
            ).tolist(),
            "fisher_threshold": self.fisher_threshold,
            "epsilon": self.epsilon,
        }
        path = Path(self.save_dir) / "vkd_selection.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self.preparation,
                "forget_mask": self.forget_mask,
                "pruned_neurons": pruning,
                "activation_ratios": ratios,
            },
            path,
        )
        # Materialize pruning in weights so save/reload needs no inference hooks.
        with torch.no_grad():
            for name, module in self.layout.projections.items():
                module.weight.mul_((~pruning[name]).to(module.weight.device)[None, :])
        if self.calibrate_loss_weights:
            self._calibrate_objective()
        self.preparation["calibration_seconds"] = time.perf_counter() - started
        (Path(self.save_dir) / "vkd_calibration.json").write_text(
            json.dumps(self.preparation, indent=2)
        )
        logger.info("VKD calibration and pruning: %s", self.preparation)
        return self.preparation

    def _gradient_stats(self, forget=None):
        squared = dot = 0.0
        for name, parameter in self.layout.visual.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            squared += float(gradient.square().sum())
            if forget is not None and name in forget:
                dot += float((gradient.cpu() * forget[name]).sum())
        return math.sqrt(squared), dot

    def _calibrate_objective(self):
        """Measure each component independently before any optimizer update."""
        self.model.eval()
        self.layout.visual.train()
        records = []
        for index, (forget, retain) in enumerate(
            zip(self.forget_loader, self.retain_loader)
        ):
            if index >= self.loss_calibration_batches:
                break
            forget = forget.get("multimodal", forget)
            retain = retain.get("multimodal", retain)
            self.optimizer.zero_grad(set_to_none=True)
            loss = -self.model(**forget).loss
            self.accelerator.backward(loss)
            self._mask_forget_gradients()
            fn, _ = self._gradient_stats()
            fg = {
                name: p.grad.detach().float().cpu().clone()
                for name, p in self.layout.masked_parameters().items()
                if p.grad is not None
            }
            self.optimizer.zero_grad(set_to_none=True)
            ce = self.model(**retain).loss
            self.accelerator.backward(ce)
            rn, fr = self._gradient_stats(fg)
            self.optimizer.zero_grad(set_to_none=True)
            with self.accelerator.autocast():
                student = self.layout.features(self.layout.visual, retain)
                with torch.no_grad():
                    teacher = self.layout.features(self.teacher, retain)
            kd = (student.float() - teacher.float()).square().sum(-1).mean()
            self.accelerator.backward(kd)
            kn, fk = self._gradient_stats(fg)
            records.append(
                dict(
                    forget_norm=fn,
                    retain_norm=rn,
                    distill_norm=kn,
                    forget_retain_dot=fr,
                    forget_distill_dot=fk,
                    forget_ce=-float(loss.detach()),
                    retain_ce=float(ce.detach()),
                    feature_squared_l2=float(kd.detach()),
                )
            )
            self.optimizer.zero_grad(set_to_none=True)
            del fg, loss, ce, kd, student, teacher
        self.forget_alpha, self.distill_beta = calibrated_loss_weights(
            records, self.retain_gradient_fraction, self.distill_gradient_fraction
        )
        for row in records:
            row["forget_descent_projection"] = (
                self.forget_alpha * row["forget_norm"] ** 2
                + row["forget_retain_dot"]
                + self.distill_beta * row["forget_distill_dot"]
            )
            if row["forget_descent_projection"] <= 0:
                raise ValueError(
                    "Calibrated VKD gradient does not descend the forget objective"
                )
        self.preparation["loss_calibration"] = dict(
            source="local_training_gradient_calibration_not_author_defaults",
            batches=len(records),
            forget_alpha=self.forget_alpha,
            distill_beta=self.distill_beta,
            retain_gradient_fraction=self.retain_gradient_fraction,
            distill_gradient_fraction=self.distill_gradient_fraction,
            records=records,
        )

    def _checkpoint_contract(self):
        return {**super()._checkpoint_contract(), "vkd": self._vkd_contract}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        calibration = (self.preparation or {}).get("loss_calibration")
        if calibration is not None:
            self.forget_alpha = calibration["forget_alpha"]
            self.distill_beta = calibration["distill_beta"]

    def _mask_forget_gradients(self):
        """Mask only the fresh forget component, before retain/KD backward."""
        for name, parameter in self.layout.visual.named_parameters():
            if parameter.grad is None:
                continue
            mask = self.forget_mask.get(name)
            if mask is None:
                parameter.grad.zero_()
            else:
                parameter.grad.mul_(mask.to(parameter.grad.device))

    def compute_loss(self, model, batch):
        """Return the forget objective; train() adds unmasked preservation terms."""
        return -self.forget_alpha * model(**batch).loss

    def _retain_loss(self, batch):
        captured = []
        handle = self.layout.visual.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        try:
            ce = self.model(**batch).loss
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("VKD expects one visual forward per retain batch")
        # Accelerator wraps the student's model forward in autocast. The
        # standalone teacher is not prepared by Accelerator, so it must enter
        # the same context explicitly; otherwise identical weights can incur a
        # spurious distillation loss before any pruning or optimizer update.
        with torch.no_grad(), self.accelerator.autocast():
            target = self.layout.features(self.teacher, batch)
        student = captured[0]
        if student.shape != target.shape:
            raise ValueError("Teacher and student visual feature shapes differ")
        # Squared L2 per projected token, averaged over tokens. No logits KL.
        kd = (student.float() - target.float()).square().sum(-1).mean()
        return ce + self.distill_beta * kd, ce, kd

    def train(self):
        """Run forget-driven paired updates with a cycling shuffled retain loader."""
        state = self._restore_training()
        self.prepare_unlearning()
        steps = state.get("optimizer_steps", 0)
        epoch_losses = state.get("epoch_losses", [])
        epoch_kd = state.get("epoch_kd", [])
        epoch_forget = state.get("epoch_forget", [])
        epoch_retain = state.get("epoch_retain", [])
        last_grad_norm = state.get("last_grad_norm")
        optimization_seconds = 0.0
        measured_steps = measured_examples = 0
        for _epoch in range(state.get("next_epoch", 0), self.num_epochs):
            # Keep frozen LLM dropout off; visual layers still train/checkpoint.
            self.model.eval()
            self.layout.visual.train()
            retain_iter = iter(self.retain_loader)
            total_forget = total_retain = total_kd = 0.0
            batches = 0
            for forget_batch in self.forget_loader:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)
                forget_batch = forget_batch.get("multimodal", forget_batch)
                retain_batch = retain_batch.get("multimodal", retain_batch)
                step_started = time.perf_counter()
                self.optimizer.zero_grad(set_to_none=True)
                forget_loss = self.compute_loss(self.model, forget_batch)
                if not torch.isfinite(forget_loss):
                    raise ValueError("Non-finite VKD forget loss")
                self.accelerator.backward(forget_loss)
                self._mask_forget_gradients()
                preserve_loss, retain_ce, kd = self._retain_loss(retain_batch)
                if not torch.isfinite(preserve_loss):
                    raise ValueError("Non-finite VKD preservation loss")
                self.accelerator.backward(preserve_loss)
                last_grad_norm = self._clip_grad_norm()
                if last_grad_norm is not None and not math.isfinite(last_grad_norm):
                    raise ValueError("Non-finite VKD gradient norm")
                if self.max_grad_norm is None and any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in self.layout.visual.parameters()
                ):
                    raise ValueError("Non-finite VKD gradient")
                steps += int(self._step_optimizer())
                batches += 1
                total_forget += float(forget_loss.detach())
                total_retain += float(retain_ce.detach())
                total_kd += float(kd.detach())
                optimization_seconds += time.perf_counter() - step_started
                measured_steps += 1
                measured_examples += int(forget_batch["input_ids"].shape[0]) + int(
                    retain_batch["input_ids"].shape[0]
                )
                if self._budget_reached(steps):
                    break
            epoch_forget.append(total_forget / batches)
            epoch_retain.append(total_retain / batches)
            epoch_kd.append(total_kd / batches)
            epoch_losses.append(
                (total_forget + total_retain + self.distill_beta * total_kd) / batches
            )
            if batches == len(self.forget_loader) and self._finish_epoch(
                _epoch,
                {
                    "optimizer_steps": steps,
                    "epoch_losses": epoch_losses,
                    "epoch_kd": epoch_kd,
                    "epoch_forget": epoch_forget,
                    "epoch_retain": epoch_retain,
                    "last_grad_norm": last_grad_norm,
                },
            ):
                break
            if self._budget_reached(steps):
                break
        return {
            "training_complete": len(epoch_losses) == self.num_epochs
            or self._budget_reached(steps),
            "epochs": len(epoch_losses),
            "max_steps": self.max_steps,
            "optimizer_steps": steps,
            "epoch_losses": epoch_losses,
            "epoch_forget_losses": epoch_forget,
            "epoch_retain_losses": epoch_retain,
            "epoch_feature_squared_l2": epoch_kd,
            "last_grad_norm": last_grad_norm,
            "preparation": self.preparation,
            "language_model_frozen": True,
            "teacher_parameters": sum(p.numel() for p in self.teacher.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "update_schedule": "forget_driven_paired",
            "effective_forget_alpha": self.forget_alpha,
            "effective_distill_beta": self.distill_beta,
            "optimization_seconds": optimization_seconds,
            "seconds_per_optimizer_step": optimization_seconds / measured_steps
            if measured_steps
            else None,
            "paired_examples_per_second": measured_examples / optimization_seconds
            if optimization_seconds
            else None,
            "peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None,
        }
