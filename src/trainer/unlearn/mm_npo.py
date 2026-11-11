"""多模态 NPO (Negative Preference Optimization) 遗忘方法。

支持基准原有目标，以及原始 NPO 的逐答案序列目标和 NPO+RT 保留项。
参考模型始终冻结；原始 NPO 使用 Vanilla，MIP 配置使用 retain-only oracle。
"""

import logging
import math

import torch
import torch.nn.functional as F

from trainer.unlearn.mm_base import MMUnlearnBase
from trainer.unlearn.mm_grad_diff import MMGradDiff

logger = logging.getLogger(__name__)


def clear_probability_npo_loss(
    current_logits: torch.Tensor,
    oracle_logits: torch.Tensor,
    *,
    beta: float,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Compute CLEAR's operational probability-ratio NPO objective exactly.

    The public CLEAR trainer flattens every token/vocabulary probability and
    averages ``(2/beta) * log(1 + (p/p_ref)**beta)``.  Chunking is only over
    token rows; softmax still spans the complete vocabulary.
    """

    if current_logits.shape != oracle_logits.shape:
        raise ValueError(
            "CLEAR NPO current/oracle logits must have identical shapes; got "
            f"{tuple(current_logits.shape)} and {tuple(oracle_logits.shape)}"
        )
    if beta <= 0:
        raise ValueError("NPO beta must be positive")
    if chunk_size <= 0:
        raise ValueError("CLEAR NPO chunk_size must be positive")
    vocab_size = current_logits.shape[-1]
    current_flat = current_logits.reshape(-1, vocab_size)
    oracle_flat = oracle_logits.reshape(-1, vocab_size)
    total = None
    element_count = 0
    for start in range(0, current_flat.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), current_flat.shape[0])
        current_probs = F.softmax(current_flat[start:end], dim=-1)
        oracle_probs = F.softmax(
            oracle_flat[start:end].to(current_logits.device), dim=-1
        )
        ratios = current_probs / (oracle_probs + 1e-8)
        chunk_sum = torch.log1p(torch.pow(ratios, beta)).sum()
        total = chunk_sum if total is None else total + chunk_sum
        element_count += ratios.numel()
    if total is None or element_count == 0:
        raise ValueError("CLEAR NPO received empty logits")
    return (2.0 / beta) * total / element_count


def answer_sequence_nll(
    logits: torch.Tensor, labels: torch.Tensor, *, chunk_size: int = 8
) -> torch.Tensor:
    """Sum causal answer-token NLL per example, as in original NPO.

    Labels use -100 for prompt/image/padding positions. Only supervised rows
    enter cross entropy; chunking avoids casting all visual-token logits to
    float32. Answer lengths are deliberately not averaged before NPO.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("NPO logits and labels must have matching batch/sequence axes")
    if chunk_size <= 0:
        raise ValueError("NPO chunk_size must be positive")
    targets = labels[:, 1:].to(logits.device)
    valid = targets != -100
    if targets.shape[0] == 0 or not valid.any(dim=-1).all():
        raise ValueError(
            "NPO requires at least one supervised answer token per example"
        )
    batch_indices, token_indices = valid.nonzero(as_tuple=True)
    answer_logits = logits[batch_indices, token_indices]
    answer_targets = targets[batch_indices, token_indices]
    losses = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float32)
    for start in range(0, batch_indices.numel(), chunk_size):
        rows = batch_indices[start : start + chunk_size]
        token_losses = F.cross_entropy(
            answer_logits[start : start + chunk_size].float(),
            answer_targets[start : start + chunk_size],
            reduction="none",
        )
        losses = losses.index_add(0, rows, token_losses)
    return losses


class MMNPO(MMGradDiff):
    """NPO, optionally paired with positive answer CE on the retain set (RT)."""

    def __init__(
        self,
        *args,
        oracle_model=None,
        beta: float = 0.4,
        oracle_device: str | None = None,
        benchmark: str | None = None,
        objective_mode: str = "scalar_loss_ratio",
        probability_chunk_size: int = 8,
        retain_loss_type: str = "none",
        retain_loss_weight: float = 1.0,
        logical_microbatch_size: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, benchmark=benchmark, schedule_mode="paired", **kwargs)
        self.beta = float(beta)
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("MMNPO beta must be positive")
        self.retain_loss_type = str(retain_loss_type).lower()
        if self.retain_loss_type not in {"none", "nll"}:
            raise ValueError("MMNPO retain_loss_type must be none or NLL")
        self.retain_loss_weight = float(retain_loss_weight)
        if not math.isfinite(self.retain_loss_weight) or self.retain_loss_weight <= 0:
            raise ValueError(
                "MMNPO retain_loss_weight must be positive; use type=none to disable RT"
            )
        if self.retain_loss_type == "nll" and (
            self.retain_loader is None or len(self.retain_loader) == 0
        ):
            raise ValueError("MMNPO+RT requires a non-empty retain_loader")
        self.oracle_device = oracle_device
        self.benchmark = str(benchmark).lower() if benchmark is not None else None
        self.objective_mode = str(objective_mode).lower()
        if self.objective_mode == "auto":
            if self.benchmark not in {"mllmu", "clear"}:
                raise ValueError(
                    "MMNPO objective_mode=auto requires benchmark=mllmu|clear"
                )
            self.objective_mode = (
                "clear_probability_ratio"
                if self.benchmark == "clear"
                else "scalar_loss_ratio"
            )
        if self.objective_mode not in {
            "scalar_loss_ratio",
            "clear_probability_ratio",
            "sequence_log_ratio",
        }:
            raise ValueError(
                "MMNPO objective_mode must be auto, scalar_loss_ratio, or "
                "clear_probability_ratio, or sequence_log_ratio"
            )
        self.probability_chunk_size = int(probability_chunk_size)
        self.logical_microbatch_size = logical_microbatch_size
        if logical_microbatch_size is not None and (
            int(logical_microbatch_size) < 1
            or self.objective_mode != "sequence_log_ratio"
        ):
            raise ValueError(
                "Logical microbatches require a positive size and sequence_log_ratio"
            )
        if self.probability_chunk_size <= 0:
            raise ValueError("MMNPO probability_chunk_size must be positive")
        if oracle_model is None:
            raise ValueError(
                "MMNPO requires oracle_model. "
                "Pass a pre-loaded frozen model or specify oracle_model_path in config."
            )
        self.oracle_model = oracle_model
        self.oracle_model.eval()
        for p in self.oracle_model.parameters():
            p.requires_grad = False
        if oracle_device:
            self.oracle_model = self.oracle_model.to(oracle_device)
            logger.info(
                "NPO: oracle model loaded on dedicated device %s (frozen), beta=%.2f",
                oracle_device,
                self.beta,
            )
        else:
            self.oracle_model = self.accelerator.prepare(self.oracle_model)
            logger.info("NPO: oracle model loaded (frozen), beta=%.2f", self.beta)

    def _oracle_batch(self, batch: dict) -> dict:
        if not self.oracle_device:
            return batch
        return {
            key: value.to(self.oracle_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)

        with torch.no_grad():
            oracle_outputs = self.oracle_model(**self._oracle_batch(batch))
        if self.objective_mode == "clear_probability_ratio":
            return clear_probability_npo_loss(
                outputs.logits,
                oracle_outputs.logits,
                beta=self.beta,
                chunk_size=self.probability_chunk_size,
            )

        if self.objective_mode == "sequence_log_ratio":
            current_loss = answer_sequence_nll(
                outputs.logits, batch["labels"], chunk_size=self.probability_chunk_size
            )
            oracle_loss = answer_sequence_nll(
                oracle_outputs.logits,
                batch["labels"],
                chunk_size=self.probability_chunk_size,
            ).to(current_loss.device)
        else:
            current_loss = outputs.loss
            oracle_loss = oracle_outputs.loss.to(current_loss.device)
        neg_log_ratios = current_loss - oracle_loss
        values = -F.logsigmoid(self.beta * neg_log_ratios) * 2 / self.beta
        loss = self._logical_mean(values)
        return loss

    def _logical_mean(self, values):
        """Keep the original microbatch weighting when packing physical batches."""
        if self.logical_microbatch_size is None:
            return values.mean()
        return torch.stack(
            [group.mean() for group in values.split(int(self.logical_microbatch_size))]
        ).mean()

    def compute_forget_loss(self, model, batch) -> torch.Tensor:
        """Use NPO, rather than the parent's GA, in each paired update."""
        return self.compute_loss(model, batch)

    def compute_retain_loss(self, model, batch) -> torch.Tensor:
        """Preserve correct retain answers with token-mean CE, as in NPO+RT."""
        outputs = model(**batch)
        if self.logical_microbatch_size is None:
            return self.retain_loss_weight * outputs.loss
        nll = answer_sequence_nll(
            outputs.logits, batch["labels"], chunk_size=self.probability_chunk_size
        )
        counts = (batch["labels"][:, 1:] != -100).sum(-1).to(nll.device)
        group = int(self.logical_microbatch_size)
        loss = torch.stack(
            [
                values.sum() / tokens.sum()
                for values, tokens in zip(nll.split(group), counts.split(group))
            ]
        ).mean()
        return self.retain_loss_weight * loss

    def train(self):
        summary = (
            super().train()
            if self.retain_loss_type == "nll"
            else MMUnlearnBase.train(self)
        )
        summary.update(
            {
                "objective_mode": self.objective_mode,
                "benchmark_protocol": self.benchmark,
                "beta": self.beta,
                "probability_chunk_size": self.probability_chunk_size,
                "variant": "NPO+RT" if self.retain_loss_type == "nll" else "NPO",
                "retain_loss_type": self.retain_loss_type,
                "logical_microbatch_size": self.logical_microbatch_size,
                "retain_loss_weight": (
                    self.retain_loss_weight if self.retain_loss_type == "nll" else 0.0
                ),
            }
        )
        return summary
