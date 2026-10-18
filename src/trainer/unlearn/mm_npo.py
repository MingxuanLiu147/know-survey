"""多模态 NPO (Negative Preference Optimization) 遗忘方法。

使用 oracle（未遗忘）模型作参考，在 forget 数据上优化 NPO 目标。
oracle 模型需要单独指定路径，frozen 状态参与前向计算。
"""

import logging

import torch
import torch.nn.functional as F

from trainer.unlearn.mm_base import MMUnlearnBase

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


class MMNPO(MMUnlearnBase):
    """NPO: -logsigmoid(beta * (current_loss - oracle_loss)).mean() * 2 / beta。"""

    def __init__(
        self,
        *args,
        oracle_model=None,
        beta: float = 0.4,
        oracle_device: str | None = None,
        benchmark: str | None = None,
        objective_mode: str = "scalar_loss_ratio",
        probability_chunk_size: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.beta = float(beta)
        if self.beta <= 0:
            raise ValueError("MMNPO beta must be positive")
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
        }:
            raise ValueError(
                "MMNPO objective_mode must be auto, scalar_loss_ratio, or "
                "clear_probability_ratio"
            )
        self.probability_chunk_size = int(probability_chunk_size)
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

        current_loss = outputs.loss
        oracle_loss = oracle_outputs.loss.to(current_loss.device)
        neg_log_ratios = current_loss - oracle_loss
        loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
        return loss

    def train(self):
        summary = super().train()
        summary.update(
            {
                "objective_mode": self.objective_mode,
                "benchmark_protocol": self.benchmark,
                "beta": self.beta,
                "probability_chunk_size": self.probability_chunk_size,
            }
        )
        return summary
