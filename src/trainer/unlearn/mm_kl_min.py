"""多模态 KL_Min 遗忘方法。

每个 forget batch 配一个 retain batch，联合执行 forget 梯度上升并最小化
当前模型与 vanilla 模型在 retain 数据上的 KL 散度。
vanilla 模型通过 deepcopy 创建，frozen 状态用于计算参考 logits。
"""

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from trainer.unlearn.mm_grad_diff import MMGradDiff

logger = logging.getLogger(__name__)


def kl_min_objective(
    forget_nll: torch.Tensor,
    retain_kl: torch.Tensor,
    retain_ce: torch.Tensor | None = None,
    *,
    retain_ce_weight: float = 0.0,
) -> torch.Tensor:
    """Return an explicit benchmark KL objective.

    CLEAR uses ``-L_forget + KL``.  The original MLLMU-Bench script applies
    forget ascent and retain updates sequentially, with the retain update
    minimizing ``CE + KL``.  ``retain_ce_weight`` makes that difference
    auditable rather than hiding it in a shared trainer.
    """

    objective = -forget_nll + retain_kl
    if retain_ce is not None:
        objective = objective + float(retain_ce_weight) * retain_ce
    return objective


def retain_kl_divergence(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    batch: dict,
    *,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Compute exact token-mean KL in bounded-memory token chunks.

    Softmax still spans the full vocabulary for every valid token; chunking
    only limits how many token rows are materialized simultaneously.
    """

    if current_logits.shape != reference_logits.shape:
        raise ValueError(
            "Current and reference retain logits must have the same shape; got "
            f"{tuple(current_logits.shape)} and {tuple(reference_logits.shape)}"
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"KL token chunk size must be positive; got {chunk_size}")

    token_mask = batch.get("attention_mask")
    if token_mask is None:
        labels = batch.get("labels")
        if labels is not None:
            token_mask = labels.ne(-100)
    if token_mask is not None and tuple(token_mask.shape) == tuple(
        current_logits.shape[:-1]
    ):
        token_mask = token_mask.to(device=current_logits.device, dtype=torch.bool)
        if not token_mask.any():
            raise ValueError("Retain batch has no valid tokens for KL minimization")
        valid_indices = token_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    else:
        valid_indices = None

    vocab_size = current_logits.shape[-1]
    current_flat = current_logits.reshape(-1, vocab_size)
    reference_flat = reference_logits.reshape(-1, vocab_size)
    token_count = (
        valid_indices.numel() if valid_indices is not None else current_flat.shape[0]
    )
    if token_count == 0:
        raise ValueError("Retain batch has no tokens for KL minimization")

    total_kl = None
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        if valid_indices is None:
            current_chunk = current_flat[start:end]
            reference_chunk = reference_flat[start:end]
        else:
            chunk_indices = valid_indices[start:end]
            current_chunk = current_flat.index_select(0, chunk_indices)
            reference_chunk = reference_flat.index_select(0, chunk_indices)

        current_log_probs = F.log_softmax(current_chunk, dim=-1)
        reference_probs = F.softmax(reference_chunk, dim=-1)
        chunk_kl = F.kl_div(
            current_log_probs,
            reference_probs,
            reduction="sum",
        )
        total_kl = chunk_kl if total_kl is None else total_kl + chunk_kl

    return total_kl / token_count


class MMKLMin(MMGradDiff):
    """Benchmark-aware forget ascent plus retain KL minimization."""

    def __init__(
        self,
        model,
        processor,
        forget_loader,
        retain_loader=None,
        *,
        kl_chunk_size: int = 16,
        reference_model=None,
        reference_device: str | None = None,
        schedule_mode: str = "paired",
        benchmark: str | None = None,
        retain_ce_weight: float | None = None,
        **kwargs,
    ):
        self.kl_chunk_size = int(kl_chunk_size)
        self.reference_device = reference_device
        normalized_benchmark = str(benchmark).lower() if benchmark is not None else None
        self.retain_ce_weight = (
            1.0
            if retain_ce_weight is None and normalized_benchmark == "mllmu"
            else float(retain_ce_weight or 0.0)
        )
        if self.kl_chunk_size <= 0:
            raise ValueError(
                f"MMKLMin kl_chunk_size must be positive; got {self.kl_chunk_size}"
            )
        if reference_device and reference_model is None:
            raise ValueError(
                "MMKLMin reference_device requires a separately supplied "
                "reference_model to avoid deep-copying the student on its GPU"
            )
        if self.retain_ce_weight < 0:
            raise ValueError("MMKLMin retain_ce_weight must be non-negative")
        super().__init__(
            model=model,
            processor=processor,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            schedule_mode=schedule_mode,
            benchmark=normalized_benchmark,
            **kwargs,
        )
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.vanilla_model = (
            deepcopy(unwrapped) if reference_model is None else reference_model
        )
        self.vanilla_model.eval()
        for p in self.vanilla_model.parameters():
            p.requires_grad = False
        if reference_device:
            self.vanilla_model = self.vanilla_model.to(reference_device)
        else:
            self.vanilla_model = self.accelerator.prepare(self.vanilla_model)
        logger.info(
            "KL_Min: frozen reference ready (source=%s, device=%s), token chunk size=%d",
            "deepcopy" if reference_model is None else "supplied_model",
            reference_device or "accelerator_device",
            self.kl_chunk_size,
        )

    def _reference_batch(self, batch: dict) -> dict:
        if not self.reference_device:
            return batch
        return {
            key: value.to(self.reference_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def compute_retain_loss(self, model, batch) -> torch.Tensor:
        """Compute the benchmark-selected retain objective."""

        retain_out_current = model(**batch)
        with torch.no_grad():
            retain_out_vanilla = self.vanilla_model(**self._reference_batch(batch))

        retain_kl = retain_kl_divergence(
            retain_out_current.logits,
            retain_out_vanilla.logits.to(retain_out_current.logits.device),
            batch,
            chunk_size=self.kl_chunk_size,
        )
        return retain_kl + self.retain_ce_weight * retain_out_current.loss

    def train(self):
        summary = super().train()
        summary.update(
            {
                "retain_objective": (
                    "KL(reference||current)"
                    if self.retain_ce_weight == 0
                    else f"{self.retain_ce_weight:g}*CE + KL(reference||current)"
                ),
                "retain_ce_weight": self.retain_ce_weight,
                "kl_chunk_size": self.kl_chunk_size,
            }
        )
        return summary
