"""
Knowledge Injection (知识注入) 训练器基类
=========================================

本模块实现参数高效微调（PEFT）的基础训练器。
支持 LoRA、DoRA、AdaLoRA 等方法向模型注入新知识。

核心功能：
1. 集成 peft 库的适配器配置
2. 支持训练过程中的梯度检查点
3. 提供统一的模型保存和加载接口
"""

import logging
from typing import Optional, Any

import torch
import torch.nn.functional as F

from trainer.base import FinetuneTrainer

logger = logging.getLogger(__name__)


class InjectTrainer(FinetuneTrainer):
    """知识注入训练器基类

    继承自 FinetuneTrainer，添加 PEFT 相关功能支持。
    所有参数高效微调方法（LoRA、DoRA 等）都应继承此类。

    Attributes:
        peft_config: PEFT 配置对象
        adapter_name: 适配器名称
    """

    def __init__(
        self,
        peft_config: Optional[Any] = None,
        adapter_name: str = "default",
        *args,
        **kwargs,
    ):
        """初始化知识注入训练器

        Args:
            peft_config: PEFT 配置对象（LoraConfig、DoraConfig 等）
            adapter_name: 适配器名称，用于多适配器场景
            *args, **kwargs: 传递给父类 FinetuneTrainer 的参数
        """
        self.peft_config = peft_config
        self.adapter_name = adapter_name
        self._loss_group_totals = None
        super().__init__(*args, **kwargs)
        # Request Trainer's token count for the whole accumulation window.
        # compute_loss consumes it here instead of forwarding it to the model.
        self.model_accepts_loss_kwargs = True

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # These fields belong to the objective, not the model's forward API.
        for key in ("sample_weight", "loss_group"):
            if key not in self._signature_columns:
                self._signature_columns.append(key)

    @staticmethod
    def _sample_weights(inputs, labels):
        """Return one finite, nonnegative token multiplier per sequence."""
        value = inputs.get("sample_weight")
        if value is None:
            return torch.ones(labels.shape[0], device=labels.device)
        weights = torch.as_tensor(value, device=labels.device, dtype=torch.float32)
        if weights.ndim != 1 or weights.shape[0] != labels.shape[0]:
            raise ValueError("sample_weight must contain one value per sequence.")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("sample_weight must be finite and nonnegative.")
        return weights

    def _get_num_items_in_batch(self, batch_samples, device):
        if not batch_samples or "labels" not in batch_samples[0]:
            return None
        denominator = torch.zeros((), device=device)
        for batch in batch_samples:
            counts = batch["labels"][..., 1:].ne(-100).sum(dim=-1)
            if "sample_weight" in batch:
                counts = counts * self._sample_weights(batch, batch["labels"])
            denominator += counts.sum().to(device)
        if self.args.average_tokens_across_devices and self.args.world_size >= 1:
            denominator = self.accelerator.gather(denominator).sum()
        # CE is computed here on gathered logits, including DataParallel's
        # complete batch. Do not divide fractional weights by local GPU count.
        parallelism = getattr(self.accelerator, "parallelism_config", None)
        if parallelism is not None:
            denominator = denominator / parallelism.non_data_parallel_size
        return denominator

    def _normalize_token_loss(
        self,
        loss,
        labels,
        num_items_in_batch=None,
        regularization=None,
        *,
        sample_weight=None,
        loss_is_sum=False,
    ):
        """Normalize token loss globally and regularization per optimizer step."""
        if num_items_in_batch is not None or loss_is_sum:
            counts = labels[..., 1:].ne(-100).sum(dim=-1).to(loss.device)
            if sample_weight is not None:
                counts = counts * sample_weight.to(loss.device)
            token_count = counts.sum()
            denominator = torch.as_tensor(
                token_count if num_items_in_batch is None else num_items_in_batch,
                device=loss.device,
            )
            if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
                raise ValueError(
                    "Injection batch has no positive-weight target tokens."
                )
            loss = (loss if loss_is_sum else loss * token_count) / denominator
            if (
                num_items_in_batch is not None
                and self.args.average_tokens_across_devices
            ):
                # DDP averages gradients, while the denominator is a global sum.
                loss = loss * self.accelerator.num_processes
        if regularization is not None:
            if (
                num_items_in_batch is not None
                and self.is_in_train
                and self.model.training
            ):
                regularization = (
                    regularization / self.current_gradient_accumulation_steps
                )
            loss = loss + regularization
        return loss

    def _compute_causal_loss(
        self, logits, inputs, num_items_in_batch=None, regularization=None
    ):
        """Compute sum(weight * token CE) / sum(weight * valid target tokens).

        The supplied denominator covers the entire accumulation window and,
        when requested, all ranks. Call the model without labels so the large
        vocabulary cross entropy is allocated only once. Group logs report
        unweighted token means, excluding any method-specific regularization.
        """
        labels = inputs["labels"].to(logits.device)
        weights = self._sample_weights(inputs, labels)
        shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
        token_losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        loss_sum = (token_losses * weights[:, None]).sum()
        self._record_loss_groups(token_losses, shifted_labels, inputs.get("loss_group"))
        return self._normalize_token_loss(
            loss_sum,
            labels,
            num_items_in_batch,
            regularization,
            sample_weight=weights,
            loss_is_sum=True,
        )

    def _record_loss_groups(self, token_losses, shifted_labels, loss_group):
        if loss_group is None or not self.is_in_train or not self.model.training:
            return
        groups = torch.as_tensor(loss_group, device=token_losses.device)
        if (
            groups.shape != token_losses.shape[:1]
            or not ((groups == -1) | (groups == 0) | (groups == 1)).all()
        ):
            raise ValueError(
                "loss_group must be -1 (unspecified), 0 (fact), or 1 (QA)."
            )
        per_example_loss = token_losses.detach().sum(dim=-1)
        per_example_count = shifted_labels.ne(-100).sum(dim=-1)
        totals = torch.stack(
            [
                value[groups == group].sum()
                for group in (0, 1)
                for value in (per_example_loss, per_example_count)
            ]
        ).to(torch.float64)
        if self._loss_group_totals is None:
            self._loss_group_totals = totals
        else:
            self._loss_group_totals += totals

    def log(self, logs, start_time=None):
        if "loss" in logs or "train_loss" in logs:
            totals = self._loss_group_totals
            if totals is None:
                totals = torch.zeros(4, dtype=torch.float64, device=self.args.device)
            # Every rank participates, including ranks with no grouped examples.
            totals = self.accelerator.reduce(totals, reduction="sum").cpu().tolist()
            for index, name in enumerate(("ntp", "qa")):
                loss_sum, token_count = totals[index * 2 : index * 2 + 2]
                if token_count:
                    logs[f"{name}_loss"] = loss_sum / token_count
                    logs[f"{name}_supervised_tokens"] = int(token_count)
            self._loss_group_totals = None
        return super().log(logs, start_time)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        """计算标准的 causal language modeling 损失

        Args:
            model: 模型实例
            inputs: 输入 batch，包含 input_ids, attention_mask, labels
            return_outputs: 是否返回模型输出

        Returns:
            loss 或 (loss, outputs) 元组
        """
        # 标准的语言模型损失计算
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        loss = self._compute_causal_loss(outputs.logits, inputs, num_items_in_batch)
        outputs.loss = loss

        return (loss, outputs) if return_outputs else loss

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        """保存模型（包括 PEFT 适配器）

        如果模型使用了 PEFT，则只保存适配器权重。
        否则保存完整模型权重。

        Args:
            output_dir: 输出目录
            _internal_call: 是否为内部调用
        """
        if not self.is_world_process_zero():
            return
        if output_dir is None:
            output_dir = self.args.output_dir

        # 检查是否为 PEFT 模型
        if hasattr(self.model, "save_pretrained") and hasattr(
            self.model, "peft_config"
        ):
            # PEFT 模型：只保存适配器
            self.model.save_pretrained(output_dir)
            logger.info(f"PEFT adapter saved to {output_dir}")
        else:
            # 普通模型：调用父类方法保存完整权重
            super().save_model(output_dir, _internal_call)

        # 保存 tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
