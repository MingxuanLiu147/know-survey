"""
LoRA 训练器
===========

实现 Low-Rank Adaptation (LoRA) 微调方法。
LoRA 通过在 Transformer 层中添加低秩分解矩阵来实现参数高效微调。

参考论文: LoRA: Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685
"""

import logging
from typing import Optional, List

from trainer.inject.base import InjectTrainer

logger = logging.getLogger(__name__)


class LoRATrainer(InjectTrainer):
    """LoRA 微调训练器

    使用 peft 库的 LoRA 配置对模型进行参数高效微调。

    LoRA 核心思想：
    - 冻结预训练权重 W
    - 添加低秩分解 ΔW = BA，其中 B ∈ R^(d×r), A ∈ R^(r×k)
    - 推理时合并：W' = W + ΔW

    Attributes:
        r: 低秩矩阵的秩
        lora_alpha: LoRA 缩放因子
        lora_dropout: Dropout 比例
        target_modules: 应用 LoRA 的目标模块
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        *args,
        **kwargs,
    ):
        """初始化 LoRA 训练器

        Args:
            r: 低秩矩阵的秩，控制可训练参数量
            lora_alpha: 缩放因子，实际缩放为 lora_alpha/r
            lora_dropout: Dropout 比例，用于正则化
            target_modules: 应用 LoRA 的模块名称列表，如 ["q_proj", "v_proj"]
            bias: 是否训练 bias，可选 "none", "all", "lora_only"
            task_type: 任务类型，如 "CAUSAL_LM"
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = (
            list(target_modules) if target_modules else ["q_proj", "v_proj"]
        )
        self.bias = bias
        self.task_type = task_type

        super().__init__(*args, **kwargs)

        # 在初始化后应用 LoRA 配置
        self._apply_lora_config()

    def _apply_lora_config(self):
        """应用 LoRA 配置到模型

        使用 peft 库创建 LoRA 配置并应用到模型。
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # 创建 LoRA 配置
            task_type_map = {
                "CAUSAL_LM": TaskType.CAUSAL_LM,
                "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
                "SEQ_CLS": TaskType.SEQ_CLS,
            }

            lora_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias=self.bias,
                task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            )

            # 应用 LoRA 到模型
            self.model = get_peft_model(self.model, lora_config)
            self.model_wrapped = self.model

            # 打印可训练参数信息
            self.model.print_trainable_parameters()
            logger.info(f"LoRA config applied: r={self.r}, alpha={self.lora_alpha}")

        except ImportError:
            logger.error("peft library not installed. Please run: pip install peft")
            raise

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 LoRA 微调损失

        标准的 causal language modeling 损失。

        Args:
            model: PEFT 包装后的模型
            inputs: 输入 batch
            return_outputs: 是否返回输出

        Returns:
            loss 或 (loss, outputs)
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
