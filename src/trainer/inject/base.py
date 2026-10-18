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
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
            labels=inputs["labels"],
        )
        loss = outputs.loss

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
