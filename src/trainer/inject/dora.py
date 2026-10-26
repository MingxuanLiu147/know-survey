"""
DoRA 训练器
===========

实现 Weight-Decomposed Low-Rank Adaptation (DoRA) 微调方法。
DoRA 将权重分解为幅度和方向两部分，分别进行调整。

参考论文: DoRA: Weight-Decomposed Low-Rank Adaptation
https://arxiv.org/abs/2402.09353
"""

import logging
from typing import Optional, List

from trainer.inject.base import InjectTrainer

logger = logging.getLogger(__name__)


class DoRATrainer(InjectTrainer):
    """DoRA 微调训练器

    使用 peft 库的 DoRA 配置对模型进行参数高效微调。

    DoRA 核心思想：
    - 将权重 W 分解为幅度 m 和方向 V：W = m * (V / ||V||)
    - 对方向部分应用 LoRA：V' = V + ΔV
    - 幅度 m 单独训练

    相比 LoRA 的优势：
    - 更好地保持预训练权重的方向信息
    - 在相同参数量下通常有更好的性能
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str] | str] = None,
        use_dora: bool = True,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        *args,
        **kwargs,
    ):
        """初始化 DoRA 训练器

        Args:
            r: 低秩矩阵的秩
            lora_alpha: 缩放因子
            lora_dropout: Dropout 比例
            target_modules: 应用 DoRA 的模块名称列表
            use_dora: 是否启用 DoRA（设为 True 使用 DoRA，False 退化为 LoRA）
            bias: 是否训练 bias
            task_type: 任务类型
        """
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = (
            target_modules
            if isinstance(target_modules, str)
            else list(target_modules)
            if target_modules
            else ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.use_dora = use_dora
        self.bias = bias
        self.task_type = task_type

        super().__init__(*args, **kwargs)

        # 应用 DoRA 配置
        self._apply_dora_config()

    def _apply_dora_config(self):
        """应用 DoRA 配置到模型"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            task_type_map = {
                "CAUSAL_LM": TaskType.CAUSAL_LM,
                "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
                "SEQ_CLS": TaskType.SEQ_CLS,
            }

            # DoRA 通过 LoraConfig 的 use_dora 参数启用
            dora_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias=self.bias,
                task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
                use_dora=self.use_dora,  # 启用 DoRA
            )

            # 应用到模型
            self.model = get_peft_model(self.model, dora_config)
            self.model_wrapped = self.model

            self.model.print_trainable_parameters()
            logger.info(
                f"DoRA config applied: r={self.r}, alpha={self.lora_alpha}, use_dora={self.use_dora}"
            )

        except ImportError:
            logger.error("peft library not installed. Please run: pip install peft")
            raise

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 DoRA 微调损失"""
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
