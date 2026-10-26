"""
AdaLoRA 训练器
==============

实现 Adaptive Low-Rank Adaptation (AdaLoRA) 微调方法。
AdaLoRA 通过自适应地调整不同层的秩来优化参数效率。

参考论文: AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
https://arxiv.org/abs/2303.10512

核心思想：
1. 使用 SVD 分解来表示增量更新
2. 根据重要性得分动态调整每层的秩
3. 在训练过程中逐步剪枝不重要的奇异值
"""

import logging
import math
from typing import Optional, List

from transformers import TrainerCallback

from trainer.inject.base import InjectTrainer

logger = logging.getLogger(__name__)


class _AdaLoraRankCallback(TrainerCallback):
    """Run PEFT's rank allocator while gradients are still available."""

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        model = getattr(model, "module", model)
        base_model = getattr(model, "base_model", None)
        update = getattr(base_model, "update_and_allocate", None)
        if update is not None:
            config = next(iter(base_model.peft_config.values()))
            if state.global_step == 0 and config.total_step - config.tfinal == 1:
                # ponytail: prime PEFT's importance cache for one-step smoke runs.
                update(0)
            update(state.global_step + 1)
        return control


class AdaLoRATrainer(InjectTrainer):
    """AdaLoRA 自适应微调训练器

    通过自适应秩分配实现更高效的参数微调。

    AdaLoRA vs LoRA：
    - LoRA：固定秩，所有层使用相同的 r
    - AdaLoRA：自适应秩，根据重要性动态调整

    Attributes:
        init_r: 初始秩（训练开始时的秩）
        target_r: 目标秩（训练结束时的秩）
        tinit: 秩调整的初始训练步数
        tfinal: 秩调整的结束训练步数
        deltaT: 秩更新的间隔步数
    """

    def __init__(
        self,
        init_r: int = 12,
        target_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str] | str] = None,
        tinit: int = 0,
        tfinal: int = 0,
        deltaT: int = 1,
        beta1: float = 0.85,
        beta2: float = 0.85,
        orth_reg_weight: float = 0.5,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        *args,
        **kwargs,
    ):
        """初始化 AdaLoRA 训练器

        Args:
            init_r: 初始秩（开始训练时）
            target_r: 目标秩（训练结束时，通过剪枝达到）
            lora_alpha: 缩放因子
            lora_dropout: Dropout 比例
            target_modules: 应用 AdaLoRA 的模块
            tinit: 开始调整秩的训练步数
            tfinal: 停止调整秩的训练步数
            deltaT: 秩更新间隔
            beta1: 重要性得分的 EMA 系数（奇异值）
            beta2: 重要性得分的 EMA 系数（敏感度）
            orth_reg_weight: 正交正则化权重
            bias: 是否训练 bias
            task_type: 任务类型
        """
        self.init_r = init_r
        self.target_r = target_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = (
            target_modules
            if isinstance(target_modules, str)
            else list(target_modules)
            if target_modules
            else ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.tinit = tinit
        self.tfinal = tfinal
        self.deltaT = deltaT
        self.beta1 = beta1
        self.beta2 = beta2
        self.orth_reg_weight = orth_reg_weight
        self.bias = bias
        self.task_type = task_type

        super().__init__(*args, **kwargs)

        # 应用 AdaLoRA 配置
        self._apply_adalora_config()
        self.add_callback(_AdaLoraRankCallback())

    def _get_total_training_steps(self) -> int:
        """为 AdaLoRA 推导总训练步数。"""
        if getattr(self.args, "max_steps", -1) and self.args.max_steps > 0:
            return self.args.max_steps

        if self.train_dataset is None:
            raise ValueError(
                "AdaLoRA requires `max_steps` or a sized train_dataset to infer "
                "`total_step`."
            )

        world_size = max(getattr(self.args, "world_size", 1), 1)
        effective_batch = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * world_size
        )
        steps_per_epoch = max(math.ceil(len(self.train_dataset) / effective_batch), 1)
        return max(math.ceil(self.args.num_train_epochs * steps_per_epoch), 1)

    def _apply_adalora_config(self):
        """应用 AdaLoRA 配置到模型"""
        try:
            from peft import AdaLoraConfig, get_peft_model, TaskType

            task_type_map = {
                "CAUSAL_LM": TaskType.CAUSAL_LM,
                "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
                "SEQ_CLS": TaskType.SEQ_CLS,
            }

            # 创建 AdaLoRA 配置
            adalora_config = AdaLoraConfig(
                init_r=self.init_r,
                target_r=self.target_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                tinit=self.tinit,
                tfinal=self.tfinal,
                deltaT=self.deltaT,
                total_step=self._get_total_training_steps(),
                beta1=self.beta1,
                beta2=self.beta2,
                orth_reg_weight=self.orth_reg_weight,
                bias=self.bias,
                task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            )

            # 应用到模型
            self.model = get_peft_model(self.model, adalora_config)
            self.model_wrapped = self.model

            self.model.print_trainable_parameters()
            logger.info(
                f"AdaLoRA config applied: init_r={self.init_r}, "
                f"target_r={self.target_r}, alpha={self.lora_alpha}"
            )

        except ImportError:
            logger.error("peft library not installed. Please run: pip install peft")
            raise

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 AdaLoRA 微调损失。

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
        # PEFT's AdaLoraModel.forward already adds the configured orthogonal
        # regularizer. Adding it again here changes the published objective.
        return (outputs.loss, outputs) if return_outputs else outputs.loss
