"""
GradDiff (Gradient Difference) Unlearning 算法
==============================================

GradDiff 是一种经典的 unlearning 方法，通过梯度差分实现选择性遗忘。

核心思想：
    1. Forget 数据：使用负梯度（梯度上升）降低模型对该数据的拟合
    2. Retain 数据：使用正梯度（正常训练）保持模型性能
    3. 两个梯度的差分效应实现选择性遗忘

损失函数：
    Total Loss = γ × (-NLL_forget) + α × Loss_retain

    其中：
    - γ × (-NLL_forget): 梯度上升，增加 forget 数据的损失
    - α × Loss_retain: 正常训练，保持 retain 数据的性能

参数说明：
    - gamma (γ): forget 损失的权重，控制"遗忘"强度
    - alpha (α): retain 损失的权重，控制"保持"强度
    - retain_loss_type: "NLL" 或 "KL"，选择 retain 数据的损失类型
"""

import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer


class GradDiff(UnlearnTrainer):
    """梯度差分 (Gradient Difference) Unlearning Trainer

    这是许多 unlearning 方法的基类（如 SimNPO、NPO 等都继承自它）。

    提供的核心功能：
    1. 加权组合 forget 和 retain 损失
    2. 支持两种 retain 损失类型：NLL 和 KL 散度
    3. 可选的参考模型（用于 KL 散度计算）
    """

    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        """初始化 GradDiff Trainer

        Args:
            gamma (float): forget 损失的权重系数，默认 1.0
                - gamma 越大：遗忘越强（但可能损害模型整体性能）
                - gamma 越小：遗忘越弱（但 retain 数据保持得更好）
                - 典型值：0.1 ~ 1.0

            alpha (float): retain 损失的权重系数，默认 1.0
                - alpha 越大：更注重保持 retain 数据的性能
                - alpha 越小：更注重遗忘 forget 数据
                - 典型值：0.5 ~ 2.0

            retain_loss_type (str): retain 数据的损失类型
                - "NLL": 负对数似然（标准交叉熵损失）
                  → 更简单，无需参考模型，但可能导致过拟合
                - "KL": 与参考模型的 KL 散度
                  → 更稳定，保持模型输出分布不变，但需要额外显存

            *args, **kwargs: 传递给父类 UnlearnTrainer 的参数
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type

        # 参考模型（仅在 retain_loss_type="KL" 时使用）
        # ref_model 是训练开始前的模型副本，用于计算 KL 散度
        # 目的：确保模型在 retain 数据上的输出分布不发生太大变化
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        """准备参考模型（用于 KL 散度计算）

        创建当前模型的深拷贝，并设置为评估模式。
        参考模型的参数在整个训练过程中保持不变。

        Args:
            model: 当前待训练的模型

        Returns:
            ref_model: 冻结的参考模型副本

        注意事项：
            - 参考模型会占用额外的 GPU 显存（与原模型大小相同）
            - 如果使用 DeepSpeed ZeRO-3，参考模型也会被分片
        """
        # 深拷贝模型（包括所有参数）
        ref_model = copy.deepcopy(model).to(self.accelerator.device)

        # 设置为评估模式（关闭 dropout、batch norm 等）
        ref_model.eval()

        # 根据是否使用 DeepSpeed 进行不同的准备
        if self.is_deepspeed_enabled:
            # DeepSpeed 模式：需要特殊的初始化流程
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            # 标准 Accelerate 模式
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)

        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        """计算 retain 数据的损失

        根据 retain_loss_type 选择不同的损失计算方式：
        - "NLL": 标准的负对数似然（交叉熵）
        - "KL": 与参考模型输出的 KL 散度

        Args:
            model: 当前模型
            retain_inputs (dict): retain 数据的输入
                - input_ids: token IDs
                - attention_mask: 注意力掩码
                - labels: 目标标签

        Returns:
            retain_loss (Tensor): 标量损失值
        """
        # 前向传播计算模型输出
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0

        if self.retain_loss_type == "NLL":
            # 负对数似然损失（标准的语言模型训练损失）
            # retain_outputs.loss 已经是平均过的交叉熵损失
            retain_loss += retain_outputs.loss

        elif self.retain_loss_type == "KL":
            # KL 散度：D_KL(P_ref || P_current)
            # 衡量当前模型输出与参考模型输出的差异
            # 目的：确保模型在 retain 数据上的行为不发生太大变化
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss

        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )

        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 GradDiff 的总损失（核心训练逻辑）

        这是标准的 GradDiff 损失函数，许多变体（如 SimNPO）会覆盖此方法。

        Args:
            model: 待训练的模型
            inputs (dict): 包含 forget 和 retain 数据的字典
            return_outputs (bool): 是否返回模型输出

        Returns:
            loss (Tensor): 标量损失值
            outputs (optional): 模型输出

        损失计算公式：
            loss = γ × (-NLL_forget) + α × loss_retain

        解释：
            - -NLL_forget: 负的负对数似然 = 正的对数似然
              → 梯度方向与正常训练相反（梯度上升）
              → 增加模型对 forget 数据的损失（降低拟合度）
            - loss_retain: 正常的训练损失
              → 保持模型在 retain 数据上的性能
        """

        # ==================== 处理 Forget 数据 ====================
        forget_inputs = inputs["forget"]

        # 提取必要的输入字段
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        # 计算 forget 数据的负对数似然
        forget_outputs = model(**forget_inputs)

        # 关键：取负号！这使得梯度方向相反
        # 原始：minimize NLL → 模型学习 forget 数据
        # 取负：minimize (-NLL) → 模型"遗忘" forget 数据
        forget_loss = -forget_outputs.loss

        # ==================== 处理 Retain 数据 ====================
        retain_inputs = inputs["retain"]

        # 提取必要的输入字段
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }

        # 计算 retain 损失（NLL 或 KL）
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # ==================== 组合损失 ====================
        # 加权求和：平衡"遗忘"和"保持"
        # 典型配置：gamma=0.1~1.0, alpha=1.0
        # → 更注重保持 retain 数据的性能
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
