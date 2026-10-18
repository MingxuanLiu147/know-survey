"""
SimNPO (Simplified Negative Preference Optimization) Unlearning 算法
====================================================================

SimNPO 是一种基于偏好优化的 unlearning 方法，通过负面偏好优化来"遗忘"特定数据。

核心思想：
    1. 对于 forget 数据：使模型输出该数据的概率降低（负面偏好）
    2. 对于 retain 数据：保持模型性能不变（正常训练）
    3. 使用 logsigmoid 函数平滑优化过程，避免梯度爆炸

损失函数：
    Total Loss = γ × Forget Loss + α × Retain Loss

    其中：
    - Forget Loss = -logsigmoid(β × (NLL - δ)) × 2/β
      → 鼓励模型对 forget 数据产生高 NLL（低概率）
    - Retain Loss = 标准的负对数似然损失（NLL）或 KL 散度
      → 保持模型在 retain 数据上的性能

参数说明：
    - delta (δ): NLL 的目标偏移量，控制"遗忘"的程度
    - beta (β): 温度参数，控制优化的平滑程度（β 越大，梯度越陡峭）
    - gamma (γ): forget 损失的权重（继承自 GradDiff）
    - alpha (α): retain 损失的权重（继承自 GradDiff）
"""

import torch.nn.functional as F

from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff


class SimNPO(GradDiff):
    """简化的负面偏好优化 (Simplified Negative Preference Optimization)

    继承自 GradDiff 基类，在其基础上引入了：
    - delta: NLL 偏移量（控制遗忘强度）
    - beta: 温度参数（控制优化平滑度）

    SimNPO 通过 logsigmoid 函数将 forget loss 转换为偏好优化形式，
    相比简单的梯度上升更加稳定。
    """

    def __init__(self, delta=0.0, beta=1.0, *args, **kwargs):
        """初始化 SimNPO Trainer

        Args:
            delta (float): NLL 偏移量，默认 0.0
                - delta=0: 标准 SimNPO
                - delta>0: 更强的遗忘（允许更高的 NLL）
                - delta<0: 更弱的遗忘
            beta (float): 温度参数，默认 1.0
                - beta 越大：梯度越陡峭，遗忘越快（但可能不稳定）
                - beta 越小：梯度越平滑，遗忘越慢（但更稳定）
            *args, **kwargs: 传递给父类的参数
                - gamma: forget loss 权重（从 GradDiff 继承）
                - alpha: retain loss 权重（从 GradDiff 继承）
                - retain_loss_type: "NLL" 或 "KL"（从 GradDiff 继承）
        """
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """计算 SimNPO 的总损失

        这是训练循环中每个 batch 都会调用的核心函数。

        Args:
            model: 待训练的语言模型
            inputs (dict): 包含 forget 和 retain 两个键的字典
                - inputs["forget"]: forget 数据的 batch
                - inputs["retain"]: retain 数据的 batch
            return_outputs (bool): 是否返回模型输出（用于日志记录）

        Returns:
            loss (Tensor): 标量损失值（用于反向传播）
            outputs (optional): 模型输出（如果 return_outputs=True）

        处理流程：
            1. 计算 forget 数据的负面偏好损失
            2. 计算 retain 数据的保持损失
            3. 加权组合两个损失
        """

        # ==================== 处理 Forget 数据 ====================
        forget_inputs = inputs["forget"]

        # 提取标签（用于计算有效 token 数量）
        # labels 中 -100 表示不参与损失计算的 token（如 padding）
        forget_labels = forget_inputs["labels"]
        loss_mask = forget_labels != -100  # 标记哪些 token 参与计算

        # 计算每个样本的负对数似然 (NLL)
        # NLL 越高 → 模型对该样本的概率越低 → 越"遗忘"
        # forget_loss shape: (batch_size,)
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)

        # 归一化：除以有效 token 数量，得到平均 per-token NLL
        forget_loss = forget_loss / loss_mask.sum(-1)  # shape: (batch_size,)

        # 应用 delta 偏移
        # delta > 0 时，相当于提高了"遗忘"的目标 NLL
        forget_loss = forget_loss - self.delta

        # SimNPO 的核心转换：
        # -logsigmoid(β × NLL) = log(1 + exp(-β × NLL))
        # 效果：将 NLL 转换为偏好优化形式
        # - NLL 高 → logsigmoid 接近 0 → 损失接近 0（已经"遗忘"）
        # - NLL 低 → logsigmoid 接近 -∞ → 损失高（需要继续"遗忘"）
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta

        # ==================== 处理 Retain 数据 ====================
        retain_inputs = inputs["retain"]

        # 提取必要的输入字段
        # （某些 unlearning 方法可能在 inputs 中包含额外字段）
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],  # token IDs
            "attention_mask": retain_inputs["attention_mask"],  # 注意力掩码
            "labels": retain_inputs["labels"],  # 目标标签
        }

        # 计算 retain 损失（继承自 GradDiff）
        # 可以是标准的 NLL 或与参考模型的 KL 散度
        # 目的：保持模型在 retain 数据上的性能
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # ==================== 组合损失 ====================
        # 总损失 = gamma × forget损失 + alpha × retain损失
        # - gamma (self.gamma): 控制"遗忘"的强度
        # - alpha (self.alpha): 控制"保持"的强度
        # 典型值：gamma=0.125, alpha=1.0 → 更关注保持性能
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
