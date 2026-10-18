"""
ForgetRetainDataset - Unlearning 场景的组合数据集
================================================

在 unlearning 任务中，我们需要同时处理两类数据：
1. Forget 数据：需要被"遗忘"的敏感或不希望模型记住的数据
2. Retain 数据：需要保持模型性能的正常数据

ForgetRetainDataset 将这两类数据组合在一起，每次返回一对样本：
    {"forget": forget_sample, "retain": retain_sample}

这种设计使得 unlearning 算法可以在一个训练循环中同时：
- 对 forget 数据执行"遗忘"操作（如梯度上升）
- 对 retain 数据执行正常训练（保持性能）

数据采样策略：
    通过 anchor 参数控制采样行为：
    - anchor="forget": 顺序遍历 forget 数据，随机采样 retain 数据
    - anchor="retain": 顺序遍历 retain 数据，随机采样 forget 数据
"""

import torch
from torch.utils.data import Dataset


class ForgetRetainDataset(Dataset):
    """遗忘场景下使用的组合数据集封装

    将 forget 和 retain 两个独立的数据集组合成一个数据集，
    每次 __getitem__ 返回一对样本，供 unlearning 算法同时处理。

    示例：
        >>> forget_ds = QADataset(...)  # 100 条需要遗忘的数据
        >>> retain_ds = QADataset(...)  # 900 条需要保持的数据
        >>> unlearn_ds = ForgetRetainDataset(forget_ds, retain_ds, anchor="forget")
        >>> len(unlearn_ds)  # 100（锚定在 forget）
        >>> item = unlearn_ds[0]
        >>> # item = {
        >>> #     "forget": forget_ds[0],      # 第0条 forget 数据
        >>> #     "retain": retain_ds[random]   # 随机一条 retain 数据
        >>> # }
    """

    # 参考实现：https://github.com/OPTML-Group/SOUL/blob/main/src/dataset/Base.py
    def __init__(self, forget, retain, anchor="forget"):
        """初始化组合数据集

        Args:
            forget (Dataset): Forget 数据集
                - 包含需要被"遗忘"的样本
                - 例如：特定作者的问答对、敏感信息等

            retain (Dataset): Retain 数据集
                - 包含需要保留模型性能的正常样本
                - 例如：其他作者的问答对、通用知识等

            anchor (str): 锚定的数据集，可选 "forget" 或 "retain"
                - "forget": 数据集长度 = len(forget)
                  → 每个 epoch 完整遍历所有 forget 数据
                  → retain 数据随机采样（可能重复或遗漏某些样本）

                - "retain": 数据集长度 = len(retain)
                  → 每个 epoch 完整遍历所有 retain 数据
                  → forget 数据随机采样

        TOFU 示例配置：
            forget10: 176 条样本（10% 作者）
            retain90: 1584 条样本（90% 作者）
            anchor="forget" → 每个 epoch 训练 176 步
        """
        self.forget = forget
        self.retain = retain
        # anchor 只允许取 "forget" 或 "retain"
        self.anchor = anchor

    def __len__(self):
        """返回数据集长度（等于锚定数据集的长度）

        Returns:
            int: 数据集长度

        训练影响：
            - 决定每个 epoch 的训练步数
            - 步数 = len(dataset) // (batch_size × num_gpus × grad_accum_steps)
        """
        if self.anchor == "forget":
            assert self.forget is not None, ValueError(
                "forget dataset can't be None when anchor=forget"
            )
            # 长度等于 forget 的样本数
            return len(self.forget)
        elif self.anchor == "retain":
            assert self.retain is not None, ValueError(
                "retain dataset can't be None when anchor=retain"
            )
            # 长度等于 retain 的样本数
            return len(self.retain)
        else:
            raise NotImplementedError(f"{self.anchor} can be only forget or retain")

    def __getitem__(self, idx):
        """获取一对 (forget, retain) 样本

        Args:
            idx (int): 索引（范围：0 到 len(self)-1）

        Returns:
            dict: 包含 forget 和 retain 样本的字典
                {
                    "forget": {
                        "input_ids": Tensor,      # shape: (seq_len,)
                        "attention_mask": Tensor,  # shape: (seq_len,)
                        "labels": Tensor,          # shape: (seq_len,)
                    },
                    "retain": {
                        "input_ids": Tensor,
                        "attention_mask": Tensor,
                        "labels": Tensor,
                    }
                }

        采样策略：
            anchor="forget":
                - forget: 按顺序访问 forget[idx]（确保所有数据都被访问）
                - retain: 随机采样 retain[random]（每次可能不同）

            anchor="retain":
                - retain: 按顺序访问 retain[idx]
                - forget: 随机采样 forget[random]

        为什么要随机采样非锚定数据？
            1. 避免固定配对：防止模型学到 forget-retain 之间的虚假关联
            2. 数据增强：同一个锚定样本在不同 epoch 配对不同的样本
            3. 平衡数据：当两个数据集大小不同时，通过随机采样平衡
        """
        item = {}
        if self.anchor == "forget":
            # 锚定 forget：按顺序遍历 forget，retain 每次随机采一个
            item["forget"] = self.forget[idx]
            if self.retain:
                # 从 retain 数据集中随机采样一个索引
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
        elif self.anchor == "retain":
            # 锚定 retain：反过来，retain 顺序，forget 随机
            item["retain"] = self.retain[idx]
            if self.forget:
                # 从 forget 数据集中随机采样一个索引
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
        return item
