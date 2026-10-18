"""多模态 Gradient Ascent 遗忘方法。

在 forget 数据上做梯度上升（loss = -outputs.loss），使模型"遗忘"对应知识。
"""

import torch

from trainer.unlearn.mm_base import MMUnlearnBase


class MMGradAscent(MMUnlearnBase):
    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        return -outputs.loss
