# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any

logger = logging.getLogger(__name__)


class FinetuneTrainer(Trainer):
    """
    通用微调训练器类，继承自 HuggingFace Transformers 的 Trainer。

    主要功能扩展：
    1. 支持自定义评估器 (evaluators)：允许在评估阶段运行复杂的自定义评估逻辑，而不仅仅是计算 Loss。
    2. 结果保存：将自定义评估的结果自动保存到对应的 Checkpoint 目录下。
    """

    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        """
        初始化 FinetuneTrainer。

        Args:
            evaluators (Dict, optional): 自定义评估器字典。
            template_args (Any, optional): 模板参数，用于传递给评估器。
            *args, **kwargs: 传递给父类 Trainer 的参数。
        """
        self.evaluators = evaluators
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """
        重写 evaluate 方法，优先支持自定义评估流程。

        流程：
        1. 如果存在自定义评估器 (self.evaluators)：
           - 仅在主进程中运行评估（避免多进程重复计算）。
           - 创建评估结果输出目录 (checkpoint_folder/evals)。
           - 遍历所有评估器运行评估，并收集指标。
           - 记录日志。
        2. 如果没有自定义评估器但提供了 eval_dataset：
           - 回退到 HuggingFace Trainer 的默认评估逻辑。
        3. 否则返回空字典。
        """
        # Run a custom evaluator and save results
        if self.evaluators:
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    eval_metrics = {}
                    for _, evaluator in self.evaluators.items():
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        eval_metrics.update(evaluator.evaluate(**eval_args))
                    self.log(eval_metrics)
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
