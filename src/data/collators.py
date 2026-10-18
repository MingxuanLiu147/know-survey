import torch
import transformers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX


class DataCollatorForSupervisedDataset(object):
    """用于有监督微调的通用 Collator。

    既支持“扁平”样本（直接包含 input_ids/labels），也支持嵌套 dict 结构，
    例如遗忘场景下的 {"forget": {...}, "retain": {...}}，通过递归方式对每个子键单独做 padding。
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        # 用于决定 pad token id 以及后处理
        self.tokenizer = tokenizer
        # left/right padding，由模型和训练配置决定
        self.padding_side = padding_side
        # 可选：如果不为 None，则会从样本中抽取该字段并堆叠（如 "index"）
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        """从一批样本中抽取同名子字段，形成新的 list，方便递归处理。"""
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        """对一批不等长序列做 padding，支持左右两种 padding 方式。"""
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            # 左 padding：先翻转、右侧 padding，再翻转回来
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def _stack_optional_scalar(self, instances, key: str):
        if key not in instances[0]:
            return None
        values = [instance[key] for instance in instances]
        if key == "sample_weight":
            return torch.tensor(values, dtype=torch.float32)
        return torch.tensor(values, dtype=torch.long)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """将若干条样本打包为一个 batch。

        - 如果样本是嵌套 dict（没有 input_ids），则对每个子键递归调用自身
        - 如果样本已经是叶子（包含 input_ids），则做真正的 padding/堆叠
        """
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            # 嵌套结构：例如 {"forget": {...}, "retain": {...}}
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                # 对每个子键单独 collate，形成多层嵌套的 batch dict
                return_dct[key] = self(key_instances)
        else:
            # 叶子结构：真正执行 padding 和 stack
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                # labels 的 padding 使用 IGNORE_INDEX，以免影响损失
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            for scalar_key in (
                "prompt_length",
                "response_start",
                "train_target_span",
                "sample_weight",
            ):
                scalar_value = self._stack_optional_scalar(instances, scalar_key)
                if scalar_value is not None:
                    return_dct.update({scalar_key: scalar_value})
            if self.index:
                # 如果需要把 index 一并 collate 出来（如样本追踪）
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    # 仅做 Warning，不直接中断训练
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


class DataCollatorForReFTDataset(DataCollatorForSupervisedDataset):
    """为 ReFT 训练补充最后一个 prompt token 的 intervention 位置。"""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        if "input_ids" in batch and "prompt_length" in batch:
            prompt_lengths = torch.clamp(batch["prompt_length"] - 1, min=0)
            batch["unit_locations"] = {
                "sources->base": (
                    None,
                    [[[int(position.item())]] for position in prompt_lengths],
                )
            }
        return batch
