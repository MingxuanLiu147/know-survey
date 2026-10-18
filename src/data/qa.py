import torch
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index


class QADataset(Dataset):
    """面向问答类（QA）数据的通用 Dataset。

    典型用途：TOFU 等基准的 question/answer 数据。
    - 支持 few-shot：可以在目标 QA 前拼接若干 in-context 示例
    - 输出：满足监督微调格式的 input_ids / labels / attention_mask + index
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 从 HuggingFace Hub 或本地缓存中加载原始 QA 数据
        # 示例：hf_args = {"path": "locuslab/TOFU", "name": "forget10", "split": "train"}
        self.data = load_hf_dataset(**hf_args)
        # 为每条数据添加一个 "index" 字段，便于在训练/评估时追踪到原始样本
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        # few-shot 数据（可选）：当提供 few_shot_dataset_hf_args 时，
        # 会把该数据集中的 QA 作为 in-context 示例拼接到当前样本前面
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        """给定单个问答对，构造一条可喂给模型的样本。"""
        if self.fs_data is None:
            # 无 few-shot：只有当前 question/answer
            prompt_msgs, response_msgs = [question], [answer]
        else:
            # 有 few-shot：前面是 in-context 示例，最后一对是当前 QA
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        # 核心预处理：应用聊天模板 + tokenize + 构造 labels/attention_mask
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        # 按 Trainer 期望的格式组织输出
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        """支持两种情况：
        - answer 为单个字符串：返回单条样本
        - answer 为列表：对每个答案都生成一条样本，放在一个 dict 里
        """
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            # 当标注格式不是 str 或 list 时，不支持该类型
            raise NotImplementedError("answer format not found")
        return item


class QAwithIdkDataset(QADataset):
    """在原始 QA 基础上，额外拼接一条“我不知道(idk)”式回答的 Dataset。

    用途：DPO / NPO 等方法中，构造 original vs idk 的偏好对。
    """

    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        # 预加载所有 idk 风格回复（逐行读取）
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        """给定一个 question，随机采样一条 idk 回复并处理成训练样本。"""
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        """返回 original / alternate 两条样本：
        - original: 原始数据集中的答案
        - alternate: 随机采样的一条 idk 风格回答
        """
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            # 单答案场景：直接构造一个 dict 包含两种回答
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        # 根据配置决定是返回 pair，还是只返回 alternate 版本
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    """从同一条样本中读取另一个字段作为 alternate 答案。

    例如 TOFU 中的 paraphrased_answer，可以作为 alternate 回答。
    """

    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]
