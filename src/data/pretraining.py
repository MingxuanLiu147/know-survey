from torch.utils.data import Dataset
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    """用于 prefix + completion 形式的数据集（类似 GPT-style completion 任务）。

    - prefix_key: 前缀字段（如 "prompt"）
    - text_key:   需要模型续写的正文字段（如 "text"）
    """

    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载原始数据并添加 index 列，index 可在评估/日志中使用
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # 如果后续访问不到对应 key，将使用 "" 作为默认前缀/正文
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        """给定一条 prefix + 文本内容，构造成可训练样本。"""
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        """支持缺省字段的健壮访问：
        - 如果 prefix_key 不存在，使用空字符串代替
        - 如果 text_key 不存在，同样使用空字符串
        """
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 预训练场景：把大量 text 拼成一长串，再按固定长度切成多个 chunk
        self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _chunk_raw_text(self, raw_text):
        """将原始多条文本合并并按 token 长度切块。

        处理步骤：
        1. 用两个换行拼接所有样本，形成连续语料
        2. 整体 tokenize 成一个长的 token 序列
        3. 按 max_length 把 token 序列等长切分成多个 chunk
        4. 再 decode 回文本，后面每个 chunk 单独走 preprocess_pretraining_instance
        """
        # 1）样本间加空行，弱化样本边界
        raw_text = "\n\n".join(raw_text)
        # 2）整体编码，不加特殊 token
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        # 3）按 max_length 计算 chunk 数量
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            # 4）切分出每一段 token，并 decode 回文本
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        # 整个语料被切成多少个 chunk，就有多少个训练样本
        return len(self.chunks)

    def __getitem__(self, idx):
        # prefix 为空串，表示对整段 chunk 的所有 token 计算损失
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )
