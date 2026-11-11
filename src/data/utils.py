import os
from pathlib import Path

import datasets
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Union

IGNORE_INDEX = (
    -100
)  # 训练中不计算损失的位置统一用 -100（与 Transformers 默认约定保持一致）

logger = logging.getLogger("data")


def tokenize_causal_pair(
    tokenizer,
    prompt: str,
    response: str,
    *,
    add_special_tokens: bool = True,
) -> tuple[list[int], list[int]]:
    """Encode a causal pair while preserving the exact generation prompt IDs.

    A jointly encoded continuation is reused only when its prompt IDs match
    exactly, preserving safe SentencePiece and whitespace behavior. Otherwise
    encode the response separately to prevent a boundary merge from absorbing
    its first token. Explicit EOS tokens are preserved. Truncation belongs to
    the caller.
    """

    def encode(text, special_tokens):
        ids = tokenizer(text, add_special_tokens=special_tokens)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("Causal pair tokenization requires one sequence.")
            ids = ids[0]
        return list(ids)

    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("Causal prompt and response must be strings.")
    prompt_ids = encode(prompt, add_special_tokens)
    full_ids = encode(prompt + response, add_special_tokens)
    if full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids, full_ids[len(prompt_ids) :]
    return prompt_ids, encode(response, False)


def append_eos_target(prompt_ids, response_ids, tokenizer, max_length):
    """Terminate a complete target, rejecting truncation that would teach a false end."""
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("append_eos requires tokenizer.eos_token_id.")
    response_ids = list(response_ids)
    if not response_ids or response_ids[-1] != eos:
        response_ids.append(eos)
    if len(prompt_ids) + len(response_ids) > max_length:
        raise ValueError(
            "Complete training target plus EOS exceeds max_length; increase the "
            "limit or explicitly filter this sample instead of truncating its end."
        )
    return response_ids


def load_hf_dataset(path, **kwargs):
    """封装 datasets.load_dataset 的简单辅助函数。

    这里默认开启 force_redownload，以避免部分环境下 cache 校验失败的问题。
    """
    # TOFU's public JSON snapshot can be used without Hub metadata/network. This
    # keeps the canonical configs (and their named splits) unchanged while an
    # auditable run records the exact snapshot directory in TOFU_LOCAL_DIR.
    tofu_local_dir = os.getenv("TOFU_LOCAL_DIR")
    if path == "locuslab/TOFU" and tofu_local_dir:
        config_name = kwargs.pop("name", None)
        if not config_name:
            raise ValueError("A named TOFU split is required with TOFU_LOCAL_DIR")
        data_file = Path(tofu_local_dir).expanduser() / f"{config_name}.json"
        if not data_file.is_file():
            raise FileNotFoundError(
                f"TOFU_LOCAL_DIR does not contain split `{config_name}`: {data_file}"
            )
        path = "json"
        kwargs["data_files"] = str(data_file)

    # Force redownload to avoid consistency check errors due to network/cache issues
    if "download_mode" not in kwargs:
        kwargs["download_mode"] = "force_redownload"
    dataset = datasets.load_dataset(path, **kwargs)
    return dataset


def preprocess_chat_instance(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    max_length: int,
    predict_with_generate: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_config (Dict[str, Any]): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    if isinstance(prompt_msgs, str):
        if not isinstance(response_msgs, str):
            raise ValueError("A string prompt requires a string response.")
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]
    if isinstance(response_msgs, str) or len(prompt_msgs) != len(response_msgs):
        raise ValueError("Prompt and response messages must be paired.")
    if not prompt_msgs:
        raise ValueError("At least one prompt/response pair is required.")
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")

    # 分两种模式：
    # 1）使用 tokenizer.apply_chat_template（推荐，适配 Llama 等 chat 模型）
    # 2）手工拼接 user/asst 起止 tag
    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]
        # 某些模板会使用日期信息（可选）
        date_str = template_config.get("date_string", None)
        date_info = {"date_string": date_str} if date_str is not None else {}
        full_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=False, **date_info
        )
        # all except last response are in-context examples
        # wrapped_prompt：只包含到“最后一条 user 提示”为止的文本，用于构造 prompt_ids
        wrapped_prompt = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True, **date_info
        )
        if not full_text.startswith(wrapped_prompt):
            raise ValueError(
                "The chat template does not preserve the generation prompt "
                "as a prefix of the full conversation."
            )
        prompt_ids, response_ids = tokenize_causal_pair(
            tokenizer,
            wrapped_prompt,
            full_text[len(wrapped_prompt) :],
            add_special_tokens=False,
        )
    else:
        # 不使用内置 chat_template，手动按照自定义模板拼接
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_config.get(
            "system_prompt_with_special_tokens", None
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        # add in-context examples
        n_few_shot = len(prompt_msgs) - 1
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_start_tag"]
                + fs_response
                + template_config["asst_end_tag"]
            )

        # add actual example
        # 最后一对是“真正要训练/评估”的 QA，其余视为 in-context 示例
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + final_prompt
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
        )
        prompt_ids, response_ids = tokenize_causal_pair(
            tokenizer,
            wrapped_prompt,
            final_response,
            add_special_tokens=True,
        )

    chat_ids = prompt_ids + response_ids
    # 确保以 EOS 结尾，便于模型学习停止位置
    if not chat_ids:
        raise ValueError("The prompt/response pair produced no tokens.")
    if tokenizer.eos_token_id is not None and chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]
    chat_ids = chat_ids[:max_length]

    len_matched = len(prompt_ids)
    if len_matched >= len(chat_ids):
        raise ValueError("The complete prompt leaves no trainable response tokens.")

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
        labels = chat_ids  # contains the entire conversation
    else:
        # 训练模式：input_ids = 完整对话；labels 只在“答案 token”上计算损失
        item["input_ids"] = chat_ids
        len_matched = max(len_matched, 1)
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
        if not any(label != IGNORE_INDEX for label in labels[1:]):
            raise ValueError("The prompt/response pair has no causal target tokens.")

    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def preprocess_pretraining_instance(
    tokenizer,
    prefix: str,
    text_content: str,
    max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a pretraining instance for training or generation.
    When in training, both the returned `input_ids` and `labels` are over the entire token sequence. `input_ids` has no padding, `labels` assigns `IGNORE_INDEX` to ignore all tokens that we don't compute loss over (i.e. the the 0th index token, all prefix tokens)
    When in generation, `input_ids` are returned only until the prefix portion. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full input token sequence.
    Args:
        tokenizer: Tokenizer to apply on text
        prefix (str): The prefix string to prepend to the content.
        text_content (str): The main text content (following the prefix) to be tokenized.
        max_length (int): Maximum text content length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.
        insert_space (bool, optional): Whether to insert a space between prefix and content.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    if max_length < 1:
        raise ValueError("max_length must allow at least one content token.")
    prefix_ids, content_ids = tokenize_causal_pair(
        tokenizer,
        prefix,
        (" " if insert_space else "") + text_content,
        add_special_tokens=True,
    )
    prefix_len = len(prefix_ids)
    full_seq_ids = prefix_ids + content_ids[:max_length]

    len_matched = prefix_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    # prefix 部分全部置为 IGNORE_INDEX，只在正文部分计算损失
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError("Pretraining sample has no causal target tokens.")
    item = {}
    if predict_with_generate:
        item["input_ids"] = prefix_ids
    else:
        # 训练模式：input_ids 是整段序列
        item["input_ids"] = full_seq_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    # 转成 torch.Tensor，方便 DataCollator 后续堆叠
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


# 为 HF Dataset 添加一个自增 index 列，便于在训练/评测阶段追踪样本来源
def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset
