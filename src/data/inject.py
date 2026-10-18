"""
Knowledge Injection 数据集
=========================

支持知识注入（微调）任务的数据集类。
支持 Alpaca、ShareGPT 等主流数据格式。

数据格式支持：
- Alpaca: instruction, input, output
- ShareGPT: conversations (多轮对话)
- 自定义: 可配置的字段映射
"""

import json
import logging
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)
_MISSING = object()


def _align_chat_training_text(
    prompt: str,
    rendered_full_text: str,
    output: str,
    *,
    tokenizer,
    template_args: Dict[str, Any],
) -> str:
    """Keep training text prefixed by the exact generation prompt."""
    if rendered_full_text.startswith(prompt):
        return rendered_full_text
    response_prefix = template_args.get("response_prefix", "")
    return prompt + response_prefix + output + (tokenizer.eos_token or "")


def render_alpaca_texts(
    instruction: str,
    input_text: str = "",
    output: str | None = None,
    *,
    tokenizer=None,
    template_args: Optional[Dict[str, Any]] = None,
) -> tuple[str, str | None]:
    """Render one Alpaca item identically for training and evaluation."""
    template_args = template_args or {}
    if (
        tokenizer is not None
        and template_args.get("apply_chat_template", False)
        and hasattr(tokenizer, "apply_chat_template")
    ):
        user_content = "\n\n".join(part for part in (instruction, input_text) if part)
        messages = [{"role": "user", "content": user_content}]
        system_prompt = template_args.get("system_prompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        def render(items, add_generation_prompt):
            try:
                return tokenizer.apply_chat_template(
                    items,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except TypeError:
                return tokenizer.apply_chat_template(items, tokenize=False)

        prompt = render(messages, True)
        full_text = None
        if output is not None:
            full_text = render(
                messages + [{"role": "assistant", "content": output}], False
            )
            full_text = _align_chat_training_text(
                prompt,
                full_text,
                output,
                tokenizer=tokenizer,
                template_args=template_args,
            )
        return prompt, full_text

    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
            "\n\n### Response:\n"
        )
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt, None if output is None else prompt + output


def tokenize_inject_pair(
    prompt: str,
    full_text: str,
    *,
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """Tokenize one injection pair while reserving room for its response."""
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    response_ids = full_ids[len(prompt_ids) :]
    if response_ids.numel() == 0:
        raise ValueError("Injection sample has no response tokens to train on.")

    response_budget = min(response_ids.numel(), max(max_length - 1, 1))
    prompt_budget = min(prompt_ids.numel(), max_length - response_budget)
    response_budget = max_length - prompt_budget
    input_ids = torch.cat((prompt_ids[:prompt_budget], response_ids[:response_budget]))
    prompt_len = int(prompt_budget)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone().unsqueeze(0)
    labels[0, :prompt_len] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels.squeeze(0),
        "prompt_length": prompt_len,
        "response_start": prompt_len,
        "train_target_span": int((labels != -100).sum().item()),
    }


class InjectDataset(Dataset):
    """知识注入数据集基类

    支持多种数据格式，用于参数高效微调。

    Attributes:
        data: 样本列表
        tokenizer: 分词器
        max_length: 最大序列长度
        format_type: 数据格式类型 (alpaca/sharegpt/custom)
    """

    def __init__(
        self,
        hf_args: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        format_type: str = "alpaca",
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        conversations_key: str = "conversations",
        tokenizer=None,
        max_length: int = 2048,
        template_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """初始化注入数据集

        Args:
            hf_args: HuggingFace 数据集参数
            data_path: 本地数据路径
            format_type: 数据格式 (alpaca/sharegpt/custom)
            instruction_key: 指令字段名，支持点路径（如 question.problem）
            input_key: 输入字段名，支持点路径
            output_key: 输出字段名，支持点路径
            conversations_key: 对话字段名（ShareGPT 格式）
            tokenizer: 分词器
            max_length: 最大长度
            template_args: 模板参数
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args or {}
        self.format_type = format_type

        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.conversations_key = conversations_key

        # 加载数据
        self.data = self._load_data(hf_args, data_path)

        logger.info(
            f"InjectDataset loaded with {len(self.data)} samples, format={format_type}"
        )

    def _load_data(
        self, hf_args: Optional[Dict[str, Any]], data_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """加载数据"""
        if data_path:
            with open(data_path, "r", encoding="utf-8") as f:
                if data_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            return data
        elif hf_args:
            dataset = load_dataset(**hf_args)
            return list(dataset)
        else:
            logger.warning("No data source specified")
            return []

    def _resolve_field_value(self, item: Any, key: str) -> Any:
        """解析字段值，支持 a.b.0 这样的点路径。"""
        if not isinstance(key, str):
            return _MISSING

        if isinstance(item, dict) and key in item:
            return item[key]

        current = item
        for part in key.split("."):
            if isinstance(current, dict):
                if part not in current:
                    return _MISSING
                current = current[part]
                continue

            if isinstance(current, list):
                try:
                    index = int(part)
                except (TypeError, ValueError):
                    return _MISSING
                if index < 0 or index >= len(current):
                    return _MISSING
                current = current[index]
                continue

            return _MISSING

        return current

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        item = self.data[idx]

        if self.format_type == "alpaca":
            features = self._process_alpaca(item, idx)
        elif self.format_type == "sharegpt":
            features = self._process_sharegpt(item, idx)
        else:
            features = self._process_custom(item)

        sample_weight = item.get("sample_weight", item.get("weight"))
        if sample_weight is not None:
            try:
                features["sample_weight"] = float(sample_weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"InjectDataset[{idx}] sample weight must be numeric, "
                    f"got {sample_weight!r}."
                ) from exc

        return features

    def _require_text_field(
        self,
        item: Dict[str, Any],
        key: str,
        idx: int,
        *,
        allow_empty: bool = False,
    ) -> str:
        """校验并返回字符串字段，尽早暴露坏样本。"""
        value = self._resolve_field_value(item, key)
        if value is _MISSING:
            raise ValueError(
                f"InjectDataset[{idx}] missing required field '{key}' "
                f"for format '{self.format_type}'."
            )

        if value is None:
            raise ValueError(
                f"InjectDataset[{idx}] field '{key}' is None "
                f"for format '{self.format_type}'."
            )

        if not isinstance(value, str):
            value = str(value)
        value = value.strip()

        if not allow_empty and not value:
            raise ValueError(
                f"InjectDataset[{idx}] field '{key}' is empty "
                f"for format '{self.format_type}'."
            )
        return value

    def _optional_text_field(self, item: Dict[str, Any], key: str) -> str:
        value = self._resolve_field_value(item, key)
        if value is _MISSING or value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    def _should_apply_chat_template(self) -> bool:
        return bool(
            self.tokenizer is not None
            and self.template_args.get("apply_chat_template", False)
            and hasattr(self.tokenizer, "apply_chat_template")
        )

    def _prepend_system_message(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        system_prompt = self.template_args.get("system_prompt")
        if not system_prompt or any(msg["role"] == "system" for msg in messages):
            return messages
        return [{"role": "system", "content": system_prompt}] + messages

    def _render_chat_messages(
        self, messages: List[Dict[str, str]], *, add_generation_prompt: bool
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def _render_sharegpt_plain_messages(
        self, messages: List[Dict[str, str]], *, add_generation_prompt: bool
    ) -> str:
        role_prefix = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }
        rendered = [f"{role_prefix[msg['role']]}: {msg['content']}" for msg in messages]
        if add_generation_prompt:
            rendered.append("Assistant: ")
        return "\n".join(rendered)

    def _process_alpaca(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """处理 Alpaca 格式数据

        Alpaca 格式：
        {
            "instruction": "任务指令",
            "input": "可选的输入内容",
            "output": "期望的输出"
        }
        """
        instruction = self._require_text_field(item, self.instruction_key, idx)
        input_text = self._optional_text_field(item, self.input_key)
        output = self._require_text_field(item, self.output_key, idx)

        prompt, full_text = render_alpaca_texts(
            instruction,
            input_text,
            output,
            tokenizer=self.tokenizer,
            template_args=self.template_args,
        )
        return self._tokenize(prompt, full_text)

    def _process_sharegpt(self, item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """处理 ShareGPT 格式数据

        ShareGPT 格式：
        {
            "conversations": [
                {"from": "human", "value": "用户输入"},
                {"from": "gpt", "value": "助手回复"},
                ...
            ]
        }
        """
        if self.conversations_key not in item:
            raise ValueError(
                f"InjectDataset[{idx}] missing required field "
                f"'{self.conversations_key}' for format 'sharegpt'."
            )
        conversations = item[self.conversations_key]
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(
                f"InjectDataset[{idx}] field '{self.conversations_key}' must be a "
                "non-empty list for format 'sharegpt'."
            )

        normalized_messages = []

        for turn_idx, turn in enumerate(conversations):
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()

            if role in ["human", "user"]:
                normalized_messages.append({"role": "user", "content": content})
            elif role in ["gpt", "assistant"]:
                normalized_messages.append({"role": "assistant", "content": content})
            elif role == "system":
                normalized_messages.append({"role": "system", "content": content})

            if role in ["human", "user", "gpt", "assistant", "system"] and not content:
                raise ValueError(
                    f"InjectDataset[{idx}] conversation turn {turn_idx} has empty "
                    f"content for role '{role}'."
                )

        if not normalized_messages:
            raise ValueError(
                f"InjectDataset[{idx}] has no usable conversation turns in "
                f"'{self.conversations_key}'."
            )
        if normalized_messages[-1]["role"] != "assistant":
            raise ValueError(
                f"InjectDataset[{idx}] ShareGPT sample must end with an assistant "
                "message so labels can be computed."
            )

        prompt_messages = normalized_messages[:-1]

        if self._should_apply_chat_template():
            prompt = self._render_chat_messages(
                self._prepend_system_message(prompt_messages),
                add_generation_prompt=True,
            )
            full_text = self._render_chat_messages(
                self._prepend_system_message(normalized_messages),
                add_generation_prompt=False,
            )
            full_text = _align_chat_training_text(
                prompt,
                full_text,
                normalized_messages[-1]["content"],
                tokenizer=self.tokenizer,
                template_args=self.template_args,
            )
            return self._tokenize(prompt, full_text)

        prompt = self._render_sharegpt_plain_messages(
            prompt_messages,
            add_generation_prompt=True,
        )
        full_text = self._render_sharegpt_plain_messages(
            normalized_messages,
            add_generation_prompt=False,
        )

        return self._tokenize(prompt, full_text)

    def _process_custom(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理自定义格式数据"""
        # 尝试多种常见字段名
        text = item.get("text", "")
        if not text:
            text = item.get("content", "")
        if not text:
            # 拼接所有字符串字段
            text = " ".join(str(v) for v in item.values() if isinstance(v, str))

        return self._tokenize("", text)

    def _tokenize(self, prompt: str, full_text: str) -> Dict[str, Any]:
        """Tokenize 文本

        Args:
            prompt: 输入提示（不计入损失）
            full_text: 完整文本（包含输出）

        Returns:
            tokenized 字典
        """
        if self.tokenizer is None:
            return {"text": full_text, "prompt": prompt}

        return tokenize_inject_pair(
            prompt,
            full_text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )


class AlpacaDataset(InjectDataset):
    """Alpaca 格式数据集"""

    def __init__(self, tokenizer=None, **kwargs):
        kwargs.pop("format_type", None)
        super().__init__(format_type="alpaca", tokenizer=tokenizer, **kwargs)


class ShareGPTDataset(InjectDataset):
    """ShareGPT 格式数据集"""

    def __init__(self, tokenizer=None, **kwargs):
        kwargs.pop("format_type", None)
        super().__init__(format_type="sharegpt", tokenizer=tokenizer, **kwargs)
