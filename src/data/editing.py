"""
Knowledge Editing 数据集
=======================

统一适配知识编辑 benchmark 到当前仓库的 `EditRequest` 结构。

核心约束：
- `prompt` 与 `subject` 必须能被 ROME / MEMIT 直接消费；
- locality / portability 需要被当前 `src/evals/edit.py` 识别；
- 兼容本地规范化文件与 Hugging Face 原始数据源两种加载方式。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class EditingSample:
    """编辑样本数据结构。"""

    prompt: str
    subject: str
    target_new: str
    target_old: Optional[str] = None
    rephrase_prompts: Optional[List[str]] = None
    locality_inputs: Optional[Dict[str, List[Dict[str, str]]]] = None
    portability_inputs: Optional[Dict[str, List[Dict[str, str]]]] = None
    # Multimodal fields (Optional, backward-compatible with text-only editing)
    image: Optional[Any] = None
    image_rephrase: Optional[Any] = None
    multimodal_locality_inputs: Optional[Dict[str, Any]] = None
    scope_training_prompts: Optional[List[str]] = None
    scope_training_negatives: Optional[List[str]] = None
    prompt_template: Optional[str] = None


class EditingDataset(Dataset):
    """知识编辑数据集基类。"""

    def __init__(
        self,
        hf_args: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        prompt_key: str = "prompt",
        subject_key: str = "subject",
        target_new_key: str = "target_new",
        target_old_key: str = "target_true",
        tokenizer=None,
        max_length: int = 512,
        template_args: Optional[Dict[str, Any]] = None,
        apply_chat_template: bool = False,
        split: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        expected_num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args or {}
        self.apply_chat_template = apply_chat_template
        self.prompt_key = prompt_key
        self.subject_key = subject_key
        self.target_new_key = target_new_key
        self.target_old_key = target_old_key
        self.split = split
        self.extra_args = kwargs
        self.data_path = self._resolve_data_path(data_path, split)
        if expected_sha256 is not None:
            # A pinned benchmark must never fall back to a different HF dataset.
            if self.data_path is None or not self.data_path.is_file():
                raise FileNotFoundError(
                    "Pinned editing data is missing. Set data_path to the benchmark "
                    "JSON/JSONL matching expected_sha256. / 请指定与基准校验值一致的数据文件。"
                )
            actual_sha256 = hashlib.sha256(self.data_path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"Editing dataset SHA256 mismatch: expected {expected_sha256}, "
                    f"got {actual_sha256}. Set data_path to the matching benchmark "
                    "snapshot. / 数据文件与实验配置不一致，请使用对应的完整基准数据。"
                )
        self.data = self._load_data(hf_args)
        if expected_num_samples is not None and len(self.data) != expected_num_samples:
            raise ValueError(
                f"Editing dataset requires {expected_num_samples} normalized samples, "
                f"got {len(self.data)}. / 规范化后的样本数与实验配置不一致。"
            )
        logger.info(
            "%s loaded with %d samples from %s",
            self.__class__.__name__,
            len(self.data),
            self.data_path or hf_args,
        )

    @classmethod
    def project_root(cls) -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def edit_data_dir(cls) -> Path:
        return cls.project_root() / "data" / "edit"

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return None

    def _resolve_data_path(
        self, data_path: Optional[str], split: Optional[str]
    ) -> Optional[Path]:
        candidate = data_path or self._default_data_path(split)
        if candidate is None:
            return None
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = self.project_root() / path
        return path

    def _load_raw_data(self, hf_args: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.data_path is not None:
            if self.data_path.exists():
                suffix = self.data_path.suffix.lower()
                if suffix == ".json":
                    with open(self.data_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                elif suffix == ".jsonl":
                    with open(self.data_path, "r", encoding="utf-8") as f:
                        data = [json.loads(line) for line in f if line.strip()]
                else:
                    raise ValueError(
                        f"Unsupported editing dataset format: {self.data_path}"
                    )
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                return list(data)

            if not hf_args:
                raise FileNotFoundError(
                    f"Editing dataset file not found: {self.data_path}. "
                    "Set data_path to an existing JSON/JSONL file or provide hf_args."
                )
            logger.warning(
                "Local edit file missing (%s); falling back to HuggingFace %s",
                self.data_path,
                hf_args,
            )

        if hf_args:
            dataset = load_dataset(**hf_args)
            if hasattr(dataset, "column_names"):
                return list(dataset)
            if hasattr(dataset, "keys"):
                first_split = next(iter(dataset.keys()))
                return list(dataset[first_split])

        logger.warning(
            "No editing data source specified for %s", self.__class__.__name__
        )
        return []

    def _load_data(self, hf_args: Optional[Dict[str, Any]]) -> List[EditingSample]:
        raw_data = self._load_raw_data(hf_args)
        samples: List[EditingSample] = []
        for index, item in enumerate(raw_data):
            normalized = self.normalize_record(item, index)
            if normalized is None:
                continue
            normalized_samples = (
                normalized if isinstance(normalized, list) else [normalized]
            )
            for sample in normalized_samples:
                if sample is None:
                    continue
                for field in ("scope_training_prompts", "scope_training_negatives"):
                    prompts = self._normalize_prompt_list(item.get(field))
                    setattr(
                        sample,
                        field,
                        [self._format_model_prompt(p) for p in prompts] or None,
                    )
                samples.append(sample)
        return samples

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        return self._build_sample(
            prompt=item.get(self.prompt_key, ""),
            subject=item.get(self.subject_key, ""),
            target_new=item.get(self.target_new_key, ""),
            target_old=item.get(
                self.target_old_key,
                item.get("ground_truth", item.get("answer", item.get("target_true"))),
            ),
            rephrase_prompts=item.get(
                "rephrase_prompts",
                item.get("rephrase_prompt", item.get("rephrase")),
            ),
            locality_inputs=item.get("locality_inputs", item.get("locality")),
            portability_inputs=item.get("portability_inputs", item.get("portability")),
            prompt_template=item.get("prompt_template"),
        )

    def _build_sample(
        self,
        prompt: Any,
        subject: Any,
        target_new: Any,
        target_old: Any = None,
        rephrase_prompts: Any = None,
        locality_inputs: Any = None,
        portability_inputs: Any = None,
        prompt_template: Any = None,
    ) -> Optional[EditingSample]:
        prompt_text = self._flatten_text(prompt).strip()
        target_new_text = self._flatten_text(target_new).strip()
        if not prompt_text or not target_new_text:
            return None

        explicit_subject = self._flatten_text(subject).strip()
        subject_text = explicit_subject or self._infer_subject_from_prompt(prompt_text)
        template_text = self._flatten_text(prompt_template).strip() or None
        if template_text is None and explicit_subject and "{}" in prompt_text:
            template_text = prompt_text
        if template_text is not None:
            if template_text.count("{}") != 1 or not subject_text:
                raise ValueError("prompt_template requires one {} and a subject")
            rendered = template_text.format(subject_text)
            if prompt_text not in (template_text, rendered):
                raise ValueError("prompt_template does not match the supplied prompt")
            prompt_text = rendered
        if subject_text and subject_text not in prompt_text:
            prompt_text = f"Subject: {subject_text}\n{prompt_text}"

        prompt_text = self._format_model_prompt(prompt_text)
        normalized_rephrases = [
            self._format_model_prompt(value)
            for value in self._normalize_prompt_list(rephrase_prompts)
        ]

        return EditingSample(
            prompt=prompt_text,
            subject=subject_text or prompt_text,
            target_new=target_new_text,
            prompt_template=(
                self._format_model_prompt(template_text) if template_text else None
            ),
            target_old=self._flatten_text(target_old).strip() or None,
            rephrase_prompts=normalized_rephrases or None,
            locality_inputs=self._normalize_eval_groups(locality_inputs, "locality"),
            portability_inputs=self._normalize_eval_groups(
                portability_inputs, "portability"
            ),
        )

    def _format_model_prompt(self, prompt: str) -> str:
        """Optionally materialise the tokenizer's official chat prompt.

        AnyEdit's released Qwen/Llama dataset adapters wrap every question in
        the model chat template before computing autoregressive edit targets.
        Keeping this opt-in preserves the raw prompts used by ROME/MEMIT on
        classic factual benchmarks while allowing official AnyEdit-style runs.
        """
        prompt = str(prompt).strip()
        if not prompt or not self.apply_chat_template:
            return prompt
        # Avoid wrapping probes that were already normalized by a helper.
        if prompt.startswith(("<|im_start|>", "<|begin_of_text|>", "[INST]")):
            return prompt
        if self.tokenizer is None or not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                "apply_chat_template=true requires a tokenizer with "
                "apply_chat_template support"
            )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _normalize_prompt_list(self, value: Any) -> List[str]:
        from know_surgery.data_contract import rephrase_prompts

        return rephrase_prompts(value)

    def _flatten_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            for item in value:
                flattened = self._flatten_text(item).strip()
                if flattened:
                    return flattened
            return ""
        if isinstance(value, dict):
            for key in ("str", "text", "name", "answer", "target", "value", "label"):
                if key in value:
                    flattened = self._flatten_text(value[key]).strip()
                    if flattened:
                        return flattened
            return ""
        return str(value)

    def _expected_from_item(
        self, item: Dict[str, Any], default_target: Any = None
    ) -> str:
        for key in (
            "ground_truth",
            "expected",
            "answer",
            "target",
            "target_new",
            "target_true",
        ):
            if key in item:
                value = self._flatten_text(item[key]).strip()
                if value:
                    return value
        return self._flatten_text(default_target).strip()

    def _make_eval_item(self, prompt: Any, target: Any) -> Optional[Dict[str, str]]:
        prompt_text = self._flatten_text(prompt).strip()
        target_text = self._flatten_text(target).strip()
        if not prompt_text or not target_text:
            return None
        return {
            "prompt": self._format_model_prompt(prompt_text),
            "ground_truth": target_text,
        }

    def _coerce_eval_entries(
        self, payload: Any, default_target: Any = None
    ) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []

        if payload is None:
            return entries

        if isinstance(payload, str):
            item = self._make_eval_item(payload, default_target)
            return [item] if item else []

        if isinstance(payload, dict):
            if any(key in payload for key in ("prompt", "question")) and not isinstance(
                payload.get("prompt", payload.get("question")), list
            ):
                item = self._make_eval_item(
                    payload.get("prompt", payload.get("question")),
                    self._expected_from_item(payload, default_target),
                )
                return [item] if item else []

            prompts = payload.get("prompt", payload.get("question"))
            targets = payload.get(
                "ground_truth",
                payload.get(
                    "expected",
                    payload.get("answer", payload.get("target", default_target)),
                ),
            )
            if prompts is not None:
                if isinstance(prompts, str):
                    prompts = [prompts]
                if isinstance(targets, str):
                    targets = [targets] * len(prompts)
                elif not isinstance(targets, list):
                    targets = [targets] * len(prompts)
                if len(prompts) != len(targets):
                    raise ValueError("Probe prompt and target counts differ.")
                for prompt, target in zip(prompts, targets):
                    item = self._make_eval_item(prompt, target)
                    if item:
                        entries.append(item)
                return entries

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    entry = self._make_eval_item(
                        item.get("prompt", item.get("question")),
                        self._expected_from_item(item, default_target),
                    )
                else:
                    entry = self._make_eval_item(item, default_target)
                if entry:
                    entries.append(entry)

        return entries

    def _normalize_eval_groups(
        self, raw_inputs: Any, default_group: str
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        if raw_inputs is None:
            return None

        grouped: Dict[str, List[Dict[str, str]]] = {}
        if isinstance(raw_inputs, dict) and not any(
            key in raw_inputs
            for key in (
                "prompt",
                "question",
                "ground_truth",
                "expected",
                "answer",
                "target",
            )
        ):
            for group, payload in raw_inputs.items():
                entries = self._coerce_eval_entries(payload)
                if entries:
                    grouped[str(group)] = entries
            return grouped or None

        entries = self._coerce_eval_entries(raw_inputs)
        if entries:
            grouped[default_group] = entries
        return grouped or None

    def _format_subject_prompt(self, prompt_template: str, subject: str) -> str:
        prompt_template = self._flatten_text(prompt_template).strip()
        if "{}" in prompt_template:
            return prompt_template.format(subject)
        return prompt_template

    def _requested_rewrite_to_sample(
        self,
        requested_rewrite: Dict[str, Any],
        rephrase_prompts: Any = None,
        locality_inputs: Any = None,
        portability_inputs: Any = None,
    ) -> Optional[EditingSample]:
        subject = self._flatten_text(requested_rewrite.get("subject")).strip()
        prompt = requested_rewrite.get("prompt", "")
        target_new = requested_rewrite.get("target_new")
        target_old = requested_rewrite.get(
            "target_true", requested_rewrite.get("target_old")
        )
        return self._build_sample(
            prompt=prompt,
            subject=subject,
            target_new=target_new,
            target_old=target_old,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
        )

    def _infer_subject_from_prompt(self, prompt: str) -> str:
        patterns = [
            r"(?i)(?:what|who|which|where|when|why|how)\s+(?:is|was|are|were|does|did|do)\s+(.+?)'s\b",
            r"(?i)(?:what|who|which)\s+family does\s+(.+?)\s+belong to",
            r"(?i)who is the head of state of the country where\s+(.+?)\s+holds",
            r"(?i)who is\s+(.+?)\?",
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return match.group(1).strip()
        return prompt

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        input_text = sample.prompt
        target_text = f"{sample.prompt} {sample.target_new}"

        if self.tokenizer:
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": targets["input_ids"].squeeze(0),
                "subject": sample.subject,
                "target_new": sample.target_new,
                "target_old": sample.target_old,
            }

        return {
            "prompt": sample.prompt,
            "subject": sample.subject,
            "target_new": sample.target_new,
            "target_old": sample.target_old,
        }

    def to_edit_requests(self, limit: Optional[int] = None):
        """将数据集样本转换为 EditRequest 列表。"""
        from trainer.edit.base import EditRequest

        samples = self.data[:limit] if limit is not None else self.data
        return [
            EditRequest(
                prompt=sample.prompt,
                subject=sample.subject,
                target_new=sample.target_new,
                target_old=sample.target_old,
                rephrase_prompts=sample.rephrase_prompts,
                locality_inputs=sample.locality_inputs,
                portability_inputs=sample.portability_inputs,
                image=sample.image,
                image_rephrase=sample.image_rephrase,
                multimodal_locality_inputs=sample.multimodal_locality_inputs,
                scope_training_prompts=sample.scope_training_prompts,
                scope_training_negatives=sample.scope_training_negatives,
                prompt_template=sample.prompt_template,
            )
            for sample in samples
        ]


class ZSREDataset(EditingDataset):
    """ZSRE 数据集。"""

    LOCAL_FILES = {
        "train": "data/edit/zsre/zsre_mend_train.json",
        "test": "data/edit/zsre/ZsRE-test-all.json",
        "all": "data/edit/zsre/ZsRE-test-all.json",
    }

    def __init__(
        self,
        tokenizer=None,
        split: str = "test",
        hf_args: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        default_hf_args = {
            "path": "zjunlp/KnowEdit",
            "name": "zsre",
            "split": split,
        }
        super().__init__(
            hf_args=hf_args or default_hf_args,
            data_path=data_path,
            tokenizer=tokenizer,
            prompt_key="prompt",
            subject_key="subject",
            target_new_key="target_new",
            target_old_key="ground_truth",
            split=split,
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.LOCAL_FILES.get(split or "test", self.LOCAL_FILES["test"])

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        if "src" in item and "alt" in item:
            locality_inputs = None
            loc_prompt = self._flatten_text(item.get("loc")).strip()
            loc_answer = self._flatten_text(item.get("loc_ans")).strip()
            if loc_prompt and loc_answer:
                locality_inputs = {
                    "Relation_Specificity": [
                        {"prompt": loc_prompt, "ground_truth": loc_answer}
                    ]
                }
            return self._build_sample(
                prompt=item.get("src", ""),
                subject=item.get("subject", ""),
                target_new=item.get("alt", ""),
                target_old=item.get("answers") or item.get("pred"),
                rephrase_prompts=item.get("rephrase"),
                locality_inputs=locality_inputs,
            )
        return super().normalize_record(item, index)


class CounterFactDataset(EditingDataset):
    """CounterFact 数据集。"""

    LOCAL_FILES = {
        "train": "data/edit/counterfact/counterfact_train.json",
        "test": "data/edit/counterfact/counterfact_test.json",
    }

    def __init__(
        self,
        tokenizer=None,
        split: str = "train",
        hf_args: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        default_hf_args = {
            "path": "zjunlp/KnowEdit",
            "name": "counterfact",
            "split": split,
        }
        super().__init__(
            hf_args=hf_args or default_hf_args,
            data_path=data_path,
            tokenizer=tokenizer,
            prompt_key="prompt",
            subject_key="subject",
            target_new_key="target_new",
            target_old_key="ground_truth",
            split=split,
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.LOCAL_FILES.get(split or "train", self.LOCAL_FILES["train"])

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        if "requested_rewrite" in item:
            rewrite = item.get("requested_rewrite") or {}
            target_old = self._flatten_text(
                rewrite.get("target_true", rewrite.get("target_old"))
            )
            # CounterFact generation_prompts are open-ended fluency prefixes,
            # not labeled portability QA. Assigning target_new as their answer
            # fabricates ground truth (for example, a birthplace prefix may
            # accompany a mother-tongue rewrite), so they remain unscored.
            # CounterFact neighborhood prompts: locality expects original answer
            locality_inputs = None
            neighborhood = item.get("neighborhood_prompts") or item.get(
                "attribute_prompts"
            )
            if neighborhood and target_old:
                locality_inputs = {
                    "Neighborhood": [
                        {"prompt": prompt, "ground_truth": target_old}
                        for prompt in neighborhood
                        if self._flatten_text(prompt).strip()
                    ]
                }
            return self._requested_rewrite_to_sample(
                rewrite,
                rephrase_prompts=item.get("paraphrase_prompts"),
                locality_inputs=locality_inputs,
            )
        return super().normalize_record(item, index)


class ELKENDataset(EditingDataset):
    """ELKEN 事件级知识编辑数据集。"""

    LOCAL_FILES = {
        "train": "data/edit/elken/train.json",
        "test": "data/edit/elken/test_repaired_drop_incomplete_tail.json",
    }

    def __init__(
        self,
        tokenizer=None,
        split: str = "train",
        data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer, data_path=data_path, split=split, **kwargs
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.LOCAL_FILES.get(split or "train", self.LOCAL_FILES["train"])

    def _format_mcq_prompt(self, event: str, qa: Dict[str, Any]) -> str:
        question = self._flatten_text(qa.get("question")).strip()
        candidate = self._flatten_text(qa.get("candidate")).strip()
        return f"Event: {event}\nQuestion: {question}\nOptions: {candidate}\nAnswer:"

    def _qa_entries(
        self, event: str, qas: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []
        for qa in qas or []:
            prompt = self._format_mcq_prompt(event, qa)
            answer = self._flatten_text(qa.get("answer")).strip()
            item = self._make_eval_item(prompt, answer)
            if item:
                entries.append(item)
        return entries

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        event = self._flatten_text(item.get("event")).strip()
        samples: List[EditingSample] = []

        for bucket_name in ("fact", "tendency"):
            bucket = item.get(bucket_name, {}) or {}
            qas = bucket.get("qas", []) or []
            local_qas = bucket.get("local_qas", []) or []
            locality_inputs = None
            if local_qas:
                locality_inputs = {
                    f"{bucket_name}_locality": self._qa_entries(event, local_qas)
                }

            for qa_index, qa in enumerate(qas):
                prompt = self._format_mcq_prompt(event, qa)
                target = self._flatten_text(qa.get("answer")).strip()
                sibling_qas = [
                    candidate for i, candidate in enumerate(qas) if i != qa_index
                ]
                portability_inputs = None
                sibling_entries = self._qa_entries(event, sibling_qas)
                if sibling_entries:
                    portability_inputs = {
                        f"{bucket_name}_event_transfer": sibling_entries
                    }
                sample = self._build_sample(
                    prompt=prompt,
                    subject=event,
                    target_new=target,
                    locality_inputs=locality_inputs,
                    portability_inputs=portability_inputs,
                )
                if sample:
                    samples.append(sample)

        return samples


class UnKEDataset(EditingDataset):
    """UnKE / UnKEBench 数据集。"""

    VERSION_FILES = {
        "v2": "data/edit/unke/final_data_v2.json",
        "v3": "data/edit/unke/final_data_v3.json",
    }

    def __init__(
        self,
        tokenizer=None,
        version: str = "v3",
        data_path: Optional[str] = None,
        **kwargs,
    ):
        self.version = version
        super().__init__(tokenizer=tokenizer, data_path=data_path, **kwargs)

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.VERSION_FILES.get(self.version, self.VERSION_FILES["v3"])

    def _format_choice_prompt(self, question: str, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F"]
        options = " ".join(
            f"({labels[idx]}) {choice}" for idx, choice in enumerate(choices or [])
        )
        return f"{question}\nOptions: {options}\nAnswer:"

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        subject = self._infer_subject_from_prompt(
            self._flatten_text(item.get("question"))
        )

        portability_inputs = {
            "FactualCorrectness": [
                {"prompt": prompt, "ground_truth": answer}
                for prompt, answer in zip(
                    item.get("sub_question", []) or [],
                    item.get("sub_answer", []) or [],
                )
                if self._flatten_text(prompt).strip()
                and self._flatten_text(answer).strip()
            ]
        }
        if not portability_inputs["FactualCorrectness"]:
            portability_inputs = None

        locality_entries = []
        for prompt, choices, answer_idx in zip(
            item.get("mmlu_questions", []) or [],
            item.get("mmlu_choices", []) or [],
            item.get("mmlu_answer", []) or [],
        ):
            if not isinstance(choices, list):
                continue
            try:
                expected_index = int(answer_idx)
                expected = ["A", "B", "C", "D", "E", "F"][expected_index]
            except (TypeError, ValueError, IndexError):
                continue
            locality_entries.append(
                {
                    "prompt": self._format_choice_prompt(
                        self._flatten_text(prompt), choices
                    ),
                    "ground_truth": self._flatten_text(expected),
                }
            )
        locality_inputs = (
            {"GeneralAbility_MMLU": locality_entries} if locality_entries else None
        )

        return self._build_sample(
            prompt=item.get("question", ""),
            subject=subject,
            target_new=item.get("answer", ""),
            rephrase_prompts=item.get("para_question"),
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
        )


class ConceptEditDataset(EditingDataset):
    """ConceptEdit 数据集。"""

    def __init__(
        self,
        tokenizer=None,
        variant: str = "intra",
        data_path: Optional[str] = None,
        **kwargs,
    ):
        self.variant = variant.lower()
        super().__init__(tokenizer=tokenizer, data_path=data_path, **kwargs)

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return "data/edit/conceptedit/concept_data.json"

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        module_key = "module_inter" if self.variant == "inter" else "module_intra"
        module_data = item.get(module_key, {}) or {}
        concept_name = self._flatten_text(item.get("concept_name")).strip()
        prompt = f"The definition of {concept_name} is"
        locality_inputs = None
        locality_prompt = self._flatten_text(item.get("locality_prompt")).strip()
        locality_answer = self._flatten_text(item.get("locality_answer")).strip()
        if locality_prompt and locality_answer:
            locality_inputs = {
                "Concept_Locality": [
                    {"prompt": locality_prompt, "ground_truth": locality_answer}
                ]
            }
        rephrase_prompts = [f"What is the definition of {concept_name}?"]
        return self._build_sample(
            prompt=prompt,
            subject=concept_name,
            target_new=module_data.get("replace_def", ""),
            target_old=item.get("concept_def"),
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
        )


class AKEWDataset(EditingDataset):
    """AKEW 数据集适配器。"""

    SUBSET_FILES = {
        "counterfact": "data/edit/akew/CounterFact.json",
        "mquake-cf": "data/edit/akew/MQuAKE-CF.json",
        "wikiupdate": "data/edit/akew/WikiUpdate.json",
    }

    def __init__(
        self,
        tokenizer=None,
        subset: str = "counterfact",
        target_mode: str = "structured",
        data_path: Optional[str] = None,
        **kwargs,
    ):
        self.subset = subset.lower()
        self.target_mode = target_mode.lower()
        if self.target_mode not in {"structured", "unstructured"}:
            raise ValueError("AKEW target_mode must be 'structured' or 'unstructured'")
        super().__init__(tokenizer=tokenizer, data_path=data_path, **kwargs)

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.SUBSET_FILES.get(self.subset, self.SUBSET_FILES["counterfact"])

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        if self.subset == "mquake-cf":
            rewrite = (item.get("requested_rewrite") or [{}])[0]
            portability_inputs = self._akew_portability(item, rewrite)
            if self.target_mode == "unstructured":
                return self._unstructured_rewrite_to_sample(
                    rewrite,
                    portability_inputs=portability_inputs,
                )
            return self._requested_rewrite_to_sample(
                rewrite,
                portability_inputs=portability_inputs,
            )

        if self.subset == "counterfact":
            rewrite = item.get("requested_rewrite") or {}
            if self.target_mode == "unstructured":
                return self._unstructured_rewrite_to_sample(
                    rewrite,
                    rephrase_prompts=item.get("paraphrase_prompts"),
                    locality_inputs=self._counterfact_locality(item, rewrite),
                    portability_inputs=self._akew_portability(item, rewrite),
                )
            return self._requested_rewrite_to_sample(
                rewrite,
                rephrase_prompts=item.get("paraphrase_prompts"),
                locality_inputs=self._counterfact_locality(item, rewrite),
            )

        if self.target_mode == "unstructured":
            rewrite = item.get("requested_rewrite", {})
            return self._unstructured_rewrite_to_sample(
                rewrite,
                portability_inputs=self._akew_portability(item, rewrite),
            )
        return self._requested_rewrite_to_sample(item.get("requested_rewrite", {}))

    def _counterfact_locality(
        self, item: Dict[str, Any], rewrite: Dict[str, Any]
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        target_old = self._flatten_text(
            rewrite.get("target_true", rewrite.get("target_old"))
        )
        neighborhood = item.get("neighborhood_prompts") or item.get("attribute_prompts")
        if not neighborhood or not target_old:
            return None
        entries = [
            self._make_eval_item(prompt, target_old)
            for prompt in neighborhood
            if self._flatten_text(prompt).strip()
        ]
        return {"Neighborhood": [entry for entry in entries if entry]} or None

    def _akew_portability(
        self, item: Dict[str, Any], rewrite: Dict[str, Any]
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        groups: Dict[str, List[Dict[str, str]]] = {}

        triplet_entries = []
        for triplet in rewrite.get("unsfact_triplets_GPT", []) or []:
            subject = self._flatten_text(triplet.get("subject")).strip()
            prompt = self._format_subject_prompt(triplet.get("prompt", ""), subject)
            entry = self._make_eval_item(prompt, triplet.get("target"))
            if entry:
                triplet_entries.append(entry)
        if triplet_entries:
            groups["UnstructuredFact"] = triplet_entries[:5]

        multihop_entries = [
            self._make_eval_item(prompt, item.get("new_answer", ""))
            for prompt in item.get("questions", []) or []
        ]
        multihop_entries = [entry for entry in multihop_entries if entry]
        if multihop_entries:
            groups["MultiHop"] = multihop_entries
        return groups or None

    def _unstructured_rewrite_to_sample(
        self,
        rewrite: Dict[str, Any],
        *,
        rephrase_prompts: Any = None,
        locality_inputs: Any = None,
        portability_inputs: Any = None,
    ) -> Optional[EditingSample]:
        """Map AKEW's released long-form fields exactly as AnyEdit does."""
        subject = self._flatten_text(rewrite.get("subject")).strip()
        if self.subset == "counterfact":
            prompt = self._flatten_text(
                rewrite.get("prompt_full", rewrite.get("question", ""))
            ).strip()
        elif self.subset == "mquake-cf":
            # This is the exact input used by AnyEdit's released MQuAKE adapter.
            prompt = self._flatten_text(rewrite.get("question", "")).strip()
        else:
            prompt = ""
        if not prompt:
            prompt = self._format_subject_prompt(rewrite.get("prompt", ""), subject)
        target = self._flatten_text(rewrite.get("fact_new_uns")).strip()
        if not target:
            raise ValueError(
                f"AKEW {self.subset} unstructured record is missing fact_new_uns"
            )
        return self._build_sample(
            prompt=prompt,
            subject=subject,
            target_new=target,
            target_old=rewrite.get("target_true", rewrite.get("target_old")),
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
        )


class LEMEDataset(EditingDataset):
    """LEME 数据集适配器。"""

    SUBSET_FILES = {
        "counterfact": "data/edit/leme/counterfact_with_coupled_entities.json",
        "zsre": "data/edit/leme/zsre_mend_eval_with_coupled_entities.json",
    }

    def __init__(
        self,
        tokenizer=None,
        subset: str = "counterfact",
        data_path: Optional[str] = None,
        **kwargs,
    ):
        self.subset = subset.lower()
        super().__init__(tokenizer=tokenizer, data_path=data_path, **kwargs)

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.SUBSET_FILES.get(self.subset, self.SUBSET_FILES["counterfact"])

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        if self.subset == "zsre":
            locality_inputs = None
            if item.get("loc") and item.get("loc_ans"):
                locality_inputs = {
                    "Relation_Specificity": [
                        {
                            "prompt": item.get("loc", ""),
                            "ground_truth": item.get("loc_ans", ""),
                        }
                    ]
                }
            return self._build_sample(
                prompt=item.get("src", ""),
                subject=item.get("subject", ""),
                target_new=item.get("alt", ""),
                target_old=item.get("answers", []),
                rephrase_prompts=item.get("rephrase"),
                locality_inputs=locality_inputs,
            )

        return self._requested_rewrite_to_sample(
            item.get("requested_rewrite", {}),
            rephrase_prompts=item.get("paraphrase_prompts"),
        )


class EditEveryDataset(EditingDataset):
    """EditEvery dataset from AnyEdit -- general-purpose editing benchmark.

    Paper / project: https://github.com/jianghoucheng/AnyEdit

    Contains 552 long-form QA pairs across categories such as math and
    reasoning.  Each sample only has *prompt* and *target_new*; no rephrase,
    locality, or portability annotations.
    """

    def __init__(
        self,
        tokenizer=None,
        data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(tokenizer=tokenizer, data_path=data_path, **kwargs)

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return "data/edit/editevery/editevery.json"

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        question = item.get("question", item.get("prompt", ""))
        answer = item.get("answer", item.get("target_new", ""))
        # Prefer explicit eval fields when present; otherwise leave None so
        # Suite reports 0 evaluated cases instead of inventing scores.
        rephrase = item.get("rephrase_prompts") or item.get("para_question")
        locality = item.get("locality_inputs") or item.get("locality")
        portability = item.get("portability_inputs") or item.get("portability")
        return self._build_sample(
            prompt=question,
            # Category is metadata in the official release; it must not be
            # injected into the model prompt as a synthetic ``Subject:`` line.
            subject=question,
            target_new=answer,
            rephrase_prompts=rephrase,
            locality_inputs=locality,
            portability_inputs=portability,
        )
