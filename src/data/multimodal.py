"""多模态数据集和 collate 函数（sidecar，不修改现有 data/__init__.py）。

复用上游 MMUnlearner 的 MLLMU 数据格式（Parquet + image bytes + metadata JSON）。
collate 使用 transformers 原生 processor.apply_chat_template，不依赖 qwen_vl_utils。
"""

import glob
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


def _mllmu_profile_name(biography) -> str:
    """Extract the profile name stored in MLLMU's biography field."""

    if isinstance(biography, str):
        try:
            biography = json.loads(biography)
        except json.JSONDecodeError:
            return ""
    if not isinstance(biography, dict):
        return ""
    return str(biography.get("Name") or "").strip()


def _mllmu_text_question(question: str, biography) -> str:
    """Anchor an image-referential MLLMU question to its named profile.

    The bundled MANU data path uses the biography name for its text-only view.
    Keep the multimodal question unchanged and expose this rendering only via
    ``text_question`` so all existing baseline loaders retain their behavior.
    """

    name = _mllmu_profile_name(biography)
    question = str(question).strip()
    if not name:
        return question
    anchored = question
    for reference in (
        "this person in the image",
        "the person in the image",
        "this individual in the image",
        "the individual in the image",
        "this person",
        "the person",
        "this individual",
        "the individual",
    ):
        anchored = anchored.replace(reference, name)
        anchored = anchored.replace(reference.capitalize(), name)
    if name not in anchored:
        anchored = f"{anchored} {name}".strip()
    return anchored


def load_mllmu_frame(path_or_dir: str) -> pd.DataFrame:
    """Load one parquet file or a parquet directory into a single dataframe."""
    if not os.path.exists(path_or_dir):
        raise FileNotFoundError(f"MLLMU data path not found: {path_or_dir}")

    if os.path.isfile(path_or_dir):
        return pd.read_parquet(path_or_dir)

    parquet_files = sorted(glob.glob(os.path.join(path_or_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {path_or_dir}")

    frames = [pd.read_parquet(parquet_file) for parquet_file in parquet_files]
    return pd.concat(frames, ignore_index=True)


def _maybe_limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


class MLLMUDataset(Dataset):
    """MLLMU-Bench 数据集，从 Parquet 读取图像+QA 对并展平为单条样本。"""

    def __init__(self, parquet_path: str, *, text_only: bool = False):
        super().__init__()
        self.text_only = text_only
        df = load_mllmu_frame(parquet_path)
        self.samples = self._flatten(df)
        logger.info(
            "MLLMUDataset loaded %d samples from %s (text_only=%s)",
            len(self.samples),
            parquet_path,
            text_only,
        )

    def _flatten(self, df: pd.DataFrame) -> List[Dict]:
        data = []
        has_metadata = "metadata" in df.columns
        for idx, row in df.iterrows():
            image = None
            if not self.text_only:
                image_bytes = row["image"].get("bytes")
                try:
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception as e:
                    logger.warning("Skipping row %d: image decode error: %s", idx, e)
                    continue

            if has_metadata:
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError as e:
                    logger.warning("Skipping row %d: metadata JSON error: %s", idx, e)
                    continue
                for qa in metadata:
                    q, a = qa.get("Question", ""), qa.get("Answer", "")
                    if q and a:
                        data.append(
                            {
                                "image": image,
                                "question": q,
                                "text_question": _mllmu_text_question(
                                    q, row.get("biography")
                                ),
                                "answer": a,
                            }
                        )
            else:
                q = row.get("question", "")
                a = row.get("answer", "")
                if q and a:
                    data.append(
                        {
                            "image": image,
                            "question": q,
                            "text_question": _mllmu_text_question(
                                q, row.get("biography")
                            ),
                            "answer": a,
                        }
                    )
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class MLLMUTextKnowledgeDataset(Dataset):
    """Pure-text profile knowledge used by MMUnlearner's language mask.

    MLLMU ``Full_Set`` stores these probes under
    ``Classification_Task.Pure_Text_Questions`` rather than the training
    ``question``/``answer`` columns.  Treating Full_Set as a normal
    :class:`MLLMUDataset` silently produced an empty preservation loader.
    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        *,
        max_samples: int | None = None,
    ):
        super().__init__()
        frame = (
            load_mllmu_frame(data_source)
            if isinstance(data_source, str)
            else data_source.copy()
        )
        self.samples: list[dict] = []
        for record in frame.to_dict("records"):
            task = record.get("Classification_Task") or {}
            questions = (
                task.get("Pure_Text_Questions", []) if isinstance(task, dict) else []
            )
            for item in questions:
                if max_samples is not None and len(self.samples) >= max_samples:
                    break
                if not isinstance(item, dict):
                    continue
                question = str(item.get("Question") or "").strip()
                correct = item.get("Correct_Answer")
                options = item.get("Options") or {}
                answer = None
                if isinstance(options, dict):
                    answer = options.get(correct)
                    if answer is None:
                        answer = options.get(str(correct))
                elif isinstance(options, (list, tuple)):
                    try:
                        index = int(correct)
                    except (TypeError, ValueError):
                        index = ord(str(correct).upper()) - ord("A")
                    if 0 <= index < len(options):
                        answer = options[index]
                answer = str(answer or "").strip()
                if question and answer:
                    self.samples.append(
                        {"image": None, "question": question, "answer": answer}
                    )
            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def mm_collate_fn(examples, processor, ans_only=False):
    """将 MLLMUDataset 样本列表编码为模型可用的 batch。

    使用 processor.apply_chat_template 构建 messages，
    再调用 processor() 编码图像+文本，不依赖 qwen_vl_utils。

    Args:
        examples: 样本列表，每个含 image / question / answer。
        processor: AutoProcessor 实例。
        ans_only: 是否只在 answer token 上计算 loss。
    """
    images = []
    texts = []
    prompt_texts = []
    answer_tokens_list: list[list[list[int]]] = []

    def token_ids(text: str) -> list[int]:
        encoded = processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None,
        )["input_ids"]
        if torch.is_tensor(encoded):
            encoded = encoded.detach().cpu().tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]

    for ex in examples:
        example_images = ex.get("images")
        if example_images is None:
            example_images = [ex["image"]] if ex.get("image") is not None else []
        question = ex["question"]
        answer = ex["answer"]

        images.extend(example_images)
        user_content = [{"type": "image"} for _ in example_images]
        user_content.append({"type": "text", "text": question})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text)

        if ans_only:
            prompt_text = processor.apply_chat_template(
                messages[:1], add_generation_prompt=True
            )
            if not text.startswith(prompt_text):
                raise ValueError(
                    "The chat template must render the user turn and assistant "
                    "generation prompt as a prefix of the full conversation."
                )
            prompt_texts.append(prompt_text)
            candidates: list[list[int]] = []
            for candidate_text in (answer, f" {answer}"):
                try:
                    candidate_ids = token_ids(candidate_text)
                except (KeyError, TypeError, ValueError):
                    continue
                if candidate_ids and candidate_ids not in candidates:
                    candidates.append(candidate_ids)
            if not candidates:
                raise ValueError(f"Answer tokenization produced no tokens: {answer!r}")
            answer_tokens_list.append(candidates)

    batch = processor(
        text=texts,
        images=images if images else None,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()

    if ans_only and answer_tokens_list:
        # Process the prompt with the same images: VLM processors expand image
        # placeholders, so plain-text token counts cannot locate the answer.
        prompt_batch = processor(
            text=prompt_texts,
            images=images if images else None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        for i, candidate_sequences in enumerate(answer_tokens_list):
            positions = batch["attention_mask"][i].bool().nonzero().flatten()
            sequence = labels[i, positions].detach().cpu().tolist()
            prompt_ids = (
                prompt_batch["input_ids"][i][prompt_batch["attention_mask"][i].bool()]
                .detach()
                .cpu()
                .tolist()
            )
            prompt_length = len(prompt_ids)
            if sequence[:prompt_length] != prompt_ids:
                raise ValueError(
                    "Assistant answer boundary cannot be aligned; the prompt may "
                    "have been truncated or the template tokenization differs."
                )
            best_match: tuple[int, int] | None = None
            for answer_ids in candidate_sequences:
                answer_length = len(answer_ids)
                for start in range(
                    len(sequence) - answer_length, prompt_length - 1, -1
                ):
                    if sequence[start : start + answer_length] == answer_ids:
                        candidate_match = (start, answer_length)
                        if best_match is None or start > best_match[0]:
                            best_match = candidate_match
                        break
            if best_match is None:
                raise ValueError(
                    "Complete answer token span was not found in the assistant turn "
                    "(it may have been truncated); "
                    f"answer={examples[i]['answer']!r}"
                )
            start, answer_length = best_match
            answer_positions = positions[start : start + answer_length]
            answer_labels = labels[i, answer_positions].clone()
            labels[i].fill_(-100)
            labels[i, answer_positions] = answer_labels
    else:
        labels[batch["attention_mask"] == 0] = -100

    batch["labels"] = labels
    return dict(batch)


class CanonicalMMInjectionDataset(Dataset):
    """Load local image/text targets; resolve image paths beside the JSON file."""

    def __init__(
        self,
        path: str,
        *,
        image_root: str | None = None,
        require_images: bool = True,
        max_samples: int | None = None,
    ):
        source = Path(path).expanduser().resolve()
        text = source.read_text(encoding="utf-8")
        records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if source.suffix.lower() == ".jsonl"
            else json.loads(text)
        )
        if not isinstance(records, list) or not records:
            raise ValueError("Multimodal injection needs a nonempty JSON list or JSONL")
        root = Path(image_root).expanduser().resolve() if image_root else source.parent
        self.samples = []
        for index, record in enumerate(records):
            if max_samples is not None and max_samples > 0 and index >= max_samples:
                break
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at record {index}")
            question = next(
                (
                    record.get(k)
                    for k in ("prompt", "question", "instruction", "input")
                    if record.get(k)
                ),
                None,
            )
            if (
                record.get("instruction")
                and record.get("input")
                and question == record["instruction"]
            ):
                question = f"{question}\n{record['input']}"
            answer = next(
                (
                    record.get(k)
                    for k in ("target_new", "answer", "output")
                    if record.get(k)
                ),
                None,
            )
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(answer, str)
                or not answer.strip()
            ):
                raise ValueError(f"Missing nonempty prompt/target at record {index}")
            paths = record.get("images", record.get("image", []))
            if isinstance(paths, str):
                paths = [paths]
            if not isinstance(paths, list) or (require_images and not paths):
                raise ValueError(f"Missing image path(s) at record {index}")
            resolved = []
            for value in paths:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Invalid image path at record {index}")
                path_value = Path(value).expanduser()
                image_path = (
                    path_value if path_value.is_absolute() else root / path_value
                ).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image at record {index} not found: {image_path}"
                    )
                resolved.append(image_path)
            self.samples.append(
                {"question": question, "answer": answer, "image_paths": resolved}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        images = []
        for path in record["image_paths"]:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        return {
            "question": record["question"],
            "answer": record["answer"],
            "images": images,
        }


def get_mm_inject_data(data_cfg: DictConfig, processor):
    """Return training and optional retention loaders for image/text injection."""
    if str(data_cfg.get("mode")) != "inject":
        raise ValueError("Canonical_MM_inject requires data.mode=inject")
    loaders = []
    for role in ("train", "retain"):
        source = data_cfg.get(f"{role}_path")
        if not source:
            if role == "train":
                raise ValueError("Multimodal injection requires data.train_path")
            loaders.append(None)
            continue
        dataset = CanonicalMMInjectionDataset(
            str(source),
            image_root=data_cfg.get("image_root"),
            require_images=role == "train",
            max_samples=data_cfg.get("max_samples"),
        )
        loaders.append(
            _make_loader(
                dataset,
                processor,
                batch_size=int(data_cfg.get("batch_size", 1)),
                ans_only=bool(data_cfg.get("ans_only", True)),
                shuffle=role == "train",
            )
        )
    return tuple(loaders)


def mm_paired_modality_collate_fn(examples, processor, ans_only=False):
    """Encode the same raw QA examples as independent text and image-text views.

    MANU compares language-neuron activations across modalities.  A valid text
    view cannot be obtained by merely removing ``pixel_values`` from an already
    tokenized multimodal batch because its ``input_ids`` still contain image
    placeholders.  This opt-in collator therefore invokes the processor twice
    on the same examples.  Existing trainers keep using :func:`mm_collate_fn`.

    Samples may provide ``text_question``/``text_answer`` when the benchmark has
    a modality-specific textual rendering; otherwise the original QA is reused.
    """

    if not examples:
        raise ValueError("Paired multimodal collate received an empty example list")
    missing_images = [
        index for index, example in enumerate(examples) if example.get("image") is None
    ]
    if missing_images:
        raise ValueError(
            "Paired modality MANU batches require an image for every example; "
            f"missing at batch indices {missing_images}"
        )

    multimodal_batch = mm_collate_fn(examples, processor, ans_only=ans_only)
    text_examples = [
        {
            **example,
            "image": None,
            "question": example.get("text_question", example["question"]),
            "answer": example.get("text_answer", example["answer"]),
        }
        for example in examples
    ]
    text_batch = mm_collate_fn(text_examples, processor, ans_only=ans_only)
    return {"text": text_batch, "multimodal": multimodal_batch}


def _make_loader(
    dataset: Dataset,
    processor,
    *,
    batch_size: int,
    ans_only: bool,
    shuffle: bool,
    paired_modalities: bool = False,
) -> DataLoader:
    collate_fn = mm_paired_modality_collate_fn if paired_modalities else mm_collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, processor, ans_only=ans_only),
    )


def _resolve_mllmu_path(data_cfg: DictConfig, split_name: str) -> str:
    data_dir = str(data_cfg.data_dir)
    forget_ratio = int(data_cfg.get("forget_split_ratio", 5))
    retain_ratio = 100 - forget_ratio
    mapping = {
        "forget": os.path.join(data_dir, f"forget_{forget_ratio}"),
        "retain": os.path.join(data_dir, f"retain_{retain_ratio}"),
        "retain_shared": os.path.join(data_dir, f"retain_{retain_ratio}"),
        "retain_celebrity": str(
            data_cfg.get("retain_set_parquet", os.path.join(data_dir, "Retain_Set"))
        ),
        "full": str(
            data_cfg.get(
                "full_set_parquet",
                os.path.join(data_dir, "Full_Set", "train-00000-of-00001.parquet"),
            )
        ),
    }
    if split_name not in mapping:
        raise ValueError(f"Unsupported MLLMU split_name={split_name}")
    return mapping[split_name]


def get_mm_data(data_cfg: DictConfig, processor):
    """根据 Hydra 数据配置构建训练 DataLoader。

    mode=unlearn:
        返回 (forget_loader, retain_loader)
    mode=reference:
        返回 (reference_loader, None)，供 retain-only reference/oracle 流程复用。
    """
    mode = str(data_cfg.get("mode", "unlearn")).lower()
    batch_size = int(data_cfg.get("batch_size", 4))
    ans_only = bool(data_cfg.get("ans_only", False))
    max_samples = data_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    paired_modalities = bool(data_cfg.get("paired_modalities", False))

    if mode == "reference":
        source = str(data_cfg.get("reference_source", "retain")).lower()
        text_only = bool(data_cfg.get("reference_text_only", False))
        dataset = MLLMUDataset(
            _resolve_mllmu_path(data_cfg, source), text_only=text_only
        )
        dataset = _maybe_limit_dataset(dataset, max_samples)
        return _make_loader(
            dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
            paired_modalities=paired_modalities,
        ), None

    forget_dataset = MLLMUDataset(_resolve_mllmu_path(data_cfg, "forget"))
    forget_dataset = _maybe_limit_dataset(forget_dataset, max_samples)
    forget_loader = _make_loader(
        forget_dataset,
        processor,
        batch_size=batch_size,
        ans_only=ans_only,
        shuffle=True,
        paired_modalities=paired_modalities,
    )

    retain_loader = None
    retain_path = _resolve_mllmu_path(data_cfg, "retain")
    if os.path.exists(retain_path):
        retain_dataset = MLLMUDataset(retain_path)
        retain_dataset = _maybe_limit_dataset(retain_dataset, max_samples)
        retain_loader = _make_loader(
            retain_dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
            paired_modalities=paired_modalities,
        )

    full_parquet = data_cfg.get("full_set_parquet", None)
    if full_parquet and not retain_loader and os.path.exists(full_parquet):
        retain_dataset = MLLMUDataset(str(full_parquet))
        retain_dataset = _maybe_limit_dataset(retain_dataset, max_samples)
        retain_loader = _make_loader(
            retain_dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
            paired_modalities=paired_modalities,
        )

    return forget_loader, retain_loader


def get_mm_mask_data(data_cfg: DictConfig, processor) -> Dict[str, DataLoader]:
    """构建 MLLMU 的三掩码工作流所需 DataLoader。"""
    batch_size = int(data_cfg.get("mask_batch_size", data_cfg.get("batch_size", 4)))
    ans_only = bool(data_cfg.get("mask_ans_only", True))
    max_samples = data_cfg.get("mask_max_samples")
    max_samples = int(max_samples) if max_samples is not None else None

    forget_dataset = _maybe_limit_dataset(
        MLLMUDataset(_resolve_mllmu_path(data_cfg, "forget")), max_samples
    )
    retain_dataset = _maybe_limit_dataset(
        MLLMUDataset(_resolve_mllmu_path(data_cfg, "retain")), max_samples
    )
    full_text_dataset = MLLMUTextKnowledgeDataset(
        _resolve_mllmu_path(data_cfg, "full"),
        max_samples=max_samples,
    )

    vision_preserve_dataset = retain_dataset
    language_preserve_dataset = ConcatDataset([full_text_dataset, retain_dataset])

    return {
        "forget": _make_loader(
            forget_dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
        ),
        "vision_preserve": _make_loader(
            vision_preserve_dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
        ),
        "language_preserve": _make_loader(
            language_preserve_dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
        ),
    }
