"""CLEAR benchmark data adapter for sidecar multimodal unlearning."""

import glob
import logging
import os
import random
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from data.multimodal import mm_collate_fn, mm_paired_modality_collate_fn

logger = logging.getLogger(__name__)

IMAGE_CAPTION_QUESTIONS = [
    "What can you see in this picture?",
    "Tell me about the content of this image.",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]

CAPTION_MODE = "caption"
RECOGNITION_MODE = "recognition"
TEXT_MODE = "text"

_TRAIN_MODE_MAP = {
    "caption": CAPTION_MODE,
    "recognition": RECOGNITION_MODE,
    "text": TEXT_MODE,
}


def _maybe_limit_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def _resolve_clear_path(data_cfg, split_name: str) -> str:
    data_dir = str(data_cfg.data_dir)
    forget_ratio = int(data_cfg.get("forget_split_ratio", 5))
    retain_ratio = 100 - forget_ratio
    training_suffix = str(data_cfg.get("training_split_suffix", ""))
    if training_suffix not in {"", "+tofu"}:
        raise ValueError(
            "CLEAR training_split_suffix must be empty or '+tofu'; "
            f"got {training_suffix!r}"
        )
    mapping = {
        "forget": os.path.join(data_dir, f"forget{forget_ratio:02d}{training_suffix}"),
        "retain": os.path.join(data_dir, f"retain{retain_ratio}{training_suffix}"),
        "full": str(data_cfg.get("full_dir", os.path.join(data_dir, "full"))),
        "full_tofu": str(
            data_cfg.get("full_tofu_dir", os.path.join(data_dir, "full+tofu"))
        ),
    }
    if split_name not in mapping:
        raise ValueError(f"Unsupported CLEAR split_name={split_name}")
    return mapping[split_name]


def _read_parquet_rows(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Read at most ``max_rows`` without materializing every image shard row."""

    if max_rows is None or max_rows <= 0:
        return pd.read_parquet(path)
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        first_batch = next(parquet.iter_batches(batch_size=max_rows), None)
        if first_batch is None:
            return pd.DataFrame()
        return first_batch.to_pandas()
    except (ImportError, StopIteration):
        return pd.read_parquet(path).head(max_rows)


def load_clear_dataframe(path_or_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load a CLEAR parquet file/directory, optionally bounded by row count."""
    if not os.path.exists(path_or_dir):
        raise FileNotFoundError(f"CLEAR data path not found: {path_or_dir}")

    if os.path.isfile(path_or_dir):
        return _read_parquet_rows(path_or_dir, max_rows=max_rows)

    parquet_files = sorted(glob.glob(os.path.join(path_or_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {path_or_dir}")

    frames = []
    remaining = max_rows if max_rows is not None and max_rows > 0 else None
    for parquet_file in parquet_files:
        frame = _read_parquet_rows(parquet_file, max_rows=remaining)
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)
            if remaining <= 0:
                break
    return pd.concat(frames, ignore_index=True)


def decode_clear_image(image_value: Any):
    """Decode CLEAR image cell into a PIL image when possible."""
    if image_value is None:
        return None

    try:
        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes")
            if image_bytes is not None:
                return Image.open(BytesIO(image_bytes)).convert("RGB")
            image_path = image_value.get("path")
            if image_path and os.path.exists(image_path):
                return Image.open(image_path).convert("RGB")
        if isinstance(image_value, str) and os.path.exists(image_value):
            return Image.open(image_value).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to decode CLEAR image: %s", exc)
    return None


class CLEARDataset(Dataset):
    """Build CLEAR training/eval samples into image/question/answer triplets."""

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        mode: str = CAPTION_MODE,
        max_samples: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.max_samples = max_samples
        self.df = (
            load_clear_dataframe(data_source, max_rows=max_samples)
            if isinstance(data_source, str)
            else data_source.copy()
        )
        self.samples = self._build_samples()
        logger.info(
            "CLEARDataset loaded %d samples in %s mode", len(self.samples), mode
        )

    def _build_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []

        for record in self.df.to_dict("records"):
            if self.max_samples is not None and len(samples) >= self.max_samples:
                break
            image = decode_clear_image(record.get("image"))
            question = str(record.get("question") or "").strip()
            answer = str(record.get("answer") or "").strip()
            caption = str(record.get("caption") or "").strip()
            name = str(record.get("name") or "").strip()

            if question and answer:
                if self.mode == TEXT_MODE and image is not None:
                    continue
                samples.append(
                    {
                        "image": None if self.mode == TEXT_MODE else image,
                        "question": question,
                        "answer": answer,
                    }
                )
                continue

            if self.mode == CAPTION_MODE and image is not None and caption:
                text_question = (
                    f"Describe what is known about {name}."
                    if name
                    else "Describe what is known about the subject."
                )
                samples.append(
                    {
                        "image": image,
                        "question": random.choice(IMAGE_CAPTION_QUESTIONS),
                        "text_question": text_question,
                        "answer": caption,
                    }
                )
                continue

            if self.mode == RECOGNITION_MODE and image is not None and name:
                samples.append(
                    {
                        "image": image,
                        "question": "What is the name of the person in the image?",
                        "answer": name,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def _make_loader(
    dataset: Dataset,
    processor,
    *,
    batch_size: int,
    ans_only: bool,
    shuffle: bool,
    paired_modalities: bool = False,
):
    collate_fn = mm_paired_modality_collate_fn if paired_modalities else mm_collate_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: collate_fn(x, processor, ans_only=ans_only),
    )


def get_clear_data(data_cfg, processor):
    """Create CLEAR training dataloaders.

    mode=unlearn:
        返回 (forget_loader, retain_loader)
    mode=reference:
        返回 (reference_loader, None)，供 retain-only reference/oracle 流程复用。
    """
    batch_size = int(data_cfg.get("batch_size", 4))
    ans_only = bool(data_cfg.get("ans_only", False))
    train_mode = str(data_cfg.get("train_mode", "caption")).lower()
    mode = str(data_cfg.get("mode", "unlearn")).lower()
    max_samples = data_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    paired_modalities = bool(data_cfg.get("paired_modalities", False))

    dataset_mode = _TRAIN_MODE_MAP.get(train_mode)
    if dataset_mode is None:
        raise ValueError(
            f"Unknown CLEAR train_mode={train_mode}. Available: {sorted(_TRAIN_MODE_MAP)}"
        )

    if mode == "reference":
        source = str(data_cfg.get("reference_source", "retain")).lower()
        source_mode = str(data_cfg.get("reference_train_mode", train_mode)).lower()
        reference_mode = _TRAIN_MODE_MAP.get(source_mode)
        if reference_mode is None:
            raise ValueError(
                f"Unknown CLEAR reference_train_mode={source_mode}. "
                f"Available: {sorted(_TRAIN_MODE_MAP)}"
            )
        reference_dataset = CLEARDataset(
            _resolve_clear_path(data_cfg, source),
            mode=reference_mode,
            max_samples=max_samples,
        )
        reference_dataset = _maybe_limit_dataset(reference_dataset, max_samples)
        return _make_loader(
            reference_dataset,
            processor,
            batch_size=batch_size,
            ans_only=ans_only,
            shuffle=True,
            paired_modalities=paired_modalities,
        ), None

    forget_dataset = CLEARDataset(
        _resolve_clear_path(data_cfg, "forget"),
        mode=dataset_mode,
        max_samples=max_samples,
    )
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
    retain_dir = _resolve_clear_path(data_cfg, "retain")
    if os.path.exists(retain_dir):
        retain_dataset = CLEARDataset(
            retain_dir, mode=dataset_mode, max_samples=max_samples
        )
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


def get_clear_mask_data(data_cfg, processor) -> dict[str, DataLoader]:
    """Construct benchmark-aware preserve loaders for CLEAR mask generation."""
    batch_size = int(data_cfg.get("mask_batch_size", data_cfg.get("batch_size", 4)))
    ans_only = bool(data_cfg.get("mask_ans_only", False))
    max_samples = data_cfg.get("mask_max_samples")
    max_samples = int(max_samples) if max_samples is not None else None

    # MMUnlearner mask generation follows the later method repository and uses
    # visual-only forget/retain splits even when a CLEAR-native baseline trains
    # on the benchmark's mixed `+tofu` splits.
    mask_cfg = data_cfg.copy()
    mask_cfg.training_split_suffix = str(data_cfg.get("mask_training_split_suffix", ""))
    forget_dataset = _maybe_limit_dataset(
        CLEARDataset(
            _resolve_clear_path(mask_cfg, "forget"),
            mode=CAPTION_MODE,
            max_samples=max_samples,
        ),
        max_samples,
    )
    retain_caption_dataset = _maybe_limit_dataset(
        CLEARDataset(
            _resolve_clear_path(mask_cfg, "retain"),
            mode=CAPTION_MODE,
            max_samples=max_samples,
        ),
        max_samples,
    )
    full_tofu_text_dataset = _maybe_limit_dataset(
        CLEARDataset(
            _resolve_clear_path(data_cfg, "full_tofu"),
            mode=TEXT_MODE,
            max_samples=max_samples,
        ),
        max_samples,
    )

    vision_preserve_dataset = retain_caption_dataset
    language_preserve_dataset = ConcatDataset(
        [full_tofu_text_dataset, retain_caption_dataset]
    )

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
