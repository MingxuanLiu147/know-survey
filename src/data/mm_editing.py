"""
Multimodal Knowledge Editing Datasets
======================================

Adapts multimodal editing benchmarks (MMEdit E-VQA / E-IC, VLKEB,
MMKE-Bench) into the ``EditingSample`` / ``EditRequest`` pipeline used by
open-unlearning.

Image fields are stored as ``PIL.Image`` inside ``EditingSample`` so that
``MMEditMixin.mm_tokenize`` can consume them directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from data.editing import EditingDataset, EditingSample

logger = logging.getLogger(__name__)


def _safe_load_image(path: Path, *, required: bool = False) -> Optional[Image.Image]:
    """Load an RGB image and retain its resolved filename for audit output."""
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Multimodal editing image not found: {path}")
        logger.warning("Optional multimodal editing image not found: %s", path)
        return None
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        image.filename = str(path)
        return image
    except Exception as exc:
        if required:
            raise ValueError(
                f"Cannot load multimodal editing image {path}: {exc}"
            ) from exc
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


class MMEditingDataset(EditingDataset):
    """Base class for multimodal editing datasets.

    MMEdit E-VQA, E-IC, and VLKEB share one normalization path. MMKE-Bench
    keeps its own schema adapter below.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        rephrase_image_dir: Optional[str] = None,
        strict_images: bool = True,
        portability_hop: Optional[int] = None,
        **kwargs,
    ):
        self.image_dir = self._to_path(image_dir)
        self.rephrase_image_dir = self._to_path(rephrase_image_dir)
        self.strict_images = strict_images
        if portability_hop is not None:
            portability_hop = int(portability_hop)
            if portability_hop not in {1, 2, 3, 4}:
                raise ValueError("portability_hop must be one of 1, 2, 3, or 4")
        self.portability_hop = portability_hop
        super().__init__(**kwargs)

    def _to_path(self, p: Optional[str]) -> Optional[Path]:
        if p is None:
            return None
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = self.project_root() / path
        return path

    def _resolve_image(
        self,
        value: Any,
        base_dir: Optional[Path] = None,
        *,
        required: bool = False,
        field: str = "image",
        index: Optional[int] = None,
    ) -> Optional[Image.Image]:
        if hasattr(value, "convert"):
            filename = getattr(value, "filename", None)
            image = value.convert("RGB")
            image.load()
            if filename:
                image.filename = filename
            return image

        name = self._flatten_text(value).strip()
        if not name:
            if required:
                record = f" at record {index}" if index is not None else ""
                raise ValueError(f"Missing required {field}{record}")
            return None

        base = base_dir or self.image_dir
        path = Path(name).expanduser()
        if not path.is_absolute():
            if base is None:
                if required:
                    raise ValueError(
                        f"Cannot resolve required {field} {name!r}: image_dir is unset"
                    )
                return None
            path = base / path
        return _safe_load_image(path.resolve(), required=required)

    def _normalize_port_new(
        self, value: Any
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """Normalize VLKEB portability without mixing its two protocols.

        ``eval_multihop.json`` stores official Table-2/Table-12 probes as
        ``port_type`` plus a nested ``Q&A`` object.  In contrast,
        ``eval_edit_onehop.json`` stores the extra edit and test questions used
        by the paper's Table 3.  A requested ``portability_hop`` selects only
        the former schema so the two experiments cannot be averaged together.
        """
        if not isinstance(value, list):
            return None

        groups: Dict[str, List[Dict[str, str]]] = {}
        hop_label = (
            f"{self.portability_hop}-hop" if self.portability_hop is not None else None
        )
        multihop_entries = []
        for item in value:
            if not isinstance(item, dict) or "port_type" not in item:
                continue
            if hop_label is not None and item.get("port_type") != hop_label:
                continue
            qa = item.get("Q&A")
            if not isinstance(qa, dict):
                continue
            entry = self._make_eval_item(qa.get("Question"), qa.get("Answer"))
            if entry is not None:
                multihop_entries.append(entry)
        if multihop_entries:
            groups["multimodal_portability"] = multihop_entries

        # The flat fields belong to the distinct "edit one-hop knowledge"
        # protocol.  Preserve legacy loading only when no hop filter is active.
        if self.portability_hop is not None:
            return groups or None

        field_pairs = {
            "text_portability": ("port_edit_q", "port_edit_a"),
            "multimodal_portability": ("port_q", "port_a"),
        }
        for group, (prompt_key, answer_key) in field_pairs.items():
            entries = [
                entry
                for item in value
                if isinstance(item, dict)
                for entry in [
                    self._make_eval_item(item.get(prompt_key), item.get(answer_key))
                ]
                if entry is not None
            ]
            if entries:
                groups[group] = entries
        return groups or None

    @staticmethod
    def _merge_eval_groups(
        first: Optional[Dict[str, List[Dict[str, str]]]],
        second: Optional[Dict[str, List[Dict[str, str]]]],
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        if not first:
            return second
        if not second:
            return first
        merged = {key: list(entries) for key, entries in first.items()}
        for key, entries in second.items():
            merged.setdefault(key, []).extend(entries)
        return merged

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        prompt = self._flatten_text(item.get("src", item.get("prompt", ""))).strip()
        target_new = self._flatten_text(
            item.get("alt", item.get("target_new", ""))
        ).strip()
        if not prompt or not target_new:
            return None

        portability_inputs = self._normalize_eval_groups(
            item.get("portability_inputs", item.get("portability")), "portability"
        )
        port_new = self._normalize_port_new(item.get("port_new"))
        if self.portability_hop is not None and not port_new:
            return None
        portability_inputs = self._merge_eval_groups(portability_inputs, port_new)

        image = self._resolve_image(
            item.get("image"), required=True, field="image", index=index
        )
        image_rephrase_value = item.get("image_rephrase")
        image_rephrase = self._resolve_image(
            image_rephrase_value,
            self.rephrase_image_dir,
            required=self.strict_images and bool(image_rephrase_value),
            field="image_rephrase",
            index=index,
        )

        rephrase_text = self._flatten_text(item.get("rephrase", "")).strip()
        loc_prompt = self._flatten_text(item.get("loc", "")).strip()
        loc_answer = self._flatten_text(item.get("loc_ans", "")).strip()
        locality_inputs = None
        if loc_prompt and loc_answer:
            locality_inputs = {
                "text_locality": [{"prompt": loc_prompt, "ground_truth": loc_answer}]
            }

        mm_locality = None
        m_loc_q = self._flatten_text(item.get("m_loc_q", "")).strip()
        m_loc_a = self._flatten_text(item.get("m_loc_a", "")).strip()
        if m_loc_q and m_loc_a:
            mm_locality = {
                "image": self._resolve_image(
                    item.get("m_loc"),
                    required=self.strict_images,
                    field="m_loc",
                    index=index,
                ),
                "prompt": m_loc_q,
                "ground_truth": m_loc_a,
            }

        return EditingSample(
            prompt=prompt,
            subject=self._flatten_text(item.get("subject", "")).strip() or prompt,
            target_new=target_new,
            target_old=self._flatten_text(item.get("pred", "")).strip() or None,
            rephrase_prompts=[rephrase_text] if rephrase_text else None,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            image=image,
            image_rephrase=image_rephrase,
            multimodal_locality_inputs=mm_locality,
        )


class MMEditVQADataset(MMEditingDataset):
    """MMEdit E-VQA dataset (EMNLP 2023).

    Expected annotation format (JSON list)::

        {
          "image": "relative/path.jpg",
          "src": "What sport ...",
          "pred": "surfing",
          "alt": "skateboarding",
          "rephrase": "Which sport ...",
          "image_rephrase": "rephrase_img.jpg",
          "loc": "locality question",
          "loc_ans": "locality answer",
          "m_loc": "multimodal_locality_img.jpg",
          "m_loc_q": "multimodal locality question",
          "m_loc_a": "multimodal locality answer"
        }
    """

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        filename = "vqa_train.json" if split == "train" else "vqa_eval.json"
        return f"data/edit/mmedit/vqa/{filename}"


class MMEditCaptionDataset(MMEditingDataset):
    """MMEdit E-IC (Image Captioning) dataset (EMNLP 2023).

    Annotation format is identical to E-VQA; the only difference is that the
    prompt is a captioning instruction rather than a visual question.
    """

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        filename = (
            "caption_train_edit.json" if split == "train" else "caption_eval_edit.json"
        )
        return f"data/edit/mmedit/caption/{filename}"


class MMKEBenchDataset(MMEditingDataset):
    """MMKE-Bench dataset (ICLR 2025).

    Three editing task subsets:
      - visual_entity: editing visual entity knowledge
      - visual_semantic: editing visual semantic knowledge
      - user_specific: user-specific editing

    Data can be loaded from a local JSON file or via HuggingFace datasets.
    """

    SUBSET_FILES = {
        "visual_entity": "data/edit/mmke_bench/visual_entity.json",
        "visual_semantic": "data/edit/mmke_bench/visual_semantic.json",
        "user_specific": "data/edit/mmke_bench/user_specific.json",
    }

    def __init__(
        self,
        subset: str = "visual_entity",
        data_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        hf_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.subset = subset.lower()
        default_hf = None
        if hf_args is None and data_path is None:
            default_hf = {
                "path": "MMKE-Bench-dataset",
                "name": self.subset,
                "split": "test",
            }
        super().__init__(
            data_path=data_path,
            image_dir=image_dir,
            hf_args=hf_args or default_hf,
            **kwargs,
        )

    def _default_data_path(self, split: Optional[str]) -> Optional[str]:
        return self.SUBSET_FILES.get(self.subset)

    def normalize_record(
        self, item: Dict[str, Any], index: int
    ) -> Optional[Union[EditingSample, List[EditingSample]]]:
        prompt = self._flatten_text(
            item.get("prompt", item.get("question", item.get("src", "")))
        ).strip()
        target_new = self._flatten_text(
            item.get("target_new", item.get("alt", item.get("answer", "")))
        ).strip()
        if not prompt or not target_new:
            return None

        subject = self._flatten_text(item.get("subject", "")).strip() or prompt
        target_old = (
            self._flatten_text(
                item.get("target_old", item.get("ground_truth", ""))
            ).strip()
            or None
        )

        image_field = item.get("image", item.get("image_path", ""))
        image = None
        if isinstance(image_field, str) and image_field:
            image = self._resolve_image(image_field)
        elif hasattr(image_field, "convert"):
            image = image_field.convert("RGB")

        rephrase_text = self._flatten_text(
            item.get("rephrase", item.get("rephrase_prompt", ""))
        ).strip()
        rephrase_prompts = [rephrase_text] if rephrase_text else None

        locality_inputs = self._normalize_eval_groups(
            item.get("locality_inputs", item.get("locality")), "locality"
        )
        portability_inputs = self._normalize_eval_groups(
            item.get("portability_inputs", item.get("portability")), "portability"
        )

        return EditingSample(
            prompt=prompt,
            subject=subject,
            target_new=target_new,
            target_old=target_old,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            image=image,
        )
