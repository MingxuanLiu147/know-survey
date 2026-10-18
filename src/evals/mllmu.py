"""MLLMU-Bench evaluator aligned with the benchmark's public evaluator.

The native protocol is taken from ``franciscoliu/MLLMU-Bench`` at commit
``0e3098e5ea7e4878ce4f19cdf873de1410b4d427``.  The upstream evaluator defines
classification, fill-in-the-blank (cloze), and generation tasks over Forget,
Test, Retain Shared, and Real Celebrity splits.  MMMU, LLaVA-Bench, and the
optional GPT factuality judge are external evaluations in the upstream repo;
this module reports them as unavailable instead of fabricating proxy scores.
"""

import glob
import json
import logging
import os
import random
import re
from io import BytesIO
from typing import Any

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
random.seed(42)

UPSTREAM_REPOSITORY = "https://github.com/franciscoliu/MLLMU-Bench"
UPSTREAM_COMMIT = "0e3098e5ea7e4878ce4f19cdf873de1410b4d427"

SPLIT_SEMANTICS = {
    "forget": {
        "role": "unlearning_efficacy",
        "direction": "lower_is_better_after_unlearning",
    },
    "test": {
        "role": "unlearning_generalization_to_alternate_images",
        "direction": "lower_is_better_after_unlearning",
    },
    "retain_shared": {
        "role": "retained_fictitious_profile_utility",
        "direction": "higher_is_better",
    },
    "retain": {
        "role": "retained_fictitious_profile_utility",
        "direction": "higher_is_better",
    },
    "retain_celebrity": {
        "role": "real_celebrity_utility",
        "direction": "higher_is_better",
    },
    "real": {
        "role": "real_celebrity_utility",
        "direction": "higher_is_better",
    },
}


def _unavailable(reason: str, **extra) -> dict[str, Any]:
    """Return a JSON-safe unavailable result without substituting zero."""
    return {"status": "unavailable", "value": None, "reason": reason, **extra}


def _task_unavailable(reason: str) -> dict[str, Any]:
    """Return the common empty-task contract used by all MLLMU tasks."""
    return _unavailable(
        reason,
        numerator=0,
        denominator=0,
        total=0,
        per_example_count=0,
    )


def _load_parquet_frame(path_or_dir: str) -> pd.DataFrame:
    if os.path.isfile(path_or_dir):
        return pd.read_parquet(path_or_dir)

    parquet_files = sorted(glob.glob(os.path.join(path_or_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {path_or_dir}")
    return pd.concat(
        [pd.read_parquet(parquet_file) for parquet_file in parquet_files],
        ignore_index=True,
    )


def _decode_image(image_value: Any):
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
        logger.warning("Skipping image decode failure: %s", exc)
    return None


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _select_record_image(record: dict[str, Any]):
    if record.get("image") is not None:
        return _decode_image(record.get("image"))

    candidates = [_decode_image(item) for item in _to_list(record.get("images"))]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None
    return random.choice(candidates)


def _prepare_inputs_from_content(
    processor, model, content: list[dict[str, str]], images: list[Any] | None
):
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images or None, return_tensors="pt")
    return {key: value.to(model.device) for key, value in inputs.items()}


def _generate_from_content(
    processor,
    model,
    content: list[dict[str, str]],
    *,
    images: list[Any] | None = None,
    max_new_tokens: int = 50,
) -> str:
    inputs = _prepare_inputs_from_content(processor, model, content, images)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    output_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
    return processor.tokenizer.decode(
        output_tokens[0], skip_special_tokens=True
    ).strip()


def _generate_text(
    processor, model, question: str, image=None, *, max_new_tokens: int = 50
) -> str:
    content = [{"type": "text", "text": question}]
    images = None
    if image is not None:
        content.insert(0, {"type": "image"})
        images = [image]
    return _generate_from_content(
        processor,
        model,
        content,
        images=images,
        max_new_tokens=max_new_tokens,
    )


def _formulate_prompt_with_options(question: str, options: dict[str, str]) -> str:
    options_text = "\n".join([f"{key}: {value}" for key, value in options.items()])
    return f"{question}\n{options_text}\nJust give ONE letter representing the answer directly."


def _normalize_blank_prompt(question: str) -> str:
    return (
        question.replace("__", "[Blank]")
        + "\nPlease ONLY provide the correct answer that should replace the [Blank]."
    )


def _safe_rouge_scorer():
    try:
        from rouge_score import rouge_scorer

        return rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    except ImportError:
        logger.warning(
            "rouge_score is not installed; MLLMU generation ROUGE metrics are unavailable."
        )
        return None


def _compute_bleu(reference: str, prediction: str) -> float | None:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=SmoothingFunction().method1,
        )
    except ImportError:
        logger.warning("nltk is not installed; MLLMU BLEU metrics are unavailable.")
        return None


def _new_counter() -> dict[str, float]:
    return {"correct": 0, "total": 0}


def _new_generation_counter() -> dict[str, float]:
    return {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bleu": 0.0,
        "total": 0,
        "rouge_count": 0,
        "bleu_count": 0,
    }


def _finalize_counter(counter: dict[str, float]) -> dict[str, Any]:
    total = int(counter["total"])
    correct = int(counter["correct"])
    if total == 0:
        return {
            **_task_unavailable("zero_valid_questions"),
            "accuracy": None,
            "correct": correct,
            "metric_definition": "correct / denominator",
            "range": [0.0, 1.0],
        }
    return {
        "status": "available",
        "reason": None,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "numerator": correct,
        "denominator": total,
        "per_example_count": total,
        "metric_definition": "correct / denominator",
        "range": [0.0, 1.0],
    }


def _finalize_generation_counter(counter: dict[str, float]) -> dict[str, Any]:
    total = int(counter["total"])
    if total == 0:
        return {
            **_task_unavailable("zero_valid_questions"),
            "rouge1": None,
            "rouge2": None,
            "rougeL": None,
            "bleu": None,
            "metric_definition": "macro mean over valid generation questions",
            "range": [0.0, 1.0],
        }

    rouge_count = int(counter["rouge_count"])
    bleu_count = int(counter["bleu_count"])
    missing = []
    if rouge_count == 0:
        missing.append("rouge_score_dependency_missing")
    if bleu_count == 0:
        missing.append("nltk_dependency_missing")
    return {
        "status": "partial" if missing else "available",
        "reason": "; ".join(missing) if missing else None,
        "rouge1": counter["rouge1"] / rouge_count if rouge_count else None,
        "rouge2": counter["rouge2"] / rouge_count if rouge_count else None,
        "rougeL": counter["rougeL"] / rouge_count if rouge_count else None,
        "bleu": counter["bleu"] / bleu_count if bleu_count else None,
        "total": total,
        "denominator": total,
        "rouge_denominator": rouge_count,
        "bleu_denominator": bleu_count,
        "per_example_count": total,
        "metric_definition": "macro mean over valid generation questions",
        "range": [0.0, 1.0],
    }


def _maybe_filter_frame(df: pd.DataFrame, filter_ids: set[str] | None) -> pd.DataFrame:
    if not filter_ids or "ID" not in df.columns:
        return df
    return df[df["ID"].astype(str).isin(filter_ids)].reset_index(drop=True)


def _maybe_limit_frame(df: pd.DataFrame, max_records: int | None) -> pd.DataFrame:
    if max_records is None or max_records <= 0 or len(df) <= max_records:
        return df
    return df.head(max_records).reset_index(drop=True)


def _parse_shot_num(shot_num: str) -> int:
    normalized = str(shot_num).strip().lower()
    if normalized in {"zero", "zero_shot", "0"}:
        return 0
    if normalized in {"one", "one_shot"}:
        return 1
    digits = re.findall(r"\d+", normalized)
    if digits:
        return int(digits[0])
    raise ValueError(f"Unsupported shot_num={shot_num}")


def _sample_few_shot_ids(df: pd.DataFrame, shot_count: int) -> list[str]:
    if shot_count <= 0 or "ID" not in df.columns:
        return []
    ids = df["ID"].astype(str).unique().tolist()
    if not ids:
        return []
    rng = random.Random(42)
    return rng.sample(ids, min(shot_count, len(ids)))


def _build_text_only_prompt(prompt: str, exemplars: list[dict[str, str]]) -> str:
    blocks = []
    for idx, exemplar in enumerate(exemplars):
        blocks.append(
            f"Example {idx + 1}\n{exemplar['prompt']}\nCorrect Answer: {exemplar['answer']}"
        )
    blocks.append(prompt)
    return "\n\n".join(blocks)


def _build_image_content(prompt: str, image, exemplars: list[dict[str, Any]]):
    content: list[dict[str, str]] = []
    images: list[Any] = []

    for idx, exemplar in enumerate(exemplars):
        exemplar_image = exemplar.get("image")
        if exemplar_image is not None:
            content.append({"type": "image"})
            images.append(exemplar_image)
        content.append(
            {
                "type": "text",
                "text": (
                    f"Example {idx + 1}\n{exemplar['prompt']}\n"
                    f"Correct Answer: {exemplar['answer']}"
                ),
            }
        )

    if image is not None:
        content.append({"type": "image"})
        images.append(image)
    content.append({"type": "text", "text": prompt})

    return content, images or None


def _build_classification_few_shots(df: pd.DataFrame, selected_ids: list[str]):
    exemplars = {"image_textual": [], "pure_text": []}
    skip_map: dict[str, dict[str, set[int]]] = {}
    if not selected_ids:
        return exemplars, skip_map

    filtered = df[df["ID"].astype(str).isin(selected_ids)]
    for _, row in filtered.iterrows():
        record = row.to_dict()
        row_id = str(record.get("ID") or "")
        image = _select_record_image(record)
        tasks = record.get("Classification_Task") or {}
        skip_map.setdefault(row_id, {"image_textual": set(), "pure_text": set()})

        for idx, question_data in enumerate(
            _to_list(tasks.get("Image_Textual_Questions"))
        ):
            question = str(question_data.get("Question") or "").strip()
            options = dict(question_data.get("Options") or {})
            correct = str(question_data.get("Correct_Answer") or "").strip().upper()
            if not question or not options or not correct:
                continue
            exemplars["image_textual"].append(
                {
                    "image": image,
                    "prompt": _formulate_prompt_with_options(question, options),
                    "answer": correct,
                }
            )
            skip_map[row_id]["image_textual"].add(idx)

        for idx, question_data in enumerate(_to_list(tasks.get("Pure_Text_Questions"))):
            question = str(question_data.get("Question") or "").strip()
            options = dict(question_data.get("Options") or {})
            correct = str(question_data.get("Correct_Answer") or "").strip().upper()
            if not question or not options or not correct:
                continue
            exemplars["pure_text"].append(
                {
                    "prompt": _formulate_prompt_with_options(question, options),
                    "answer": correct,
                }
            )
            skip_map[row_id]["pure_text"].add(idx)

    return exemplars, skip_map


def _build_fill_blank_few_shots(df: pd.DataFrame, selected_ids: list[str]):
    exemplars = {"image_textual": [], "pure_text": []}
    skip_map: dict[str, dict[str, set[int]]] = {}
    if not selected_ids:
        return exemplars, skip_map

    filtered = df[df["ID"].astype(str).isin(selected_ids)]
    for _, row in filtered.iterrows():
        record = row.to_dict()
        row_id = str(record.get("ID") or "")
        image = _select_record_image(record)
        skip_map.setdefault(row_id, {"image_textual": set(), "pure_text": set()})

        for idx, question_data in enumerate(_to_list(record.get("Mask_Task"))):
            question = str(question_data.get("Question") or "").strip()
            answer = str(question_data.get("Ground_Truth") or "").strip()
            qa_type = str(question_data.get("Type") or "").strip()
            if (
                not question
                or not answer
                or qa_type not in {"Image_Textual", "Pure_Text"}
            ):
                continue

            label = "image_textual" if qa_type == "Image_Textual" else "pure_text"
            exemplars[label].append(
                {
                    "image": image if label == "image_textual" else None,
                    "prompt": _normalize_blank_prompt(question),
                    "answer": answer,
                }
            )
            skip_map[row_id][label].add(idx)

    return exemplars, skip_map


def evaluate_classification(
    parquet_path: str,
    processor,
    model,
    *,
    max_new_tokens: int = 50,
    filter_ids: set[str] | None = None,
    few_shot_df: pd.DataFrame | None = None,
    shot_count: int = 0,
    max_records: int | None = None,
    return_records: bool = False,
):
    df = _maybe_limit_frame(
        _maybe_filter_frame(_load_parquet_frame(parquet_path), filter_ids), max_records
    )
    selected_ids = _sample_few_shot_ids(df, shot_count)
    few_shot_examples, skip_map = (
        _build_classification_few_shots(few_shot_df, selected_ids)
        if shot_count > 0 and few_shot_df is not None
        else ({"image_textual": [], "pure_text": []}, {})
    )

    counters = {
        "image_textual": _new_counter(),
        "pure_text": _new_counter(),
    }
    records: list[dict[str, Any]] = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Classification"):
        record = row.to_dict()
        row_id = str(record.get("ID") or "")
        image = _select_record_image(record)
        tasks = record.get("Classification_Task") or {}

        for idx, question_data in enumerate(
            _to_list(tasks.get("Image_Textual_Questions"))
        ):
            if idx in skip_map.get(row_id, {}).get("image_textual", set()):
                continue
            question = str(question_data.get("Question") or "").strip()
            options = dict(question_data.get("Options") or {})
            correct = str(question_data.get("Correct_Answer") or "").strip().upper()
            if not question or not options or not correct:
                continue

            prompt = _formulate_prompt_with_options(question, options)
            content, images = _build_image_content(
                prompt, image, few_shot_examples["image_textual"]
            )
            generated = _generate_from_content(
                processor,
                model,
                content,
                images=images,
                max_new_tokens=max_new_tokens,
            )
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", generated)
            predicted = cleaned[0].upper() if cleaned else None
            if predicted == correct:
                counters["image_textual"]["correct"] += 1
            counters["image_textual"]["total"] += 1
            records.append(
                {
                    "row_index": int(row_index),
                    "id": row_id,
                    "task": "classification",
                    "modality": "image_textual",
                    "question_index": idx,
                    "question": question,
                    "options": options,
                    "prediction": predicted,
                    "raw_prediction": generated,
                    "ground_truth": correct,
                    "correct": predicted == correct,
                }
            )

        for idx, question_data in enumerate(_to_list(tasks.get("Pure_Text_Questions"))):
            if idx in skip_map.get(row_id, {}).get("pure_text", set()):
                continue
            question = str(question_data.get("Question") or "").strip()
            options = dict(question_data.get("Options") or {})
            correct = str(question_data.get("Correct_Answer") or "").strip().upper()
            if not question or not options or not correct:
                continue

            prompt = _build_text_only_prompt(
                _formulate_prompt_with_options(question, options),
                few_shot_examples["pure_text"],
            )
            generated = _generate_text(
                processor,
                model,
                prompt,
                image=None,
                max_new_tokens=max_new_tokens,
            )
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", generated)
            predicted = cleaned[0].upper() if cleaned else None
            if predicted == correct:
                counters["pure_text"]["correct"] += 1
            counters["pure_text"]["total"] += 1
            records.append(
                {
                    "row_index": int(row_index),
                    "id": row_id,
                    "task": "classification",
                    "modality": "pure_text",
                    "question_index": idx,
                    "question": question,
                    "options": options,
                    "prediction": predicted,
                    "raw_prediction": generated,
                    "ground_truth": correct,
                    "correct": predicted == correct,
                }
            )

    total_counter = {
        "correct": counters["image_textual"]["correct"]
        + counters["pure_text"]["correct"],
        "total": counters["image_textual"]["total"] + counters["pure_text"]["total"],
    }
    metrics = {
        "image_textual": _finalize_counter(counters["image_textual"]),
        "pure_text": _finalize_counter(counters["pure_text"]),
        "total": _finalize_counter(total_counter),
    }
    return (metrics, records) if return_records else metrics


def evaluate_fill_in_the_blank(
    parquet_path: str,
    processor,
    model,
    *,
    max_new_tokens: int = 50,
    filter_ids: set[str] | None = None,
    few_shot_df: pd.DataFrame | None = None,
    shot_count: int = 0,
    max_records: int | None = None,
    return_records: bool = False,
):
    df = _maybe_limit_frame(
        _maybe_filter_frame(_load_parquet_frame(parquet_path), filter_ids), max_records
    )
    selected_ids = _sample_few_shot_ids(df, shot_count)
    few_shot_examples, skip_map = (
        _build_fill_blank_few_shots(few_shot_df, selected_ids)
        if shot_count > 0 and few_shot_df is not None
        else ({"image_textual": [], "pure_text": []}, {})
    )

    counters = {
        "image_textual": _new_counter(),
        "pure_text": _new_counter(),
    }
    records: list[dict[str, Any]] = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="FillBlank"):
        record = row.to_dict()
        row_id = str(record.get("ID") or "")
        image = _select_record_image(record)

        for idx, question_data in enumerate(_to_list(record.get("Mask_Task"))):
            question = str(question_data.get("Question") or "").strip()
            answer = str(question_data.get("Ground_Truth") or "").strip()
            qa_type = str(question_data.get("Type") or "").strip()
            if (
                not question
                or not answer
                or qa_type not in {"Image_Textual", "Pure_Text"}
            ):
                continue

            label = "image_textual" if qa_type == "Image_Textual" else "pure_text"
            if idx in skip_map.get(row_id, {}).get(label, set()):
                continue

            prompt = _normalize_blank_prompt(question)
            if label == "image_textual":
                content, images = _build_image_content(
                    prompt, image, few_shot_examples[label]
                )
                generated = _generate_from_content(
                    processor,
                    model,
                    content,
                    images=images,
                    max_new_tokens=max_new_tokens,
                )
            else:
                generated = _generate_text(
                    processor,
                    model,
                    _build_text_only_prompt(prompt, few_shot_examples[label]),
                    image=None,
                    max_new_tokens=max_new_tokens,
                )

            is_correct = answer.lower() in generated.lower()
            if is_correct:
                counters[label]["correct"] += 1
            counters[label]["total"] += 1
            records.append(
                {
                    "row_index": int(row_index),
                    "id": row_id,
                    "task": "fill_in_the_blank",
                    "modality": label,
                    "question_index": idx,
                    "question": question,
                    "prediction": generated,
                    "ground_truth": answer,
                    "correct": is_correct,
                    "match_rule": "case_insensitive_ground_truth_substring",
                }
            )

    total_counter = {
        "correct": counters["image_textual"]["correct"]
        + counters["pure_text"]["correct"],
        "total": counters["image_textual"]["total"] + counters["pure_text"]["total"],
    }
    metrics = {
        "image_textual": _finalize_counter(counters["image_textual"]),
        "pure_text": _finalize_counter(counters["pure_text"]),
        "total": _finalize_counter(total_counter),
    }
    return (metrics, records) if return_records else metrics


def evaluate_generation(
    parquet_path: str,
    processor,
    model,
    *,
    output_dir: str,
    split_name: str,
    max_new_tokens: int = 200,
    filter_ids: set[str] | None = None,
    max_records: int | None = None,
):
    df = _maybe_limit_frame(
        _maybe_filter_frame(_load_parquet_frame(parquet_path), filter_ids), max_records
    )
    scorer = _safe_rouge_scorer()
    counters = {
        "image_textual": _new_generation_counter(),
        "pure_text": _new_generation_counter(),
    }
    records = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Generation"):
        record = row.to_dict()
        image = _select_record_image(record)

        for question_index, qa in enumerate(_to_list(record.get("Generation_Task"))):
            question = str(qa.get("Question") or "").strip()
            answer = str(qa.get("Ground_Truth") or "").strip()
            qa_type = str(qa.get("Type") or "").strip()
            if (
                not question
                or not answer
                or qa_type not in {"Image_Textual", "Pure_Text"}
            ):
                continue

            label = "image_textual" if qa_type == "Image_Textual" else "pure_text"
            generated = _generate_text(
                processor,
                model,
                question,
                image=image if label == "image_textual" else None,
                max_new_tokens=max_new_tokens,
            )

            example_scores = {
                "rouge1": None,
                "rouge2": None,
                "rougeL": None,
                "bleu": None,
            }
            if scorer is not None:
                rouge_scores = scorer.score(answer, generated)
                for metric_name in ("rouge1", "rouge2", "rougeL"):
                    value = rouge_scores[metric_name].fmeasure
                    counters[label][metric_name] += value
                    example_scores[metric_name] = value
                counters[label]["rouge_count"] += 1
            bleu = _compute_bleu(answer, generated)
            if bleu is not None:
                counters[label]["bleu"] += bleu
                counters[label]["bleu_count"] += 1
                example_scores["bleu"] = bleu
            counters[label]["total"] += 1

            records.append(
                {
                    "row_index": int(row_index),
                    "id": str(record.get("ID") or ""),
                    "task": "generation",
                    "modality": label,
                    "question_index": question_index,
                    "question": question,
                    "prediction": generated,
                    "ground_truth": answer,
                    "scores": example_scores,
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    detail_file = os.path.join(output_dir, f"{split_name}_generation_results.json")
    with open(detail_file, "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    total_counter = _new_generation_counter()
    for label in ("image_textual", "pure_text"):
        for key in (
            "rouge1",
            "rouge2",
            "rougeL",
            "bleu",
            "total",
            "rouge_count",
            "bleu_count",
        ):
            total_counter[key] += counters[label][key]

    return {
        "image_textual": _finalize_generation_counter(counters["image_textual"]),
        "pure_text": _finalize_generation_counter(counters["pure_text"]),
        "total": _finalize_generation_counter(total_counter),
    }


class MLLMUEvaluator:
    """Config-driven MLLMU evaluator used by mm_eval.py."""

    def __init__(self, eval_cfg, model, processor):
        self.eval_cfg = eval_cfg
        self.model = model
        self.processor = processor
        self.data_dir = eval_cfg.get("data_dir", "data/MLLMU-Bench")
        self.output_dir = eval_cfg.get("output_dir", "saves/mm_eval/mllmu")
        self.forget_ratio = int(eval_cfg.get("forget_split_ratio", 5))
        self.tasks = list(
            eval_cfg.get(
                "tasks",
                ["classification", "fill_in_the_blank", "generation"],
            )
        )
        self.splits = list(
            eval_cfg.get(
                "splits",
                ["forget", "retain_shared", "retain_celebrity", "test"],
            )
        )
        self.max_new_tokens = int(eval_cfg.get("max_new_tokens", 50))
        self.generation_max_new_tokens = int(
            eval_cfg.get("generation_max_new_tokens", 200)
        )
        self.test_data_dir = str(
            eval_cfg.get("test_data_dir", os.path.join(self.data_dir, "Test_Set"))
        )
        self.retain_celebrity_data = str(
            eval_cfg.get(
                "retain_celebrity_data",
                os.path.join(
                    self.data_dir, "Retain_Set", "train-00000-of-00001.parquet"
                ),
            )
        )
        self.few_shot_parquet = str(
            eval_cfg.get(
                "few_shot_parquet",
                os.path.join(self.data_dir, "Full_Set", "train-00000-of-00001.parquet"),
            )
        )
        self.shot_num = str(eval_cfg.get("shot_num", "zero_shot"))
        self.shot_count = _parse_shot_num(self.shot_num)
        self.task_semantics = str(
            eval_cfg.get("task_semantics", "full_multimodal_deletion")
        ).lower()
        if self.task_semantics not in {
            "full_multimodal_deletion",
            "visual_entity_unlearning",
        }:
            raise ValueError(
                "MLLMU task_semantics must be full_multimodal_deletion or "
                "visual_entity_unlearning"
            )
        self.max_records = eval_cfg.get("max_records")
        self.max_records = (
            int(self.max_records) if self.max_records is not None else None
        )

    def _benchmark_metadata(self) -> dict[str, Any]:
        """Describe exactly what is native, external, and optional upstream."""
        return {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "native_tasks": {
                "classification": {
                    "metric": "accuracy",
                    "formula": "correct / valid multiple-choice questions",
                    "range": [0.0, 1.0],
                },
                "fill_in_the_blank": {
                    "metric": "substring accuracy",
                    "formula": (
                        "case-insensitive ground-truth substring matches / "
                        "valid cloze questions"
                    ),
                    "range": [0.0, 1.0],
                },
                "generation": {
                    "metrics": ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1", "BLEU"],
                    "aggregation": "macro mean over valid questions",
                    "range": [0.0, 1.0],
                },
            },
            "input_schema": {
                "required_profile_fields": [
                    "ID",
                    "Classification_Task",
                    "Mask_Task",
                    "Generation_Task",
                ],
                "image_fields": ["image", "images"],
                "classification_fields": [
                    "Question",
                    "Options",
                    "Correct_Answer",
                ],
                "cloze_generation_fields": ["Question", "Ground_Truth", "Type"],
            },
            "shot_num": self.shot_num,
            "task_semantics": self.task_semantics,
            "modality_aggregation_policy": (
                "report_image_textual_and_pure_text_separately"
                if self.task_semantics == "visual_entity_unlearning"
                else "benchmark_native_total_and_per_modality"
            ),
            "empty_split_policy": "metric value is null with an explicit reason",
        }

    def _split_semantics(self, split_name: str) -> dict[str, Any] | None:
        base = SPLIT_SEMANTICS.get(split_name)
        if base is None:
            return None
        semantics = dict(base)
        if self.task_semantics == "visual_entity_unlearning" and split_name in {
            "forget",
            "test",
        }:
            semantics.update(
                {
                    "role": "visual_deletion_with_target_text_preservation",
                    "direction": "modality_dependent",
                    "per_modality_direction": {
                        "image_textual": "lower_is_better_after_unlearning",
                        "pure_text": "higher_is_better_for_preservation",
                    },
                    "total_metric_comparability": "do_not_combine_opposite_directions",
                }
            )
        return semantics

    @staticmethod
    def _write_per_example(
        output_dir: str,
        split_name: str,
        task_name: str,
        records: list[dict[str, Any]],
    ) -> str:
        detail_file = os.path.join(output_dir, f"{split_name}_{task_name}_results.json")
        with open(detail_file, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        return detail_file

    def _missing_split_results(self, reason: str) -> dict[str, Any]:
        return {
            task: _task_unavailable(reason)
            for task in self.tasks
            if task in {"classification", "fill_in_the_blank", "generation"}
        }

    def _split_path(self, split_name: str) -> str:
        retain_ratio = 100 - self.forget_ratio
        mapping = {
            "forget": os.path.join(self.data_dir, f"forget_{self.forget_ratio}"),
            "retain": os.path.join(self.data_dir, f"retain_{retain_ratio}"),
            "retain_shared": os.path.join(self.data_dir, f"retain_{retain_ratio}"),
            "retain_celebrity": self.retain_celebrity_data,
            "real": self.retain_celebrity_data,
            "test": self.test_data_dir,
        }
        if split_name not in mapping:
            raise ValueError(f"Unsupported MLLMU split={split_name}")
        return mapping[split_name]

    def _forget_ids(self) -> set[str]:
        forget_path = self._split_path("forget")
        df = _load_parquet_frame(forget_path)
        if "ID" not in df.columns:
            return set()
        return set(df["ID"].astype(str).tolist())

    def evaluate(self):
        os.makedirs(self.output_dir, exist_ok=True)
        results: dict[str, Any] = {
            "benchmark_metadata": self._benchmark_metadata(),
        }

        forget_ids = None
        test_filter_issue = None
        if "test" in self.splits:
            try:
                forget_ids = self._forget_ids()
                if not forget_ids:
                    test_filter_issue = "forget_split_contains_no_ID_values"
            except FileNotFoundError as exc:
                test_filter_issue = (
                    f"forget_split_required_for_test_filter_missing: {exc}"
                )

        few_shot_df = None
        few_shot_issue = None
        if self.shot_count > 0:
            try:
                few_shot_df = _load_parquet_frame(self.few_shot_parquet)
            except FileNotFoundError as exc:
                few_shot_issue = f"requested_few_shot_data_missing: {exc}"

        for split_name in self.splits:
            split_path = self._split_path(split_name)
            if not os.path.exists(split_path):
                reason = f"split_path_missing: {split_path}"
                logger.warning("MLLMU %s unavailable: %s", split_name, reason)
                results[split_name] = {
                    "status": "unavailable",
                    "reason": reason,
                    "semantics": self._split_semantics(split_name),
                    **self._missing_split_results(reason),
                }
                continue

            if split_name == "test" and test_filter_issue:
                results[split_name] = {
                    "status": "unavailable",
                    "reason": test_filter_issue,
                    "semantics": self._split_semantics(split_name),
                    **self._missing_split_results(test_filter_issue),
                }
                continue

            if few_shot_issue:
                results[split_name] = {
                    "status": "unavailable",
                    "reason": few_shot_issue,
                    "semantics": self._split_semantics(split_name),
                    **self._missing_split_results(few_shot_issue),
                }
                continue

            split_results: dict[str, Any] = {
                "status": "available",
                "reason": None,
                "semantics": self._split_semantics(split_name),
                "data_path": split_path,
            }
            split_filter_ids = forget_ids if split_name == "test" else None

            if "classification" in self.tasks:
                classification, records = evaluate_classification(
                    split_path,
                    self.processor,
                    self.model,
                    max_new_tokens=self.max_new_tokens,
                    filter_ids=split_filter_ids,
                    few_shot_df=few_shot_df,
                    shot_count=self.shot_count,
                    max_records=self.max_records,
                    return_records=True,
                )
                split_results["classification"] = classification
                self._write_per_example(
                    self.output_dir, split_name, "classification", records
                )
            if "fill_in_the_blank" in self.tasks:
                fill_in_the_blank, records = evaluate_fill_in_the_blank(
                    split_path,
                    self.processor,
                    self.model,
                    max_new_tokens=self.max_new_tokens,
                    filter_ids=split_filter_ids,
                    few_shot_df=few_shot_df,
                    shot_count=self.shot_count,
                    max_records=self.max_records,
                    return_records=True,
                )
                split_results["fill_in_the_blank"] = fill_in_the_blank
                self._write_per_example(
                    self.output_dir, split_name, "fill_in_the_blank", records
                )
            if "generation" in self.tasks:
                split_results["generation"] = evaluate_generation(
                    split_path,
                    self.processor,
                    self.model,
                    output_dir=self.output_dir,
                    split_name=split_name,
                    max_new_tokens=self.generation_max_new_tokens,
                    filter_ids=split_filter_ids,
                    max_records=self.max_records,
                )
            results[split_name] = split_results

        results["external_evaluations"] = {
            "MMMU": _unavailable(
                "external_benchmark_runner_not_bundled_by_MLLMU_Bench",
                upstream_guidance="run the official MMMU repository",
            ),
            "LLaVA-Bench": _unavailable(
                "external_benchmark_runner_not_bundled_by_MLLMU_Bench",
                upstream_guidance="run the official LLaVA-Bench repository",
            ),
            "generation_factuality": _unavailable(
                "optional_external_LLM_judge_not_run",
                upstream_script="eval_gpt.py",
            ),
        }

        output_file = os.path.join(self.output_dir, "MLLMU_EVAL.json")
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        logger.info("MLLMU eval results saved to %s", output_file)
        return results
