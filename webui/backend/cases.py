"""Read saved evaluator examples without reconstructing or inventing model answers."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

MAX_CASE_FILE_BYTES = 12 * 1024 * 1024


def _number(value: Any) -> int | float | None:
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _text(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        if isinstance(record.get(key), str):
            return record[key]
    return None


def _pointer(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _pair_key(record: dict[str, Any]) -> str | None:
    # Position in a filtered list is insufficient. Verify identity and text.
    prompt = _text(record, "prompt", "input")
    expected = _text(record, "expected", "ground_truth", "reference")
    identity = record.get("id", record.get("case_index", record.get("index")))
    if prompt is None:
        return None
    return json.dumps(
        [
            identity,
            prompt,
            expected,
            record.get("source_dataset"),
            record.get("source_row_index"),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def _record(
    item: dict[str, Any],
    *,
    source: str,
    pointer: str,
    metric: dict[str, Any],
    context: dict[str, Any],
    before_item: dict[str, Any] | None = None,
    pairing_warning: str | None = None,
) -> dict[str, Any]:
    before = _text(
        item, "pre_prediction_text", "pre_prediction", "before_response", "before"
    )
    after = _text(
        item,
        "post_prediction_text",
        "prediction_text",
        "post_prediction",
        "prediction",
        "generated",
        "generation",
        "after_response",
        "after",
    )
    if before is None and before_item is not None:
        before = _text(
            before_item, "generated", "generation", "prediction", "prediction_text"
        )
    score = next(
        (
            _number(item[key])
            for key in (
                "score",
                "prob",
                "target_nll",
                "avg_loss",
                "rougeL_recall",
                "exact_match",
            )
            if _number(item.get(key)) is not None
        ),
        None,
    )
    before_score = _number(item.get("pre_correct", item.get("before_score")))
    after_score = _number(item.get("post_correct", item.get("after_score", score)))
    if isinstance(item.get("correct"), bool):
        after_score = float(item["correct"])
    if before_item is not None:
        if isinstance(before_item.get("correct"), bool):
            before_score = float(before_item["correct"])
        elif _number(before_item.get("target_nll")) is not None:
            before_score = before_item["target_nll"]
    kind = item.get("measurement_kind")
    if "prediction_text" in item or "post_prediction_text" in item:
        kind = kind or "teacher_forced_token_prediction"
    elif any(
        key in item
        for key in ("generated", "generation", "prediction", "post_prediction")
    ):
        kind = kind or "generation"
    elif "target_nll" in item:
        kind = kind or "target_token_nll"
    elif "prob" in item:
        kind = kind or "target_probability"
    kind = kind or context.get("measurement_kind") or metric.get("measurement_kind")
    reason = item.get("unavailable_reason")
    if reason is None:
        reason = (
            "responses_not_recorded"
            if before is None and after is None
            else "before_not_recorded"
            if before is None
            else "after_not_recorded"
            if after is None
            else None
        )
    status = (
        "correct"
        if item.get("correct") is True
        else "incorrect"
        if item.get("correct") is False
        else "measured"
        if score is not None or after_score is not None
        else "recorded"
        if before is not None or after is not None
        else "unavailable"
    )
    return {
        "id": hashlib.sha256(f"{source}#{pointer}".encode()).hexdigest()[:20],
        "metric_id": metric["id"],
        "evaluator": metric["evaluator"],
        "phase": metric["phase"],
        "case_index": item.get("case_index", item.get("index", item.get("id"))),
        "probe_index": item.get("probe_index"),
        "group": item.get("group"),
        "prompt": _text(item, "prompt", "input"),
        "expected": _text(item, "expected", "ground_truth", "reference"),
        "before": before,
        "after": after,
        "status": status,
        "measurement_kind": kind,
        "score": score,
        "before_score": before_score,
        "after_score": after_score,
        "source": source,
        "source_pointer": pointer,
        "unavailable_reason": reason,
        "warnings": [pairing_warning] if pairing_warning else [],
        "protocol_version": context.get(
            "protocol_version", metric.get("protocol_version")
        ),
        "scoring": context.get("scoring", metric.get("scoring")),
    }


def read_cases(
    run_dir: Path, metrics: list[dict[str, Any]], *, checkpoint_eval: Path | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Adapt text edit, inject-v2 and saved unlearning examples; no model loading.

    Before/after lists are paired only with unique matching identity and text.
    Score-only records retain null text. External input files and model caches
    are never read to fill missing fields.
    """
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    paths = sorted(run_dir.glob("*_EVAL.json"))
    if checkpoint_eval is not None:
        paths.extend(sorted(checkpoint_eval.glob("*_EVAL.json")))
    if (run_dir / "unlearning_metrics.json").is_file():
        paths.append(run_dir / "unlearning_metrics.json")
    for path in paths:
        source = path.relative_to(run_dir).as_posix()
        if not path.resolve().is_relative_to(run_dir.resolve()):
            warnings.append(f"Skipped artifact outside the run directory: {source}")
            continue
        try:
            if path.stat().st_size > MAX_CASE_FILE_BYTES:
                warnings.append(
                    f"Artifact exceeds the case viewer size limit: {source}"
                )
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            warnings.append(f"Could not read case artifact: {source}")
            continue
        if not isinstance(payload, dict):
            continue
        summary_source = source.replace("_EVAL.json", "_SUMMARY.json")
        available = [metric for metric in metrics if metric["source"] == summary_source]

        def select(name):
            return next(
                (metric for metric in available if metric["metric_name"] == name), None
            )

        def add_examples(examples, pointer, metric, context, references=None):
            if metric is None or not isinstance(examples, list):
                return
            reference_map = defaultdict(list)
            for reference in references or []:
                if isinstance(reference, dict):
                    key = _pair_key(reference)
                    if key is not None:
                        reference_map[key].append(reference)
            current_counts = defaultdict(int)
            for item in examples:
                if isinstance(item, dict):
                    current_counts[_pair_key(item)] += 1
            for index, item in enumerate(examples):
                if not isinstance(item, dict):
                    continue
                before_item = None
                warning = None
                if references is not None:
                    key = _pair_key(item)
                    candidates = reference_map.get(key, [])
                    if (
                        key is not None
                        and len(candidates) == 1
                        and current_counts[key] == 1
                    ):
                        before_item = candidates[0]
                    else:
                        warning = "reference_example_not_uniquely_matched"
                rows.append(
                    _record(
                        item,
                        source=source,
                        pointer=f"{pointer}/{index}",
                        metric=metric,
                        context=context,
                        before_item=before_item,
                        pairing_warning=warning,
                    )
                )

        if payload.get("protocol_version") == "inject-v2":
            add_examples(
                payload.get("examples"), "/examples", select("task_accuracy"), payload
            )
            add_examples(
                payload.get("current_examples"),
                "/current_examples",
                select("knowledge_retention") or select("current_score"),
                payload,
                payload.get("original_examples", []),
            )
            continue
        if path.name == "unlearning_metrics.json":
            # This artifact pairs the same explicit indexed split within one evaluation.
            for split in ("forget", "retain"):
                metric = select(f"edited_{split}_nll")
                base = payload.get("base", {}).get(split, {}).get("per_example", [])
                edited = payload.get("edited", {}).get(split, {}).get("per_example", [])
                if (
                    metric is None
                    or not isinstance(base, list)
                    or not isinstance(edited, list)
                ):
                    continue
                indexed = defaultdict(list)
                for item in base:
                    if isinstance(item, dict) and type(item.get("index")) is int:
                        indexed[item["index"]].append(item)
                seen = [item.get("index") for item in edited if isinstance(item, dict)]
                for index, item in enumerate(edited):
                    if not isinstance(item, dict):
                        continue
                    candidates = indexed.get(item.get("index"), [])
                    paired = (
                        candidates[0]
                        if len(candidates) == 1 and seen.count(item.get("index")) == 1
                        else None
                    )
                    rows.append(
                        _record(
                            {**item, "group": split},
                            source=source,
                            pointer=f"/edited/{split}/per_example/{index}",
                            metric=metric,
                            context=payload,
                            before_item=paired,
                            pairing_warning=None
                            if paired
                            else "reference_example_not_uniquely_matched",
                        )
                    )
            continue
        for name, group in payload.items():
            if not isinstance(group, dict):
                continue
            metric = select(
                {"long_form": "rouge_l", "comprehensive": "overall_score"}.get(
                    name, name
                )
            )
            context = {**payload, **group}
            add_examples(
                group.get("per_example"),
                f"/{_pointer(name)}/per_example",
                metric,
                context,
            )
            values = group.get("value_by_index")
            if metric is not None and isinstance(values, dict):
                for index, item in values.items():
                    if isinstance(item, dict):
                        rows.append(
                            _record(
                                {"index": index, **item},
                                source=source,
                                pointer=f"/{_pointer(name)}/value_by_index/{_pointer(index)}",
                                metric=metric,
                                context=context,
                            )
                        )
                    elif _number(item) is not None:
                        rows.append(
                            _record(
                                {"index": index, "score": item},
                                source=source,
                                pointer=f"/{_pointer(name)}/value_by_index/{_pointer(index)}",
                                metric=metric,
                                context=context,
                            )
                        )
    if not rows:
        warnings.append("No supported per-example records were saved for this result.")
    return rows, warnings
