"""
Knowledge Injection 评估器
=========================

实现知识注入（微调）的评估指标：
- Task Accuracy（任务准确率）：微调任务的性能
- Knowledge Retention（知识保持）：原有知识的保持率
"""

import logging
import re
import string
from collections import Counter
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import torch
from data.inject import render_alpaca_texts, tokenize_inject_pair
from evals.base import Evaluator
from model.reft_positions import (
    build_loreft_unit_locations,
    get_loreft_position_config,
)
from evals.fact_answers import FACT_ANSWER_SCORING_VERSION, fact_answer_scores

logger = logging.getLogger(__name__)

_COST_RE = re.compile(
    r"(?:total\s+cost\s+is\s*\$?|\$)\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


GSM8K_SCORING_VERSION = "final-conclusion-v3"


def gsm8k_final_answer(text, *, require_marker=True):
    """Read explicit answer conclusions in order, without using intermediate numbers.

    Strict scoring requires a valid final ``####`` marker and rejects any later
    conflicting or invalid explicit conclusion. Flexible scoring uses the last
    explicit answer, including an invalid one, so it cannot fall back to an
    earlier plausible number. Empty trailing markers are flexible-only tolerance.
    """
    text = str(text).strip()
    if not require_marker:
        text = re.sub(r"(?:\s*####\s*)+$", "", text).strip()
    number = r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?"

    def scalar(value):
        match = re.fullmatch(r"(" + number + r")[.!?]?", value.strip())
        return Decimal(match.group(1).replace(",", "")) if match else None

    def conclusion(value):
        # Reject ambiguous numeric alternatives and partial forms such as 4e2
        # or 4.2.3 instead of extracting a valid-looking numeric fragment.
        values = re.findall(r"(?<![\w.])" + number + r"(?!\w|\.\d)", value)
        if len(values) == 1:
            return scalar(values[0])
        # Preserve an unambiguous boxed answer repeated verbatim in the same
        # conclusion; different numeric alternatives still invalidate it.
        if values and re.search(r"\\boxed\s*\{", value):
            unique = {scalar(item) for item in values}
            if len(unique) == 1:
                return unique.pop()
        return None

    answers = []
    covered_spans = []
    marker_matches = list(re.finditer(r"####", text))
    for match in marker_matches:
        remainder = text[match.end() :].lstrip()
        line = remainder.splitlines()[0] if remainder else ""
        answers.append((match.start(), "marker", scalar(line)))
        covered_spans.append((match.start(), len(text) - len(remainder) + len(line)))
    # Strong answer statements can follow an earlier boxed value on the same
    # line. Weak discourse cues remain restricted to the beginning of a line.
    explicit = re.compile(
        r"(?:^[ \t*_]*(?:therefore|thus|hence|so)\b|"
        r"\b(?:the (?:final )?answer(?: is)?|final answer)\b)",
        re.IGNORECASE | re.MULTILINE,
    )
    statements = list(explicit.finditer(text))
    for index, match in enumerate(statements):
        line_end = text.find("\n", match.end())
        end = line_end if line_end >= 0 else len(text)
        # A later explicit statement or marker starts a new conclusion. A boxed
        # value inside this conclusion is part of it, not an independent escape
        # from an ambiguous statement such as "Final answer: boxed42 or43".
        later = [
            item.start() for item in marker_matches if item.start() > match.start()
        ]
        if index + 1 < len(statements):
            later.append(statements[index + 1].start())
        if later:
            end = min(end, min(later))
        answers.append(
            (match.start(), "conclusion", conclusion(text[match.end() : end]))
        )
        covered_spans.append((match.start(), end))
    for match in re.finditer(r"\\boxed\s*\{", text):
        if any(start <= match.start() < end for start, end in covered_spans):
            continue
        end = text.find("}", match.end())
        value = scalar(text[match.end() : end]) if end >= 0 else None
        answers.append((match.start(), "boxed", value))
    if text:
        last_line = text.splitlines()[-1]
        bare = re.fullmatch(r"\$?(" + number + r")\.?", last_line)
        if bare:
            answers.append((len(text) - len(last_line), "bare", scalar(bare.group(1))))
    answers.sort(key=lambda answer: answer[0])
    if require_marker:
        markers = [answer for answer in answers if answer[1] == "marker"]
        if not markers or markers[-1][2] is None:
            return None
        position, _, value = markers[-1]
        if any(other != value for start, _, other in answers if start > position):
            return None
        return value
    return answers[-1][2] if answers else None


def generation_token_details(token_ids, model, tokenizer, max_new_tokens):
    """Observe EOS versus budget exhaustion without modifying generated text."""
    config = getattr(model, "generation_config", None)
    if config is None:
        config = getattr(getattr(model, "model", None), "generation_config", None)
    eos = getattr(config, "eos_token_id", getattr(tokenizer, "eos_token_id", None))
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    stop = next((index for index, value in enumerate(ids) if value in eos_ids), None)
    count = stop + 1 if stop is not None else len(ids)
    return {
        "output_token_count": count,
        "eos_terminated": stop is not None,
        "finish_reason": "eos"
        if stop is not None
        else ("length" if count >= max_new_tokens else "other"),
    }


def short_answer_scores(generated, references):
    """Score full answers plus first-line exact/token and legacy substring diagnostics.

    ``first_line_match`` retains raw substring matching for compatibility. The
    added normalized exact/token fields reject empty references and substrings
    inside another word, such as Bini inside combining.
    """

    def tokens(text):
        text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predicted = tokens(generated)
    references = references if isinstance(references, list) else [references]
    normalized = [tokens(reference) for reference in references]
    scores = []
    for reference in normalized:
        overlap = sum((Counter(predicted) & Counter(reference)).values())
        scores.append(
            2 * overlap / (len(predicted) + len(reference)) if overlap else 0.0
        )
    first_line = generated.strip().splitlines()[0] if generated.strip() else ""
    first_tokens = tokens(first_line)
    first_line_exact = bool(first_tokens) and first_tokens in normalized
    first_line_token = any(
        reference
        and any(
            first_tokens[start : start + len(reference)] == reference
            for start in range(len(first_tokens) - len(reference) + 1)
        )
        for reference in normalized
    )
    return {
        "answer_exact_match": bool(predicted) and predicted in normalized,
        "answer_token_f1": max(scores, default=0.0),
        "first_line_exact_match": first_line_exact,
        "first_line_token_match": first_line_token,
        "first_line_match": any(
            str(answer).strip().lower() in first_line.lower() for answer in references
        ),
    }


class InjectEvaluator(Evaluator):
    """知识注入评估器基类"""

    def __init__(self, eval_cfg, **kwargs):
        self.name = "inject"
        super().__init__(self.name, eval_cfg, **kwargs)

    def _is_strict(self) -> bool:
        return bool(self.eval_cfg.get("strict", False))

    def _get_model_device(self, model) -> torch.device:
        if hasattr(model, "device"):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _normalize_text(self, value: Any) -> str:
        while isinstance(value, list):
            if not value:
                return ""
            value = value[0]
        if value is None:
            return ""
        return str(value).strip()

    def _join_prompt_target(self, prompt: str, target: str) -> str:
        if not prompt:
            return target
        if not target:
            return prompt
        if prompt[-1].isspace() or target[0].isspace():
            return prompt + target
        return f"{prompt} {target}"

    def _get_pad_token_id(self, tokenizer):
        if tokenizer.pad_token_id is not None:
            return tokenizer.pad_token_id
        return tokenizer.eos_token_id

    def _render_prompt(self, item, tokenizer, template_args=None) -> str:
        prompt = self._normalize_text(item.get("prompt"))
        if prompt:
            return prompt
        instruction = self._normalize_text(
            item.get("instruction", item.get("question", ""))
        )
        input_text = self._normalize_text(item.get("input", ""))
        if not instruction:
            return ""
        prompt, _ = render_alpaca_texts(
            instruction,
            input_text,
            tokenizer=tokenizer,
            template_args=template_args,
        )
        return prompt

    def _tokenize_prompt(self, tokenizer, prompt: str, full_text: str | None = None):
        max_length = int(self.eval_cfg.get("args", {}).get("max_length", 2048))
        if full_text is not None:
            features = tokenize_inject_pair(
                prompt,
                full_text,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            prompt_length = features["prompt_length"]
            input_ids = features["input_ids"][:prompt_length].unsqueeze(0)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

    def _render_target(self, expected: str, tokenizer, template_args=None) -> str:
        template_args = template_args or {}
        target = template_args.get("response_prefix", "") + expected
        if template_args.get("apply_chat_template", False) and tokenizer.eos_token:
            target += tokenizer.eos_token
        return target

    def _render_training_text(
        self, item, prompt: str, expected: str, tokenizer, template_args=None
    ) -> str:
        if not self._normalize_text(item.get("prompt")):
            instruction = self._normalize_text(
                item.get("instruction", item.get("question", ""))
            )
            input_text = self._normalize_text(item.get("input", ""))
            if instruction:
                _, full_text = render_alpaca_texts(
                    instruction,
                    input_text,
                    expected,
                    tokenizer=tokenizer,
                    template_args=template_args,
                )
                if full_text is not None:
                    return full_text
        target = self._render_target(expected, tokenizer, template_args)
        return self._join_prompt_target(prompt, target)

    def _generate(self, model, inputs, prompt_token_len, **generation_args):
        if model.__class__.__name__ != "ReftModel":
            return model.generate(**inputs, **generation_args)[0]

        unit_locations = build_loreft_unit_locations(
            inputs["attention_mask"],
            [prompt_token_len],
            len(model.sorted_keys),
            **get_loreft_position_config(model),
        )
        _, output_ids = model.generate(
            base=inputs,
            unit_locations=unit_locations,
            intervene_on_prompt=True,
            **generation_args,
        )
        return output_ids[0]

    def _extract_currency(self, text: str) -> Optional[float]:
        match = _COST_RE.search(text or "")
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _scoring_profile(self, item=None):
        """Resolve explicit GSM evaluation profiles over legacy GSM row labels."""
        item = item or {}
        configured = str(self.eval_cfg.get("args", {}).get("scoring") or "").lower()
        row_profile = str(item.get("scoring") or "").lower()
        source = str(item.get("source_dataset") or "").lower()
        gsm_profiles = {"gsm8k", "gsm8k_numeric"}
        if configured in gsm_profiles and (
            source == "gsm8k" or row_profile in gsm_profiles
        ):
            return configured
        profile = row_profile or configured
        if source == "gsm8k" and profile not in {
            "exact_match",
            "short_answer",
            "gsm8k_numeric",
        }:
            return "gsm8k"
        return profile

    def _score_prediction(
        self, expected: str, generated: str, item: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Score a prediction; RuleArena-Airline uses numeric cost matching."""
        item = item or {}
        source = str(item.get("source_dataset", "")).lower()
        domain = str(item.get("domain", "")).lower()
        scoring = self._scoring_profile(item)

        if scoring == "exact_match":
            return expected.strip().casefold() == generated.strip().casefold()

        if scoring == "short_answer":
            return short_answer_scores(generated, item.get("references", [expected]))[
                "answer_exact_match"
            ]

        if scoring == "fact_answer":
            # A lexical answer metric, not a semantic judge. The full-output
            # format score and review flags are persisted separately below.
            return fact_answer_scores(generated, item.get("references", [expected]))[
                "first_answer_token_match"
            ]

        if scoring in {"gsm8k", "gsm8k_numeric"}:
            reference = gsm8k_final_answer(expected, require_marker=False)
            answer = gsm8k_final_answer(generated, require_marker=scoring == "gsm8k")
            return reference is not None and answer is not None and answer == reference

        if source == "ruletaker":
            labels = re.findall(r"\b(?:not entailment|entailment)\b", generated.lower())
            return bool(labels) and labels[-1] == expected.lower()

        use_cost = (
            scoring in {"rulearena", "cost", "currency"}
            or source == "rulearena"
            or domain == "airline"
        )
        if use_cost:
            expected_cost = self._extract_currency(expected)
            if expected_cost is None:
                try:
                    expected_cost = float(
                        str(item.get("answer", expected)).replace("$", "")
                    )
                except ValueError:
                    expected_cost = None
            generated_cost = self._extract_currency(generated)
            if expected_cost is not None and generated_cost is not None:
                return abs(expected_cost - generated_cost) <= 0.01
            # Fall through to substring match if parsing fails.

        return expected.lower() in generated.lower()

    def _load_cached_result(self, output_dir=None):
        """Recompute results until cache files bind the full evaluation identity.

        Existing files do not identify model weights, input records, templates,
        and decoding settings together. A matching protocol/scoring version or
        output directory therefore cannot establish that a result is reusable.
        Keep those files readable for reporting, but never treat them as a
        substitute for evaluating the current inputs.
        """
        return None

    def _summary_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        for metric in ("task_accuracy", "knowledge_retention", "perplexity"):
            if metric in result:
                value = result[metric]
                summary = {metric: None if value is None else float(value)}
                if "retention_available" in result:
                    summary["retention_available"] = bool(result["retention_available"])
                if result.get("unavailable_reason") is not None:
                    summary["unavailable_reason"] = str(result["unavailable_reason"])
                for key in (
                    "current_score",
                    "original_score",
                    "correct",
                    "total",
                    "current_correct",
                    "current_total",
                    "original_correct",
                    "original_total",
                    "total_tokens",
                    "skipped",
                    "protocol_version",
                    "scoring",
                    "score_semantics",
                    "task_accuracy_metric",
                    "answer_exact_match",
                    "answer_token_f1",
                    "first_line_match",
                    "first_line_exact_match",
                    "first_line_token_match",
                    "first_answer_exact_match",
                    "first_answer_token_match",
                    "fact_answer_scoring_version",
                    "fact_answer_count",
                    "review_flag_rate",
                    "group_metrics",
                    "answer_extraction_rate",
                    "numeric_extraction_rate",
                    "strict_marker_extraction_rate",
                    "strict_marker_accuracy",
                    "numeric_flexible_accuracy",
                    "gsm8k_scoring_version",
                    "length_limit_rate",
                    "eos_termination_rate",
                    "mean_output_tokens",
                ):
                    if key in result:
                        summary[key] = result[key]
                return summary
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    def _persist_result(self, result: Dict[str, Any], output_dir=None):
        result["protocol_version"] = "inject-v2"
        if "perplexity" in result:
            result["scoring"] = "causal-token-nll-v2"
        else:
            result.setdefault(
                "scoring", self._scoring_profile() or "dataset-aware-substring-v1"
            )
            if result["scoring"] == "fact_answer":
                result["fact_answer_scoring_version"] = FACT_ANSWER_SCORING_VERSION
                result["score_semantics"] = (
                    "first-answer token-span presence; not semantic accuracy; "
                    "negation/correction/continuation flags require review"
                )
        target_output_dir = (
            output_dir if output_dir else self.eval_cfg.get("output_dir")
        )
        if target_output_dir is None:
            logger.warning(
                "No output_dir configured for `%s`; skip persisting results.",
                self.name,
            )
            return
        logs_file = self.get_logs_file_path(target_output_dir)
        summary_file = self.get_logs_file_path(target_output_dir, suffix="SUMMARY")
        self.save_logs(result, logs_file)
        self.save_logs(self._summary_from_result(result), summary_file)
        logger.info("Saved `%s` evaluation to %s", self.name, logs_file)


class InjectAccuracyEvaluator(InjectEvaluator):
    """微调任务准确率评估器"""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "inject_accuracy"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        eval_data = kwargs.get("eval_data", [])
        template_args = kwargs.get("template_args")
        eval_method = self.eval_cfg.get("args", {}).get("eval_method", "generation")

        cached = self._load_cached_result(output_dir=output_dir)
        if cached is not None:
            return cached

        if tokenizer is None or not eval_data:
            if self._is_strict():
                missing = []
                if tokenizer is None:
                    missing.append("tokenizer")
                if not eval_data:
                    missing.append("eval_data / eval_data_path")
                raise ValueError(
                    "InjectAccuracyEvaluator requires "
                    + ", ".join(missing)
                    + ". Refusing to write a fake 0.0 summary (strict=true)."
                )
            reason = "missing_tokenizer" if tokenizer is None else "missing_eval_data"
            if eval_method == "perplexity":
                result = {
                    "perplexity": None,
                    "avg_loss": None,
                    "total_tokens": 0,
                    "unavailable_reason": reason,
                }
            else:
                result = {
                    "task_accuracy": None,
                    "correct": 0,
                    "total": 0,
                    "unavailable_reason": reason,
                }
            self._persist_result(result, output_dir=output_dir)
            return result

        model = self.prepare_model(model)
        if eval_method == "generation":
            result = self._eval_generation(
                model, tokenizer, eval_data, template_args=template_args
            )
        elif eval_method == "perplexity":
            result = self._eval_perplexity(
                model, tokenizer, eval_data, template_args=template_args
            )
        else:
            raise ValueError(
                f"Unsupported eval_method `{eval_method}` for InjectAccuracyEvaluator."
            )

        self._persist_result(result, output_dir=output_dir)
        return result

    def _eval_generation(
        self,
        model,
        tokenizer,
        eval_data: List[Dict[str, Any]],
        template_args=None,
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        examples = []
        device = self._get_model_device(model)
        pad_token_id = self._get_pad_token_id(tokenizer)
        max_new_tokens = int(self.eval_cfg.get("args", {}).get("max_new_tokens", 100))

        for item in eval_data:
            prompt = self._render_prompt(item, tokenizer, template_args)
            expected = self._normalize_text(
                item.get("expected", item.get("output", item.get("answer", "")))
            )
            if not prompt or not expected:
                continue

            inputs = self._tokenize_prompt(
                tokenizer,
                prompt,
                self._render_training_text(
                    item, prompt, expected, tokenizer, template_args
                ),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_token_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self._generate(
                    model,
                    inputs,
                    prompt_token_len,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

            generated = tokenizer.decode(
                output_ids[prompt_token_len:], skip_special_tokens=True
            ).strip()
            is_correct = self._score_prediction(expected, generated, item)
            if is_correct:
                correct += 1
            examples.append(
                {
                    "index": total,
                    "source_dataset": item.get("source_dataset"),
                    "source_row_index": item.get(
                        "source_row_index", item.get("row_index")
                    ),
                    "prompt": prompt,
                    "expected": expected,
                    "references": item.get("references", [expected]),
                    "generated": generated,
                    "correct": is_correct,
                    **generation_token_details(
                        output_ids[prompt_token_len:], model, tokenizer, max_new_tokens
                    ),
                }
            )
            # Keep cohort identity beside the prediction so paired analyses do
            # not depend on row order or silently lose data provenance.
            for key in (
                "id",
                "case_id",
                "fact_id",
                "question_variant_id",
                "split",
                "group",
            ):
                if key in item:
                    examples[-1][key] = item[key]
            scoring = self._scoring_profile(item)
            examples[-1]["scoring"] = scoring or "dataset-aware-substring-v1"
            if scoring in {"short_answer", "fact_answer"}:
                examples[-1].update(
                    short_answer_scores(generated, item.get("references", [expected]))
                )
                if scoring == "fact_answer":
                    examples[-1].update(
                        fact_answer_scores(
                            generated, item.get("references", [expected])
                        )
                    )
            elif scoring in {"gsm8k", "gsm8k_numeric"}:
                strict_answer = gsm8k_final_answer(generated)
                flexible = gsm8k_final_answer(generated, require_marker=False)
                extracted = flexible if scoring == "gsm8k_numeric" else strict_answer
                examples[-1]["extracted_final_answer"] = (
                    str(extracted) if extracted is not None else None
                )
                examples[-1]["strict_marker_final_answer"] = (
                    str(strict_answer) if strict_answer is not None else None
                )
                reference = gsm8k_final_answer(expected, require_marker=False)
                examples[-1]["flexible_final_answer"] = (
                    str(flexible) if flexible is not None else None
                )
                examples[-1]["strict_marker_correct"] = (
                    strict_answer is not None
                    and reference is not None
                    and strict_answer == reference
                )
                examples[-1]["numeric_flexible_correct"] = (
                    flexible is not None
                    and reference is not None
                    and flexible == reference
                )
            total += 1

        accuracy = correct / total if total > 0 else None
        if accuracy is None and self._is_strict():
            raise ValueError(
                "InjectAccuracyEvaluator found no valid evaluation samples."
            )
        logger.info("Inject Task Accuracy: %s (%d/%d)", accuracy, correct, total)
        result = {
            "task_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "eval_method": "generation",
            "examples": examples,
            "skipped": len(eval_data) - total,
        }
        profiles = {row["scoring"] for row in examples}
        if profiles:
            result["scoring"] = profiles.pop() if len(profiles) == 1 else "mixed"
        if total:
            result.update(
                length_limit_rate=sum(
                    row["finish_reason"] == "length" for row in examples
                )
                / total,
                eos_termination_rate=sum(row["eos_terminated"] for row in examples)
                / total,
                mean_output_tokens=sum(row["output_token_count"] for row in examples)
                / total,
            )
            for key in (
                "answer_exact_match",
                "answer_token_f1",
                "first_line_match",
                "first_line_exact_match",
                "first_line_token_match",
                "first_answer_exact_match",
                "first_answer_token_match",
            ):
                available = [row[key] for row in examples if key in row]
                if available:
                    result[key] = sum(available) / len(available)
            fact_rows = [row for row in examples if row["scoring"] == "fact_answer"]
            if fact_rows:
                result["fact_answer_scoring_version"] = FACT_ANSWER_SCORING_VERSION
                result["fact_answer_count"] = len(fact_rows)
                result["review_flag_rate"] = sum(
                    row["needs_review"] for row in fact_rows
                ) / len(fact_rows)
                if len(fact_rows) == total:
                    result["task_accuracy_metric"] = "first_answer_token_match"
            groups = sorted({row["group"] for row in examples if row.get("group")})
            if groups:
                result["group_metrics"] = {}
                for group in groups:
                    rows = [row for row in examples if row.get("group") == group]
                    group_result = {
                        "total": len(rows),
                        "correct": sum(row["correct"] for row in rows),
                        "task_accuracy": sum(row["correct"] for row in rows)
                        / len(rows),
                    }
                    for key in (
                        "answer_exact_match",
                        "first_answer_token_match",
                        "needs_review",
                    ):
                        observed = [row[key] for row in rows if key in row]
                        if observed:
                            group_result[key] = sum(observed) / len(observed)
                    result["group_metrics"][group] = group_result
            extracted_rows = [
                row for row in examples if "extracted_final_answer" in row
            ]
            if extracted_rows:
                result["gsm8k_scoring_version"] = GSM8K_SCORING_VERSION
                result["numeric_extraction_rate"] = sum(
                    row["flexible_final_answer"] is not None for row in extracted_rows
                ) / len(extracted_rows)
                result["strict_marker_extraction_rate"] = sum(
                    row["strict_marker_final_answer"] is not None
                    for row in extracted_rows
                ) / len(extracted_rows)
                result["strict_marker_accuracy"] = sum(
                    row["strict_marker_correct"] for row in extracted_rows
                ) / len(extracted_rows)
                result["answer_extraction_rate"] = sum(
                    row["extracted_final_answer"] is not None for row in extracted_rows
                ) / len(extracted_rows)
                result["numeric_flexible_accuracy"] = sum(
                    row["numeric_flexible_correct"] for row in extracted_rows
                ) / len(extracted_rows)
        if accuracy is None:
            result["unavailable_reason"] = "no_valid_eval_samples"
        return result

    def _eval_perplexity(
        self,
        model,
        tokenizer,
        eval_data: List[Dict[str, Any]],
        template_args=None,
    ) -> Dict[str, Any]:
        total_loss = 0.0
        total_tokens = 0
        device = self._get_model_device(model)
        max_length = int(self.eval_cfg.get("args", {}).get("max_length", 2048))

        for item in eval_data:
            prompt = self._render_prompt(item, tokenizer, template_args)
            expected = self._normalize_text(
                item.get("expected", item.get("output", item.get("answer", "")))
            )
            text = self._normalize_text(item.get("text", ""))

            prompt_token_len = 0
            text_only = bool(text and not prompt and not expected)
            if text_only:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = inputs["input_ids"].clone()
            else:
                if not prompt or not expected:
                    continue
                full_text = self._render_training_text(
                    item, prompt, expected, tokenizer, template_args
                )
                features = tokenize_inject_pair(
                    prompt, full_text, tokenizer=tokenizer, max_length=max_length
                )
                prompt_token_len = features["prompt_length"]
                inputs = {
                    key: features[key].unsqueeze(0).to(device)
                    for key in ("input_ids", "attention_mask")
                }
                labels = features["labels"].unsqueeze(0).to(device)
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

            # Causal LM loss predicts token t from t-1; the first label is
            # never scored, including for text-only evaluation records.
            num_tokens = int((labels[:, 1:] != -100).sum().item())
            if num_tokens <= 0:
                continue

            with torch.no_grad():
                if model.__class__.__name__ == "ReftModel":
                    unit_locations = build_loreft_unit_locations(
                        inputs["attention_mask"],
                        [1 if text_only else prompt_token_len],
                        len(model.sorted_keys),
                        # The first text-only target has no causal LM loss;
                        # only explicit legacy mode may intervene there.
                        labels=None if text_only else labels,
                        loss_group=[0] if text_only else None,
                        **get_loreft_position_config(model),
                    )
                    _, outputs = model(
                        base=inputs,
                        unit_locations=unit_locations,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = outputs.loss
                else:
                    loss = model(**inputs, labels=labels).loss

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        if total_tokens == 0 and self._is_strict():
            raise ValueError(
                "InjectAccuracyEvaluator found no evaluable response tokens."
            )
        avg_loss = total_loss / total_tokens if total_tokens > 0 else None
        perplexity = (
            torch.exp(torch.tensor(avg_loss)).item() if total_tokens > 0 else None
        )
        logger.info("Inject Perplexity: %s", perplexity)
        result = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "eval_method": "perplexity",
        }
        if perplexity is None:
            result["unavailable_reason"] = "no_evaluable_response_tokens"
        return result


class InjectRetentionEvaluator(InjectEvaluator):
    """知识保持评估器"""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "inject_retention"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        benchmark_data = kwargs.get("benchmark_data", [])
        original_model = kwargs.get("original_model")
        template_args = kwargs.get("template_args")

        cached = self._load_cached_result(output_dir=output_dir)
        if cached is not None:
            return cached

        if tokenizer is None or not benchmark_data:
            if self._is_strict():
                missing = []
                if tokenizer is None:
                    missing.append("tokenizer")
                if not benchmark_data:
                    missing.append("benchmark_data / benchmark_data_path")
                raise ValueError(
                    "InjectRetentionEvaluator requires "
                    + ", ".join(missing)
                    + ". Refusing to write a fake 0.0 summary (strict=true)."
                )
            result = {
                "knowledge_retention": None,
                "current_score": None,
                "current_correct": 0,
                "current_total": 0,
                "retention_available": False,
                "unavailable_reason": (
                    "missing_tokenizer"
                    if tokenizer is None
                    else "missing_benchmark_data"
                ),
            }
            self._persist_result(result, output_dir=output_dir)
            return result

        model = self.prepare_model(model)
        current_score, current_correct, current_total, current_examples = (
            self._compute_scores(
                model, tokenizer, benchmark_data, template_args=template_args
            )
        )

        result = {
            "current_score": current_score,
            "current_correct": current_correct,
            "current_total": current_total,
            "current_examples": current_examples,
            "skipped": len(benchmark_data) - current_total,
        }

        if current_total == 0:
            if self._is_strict():
                raise ValueError(
                    "InjectRetentionEvaluator found no valid benchmark samples."
                )
            result.update(
                knowledge_retention=None,
                retention_available=False,
                unavailable_reason="no_valid_benchmark_samples",
            )
            self._persist_result(result, output_dir=output_dir)
            return result

        if original_model is not None:
            original_model = self.prepare_model(original_model)
            (
                original_score,
                original_correct,
                original_total,
                original_examples,
            ) = self._compute_scores(
                original_model,
                tokenizer,
                benchmark_data,
                template_args=template_args,
            )
            if original_score is not None and original_score > 0:
                retention = current_score / original_score
                retention_available = True
                unavailable_reason = None
            else:
                retention = None
                retention_available = False
                unavailable_reason = (
                    "no_valid_reference_samples"
                    if original_total == 0
                    else "original_score_non_positive"
                )
            result.update(
                {
                    "knowledge_retention": retention,
                    "retention_available": retention_available,
                    "unavailable_reason": unavailable_reason,
                    "original_score": original_score,
                    "original_correct": original_correct,
                    "original_total": original_total,
                    "original_examples": original_examples,
                }
            )
        else:
            result.update(
                knowledge_retention=None,
                retention_available=False,
                unavailable_reason="missing_original_model",
            )

        if result.get("retention_available", True):
            logger.info("Knowledge Retention: %.4f", result["knowledge_retention"])
        else:
            logger.info(
                "Knowledge Retention unavailable: %s",
                result["unavailable_reason"],
            )
        self._persist_result(result, output_dir=output_dir)
        return result

    def _compute_scores(
        self, model, tokenizer, data: List[Dict[str, Any]], template_args=None
    ) -> Tuple[Optional[float], int, int, List[Dict[str, Any]]]:
        correct = 0
        total = 0
        examples = []
        device = self._get_model_device(model)
        pad_token_id = self._get_pad_token_id(tokenizer)
        max_new_tokens = int(self.eval_cfg.get("args", {}).get("max_new_tokens", 50))

        for item in data:
            prompt = self._render_prompt(item, tokenizer, template_args)
            expected = self._normalize_text(
                item.get("expected", item.get("output", item.get("answer", "")))
            )
            if not prompt or not expected:
                continue

            inputs = self._tokenize_prompt(
                tokenizer,
                prompt,
                self._render_training_text(
                    item, prompt, expected, tokenizer, template_args
                ),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            prompt_token_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self._generate(
                    model,
                    inputs,
                    prompt_token_len,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

            generated = tokenizer.decode(
                output_ids[prompt_token_len:], skip_special_tokens=True
            ).strip()
            is_correct = self._score_prediction(expected, generated, item)
            if is_correct:
                correct += 1
            examples.append(
                {
                    "index": total,
                    "source_dataset": item.get("source_dataset"),
                    "source_row_index": item.get(
                        "source_row_index", item.get("row_index")
                    ),
                    "prompt": prompt,
                    "expected": expected,
                    "references": item.get("references", [expected]),
                    "generated": generated,
                    "correct": is_correct,
                }
            )
            if self._scoring_profile(item) == "fact_answer":
                examples[-1].update(
                    fact_answer_scores(generated, item.get("references", [expected]))
                )
            total += 1

        score = correct / total if total > 0 else None
        return score, correct, total, examples
