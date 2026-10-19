"""
Knowledge Injection 评估器
=========================

实现知识注入（微调）的评估指标：
- Task Accuracy（任务准确率）：微调任务的性能
- Knowledge Retention（知识保持）：原有知识的保持率
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from data.inject import render_alpaca_texts, tokenize_inject_pair
from evals.base import Evaluator

logger = logging.getLogger(__name__)

_COST_RE = re.compile(
    r"(?:total\s+cost\s+is\s*\$?|\$)\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


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

        unit_locations = {"sources->base": (None, [[[max(prompt_token_len - 1, 0)]]])}
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

    def _score_prediction(
        self, expected: str, generated: str, item: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Score a prediction; RuleArena-Airline uses numeric cost matching."""
        item = item or {}
        source = str(item.get("source_dataset", "")).lower()
        domain = str(item.get("domain", "")).lower()
        scoring = str(
            item.get("scoring") or self.eval_cfg.get("args", {}).get("scoring", "")
        ).lower()

        if scoring == "exact_match":
            return expected.strip().casefold() == generated.strip().casefold()

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
        if self.eval_cfg.get("overwrite", False):
            return None
        target_output_dir = (
            output_dir if output_dir else self.eval_cfg.get("output_dir")
        )
        if target_output_dir is None:
            return None
        logs_file = self.get_logs_file_path(target_output_dir)
        cached = self.load_logs_from_file(logs_file)
        if cached and cached.get("protocol_version") == "inject-v2":
            logger.info(
                "Skipping `%s`, loaded cached result from %s", self.name, logs_file
            )
            return cached
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
                ):
                    if key in result:
                        summary[key] = result[key]
                return summary
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}

    def _persist_result(self, result: Dict[str, Any], output_dir=None):
        result["protocol_version"] = "inject-v2"
        if "perplexity" in result:
            result["scoring"] = "causal-token-nll-v1"
        else:
            result["scoring"] = str(
                self.eval_cfg.get("args", {}).get("scoring")
                or "dataset-aware-substring-v1"
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
                    "source_row_index": item.get("row_index"),
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated,
                    "correct": is_correct,
                }
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
            if text and not prompt and not expected:
                full_text = text
            else:
                if not prompt or not expected:
                    continue
                target = self._render_target(expected, tokenizer, template_args)
                full_text = self._join_prompt_target(prompt, target)
                prompt_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=max_length
                )["input_ids"]
                prompt_token_len = prompt_ids.shape[1]

            inputs = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            if prompt_token_len >= input_ids.shape[1]:
                continue

            labels = input_ids.clone()
            if prompt_token_len > 0:
                labels[:, :prompt_token_len] = -100
            if "attention_mask" in inputs:
                labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

            # Causal LM loss predicts token t from t-1; the first label is
            # never scored, including for text-only evaluation records.
            num_tokens = int((labels[:, 1:] != -100).sum().item())
            if num_tokens <= 0:
                continue

            with torch.no_grad():
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
                    "source_row_index": item.get("row_index"),
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated,
                    "correct": is_correct,
                }
            )
            total += 1

        score = correct / total if total > 0 else None
        return score, correct, total, examples
