"""
Knowledge Editing evaluator:
- Reliability: rewrite accuracy
- Generalization: rephrase accuracy
- Locality: token-level invariance before/after edit
- Portability: grouped transfer accuracy
"""

import logging
import math
import re
from collections import Counter
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from evals.base import Evaluator

logger = logging.getLogger(__name__)


class EditEvaluator(Evaluator):
    """Base helper for edit evaluators."""

    def __init__(self, eval_cfg, **kwargs):
        self.name = "edit"
        super().__init__(self.name, eval_cfg, **kwargs)

    def _get_model_device(self, model) -> torch.device:
        get_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_embeddings):
            embeddings = get_embeddings()
            if embeddings is not None:
                return embeddings.weight.device
        if hasattr(model, "device"):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _apply_edit_runtime(self, model, prompt: str):
        """Resolve non-parametric editors (IKE/SERAC) for one prompt.

        Parametric editors simply have no ``_edit_runtime`` attribute and pass
        through unchanged.  This keeps all five text editors on the exact same
        EditSuite metric implementation.
        """
        runtime = getattr(model, "_edit_runtime", None)
        if runtime is None:
            return model, prompt
        selected_model = (
            runtime.select_model(prompt) if hasattr(runtime, "select_model") else model
        )
        transformed_prompt = (
            runtime.transform_prompt(prompt)
            if hasattr(runtime, "transform_prompt")
            else prompt
        )
        return selected_model, transformed_prompt

    def _resolve_edit_runtime(self, model, tokenizer, prompt: str):
        """Resolve hard generation routing together with its actual tokenizer."""
        runtime = getattr(model, "_edit_runtime", None)
        if runtime is not None and hasattr(runtime, "resolve_query"):
            return runtime.resolve_query(prompt)
        model, prompt = self._apply_edit_runtime(model, prompt)
        return model, tokenizer, prompt

    def _mean(self, values: List[float], default: float) -> float:
        return sum(values) / len(values) if values else default

    @staticmethod
    def _causal_model_inputs(batch):
        return {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }

    def _causal_prompt_target_inputs(self, tokenizer, prompt, target, device):
        """Tokenize a causal prompt/target with an explicit edit boundary."""

        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        separator = "" if prompt.endswith((" ", "\n", "\t")) else " "
        target_ids = tokenizer(
            separator + target,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        prompt_ids = prompt_inputs["input_ids"]
        full_inputs = {
            "input_ids": torch.cat((prompt_ids, target_ids), dim=1),
            "attention_mask": torch.cat(
                (
                    prompt_inputs.get("attention_mask", torch.ones_like(prompt_ids)),
                    torch.ones_like(target_ids),
                ),
                dim=1,
            ),
        }
        return full_inputs, prompt_ids.shape[1], target_ids[0]

    @contextmanager
    def _grace_prompt_boundary(self, model, prompt_length: int):
        """Make full-sequence teacher forcing use GRACE's edit key position.

        GRACE normally sees the last prompt token during autoregressive
        generation. A full prompt+target forward would otherwise route on the
        final target token and fail to evaluate the learned codebook. This
        temporary boundary preserves the same query used by editing and
        generation without changing non-GRACE models.
        """

        adapters = []
        for module in model.modules():
            if module.__class__.__name__ == "GRACEAdapter" and hasattr(
                module, "key_id"
            ):
                adapters.append((module, module.key_id))
                module.key_id = max(prompt_length - 1, 0)
        try:
            yield
        finally:
            for module, key_id in adapters:
                module.key_id = key_id

    def _flatten_target(self, target: Any) -> str:
        while isinstance(target, list):
            if not target:
                return ""
            target = target[0]
        return "" if target is None else str(target)

    def _normalize_prompt_target(self, prompt: Any, target: Any) -> Tuple[str, str]:
        return (
            "" if prompt is None else str(prompt).strip(),
            self._flatten_target(target).strip(),
        )

    def _normalize_rephrase_prompts(self, item: Dict[str, Any]) -> List[str]:
        raw = item.get(
            "rephrase_prompts",
            item.get("rephrase_prompt", item.get("rephrase", [])),
        )
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [str(v).strip() for v in raw if v is not None and str(v).strip()]

    def _extract_expected(self, item: Dict[str, Any]) -> str:
        return self._flatten_target(
            item.get(
                "expected",
                item.get("ground_truth", item.get("answer", item.get("target", ""))),
            )
        )

    def _iter_eval_pairs(
        self, raw_inputs: Any, default_group: str
    ) -> List[Tuple[str, str, str]]:
        """Normalize locality/portability inputs into (group, prompt, expected)."""
        if raw_inputs is None:
            return []

        pairs: List[Tuple[str, str, str]] = []

        if isinstance(raw_inputs, dict):
            # single item dict: {"prompt": ..., "ground_truth": ...}
            if any(
                k in raw_inputs
                for k in ("prompt", "expected", "ground_truth", "answer", "target")
            ):
                raw_inputs = [raw_inputs]
            else:
                for group_key, payload in raw_inputs.items():
                    group = str(group_key)
                    if payload is None:
                        continue

                    # {group: {"prompt": [...], "ground_truth": [...]}}
                    if isinstance(payload, dict):
                        prompts = payload.get("prompt", [])
                        targets = payload.get(
                            "ground_truth",
                            payload.get(
                                "expected",
                                payload.get("answer", payload.get("target", [])),
                            ),
                        )

                        if isinstance(prompts, str):
                            prompts = [prompts]
                        elif not isinstance(prompts, list):
                            prompts = []

                        if isinstance(targets, str):
                            targets = [targets]
                        elif not isinstance(targets, list):
                            targets = [targets]

                        for prompt, target in zip(prompts, targets):
                            p, t = self._normalize_prompt_target(prompt, target)
                            if p and t:
                                pairs.append((group, p, t))
                        continue

                    # {group: [{"prompt": ..., "ground_truth": ...}, ...]}
                    if isinstance(payload, list):
                        for item in payload:
                            if not isinstance(item, dict):
                                continue
                            p, t = self._normalize_prompt_target(
                                item.get("prompt", ""), self._extract_expected(item)
                            )
                            if p and t:
                                pairs.append((group, p, t))
                        continue
                return pairs

        if isinstance(raw_inputs, list):
            for item in raw_inputs:
                if not isinstance(item, dict):
                    continue
                group = str(
                    item.get("type")
                    or item.get("category")
                    or item.get("key")
                    or default_group
                )
                p, t = self._normalize_prompt_target(
                    item.get("prompt", ""), self._extract_expected(item)
                )
                if p and t:
                    pairs.append((group, p, t))

        return pairs

    def _teacher_forcing_prediction_and_labels(
        self, model, tokenizer, prompt: str, target: str
    ) -> Tuple[List[int], List[int]]:
        runtime = getattr(model, "_edit_runtime", None)
        if runtime is not None and hasattr(runtime, "target_distribution"):
            with torch.no_grad():
                log_probs, gold = runtime.target_distribution(prompt, target)
            return log_probs.argmax(dim=-1).cpu().tolist(), gold.cpu().tolist()
        model, tokenizer, prompt = self._resolve_edit_runtime(model, tokenizer, prompt)
        device = self._get_model_device(model)

        if getattr(model.config, "is_encoder_decoder", False):
            src = tokenizer(prompt, return_tensors="pt").to(device)
            trg = tokenizer(target, return_tensors="pt").to(device)
            target_ids = trg["input_ids"][0]
            if target_ids.numel() <= 1:
                return [], []
            with torch.no_grad():
                outputs = model(
                    input_ids=src["input_ids"],
                    attention_mask=src.get("attention_mask", None),
                    decoder_input_ids=trg["input_ids"],
                    decoder_attention_mask=trg.get("attention_mask", None),
                )
                logits = outputs.logits
            pred = torch.argmax(logits[0, :-1, :], dim=-1)
            gold = target_ids[:-1]
            return pred.detach().cpu().tolist(), gold.detach().cpu().tolist()

        full_inputs, prompt_len, gold = self._causal_prompt_target_inputs(
            tokenizer, prompt, target, device
        )
        if gold.numel() == 0:
            return [], []

        with self._grace_prompt_boundary(model, prompt_len), torch.no_grad():
            logits = model(**self._causal_model_inputs(full_inputs)).logits

        pred = torch.argmax(logits[0, prompt_len - 1 : -1, :], dim=-1)
        return pred.detach().cpu().tolist(), gold.detach().cpu().tolist()

    def compute_token_accuracy(
        self, model, tokenizer, prompt: str, target: str, **kwargs
    ) -> float:
        return self._token_accuracy_detail(model, tokenizer, prompt, target)["score"]

    def _token_accuracy_detail(
        self, model, tokenizer, prompt: str, target: str
    ) -> Dict[str, Any]:
        """Return token accuracy with enough detail to recompute it."""

        pred_tokens, gold_tokens = self._teacher_forcing_prediction_and_labels(
            model, tokenizer, prompt, target
        )
        overlap = min(len(pred_tokens), len(gold_tokens))
        correct = sum(
            1 for index in range(overlap) if pred_tokens[index] == gold_tokens[index]
        )
        denominator = len(gold_tokens)
        return {
            "score": correct / denominator if denominator else 0.0,
            "correct_tokens": correct,
            "total_target_tokens": denominator,
            "prediction_token_ids": pred_tokens,
            "expected_token_ids": gold_tokens,
            "prediction_text": tokenizer.decode(pred_tokens, skip_special_tokens=False),
            "expected_text": tokenizer.decode(gold_tokens, skip_special_tokens=False),
        }

    def token_match_ratio(self, pre_tokens: List[int], post_tokens: List[int]) -> float:
        if not pre_tokens and not post_tokens:
            return 1.0
        if not pre_tokens or not post_tokens:
            return 0.0
        overlap = min(len(pre_tokens), len(post_tokens))
        same = sum(1 for i in range(overlap) if pre_tokens[i] == post_tokens[i])
        # Locality compares against pre-edit output length.
        return same / len(pre_tokens)

    def compute_target_probability(
        self, model, tokenizer, prompt: str, target: str, **kwargs
    ) -> float:
        runtime = getattr(model, "_edit_runtime", None)
        if runtime is not None and hasattr(runtime, "target_distribution"):
            with torch.no_grad():
                log_probs, gold = runtime.target_distribution(prompt, target)
                if not gold.numel():
                    return 0.0
                return float(
                    log_probs.gather(-1, gold.to(log_probs.device).unsqueeze(-1))
                    .exp()
                    .mean()
                    .item()
                )
        model, tokenizer, prompt = self._resolve_edit_runtime(model, tokenizer, prompt)
        device = self._get_model_device(model)

        if getattr(model.config, "is_encoder_decoder", False):
            src = tokenizer(prompt, return_tensors="pt").to(device)
            trg = tokenizer(target, return_tensors="pt").to(device)
            target_ids = trg["input_ids"][0]
            if target_ids.numel() <= 1:
                return 0.0
            with torch.no_grad():
                logits = model(
                    input_ids=src["input_ids"],
                    attention_mask=src.get("attention_mask", None),
                    decoder_input_ids=trg["input_ids"],
                    decoder_attention_mask=trg.get("attention_mask", None),
                ).logits
            probs = F.softmax(logits[0, :-1, :], dim=-1)
            gold = target_ids[:-1].to(probs.device)
            target_probs = probs.gather(1, gold.unsqueeze(1)).squeeze()
            return (
                float(target_probs.item())
                if target_probs.dim() == 0
                else float(target_probs.mean().item())
            )

        full_inputs, prompt_len, target_ids = self._causal_prompt_target_inputs(
            tokenizer, prompt, target, device
        )
        if target_ids.numel() == 0:
            return 0.0

        with self._grace_prompt_boundary(model, prompt_len), torch.no_grad():
            logits = model(**self._causal_model_inputs(full_inputs)).logits
        target_logits = logits[0, prompt_len - 1 : -1, :]
        probs = F.softmax(target_logits, dim=-1)
        target_probs = probs.gather(
            1, target_ids.to(probs.device).unsqueeze(1)
        ).squeeze()
        return (
            float(target_probs.item())
            if target_probs.dim() == 0
            else float(target_probs.mean().item())
        )

    def _score_prompt_target(
        self, model, tokenizer, prompt: str, target: str, use_probability: bool = False
    ) -> float:
        if use_probability:
            return self.compute_target_probability(model, tokenizer, prompt, target)
        return self.compute_token_accuracy(model, tokenizer, prompt, target)

    def _aggregate_case_group_scores(
        self,
        raw_inputs: Any,
        default_group: str,
        score_fn: Callable[[str, str], float],
    ) -> Tuple[Optional[float], int, Dict[str, float]]:
        grouped_scores = defaultdict(list)
        for group, prompt, expected in self._iter_eval_pairs(raw_inputs, default_group):
            grouped_scores[group].append(float(score_fn(prompt, expected)))

        case_group_means = {
            group: self._mean(scores, 0.0)
            for group, scores in grouped_scores.items()
            if scores
        }
        case_score = (
            self._mean(list(case_group_means.values()), 0.0)
            if case_group_means
            else None
        )
        sample_count = sum(len(scores) for scores in grouped_scores.values())
        return case_score, sample_count, case_group_means

    def _merge_group_case_means(
        self, dst: Dict[str, List[float]], src: Dict[str, float]
    ) -> None:
        for key, value in src.items():
            dst[key].append(value)

    def _finalize_group_acc(
        self, grouped_case_means: Dict[str, List[float]]
    ) -> Dict[str, float]:
        return {
            f"{key}_acc": self._mean(values, 0.0)
            for key, values in grouped_case_means.items()
        }


class EditSuiteEvaluator(Evaluator):
    """Suite-style edit evaluator (single handler + multiple metric evaluators).

    This mirrors unlearning eval flow (`tofu`/`muse`): one evaluator node orchestrates
    multiple metric entries configured under `eval.edit.metrics.*`.
    """

    PRIMARY_VALUE_KEYS = {
        "reliability": "reliability",
        "generalization": "generalization",
        "locality": "locality",
        "portability": "portability",
        "comprehensive": "overall_score",
        "long_form": "rouge_l",
    }

    def __init__(self, eval_cfg, **kwargs):
        self.name = "edit"
        self.eval_cfg = eval_cfg
        self.metrics_cfg = self.eval_cfg.get("metrics", {}) or {}
        self.metrics = {}
        output_dir = self.eval_cfg.get("output_dir", None)
        if output_dir is not None:
            logger.info(
                "Evaluations stored in the experiment directory: %s", output_dir
            )
        else:
            logger.info(
                "Evaluator `%s` initialized without output_dir "
                "(standalone evaluator mode).",
                self.name,
            )

    def _handler_registry(self):
        return {
            "EditReliabilityEvaluator": EditReliabilityEvaluator,
            "EditGeneralizationEvaluator": EditGeneralizationEvaluator,
            "EditLocalityEvaluator": EditLocalityEvaluator,
            "EditPortabilityEvaluator": EditPortabilityEvaluator,
            "EditComprehensiveEvaluator": EditComprehensiveEvaluator,
            "EditLongFormEvaluator": EditLongFormEvaluator,
        }

    def summarize(self, logs):
        summary = {}
        for metric_name, metric_result in logs.items():
            if not isinstance(metric_result, dict):
                summary[metric_name] = metric_result
                continue
            key = self.PRIMARY_VALUE_KEYS.get(metric_name, metric_name)
            if key in metric_result:
                summary[key] = metric_result[key]
            elif metric_name in metric_result:
                summary[metric_name] = metric_result[metric_name]
            if metric_name == "long_form":
                for long_form_key in (
                    "bleu4",
                    "rouge1",
                    "rouge2",
                    "rouge_l",
                    "token_f1",
                    "semantic_similarity",
                    "mmlu_pre_accuracy",
                    "mmlu_post_accuracy",
                    "mmlu_accuracy_delta",
                ):
                    if long_form_key in metric_result:
                        summary[long_form_key] = metric_result[long_form_key]
        return summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        overwrite = (
            self.eval_cfg.get("overwrite", False) if overwrite is None else overwrite
        )
        model = self.prepare_model(model)

        output_dir = output_dir if output_dir else self.eval_cfg.get("output_dir", None)
        logs_file_path = None
        summary_file_path = None
        logs = {}
        if output_dir:
            logs_file_path = self.get_logs_file_path(output_dir)
            summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")
            logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        if not self.metrics_cfg:
            logger.warning("No edit metrics configured under `eval.edit.metrics`.")
            return {}

        handler_registry = self._handler_registry()
        tokenizer = kwargs.get("tokenizer", None)
        edit_data = kwargs.get("edit_data", None)
        original_model = kwargs.get("original_model", None)
        if tokenizer is None:
            raise ValueError("EditSuiteEvaluator requires `tokenizer`.")
        if edit_data is None:
            raise ValueError(
                "EditSuiteEvaluator requires `edit_data` or `edit_data_path`."
            )
        if not edit_data:
            raise ValueError("EditSuiteEvaluator received empty `edit_data`.")

        for metric_name, metric_cfg in self.metrics_cfg.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info("Skipping %s, already evaluated.", metric_name)
                continue

            handler_name = metric_cfg.get("handler", None)
            evaluator_cls = handler_registry.get(handler_name)
            if evaluator_cls is None:
                raise NotImplementedError(
                    f"{handler_name} not implemented or not registered for edit suite"
                )

            metric_evaluator = evaluator_cls(metric_cfg)
            metric_result = metric_evaluator.evaluate(
                model=model,
                tokenizer=tokenizer,
                edit_data=edit_data,
                original_model=original_model,
                output_dir=output_dir,
            )
            logs[metric_name] = metric_result

            if output_dir:
                self.save_logs(logs, logs_file_path)
                self.save_logs(self.summarize(logs), summary_file_path)

        return self.summarize(logs)


class EditReliabilityEvaluator(EditEvaluator):
    """Reliability: mean rewrite accuracy."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_reliability"
        args = eval_cfg.get("args", {})
        self.success_threshold = float(args.get("success_threshold", 1.0))

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        if tokenizer is None or not edit_data:
            return {"reliability": 0.0}

        model = self.prepare_model(model)
        scores = []
        examples = []
        for case_index, item in enumerate(edit_data):
            prompt, target_new = self._normalize_prompt_target(
                item.get("prompt", ""), item.get("target_new", "")
            )
            if prompt and target_new:
                detail = self._token_accuracy_detail(
                    model, tokenizer, prompt, target_new
                )
                scores.append(detail["score"])
                examples.append(
                    {
                        "case_index": case_index,
                        "prompt": prompt,
                        "expected": target_new,
                        **detail,
                    }
                )

        reliability = self._mean(scores, 0.0)
        result = {
            "reliability": reliability,
            "total_samples": len(scores),
            "successful_edits": int(
                sum(1 for score in scores if score >= self.success_threshold)
            ),
            "per_example": examples,
        }
        logger.info("Edit Reliability: %.4f", reliability)
        return result


class EditGeneralizationEvaluator(EditEvaluator):
    """Generalization: mean rephrase accuracy (case-level macro)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_generalization"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        if tokenizer is None:
            return {"generalization": 0.0}

        model = self.prepare_model(model)
        case_scores = []
        total_rephrase_samples = 0
        examples = []
        for case_index, item in enumerate(edit_data):
            _, target_new = self._normalize_prompt_target(
                item.get("prompt", ""), item.get("target_new", "")
            )
            per_case = []
            case_examples = []
            for probe_index, rephrase in enumerate(
                self._normalize_rephrase_prompts(item)
            ):
                prompt, target = self._normalize_prompt_target(rephrase, target_new)
                if prompt and target:
                    detail = self._token_accuracy_detail(
                        model, tokenizer, prompt, target
                    )
                    per_case.append(detail["score"])
                    case_examples.append(
                        {
                            "case_index": case_index,
                            "probe_index": probe_index,
                            "prompt": prompt,
                            "expected": target,
                            **detail,
                        }
                    )
                    total_rephrase_samples += 1
            if per_case:
                case_score = self._mean(per_case, 0.0)
                case_scores.append(case_score)
                for example in case_examples:
                    example["case_score"] = case_score
                examples.extend(case_examples)

        generalization = self._mean(case_scores, 0.0) if case_scores else None
        result = {
            "generalization": generalization,
            "total_rephrase_samples": total_rephrase_samples,
            "evaluated_cases": len(case_scores),
            "per_example": examples,
        }
        if generalization is None:
            result["generalization_unavailable_reason"] = "missing_rephrase_prompts"
            logger.warning("Edit Generalization unavailable: no rephrase prompts")
        else:
            logger.info("Edit Generalization: %.4f", generalization)
        return result


class EditLocalityEvaluator(EditEvaluator):
    """Locality: pre/post token invariance (case-level macro + by_key)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_locality"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")
        if tokenizer is None:
            return {
                "locality": None,
                "total_locality_samples": 0,
                "locality_unavailable_reason": "missing_tokenizer",
            }
        if original_model is None:
            logger.warning(
                "EditLocalityEvaluator requires original_model; "
                "returning locality=null (not a fake 1.0)."
            )
            return {
                "locality": None,
                "total_locality_samples": 0,
                "locality_unavailable_reason": "missing_original_model",
            }

        model = self.prepare_model(model)
        original_model = self.prepare_model(original_model)

        def _loc_detail(prompt: str, expected: str) -> Dict[str, Any]:
            pre_tokens, _ = self._teacher_forcing_prediction_and_labels(
                original_model, tokenizer, prompt, expected
            )
            post_tokens, _ = self._teacher_forcing_prediction_and_labels(
                model, tokenizer, prompt, expected
            )
            overlap = min(len(pre_tokens), len(post_tokens))
            matching = sum(
                1 for index in range(overlap) if pre_tokens[index] == post_tokens[index]
            )
            return {
                "score": self.token_match_ratio(pre_tokens, post_tokens),
                "matching_tokens": matching,
                "total_pre_edit_tokens": len(pre_tokens),
                "pre_prediction_token_ids": pre_tokens,
                "post_prediction_token_ids": post_tokens,
                "pre_prediction_text": tokenizer.decode(
                    pre_tokens, skip_special_tokens=False
                ),
                "post_prediction_text": tokenizer.decode(
                    post_tokens, skip_special_tokens=False
                ),
            }

        case_scores = []
        total_locality_samples = 0
        grouped_case_means = defaultdict(list)
        examples = []
        for case_index, item in enumerate(edit_data):
            case_group_scores = defaultdict(list)
            case_examples = []
            for probe_index, (group, prompt, expected) in enumerate(
                self._iter_eval_pairs(
                    item.get("locality_inputs", item.get("locality", None)),
                    "locality",
                )
            ):
                detail = _loc_detail(prompt, expected)
                case_group_scores[group].append(detail["score"])
                case_examples.append(
                    {
                        "case_index": case_index,
                        "probe_index": probe_index,
                        "group": group,
                        "prompt": prompt,
                        "expected": expected,
                        **detail,
                    }
                )
            case_group_means = {
                group: self._mean(scores, 0.0)
                for group, scores in case_group_scores.items()
                if scores
            }
            case_score = (
                self._mean(list(case_group_means.values()), 0.0)
                if case_group_means
                else None
            )
            total_locality_samples += len(case_examples)
            self._merge_group_case_means(grouped_case_means, case_group_means)
            if case_score is not None:
                case_scores.append(case_score)
                for example in case_examples:
                    example["case_score"] = case_score
                examples.extend(case_examples)

        locality = self._mean(case_scores, 1.0) if case_scores else None
        result = {
            "locality": locality,
            "total_locality_samples": total_locality_samples,
            "evaluated_cases": len(case_scores),
            "locality_by_key": self._finalize_group_acc(grouped_case_means),
            "per_example": examples,
        }
        if locality is None:
            result["locality_unavailable_reason"] = "missing_locality_inputs"
            logger.warning("Edit Locality unavailable: no locality inputs")
        else:
            logger.info("Edit Locality: %.4f", locality)
        return result


class EditPortabilityEvaluator(EditEvaluator):
    """Portability: grouped transfer accuracy (case-level macro + by_key)."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_portability"

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        if tokenizer is None:
            return {"portability": 0.0}

        model = self.prepare_model(model)

        case_scores = []
        total_portability_samples = 0
        grouped_case_means = defaultdict(list)
        examples = []
        for case_index, item in enumerate(edit_data):
            case_group_scores = defaultdict(list)
            case_examples = []
            for probe_index, (group, prompt, expected) in enumerate(
                self._iter_eval_pairs(
                    item.get("portability_inputs", item.get("portability", None)),
                    "portability",
                )
            ):
                detail = self._token_accuracy_detail(model, tokenizer, prompt, expected)
                case_group_scores[group].append(detail["score"])
                case_examples.append(
                    {
                        "case_index": case_index,
                        "probe_index": probe_index,
                        "group": group,
                        "prompt": prompt,
                        "expected": expected,
                        **detail,
                    }
                )
            case_group_means = {
                group: self._mean(scores, 0.0)
                for group, scores in case_group_scores.items()
                if scores
            }
            case_score = (
                self._mean(list(case_group_means.values()), 0.0)
                if case_group_means
                else None
            )
            total_portability_samples += len(case_examples)
            self._merge_group_case_means(grouped_case_means, case_group_means)
            if case_score is not None:
                case_scores.append(case_score)
                for example in case_examples:
                    example["case_score"] = case_score
                examples.extend(case_examples)

        portability = self._mean(case_scores, 0.0) if case_scores else None
        result = {
            "portability": portability,
            "total_portability_samples": total_portability_samples,
            "evaluated_cases": len(case_scores),
            "portability_by_key": self._finalize_group_acc(grouped_case_means),
            "per_example": examples,
        }
        if portability is None:
            result["portability_unavailable_reason"] = "missing_portability_inputs"
            logger.warning("Edit Portability unavailable: no portability inputs")
        else:
            logger.info("Edit Portability: %.4f", portability)
        return result


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)


def _ngram_counts(tokens: List[str], order: int) -> Counter:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def _f1(overlap: int, predicted: int, reference: int) -> float:
    if predicted == 0 or reference == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / reference
    return 2 * precision * recall / (precision + recall)


def _rouge_n(prediction: List[str], reference: List[str], order: int) -> float:
    pred_counts = _ngram_counts(prediction, order)
    ref_counts = _ngram_counts(reference, order)
    overlap = sum((pred_counts & ref_counts).values())
    return _f1(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def _lcs_length(left: List[str], right: List[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _sentence_bleu(prediction: List[str], reference: List[str]) -> float:
    """Deterministic token-level BLEU-4 with add-one smoothing."""
    if not prediction or not reference:
        return 0.0
    log_precisions = []
    for order in range(1, 5):
        pred_counts = _ngram_counts(prediction, order)
        ref_counts = _ngram_counts(reference, order)
        overlap = sum((pred_counts & ref_counts).values())
        total = sum(pred_counts.values())
        log_precisions.append(math.log((overlap + 1) / (total + 1)))
    brevity_penalty = (
        1.0
        if len(prediction) > len(reference)
        else math.exp(1.0 - len(reference) / max(len(prediction), 1))
    )
    return brevity_penalty * math.exp(sum(log_precisions) / 4)


class EditLongFormEvaluator(EditEvaluator):
    """Generated-output metrics for AnyEdit/UnKE long-form benchmarks.

    Metrics are computed on word/punctuation tokens, not characters.  The
    result retains every generated prediction and reports missing groups as
    ``null`` rather than fabricating zero-valued evidence.
    """

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_long_form"
        args = eval_cfg.get("args", {})
        self.max_new_tokens = int(args.get("max_new_tokens", 512))
        self.generation_margin = int(args.get("generation_margin", 16))
        self.max_pairs_per_group = int(args.get("max_pairs_per_group", 5))
        self.semantic_model_name = args.get("semantic_model_name", None)

    def _generate(self, model, tokenizer, prompt: str, target: str) -> str:
        model, tokenizer, prompt = self._resolve_edit_runtime(model, tokenizer, prompt)
        device = self._get_model_device(model)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {
            key: value.to(device)
            for key, value in self._causal_model_inputs(encoded).items()
        }
        target_length = len(
            tokenizer(target, add_special_tokens=False).get("input_ids", [])
        )
        max_new_tokens = min(
            self.max_new_tokens, max(target_length + self.generation_margin, 1)
        )
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        prompt_length = encoded["input_ids"].shape[1]
        return tokenizer.decode(
            generated[0, prompt_length:], skip_special_tokens=True
        ).strip()

    def _score(self, prediction: str, reference: str) -> Dict[str, float]:
        pred_tokens = _word_tokens(prediction)
        ref_tokens = _word_tokens(reference)
        unigram_overlap = sum(
            (_ngram_counts(pred_tokens, 1) & _ngram_counts(ref_tokens, 1)).values()
        )
        return {
            "bleu4": _sentence_bleu(pred_tokens, ref_tokens),
            "rouge1": _rouge_n(pred_tokens, ref_tokens, 1),
            "rouge2": _rouge_n(pred_tokens, ref_tokens, 2),
            "rouge_l": _f1(
                _lcs_length(pred_tokens, ref_tokens),
                len(pred_tokens),
                len(ref_tokens),
            ),
            "token_f1": _f1(unigram_overlap, len(pred_tokens), len(ref_tokens)),
            "exact_match": float(prediction.strip() == reference.strip()),
        }

    def _semantic_scores(
        self, predictions: List[str], references: List[str]
    ) -> tuple[Optional[List[float]], Optional[str]]:
        if not self.semantic_model_name:
            return None, "semantic_model_not_configured"
        try:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(self.semantic_model_name)
            pred_embeddings = encoder.encode(
                predictions, convert_to_tensor=True, normalize_embeddings=True
            )
            ref_embeddings = encoder.encode(
                references, convert_to_tensor=True, normalize_embeddings=True
            )
            values = (pred_embeddings * ref_embeddings).sum(dim=-1)
            return values.detach().cpu().tolist(), None
        except Exception as exc:
            return None, f"semantic_model_unavailable: {exc}"

    @staticmethod
    def _choice_correct(prediction: str, expected: str) -> float:
        expected = str(expected).strip().upper()
        match = re.search(
            r"(?:^|\s|\()([A-F])(?:\)|\.|:|\s|$)",
            str(prediction).strip().upper(),
        )
        return float(bool(match and match.group(1) == expected))

    def evaluate(self, model, output_dir=None, **kwargs):
        del output_dir
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")
        if tokenizer is None or not edit_data:
            return {
                "bleu4": None,
                "rouge1": None,
                "rouge2": None,
                "rouge_l": None,
                "token_f1": None,
                "exact_match": None,
                "total_samples": 0,
                "unavailable_reason": "missing_tokenizer_or_edit_data",
            }

        model = self.prepare_model(model)
        pairs: List[tuple[str, str, str]] = []
        mmlu_pairs: List[tuple[str, str]] = []
        for item in edit_data:
            prompt, target = self._normalize_prompt_target(
                item.get("prompt", ""), item.get("target_new", "")
            )
            if prompt and target:
                pairs.append(("rewrite", prompt, target))
            for rephrase in self._normalize_rephrase_prompts(item):
                rp, rt = self._normalize_prompt_target(rephrase, target)
                if rp and rt:
                    pairs.append(("rephrase", rp, rt))
            pairs.extend(
                self._iter_eval_pairs(
                    item.get("portability_inputs", item.get("portability")),
                    "portability",
                )
            )
            for group, prompt, expected in self._iter_eval_pairs(
                item.get("locality_inputs", item.get("locality")), "locality"
            ):
                if group == "GeneralAbility_MMLU":
                    mmlu_pairs.append((prompt, expected))

        group_counts: Dict[str, int] = defaultdict(int)
        bounded_pairs = []
        for group, prompt, target in pairs:
            if group_counts[group] >= self.max_pairs_per_group:
                continue
            bounded_pairs.append((group, prompt, target))
            group_counts[group] += 1

        examples = []
        for group, prompt, reference in bounded_pairs:
            prediction = self._generate(model, tokenizer, prompt, reference)
            examples.append(
                {
                    "group": group,
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    **self._score(prediction, reference),
                }
            )

        metric_names = (
            "bleu4",
            "rouge1",
            "rouge2",
            "rouge_l",
            "token_f1",
            "exact_match",
        )
        result: Dict[str, Any] = {
            name: self._mean([example[name] for example in examples], 0.0)
            if examples
            else None
            for name in metric_names
        }
        result.update(
            {
                "total_samples": len(examples),
                "group_counts": dict(group_counts),
                "per_example": examples,
            }
        )

        semantic, reason = self._semantic_scores(
            [example["prediction"] for example in examples],
            [example["reference"] for example in examples],
        )
        result["semantic_similarity"] = (
            self._mean([float(value) for value in semantic], 0.0) if semantic else None
        )
        if reason:
            result["semantic_similarity_unavailable_reason"] = reason
        if not examples:
            result["unavailable_reason"] = "missing_long_form_pairs"

        mmlu_examples = []
        for prompt, expected in mmlu_pairs[: self.max_pairs_per_group]:
            post_prediction = self._generate(model, tokenizer, prompt, expected)
            pre_prediction = None
            if original_model is not None:
                pre_prediction = self._generate(
                    original_model, tokenizer, prompt, expected
                )
            mmlu_examples.append(
                {
                    "prompt": prompt,
                    "expected_label": expected,
                    "pre_prediction": pre_prediction,
                    "post_prediction": post_prediction,
                    "pre_correct": (
                        self._choice_correct(pre_prediction, expected)
                        if pre_prediction is not None
                        else None
                    ),
                    "post_correct": self._choice_correct(post_prediction, expected),
                }
            )
        post_values = [example["post_correct"] for example in mmlu_examples]
        pre_values = [
            example["pre_correct"]
            for example in mmlu_examples
            if example["pre_correct"] is not None
        ]
        result["mmlu_post_accuracy"] = (
            self._mean(post_values, 0.0) if post_values else None
        )
        result["mmlu_pre_accuracy"] = (
            self._mean(pre_values, 0.0) if pre_values else None
        )
        result["mmlu_accuracy_delta"] = (
            result["mmlu_post_accuracy"] - result["mmlu_pre_accuracy"]
            if result["mmlu_post_accuracy"] is not None
            and result["mmlu_pre_accuracy"] is not None
            else None
        )
        result["mmlu_total_samples"] = len(mmlu_examples)
        result["mmlu_per_example"] = mmlu_examples
        if not mmlu_examples:
            result["mmlu_unavailable_reason"] = "missing_mmlu_probes"
        elif original_model is None:
            result["mmlu_pre_accuracy_unavailable_reason"] = "missing_original_model"
        return result


class EditComprehensiveEvaluator(EditEvaluator):
    """Comprehensive scorer for all four metrics."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "edit_comprehensive"
        args = eval_cfg.get("args", {})
        self.reliability_weight = float(args.get("reliability_weight", 0.25))
        self.generalization_weight = float(args.get("generalization_weight", 0.25))
        self.locality_weight = float(args.get("locality_weight", 0.25))
        self.portability_weight = float(args.get("portability_weight", 0.25))
        self.use_probability = bool(args.get("use_probability", False))
        self.require_all_metrics = bool(args.get("require_all_metrics", True))

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        edit_data = kwargs.get("edit_data", [])
        original_model = kwargs.get("original_model")
        if tokenizer is None or not edit_data:
            return self._empty_result()

        model = self.prepare_model(model)
        if original_model is not None:
            original_model = self.prepare_model(original_model)

        reliability_case_scores = []
        generalization_case_scores = []
        locality_case_scores = []
        portability_case_scores = []

        reliability_samples = 0
        generalization_samples = 0
        locality_samples = 0
        portability_samples = 0

        grouped_locality_case_means = defaultdict(list)
        grouped_portability_case_means = defaultdict(list)

        def _loc_score(prompt: str, expected: str) -> float:
            pre_tokens, _ = self._teacher_forcing_prediction_and_labels(
                original_model, tokenizer, prompt, expected
            )
            post_tokens, _ = self._teacher_forcing_prediction_and_labels(
                model, tokenizer, prompt, expected
            )
            return self.token_match_ratio(pre_tokens, post_tokens)

        def _port_score(prompt: str, expected: str) -> float:
            return self.compute_token_accuracy(model, tokenizer, prompt, expected)

        for item in edit_data:
            prompt, target_new = self._normalize_prompt_target(
                item.get("prompt", ""), item.get("target_new", "")
            )

            if prompt and target_new:
                reliability_case_scores.append(
                    self._score_prompt_target(
                        model, tokenizer, prompt, target_new, self.use_probability
                    )
                )
                reliability_samples += 1

            rephrase_scores = []
            for rephrase in self._normalize_rephrase_prompts(item):
                rp, target = self._normalize_prompt_target(rephrase, target_new)
                if rp and target:
                    rephrase_scores.append(
                        self._score_prompt_target(
                            model, tokenizer, rp, target, self.use_probability
                        )
                    )
                    generalization_samples += 1
            if rephrase_scores:
                generalization_case_scores.append(self._mean(rephrase_scores, 0.0))

            if original_model is not None:
                loc_case_score, loc_samples, loc_case_group_means = (
                    self._aggregate_case_group_scores(
                        item.get("locality_inputs", item.get("locality", None)),
                        "locality",
                        _loc_score,
                    )
                )
                locality_samples += loc_samples
                self._merge_group_case_means(
                    grouped_locality_case_means, loc_case_group_means
                )
                if loc_case_score is not None:
                    locality_case_scores.append(loc_case_score)

            port_case_score, port_samples, port_case_group_means = (
                self._aggregate_case_group_scores(
                    item.get("portability_inputs", item.get("portability", None)),
                    "portability",
                    _port_score,
                )
            )
            portability_samples += port_samples
            self._merge_group_case_means(
                grouped_portability_case_means, port_case_group_means
            )
            if port_case_score is not None:
                portability_case_scores.append(port_case_score)

        reliability = (
            self._mean(reliability_case_scores, 0.0)
            if reliability_case_scores
            else None
        )
        generalization = (
            self._mean(generalization_case_scores, 0.0)
            if generalization_case_scores
            else None
        )
        locality = (
            self._mean(locality_case_scores, 1.0) if locality_case_scores else None
        )
        portability = (
            self._mean(portability_case_scores, 0.0)
            if portability_case_scores
            else None
        )

        weighted = [
            (reliability, self.reliability_weight),
            (generalization, self.generalization_weight),
            (locality, self.locality_weight),
            (portability, self.portability_weight),
        ]
        available = [(value, weight) for value, weight in weighted if value is not None]
        weight_sum = sum(weight for _, weight in available)
        if self.require_all_metrics and len(available) != len(weighted):
            overall_score = None
        else:
            overall_score = (
                sum(value * weight for value, weight in available) / weight_sum
                if weight_sum > 0
                else None
            )

        result = {
            "reliability": reliability,
            "generalization": generalization,
            "locality": locality,
            "portability": portability,
            "overall_score": overall_score,
            "total_edits": len(edit_data),
            "reliability_samples": reliability_samples,
            "generalization_samples": generalization_samples,
            "locality_samples": locality_samples,
            "portability_samples": portability_samples,
            "reliability_cases": len(reliability_case_scores),
            "generalization_cases": len(generalization_case_scores),
            "locality_cases": len(locality_case_scores),
            "portability_cases": len(portability_case_scores),
            "locality_by_key": self._finalize_group_acc(grouped_locality_case_means),
            "portability_by_key": self._finalize_group_acc(
                grouped_portability_case_means
            ),
            "unavailable_metrics": [
                name
                for name, value in (
                    ("reliability", reliability),
                    ("generalization", generalization),
                    ("locality", locality),
                    ("portability", portability),
                )
                if value is None
            ],
        }
        if overall_score is None and available:
            result["overall_unavailable_reason"] = (
                "missing_component_metrics"
                if self.require_all_metrics
                else "no_available_metrics"
            )

        logger.info(
            "Edit Comprehensive: reliability=%s generalization=%s locality=%s "
            "portability=%s overall=%s",
            reliability,
            generalization,
            locality,
            portability,
            overall_score,
        )
        return result

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "reliability": None,
            "generalization": None,
            "locality": None,
            "portability": None,
            "overall_score": None,
            "total_edits": 0,
            "reliability_samples": 0,
            "generalization_samples": 0,
            "locality_samples": 0,
            "portability_samples": 0,
            "reliability_cases": 0,
            "generalization_cases": 0,
            "locality_cases": 0,
            "portability_cases": 0,
            "locality_by_key": {},
            "portability_by_key": {},
            "unavailable_metrics": [
                "reliability",
                "generalization",
                "locality",
                "portability",
            ],
        }
