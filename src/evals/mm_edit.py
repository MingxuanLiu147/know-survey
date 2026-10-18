"""Image-aware reliability, generalization, locality and portability metrics."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from evals.edit import EditEvaluator
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class _MMEditEvalHelper(EditEvaluator):
    """Shared helpers for multimodal edit evaluators."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self._mm = MMEditMixin()

    def _init_mm(self, processor):
        self._mm.init_mm(processor)

    @staticmethod
    def _apply_mm_runtime(model, prompt: str, image=None):
        """Apply an IKE/SERAC-style runtime without changing image objects."""
        runtime = getattr(model, "_edit_runtime", None)
        if runtime is None:
            return model, prompt

        selected_model = (
            runtime.select_model(prompt) if hasattr(runtime, "select_model") else model
        )
        if hasattr(runtime, "transform_mm_prompt"):
            prompt = runtime.transform_mm_prompt(prompt, image=image)
        elif hasattr(runtime, "transform_prompt"):
            prompt = runtime.transform_prompt(prompt)
        return selected_model, prompt

    def _mm_token_accuracy(self, model, processor, prompt, target, image=None):
        model, prompt = self._apply_mm_runtime(model, prompt, image)
        self._mm.model = model
        if image is not None:
            inputs = self._mm.mm_tokenize(prompt, image, target)
        else:
            inputs = self._mm.mm_tokenize_text_only(prompt, target)

        with torch.no_grad():
            outputs = self._mm.mm_forward(model, inputs)

        logits = outputs.logits
        labels = inputs["labels"]
        preds = logits.argmax(dim=-1)
        shift_preds = preds[:, :-1]
        shift_labels = labels[:, 1:]
        shift_mask = shift_labels != -100
        if not shift_mask.any():
            return 0.0

        correct = ((shift_preds == shift_labels) & shift_mask).sum().item()
        total = shift_mask.sum().item()
        return correct / total if total > 0 else 0.0

    def _mm_pred_tokens(self, model, processor, prompt, target, image=None):
        ranked_tokens = self._mm_pred_topk_tokens(
            model,
            processor,
            prompt,
            target,
            image,
            top_k=1,
        )
        return [row[0] for row in ranked_tokens if row]

    def _mm_pred_topk_tokens(
        self,
        model,
        processor,
        prompt,
        target,
        image=None,
        *,
        top_k: int,
    ) -> list[list[int]]:
        """Return ordered top-k token ids at teacher-forced target positions."""
        model, prompt = self._apply_mm_runtime(model, prompt, image)
        self._mm.model = model
        if image is not None:
            inputs = self._mm.mm_tokenize(prompt, image, target)
        else:
            inputs = self._mm.mm_tokenize_text_only(prompt, target)

        with torch.no_grad():
            outputs = self._mm.mm_forward(model, inputs)

        logits = outputs.logits
        labels = inputs["labels"]
        effective_top_k = min(max(int(top_k), 1), int(logits.shape[-1]))
        # Softmax is monotonic, so ranking logits matches the official VLKEB
        # IKE implementation's ranking over softmax probabilities.
        shift_ranked = logits[:, :-1].topk(effective_top_k, dim=-1).indices
        shift_labels = labels[:, 1:]
        shift_mask = shift_labels != -100
        if not shift_mask.any():
            return []
        return shift_ranked[0][shift_mask[0]].detach().cpu().tolist()

    @staticmethod
    def _flatten_ranked_tokens(ranked_tokens: list[list[int]]) -> list[int]:
        return [token for row in ranked_tokens for token in row]

    @staticmethod
    def _iter_multimodal_locality(item: dict[str, Any]) -> list[dict[str, Any]]:
        raw = item.get("multimodal_locality_inputs")
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, list):
            return [probe for probe in raw if isinstance(probe, dict)]
        return []


class MMEditReliabilityEvaluator(_MMEditEvalHelper):
    """Multimodal reliability: rewrite accuracy with image input."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_reliability"

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        if processor is None or not edit_data:
            return {
                "mm_reliability": None,
                "total_samples": 0,
                "per_example": [],
                "reliability_unavailable_reason": (
                    "missing_processor" if processor is None else "missing_edit_data"
                ),
            }

        self._init_mm(processor)
        model = self.prepare_model(model)
        scores = []
        per_example = []
        for index, item in enumerate(edit_data):
            prompt, target_new = self._normalize_prompt_target(
                item.get("prompt", ""), item.get("target_new", "")
            )
            image = item.get("image")
            if prompt and target_new:
                score = self._mm_token_accuracy(
                    model, processor, prompt, target_new, image
                )
                scores.append(score)
                per_example.append({"index": index, "score": score})

        mm_reliability = self._mean(scores, 0.0) if scores else None
        result = {
            "mm_reliability": mm_reliability,
            "total_samples": len(scores),
            "per_example": per_example,
        }
        if mm_reliability is None:
            result["reliability_unavailable_reason"] = "missing_reliability_inputs"
            logger.warning("MM reliability unavailable: no edit inputs were evaluated.")
        else:
            logger.info("MM Edit Reliability: %.4f", mm_reliability)
        return result


class MMEditGeneralizationEvaluator(_MMEditEvalHelper):
    """Multimodal generalization over text and image rephrases."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_generalization"

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        if processor is None:
            return {
                "mm_generalization": None,
                "text_rephrase_acc": None,
                "image_rephrase_acc": None,
                "text_rephrase_samples": 0,
                "image_rephrase_samples": 0,
                "total_samples": 0,
                "per_example": [],
                "generalization_unavailable_reason": "missing_processor",
            }

        self._init_mm(processor)
        model = self.prepare_model(model)
        text_scores = []
        image_scores = []
        per_example = []

        for index, item in enumerate(edit_data):
            _, target_new = self._normalize_prompt_target(
                item.get("prompt", ""), item.get("target_new", "")
            )
            image = item.get("image")
            case_text_scores = []
            case_image_scores = []

            for rephrase in self._normalize_rephrase_prompts(item):
                prompt, target = self._normalize_prompt_target(rephrase, target_new)
                if prompt and target:
                    score = self._mm_token_accuracy(
                        model, processor, prompt, target, image
                    )
                    text_scores.append(score)
                    case_text_scores.append(score)

            image_rephrase = item.get("image_rephrase")
            prompt, target = self._normalize_prompt_target(
                item.get("prompt", ""), target_new
            )
            if image_rephrase is not None and prompt and target:
                score = self._mm_token_accuracy(
                    model, processor, prompt, target, image_rephrase
                )
                image_scores.append(score)
                case_image_scores.append(score)

            case_scores = case_text_scores + case_image_scores
            if case_scores:
                per_example.append(
                    {
                        "index": index,
                        "score": self._mean(case_scores, 0.0),
                        "text_rephrase_acc": self._mean(case_text_scores, 0.0)
                        if case_text_scores
                        else None,
                        "image_rephrase_acc": self._mean(case_image_scores, 0.0)
                        if case_image_scores
                        else None,
                        "text_rephrase_samples": len(case_text_scores),
                        "image_rephrase_samples": len(case_image_scores),
                    }
                )

        all_scores = text_scores + image_scores
        result = {
            "mm_generalization": self._mean(all_scores, 0.0) if all_scores else None,
            "text_rephrase_acc": self._mean(text_scores, 0.0) if text_scores else None,
            "image_rephrase_acc": self._mean(image_scores, 0.0)
            if image_scores
            else None,
            "text_rephrase_samples": len(text_scores),
            "image_rephrase_samples": len(image_scores),
            "total_samples": len(all_scores),
            "per_example": per_example,
        }
        if result["mm_generalization"] is None:
            result["generalization_unavailable_reason"] = (
                "missing_generalization_inputs"
            )
            logger.warning(
                "MM generalization unavailable: no rephrase probes were evaluated."
            )
        else:
            logger.info("MM Edit Generalization: %.4f", result["mm_generalization"])
        return result


class MMEditLocalityEvaluator(_MMEditEvalHelper):
    """Pre/post invariance using predictions cached before the edit."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_locality"

    def capture_baseline(self, model, processor, edit_data):
        """Cache pre-edit locality predictions without cloning the model."""
        if processor is None:
            return None

        self._init_mm(processor)
        model = self.prepare_model(model)
        cases = []
        total_samples = 0
        for item in edit_data:
            text_predictions = []
            for group, prompt, expected in self._iter_eval_pairs(
                item.get("locality_inputs", item.get("locality")),
                "text_locality",
            ):
                tokens = self._mm_pred_tokens(model, processor, prompt, expected)
                text_predictions.append(
                    {
                        "group": group,
                        "tokens": tokens,
                    }
                )
                total_samples += bool(tokens)

            multimodal_predictions = []
            for probe in self._iter_multimodal_locality(item):
                prompt, expected = self._normalize_prompt_target(
                    probe.get("prompt", ""),
                    probe.get("ground_truth", probe.get("expected", "")),
                )
                if prompt and expected:
                    ranked_tokens = self._mm_pred_topk_tokens(
                        model,
                        processor,
                        prompt,
                        expected,
                        probe.get("image"),
                        top_k=10,
                    )
                    multimodal_predictions.append(
                        {
                            "top1_tokens": [row[0] for row in ranked_tokens if row],
                            "top10_tokens": ranked_tokens,
                        }
                    )
                    total_samples += bool(ranked_tokens)

            cases.append(
                {
                    "text": text_predictions,
                    "multimodal": multimodal_predictions,
                }
            )
        return {"cases": cases, "total_samples": total_samples}

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        baseline = kwargs.get("locality_baseline")
        original_model = kwargs.get("original_model")
        if processor is None:
            return {
                "mm_locality": None,
                "total_samples": 0,
                "per_example": [],
                "locality_unavailable_reason": "missing_processor",
            }
        if baseline is None and original_model is not None:
            baseline = self.capture_baseline(original_model, processor, edit_data)
        if baseline is None:
            logger.warning(
                "MM locality unavailable: no pre-edit prediction baseline was provided."
            )
            return {
                "mm_locality": None,
                "total_samples": 0,
                "per_example": [],
                "locality_unavailable_reason": "missing_pre_edit_baseline",
            }

        self._init_mm(processor)
        model = self.prepare_model(model)
        baseline_cases = baseline.get("cases", [])
        text_scores = []
        multimodal_scores = []
        multimodal_top10_scores = []
        per_example = []
        missing_baseline_samples = 0
        missing_top10_baseline_samples = 0

        for index, item in enumerate(edit_data):
            cached = baseline_cases[index] if index < len(baseline_cases) else {}
            cached_text = cached.get("text", [])
            cached_mm = cached.get("multimodal", [])
            case_text_scores = []
            case_mm_scores = []
            case_mm_top10_scores = []

            text_pairs = self._iter_eval_pairs(
                item.get("locality_inputs", item.get("locality")),
                "text_locality",
            )
            for probe_index, (_group, prompt, expected) in enumerate(text_pairs):
                if probe_index >= len(cached_text):
                    missing_baseline_samples += 1
                    continue
                pre = cached_text[probe_index].get("tokens", [])
                if not pre:
                    missing_baseline_samples += 1
                    continue
                post = self._mm_pred_tokens(model, processor, prompt, expected)
                score = self.token_match_ratio(pre, post)
                text_scores.append(score)
                case_text_scores.append(score)

            mm_probes = self._iter_multimodal_locality(item)
            for probe_index, probe in enumerate(mm_probes):
                prompt, expected = self._normalize_prompt_target(
                    probe.get("prompt", ""),
                    probe.get("ground_truth", probe.get("expected", "")),
                )
                if not prompt or not expected:
                    continue
                if probe_index >= len(cached_mm):
                    missing_baseline_samples += 1
                    missing_top10_baseline_samples += 1
                    continue
                cached_prediction = cached_mm[probe_index]
                if isinstance(cached_prediction, dict):
                    pre = cached_prediction.get("top1_tokens", [])
                    pre_top10 = cached_prediction.get("top10_tokens", [])
                else:
                    # Backward-compatible with baselines captured before the
                    # dual top-1/top-10 locality protocol was introduced.
                    pre = cached_prediction
                    pre_top10 = []
                if not pre:
                    missing_baseline_samples += 1
                    continue
                post_top10 = self._mm_pred_topk_tokens(
                    model,
                    processor,
                    prompt,
                    expected,
                    probe.get("image"),
                    top_k=10,
                )
                post = [row[0] for row in post_top10 if row]
                score = self.token_match_ratio(pre, post)
                multimodal_scores.append(score)
                case_mm_scores.append(score)
                if pre_top10:
                    top10_score = self.token_match_ratio(
                        self._flatten_ranked_tokens(pre_top10),
                        self._flatten_ranked_tokens(post_top10),
                    )
                    multimodal_top10_scores.append(top10_score)
                    case_mm_top10_scores.append(top10_score)
                else:
                    missing_top10_baseline_samples += 1

            case_scores = case_text_scores + case_mm_scores
            if case_scores:
                per_example.append(
                    {
                        "index": index,
                        "score": self._mean(case_scores, 0.0),
                        "text_locality_acc": self._mean(case_text_scores, 0.0)
                        if case_text_scores
                        else None,
                        "multimodal_locality_acc": self._mean(case_mm_scores, 0.0)
                        if case_mm_scores
                        else None,
                        "multimodal_locality_acc_top10": self._mean(
                            case_mm_top10_scores, 0.0
                        )
                        if case_mm_top10_scores
                        else None,
                        "text_locality_samples": len(case_text_scores),
                        "multimodal_locality_samples": len(case_mm_scores),
                        "multimodal_locality_top10_samples": len(case_mm_top10_scores),
                    }
                )

        all_scores = text_scores + multimodal_scores
        mm_locality = self._mean(all_scores, 0.0) if all_scores else None
        result = {
            "mm_locality": mm_locality,
            "text_locality_acc": self._mean(text_scores, 0.0) if text_scores else None,
            "multimodal_locality_acc": self._mean(multimodal_scores, 0.0)
            if multimodal_scores
            else None,
            "multimodal_locality_acc_top10": self._mean(multimodal_top10_scores, 0.0)
            if multimodal_top10_scores
            else None,
            "text_locality_samples": len(text_scores),
            "multimodal_locality_samples": len(multimodal_scores),
            "multimodal_locality_top10_samples": len(multimodal_top10_scores),
            "total_samples": len(all_scores),
            "missing_baseline_samples": missing_baseline_samples,
            "missing_top10_baseline_samples": missing_top10_baseline_samples,
            "locality_protocol": {
                "text": "ordered_top1_token_invariance",
                "multimodal_primary": "ordered_top1_token_invariance",
                "multimodal_vlkeb_ike_compatible": ("ordered_top10_token_invariance"),
            },
            "per_example": per_example,
        }
        if mm_locality is None:
            result["locality_unavailable_reason"] = "missing_locality_inputs"
            logger.warning(
                "MM locality unavailable: no locality probes were evaluated."
            )
        else:
            logger.info("MM Edit Locality: %.4f", mm_locality)
        return result


class MMEditPortabilityEvaluator(_MMEditEvalHelper):
    """Grouped transfer accuracy for multimodal edit requests."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "mm_edit_portability"

    def evaluate(self, model, output_dir=None, **kwargs):
        processor = kwargs.get("processor")
        edit_data = kwargs.get("edit_data", [])
        if processor is None:
            return {
                "mm_portability": None,
                "total_samples": 0,
                "per_example": [],
                "portability_unavailable_reason": "missing_processor",
            }

        self._init_mm(processor)
        model = self.prepare_model(model)
        case_scores = []
        total_samples = 0
        grouped_case_means = defaultdict(list)
        per_example = []

        for index, item in enumerate(edit_data):
            image = item.get("image")
            raw = item.get("portability_inputs", item.get("portability"))
            grouped_scores = defaultdict(list)
            for group, prompt, expected in self._iter_eval_pairs(raw, "portability"):
                probe_image = None if group.lower().startswith("text_") else image
                grouped_scores[group].append(
                    self._mm_token_accuracy(
                        model, processor, prompt, expected, probe_image
                    )
                )

            case_group_means = {
                group: self._mean(scores, 0.0)
                for group, scores in grouped_scores.items()
            }
            sample_count = sum(len(scores) for scores in grouped_scores.values())
            case_score = (
                self._mean(list(case_group_means.values()), 0.0)
                if case_group_means
                else None
            )
            total_samples += sample_count
            self._merge_group_case_means(grouped_case_means, case_group_means)
            if case_score is not None:
                case_scores.append(case_score)
                per_example.append(
                    {
                        "index": index,
                        "score": case_score,
                        "samples": sample_count,
                        "by_key": case_group_means,
                    }
                )

        mm_portability = self._mean(case_scores, 0.0) if case_scores else None
        result = {
            "mm_portability": mm_portability,
            "total_samples": total_samples,
            "evaluated_cases": len(case_scores),
            "portability_by_key": self._finalize_group_acc(grouped_case_means),
            "per_example": per_example,
        }
        if mm_portability is None:
            result["portability_unavailable_reason"] = "missing_portability_inputs"
            logger.warning("MM portability unavailable: no portability probes found.")
        else:
            logger.info("MM Edit Portability: %.4f", mm_portability)
        return result
