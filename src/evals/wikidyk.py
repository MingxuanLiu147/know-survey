"""Two WikiDYK score views over one set of generated answers.

Toolkit: acquisition, paraphrase generalization, relative knowledge retention.
Native: reliability/generality/paraphrase/portability/locality Match and F1.
Scoring follows upstream src/utils/metrics.py for the native view, including its
primary-reference F1 behavior. All saved scalar scores are fractions, not percent.
"""

from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import logging
import math
import re
import string

import torch

from data.wikidyk import QUESTION_TYPES, answer_options, evaluation_examples
from evals.inject import InjectEvaluator, generation_token_details
from model.reft_positions import (
    build_loreft_unit_locations,
    get_loreft_position_config,
)
from evals.fact_answers import FACT_ANSWER_SCORING_VERSION, fact_answer_scores

logger = logging.getLogger(__name__)


def _f1_tokens(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\b(a|an|the)\b", " ", text).split()


def score_answer(generated, expected):
    """Return toolkit substring accuracy and the author's native Match/F1 scores."""
    references = answer_options(expected)
    predicted_tokens = _f1_tokens(generated)
    reference_tokens = _f1_tokens(references[0])
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    f1 = (
        2 * overlap / (len(predicted_tokens) + len(reference_tokens))
        if overlap
        else 0.0
    )
    return {
        "toolkit_correct": references[0].strip().lower() in generated.lower(),
        "native_match": any(reference in generated for reference in references),
        "native_f1": f1,
    }


def summarize_predictions(predictions, reference_predictions=None):
    """Build both result views and compare retention on identical locality probes.

    Missing/zero reference accuracy makes retention unavailable. Ratios above one
    are retained. No average across the five different question types is created.
    """
    groups = {key: [] for key in QUESTION_TYPES}
    identifiers = [item["id"] for item in predictions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("WikiDYK prediction IDs must be unique.")
    for item in predictions:
        groups[item["question_type"]].append(item)
    if any(not rows for rows in groups.values()):
        raise ValueError("Both score views require all five WikiDYK question types.")
    native = {}
    for key, rows in groups.items():
        native[key] = {
            "match": sum(row["native_match"] for row in rows) / len(rows),
            "f1": sum(row["native_f1"] for row in rows) / len(rows),
            "count": len(rows),
        }

    def accuracy(rows):
        return sum(row["toolkit_correct"] for row in rows) / len(rows)

    current = groups["locality"]
    toolkit = {
        "acquisition": accuracy(groups["reliability"]),
        "generalization": accuracy(groups["paraphrase"]),
        "retention": None,
        "locality_after": accuracy(current),
        "locality_before": None,
        "counts": {
            "acquisition": len(groups["reliability"]),
            "generalization": len(groups["paraphrase"]),
            "retention": len(current),
        },
        "generalization_probe": "paraphrase",
        "unavailable_reason": "missing_reference_predictions",
    }
    if reference_predictions is not None:
        reference = {row["id"]: row for row in reference_predictions}
        if len(reference) != len(reference_predictions) or set(reference) != {
            row["id"] for row in current
        }:
            raise ValueError("Retention reference IDs must exactly match locality IDs.")
        for row in current:
            before = reference[row["id"]]
            if any(
                before.get(key) != row.get(key)
                for key in ("prompt", "expected", "question_type")
            ):
                raise ValueError(f"Retention reference probe differs for {row['id']}.")
        original_accuracy = accuracy(list(reference.values()))
        toolkit["locality_before"] = original_accuracy
        toolkit["unavailable_reason"] = (
            "zero_reference_accuracy" if original_accuracy == 0 else None
        )
        if original_accuracy:
            toolkit["retention"] = toolkit["locality_after"] / original_accuracy
    probe_statistics = {}
    for key, rows in groups.items():
        unique = {
            (row["prompt"], tuple(answer_options(row["expected"]))) for row in rows
        }
        probe_statistics[key] = {
            "rows": len(rows),
            "unique_prompt_answer_pairs": len(unique),
            "duplicate_rows": len(rows) - len(unique),
        }
    result = {
        "toolkit": toolkit,
        "native": native,
        "units": "fraction",
        "probe_statistics": probe_statistics,
    }
    # Additional diagnostics never replace the native/toolkit benchmark views.
    lexical = {}
    for key, rows in groups.items():
        diagnostics = [
            row["fact_diagnostics"] for row in rows if "fact_diagnostics" in row
        ]
        if diagnostics:
            lexical[key] = {"count": len(diagnostics)}
            for metric in (
                "answer_exact_match",
                "first_answer_token_match",
                "needs_review",
            ):
                lexical[key][metric] = sum(row[metric] for row in diagnostics) / len(
                    diagnostics
                )
    if lexical:
        result["fact_diagnostics"] = {
            "version": FACT_ANSWER_SCORING_VERSION,
            "semantics": "lexical first-answer match; not semantic accuracy",
            "by_question_type": lexical,
        }
    observed = [row for row in predictions if "finish_reason" in row]
    if observed:
        result["generation_diagnostics"] = {
            "observed_count": len(observed),
            "length_limit_rate": sum(
                row["finish_reason"] == "length" for row in observed
            )
            / len(observed),
            "eos_termination_rate": sum(
                row["finish_reason"] == "eos" for row in observed
            )
            / len(observed),
            "mean_output_tokens": sum(row["output_token_count"] for row in observed)
            / len(observed),
        }
        repetition = []
        for row in observed:
            words = re.findall(r"\w+", row["generated"].lower())
            grams = [tuple(words[i : i + 4]) for i in range(max(len(words) - 3, 0))]
            repetition.append(
                (len(grams) - len(set(grams))) / len(grams) if grams else 0.0
            )
        result["generation_diagnostics"].update(
            repeated_4gram_fraction_mean=sum(repetition) / len(repetition),
            severe_repetition_rate=sum(value >= 0.5 for value in repetition)
            / len(repetition),
            repetition_definition="word 4-grams; (occurrences - unique) / occurrences; severe >= 0.5",
        )
    return result


class WikiDYKEvaluator(InjectEvaluator):
    """Generate once per question and persist both scoring views plus predictions."""

    def __init__(self, eval_cfg, **kwargs):
        super().__init__(eval_cfg, **kwargs)
        self.name = "wikidyk"

    def _generation_kwargs(self, tokenizer):
        """Resolve explicit decoding settings without changing the legacy default."""
        args = self.eval_cfg.get("args", {})
        do_sample = args.get("do_sample", False)
        if not isinstance(do_sample, bool):
            raise ValueError("WikiDYK do_sample must be a boolean.")
        generation = {
            "max_new_tokens": int(args.get("max_new_tokens", 100)),
            "do_sample": do_sample,
            "pad_token_id": self._get_pad_token_id(tokenizer),
        }
        if generation["max_new_tokens"] < 1:
            raise ValueError("WikiDYK max_new_tokens must be positive.")
        if do_sample:
            # HF uses top_k=0 to disable filtering (vLLM also accepts -1). Explicit
            # values also prevent a checkpoint's top_k=50 from changing QA
            # sampling when matching upstream SamplingParams defaults.
            temperature = float(args.get("temperature", 1.0))
            top_p = float(args.get("top_p", 1.0))
            top_k = args.get("top_k", 0)
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("WikiDYK sampling temperature must be positive.")
            if not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("WikiDYK top_p must be in (0, 1].")
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
                raise ValueError("WikiDYK top_k must be a nonnegative integer.")
            seed = args.get("generation_seed", 42)
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed < 2**63
            ):
                raise ValueError(
                    "WikiDYK generation_seed must be an integer in [0, 2^63)."
                )
            generation.update(temperature=temperature, top_p=top_p, top_k=top_k)
        return generation

    def _predict(self, model, tokenizer, examples):
        is_reft = model.__class__.__name__ == "ReftModel"
        if is_reft:
            position_config = get_loreft_position_config(model)
            requested_position = self.eval_cfg.get("loreft_position")
            if requested_position not in {
                None,
                "last_prompt_token",
                "last_k_prompt_tokens",
            }:
                raise ValueError("Unsupported WikiDYK LoReFT position protocol.")
            if requested_position == "last_prompt_token" and (
                position_config["last_k_prompt_tokens"] != 1
            ):
                raise ValueError(
                    "WikiDYK last_prompt_token conflicts with saved LoReFT k."
                )
            if requested_position is None and not hasattr(
                model, "_loreft_position_config"
            ):
                raise ValueError(
                    "WikiDYK LoReFT without position metadata requires explicit "
                    "last_prompt_token protocol."
                )
        model = self.prepare_model(model)
        args = self.eval_cfg.get("args", {})
        batch_size = int(args.get("batch_size", 16))
        max_length = int(args.get("max_length", 4096))
        if batch_size < 1:
            raise ValueError("WikiDYK evaluation batch_size must be positive.")
        generation_args = self._generation_kwargs(tokenizer)
        device = self._get_model_device(model)
        predictions = []
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                inputs = tokenizer(
                    [row["prompt"] for row in batch],
                    padding=True,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                if inputs["input_ids"].shape[1] > max_length:
                    raise ValueError(
                        "WikiDYK evaluation prompt exceeds max_length; increase it instead of truncating the probe."
                    )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                length = inputs["input_ids"].shape[1]
                rng_devices = (
                    [
                        device.index
                        if device.index is not None
                        else torch.cuda.current_device()
                    ]
                    if device.type == "cuda"
                    else []
                )
                rng_context = (
                    torch.random.fork_rng(devices=rng_devices)
                    if generation_args["do_sample"]
                    else nullcontext()
                )
                with torch.inference_mode(), rng_context:
                    if generation_args["do_sample"]:
                        # Pair reference/updated locality with identical random
                        # draws and isolate each batch from earlier generations.
                        seed = args.get("generation_seed", 42) + start
                        torch.random.default_generator.manual_seed(seed)
                        if device.type == "cuda":
                            with torch.cuda.device(device):
                                torch.cuda.manual_seed(seed)
                    if is_reft:
                        # These inputs contain prompts only. Real-token counts
                        # are therefore the complete unpadded prompt lengths.
                        unit_locations = build_loreft_unit_locations(
                            inputs["attention_mask"],
                            inputs["attention_mask"].sum(dim=1),
                            len(model.sorted_keys),
                            **position_config,
                        )
                        _, generated = model.generate(
                            base=inputs,
                            unit_locations=unit_locations,
                            intervene_on_prompt=True,
                            **generation_args,
                        )
                    else:
                        generated = model.generate(**inputs, **generation_args)
                decoded = tokenizer.batch_decode(
                    generated[:, length:], skip_special_tokens=True
                )
                for row, output, token_ids in zip(
                    batch, decoded, generated[:, length:]
                ):
                    predictions.append(
                        {
                            **row,
                            "generated": output,
                            **score_answer(output, row["expected"]),
                            "fact_diagnostics": fact_answer_scores(
                                output, row["expected"]
                            ),
                            **generation_token_details(
                                token_ids,
                                model,
                                tokenizer,
                                generation_args["max_new_tokens"],
                            ),
                        }
                    )
                if (
                    start == 0
                    or (start // batch_size + 1) % 100 == 0
                    or start + batch_size >= len(examples)
                ):
                    logger.info(
                        "WikiDYK %s: generated %s/%s answers with %s",
                        batch[0]["question_type"],
                        len(predictions),
                        len(examples),
                        model.__class__.__name__,
                    )
        finally:
            tokenizer.padding_side = padding_side
        return predictions

    def evaluate(self, model, output_dir=None, **kwargs):
        tokenizer = kwargs.get("tokenizer")
        original_model = kwargs.get("original_model")
        if tokenizer is None or original_model is None:
            raise ValueError(
                "WikiDYK dual evaluation requires tokenizer and an explicit original_model/reference checkpoint."
            )
        examples = evaluation_examples(kwargs.get("eval_data"))
        locality = [row for row in examples if row["question_type"] == "locality"]
        reference = self._predict(original_model, tokenizer, locality)
        # Preserve the same locality batch composition for the updated and base
        # models, including Original. Both score views still share one answer.
        predictions = [
            row
            for question_type in QUESTION_TYPES
            for row in self._predict(
                model,
                tokenizer,
                [
                    example
                    for example in examples
                    if example["question_type"] == question_type
                ],
            )
        ]
        summary = summarize_predictions(predictions, reference)
        fingerprint = hashlib.sha256(
            json.dumps(examples, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        result = {
            **summary,
            "protocol_version": "wikidyk-dual-v1",
            "question_sha256": fingerprint,
            "generation": {
                "backend": "transformers",
                "batching": "question_type",
                "batch_size": int(self.eval_cfg.get("args", {}).get("batch_size", 16)),
                "max_length": int(
                    self.eval_cfg.get("args", {}).get("max_length", 4096)
                ),
                **self._generation_kwargs(tokenizer),
                **(
                    {
                        "generation_seed": self.eval_cfg.get("args", {}).get(
                            "generation_seed", 42
                        ),
                        "seed_scope": "batch",
                    }
                    if self.eval_cfg.get("args", {}).get("do_sample", False)
                    else {}
                ),
            },
            "intervention": (
                {
                    "position": "last_prompt_token",
                    "count": len(model.sorted_keys),
                    "intervene_on_prompt": True,
                }
                if model.__class__.__name__ == "ReftModel"
                else None
            ),
            "scoring": {
                "toolkit": "case-insensitive-primary-answer-substring",
                "native_match": "case-sensitive-any-answer-substring",
                "native_f1": "normalized-token-f1-primary-answer",
                "fact_diagnostics": FACT_ANSWER_SCORING_VERSION,
            },
            "predictions": predictions,
            "reference_predictions": reference,
        }
        directory = output_dir or self.eval_cfg.get("output_dir")
        if directory is None:
            raise ValueError("WikiDYK evaluation requires output_dir.")
        self.save_logs(result, self.get_logs_file_path(directory))
        self.save_logs(summary, self.get_logs_file_path(directory, "SUMMARY"))
        for view in ("toolkit", "native"):
            self.save_logs(
                {
                    view: summary[view],
                    "units": "fraction",
                    "question_sha256": fingerprint,
                },
                self.get_logs_file_path(directory, f"{view.upper()}_SUMMARY"),
            )
        return result
