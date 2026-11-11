"""WikiDYK's released fact/QA schema, without instruction-template conversion.

Source: https://github.com/zhang-yu-wei/WikiDYK (ACL 2026).
Training reads facts and generated_qas; evaluation questions come only from eval.
Repeated examples are indexed lazily. The trainer supplies the shuffled sampler.
"""

import bisect
import json
import math
import random

import torch
from torch.utils.data import Dataset

from data.utils import append_eos_target, tokenize_causal_pair

QUESTION_TYPES = ("reliability", "generality", "paraphrase", "portability", "locality")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"WikiDYK requires a nonempty string for {field}.")
    return value


def answer_options(value):
    """Preserve reference order, including the primary answer used by the toolkit."""
    options = value if isinstance(value, list) else [value]
    if not options:
        raise ValueError("WikiDYK answer list is empty.")
    return [_text(answer, "answer") for answer in options]


def evaluation_examples(records):
    """Expand the five native question types; never attach facts to model prompts."""
    if not isinstance(records, list) or not records:
        raise ValueError("WikiDYK evaluation requires a nonempty list of records.")
    examples = []
    for row, record in enumerate(records):
        probes = record.get("eval")
        if not isinstance(probes, dict):
            raise ValueError(f"WikiDYK row {row} has no eval mapping.")
        for question_type in QUESTION_TYPES:
            if question_type not in probes:
                continue
            probe = probes[question_type]
            question = _text(probe.get("prompt"), f"{question_type}.prompt")
            examples.append(
                {
                    "id": f"{row}:{question_type}",
                    "case_id": record.get("case_id"),
                    "question_type": question_type,
                    "prompt": question + "\nAnswer:",
                    "expected": answer_options(probe.get("answer")),
                }
            )
    missing = set(QUESTION_TYPES) - {item["question_type"] for item in examples}
    if missing:
        raise ValueError(
            f"WikiDYK evaluation is missing question types: {sorted(missing)}"
        )
    return examples


class WikiDYKDataset(Dataset):
    """Fact NTP, fact + synthetic QA, or QA-only training for causal base models.

    ``synthetic_qa`` corresponds to the upstream ``qa_data_ratio=-1`` recipe:
    each fact/augmentation is repeated num_upsample times, and each fact's QAs
    cycle until there are at least num_upsample QA examples. Evaluation probes
    are never used as training QAs. QA prompts retain their inference tokenization;
    sampler order is controlled by the repository trainer.
    ``append_eos`` supervises the end of BOTH fact and QA targets and rejects
    samples whose complete target would be truncated. Legacy recipes opt out.
    ``loss_normalization=sequence`` gives each example its configured weight
    regardless of answer length. The default ``token`` preserves the original
    objective. Neither policy changes the labels or reads evaluation answers.
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        template_args=None,
        max_length=4096,
        num_upsample=1000,
        training_recipe="synthetic_qa",
        append_eos=False,
        fact_loss_weight=1.0,
        qa_loss_weight=1.0,
        loss_normalization="token",
    ):
        if (template_args or {}).get("apply_chat_template", False):
            raise ValueError("WikiDYK's base-model recipe requires chat templates off.")
        if training_recipe not in {"ntp", "synthetic_qa", "qa_only"}:
            raise ValueError("training_recipe must be ntp, synthetic_qa, or qa_only.")
        if (
            not isinstance(num_upsample, int)
            or isinstance(num_upsample, bool)
            or num_upsample < 1
        ):
            raise ValueError("num_upsample must be a positive integer.")
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        with open(data_path, encoding="utf-8") as stream:
            self.data = json.load(stream)
        if not isinstance(self.data, list) or not self.data:
            raise ValueError("WikiDYK training requires a nonempty JSON list.")
        self.tokenizer = tokenizer
        self.data_path = str(data_path)
        self.training_recipe = training_recipe
        self.max_length = max_length
        self.append_eos = append_eos
        if loss_normalization not in {"token", "sequence"}:
            raise ValueError("loss_normalization must be token or sequence.")
        self.loss_normalization = loss_normalization
        self.loss_weights = (float(fact_loss_weight), float(qa_loss_weight))
        if any(not math.isfinite(w) or w < 0 for w in self.loss_weights):
            raise ValueError("WikiDYK loss weights must be finite and nonnegative.")
        active_weights = (
            self.loss_weights
            if training_recipe == "synthetic_qa"
            else [self.loss_weights[int(training_recipe == "qa_only")]]
        )
        if not any(active_weights):
            raise ValueError("WikiDYK training requires a positive active loss weight.")
        self.blocks = []
        self.ends = []
        self.fact_examples = 0
        self.qa_examples = 0
        for row, record in enumerate(self.data):
            fact = _text(record.get("fact"), f"row {row}.fact")
            if fact.startswith("that"):
                fact = fact[4:]
            if fact.endswith("?"):
                fact = fact[:-1]
            facts = [_text(fact.strip(), f"row {row}.fact")]
            for field, key in (
                ("paraphrases", "paraphrased_fact"),
                ("reordered_facts", "reordered_fact"),
            ):
                for item in record.get(field, []):
                    facts.append(_text(item.get(key), f"row {row}.{key}"))
            if training_recipe != "qa_only":
                for text in facts:
                    self._add_block([(text, None)], num_upsample)
                    self.fact_examples += num_upsample
            if training_recipe in {"synthetic_qa", "qa_only"}:
                qas = record.get("generated_qas")
                if not isinstance(qas, list) or not qas:
                    raise ValueError(
                        f"WikiDYK row {row} has no generated_qas for QA training."
                    )
                pairs = [
                    (
                        _text(qa.get("question"), "generated question") + "\nAnswer:",
                        answer_options(qa.get("answer")),
                    )
                    for qa in qas
                ]
                count = max(len(pairs), num_upsample)
                self._add_block(pairs, count)
                self.qa_examples += count

    def _add_block(self, pairs, count):
        self.blocks.append(pairs)
        self.ends.append((self.ends[-1] if self.ends else 0) + count)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        block = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[block - 1] if block else 0)
        pairs = self.blocks[block]
        prompt, options = pairs[offset % len(pairs)]
        prefix = (self.tokenizer.bos_token or "") + prompt
        if options is not None:
            response = random.choice(options)
            # Joint tokenization can merge the final ':' with the answer's
            # first word. Preserve the exact inference prefix and supervise
            # every response token instead of masking by a separate length.
            prompt_ids, response_ids = tokenize_causal_pair(
                self.tokenizer, prefix, response, add_special_tokens=False
            )
            if self.append_eos:
                response_ids = append_eos_target(
                    prompt_ids, response_ids, self.tokenizer, self.max_length
                )
            input_ids = torch.tensor(
                (prompt_ids + response_ids)[: self.max_length], dtype=torch.long
            )
            prompt_length = min(len(prompt_ids), len(input_ids))
        else:
            input_ids = self.tokenizer(
                prefix,
                return_tensors="pt",
                padding=False,
                truncation=not self.append_eos,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"][0]
            if self.append_eos:
                input_ids = torch.tensor(
                    append_eos_target(
                        [], input_ids.tolist(), self.tokenizer, self.max_length
                    ),
                    dtype=torch.long,
                )
            # A padding token must never be trained to predict the first fact
            # token. This makes the NTP loss independent of batch padding.
            prompt_length = 1
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        target_count = int((labels[1:] != -100).sum())
        if not target_count:
            raise ValueError(
                f"WikiDYK sample {index} has no trainable target after truncation."
            )
        sample_weight = self.loss_weights[int(options is not None)]
        if self.loss_normalization == "sequence":
            # The shared trainer sums weight * CE and divides by weighted
            # target tokens over the complete accumulation window / DDP group.
            # w / n therefore yields sum(w * mean_CE) / sum(w), without
            # making short-answer QA depend on the length of the NTP branch.
            sample_weight /= target_count
        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_length": prompt_length,
            "loss_group": int(options is not None),
            "sample_weight": sample_weight,
        }
