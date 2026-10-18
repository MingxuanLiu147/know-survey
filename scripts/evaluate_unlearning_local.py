#!/usr/bin/env python3
"""Evaluate target-token NLL before and after a local unlearning smoke.

This bounded evaluator uses fixed dataset-provided QA targets. It is not the
official TOFU evaluator and must not be reported as a full benchmark result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BOUNDARY_CONTRACT = {
    "version": "separate_prompt_target_v1",
    "prompt_encoding": "tokenizer(prompt, add_special_tokens=True)",
    "target_encoding": (
        "tokenizer(answer, add_special_tokens=False), followed by exactly one EOS"
    ),
    "concatenation": "input_ids = prompt_ids + target_ids",
    "labels": "-100 for every prompt token; target_ids for every target token",
    "truncation": (
        "Preserve at least one prompt token and EOS; left-truncate the prompt and "
        "right-truncate answer tokens only when max_length requires it."
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--edited-model", required=True)
    parser.add_argument("--forget-data", required=True)
    parser.add_argument("--retain-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records_with_identity(path: str) -> tuple[list[dict[str, Any]], dict]:
    resolved = Path(path).expanduser().resolve(strict=True)
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{resolved} must contain a non-empty JSON list")
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"{resolved}: record {index} must be a JSON object")
        if "question" not in record or "answer" not in record:
            raise ValueError(
                f"{resolved}: record {index} must contain question and answer"
            )
    return payload, {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(payload),
    }


def _model_identity(path: str) -> dict[str, Any]:
    requested = Path(path).expanduser()
    if not requested.is_dir():
        raise ValueError(
            "This local evaluator requires a concrete local model directory; "
            f"got {path!r}"
        )
    resolved = requested.resolve(strict=True)
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config missing: {config_path}")

    weight_paths = sorted(resolved.glob("*.safetensors"))
    if not weight_paths:
        weight_paths = sorted(resolved.glob("pytorch_model*.bin"))
    if not weight_paths:
        raise FileNotFoundError(f"No local model weights found in {resolved}")

    weights = [
        {
            "name": weight_path.name,
            "path": str(weight_path.resolve(strict=True)),
            "bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
        }
        for weight_path in weight_paths
    ]
    identity_material = {
        "config_sha256": _sha256(config_path),
        "weights": [
            {
                "name": item["name"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in weights
        ],
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "requested_path": path,
        "resolved_path": str(resolved),
        **identity_material,
        "weight_files": weights,
        "identity_sha256": identity_sha256,
    }


def _edited_safetensors_sha256(identity: dict[str, Any]) -> str:
    matching = [
        item["sha256"]
        for item in identity["weight_files"]
        if item["name"] == "model.safetensors"
    ]
    if len(matching) != 1:
        raise ValueError(
            "Edited checkpoint must contain exactly one model.safetensors file"
        )
    return matching[0]


def _encode_example(tokenizer, record: dict[str, Any], max_length: int):
    """Encode prompt and target separately so the supervision boundary is exact."""
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    question = str(record["question"])
    answer = str(record["answer"])
    if not question.strip() or not answer.strip():
        raise ValueError("Question and answer text must both be non-empty")

    prompt_ids = tokenizer(
        question,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if not prompt_ids or not answer_ids:
        raise ValueError("Tokenization produced an empty prompt or answer")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    eos_token_id = int(tokenizer.eos_token_id)
    eos_appended = not answer_ids or answer_ids[-1] != eos_token_id
    target_ids = list(answer_ids)
    if eos_appended:
        target_ids.append(eos_token_id)

    target_truncated = False
    max_target_tokens = max_length - 1
    if len(target_ids) > max_target_tokens:
        target_truncated = True
        target_ids = target_ids[: max_target_tokens - 1] + [eos_token_id]

    prompt_budget = max_length - len(target_ids)
    prompt_truncated = len(prompt_ids) > prompt_budget
    if prompt_truncated:
        prompt_ids = prompt_ids[-prompt_budget:]
    if not prompt_ids or len(target_ids) < 2:
        raise ValueError("Truncation removed the prompt or all answer tokens")

    input_ids = list(prompt_ids) + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    boundary = {
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": len(target_ids) - 1,
        "target_tokens": len(target_ids),
        "sequence_tokens": len(input_ids),
        "eos_appended": eos_appended,
        "prompt_truncated": prompt_truncated,
        "target_truncated": target_truncated,
    }
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
        boundary,
    )


def _evaluate_split(model, tokenizer, records, max_length: int, device):
    total_nll = 0.0
    total_tokens = 0
    per_example = []
    model.eval()
    for index, record in enumerate(records):
        input_ids, labels, boundary = _encode_example(tokenizer, record, max_length)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            loss = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                labels=labels,
            ).loss
        loss_value = float(loss)
        if not math.isfinite(loss_value):
            raise ValueError(f"Non-finite target NLL for example {index}")
        nll_sum = loss_value * boundary["target_tokens"]
        total_nll += nll_sum
        total_tokens += boundary["target_tokens"]
        per_example.append(
            {
                "index": index,
                **boundary,
                "target_nll": loss_value,
                "target_nll_sum": nll_sum,
            }
        )
    mean_target_nll = total_nll / total_tokens
    if not math.isfinite(mean_target_nll):
        raise ValueError("Evaluation produced a non-finite mean target NLL")
    return {
        "mean_target_nll": mean_target_nll,
        "target_nll_sum": total_nll,
        "target_tokens": total_tokens,
        "examples": len(records),
        "indices": [item["index"] for item in per_example],
        "per_example": per_example,
    }


def _load_model(path: str, device):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    return model.to(device)


def main() -> int:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_records, forget_identity = _load_records_with_identity(args.forget_data)
    retain_records, retain_identity = _load_records_with_identity(args.retain_data)
    base_identity = _model_identity(args.base_model)
    edited_identity = _model_identity(args.edited_model)

    base_model = _load_model(args.base_model, device)
    base = {
        "forget": _evaluate_split(
            base_model, tokenizer, forget_records, args.max_length, device
        ),
        "retain": _evaluate_split(
            base_model, tokenizer, retain_records, args.max_length, device
        ),
    }
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    edited_model = _load_model(args.edited_model, device)
    edited = {
        "forget": _evaluate_split(
            edited_model, tokenizer, forget_records, args.max_length, device
        ),
        "retain": _evaluate_split(
            edited_model, tokenizer, retain_records, args.max_length, device
        ),
    }

    summary = {
        "schema_version": 2,
        "status": "local_validation",
        "operation": "unlearning",
        "scope": f"bounded_{len(forget_records)}_forget_{len(retain_records)}_retain_slice",
        "max_length": args.max_length,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "boundary_contract": BOUNDARY_CONTRACT,
        "data": {
            "forget": forget_identity,
            "retain": retain_identity,
        },
        "base_model": args.base_model,
        "edited_model": args.edited_model,
        "base_model_identity": base_identity,
        "edited_model_identity": edited_identity,
        "edited_model_safetensors_sha256": _edited_safetensors_sha256(edited_identity),
        "base_forget_nll": base["forget"]["mean_target_nll"],
        "edited_forget_nll": edited["forget"]["mean_target_nll"],
        "forget_nll_change": (
            edited["forget"]["mean_target_nll"] - base["forget"]["mean_target_nll"]
        ),
        "base_retain_nll": base["retain"]["mean_target_nll"],
        "edited_retain_nll": edited["retain"]["mean_target_nll"],
        "retain_nll_change": (
            edited["retain"]["mean_target_nll"] - base["retain"]["mean_target_nll"]
        ),
        "base": base,
        "edited": edited,
        "claim_boundary": (
            "Local small-model engineering validation only; not an official "
            "TOFU benchmark result or multi-seed comparison."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    concise = {k: v for k, v in summary.items() if not isinstance(v, dict)}
    print(json.dumps(concise, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
