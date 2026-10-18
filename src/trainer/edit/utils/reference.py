"""Explicit reference protocols shared by ROME, MEMIT and AlphaEdit.

Equations and context policies follow kmeng01/rome 0874014, kmeng01/memit
80426fd and jianghoucheng/AlphaEdit b84624f. This adapter preserves local device,
tokenizer and transactional handling; it does not assert benchmark reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata

import torch

from trainer.edit.utils.compute_z import (
    _frozen_model,
    _input_device,
    _pad_token_rows,
    _token_ids,
    context_template_groups,
)

REFERENCE_VARIANT = "official_reference"


def validate_reference_profile(editor, method: str) -> None:
    """Reject a mislabeled profile rather than silently changing its objective."""
    if editor.implementation_variant not in {"native_adapted", REFERENCE_VARIANT}:
        raise ValueError(
            "implementation_variant must be native_adapted or official_reference"
        )
    if editor.implementation_variant != REFERENCE_VARIANT:
        return
    expected_readout = "final_logits" if method == "ROME" else "intermediate_logits"
    if (
        editor.v_loss_mode != expected_readout
        or editor.kl_direction != "edited_to_base"
    ):
        raise ValueError(
            f"{method} official_reference requires {expected_readout} and edited_to_base KL"
        )
    if method == "MEMIT" and editor.edit_weight != 1.0:
        raise ValueError("MEMIT official_reference requires edit_weight=1.0")
    if hasattr(editor, "mom2_adjustment") and not editor.mom2_adjustment:
        raise ValueError("official_reference requires real preservation second moments")
    if editor.context_templates is not None:
        groups = context_template_groups(editor.context_templates)
        if groups[0][0] != "{}":
            raise ValueError(
                "official_reference contexts must start with the bare {} request"
            )


def reference_stats_kwargs(editor) -> dict:
    """Keep native cache identities intact; reference statistics opt in explicitly."""
    if editor.implementation_variant != REFERENCE_VARIANT:
        return {}
    return {
        "dataset_config": editor.mom2_dataset_config,
        "sample_policy": "upstream_random",
        "sample_seed": editor.mom2_sample_seed,
        "batch_tokens": editor.mom2_batch_tokens,
    }


def _generate_contexts(model, tok, prompts, count, length):
    """Top-5 multinomial contexts, bounded by total length as in generate_fast.

    Full forwards avoid dependencies on version-specific KV cache objects.
    Modern tokenizer/device adapters may change RNG consumption vs upstream;
    the realized strings are persisted and can be supplied verbatim next run.
    """
    rows = [
        _token_ids(tok, prompt, add_special_tokens=True)
        for prompt in prompts
        for _ in range(count)
    ]
    if any(not row for row in rows):
        raise ValueError("Context generation prompts must tokenize to non-empty rows")
    device = _input_device(model)
    while any(len(row) < length for row in rows):
        batch, _ = _pad_token_rows(tok, rows, device)
        logits = model(**batch).logits
        positions = torch.tensor([len(row) - 1 for row in rows], device=logits.device)
        probabilities = logits[
            torch.arange(len(rows), device=logits.device), positions
        ].softmax(-1)
        values, indices = probabilities.topk(min(5, probabilities.shape[-1]), dim=-1)
        sampled = indices.gather(
            1, torch.multinomial(values / values.sum(-1, keepdim=True), 1)
        )
        for row, token in zip(rows, sampled[:, 0].tolist()):
            if len(row) < length:
                row.append(token)
    return [
        unicodedata.normalize("NFKD", tok.decode(row))
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for row in rows
    ]


def ensure_reference_contexts(editor, method: str) -> None:
    """Generate once on the original model and persist exact reproducible contexts."""
    if editor.implementation_variant != REFERENCE_VARIANT:
        return
    if editor.context_templates is None:
        device = _input_device(editor.model)
        devices = [device.index or 0] if device.type == "cuda" else []
        with _frozen_model(editor.model), torch.no_grad(), torch.random.fork_rng(
            devices=devices
        ):
            torch.manual_seed(editor.args.seed)
            if method == "ROME":
                seed = editor.tokenizer.eos_token or editor.tokenizer.bos_token
                if not seed:
                    raise ValueError(
                        "ROME context generation needs an EOS/BOS seed token"
                    )
                generated = [
                    text
                    for length, count in editor.context_template_length_params
                    for text in _generate_contexts(
                        editor.model, editor.tokenizer, [seed], count, length
                    )
                ]
                editor.context_templates = ["{}"] + [
                    text.replace("{", " ").replace("}", " ") + ". {}"
                    for text in generated
                ]
            else:
                generated = _generate_contexts(
                    editor.model,
                    editor.tokenizer,
                    ["The", "Therefore", "Because", "I", "You"],
                    1,
                    10,
                )
                editor.context_templates = [
                    ["{}"],
                    [
                        text.replace("{", " ").replace("}", " ") + ". {}"
                        for text in generated
                    ],
                ]
    destination = Path(editor.args.output_dir) / "reference_contexts.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "method": method,
                "seed": editor.args.seed,
                "context_templates": editor.context_templates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


__all__ = [
    "REFERENCE_VARIANT",
    "ensure_reference_contexts",
    "reference_stats_kwargs",
    "validate_reference_profile",
]
