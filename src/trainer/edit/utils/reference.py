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


def statistics_execution_options(editor) -> dict:
    """Expose an explicit statistics implementation without changing edit attention."""
    attention = getattr(editor, "mom2_attention_policy", "model_default")
    placement = getattr(editor, "mom2_placement_policy", "model_default")
    if attention == "model_default" and placement == "model_default":
        return {}
    options = {
        "attention_policy": attention,
        "query_chunk_size": getattr(editor, "mom2_query_chunk_size", 256),
        "placement_policy": placement,
    }
    device = getattr(editor, "mom2_execution_device", None)
    if device is not None:
        options["execution_backend"] = torch.device(device).type
    return options


def statistics_execution_metadata(editor) -> dict:
    """Bind alternative C0/projection computation to its actual implementation."""
    options = statistics_execution_options(editor)
    if not options:
        return {}
    from trainer.edit.utils.statistics_context import statistics_execution_identity

    return {
        "statistics_execution": statistics_execution_identity(editor.model, **options)
    }


def validate_statistics_configuration(editor, *, bind: bool = True) -> None:
    """Keep fixed base-model statistics from crossing a changed editor setup.

    Sequential weight updates intentionally retain C0. Replacing the model,
    tokenizer or corpus requires a fresh editor, rather than reusing that C0.
    """
    from trainer.edit.utils.layer_stats import statistics_data_identity

    identity = {
        "model_instance": id(editor.model),
        "tokenizer_instance": id(editor.tokenizer),
        "model_dtype": str(next(editor.model.parameters()).dtype),
        "attention_implementation": getattr(
            editor.model.config, "_attn_implementation", None
        ),
        "local_source": statistics_data_identity(
            getattr(editor, "mom2_data_path", None)
        ),
        **statistics_execution_metadata(editor),
        **{
            name: getattr(editor, name, None)
            for name in (
                "implementation_variant",
                "mom2_dataset",
                "mom2_dataset_config",
                "mom2_n_samples",
                "mom2_dtype",
                "mom2_sample_policy",
                "mom2_sample_seed",
                "mom2_batch_tokens",
                "mom2_max_length",
                "mom2_attention_policy",
                "mom2_query_chunk_size",
                "mom2_placement_policy",
                "rewrite_module_tmp",
                "covariance_ridge",
            )
        },
    }
    previous = getattr(editor, "_statistics_configuration", None)
    if previous is not None and previous != identity:
        raise ValueError(
            "Model/tokenizer/statistics configuration or attention policy changed after C0 initialization; "
            "restore the original policy or prepare a fresh editor/state from the base model"
        )
    if bind:
        editor._statistics_configuration = identity


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
    """Honor explicit corpus/resource choices in native and reference profiles."""
    options = {
        key: value
        for key, value in (
            ("dataset_config", getattr(editor, "mom2_dataset_config", None)),
            ("data_path", getattr(editor, "mom2_data_path", None)),
            ("batch_tokens", getattr(editor, "mom2_batch_tokens", None)),
            ("max_length", getattr(editor, "mom2_max_length", None)),
            ("execution_device", getattr(editor, "mom2_execution_device", None)),
            (
                "checkpoint_every_batches",
                getattr(editor, "mom2_checkpoint_every_batches", None),
            ),
        )
        if value is not None
    }
    if editor.implementation_variant == REFERENCE_VARIANT:
        options.update(
            sample_policy="upstream_random", sample_seed=editor.mom2_sample_seed
        )
    options.update(statistics_execution_options(editor))
    if getattr(editor, "mom2_require_cached", False):
        options["cache_only"] = True
    return options


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
