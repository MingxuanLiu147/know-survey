"""MIT MEMIT value optimisation, adapted to Know-Surgery's public editor API.

Based on kmeng01/memit, revision 80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b,
``memit/compute_z.py``. Copyright (c) 2022 Kevin Meng; see LICENSES/MEMIT.txt.
The implementation was rebuilt from that licensed source in September 2026.
Device placement, token alignment, state restoration and selectable objectives
are local adaptations, not a claim of exact upstream reproduction.
See THIRD_PARTY_NOTICES.md for attribution and docs/editing.md for usage.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from trainer.edit.utils import nethook

logger = logging.getLogger(__name__)


def _module_placement(module, fallback: torch.Tensor | None = None):
    """Resolve a module's execution device without assuming the first GPU."""
    parameter = next(module.parameters(), None)
    hook_device = getattr(getattr(module, "_hf_hook", None), "execution_device", None)
    if hook_device is not None:
        device = torch.device(hook_device)
    elif parameter is not None and parameter.device.type != "meta":
        device = parameter.device
    elif fallback is not None:
        device = fallback.device
    else:
        raise ValueError("Cannot resolve module execution device")
    dtype = parameter.dtype if parameter is not None else None
    return device, dtype


def _input_device(model: PreTrainedModel) -> torch.device:
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter) and (embedding := getter()) is not None:
        return _module_placement(embedding)[0]
    for path in ("model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"):
        try:
            return _module_placement(nethook.get_module(model, path))[0]
        except LookupError:
            pass
    return _module_placement(model)[0]


def _hidden_state(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError("Traced module must return a tensor or a tensor-first sequence")
    return value


def format_request_prompt(prompt: str, subject: str) -> str:
    """Fill the subject placeholder, or keep an already materialised prompt."""
    return str(prompt).format(str(subject)) if "{}" in str(prompt) else str(prompt)


def _subject_start(prompt: str, subject: str) -> int:
    """Locate the inserted subject, retaining first-match behavior for plain text."""
    if "{}" in prompt:
        return len(prompt.split("{}", 1)[0].format(subject))
    return prompt.find(subject) if subject else -1


def _contextualize_request(prompt: str, subject: str, context: str):
    sentence = format_request_prompt(prompt, subject)
    start = _subject_start(prompt, subject)
    # The context may mention the subject too. Carry the request's position
    # through formatting instead of searching the fully materialized string.
    if start >= 0:
        start += len(context.split("{}", 1)[0].format(""))
    return context.format(sentence), start


def _token_ids(tok, text: str, *, add_special_tokens: bool) -> list[int]:
    values = tok(text, add_special_tokens=add_special_tokens)["input_ids"]
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: PreTrainedTokenizerBase,
    fact_token: str = "subject_last",
    *,
    subject_start: int | None = None,
) -> int:
    """Return a subject boundary token index including leading special tokens."""
    sentence = format_request_prompt(prompt, subject)
    if fact_token == "last":
        tokens = _token_ids(tok, sentence, add_special_tokens=True)
        if not tokens:
            raise ValueError("Cannot look up a token in an empty prompt")
        return len(tokens) - 1
    if fact_token not in ("subject_first", "subject_last"):
        raise ValueError(f"Unsupported fact_token={fact_token!r}")
    begin = _subject_start(prompt, subject) if subject_start is None else subject_start
    if not subject or begin < 0 or sentence[begin : begin + len(subject)] != subject:
        raise ValueError(f"Subject {subject!r} is not present in prompt {sentence!r}")
    end = begin + len(subject)
    try:
        spans = tok(sentence, return_offsets_mapping=True, add_special_tokens=True).get(
            "offset_mapping"
        )
    except (TypeError, NotImplementedError, ValueError):
        spans = None
    if spans is not None:
        indices = [
            index
            for index, (left, right) in enumerate(spans)
            if left < end and right > begin and right > left
        ]
        if indices:
            return indices[0 if fact_token == "subject_first" else -1]
    # Slow tokenizers have no character offsets. Use the subject-prefix boundary
    # as in MEMIT's word-token lookup, keeping tokenizer-added leading tokens.
    through = _token_ids(tok, sentence[:end], add_special_tokens=True)
    # Byte-level tokenizers attach a preceding space to the first subject
    # token. A standalone trailing-space token must not shift subject_first.
    prefix_text = sentence[:begin]
    if prefix_text.endswith(" "):
        prefix_text = prefix_text[:-1]
    prefix = _token_ids(tok, prefix_text, add_special_tokens=True)
    if not through:
        raise ValueError("Subject tokenized to an empty sequence")
    return (
        min(len(prefix), len(through) - 1)
        if fact_token == "subject_first"
        else len(through) - 1
    )


def _pad_token_rows(tok, rows: Sequence[Sequence[int]], device, *, left=False):
    if not rows or any(not row for row in rows):
        raise ValueError("At least one non-empty token sequence is required per prompt")
    lengths = torch.tensor([len(row) for row in rows], device=device)
    width = int(lengths.max().item())
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tok, "eos_token_id", None)
    ids = torch.full(
        (len(rows), width),
        0 if pad_id is None else pad_id,
        device=device,
        dtype=torch.long,
    )
    offsets = width - lengths if left else torch.zeros_like(lengths)
    for index, (row, offset) in enumerate(zip(rows, offsets.tolist())):
        ids[index, offset : offset + len(row)] = torch.tensor(
            row, dtype=torch.long, device=device
        )
    positions = torch.arange(width, device=device).unsqueeze(0)
    # Token identity is not padding identity: a real EOS must remain visible even
    # when the tokenizer uses eos_token_id as pad_token_id.
    mask = (positions >= offsets[:, None]) & (positions < (offsets + lengths)[:, None])
    return {"input_ids": ids, "attention_mask": mask.long()}, offsets.tolist()


def _prompt_batch(tok, prompts: Sequence[str], device):
    # Representation lookup is a forward pass, not autoregressive generation.
    # Right padding keeps real tokens at their single-example positions even in
    # models with learned absolute embeddings (GPT-2). Do not mutate the caller's
    # tokenizer.padding_side, which may legitimately be left for generation.
    return _pad_token_rows(
        tok,
        [_token_ids(tok, prompt, add_special_tokens=True) for prompt in prompts],
        device,
    )


def _select_rows(hidden: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[0] != len(indices):
        raise ValueError("Expected batch-first [batch, tokens, hidden] representations")
    rows = torch.arange(len(indices), device=hidden.device)
    columns = torch.tensor(indices, device=hidden.device)
    return hidden[rows, columns]


@contextmanager
def _frozen_model(model):
    """Suspend parameter gradients/dropout and restore each original state."""
    parameters = [
        (parameter, parameter.requires_grad) for parameter in model.parameters()
    ]
    modes = [(module, module.training) for module in model.modules()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        model.eval()
        yield
    finally:
        for parameter, state in parameters:
            parameter.requires_grad_(state)
        for module, state in modes:
            module.training = state


def get_module_input_output_at_words(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    subjects: Sequence[str],
    module_name: str,
    *,
    fact_token: str = "subject_last",
    context_templates: Optional[Sequence[str | Sequence[str]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average traced subject-token input/output over contexts for each request."""
    if len(prompts) != len(subjects) or not prompts:
        raise ValueError("prompts and subjects must have identical non-zero lengths")
    groups = context_template_groups(context_templates)
    templates = [template for group in groups for template in group]
    expanded = [
        (*_contextualize_request(prompt, subject, context), subject)
        for prompt, subject in zip(prompts, subjects)
        for context in templates
    ]
    batch, offsets = _prompt_batch(
        tok, [text for text, _, _ in expanded], _input_device(model)
    )
    indices = [
        find_fact_lookup_idx(text, subject, tok, fact_token, subject_start=start)
        + offset
        for (text, start, subject), offset in zip(expanded, offsets)
    ]
    with _frozen_model(model), torch.no_grad(), nethook.Trace(
        model,
        module_name,
        retain_input=True,
        retain_output=True,
        clone=True,
        detach=True,
    ) as captured:
        model(**batch)
    gathered = (
        _select_rows(_hidden_state(captured.input), indices),
        _select_rows(_hidden_state(captured.output), indices),
    )
    # MEMIT/AlphaEdit weight context *types* equally, even if group sizes differ.
    # A flat list remains one group, preserving native/ROME averaging.
    return tuple(
        torch.stack(
            [
                chunk.mean(1)
                for chunk in value.reshape(len(prompts), len(templates), -1).split(
                    [len(group) for group in groups], dim=1
                )
            ],
            dim=1,
        ).mean(1)
        for value in gathered
    )


def context_template_groups(templates=None) -> list[list[str]]:
    """Normalize flat ROME contexts or grouped MEMIT/AlphaEdit contexts."""
    if templates is None:
        return [["{}"]]
    values = list(templates)
    if not values:
        raise ValueError("context_templates cannot be empty")
    groups = [values] if all(isinstance(item, str) for item in values) else values
    if any(
        isinstance(group, str)
        or not group
        or any(not isinstance(item, str) or item.count("{}") != 1 for item in group)
        for group in groups
    ):
        raise ValueError(
            "Contexts must be non-empty groups with one {} placeholder per template"
        )
    return [list(group) for group in groups]


def get_layer_outputs_at_words(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    subjects: Sequence[str],
    layer: int,
    *,
    layer_module_tmp: str = "model.layers.{}",
    fact_token: str = "subject_last",
) -> torch.Tensor:
    """Return per-request layer output at the chosen fact token."""
    return get_module_input_output_at_words(
        model,
        tok,
        prompts,
        subjects,
        layer_module_tmp.format(layer),
        fact_token=fact_token,
    )[1]


def _build_rewrite_batch(
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    target_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Right-pad causal teacher-forcing rows and align next-token target labels."""
    targets = target_ids.detach().cpu().tolist()
    if not targets:
        raise ValueError("target_new tokenized to an empty sequence")
    prefixes = [_token_ids(tok, prompt, add_special_tokens=True) for prompt in prompts]
    if any(not prefix for prefix in prefixes):
        raise ValueError("Rewrite prompts must contain at least one token")
    batch, _ = _pad_token_rows(
        tok, [prefix + targets[:-1] for prefix in prefixes], device
    )
    labels = torch.full_like(batch["input_ids"], -100)
    for row, prefix in enumerate(prefixes):
        start = len(prefix) - 1
        labels[row, start : start + len(targets)] = torch.tensor(targets, device=device)
    return batch, labels


def _readout(module, hidden: torch.Tensor) -> torch.Tensor:
    device, dtype = _module_placement(module, hidden)
    return module(hidden.to(device=device, dtype=dtype or hidden.dtype))


def _rewrite_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(logits.device)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _kl_penalty(base_log_probs, edited_log_probs, direction: str):
    if direction == "base_to_edited":
        return F.kl_div(
            edited_log_probs, base_log_probs, log_target=True, reduction="sum"
        )
    return F.kl_div(base_log_probs, edited_log_probs, log_target=True, reduction="sum")


def compute_z(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    request: Dict[str, Any],
    layer: int,
    *,
    layer_module_tmp: str = "model.layers.{}",
    lm_head_module: str = "lm_head",
    ln_f_module: str = "model.norm",
    v_lr: float = 5e-1,
    v_num_grad_steps: int = 20,
    v_weight_decay: float = 0.5,
    v_loss_layer: int = 31,
    v_loss_mode: str = "final_logits",
    kl_direction: str = "base_to_edited",
    clamp_norm_factor: float = 4.0,
    kl_factor: float = 0.0625,
    fact_token: str = "subject_last",
    context_templates: Optional[Sequence[str | Sequence[str]]] = None,
    intervention_module_tmp: str | None = None,
    optimization_convention: str = "native",
    prompt_key: str = "prompt",
    target_key: str = "target_new",
) -> torch.Tensor:
    """Optimise MEMIT's shared latent displacement, returning the desired value.

    Final-model logits and KL(base || edited) remain the native defaults.
    ``intermediate_logits`` reads the explicitly selected layer through norm/head;
    ``edited_to_base`` selects the licensed upstream KL orientation. Optimisation
    updates only the latent displacement and never the model's parameters.
    """
    if v_num_grad_steps < 1:
        raise ValueError("v_num_grad_steps must be positive")
    if v_loss_mode not in ("final_logits", "intermediate_logits"):
        raise ValueError(f"Unsupported v_loss_mode: {v_loss_mode}")
    if kl_direction not in ("base_to_edited", "edited_to_base"):
        raise ValueError(f"Unsupported kl_direction: {kl_direction}")
    if min(v_lr, v_weight_decay, clamp_norm_factor, kl_factor) < 0:
        raise ValueError("Optimisation coefficients must be non-negative")
    if optimization_convention not in {"native", "upstream"}:
        raise ValueError(
            f"Unsupported optimization_convention: {optimization_convention}"
        )
    edited_layer = (intervention_module_tmp or layer_module_tmp).format(layer)
    nethook.get_module(model, edited_layer)
    effective_loss_layer = (
        max(layer, v_loss_layer)
        if optimization_convention == "upstream"
        else v_loss_layer
    )
    loss_layer = layer_module_tmp.format(effective_loss_layer)
    if v_loss_mode == "intermediate_logits":
        if effective_loss_layer < layer:
            raise ValueError("v_loss_layer must be at or after the edited layer")
        nethook.get_module(model, loss_layer)
        norm = nethook.get_module(model, ln_f_module)
        head = nethook.get_module(model, lm_head_module)
    subject = str(request.get("subject") or "")
    if not subject:
        raise ValueError("compute_z requires request['subject'] for fact lookup")
    prompt = str(request[prompt_key])
    target_text = str(request[target_key])
    target_tokens = _token_ids(
        tok,
        target_text if target_text.startswith(" ") else " " + target_text,
        add_special_tokens=False,
    )
    if not target_tokens:
        raise ValueError("target_new tokenized to an empty sequence")
    contextualized = [
        _contextualize_request(prompt, subject, template)
        for group in context_template_groups(context_templates)
        for template in group
    ]
    contexts = [text for text, _ in contextualized]
    device = _input_device(model)
    batch, labels = _build_rewrite_batch(
        tok, contexts, torch.tensor(target_tokens), device
    )
    lookup = [
        find_fact_lookup_idx(text, subject, tok, fact_token, subject_start=start)
        for text, start in contextualized
    ]
    kl_text = f"{subject} is a"
    kl_batch, offsets = _prompt_batch(tok, [kl_text], device)
    kl_lookup = [find_fact_lookup_idx(kl_text, subject, tok, fact_token) + offsets[0]]

    with _frozen_model(model):
        # Record a true unmodified reference before creating the intervention.
        with torch.no_grad(), nethook.Trace(
            model, edited_layer, detach=True, clone=True
        ) as clean:
            model(**batch)
        initial_value = (
            _select_rows(_hidden_state(clean.output), lookup)[0].detach().clone()
        )
        displacement = torch.zeros_like(
            initial_value, dtype=torch.float32, requires_grad=True
        )
        if kl_factor:
            with torch.no_grad():
                base_kl = (
                    _select_rows(model(**kl_batch).logits, kl_lookup)
                    .float()
                    .log_softmax(-1)
                    .detach()
                )

        def forward_with_displacement(inputs, positions, intermediate=False):
            def intervene(output):
                hidden = _hidden_state(output)
                updated = hidden.clone()
                row_ids = torch.arange(len(positions), device=hidden.device)
                token_ids = torch.tensor(positions, device=hidden.device)
                updated[row_ids, token_ids] = hidden[
                    row_ids, token_ids
                ] + displacement.to(hidden)
                if isinstance(output, tuple):
                    return (updated, *output[1:])
                if isinstance(output, list):
                    return [updated, *output[1:]]
                return updated

            with ExitStack() as hooks:
                hooks.enter_context(
                    nethook.Trace(
                        model, edited_layer, retain_output=False, edit_output=intervene
                    )
                )
                capture = (
                    hooks.enter_context(nethook.Trace(model, loss_layer))
                    if intermediate
                    else None
                )
                output = model(**inputs)
                if capture is None:
                    return output.logits
                return _readout(head, _readout(norm, _hidden_state(capture.output)))

        optimizer = torch.optim.Adam([displacement], lr=v_lr)
        initial_norm = initial_value.float().norm()
        for iteration in range(v_num_grad_steps):
            optimizer.zero_grad(set_to_none=True)
            rewrite_logits = forward_with_displacement(
                batch, lookup, v_loss_mode == "intermediate_logits"
            )
            nll = _rewrite_nll(rewrite_logits, labels)
            divergence = nll.new_zeros(())
            if kl_factor:
                logits = forward_with_displacement(kl_batch, kl_lookup)
                edited_kl = _select_rows(logits, kl_lookup).float().log_softmax(-1)
                divergence = kl_factor * _kl_penalty(
                    base_kl.to(edited_kl.device), edited_kl, kl_direction
                )
            regularizer = (
                v_weight_decay
                * displacement.norm()
                / initial_norm.square().clamp_min(1e-12)
            )
            objective = nll + divergence.to(nll.device) + regularizer.to(nll.device)
            if not torch.isfinite(objective):
                raise RuntimeError("Non-finite value optimisation objective")
            # Official ROME/MEMIT evaluate the last iteration without another
            # Adam step, and test the stopping threshold before any update.
            if optimization_convention == "upstream" and (
                objective.item() < 0.05 or iteration == v_num_grad_steps - 1
            ):
                break
            objective.backward()
            optimizer.step()
            with torch.no_grad():
                radius = clamp_norm_factor * initial_norm
                displacement.mul_(
                    (radius / displacement.norm().clamp_min(1e-12)).clamp(max=1)
                )
            logger.debug(
                "value optimisation layer=%s step=%d nll=%.5f kl=%.5f",
                edited_layer,
                iteration + 1,
                nll.item(),
                divergence.item(),
            )
            if objective.item() < 0.05:
                break
        return initial_value + displacement.detach().to(initial_value)


def compute_ks(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    layer: int,
    *,
    layer_module_tmp: str = "model.layers.{}",
) -> Tuple[torch.Tensor, list[int]]:
    """Compatibility API for collecting the last real token's layer output."""
    batch, _ = _prompt_batch(tok, prompts, _input_device(model))
    mask = batch["attention_mask"].bool()
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask, -1).max(1).values.tolist()
    with _frozen_model(model), torch.no_grad(), nethook.Trace(
        model, layer_module_tmp.format(layer), detach=True, clone=True
    ) as captured:
        model(**batch)
    return _select_rows(_hidden_state(captured.output), indices), indices


__all__ = [
    "compute_ks",
    "compute_z",
    "find_fact_lookup_idx",
    "format_request_prompt",
    "get_layer_outputs_at_words",
    "get_module_input_output_at_words",
]
