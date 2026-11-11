"""One prompt-position contract for LoReFT training and inference.

Positions count real (non-padding) tokens in the prompt prefix, never tokens in
the supervised answer. Every row must contain at least k prompt tokens: PyVene
requires rectangular locations, so short prompts fail instead of duplicating a
position or silently changing k for the entire batch.
"""

from numbers import Integral

import torch


POSITION_CONTRACT_VERSION = 1


def loreft_position_config(last_k_prompt_tokens=1, allow_legacy_ntp=False):
    """Validate the shared position settings without changing the ReFT formula."""
    if (
        isinstance(last_k_prompt_tokens, bool)
        or not isinstance(last_k_prompt_tokens, Integral)
        or last_k_prompt_tokens < 1
    ):
        raise ValueError("LoReFT last_k_prompt_tokens must be a positive integer.")
    if not isinstance(allow_legacy_ntp, bool):
        raise ValueError("LoReFT allow_legacy_ntp must be a boolean.")
    if allow_legacy_ntp and last_k_prompt_tokens != 1:
        raise ValueError(
            "LoReFT legacy NTP is supported only with last_k_prompt_tokens=1."
        )
    return {
        "last_k_prompt_tokens": int(last_k_prompt_tokens),
        "allow_legacy_ntp": allow_legacy_ntp,
    }


def position_config_from_metadata(metadata):
    """Restore new settings, or the historical last1/NTP policy for old artifacts."""
    if not metadata.get("intervene_on_prompt", True):
        raise ValueError("LoReFT checkpoints must use intervene_on_prompt=true.")
    version = metadata.get("position_contract_version")
    if version is not None and version != POSITION_CONTRACT_VERSION:
        raise ValueError(f"Unsupported LoReFT position contract version: {version}")
    position = metadata.get("position", "last_prompt_token")
    if position not in {"last_prompt_token", "last_k_prompt_tokens"}:
        raise ValueError(f"Unsupported LoReFT position policy: {position}")
    if "last_k_prompt_tokens" not in metadata:
        if position != "last_prompt_token" or version is not None:
            raise ValueError(
                "LoReFT position metadata is missing last_k_prompt_tokens."
            )
        return loreft_position_config(1, metadata.get("allow_legacy_ntp", True))
    config = loreft_position_config(
        metadata["last_k_prompt_tokens"], metadata.get("allow_legacy_ntp", False)
    )
    if position == "last_prompt_token" and config["last_k_prompt_tokens"] != 1:
        raise ValueError("LoReFT last_prompt_token metadata conflicts with k != 1.")
    return config


def get_loreft_position_config(model):
    """Read attached checkpoint/trainer settings; raw old ReftModels use last1."""
    config = getattr(model, "_loreft_position_config", None)
    return (
        loreft_position_config(**config)
        if config is not None
        else loreft_position_config(1, True)
    )


def build_loreft_unit_locations(
    attention_mask,
    prompt_lengths,
    num_interventions,
    *,
    last_k_prompt_tokens=1,
    allow_legacy_ntp=False,
    labels=None,
    loss_group=None,
):
    """Return PyVene locations shaped [intervention, batch, k].

    ``prompt_lengths`` counts unpadded prefix tokens. ``attention_mask`` only
    maps those tokens to padded columns; its full length must never substitute
    for prompt length when answer tokens are present. ``loss_group=0`` explicitly
    identifies pure next-token text (a first-token prefix contains no QA context).
    ``allow_legacy_ntp`` exists solely to reproduce that historical k=1 recipe.
    Training/NLL callers supply labels so a supervised answer cannot be selected.
    """
    config = loreft_position_config(last_k_prompt_tokens, allow_legacy_ntp)
    k = config["last_k_prompt_tokens"]
    if (
        isinstance(num_interventions, bool)
        or not isinstance(num_interventions, Integral)
        or num_interventions < 1
    ):
        raise ValueError("LoReFT requires a positive integer number of interventions.")
    if attention_mask is None or prompt_lengths is None:
        raise ValueError("LoReFT requires attention_mask and unpadded prompt_length.")
    mask = torch.as_tensor(attention_mask).detach().cpu()
    lengths = torch.as_tensor(prompt_lengths).detach().cpu()
    if mask.ndim != 2 or not mask.shape[0] or not mask.shape[1]:
        raise ValueError("LoReFT attention_mask must be a nonempty batch of sequences.")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("LoReFT attention_mask must contain only 0 and 1.")
    if (
        lengths.ndim != 1
        or len(lengths) != len(mask)
        or lengths.dtype == torch.bool
        or lengths.is_floating_point()
        or lengths.is_complex()
    ):
        raise ValueError("LoReFT prompt_length must contain one integer per example.")
    if loss_group is not None:
        groups = torch.as_tensor(loss_group).detach().cpu()
        if groups.shape != lengths.shape:
            raise ValueError("LoReFT loss_group must contain one value per example.")
        if (groups == 0).any() and not allow_legacy_ntp:
            raise ValueError(
                "LoReFT QA prompt positions cannot use pure NTP rows (loss_group=0): "
                "the first token has no question context. Use QA-only data, or "
                "explicit allow_legacy_ntp=true with k=1 for historical reproduction."
            )
    if labels is not None:
        labels = torch.as_tensor(labels).detach().cpu()
        if labels.shape != mask.shape:
            raise ValueError("LoReFT labels must match attention_mask shape.")
    rows = []
    for index, (row_mask, prompt_length) in enumerate(zip(mask, lengths.tolist())):
        real_positions = row_mask.nonzero(as_tuple=True)[0]
        if prompt_length < k:
            raise ValueError(
                f"LoReFT example {index} has {prompt_length} prompt tokens, fewer than "
                f"last_k_prompt_tokens={k}; lower k instead of duplicating positions."
            )
        if prompt_length > len(real_positions):
            raise ValueError(
                "LoReFT prompt_length exceeds the number of real input tokens."
            )
        prefix = real_positions[:prompt_length]
        if labels is not None and (labels[index, prefix] != -100).any():
            raise ValueError(
                "LoReFT prompt_length includes supervised answer tokens; "
                "all prompt labels must be -100."
            )
        rows.append(prefix[-k:].tolist())
    locations = [[row.copy() for row in rows] for _ in range(num_interventions)]
    return {"sources->base": (None, locations)}
