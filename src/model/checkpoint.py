"""Restore saved full models, PEFT adapters and method-specific runtime state."""

import json
import logging
import os
from pathlib import Path
from copy import deepcopy
from types import MethodType
from omegaconf import DictConfig, open_dict

import torch

from model.reft_positions import position_config_from_metadata

logger = logging.getLogger(__name__)


def _fp32_loreft_forward(intervention, base, source=None, subspaces=None):
    """Keep the small LoReFT projection in FP32 and preserve the base dtype.

    PyReFT 0.0.7 directly applies learned_source to base without a dtype cast.
    Isolating this projection from outer autocast makes FP32 trainable weights
    safe with a BF16 base during both training and ordinary inference.
    """
    with torch.autocast(device_type=base.device.type, enabled=False):
        hidden = base.float()
        rotated = intervention.rotate_layer(hidden)
        source_hidden = intervention.act_fn(intervention.learned_source(hidden))
        output = hidden + torch.matmul(
            source_hidden - rotated, intervention.rotate_layer.weight.T
        )
    return intervention.dropout(output.to(base.dtype))


def configure_loreft_precision(intervention, policy):
    """Apply a saved precision policy without changing PyReFT's artifact type."""
    if policy == "legacy_model_dtype":
        return intervention
    if policy != "fp32_intervention":
        raise ValueError(f"Unsupported LoReFT precision policy: {policy}")
    intervention.float()
    # Keep the upstream intervention class in config.json. An instance-local
    # forward is restored explicitly from loreft_config.json by our loader.
    intervention.forward = MethodType(_fp32_loreft_forward, intervention)
    return intervention


def _restore_adalora_inference(model, model_path):
    """Preserve saved adapter precision after PEFT's rank resizing.

    PEFT can create the resized matrices in the base model's bf16 dtype before
    casting them back to fp32. Recopy the saved values into the already resized
    adapter, without applying the original rank mask for a second time.
    """
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    config = model.peft_config["default"]
    rank_pattern = config.rank_pattern
    weights = load_peft_weights(str(model_path), device="cpu")
    try:
        config.rank_pattern = None
        set_peft_model_state_dict(model, weights, adapter_name="default")
    finally:
        config.rank_pattern = rank_pattern
    # This loader restores inference state. AdaLoraModel.forward adds a training
    # orthogonality penalty whenever labels are supplied, and inference adapters
    # have no trainable_adapter_name. Evaluation NLL must contain only LM loss.
    model.base_model.forward = model.base_model.model.forward


def _load_loreft_checkpoint(model_path, base_model):
    """Restore local LoReFT tensors on the requested device and saved dtype."""
    import pyreft
    from pyvene.models.configuration_intervenable_model import RepresentationConfig

    config = pyreft.ReftConfig.from_pretrained(str(model_path))
    metadata_path = Path(model_path) / "loreft_config.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    precision_policy = metadata.get("precision_policy", "legacy_model_dtype")
    position_config = position_config_from_metadata(metadata)
    if not config.sorted_keys or len(config.sorted_keys) != len(config.representations):
        raise ValueError("LoReFT checkpoint needs one saved key per intervention.")
    representations = []
    states = []
    for index, (key, representation) in enumerate(
        zip(config.sorted_keys, config.representations)
    ):
        if config.intervention_types[index] != str(pyreft.LoreftIntervention):
            raise ValueError("LoReFT checkpoint contains an unsupported intervention.")
        if Path(key).name != key:
            raise ValueError("Invalid LoReFT intervention key.")
        state = torch.load(
            Path(model_path) / f"intkey_{key}.bin",
            map_location="cpu",
            weights_only=True,
        )
        weight = state["weight"]
        if precision_policy == "fp32_intervention" and any(
            tensor.dtype != torch.float32 for tensor in state.values()
        ):
            raise ValueError("LoReFT FP32 policy does not match checkpoint tensors.")
        intervention = pyreft.LoreftIntervention(
            embed_dim=weight.shape[1],
            low_rank_dimension=weight.shape[0],
            dtype=weight.dtype,
        )
        configure_loreft_precision(intervention, precision_policy)
        representation = RepresentationConfig(*representation)._replace(
            intervention_type=None, intervention=intervention
        )
        representations.append(representation)
        states.append(state)
    config.representations = representations
    config.intervention_types = [pyreft.LoreftIntervention] * len(representations)
    model = pyreft.get_reft_model(base_model, config)
    model._loreft_position_config = position_config
    for index, (key, state) in enumerate(zip(config.sorted_keys, states)):
        intervention = model.interventions[key]
        intervention.load_state_dict(state)
        if config.intervention_constant_sources is not None:
            intervention.is_source_constant = config.intervention_constant_sources[
                index
            ]
        if config.intervention_dimensions is not None:
            dimension = config.intervention_dimensions[index]
            if dimension is not None:
                intervention.set_interchange_dim(dimension)
    model.set_device(str(next(base_model.parameters()).device))
    return model


def write_training_artifact(output_dir, cfg):
    """Describe the saved inference state without claiming optimizer recovery."""
    root = Path(output_dir)
    artifact_type = "full_model"
    base_model = str(cfg.model.model_args.pretrained_model_name_or_path)
    if (root / "adapter_config.json").is_file():
        artifact_type = "peft_adapter"
        adapter = json.loads((root / "adapter_config.json").read_text())
        base_model = adapter.get("base_model_name_or_path") or base_model
    elif (root / "inject_artifact.json").is_file():
        artifact = json.loads((root / "inject_artifact.json").read_text())
        if artifact.get("method") == "loreft":
            artifact_type = "loreft_intervention"
    manifest = {
        "schema_version": 1,
        "operation": str(cfg.get("operation", cfg.get("mode", "train"))),
        "method": str(cfg.trainer.handler),
        "artifact_type": artifact_type,
        "base_model_name_or_path": base_model,
        "tokenizer_saved": (root / "tokenizer_config.json").is_file(),
        "restore_entrypoint": "model.checkpoint.load_checkpoint",
        "purpose": "inference",
    }
    (root / "training_artifact.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_checkpoint(model_cfg: DictConfig):
    """Load a Hugging Face/PEFT model or a saved pyreft intervention."""
    from model import get_model

    model_cfg = deepcopy(model_cfg)
    model_path = model_cfg.model_args.get("pretrained_model_name_or_path")
    if os.path.isfile(os.path.join(str(model_path), "tokenizer_config.json")):
        with open_dict(model_cfg.tokenizer_args):
            model_cfg.tokenizer_args.pretrained_model_name_or_path = str(model_path)
    from trainer.edit.artifacts import load_edit_artifacts, read_edit_artifact

    edit_artifact = read_edit_artifact(str(model_path))
    if edit_artifact is not None:
        edited_cfg = deepcopy(model_cfg)
        # The saved tokenizer is part of the editing checkpoint identity.
        with open_dict(edited_cfg.tokenizer_args):
            edited_cfg.tokenizer_args.pretrained_model_name_or_path = str(model_path)
        model, tokenizer = get_model(edited_cfg)
        load_edit_artifacts(model, tokenizer, model_path, manifest=edit_artifact)
        logger.info("Restored %s runtime from %s", edit_artifact["method"], model_path)
        return model, tokenizer
    adapter_config_path = os.path.join(str(model_path), "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        with open(adapter_config_path, encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(
                f"PEFT adapter {adapter_config_path} does not record its base model."
            )
        base_cfg = deepcopy(model_cfg)
        with open_dict(base_cfg.model_args):
            base_cfg.model_args.pretrained_model_name_or_path = base_model_path
        base_model, tokenizer = get_model(base_cfg)

        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(model_path))
        if adapter_config.get("peft_type") == "ADALORA":
            _restore_adalora_inference(model, model_path)
        logger.info("Loaded PEFT adapter from %s", model_path)
        return model, tokenizer

    artifact_path = os.path.join(str(model_path), "inject_artifact.json")
    if not os.path.isfile(artifact_path):
        return get_model(model_cfg)

    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("method") != "loreft":
        return get_model(model_cfg)

    base_model_path = artifact.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(
            f"LoReFT artifact {artifact_path} does not record its base model."
        )

    base_cfg = deepcopy(model_cfg)
    with open_dict(base_cfg.model_args):
        base_cfg.model_args.pretrained_model_name_or_path = base_model_path
    base_model, tokenizer = get_model(base_cfg)

    model = _load_loreft_checkpoint(model_path, base_model)
    logger.info("Loaded LoReFT intervention from %s", model_path)
    return model, tokenizer
