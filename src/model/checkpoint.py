"""Restore saved full models, PEFT adapters and method-specific runtime state."""

import json
import logging
import os
from pathlib import Path
from copy import deepcopy
from omegaconf import DictConfig, open_dict

logger = logging.getLogger(__name__)


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

    import pyreft

    model = pyreft.ReftModel.load(str(model_path), base_model)
    model.set_device(str(next(base_model.parameters()).device))
    logger.info("Loaded LoReFT intervention from %s", model_path)
    return model, tokenizer
