"""Complete AdaLoRA training state, separate from PEFT's pruned inference adapter.

Optimizer, scheduler, RNG and data progress remain owned by Transformers Trainer.
This module restores method parameters in place so its optimizer references stay
valid. It supports ordinary, unsharded Trainer checkpoints. The current native
data-skip path permits exact replay within the first epoch or at epoch boundaries;
mid-epoch resumes in later epochs fail instead of changing the sampled permutation.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


METADATA_FILE = "adalora_training.json"
TENSORS_FILE = "adalora_training.safetensors"
_GROUPS = ("ipt", "exp_avg_ipt", "exp_avg_unc")
_CONFIG_KEYS = (
    "init_r",
    "target_r",
    "total_step",
    "tinit",
    "tfinal",
    "deltaT",
    "beta1",
    "beta2",
    "lora_alpha",
    "lora_dropout",
    "orth_reg_weight",
    "target_modules",
    "bias",
    "task_type",
    "fan_in_fan_out",
    "init_lora_weights",
)
_TRAINING_KEYS = (
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "world_size",
    "max_steps",
    "num_train_epochs",
    "seed",
    "data_seed",
    "lr_scheduler_type",
    "warmup_steps",
    "warmup_ratio",
    "optim",
    "optim_args",
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "dataloader_drop_last",
    "dataloader_num_workers",
    "group_by_length",
    "ignore_data_skip",
    "bf16",
    "fp16",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
)


def _json_value(value):
    return json.loads(json.dumps(value, default=lambda item: sorted(item)))


def _parts(trainer, model=None):
    if trainer.is_deepspeed_enabled or trainer.is_fsdp_enabled:
        raise ValueError("Complete AdaLoRA state requires an unsharded Trainer model.")
    model = trainer.accelerator.unwrap_model(trainer.model if model is None else model)
    base = model.base_model
    adapter = base.trainable_adapter_name
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or name.endswith(f".ranknum.{adapter}")
    }
    config = base.peft_config[adapter]
    return base, adapter, parameters, config, base.rankallocator


def _training_identity(trainer):
    identity = {key: getattr(trainer.args, key) for key in _TRAINING_KEYS}
    identity["global_batch"] = (
        trainer.args.per_device_train_batch_size
        * trainer.args.gradient_accumulation_steps
        * trainer.args.world_size
    )
    identity["dataset_length"] = len(trainer.train_dataset)
    identity["use_seedable_sampler"] = (
        trainer.accelerator.dataloader_config.use_seedable_sampler
    )
    return _json_value(identity)


def save_adalora_training_state(trainer, output_dir):
    """Save full-rank parameters, rank assignment and allocator history on rank 0."""
    if not trainer.is_world_process_zero():
        return
    _, adapter, parameters, config, allocator = _parts(trainer)
    metadata = {
        "schema_version": 1,
        "method": "adalora",
        "adapter": adapter,
        "global_step": trainer.state.global_step,
        "config": _json_value({key: getattr(config, key) for key in _CONFIG_KEYS}),
        "training": _training_identity(trainer),
        "rank_pattern": config.rank_pattern,
        "allocator": {
            key: getattr(allocator, key)
            for key in ("init_bgt", "target_bgt", "name_set")
        },
        "parameters": {name: p.requires_grad for name, p in parameters.items()},
    }
    tensors = {
        "parameter::" + name: p.detach().cpu().contiguous().clone()
        for name, p in parameters.items()
    }
    for group in _GROUPS:
        values = getattr(allocator, group)
        metadata[group] = sorted(values)
        for name, value in values.items():
            tensors[group + "::" + name] = value.detach().cpu().contiguous().clone()
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / (TENSORS_FILE + ".tmp")
    save_file(
        tensors, str(temporary), metadata={"global_step": str(metadata["global_step"])}
    )
    temporary.replace(path / TENSORS_FILE)
    temporary = path / (METADATA_FILE + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path / METADATA_FILE)


def load_adalora_training_state(trainer, checkpoint, model=None):
    """Validate a complete checkpoint before copying any parameters in place.

    A PEFT inference adapter alone is insufficient. Missing allocator history,
    changed schedules/training settings, and incompatible tensors fail explicitly.
    Frozen base and input-file identity are the caller's responsibility.
    """
    path = Path(checkpoint)
    rng_name = (
        f"rng_state_{trainer.args.process_index}.pth"
        if trainer.args.world_size > 1
        else "rng_state.pth"
    )
    required = [
        METADATA_FILE,
        TENSORS_FILE,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        rng_name,
    ]
    if trainer.accelerator.scaler is not None:
        required.append("scaler.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete AdaLoRA training checkpoint; missing {missing}.")
    metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
    base, adapter, parameters, config, allocator = _parts(trainer, model)
    expected_config = _json_value({key: getattr(config, key) for key in _CONFIG_KEYS})
    if (
        metadata.get("schema_version"),
        metadata.get("method"),
        metadata.get("adapter"),
    ) != (1, "adalora", adapter):
        raise ValueError("AdaLoRA training checkpoint identity differs.")
    if metadata.get("config") != expected_config:
        raise ValueError(
            "AdaLoRA method configuration or schedule changed during resume."
        )
    if metadata.get("training") != _training_identity(trainer):
        raise ValueError(
            "AdaLoRA training settings or dataset length changed during resume."
        )
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    step = metadata.get("global_step")
    if (
        step != state.get("global_step")
        or not isinstance(step, int)
        or not 0 <= step <= config.total_step
    ):
        raise ValueError("AdaLoRA method step differs from Trainer state or schedule.")
    if (
        trainer.args.ignore_data_skip
        or not trainer.accelerator.dataloader_config.use_seedable_sampler
    ):
        raise ValueError(
            "AdaLoRA resume requires data skipping and a seedable sampler."
        )
    epoch = float(state.get("epoch") or 0.0)
    if epoch >= 1 and not epoch.is_integer():
        # Accelerate's skip_first_batches constructs a new DataLoaderShard with
        # iteration=0. Its __iter__ then resets the epoch of the seeded sampler.
        # Do not silently resume from the wrong permutation in this native path.
        raise ValueError(
            "AdaLoRA exact resume inside a later epoch is unsupported by the "
            "native data-skip path; use an epoch-boundary checkpoint."
        )
    expected_allocator = {
        key: getattr(allocator, key) for key in ("init_bgt", "target_bgt", "name_set")
    }
    if metadata.get("allocator") != expected_allocator:
        raise ValueError("AdaLoRA allocator structure differs.")
    if metadata.get("parameters") != {
        name: p.requires_grad for name, p in parameters.items()
    }:
        raise ValueError("AdaLoRA trainable parameter inventory differs.")

    # Allocator keys are relative to base.model; parameter keys use the PEFT root.
    method_parameters = dict(base.model.named_parameters())
    expected_history = {
        name for name in method_parameters if "lora_" in name and adapter in name
    }
    history_required = 0 < step < config.total_step - config.tfinal
    for group in _GROUPS:
        names = metadata.get(group)
        if not isinstance(names, list) or set(names) != (
            expected_history if history_required else set()
        ):
            raise ValueError(f"AdaLoRA {group} history is missing or incompatible.")
    if "rank_pattern" not in metadata:
        raise ValueError("AdaLoRA checkpoint is missing its rank pattern metadata.")
    pattern = metadata["rank_pattern"]
    if pattern is not None:
        if not isinstance(pattern, dict):
            raise ValueError("AdaLoRA rank pattern must be a mapping.")
        expected_ranks = {
            name.replace(f".{adapter}", ""): p.shape[0]
            for name, p in method_parameters.items()
            if f"lora_E.{adapter}" in name
        }
        normalized = {
            name.replace(f".{adapter}", ""): mask for name, mask in pattern.items()
        }
        if set(normalized) != set(expected_ranks) or any(
            not isinstance(mask, list)
            or len(mask) != expected_ranks[name]
            or any(not isinstance(value, bool) for value in mask)
            for name, mask in normalized.items()
        ):
            raise ValueError(
                "AdaLoRA rank pattern is incompatible with full-rank parameters."
            )
    elif step >= config.total_step - config.tfinal or (
        step - step % config.deltaT > config.tinit
    ):
        raise ValueError("AdaLoRA checkpoint is missing its allocated rank pattern.")

    expected = {"parameter::" + name: p for name, p in parameters.items()}
    expected.update(
        {
            group + "::" + name: method_parameters[name]
            for group in _GROUPS
            for name in metadata[group]
        }
    )
    with safe_open(str(path / TENSORS_FILE), framework="pt", device="cpu") as stream:
        if (stream.metadata() or {}).get("global_step") != str(step) or set(
            stream.keys()
        ) != set(expected):
            raise ValueError("AdaLoRA tensor inventory or saved step differs.")
        tensors = {name: stream.get_tensor(name) for name in stream.keys()}
    for name, target in expected.items():
        saved = tensors[name]
        if (
            saved.shape != target.shape
            or saved.dtype != target.dtype
            or not torch.isfinite(saved).all()
        ):
            raise ValueError(f"AdaLoRA tensor shape, dtype or values differ: {name}.")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(tensors["parameter::" + name].to(parameter.device))
    config.rank_pattern = pattern
    for group in _GROUPS:
        setattr(
            allocator,
            group,
            {
                name: tensors[group + "::" + name].to(method_parameters[name].device)
                for name in metadata[group]
            },
        )
    trainer._adalora_resume_loaded = {
        "global_step": step,
        "parameter_objects_preserved": True,
    }
