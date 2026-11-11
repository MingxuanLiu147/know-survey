"""Minimal LoReFT trainer integration for the inject pipeline."""

import json
import logging
import os
from typing import Any, List, Optional

import torch

from model.checkpoint import configure_loreft_precision
from model.reft_positions import (
    POSITION_CONTRACT_VERSION,
    build_loreft_unit_locations,
    loreft_position_config,
)
from trainer.inject.base import InjectTrainer

logger = logging.getLogger(__name__)


class LoReFTTrainer(InjectTrainer):
    """Train a low-rank ReFT intervention on top of the base causal LM."""

    def __init__(
        self,
        layer: Any = 8,
        component: str = "block_output",
        low_rank_dimension: int = 4,
        intervene_on_prompt: bool = True,
        trainable_dtype: str = "float32",
        last_k_prompt_tokens: int = 1,
        allow_legacy_ntp: bool = False,
        *args,
        **kwargs,
    ):
        if not intervene_on_prompt:
            raise ValueError(
                "LoReFT currently supports prompt-only interventions. "
                "Set trainer.method_args.intervene_on_prompt=true."
            )
        self.layers = self._normalize_layers(layer)
        self.component = component
        self.low_rank_dimension = low_rank_dimension
        self.intervene_on_prompt = intervene_on_prompt
        self.loreft_position_config = loreft_position_config(
            last_k_prompt_tokens, allow_legacy_ntp
        )
        if trainable_dtype not in {"float32", "model"}:
            raise ValueError("LoReFT trainable_dtype must be 'float32' or 'model'.")
        self.trainable_dtype = trainable_dtype
        self.precision_policy = (
            "fp32_intervention"
            if trainable_dtype == "float32"
            else "legacy_model_dtype"
        )
        super().__init__(*args, **kwargs)
        collator_config = getattr(self.data_collator, "loreft_position_config", None)
        if collator_config is not None and (
            collator_config != self.loreft_position_config
            or self.data_collator.num_interventions != len(self.layers)
        ):
            raise ValueError(
                "LoReFT trainer and collator position settings must agree."
            )
        if self.args.gradient_checkpointing:
            raise ValueError(
                "LoReFT's current PyReFT dynamic hooks do not support gradient "
                "checkpoint recomputation. Set trainer.args.gradient_checkpointing=false."
            )
        self._apply_loreft_config()

    @staticmethod
    def _normalize_layers(layer: Any) -> List[int]:
        if isinstance(layer, int):
            return [layer]
        return [int(value) for value in layer]

    def _get_hidden_size(self) -> int:
        if hasattr(self.model.config, "hidden_size"):
            return int(self.model.config.hidden_size)
        if hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config, "hidden_size"
        ):
            return int(self.model.config.text_config.hidden_size)
        raise AttributeError("Unable to infer hidden_size for LoReFT integration.")

    def _build_representations(self, pyreft_module):
        hidden_size = self._get_hidden_size()
        model_dtype = next(self.model.parameters()).dtype
        parameter_dtype = (
            torch.float32 if self.trainable_dtype == "float32" else model_dtype
        )
        representations = []
        for layer in self.layers:
            representations.append(
                {
                    "layer": int(layer),
                    "component": self.component,
                    "low_rank_dimension": self.low_rank_dimension,
                    "intervention": configure_loreft_precision(
                        pyreft_module.LoreftIntervention(
                            embed_dim=hidden_size,
                            low_rank_dimension=self.low_rank_dimension,
                            dtype=parameter_dtype,
                        ),
                        self.precision_policy,
                    ),
                }
            )
        if len(representations) == 1:
            return representations[0]
        return representations

    def _apply_loreft_config(self):
        try:
            import pyreft
        except ImportError:
            logger.error(
                "pyreft is required for LoReFT. Install it with pip install pyreft==0.0.7"
            )
            raise

        reft_config = pyreft.ReftConfig(
            representations=self._build_representations(pyreft)
        )
        self.model = pyreft.get_reft_model(self.model, reft_config)
        self.model._loreft_position_config = dict(self.loreft_position_config)
        self.model_wrapped = self.model
        self.reft_config = reft_config
        self._initial_intervention_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

        if hasattr(self.model, "print_trainable_parameters"):
            try:
                self.model.print_trainable_parameters()
            except Exception as exc:
                logger.warning(
                    "pyreft print_trainable_parameters() failed, continuing without it: %s",
                    exc,
                )
        logger.info(
            "LoReFT config applied: layers=%s component=%s low_rank_dimension=%s",
            self.layers,
            self.component,
            self.low_rank_dimension,
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        unit_locations = build_loreft_unit_locations(
            inputs.get("attention_mask"),
            inputs.get("prompt_length"),
            len(self.layers),
            labels=inputs["labels"],
            loss_group=inputs.get("loss_group"),
            **self.loreft_position_config,
        )
        supplied_locations = inputs.get("unit_locations", unit_locations)
        supplied_pair = supplied_locations.get("sources->base")
        # DataLoader's pin-memory traversal converts ordinary tuples to lists.
        # Compare the source/base pair by content while still rejecting any
        # changed indices, extra keys, or supervised-answer locations.
        if (
            set(supplied_locations) != {"sources->base"}
            or not isinstance(supplied_pair, (tuple, list))
            or tuple(supplied_pair) != unit_locations["sources->base"]
        ):
            raise ValueError(
                "LoReFT unit_locations disagree with the prompt position contract."
            )
        base_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }
        _, outputs = model(
            base=base_inputs,
            unit_locations=unit_locations,
            use_cache=False,
        )
        loss = self._compute_causal_loss(outputs.logits, inputs, num_items_in_batch)
        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        if not self.is_world_process_zero():
            return
        if output_dir is None:
            output_dir = self.args.output_dir
        output_dir = os.fspath(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir, include_model=False)
        base_model = getattr(getattr(self.model, "model", None), "config", None)
        base_model_name = getattr(base_model, "_name_or_path", None)
        if base_model_name is not None:
            base_model_name = str(base_model_name)

        with open(
            os.path.join(output_dir, "loreft_config.json"), "w", encoding="utf-8"
        ) as fout:
            delta_squared = sum(
                float(
                    (
                        parameter.detach().cpu().float()
                        - self._initial_intervention_state[name].float()
                    )
                    .square()
                    .sum()
                )
                for name, parameter in self.model.named_parameters()
                if name in self._initial_intervention_state
            )
            json.dump(
                {
                    "method": "loreft",
                    "base_model_name_or_path": base_model_name,
                    "layers": self.layers,
                    "component": self.component,
                    "low_rank_dimension": self.low_rank_dimension,
                    "intervene_on_prompt": self.intervene_on_prompt,
                    "trainable_dtype": self.trainable_dtype,
                    "precision_policy": self.precision_policy,
                    "trainable_parameter_dtypes": sorted(
                        {
                            str(parameter.dtype)
                            for parameter in self.model.parameters()
                            if parameter.requires_grad
                        }
                    ),
                    "position": "last_k_prompt_tokens",
                    "position_contract_version": POSITION_CONTRACT_VERSION,
                    **self.loreft_position_config,
                    "trainable_parameters": sum(
                        value.numel()
                        for value in self._initial_intervention_state.values()
                    ),
                    "intervention_delta_l2": delta_squared**0.5,
                    "intervention_weights_changed": delta_squared > 0,
                },
                fout,
                ensure_ascii=False,
                indent=2,
            )

        with open(
            os.path.join(output_dir, "inject_artifact.json"), "w", encoding="utf-8"
        ) as fout:
            json.dump(
                {
                    "method": "loreft",
                    "path": output_dir,
                    "base_model_name_or_path": base_model_name,
                },
                fout,
                ensure_ascii=False,
                indent=2,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
