"""Minimal LoReFT trainer integration for the inject pipeline."""

import json
import logging
import os
from typing import Any, List, Optional

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
        *args,
        **kwargs,
    ):
        self.layers = self._normalize_layers(layer)
        self.component = component
        self.low_rank_dimension = low_rank_dimension
        self.intervene_on_prompt = intervene_on_prompt
        super().__init__(*args, **kwargs)
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
        representations = []
        for layer in self.layers:
            representations.append(
                {
                    "layer": int(layer),
                    "component": self.component,
                    "low_rank_dimension": self.low_rank_dimension,
                    "intervention": pyreft_module.LoreftIntervention(
                        embed_dim=hidden_size,
                        low_rank_dimension=self.low_rank_dimension,
                        dtype=model_dtype,
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
        self.model_wrapped = self.model
        self.reft_config = reft_config

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        base_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }
        _, outputs = model(
            base=base_inputs,
            unit_locations=inputs.get("unit_locations"),
            labels=inputs.get("labels"),
            use_cache=False,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
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
            json.dump(
                {
                    "method": "loreft",
                    "base_model_name_or_path": base_model_name,
                    "layers": self.layers,
                    "component": self.component,
                    "low_rank_dimension": self.low_rank_dimension,
                    "intervene_on_prompt": self.intervene_on_prompt,
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
