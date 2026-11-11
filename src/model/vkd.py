"""Visual-only adapters for VKD; language weights never belong to this module."""

from __future__ import annotations

import math

import torch
from torch import nn


class QwenVKDVisual:
    """Locate Qwen2/2.5-VL's encoder, merger and inner MLP projections.

    The visual forward returns projected tokens in image order (after Qwen2.5's
    window reordering), exactly the features passed into the language model.
    Qwen3-VL and other architectures require separate, verified adapters.
    """

    def __init__(self, model, layer_fraction: float):
        family = getattr(model.config, "model_type", None)
        if family not in {"qwen2_vl", "qwen2_5_vl"}:
            raise ValueError("MMVKD currently supports Qwen2-VL and Qwen2.5-VL only")
        if getattr(model, "peft_config", None):
            raise ValueError("MMVKD requires a merged model with LoRA disabled")
        if not 0 < layer_fraction <= 1:
            raise ValueError("vision_layer_fraction must be in (0, 1]")
        core = getattr(model, "model", model)
        self.visual = getattr(core, "visual", None)
        if self.visual is None:
            self.visual = getattr(model, "visual", None)
        if self.visual is None or not hasattr(self.visual, "merger"):
            raise ValueError("Cannot locate Qwen vision encoder and projector")
        blocks = self.visual.blocks
        start = len(blocks) - math.ceil(len(blocks) * layer_fraction)
        projection = "down_proj" if family == "qwen2_5_vl" else "fc2"
        self.mlps = {
            f"blocks.{i}.mlp": blocks[i].mlp for i in range(start, len(blocks))
        }
        self.projections = {
            f"{name}.{projection}": getattr(mlp, projection)
            for name, mlp in self.mlps.items()
        }
        if not self.projections or not all(
            isinstance(module, nn.Linear) for module in self.projections.values()
        ):
            raise ValueError("VKD requires plain linear vision MLP output projections")

    def masked_parameters(self):
        """Return only selected encoder MLP parameters, with visual-relative names."""
        return {
            f"{prefix}.{name}": parameter
            for prefix, mlp in self.mlps.items()
            for name, parameter in mlp.named_parameters()
        }

    @staticmethod
    def features(visual, batch):
        """Run a standalone visual teacher without allocating a second LLM."""
        dtype = next(visual.parameters()).dtype
        return visual(batch["pixel_values"].to(dtype), grid_thw=batch["image_grid_thw"])

    @staticmethod
    def examples(batch):
        """Split one-image-per-QA batches for per-example empirical Fisher scores.

        Qwen packs variable-sized image patches along dimension zero, so slicing
        pixel_values as though it had a batch dimension would corrupt the inputs.
        """
        batch = batch.get("multimodal", batch)
        required = {"input_ids", "labels", "pixel_values", "image_grid_thw"}
        if not required.issubset(batch) or "pixel_values_videos" in batch:
            raise ValueError("MMVKD requires image-text QA with labels, without video")
        count = batch["input_ids"].shape[0]
        grid = batch["image_grid_thw"]
        if grid.shape != (count, 3) or torch.any(grid <= 0):
            raise ValueError("MMVKD calibration requires exactly one image per QA")
        sizes = grid.prod(-1).tolist()
        if sum(sizes) != batch["pixel_values"].shape[0]:
            raise ValueError("Image patch count does not match image_grid_thw")
        offset = 0
        for i, size in enumerate(sizes):
            example = {}
            for key, value in batch.items():
                if key == "pixel_values":
                    example[key] = value[offset : offset + size]
                elif isinstance(value, torch.Tensor) and value.shape[0] == count:
                    example[key] = value[i : i + 1]
                else:
                    raise ValueError(f"Unsupported VKD calibration batch field: {key}")
            if not torch.any(example["labels"][:, 1:] != -100):
                raise ValueError("VKD calibration QA has no supervised answer tokens")
            yield example
            offset += size
