"""
Multimodal Editing Mixin
========================

Provides multimodal tokenization and forward-pass utilities for MM editing
methods.  All MM editors (MM-GRACE, MM-WISE, etc.) inherit from this mixin
alongside their text-only parent to gain image processing capabilities.

Target models: Qwen2-VL / Qwen2.5-VL / LLaVA-OneVision (HF Processor path).
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class MMEditMixin:
    """Mixin that injects multimodal capabilities into text-only editors.

    Expects the concrete class to expose ``self.model`` and sets
    ``self.processor`` during ``init_mm()``.
    """

    processor: Any = None

    _LANGUAGE_LAYER_PATHS = (
        "model.language_model.layers",  # Qwen3-VL
        "model.language_model.model.layers",  # wrapped LLaVA/Llama
        "model.language_model.model.decoder.layers",  # wrapped BLIP2-OPT
        "language_model.model.layers",  # LLaVA/Llama
        "language_model.model.decoder.layers",  # BLIP2-OPT
        "language_model.layers",
        "model.model.layers",
        "model.layers",  # Qwen2-VL/Qwen2.5-VL
    )
    _LANGUAGE_PROJECTIONS = (
        "mlp.down_proj",
        "mlp.c_proj",
        "mlp.fc2",
        "fc2",
        "mlp.dense_4h_to_h",
    )

    def init_mm(self, processor: Any) -> None:
        """Attach the multimodal processor (call once after __init__)."""
        if processor is None:
            raise ValueError("A multimodal processor is required")
        self.processor = processor

    def capture_mm_independent_state(self) -> Dict[str, Any]:
        """Capture the smallest state needed to reset one independent edit.

        Multimodal benchmarks evaluate every case from the same base model.
        Copying an entire 7B checkpoint before every case is prohibitively
        expensive, while the currently registered MM editors mutate much
        smaller state:

        * IKE replaces immutable memory-list/tensor references transactionally;
        * LoRA updates only ``lora_*`` parameters; and
        * GRACE/WISE install a removable adapter around one language layer.

        Unknown editors deliberately fall back to a full CPU state snapshot so
        correctness is preserved until they declare a narrower reset contract.
        """
        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError("Cannot snapshot an MM editor without a model")

        snapshot_runtime = getattr(self, "_snapshot_runtime_state", None)
        restore_runtime = getattr(self, "_restore_runtime_state", None)
        if callable(snapshot_runtime) and callable(restore_runtime):
            return {"mode": "runtime", "state": snapshot_runtime()}

        lora_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        }
        if lora_parameters:
            return {"mode": "parameters", "state": lora_parameters}

        if callable(getattr(self, "reset_layer", None)):
            return {"mode": "reset_layer", "state": None}

        return {
            "mode": "full_model",
            "state": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
        }

    def restore_mm_independent_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore a state produced by :meth:`capture_mm_independent_state`."""
        if not isinstance(snapshot, dict) or "mode" not in snapshot:
            raise TypeError("Invalid multimodal independent-edit snapshot")

        model = getattr(self, "model", None)
        if model is None:
            raise RuntimeError("Cannot restore an MM editor without a model")

        mode = snapshot["mode"]
        if mode == "runtime":
            self._restore_runtime_state(snapshot["state"])
        elif mode == "parameters":
            current = dict(model.named_parameters())
            expected = snapshot["state"]
            missing = sorted(set(expected) - set(current))
            if missing:
                raise RuntimeError(
                    "Cannot restore MM-LoRA parameters missing from the model: "
                    + ", ".join(missing)
                )
            with torch.no_grad():
                for name, value in expected.items():
                    parameter = current[name]
                    parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))
        elif mode == "reset_layer":
            self.reset_layer()
        elif mode == "full_model":
            reset_layer = getattr(self, "reset_layer", None)
            if callable(reset_layer):
                reset_layer()
            model.load_state_dict(snapshot["state"], strict=True)
        else:
            raise ValueError(f"Unsupported MM independent restore mode: {mode}")

        # Remove state that must never leak into the next independent case.
        if mode != "runtime":
            if hasattr(model, "_edit_runtime"):
                delattr(model, "_edit_runtime")
            reset_independent = getattr(self, "reset_independent_state", None)
            if callable(reset_independent):
                reset_independent()
        for attribute in ("edit_history", "_edit_history"):
            value = getattr(self, attribute, None)
            if hasattr(value, "clear"):
                value.clear()
        model.zero_grad(set_to_none=True)

    def configure_mm_target_layer(self) -> str:
        """Resolve ``inner_params: auto`` to a language-model projection.

        Vision encoders also contain transformer blocks.  The explicit path
        order below keeps GRACE/WISE on the language backbone for the model
        families covered by the multimodal configs.
        """
        if getattr(self, "inner_params", None) != "auto":
            return self.target_layer

        requested_layers = getattr(self, "layers", None) or [9]
        checked = []
        for container_path in self._LANGUAGE_LAYER_PATHS:
            try:
                layers = self._mm_get_module(self.model, container_path)
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            if not hasattr(layers, "__len__") or not len(layers):
                continue

            layer_idx = next(
                (idx for idx in requested_layers if 0 <= idx < len(layers)),
                len(layers) - 1,
            )
            for projection in self._LANGUAGE_PROJECTIONS:
                target_path = f"{container_path}.{layer_idx}.{projection}"
                checked.append(target_path)
                try:
                    target = self._mm_get_module(self.model, target_path)
                except (AttributeError, IndexError, KeyError, TypeError):
                    continue
                weight = getattr(target, "weight", None)
                if weight is None or weight.ndim != 2:
                    continue

                self.layers = [layer_idx]
                self.inner_params = target_path
                self.target_layer = target_path
                logger.info("Resolved multimodal edit target to %s", target_path)
                return target_path

        raise ValueError(
            "Cannot find a supported language projection for multimodal editing. "
            f"Checked: {', '.join(checked) or 'no language layer containers found'}"
        )

    def _run_mm_edit_requests(
        self,
        requests: Any,
        apply_edit: Callable[[Any], None],
    ) -> Dict[str, Any]:
        """Validate and apply MM edits without swallowing the first failure."""
        requests = requests if isinstance(requests, list) else [requests]
        if not requests:
            raise ValueError("At least one multimodal edit request is required")
        for request in requests:
            self._validate_mm_request(request)

        self._ensure_adapter()
        for request in requests:
            apply_edit(request)
        return {"success": True, "edited_count": len(requests), "metrics": {}}

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def mm_tokenize(
        self,
        prompts: Union[str, List[str]],
        images: Union[Any, List[Any]],
        targets: Union[str, List[str]],
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build model inputs from prompt + image + target.

        Uses ``processor.apply_chat_template`` to construct the chat text,
        appends the target answer, then calls the processor to encode
        everything.  Labels are masked so that only the target (answer)
        tokens contribute to the loss.

        Returns a dict with at least ``input_ids``, ``attention_mask``,
        ``pixel_values``, and ``labels``.
        """
        self._require_processor()

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(images, tuple):
            images = list(images)
        elif not isinstance(images, list):
            images = [images]
        if not (len(prompts) == len(images) == len(targets)):
            raise ValueError("prompts, images, and targets must have the same length")
        if any(image is None for image in images):
            raise ValueError("Every multimodal edit request must include an image")
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("Every multimodal edit request must include target text")

        if device is None:
            device = self._mm_device()

        texts: List[str] = []
        valid_images: List[Any] = []

        for prompt, image, target in zip(prompts, images, targets):
            user_content: list = [{"type": "image"}]
            valid_images.append(self._load_image(image))
            user_content.append({"type": "text", "text": prompt})

            messages = [
                {"role": "user", "content": user_content},
            ]
            text = self._render_prompt(messages, prompt)
            answer_text = self._answer_text(target)
            texts.append(text + answer_text)

        batch = self.processor(
            text=texts,
            images=valid_images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self._build_answer_labels(
            batch["input_ids"],
            targets,
            attention_mask=batch.get("attention_mask"),
        )
        batch["labels"] = labels

        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def mm_tokenize_text_only(
        self,
        prompts: Union[str, List[str]],
        targets: Union[str, List[str]],
        *,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text-only inputs through the processor (for locality tests)."""
        self._require_processor()
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(targets, str):
            targets = [targets]
        if len(prompts) != len(targets):
            raise ValueError("prompts and targets must have the same length")
        if device is None:
            device = self._mm_device()

        texts = []
        for prompt, target in zip(prompts, targets):
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            text = self._render_prompt(messages, prompt)
            answer_text = self._answer_text(target)
            texts.append(text + answer_text)

        batch = self.processor(
            text=texts,
            images=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self._build_answer_labels(
            batch["input_ids"],
            targets,
            attention_mask=batch.get("attention_mask"),
        )
        batch["labels"] = labels

        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def mm_forward(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Any:
        """Run a multimodal forward pass.  Filters out non-tensor keys."""
        tensor_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        return model(**tensor_inputs, **kwargs)

    def mm_loss(
        self,
        model: Any,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute cross-entropy loss on the answer tokens."""
        outputs = self.mm_forward(model, inputs)
        return outputs.loss

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: Any) -> Image.Image:
        """Accept PIL.Image, file path, or bytes; return RGB PIL.Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, bytes):
            return Image.open(BytesIO(image)).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_processor(self) -> None:
        if self.processor is None:
            raise RuntimeError("Call init_mm(processor) before multimodal tokenization")
        if getattr(self.processor, "tokenizer", None) is None:
            raise TypeError("The multimodal processor does not expose a tokenizer")

    def _render_prompt(self, messages: list[dict[str, Any]], prompt: str) -> str:
        """Use a chat template when available, else the processor's plain text path."""
        apply_template = getattr(self.processor, "apply_chat_template", None)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if callable(apply_template) and getattr(tokenizer, "chat_template", None):
            return apply_template(messages, add_generation_prompt=True, tokenize=False)
        return prompt

    @staticmethod
    def _answer_text(target: str) -> str:
        """Separate an answer from its prompt as in the official VLKEB wrapper."""
        return target if target[:1].isspace() else f" {target}"

    def _validate_mm_request(self, request: Any) -> None:
        self._require_processor()
        if request is None:
            raise TypeError("A multimodal edit request is required")
        if getattr(request, "image", None) is None:
            raise ValueError("Multimodal GRACE/WISE requires request.image")
        target = getattr(request, "target_new", None)
        if not isinstance(target, str) or not target:
            raise ValueError("Multimodal GRACE/WISE requires request.target_new")

    @staticmethod
    def _mm_get_module(root: Any, name: str) -> Any:
        module = root
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module

    def _mm_device(self) -> torch.device:
        """Best-effort device detection from ``self.model``."""
        model = getattr(self, "model", None)
        if model is None:
            return torch.device("cpu")
        if hasattr(model, "device"):
            return model.device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_answer_labels(
        self,
        input_ids: torch.Tensor,
        targets: List[str],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create labels that only supervise the answer span.

        Encodes each target string, finds its token ids at the *end* of the
        corresponding input_ids row, and masks everything else with -100.
        """
        labels = torch.full_like(input_ids, -100)

        for i, target in enumerate(targets):
            target_ids = self.processor.tokenizer(
                target, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0]

            seq_len = input_ids.size(1)
            tgt_len = len(target_ids)

            if tgt_len == 0:
                raise ValueError(f"Target tokenization is empty for sample {i}")
            if tgt_len > seq_len:
                raise ValueError(
                    f"Target token span is longer than encoded sample {i}; "
                    "increase the processor's maximum length"
                )

            matched = False
            for start in range(seq_len - tgt_len, -1, -1):
                if torch.equal(
                    input_ids[i, start : start + tgt_len],
                    target_ids.to(input_ids.device),
                ):
                    labels[i, start : start + tgt_len] = input_ids[
                        i, start : start + tgt_len
                    ]
                    matched = True
                    break

            if not matched:
                decoded_span = self._decoded_answer_span(
                    input_ids[i],
                    target,
                    target_token_count=tgt_len,
                    attention_mask=(
                        attention_mask[i] if attention_mask is not None else None
                    ),
                )
                if decoded_span is not None:
                    start, end = decoded_span
                    labels[i, start:end] = input_ids[i, start:end]
                    logger.warning(
                        "Recovered context-sensitive target boundary for sample %d "
                        "using decoded suffix validation (%d tokens).",
                        i,
                        end - start,
                    )
                    matched = True

            if not matched:
                raise ValueError(
                    f"Target token span was not found for sample {i}; "
                    "the answer may have been truncated or tokenized across a boundary"
                )

        return labels

    def _decoded_answer_span(
        self,
        row: torch.Tensor,
        target: str,
        *,
        target_token_count: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[tuple[int, int]]:
        """Recover a context-sensitive suffix only when it decodes exactly."""
        tokenizer = self.processor.tokenizer
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            return None

        if attention_mask is None:
            valid_positions = list(range(len(row)))
        else:
            valid_positions = attention_mask.nonzero(as_tuple=False).view(-1).tolist()
        if not valid_positions:
            return None

        valid_start = valid_positions[0]
        valid_end = valid_positions[-1] + 1
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        while valid_end > valid_start and int(row[valid_end - 1]) in special_ids:
            valid_end -= 1

        target_text = target.strip()
        search_start = max(valid_start, valid_end - target_token_count - 8)
        for start in range(valid_end - 1, search_start - 1, -1):
            candidate = row[start:valid_end].detach().cpu().tolist()
            decoded = str(decode(candidate, skip_special_tokens=True)).strip()
            if decoded == target_text:
                return start, valid_end
        return None
