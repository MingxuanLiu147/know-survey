"""Checkpoint-gated instruction-conditioned compact MEND editor.

InstructEdit trains one learned editor across task-specific instructions.  The
public EasyEdit path wraps the primary edit prompt; this repository deliberately
extends instruction conditioning to rephrase and portability prompts so its
editing and evaluation protocol remains internally consistent.  That extension
is a Know-Surgery compact protocol, not an exact copy of EasyEdit's data flow.

Consequently this class is a compact representative path, not an official
InstructEdit checkpoint reproduction.  A dedicated InstructEdit checkpoint is
mandatory by default; plain MEND checkpoints are rejected so a randomly
initialised or differently trained edit network cannot be reported as an
InstructEdit result.

Paper: InstructEdit: Instruction-Based Knowledge Editing for Large Language
Models (https://arxiv.org/abs/2402.16123).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Union

from trainer.edit.base import EditRequest
from trainer.edit.mend import MENDEditor

logger = logging.getLogger(__name__)

DEFAULT_TASK_DESCRIPTIONS: Dict[str, List[str]] = {
    "counterfact": [
        "A benchmark for editing factual associations involving tail entities.",
        "Update a counterfactual subject-relation-object association while "
        "preserving unrelated knowledge.",
    ],
    "zsre": [
        "A question-answering task that predicts an object from a subject and "
        "relation without additional context.",
        "Edit the answer to a zero-shot relation question while preserving "
        "unrelated answers.",
    ],
    "wikirecent": [
        "A factual editing task built from recent Wikidata relations.",
        "Update a recently changed factual relation while preserving unrelated "
        "knowledge.",
    ],
    "convsent": [
        "Control the response sentiment for one topic without changing behavior "
        "on other topics.",
    ],
    "default": [
        "Apply the requested targeted knowledge edit while preserving unrelated "
        "model behavior.",
    ],
}

INSTRUCTION_TEMPLATE = "Task: {task_type}\nDescription: {description}\nInput: {prompt}"


class InstructEditEditor(MENDEditor):
    """Instruction-conditioned compact MEND editor with reproducible prompts."""

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        edit_lr: float = 1e-4,
        n_hidden: int = 128,
        rank: int = 64,
        task_descriptions: Optional[Dict[str, List[str]]] = None,
        default_task_type: str = "default",
        instruction_index: int = 0,
        *args,
        **kwargs,
    ):
        descriptions = task_descriptions or DEFAULT_TASK_DESCRIPTIONS
        self.task_descriptions = self._normalise_descriptions(descriptions)
        if default_task_type not in self.task_descriptions:
            raise ValueError(
                f"Unknown InstructEdit default_task_type {default_task_type!r}; "
                f"available={sorted(self.task_descriptions)}"
            )
        if instruction_index < 0:
            raise ValueError("InstructEdit instruction_index must be non-negative")
        self.default_task_type = default_task_type
        self.instruction_index = instruction_index
        super().__init__(
            layers=layers,
            edit_lr=edit_lr,
            n_hidden=n_hidden,
            rank=rank,
            *args,
            **kwargs,
        )

    @staticmethod
    def _normalise_descriptions(
        descriptions: Mapping[str, List[str]],
    ) -> Dict[str, List[str]]:
        normalised: Dict[str, List[str]] = {}
        for task_type, values in descriptions.items():
            if not isinstance(values, list):
                raise TypeError(
                    f"Descriptions for {task_type!r} must be a list of strings"
                )
            cleaned = [str(value).strip() for value in values if str(value).strip()]
            if not cleaned:
                raise ValueError(
                    f"Descriptions for InstructEdit task {task_type!r} are empty"
                )
            normalised[str(task_type)] = cleaned
        if "default" not in normalised:
            raise ValueError("InstructEdit task descriptions must define 'default'")
        return normalised

    # ------------------------------------------------------------------
    # Dedicated checkpoint identity
    # ------------------------------------------------------------------

    def _checkpoint_editor_type(self) -> str:
        return "instructedit"

    def _checkpoint_editor_metadata(self) -> Dict[str, Any]:
        return {
            "instruction_template": INSTRUCTION_TEMPLATE,
            "task_descriptions": self.task_descriptions,
            "default_task_type": self.default_task_type,
            "instruction_index": self.instruction_index,
        }

    def _validate_checkpoint_identity(self, payload: Mapping[str, Any]) -> None:
        # Reject a preprocessing-contract mismatch before streaming a potentially
        # large base model through the exact-weight hash. Cross-editor checkpoints
        # still defer to MEND so they retain the more specific editor_type error.
        if payload.get("editor_type") == self._checkpoint_editor_type():
            expected = self._checkpoint_editor_metadata()
            actual = payload.get("editor_metadata")
            if actual != expected:
                raise ValueError(
                    "InstructEdit checkpoint instruction metadata does not match the "
                    "current template/task descriptions/index. Reuse the training "
                    "configuration or pretrain a matching checkpoint."
                )
        super()._validate_checkpoint_identity(payload)

    # ------------------------------------------------------------------
    # Instruction-conditioned execution and training
    # ------------------------------------------------------------------

    def prepare_requests(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        task_type: Optional[str] = None,
    ) -> List[EditRequest]:
        """Return deterministically instruction-wrapped edit/evaluation requests."""

        if isinstance(requests, EditRequest):
            requests = [requests]
        prepared = []
        for request in requests:
            resolved_type = task_type or self._infer_task_type(request)
            prepared.append(self._wrap_with_instruction(request, resolved_type))
        return prepared

    def prepare_evaluation_requests(
        self, requests: Union[EditRequest, List[EditRequest]]
    ) -> List[EditRequest]:
        """Pipeline hook: evaluation must use the same instruction contract."""

        return self.prepare_requests(requests)

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        task_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        wrapped_requests = self.prepare_requests(requests, task_type=task_type)
        logger.info(
            "InstructEdit prepared %d instruction-conditioned requests",
            len(wrapped_requests),
        )
        result = super().edit(wrapped_requests, **kwargs)
        result["implementation"] = "compact_native_instructedit"
        result["instruction_index"] = self.instruction_index
        return result

    def train_edit_network(
        self,
        train_requests: List[EditRequest],
        num_epochs: int = 10,
        batch_size: int = 4,
        lr: Optional[float] = None,
        task_type: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """Train compact MEND on deterministically instruction-wrapped cases."""

        return super().train_edit_network(
            self.prepare_requests(train_requests, task_type=task_type),
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Prompt/probe preparation
    # ------------------------------------------------------------------

    def _wrap_with_instruction(
        self, request: EditRequest, task_type: str
    ) -> EditRequest:
        if task_type not in self.task_descriptions:
            task_type = self.default_task_type
        description = self._select_description(task_type)

        def wrap(prompt: Any) -> str:
            return INSTRUCTION_TEMPLATE.format(
                task_type=task_type,
                description=description,
                prompt=str(prompt),
            )

        rephrases = (
            [wrap(prompt) for prompt in request.rephrase_prompts]
            if request.rephrase_prompts
            else None
        )
        portability = self._wrap_probe_payload(request.portability_inputs, wrapper=wrap)
        return replace(
            request,
            prompt=wrap(request.prompt),
            rephrase_prompts=rephrases,
            portability_inputs=portability,
        )

    def _wrap_probe_payload(self, payload: Any, *, wrapper) -> Any:
        """Wrap every portability ``prompt`` while preserving nested schema."""

        if payload is None:
            return None
        cloned = copy.deepcopy(payload)
        if isinstance(cloned, list):
            return [self._wrap_probe_payload(item, wrapper=wrapper) for item in cloned]
        if isinstance(cloned, dict):
            for key, value in list(cloned.items()):
                if key == "prompt":
                    if isinstance(value, list):
                        cloned[key] = [wrapper(prompt) for prompt in value]
                    else:
                        cloned[key] = wrapper(value)
                elif isinstance(value, (dict, list)):
                    cloned[key] = self._wrap_probe_payload(value, wrapper=wrapper)
        return cloned

    def _select_description(self, task_type: str) -> str:
        descriptions = self.task_descriptions.get(
            task_type, self.task_descriptions[self.default_task_type]
        )
        return descriptions[self.instruction_index % len(descriptions)]

    def _infer_task_type(self, request: EditRequest) -> str:
        # The benchmark task is a run-level setting in official InstructEdit.
        # Avoid content heuristics that silently vary preprocessing case by case.
        del request
        return self.default_task_type

    def register_task_type(self, task_type: str, descriptions: List[str]) -> None:
        """Register preprocessing for future training; loaded checkpoints reject it."""

        normalised = self._normalise_descriptions(
            {"default": self.task_descriptions["default"], task_type: descriptions}
        )
        self.task_descriptions[task_type] = normalised[task_type]
        self._pretrained_loaded = False
        self.edit_network_path = None
        logger.info(
            "InstructEdit registered task type %s with %d descriptions; a new "
            "checkpoint is required before editing",
            task_type,
            len(descriptions),
        )
