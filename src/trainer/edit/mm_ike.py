"""
MM-IKE: Multimodal In-Context Knowledge Editing
================================================

Extends IKEEditor with multimodal capabilities.  Does not modify model
weights -- instead, stores multimodal edit facts and retrieves them as
ICL demonstrations during inference.

Reference:
  - IKE: https://arxiv.org/abs/2305.12740
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from trainer.edit.base import EditRequest
from trainer.edit.ike import IKEEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)

VLKEB_ICL_TEMPLATE = "New Fact: {fact}\nPrompt: {fact}\n\n"


class MMIKEEditor(IKEEditor, MMEditMixin):
    """Multimodal IKE editor.

    Stores multimodal edit facts (text descriptions of the image context)
    and retrieves them as ICL demonstrations.  The sentence-transformer
    encodes only the textual part of each fact for retrieval.
    """

    def __init__(
        self,
        processor=None,
        image_description_template: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)
        # VLKEB retrieves text-only demonstrations.  Keeping the question text
        # unchanged also makes the persisted MiniLM index match Appendix B.3.
        self.image_description_template = image_description_template or "{prompt}"
        self._active_mm_context: Optional[Dict[str, str]] = None

    def _snapshot_runtime_state(self) -> Dict[str, Any]:
        """Include the fixed single-edit ICL context in independent resets."""
        state = super()._snapshot_runtime_state()
        state["active_mm_context"] = self._active_mm_context
        return state

    def _restore_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        super()._restore_runtime_state(snapshot)
        self._active_mm_context = snapshot.get("active_mm_context")

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Store multimodal edits.

        For each request that carries an image, the fact string is
        augmented with an image description prefix so that retrieval can
        distinguish visual from text-only edits.
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "icl_examples": [],
        }

        # VLKEB single editing retrieves k demonstrations from the training
        # memory *before* the current fact is added, then fixes that context for
        # every probe of this case.  Dynamic per-probe retrieval can omit the
        # edited fact and is not the protocol reported in the paper.
        fixed_context = len(requests) == 1

        for request in requests:
            try:
                if request.image is not None:
                    enriched_prompt = self.image_description_template.format(
                        prompt=request.prompt
                    )
                    import dataclasses

                    enriched_request = dataclasses.replace(
                        request, prompt=enriched_prompt
                    )
                    retrieved = (
                        self.get_icl_prompt(enriched_prompt)
                        if self.knowledge_store
                        else ""
                    )
                    self._store_edit(enriched_request)
                else:
                    enriched_prompt = request.prompt
                    retrieved = (
                        self.get_icl_prompt(enriched_prompt)
                        if self.knowledge_store
                        else ""
                    )
                    self._store_edit(request)

                results["edited_count"] += 1
                fact = f"{enriched_prompt} {request.target_new}"
                template = getattr(self, "icl_prompt_template", VLKEB_ICL_TEMPLATE)
                current_fact = template.format(
                    fact=fact,
                    prompt=enriched_prompt,
                    target=request.target_new,
                )
                icl = f"{retrieved}{current_fact}"
                results["icl_examples"].append(icl)
                if fixed_context:
                    self._active_mm_context = {
                        "examples": icl,
                        "fact": fact,
                    }
            except Exception as e:
                logger.error("MM-IKE edit failed: %s", e)
                results["success"] = False

        # MM evaluators discover non-parametric editors through the same
        # lightweight runtime protocol used by text IKE.  No model weights are
        # copied or changed.
        self.model._edit_runtime = self
        return results

    def get_mm_icl_prompt(self, query_prompt: str, image=None) -> str:
        """Retrieve ICL prompt, optionally enriched with image context."""
        if image is not None:
            query = self.image_description_template.format(prompt=query_prompt)
        else:
            query = query_prompt
        return self.get_icl_prompt(query)

    def transform_mm_prompt(self, prompt: str, image=None) -> str:
        """Prepend the fixed VLKEB single-edit context during evaluation."""
        context = getattr(self, "_active_mm_context", None)
        if context:
            return (
                f"{context['examples']}"
                f"New Fact: {context['fact']}\nPrompt: {prompt}"
            )

        # Compatibility path for non-independent or legacy multi-edit callers.
        examples = self.get_mm_icl_prompt(prompt, image=image)
        if not examples:
            return prompt
        return f"{examples}Prompt: {prompt}\nAnswer:"
