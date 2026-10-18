"""Experimental multimodal compact scope/prefix runtime, not full SERAC.

The shared SERAC implementation already provides external memory, learned
scope routing, archive loading, and prompt transformation.  This subclass
only attaches the multimodal processor and exposes the same runtime transform
to image-aware evaluators; it does not allocate or fine-tune a second model.
"""

from __future__ import annotations

from trainer.edit.mm_mixin import MMEditMixin
from trainer.edit.serac_compact import SERACCompactEditor


class MMSERACEditor(SERACCompactEditor, MMEditMixin):
    """Use SERAC's text scope router while retaining the evaluator's image."""

    def __init__(self, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)

    def transform_mm_prompt(self, prompt: str, image=None) -> str:
        """Route by text and leave the live image untouched for model input."""
        del image
        return self.transform_prompt(prompt)
