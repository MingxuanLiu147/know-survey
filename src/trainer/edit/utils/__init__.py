"""Shared editing instrumentation and representation utilities.

Instrumentation derives from the MIT-licensed ROME/MEMIT projects. Value
optimisation and second-moment collection were rebuilt from the pinned MIT
MEMIT source; see THIRD_PARTY_NOTICES.md and LICENSES/MEMIT.txt.
"""

from trainer.edit.utils.nethook import (
    StopForward,
    Trace,
    TraceDict,
    get_module,
    get_parameter,
    recursive_copy,
    replace_module,
    set_requires_grad,
)

__all__ = [
    "Trace",
    "TraceDict",
    "StopForward",
    "get_module",
    "get_parameter",
    "replace_module",
    "set_requires_grad",
    "recursive_copy",
]
