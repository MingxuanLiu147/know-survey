#!/usr/bin/env python3
"""Dispatch a repository Skill to the deterministic shared runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path


COMMANDS: dict[str, list[str]] = {
    "setup-env": ["environment"],
    "inspect-repo": ["inspect"],
    "plan-surgery": ["plan"],
    "add-model": ["validate-config", "--kind", "model"],
    "add-dataset": ["validate-config", "--kind", "dataset"],
    "normalize-knowledge": ["normalize"],
    "build-cases": ["build-cases"],
    "add-multimodal-case": ["validate-mm"],
    "run-inject": ["run-operation", "--operation", "inject"],
    "run-unlearn": ["run-operation", "--operation", "unlearning"],
    "run-edit": ["run-operation", "--operation", "edit"],
    "smoke-test": ["smoke"],
    "batch-sweep": ["sweep"],
    "aggregate-results": ["aggregate"],
    "compare-methods": ["compare"],
    "generate-report": ["report"],
    "reproduce-run": ["reproduce"],
    "debug-ui": ["debug-ui"],
    "release-checklist": ["release"],
}


def main() -> int:
    invoked = Path(__file__).absolute()
    skill_name = invoked.parents[1].name
    if skill_name == "_shared":
        raise SystemExit("Invoke this wrapper through a Skill scripts/run.py link.")
    root = invoked.parents[4]
    command = COMMANDS.get(skill_name)
    if command is None:
        raise SystemExit(f"No runtime mapping for Skill: {skill_name}")
    if skill_name in {"run-edit", "run-inject", "run-unlearn"}:
        command = [*command, "--output", f"artifacts/{skill_name}/trace.json"]
    argv = [
        sys.executable,
        str(root / "scripts/skill_runtime.py"),
        *command,
        *sys.argv[1:],
    ]
    os.execv(sys.executable, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
