---
name: run-edit
description: Preview or execute a registered knowledge editing plan through the unified surgery entrypoint.
---

# Run Edit

Preview with `python .claude/skills/run-edit/scripts/run.py --plan <plan.json>`.
The plan must use `operation=edit`. Optionally select `--method <id>` and use
`--output <trace.json>` or `--timeout <seconds>`.

Inspect the composed model, dataset, edit count and evaluator settings. Verify
model-specific module mappings, statistics and auxiliary checkpoints required by
the method. With user authorization, add `--execute` and optionally `--cuda <id>`;
CPU-compatible workflows do not require a CUDA argument.

Output defaults to `artifacts/run-edit/trace.json`; actual model outputs follow the
resolved Hydra output directory. Check predictions and side-effect metrics, not
only the process exit code. A compose preview changes no model weights.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
