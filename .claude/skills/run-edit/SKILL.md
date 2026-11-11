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

Before execution, run the same selection with `--preflight`. Ask the user to
check or change the actual layer indices, module mappings, method parameters and
input paths. Forward their explicit confirmation with
`--execute --reviewed-config METHOD=HASH`, using the report's `review_hash`.
Never copy a preview hash into an execution command as automatic consent.
Changes to the model, method, data, auxiliary assets or parameters need a new review.

Output defaults to `artifacts/run-edit/trace.json`; actual model outputs follow the
resolved Hydra output directory. Check predictions and side-effect metrics, not
only the process exit code. A compose preview changes no model weights.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
