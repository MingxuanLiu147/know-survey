---
name: run-unlearn
description: Preview or execute an unlearning plan with explicit forget and retain inputs through the unified surgery entrypoint.
---

# Run Unlearn

Preview with `python .claude/skills/run-unlearn/scripts/run.py --plan <plan.json>`.
The plan must use `operation=unlearning`. Optional arguments are `--method <id>`,
`--output <trace.json>`, `--timeout <seconds>` and `--cuda <id>`.

Check that forget and retain scopes are disjoint and the intended loss, checkpoint
and evaluation are composed. Add `--execute` only for an authorized run; CPU paths
do not require a CUDA argument.

First use `--preflight` and have the user check the current model, layer/method
settings, forget/retain data and budget. Pass their explicit confirmation as
`--execute --reviewed-config METHOD=HASH`. Never auto-confirm a report; changing
the configuration or inputs requires another review.

Output defaults to `artifacts/run-unlearn/trace.json`; model artifacts follow the
resolved Hydra directory. Examine forget efficacy and retained utility together.
Reduced answer quality is not a guarantee of complete knowledge removal. Missing
retention measurements must remain unavailable.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
