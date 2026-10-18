---
name: smoke-test
description: Compose a plan and optionally run at most two cases to check data, model loading and output creation before a larger execution.
---

# Smoke Test

Run `python .claude/skills/smoke-test/scripts/run.py --plan <plan.json>` for
configuration-only validation. Optionally select `--method <id>`, `--output
<trace.json>` and `--timeout <seconds>`.

Inspect every resolved job. For an authorized bounded model run, add `--execute`
and optionally `--cuda <id>`. The helper limits a copy of the plan to two cases;
it does not authorize or launch the full matrix.

Output defaults to `artifacts/smoke-test/trace.json`. Read input counts, generated
files and failure messages. Compose evidence is `configured`; successful real
smoke evidence is `verified_smoke`, not a full-dataset result.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
