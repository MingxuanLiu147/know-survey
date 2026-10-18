---
name: aggregate-results
description: Collect machine-readable metrics from existing run directories while keeping source references and unavailable values.
---

# Aggregate Results

Run `python .claude/skills/aggregate-results/scripts/run.py <run-dir> [<run-dir> ...] --output <aggregate.json>`.
Inspect each run's command, resolved config, input count, trace and metric files.
The helper reads supported summary formats and preserves null values; it does
not infer metrics from console logs or certify a complete benchmark.

Check scope and metric denominators before interpreting the aggregate. Use
`compare-methods` for comparability checks; do not mix smoke and full runs in a
claimed ranking. Missing run directories are errors.

Output defaults to `artifacts/aggregate-results/aggregate.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
