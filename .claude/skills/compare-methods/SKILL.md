---
name: compare-methods
description: Compare metrics from existing run directories and identify differences in data, model, seed, protocol or hardware before ranking.
---

# Compare Methods

Run `python .claude/skills/compare-methods/scripts/run.py <run-dir> <run-dir> --output <comparison.json>`.
Inputs are run directories, not an aggregate JSON file. The helper returns metric
rows and checks the recorded fairness fields. If any required identity is missing
or differs, `comparable` remains false with the field-level reasons.
Empty data hashes are unknown. Completed runs also need common metrics with
recorded, matching scoring semantics. Noncomparable rows remain available for
inspection; identical folder names do not establish identical inputs.

Review denominators and metric directions before ranking. Runtime comparisons
need matching hardware and work scope. The helper writes comparison data; charts
and a claimed ranking require a separate explicit interpretation.

Output defaults to `artifacts/compare-methods/comparison.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
