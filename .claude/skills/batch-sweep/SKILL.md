---
name: batch-sweep
description: Expand a plan into reproducible method-by-seed commands with separate output task names; this skill does not execute the matrix.
---

# Batch Sweep

Run `python .claude/skills/batch-sweep/scripts/run.py --plan <plan.json> --seeds 42,43,44 --output <sweep.json>`.
The helper validates registered components, deduplicates repeated seeds and writes
one pending command per method/seed with a reproducibility hash. Different seeds
receive different task names and explicit output directories, including
multimodal routes. It does not inspect prior results or mark runs done.

Review compatibility, input identity and total cost. Perform a representative
compose and bounded model check before executing a large matrix. Execution uses
the normal run path and requires the user's authorization; no cluster jobs are
submitted by this helper.

Output defaults to `artifacts/batch-sweep/sweep_manifest.json`. Preserve failed
and incomplete runs when comparing results.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
