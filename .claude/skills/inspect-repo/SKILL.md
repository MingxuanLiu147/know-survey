---
name: inspect-repo
description: List the source checkout's entrypoints, registered models, datasets, trainers, recipes and skills before choosing or extending a workflow.
---

# Inspect Repo

Run `python .claude/skills/inspect-repo/scripts/run.py --output <inventory.json>`.
Without `--output`, the inventory is written to
`artifacts/inspect-repo/repo_inventory.json`.

Use the registry IDs in this output when creating a plan. Inspect the matching
`configs/` files and implementation when a combination is uncertain. Inventory
counts describe configuration coverage; they do not prove a model run succeeded.
Do not modify source or existing run outputs while taking inventory.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
