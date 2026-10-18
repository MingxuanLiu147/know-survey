---
name: reproduce-run
description: Recover an exact saved command and configuration references when inspecting or rerunning a previous Know-Surgery result.
---

# Reproduce Run

Run `python .claude/skills/reproduce-run/scripts/run.py --run-dir <run-dir> --output <reproduction.json>`.
A nonempty `command.txt` is required. The helper also references resolved config
and environment files when present, and does not execute the command.

Check the working directory, data/model versions and available resources before
rerunning. Preview the recovered command first when execution is authorized.
Changed input data or checkpoint constitutes a new experiment. Do not guess a
missing command from a folder name.

Output defaults to `artifacts/reproduce-run/reproduction.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
