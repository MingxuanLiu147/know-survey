---
name: release-checklist
description: Check source entrypoints, model/data/trainer configs, installed dependencies and skill wrappers before sharing or using a checkout.
---

# Release Checklist

Run `python .claude/skills/release-checklist/scripts/run.py --output <checklist.json>`.
Inspect failed source/config, Python/dependency, license-file and skill-wrapper
checks. Optional UI and Codex discovery components are listed separately. The
helper does not require historical run manifests, a test directory, a paper,
cluster access, Docker or GPU hardware.

A passed result means this structural check passed. It does not certify licenses,
model compatibility or benchmark quality, and it does not upload anything. Use
`setup-env`, a compose preview and a bounded model run for the intended workflow.
Review data/model rights and private inputs before sharing actual artifacts.

Output defaults to `artifacts/release-checklist/checklist.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
