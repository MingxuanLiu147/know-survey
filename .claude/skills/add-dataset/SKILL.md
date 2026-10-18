---
name: add-dataset
description: Register or validate a dataset configuration and map its records to the shared data contract for an intended operation.
---

# Add Dataset

Inspect existing `src/data/` handlers and `configs/data/datasets/` configs before
adding an adapter. Record source, license, version and split with the data or
configuration; do not redistribute restricted raw records.

Run `python .claude/skills/add-dataset/scripts/run.py --path configs/data/datasets/<id>.yaml --output <validation.json>`.
This helper checks YAML is a mapping, not provenance or runtime correctness.
Review the handler and arguments, normalize a small permitted sample, build cases,
and compose the intended operation to check the integration.

Reuse canonical adapters where the semantics match. Preserve rejected rows and
never replace missing targets, timestamps or conditions with fabricated values.
Output defaults to `artifacts/config_validation.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
