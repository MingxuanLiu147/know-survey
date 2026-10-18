---
name: add-model
description: Register or validate a model configuration when adding a local or Hugging Face model to Know-Surgery.
---

# Add Model

Inspect existing `configs/model/` entries and reuse their adapter structure.
Record weight/tokenizer sources, precision, architecture-specific target modules
and applicable model license in the configuration or user-facing documentation.
Prefer an existing local model supplied by the user over a duplicate download.

Run `python .claude/skills/add-model/scripts/run.py --path configs/model/<id>.yaml --output <validation.json>`.
This helper checks YAML is a mapping and reports its keys; it does not load the
model or prove required fields, licensing or algorithm compatibility. Review
`model_args`/`tokenizer_args`, then compose the intended operation. If execution is
authorized, run a bounded load/forward/generate check with `smoke-test`.

Keep configuration validity, model loading and editor compatibility separate.
Output defaults to `artifacts/config_validation.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
