---
name: add-multimodal-case
description: Validate local images referenced by JSON or JSONL knowledge cases before configuring a multimodal operation.
---

# Add Multimodal Case

Run `python .claude/skills/add-multimodal-case/scripts/run.py --input <records.jsonl> --output <validation.json>`.
Each record must contain `image` or `images`. Use absolute paths or paths relative
to the repository root. The helper checks files exist and can be decoded; it does
not judge image/text alignment or run the model. Empty input and corrupt images
return an error with the record index.

Review text targets, processor compatibility and image rights separately. Prepare
visual efficacy and text-preservation probes, then compose the matching `mm_*`
entrypoint. Do not include private images in a shared report.

Output defaults to `artifacts/add-multimodal-case/validation.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
