---
name: normalize-knowledge
description: Convert JSON, JSONL, CSV or TXT inputs to operation-aware knowledge records before case construction or custom uploads.
---

# Normalize Knowledge

Run `python .claude/skills/normalize-knowledge/scripts/run.py --input <file> --operation edit --knowledge-type fact --output <normalized.jsonl>`.
`--operation` accepts `unlearning`, `inject` or `edit`. For unlearning, preserve
`dataset_role` in each record or use `--role forget` / `--role retain` on separate
inputs. Injection uses the train role; editing uses the edit role.

Read the accepted JSONL, sibling `.rejected.json` and `.manifest.json`. A nonzero
exit indicates rejected or empty input; fix the reported records before running.
TXT import splits blank-line paragraphs only. It does not infer missing targets
or perform semantic knowledge extraction. Keep provenance and stable IDs.

Without `--output`, files go under `artifacts/normalize-knowledge/`.
Never invent targets, images, conditions or dates to make validation pass.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
