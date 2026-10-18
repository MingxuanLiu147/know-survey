---
name: build-cases
description: Convert normalized records into runnable edit, injection or forget/retain cases and count available evaluation probes.
---

# Build Cases

Run `python .claude/skills/build-cases/scripts/run.py --input <normalized.jsonl> --operation edit --output <cases.jsonl>`.
Choose the same operation used in normalization. Inspect the sibling
`.manifest.json` for case counts, missing probes and source identity. Unlearning
also writes separate `.forget.jsonl` and `.retain.jsonl` files.

Pass the built cases to `plan-surgery --cases <cases.jsonl>` to bind them to a run.
The helper preserves supplied rephrase/locality/portability probes; it does not
fabricate questions or reference answers. Missing side-effect probes remain
explicitly missing and must not become perfect scores.

Output defaults to `artifacts/build-cases/cases.jsonl`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
