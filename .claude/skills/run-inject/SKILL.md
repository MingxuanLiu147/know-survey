---
name: run-inject
description: Preview or execute a registered knowledge injection plan through the unified surgery entrypoint.
---

# Run Inject

Preview with `python .claude/skills/run-inject/scripts/run.py --plan <plan.json>`.
The plan must use `operation=inject`. Optional arguments are `--method <id>`,
`--output <trace.json>`, `--timeout <seconds>` and `--cuda <id>`.

Inspect the training data, model, adapter, sample budget and retention evaluation.
Image-text plans use `modality=multimodal`, `inject/MM_LoRA`, and the canonical
multimodal data route; bound train cases become `data.train_path`. Text defaults
are unchanged. Check target and retain probe availability in the actual output;
training completion alone does not establish retention. For image-text plans,
`overrides.data.retain_path` may point to a separate local probe file; the runtime
records the consumed input hashes. Training-set before/after probes are bounded
diagnostics, not a held-out benchmark.
Add `--execute` only when the user's request authorizes execution; a CUDA argument
is optional. Check local resource availability before a model run.

Output defaults to `artifacts/run-inject/trace.json`; training artifacts follow
the resolved Hydra directory. Report target improvement and retention separately,
and preserve missing metrics as missing. A successful exit alone is not a
completed evaluation.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
