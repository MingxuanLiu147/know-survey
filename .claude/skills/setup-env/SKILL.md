---
name: setup-env
description: Inspect the active Python environment, installed packages, visible GPUs and local model caches before execution or when diagnosing setup.
---

# Setup Env

Run `python .claude/skills/setup-env/scripts/run.py` from the source checkout.
Use `--model-cache /path/to/model` (repeatable) to inspect explicit local paths;
otherwise the helper uses the standard Hugging Face cache environment settings.
Use `--output <report.json>` to choose the report location.

Read package failures and `gpu_query` before choosing an execution device. A CPU
machine or missing `nvidia-smi` is recorded in the report and does not prevent
configuration planning. The helper inspects the environment; it installs nothing
and downloads no weights. Record the interpreter used for later commands.

Output: `artifacts/setup-env/environment_report.json`. This local diagnostic may
contain filesystem paths; review it before sharing. Package availability alone
does not establish model or editor compatibility.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
