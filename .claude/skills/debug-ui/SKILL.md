---
name: debug-ui
description: Check a running Know-Surgery backend health endpoint and guide focused UI/CLI behavior checks after interface changes.
---

# Debug Ui

Run `python .claude/skills/debug-ui/scripts/run.py --url http://127.0.0.1:7860 --output <health.json>`.
Use the actual backend URL; the helper requests `/api/v1/health`. It reports HTTP
health only, not browser usability or successful model execution.

For a UI change, additionally exercise the affected control and compare the
exported command with the CLI preview. Check language, validation messages and
result rendering where relevant. Do not launch training just to check layout.
Keep provider keys out of logs, browser exports and reports.

Output defaults to `artifacts/debug-ui/health.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
