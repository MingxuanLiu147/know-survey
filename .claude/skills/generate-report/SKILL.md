---
name: generate-report
description: Render existing aggregate, comparison or trace JSON into Markdown and JSON for review and reproduction.
---

# Generate Report

Run `python .claude/skills/generate-report/scripts/run.py --input <aggregate.json> --markdown-output <report.md> --json-output <report.json>`.
The helper preserves the input payload and renders it in Markdown. It does not
invent an explanation, translate the report automatically or fetch missing runs.
Add a concise interpretation only after inspecting the actual input records.

Keep missing metrics as null/N/A and distinguish configuration previews, bounded
checks and completed protocols. Include available rerun commands and limitations.
These reports may contain local paths from the input; review or redact private
values before sharing them publicly.

Defaults: `artifacts/generate-report/report.md` and `report.json`.

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
