---
name: plan-surgery
description: Turn a natural-language unlearning, injection or editing request into an editable plan using registered model, dataset and method IDs.
---

# Plan Surgery

Run `python .claude/skills/plan-surgery/scripts/run.py --intent "Edit one CounterFact fact with GPT2 and FT"`.
Optional overrides are `--operation`, `--model-id`, `--dataset-id`, `--methods FT,ROME`,
`--num-cases` and `--output <plan.json>`. Public operations are exactly
`unlearning`, `inject` and `edit`.

For custom data, normalize and build cases first, then pass `--cases <cases.jsonl>`.
This binds role-specific inputs and their hashes to the plan. Unlearning needs
both forget and retain records. Rank editors need explicit subjects.

Output defaults to `artifacts/plan-surgery/surgery_plan.json`; review its methods,
experiment recipes, sample budget, inputs and warnings. A missing exact recipe
needs a checked compose preview, not an invented registration. The helper only
plans. Use the appropriate run skill to preview, and execute only when the user's
request authorizes the actual run. Existing authorization need not be requested again.

The plan schema is [surgery-plan.schema.json](../_shared/references/surgery-plan.schema.json).

For output conventions and evidence labels, see
[artifact layout](../_shared/references/artifact-layout.md) and
[evidence levels](../_shared/references/evidence-levels.md).
