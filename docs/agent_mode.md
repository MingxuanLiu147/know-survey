# Agent usage

[Documentation](README.md) · [Custom data](components.md)

An Agent can turn a natural-language request into a plan, configuration, data preparation steps and executable commands. Know-Surgery offers a built-in provider loop and shared workflows for external Codex/Claude Code agents.

## Built-in Agent

Configure one provider on the server using `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`, then start in preview mode:

```bash
know-surgery agent --root . --provider deepseek \
  --intent "Inspect recipes and preview one GPT-2 CounterFact edit with FT" \
  --max-steps 12 --max-cases 2 --timeout 600
```

Use `--provider openai` for OpenAI. Do not place API keys in prompts or data files. The WebUI Agent panel uses the same runtime and lets you inspect tool calls, commands, logs and results.

Preview is the default. Adding `--execute --cuda 0` enables bounded model execution after configuration preview and prerequisite checks. Choose an available GPU and prepared local inputs. The Agent's tools cover repository inspection, environment checks, planning, data normalization, case building, limited runs, aggregation and reporting; they do not execute arbitrary provider-generated shell commands.

A useful request names the operation, model, data, method, size and desired evaluation:

> Use my local GPT-2 model and CounterFact JSON. Preview one FT edit and report target accuracy, paraphrase accuracy and locality. Show the command before running.

Sessions are saved under `artifacts/agent/<session-id>/`. The trace contains tool decisions, observations and commands. Model and evaluation outputs remain in the task's `saves/` directory.

## Use the WebUI configuration

The WebUI has a configuration-bound Agent flow. Compose the task first, then enable
execution in the Agent panel. The session receives the same model, method, data,
seed, evaluation and limits as the Run button. Its execution tool takes no replacement
configuration from the provider. A run appears in the shared Runs page and retains
its own command, output directory and result ID.

These sessions use the server's device environment. Their default is preview only;
the CLI `--max-cases` default does not replace the WebUI configuration's budget.
If a local input file changes after preview, compose it again before execution.
The CLI's independent planning flow above remains available for requests that do
not start from a WebUI configuration.

## External coding agents

Open the repository in Codex or Claude Code. Shared workflows live in `.claude/skills/`; `.agents/skills` points to the same directory. Typical workflows are:

| Task | Skills |
|---|---|
| Choose a setup | `setup-env`, `inspect-repo`, `plan-surgery` |
| Prepare input | `normalize-knowledge`, `build-cases` |
| Add components | `add-model`, `add-dataset`, `add-multimodal-case` |
| Execute | `run-unlearn`, `run-inject`, `run-edit`, `smoke-test` |
| Compare and explain | `aggregate-results`, `compare-methods`, `generate-report` |
| Rerun | `reproduce-run` |

Skill wrappers call the same runtime used by the CLI and workbench. For example:

```bash
python .claude/skills/inspect-repo/scripts/run.py \
  --output artifacts/my_task/components.json
```

An external coding agent can help add configurations or prepare data; expensive runs still need an explicit task scope and appropriate resources.

## Bind your own data to execution

Normalize input, build cases, and bind those cases to the plan so the run uses the reviewed file:

```bash
python scripts/skill_runtime.py normalize --operation edit \
  --input /path/to/knowledge.jsonl --output artifacts/my_task/normalized.jsonl

python scripts/skill_runtime.py build-cases --operation edit \
  --input artifacts/my_task/normalized.jsonl --output artifacts/my_task/cases.jsonl

python scripts/skill_runtime.py plan \
  --intent "Edit one GPT-2 fact with FT and evaluate its side effects" \
  --operation edit --model-id gpt2 --dataset-id CounterFact_edit \
  --methods FT --num-cases 1 --cases artifacts/my_task/cases.jsonl \
  --output artifacts/my_task/plan.json

python scripts/skill_runtime.py smoke --plan artifacts/my_task/plan.json \
  --output artifacts/my_task/preview.json
```

Review the plan's model paths, method parameters, data files and evaluation settings. Add `--execute --cuda 0` to the last command for a limited run. `smoke` accepts at most two cases; it is a convenient first run before increasing the experiment size.

Input conversion preserves supplied source metadata, event times, rules, steps and probes. TXT splitting uses blank lines. Unlearning requires separate `forget` and `retain` records with answers. A source file changed after planning must be normalized and planned again.

## Results and reruns

Use the saved configuration and command to repeat a task, or ask the runtime to reconstruct them:

```bash
python scripts/skill_runtime.py reproduce --run-dir saves/edit/my_edit \
  --output artifacts/my_task/reproduction.json
```

Inspect per-example outputs as well as summaries. A completed process may still have weak target accuracy or large side effects. Missing evaluation inputs should be supplied before interpreting a run as a successful knowledge update. See [evaluation](evaluation.md).
