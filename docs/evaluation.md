# Evaluation and results

[Documentation](README.md) · [Configuration](experiments.md)

Evaluate both the requested knowledge change and behavior that should remain unchanged. Keep the input split, starting model, seed, method budget and metric settings fixed when comparing methods.

## Editing metrics

`eval=edit` uses `EditSuiteEvaluator`. `eval=edit_benchmark` selects a smaller suite containing reliability, generalization, locality and portability.

| Metric | Meaning | Required input |
|---|---|---|
| Reliability | Teacher-forced accuracy on target tokens | Edit prompt and new answer |
| Generalization | Target accuracy on paraphrased prompts | Independent rephrases |
| Locality | Agreement with the original model on unrelated prompts | Locality probes and a pre-edit model |
| Portability | Accuracy on related questions | Questions with expected answers |
| Long-form generation | Generated-answer exact match, F1, ROUGE-L and optional semantic similarity | Reference answers and generation limits |

Teacher-forced accuracy is different from free generation. A high reliability value does not imply that a model produces the entire new answer by itself. Missing probes produce unavailable values rather than perfect scores. Check per-example predictions and denominators alongside the summary.

During inline text-edit evaluation, the pipeline keeps a pre-edit reference model for locality. `edit_reference_device=cpu` is the default and can reduce GPU memory use. Standalone evaluation must receive the original checkpoint explicitly:

```bash
python src/eval.py model=gpt2 eval=edit task_name=reevaluate_edit \
  model.model_args.pretrained_model_name_or_path=/path/to/saved_edit \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  eval.edit.edit_data_path=/path/to/edit_requests_preview.json \
  eval.edit.original_model_path=/path/to/gpt2 --cfg job
```

Use the normalized request file from the run, not an arbitrary original dataset schema. The saved edit directory must include the method's state and runtime metadata. Keep required auxiliary encoder/model weights available when restoring IKE, SERAC or WISE.

## Injection evaluation

Use separate data to measure target acquisition and retained knowledge. Both accept JSON records containing a prompt/instruction and expected output/answer. The standalone evaluator reloads full models, PEFT adapters and LoReFT state as appropriate:

```bash
python src/eval.py model=Qwen2.5-3B-Instruct eval=inject \
  task_name=inject_evaluation \
  model.model_args.pretrained_model_name_or_path=/path/to/saved_injection \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/base_model \
  eval.inject_metrics.task_accuracy.eval_data_path=/path/to/target_eval.json \
  eval.inject_metrics.knowledge_retention.benchmark_data_path=/path/to/retention_eval.json \
  eval.inject_metrics.knowledge_retention.original_model_path=/path/to/base_model --cfg job
```

Remove `--cfg job` after checking inputs. Do not use training examples as held-out acquisition questions. Keep the base model accessible for adapter loading and before/after retention comparison.

## Unlearning and downstream evaluation

Use a benchmark matching the training data, for example `eval=tofu` or `eval=muse`. Their YAML files under `configs/eval/` define metrics and reference inputs. Some metrics require logs from a retained-data reference model; provide `retain_logs_path` in the corresponding experiment rather than substituting an unrelated baseline.

```bash
python src/eval.py experiment=eval/tofu/default task_name=tofu_evaluation \
  model.model_args.pretrained_model_name_or_path=/path/to/unlearned_model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/tokenizer --cfg job
```

Install `.[lm-eval]` to use `eval=lm_eval` and select tasks in `configs/eval/lm_eval.yaml`. Keep the same task version, prompt formatting and sample budget before and after a knowledge update. Multimodal evaluation uses the matching processor and multimodal route.

## Reading outputs

| Output | Content |
|---|---|
| `resolved_config.yaml`, `command.txt` | Parameters and invocation |
| `input_manifest.json`, `environment.json` | Input identities and runtime information |
| `edit_requests_preview.json` | Requests used by the editor |
| `edit_results.json` | Editing mode and method results |
| `edit_inline_SUMMARY.json` | Inline text-edit metrics |
| `independent_SUMMARY.json` | Aggregation for edit/evaluate/restore mode |
| `history/step_*/` | Earlier-request evaluation during sequential editing |
| Evaluator-specific `*_EVAL.json`, `*_SUMMARY.json` | Detailed predictions and aggregate metrics |

Available files depend on the operation, evaluation and save settings. In sequential runs, growing-prefix metrics contain different numbers of requests; compare a fixed cohort to measure how earlier edits are retained.

For runs with complete supported summaries:

```bash
python scripts/skill_runtime.py aggregate saves/edit/run_a saves/edit/run_b \
  --output artifacts/comparison/aggregate.json
python scripts/skill_runtime.py report --input artifacts/comparison/aggregate.json \
  --markdown-output artifacts/comparison/report.md \
  --json-output artifacts/comparison/report.json
```

Aggregation preserves missing values. It does not make runs with different models, inputs or budgets directly comparable.

## Adding a metric

Implement the evaluator under `src/evals/`, register its handler and add a YAML configuration under `configs/eval/`. State the score direction, denominator, aggregation rule and behavior for absent probes. Preserve per-example results so users can inspect why a summary changed. Existing TOFU/MUSE metrics also support `pre_compute` dependencies for reusing intermediate metric outputs.
