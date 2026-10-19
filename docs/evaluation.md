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

Training and standalone evaluation share the same input resolver for
`eval_data` / `eval_data_path`, `benchmark_data` / `benchmark_data_path`, and
`original_model_path`. To evaluate during injection training, enable evaluation
and provide separate held-out target and retention inputs plus the original
checkpoint:

```bash
know-surgery operation=inject experiment=inject/ruletaker/default \
  trainer=inject/LoRA task_name=rules_lora \
  trainer.args.do_eval=true \
  eval.inject_metrics.task_accuracy.eval_data_path=/path/to/target_eval.json \
  eval.inject_metrics.knowledge_retention.benchmark_data_path=/path/to/retention_eval.json \
  eval.inject_metrics.knowledge_retention.original_model_path=/path/to/base_model --cfg job
```

Both evaluation inputs accept records containing a prompt/instruction and
expected output/answer. JSON, JSONL and YAML files are supported. You can also
evaluate a saved artifact in a separate process. The shared checkpoint loader
restores full models, PEFT adapters, text editing state and LoReFT interventions
as appropriate, preferring the tokenizer saved alongside the artifact:

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

`strict=true` rejects missing evaluation data/tokenizers and empty effective
evaluation sets. With
`strict=false`, unavailable metrics are saved as `null` with an
`unavailable_reason`; a measured accuracy of zero remains `0.0`. Use a new
evaluation `task_name` or set the metric's `overwrite=true` when inputs change.

Injection summaries use `protocol_version=inject-v2`. `knowledge_retention` is
the ratio `current_score / original_score` on the same retention inputs; the two
absolute scores are recorded separately. A ratio of 1 means equal accuracy and
a ratio above 1 means improved accuracy. If the original model is absent or its
score is zero, the ratio is `null` with a reason, even if `current_score` is
available. Old summaries are not automatically converted to this protocol.

For configured `data.eval` and `data.test` splits, `+max_eval_samples=N` controls
their sample limit separately from `+max_train_samples=N`. A training limit does
not truncate these held-out splits. Explicit evaluator files retain their own
input size and metric settings.

## Unlearning and downstream evaluation

Use a benchmark matching the training data, for example `eval=tofu` or `eval=muse`. Their YAML files under `configs/eval/` define metrics and reference inputs. Some metrics require logs from a retained-data reference model; provide `retain_logs_path` in the corresponding experiment rather than substituting an unrelated baseline.

```bash
python src/eval.py experiment=eval/tofu/default task_name=tofu_evaluation \
  model.model_args.pretrained_model_name_or_path=/path/to/unlearned_model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/tokenizer --cfg job
```

Install `.[lm-eval]` to use `eval=lm_eval` and select tasks in `configs/eval/lm_eval.yaml`. Keep the same task version, prompt formatting and sample budget before and after a knowledge update. Multimodal evaluation uses the matching processor and multimodal route.

## Multimodal edited artifacts

Multimodal memory and adapter restoration remains experimental. In particular,
MM-IKE, MM-GRACE, MM-WISE and MM-SERAC may write method-state files, but the
multimodal model loader does not yet reconstruct those editing runtimes from a
saved directory. Loading the model weights alone does not restore the edit.
Processor assets and the fixed MM-IKE context also need a multimodal artifact
contract. Keep the original inputs, processor, configuration and full output
directory; evaluate during the supported editing run until a save/reload path
has been verified for the exact method and model.

## Reading outputs

| Output | Content |
|---|---|
| `resolved_config.yaml`, `command.txt` | Parameters and invocation |
| `training_artifact.json` | Saved training artifact type, base model and inference restoration entrypoint |
| `input_manifest.json`, `environment.json` | Input identities and runtime information |
| `edit_requests_preview.json` | Requests used by the editor |
| `edit_results.json` | Editing mode and method results |
| `edit_inline_SUMMARY.json` | Inline text-edit metrics |
| `independent_SUMMARY.json` | Aggregation for edit/evaluate/restore mode |
| `history/step_*/` | Earlier-request evaluation during sequential editing |
| Evaluator-specific `*_EVAL.json`, `*_SUMMARY.json` | Detailed predictions and aggregate metrics |

Available files depend on the operation, evaluation and save settings. In sequential runs, growing-prefix metrics contain different numbers of requests; compare a fixed cohort to measure how earlier edits are retained.

`training_artifact.json` describes inference restoration; it does not promise
optimizer or training-step recovery. Keep the full output directory and any base
checkpoint required by an adapter or intervention.

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
