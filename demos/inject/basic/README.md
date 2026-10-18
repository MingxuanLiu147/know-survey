# Injection examples

Small rule, procedure and retention examples are stored in `data/`.
Source revisions are recorded with the records. These files are illustrative
evaluation inputs; they do not replace a full benchmark or training split.

| File | Purpose |
|---|---|
| `rule_ruletaker.json` | Rule entailment questions |
| `rule_rulearena.json` | Airline baggage-rule questions |
| `procedure_wikihow.json` | Procedure summaries |
| `retention_benchmark.json` | Unrelated knowledge questions |

Preview a RuleTaker LoRA training configuration:

```bash
python src/surgery.py --config-name=surgery.yaml \
  operation=inject experiment=inject/ruletaker/default \
  trainer=inject/LoRA task_name=my_rule_injection --cfg job
```

Review the training dataset and model paths before removing `--cfg job` to run.
Use `experiment=inject/wikihow/default` for procedure data. For LoReFT, also set
`trainer=inject/LoReFT collator=DataCollatorForReFTDataset`.

Evaluate a saved model or adapter:

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/inject/default task_name=my_rule_eval \
  inject_model_path=saves/inject/my_rule_injection \
  eval_data_path=demos/inject/basic/data/rule_ruletaker.json \
  benchmark_data_path=demos/inject/basic/data/retention_benchmark.json \
  original_model_path=/path/to/base-model
```

Select the same `model` configuration used for training. The original model is
needed to measure a retention change; otherwise only the current score is
available. See [evaluation](../../../docs/evaluation.md) for metric configuration
and [experiments](../../../docs/experiments.md) for custom data.
