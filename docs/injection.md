# Knowledge injection / 知识注入

[Documentation](README.md) · [Editing](editing.md) · [Unlearning](unlearning.md)

Use `operation=inject` to learn new information through supervised training.
Full FT, LoRA, DoRA, AdaLoRA and LoReFT share data loading, training loss and
evaluation. The current paper uses **WikiDYK and Qwen2.5-1.5B Base**, without
an instruction chat template. / 注入用于学习新知识；论文采用 Base 模型，不能换成 Instruct 模型后沿用结果。

## Prepare data

For custom QA, instructions or conversations, see [data adapters](components.md).
Keep training targets, held-out acquisition questions, paraphrases and unrelated
retention probes separate. Images use [image–text injection](components.md#imagetext-injection).

| Paper input | Content | Environment variable |
|---|---|---|
| Training | 12,290 facts with 51,957 retained `generated_qas`; no evaluation answers | `WIKIDYK_QA_TRAIN_PATH` |
| Evaluation | Frozen `eval` questions for acquisition, paraphrase and retention | `WIKIDYK_EVAL_PATH` |
| Base model | Original Qwen2.5-1.5B checkpoint and tokenizer | `WIKIDYK_BASE_MODEL` |

Training SHA256: `356b88d3bfc87d17da34cbd17993e146d3f8a3f06df90248ddcd6c08eda6ab2d`.
Evaluation SHA256: `2a80a018f03a126ef5c31fced1d4596e312018590ff4c018227753d67fbeed61`.
These identify prepared inputs; a raw dataset download does not recreate that
preparation. Weights and full datasets are supplied separately.

The dataset is registered as `WikiDYK_train_inject` with `WikiDYKDataset`;
`eval=wikidyk` selects the matching evaluator. CLI, WebUI and Agent Skills share
the five `builtin-wikidyk-<method>-qwen25-15b-base` recipes. Source and data terms:
[WikiDYK repository](https://github.com/zhang-yu-wei/WikiDYK) and
[dataset files](https://huggingface.co/datasets/YWZBrandon/wikidyk_exp/tree/main).
Download the JSON files explicitly; the dataset viewer's inferred split is not
this toolkit's prepared QA-only split.

## Select a method

Canonical recipes: `inject/wikidyk/{ft,lora,dora,adalora,loreft}`.
`inject/wikidyk/default` selects Full FT. All use QA-only supervision of answer
and EOS, token-normalized loss, seed 42, maximum length 256 and one epoch.
Each fact cycles through its training QA for 1,000 exposures: 12,290,000 examples,
48,008 optimizer updates at effective batch 256, including the final 208 examples.

| Method | Learning rate | Batch × accumulation, one GPU | Method settings |
|---|---:|---:|---|
| Full FT | `2e-5` | 128 × 2 | FP32 weights, BF16 autocast |
| LoRA | `1e-4` | 256 × 1 | rank 32, alpha 16, dropout 0 |
| DoRA | `1e-4` | 128 × 2 | rank 32, alpha 16, dropout 0 |
| AdaLoRA | `1e-4` | 256 × 1 | initial/target rank 48/32; initial/final phase 4,800 steps |
| LoReFT | `2e-4` | 256 × 1 | rank 8, blocks 0–27, last prompt token |

All use AdamW, linear scheduling, warmup ratio 0.05 and weight decay 0.
Adapters/interventions use FP32 over a BF16 base. Physical batches above come
from the A100 configuration, with gradient checkpointing disabled. On smaller
GPUs, reduce physical batch and increase accumulation while preserving
`batch × accumulation × GPU count = 256`; validate memory before a full run.
LoReFT requires the `[reft]` installation extra.

## Preview and run

```bash
export WIKIDYK_BASE_MODEL=/path/to/Qwen2.5-1.5B
export WIKIDYK_QA_TRAIN_PATH=/path/to/train_qa.json
export WIKIDYK_EVAL_PATH=/path/to/eval.json

know-surgery operation=inject experiment=inject/wikidyk/lora \
  task_name=wikidyk_lora --cfg job

know-surgery operation=inject experiment=inject/wikidyk/lora \
  task_name=wikidyk_lora --preflight
```

Review the resolved settings and inputs, then repeat with
`--reviewed-config YOUR_REVIEW_HASH` in place of `--preflight`.
For a short run, set `max_train_samples` and `trainer.args.max_steps` explicitly
and use a new task name; these limits define a different experiment.
See [configuration and execution](experiments.md).

## Evaluate the saved update

Training disables periodic evaluation. Evaluate saved updates using the same
questions and original checkpoint for every method:

```bash
python src/eval.py experiment=eval/inject/wikidyk \
  inject_model_path=/path/to/saved_update \
  task_name=wikidyk_lora_eval --cfg job
```

After checking the configuration, remove `--cfg job` to evaluate. The loader
restores full weights, PEFT adapters or LoReFT interventions. Keep the complete
output directory and original model available.

The paper uses greedy generation, at most 100 new tokens, no chat template and
the same `Answer:` suffix for Original and updated models. Select `toolkit` scores:

| Paper metric | Definition |
|---|---|
| Acquisition | Case-insensitive primary-answer substring match on original questions |
| Generalization | The same match on paraphrased questions |
| Relative Retention | Updated locality accuracy / Original locality accuracy |

Raw scores are fractions; multiply by 100 for percentages. Retention is a ratio;
also inspect the two absolute accuracies. The locality set has 8,500 rows but
only 40 distinct question/answer pairs with unequal multiplicities. The paper
preserves this row weighting; these are not 8,500 independent retained facts.
Native Match/F1 fields use different rules and must not silently replace the
paper columns. Runs save configurations, manifests, predictions and
`WIKIDYK_*SUMMARY.json`; see [evaluation](evaluation.md).
