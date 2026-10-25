# Configuring and running tasks

[Documentation](README.md) · [Models and data](components.md) · [Editing methods](editing.md)

Run commands from the repository root after installation. The common form is:

```bash
know-surgery operation=edit experiment=edit/model_coverage/gpt2_ft_counterfact \
  task_name=my_edit --cfg job
```

`--cfg job` prints the configuration and exits without loading a model. Remove it to execute after preparing the selected model, data and method assets. Choose a new `task_name` for each run.

## Choose an operation and experiment

| Operation | Base configuration | Example experiment |
|---|---|---|
| `unlearning` | `configs/unlearn.yaml` | `unlearn/tofu/default` |
| `inject` | `configs/inject.yaml` | `inject/ruletaker/default` |
| `edit` | `configs/edit.yaml` | `edit/model_coverage/gpt2_ft_counterfact` |

Experiment IDs are paths below `configs/experiment/`, without `.yaml`. A complete recipe chooses model, trainer, datasets and evaluation. Use `configs/extensions/recipes.yaml` to find curated combinations.

```bash
# TOFU: start from a checkpoint trained on the full knowledge set.
know-surgery operation=unlearning experiment=unlearn/tofu/default \
  trainer=NPO forget_split=forget10 retain_split=retain90 \
  task_name=tofu_npo --cfg job

# RuleTaker: supervised injection using LoRA.
know-surgery operation=inject experiment=inject/ruletaker/default \
  trainer=inject/LoRA task_name=rules_lora --cfg job

# ROME: GPT-2 model mapping with reference-method settings.
know-surgery operation=edit experiment=edit/official/gpt2_rome \
  task_name=rome_edit max_edits=1 --cfg job
```

For unlearning, use disjoint forget and retain data and the correct starting checkpoint. Some TOFU/MUSE metrics also require reference logs. Injection recipes can disable evaluation during training; evaluate the saved model with explicit target and retention inputs as described in [evaluation](evaluation.md).

## Override parameters

| Setting | Meaning |
|---|---|
| `model` | Select a file under `configs/model/` |
| `model.model_args.pretrained_model_name_or_path` | Model ID or local weights |
| `model.tokenizer_args.pretrained_model_name_or_path` | Matching tokenizer |
| `trainer` | Method configuration, such as `NPO`, `inject/LoRA`, `edit/MEMIT` |
| `trainer.args` | Training/runtime arguments: batch size, epochs, precision, seed, output |
| `trainer.method_args` | Method parameters: edit layers, adapter rank, statistics, checkpoints |
| `data/datasets@data.edit` | Dataset configuration assigned to the edit role |
| `eval` | Evaluation configuration, such as `edit` or `edit_benchmark` |
| `max_edits` | Limit editing requests |
| `+max_train_samples` | Limit training examples for injection/unlearning |
| `paths.output_dir` | Output directory shared by training and evaluation |

Override existing keys with `key=value`; add missing keys with `+key=value`. Use quotes around list values, for example `'trainer.method_args.layers=[3,4,5]'`. A method/model mapping is more than a weight path: changing architecture may require different layers, modules, tokenizer handling and statistics.

Local weights example:

```bash
know-surgery operation=edit experiment=edit/model_coverage/gpt2_ft_counterfact \
  task_name=local_edit \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  +model.model_args.local_files_only=true \
  +model.tokenizer_args.local_files_only=true \
  +data.edit.CounterFact_edit.args.data_path=/path/to/counterfact.json --cfg job
```

`local_files_only` covers model/tokenizer loading. Methods that collect preservation statistics may still need a locally cached dataset. `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` disable remote lookup when all required assets are already available.

## Editing modes and saved outputs

| `edit_type` | Use |
|---|---|
| `single` | One edit |
| `batch` | One method call with a list of requests |
| `sequential` | Accumulate edits on the updated model |
| `independent` | Edit/evaluate/restore each request separately |

The selected method determines whether a batch uses a joint solve. Reference IKE supports one active edit; use `independent` for multiple evaluation cases.

For sequential editing, evaluate earlier requests at chosen steps:

```bash
know-surgery operation=edit experiment=edit/official/gpt2_alphaedit \
  task_name=sequential_edits edit_type=sequential max_edits=100 \
  'edit_history_eval_steps=[1,20,50,100]' --cfg job
```

Set `trainer.args.do_eval=true` to enable these evaluations. Results appear under `history/step_*/`. Compare the same request cohort across steps when measuring retention.

Default output directories are `saves/unlearn/<task_name>/`, `saves/inject/<task_name>/` and `saves/edit/<task_name>/`. The public operation `unlearning` uses the internal directory name `unlearn`. `save_edited_model=false` skips the final model save, while summaries can still be written.

## Image–text tasks

Use `modality=multimodal` with a compatible editing, injection or unlearning configuration. Multimodal configurations use `model.model_path` and `model.processor_path`, rather than the text model's `model_args`/`tokenizer_args` paths:

```bash
know-surgery operation=edit modality=multimodal \
  experiment=edit/multimodal/mmedit_grace_smoke task_name=image_edit \
  model.model_path=/path/to/Qwen2-VL-2B-Instruct \
  model.processor_path=/path/to/Qwen2-VL-2B-Instruct \
  data.edit.MMEdit_VQA.args.data_path=/path/to/vqa_eval.json \
  data.edit.MMEdit_VQA.args.image_dir=/path/to/images \
  data.edit.MMEdit_VQA.args.rephrase_image_dir=/path/to/rephrase_images --cfg job
```

This example uses a one-step GRACE budget. Supply readable images and the correct processor, then adjust the method budget for the intended task. Image–text methods are experimental and compatibility depends on the exact model and dataset. For LoRA injection, use `operation=inject modality=multimodal` with local training images and targets; see the [data and execution example](components.md#imagetext-injection).

Unlearning experiments are under `configs/experiment/unlearn/multimodal/`. MLLMU-Bench/CLEAR require the corresponding full-data-trained starting checkpoint; methods may additionally need a retain-only reference or a teacher. `MMUnlearner` requires checkpoint-bound gradient masks: prepare them with `scripts/run_mm_generate_mask.py` and set `trainer.method_args.grad_mask_path`. Use that script's `--help` for the benchmark, model family, dataset and output arguments. Generating masks can require substantially more memory than LoRA training.

## Multiple GPUs

Use a single process for editing. Model sharding via `model.model_args.device_map` is method-dependent and is different from distributed training.

Training methods can use `torchrun`; choose devices, a suitable memory budget and a supported training setup:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  src/surgery.py operation=inject experiment=inject/ruletaker/default \
  trainer=inject/LoRA task_name=rules_lora_ddp \
  trainer.args.do_eval=false
```

Do not combine distributed data parallel training with `device_map=auto`. Evaluate saved checkpoints separately in one process. Merely listing multiple visible GPUs does not launch distributed workers.
