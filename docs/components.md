# Using your own models and data

[Documentation](README.md) · [Configuration](experiments.md)

Reuse an existing handler whenever it accepts your model or data format. Most custom inputs need only a YAML file or command override.

## Models

Select a model configuration with `model=<name>`, then override both weight and tokenizer paths:

```bash
know-surgery operation=edit experiment=edit/model_coverage/gpt2_ft_counterfact \
  task_name=local_model \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 --cfg job
```

For a new causal language model, create `configs/model/MyModel.yaml`:

```yaml
model_handler: AutoModelForCausalLM
model_args:
  pretrained_model_name_or_path: /path/to/model
  torch_dtype: bfloat16
  attn_implementation: eager
tokenizer_args:
  pretrained_model_name_or_path: /path/to/model
template_args:
  apply_chat_template: false
  user_start_tag: ""
  user_end_tag: ""
  asst_start_tag: ""
  asst_end_tag: ""
```

Choose precision supported by your device. Set chat formatting to match the model and task. An AutoModel-compatible model can load without necessarily supporting every editor: ROME/MEMIT/AlphaEdit/AnyEdit need explicit layer and projection mappings; LoRA needs valid target modules. See [editing configuration](editing.md).

## Local editing data

The `Canonical_edit` adapter reads JSON lists or JSONL records. Example record:

```json
{
  "id": "office-location",
  "prompt": "The office of Northstar is in",
  "subject": "Northstar",
  "target_new": "Paris",
  "target_old": "London",
  "rephrase_prompts": ["Where is Northstar's office?"],
  "locality_inputs": {
    "unrelated": [{"prompt": "The capital of Italy is", "ground_truth": "Rome"}]
  },
  "portability_inputs": {
    "one_hop": [{"prompt": "Northstar's office is in which country?", "ground_truth": "France"}]
  }
}
```

Supply a real subject span for subject-based editors. Provide probes separately from training data; do not invent missing answers. The example is an input illustration, not a factual benchmark.

```bash
know-surgery operation=edit model=gpt2 trainer=edit/FT \
  data/datasets@data.edit=Canonical_edit \
  data.edit.Canonical_edit.args.data_path=/path/to/edits.jsonl \
  task_name=custom_edit max_edits=1 trainer.args.bf16=false --cfg job
```

For original dataset formats, select the existing adapter instead, such as `CounterFact_edit` or `ZSRE_edit`, and provide its `args.data_path`. Keep source, version and split information with your data.

## Injection and unlearning data

Injection's `Canonical_inject` expects Alpaca-style JSON/JSONL:

```json
{"instruction": "Explain the office access rule.", "input": "", "output": "Visitors must sign in at reception."}
```

```bash
know-surgery operation=inject model=Qwen2.5-3B-Instruct trainer=inject/LoRA \
  data/datasets@data.train=Canonical_inject \
  data.train.Canonical_inject.args.hf_args.data_files=/path/to/train.jsonl \
  task_name=custom_inject --cfg job
```

For full supervised training, use `trainer=inject/base_inject`. Evaluation needs separate target and retention files; see [evaluation](evaluation.md).

The `Canonical_forget` and `Canonical_retain` adapters read question/answer records:

```json
{"question": "What information should this set contain?", "answer": "The corresponding target answer."}
```

```bash
know-surgery operation=unlearning model=Qwen2.5-3B-Instruct trainer=GradDiff \
  data/datasets@data.forget=Canonical_forget \
  data/datasets@data.retain=Canonical_retain \
  data.forget.Canonical_forget.args.hf_args.data_files=/path/to/forget.jsonl \
  data.retain.Canonical_retain.args.hf_args.data_files=/path/to/retain.jsonl \
  trainer.args.do_eval=false task_name=custom_unlearn --cfg job
```

This last command configures training only. Replace the default evaluation with a protocol appropriate to your custom data before evaluating. Forget and retain records must be disjoint.

## Normalize structured input

CLI and WebUI share a data contract for facts, events, procedures, rules and free text. JSON/JSONL/CSV/TXT can be normalized before building operation-specific cases:

```bash
python scripts/skill_runtime.py normalize --operation edit \
  --input /path/to/knowledge.jsonl --output artifacts/my_data/normalized.jsonl
python scripts/skill_runtime.py build-cases --operation edit \
  --input artifacts/my_data/normalized.jsonl --output artifacts/my_data/cases.jsonl
```

Structured metadata, sources, timestamps, conditions, steps and probes are retained where supplied. The required answer fields depend on the operation. TXT is split on blank lines; it does not infer semantic boundaries or automatically generate reliable evaluation answers. [Agent usage](agent_mode.md) shows how to bind the resulting cases to a run.

## New handlers and experiment recipes

| Component | Code registration | Configuration |
|---|---|---|
| Model | `src/model/__init__.py` | `configs/model/<name>.yaml` |
| Dataset | `src/data/__init__.py` | `configs/data/datasets/<name>.yaml` |
| Trainer / editor | `src/trainer/__init__.py` and method package | `configs/trainer/<name>.yaml` for unlearning; `inject/` or `edit/` subfolders for the other operations |
| Evaluator | `src/evals/__init__.py` | `configs/eval/<name>.yaml` |
| Complete task | Existing registered components | `configs/experiment/<operation>/<dataset>/<name>.yaml` |

A new editing method implements `EditTrainer.edit(requests)` and declares its model mapping, supported modes, required assets and state saving behavior. Keep request conversion in dataset adapters so different methods can use the same input.

Register curated model/data/method pairings in `configs/extensions/recipes.yaml` for discovery by the UI and Agent. The WebUI extension wizard exports configuration/scaffold bundles for review; it does not install uploaded executable code. Multimodal models also require a processor, valid images and method-specific token handling, so a text YAML alone is insufficient.
