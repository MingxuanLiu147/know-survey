# Hydra configuration

Know-Surgery composes YAML files from `configs/`. The public entrypoint selects
one of three operations: `unlearning`, `inject` or `edit`.

```bash
python src/surgery.py --config-name=surgery.yaml \
  operation=edit experiment=examples/edit_replacement_intent_demo --cfg job
```

`--cfg job` prints the combined configuration without loading model weights.
Remove it to execute after reviewing the inputs and output directory.

## Select components

- `model=gpt2` selects `configs/model/gpt2.yaml`.
- `trainer=edit/FT` selects `configs/trainer/edit/FT.yaml`.
- `experiment=examples/edit_replacement_intent_demo` selects the example above.
- `task_name=my_edit` names the run and its output directory.

Experiments use `defaults` to select components and nested mappings to override
their parameters. For example:

```yaml
# @package _global_
defaults:
  - override /model: gpt2
  - override /trainer: edit/FT
  - override /data: edit
  - override /eval: edit

trainer:
  method_args:
    layers: [4, 5]
    num_steps: 10
task_name: my_edit
```

Supply an operation-appropriate dataset as shown in [experiments](experiments.md).
Inspect the composed tree before changing a nested key: different operations
have different data and evaluation structures.

## Override values

Use dot notation to replace an existing value:

```bash
python src/surgery.py --config-name=surgery.yaml \
  operation=edit experiment=examples/edit_replacement_intent_demo \
  model.model_args.pretrained_model_name_or_path=/path/to/model \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/model \
  trainer.method_args.num_steps=5 task_name=my_edit --cfg job
```

Use `+key=value` to introduce a new key; use `'~key'` to remove one. Quote values
containing spaces, lists or shell-sensitive characters. `${name}` in YAML
references another configuration value; `???` marks a required value that the
caller must supply. Model and tokenizer paths should refer to the same checkpoint.

See [components](components.md) for adding configurations and
[evaluation](evaluation.md) for evaluator overrides.
