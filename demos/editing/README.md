# Editing examples

Each directory contains one illustrative JSONL input:

| Directory | Request |
|---|---|
| `replacement/` | Replace an existing association |
| `addition/` | Add an association through an editor |
| `absence/` | Rewrite an answer toward absence |

All three use `operation=edit`. These are request semantics, not additional
public operations; the absence example does not establish model unlearning.
The inputs include held-out probes and contain no precomputed results.

Preview the GPT-2 + FT example:

```bash
python src/surgery.py --config-name=surgery.yaml \
  operation=edit experiment=examples/edit_replacement_intent_demo --cfg job
```

For the other inputs, replace `replacement` with `addition` or `absence` in the
experiment name. To execute, remove `--cfg job` and supply your checkpoint:

```bash
python src/surgery.py --config-name=surgery.yaml \
  operation=edit experiment=examples/edit_replacement_intent_demo \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  task_name=my_first_edit
```

Inspect the generated predictions and locality metrics under
`saves/edit/my_first_edit/`. See [editing](../../docs/editing.md) for other methods
and their required assets.
