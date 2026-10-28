# Editing methods

[Documentation](README.md) · [Configuration](experiments.md) · [Custom data](components.md) · [Evaluation](evaluation.md)

All commands below are run from the repository root. Replace `/path/to/...` with your own model and data paths. Start with `--cfg job`; remove it only after the inputs and method parameters are ready. Choose an available GPU with `CUDA_VISIBLE_DEVICES`.

Locality evaluation keeps a pristine reference model on CPU by default
(`edit_reference_device=cpu`). When the editing model uses FlashAttention, the CPU
copy switches to eager attention; the editing model retains its original attention
implementation. Models that cannot switch attention dynamically require a compatible
model implementation or an explicit CUDA reference device, such as
`edit_reference_device=cuda:1`.

Independent multimodal runs with `resume_independent=true` reuse cases only when
the configuration, loaded checkpoint, processor and request contents match
`independent_identity.json`. Image checks include pixels in edit, rephrase and
locality probes, so replacing an image or checkpoint at the same path invalidates
the resume. The checkpoint is hashed once per invocation, before editing. Use a
new output directory for changed inputs or older runs without this content identity;
their existing results are preserved.

## Choose a method

| Method / trainer | Required preparation | Typical use |
|---|---|---|
| `edit/FT` | Training budget and editable layers | Direct fine-tuning on edit targets |
| `edit/ROME` | Subject spans, one layer, preservation statistics | Factual weight edits |
| `edit/MEMIT` | Subject spans, multiple layers, statistics | Multi-request weight updates |
| `edit/AlphaEdit` | Statistics, null-space settings and continuation state | Projected weight updates |
| `edit/AnyEditOfficial` | Long targets, chunk settings and statistics | Joint MEMIT/AlphaEdit ARE updates |
| `edit/IKE_official` | Independent demonstration corpus and sentence encoder | One active edit through in-context demonstrations |
| `edit/GRACE` | Model-specific layer and adapter settings | Learned key/value adapter entries |
| `edit/WISE` | Model-specific projection and independent scope negatives | WISE-Merge side-memory editing |
| `edit/SERAC` | Trained scope encoder and counterfactual model archive | Retrieval and counterfactual routing |
| `edit/MEND`, `edit/MALMEN`, `edit/InstructEdit` | Matching auxiliary edit-network checkpoint | Compact auxiliary-network implementations |

The same dataset can be reused through its adapter. Supported modes and model mappings differ by method; a model that loads successfully is not automatically compatible with every editor. `WISEAdapted` and `SERACCompact` are separate alternative implementations and are not interchangeable with `WISE` and `SERAC` checkpoints.

## ROME, MEMIT and AlphaEdit

For new ROME/MEMIT reproductions, select `ROME_Official` or `MEMIT_Official`. These profiles implement the original authors' target optimization, context generation and update formulas. `AlphaEdit_Official` supplies the corresponding AlphaEdit profile. The default trainer configurations retain `implementation_variant=native_adapted` for existing experiments; select the reference profile explicitly. Model adapters remain necessary.

The complete GPT-2 and GPT-2-XL experiments are under `configs/experiment/edit/official/`:

```bash
know-surgery operation=edit experiment=edit/official/gpt2_rome \
  task_name=rome_reference max_edits=1 trainer.args.bf16=false \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  +data.edit.CounterFact_edit.args.data_path=/path/to/counterfact.json --cfg job
```

Change the experiment to `edit/official/gpt2_memit` or `edit/official/gpt2_alphaedit` for the corresponding method. Select `edit_type=batch` for one method call with multiple requests, or `edit_type=sequential` for cumulative updates. Reference profiles describe mechanisms, not a guarantee of published benchmark scores.

The `gpt2-xl_rome` and `gpt2-xl_memit` experiments load weights in FP32, matching the original authors' model-loading precision. Their smaller `gpt2_*` counterparts adapt the same algorithms to GPT-2. When a subject appears more than once, use a prompt such as `Paris and {} are different` with `subject=Paris` to identify the intended occurrence. The inserted subject is used consistently for target optimization and key extraction, including when generated context also mentions it. Plain prompts without `{}` retain first-occurrence selection.

CounterFact adapters retain this template automatically. Canonical records can also provide an expanded `prompt` plus a matching `prompt_template`; displayed questions and evaluation continue to use the expanded prompt.

### Qwen2.5-3B on zsRE

Use `edit/zsre/qwen25_3b/rome` or `edit/zsre/qwen25_3b/memit`. These existing recipe names now select `ROME_Official` and `MEMIT_Official` directly; they do not inherit the Qwen7B experiments. Leave `trainer` unset in direct CLI commands so the experiment selects its reference implementation. Skill and WebUI plans selecting ROME or MEMIT for this model/dataset use the same recipes and reference profiles.

Both recipes use the same Qwen2.5-3B-Instruct checkpoint, BF16/eager attention, seed 42, zsRE test split, 1,301 independent edits, and `edit_benchmark` evaluator (Reliability, Generalization, Locality and Portability). Each edit restores the original model. The editable layers remain ROME `[5]` and MEMIT `[4,5,6,7,8]`; the loss readout is configured at block 35 for Qwen3B's 36-block architecture. ROME evaluates final logits, while MEMIT uses the reference intermediate readout.

The reference profiles supply target optimization (20 steps, learning rate 0.5, weight decay 0.5), edited-to-base KL, generated contexts, and real Wikipedia `20200501.en` second moments from 100,000 sampled texts (statistics seed 1). MEMIT uses preservation weight 20,000 and clamp factor 0.75. These are the reference algorithms adapted to Qwen, not published Qwen3B benchmark scores from the original authors.

Preview the exact configuration before execution:

```bash
know-surgery operation=edit experiment=edit/zsre/qwen25_3b/rome \
  task_name=qwen3b_zsre_rome_reference \
  model.model_args.pretrained_model_name_or_path=/path/to/Qwen2.5-3B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/Qwen2.5-3B-Instruct \
  data.edit.ZSRE_edit.args.data_path=/path/to/zsre-test-1301.json \
  --cfg job
```

Use `edit/zsre/qwen25_3b/memit` and a new task name for MEMIT. The dataset must contain 1,301 normalized cases and match SHA256 `0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c`, the zsRE snapshot used by this benchmark. The loader rejects changed or short example files instead of silently evaluating fewer cases. To use another dataset, explicitly update `expected_sha256` and `expected_num_samples` and report it as a different setting.

Keep the same model and dataset snapshots across methods. Existing results retain their saved configuration; changing a recipe does not update their metrics. Report new results only after a complete run, using its saved `resolved_config.yaml`, input manifest and per-case metrics.

Important `trainer.method_args` settings:

| Field | Use |
|---|---|
| `layers` | Editable block indices; ROME uses one layer |
| `rewrite_module_tmp` | Projection to update, such as `transformer.h.{}.mlp.c_proj` for GPT-2 |
| `layer_module_tmp`, `v_loss_layer` | Block lookup and target-loss readout |
| `ln_f_module`, `lm_head_module` | Final normalization and output head |
| `mlp_module_tmp` | ROME's MLP intervention location |
| `fact_token` | Subject-token selection; the subject must occur in the prompt |
| `mom2_dataset`, `mom2_dataset_config` | Corpus used for preservation statistics |
| `mom2_n_samples`, `mom2_sample_seed`, `mom2_batch_tokens` | Statistics sample count, random seed and token budget in reference profiles |
| `stats_dir` | Writable statistics cache |
| `v_num_grad_steps`, `v_lr` | Target-vector optimization budget |
| `covariance_ridge` | ROME/MEMIT solve regularization |
| `nullspace_threshold`, `l2` | AlphaEdit projection and update regularization |

Reference rank-editor profiles default to Wikipedia `20200501.en` and collect real layer-input second moments. This can cost substantially more than the edits themselves. Prepare the dataset/cache and enough storage, or explicitly choose another available statistics corpus and budget. Changing them changes the experimental setting. Do not replace statistics with an identity matrix when interpreting method behavior.

For example, reference ROME/MEMIT/AlphaEdit accept these overrides for a smaller Wikitext statistics collection:

```text
trainer.method_args.mom2_dataset=wikitext
trainer.method_args.mom2_dataset_config=wikitext-103-raw-v1
trainer.method_args.mom2_n_samples=128
trainer.method_args.mom2_batch_tokens=512
trainer.method_args.stats_dir=data/stats
```

Cache identity includes the model, layer and statistics settings. Keep each experiment's generated context templates and state with the model. Reusing incompatible statistics or state is rejected.

## AnyEdit: long targets and joint batches

AnyEdit divides the target into token chunks, optimizes their boundary representations, then applies a joint layer update. Public backends are `memit_are` and `alphaedit_are`; select them with `trainer.method_args.strategy`.

```bash
know-surgery operation=edit experiment=edit/akew/gpt2_anyedit_joint_smoke \
  task_name=anyedit_example edit_type=batch max_edits=2 \
  trainer.method_args.strategy=memit_are \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  +data.edit.AKEW_counterfact_unstructured_edit.args.data_path=/path/to/CounterFact.json \
  --cfg job
```

This GPT-2 example deliberately uses a small optimization/statistics budget. Adjust it before a larger experiment. For the AlphaEdit backend, select `strategy=alphaedit_are` and a different task name.

`AnyEditOfficial` contains Qwen2.5-7B settings: edit layers `[4,5,6,7,8]`, loss layer `27`, window size `50` and `overlap=0`. Adapt module names and layer indices when using a different model. `window_size` and `overlap` count tokens; `target_prefix` controls the answer boundary and must match the tokenizer. `edit_type=batch` jointly solves requests, while `sequential` updates the model request by request.

AnyEdit's reference statistics use Wikipedia `20220301.en`, `mom2_sample_policy=upstream_random` and seed `1`. For Qwen2 the context cap is `4096`; `mom2_max_length` and `mom2_batch_tokens` can override the cap and collection budget. These settings differ from the ordinary rank-editor reference profiles.

## Prepare independent inputs for IKE, SERAC and WISE

Use independent auxiliary and held-out JSON files in CounterFact, zsRE or canonical format. Each demonstration needs a prompt, new answer, rephrase and locality question with an answer.

```bash
python scripts/prepare_memory_editor_data.py \
  --auxiliary /path/to/auxiliary.json --heldout /path/to/heldout.json \
  --output artifacts/memory_inputs \
  --train-count 64 --validation-count 16 --negative-count 8 --heldout-count 16
```

The command produces `ike_demonstrations.json`, `serac_train.json`, `serac_validation.json` and `heldout_edit.json`. It reserves the held-out records, separates auxiliary training/validation and WISE negative sources, and rejects overlapping or incomplete inputs. The requested counts must fit the eligible data. Exact matching cannot establish semantic independence; choose source splits carefully.

### IKE

`IKE_official` uses `protocol=paper_copy_update_retain`: 32 different demonstration facts, organized as copy/update/retain examples, plus the current edit. It requires at least 32 independent corpus entries and a local or accessible sentence-transformer model.

```bash
know-surgery operation=edit experiment=edit/zsre/ike_official \
  task_name=ike_example edit_type=independent max_edits=2 \
  model.model_args.pretrained_model_name_or_path=/path/to/Qwen2.5-1.5B \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/Qwen2.5-1.5B \
  trainer.method_args.sentence_model_name=/path/to/sentence_encoder \
  trainer.method_args.demonstration_path=artifacts/memory_inputs/ike_demonstrations.json \
  +data.edit.ZSRE_edit.args.data_path=artifacts/memory_inputs/heldout_edit.json --cfg job
```

Use a model with enough context for all demonstrations and the query; GPT-2's 1024-token window is usually insufficient. The reference protocol permits one active edit. Use `independent` to evaluate multiple edits rather than accumulating them. The native `edit/IKE` configuration uses an edit-memory protocol; it is a different setting.

### SERAC

Train the scope encoder and counterfactual model before online editing. Use a counterfactual model with the base model's tokenizer vocabulary. The following GPT-2 example first checks the input splits without training:

```bash
python scripts/train_serac.py \
  --model /path/to/gpt2 --counterfactual-model /path/to/gpt2 \
  --scope-model /path/to/roberta-base --local-files-only \
  --data artifacts/memory_inputs/serac_train.json \
  --validation artifacts/memory_inputs/serac_validation.json \
  --heldout artifacts/memory_inputs/heldout_edit.json \
  --max-records 64 --epochs 1 --device cuda:0 \
  --output saves/edit/serac_auxiliary --validate-only
```

Remove `--validate-only` to train. Tune the budget on the independent validation set; the example budget is not a trained model quality guarantee. Training writes `serac_auxiliary.pt`. Then load it through the editing entrypoint:

```bash
know-surgery operation=edit experiment=edit/zsre/serac_prepared \
  task_name=serac_example \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  trainer.method_args.archive=saves/edit/serac_auxiliary/serac_auxiliary.pt \
  +data.edit.ZSRE_edit.args.data_path=artifacts/memory_inputs/heldout_edit.json --cfg job
```

SERAC's online edit adds a cache entry; it does not train the frozen base model. The implementation provides supervised bi-encoder routing with a counterfactual causal LM, not every optional upstream architecture. Auxiliary training requires explicit `scope_training_prompts` and `scope_training_negatives`; do not use evaluation probes as training labels.

### WISE

`edit/WISE` selects WISE-Merge. Supply independent `scope_training_negatives`; unrelated evaluation probes must stay out of its routing objective. The preparation command above adds the required inputs to `heldout_edit.json`.

```bash
know-surgery operation=edit experiment=edit/zsre/wise_prepared \
  task_name=wise_example trainer.method_args.n_iter=40 \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  +data.edit.ZSRE_edit.args.data_path=artifacts/memory_inputs/heldout_edit.json --cfg job
```

Check `inner_params` for the chosen model. `save_freq`, `merge_freq`, `mask_ratio` and `merge_strategy` control side-memory updates and merging. WISE-Retrieve is not this configuration.

## Train MEND, MALMEN or InstructEdit auxiliary networks

These are compact native implementations. Their checkpoints must match the model, layers, rank and network dimensions; they are not interchangeable with arbitrary upstream archives. Use an auxiliary training split separate from evaluation data.

```bash
python scripts/train_mend_edit_network.py \
  --model /path/to/gpt2 --data /path/to/auxiliary_zsre.json \
  --output saves/edit/mend_network.pt \
  --layers 5 --rank 8 --n-hidden 16 --edit-lr 0.001 --max-edits 16 --epochs 3

know-surgery operation=edit experiment=edit/zsre/gpt2_mend_real \
  task_name=mend_example \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  trainer.method_args.edit_network_path=saves/edit/mend_network.pt \
  +data.edit.ZSRE_edit.args.data_path=/path/to/heldout_zsre.json --cfg job
```

For InstructEdit, add `--editor instructedit --task-type zsre` to the training command and select `trainer=edit/InstructEdit` with `trainer.method_args.default_task_type=zsre` at execution. Match its layers, rank, hidden size and learning rate to the trained network.

```bash
python scripts/train_malmen_auxiliary_network.py \
  --model /path/to/gpt2 --data /path/to/auxiliary_zsre.json \
  --output saves/edit/malmen_network.pt \
  --layers 5 --rank 16 --max-edits 16 --epochs 3

know-surgery operation=edit experiment=edit/zsre/gpt2_malmen_smoke \
  task_name=malmen_example \
  model.model_args.pretrained_model_name_or_path=/path/to/gpt2 \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/gpt2 \
  trainer.method_args.archive=saves/edit/malmen_network.pt \
  data.edit.ZSRE_edit.args.data_path=/path/to/heldout_zsre.json --cfg job
```

Keep `require_pretrained=true`; bypassing it does not replace training an auxiliary network.

## Saving and evaluation

Use `save_edited_model=true` when you need to restore a run; some example experiments disable saving by default. Keep the entire output directory, including edit-state files and metadata. IKE embeds its demonstration corpus in saved state but still needs sentence-encoder weights. SERAC needs compatible auxiliary networks; WISE needs its side-memory state. AlphaEdit/AnyEdit continuation requires matching method and statistics settings.

Use the project evaluator to restore runtime state and compare to the original model. Copying only Hugging Face weight files is insufficient for memory-based methods. See [evaluation](evaluation.md) for commands and metrics, and [third-party notices](../THIRD_PARTY_NOTICES.md) for method-specific terms.
