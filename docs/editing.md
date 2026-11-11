# Editing methods

[Documentation](README.md) · [Configuration](experiments.md) · [Custom data](components.md) · [Evaluation](evaluation.md)

All commands below are run from the repository root. Replace `/path/to/...` with your own model and data paths. Use `--cfg job` for composition, then `--preflight` to inspect inputs and parameters. Execute with `--reviewed-config YOUR_REVIEW_HASH` after review. Choose an available GPU with `CUDA_VISIBLE_DEVICES`.

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

### Qwen2.5-7B on zsRE: the paper configuration

The paper uses **Qwen2.5-7B-Instruct**, BF16/eager, seed 42 and 1,301 independent
edits. Each case starts from the original weights. Main recipes are under
`edit/zsre/qwen25_7b/`; the separate 3B recipes are model-size extensions, not
the source of the current table. / 正文采用 7B；3B 配方供扩展使用，不对应正文结果。

| Recipe | Implementation | Required resources |
|---|---|---|
| `ft` | Layer-restricted target-only FT | Layer 23, configured optimization budget |
| `rome` | `ROME_Official` | Layer 5; matching covariance statistics |
| `memit` | `MEMIT_Official` | Layers 4–8; matching covariance statistics |
| `alphaedit` | `AlphaEdit_Official` | Layers 4–8; covariance and null-space projection |
| `mend` | `MEND_official` factor network | Trained network for layers 25–27 and its metadata |
| `ike` | Semantic copy/update/retain demonstrations | Independent demonstration file and sentence encoder |
| `wise` | WISE-Merge side memory | Layer 23 and separate scope-training negatives |
| `anyedit` | `AnyEditOfficial` / MEMIT-ARE | Long-target settings and preservation statistics; full table result pending |

The seven completed methods above match the paper's method set. Auxiliary
weights and statistics are external inputs, not bundled model assets. A recipe
alone does not reproduce the table without the same input and auxiliary assets.
SERAC is available separately and is not a row in the current table.

The test JSON must contain 1,301 records with SHA256
`0e0214dda853a906ef02bef58c7ce2978af5a962e91fc4a10827e0125459905c`.
The loader checks both count and fingerprint; a small fixture is not a substitute.
Supply `data.edit.ZSRE_edit.args.data_path`, or configure `KNOW_SURGERY_DATA_ROOTS`.

```bash
know-surgery operation=edit experiment=edit/zsre/qwen25_7b/rome \
  model.model_args.pretrained_model_name_or_path=/path/to/Qwen2.5-7B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/Qwen2.5-7B-Instruct \
  data.edit.ZSRE_edit.args.data_path=/path/to/ZsRE-test-all.json \
  trainer.method_args.mom2_data_path=/path/to/wikipedia \
  trainer.method_args.stats_dir=/path/to/statistics \
  task_name=zsre_rome --cfg job
```

Run the same command with `--preflight`, inspect the layers, model metadata,
statistics and data, then replace it with `--reviewed-config YOUR_REVIEW_HASH`.
Changed model/method/input settings require another review. For Qwen7B, the
statistics-only attention path uses query chunks of 256 and a 32,768-token cap;
editing still uses eager attention. Prepare matching 100,000-document statistics
before execution. `scripts/prepare_rank_statistics.py --help` describes local
corpus inputs and resumable statistics collection. Different models and corpora
need different statistics; collection itself is a substantial computation.

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
know-surgery operation=edit experiment=edit/akew/anyedit \
  task_name=anyedit_example edit_type=batch max_edits=2 \
  trainer.method_args.strategy=memit_are \
  model.model_args.pretrained_model_name_or_path=/path/to/Qwen2.5-7B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=/path/to/Qwen2.5-7B-Instruct \
  +data.edit.AKEW_counterfact_unstructured_edit.args.data_path=/path/to/CounterFact.json \
  --cfg job
```

For the AlphaEdit backend, select `strategy=alphaedit_are` and a different task name.

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

## Auxiliary networks

The paper's MEND uses `edit/MEND_official`: activation/backpropagation factors,
FP32 auxiliary networks and a model-bound checkpoint. It is distinct from the
compact `edit/MEND` and `MM_MEND` paths. Use
`python scripts/train_mend_edit_network.py --help` for factor training, independent
validation, preview and full-state resume options. The exported network and its
`.metadata.json` must match the base checkpoint, edited matrices and method
settings. An untrained network or compact checkpoint is rejected.

### Compact MEND, MALMEN and InstructEdit

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

For MALMEN, train the auxiliary network with
`scripts/train_malmen_auxiliary_network.py --help`, then select
`trainer=edit/MALMEN` in a matching editing experiment. Set
`trainer.method_args.archive` to the resulting checkpoint and use the same
layers, module names, rank and block count as auxiliary training.

Keep `require_pretrained=true`; bypassing it does not replace training an auxiliary network.

## Saving and evaluation

Use `save_edited_model=true` when you need to restore a run; some example experiments disable saving by default. Keep the entire output directory, including edit-state files and metadata. IKE embeds its demonstration corpus in saved state but still needs sentence-encoder weights. SERAC needs compatible auxiliary networks; WISE needs its side-memory state. AlphaEdit/AnyEdit continuation requires matching method and statistics settings.

Use the project evaluator to restore runtime state and compare to the original model. Copying only Hugging Face weight files is insufficient for memory-based methods. See [evaluation](evaluation.md) for commands and metrics, and [third-party notices](../THIRD_PARTY_NOTICES.md) for method-specific terms.
