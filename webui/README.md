# WebUI

The workbench uses FastAPI and React, with the same surgery commands and
configurations as the CLI and Agent.

## Start the workbench

From the repository root, install the Python dependencies and build the frontend:

```bash
pip install -e '.[webui]'
cd webui/frontend
npm ci
npm run build
cd ../..
python -m webui.backend.app --host 127.0.0.1 --port 17860
```

Open http://127.0.0.1:17860. Use the language button to switch English/Chinese.

1. **Workbench**: describe the desired knowledge change, generate a plan or
   select a recipe, then review the configuration and equivalent command.
2. **Runs**: follow the process log, stop an active task and open its linked result.
3. **Compare**: compare completed runs, check their evaluation settings and inspect
   the saved inputs, reference answers and model outputs.
4. **Extensions**: browse registered models, datasets, methods and Skills.

The current task stays in memory when you move between these pages. Reloading the
browser starts a new draft. Changing the configuration invalidates its previous
preview; preview again before running. Repeated submission of the same request
returns the existing run. To repeat a completed configuration, choose a new task
name and preview it again.

The workbench uses the server's device environment. To select devices, set
`CUDA_VISIBLE_DEVICES` when starting the backend; there is no GPU management panel.
Preview composes the configuration without loading model weights. Execution needs
the configured inputs, dependencies and compute. The one-edit/one-step switch
sets exactly that limit; turn it off to use the configuration's full budget.

Text tasks open with a complete example request and the matching recipe. The
one-edit/one-step limit starts off, so preview uses the recipe's stated budget.
The examples describe the input data, update method, evaluation and saved outputs;
changing the operation loads its corresponding defaults.

- **Unlearning** — `unlearn/tofu/gpt2_quickstart`: GPT-2 with GradAscent for
  20 training steps. The local example has 20 forget and 20 retain questions;
  both sets are evaluated for target-answer probability.
- **Injection** — `inject/ruletaker/workbench`: Qwen2.5-1.5B (base) with LoRA
  for 20 training steps on 40 RuleTaker training records, with 20 separate test questions and five
  unrelated factual probes for comparison with the base model. The trained
  adapter, configuration and evaluation outputs are saved.
- **Editing** — `edit/counterfact/workbench`: GPT-2 with FT on four CounterFact
  requests, up to 25 update steps per request. Each edit starts from the same
  original weights. Target reliability, paraphrase generalization and locality
  are evaluated separately for every request; no combined edited model is saved.

Prepare the referenced local training data and model weights before execution.
The stated record counts describe these examples, not the full source datasets.
Long-text preset buttons also apply their registered recipe and parameters.

## Configure a task

The dataset and evaluation choices follow the selected operation. Unlearning uses
separate forget and retain scopes. Preview also checks known model-specific module
mappings, required editor checkpoints, and explicitly configured local inputs.
A valid preview checks these prerequisites; model loading and execution still
appear as separate run states.

For standard Transformers models, preview resolves a Hugging Face model ID and
revision to a complete local cache snapshot. The displayed command and execution
both use that directory with local loading enabled, avoiding online metadata
requests. Missing or incomplete local weights stop the task before execution;
download the checkpoint separately or provide its local path. The model registry
lists configurations, not downloaded weights.

Set `HF_HOME` or `HF_HUB_CACHE` in the backend's environment to use an existing
Hugging Face cache. A background service needs these settings in its own service
configuration; it does not inherit variables from a later terminal session.
`KNOW_SURGERY_MODEL_CACHE_ROOTS` optionally adds cache roots separated by the
platform path separator. Each cache must retain its repository and revision layout.

AnyEdit's MEMIT strategy also needs the configured statistics corpus when
`mom2_adjustment=true`. Preview checks the exact Wikipedia or WikiText variant
in the server's datasets cache (`HF_DATASETS_CACHE`, or `HF_HOME/datasets`).
Prepare that corpus before starting the task. Precomputed covariance files are
optional: the method can calculate them from the corpus using its configured
budget. `max_edits` limits knowledge-edit requests, not the number of corpus rows
used to estimate covariance; `mom2_n_samples` controls that separate budget.

The `edit/unke/anyedit` recipe uses Wikipedia `20200501.en` for covariance
statistics: 200 rows, at most 512 tokens per row and 512 padded tokens per batch.
Its editing inputs remain UnKEBench; the model remains Qwen2.5-7B-Instruct, with
covariance adjustment enabled on all five configured layers. These settings are
visible in the resolved configuration. Official-reference recipes retain their
own corpora and sampling budgets. Prepared statistics corpora are read directly
from the local datasets cache, without an online metadata check. This recipe uses
PyTorch SDPA attention and does not require FlashAttention.

Open **Parameters and local paths** to edit individual Hydra overrides as a JSON
object, for example:

```json
{
  "trainer.args.learning_rate": 0.0001,
  "model.model_args.pretrained_model_name_or_path": "/path/to/model",
  "model.tokenizer_args.pretrained_model_name_or_path": "/path/to/model"
}
```

Apply the parameters and preview again before execution. Unapplied edits disable
preview and execution; **Restore applied parameters** returns to the previous
configuration. Operation, model/trainer selection, task name, seed and output
location remain managed by the task fields. Parameters use the same keys in both
languages and appear in the equivalent command.

The GPT-2 TOFU starter evaluates answer probability on its forget and retain
inputs after training. Uploaded injection data needs separate evaluation inputs
for target accuracy and retention; configure these paths and enable
`trainer.args.do_eval` when those probes are available.

## Agent execution

The local planner works without an API key. For model-selected Skills, configure
OpenAI or DeepSeek on the server as described in [Agent usage](../docs/agent_mode.md).
Generate or edit the task configuration, preview it, then open **Use Agent for
this task**. The Agent uses that configuration, including its model, input data,
seed and limits. It cannot replace these through a tool argument. Preview is the
default; enable **Execute the previewed configuration** to start a model run.

The execution trace shows the chosen Skills and their observations. A created run
links directly to **Runs**, where its command, logs and results are available.
Stopping the session also stops its active run. Provider credentials remain on
the server and are not exported with plans.

## Read a result

A running task can have a pending result. A finished task without saved metrics
shows that state separately from an unknown result ID or a connection failure.
The result identifier comes from the task's reserved output directory.

Sample inspection reads saved per-example files. Reference answers and actual
model outputs are separate; absent before/after answers stay absent. Token-level
predictions are labelled separately from generated answers. Metric details carry
the evaluator, phase, protocol and scoring information when recorded. Comparison
checks these settings in addition to the model, dataset and seed; results with
missing settings are not automatically treated as equivalent.

Sequential-edit history entries identify their method and edit step. When a saved
history index links a checkpoint to its parent run, shared model, data and seed
information comes from that run; the sample scope remains the checkpoint's scope.
Missing checkpoint metadata stays unknown.

Locality can measure whether predictions stay unchanged after editing. In that
case, a score of 1 means the predictions agree, even if both differ from the
reference answer. Use reliability or answer accuracy to assess correctness.

## Custom inputs

Upload UTF-8 JSON, JSONL or CSV, review the normalized rows and register the
configuration. Editing requires a subject and new answer; injection requires a
prompt and new answer. Unlearning requires separate forget and retain inputs.
Review the generated command and resolved paths before execution.

Registry IDs stay the same across languages. Copied commands can be run from the
repository root. The command and resolved configuration are available after
preview; result reports can be downloaded from the Runs page.

To validate a separately prepared extension bundle before manual installation:

```bash
python scripts/validate_extension.py path/to/extension.zip
python scripts/validate_extension.py --installed my-extension
```

An optional `--smoke` checks a configured model workflow and may load weights.
See [components](../docs/components.md) for installation and interfaces.

## Interface appearance

The workbench uses a violet palette with light and dark themes. English text uses
IBM Plex Sans, Chinese text uses a Noto Sans SC subset, and commands use IBM Plex
Mono. Fonts are bundled locally; no external font service is required. The
operating-table alpaca appears in the sidebar, with a compact mark for the browser
tab and mobile header.

## Source layout

- `backend/`: API, task persistence, uploads and result access.
- `frontend/`: React interface.
- `backend/data_import.py`: bounded upload parsing and operation-aware normalization.
