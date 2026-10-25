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
