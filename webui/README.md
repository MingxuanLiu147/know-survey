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

1. **Workbench**: describe a knowledge update, choose a recipe and review the
   proposed model, dataset and method. Compose the configuration before running.
2. **Runs**: inspect logs, stop a running job and open its outputs.
3. **Compare**: select completed runs and inspect target effects and side effects.
4. **Extensions**: browse registered models, datasets, methods and Skills.

The Agent workflow supports deterministic planning and optional OpenAI/DeepSeek
providers. Configure credentials on the server as described in
[Agent workflows](../docs/agent_mode.md); credentials are not exported with plans.
Preview does not load a model. Execution requires local inputs or accessible
model/data sources and enough compute for the chosen method.

## Custom inputs

Upload UTF-8 JSON, JSONL or CSV, review the normalized rows and register the
configuration. Editing requires a subject and new answer; injection requires a
prompt and new answer. Unlearning requires separate forget and retain inputs.
Review the generated command and resolved paths before execution.

Configuration import/export preserves registry IDs across languages. Copied
commands can be run from the repository root. Missing results remain empty
until a real run produces them.

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
