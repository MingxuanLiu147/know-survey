# WebUI

The main interface uses FastAPI and React. The Gradio client remains available
for compatibility; both use the same surgery commands and configurations.

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
4. **Extensions**: browse Skills and create model, dataset or algorithm scaffolds.

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

The extension wizard exports a ZIP scaffold for manual review and installation:

```bash
python scripts/validate_extension.py path/to/extension.zip
python scripts/validate_extension.py --installed my-extension
```

An optional `--smoke` checks a configured model workflow and may load weights.
See [components](../docs/components.md) for installation and interfaces.

## Gradio client

```bash
python webui/app.py --host 127.0.0.1 --port 7861
```

Select an operation and recipe, review the equivalent command, then execute and
inspect outputs. Evaluation uses `src/eval.py`; the three surgery operations
remain `unlearning`, `inject` and `edit`.

## Source layout

- `backend/`: API, task persistence, uploads and result access.
- `frontend/`: React interface.
- `app.py` and `components/`: Gradio client.
- `utils/`: shared configuration, execution and extension helpers.
