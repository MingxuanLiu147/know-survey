<div align="center">

<img src="webui/frontend/public/know-surgery-mark.png" alt="Know-Surgery alpaca mark" width="128" height="128" />

# Know-Surgery

**A workbench for knowledge unlearning, injection, and editing.**

**English** | [简体中文](README.zh-CN.md)

[Quickstart](#quickstart) · [Methods](#methods-and-data) · [WebUI](#webui) · [Agent](#agent) · [Documentation](docs/README.md)

</div>

Know-Surgery brings model knowledge updates into one configurable workflow. Choose a model, data and method; run through the CLI, WebUI or an Agent; inspect both the intended change and its side effects. The project builds on [OpenUnlearning](https://github.com/locuslab/open-unlearning), with editing integrations informed by [EasyEdit](https://github.com/zjunlp/EasyEdit) and individual method implementations.

| Goal | Operation | Main inputs |
|---|---|---|
| Forget selected knowledge | `unlearning` | Separate forget and retain sets |
| Learn new knowledge | `inject` | Supervised examples and evaluation data |
| Correct existing knowledge | `edit` | New answers, paraphrases and unrelated probes |

## Quickstart

Use Python **3.11+** and a PyTorch environment suited to your hardware. Install from the [GitHub repository](https://github.com/MingxuanLiu147/know-survey):

```bash
git clone https://github.com/MingxuanLiu147/know-survey.git
cd know-survey
python -m venv .venv
source .venv/bin/activate
pip install -e ".[webui]"
```

Optional extras are `[gpu]` for DeepSpeed/bitsandbytes, `[reft]` for PyReFT and `[lm-eval]` for downstream evaluation. Model weights and datasets are prepared separately.

Preview a configuration without loading a model:

```bash
know-surgery operation=edit \
  experiment=edit/model_coverage/gpt2_ft_counterfact \
  task_name=first_edit --cfg job
```

To run this example, set paths to a local GPT-2 checkpoint and CounterFact JSON file. Select an available GPU, review the configuration, then remove `--cfg job`:

```bash
export MODEL_PATH=/path/to/gpt2
export EDIT_DATA=/path/to/counterfact.json
CUDA_VISIBLE_DEVICES=0 know-surgery operation=edit \
  experiment=edit/model_coverage/gpt2_ft_counterfact \
  task_name=first_edit max_edits=1 save_edited_model=false \
  trainer.args.bf16=false \
  model.model_args.pretrained_model_name_or_path="$MODEL_PATH" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_PATH" \
  +data.edit.CounterFact_edit.args.data_path="$EDIT_DATA" --cfg job
```

`know-surgery`, `python -m know_surgery` and `python src/surgery.py` share the same entrypoint. A preview checks configuration composition; execution also needs readable inputs and method-specific assets. Results normally appear in `saves/<mode>/<task_name>/`.

Other operation previews:

```bash
know-surgery operation=unlearning experiment=unlearn/tofu/default \
  trainer=GradAscent forget_split=forget10 retain_split=retain90 \
  task_name=forget_example --cfg job

know-surgery operation=inject experiment=inject/ruletaker/default \
  trainer=inject/LoRA task_name=inject_example --cfg job
```

TOFU unlearning requires the intended starting checkpoint. The RuleTaker recipe disables evaluation by default; supply target and retention data to enable it. See [experiment configuration](docs/experiments.md).

## Methods and data

| Family | Methods | Before running |
|---|---|---|
| Unlearning | GradAscent, GradDiff, NPO, SimNPO, DPO, RMU, UNDIAL, CEU, SatImp, WGA, PDU | Select compatible forget/retain data and evaluation |
| Injection | Full FT, LoRA, DoRA, AdaLoRA, LoReFT | Set supervised data and model-specific adapter modules |
| Weight editing | FT, ROME, MEMIT, AlphaEdit | Check layers and statistics; ROME/MEMIT/AlphaEdit offer explicit reference profiles |
| Long-target editing | AnyEdit with MEMIT/AlphaEdit ARE | Set target chunks, model mappings and preservation statistics |
| Memory / adapter editing | IKE, GRACE, WISE, SERAC | IKE needs demonstrations; WISE needs scope negatives; SERAC needs trained auxiliary networks |
| Auxiliary-network editing | MEND, MALMEN, InstructEdit | Compact implementations requiring matching trained checkpoints |
| Image–text updates | Multimodal unlearning, LoRA injection and editing adapters | Experimental, model/method-specific; [local image/text injection](docs/components.md#imagetext-injection) |

Text configurations cover GPT-2/XL, GPT-J, Llama, Qwen2.5, Phi, Gemma, Zephyr and DeepSeek distillations. Dataset adapters include TOFU, CounterFact, zsRE, LEME, ELKEN, AKEW, WikiHow, RuleTaker, RuleArena, Alpaca/ShareGPT, ConceptEdit and image–text datasets. A configuration does not guarantee compatibility with every method.

Use [editing methods](docs/editing.md) for prerequisites and commands, [components](docs/components.md) for your own model or data, and [evaluation](docs/evaluation.md) to interpret results. Reference profiles reproduce specified mechanisms with model adapters; they do not promise the original paper's scores on every model.

## WebUI

Build the React frontend, then start the FastAPI workbench:

```bash
cd webui/frontend
npm ci
npm run build
cd ../..
python -m webui.backend.app --host 127.0.0.1 --port 17860
```

Open [localhost:17860](http://127.0.0.1:17860). Select a recipe or describe a goal, preview inputs and the equivalent command, then run and inspect results. The interface supports English and Chinese. See the [WebUI guide](webui/README.md).

## Agent

The built-in Agent chooses tools for planning, data preparation, execution and reporting:

```bash
know-surgery agent --root . --provider deepseek \
  --intent "Inspect recipes and preview one GPT-2 CounterFact edit with FT" \
  --max-steps 12 --max-cases 2 --timeout 600
```

Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` on the server. Preview is the default; add `--execute --cuda 0` for bounded model execution. Codex and Claude Code can also use the shared workflows in `.claude/skills/`. See [Agent usage](docs/agent_mode.md).

## Project layout

```text
src/           CLI, models, datasets, trainers, evaluation and Agent runtime
configs/       Model, data, method, evaluation and experiment YAML
webui/         FastAPI + React workbench
.claude/skills/ Shared Agent workflows
scripts/       Data preparation, auxiliary training and result helpers
demos/         Small input and usage examples
docs/          Configuration and usage guides
```

See [architecture](docs/architecture.md) for the execution flow and [documentation](docs/README.md) for all guides. Saved editing runs may need method state in addition to weights; keep the entire output directory when restoring an editor.

## Acknowledgements and license

Know-Surgery preserves OpenUnlearning attribution. Cite the methods and datasets used in your experiment. Project code is distributed under the [MIT license](LICENSE), with third-party terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `LICENSES/`. Model weights and datasets retain their own terms.
