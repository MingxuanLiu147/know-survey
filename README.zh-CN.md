<div align="center">

<img src="webui/frontend/public/know-surgery-mark.png" alt="Know-Surgery 羊驼精修标志" width="128" height="128" />

# Know-Surgery

**统一的模型知识遗忘、注入与编辑工作台。**

[English](README.md) | **简体中文**

[快速开始](#快速开始) · [方法与数据](#方法与数据) · [WebUI](#webui) · [Agent](#agent) · [文档](docs/README.md)

</div>

Know-Surgery 将模型知识更新整理为统一的配置与执行流程：选择模型、数据和方法，通过 CLI、WebUI 或 Agent 运行，同时查看目标效果与副作用。项目基于 [OpenUnlearning](https://github.com/locuslab/open-unlearning) 扩展，编辑方法参考 [EasyEdit](https://github.com/zjunlp/EasyEdit) 及各算法实现。

| 目标 | 操作 | 主要输入 |
|---|---|---|
| 遗忘指定知识 | `unlearning` | 相互独立的遗忘集与保留集 |
| 学习新知识 | `inject` | 监督训练样本与评估数据 |
| 修改已有知识 | `edit` | 新答案、改写问题与无关知识探针 |

## 快速开始

使用 **Python 3.11+**，并准备适合当前硬件的 PyTorch 环境。从 [GitHub 仓库](https://github.com/MingxuanLiu147/know-survey) 安装：

```bash
git clone https://github.com/MingxuanLiu147/know-survey.git
cd know-survey
python -m venv .venv
source .venv/bin/activate
pip install -e ".[webui]"
```

可选依赖：`[gpu]` 提供 DeepSpeed/bitsandbytes，`[reft]` 提供 PyReFT，`[lm-eval]` 用于下游评估。模型权重和数据集需要单独准备。

先预览配置，此步骤不会加载模型：

```bash
know-surgery operation=edit \
  experiment=edit/model_coverage/gpt2_ft_counterfact \
  task_name=first_edit --cfg job
```

执行示例需要本地 GPT-2 权重和 CounterFact JSON 文件。设置路径，选择空闲 GPU，确认配置后去掉 `--cfg job`：

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

`know-surgery`、`python -m know_surgery` 和 `python src/surgery.py` 使用同一入口。预览只检查配置组合；运行还需要实际数据、权重和方法专用资源。结果通常写入 `saves/<mode>/<task_name>/`。

另外两种操作的配置预览：

```bash
know-surgery operation=unlearning experiment=unlearn/tofu/default \
  trainer=GradAscent forget_split=forget10 retain_split=retain90 \
  task_name=forget_example --cfg job

know-surgery operation=inject experiment=inject/ruletaker/default \
  trainer=inject/LoRA task_name=inject_example --cfg job
```

TOFU 遗忘需要使用正确的起始权重。RuleTaker 配置默认关闭评估，开启前需要提供目标与保留数据。详见[实验配置](docs/experiments.md)。

## 方法与数据

| 类别 | 方法 | 运行前准备 |
|---|---|---|
| 遗忘 | GradAscent、GradDiff、NPO、SimNPO、DPO、RMU、UNDIAL、CEU、SatImp、WGA、PDU | 匹配的遗忘集、保留集与评估配置 |
| 注入 | Full FT、LoRA、DoRA、AdaLoRA、LoReFT | 监督数据及模型对应的适配器模块 |
| 权重编辑 | FT、ROME、MEMIT、AlphaEdit | 层映射和统计量；ROME/MEMIT/AlphaEdit 提供显式参考配置 |
| 长目标编辑 | AnyEdit 的 MEMIT/AlphaEdit ARE 后端 | 目标分块、模型映射和保留统计量 |
| 记忆／适配器编辑 | IKE、GRACE、WISE、SERAC | IKE 需要演示库，WISE 需要独立负例，SERAC 需要训练辅助网络 |
| 辅助网络编辑 | MEND、MALMEN、InstructEdit | 紧凑实现，需要匹配的预训练编辑网络 |
| 图文更新 | 多模态遗忘、LoRA 注入与编辑适配器 | 实验性能力，按模型与方法选择；[本地图文注入说明](docs/components.md#imagetext-injection) |

文本模型配置覆盖 GPT-2/XL、GPT-J、Llama、Qwen2.5、Phi、Gemma、Zephyr 和 DeepSeek 蒸馏模型。数据适配器包括 TOFU、CounterFact、zsRE、LEME、ELKEN、AKEW、WikiHow、RuleTaker、RuleArena、Alpaca/ShareGPT、ConceptEdit 及图文数据集。配置存在不表示可以任意组合模型与方法。

[编辑方法](docs/editing.md)介绍前置资源和运行命令，[组件接入](docs/components.md)介绍自定义模型与数据，[评估说明](docs/evaluation.md)解释指标与结果。参考配置实现特定算法机制并适配模型，不承诺在所有模型上得到原论文分数。

## WebUI

构建 React 前端后启动 FastAPI 工作台：

```bash
cd webui/frontend
npm ci
npm run build
cd ../..
python -m webui.backend.app --host 127.0.0.1 --port 17860
```

打开 [localhost:17860](http://127.0.0.1:17860)，选择配置或描述目标，预览输入与等价命令，然后执行并查看结果。界面支持中英切换。详见 [WebUI 使用说明](webui/README.md)。

## Agent

内置 Agent 可以选择工具，完成规划、数据准备、执行与报告：

```bash
know-surgery agent --root . --provider deepseek \
  --intent "Inspect recipes and preview one GPT-2 CounterFact edit with FT" \
  --max-steps 12 --max-cases 2 --timeout 600
```

在服务端设置 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。默认只预览，添加 `--execute --cuda 0` 后允许限量运行模型。Codex、Claude Code 也可使用 `.claude/skills/` 中的共享工作流。详见 [Agent 使用说明](docs/agent_mode.md)。

## 目录结构

```text
src/           CLI、模型、数据集、训练器、评估与 Agent
configs/       模型、数据、方法、评估与实验 YAML
webui/         FastAPI + React 工作台
.claude/skills/ 共享 Agent 工作流
scripts/       数据准备、辅助网络训练与结果处理
demos/         小型输入和用法示例
docs/          配置与使用说明
```

[架构说明](docs/architecture.md)介绍执行流程，[文档索引](docs/README.md)提供各主题入口。恢复编辑器时可能同时需要权重与方法状态，请保留完整输出目录。

## 致谢与许可

Know-Surgery 保留 OpenUnlearning 的原始署名；实验中请引用实际使用的方法和数据集。项目代码采用 [MIT 许可](LICENSE)，第三方条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 `LICENSES/`。模型权重与数据集遵循各自条款。
