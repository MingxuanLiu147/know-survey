# Python 接口

`run_surgery` 将简短的 Python 配置交给现有 Hydra / CLI 流程，同步执行知识更新并读取原生评估摘要。需要 Python 3.11 或更新版本，并在已安装项目依赖的环境中使用。

```python
from know_surgery import run_surgery

config = {
    "operation": "unlearning",
    "model": "Llama-3.2-1B",
    "dataset": "TOFU:forget10",
    "method": "NPO",
}
preview = run_surgery(config, preview=True)
print(preview["resolved_config"])
print(preview["preflight"])

# 检查并修改模型、层数、方法参数和数据后，由使用者粘贴确认值。
# 不自动从 preview 取 hash 后直接启动。
reviewed_hash = input("检查参数后粘贴当前 review_hash：").strip()
result = run_surgery(config, reviewed_config=reviewed_hash)
print(result["metrics"])
```

`preview=True` 组合配置并检查本地元数据，不加载权重、创建输出目录或启动训练。执行需显式传入 `reviewed_config`；更换模型、方法、数据或参数后旧确认失效。默认 `evaluate=True`，执行结束后返回真实评估文件中的指标。首版 Python 接口支持文本配方；多模态仍使用原有 CLI。

## 示例预设

| 操作 | 模型简名 | 数据集 | 方法 | 对应实验配方 |
|---|---|---|---|---|
| `inject` | `Qwen2.5-3B` | `RuleTaker` | `LoRA` | `inject/ruletaker/python_api` |
| `edit` | `Qwen2.5-7B` | `UnKE` | `AnyEdit` | `edit/unke/anyedit` |
| `unlearning` | `Llama-3.2-1B` | `TOFU:forget10` | `NPO` | `unlearn/tofu/default`，覆盖 `trainer=NPO` |

模型简名映射到相应的 Instruct 配置。TOFU 从 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` 开始，使用 `forget10 / retain90`；`TOFU` 是 `TOFU:forget10` 的简写。AnyEdit 沿用配方中的 `memit_are` 实现与统计设置。

RuleTaker 示例使用现有训练数据、40 条测试样本和 5 条无关事实保留探针。该小规模评估用于展示接口，正式研究应配置相应的评估集和保留能力基准。实际运行前须准备配方所需的模型、数据和方法资源；预览通过不等同于完整实验通过。

预设定义位于 [presets.yaml](../configs/api/presets.yaml)。它只提供上述三种便捷组合，不会推断其他模型、数据集与算法的兼容性。

## 自定义已注册配方

其他文本任务使用 `operation`、`experiment` 和 Hydra 覆盖项；此时不混用 `model`、`dataset`、`method` 简写字段。

```python
config = {
    "operation": "edit",
    "experiment": "edit/model_coverage/gpt2_ft_counterfact",
    "overrides": ["max_edits=2", "trainer.method_args.num_steps=1"],
    "output_dir": "saves/edit/python-example",
}
preview = run_surgery(config, preview=True)
```

`overrides` 接收 Hydra `key=value` 字符串列表，可指定本地模型、数据与评估路径。注入评估也接受原生的内联 `eval_data` 和 `benchmark_data`。只训练时设置 `evaluate=False`，接口同时关闭评估回调并返回空指标；独立编辑模式需要评估，不能使用该选项。

## 返回结果

| 字段 | 含义 |
|---|---|
| `status` | `preview` 或成功完成后的 `completed` |
| `operation` | 本次操作 |
| `resolved_config` | 完整解析后的配置 |
| `preflight` | 模型层数、参数、输入指纹、问题和待使用者确认的 `review_hash` |
| `command` | 调用现有 CLI 的命令参数列表 |
| `output_dir` | 运行产物目录 |
| `metrics` | 按摘要来源分组的原生指标，例如 `edit_inline` |
| `summary_files` | 本次读取的指标文件 |

接口保留原生指标名称、空值与协议元数据；例如缺少 Retain 参考结果时，不将 TOFU 的 FQ 缺失或空值填成零。接口不返回内存中的模型对象，模型与编辑产物按所选配方保存；独立编辑逐例恢复原模型，不生成单一合并模型。

每次运行默认生成独立目录；显式指定的目录必须为空，以免混入旧结果。目录中的 `python_api.json` 记录配置、命令、输出和执行状态。训练失败或评估未产生摘要会抛出异常。不要在已初始化的 Hydra 上下文内调用此接口。

相关入口：[三类任务与配置](experiments.md) · [Agent Skills](agent_mode.md)。
