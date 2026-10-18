"""
动态参数面板组件
================

根据选择的 trainer 动态生成参数编辑面板。
"""
# ruff: noqa: E402

import gradio as gr
from typing import Dict, Any, Tuple
import sys
from pathlib import Path

# 添加 webui 目录到路径
webui_dir = Path(__file__).parent.parent
if str(webui_dir) not in sys.path:
    sys.path.insert(0, str(webui_dir))

from utils.config_loader import ConfigLoader


def create_params_panel(config_loader: ConfigLoader) -> Dict[str, Any]:
    """创建参数面板

    Args:
        config_loader: 配置加载器

    Returns:
        组件字典
    """
    components = {}

    with gr.Accordion("高级参数", open=False):
        gr.Markdown("### 训练参数 (trainer.args)")

        with gr.Row():
            components["learning_rate"] = gr.Textbox(
                value="1e-5", label="学习率 (learning_rate)", scale=1
            )
            components["num_epochs"] = gr.Number(
                value=3, label="训练轮数 (num_train_epochs)", precision=0, scale=1
            )

        with gr.Row():
            components["batch_size"] = gr.Number(
                value=4,
                label="批次大小 (per_device_train_batch_size)",
                precision=0,
                scale=1,
            )
            components["gradient_accumulation"] = gr.Number(
                value=4, label="梯度累积步数", precision=0, scale=1
            )

        with gr.Row():
            components["max_length"] = gr.Number(
                value=512, label="最大序列长度", precision=0, scale=1
            )
            components["warmup_ratio"] = gr.Number(
                value=0.1, label="预热比例 (warmup_ratio)", scale=1
            )

        # 方法特定参数
        gr.Markdown("### 方法参数 (method_args)")
        components["method_args_json"] = gr.Textbox(
            value="{}", label="方法特定参数 (JSON 格式)", lines=5, interactive=True
        )

        # 高级 override 编辑器
        gr.Markdown("### Override 编辑器")
        components["overrides"] = gr.Textbox(
            value="",
            label="额外 Hydra Overrides",
            info="每行一个 override，如: trainer.args.weight_decay=0.01",
            lines=3,
            placeholder="trainer.args.weight_decay=0.01\nmodel.model_args.torch_dtype=float16",
        )

    return components


def get_trainer_params(trainer_name: str, config_loader: ConfigLoader) -> Dict:
    """获取 trainer 的默认参数

    Args:
        trainer_name: trainer 名称
        config_loader: 配置加载器

    Returns:
        参数字典，包含 args 和 method_args
    """
    config = config_loader.get_trainer_config(trainer_name)

    result = {
        "handler": config.get("handler", trainer_name),
        "args": {},
        "method_args": config.get("method_args", {}),
    }

    # 提取 args（如果存在）
    if "args" in config:
        result["args"] = config["args"]

    return result


def update_params_from_trainer(
    trainer_name: str, config_loader: ConfigLoader
) -> Tuple[gr.update, ...]:
    """根据选择的 trainer 更新参数面板

    Args:
        trainer_name: trainer 名称
        config_loader: 配置加载器

    Returns:
        各参数组件的更新值
    """
    import json

    params = get_trainer_params(trainer_name, config_loader)
    method_args = params.get("method_args", {})

    # 格式化 method_args 为 JSON
    method_args_json = json.dumps(method_args, indent=2, ensure_ascii=False)

    return (gr.update(value=method_args_json),)


def parse_overrides(
    learning_rate: str,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    max_length: int,
    warmup_ratio: float,
    method_args_json: str,
    overrides_text: str,
) -> Dict[str, Any]:
    """解析参数面板的值为 overrides 字典

    只有当用户修改了默认值时才添加 override，避免覆盖配置中的值。

    Returns:
        Hydra overrides 字典
    """
    import json

    # 默认值定义
    DEFAULTS = {
        "learning_rate": "1e-5",
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "max_length": 512,
        "warmup_ratio": 0.1,
    }

    overrides = {}

    # 只有当值与默认值不同时才添加 override
    if learning_rate and learning_rate != DEFAULTS["learning_rate"]:
        overrides["trainer.args.learning_rate"] = learning_rate
    if num_epochs and int(num_epochs) != DEFAULTS["num_epochs"]:
        overrides["trainer.args.num_train_epochs"] = int(num_epochs)
    if batch_size and int(batch_size) != DEFAULTS["batch_size"]:
        overrides["trainer.args.per_device_train_batch_size"] = int(batch_size)
    if (
        gradient_accumulation
        and int(gradient_accumulation) != DEFAULTS["gradient_accumulation"]
    ):
        overrides["trainer.args.gradient_accumulation_steps"] = int(
            gradient_accumulation
        )
    # 注意：max_length 的配置路径在各数据集内部，这里不单独处理
    # 如需修改，请使用 Override 编辑器手动指定完整路径
    if warmup_ratio is not None and warmup_ratio != DEFAULTS["warmup_ratio"]:
        overrides["trainer.args.warmup_ratio"] = warmup_ratio

    # 方法参数
    if method_args_json and method_args_json != "{}":
        try:
            method_args = json.loads(method_args_json)
            for key, value in method_args.items():
                overrides[f"trainer.method_args.{key}"] = value
        except json.JSONDecodeError:
            pass  # 忽略无效 JSON

    # 额外 overrides（用户手动输入的始终生效）
    if overrides_text:
        for line in overrides_text.strip().split("\n"):
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                overrides[key.strip()] = value.strip()

    return overrides
