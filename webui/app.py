# -*- coding: utf-8 -*-
# ruff: noqa: E402
"""
Know-Surgery WebUI
==================

基于 Gradio 的大模型知识可控更新工具包图形界面。
支持 Unlearn(删)、Inject(增)、Edit(改) 三大操作的配置和可视化运行。

启动方式:
    python webui/app.py

或指定端口:
    python webui/app.py --port 7860 --share
"""

import sys
import json
import tempfile
import time
from pathlib import Path

import gradio as gr

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import ConfigLoader
from utils.runner import CommandRunner, with_dataset_overrides
from components.config_panel import (
    create_config_panel,
    update_config_on_mode_change,
    update_config_on_recipe_change,
)
from components.run_panel import (
    create_run_panel,
    generate_command_preview,
    update_status,
    load_eval_results,
)
from components.params import (
    create_params_panel,
    parse_overrides,
    update_params_from_trainer,
)
from components.extensions import (
    create_custom_extensions_panel,
    refresh_catalog_callback,
)


# 全局配置加载器和命令执行器
config_loader = ConfigLoader(PROJECT_ROOT / "configs")
runner = CommandRunner(PROJECT_ROOT)


def build_ui() -> gr.Blocks:
    """构建完整的 WebUI 界面"""

    with gr.Blocks(title="Know-Surgery WebUI") as app:
        # 标题
        gr.Markdown("""
        # 🔬 Know-Surgery WebUI
        **大模型知识可控更新工具包** - 支持 Unlearn(删)、Inject(增)、Edit(改) 三大操作
        """)

        with gr.Row():
            # ==================== 左侧配置区 ====================
            with gr.Column(scale=1, elem_classes="config-column"):
                config_components = create_config_panel(config_loader)

                # 添加高级参数面板
                params_components = create_params_panel(config_loader)

            # ==================== 右侧运行区 ====================
            with gr.Column(scale=1, elem_classes="run-column"):
                run_components = create_run_panel()

        extension_components = create_custom_extensions_panel(
            PROJECT_ROOT,
            runner,
            catalog_records=config_loader.get_catalog_records(),
        )

        # ==================== 事件绑定 ====================

        catalog_outputs = [
            config_components["experiment"],
            config_components["model"],
            config_components["trainer"],
            config_components["unlearn_datasets_group"],
            config_components["inject_datasets_group"],
            config_components["edit_datasets_group"],
            config_components["forget_dataset"],
            config_components["retain_dataset"],
            config_components["train_dataset"],
            config_components["edit_dataset"],
            config_components["eval_suite"],
            config_components["train_eval_group"],
            config_components["eval_mode_group"],
        ]

        def update_catalog_lane(mode, catalog_mode, refresh=False):
            if refresh:
                config_loader.refresh_catalog()
            updates = update_config_on_mode_change(mode, config_loader, catalog_mode)
            experiments = config_loader.get_experiments(
                mode, experimental=catalog_mode == "experimental"
            )
            recipe_update = update_config_on_recipe_change(
                experiments[0] if experiments else None, config_loader
            )[-1]
            return (
                *updates,
                recipe_update,
                gr.update(value=config_loader.get_catalog_diagnostics()),
            )

        catalog_outputs_with_feedback = catalog_outputs + [
            config_components["recipe_details"],
            config_components["catalog_diagnostics"],
        ]

        # Mode and trust-lane changes replace every bound component atomically.
        for trigger in (
            config_components["mode"],
            config_components["catalog_mode"],
        ):
            trigger.change(
                fn=lambda mode, lane: update_catalog_lane(mode, lane),
                inputs=[
                    config_components["mode"],
                    config_components["catalog_mode"],
                ],
                outputs=catalog_outputs_with_feedback,
            )

        recipe_outputs = [
            config_components["model"],
            config_components["trainer"],
            config_components["forget_dataset"],
            config_components["retain_dataset"],
            config_components["train_dataset"],
            config_components["edit_dataset"],
            config_components["eval_suite"],
            config_components["recipe_details"],
        ]
        config_components["experiment"].change(
            fn=lambda experiment: update_config_on_recipe_change(
                experiment, config_loader
            ),
            inputs=[config_components["experiment"]],
            outputs=recipe_outputs,
        )

        # One refresh snapshot feeds both the run panel and extension catalog.
        for refresh_button in (
            config_components["refresh_components"],
            extension_components["catalog_refresh"],
        ):
            refresh_button.click(
                fn=lambda mode, lane: update_catalog_lane(mode, lane, refresh=True),
                inputs=[
                    config_components["mode"],
                    config_components["catalog_mode"],
                ],
                outputs=catalog_outputs_with_feedback,
            )
        config_components["refresh_components"].click(
            fn=lambda: refresh_catalog_callback(str(PROJECT_ROOT)),
            inputs=[],
            outputs=[extension_components["catalog_output"]],
        )

        # 刷新已保存模型列表
        def refresh_saved_models():
            saved_models = config_loader.get_saved_models()
            return gr.update(
                choices=saved_models, value=saved_models[0] if saved_models else None
            )

        config_components["refresh_saved_models"].click(
            fn=refresh_saved_models,
            inputs=[],
            outputs=[config_components["saved_model_path"]],
        )

        # 2. Trainer 改变时更新方法参数
        config_components["trainer"].change(
            fn=lambda trainer: update_params_from_trainer(trainer, config_loader),
            inputs=[config_components["trainer"]],
            outputs=[params_components["method_args_json"]],
        )

        # 3. 配置改变时更新命令预览
        config_inputs = [
            config_components["mode"],
            config_components["model"],
            config_components["trainer"],
            config_components["experiment"],
            config_components["task_name"],
            config_components["seed"],
            config_components["forget_dataset"],
            config_components["retain_dataset"],
            config_components["train_dataset"],
            config_components["edit_dataset"],
            config_components["eval_suite"],
            config_components["eval_suite_select"],
            config_components["saved_model_path"],
        ]

        # 高级参数输入
        params_inputs = [
            params_components["learning_rate"],
            params_components["num_epochs"],
            params_components["batch_size"],
            params_components["gradient_accumulation"],
            params_components["max_length"],
            params_components["warmup_ratio"],
            params_components["method_args_json"],
            params_components["overrides"],
        ]

        all_preview_inputs = config_inputs + params_inputs

        def update_command(*args):
            # 解析配置参数
            (
                mode,
                model,
                trainer,
                experiment,
                task_name,
                seed,
                forget,
                retain,
                train,
                edit,
                train_eval_suite,
                eval_suite,
                saved_model,
            ) = args[:13]
            # 解析高级参数
            (
                learning_rate,
                num_epochs,
                batch_size,
                gradient_accumulation,
                max_length,
                warmup_ratio,
                method_args_json,
                overrides_text,
            ) = args[13:]

            # 构建 overrides
            overrides = parse_overrides(
                learning_rate,
                num_epochs,
                batch_size,
                gradient_accumulation,
                max_length,
                warmup_ratio,
                method_args_json,
                overrides_text,
            )

            overrides = with_dataset_overrides(
                mode,
                experiment,
                overrides,
                forget_dataset=forget,
                retain_dataset=retain,
                train_dataset=train,
                edit_dataset=edit,
            )

            return generate_command_preview(
                mode=mode,
                model=model,
                trainer=trainer,
                experiment=experiment,
                task_name=task_name,
                seed=int(seed) if seed else 42,
                eval_suite=eval_suite if mode == "eval" else train_eval_suite,
                saved_model_path=saved_model,
                overrides=overrides,
            )

        for component in all_preview_inputs:
            component.change(
                fn=update_command,
                inputs=all_preview_inputs,
                outputs=[run_components["command_preview"]],
            )

        # 4. 运行按钮事件
        def run_training(
            mode,
            model,
            trainer,
            experiment,
            task_name,
            seed,
            forget_dataset,
            retain_dataset,
            train_dataset,
            edit_dataset,
            train_eval_suite,
            eval_suite_select,
            saved_model_path,
            cuda_devices,
            learning_rate,
            num_epochs,
            batch_size,
            gradient_accumulation,
            max_length,
            warmup_ratio,
            method_args_json,
            overrides_text,
        ):
            """执行训练/评估任务（生成器函数，用于实时更新日志）"""

            # eval 模式特殊处理
            if mode == "eval":
                # 构建 eval 命令
                command = runner.build_command(
                    mode="eval",
                    model=model,
                    task_name=task_name,
                    eval_suite=eval_suite_select,
                    model_path=saved_model_path,
                    overrides={},
                )
            else:
                # 构建 overrides
                overrides = parse_overrides(
                    learning_rate,
                    num_epochs,
                    batch_size,
                    gradient_accumulation,
                    max_length,
                    warmup_ratio,
                    method_args_json,
                    overrides_text,
                )

                use_experiment = experiment and not experiment.startswith("(")
                overrides = with_dataset_overrides(
                    mode,
                    experiment,
                    overrides,
                    forget_dataset=forget_dataset,
                    retain_dataset=retain_dataset,
                    train_dataset=train_dataset,
                    edit_dataset=edit_dataset,
                )

                # 构建命令
                command = runner.build_command(
                    mode=mode,
                    model=model,
                    trainer=trainer,
                    experiment=use_experiment and experiment or None,
                    task_name=task_name,
                    seed=int(seed) if seed else 42,
                    eval_suite=train_eval_suite,
                    overrides=overrides,
                )

            # 日志收集
            log_lines = []

            def log_callback(line: str):
                log_lines.append(line)

            # 设置环境变量
            env = {"CUDA_VISIBLE_DEVICES": cuda_devices} if cuda_devices else {}

            # 启动运行
            runner.run(command, log_callback=log_callback, env=env)

            # 等待进程启动
            time.sleep(0.5)

            # 实时更新日志
            last_log_count = 0
            while runner.is_running():
                if len(log_lines) > last_log_count:
                    last_log_count = len(log_lines)
                    output_dir = runner.infer_output_dir(command, "\n".join(log_lines))
                    yield (
                        "\n".join(log_lines),
                        *update_status(running=True),
                        output_dir,
                    )
                time.sleep(0.3)

            # 最终结果
            output_dir = runner.infer_output_dir(command, "\n".join(log_lines))
            yield (
                "\n".join(log_lines),
                *update_status(running=False, exit_code=runner.status.exit_code),
                output_dir,
            )

        run_components["run_btn"].click(
            fn=run_training,
            inputs=[
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
                config_components["eval_suite"],
                config_components["eval_suite_select"],
                config_components["saved_model_path"],
                run_components["cuda_devices"],
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                params_components["max_length"],
                params_components["warmup_ratio"],
                params_components["method_args_json"],
                params_components["overrides"],
            ],
            outputs=[
                run_components["log_output"],
                run_components["status"],
                run_components["run_btn"],
                run_components["stop_btn"],
                run_components["output_dir"],
            ],
        )

        # 加载结果按钮事件
        run_components["load_results_btn"].click(
            fn=load_eval_results,
            inputs=[run_components["output_dir"]],
            outputs=[run_components["result_summary"]],
        )

        # 5. 停止按钮事件
        def stop_training():
            runner.stop()
            return update_status(running=False, exit_code=-1)

        run_components["stop_btn"].click(
            fn=stop_training,
            inputs=[],
            outputs=[
                run_components["status"],
                run_components["run_btn"],
                run_components["stop_btn"],
            ],
        )

        # 6. 清空日志
        run_components["clear_log_btn"].click(
            fn=lambda: "", inputs=[], outputs=[run_components["log_output"]]
        )

        # 7. 复制命令（通过 JavaScript）
        run_components["copy_btn"].click(
            fn=None,
            inputs=[run_components["command_preview"]],
            outputs=[],
            js="(text) => { navigator.clipboard.writeText(text); }",
        )

        # 8. 导出配置
        def export_config(
            mode,
            model,
            trainer,
            experiment,
            task_name,
            seed,
            forget_dataset,
            retain_dataset,
            train_dataset,
            edit_dataset,
            learning_rate,
            num_epochs,
            batch_size,
            gradient_accumulation,
            max_length,
            warmup_ratio,
            method_args_json,
            overrides_text,
        ):
            """导出当前配置为 YAML 文件"""
            import yaml

            config = {
                "mode": mode,
                "model": model,
                "trainer": trainer,
                "task_name": task_name,
                "seed": int(seed) if seed else 42,
            }

            if experiment and not experiment.startswith("("):
                config["experiment"] = experiment

            # 数据集配置
            if mode == "unlearn":
                config["data"] = {
                    "forget": forget_dataset,
                    "retain": retain_dataset,
                }
            elif mode == "inject":
                config["data"] = {"train": train_dataset}
            elif mode == "edit":
                config["data"] = {"edit": edit_dataset}

            # 训练参数
            config["trainer_args"] = {
                "learning_rate": learning_rate,
                "num_train_epochs": int(num_epochs) if num_epochs else 3,
                "per_device_train_batch_size": int(batch_size) if batch_size else 4,
                "gradient_accumulation_steps": int(gradient_accumulation)
                if gradient_accumulation
                else 4,
                "warmup_ratio": warmup_ratio,
            }

            # 方法参数
            if method_args_json and method_args_json != "{}":
                try:
                    config["method_args"] = json.loads(method_args_json)
                except json.JSONDecodeError:
                    pass

            # 额外 overrides
            if overrides_text:
                config["overrides"] = overrides_text

            # 写入临时文件
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
                return f.name

        config_components["export_btn"].click(
            fn=export_config,
            inputs=[
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
                params_components["learning_rate"],
                params_components["num_epochs"],
                params_components["batch_size"],
                params_components["gradient_accumulation"],
                params_components["max_length"],
                params_components["warmup_ratio"],
                params_components["method_args_json"],
                params_components["overrides"],
            ],
            outputs=[config_components["config_download"]],
        )

        # 9. 导入配置
        def import_config(file):
            """从上传的 YAML 文件导入配置"""
            if file is None:
                return [gr.update()] * 10

            import yaml

            try:
                with open(file.name, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                return (
                    gr.update(value=config.get("mode", "unlearn")),
                    gr.update(value=config.get("model")),
                    gr.update(value=config.get("trainer")),
                    gr.update(value=config.get("experiment")),
                    gr.update(value=config.get("task_name", "my_experiment")),
                    gr.update(value=config.get("seed", 42)),
                    gr.update(value=config.get("data", {}).get("forget")),
                    gr.update(value=config.get("data", {}).get("retain")),
                    gr.update(value=config.get("data", {}).get("train")),
                    gr.update(value=config.get("data", {}).get("edit")),
                )
            except Exception as e:
                gr.Warning(f"导入配置失败: {e}")
                return [gr.update()] * 10

        config_components["import_btn"].click(
            fn=lambda: gr.update(visible=True),
            inputs=[],
            outputs=[config_components["config_upload"]],
        )

        config_components["config_upload"].change(
            fn=import_config,
            inputs=[config_components["config_upload"]],
            outputs=[
                config_components["mode"],
                config_components["model"],
                config_components["trainer"],
                config_components["experiment"],
                config_components["task_name"],
                config_components["seed"],
                config_components["forget_dataset"],
                config_components["retain_dataset"],
                config_components["train_dataset"],
                config_components["edit_dataset"],
            ],
        )

        # 初始化命令预览
        def initial_command_preview():
            experiments = config_loader.get_experiments("unlearn")
            experiment = experiments[0] if experiments else None
            recipe = config_loader.get_recipe_metadata(experiment)
            return generate_command_preview(
                mode="unlearn",
                model=recipe.get("model"),
                trainer=recipe.get("trainer"),
                experiment=experiment,
                task_name="my_experiment",
                seed=42,
                eval_suite=recipe.get("evaluation"),
                overrides={},
            )

        app.load(
            fn=initial_command_preview,
            inputs=[],
            outputs=[run_components["command_preview"]],
        )

    return app


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Know-Surgery WebUI")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务地址")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Know-Surgery WebUI                              ║
    ║   大模型知识可控更新工具包 - 图形化界面                       ║
    ╚═══════════════════════════════════════════════════════════╝
    
    启动参数:
      - Host: {args.host}
      - Port: {args.port}
      - Share: {args.share}
    
    项目根目录: {PROJECT_ROOT}
    配置目录: {PROJECT_ROOT / "configs"}
    """)

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
