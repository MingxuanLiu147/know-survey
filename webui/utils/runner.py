# -*- coding: utf-8 -*-
"""
命令执行器
==========

负责执行训练命令，管理进程，捕获输出日志。
"""

import subprocess
import threading
import signal
import os
import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from dataclasses import dataclass


SURGERY_OPERATION_BY_MODE = {
    "unlearn": "unlearning",
    "unlearning": "unlearning",
    "inject": "inject",
    "edit": "edit",
}
OUTPUT_MODE_BY_OPERATION = {
    "unlearning": "unlearn",
    "inject": "inject",
    "edit": "edit",
}


def surgery_operation_for_mode(mode: str) -> str:
    """Translate an internal UI mode to the public surgery operation name."""
    try:
        return SURGERY_OPERATION_BY_MODE[mode]
    except KeyError as exc:
        valid = ", ".join(sorted(SURGERY_OPERATION_BY_MODE))
        raise ValueError(
            f"Unsupported training mode `{mode}`. Expected: {valid}"
        ) from exc


def _hydra_value(value: Any) -> str:
    """Serialize one override value without relying on shell quoting."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def format_argv(argv: Sequence[str]) -> str:
    """Render argv for copy/paste while preserving exact token boundaries."""
    return " \\\n  ".join(shlex.quote(str(part)) for part in argv)


def with_dataset_overrides(
    mode: str,
    experiment: str | None,
    overrides: Mapping[str, Any] | None = None,
    *,
    forget_dataset: str | None = None,
    retain_dataset: str | None = None,
    train_dataset: str | None = None,
    edit_dataset: str | None = None,
) -> dict[str, Any]:
    """Add operation-specific dataset roles for manual compositions.

    Curated experiment recipes already own their data roles.  Adding another
    Hydra package override on top of a recipe is both ambiguous and a common
    source of UI/CLI drift, so dataset selectors are applied only when no
    experiment recipe is selected.
    """
    merged = dict(overrides or {})
    if mode == "eval":
        return merged
    if experiment and not experiment.startswith("("):
        return merged

    operation = surgery_operation_for_mode(mode)
    if operation == "unlearning":
        if forget_dataset:
            merged["data/datasets@data.forget"] = forget_dataset
        if retain_dataset:
            merged["data/datasets@data.retain"] = retain_dataset
    elif operation == "inject" and train_dataset:
        merged["data/datasets@data.train"] = train_dataset
    elif operation == "edit" and edit_dataset:
        merged["data/datasets@data.edit"] = edit_dataset
    return merged


@dataclass
class RunStatus:
    """运行状态"""

    running: bool = False
    exit_code: Optional[int] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None


class CommandRunner:
    """命令执行器，支持实时日志输出和进程管理"""

    def __init__(self, working_dir: str = None):
        """初始化执行器

        Args:
            working_dir: 工作目录，默认为项目根目录
        """
        if working_dir is None:
            working_dir = Path(__file__).parent.parent.parent
        self.working_dir = Path(working_dir)
        self.process: Optional[subprocess.Popen] = None
        self.status = RunStatus()
        self._log_callback: Optional[Callable[[str], None]] = None
        self._thread: Optional[threading.Thread] = None

    def build_argv(
        self,
        mode: str,
        model: str = None,
        trainer: str = None,
        experiment: str = None,
        task_name: str = "experiment",
        overrides: dict = None,
        eval_suite: str = None,
        model_path: str = None,
        seed: int | None = None,
    ) -> list[str]:
        """Build the exact argv used by both preview and execution.

        Args:
            mode: 运行模式 (unlearn/inject/edit/eval)
            model: 模型名称（HuggingFace 模型或配置名）
            trainer: 训练方法
            experiment: 实验模板
            task_name: 任务名称
            overrides: 参数覆盖字典
            eval_suite: 评测套件（eval 模式）
            model_path: 已保存模型路径（eval 模式）

        Returns:
            Argument vector suitable for ``subprocess.Popen(shell=False)``.
        """
        # eval 模式使用 eval.py
        if mode == "eval":
            cmd_parts = [
                "python",
                "src/eval.py",
                "--config-name=eval.yaml",
            ]

            # 添加评测套件
            if eval_suite:
                cmd_parts.append(f"eval={eval_suite}")

            # 添加模型路径或模型配置
            if model_path:
                cmd_parts.append(
                    f"model.model_args.pretrained_model_name_or_path={model_path}"
                )
            elif model:
                cmd_parts.append(f"model={model}")

            # 添加任务名称
            cmd_parts.append(f"task_name={task_name}")
        else:
            # 训练模式统一使用 surgery.py；UI 的 unlearn mode 对外叫 unlearning。
            operation = surgery_operation_for_mode(mode)
            cmd_parts = [
                "python",
                "src/surgery.py",
                "--config-name=surgery.yaml",
                f"operation={operation}",
            ]

            # 添加实验模板
            if experiment and not experiment.startswith("("):
                cmd_parts.append(f"experiment={experiment}")

            # 添加模型
            if model:
                cmd_parts.append(f"model={model}")

            # 添加训练方法
            if trainer:
                cmd_parts.append(f"trainer={trainer}")

            if eval_suite:
                cmd_parts.append(f"eval={eval_suite}")

            # 添加任务名称
            cmd_parts.append(f"task_name={task_name}")

            if seed is not None:
                cmd_parts.append(f"trainer.args.seed={int(seed)}")

        # 添加参数覆盖
        if overrides:
            for key, value in overrides.items():
                if value is not None and value != "":
                    cmd_parts.append(f"{key}={_hydra_value(value)}")

        return cmd_parts

    def build_command(self, *args, **kwargs) -> str:
        """Build a copy/paste command from :meth:`build_argv`."""
        return format_argv(self.build_argv(*args, **kwargs))

    def run(
        self,
        command: str | Sequence[str],
        log_callback: Callable[[str], None] = None,
        env: dict = None,
    ) -> bool:
        """执行命令

        Args:
            command: 要执行的命令
            log_callback: 日志回调函数，每行输出调用一次
            env: 额外的环境变量

        Returns:
            是否成功启动
        """
        if self.status.running:
            return False

        self._log_callback = log_callback
        self.status = RunStatus(running=True)

        # 准备环境变量
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # 启动后台线程执行命令
        self._thread = threading.Thread(
            target=self._run_process, args=(command, run_env), daemon=True
        )
        self._thread.start()

        return True

    def _run_process(self, command: str | Sequence[str], env: dict):
        """后台执行进程"""
        try:
            if isinstance(command, str):
                cmd_args = shlex.split(command.replace("\\\n", " "))
            else:
                cmd_args = [str(part) for part in command]
            cmd_display = format_argv(cmd_args).replace("\\\n", " ")

            self._log(f"[INFO] 工作目录: {self.working_dir}")
            self._log(f"[INFO] 执行命令: {cmd_display}\n")

            self.process = subprocess.Popen(
                cmd_args,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self.working_dir),
                env=env,
                text=True,
                bufsize=1,
            )

            # 实时读取输出
            for line in iter(self.process.stdout.readline, ""):
                if not self.status.running:
                    break
                self._log(line.rstrip("\n"))

            self.process.wait()
            self.status.exit_code = self.process.returncode

            if self.status.exit_code == 0:
                self._log("\n[SUCCESS] 命令执行完成")
            else:
                self._log(f"\n[ERROR] 命令执行失败，退出码: {self.status.exit_code}")

        except Exception as e:
            self.status.error = str(e)
            self._log(f"\n[ERROR] 执行异常: {e}")
        finally:
            self.status.running = False
            self.process = None

    def _log(self, message: str):
        """发送日志"""
        if self._log_callback:
            self._log_callback(message)

    def stop(self) -> bool:
        """停止当前运行的命令

        Returns:
            是否成功停止
        """
        if not self.status.running or self.process is None:
            return False

        try:
            self._log("\n[INFO] 正在停止进程...")
            # 发送 SIGINT 信号
            self.process.send_signal(signal.SIGINT)
            # 等待进程结束
            self.process.wait(timeout=5)
            self.status.running = False
            self._log("[INFO] 进程已停止")
            return True
        except subprocess.TimeoutExpired:
            # 强制终止
            self.process.kill()
            self.status.running = False
            self._log("[INFO] 进程已强制终止")
            return True
        except Exception as e:
            self._log(f"[ERROR] 停止失败: {e}")
            return False

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self.status.running

    def infer_output_dir(self, command: str, logs: str = "") -> str:
        """Infer the likely output directory from command/logs for UI display."""
        text = f"{command}\n{logs}"

        for pattern in (
            r"Done\. Model saved to\s+([^\s]+)",
            r"Model saved to\s+([^\s]+)",
            r"output_dir['\"]?:\s*['\"]?([^'\"\s,}]+)",
            r"trainer\.args\.output_dir=([^\s]+)",
            r"paths\.output_dir=([^\s]+)",
        ):
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip("\"'")

        task_match = re.search(r"(?:^|\s)task_name=([^\s]+)", command)
        config_match = re.search(r"--config-name=([A-Za-z0-9_./-]+)\.yaml", command)
        if task_match and config_match:
            mode = Path(config_match.group(1)).stem
            task_name = task_match.group(1).strip("\"'")
            if mode == "eval":
                return f"saves/eval/{task_name}"
            if mode == "surgery":
                operation_match = re.search(
                    r"(?:^|\s)operation=(unlearning|inject|edit)(?:\s|$)", command
                )
                if operation_match is None:
                    return ""
                mode = OUTPUT_MODE_BY_OPERATION[operation_match.group(1)]
            return f"saves/{mode}/{task_name}"

        return ""
