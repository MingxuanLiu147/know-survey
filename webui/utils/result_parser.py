"""
结果解析器
==========

解析评估结果文件（*_SUMMARY.json），用于在 UI 中展示指标卡片。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    """评估结果"""

    name: str  # 评估名称（如 TOFU, MUSE）
    metrics: Dict[str, float]  # 指标名 -> 值
    file_path: str  # 结果文件路径


class ResultParser:
    """评估结果解析器"""

    # 指标说明映射，用于 UI 展示
    METRIC_DESCRIPTIONS = {
        # TOFU 指标
        "forget_quality": "遗忘质量",
        "model_utility": "模型效用",
        "forget_Q_A_Prob": "遗忘-QA概率",
        "forget_Q_A_ROUGE": "遗忘-QA ROUGE",
        "forget_Truth_Ratio": "遗忘-真值比",
        "privleak": "隐私泄露",
        "extraction_strength": "提取强度",
        # MUSE 指标
        "exact_memorization": "精确记忆",
        "forget_knowmem_ROUGE": "遗忘-知识记忆",
        "forget_verbmem_ROUGE": "遗忘-逐字记忆",
        "retain_knowmem_ROUGE": "保留-知识记忆",
        # Edit 指标
        "reliability": "可靠性",
        "generalization": "泛化性",
        "locality": "局部性",
        "portability": "可移植性",
        # Inject 指标
        "task_accuracy": "任务准确率",
        "knowledge_retention": "知识保持",
    }

    @staticmethod
    def find_summary_files(output_dir: str) -> List[str]:
        """在输出目录中查找所有 SUMMARY.json 文件

        Args:
            output_dir: 输出目录路径

        Returns:
            SUMMARY.json 文件路径列表
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            return []

        summary_files = []
        # 查找当前目录
        for f in output_path.glob("*_SUMMARY.json"):
            summary_files.append(str(f))

        # 如果没找到，查找子目录
        if not summary_files:
            for f in output_path.rglob("*_SUMMARY.json"):
                summary_files.append(str(f))

        return sorted(summary_files)

    @staticmethod
    def parse_summary_file(file_path: str) -> Optional[EvalResult]:
        """解析单个 SUMMARY.json 文件

        Args:
            file_path: 文件路径

        Returns:
            EvalResult 对象，解析失败返回 None
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 从文件名提取评估名称（如 TOFU_SUMMARY.json -> TOFU）
            filename = os.path.basename(file_path)
            eval_name = filename.replace("_SUMMARY.json", "")

            return EvalResult(name=eval_name, metrics=data, file_path=file_path)
        except Exception:
            return None

    @classmethod
    def parse_results(cls, output_dir: str) -> List[EvalResult]:
        """解析输出目录中的所有评估结果

        Args:
            output_dir: 输出目录路径

        Returns:
            EvalResult 列表
        """
        results = []
        summary_files = cls.find_summary_files(output_dir)

        for file_path in summary_files:
            result = cls.parse_summary_file(file_path)
            if result:
                results.append(result)

        return results

    @classmethod
    def format_metric_value(cls, value: Any) -> str:
        """格式化指标值用于显示

        Args:
            value: 指标值

        Returns:
            格式化后的字符串
        """
        if isinstance(value, float):
            if abs(value) < 0.01 or abs(value) > 1000:
                return f"{value:.4e}"
            return f"{value:.4f}"
        return str(value)

    @classmethod
    def render_metrics_markdown(cls, results: List[EvalResult]) -> str:
        """将评估结果渲染为 Markdown 格式

        Args:
            results: EvalResult 列表

        Returns:
            Markdown 字符串
        """
        if not results:
            return "*暂无评估结果*"

        lines = []

        for result in results:
            lines.append(f"### {result.name} 评估结果\n")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")

            for metric_name, value in sorted(result.metrics.items()):
                display_name = cls.METRIC_DESCRIPTIONS.get(metric_name, metric_name)
                formatted_value = cls.format_metric_value(value)
                lines.append(f"| {display_name} | {formatted_value} |")

            lines.append("")

        return "\n".join(lines)

    @classmethod
    def render_metrics_html(cls, results: List[EvalResult]) -> str:
        """将评估结果渲染为 HTML 卡片格式（用于 Gradio）

        Args:
            results: EvalResult 列表

        Returns:
            HTML 字符串
        """
        if not results:
            return "<p style='color: #888;'>暂无评估结果</p>"

        html_parts = []

        for result in results:
            cards = []
            for metric_name, value in sorted(result.metrics.items()):
                display_name = cls.METRIC_DESCRIPTIONS.get(metric_name, metric_name)
                formatted_value = cls.format_metric_value(value)

                # 根据值的正负和大小选择颜色
                if isinstance(value, (int, float)):
                    if value < 0:
                        color = "#dc3545"  # 红色
                    elif value > 0.8:
                        color = "#28a745"  # 绿色
                    else:
                        color = "#007bff"  # 蓝色
                else:
                    color = "#6c757d"  # 灰色

                card = f"""
                <div style="display: inline-block; margin: 5px; padding: 10px; 
                            border: 1px solid #ddd; border-radius: 8px; 
                            min-width: 120px; text-align: center;">
                    <div style="font-size: 0.85em; color: #666;">{display_name}</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: {color};">{formatted_value}</div>
                </div>
                """
                cards.append(card)

            html_parts.append(f"""
            <div style="margin-bottom: 15px;">
                <h4 style="margin-bottom: 10px;">{result.name} 评估结果</h4>
                <div style="display: flex; flex-wrap: wrap;">
                    {"".join(cards)}
                </div>
            </div>
            """)

        return "".join(html_parts)
