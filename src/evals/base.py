import os
import json
import logging
from evals.metrics import get_metrics

logger = logging.getLogger("evaluator")


class Evaluator:
    """Base evaluator for unlearning metrics.

    评估器基类：根据评估配置加载各类指标（metrics），调度模型完成评估，
    并将每个指标的细粒度结果和聚合结果分别写入 JSON 文件，便于对比不同实验。
    """

    def __init__(self, name, eval_cfg, **kwargs):
        self.name = name
        self.eval_cfg = eval_cfg
        # Some evaluator handlers (e.g. edit/inject leaf evaluators) are
        # standalone and do not define nested `metrics` configs.
        self.metrics_cfg = self.eval_cfg.get("metrics", {})
        self.metrics = (
            self.load_metrics(self.metrics_cfg)
            if self.metrics_cfg is not None and len(self.metrics_cfg) > 0
            else {}
        )
        output_dir = self.eval_cfg.get("output_dir", None)
        if output_dir is not None:
            logger.info(f"Evaluations stored in the experiment directory: {output_dir}")
        else:
            logger.info(
                f"Evaluator `{self.name}` initialized without output_dir "
                "(standalone evaluator mode)."
            )

    def get_logs_file_path(self, output_dir, suffix="EVAL"):
        """Returns the path to json file to store results.

        根据评估名称和后缀生成日志文件路径，用于区分原始评估结果（EVAL）和汇总结果（SUMMARY）。"""
        logs_filename = os.path.join(output_dir, f"{self.name}_{suffix}.json")
        return logs_filename

    def load_logs_from_file(self, file):
        """Returns the cache of existing results.

        如果对应的 JSON 文件已经存在，则把历史评估结果读入内存，作为后续增量评估的缓存。"""
        logs = {}
        if os.path.exists(file):
            logger.info(f"Loading existing evaluations from {file}")
            with open(file, "r") as f:
                logs = json.load(f)
        return logs

    def save_logs(self, logs, file):
        """Save the logs in a json file.

        先按 metric 名称排序，保证不同运行之间的键顺序稳定，再安全地写入磁盘。"""
        logs = dict(sorted(logs.items()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save {file}: {e}")

    def prepare_model(self, model):
        """Prepare model for evaluation.

        统一将模型切换到 eval() 模式，关闭 dropout 等训练行为，保证评估过程可复现。"""
        model.eval()
        return model

    def load_metrics(self, metrics_cfg):
        """Load metrics for evaluation.

        通过 `evals.metrics.get_metrics` 根据配置构造各类 `UnlearningMetric` 对象，
        这些对象内部负责数据加载、预计算和具体评价逻辑，这里只做调度。"""
        metrics = get_metrics(metrics_cfg)
        return metrics

    def summarize(self, logs):
        """Summarize the metrics results.

        只抽取每个 metric 结果中的聚合值 `agg_value`，生成一个紧凑的概要字典写到 SUMMARY 文件。"""
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            if metric_name not in self.metrics:
                continue
            agg_value = metric_results.get("agg_value", None)
            if agg_value is not None:
                metric_summary[metric_name] = agg_value
        return metric_summary

    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        """Run evaluation for all configured metrics.

        评估流程：
        1）根据 overwrite 标志决定是否复用已有日志作为 cache；
        2）遍历所有 metric，按需跳过已完成的指标；
        3）将同一个 `logs` 字典作为 cache 传给各个 metric，使其在内部可以共享预计算结果；
        4）每次更新完 logs 后，分别更新细粒度日志（EVAL）和聚合概要（SUMMARY）。"""
        # set flag to overwrite metrics（外部未显式传入时，优先使用配置中的默认策略）
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation（确保进入评估模式）
        model = self.prepare_model(model)

        # Set output_dir and file to store results（决定当前评估结果写到哪个实验目录）
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        # Load existing results from file if any.
        # 这里的 logs 同时充当「缓存」和「细粒度结果容器」，用于给 metric 复用历史结果和增量写入。
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )
        for metric_name, metric_fn in self.metrics.items():
            # 如果选择不覆盖且该 metric 已有非空结果，则直接跳过计算，只更新 SUMMARY。
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info(f"Skipping {metric_name}, already evaluated.")
                if "agg_value" in logs[metric_name]:
                    logger.info(
                        f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                    )
                # 即使跳过单个 metric，也保持 SUMMARY 文件与当前 logs 一致
                self.save_logs(self.summarize(logs), summary_file_path)
                continue
            # 若允许覆盖，先删除该 metric 旧结果，避免旧字段与新结构混在一起
            _ = logs.pop(metric_name, None)  # overwriting existing evals if present
            kwargs = {
                # 运行时依赖：分词器和 prompt 模板信息，供 metric 内部的数据加载与生成使用
                "tokenizer": kwargs.get("tokenizer", None),
                "template_args": kwargs.get("template_args", None),
            }
            # 静态配置参数：在 Hydra 配置中为该 metric 指定的数据集、collator、预计算等信息
            metrics_args = self.eval_cfg.metrics[metric_name]
            result = metric_fn(
                model,
                metric_name=metric_name,
                cache=logs,  # 将 logs 作为 cache 传入，使 metric 内部可以跳过已完成的 pre_compute 并统一写回结果
                **kwargs,
                **metrics_args,
            )
            if "agg_value" in result:
                # 每个 metric 完成后，优先打印聚合指标，方便在日志中快速对比
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            # 先落盘细粒度结果（EVAL），再根据最新 logs 生成概要结果（SUMMARY）
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        return self.summarize(logs)
