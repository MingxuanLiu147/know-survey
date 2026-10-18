"""指标系统的基础抽象。

这里定义了 ``UnlearningMetric`` 统一封装：
- 数据集与 collator 的按需加载
- pre_compute 前置指标的递归执行
- 参考日志的加载
- 指标结果缓存

同时提供 ``unlearning_metric`` 装饰器，便于把普通函数包装成统一的指标对象。
"""

import os
import json
import logging
from typing import Callable, Any, Dict
from data import get_datasets, get_collators

logger = logging.getLogger("metrics")


class UnlearningMetric:
    """统一的指标封装类，屏蔽不同指标之间的调用差异。"""

    def __init__(
        self,
        name: str,
        metric_fn: Callable[..., Any],
    ):
        self.name = name
        self._metric_fn = metric_fn
        self.data = None
        self.collators = None
        self.pre_compute_metrics: Dict[str, Callable] = {}

    def get_datasets(self, dataset_cfgs=None, **kwargs):
        """Load the datasets from config"""
        if self.data:
            return self.data
        data = get_datasets(
            tokenizer=kwargs.get("tokenizer", None),
            template_args=kwargs.get("template_args", None),
            dataset_cfgs=dataset_cfgs,
        )
        return data

    def get_collators(self, collator_cfgs=None, **kwargs):
        """Load the collators from config"""
        if self.collators:
            return self.collators
        collators = get_collators(
            tokenizer=kwargs.get("tokenizer", None), collator_cfgs=collator_cfgs
        )
        return collators

    def set_pre_compute_metrics(self, metrics: Dict[str, Callable]):
        """记录当前指标依赖的前置指标。"""
        self.pre_compute_metrics.update(metrics)

    def evaluate_metric(self, model, metric_name, **kwargs):
        """在参数准备完成后，真正调用底层指标函数。"""
        logger.info(f"Evaluating {metric_name}")
        results = self._metric_fn(model, **kwargs)
        return results

    def load_logs_from_file(self, file):
        """Load a logs file, assumes json"""
        logs = {}
        if os.path.exists(file):
            logger.info(f"Loading evaluations from {file}")
            with open(file, "r") as f:
                logs = json.load(f)
        else:
            raise ValueError(f"{file} doesn't exist!")
        return logs

    def prepare_kwargs_evaluate_metric(self, model, metric_name, cache={}, **kwargs):
        """Prepare the kwargs required to call the metric_fn defined by user.
        - Loads datasets, collators, results for pre_compute metrics
        Returns:
            Dict: Updated kwargs with datasets, collators, pre_compute results loaded
        """
        # Load datasets
        dataset_cfgs = kwargs.pop("datasets", None)
        if dataset_cfgs is not None:
            data = self.get_datasets(dataset_cfgs=dataset_cfgs, **kwargs)
            kwargs.update({"data": data})

        # Load collators
        collator_cfgs = kwargs.pop("collators", None)
        if collator_cfgs is not None:
            collators = self.get_collators(collator_cfgs=collator_cfgs, **kwargs)
            kwargs.update({"collators": collators})

        # Evaluate precompute and load results
        pre_compute_cfgs = kwargs.pop("pre_compute", {})
        pre_metric_results = {}
        for pre_metric_name, pre_metric_cfg in pre_compute_cfgs.items():
            access_name = pre_metric_cfg.get("access_key", pre_metric_name)
            _results = {}
            # 前置指标往往计算代价较高，因此优先复用缓存，避免重复前向。
            if pre_metric_name in cache:
                logger.info(
                    f"Skipping {metric_name}'s precompute {pre_metric_name}, already evaluated."
                )
                _results = cache[pre_metric_name]
            else:
                pre_metric = self.pre_compute_metrics.get(pre_metric_name, None)
                assert pre_metric is not None, ValueError(
                    f"No pre_compute metric of name {pre_metric_name}"
                )
                pre_metric_kwargs = kwargs.copy()
                pre_metric_kwargs.update(**pre_metric_cfg)
                _results = pre_metric.evaluate(
                    model, pre_metric_name, cache=cache, **pre_metric_kwargs
                )
            pre_metric_results.update({access_name: _results})
        if pre_metric_results:
            kwargs.update({"pre_compute": pre_metric_results})

        # Load reference logs
        reference_logs_cfgs = kwargs.pop("reference_logs", {})
        reference_logs = {}
        for reference_log_name, reference_log_cfg in reference_logs_cfgs.items():
            path = reference_log_cfg.get("path", None)
            if path is None:
                continue
            include_cfgs = reference_log_cfg.get("include", None)
            assert path is not None, ValueError(
                "path not specified for {reference_log_name} in {metric_name}"
            )
            _logs = self.load_logs_from_file(path)
            reference_logs[reference_log_name] = {}
            for key, include_cfg in include_cfgs.items():
                access_name = include_cfg.get("access_key", key)
                # 允许配置层为导入的参考指标结果重新命名，方便下游访问。
                _results = _logs.get(key, None)
                reference_logs[reference_log_name][access_name] = _results
                if _results is None:
                    logger.warning(
                        f"{key} evals not present in the {path}, setting it to None, may result in error soon if code attempts to access."
                    )
        if reference_logs:
            kwargs.update({"reference_logs": reference_logs})

        return kwargs

    def evaluate(self, model, metric_name, cache, **kwargs):
        """Evaluates a metric including its pre_compute metrics"""
        if metric_name in cache:
            logger.info(f"Skipping {metric_name}, already evaluated.")

        metric_kwargs = self.prepare_kwargs_evaluate_metric(
            model, metric_name, cache, **kwargs
        )
        results = self.evaluate_metric(model, metric_name, **metric_kwargs)
        cache.update({metric_name: results})
        return results

    def __call__(self, model, **kwargs):
        return self.evaluate(model, **kwargs)

    def __repr__(self) -> str:
        """Represents class object as string

        Returns:
            str: string representation of the class object
        """
        return f"{type(self).__name__} {self.name}"


# 这个装饰器把普通的 Python 指标函数包装成统一的 UnlearningMetric 对象。
class unlearning_metric:
    """将普通函数包装为 ``UnlearningMetric`` 的轻量装饰器。"""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, metric_fn: Callable[..., Any]) -> UnlearningMetric:
        name = self.name or metric_fn.__name__
        return UnlearningMetric(name=name, metric_fn=metric_fn)
