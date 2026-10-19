"""评估入口

本模块刻意保持精简：只负责把配置、模型加载与评估器执行串起来。
真正的评估逻辑（指标计算、日志写入、缓存复用、任务差异）都在 `src/evals/` 内部完成

整体流程：
1) 固定随机种子，保证评估可复现。
2) 从配置加载模型与分词器。
3) 从 `cfg.eval` 构建评估器（TOFU/MUSE/LM‑Eval/编辑/注入等）。
4) 共享同一模型与模板信息，依次执行各评估器。

这样设计的原因：
- 复用同一模型实例，避免重复加载，保证评估之间的可比性。
- 通过 Hydra 保持评估逻辑可配置（换数据集/指标无需改代码）。
- 把指标与日志逻辑集中在评估器里，入口保持简单且可预测。
"""

import logging
from copy import deepcopy

import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model.checkpoint import load_checkpoint as _load_eval_model
from evals.runtime import EvaluationRuntime
from evals import get_evaluators

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """评估入口函数。

    Args:
        cfg (DictConfig): Hydra 加载的评估配置。
    """
    # 1) 可复现性：先固定随机种子，避免后续操作污染随机状态。
    seed_everything(cfg.seed)

    # 2) 一次性加载模型与分词器并在各评估器之间共享。
    #    这样既省时，也保证评估之间的比较一致。
    model_cfg = deepcopy(cfg.model)
    base_model_cfg = deepcopy(cfg.model)
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    template_args = model_cfg.template_args
    model, tokenizer = _load_eval_model(model_cfg)

    # 3) 根据配置构建评估器。
    #    每个评估器内部管理一组指标与日志输出逻辑。
    #    具体指标在 `cfg.eval.<name>.metrics` 下配置。
    #    常见类别包括：
    #    - 记忆/遗忘：probability、ROUGE、truth_ratio 等；
    #    - 隐私/MIA：ks_test、privleak、MIA AUC 系列等；
    #    - 效用：classifier_prob、hm_aggregate 等；
    #    - 知识编辑/注入：reliability、locality、portability 等。
    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)

    # 4) 执行各评估器。
    #    传入共享的运行时上下文（model/tokenizer/template_args），
    #    每个评估器会把细粒度日志与汇总结果写到各自的 output_dir。
    with EvaluationRuntime(base_model_cfg) as runtime:
        for evaluator_name, evaluator in evaluators.items():
            eval_args = {
                # 模板参数用于统一数据集 prompt 的格式。
                "template_args": template_args,
                "model": model,
                "tokenizer": tokenizer,
            }
            runtime_args = runtime.resolve(evaluator.eval_cfg)
            if runtime_args:
                logger.info(
                    "Evaluator `%s` loaded runtime args: %s",
                    evaluator_name,
                    ", ".join(sorted(runtime_args.keys())),
                )
                eval_args.update(runtime_args)
            # `evaluate` 会返回摘要字典；这里不使用它，
            # 因为评估器已经将结果写入配置的 output_dir。
            _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
