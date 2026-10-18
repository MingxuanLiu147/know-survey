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

import json
import logging
import os
from copy import deepcopy

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators

logger = logging.getLogger(__name__)


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_structured_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Evaluation input file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    if ext == ".jsonl":
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext in {".yaml", ".yml"}:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raise ValueError(
        f"Unsupported evaluation input format for {path}. Use .json/.jsonl/.yaml/.yml."
    )


def _get_optional_evaluator_input(eval_cfg: DictConfig, key: str):
    value = eval_cfg.get(key, None)
    if value is not None:
        return _to_plain(value)
    path = eval_cfg.get(f"{key}_path", None)
    if path is None:
        return None
    return _load_structured_data(path)


def _load_eval_model(model_cfg: DictConfig):
    """Load a Hugging Face/PEFT model or a saved pyreft intervention."""
    model_path = model_cfg.model_args.get("pretrained_model_name_or_path")
    from trainer.edit.artifacts import load_edit_artifacts, read_edit_artifact

    edit_artifact = read_edit_artifact(str(model_path))
    if edit_artifact is not None:
        edited_cfg = deepcopy(model_cfg)
        # The saved tokenizer is part of the editing checkpoint identity.
        with open_dict(edited_cfg.tokenizer_args):
            edited_cfg.tokenizer_args.pretrained_model_name_or_path = str(model_path)
        model, tokenizer = get_model(edited_cfg)
        load_edit_artifacts(model, tokenizer, model_path, manifest=edit_artifact)
        logger.info("Restored %s runtime from %s", edit_artifact["method"], model_path)
        return model, tokenizer
    adapter_config_path = os.path.join(str(model_path), "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        with open(adapter_config_path, encoding="utf-8") as handle:
            adapter_config = json.load(handle)
        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            raise ValueError(
                f"PEFT adapter {adapter_config_path} does not record its base model."
            )
        base_cfg = deepcopy(model_cfg)
        with open_dict(base_cfg.model_args):
            base_cfg.model_args.pretrained_model_name_or_path = base_model_path
        base_model, tokenizer = get_model(base_cfg)

        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(model_path))
        logger.info("Loaded PEFT adapter from %s", model_path)
        return model, tokenizer

    artifact_path = os.path.join(str(model_path), "inject_artifact.json")
    if not os.path.isfile(artifact_path):
        return get_model(model_cfg)

    with open(artifact_path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("method") != "loreft":
        return get_model(model_cfg)

    base_model_path = artifact.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(
            f"LoReFT artifact {artifact_path} does not record its base model."
        )

    base_cfg = deepcopy(model_cfg)
    with open_dict(base_cfg.model_args):
        base_cfg.model_args.pretrained_model_name_or_path = base_model_path
    base_model, tokenizer = get_model(base_cfg)

    import pyreft

    model = pyreft.ReftModel.load(str(model_path), base_model)
    model.set_device(str(next(base_model.parameters()).device))
    logger.info("Loaded LoReFT intervention from %s", model_path)
    return model, tokenizer


def _resolve_evaluator_runtime_args(
    evaluator_cfg: DictConfig,
    base_model_cfg: DictConfig,
    original_model_cache: dict,
):
    runtime_args = {}
    for key in ("edit_data", "eval_data", "benchmark_data"):
        value = _get_optional_evaluator_input(evaluator_cfg, key)
        if value is not None:
            runtime_args[key] = value

    original_model = evaluator_cfg.get("original_model", None)
    if original_model is not None:
        runtime_args["original_model"] = original_model

    original_model_path = evaluator_cfg.get("original_model_path", None)
    if original_model_path is not None:
        if original_model_path not in original_model_cache:
            ref_model_cfg = deepcopy(base_model_cfg)
            with open_dict(ref_model_cfg.model_args):
                ref_model_cfg.model_args.pretrained_model_name_or_path = (
                    original_model_path
                )
            ref_model, _ = get_model(ref_model_cfg)
            original_model_cache[original_model_path] = ref_model
            logger.info("Loaded original_model from %s", original_model_path)
        runtime_args["original_model"] = original_model_cache[original_model_path]

    return runtime_args


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
    original_model_cache = {}
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            # 模板参数用于统一数据集 prompt 的格式。
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        runtime_args = _resolve_evaluator_runtime_args(
            evaluator.eval_cfg, base_model_cfg, original_model_cache
        )
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
