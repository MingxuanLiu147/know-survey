"""多模态 unlearning 评估入口（sidecar，不修改现有 eval.py）。

根据 benchmark 配置选择 MLLMU 或 CLEAR evaluator，执行评估并输出 JSON 结果。

使用示例：
    python src/mm_eval.py \
        model.model_path=saves/mm_unlearn/MMGradAscent/latest \
        eval=mllmu
"""

import json
import logging
import shlex
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "src")

from mm_registry import (  # noqa: E402
    MM_UNLEARNING_DATASETS,
    get_mm_dataset,
    import_symbol,
)

from know_surgery.runtime import resolve_device as _resolve_runtime_device

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {
    key: registration.evaluator for key, registration in MM_UNLEARNING_DATASETS.items()
}


def _incomplete_native_results(eval_cfg: DictConfig, results: dict) -> list[str]:
    """Identify missing requested MLLMU results without gating optional judges."""
    if not eval_cfg.get("require_all_splits", False):
        return []
    missing = []
    for split in eval_cfg.splits:
        for task in eval_cfg.tasks:
            for modality in ("image_textual", "pure_text", "total"):
                metric = results.get(split, {}).get(task, {}).get(modality, {})
                if metric.get("status") != "available" or not metric.get("denominator"):
                    missing.append(f"{split}/{task}/{modality}")
    return missing


def _import_cls(dotpath: str):
    return import_symbol(dotpath)


@hydra.main(version_base=None, config_path="../configs", config_name="mm_eval.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model
    from transformers import set_seed

    device = _resolve_runtime_device(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    benchmark = str(cfg.eval.get("benchmark", "mllmu"))
    registration = get_mm_dataset(benchmark)
    if not registration.evaluator:
        raise ValueError(
            f"{registration.name} has no standalone multimodal evaluator. "
            "Use the before/after train and retain probes recorded by MMLoRA "
            "in run_summary.json."
        )

    backend = str(cfg.eval.get("backend", "transformers"))
    if backend == "vllm":
        if (
            benchmark != "mllmu"
            or str(cfg.eval.get("shot_num", "zero_shot")) != "zero_shot"
        ):
            raise ValueError("vLLM evaluation currently supports zero-shot MLLMU only")
        from model.mm_vllm import MMVLLM

        model = MMVLLM(cfg.model, cfg.eval, seed)
        processor = model.processor
    elif backend == "transformers":
        logger.info("Loading model for evaluation...")
        model, processor = get_mm_model(cfg.model, for_training=False)
        model = model.to(device)
        model.eval()
    else:
        raise ValueError(f"Unknown multimodal evaluation backend: {backend}")

    data_dir = Path(str(cfg.eval.get("data_dir", "")))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{registration.name} evaluation data directory not found: {data_dir}. "
            f"Registered public source: {registration.source}"
        )
    eval_dotpath = registration.evaluator

    eval_cls = _import_cls(eval_dotpath)
    evaluator = eval_cls(cfg.eval, model, processor)

    logger.info("Running %s evaluation...", benchmark)
    # Engine initialization may consume Python RNG. Test image selection must
    # start from the same seed for every model and inference backend.
    set_seed(seed)
    start_time = time.perf_counter()
    results = evaluator.evaluate()
    incomplete = _incomplete_native_results(cfg.eval, results)
    elapsed_seconds = time.perf_counter() - start_time
    output_dir = Path(str(cfg.eval.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    (output_dir / "eval_summary.json").write_text(
        json.dumps(
            {
                "status": "incomplete" if incomplete else "completed",
                "incomplete_native_results": incomplete,
                "operation": "unlearning",
                "modality": "multimodal",
                "benchmark": registration.key,
                "inference_backend": backend,
                "model_path": str(cfg.model.model_path),
                "seed": seed,
                "elapsed_seconds": elapsed_seconds,
                "command": shlex.join(["python", "src/mm_eval.py", *sys.argv[1:]]),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if incomplete:
        raise RuntimeError(
            "Requested native evaluation is incomplete; details saved in eval_summary.json: "
            + ", ".join(incomplete)
        )
    logger.info("Evaluation complete. Results: %s", results)


if __name__ == "__main__":
    main()
