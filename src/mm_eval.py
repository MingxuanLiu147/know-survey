"""多模态 unlearning 评估入口（sidecar，不修改现有 eval.py）。

根据 benchmark 配置选择 MLLMU 或 CLEAR evaluator，执行评估并输出 JSON 结果。

注意：
    当前环境下需要在导入 Hydra 前先导入 torch，否则直接执行
    `python src/mm_eval.py ...` 可能导致 CUDA 不可用。

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
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "src")

from mm_registry import (  # noqa: E402
    MM_UNLEARNING_DATASETS,
    get_mm_dataset,
    import_symbol,
)

logger = logging.getLogger(__name__)

EVALUATOR_REGISTRY = {
    key: registration.evaluator for key, registration in MM_UNLEARNING_DATASETS.items()
}


def _import_cls(dotpath: str):
    return import_symbol(dotpath)


def _resolve_runtime_device(cfg: DictConfig) -> str:
    device = str(cfg.get("device", "cuda"))
    require_cuda = bool(cfg.get("require_cuda", device.startswith("cuda")))
    cuda_available = torch.cuda.is_available()

    logger.info(
        "Runtime device check: device=%s, require_cuda=%s, cuda_available=%s, cuda_device_count=%s",
        device,
        require_cuda,
        cuda_available,
        torch.cuda.device_count(),
    )

    if device.startswith("cuda") and not cuda_available:
        hint = (
            "CUDA is not available for mm_eval. "
            "In this environment avoid `python -u`, stdout/stderr redirection, "
            "`CUDA_VISIBLE_DEVICES=*`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`. "
            "If you intentionally want CPU, override with `device=cpu require_cuda=false`."
        )
        if require_cuda:
            raise RuntimeError(hint)
        logger.warning("%s", hint)
        return "cpu"

    return device


@hydra.main(version_base=None, config_path="../configs", config_name="mm_eval.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model
    from transformers import set_seed

    device = _resolve_runtime_device(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    logger.info("Loading model for evaluation...")
    model, processor = get_mm_model(cfg.model)
    if device.startswith("cuda"):
        model = model.to(device)
        logger.info("Moved evaluation model to %s", device)
    else:
        logger.warning("Evaluation will run on CPU")
    model.eval()

    benchmark = str(cfg.eval.get("benchmark", "mllmu"))
    registration = get_mm_dataset(benchmark)
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
    start_time = time.perf_counter()
    results = evaluator.evaluate()
    elapsed_seconds = time.perf_counter() - start_time
    output_dir = Path(str(cfg.eval.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    (output_dir / "eval_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "operation": "unlearning",
                "modality": "multimodal",
                "benchmark": registration.key,
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
    logger.info("Evaluation complete. Results: %s", results)


if __name__ == "__main__":
    main()
