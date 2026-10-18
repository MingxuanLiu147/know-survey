"""多模态 unlearning 训练入口（sidecar，不修改现有 train.py）。

通过 Hydra 配置驱动，加载多模态模型 + MLLMU 数据 + 遗忘方法，执行训练并保存 checkpoint。

注意：
    当前环境下需要在导入 Hydra 前先导入 torch，否则直接执行
    `python src/mm_train.py ...` 可能导致 CUDA 不可用。

使用示例：
    python src/mm_train.py \
        model=Qwen2VL-2B \
        data=mllmu \
        trainer=MMGradAscent
"""

import json
import logging
import shlex
import sys
import time
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, "src")

from mm_registry import (  # noqa: E402
    MM_UNLEARNING_ALGORITHMS,
    get_mm_algorithm,
    get_mm_dataset,
    import_symbol,
)

logger = logging.getLogger(__name__)

MM_TRAINER_REGISTRY = {}
for canonical_name, registration in MM_UNLEARNING_ALGORITHMS.items():
    if not registration.runnable:
        continue
    for registered_name in (canonical_name, *registration.aliases):
        MM_TRAINER_REGISTRY[registered_name] = registration.implementation


def _import_trainer_cls(dotpath: str):
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
            "CUDA is not available for mm_train. "
            "In this environment avoid `python -u`, stdout/stderr redirection, "
            "`CUDA_VISIBLE_DEVICES=*`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`. "
            "If you intentionally want CPU, override with `device=cpu require_cuda=false`."
        )
        if require_cuda:
            raise RuntimeError(hint)
        logger.warning("%s", hint)
        return "cpu"

    return device


def _get_data_loaders(cfg: DictConfig, processor):
    benchmark = str(cfg.data.get("benchmark", "mllmu")).lower()
    registration = get_mm_dataset(benchmark)
    loader = import_symbol(registration.loader)
    logger.info("Loading %s data...", registration.name)
    return loader(cfg.data, processor)


def _validate_method_data_mode(trainer_name: str, data_cfg: DictConfig) -> None:
    """Prevent methods from silently training on the wrong primary split."""

    mode = str(data_cfg.get("mode", "unlearn")).lower()
    if trainer_name == "MMRetainFT" and mode != "reference":
        raise ValueError(
            "MMRetainFT requires data.mode=reference so the primary loader contains "
            "retain/reference examples rather than forget examples."
        )
    if trainer_name != "MMRetainFT" and mode != "unlearn":
        raise ValueError(
            f"{trainer_name} requires data.mode=unlearn; received data.mode={mode}."
        )


def _resolve_algorithm(cfg: DictConfig):
    """Resolve a trainer alias and normalize the runtime config to its canonical name."""

    requested_name = str(cfg.trainer.handler)
    algorithm = get_mm_algorithm(requested_name, require_runnable=True)
    cfg.trainer.handler = algorithm.name
    return requested_name, algorithm


def _write_run_trace(
    cfg: DictConfig,
    output_dir: str,
    *,
    status: str,
    elapsed_seconds: float | None = None,
    train_summary: dict | None = None,
    error: BaseException | None = None,
) -> None:
    """Persist the resolved config and an auditable run summary."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    invocation_command = str(
        cfg.get(
            "invocation_command",
            shlex.join(["python", "src/mm_train.py", *sys.argv[1:]]),
        )
    )
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "visible_index": index,
                    "name": torch.cuda.get_device_name(index),
                    "peak_memory_mb": torch.cuda.max_memory_allocated(index)
                    / (1024**2),
                }
            )
    summary = {
        "status": status,
        "operation": str(cfg.get("operation", "unlearning")),
        "modality": "multimodal",
        "trainer": str(cfg.trainer.handler),
        "benchmark": str(cfg.data.benchmark),
        "model_path": str(cfg.model.model_path),
        "seed": int(cfg.get("seed", 42)),
        "output_dir": str(output_dir),
        "model_save_requested": bool(cfg.get("save_model", True)),
        "cuda_devices": cuda_devices,
        "elapsed_seconds": elapsed_seconds,
        "command": invocation_command,
        "train": train_summary or {},
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }
    (output_path / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="mm_train.yaml")
def main(cfg: DictConfig):
    from model.multimodal import get_mm_model
    from transformers import set_seed

    device = _resolve_runtime_device(cfg)
    requested_trainer_name, algorithm = _resolve_algorithm(cfg)
    trainer_name = algorithm.name
    benchmark = get_mm_dataset(str(cfg.data.get("benchmark", "mllmu")))
    if benchmark.key not in algorithm.supported_benchmarks:
        raise ValueError(
            f"{requested_trainer_name} does not declare support for {benchmark.name}. "
            f"Supported: {list(algorithm.supported_benchmarks)}"
        )
    _validate_method_data_mode(trainer_name, cfg.data)

    data_dir = Path(str(cfg.data.data_dir))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{benchmark.name} data directory not found: {data_dir}. "
            f"Registered public source: {benchmark.source}"
        )

    set_seed(int(cfg.get("seed", 42)))

    if trainer_name in {"MMUnlearner", "MMMANU"} and bool(
        cfg.model.get("lora", {}).get("enabled", False)
    ):
        raise ValueError(
            f"{trainer_name} requires `model.lora.enabled=false` because it edits "
            "the full-parameter model space rather than a LoRA adapter."
        )
    if trainer_name == "MMMANU" and not bool(cfg.data.get("paired_modalities", False)):
        raise ValueError(
            "MMMANU requires data.paired_modalities=true so every activation "
            "statistic has aligned text-only and image-text inputs."
        )
    if trainer_name == "MMUnlearner":
        grad_mask_path = cfg.trainer.get("method_args", {}).get("grad_mask_path")
        if not grad_mask_path:
            raise ValueError(
                "MMUnlearner requires trainer.method_args.grad_mask_path. "
                "Generate a benchmark-aware saliency mask before training."
            )
        if not Path(str(grad_mask_path)).is_file():
            raise FileNotFoundError(
                f"MMUnlearner grad mask not found: {grad_mask_path}"
            )

    logger.info("Loading multimodal model...")
    model, processor = get_mm_model(cfg.model)
    if device.startswith("cuda"):
        model = model.to(device)
        logger.info("Moved training model to %s", device)

    forget_loader, retain_loader = _get_data_loaders(cfg, processor)
    if (
        "retain_set" in algorithm.requirements
        and trainer_name != "MMRetainFT"
        and retain_loader is None
    ):
        raise ValueError(
            f"{trainer_name} requires a retain split, but {benchmark.name} did not "
            "produce a retain loader for the resolved data configuration."
        )

    trainer_cls = _import_trainer_cls(algorithm.implementation)

    method_args = dict(cfg.trainer.get("method_args", {}))
    trainer_args = dict(cfg.trainer.get("args", {}))

    output_dir = cfg.get("output_dir", "saves/mm_unlearn")

    if trainer_name in {"MMNPO", "MMKLMin", "MMSCRUB"}:
        reference_contract = {
            "MMNPO": ("oracle_model_path", "oracle_model"),
            "MMKLMin": ("reference_model_path", "reference_model"),
            "MMSCRUB": ("teacher_model_path", "teacher_model"),
        }
        path_key, model_key = reference_contract[trainer_name]
        reference_path = method_args.pop(path_key, None)
        if trainer_name == "MMNPO" and reference_path is None:
            raise ValueError("MMNPO requires method_args.oracle_model_path")
        if reference_path is None:
            reference_path = None

    if trainer_name in {"MMNPO", "MMKLMin", "MMSCRUB"} and reference_path is not None:
        from model.multimodal import get_mm_model as _load

        reference_cfg = OmegaConf.create(
            {
                "model_path": reference_path,
                "processor_path": cfg.model.get("processor_path", cfg.model.model_path),
                "model_family": cfg.model.get("model_family", "qwen2vl"),
                "torch_dtype": cfg.model.get("torch_dtype", "bfloat16"),
                "attn_implementation": cfg.model.get("attn_implementation", None),
                "trust_remote_code": cfg.model.get("trust_remote_code", False),
                "local_files_only": cfg.model.get("local_files_only", False),
                "processor_args": dict(cfg.model.get("processor_args", {})),
                "lora": {"enabled": False},
            }
        )
        reference_model, _ = _load(reference_cfg)
        method_args[model_key] = reference_model

    trainer = trainer_cls(
        model=model,
        processor=processor,
        forget_loader=forget_loader,
        retain_loader=retain_loader,
        lr=trainer_args.get("lr", 5e-4),
        num_epochs=trainer_args.get("num_epochs", 5),
        max_grad_norm=trainer_args.get("max_grad_norm", 1.0),
        gradient_accumulation_steps=trainer_args.get("gradient_accumulation_steps", 1),
        warmup_steps=trainer_args.get("warmup_steps", 0),
        save_dir=output_dir,
        **method_args,
    )

    _write_run_trace(cfg, output_dir, status="running")
    start_time = time.perf_counter()
    train_summary = None
    try:
        logger.info("Starting training with %s...", trainer_name)
        train_summary = trainer.train()

        if bool(cfg.get("save_model", True)):
            logger.info("Saving model...")
            trainer.save_model(output_dir)
        else:
            logger.info("Skipping model save because save_model=false")
    except Exception as exc:
        elapsed_seconds = time.perf_counter() - start_time
        _write_run_trace(
            cfg,
            output_dir,
            status="failed",
            elapsed_seconds=elapsed_seconds,
            train_summary=train_summary,
            error=exc,
        )
        logger.exception("Multimodal unlearning failed")
        raise
    elapsed_seconds = time.perf_counter() - start_time
    _write_run_trace(
        cfg,
        output_dir,
        status="completed",
        elapsed_seconds=elapsed_seconds,
        train_summary=train_summary,
    )
    logger.info("Done. Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
