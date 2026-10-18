"""多模态模型加载模块（sidecar，不修改现有 model/__init__.py）。

提供 get_mm_model()，返回 (model, processor)。
支持 Qwen2VL / Qwen2.5VL / Qwen3VL 系列，不依赖 qwen_vl_utils。
"""

import logging
from typing import List, Set

import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # transformers < 4.45 compatibility for config/unit checks
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

logger = logging.getLogger(__name__)


def find_all_linear_names(model) -> List[str]:
    """自动发现所有适合 LoRA 的 Linear 模块名，排除 embedding / lm_head / patch_embed。"""
    cls = torch.nn.Linear
    lora_module_names: Set[str] = set()
    exclude_keywords = {"embeddings", "embed_tokens", "patch_embed"}
    for name, module in model.named_modules():
        if any(kw in name for kw in exclude_keywords):
            continue
        if name.rsplit(".", 1)[-1] == "lm_head":
            continue
        if isinstance(module, cls):
            parts = name.split(".")
            lora_module_names.add(parts[0] if len(parts) == 1 else ".".join(parts[-2:]))
    return sorted(lora_module_names)


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def get_mm_model(model_cfg: DictConfig):
    """加载多模态模型和 processor。

    Args:
        model_cfg: Hydra 模型配置，包含 model_path / processor_path / lora / dtype 等字段。

    Returns:
        (model, processor) 二元组。
    """
    model_path = str(model_cfg.model_path)
    processor_path = str(model_cfg.get("processor_path", model_path))
    torch_dtype = _get_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = model_cfg.get("attn_implementation", None)
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    local_files_only = bool(model_cfg.get("local_files_only", False))

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    model_family = model_cfg.get("model_family", type(model).__name__)

    logger.info("Loaded %s from %s (dtype=%s)", model_family, model_path, torch_dtype)

    lora_cfg = model_cfg.get("lora", None)
    if lora_cfg and lora_cfg.get("enabled", False):
        configured_targets = lora_cfg.get("target_modules", None)
        if isinstance(configured_targets, str):
            # PEFT treats a string as a regex; keep it intact for model-specific
            # language-backbone targeting (for example LLaVA).
            target_modules = configured_targets
        else:
            target_modules = (
                list(configured_targets)
                if configured_targets
                else find_all_linear_names(model)
            )
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.print_trainable_parameters()

    processor_kwargs = dict(model_cfg.get("processor_args", {}))
    processor_kwargs.update(
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    processor = AutoProcessor.from_pretrained(processor_path, **processor_kwargs)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise TypeError(
            f"Processor loaded from '{processor_path}' does not expose a tokenizer"
        )
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise TypeError(
                f"Tokenizer loaded from '{processor_path}' has no pad or EOS token"
            )
        tokenizer.pad_token = eos_token

    model.eval()

    return model, processor
