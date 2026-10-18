"""模型与分词器加载辅助模块。

本模块统一负责：
1) 模型类注册（`MODEL_REGISTRY`）；
2) 加载模型时的数据类型（dtype）规范化；
3) 基于 Hydra 配置安全地构建模型与分词器。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

# 可选：Hugging Face 下载缓存目录。
hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    """按类名注册模型类，便于通过配置动态查找。"""
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    """把配置里的字符串 dtype 转成 torch dtype 对象。

    这样做的原因：
    - Hydra/OmegaConf 常把 dtype 存成字符串；
    - `from_pretrained` 需要的是 torch 的 dtype 对象。
    """
    # `open_dict` 允许临时修改结构化的 OmegaConf 对象。
    with open_dict(model_args):
        # 先移除 torch_dtype，避免后续 kwargs 重复传参。
        torch_dtype = model_args.pop("torch_dtype", None)

    # 在当前加载路径里，flash_attention_2 仅支持 fp16/bf16。
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    """根据配置构建模型与分词器。

    期望配置包含：
    - model_cfg.model_args
    - model_cfg.tokenizer_args
    - 可选 model_cfg.model_handler（在 MODEL_REGISTRY 中的类名）
    """
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        # 模型路径单独取出，不与通用 kwargs 混在一起。
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **model_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )
    # 分词器单独加载，保证函数始终返回 (model, tokenizer) 二元组。
    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    """确保分词器存在可用的 EOS token。

    - 若原本没有 EOS，则新增；
    - 若已有 EOS，则替换为指定值。
    """
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    """加载分词器，并补齐最基本的特殊 token 安全默认值。"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    # 部分检查点缺少 EOS，这里补一个通用兜底值。
    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    # 许多训练/评测流程依赖 pad_token；若缺失则复用 EOS。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# 模块导入时完成可用模型处理器注册。
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
