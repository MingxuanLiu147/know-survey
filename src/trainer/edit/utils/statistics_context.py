"""Explicit bounded Qwen2 statistics execution, with transactional restoration.

The query256 path is an approved numerical implementation variant, not an
assertion of bitwise equivalence to the unchunked Transformers eager kernel.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import re
import threading

import torch
import transformers

from trainer.edit.utils.compute_z import _frozen_model, _input_device

_CONTEXT_LOCK = threading.RLock()
POLICY_VERSION = "qwen2_query_chunked_eager_real_prefix_v1"


def _source_hash(value) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def statistics_execution_identity(
    model,
    *,
    attention_policy="model_default",
    query_chunk_size=256,
    placement_policy="model_default",
    execution_backend="cuda",
) -> dict:
    """Describe actual statistics arithmetic independently of model placement."""
    if attention_policy not in {"model_default", "qwen2_eager_query_chunked"}:
        raise ValueError(f"Unknown statistics attention_policy: {attention_policy}")
    if placement_policy not in {"model_default", "qwen2_prefix"}:
        raise ValueError(f"Unknown statistics placement_policy: {placement_policy}")
    if (
        isinstance(query_chunk_size, bool)
        or not isinstance(query_chunk_size, int)
        or query_chunk_size <= 0
    ):
        raise ValueError("query_chunk_size must be a positive integer")
    if (
        attention_policy != "model_default" or placement_policy != "model_default"
    ) and execution_backend not in {"cpu", "cuda"}:
        raise ValueError(
            "execution_backend must be cpu or cuda for explicit statistics policies"
        )
    identity = {
        "attention_policy": attention_policy,
        "placement_policy": placement_policy,
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "accumulation": "one_FP32_GEMM_per_padded_batch_then_ordered_out_of_place_add_v1",
        "implementation_sources_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("statistics_context.py", "layer_stats.py")
        },
    }
    if attention_policy != "model_default" or placement_policy != "model_default":
        identity["execution_backend"] = execution_backend
    if attention_policy == "qwen2_eager_query_chunked":
        import transformers.models.qwen2.modeling_qwen2 as qwen

        if getattr(model.config, "model_type", None) != "qwen2":
            raise ValueError("qwen2_eager_query_chunked requires a Qwen2 model")
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError(
                "qwen2_eager_query_chunked requires the base eager attention backend"
            )
        identity.update(
            version=POLICY_VERSION,
            query_chunk_size=query_chunk_size,
            eager_source_sha256=_source_hash(qwen.eager_attention_forward),
            repeat_kv_source_sha256=_source_hash(qwen.repeat_kv),
            key_value_policy="full_unsliced_repeat_kv",
            attention_weights="not_requested",
        )
    if (
        placement_policy == "qwen2_prefix"
        and getattr(model.config, "model_type", None) != "qwen2"
    ):
        raise ValueError("qwen2_prefix requires a Qwen2 model")
    return identity


def _query_chunked_eager(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    *,
    query_chunk_size=256,
    **kwargs,
):
    import transformers.models.qwen2.modeling_qwen2 as qwen

    if torch.is_grad_enabled() or module.training or dropout:
        raise RuntimeError(
            "Chunked eager statistics require frozen eval/no_grad and dropout=0"
        )
    if kwargs.get("output_attentions", False):
        raise ValueError("Chunked statistics do not return attention weights")
    key_states = qwen.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen.repeat_kv(value, module.num_key_value_groups)
    chunks = []
    for begin in range(0, query.shape[-2], query_chunk_size):
        end = min(begin + query_chunk_size, query.shape[-2])
        weights = (
            torch.matmul(query[:, :, begin:end, :], key_states.transpose(2, 3))
            * scaling
        )
        if attention_mask is not None:
            causal_mask = (
                attention_mask[:, :, :, : key_states.shape[-2]]
                if attention_mask.shape[-2] == 1
                else attention_mask[:, :, begin:end, : key_states.shape[-2]]
            )
            weights = weights + causal_mask
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        weights = torch.nn.functional.dropout(
            weights, p=dropout, training=module.training
        )
        chunks.append(torch.matmul(weights, value_states))
        del weights
    return torch.cat(chunks, dim=2).transpose(1, 2).contiguous(), None


@contextmanager
def statistics_execution_context(
    model,
    layer_name,
    *,
    attention_policy="model_default",
    query_chunk_size=256,
    placement_policy="model_default",
    execution_device=None,
    execution_backend="cuda",
):
    """Restore parameters, buffers, backend, callable and flags even on failure.

    Real tensor transfers preserve values; no meta or Accelerate offload is used.
    The lock prevents overlapping process-local patches. This is intended for a
    single synchronous statistics worker, not a concurrent serving model.
    """
    statistics_execution_identity(
        model,
        attention_policy=attention_policy,
        query_chunk_size=query_chunk_size,
        placement_policy=placement_policy,
        execution_backend=execution_backend,
    )
    if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
        raise ValueError("Statistics context refuses meta parameters or buffers")
    if attention_policy == "model_default" and placement_policy == "model_default":
        device = _input_device(model)
        if execution_device is not None and torch.device(execution_device) != device:
            raise ValueError(
                "execution_device transfers require explicit qwen2_prefix placement"
            )
        with _frozen_model(model), torch.no_grad():
            yield device
        return
    if any(getattr(module, "_hf_hook", None) is not None for module in model.modules()):
        raise ValueError(
            "Statistics context requires ordinary real tensors without Accelerate hooks"
        )
    prefix_end = None
    if placement_policy == "qwen2_prefix":
        match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.down_proj", layer_name)
        if match is None or int(match[1]) >= len(model.model.layers):
            raise ValueError(
                "qwen2_prefix requires an existing model.layers.N.mlp.down_proj"
            )
        prefix_end = int(match[1]) + 1
    device = (
        torch.device(execution_device)
        if execution_device is not None
        else _input_device(model)
    )
    if device.type != execution_backend:
        raise ValueError(
            f"Statistics execution_backend={execution_backend} cannot compute on {device.type}; declare a separate backend identity"
        )
    if device.type == "meta":
        raise ValueError("Statistics execution_device cannot be meta")
    with _CONTEXT_LOCK:
        # Save direct owning slots so ordinary .to replacements of buffers are
        # also restored. Parameters retain their original objects and aliases.
        slots = [
            (
                module,
                key,
                parameter,
                parameter.device,
                parameter.dtype,
                parameter.requires_grad,
                None
                if parameter.grad is None
                else (parameter.grad.device, parameter.grad.dtype),
            )
            for module in model.modules()
            for key, parameter in module._parameters.items()
            if parameter is not None
        ]
        buffers = [
            (module, key, value, value.device, value.dtype)
            for module in model.modules()
            for key, value in module._buffers.items()
            if value is not None
        ]
        modes = [(module, module.training) for module in model.modules()]
        configurations = {
            id(module.config): (
                module.config,
                getattr(module.config, "_attn_implementation", None),
            )
            for module in model.modules()
            if hasattr(module, "config")
        }
        import transformers.models.qwen2.modeling_qwen2 as qwen

        original_eager = qwen.eager_attention_forward
        try:
            if prefix_end is not None:
                model.to("cpu")
                model.get_input_embeddings().to(device)
                for module in model.model.layers[:prefix_end]:
                    module.to(device)
            elif execution_device is not None and device != _input_device(model):
                raise ValueError(
                    "execution_device transfers require explicit qwen2_prefix placement"
                )
            if attention_policy == "qwen2_eager_query_chunked":

                def chunked(*args, **kwargs):
                    return _query_chunked_eager(
                        *args, query_chunk_size=query_chunk_size, **kwargs
                    )

                qwen.eager_attention_forward = chunked
            with _frozen_model(model), torch.no_grad():
                yield device
        finally:
            qwen.eager_attention_forward = original_eager
            for config, backend in configurations.values():
                config._attn_implementation = backend
            with torch.no_grad():
                for (
                    module,
                    key,
                    original,
                    original_device,
                    dtype,
                    requires_grad,
                    grad_spec,
                ) in slots:
                    current = module._parameters[key]
                    original.data = current.data.to(device=original_device, dtype=dtype)
                    if current.grad is not None and grad_spec is not None:
                        original.grad = current.grad.to(
                            device=grad_spec[0], dtype=grad_spec[1]
                        )
                    original.requires_grad_(requires_grad)
                    module._parameters[key] = original
                for module, key, original, original_device, dtype in buffers:
                    original.data = module._buffers[key].data.to(
                        device=original_device, dtype=dtype
                    )
                    module._buffers[key] = original
            for module, training in modes:
                module.training = training
