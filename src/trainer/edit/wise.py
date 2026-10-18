"""
WISE 知识编辑器
===============

实现 WISE (Knowledge Memory for Lifelong Model Editing) 方法。
WISE 在 GRACE 基础上增加知识分片与合并机制，支持大规模持续编辑。

参考论文: WISE: Rethinking the Knowledge Memory for Lifelong Model Editing
of Large Language Models
https://arxiv.org/abs/2405.14768

核心思想：
1. 使用 WISEAdapter 包装目标层，维护可训练的新权重矩阵
2. 编辑时训练新权重，利用激活距离损失保持局部性
3. 累积编辑后，通过合并策略 (slerp / ties / linear) 整合权重
4. 推理时根据激活距离判断使用原始权重还是编辑后权重

本实现覆盖 WISE 的权重分片、稀疏梯度掩码、activation-margin、检索路由与
周期合并核心路径。EasyEdit 的 context-template 扩充、逐分片 activation
statistics 和 replay 属于尚未接入的扩展，不在当前实现范围内。
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)
from trainer.edit.utils.nethook import get_module, replace_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------


def _slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dot_threshold: float = 0.9995,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical linear interpolation between two weight tensors."""
    v0_np = v0.detach().cpu().float().numpy()
    v1_np = v1.detach().cpu().float().numpy()
    v0_copy, v1_copy = np.copy(v0_np), np.copy(v1_np)

    norm0, norm1 = np.linalg.norm(v0_np), np.linalg.norm(v1_np)
    if norm0 > eps:
        v0_np = v0_np / norm0
    if norm1 > eps:
        v1_np = v1_np / norm1

    dot = np.sum(v0_np * v1_np)
    if np.abs(dot) > dot_threshold:
        return torch.from_numpy((1 - t) * v0_copy + t * v1_copy).to(
            device=v0.device,
            dtype=v0.dtype,
        )

    theta_0 = np.arccos(np.clip(dot, -1, 1))
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    return torch.from_numpy(s0 * v0_copy + s1 * v1_copy).to(
        device=v0.device,
        dtype=v0.dtype,
    )


def _ties_merge(
    weights: List[float],
    base: torch.Tensor,
    tensors: List[torch.Tensor],
    density: float = 0.5,
) -> torch.Tensor:
    """TIES merge: magnitude pruning + sign consensus + weighted sum."""
    deltas = [t - base for t in tensors]

    pruned = []
    for d in deltas:
        flat = d.abs().view(-1)
        threshold = torch.quantile(flat.float(), 1.0 - density)
        pruned.append(d * (d.abs() >= threshold))

    stacked = torch.stack(pruned)
    weight_t = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while weight_t.dim() < stacked.dim():
        weight_t = weight_t.unsqueeze(-1)

    weighted = stacked * weight_t
    sign_sum = stacked.sign().sum(dim=0)
    majority = (sign_sum >= 0).to(stacked.dtype) * 2 - 1
    mask = (stacked.sign() == majority).to(stacked.dtype)

    mixed = (weighted * mask).sum(dim=0)
    divisor = (weight_t * mask).sum(dim=0)
    divisor[divisor == 0] = 1
    return (base + mixed / divisor).to(base.dtype)


def _linear_merge(t: float, base: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    return (1.0 - t) * base + t * new


def _activation_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    per_sample: bool = False,
) -> torch.Tensor:
    distances = torch.norm(a - b, p=2, dim=-1)
    if distances.ndim > 1:
        if attention_mask is None:
            distances = distances.mean(dim=-1)
        else:
            mask = attention_mask[:, -distances.shape[-1] :].to(distances)
            if mask.shape != distances.shape:
                raise ValueError(
                    "WISE attention mask must match the hidden-state batch"
                )
            distances = (distances * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return distances if per_sample else distances.mean()


# ---------------------------------------------------------------------------
# WISEAdapter
# ---------------------------------------------------------------------------


class WISEAdapter(nn.Module):
    """Knowledge-sharding adapter that wraps a single linear layer.

    Maintains a trainable ``new_weight`` matrix that is optimised during editing.
    Accumulated edits are periodically merged into the base layer using a
    configurable strategy (slerp / ties / linear).
    """

    def __init__(
        self,
        layer: nn.Module,
        merge_strategy: str = "slerp",
        merge_freq: int = 5,
        save_freq: int = 1,
        mask_ratio: float = 0.5,
        act_ratio: float = 1.0,
        merge_weight: float = 0.5,
        density: float = 0.5,
        transpose: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.weight = layer.weight
        self.device = layer.weight.device
        self.merge_strategy = merge_strategy
        self.merge_freq = merge_freq
        self.save_freq = save_freq
        self.mask_ratio = mask_ratio
        self.act_ratio = act_ratio
        self.merge_weight_factor = merge_weight
        self.density = density
        self.transpose = transpose

        if merge_strategy not in {"slerp", "ties", "linear"}:
            raise ValueError(f"Unknown WISE merge strategy: {merge_strategy}")
        if not 0.0 < mask_ratio <= 1.0:
            raise ValueError("WISE mask_ratio must be in (0, 1]")
        if merge_freq <= 0 or save_freq <= 0:
            raise ValueError("WISE merge_freq and save_freq must be positive")

        self._original_weight_cpu = self.weight.detach().cpu().clone()
        # WISE routing compares candidates against an immutable base branch.
        # Sharing this object with ``layer`` makes merges silently mutate the
        # supposed base and collapses the activation-distance signal.
        self.original_layer = copy.deepcopy(layer).requires_grad_(False)
        torch.cuda.empty_cache()
        self.new_weight = torch.nn.Parameter(self.weight.detach().clone())
        self.memory_weight: List[torch.Tensor] = []

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]

        self.editing: bool = False
        self.editing_total_cnt: int = 0
        self.weight_mask: Optional[torch.Tensor] = None
        self._min_act: float = float("inf")

        self.original_layer_output: Optional[torch.Tensor] = None
        self.new_weight_layer_output: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def generate_activation_mask(self):
        flat_size = self.new_weight.reshape(-1).size()[0]
        mask_np = np.random.choice(
            [1, 0],
            size=flat_size,
            p=[self.mask_ratio, 1 - self.mask_ratio],
        )
        self.weight_mask = torch.from_numpy(mask_np).to(self.device)

    def mask_new_weight_gradient(self):
        if self.new_weight.grad is None or self.weight_mask is None:
            return
        p_size = self.new_weight.grad.size()
        self.new_weight.grad = (
            (self.new_weight.grad.reshape(-1) * self.weight_mask)
            .view(p_size)
            .to(self.new_weight.grad.dtype)
        )

    def _weight_forward(self, x: Tensor, weight: torch.Tensor) -> Tensor:
        """Apply an nn.Linear or GPT-2 Conv1D-shaped weight with its bias."""
        bias = getattr(self.layer, "bias", None)
        if self.transpose:
            return F.linear(x, weight, bias)
        # transformers.pytorch_utils.Conv1D stores [in_features, out_features].
        output = torch.matmul(x, weight)
        return output if bias is None else output + bias

    def _new_weight_forward(self, x: Tensor) -> Tensor:
        return self._weight_forward(x, self.new_weight)

    def save_weight(self):
        self.memory_weight.append(self.new_weight.detach().clone())
        self.new_weight = torch.nn.Parameter(
            self._original_weight_cpu.clone().to(self.device)
        )

    def save_editing_activation(self):
        if self.original_layer_output is None or self.new_weight_layer_output is None:
            return
        # The edit request is always the first batch item; any following rows
        # are locality probes and must not lower the in-scope routing threshold.
        dist = _activation_distance(
            self.original_layer_output[:1],
            self.new_weight_layer_output[:1],
            self.attention_mask[:1] if self.attention_mask is not None else None,
        )
        self._min_act = min(self._min_act, dist.item())

    def merge_weight(self):
        """Merge shards into the latest merged branch, preserving prior rounds."""
        base = self.layer.weight.detach()
        # A merge can occur between saves (e.g. save=3, merge=5). Include the
        # unsaved active shard before resetting it, without adding an empty one.
        original = self._original_weight_cpu.to(self.new_weight)
        if self.memory_weight and not torch.equal(self.new_weight.detach(), original):
            self.memory_weight.append(self.new_weight.detach().clone())
        if not self.memory_weight:
            merged = self._apply_merge(
                self.merge_weight_factor,
                base,
                [self.new_weight.data],
            )
            self.layer.weight = nn.Parameter(merged, requires_grad=False)
            self.weight = self.layer.weight
            self.new_weight = torch.nn.Parameter(
                self._original_weight_cpu.clone().to(self.device)
            )
            return

        weights = [
            self.merge_weight_factor / len(self.memory_weight)
            for _ in self.memory_weight
        ]
        merged = self._apply_merge(
            weights,
            base,
            self.memory_weight,
        )
        self.layer.weight = nn.Parameter(merged.to(self.device), requires_grad=False)
        self.weight = self.layer.weight
        self.new_weight = torch.nn.Parameter(
            self._original_weight_cpu.clone().to(self.device)
        )
        self.memory_weight.clear()

    def _apply_merge(self, weights, base, tensors):
        if not tensors:
            return base.detach().clone()

        if isinstance(weights, (float, int)):
            coefficients = [float(weights) / len(tensors) for _ in tensors]
        else:
            coefficients = [float(weight) for weight in weights]
            if len(coefficients) != len(tensors):
                raise ValueError(
                    "WISE merge weights must align one-to-one with memory shards"
                )

        if self.merge_strategy == "slerp":
            total_weight = sum(coefficients)
            if total_weight == 0:
                return base.detach().clone()
            target = torch.zeros_like(base)
            for coefficient, tensor in zip(coefficients, tensors):
                target.add_(tensor.to(base), alpha=coefficient / total_weight)
            return _slerp(total_weight, base, target)
        elif self.merge_strategy == "ties":
            return _ties_merge(coefficients, base, tensors, density=self.density)
        elif self.merge_strategy == "linear":
            merged = base.detach().clone()
            for coefficient, tensor in zip(coefficients, tensors):
                merged.add_(tensor.to(base) - base, alpha=coefficient)
            return merged
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def forward(self, *args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise TypeError("WISEAdapter expects one hidden-state tensor input")
        if self.editing:
            layer_out = self._new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            original_out = self.original_layer(*args)
            merged_out = self.layer(*args)
            new_out = self._new_weight_forward(*args)
            candidates = [merged_out, new_out]
            distances = [
                _activation_distance(
                    original_out, candidate, self.attention_mask, per_sample=True
                )
                for candidate in candidates
            ]

            for mem_w in self.memory_weight:
                mem_out = self._weight_forward(args[0], mem_w)
                candidates.append(mem_out)
                distances.append(
                    _activation_distance(
                        original_out, mem_out, self.attention_mask, per_sample=True
                    )
                )

            threshold = (
                self._min_act * self.act_ratio if self._min_act < float("inf") else 0.0
            )
            best_distance, best_index = torch.stack(distances).max(dim=0)
            layer_out = original_out
            for index, candidate in enumerate(candidates):
                select = (best_index == index) & (best_distance >= threshold)
                select = select.reshape(-1, *([1] * (candidate.ndim - 1)))
                layer_out = torch.where(select, candidate, layer_out)

        return layer_out


# ---------------------------------------------------------------------------
# WISEEditor
# ---------------------------------------------------------------------------


class WISEEditor(EditTrainer):
    """WISE 知识编辑器

    基于知识分片与合并的持续知识编辑方法。
    支持 slerp / ties / linear 三种权重合并策略。

    Paper: https://arxiv.org/abs/2405.14768
    """

    variant = "native_sharded_routing_v3"
    adapter_class = WISEAdapter

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        inner_params: Union[str, List[str]] = "auto",
        edit_lr: float = 1e-4,
        n_iter: int = 70,
        merge_strategy: Literal["slerp", "ties", "linear"] = "slerp",
        merge_freq: int = 5,
        save_freq: int = 1,
        mask_ratio: float = 0.5,
        norm_constraint: Optional[float] = None,
        merge_weight: float = 0.5,
        density: float = 0.5,
        act_ratio: float = 1.0,
        gamma: float = 5.0,
        alpha: float = 20.0,
        beta: float = 5.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            layers: 编辑的目标层索引列表（兼容基类接口）
            inner_params: 目标层路径，如 ``"model.layers.9.mlp.down_proj"``
            edit_lr: 新权重优化学习率
            n_iter: 每次编辑的训练迭代次数
            merge_strategy: 权重合并策略 (``"slerp"`` | ``"ties"`` | ``"linear"``)
            merge_freq: 每隔多少次编辑执行一次合并
            save_freq: 每隔多少次编辑保存权重快照
            mask_ratio: 梯度掩码中被激活的比例
            norm_constraint: 权重更新的范数约束（None 表示不约束）
            merge_weight: 合并时新权重的插值系数
            density: TIES 合并的密度参数
            act_ratio: 激活距离阈值缩放因子
            gamma: margin loss 间隔参数
            alpha: out-of-scope 距离上界
            beta: in-scope 距离下界
        """
        super().__init__(layers=layers, *args, **kwargs)

        if isinstance(inner_params, list):
            if len(inner_params) != 1:
                raise ValueError("WISE supports exactly one inner parameter")
            inner_params = inner_params[0]
        self.inner_params = inner_params
        self.edit_lr = edit_lr
        self.n_iter = n_iter
        self.merge_strategy = merge_strategy
        self.merge_freq = merge_freq
        self.save_freq = save_freq
        self.mask_ratio = mask_ratio
        self.norm_constraint = norm_constraint
        self.merge_weight = merge_weight
        self.density = density
        self.act_ratio = act_ratio
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        if n_iter <= 0:
            raise ValueError("WISE n_iter must be positive")

        self.target_layer = self._resolve_target_layer(inner_params)

        self._adapter_installed = False
        self._original_layer = None
        self._requires_grad_state: Dict[str, bool] = {}
        self._edit_history: List[Dict[str, torch.Tensor]] = []
        self._attention_mask_hook = None
        self._base_model_identity = None

    def _resolve_target_layer(self, inner_params: str) -> str:
        """Resolve one editable MLP projection across GPT/Qwen/Llama layouts."""
        if inner_params != "auto":
            normalized = inner_params.replace("[", ".").replace("]", "")
            suffixes = (".weight", ".bias")
            return (
                normalized.rsplit(".", 1)[0]
                if normalized.endswith(suffixes)
                else normalized
            )

        layer_idx = self._resolve_layer_indices(self.layers)[-1]
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return f"transformer.h.{layer_idx}.mlp.c_fc"
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return f"model.layers.{layer_idx}.mlp.down_proj"
        if hasattr(self.model, "gpt_neox"):
            return f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h"
        raise ValueError(
            "Cannot auto-resolve a WISE target for this model architecture; "
            "set trainer.method_args.inner_params explicitly"
        )

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def _ensure_adapter(self):
        if self._adapter_installed:
            return

        model = self.model
        try:
            target = get_module(model, self.target_layer)
        except LookupError:
            raise ValueError(f"Target layer '{self.target_layer}' not found")

        if not isinstance(target, WISEAdapter):
            from transformers.pytorch_utils import Conv1D

            if not hasattr(target, "weight"):
                raise TypeError(
                    f"WISE target '{self.target_layer}' has no weight parameter"
                )
            self._base_model_identity = build_base_model_identity(model)
            transpose = not isinstance(target, Conv1D)
            adapter = self.adapter_class(
                layer=target,
                merge_strategy=self.merge_strategy,
                merge_freq=self.merge_freq,
                save_freq=self.save_freq,
                mask_ratio=self.mask_ratio,
                act_ratio=self.act_ratio,
                merge_weight=self.merge_weight,
                density=self.density,
                transpose=transpose,
            ).to(target.weight.device)

            self._original_layer = copy.deepcopy(target).cpu()
            self._requires_grad_state = {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            }
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            replace_module(model, self.target_layer, adapter)

            def capture_attention_mask(module, args, kwargs):
                del module, args
                adapter.attention_mask = kwargs.get("attention_mask")

            self._attention_mask_hook = model.register_forward_pre_hook(
                capture_attention_mask, with_kwargs=True
            )

        self._adapter_installed = True

    def _get_adapter(self) -> WISEAdapter:
        adapter = get_module(self.model, self.target_layer)
        assert isinstance(adapter, WISEAdapter)
        return adapter

    @staticmethod
    def _clone_optional_tensor(value: Optional[torch.Tensor]):
        return None if value is None else value.detach().clone()

    def _snapshot_transaction_state(
        self,
        adapter: WISEAdapter,
    ) -> Dict[str, Any]:
        """Capture every state component one WISE request can mutate."""
        return {
            "layer_state": {
                name: value.detach().clone()
                for name, value in adapter.layer.state_dict().items()
            },
            "layer_weight_requires_grad": adapter.layer.weight.requires_grad,
            "original_layer_state": {
                name: value.detach().clone()
                for name, value in adapter.original_layer.state_dict().items()
            },
            "original_weight_cpu": adapter._original_weight_cpu.detach().clone(),
            "new_weight": adapter.new_weight.detach().clone(),
            "new_weight_requires_grad": adapter.new_weight.requires_grad,
            "new_weight_grad": self._clone_optional_tensor(adapter.new_weight.grad),
            "memory_weight": [
                weight.detach().clone() for weight in adapter.memory_weight
            ],
            "editing": adapter.editing,
            "editing_total_cnt": adapter.editing_total_cnt,
            "weight_mask": self._clone_optional_tensor(adapter.weight_mask),
            "min_act": adapter._min_act,
            "original_layer_output": self._clone_optional_tensor(
                adapter.original_layer_output
            ),
            "new_weight_layer_output": self._clone_optional_tensor(
                adapter.new_weight_layer_output
            ),
            "edit_history": [
                {name: value.detach().clone() for name, value in entry.items()}
                for entry in self._edit_history
            ],
        }

    def _restore_transaction_state(
        self,
        adapter: WISEAdapter,
        state: Dict[str, Any],
    ) -> None:
        """Roll back a failed WISE request without disturbing prior edits."""
        adapter.layer.load_state_dict(state["layer_state"])
        adapter.layer.weight.requires_grad_(state["layer_weight_requires_grad"])
        adapter.weight = adapter.layer.weight
        adapter.original_layer.load_state_dict(state["original_layer_state"])
        adapter._original_weight_cpu = state["original_weight_cpu"]
        adapter.new_weight = nn.Parameter(
            state["new_weight"],
            requires_grad=state["new_weight_requires_grad"],
        )
        if state["new_weight_grad"] is not None:
            adapter.new_weight.grad = state["new_weight_grad"]
        adapter.memory_weight = state["memory_weight"]
        adapter.editing = state["editing"]
        adapter.editing_total_cnt = state["editing_total_cnt"]
        adapter.weight_mask = state["weight_mask"]
        adapter._min_act = state["min_act"]
        adapter.original_layer_output = state["original_layer_output"]
        adapter.new_weight_layer_output = state["new_weight_layer_output"]
        self._edit_history = state["edit_history"]

    def reset_layer(self):
        """Restore the original layer, removing the WISE adapter."""
        if self._adapter_installed and self._original_layer is not None:
            adapter = self._get_adapter()
            restored = copy.deepcopy(self._original_layer).to(
                adapter.layer.weight.device
            )
            replace_module(self.model, self.target_layer, restored)
            for name, parameter in self.model.named_parameters():
                if name in self._requires_grad_state:
                    parameter.requires_grad_(self._requires_grad_state[name])
            self._adapter_installed = False
            self._edit_history.clear()
            if self._attention_mask_hook is not None:
                self._attention_mask_hook.remove()
                self._attention_mask_hook = None

    def _state_config(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "target_layer",
                "act_ratio",
                "merge_strategy",
                "merge_freq",
                "save_freq",
                "merge_weight",
                "density",
                "mask_ratio",
                "edit_lr",
                "n_iter",
                "norm_constraint",
                "alpha",
                "beta",
                "gamma",
            )
        }

    def save_adapter(self, path: Union[str, Path]) -> Path:
        """Persist WISE shards, merged weight, routing state, and mask."""
        adapter = self._get_adapter()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        memory_weight = (
            torch.stack([weight.detach().cpu() for weight in adapter.memory_weight])
            if adapter.memory_weight
            else torch.empty(
                (0, *adapter.new_weight.shape), dtype=adapter.new_weight.dtype
            )
        )
        payload = {
            "format_version": 2,
            "variant": self.variant,
            "config": self._state_config(),
            "base_model_identity": self._base_model_identity,
            "target_layer": self.target_layer,
            "transpose": adapter.transpose,
            "base_weight": adapter.original_layer.weight.detach().cpu(),
            "merged_weight": adapter.layer.weight.detach().cpu(),
            "new_weight": adapter.new_weight.detach().cpu(),
            "memory_weight": memory_weight,
            "weight_mask": (
                adapter.weight_mask.detach().cpu()
                if adapter.weight_mask is not None
                else torch.empty(0)
            ),
            "min_activation_distance": torch.tensor(adapter._min_act),
            "editing_total_count": adapter.editing_total_cnt,
        }
        torch.save(payload, destination)
        return destination

    def load_adapter(self, path: Union[str, Path]) -> None:
        """Load WISE runtime state into an editor created from the base model."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported PyTorch releases
            payload = torch.load(path, map_location="cpu")
        if payload.get("format_version") != 2 or payload.get("variant") != self.variant:
            raise ValueError("Unsupported WISE adapter format")
        if payload.get("config") != self._state_config():
            raise ValueError(
                "WISE adapter routing/training configuration is incompatible"
            )
        if self._adapter_installed:
            if payload.get("base_model_identity") != self._base_model_identity:
                raise ValueError("WISE adapter base-model identity is incompatible")
        else:
            validate_base_model_identity(
                self.model, payload.get("base_model_identity"), checkpoint_label="WISE"
            )
        if payload.get("target_layer") != self.target_layer:
            raise ValueError(
                "WISE target mismatch: "
                f"checkpoint={payload.get('target_layer')!r}, "
                f"editor={self.target_layer!r}"
            )

        self._ensure_adapter()
        adapter = self._get_adapter()
        if bool(payload.get("transpose")) != adapter.transpose:
            raise ValueError("WISE checkpoint projection orientation is incompatible")
        device = adapter.layer.weight.device
        dtype = adapter.layer.weight.dtype
        expected_shape = adapter.layer.weight.shape
        for name in ("base_weight", "merged_weight", "new_weight"):
            if payload[name].shape != expected_shape:
                raise ValueError(
                    f"WISE {name} shape {tuple(payload[name].shape)} does not match "
                    f"target shape {tuple(expected_shape)}"
                )
        memory_weight = payload["memory_weight"]
        if memory_weight.ndim != len(expected_shape) + 1 or tuple(
            memory_weight.shape[1:]
        ) != tuple(expected_shape):
            raise ValueError("WISE memory shard dimensions are incompatible")

        adapter.original_layer.weight.data.copy_(
            payload["base_weight"].to(device=device, dtype=dtype)
        )
        adapter._original_weight_cpu = payload["base_weight"].detach().cpu().clone()
        adapter.layer.weight = nn.Parameter(
            payload["merged_weight"].to(device=device, dtype=dtype),
            requires_grad=False,
        )
        adapter.weight = adapter.layer.weight
        adapter.new_weight = nn.Parameter(
            payload["new_weight"].to(device=device, dtype=dtype),
            requires_grad=True,
        )
        adapter.memory_weight = [
            weight.to(device=device, dtype=dtype) for weight in memory_weight
        ]
        weight_mask = payload["weight_mask"]
        adapter.weight_mask = (
            weight_mask.to(device=device) if weight_mask.numel() else None
        )
        adapter._min_act = float(payload["min_activation_distance"].item())
        adapter.editing_total_cnt = int(payload["editing_total_count"])

    def save_model(self, output_dir=None, _internal_call=False):
        """Save a loadable base checkpoint plus the WISE runtime memory."""
        if not self._adapter_installed:
            return super().save_model(output_dir, _internal_call=_internal_call)

        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_adapter(output_path / "wise_adapter.pt")

        adapter = self._get_adapter()
        base_layer = copy.deepcopy(self._original_layer).to(adapter.layer.weight.device)
        replace_module(self.model, self.target_layer, base_layer)
        try:
            return super().save_model(output_path, _internal_call=_internal_call)
        finally:
            replace_module(self.model, self.target_layer, adapter)

    # ------------------------------------------------------------------
    # Edit interface
    # ------------------------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        self._ensure_adapter()

        results: Dict[str, Any] = {
            "success": True,
            "variant": self.variant,
            "edited_count": 0,
            "successful_request_indices": [],
            "metrics": {},
        }

        for request_index, request in enumerate(requests):
            adapter = self._get_adapter()
            transaction_state = self._snapshot_transaction_state(adapter)
            try:
                scope_metrics = self._apply_wise_edit(request)
                results["edited_count"] += 1
                results["successful_request_indices"].append(request_index)
                if scope_metrics is not None:
                    results["metrics"].setdefault("scope_training", []).append(
                        scope_metrics
                    )
            except Exception as e:
                self._restore_transaction_state(adapter, transaction_state)
                logger.error(
                    "WISE edit failed for '%s': %s", request.prompt, e, exc_info=True
                )
                results["success"] = False

        return results

    @staticmethod
    def _scope_training_negatives(request: EditRequest) -> List[str]:
        """Validate explicit training negatives while keeping probes held out."""

        def normalize(prompt):
            return " ".join(prompt.split()).casefold()

        def probe_prompts(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from probe_prompts(item)
            elif isinstance(value, dict):
                if "prompt" in value:
                    yield from probe_prompts(value["prompt"])
                else:
                    for group in value.values():
                        yield from probe_prompts(group)

        negatives = request.scope_training_negatives or []
        if not isinstance(negatives, list) or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in negatives
        ):
            raise ValueError("WISE scope_training_negatives must be non-empty strings")
        held_out = {
            normalize(prompt)
            for group in (
                request.locality_inputs,
                request.rephrase_prompts,
                request.portability_inputs,
            )
            for prompt in probe_prompts(group)
        }
        training = {normalize(prompt) for prompt in negatives}
        if training & held_out:
            raise ValueError(
                "WISE scope training negatives overlap held-out evaluation probes"
            )
        if normalize(request.prompt) in training:
            raise ValueError("WISE scope training negatives overlap the edit prompt")
        return list(dict.fromkeys(negatives))

    def _apply_wise_edit(self, request: EditRequest):
        model = self.model
        tokenizer = self.tokenizer
        device = self._input_device(model)
        adapter = self._get_adapter()
        negatives = self._scope_training_negatives(request)
        if not negatives:
            logger.warning(
                "WISE has no scope_training_negatives: using in-scope-only activation "
                "training; out-of-scope preservation is not trained. Evaluation probes "
                "remain held out."
            )

        prompt_ids = tokenizer(
            request.prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        separator = "" if request.prompt.endswith((" ", "\n", "\t")) else " "
        target_ids = tokenizer(
            separator + request.target_new,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        if target_ids.numel() == 0:
            raise ValueError("WISE target_new tokenized to an empty sequence")
        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = torch.ones_like(full_ids)
        labels = torch.full_like(full_ids, -100)
        labels[:, prompt_ids.shape[1] :] = target_ids

        if negatives:
            loc_ids = tokenizer(
                negatives,
                return_tensors="pt",
                padding="max_length",
                max_length=full_ids.shape[1],
                truncation=True,
            )
            full_ids = torch.cat([full_ids, loc_ids["input_ids"].to(device)], dim=0)
            attention_mask = torch.cat(
                [attention_mask, loc_ids["attention_mask"].to(device)], dim=0
            )
            labels = torch.cat(
                [
                    labels,
                    torch.full_like(loc_ids["input_ids"].to(device), -100),
                ],
                dim=0,
            )

        tokens = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        self._edit_history.append({k: v.detach().cpu() for k, v in tokens.items()})

        adapter.editing = True
        adapter.set_parameter_tunable()

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.generate_activation_mask()

        optimizer = torch.optim.SGD(
            [adapter.new_weight],
            lr=self.edit_lr,
            weight_decay=1e-5,
        )

        try:
            for _ in range(self.n_iter):
                outputs = model(**tokens)

                logits = outputs.logits
                shift_logits = logits[:1, :-1, :].contiguous()
                shift_labels = labels[:1, 1:].to(logits.device).contiguous()

                loss_fct = CrossEntropyLoss(reduction="none")
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(1, -1)

                label_mask = shift_labels.ne(-100)
                if not label_mask.any():
                    raise ValueError("WISE edit batch has no supervised target tokens")
                ft_loss = (per_token_loss * label_mask).sum() / label_mask.sum()

                act_loss = torch.tensor(0.0, device=ft_loss.device)
                orig_out = adapter.original_layer_output
                new_out = adapter.new_weight_layer_output
                if orig_out is not None and new_out is not None:
                    in_scope = _activation_distance(
                        orig_out[:1], new_out[:1], attention_mask[:1]
                    )
                    if orig_out.shape[0] > 1:
                        out_scope = _activation_distance(
                            orig_out[1:], new_out[1:], attention_mask[1:]
                        )
                        act_loss = (
                            F.relu(out_scope - in_scope + self.gamma)
                            + F.relu(out_scope - self.alpha)
                            + F.relu(self.beta - in_scope)
                        ).to(ft_loss.device)
                    else:
                        act_loss = F.relu(self.beta - in_scope).to(ft_loss.device)

                loss = ft_loss + act_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                adapter.mask_new_weight_gradient()
                optimizer.step()

                if self.norm_constraint is not None:
                    with torch.no_grad():
                        orig_w = adapter._original_weight_cpu.to(
                            device=adapter.new_weight.device,
                            dtype=adapter.new_weight.dtype,
                        )
                        adapter.new_weight.clamp_(
                            orig_w - self.norm_constraint,
                            orig_w + self.norm_constraint,
                        )

            with torch.no_grad():
                model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
            adapter.save_editing_activation()
        finally:
            adapter.editing = False
        adapter.editing_total_cnt += 1

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.save_weight()
            logger.info("WISE: saved weight snapshot to memory")

        if adapter.editing_total_cnt % self.merge_freq == 0:
            adapter.merge_weight()
            logger.info(
                "WISE: merged weights using '%s' strategy",
                self.merge_strategy,
            )
        return {
            "mode": "independent_negatives" if negatives else "in_scope_only",
            "training_negative_count": len(negatives),
            "uses_evaluation_probes": False,
        }

        logger.info(
            "WISE edit applied: '%s' -> '%s' (loss=%.4f, total_edits=%d)",
            request.prompt,
            request.target_new,
            loss.item(),
            adapter.editing_total_cnt,
        )
