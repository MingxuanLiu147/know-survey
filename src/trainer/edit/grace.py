"""
GRACE 知识编辑器
================

实现 GRACE (Lifelong Model Editing with Discrete Key-Value Adaptors) 方法。
GRACE 通过在目标层插入可学习的 key-value 适配器实现持续知识编辑。

参考论文: T. Hartvigsen et al., "Aging with GRACE: Lifelong Model Editing
with Discrete Key-Value Adaptors", NeurIPS 2023
https://arxiv.org/abs/2211.11031

核心思想：
1. 在目标层插入 GRACEAdapter，维护 (key, value, epsilon) 码本
2. 前向传播时，计算输入与码本中所有 key 的欧氏距离
3. 若距离小于 epsilon 阈值，替换输出为对应 value
4. 编辑时训练新的 key-value 对并加入码本

本实现复现论文/EasyEdit 的离散 codebook、epsilon 冲突处理和 value-only
optimization 核心路径。moving-average epsilon、early-stop 等可选变体未启用；
运行状态通过独立 ``grace_adapter.pt`` 保存，而不是伪装成普通模型权重。
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.utils.nethook import get_module, replace_module

logger = logging.getLogger(__name__)


def _euclidean_distance(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance between *key* rows and *query* rows."""
    if key.dim() < 2:
        key = key.unsqueeze(0)
    orig_dtype = key.dtype
    return torch.cdist(key.float(), query.float(), p=2).to(orig_dtype)


def _perturb_values(
    value: torch.Tensor,
    num_pert: int,
    device: torch.device,
) -> torch.Tensor:
    """Create noisy copies of a value vector for adversarial training.

    The first copy retains the original value (zero noise).
    """
    noise = torch.normal(
        0,
        1,
        (num_pert, *value.shape),
        device=device,
        dtype=value.dtype,
    )
    noise[0] = 0
    # Keep the request batch dimension stable.  Expanding it to ``num_pert``
    # makes the language-model labels dimensionally incompatible.
    return value + noise.mean(dim=0)


# ---------------------------------------------------------------------------
# GRACEAdapter
# ---------------------------------------------------------------------------


class GRACEAdapter(nn.Module):
    """Discrete key-value codebook adapter wrapping a single linear layer.

    Maintains a growing codebook of (key, value, epsilon) entries.  During
    forward pass, if the input activation at the designated token position
    falls within epsilon distance of a stored key, the layer output at that
    position is replaced with the corresponding learned value.
    """

    def __init__(
        self,
        layer: nn.Module,
        init_epsilon: float = 0.1,
        num_pert: int = 1,
        replacement: str = "replace_last",
        val_init: str = "warm",
        val_train: str = "sgd",
        transpose: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.weight = layer.weight
        self.init_epsilon = init_epsilon
        self.num_pert = num_pert
        self.replacement = replacement
        self.val_init = val_init
        self.val_train = val_train
        self.device = layer.weight.device

        if replacement not in {"replace_last", "replace_all", "replace_prompt"}:
            raise ValueError(f"Unsupported GRACE replacement strategy: {replacement}")
        if val_init not in {"warm", "cold"}:
            raise ValueError(f"Unsupported GRACE value initialization: {val_init}")
        if val_train not in {"sgd", "adv"}:
            raise ValueError(f"Unsupported GRACE value training mode: {val_train}")

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]

        self.key_id: int = -1
        self._training_edit: bool = False
        self.edit_label: Optional[torch.Tensor] = None
        self.edit_id: int = 0
        self.iter: int = 0
        self.chosen_key: Optional[torch.Tensor] = None

        # Codebook (populated lazily on first edit)
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[nn.Parameter] = None
        self.epsilons: Optional[torch.Tensor] = None
        self.key_labels: List = []
        self.edit_ids: List[int] = []

    @property
    def has_keys(self) -> bool:
        return self.keys is not None and self.keys.nelement() > 0

    @property
    def codebook_size(self) -> int:
        return self.keys.shape[0] if self.has_keys else 0

    # -- codebook management ------------------------------------------------

    def _init_key_value(self, query: torch.Tensor, value: nn.Parameter):
        """Initialise the codebook with the first key-value pair."""
        self.keys = query.detach()
        self.values = value
        self.epsilons = torch.tensor(
            [self.init_epsilon],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.key_labels = [self.edit_label]
        self.edit_ids = [self.edit_id]

    def _add_key(self, new_key: torch.Tensor, new_value: nn.Parameter):
        """Append a new key-value-epsilon entry to the codebook."""
        self.keys = torch.vstack([self.keys, new_key.detach()])
        self.values = nn.Parameter(
            torch.vstack([self.values.data, new_value.data]),
            requires_grad=True,
        )
        new_eps = torch.tensor(
            [self.init_epsilon],
            device=self.device,
            dtype=torch.float32,
        )
        self.epsilons = torch.cat([self.epsilons, new_eps])
        # Keys, values, labels, and edit ids form positional records.  They
        # must all be appended in the same order for rollback and conflict
        # resolution to remain correct after sequential edits.
        self.key_labels = self.key_labels + [self.edit_label]
        self.edit_ids = self.edit_ids + [self.edit_id]

    def _label_match(self, label_a, label_b) -> bool:
        if label_a is None or label_b is None:
            return False
        return label_a.float().mean().item() == label_b.float().mean().item()

    def delete_key(self, edit_id: int):
        """Remove a codebook entry by its *edit_id*."""
        if not self.has_keys or edit_id not in self.edit_ids:
            return
        i = self.edit_ids.index(edit_id)
        self.keys = torch.cat([self.keys[:i], self.keys[i + 1 :]])
        self.values = nn.Parameter(
            torch.cat([self.values.data[:i], self.values.data[i + 1 :]]),
            requires_grad=True,
        )
        self.epsilons = torch.cat([self.epsilons[:i], self.epsilons[i + 1 :]])
        self.key_labels = self.key_labels[:i] + self.key_labels[i + 1 :]
        self.edit_ids = self.edit_ids[:i] + self.edit_ids[i + 1 :]

    # -- forward ------------------------------------------------------------

    def forward(self, *args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise TypeError("GRACEAdapter expects one hidden-state tensor input")
        layer_out = self.layer(*args)

        if not self._training_edit and not self.has_keys:
            return layer_out

        seq_len = args[0].shape[1]
        token_to_edit = (
            (seq_len - 1) if self.key_id == -1 else min(self.key_id, seq_len - 1)
        )
        query = args[0][:, token_to_edit, :]
        if self._training_edit and query.shape[0] != 1:
            raise ValueError(
                "GRACE codebook updates currently require batch size 1; "
                f"received {query.shape[0]}"
            )

        if self.val_init == "cold":
            new_value = nn.Parameter(
                torch.rand(
                    1,
                    self.value_shape,
                    device=self.device,
                    dtype=layer_out.dtype,
                ),
                requires_grad=True,
            )
        else:  # warm
            new_value = nn.Parameter(
                layer_out[:, token_to_edit, :].detach(),
                requires_grad=True,
            )

        # -- codebook update (first iteration of a new edit only) --
        if not self.has_keys:
            self._init_key_value(query, new_value)
        elif self._training_edit and self.iter == 0:
            dists = _euclidean_distance(query, self.keys).view(-1, query.shape[0])
            smallest_dist, nearest_key = dists.min(0)

            if smallest_dist > (self.init_epsilon + self.epsilons[nearest_key]):
                self._add_key(query, new_value)
            elif not self._label_match(
                self.edit_label,
                self.key_labels[nearest_key],
            ):
                self._add_key(query, new_value)
                self.epsilons[nearest_key] = (smallest_dist / 2) - 1e-5
                self.epsilons[-1] = smallest_dist / 2
            else:
                if smallest_dist > self.epsilons[nearest_key]:
                    self.epsilons[nearest_key] = smallest_dist

        # -- distance-based output replacement --
        dists = _euclidean_distance(query, self.keys).view(-1, query.shape[0])
        if dists.nelement() == 0:
            return layer_out

        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        chosen_value = chosen_value.to(dtype=layer_out.dtype)
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        if self.val_train == "adv" and self._training_edit:
            chosen_value = _perturb_values(chosen_value, self.num_pert, self.device)

        if self.replacement == "replace_all":
            layer_out = torch.where(
                (smallest_dist <= eps).view(-1, 1, 1),
                chosen_value.unsqueeze(1).expand_as(layer_out),
                layer_out,
            )
        elif self.replacement == "replace_last":
            layer_out = layer_out.clone()
            layer_out[:, token_to_edit] = torch.where(
                smallest_dist <= eps,
                chosen_value,
                layer_out[:, token_to_edit],
            )
        elif self.replacement == "replace_prompt":
            layer_out = layer_out.clone()
            cond = (smallest_dist <= eps).view(-1, 1, 1)
            target_slice = layer_out[:, :token_to_edit]
            layer_out[:, :token_to_edit] = torch.where(
                cond.expand_as(target_slice),
                chosen_value.unsqueeze(1).expand_as(target_slice),
                target_slice,
            )

        return layer_out


# ---------------------------------------------------------------------------
# GRACEEditor
# ---------------------------------------------------------------------------


class GRACEEditor(EditTrainer):
    """GRACE 知识编辑器

    通过离散 key-value 适配器实现持续知识编辑。
    每次编辑仅向码本添加一个 key-value 对，支持无限次连续编辑。

    Paper: https://arxiv.org/abs/2211.11031
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        inner_params: Union[str, List[str]] = "auto",
        edit_lr: float = 1e-1,
        n_iter: int = 40,
        eps: float = 0.1,
        num_pert: int = 1,
        val_init: str = "warm",
        val_train: str = "sgd",
        val_reg: float = 0.0,
        dropout: float = 0.0,
        replacement: str = "replace_last",
        *args,
        **kwargs,
    ):
        """
        Args:
            layers: 编辑的目标层索引列表（兼容基类接口）
            inner_params: 目标层路径，如 ``"model.layers.9.mlp.down_proj"``
            edit_lr: 码本 value 优化学习率
            n_iter: 每次编辑的训练迭代次数
            eps: 欧氏距离阈值 (epsilon)
            num_pert: 对抗训练扰动数量
            val_init: value 初始化方式 (``"warm"`` | ``"cold"``)
            val_train: value 训练方式 (``"sgd"`` | ``"adv"``)
            val_reg: value 正则化系数
            dropout: dropout 比率（预留扩展）
            replacement: 输出替换策略
                (``"replace_last"`` | ``"replace_all"`` | ``"replace_prompt"``)
        """
        super().__init__(layers=layers, *args, **kwargs)

        if isinstance(inner_params, list):
            if len(inner_params) != 1:
                raise ValueError("GRACE supports exactly one inner parameter")
            inner_params = inner_params[0]
        self.inner_params = inner_params
        self.edit_lr = edit_lr
        self.n_iter = n_iter
        self.eps = eps
        self.num_pert = num_pert
        self.val_init = val_init
        self.val_train = val_train
        self.val_reg = val_reg
        self.dropout = dropout
        self.replacement = replacement
        if n_iter <= 0:
            raise ValueError("GRACE n_iter must be positive")
        if num_pert <= 0:
            raise ValueError("GRACE num_pert must be positive")

        self.target_layer = self._resolve_target_layer(inner_params)

        self._adapter_installed: bool = False
        self._original_layer: Optional[nn.Module] = None
        self._requires_grad_state: Dict[str, bool] = {}
        self._edit_count: int = 0

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
        if self._get_layers_container(self.model) is None:
            raise ValueError("Cannot auto-resolve a GRACE target layer")
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return f"transformer.h.{layer_idx}.mlp.c_fc"
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return f"model.layers.{layer_idx}.mlp.down_proj"
        if hasattr(self.model, "gpt_neox"):
            return f"gpt_neox.layers.{layer_idx}.mlp.dense_4h_to_h"
        raise ValueError(
            "Cannot auto-resolve a GRACE target for this model architecture; "
            "set trainer.method_args.inner_params explicitly"
        )

    # -- adapter lifecycle --------------------------------------------------

    def _ensure_adapter(self):
        if self._adapter_installed:
            return

        model = self.model
        try:
            target = get_module(model, self.target_layer)
        except LookupError:
            raise ValueError(f"Target layer '{self.target_layer}' not found in model")

        if not isinstance(target, GRACEAdapter):
            from transformers.pytorch_utils import Conv1D

            if not hasattr(target, "weight"):
                raise TypeError(
                    f"GRACE target '{self.target_layer}' has no weight parameter"
                )
            transpose = not isinstance(target, Conv1D)
            adapter = GRACEAdapter(
                layer=target,
                init_epsilon=self.eps,
                num_pert=self.num_pert,
                replacement=self.replacement,
                val_init=self.val_init,
                val_train=self.val_train,
                transpose=transpose,
            ).to(target.weight.device)

            # Keep an actual module copy: reconstructing a Transformers Conv1D
            # through ``__new__`` leaves it uninitialised and breaks reset.
            self._original_layer = copy.deepcopy(target).cpu()
            self._requires_grad_state = {
                name: parameter.requires_grad
                for name, parameter in model.named_parameters()
            }
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            replace_module(model, self.target_layer, adapter)

        self._adapter_installed = True

    def _get_adapter(self) -> GRACEAdapter:
        adapter = get_module(self.model, self.target_layer)
        assert isinstance(adapter, GRACEAdapter)
        return adapter

    def _get_grace_trainable_params(self, adapter: GRACEAdapter) -> List[nn.Parameter]:
        if adapter.values is None:
            raise RuntimeError("GRACE codebook values are not initialized")
        return [adapter.values]

    @staticmethod
    def _clone_transaction_value(value):
        """Clone one mutable adapter value without retaining autograd graphs."""
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        return copy.deepcopy(value)

    def _snapshot_adapter_state(self, adapter: GRACEAdapter) -> Dict[str, Any]:
        """Capture all request-mutable GRACE state for transactional editing."""
        values = None if adapter.values is None else adapter.values.detach().clone()
        values_grad = (
            None
            if adapter.values is None or adapter.values.grad is None
            else adapter.values.grad.detach().clone()
        )
        return {
            "keys": self._clone_transaction_value(adapter.keys),
            "values": values,
            "values_requires_grad": (
                True if adapter.values is None else adapter.values.requires_grad
            ),
            "values_grad": values_grad,
            "epsilons": self._clone_transaction_value(adapter.epsilons),
            "key_labels": [
                self._clone_transaction_value(label) for label in adapter.key_labels
            ],
            "edit_ids": list(adapter.edit_ids),
            "key_id": adapter.key_id,
            "training_edit": adapter._training_edit,
            "edit_label": self._clone_transaction_value(adapter.edit_label),
            "edit_id": adapter.edit_id,
            "iter": adapter.iter,
            "chosen_key": self._clone_transaction_value(adapter.chosen_key),
        }

    def _restore_adapter_state(
        self,
        adapter: GRACEAdapter,
        state: Dict[str, Any],
    ) -> None:
        """Restore a GRACE state snapshot after a failed request."""
        adapter.keys = state["keys"]
        if state["values"] is None:
            adapter.values = None
        else:
            adapter.values = nn.Parameter(
                state["values"],
                requires_grad=state["values_requires_grad"],
            )
            if state["values_grad"] is not None:
                adapter.values.grad = state["values_grad"]
        adapter.epsilons = state["epsilons"]
        adapter.key_labels = state["key_labels"]
        adapter.edit_ids = state["edit_ids"]
        adapter.key_id = state["key_id"]
        adapter._training_edit = state["training_edit"]
        adapter.edit_label = state["edit_label"]
        adapter.edit_id = state["edit_id"]
        adapter.iter = state["iter"]
        adapter.chosen_key = state["chosen_key"]

    def reset_layer(self):
        """Restore the original layer, removing the GRACE adapter."""
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
            self._edit_count = 0

    def save_adapter(self, path: Union[str, Path]) -> Path:
        """Save the discrete codebook needed to resume GRACE inference/editing."""
        adapter = self._get_adapter()
        if not adapter.has_keys or adapter.values is None:
            raise RuntimeError("Cannot save GRACE before at least one successful edit")
        if not (
            adapter.codebook_size
            == len(adapter.key_labels)
            == len(adapter.edit_ids)
            == adapter.values.shape[0]
            == adapter.epsilons.shape[0]
        ):
            raise RuntimeError("GRACE codebook records are not positionally aligned")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "target_layer": self.target_layer,
            "keys": adapter.keys.detach().cpu(),
            "values": adapter.values.detach().cpu(),
            "epsilons": adapter.epsilons.detach().cpu(),
            "label_signatures": torch.tensor(
                [label.float().mean().item() for label in adapter.key_labels],
                dtype=torch.float32,
            ),
            "edit_ids": torch.tensor(adapter.edit_ids, dtype=torch.long),
            "edit_count": self._edit_count,
            "key_id": adapter.key_id,
        }
        torch.save(payload, destination)
        return destination

    def load_adapter(self, path: Union[str, Path]) -> None:
        """Load a GRACE codebook into an editor created from the base model."""
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older supported PyTorch releases
            payload = torch.load(path, map_location="cpu")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported GRACE adapter format")
        if payload.get("target_layer") != self.target_layer:
            raise ValueError(
                "GRACE target mismatch: "
                f"checkpoint={payload.get('target_layer')!r}, "
                f"editor={self.target_layer!r}"
            )

        self._ensure_adapter()
        adapter = self._get_adapter()
        device = adapter.layer.weight.device
        keys = payload["keys"].to(device=device, dtype=adapter.layer.weight.dtype)
        values = payload["values"].to(device=device, dtype=adapter.layer.weight.dtype)
        epsilons = payload["epsilons"].to(device=device, dtype=torch.float32)
        signatures = payload["label_signatures"].view(-1)
        edit_ids = payload["edit_ids"].view(-1)
        count = keys.shape[0]
        if not (
            keys.ndim == 2
            and values.ndim == 2
            and keys.shape[1] == adapter.key_shape
            and values.shape[1] == adapter.value_shape
            and values.shape[0] == count
            and epsilons.numel() == count
            and signatures.numel() == count
            and edit_ids.numel() == count
        ):
            raise ValueError("Invalid or dimensionally incompatible GRACE codebook")

        adapter.keys = keys
        adapter.values = nn.Parameter(values, requires_grad=True)
        adapter.epsilons = epsilons
        adapter.key_labels = [
            torch.tensor([signature.item()], device=device) for signature in signatures
        ]
        adapter.edit_ids = [int(edit_id) for edit_id in edit_ids.tolist()]
        adapter.key_id = int(payload.get("key_id", -1))
        self._edit_count = int(payload.get("edit_count", count))

    def save_model(self, output_dir=None, _internal_call=False):
        """Save a loadable base checkpoint plus the GRACE runtime codebook."""
        if not self._adapter_installed:
            return super().save_model(output_dir, _internal_call=_internal_call)

        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.save_adapter(output_path / "grace_adapter.pt")

        adapter = self._get_adapter()
        base_layer = copy.deepcopy(self._original_layer).to(adapter.layer.weight.device)
        replace_module(self.model, self.target_layer, base_layer)
        try:
            return super().save_model(output_path, _internal_call=_internal_call)
        finally:
            replace_module(self.model, self.target_layer, adapter)

    # -- edit interface -----------------------------------------------------

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        self._ensure_adapter()

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
        }

        for request in requests:
            adapter = self._get_adapter()
            adapter_state = self._snapshot_adapter_state(adapter)
            edit_count = self._edit_count
            try:
                self._apply_grace_edit(request)
                results["edited_count"] += 1
            except Exception as e:
                self._restore_adapter_state(adapter, adapter_state)
                self._edit_count = edit_count
                logger.error(
                    "GRACE edit failed for '%s': %s",
                    request.prompt,
                    e,
                    exc_info=True,
                )
                results["success"] = False

        return results

    def _apply_grace_edit(self, request: EditRequest):
        model = self.model
        tokenizer = self.tokenizer
        device = self._input_device(model)
        adapter = self._get_adapter()

        prompt_tokens = tokenizer(
            request.prompt,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        separator = "" if request.prompt.endswith((" ", "\n", "\t")) else " "
        target_tokens = tokenizer(
            separator + request.target_new,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        if target_tokens.numel() == 0:
            raise ValueError("GRACE target_new tokenized to an empty sequence")
        full_ids = torch.cat([prompt_tokens, target_tokens], dim=1)
        labels = torch.full_like(full_ids, -100)
        labels[:, prompt_tokens.shape[1] :] = target_tokens

        tokens = {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
        }

        key_id = (labels[0] == -100).sum().item() - 1
        adapter.key_id = key_id
        adapter._training_edit = True
        adapter.edit_label = labels.detach()
        adapter.edit_id = self._edit_count

        losses: List[float] = []
        try:
            for i in range(self.n_iter):
                adapter.iter = i
                outputs = model(**tokens)

                if i == 0:
                    optimizer = torch.optim.Adam(
                        self._get_grace_trainable_params(adapter),
                        lr=self.edit_lr,
                    )

                loss = outputs.loss

                if self.val_reg > 0 and adapter.has_keys:
                    loss = loss + self.val_reg * adapter.values.norm()

                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(loss.detach().item())
        finally:
            adapter._training_edit = False
            adapter.key_id = -1
        self._edit_count += 1

        logger.info(
            "GRACE edit applied: '%s' -> '%s' (loss=%.4f, codebook_size=%d)",
            request.prompt,
            request.target_new,
            losses[-1] if losses else float("inf"),
            adapter.codebook_size,
        )
