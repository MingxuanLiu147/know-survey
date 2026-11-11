"""Factor-based MEND for causal LMs, with auditable auxiliary training.

Algorithm: Mitchell et al., Fast Model Editing at Scale (ICLR 2022).
Reference: eric-mitchell/mend @ e04fdb9cc784188906feffeb171025872933a5a8.
This native implementation uses activation/backprop factors, not an SVD of a
full gradient. Its checkpoint container is intentionally distinct from both
Know-Surgery's compact prototype and the original project's Python archive.

The residual factor transform follows the authors' MIT-licensed implementation:
Copyright (c) 2022 Eric Anthony Mitchell. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import unicodedata
from typing import Any, Mapping

import numpy as np
from omegaconf import ListConfig
import torch
from torch import nn
from torch.func import functional_call

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)
from trainer.edit.ft import FTEditor

CHECKPOINT_FORMAT = "know_surgery_factor_mend_v1"
TRAINING_FORMAT = "know_surgery_factor_mend_training_v1"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def request_fingerprint(requests: list[EditRequest]) -> str:
    """Bind an auxiliary run to the complete, ordered edit/probe contents."""
    return _json_digest([asdict(request) for request in requests])


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _parameters_digest(networks: nn.Module, edit_lrs: torch.Tensor) -> str:
    """Hash trainable weights/rates, excluding running normalization buffers."""
    digest = hashlib.sha256()
    tensors = dict(networks.named_parameters())
    tensors["__learned_edit_lrs__"] = edit_lrs
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


class _ResidualFactorLayer(nn.Module):
    """Authors' low-rank, identity-initialized residual map with mode FiLM."""

    def __init__(self, dimension: int, rank: int, modes: int):
        super().__init__()
        width = min(rank, dimension)
        self.u = nn.Parameter(torch.zeros(dimension, width))
        self.v = nn.Parameter(torch.randn(width, dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.mode_scale = nn.Embedding(modes, dimension)
        self.mode_shift = nn.Embedding(modes, dimension)
        nn.init.ones_(self.mode_scale.weight)
        nn.init.zeros_(self.mode_shift.weight)

    def forward(self, inputs: torch.Tensor, mode: int) -> torch.Tensor:
        residual = (inputs @ self.v.T) @ self.u.T + self.bias
        residual = (
            residual * self.mode_scale.weight[mode] + self.mode_shift.weight[mode]
        )
        # clamp has the authors' nonzero derivative at zero; ReLU does not.
        return inputs + residual.clamp(min=0)


class _FactorTransform(nn.Module):
    def __init__(
        self,
        x_dim: int,
        delta_dim: int,
        rank: int,
        n_hidden: int,
        modes: int,
        normalize: bool,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.normalize = normalize
        dimension = x_dim + delta_dim
        self.layers = nn.ModuleList(
            [_ResidualFactorLayer(dimension, rank, modes) for _ in range(n_hidden + 1)]
        )
        self.register_buffer("count", torch.zeros((), dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("m2", torch.zeros(dimension))

    @torch.no_grad()
    def _observe(self, inputs: torch.Tensor) -> None:
        # Parallel Welford merge computes the same sample moments without a
        # Python loop over every token. These buffers are checkpointed.
        count = inputs.shape[0]
        if not count:
            return
        mean = inputs.mean(0)
        m2 = (inputs - mean).square().sum(0)
        total = self.count + count
        shift = mean - self.mean
        self.m2.add_(m2 + shift.square() * (self.count * count / total))
        self.mean.add_(shift * (count / total))
        self.count.copy_(total)

    def forward(
        self,
        activations: torch.Tensor,
        errors: torch.Tensor,
        mode: int,
        *,
        update_stats: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = activations.reshape(-1, self.x_dim).float()
        delta = errors.reshape(-1, self.delta_dim).float()
        keep = x.ne(0).any(-1) & delta.ne(0).any(-1)
        inputs = torch.cat((x[keep], delta[keep]), dim=-1)
        if not len(inputs):
            raise ValueError("MEND has no nonzero activation/error factor pairs")
        if update_stats:
            self._observe(inputs)
        if self.normalize:
            if self.count.item() < 2:
                raise ValueError(
                    "MEND normalization requires at least two observed factors"
                )
            std = (self.m2 / (self.count - 1)).clamp_min(0).sqrt()
            inputs = (inputs - self.mean) / (std + 1e-7)
        for layer in self.layers:
            inputs = layer(inputs, mode)
        return inputs.split((self.x_dim, self.delta_dim), dim=-1)


class MENDFactorEditor(EditTrainer):
    """MEND activation/error editor with separately trained, model-bound weights.

    ``rank`` is the hidden low-rank parameterization of the factor MLP. It is
    not the rank of a pooled gradient. ``inner_params`` must name every edited
    matrix explicitly. Training uses independent validation examples and saves
    only at optimizer boundaries, making resumed sample order unambiguous.
    """

    def __init__(
        self,
        layers=None,
        inner_params=None,
        edit_lr=1e-4,
        rank=1920,
        n_hidden=1,
        normalize=True,
        shared=True,
        cedit=0.1,
        cloc=1.0,
        max_length=512,
        max_locality_samples=1,
        edit_network_path=None,
        require_pretrained=True,
        *args,
        **kwargs,
    ):
        super().__init__(layers=layers, *args, **kwargs)
        if not inner_params or not isinstance(inner_params, (list, tuple, ListConfig)):
            raise ValueError(
                "MEND requires an explicit list of inner_params matrix names"
            )
        self.inner_params = list(inner_params)
        if len(set(self.inner_params)) != len(self.inner_params):
            raise ValueError("MEND inner_params must be unique")
        self._resolve_layer_indices(self.layers)
        selected_modules = {
            id(module): index
            for index in self.layers
            for module in self._get_layers_container(self.model)[index].modules()
        }
        used_layers = set()
        for key, value, lower in (
            ("rank", rank, 1),
            ("n_hidden", n_hidden, 0),
            ("max_length", max_length, 2),
            ("max_locality_samples", max_locality_samples, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"MEND {key} must be an integer >= {lower}")
        if not all(math.isfinite(value) for value in (edit_lr, cedit, cloc)):
            raise ValueError("MEND loss/rate parameters must be finite")
        if edit_lr <= 0 or cedit <= 0 or cloc < 0:
            raise ValueError("MEND edit_lr/cedit must be positive and cloc nonnegative")
        self.initial_edit_lr = float(edit_lr)
        self.rank, self.n_hidden = rank, n_hidden
        self.normalize, self.shared = bool(normalize), bool(shared)
        self.cedit, self.cloc = float(cedit), float(cloc)
        self.max_length, self.max_locality_samples = max_length, max_locality_samples
        self.require_pretrained = require_pretrained
        self.edit_network_path = edit_network_path
        self._pretrained_loaded = False
        self.training_metadata: dict = {}
        self._base_identity = None
        parameters = dict(self.model.named_parameters())
        modules = dict(self.model.named_modules())
        self._targets = {}
        self._shape_groups = defaultdict(list)
        for name in self.inner_params:
            if name not in parameters or not name.endswith(".weight"):
                raise ValueError(f"MEND matrix does not exist: {name}")
            module = modules[name.rsplit(".", 1)[0]]
            weight = parameters[name]
            if id(module) not in selected_modules:
                raise ValueError(f"MEND matrix {name} is outside reviewed layers")
            used_layers.add(selected_modules[id(module)])
            if weight.ndim != 2:
                raise ValueError(f"MEND requires a matrix: {name}")
            if isinstance(module, nn.Linear):
                shape, transposed = (weight.shape[1], weight.shape[0]), False
            elif module.__class__.__name__ == "Conv1D":
                shape, transposed = tuple(weight.shape), True
            else:
                raise ValueError(f"Unsupported MEND projection {type(module).__name__}")
            self._targets[name] = (module, weight, transposed)
            key = f"{shape[0]}x{shape[1]}" if shared else name.replace(".", "_")
            self._shape_groups[key].append((name, shape))
        if used_layers != set(self.layers):
            raise ValueError("MEND reviewed layers and matrix layer coverage differ")
        device = next(iter(self._targets.values()))[1].device
        self.networks = nn.ModuleDict()
        self._network_for = {}
        for key, group in self._shape_groups.items():
            shape = group[0][1]
            self.networks[key] = _FactorTransform(
                *shape,
                rank,
                n_hidden,
                len(group),
                self.normalize,
            ).to(device=device, dtype=torch.float32)
            for mode, (name, _) in enumerate(group):
                self._network_for[name] = (key, mode)
        self.edit_lrs = nn.Parameter(
            torch.full(
                (len(self.inner_params),),
                self.initial_edit_lr,
                device=device,
                dtype=torch.float32,
            )
        )
        if edit_network_path:
            self.load_edit_network(edit_network_path)

    def _config(self) -> dict:
        return {
            "layers": list(self.layers),
            "inner_params": self.inner_params,
            "rank": self.rank,
            "n_hidden": self.n_hidden,
            "normalize": self.normalize,
            "shared": self.shared,
            "edit_lr": self.initial_edit_lr,
            "cedit": self.cedit,
            "cloc": self.cloc,
            "max_length": self.max_length,
            "max_locality_samples": self.max_locality_samples,
            "outer_objective": "first_rephrase_nll_and_attention_mask_locality_kl",
            "factor_dtype": "float32",
        }

    def _safe_max_length(self):
        return FTEditor._safe_max_length(self)

    def _batch(self, request: EditRequest):
        return FTEditor._build_batch(self, [request])

    @contextmanager
    def _frozen_base(self):
        flags = [(p, p.requires_grad) for p in self.model.parameters()]
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            for parameter, _ in flags:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, flag in flags:
                parameter.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode

    def capture_factors(self, request: EditRequest):
        """Return per-token activations and loss derivatives without dense SVD."""
        captured = {}
        handles = []
        with self._frozen_base(), torch.enable_grad():
            try:
                for name, (module, weight, _) in self._targets.items():
                    weight.requires_grad_(True)

                    def hook(_module, inputs, output, name=name):
                        if name in captured:
                            raise ValueError(
                                "MEND target module was called more than once"
                            )
                        captured[name] = (inputs[0].detach(), output)

                    handles.append(module.register_forward_hook(hook))
                output = self.model(**self._batch(request), use_cache=False)
                if not torch.isfinite(output.loss):
                    raise FloatingPointError("MEND inner loss is not finite")
                if set(captured) != set(self.inner_params):
                    raise ValueError("MEND forward did not visit all reviewed matrices")
                errors = torch.autograd.grad(
                    output.loss,
                    [captured[name][1] for name in self.inner_params],
                )
                return {
                    name: (captured[name][0], error.detach())
                    for name, error in zip(self.inner_params, errors)
                }
            finally:
                for handle in handles:
                    handle.remove()

    def _updates(self, request: EditRequest, *, update_stats: bool):
        factors = self.capture_factors(request)
        updates = {}
        for index, name in enumerate(self.inner_params):
            key, mode = self._network_for[name]
            network = self.networks[key]
            device = next(network.parameters()).device
            x, delta = (value.to(device) for value in factors[name])
            x, delta = network(x, delta, mode, update_stats=update_stats)
            # All token factors contribute. Linear and GPT-2 Conv1D store the
            # same gradient with opposite matrix orientations.
            pseudo_gradient = x.T @ delta if self._targets[name][2] else delta.T @ x
            updates[name] = self.edit_lrs[index] * pseudo_gradient
        if any(not torch.isfinite(update).all() for update in updates.values()):
            raise FloatingPointError("MEND produced a non-finite update")
        return updates

    def _virtual(self, updates: Mapping[str, torch.Tensor]):
        return {
            name: self._targets[name][1] + update.to(self._targets[name][1])
            for name, update in updates.items()
        }

    def _locality_requests(self, request: EditRequest):
        raw = request.locality_inputs or {}
        groups = raw.values() if isinstance(raw, dict) else [raw]
        count = 0
        for group in groups:
            for item in [group] if isinstance(group, dict) else group or []:
                prompt = item.get("prompt")
                target = item.get(
                    "ground_truth", item.get("expected", item.get("answer"))
                )
                if isinstance(target, list):
                    target = target[0] if target else None
                if (
                    isinstance(prompt, str)
                    and prompt
                    and isinstance(target, str)
                    and target
                ):
                    yield EditRequest(prompt, prompt, target)
                    count += 1
                    if count >= self.max_locality_samples:
                        return
        if count == 0:
            raise ValueError(
                "MEND auxiliary training/validation requires locality probes"
            )

    def _meta_loss(self, request: EditRequest, *, update_stats: bool):
        if not request.rephrase_prompts:
            raise ValueError("MEND outer loss requires a separate rephrase prompt")
        updates = self._updates(request, update_stats=update_stats)
        virtual = self._virtual(updates)
        outer = replace(request, prompt=request.rephrase_prompts[0])
        batch = self._batch(outer)
        post = functional_call(self.model, virtual, (), {**batch, "use_cache": False})
        edit_loss = post.loss.float()
        predictions = post.logits[:, :-1].argmax(-1)
        gold = batch["labels"][:, 1:].to(predictions.device)
        edit_accuracy = (
            predictions[gold.ne(-100)].eq(gold[gold.ne(-100)]).float().mean()
        )
        locality_losses = []
        for locality in self._locality_requests(request):
            loc_batch = self._batch(locality)
            loc_batch.pop("labels")
            with torch.no_grad():
                original = self.model(**loc_batch, use_cache=False).logits.float()
            edited = functional_call(
                self.model,
                virtual,
                (),
                {**loc_batch, "use_cache": False},
            ).logits.float()
            # Original MEND retains the entire next-token distribution at
            # every valid context position, including prompt positions.
            # This KL mask is intentionally different from answer accuracy.
            mask = loc_batch["attention_mask"].to(original.device).bool()
            original = original[mask].log_softmax(-1)
            edited = edited[mask].log_softmax(-1)
            locality_losses.append(
                (original.exp() * (original - edited)).sum(-1).mean()
            )
        locality_loss = torch.stack(locality_losses).mean()
        total = self.cedit * edit_loss + self.cloc * locality_loss
        if not torch.isfinite(total):
            raise FloatingPointError("MEND outer loss is not finite")
        return total, {
            "edit_nll": float(edit_loss.detach()),
            "locality_kl": float(locality_loss.detach()),
            "edit_token_accuracy": float(edit_accuracy.detach()),
            "total": float(total.detach()),
        }

    def edit(self, requests, **kwargs):
        if isinstance(requests, EditRequest):
            requests = [requests]
        if self.require_pretrained and not self._pretrained_loaded:
            raise RuntimeError(
                "MEND requires a trained factor-network checkpoint for this base model"
            )
        if not requests:
            raise ValueError("MEND requires a nonempty edit request")
        originals = {
            name: weight.detach().cpu().clone()
            for name, (_, weight, _) in self._targets.items()
        }
        try:
            with self._frozen_base():
                for request in requests:
                    updates = self._updates(request, update_stats=False)
                    with torch.no_grad():
                        for name, update in updates.items():
                            weight = self._targets[name][1]
                            weight.add_(update.to(weight))
                            if not torch.isfinite(weight).all():
                                raise FloatingPointError(
                                    "MEND edited weights are not finite"
                                )
            return {
                "success": True,
                "edited_count": len(requests),
                "implementation": "factor_mend",
                "checkpoint_loaded": self._pretrained_loaded,
                "inner_params": self.inner_params,
                "metrics": {},
            }
        except Exception:
            with torch.no_grad():
                for name, original in originals.items():
                    self._targets[name][1].copy_(original)
            raise

    def validate_edit_network(self, requests: list[EditRequest]) -> dict:
        """Measure held-out outer losses without changing normalization or weights."""
        if not requests:
            raise ValueError("MEND validation requires independent examples")
        rows = []
        with self._frozen_base(), torch.no_grad():
            for request in requests:
                _, row = self._meta_loss(request, update_stats=False)
                rows.append(row)
        return {
            "sample_count": len(rows),
            "fingerprint": request_fingerprint(requests),
            "per_example": rows,
            **{key: sum(row[key] for row in rows) / len(rows) for key in rows[0]},
        }

    def _payload(self):
        if self._base_identity is None:
            self._base_identity = build_base_model_identity(self.model)
        return {
            "format": CHECKPOINT_FORMAT,
            "implementation": "factor_mend",
            "config": self._config(),
            "base_model_identity": self._base_identity,
            "networks": self.networks.state_dict(),
            "edit_lrs": self.edit_lrs.detach(),
            "training_metadata": self.training_metadata,
        }

    def save_edit_network(self, path: str | Path):
        """Export only a completed, genuinely optimized and validated network."""
        meta = self.training_metadata
        if (
            meta.get("purpose") != "training"
            or not meta.get("completed")
            or meta.get("optimizer_steps", 0) < 1
            or not meta.get("validation")
        ):
            raise RuntimeError(
                "MEND export requires completed training and independent validation"
            )
        actual_parameters = _parameters_digest(self.networks, self.edit_lrs)
        if meta.get("initial_parameters_sha256") == actual_parameters:
            raise RuntimeError(
                "MEND trainable parameters did not change during training"
            )
        if meta.get("trained_parameters_sha256") != actual_parameters:
            raise RuntimeError(
                "MEND trainable parameters differ from the validated training state"
            )
        if meta.get("trained_network_sha256") != _state_digest(self.networks):
            raise RuntimeError(
                "MEND network state differs from the validated training state"
            )
        payload = self._payload()
        _atomic_save(payload, path)
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        receipt = {
            "schema_version": 1,
            "format": CHECKPOINT_FORMAT,
            "implementation": "factor_mend",
            "checkpoint_sha256": digest.hexdigest(),
            "checkpoint_size_bytes": path.stat().st_size,
            "config": payload["config"],
            "base_model_identity": payload["base_model_identity"],
            "training_metadata": payload["training_metadata"],
            "qualification": "trained_auxiliary_only_not_benchmark_verified",
        }
        sidecar = path.with_name(path.name + ".metadata.json")
        temporary = sidecar.with_name(sidecar.name + ".tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, sidecar)
        self.edit_network_path = str(path)

    def _load_payload(self, payload):
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                "Not a native factor-MEND checkpoint; compact/official Python archives cannot be cross-loaded"
            )
        if payload.get("config") != self._config():
            raise ValueError(
                "MEND checkpoint configuration differs from reviewed parameters"
            )
        validate_base_model_identity(
            self.model, payload["base_model_identity"], checkpoint_label="MEND"
        )
        states = payload.get("networks", {})
        if any(
            not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
            for value in states.values()
        ):
            raise ValueError(
                "MEND checkpoint contains invalid/non-finite network tensors"
            )
        lrs = payload.get("edit_lrs")
        if (
            not isinstance(lrs, torch.Tensor)
            or lrs.shape != self.edit_lrs.shape
            or not torch.isfinite(lrs).all()
        ):
            raise ValueError("MEND checkpoint has invalid learned edit rates")
        self.networks.load_state_dict(states, strict=True)
        with torch.no_grad():
            self.edit_lrs.copy_(lrs)
        self._base_identity = payload["base_model_identity"]
        self.training_metadata = payload.get("training_metadata", {})

    def load_edit_network(self, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        meta = payload.get("training_metadata", {}) if isinstance(payload, dict) else {}
        if (
            meta.get("purpose") != "training"
            or not meta.get("completed")
            or meta.get("optimizer_steps", 0) < 1
            or not meta.get("validation")
        ):
            raise ValueError(
                "MEND checkpoint lacks completed training/validation evidence"
            )
        self._load_payload(payload)
        actual_parameters = _parameters_digest(self.networks, self.edit_lrs)
        if (
            meta.get("initial_parameters_sha256") == actual_parameters
            or meta.get("trained_parameters_sha256") != actual_parameters
        ):
            raise ValueError(
                "MEND checkpoint lacks verified trainable-parameter change"
            )
        if meta.get("trained_network_sha256") != _state_digest(self.networks):
            raise ValueError(
                "MEND checkpoint network state differs from the validated training state"
            )
        self._pretrained_loaded = True
        self.edit_network_path = str(path)

    def train_edit_network(
        self,
        train_requests: list[EditRequest],
        *,
        validation_requests: list[EditRequest],
        max_steps: int,
        batch_size: int = 10,
        lr: float = 1e-6,
        lr_lr: float = 1e-4,
        grad_clip: float = 100.0,
        seed: int = 42,
        checkpoint_path=None,
        resume_from=None,
        save_every: int = 100,
        stop_after_steps=None,
        data_provenance=None,
        purpose="training",
        event_callback=None,
    ) -> dict:
        """Train with exact optimizer-boundary resume and disjoint validation.

        ``batch_size`` is effective examples per update. Examples backpropagate
        one at a time, so increasing it does not retain all their graphs. A
        constant learning rate is explicit; there is no hidden scheduler.
        """
        if purpose not in {"training", "calibration"}:
            raise ValueError("MEND purpose must be training or calibration")
        if not train_requests or not validation_requests:
            raise ValueError("MEND requires nonempty training and validation splits")
        for value in (max_steps, batch_size, save_every):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "MEND step and batch budgets must be positive integers"
                )
        if stop_after_steps is not None and (
            type(stop_after_steps) is not int or stop_after_steps < 1
        ):
            raise ValueError("MEND stop_after_steps must be a positive integer or null")
        if not all(
            math.isfinite(value) and value > 0 for value in (lr, lr_lr, grad_clip)
        ):
            raise ValueError(
                "MEND optimizer rates and clip must be positive and finite"
            )

        def normalized(text):
            return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

        def probe_prompts(value):
            if isinstance(value, list):
                return [text for item in value for text in probe_prompts(item)]
            if isinstance(value, dict):
                if isinstance(value.get("prompt"), str):
                    return [value["prompt"]]
                return [text for item in value.values() for text in probe_prompts(item)]
            return []

        def prompts(rows):
            return {
                normalized(prompt)
                for row in rows
                for prompt in [
                    row.prompt,
                    *(row.rephrase_prompts or []),
                    *probe_prompts(row.locality_inputs),
                    *probe_prompts(row.portability_inputs),
                ]
            }

        def subjects(rows):
            return {normalized(row.subject) for row in rows} - {""}

        if prompts(train_requests) & prompts(validation_requests) or subjects(
            train_requests
        ) & subjects(validation_requests):
            raise ValueError(
                "MEND training/validation prompts, probes or subjects overlap"
            )
        for request in [*train_requests, *validation_requests]:
            if not request.rephrase_prompts:
                raise ValueError("MEND training/validation requires rephrase prompts")
            list(self._locality_requests(request))
        source_root = Path(__file__).resolve().parents[2]
        code_identity = {
            str(path.relative_to(source_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("ft.py").resolve(),
                source_root / "data/utils.py",
            )
        }
        contract = {
            "max_steps": max_steps,
            "purpose": purpose,
            "code_identity": code_identity,
            "train_fingerprint": request_fingerprint(train_requests),
            "validation_fingerprint": request_fingerprint(validation_requests),
            "batch_size": batch_size,
            "lr": lr,
            "lr_lr": lr_lr,
            "grad_clip": grad_clip,
            "seed": seed,
            "scheduler": "constant",
            "source_sha256": {
                key: value["sha256"]
                for key, value in (data_provenance or {}).get("inputs", {}).items()
            },
        }
        optimizer = torch.optim.Adam(
            [
                {"params": self.networks.parameters(), "lr": lr},
                {"params": [self.edit_lrs], "lr": lr_lr},
            ]
        )
        generator = torch.Generator().manual_seed(seed)
        progress = {
            "step": 0,
            "cursor": 0,
            "epoch": 0,
            "order": torch.randperm(len(train_requests), generator=generator).tolist(),
            "history": [],
        }
        if resume_from:
            saved = torch.load(resume_from, map_location="cpu", weights_only=True)
            if (
                saved.get("format") != TRAINING_FORMAT
                or saved.get("contract") != contract
            ):
                raise ValueError(
                    "MEND resume data/optimizer contract differs from checkpoint"
                )
            cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if len(saved["cuda_rng"]) != cuda_count:
                raise ValueError(
                    "MEND CUDA resume requires the saved visible-device topology"
                )
            self._load_payload(saved["editor"])
            optimizer.load_state_dict(saved["optimizer"])
            progress = saved["progress"]
            if progress["step"] > max_steps:
                raise ValueError("MEND resumed progress exceeds requested step budget")
            generator.set_state(saved["sampler_rng"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            np_state = saved["numpy_rng"]
            np.random.set_state(
                (np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:])
            )
            if saved["cuda_rng"]:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            if event_callback is not None:
                event_callback("resume_loaded", {"step": progress["step"]})
        else:
            self._base_identity = build_base_model_identity(self.model)
            self.training_metadata = {
                "completed": False,
                "purpose": purpose,
                "optimizer_steps": 0,
                "initial_network_sha256": _state_digest(self.networks),
                "initial_parameters_sha256": _parameters_digest(
                    self.networks, self.edit_lrs
                ),
                "training_contract": contract,
                "data_provenance": data_provenance or {},
            }

        def save_checkpoint():
            if checkpoint_path:
                np_state = np.random.get_state()
                _atomic_save(
                    {
                        "format": TRAINING_FORMAT,
                        "editor": self._payload(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None,
                        "contract": contract,
                        "progress": progress,
                        "sampler_rng": generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "python_rng": random.getstate(),
                        "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:]),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else [],
                    },
                    checkpoint_path,
                )
                if event_callback is not None:
                    event_callback("checkpoint_saved", {"step": progress["step"]})

        start_step = progress["step"]
        with self._frozen_base():
            while progress["step"] < max_steps:
                if (
                    stop_after_steps is not None
                    and progress["step"] - start_step >= stop_after_steps
                ):
                    break
                if event_callback is not None:
                    event_callback("step_started", {"step": progress["step"] + 1})
                optimizer.zero_grad(set_to_none=True)
                records = []
                for _ in range(batch_size):
                    if progress["cursor"] == len(progress["order"]):
                        progress["order"] = torch.randperm(
                            len(train_requests), generator=generator
                        ).tolist()
                        progress["cursor"] = 0
                        progress["epoch"] += 1
                    index = progress["order"][progress["cursor"]]
                    loss, record = self._meta_loss(
                        train_requests[index], update_stats=True
                    )
                    (loss / batch_size).backward()
                    records.append(record)
                    progress["cursor"] += 1
                    if event_callback is not None:
                        event_callback(
                            "example_completed",
                            {
                                "step": progress["step"] + 1,
                                "example_index": index,
                                "micro_step": len(records),
                                **record,
                            },
                        )
                parameters = [*self.networks.parameters(), self.edit_lrs]
                nn.utils.clip_grad_norm_(parameters, grad_clip, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in parameters):
                    raise FloatingPointError(
                        "MEND optimizer produced a non-finite parameter"
                    )
                progress["step"] += 1
                progress["history"].append(
                    {
                        "step": progress["step"],
                        **{
                            key: sum(row[key] for row in records) / len(records)
                            for key in records[0]
                        },
                    }
                )
                self.training_metadata["optimizer_steps"] = progress["step"]
                if event_callback is not None:
                    event_callback("optimizer_completed", {"step": progress["step"]})
                if progress["step"] % save_every == 0:
                    save_checkpoint()
            if event_callback is not None:
                event_callback("validation_started", {"step": progress["step"]})
            validation = self.validate_edit_network(validation_requests)
            if event_callback is not None:
                event_callback("validation_completed", {"step": progress["step"]})
        self.training_metadata.update(
            completed=progress["step"] == max_steps,
            optimizer_steps=progress["step"],
            requested_steps=max_steps,
            trained_network_sha256=_state_digest(self.networks),
            trained_parameters_sha256=_parameters_digest(self.networks, self.edit_lrs),
            validation=validation,
            history=progress["history"],
        )
        self._pretrained_loaded = bool(
            self.training_metadata["completed"]
            and purpose == "training"
            and self.training_metadata["initial_parameters_sha256"]
            != self.training_metadata["trained_parameters_sha256"]
        )
        save_checkpoint()
        return self.training_metadata
