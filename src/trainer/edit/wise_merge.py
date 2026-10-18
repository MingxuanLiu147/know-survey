"""WISE-Merge with an explicit, auditable EasyEdit-derived algorithm contract.

Reference: zjunlp/EasyEdit@14cea8245f06715684592ab55184939b99d70784,
``easyeditor/models/wise/WISE.py``, ``utils.py`` and ``merge/gta.py`` (MIT).
The upstream MIT notice is retained in LICENSES/EasyEdit.txt.

This is the WISE-Merge profile (retrieve=False, replay=False), not WISE-Retrieve.
The immutable main memory, SGD side memory, random masked subspaces, routing
margin objective, template expansion and base-relative TIES merge follow that
reference. Padding, per-row routing, held-out probe isolation, transactions and
safe persistence are repository adaptations. See docs/editing.md for
usage and compatibility limits; no benchmark reproduction is implied.
"""

from __future__ import annotations

import copy
import string
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

from trainer.edit.wise import WISEAdapter, WISEEditor, _activation_distance
from trainer.edit.base import EditTrainer
from trainer.edit.utils.nethook import replace_module


UPSTREAM_REVISION = "14cea8245f06715684592ab55184939b99d70784"


def ties_merge(weights, base, tensors, density=0.53):
    """Magnitude trim, weighted sign election and disjoint weighted mean.

    This follows EasyEdit's GTA('magnitude', 'sum', normalize=True), including
    exact top-k selection (a quantile threshold can retain too many tied values).
    """
    if not 0 < density <= 1:
        raise ValueError("WISE-Merge TIES density must be in (0, 1]")
    if len(weights) != len(tensors) or not tensors:
        raise ValueError("WISE-Merge requires one coefficient per nonempty shard")
    deltas = []
    for tensor in tensors:
        delta = tensor.to(base) - base
        if density < 1:
            k = int(density * delta.numel())
            if k < 1:
                raise ValueError("WISE-Merge TIES density would remove every value")
            mask = torch.zeros_like(delta).view(-1)
            indices = delta.abs().reshape(-1).float().topk(k).indices
            mask[indices] = 1
            delta = delta * mask.reshape_as(delta)
        deltas.append(delta)
    stacked = torch.stack(deltas)
    coefficients = torch.as_tensor(weights, dtype=base.dtype, device=base.device)
    coefficients = coefficients.reshape(-1, *([1] * base.ndim))
    weighted = stacked * coefficients
    majority = torch.where(weighted.sum(0) >= 0, 1, -1)
    agrees = weighted.sign() == majority
    divisor = (coefficients * agrees).sum(0)
    divisor = torch.where(divisor == 0, 1, divisor)
    return base + (weighted * agrees).sum(0) / divisor


def positive_mean(values):
    """EasyEdit's mean over positive margin violations (zero if all satisfied)."""
    active = values > 0
    return values[active].mean() if active.any() else values.sum() * 0


def activation_margin(in_scope, out_scope, alpha, beta, gamma):
    """WISE routing objective with each training example retained separately."""
    return (
        positive_mean(out_scope[:, None] - in_scope[None, :] + gamma)
        + positive_mean(out_scope - alpha)
        + positive_mean(beta - in_scope)
    )


class WISEMergeAdapter(WISEAdapter):
    """One merged side memory plus the active shard; no retrieval memory bank."""

    activation_name = "linear"
    positive_rows = 1

    def distance(self, a, b, mask=None, *, per_sample=False, inference=False):
        act = ACT2FN[self.activation_name]
        a, b = act(a), act(b)
        # Upstream euc uses the largest activation for long generation inputs.
        # Apply its rule per row and to valid tokens only; never mix batch rows.
        if inference and a.ndim > 2:
            distance = torch.norm(a - b, dim=-1)
            valid = torch.ones_like(distance, dtype=torch.bool)
            if mask is not None:
                valid = mask[:, -distance.shape[-1] :].to(distance.device).bool()
            lengths = valid.sum(-1)
            average = (distance * valid).sum(-1) / lengths.clamp_min(1)
            maximum = distance.masked_fill(~valid, -torch.inf).amax(-1)
            result = torch.where(lengths > 100, maximum, average)
            return result if per_sample else result.mean()
        return _activation_distance(a, b, mask, per_sample=per_sample)

    def save_editing_activation(self):
        if self.original_layer_output is None:
            return
        count = self.positive_rows
        mask = None if self.attention_mask is None else self.attention_mask[:count]
        distance = self.distance(
            self.original_layer_output[:count],
            self.new_weight_layer_output[:count],
            mask,
        )
        self._min_act = min(self._min_act, distance.item())

    def merge_weight(self):
        """Merge task vectors against the immutable base, retaining prior rounds."""
        original = self._original_weight_cpu.to(self.new_weight)
        if not torch.equal(self.new_weight.detach(), original):
            self.memory_weight.append(self.new_weight.detach().clone())
        if not self.memory_weight:
            return
        shards = list(self.memory_weight)
        if torch.equal(self.layer.weight.detach(), original):
            coefficients = [self.merge_weight_factor / len(shards)] * len(shards)
        else:
            # The upstream WISE-Merge rule preserves the previous merged memory
            # as a separate original-base task vector with weight 0.6.
            coefficients = [0.4 / len(shards)] * len(shards) + [0.6]
            shards.append(self.layer.weight.detach())
        if self.merge_strategy == "ties":
            merged = ties_merge(coefficients, original, shards, self.density)
        else:
            merged = self._apply_merge(coefficients, original, shards)
        self.layer.weight = nn.Parameter(merged, requires_grad=False)
        self.weight = self.layer.weight
        self.new_weight = nn.Parameter(original.detach().clone())
        self.memory_weight.clear()

    def forward(self, *args):
        if self.editing:
            return super().forward(*args)
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise TypeError("WISE-Merge expects one hidden-state tensor")
        # Official WISE.__call__ flushes saved shards for WISE-Merge evaluation.
        # Do it here so direct HF model calls obey the same rule.
        if self.editing_total_cnt >= self.save_freq:
            original_weight = self._original_weight_cpu.to(self.new_weight)
            if self.memory_weight or not torch.equal(
                self.new_weight.detach(), original_weight
            ):
                self.merge_weight()
        original = self.original_layer(args[0])
        candidates = [self.layer(args[0]), self._new_weight_forward(args[0])]
        distances = torch.stack(
            [
                self.distance(
                    original,
                    value,
                    self.attention_mask,
                    per_sample=True,
                    inference=True,
                )
                for value in candidates
            ]
        )
        maximum, selected = distances.max(0)
        threshold = self._min_act * self.act_ratio
        result = original
        for index, candidate in enumerate(candidates):
            use = (selected == index) & (maximum >= threshold)
            result = torch.where(
                use.reshape(-1, *([1] * (candidate.ndim - 1))), candidate, result
            )
        return result


class WISEMergeEditor(WISEEditor):
    """Executable WISE-Merge; evaluation probes are never training examples."""

    variant = "wise_merge_easyedit_14cea824_v1"
    adapter_class = WISEMergeAdapter

    def __init__(
        self,
        *args,
        context_templates=None,
        activation_scope="sequence",
        activation_metric="easyedit",
        **kwargs,
    ):
        if activation_scope not in {"sequence", "subject"}:
            raise ValueError("WISE-Merge activation_scope must be sequence or subject")
        if activation_metric not in {"easyedit", "paper"}:
            raise ValueError("WISE-Merge activation_metric must be easyedit or paper")
        if context_templates is not None:
            context_templates = list(context_templates)
            self._validate_templates(context_templates)
        self.context_templates = context_templates
        self.activation_scope = activation_scope
        self.activation_metric = activation_metric
        self._resolved_templates = None
        kwargs.setdefault("merge_strategy", "ties")
        super().__init__(*args, **kwargs)
        if self.merge_freq % self.save_freq:
            raise ValueError("WISE-Merge merge_freq must be divisible by save_freq")
        if self.merge_strategy != "ties":
            raise ValueError(
                "WISE-Merge's validated profile requires merge_strategy=ties"
            )

    @staticmethod
    def _validate_templates(templates):
        if not templates or any(not isinstance(item, str) for item in templates):
            raise ValueError("WISE-Merge context_templates must be nonempty strings")
        for template in templates:
            fields = [
                field
                for _, field, _, _ in string.Formatter().parse(template)
                if field is not None
            ]
            if fields != [""] or not template.endswith("{}"):
                raise ValueError(
                    "Each WISE-Merge template must end in one {} placeholder"
                )

    def _resolve_target_layer(self, inner_params):
        if inner_params == "auto" and hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return f"transformer.h.{self._resolve_layer_indices(self.layers)[-1]}.mlp.c_proj"
        return super()._resolve_target_layer(inner_params)

    def _ensure_adapter(self):
        if self._adapter_installed:
            return
        self._prepare_templates()
        super()._ensure_adapter()
        adapter = self._get_adapter()
        adapter.activation_name = (
            (
                getattr(self.model.config, "hidden_act", None)
                or getattr(self.model.config, "activation_function", None)
                or "linear"
            )
            if self.activation_metric == "easyedit"
            else "linear"
        )
        if adapter.activation_name not in ACT2FN:
            raise ValueError(
                f"Unsupported WISE-Merge activation {adapter.activation_name!r}"
            )

    def _prepare_templates(self):
        if self._resolved_templates is not None:
            return
        if self.context_templates is not None:
            self._resolved_templates = list(self.context_templates)
            return
        # Use the fixed upstream five seeds and two generation lengths. The
        # generated strings become checkpoint state rather than a global cache.
        tok, model = self.tokenizer, self.model
        padding_side = tok.padding_side
        states = [(module, module.training) for module in model.modules()]
        try:
            tok.padding_side = "left"
            model.eval()
            batch = tok(
                ["I", "You", "Because", "Yes", "Q: "], padding=True, return_tensors="pt"
            )
            batch = {
                key: value.to(self._input_device(model))
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask"}
            }
            contexts = []
            with torch.no_grad():
                for length in (5, 10):
                    output = model.generate(
                        **batch,
                        max_new_tokens=length,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tok.pad_token_id,
                    )
                    contexts.extend(tok.batch_decode(output, skip_special_tokens=True))
            self._resolved_templates = ["{}"] + [
                text.replace("{", "").replace("}", "") + " {}" for text in contexts
            ]
        finally:
            tok.padding_side = padding_side
            for module, training in states:
                module.training = training

    def _state_config(self):
        config = super()._state_config()
        config.update(
            context_templates=self.context_templates,
            activation_scope=self.activation_scope,
            activation_metric=self.activation_metric,
            upstream_revision=UPSTREAM_REVISION,
            retrieve=False,
            replay=False,
        )
        return config

    def save_adapter(self, path):
        destination = super().save_adapter(path)
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        payload["resolved_context_templates"] = self._resolved_templates
        torch.save(payload, destination)
        return destination

    def load_adapter(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        templates = payload.get("resolved_context_templates")
        self._validate_templates(templates)
        previous = self._resolved_templates
        try:
            self._resolved_templates = list(templates)
            super().load_adapter(path)
        except Exception:
            self._resolved_templates = previous
            raise

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the base checkpoint and a distinctly named WISE-Merge artifact."""
        if not self._adapter_installed:
            return EditTrainer.save_model(
                self, output_dir, _internal_call=_internal_call
            )
        directory = Path(output_dir or self.args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_adapter(directory / "wise_merge_adapter.pt")
        adapter = self._get_adapter()
        base = copy.deepcopy(self._original_layer).to(adapter.layer.weight.device)
        replace_module(self.model, self.target_layer, base)
        try:
            return EditTrainer.save_model(
                self, directory, _internal_call=_internal_call
            )
        finally:
            replace_module(self.model, self.target_layer, adapter)

    @staticmethod
    def _subject_mask(ids, subject, tokenizer):
        variants = [
            tokenizer.encode(subject, add_special_tokens=False),
            tokenizer.encode(" " + subject, add_special_tokens=False),
        ]
        for needle in sorted(variants, key=len, reverse=True):
            for start in range(len(ids) - len(needle) + 1):
                if needle and ids[start : start + len(needle)] == needle:
                    return start, start + len(needle)
        raise ValueError("WISE-Merge subject tokens do not occur in an edit prompt")

    def _training_batch(self, request):
        negatives = self._scope_training_negatives(request)
        if self.activation_scope == "sequence" and not negatives:
            raise ValueError(
                "WISE-Merge requires independent scope_training_negatives for its routing objective"
            )
        if self.activation_scope == "subject" and not request.subject.strip():
            raise ValueError(
                "WISE-Merge subject activation requires an explicit subject"
            )
        self._prepare_templates()
        positive = [
            template.format(request.prompt) for template in self._resolved_templates
        ]

        # User-provided templates are training data too. Reject direct reuse of
        # a held-out query rather than only checking the negative examples.
        def probes(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from probes(item)
            elif isinstance(value, dict):
                if "prompt" in value:
                    yield from probes(value["prompt"])
                else:
                    for item in value.values():
                        yield from probes(item)

        held_out = {
            " ".join(prompt.split()).casefold()
            for group in (
                request.rephrase_prompts,
                request.locality_inputs,
                request.portability_inputs,
            )
            for prompt in probes(group)
        }
        if held_out.intersection(
            " ".join(prompt.split()).casefold() for prompt in positive
        ):
            raise ValueError(
                "WISE-Merge training contexts overlap held-out evaluation probes"
            )
        tok = self.tokenizer
        inputs, labels, spans = [], [], []
        for prompt in positive:
            prompt_ids = tok.encode(prompt, add_special_tokens=True)
            separator = "" if prompt.endswith((" ", "\n", "\t")) else " "
            target_ids = tok.encode(
                separator + request.target_new, add_special_tokens=False
            )
            if not prompt_ids or not target_ids:
                raise ValueError(
                    "WISE-Merge requires nonempty prompt and target tokens"
                )
            inputs.append(prompt_ids + target_ids)
            labels.append([-100] * len(prompt_ids) + target_ids)
            spans.append(
                self._subject_mask(prompt_ids, request.subject, tok)
                if self.activation_scope == "subject"
                else None
            )
        for prompt in negatives:
            ids = tok.encode(prompt, add_special_tokens=True)
            inputs.append(ids)
            labels.append([-100] * len(ids))
        length = max(map(len, inputs))
        model_limit = getattr(
            self.model.config, "max_position_embeddings", None
        ) or getattr(self.model.config, "n_positions", None)
        if model_limit is not None and length > model_limit:
            raise ValueError(
                "WISE-Merge training text exceeds the model context window; refusing silent truncation"
            )
        pad = tok.pad_token_id
        if pad is None:
            raise ValueError("WISE-Merge requires a tokenizer pad_token_id")
        # Right-pad explicitly. Left-padding GPT-2 without explicit position IDs
        # changes its absolute positions and breaks per-row training equivalence.
        device = self._input_device(self.model)
        ids = torch.full((len(inputs), length), pad, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        targets = torch.full_like(ids, -100)
        act_mask = torch.zeros((len(positive), length), device=device, dtype=torch.long)
        for index, row in enumerate(inputs):
            ids[index, : len(row)] = torch.tensor(row, device=device)
            mask[index, : len(row)] = 1
            targets[index, : len(row)] = torch.tensor(labels[index], device=device)
            if index < len(positive):
                if spans[index] is None:
                    act_mask[index, : len(row)] = 1
                else:
                    start, end = spans[index]
                    act_mask[index, start:end] = 1
        deact_mask = mask[: len(positive)] - act_mask
        if self.activation_scope == "subject" and not deact_mask.any(dim=-1).all():
            raise ValueError("WISE-Merge subject training needs non-subject tokens")
        return (
            {"input_ids": ids, "attention_mask": mask, "labels": targets},
            act_mask,
            deact_mask,
            len(positive),
            len(negatives),
        )

    def _apply_wise_edit(self, request) -> dict[str, Any]:
        adapter = self._get_adapter()
        tokens, act_mask, deact_mask, n_positive, n_negative = self._training_batch(
            request
        )
        adapter.positive_rows = n_positive
        adapter.editing = True
        adapter.set_parameter_tunable()
        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.generate_activation_mask()
        # Apply weight decay before the random mask; torch SGD weight_decay
        # otherwise changes coordinates whose WISE mask is zero (Eq. 7).
        optimizer = torch.optim.SGD([adapter.new_weight], lr=self.edit_lr)
        states = [(module, module.training) for module in self.model.modules()]
        previous_loss = None
        last_values = {}
        iterations = 0
        try:
            self.model.eval()
            for iteration in range(self.n_iter):
                outputs = self.model(**tokens)
                logits = outputs.logits[:n_positive, :-1]
                targets = tokens["labels"][:n_positive, 1:].to(logits.device)
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape_as(targets)
                target_mask = targets.ne(-100)
                ft_loss = ((losses * target_mask).sum(-1) / target_mask.sum(-1)).mean()
                original, edited = (
                    adapter.original_layer_output,
                    adapter.new_weight_layer_output,
                )
                in_scope = adapter.distance(
                    original[:n_positive],
                    edited[:n_positive],
                    act_mask,
                    per_sample=True,
                )
                if self.activation_scope == "subject":
                    out_scope = adapter.distance(
                        original[:n_positive],
                        edited[:n_positive],
                        deact_mask,
                        per_sample=True,
                    )
                else:
                    out_scope = adapter.distance(
                        original[n_positive:],
                        edited[n_positive:],
                        tokens["attention_mask"][n_positive:],
                        per_sample=True,
                    )
                act_loss = activation_margin(
                    in_scope, out_scope, self.alpha, self.beta, self.gamma
                )
                loss = ft_loss + act_loss.to(ft_loss.device)
                if not torch.isfinite(loss):
                    raise ValueError(
                        "WISE-Merge loss is nonfinite; edit transaction aborted"
                    )
                value = float(loss.detach())
                last_values = {
                    "ft_loss": float(ft_loss.detach()),
                    "activation_loss": float(act_loss.detach()),
                    "total_loss": value,
                }
                if (
                    previous_loss is not None
                    and abs(value - previous_loss) <= 1e-4
                    and value <= 0.02
                ):
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                adapter.new_weight.grad.add_(adapter.new_weight.detach(), alpha=1e-5)
                adapter.mask_new_weight_gradient()
                optimizer.step()
                iterations = iteration + 1
                if self.norm_constraint is not None:
                    with torch.no_grad():
                        original_weight = adapter._original_weight_cpu.to(
                            adapter.new_weight
                        )
                        adapter.new_weight.clamp_(
                            original_weight - self.norm_constraint,
                            original_weight + self.norm_constraint,
                        )
                previous_loss = value
            with torch.no_grad():
                self.model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
            adapter.save_editing_activation()
        finally:
            adapter.editing = False
            for module, training in states:
                module.training = training
        adapter.editing_total_cnt += 1
        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.save_weight()
        if adapter.editing_total_cnt % self.merge_freq == 0:
            adapter.merge_weight()
        return {
            "mode": "independent_negatives"
            if self.activation_scope == "sequence"
            else "subject_mask",
            "training_negative_count": n_negative,
            "positive_context_count": n_positive,
            "context_templates": list(self._resolved_templates),
            "activation_scope": self.activation_scope,
            "activation_metric": self.activation_metric,
            "upstream_revision": UPSTREAM_REVISION,
            "uses_evaluation_probes": False,
            "optimizer_steps": iterations,
            **last_values,
        }
