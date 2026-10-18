"""Supervised SERAC with a learned scope encoder and counterfactual causal LM.

This is a native adaptation of the supervised bi-encoder mechanism in
Eric Mitchell's MIT-licensed SERAC (``algs/rep.py`` and ``SupervisedTrainer``),
revision c963a39fa77a743f054bed372056fa1a66e224a4. The auxiliary encoder and
counterfactual LM are trained offline; applying an edit only appends to a cache.
Generation and teacher-forced evaluation use the reference hard route by
default. Explicit soft_weighting mixes base/counterfactual probabilities. This is not an upstream benchmark checkpoint.
The previous two-feature/prefix approximation lives in ``serac_compact.py``.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    stream_model_persistent_state_sha256,
    validate_base_model_identity,
)
from trainer.edit.utils.cache_identity import identity_digest, tokenizer_identity


def _text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def _tokenizer_payload(tokenizer) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="serac-tokenizer-") as directory:
        tokenizer.save_pretrained(directory)
        return {
            path.name: base64.b64encode(path.read_bytes()).decode("ascii")
            for path in Path(directory).iterdir()
            if path.is_file()
        }


def _restore_tokenizer(payload):
    if not isinstance(payload, dict) or not payload:
        raise ValueError("SERAC archive has no embedded tokenizer")
    with tempfile.TemporaryDirectory(prefix="serac-tokenizer-") as directory:
        for name, content in payload.items():
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Invalid embedded SERAC tokenizer filename")
            if not isinstance(content, str):
                raise ValueError("Invalid embedded SERAC tokenizer content")
            (Path(directory) / name).write_bytes(
                base64.b64decode(content, validate=True)
            )
        return AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )


def _safe_config(model) -> dict:
    # Configs are data only: no pickled Transformers objects or remote code.
    return json.loads(model.config.to_json_string(use_diff=False))


def _new_model(config: dict, *, causal: bool, dtype: str):
    config = copy.deepcopy(config)
    model_type = config.pop("model_type")
    config.pop("_name_or_path", None)
    config.pop("auto_map", None)
    cfg = AutoConfig.for_model(model_type, **config)
    cls = AutoModelForCausalLM if causal else AutoModel
    allowed_dtypes = {
        str(value): value
        for value in (torch.float32, torch.float64, torch.float16, torch.bfloat16)
    }
    if dtype not in allowed_dtypes:
        raise ValueError("Unsupported SERAC auxiliary parameter dtype")
    return cls.from_config(
        cfg, trust_remote_code=False, torch_dtype=allowed_dtypes[dtype]
    )


def _vocabulary_identity(tokenizer):
    value = tokenizer_identity(tokenizer)
    for key in ("class", "padding_side", "truncation_side", "model_max_length"):
        value.pop(key, None)
    return identity_digest(value)


class SERACEditor(EditTrainer):
    """Offline-trained SERAC auxiliary networks with cache-only online edits."""

    variant = "supervised_biencoder_causal_v1"
    _STATE_CONFIG_FIELDS = (
        "scope_threshold",
        "distance_heads",
        "distance_scale",
        "max_length",
        "soft_weighting",
    )

    def __init__(
        self,
        archive: str | None = None,
        scope_model_name_or_path: str | None = None,
        counterfactual_model_name_or_path: str | None = None,
        aux_device: str | None = None,
        scope_threshold: float = 0.5,
        distance_heads: int = 1,
        distance_scale: float = 1.0,
        max_length: int = 512,
        soft_weighting: bool = False,
        require_pretrained: bool = True,
        local_files_only: bool = False,
        scope_model=None,
        scope_tokenizer=None,
        counterfactual_model=None,
        counterfactual_tokenizer=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not require_pretrained:
            raise ValueError(
                "SERAC requires offline-trained auxiliary networks. Use train_serac() "
                "before edit(); untrained inference is not a SERAC result."
            )
        self._validate_config(
            scope_threshold, distance_heads, distance_scale, max_length, soft_weighting
        )
        self.archive = archive
        self.scope_model_name_or_path = scope_model_name_or_path
        self.counterfactual_model_name_or_path = counterfactual_model_name_or_path
        self.aux_device = aux_device
        self.scope_threshold = float(scope_threshold)
        self.distance_heads = int(distance_heads)
        self.distance_scale = float(distance_scale)
        self.max_length = int(max_length)
        self.soft_weighting = soft_weighting
        self.require_pretrained = True
        self.local_files_only = local_files_only
        self.classifier = scope_model
        self.classifier_tok = scope_tokenizer
        self.replacement = counterfactual_model
        self.replacement_tok = counterfactual_tokenizer
        self.edit_memory: list[dict[str, str]] = []
        self.training_metadata: dict[str, Any] = {}
        self._is_initialized = False
        self._pretrained_loaded = False

    @staticmethod
    def _validate_config(threshold, heads, scale, max_length, soft_weighting):
        if not isinstance(soft_weighting, bool):
            raise ValueError("SERAC soft_weighting must be boolean")
        if not 0.0 <= threshold <= 1.0 or not math.isfinite(threshold):
            raise ValueError("SERAC scope_threshold must be in [0, 1]")
        if isinstance(heads, bool) or int(heads) != heads or heads < 1:
            raise ValueError("SERAC distance_heads must be a positive integer")
        if scale <= 0 or not math.isfinite(scale):
            raise ValueError("SERAC distance_scale must be positive and finite")
        if (
            isinstance(max_length, bool)
            or int(max_length) != max_length
            or max_length < 2
        ):
            raise ValueError("SERAC max_length must be an integer >= 2")

    @property
    def device(self):
        return (
            torch.device(self.aux_device) if self.aux_device else self._input_device()
        )

    def _initialize_auxiliary(self):
        if self._is_initialized:
            return
        if self.archive:
            self.load_archive(self.archive)
            return
        if self.classifier is None:
            if not self.scope_model_name_or_path:
                raise ValueError(
                    "SERAC needs scope_model_name_or_path for auxiliary training"
                )
            self.classifier = AutoModel.from_pretrained(
                self.scope_model_name_or_path,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
        if self.classifier_tok is None:
            if not self.scope_model_name_or_path:
                raise ValueError("SERAC needs a scope tokenizer")
            self.classifier_tok = AutoTokenizer.from_pretrained(
                self.scope_model_name_or_path,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
        if self.replacement is None:
            if not self.counterfactual_model_name_or_path:
                raise ValueError("SERAC needs counterfactual_model_name_or_path")
            self.replacement = AutoModelForCausalLM.from_pretrained(
                self.counterfactual_model_name_or_path,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
        if self.replacement_tok is None:
            if not self.counterfactual_model_name_or_path:
                raise ValueError("SERAC needs a counterfactual tokenizer")
            self.replacement_tok = AutoTokenizer.from_pretrained(
                self.counterfactual_model_name_or_path,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
        self._prepare_auxiliary()
        self._is_initialized = True

    def _prepare_auxiliary(self):
        if self.classifier is self.model or self.replacement is self.model:
            raise ValueError(
                "SERAC auxiliary models must be independent of the base model"
            )
        base_storage = {p.data_ptr() for p in self.model.parameters()}
        for auxiliary in (self.classifier, self.replacement):
            if any(p.data_ptr() in base_storage for p in auxiliary.parameters()):
                raise ValueError(
                    "SERAC auxiliary weights share storage with the base model"
                )
        hidden_size = getattr(self.classifier.config, "hidden_size", None)
        if hidden_size is None or hidden_size % self.distance_heads:
            raise ValueError(
                "SERAC scope hidden size must divide evenly into distance_heads"
            )
        self.classifier_tok = copy.deepcopy(self.classifier_tok)
        self.replacement_tok = copy.deepcopy(self.replacement_tok)
        self.classifier_tok.padding_side = "right"
        if not self.classifier_tok.sep_token:
            raise ValueError("SERAC bi-encoder needs a scope tokenizer with sep_token")
        if self.classifier_tok.pad_token_id is None:
            raise ValueError("SERAC scope tokenizer needs pad_token")
        if self.replacement_tok.pad_token_id is None:
            if self.replacement_tok.eos_token_id is None:
                raise ValueError(
                    "SERAC counterfactual tokenizer needs pad_token or eos_token"
                )
            self.replacement_tok.pad_token = self.replacement_tok.eos_token
        self.classifier.to(self.device)
        self.replacement.to(self.device)
        self._freeze_auxiliary()

    def _freeze_auxiliary(self):
        for auxiliary in (self.classifier, self.replacement):
            auxiliary.eval()
            auxiliary.requires_grad_(False)
            auxiliary.zero_grad(set_to_none=True)

    def _require_trained(self):
        if not self._is_initialized and self.archive:
            self.load_archive(self.archive)
        if not self._pretrained_loaded:
            raise RuntimeError(
                "SERAC needs an offline-trained auxiliary archive; run scripts/train_serac.py "
                "on a disjoint training split and set archive. Compact archives are incompatible."
            )

    def _scope_embeddings(self, texts):
        encoded = self.classifier_tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.classifier(**batch).last_hidden_state[:, 0]
        return hidden.float().view(len(texts), self.distance_heads, -1)

    def _scope_log_similarities(self, prompts, queries):
        contexts = [prompt + self.classifier_tok.sep_token for prompt in prompts]
        edit_embeddings = self._scope_embeddings(contexts)
        query_embeddings = self._scope_embeddings(queries)
        distances = (query_embeddings[:, None] - edit_embeddings[None]).square().sum(-1)
        return -distances.min(-1).values * self.distance_scale

    @staticmethod
    def _record(record):
        if not isinstance(record, dict):
            raise ValueError(
                "SERAC auxiliary training requires explicit training dictionaries"
            )
        for forbidden in (
            "rephrase",
            "rephrase_prompts",
            "locality",
            "locality_inputs",
            "portability_inputs",
            "portability",
        ):
            if record.get(forbidden):
                raise ValueError(
                    "SERAC auxiliary training records must not contain evaluation probes; "
                    "provide a separate training split with scope_training_* fields"
                )
        result = {name: record.get(name) for name in ("prompt", "target_new")}
        if any(
            not isinstance(value, str) or not value.strip() for value in result.values()
        ):
            raise ValueError(
                "SERAC auxiliary records need non-empty prompt and target_new"
            )
        for name in ("scope_training_prompts", "scope_training_negatives"):
            values = record.get(name)
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or not value.strip() for value in values
                )
            ):
                raise ValueError(
                    f"SERAC auxiliary records need a non-empty {name} list"
                )
            result[name] = list(values)
        positives = {
            _text_hash(result["prompt"]),
            *map(_text_hash, result["scope_training_prompts"]),
        }
        if positives & set(map(_text_hash, result["scope_training_negatives"])):
            raise ValueError("SERAC auxiliary positive and negative prompts overlap")
        if record.get("id") is not None:
            result["id"] = str(record["id"])
        return result

    def _counterfactual_prompt(self, entry, query):
        # Structured prompt/target fields avoid the upstream last-word truncation
        # bug for multi-token targets in causal cache construction.
        return f"{entry['prompt']} {entry['target_new']} {query}"

    def _causal_distribution(self, model, tokenizer, prompt, target):
        if not isinstance(target, str) or not target.strip():
            raise ValueError("SERAC target distribution needs a non-empty target")
        target_text = (
            target if target[:1].isspace() or prompt[-1:].isspace() else " " + target
        )
        prefix = tokenizer.encode(prompt, add_special_tokens=True)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not prefix or not target_ids:
            raise ValueError("SERAC prompt and target must produce tokens")
        context_limit = min(
            self.max_length,
            int(getattr(model.config, "max_position_embeddings", self.max_length)),
        )
        if len(prefix) + len(target_ids) > context_limit:
            raise ValueError(
                "SERAC context plus target exceeds max_length; increase the configured limit"
            )
        device = self._input_device(model)
        input_ids = torch.tensor([prefix + target_ids], dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        start = len(prefix) - 1
        log_probs = (
            outputs.logits[0, start : start + len(target_ids)].float().log_softmax(-1)
        )
        return log_probs, torch.tensor(
            target_ids, dtype=torch.long, device=log_probs.device
        )

    def train_serac(
        self,
        train_data,
        num_epochs: int = 1,
        batch_size: int = 1,
        lr: float = 1e-5,
        cedit: float = 1.0,
        cloc: float = 1.0,
        source: str | None = None,
    ):
        """Train scope BCE + gold-cache counterfactual NLL on explicit auxiliary data.

        Each record contains ``prompt``, ``target_new``, non-empty
        ``scope_training_prompts`` (in-scope queries) and
        ``scope_training_negatives`` (out-of-scope queries). Evaluation probes
        are rejected. Training records are never populated into the edit cache.
        """
        if num_epochs < 1 or batch_size < 1 or lr <= 0 or not math.isfinite(lr):
            raise ValueError(
                "SERAC epochs, batch size and learning rate must be positive"
            )
        if cedit <= 0 or cloc <= 0 or not all(map(math.isfinite, (cedit, cloc))):
            raise ValueError("SERAC cedit and cloc must both be positive and finite")
        records = [self._record(row) for row in train_data]
        if not records:
            raise ValueError("SERAC needs non-empty auxiliary training records")
        if self.edit_memory:
            raise ValueError("Clear the SERAC edit cache before auxiliary training")
        self._initialize_auxiliary()
        self._pretrained_loaded = False
        initial_hashes = {
            name: stream_model_persistent_state_sha256(model)["state_sha256"]
            for name, model in (
                ("scope", self.classifier),
                ("counterfactual", self.replacement),
            )
        }
        for auxiliary in (self.classifier, self.replacement):
            auxiliary.train()
            auxiliary.requires_grad_(True)
        parameters = list(self.classifier.parameters()) + list(
            self.replacement.parameters()
        )
        optimizer = torch.optim.Adam(parameters, lr=lr)
        history = []
        steps = 0
        try:
            for epoch in range(num_epochs):
                totals = {"scope_loss": 0.0, "counterfactual_nll": 0.0, "loss": 0.0}
                batches = 0
                for start in range(0, len(records), batch_size):
                    batch = records[start : start + batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    contexts = [row["prompt"] for row in batch]
                    pos_queries, pos_idxs, neg_queries, neg_idxs = [], [], [], []
                    for index, row in enumerate(batch):
                        pos_queries.extend(row["scope_training_prompts"])
                        pos_idxs.extend([index] * len(row["scope_training_prompts"]))
                        neg_queries.extend(row["scope_training_negatives"])
                        neg_idxs.extend([index] * len(row["scope_training_negatives"]))
                    similarities = self._scope_log_similarities(
                        contexts, pos_queries + neg_queries
                    )
                    indices = torch.tensor(pos_idxs + neg_idxs, device=self.device)
                    chosen = similarities[
                        torch.arange(len(indices), device=self.device), indices
                    ]
                    positive_loss = -chosen[: len(pos_idxs)].mean()
                    # log(1-exp(logsim)); expm1 retains precision near logsim=0.
                    negative_prob = -torch.expm1(chosen[len(pos_idxs) :])
                    negative_loss = (
                        -negative_prob.clamp_min(torch.finfo(torch.float32).eps)
                        .log()
                        .mean()
                    )
                    scope_loss = positive_loss + negative_loss
                    nlls = []
                    for query, index in zip(pos_queries, pos_idxs):
                        row = batch[index]
                        prefix = self._counterfactual_prompt(row, query)
                        distribution, labels = self._causal_distribution(
                            self.replacement,
                            self.replacement_tok,
                            prefix,
                            row["target_new"],
                        )
                        nlls.append(-distribution.gather(1, labels[:, None]).mean())
                    counterfactual_nll = torch.stack(nlls).mean()
                    loss = cloc * scope_loss + cedit * counterfactual_nll
                    if not torch.isfinite(loss):
                        raise ValueError(
                            "SERAC auxiliary training produced a non-finite loss"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters, 100.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    steps += 1
                    batches += 1
                    for name, value in (
                        ("scope_loss", scope_loss),
                        ("counterfactual_nll", counterfactual_nll),
                        ("loss", loss),
                    ):
                        totals[name] += float(value.detach())
                history.append(
                    {
                        "epoch": epoch + 1,
                        **{key: value / batches for key, value in totals.items()},
                    }
                )
        except Exception:
            self.training_metadata = {"status": "failed", "optimizer_steps": steps}
            raise
        finally:
            self._freeze_auxiliary()
        final_hashes = {
            name: stream_model_persistent_state_sha256(model)["state_sha256"]
            for name, model in (
                ("scope", self.classifier),
                ("counterfactual", self.replacement),
            )
        }
        if any(initial_hashes[name] == final_hashes[name] for name in initial_hashes):
            self.training_metadata = {"status": "failed", "optimizer_steps": steps}
            raise ValueError(
                "SERAC auxiliary training must update both scope and counterfactual weights"
            )
        texts = [
            text
            for row in records
            for text in [
                row["prompt"],
                *row["scope_training_prompts"],
                *row["scope_training_negatives"],
            ]
        ]
        self.training_metadata = {
            "status": "trained",
            "record_count": len(records),
            "optimizer_steps": steps,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "cedit": cedit,
            "cloc": cloc,
            "source": source,
            "input_sha256": identity_digest(records),
            "training_prompt_sha256": sorted(set(map(_text_hash, texts))),
            "objective": "balanced_scope_bce_plus_gold_counterfactual_nll",
            "initial_weights_sha256": initial_hashes,
            "trained_weights_sha256": final_hashes,
            "history": history,
        }
        self._pretrained_loaded = True
        return history

    @torch.no_grad()
    def validate_auxiliary(self, data):
        """Score independent auxiliary cases without updating weights or cache."""
        self._require_trained()
        records = [self._record(row) for row in data]
        if not records:
            raise ValueError("SERAC auxiliary validation requires non-empty records")
        trained = set(self.training_metadata["training_prompt_sha256"])
        texts = [
            text
            for row in records
            for text in [
                row["prompt"],
                *row["scope_training_prompts"],
                *row["scope_training_negatives"],
            ]
        ]
        if trained & set(map(_text_hash, texts)):
            raise ValueError("SERAC validation overlaps auxiliary training")
        correct = count = 0
        nlls = []
        self.classifier.eval()
        self.replacement.eval()
        for row in records:
            positive = row["scope_training_prompts"]
            negative = row["scope_training_negatives"]
            probabilities = self._scope_log_similarities(
                [row["prompt"]], [*positive, *negative]
            )[:, 0].exp()
            expected = torch.tensor(
                [1] * len(positive) + [0] * len(negative), device=probabilities.device
            )
            correct += int(
                ((probabilities >= self.scope_threshold) == expected.bool()).sum()
            )
            count += len(expected)
            for query in positive:
                log_probs, target = self._causal_distribution(
                    self.replacement,
                    self.replacement_tok,
                    self._counterfactual_prompt(row, query),
                    row["target_new"],
                )
                nlls.append(float(-log_probs.gather(-1, target[:, None]).mean()))
        return {
            "records": len(records),
            "scope_queries": count,
            "scope_accuracy": correct / count,
            "counterfactual_queries": len(nlls),
            "counterfactual_nll": sum(nlls) / len(nlls),
            "scope": "held-out auxiliary validation; not edit benchmark metrics",
        }

    def _check_evaluation_split(self, request):
        trained = set(self.training_metadata["training_prompt_sha256"])
        texts = [request.prompt, *(request.rephrase_prompts or [])]

        def collect(value):
            if isinstance(value, dict):
                if "prompt" in value:
                    prompts = value["prompt"]
                    if isinstance(prompts, str):
                        texts.append(prompts)
                    elif isinstance(prompts, list):
                        texts.extend(item for item in prompts if isinstance(item, str))
                    # A probe can contain answers or metadata with their own
                    # nested fields. Only its prompt(s) define evaluation input.
                    return
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(request.locality_inputs)
        collect(request.portability_inputs)
        if trained & set(map(_text_hash, texts)):
            raise ValueError(
                "SERAC edit/evaluation prompts overlap the auxiliary training split"
            )

    def edit(self, requests, **kwargs):
        """Append edit descriptors without fitting either auxiliary network."""
        del kwargs
        self._require_trained()
        requests = [requests] if isinstance(requests, EditRequest) else list(requests)
        if not requests:
            raise ValueError("SERAC requires at least one edit request")
        entries = []
        for request in requests:
            if (
                not isinstance(request, EditRequest)
                or not request.prompt.strip()
                or not request.target_new.strip()
            ):
                raise ValueError(
                    "SERAC needs non-empty EditRequest prompt and target_new"
                )
            self._check_evaluation_split(request)
            entries.append(
                {
                    "prompt": request.prompt,
                    "target_new": request.target_new,
                    "subject": request.subject,
                }
            )
        self.edit_memory.extend(entries)
        self.model._edit_runtime = self
        return {
            "success": True,
            "variant": self.variant,
            "edited_count": len(entries),
            "memory_size": len(self.edit_memory),
            "online_optimizer_steps": 0,
            "auxiliary_training_steps": self.training_metadata["optimizer_steps"],
        }

    def _retrieve(self, prompt):
        self._require_trained()
        if not self.edit_memory:
            return None, 0.0
        with torch.no_grad():
            self.classifier.eval()
            scores = self._scope_log_similarities(
                [row["prompt"] for row in self.edit_memory], [prompt]
            )[0].exp()
            score, index = scores.max(0)
        return self.edit_memory[int(index)], float(score)

    def resolve_query(self, prompt):
        """Return the model, its tokenizer and its input under hard scope routing."""
        entry, score = self._retrieve(prompt)
        if entry is not None and score > self.scope_threshold:
            return (
                self.replacement,
                self.replacement_tok,
                self._counterfactual_prompt(entry, prompt),
            )
        return self.model, self.tokenizer, prompt

    def target_distribution(self, prompt, target):
        """Return target-position log probabilities [tokens, vocab] and token IDs.

        The default hard route matches supervised SERAC's reference forward.
        Explicit soft_weighting=True mixes probability distributions. Both modes
        require matching tokenization/vocabulary so EditSuite can compare token
        IDs and decode predictions with the base tokenizer. Scope sees only the query.
        """
        entry, score = self._retrieve(prompt)
        if entry is not None and _vocabulary_identity(
            self.tokenizer
        ) != _vocabulary_identity(self.replacement_tok):
            raise ValueError(
                "SERAC token-level evaluation requires identical base/counterfactual tokenizers"
            )
        if not self.soft_weighting:
            model, tokenizer, context = self.model, self.tokenizer, prompt
            if entry is not None and score > self.scope_threshold:
                model, tokenizer = self.replacement, self.replacement_tok
                context = self._counterfactual_prompt(entry, prompt)
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    return self._causal_distribution(model, tokenizer, context, target)
            finally:
                model.train(was_training)
        base_was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                base, labels = self._causal_distribution(
                    self.model, self.tokenizer, prompt, target
                )
                if entry is None:
                    return base, labels
                replacement, replacement_labels = self._causal_distribution(
                    self.replacement,
                    self.replacement_tok,
                    self._counterfactual_prompt(entry, prompt),
                    target,
                )
                if replacement.shape != base.shape or not torch.equal(
                    labels.cpu(), replacement_labels.cpu()
                ):
                    raise ValueError(
                        "SERAC base/counterfactual target distributions are incompatible"
                    )
                replacement = replacement.to(base.device)
                probability = base.new_tensor(score).clamp(0, 1)
                mixed = torch.logaddexp(
                    base + torch.log1p(-probability), replacement + probability.log()
                )
                return mixed, labels
        finally:
            self.model.train(base_was_training)

    def target_log_probs(self, prompt, target):
        distribution, labels = self.target_distribution(prompt, target)
        return distribution.gather(1, labels[:, None]).squeeze(1)

    def predict(self, prompt, max_new_tokens=32):
        model, tokenizer, transformed = self.resolve_query(prompt)
        device = self._input_device(model)
        batch = tokenizer(transformed, return_tensors="pt")
        batch = {
            key: value.to(device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask"}
        }
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                output = model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=False
                )
        finally:
            model.train(was_training)
        return tokenizer.decode(
            output[0, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def save_archive(self, path):
        """Save a self-contained, weights-only-loadable auxiliary state and cache."""
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_version": 3,
            "variant": self.variant,
            "base_model_identity": build_base_model_identity(self.model),
            "base_tokenizer_sha256": identity_digest(
                tokenizer_identity(self.tokenizer)
            ),
            "config": {name: getattr(self, name) for name in self._STATE_CONFIG_FIELDS},
            "training_metadata": copy.deepcopy(self.training_metadata),
            "edit_memory": copy.deepcopy(self.edit_memory),
        }
        for key, model, tokenizer in (
            ("scope", self.classifier, self.classifier_tok),
            ("counterfactual", self.replacement, self.replacement_tok),
        ):
            payload[key] = {
                "model_config": _safe_config(model),
                "parameter_dtype": str(next(model.parameters()).dtype),
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "tokenizer_files": _tokenizer_payload(tokenizer),
                "tokenizer_sha256": identity_digest(tokenizer_identity(tokenizer)),
            }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_archive(self, path):
        """Restore trained auxiliary networks without downloading model/tokenizer files."""
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("state_version") != 3 or payload.get("variant") != self.variant:
            raise ValueError(
                "SERAC requires a supervised v3 archive; compact/legacy archives are incompatible"
            )
        validate_base_model_identity(
            self.model, payload.get("base_model_identity"), checkpoint_label="SERAC"
        )
        if identity_digest(tokenizer_identity(self.tokenizer)) != payload.get(
            "base_tokenizer_sha256"
        ):
            raise ValueError("SERAC archive base tokenizer is incompatible")
        config = payload.get("config", {})
        if set(config) != set(self._STATE_CONFIG_FIELDS):
            raise ValueError("SERAC archive configuration is incomplete")
        self._validate_config(
            config["scope_threshold"],
            config["distance_heads"],
            config["distance_scale"],
            config["max_length"],
            config["soft_weighting"],
        )
        metadata = payload.get("training_metadata", {})
        if (
            metadata.get("status") != "trained"
            or metadata.get("optimizer_steps", 0) < 1
            or metadata.get("record_count", 0) < 1
            or not metadata.get("input_sha256")
            or not metadata.get("training_prompt_sha256")
        ):
            raise ValueError(
                "SERAC archive has no completed auxiliary training evidence"
            )
        auxiliary = []
        for name, causal in (("scope", False), ("counterfactual", True)):
            section = payload[name]
            model = _new_model(
                section["model_config"], causal=causal, dtype=section["parameter_dtype"]
            )
            model.load_state_dict(section["state_dict"], strict=True)
            tokenizer = _restore_tokenizer(section["tokenizer_files"])
            if (
                identity_digest(tokenizer_identity(tokenizer))
                != section["tokenizer_sha256"]
            ):
                raise ValueError("SERAC embedded tokenizer identity mismatch")
            actual_hash = stream_model_persistent_state_sha256(model)["state_sha256"]
            expected_hash = metadata.get("trained_weights_sha256", {}).get(name)
            initial_hash = metadata.get("initial_weights_sha256", {}).get(name)
            if (
                actual_hash != expected_hash
                or not initial_hash
                or initial_hash == expected_hash
            ):
                raise ValueError(
                    "SERAC archive auxiliary training weight evidence is inconsistent"
                )
            auxiliary.append((model, tokenizer))
        memory = payload.get("edit_memory", [])
        if not isinstance(memory, list) or any(
            not isinstance(row, dict)
            or not row.get("prompt")
            or not row.get("target_new")
            for row in memory
        ):
            raise ValueError("SERAC archive edit cache is malformed")
        # All file/config/weight validation has completed before installing state.
        previous = {
            name: getattr(self, name)
            for name in (
                *self._STATE_CONFIG_FIELDS,
                "classifier",
                "classifier_tok",
                "replacement",
                "replacement_tok",
            )
        }
        try:
            self.classifier, self.classifier_tok = auxiliary[0]
            self.replacement, self.replacement_tok = auxiliary[1]
            for name, value in config.items():
                setattr(self, name, value)
            self._prepare_auxiliary()
        except Exception:
            for name, value in previous.items():
                setattr(self, name, value)
            raise
        self.edit_memory = copy.deepcopy(memory)
        self.training_metadata = copy.deepcopy(metadata)
        self.archive = str(path)
        self._is_initialized = True
        self._pretrained_loaded = True
        self.model._edit_runtime = self

    def save_model(self, output_dir=None, _internal_call=False):
        self._require_trained()
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        self.save_archive(Path(output_dir or self.args.output_dir) / "serac_archive.pt")

    def clear_edit_memory(self):
        self.edit_memory.clear()
        if getattr(self.model, "_edit_runtime", None) is self:
            delattr(self.model, "_edit_runtime")

    def reset_independent_state(self):
        """Clear online edits while retaining immutable offline-trained networks."""
        self.clear_edit_memory()


__all__ = ["SERACEditor"]
