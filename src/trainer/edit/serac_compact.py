"""Historical compact scope/prefix editor, explicitly not full SERAC.

Retained for reproducing the native compact variant's engineering tests.
This two-feature classifier with frozen-base prompt prefixing does not implement
SERAC's learned text encoder and trained counterfactual language model.
"""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.edit.base import EditRequest, EditTrainer
from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)

logger = logging.getLogger(__name__)


class ScopeClassifier(nn.Module):
    """Classify query/cache pairs from semantic and lexical similarity."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)


class SERACCompactEditor(EditTrainer):
    """External-memory editor with learned scope routing and context generation."""

    variant = "compact_scope_prefix_v2"
    _STATE_CONFIG_FIELDS = (
        "edit_lr",
        "cedit",
        "cloc",
        "cbase",
        "classifier_hidden_size",
        "num_edit_steps",
        "scope_threshold",
        "min_scope_overlap",
    )

    def __init__(
        self,
        archive: Optional[str] = None,
        edit_lr: float = 1e-3,
        cedit: float = 1.0,
        cloc: float = 1.0,
        cbase: float = 0.0,
        classifier_hidden_size: int = 32,
        num_edit_steps: int = 50,
        scope_threshold: float = 0.5,
        min_scope_overlap: float = 0.05,
        require_pretrained: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.archive = archive
        self.edit_lr = edit_lr
        self.cedit = cedit
        self.cloc = cloc
        self.cbase = cbase
        self.classifier_hidden_size = classifier_hidden_size
        self.num_edit_steps = num_edit_steps
        self.scope_threshold = scope_threshold
        self.min_scope_overlap = min_scope_overlap
        self.require_pretrained = require_pretrained
        if num_edit_steps <= 0:
            raise ValueError("SERAC num_edit_steps must be positive")

        self.edit_memory: List[Dict[str, Any]] = []
        self.memory_embeddings: List[torch.Tensor] = []
        self.classifier: Optional[ScopeClassifier] = None
        self._is_initialized = False
        self._pretrained_loaded = False

    @property
    def device(self) -> torch.device:
        return self._input_device()

    def _ensure_initialized(self) -> None:
        if self._is_initialized:
            return
        if self.archive:
            self.load_archive(self.archive)
        else:
            self.classifier = ScopeClassifier(self.classifier_hidden_size).to(
                self.device
            )
            self._is_initialized = True

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "of",
            "in",
            "on",
            "to",
            "for",
            "and",
        }
        return {
            token
            for token in re.findall(r"[\w']+", text.lower())
            if token not in stopwords
        }

    def _embed_prompt(self, prompt: str) -> torch.Tensor:
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        batch = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(**batch, output_hidden_states=True)
        finally:
            self.model.train(was_training)
        hidden = output.hidden_states[-1]
        mask = batch.get("attention_mask")
        if mask is None or getattr(self.tokenizer, "padding_side", "right") == "left":
            index = hidden.shape[1] - 1
        else:
            index = int(mask[0].sum().item()) - 1
        return F.normalize(hidden[0, index].float(), dim=0).detach()

    def _pair_features(
        self,
        query: str,
        memory_prompt: str,
        memory_embedding: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_embedding is None:
            query_embedding = self._embed_prompt(query)
        semantic = torch.dot(
            query_embedding, memory_embedding.to(query_embedding.device)
        )
        query_tokens, memory_tokens = self._tokens(query), self._tokens(memory_prompt)
        lexical = len(query_tokens & memory_tokens) / max(
            len(query_tokens | memory_tokens), 1
        )
        return torch.stack([semantic, semantic.new_tensor(float(lexical))], dim=0)

    def _negative_prompts(self, request: EditRequest) -> List[str]:
        # Evaluation locality probes and paraphrases stay held out. Explicit
        # scope-training examples belong to a separate, auditable input split.
        negatives = list(getattr(request, "scope_training_negatives", None) or [])
        if not negatives:
            negatives.append("The weather is pleasant today.")
        return negatives

    def _positive_prompts(self, request: EditRequest) -> List[str]:
        return [
            request.prompt,
            *(getattr(request, "scope_training_prompts", None) or []),
        ]

    def _fit_scope_for_request(self, request: EditRequest, memory_index: int) -> float:
        assert self.classifier is not None
        memory_prompt = self.edit_memory[memory_index]["prompt"]
        memory_embedding = self.memory_embeddings[memory_index]
        positives = self._positive_prompts(request)
        negatives = self._negative_prompts(request)
        held_out = set(request.rephrase_prompts or [])
        groups = request.locality_inputs or {}
        groups = groups.values() if isinstance(groups, dict) else [groups]
        for group in groups:
            group = [group] if isinstance(group, dict) else group
            held_out.update(
                item["prompt"]
                for item in group or []
                if isinstance(item, dict) and item.get("prompt")
            )
        training = set(positives[1:]) | set(negatives)
        if training & held_out:
            raise ValueError(
                "SERAC scope training prompts overlap held-out evaluation probes"
            )
        if set(positives) & set(negatives):
            raise ValueError(
                "SERAC scope positive and negative training prompts overlap"
            )
        features = torch.stack(
            [
                self._pair_features(prompt, memory_prompt, memory_embedding)
                for prompt in [*positives, *negatives]
            ],
            dim=0,
        ).to(self.device)
        labels = torch.tensor(
            [1.0] * len(positives) + [0.0] * len(negatives), device=self.device
        )
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.edit_lr)
        final_loss = 0.0
        self.classifier.train()
        for _ in range(self.num_edit_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = self.classifier(features)
            positive_loss = F.binary_cross_entropy_with_logits(
                logits[: len(positives)], labels[: len(positives)]
            )
            negative_loss = F.binary_cross_entropy_with_logits(
                logits[len(positives) :], labels[len(positives) :]
            )
            loss = self.cedit * positive_loss + self.cloc * negative_loss
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        self.classifier.eval()
        return final_loss

    def _store_request(self, request: EditRequest) -> int:
        entry = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
            "target_old": request.target_old,
            "rephrase_prompts": request.rephrase_prompts or [],
            "locality_inputs": request.locality_inputs,
        }
        embedding = self._embed_prompt(request.prompt).cpu()
        self.edit_memory.append(entry)
        self.memory_embeddings.append(embedding)
        return len(self.edit_memory) - 1

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        del kwargs
        if isinstance(requests, EditRequest):
            requests = [requests]
        if not requests:
            raise ValueError("SERAC requires at least one edit request")
        if (
            self.require_pretrained
            and not self._pretrained_loaded
            and not getattr(self, "archive", None)
        ):
            raise RuntimeError(
                "SERACCompactEditor.require_pretrained=True but no archive was loaded. "
                "Run train_serac()/scripts/train_serac.py and set archive."
            )
        self._ensure_initialized()
        if self.require_pretrained and not self._pretrained_loaded:
            raise RuntimeError(
                "SERACCompactEditor.require_pretrained=True but no archive was loaded. "
                "Run train_serac()/scripts/train_serac.py and set archive."
            )
        assert self.classifier is not None
        old_memory = list(self.edit_memory)
        old_embeddings = list(self.memory_embeddings)
        old_classifier = copy.deepcopy(self.classifier.state_dict())
        was_training = self.classifier.training
        losses = []
        try:
            for request in requests:
                if not request.prompt.strip() or not request.target_new.strip():
                    raise ValueError("SERAC requires non-empty prompt and target_new")
                index = self._store_request(request)
                losses.append(self._fit_scope_for_request(request, index))
        except Exception:
            self.edit_memory = old_memory
            self.memory_embeddings = old_embeddings
            self.classifier.load_state_dict(old_classifier)
            self.classifier.train(was_training)
            self.classifier.zero_grad(set_to_none=True)
            raise

        self.model._edit_runtime = self
        return {
            "success": True,
            "variant": self.variant,
            "edited_count": len(requests),
            "memory_size": len(self.edit_memory),
            "metrics": {"scope_loss": sum(losses) / max(len(losses), 1)},
        }

    def _retrieve(self, prompt: str) -> tuple[Optional[Dict[str, Any]], float]:
        self._ensure_initialized()
        assert self.classifier is not None
        if not self.edit_memory:
            return None, 0.0
        query_embedding = self._embed_prompt(prompt)
        features = torch.stack(
            [
                self._pair_features(prompt, entry["prompt"], embedding, query_embedding)
                for entry, embedding in zip(self.edit_memory, self.memory_embeddings)
            ],
            dim=0,
        ).to(self.device)
        self.classifier.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.classifier(features))
        # The learned classifier handles semantic scope; this small lexical
        # guard prevents an under-trained online classifier from routing wholly
        # unrelated locality prompts through the counterfactual cache.
        scores = scores.masked_fill(features[:, 1] < self.min_scope_overlap, -1.0)
        score, index = scores.max(dim=0)
        return self.edit_memory[int(index.item())], float(score.item())

    def transform_prompt(self, prompt: str) -> str:
        entry, score = self._retrieve(prompt)
        if entry is None or score < self.scope_threshold:
            return prompt
        fact = f"{entry['prompt']} {entry['target_new']}"
        return f"New Fact: {fact}\nPrompt: {prompt}\nAnswer:"

    def select_model(self, prompt: str):
        del prompt
        return self.model

    def predict(self, prompt: str, max_new_tokens: int = 32) -> str:
        transformed = self.transform_prompt(prompt)
        batch = self.tokenizer(transformed, return_tensors="pt")
        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        with torch.no_grad():
            output = self.model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated = output[0, batch["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def train_serac(
        self,
        train_data: Optional[Iterable[Union[EditRequest, Dict[str, Any]]]] = None,
        num_epochs: int = 1,
        batch_size: int = 8,
    ) -> List[Dict[str, float]]:
        """Train the scope classifier and populate the counterfactual cache."""
        del batch_size
        if num_epochs < 1:
            raise ValueError("train_serac num_epochs must be positive")
        if train_data is None:
            raise ValueError("train_serac requires non-empty training data")
        requests = []
        for item in train_data:
            if isinstance(item, EditRequest):
                requests.append(item)
            else:
                requests.append(
                    EditRequest(
                        prompt=str(item.get("prompt", item.get("src", ""))),
                        subject=str(item.get("subject", "")),
                        target_new=str(item.get("target_new", item.get("alt", ""))),
                        target_old=item.get("target_old"),
                        rephrase_prompts=item.get(
                            "rephrase_prompts", item.get("rephrase")
                        ),
                        locality_inputs=item.get(
                            "locality_inputs", item.get("locality")
                        ),
                        scope_training_prompts=item.get("scope_training_prompts"),
                        scope_training_negatives=item.get("scope_training_negatives"),
                    )
                )
        if not requests:
            raise ValueError("train_serac received no usable requests")

        self._ensure_initialized()
        assert self.classifier is not None
        old_memory = list(self.edit_memory)
        old_embeddings = list(self.memory_embeddings)
        old_classifier = copy.deepcopy(self.classifier.state_dict())
        was_training = self.classifier.training
        history = []
        try:
            indices = [self._store_request(request) for request in requests]
            for epoch in range(num_epochs):
                losses = [
                    self._fit_scope_for_request(request, index)
                    for request, index in zip(requests, indices)
                ]
                history.append(
                    {
                        "epoch": float(epoch + 1),
                        "scope_loss": sum(losses) / max(len(losses), 1),
                    }
                )
        except Exception:
            self.edit_memory = old_memory
            self.memory_embeddings = old_embeddings
            self.classifier.load_state_dict(old_classifier)
            self.classifier.train(was_training)
            self.classifier.zero_grad(set_to_none=True)
            raise
        self._pretrained_loaded = True
        self.model._edit_runtime = self
        return history

    def save_archive(self, path: str | Path) -> None:
        self._ensure_initialized()
        assert self.classifier is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "state_version": 2,
                "variant": self.variant,
                "base_model_identity": build_base_model_identity(self.model),
                "pretrained_loaded": self._pretrained_loaded,
                "config": {
                    name: getattr(self, name) for name in self._STATE_CONFIG_FIELDS
                },
                "edit_memory": self.edit_memory,
                "classifier_hidden_size": self.classifier_hidden_size,
                "scope_threshold": self.scope_threshold,
                "min_scope_overlap": self.min_scope_overlap,
            },
            path,
        )
        logger.info("Saved SERAC archive to %s", path)

    def load_archive(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SERAC archive not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("state_version") != 2 or payload.get("variant") != self.variant:
            raise ValueError("Unsupported SERAC archive; regenerate compact v2 state")
        validate_base_model_identity(
            self.model, payload.get("base_model_identity"), checkpoint_label="SERAC"
        )
        config = payload.get("config")
        if not isinstance(config, dict) or set(config) != set(
            self._STATE_CONFIG_FIELDS
        ):
            raise ValueError("SERAC archive is missing its scope configuration")
        saved_hidden = int(
            payload.get("classifier_hidden_size", self.classifier_hidden_size)
        )
        classifier = ScopeClassifier(saved_hidden).to(self.device)
        classifier.load_state_dict(payload["classifier"])
        classifier.eval()
        memory = list(payload.get("edit_memory", []))
        embeddings = [self._embed_prompt(entry["prompt"]).cpu() for entry in memory]
        # Commit only once every component has loaded successfully.
        self.classifier_hidden_size = saved_hidden
        self.classifier = classifier
        self.scope_threshold = float(
            payload.get("scope_threshold", self.scope_threshold)
        )
        self.min_scope_overlap = float(
            payload.get("min_scope_overlap", self.min_scope_overlap)
        )
        for name, value in config.items():
            setattr(self, name, value)
        self.edit_memory = memory
        self.memory_embeddings = embeddings
        self.archive = str(path)
        self._pretrained_loaded = bool(payload.get("pretrained_loaded", False))
        self._is_initialized = True
        self.model._edit_runtime = self
        logger.info("Loaded SERAC archive from %s", path)

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        target_dir = Path(output_dir or self.args.output_dir)
        self.save_archive(target_dir / "serac_archive.pt")

    def clear_edit_memory(self) -> None:
        self.edit_memory.clear()
        self.memory_embeddings.clear()
        if getattr(self.model, "_edit_runtime", None) is self:
            delattr(self.model, "_edit_runtime")

    def reset_independent_state(self) -> None:
        """Remove per-case scope state while preserving an optional archive."""
        self.clear_edit_memory()
        self.classifier = None
        self._is_initialized = False
        self._pretrained_loaded = False

        # A configured archive is immutable benchmark input.  It is reloaded
        # lazily by ``_ensure_initialized`` for the next case; online SERAC
        # starts from a fresh classifier instead of leaking trained weights.


__all__ = ["ScopeClassifier", "SERACCompactEditor"]
