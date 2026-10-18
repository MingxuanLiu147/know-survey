"""
IKE 知识编辑器（In-Context Knowledge Editing）
==============================================

基于上下文学习的知识编辑方法，不修改模型权重。
显式区分编辑记忆、EasyEdit 独立演示库和原作者 32-demo 组织协议。

参考论文: Can We Edit Factual Knowledge by In-Context Learning?
https://arxiv.org/abs/2305.12740

原作者组织协议: pkunlp-icler/IKE@da58c842cd95628f281f474bc432a81cbd1cfd1e
EasyEdit 独立演示检索: zjunlp/EasyEdit@14cea8245f06715684592ab55184939b99d70784
语料字段、持久化、不同长度记录的近邻映射是本仓库适配，非论文分数复现。
"""

import json
import hashlib
import logging
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except ImportError:
    SentenceTransformer = None
    st_util = None


class IKEEditor(EditTrainer):
    """IKE 知识编辑器

    基于检索增强的上下文学习方法，不修改模型权重。
    维护编辑事实的向量索引，推理时检索最相关的事实作为 ICL 演示。

    Attributes:
        sentence_model_name: sentence-transformer 模型名称
        k: 检索的 ICL 示例数量
        use_icl_examples: 是否启用 ICL 示例检索
    """

    variant = "edit_memory_icl_v2"
    _STATE_VERSION = 2
    _RETRIEVAL_BACKENDS = {
        "auto",
        "sentence_transformer",
        "token_overlap",
    }
    _RETRIEVAL_CONFIG_FIELDS = (
        "sentence_model_name",
        "retrieval_backend",
        "k",
        "use_icl_examples",
        "icl_prompt_template",
    )

    def __init__(
        self,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        k: int = 3,
        use_icl_examples: bool = True,
        icl_prompt_template: Optional[str] = None,
        retrieval_backend: str = "auto",
        memory_path: Optional[str] = None,
        demonstration_path: Optional[str] = None,
        protocol: str = "edit_memory",
        demonstration_seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sentence_model_name = sentence_model_name
        self.k = k
        self.use_icl_examples = use_icl_examples
        self.icl_prompt_template = icl_prompt_template or (
            "New Fact: {fact}\nPrompt: {prompt}\nAnswer: {target}\n\n"
        )
        self.retrieval_backend = retrieval_backend
        self.memory_path = memory_path
        self.demonstration_path = demonstration_path
        if protocol not in {
            "edit_memory",
            "copy_update_retain",
            "paper_copy_update_retain",
        }:
            raise ValueError("Unknown IKE demonstration protocol")
        self.protocol = protocol
        self.demonstration_seed = demonstration_seed
        if isinstance(demonstration_seed, bool) or not isinstance(
            demonstration_seed, int
        ):
            raise ValueError("IKE demonstration_seed must be an integer")
        self.demonstration_records = []
        self.demonstration_sentences = []
        self.demonstration_fact_indices = []
        self.demonstration_embeddings = None
        self.demonstration_sha256 = None
        if self._independent_protocol():
            if retrieval_backend != "sentence_transformer" or not use_icl_examples:
                raise ValueError(
                    "IKE copy/update/retain requires strict semantic retrieval"
                )
            if not demonstration_path and not memory_path:
                raise ValueError(
                    "IKE copy/update/retain requires an independent demonstration_path"
                )
            self.variant = "independent_copy_update_retain_v1"
            if protocol == "paper_copy_update_retain":
                if k != 32:
                    raise ValueError(
                        "IKE paper protocol requires k=32 (4 copy / 12 update / 16 retain)"
                    )
                self.variant = "paper_copy_update_retain_v1"
        self._validate_retrieval_config(
            self._retrieval_config(), source="IKE runtime configuration"
        )

        self.knowledge_store: List[Dict[str, str]] = []
        self.knowledge_sentences: List[str] = []
        self.knowledge_queries: List[str] = []
        self.knowledge_embeddings: Optional[torch.Tensor] = None
        self._sentence_model: Optional[Any] = None
        self._sentence_model_failed = False
        if demonstration_path or (self._independent_protocol() and memory_path):
            if not self._independent_protocol():
                raise ValueError(
                    "demonstration_path requires copy_update_retain protocol"
                )
            records = (
                json.loads(Path(demonstration_path).read_text(encoding="utf-8"))
                if demonstration_path
                else json.loads(Path(memory_path).read_text(encoding="utf-8")).get(
                    "demonstration_records"
                )
            )
            self.demonstration_records, self.demonstration_sentences = (
                self.build_demonstrations(records)
            )
            if (
                protocol == "paper_copy_update_retain"
                and len(self.demonstration_records) < 32
            ):
                raise ValueError(
                    "IKE paper protocol requires at least 32 independent source records"
                )
            self.demonstration_sha256 = self._corpus_digest(self.demonstration_records)
            if protocol == "paper_copy_update_retain":
                self.demonstration_sentences = []
                for index, row in enumerate(self.demonstration_records):
                    fact = f"{row['prompt']} {row['target_new']}"
                    examples = [(row["prompt"], row["target_new"])]
                    examples += [
                        (query, row["target_new"]) for query in row["rephrase_prompts"]
                    ]
                    examples += [
                        (query, row["locality_ground_truth"])
                        for query in row["locality_prompts"]
                    ]
                    for query, target in examples:
                        self.demonstration_sentences.append(
                            f"New Fact: {fact}\nPrompt: {query} {target}\n\n"
                        )
                        self.demonstration_fact_indices.append(index)
            self.demonstration_embeddings = self._encode_queries(
                self.demonstration_sentences
            )
        if memory_path:
            self.load_edit_state(memory_path)

    @staticmethod
    def _corpus_digest(records):
        return hashlib.sha256(
            json.dumps(records, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    @staticmethod
    def build_demonstrations(records):
        """Render copy/update/retain demonstrations from an independent corpus.

        The fields are the text-only EasyEdit IKE training-corpus contract.
        Rephrase/locality fields here belong to *training* examples, never the
        edited case or its evaluation probes. All three kinds are required.
        """
        if not isinstance(records, list) or not records:
            raise ValueError("IKE demonstration corpus must be a non-empty JSON list")
        cleaned, sentences, identities = [], [], set()
        fields = (
            "id",
            "prompt",
            "target_new",
            "rephrase_prompt",
            "locality_prompt",
            "locality_ground_truth",
        )
        for record in records:
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in fields
            ):
                raise ValueError(
                    "IKE corpus requires id, prompt, target_new, rephrase_prompt, locality_prompt and locality_ground_truth"
                )
            row = {key: record[key] for key in fields}
            for name, fallback in (
                ("rephrase_prompts", "rephrase_prompt"),
                ("locality_prompts", "locality_prompt"),
            ):
                values = record.get(name, [row[fallback]])
                if (
                    not isinstance(values, list)
                    or not values
                    or any(
                        not isinstance(value, str) or not value.strip()
                        for value in values
                    )
                ):
                    raise ValueError(f"IKE corpus has invalid {name}")
                row[name] = list(values)
            # Multiple retain prompts must share the provided ground truth;
            # converters must not flatten differently labelled neighbors.
            if row["id"] in identities:
                raise ValueError(
                    "IKE demonstration corpus contains duplicate source IDs"
                )
            identities.add(row["id"])
            if (
                len(
                    {
                        IKEEditor._normalized_text(row[key])
                        for key in ("prompt", "rephrase_prompt", "locality_prompt")
                    }
                )
                != 3
            ):
                raise ValueError("IKE copy, update and retain prompts must be distinct")
            fact = f"{row['prompt']} {row['target_new']}"
            for query, target in (
                (row["prompt"], row["target_new"]),
                (row["rephrase_prompt"], row["target_new"]),
                (row["locality_prompt"], row["locality_ground_truth"]),
            ):
                sentences.append(f"New Fact: {fact}\nPrompt: {query} {target}\n\n")
            cleaned.append(row)
        return cleaned, sentences

    def _independent_protocol(self):
        return getattr(self, "protocol", "edit_memory") != "edit_memory"

    @staticmethod
    def _normalized_text(text):
        return " ".join(text.casefold().split())

    def _check_demonstration_split(self, request):
        def prompts(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from prompts(item)
            elif isinstance(value, dict):
                if "prompt" in value:
                    yield from prompts(value["prompt"])
                else:
                    for item in value.values():
                        yield from prompts(item)

        training = {
            self._normalized_text(text)
            for row in self.demonstration_records
            for text in (
                row["prompt"],
                *row["rephrase_prompts"],
                *row["locality_prompts"],
            )
        }
        evaluation = {
            self._normalized_text(text)
            for group in (
                request.prompt,
                request.rephrase_prompts,
                request.locality_inputs,
                request.portability_inputs,
            )
            for text in prompts(group)
        }
        if training & evaluation:
            raise ValueError(
                "IKE demonstration corpus overlaps edit/evaluation prompts"
            )

    def _independent_icl_prompt(self):
        if not self.knowledge_store:
            return ""
        entry = self.knowledge_store[-1]
        query = f"New Fact: {entry['fact']}\nPrompt: {entry['prompt']}\n\n"
        embeddings = self._encode_queries([query]).to(
            self.demonstration_embeddings.device
        )
        if self.protocol == "paper_copy_update_retain":
            # Rank independent facts by their best matching demonstration,
            # avoiding duplicate facts occupying the 32-neighbor budget.
            similarities = (embeddings @ self.demonstration_embeddings.T)[0]
            owners = getattr(self, "demonstration_fact_indices", [])
            if not owners:
                owners = [index // 3 for index in range(len(similarities))]
            owners = torch.tensor(owners, device=similarities.device)
            scores = torch.stack(
                [
                    similarities[owners == index].max()
                    for index in range(len(self.demonstration_records))
                ]
            )
            indices = torch.argsort(scores, descending=True, stable=True)[:32].tolist()
            # The author's icl.py fixes these counts, shuffles the types, then
            # reverses demonstrations so the closest fact is beside the query.
            kinds = [
                2,
                1,
                2,
                0,
                1,
                2,
                2,
                0,
                2,
                2,
                1,
                0,
                2,
                1,
                2,
                0,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
            ]
            rng = random.Random(self.demonstration_seed)
            rng.shuffle(kinds)
            selected = []
            for index, kind in zip(indices, kinds):
                row = self.demonstration_records[index]
                fact = f"{row['prompt']} {row['target_new']}"
                prompt = (
                    row["prompt"]
                    if kind == 0
                    else rng.choice(
                        row["rephrase_prompts"]
                        if kind == 1
                        else row["locality_prompts"]
                    )
                )
                target = (
                    row["locality_ground_truth"] if kind == 2 else row["target_new"]
                )
                selected.append(f"New Fact: {fact}\nPrompt: {prompt} {target}\n\n")
            selected.reverse()
            selected.append(f"New Fact: {entry['fact']}\nPrompt: {entry['fact']}\n\n")
            return "".join(selected)
        hits = st_util.semantic_search(
            embeddings,
            self.demonstration_embeddings,
            score_function=st_util.dot_score,
            top_k=min(self.k, len(self.demonstration_sentences)),
        )
        return "".join(
            self.demonstration_sentences[hit["corpus_id"]] for hit in hits[0]
        )

    @property
    def sentence_model(self):
        """延迟加载 sentence-transformer 模型。"""
        if self._sentence_model is not None:
            return self._sentence_model

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for IKE. "
                "Install it with: pip install sentence-transformers"
            )

        device = next(self.model.parameters()).device
        self._sentence_model = SentenceTransformer(self.sentence_model_name).to(device)
        logger.info(
            "Loaded sentence model '%s' on %s",
            self.sentence_model_name,
            device,
        )
        return self._sentence_model

    def _retrieval_config(self) -> Dict[str, Any]:
        """Return every setting that can change retrieval or prompt semantics."""
        return {
            "sentence_model_name": self.sentence_model_name,
            "retrieval_backend": self.retrieval_backend,
            "k": self.k,
            "use_icl_examples": self.use_icl_examples,
            "icl_prompt_template": self.icl_prompt_template,
        }

    @classmethod
    def _validate_retrieval_config(cls, config: Dict[str, Any], *, source: str) -> None:
        missing = [
            field for field in cls._RETRIEVAL_CONFIG_FIELDS if field not in config
        ]
        if missing:
            raise ValueError(
                f"{source} is missing retrieval-semantic fields: {', '.join(missing)}"
            )

        sentence_model_name = config["sentence_model_name"]
        if not isinstance(sentence_model_name, str) or not sentence_model_name.strip():
            raise ValueError(f"{source} sentence_model_name must be a non-empty string")

        backend = config["retrieval_backend"]
        if backend not in cls._RETRIEVAL_BACKENDS:
            choices = ", ".join(sorted(cls._RETRIEVAL_BACKENDS))
            raise ValueError(f"{source} retrieval_backend must be one of: {choices}")

        k = config["k"]
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"{source} k must be a positive integer")

        if not isinstance(config["use_icl_examples"], bool):
            raise ValueError(f"{source} use_icl_examples must be a boolean")

        template = config["icl_prompt_template"]
        if not isinstance(template, str) or not template:
            raise ValueError(f"{source} icl_prompt_template must be a non-empty string")
        try:
            template.format(fact="fact", prompt="prompt", target="target")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"{source} icl_prompt_template cannot be rendered with "
                "fact/prompt/target"
            ) from exc

    @staticmethod
    def _validate_request(request: EditRequest) -> None:
        if not isinstance(request, EditRequest):
            raise TypeError("IKE memory accepts only EditRequest instances")
        for field in ("prompt", "subject", "target_new"):
            value = getattr(request, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"IKE request {field} must be a non-empty string")
        if request.target_old is not None and not isinstance(request.target_old, str):
            raise ValueError("IKE request target_old must be a string or None")

    def _validate_memory_collections(self) -> None:
        for name in (
            "knowledge_store",
            "knowledge_sentences",
            "knowledge_queries",
        ):
            if not isinstance(getattr(self, name), list):
                raise ValueError(f"IKE {name} must be a list")
        size = len(self.knowledge_store)
        if len(self.knowledge_sentences) != size or len(self.knowledge_queries) != size:
            raise ValueError(
                "IKE memory is inconsistent: store, sentences, and queries differ in size"
            )

    def _validate_memory_state(self) -> None:
        self._validate_memory_collections()
        size = len(self.knowledge_store)

        embeddings = self.knowledge_embeddings
        if embeddings is not None:
            if not isinstance(embeddings, torch.Tensor) or embeddings.dim() != 2:
                raise ValueError("IKE knowledge_embeddings must be a rank-2 tensor")
            if embeddings.shape[0] != size:
                raise ValueError(
                    "IKE memory is inconsistent: embedding rows do not match entries"
                )
        if self._sentence_model_failed and embeddings is not None:
            raise ValueError(
                "IKE memory is inconsistent: fallback mode cannot retain embeddings"
            )
        semantic_backend_active = (
            not self._independent_protocol()
            and self.use_icl_examples
            and self.retrieval_backend in {"auto", "sentence_transformer"}
            and not self._sentence_model_failed
        )
        if semantic_backend_active and size and embeddings is None:
            raise ValueError(
                "IKE semantic memory has entries but no matching embedding index"
            )

    def _snapshot_runtime_state(self) -> Dict[str, Any]:
        runtime_present = hasattr(self.model, "_edit_runtime")
        return {
            "knowledge_store": self.knowledge_store,
            "knowledge_sentences": self.knowledge_sentences,
            "knowledge_queries": self.knowledge_queries,
            "knowledge_embeddings": self.knowledge_embeddings,
            "sentence_model": self._sentence_model,
            "sentence_model_failed": self._sentence_model_failed,
            "runtime_present": runtime_present,
            "runtime": getattr(self.model, "_edit_runtime", None),
        }

    def _restore_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        self.knowledge_store = snapshot["knowledge_store"]
        self.knowledge_sentences = snapshot["knowledge_sentences"]
        self.knowledge_queries = snapshot["knowledge_queries"]
        self.knowledge_embeddings = snapshot["knowledge_embeddings"]
        self._sentence_model = snapshot["sentence_model"]
        self._sentence_model_failed = snapshot["sentence_model_failed"]
        if snapshot["runtime_present"]:
            self.model._edit_runtime = snapshot["runtime"]
        elif hasattr(self.model, "_edit_runtime"):
            delattr(self.model, "_edit_runtime")

    def _effective_retrieval_backend(self) -> str:
        if not self.use_icl_examples:
            return "disabled"
        if self.retrieval_backend == "token_overlap":
            return "token_overlap"
        if self._sentence_model_failed:
            return "token_overlap"
        if self.retrieval_backend == "sentence_transformer":
            return "sentence_transformer"
        if self.knowledge_embeddings is not None:
            return "sentence_transformer"
        return "unresolved"

    def _validated_state_memory(
        self, payload: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[str], List[str]]:
        required = ("knowledge_store", "knowledge_sentences", "knowledge_queries")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(
                "IKE edit state is missing memory fields: " + ", ".join(missing)
            )

        store = payload["knowledge_store"]
        sentences = payload["knowledge_sentences"]
        queries = payload["knowledge_queries"]
        if not isinstance(store, list):
            raise ValueError("IKE edit state knowledge_store must be a list")
        if not isinstance(sentences, list):
            raise ValueError("IKE edit state knowledge_sentences must be a list")
        if not isinstance(queries, list):
            raise ValueError("IKE edit state knowledge_queries must be a list")
        if len(sentences) != len(store) or len(queries) != len(store):
            raise ValueError(
                "IKE edit state memory views must contain the same number of entries"
            )
        if self._independent_protocol() and len(store) > 1:
            raise ValueError("IKE independent state may contain only one active edit")

        validated_store: List[Dict[str, Any]] = []
        validated_sentences: List[str] = []
        validated_queries: List[str] = []
        for index, (entry, sentence, query) in enumerate(
            zip(store, sentences, queries)
        ):
            if not isinstance(entry, dict):
                raise ValueError(f"IKE edit state entry {index} must be an object")
            for field in ("prompt", "subject", "target_new", "fact"):
                value = entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"IKE edit state entry {index} has invalid {field}"
                    )
            target_old = entry.get("target_old")
            if target_old is not None and not isinstance(target_old, str):
                raise ValueError(f"IKE edit state entry {index} has invalid target_old")
            if not isinstance(query, str) or query != entry["prompt"]:
                raise ValueError(
                    f"IKE edit state query {index} does not match its memory prompt"
                )
            expected_fact = f"{entry['prompt']} {entry['target_new']}"
            if entry["fact"] != expected_fact:
                raise ValueError(
                    f"IKE edit state entry {index} has an inconsistent fact"
                )
            expected_sentence = self.icl_prompt_template.format(
                fact=expected_fact,
                prompt=entry["prompt"],
                target=entry["target_new"],
            )
            if not isinstance(sentence, str) or sentence != expected_sentence:
                raise ValueError(
                    f"IKE edit state sentence {index} is incompatible with its template"
                )

            validated_store.append(dict(entry))
            validated_sentences.append(sentence)
            validated_queries.append(query)

        return validated_store, validated_sentences, validated_queries

    @staticmethod
    def _validate_effective_backend(
        effective_backend: Any,
        *,
        configured_backend: str,
        use_icl_examples: bool,
        memory_size: int,
    ) -> None:
        if not use_icl_examples:
            allowed = {"disabled"}
        elif configured_backend == "token_overlap":
            allowed = {"token_overlap"}
        elif configured_backend == "sentence_transformer":
            allowed = {"sentence_transformer"}
        elif memory_size:
            allowed = {"sentence_transformer", "token_overlap"}
        else:
            allowed = {"sentence_transformer", "token_overlap", "unresolved"}

        if effective_backend not in allowed:
            raise ValueError(
                "IKE edit state effective retrieval backend is incompatible with "
                f"its configuration: {effective_backend!r} not in {sorted(allowed)}"
            )

    def _restore_embedding_index(
        self, queries: List[str], effective_backend: str
    ) -> tuple[Optional[torch.Tensor], bool]:
        if effective_backend in {"disabled", "unresolved"}:
            return None, False
        if effective_backend == "token_overlap":
            return None, self.retrieval_backend == "auto"
        if not queries:
            return None, False
        try:
            return self._encode_queries(queries), False
        except Exception as exc:
            raise RuntimeError(
                "IKE could not restore the persisted sentence-transformer index; "
                "refusing to change retrieval semantics"
            ) from exc

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 IKE 知识编辑

        将编辑请求存入知识库并更新向量索引。
        IKE 不修改模型权重，编辑效果在推理时通过 ICL 实现。

        Args:
            requests: 编辑请求（单个或列表）

        Returns:
            包含编辑结果的字典：
            - success: 是否成功
            - edited_count: 成功编辑数
            - icl_examples: 为每个请求生成的 ICL 演示（供外部调用）
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "variant": self.variant,
            "edited_count": 0,
            "icl_examples": [],
        }

        if not isinstance(requests, list):
            logger.error("IKE edit expects an EditRequest or a list of EditRequest")
            results["success"] = False
            return results
        if not requests:
            return results
        if self._independent_protocol() and len(requests) != 1:
            raise ValueError(
                "IKE copy/update/retain evaluates one active edit; use independent mode"
            )
        if self._independent_protocol() and self.knowledge_store:
            raise ValueError("Reset IKE independent state before applying another edit")
        if not self.use_icl_examples:
            results["success"] = False
            results["error"] = (
                "IKE editing requires use_icl_examples=true; runtime is disabled"
            )
            return results

        snapshot = self._snapshot_runtime_state()
        try:
            for request in requests:
                self._store_edit(request)

            # Retrieval itself is part of the transaction.  A strict semantic
            # backend can still fail while embedding the query, and must not
            # leave a committed memory entry behind in that case.
            results["icl_examples"] = [
                self.get_icl_prompt(request.prompt) for request in requests
            ]
        except Exception as exc:
            self._restore_runtime_state(snapshot)
            logger.error("IKE edit transaction failed: %s", exc)
            results["success"] = False
            return results

        # EditSuite and generation helpers discover the IKE runtime here.  The
        # base model remains unchanged; prompts are augmented at inference.
        self.model._edit_runtime = self
        results["edited_count"] = len(requests)
        if self._independent_protocol():
            results["demonstration_protocol"] = {
                "profile": self.protocol,
                "corpus_sha256": self.demonstration_sha256,
                "corpus_records": len(self.demonstration_records),
                "seed": self.demonstration_seed,
                "retrieval_backend": "sentence_transformer",
                "demonstrations": self.k,
                "types": {"copy": 4, "update": 12, "retain": 16}
                if self.protocol == "paper_copy_update_retain"
                else "semantic_top_k",
                "active_fact_copy": self.protocol == "paper_copy_update_retain",
            }
        return results

    def _store_edit(self, request: EditRequest):
        """Stage and atomically commit one edit to memory and its index."""
        self._validate_memory_state()
        self._validate_request(request)
        if self._independent_protocol():
            self._check_demonstration_split(request)

        fact = f"{request.prompt} {request.target_new}"
        entry = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
            "target_old": request.target_old,
            "fact": fact,
        }
        sentence = self.icl_prompt_template.format(
            fact=fact, prompt=request.prompt, target=request.target_new
        )

        sentence_model_before = self._sentence_model
        failed_before = self._sentence_model_failed
        try:
            embeddings, sentence_model_failed = self._prepare_embeddings_for_append(
                [request.prompt]
            )
        except Exception:
            # Lazy model loading is also editor runtime state.  A failed edit
            # must be observationally equivalent to never having been tried.
            self._sentence_model = sentence_model_before
            self._sentence_model_failed = failed_before
            raise

        # Nothing below performs fallible model work.  Assign complete copies
        # together so the three memory views and vector index cannot diverge.
        self.knowledge_store = [*self.knowledge_store, entry]
        self.knowledge_sentences = [*self.knowledge_sentences, sentence]
        self.knowledge_queries = [*self.knowledge_queries, request.prompt]
        self.knowledge_embeddings = embeddings
        self._sentence_model_failed = sentence_model_failed

        logger.info(
            "Stored edit: %s -> %s (store size: %d)",
            request.subject,
            request.target_new,
            len(self.knowledge_store),
        )

    def _prepare_embeddings_for_append(
        self, new_queries: List[str]
    ) -> tuple[Optional[torch.Tensor], bool]:
        """Return a candidate index without mutating committed editor state."""
        if self._independent_protocol():
            return None, False
        if not self.use_icl_examples:
            return None, False

        if self.retrieval_backend == "token_overlap":
            return None, False
        if self.retrieval_backend == "auto" and self._sentence_model_failed:
            return None, True

        try:
            new_embeddings = self._encode_queries(new_queries)
            if self.knowledge_embeddings is None:
                if self.knowledge_store:
                    raise ValueError(
                        "IKE semantic memory has entries but no committed embeddings"
                    )
                embeddings = new_embeddings
            else:
                device = new_embeddings.device
                embeddings = torch.cat(
                    [self.knowledge_embeddings.to(device), new_embeddings], dim=0
                )
        except Exception as exc:
            if self.retrieval_backend == "sentence_transformer":
                raise
            logger.warning(
                "IKE sentence-transformer unavailable (%s); using token-overlap retrieval.",
                exc,
            )
            return None, True

        return embeddings, False

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode and validate a query batch for semantic retrieval."""
        if not queries:
            raise ValueError("IKE cannot encode an empty query batch")
        if st_util is None:
            raise ImportError(
                "sentence-transformers is required for semantic IKE retrieval"
            )

        device = next(self.model.parameters()).device
        embeddings = self.sentence_model.encode(
            queries,
            show_progress_bar=False,
            convert_to_tensor=True,
        )
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.as_tensor(embeddings)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.dim() != 2 or embeddings.shape[0] != len(queries):
            raise ValueError(
                "IKE embedding model returned an incompatible batch shape: "
                f"expected {len(queries)} rows, got {tuple(embeddings.shape)}"
            )
        embeddings = st_util.normalize_embeddings(embeddings.to(device))
        if not torch.isfinite(embeddings).all():
            raise ValueError("IKE embedding model returned non-finite values")
        return embeddings

    def get_icl_prompt(self, query_prompt: str) -> str:
        """根据查询 prompt 检索最相关的 ICL 演示

        Args:
            query_prompt: 用户查询 prompt

        Returns:
            拼接好的 ICL 演示字符串，可直接前置到模型输入
        """
        if self._independent_protocol():
            return self._independent_icl_prompt()
        if not self.use_icl_examples or not self.knowledge_store:
            return ""

        top_k = min(self.k, len(self.knowledge_store))
        indices: List[int]
        if self.knowledge_embeddings is not None and not self._sentence_model_failed:
            device = self.knowledge_embeddings.device
            query_emb = self._encode_queries([query_prompt]).to(device)
            hits = st_util.semantic_search(
                query_emb,
                self.knowledge_embeddings,
                score_function=st_util.dot_score,
                top_k=top_k,
            )
            indices = [hit["corpus_id"] for hit in hits[0]] if hits else []
        else:
            query_tokens = self._lexical_tokens(query_prompt)
            scored = []
            for index, stored_prompt in enumerate(self.knowledge_queries):
                stored_tokens = self._lexical_tokens(stored_prompt)
                union = query_tokens | stored_tokens
                score = len(query_tokens & stored_tokens) / max(len(union), 1)
                scored.append((score, index))
            indices = [index for _, index in sorted(scored, reverse=True)[:top_k]]

        icl_parts = [self.knowledge_sentences[index] for index in indices]
        return "".join(icl_parts)

    @staticmethod
    def _lexical_tokens(text: str) -> set[str]:
        return set(re.findall(r"[\w']+", text.lower()))

    def transform_prompt(self, prompt: str) -> str:
        """Apply the retrieved ICL demonstrations to an evaluation prompt."""
        examples = self.get_icl_prompt(prompt)
        if not examples:
            return prompt
        if self._independent_protocol():
            transformed = f"{examples}New Fact: {self.knowledge_store[-1]['fact']}\nPrompt: {prompt}"
            tokenizer = getattr(self, "processing_class", None)
            config = getattr(self.model, "config", None)
            limit = getattr(
                config, "max_position_embeddings", getattr(config, "n_positions", None)
            )
            if (
                tokenizer is not None
                and limit is not None
                and len(tokenizer.encode(transformed, add_special_tokens=True)) >= limit
            ):
                raise ValueError(
                    "IKE demonstrations exceed the model context window; use a longer-context model rather than truncating the active query"
                )
            return transformed
        return f"{examples}Prompt: {prompt}\nAnswer:"

    def select_model(self, prompt: str):
        """Runtime protocol used by EditSuite; IKE always uses the base model."""
        del prompt
        return self.model

    def rebuild_embeddings(self):
        """从头重建整个知识库的向量索引。

        当知识库被外部修改或需要更换 sentence model 时调用。
        """
        self._validate_memory_collections()
        if not self.use_icl_examples or not self.knowledge_queries:
            self.knowledge_embeddings = None
            self._sentence_model_failed = False
            return
        if self.retrieval_backend == "token_overlap":
            self.knowledge_embeddings = None
            self._sentence_model_failed = False
            return
        if self.retrieval_backend == "auto" and self._sentence_model_failed:
            self.knowledge_embeddings = None
            return

        sentence_model_before = self._sentence_model
        try:
            embeddings = self._encode_queries(self.knowledge_queries)
        except Exception as exc:
            if self.retrieval_backend == "sentence_transformer":
                self._sentence_model = sentence_model_before
                raise
            self.knowledge_embeddings = None
            self._sentence_model_failed = True
            logger.warning(
                "IKE sentence-transformer unavailable (%s); using token-overlap retrieval.",
                exc,
            )
            return

        self.knowledge_embeddings = embeddings
        self._sentence_model_failed = False
        logger.info("Rebuilt embeddings for %d entries", len(self.knowledge_sentences))

    def save_edit_state(self, path: str | Path) -> None:
        """Persist memory together with all retrieval-semantic configuration."""
        self._validate_retrieval_config(
            self._retrieval_config(), source="IKE runtime configuration"
        )
        self._validate_memory_state()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_version": self._STATE_VERSION,
            **self._retrieval_config(),
            "effective_retrieval_backend": self._effective_retrieval_backend(),
            "knowledge_store": self.knowledge_store,
            "knowledge_sentences": self.knowledge_sentences,
            "knowledge_queries": self.knowledge_queries,
        }
        if self._independent_protocol():
            payload.update(
                protocol=self.protocol,
                demonstration_seed=self.demonstration_seed,
                demonstration_sha256=self.demonstration_sha256,
                demonstration_records=self.demonstration_records,
            )
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load_edit_state(self, path: str | Path) -> None:
        """Atomically load a state only when retrieval semantics are compatible."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("IKE edit state must contain a JSON object")
        if payload.get("protocol", "edit_memory") != getattr(
            self, "protocol", "edit_memory"
        ):
            raise ValueError("IKE state protocol differs from runtime")
        if self._independent_protocol():
            if payload.get("demonstration_seed") != self.demonstration_seed:
                raise ValueError("IKE state demonstration seed differs from runtime")
            records, _ = self.build_demonstrations(payload.get("demonstration_records"))
            digest = self._corpus_digest(records)
            if (
                digest != payload.get("demonstration_sha256")
                or digest != self.demonstration_sha256
            ):
                raise ValueError("IKE state demonstration corpus differs from runtime")
        version = payload.get("state_version")
        if version != self._STATE_VERSION:
            raise ValueError(
                "IKE edit state version is incompatible: "
                f"expected {self._STATE_VERSION}, got {version!r}. "
                "Regenerate legacy state so retrieval semantics can be audited."
            )

        saved_config = {
            field: payload[field]
            for field in self._RETRIEVAL_CONFIG_FIELDS
            if field in payload
        }
        self._validate_retrieval_config(
            saved_config, source="persisted IKE retrieval configuration"
        )
        runtime_config = self._retrieval_config()
        mismatches = [
            field
            for field in self._RETRIEVAL_CONFIG_FIELDS
            if saved_config[field] != runtime_config[field]
        ]
        if mismatches:
            details = "; ".join(
                f"{field}: state={saved_config[field]!r}, "
                f"runtime={runtime_config[field]!r}"
                for field in mismatches
            )
            raise ValueError(
                "IKE edit state retrieval configuration is incompatible with "
                f"the runtime ({details})"
            )

        store, sentences, queries = self._validated_state_memory(payload)
        effective_backend = payload.get("effective_retrieval_backend")
        self._validate_effective_backend(
            effective_backend,
            configured_backend=saved_config["retrieval_backend"],
            use_icl_examples=saved_config["use_icl_examples"],
            memory_size=len(store),
        )

        sentence_model_before = self._sentence_model
        try:
            embeddings, sentence_model_failed = (
                (None, False)
                if self._independent_protocol()
                else self._restore_embedding_index(queries, effective_backend)
            )
        except Exception:
            self._sentence_model = sentence_model_before
            raise

        # Commit only after config, memory, and the complete embedding index
        # have all validated successfully.
        self.knowledge_store = store
        self.knowledge_sentences = sentences
        self.knowledge_queries = queries
        self.knowledge_embeddings = embeddings
        self._sentence_model_failed = sentence_model_failed
        self.model._edit_runtime = self

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        target_dir = Path(output_dir or self.args.output_dir)
        self.save_edit_state(target_dir / "ike_edit_state.json")

    def clear_knowledge_store(self):
        """清空知识库和向量索引。"""
        self.knowledge_store.clear()
        self.knowledge_sentences.clear()
        self.knowledge_queries.clear()
        self.knowledge_embeddings = None
        if getattr(self.model, "_edit_runtime", None) is self:
            delattr(self.model, "_edit_runtime")
        logger.info("Knowledge store cleared")

    def reset_independent_state(self) -> None:
        """Restore the configured base memory between independent cases."""
        self.clear_knowledge_store()
        if self.memory_path:
            self.load_edit_state(self.memory_path)
