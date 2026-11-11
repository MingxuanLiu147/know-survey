"""Model-free editing record normalization shared by loaders and audit tools."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

ZSRE_LOCAL_FILES = {
    "train": "data/edit/zsre/zsre_mend_train.json",
    "test": "data/edit/zsre/ZsRE-test-all.json",
    "all": "data/edit/zsre/ZsRE-test-all.json",
}

# Shared with the runtime adapters: selecting a different split/version/subset
# must bind the file actually consumed, without importing torch in preflight.
EDITING_DATA_DEFAULTS = {
    "ZSREDataset": ("split", "test", ZSRE_LOCAL_FILES),
    "CounterFactDataset": (
        "split",
        "train",
        {
            "train": "data/edit/counterfact/counterfact_train.json",
            "test": "data/edit/counterfact/counterfact_test.json",
        },
    ),
    "ELKENDataset": (
        "split",
        "train",
        {
            "train": "data/edit/elken/train.json",
            "test": "data/edit/elken/test_repaired_drop_incomplete_tail.json",
        },
    ),
    "UnKEDataset": (
        "version",
        "v3",
        {
            "v2": "data/edit/unke/final_data_v2.json",
            "v3": "data/edit/unke/final_data_v3.json",
        },
    ),
    "ConceptEditDataset": (
        "variant",
        "intra",
        {"intra": "data/edit/conceptedit/concept_data.json"},
    ),
    "AKEWDataset": (
        "subset",
        "counterfact",
        {
            "counterfact": "data/edit/akew/CounterFact.json",
            "mquake-cf": "data/edit/akew/MQuAKE-CF.json",
            "wikiupdate": "data/edit/akew/WikiUpdate.json",
        },
    ),
    "LEMEDataset": (
        "subset",
        "counterfact",
        {
            "counterfact": "data/edit/leme/counterfact_with_coupled_entities.json",
            "zsre": "data/edit/leme/zsre_mend_eval_with_coupled_entities.json",
        },
    ),
    "EditEveryDataset": ("split", "all", {"all": "data/edit/editevery/editevery.json"}),
}


def default_edit_data_path(handler: str, args) -> str | None:
    """Return an adapter's default local file for its selected split/variant."""
    if handler not in EDITING_DATA_DEFAULTS:
        return None
    selector, fallback, files = EDITING_DATA_DEFAULTS[handler]
    selected = str(args.get(selector) or fallback)
    if selector in {"variant", "subset"}:
        selected = selected.lower()
    return files.get(selected, files[fallback])


def file_sha256(path: Path) -> str:
    """Hash an input without reading the entire file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict]:
    """Read a JSON/JSONL editing dataset and reject malformed containers."""
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("data")
    else:
        raise ValueError(
            "Editing data must be JSON or JSONL. / 编辑数据须为 JSON 或 JSONL。"
        )
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(
            "Editing data must contain a list of objects. / 编辑数据须由对象记录组成。"
        )
    return rows


def resolve_data_path(explicit, default, *, root: Path, data_roots=None) -> Path | None:
    """Honor an explicit file; otherwise search configured roots before fixtures."""
    if explicit:
        path = Path(explicit).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    if not default:
        return None
    relative = Path(default)
    if relative.is_absolute():
        return relative
    roots = (
        data_roots
        if data_roots is not None
        else [
            x
            for x in os.environ.get("KNOW_SURGERY_DATA_ROOTS", "").split(os.pathsep)
            if x
        ]
    )
    if isinstance(roots, str):
        roots = [roots]
    found = []
    for directory in roots:
        base = Path(directory).expanduser()
        if not base.is_absolute():
            base = root / base
        suffixes = [relative]
        if relative.parts[0] == "data":
            suffixes.append(Path(*relative.parts[1:]))
        for suffix in suffixes:
            candidate = (base / suffix).resolve()
            if candidate.is_file() and candidate not in found:
                found.append(candidate)
    if len(found) > 1:
        raise ValueError(
            "Multiple data files match; set data_path explicitly. / 存在多个候选数据文件，请明确指定 data_path: "
            + ", ".join(map(str, found))
        )
    return found[0] if found else (root / relative).resolve()


def zsre_fields(item: dict) -> dict:
    """Map both MEND-zsRE and KnowEdit records to the loader's shared fields."""
    if "src" in item and "alt" in item:
        return {
            "prompt": item.get("src"),
            "subject": item.get("subject"),
            "target_new": item.get("alt"),
            "target_old": item.get("answers") or item.get("pred"),
            "rephrase_prompts": item.get("rephrase"),
            "locality_inputs": {
                "Relation_Specificity": [
                    {"prompt": item.get("loc"), "ground_truth": item.get("loc_ans")}
                ]
            },
            "portability_inputs": item.get(
                "portability_inputs", item.get("portability")
            ),
            "prompt_template": item.get("prompt_template"),
        }
    return {
        "prompt": item.get("prompt"),
        "subject": item.get("subject"),
        "target_new": item.get("target_new"),
        "target_old": item.get("ground_truth", item.get("target_old")),
        "rephrase_prompts": item.get(
            "rephrase_prompts", item.get("rephrase_prompt", item.get("rephrase"))
        ),
        "locality_inputs": item.get("locality_inputs", item.get("locality")),
        "portability_inputs": item.get("portability_inputs", item.get("portability")),
        "prompt_template": item.get("prompt_template"),
    }


def scope_key(text: str) -> str:
    """Normalize spacing and case when checking auxiliary/evaluation overlap."""
    return (
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()
    )


def evaluation_prompts(records: list[dict]) -> set[str]:
    """Collect every edit and evaluation question before auxiliary preparation."""
    normalizer = EditingRecordNormalizer()
    result = set()
    for record in records:
        fields = zsre_fields(record)
        result.add(scope_key(normalizer._flatten_text(fields["prompt"])))
        result.update(
            map(
                scope_key, normalizer._normalize_prompt_list(fields["rephrase_prompts"])
            )
        )
        for field in ("locality_inputs", "portability_inputs"):
            groups = normalizer._normalize_eval_groups(fields[field], field) or {}
            result.update(
                scope_key(probe["prompt"])
                for probes in groups.values()
                for probe in probes
            )
    return result


def validate_scope_sidecar(records: list[dict], sidecar: list[dict]) -> None:
    """Bind training negatives to cases without allowing evaluation-probe leakage."""
    if len(sidecar) != len(records):
        raise ValueError("Scope sidecar must cover every case exactly once.")
    forbidden = evaluation_prompts(records)
    normalizer = EditingRecordNormalizer()
    for index, (record, training) in enumerate(zip(records, sidecar)):
        fields = zsre_fields(record)
        if training.get("case_index") != index or any(
            training.get(key) != normalizer._flatten_text(fields[key]).strip()
            for key in ("prompt", "target_new")
        ):
            raise ValueError(
                f"Scope sidecar case {index} differs from the benchmark identity/order."
            )
        negatives = normalizer._normalize_prompt_list(
            training.get("scope_training_negatives")
        )
        if not negatives or any(scope_key(prompt) in forbidden for prompt in negatives):
            raise ValueError(
                f"Scope sidecar case {index} needs disjoint training negatives."
            )
        positives = normalizer._normalize_prompt_list(
            training.get("scope_training_prompts")
        )
        if any(
            scope_key(prompt) in forbidden - {scope_key(training["prompt"])}
            for prompt in positives
        ):
            raise ValueError(
                f"Scope sidecar case {index} reuses evaluation probes for training."
            )


def build_scope_sidecar(records, training, *, seed=42, negatives_per_case=5):
    """Sample negatives only from independent training questions and subjects."""
    if negatives_per_case < 1:
        raise ValueError("negatives_per_case must be positive")
    normalizer = EditingRecordNormalizer()
    forbidden = evaluation_prompts(records)
    subjects = {
        scope_key(normalizer._flatten_text(zsre_fields(row)["subject"]))
        for row in records
    }
    pool = []
    for row in training:
        fields = zsre_fields(row)
        prompt = normalizer._flatten_text(fields["prompt"]).strip()
        subject = scope_key(normalizer._flatten_text(fields["subject"]))
        if (
            prompt
            and scope_key(prompt) not in forbidden
            and subject
            and subject not in subjects
        ):
            pool.append(prompt)
    pool = sorted(set(pool))
    if len(pool) < negatives_per_case:
        raise ValueError(
            "Not enough independent training negatives after overlap exclusion."
        )
    rng = random.Random(seed)
    sidecar = []
    for index, row in enumerate(records):
        fields = zsre_fields(row)
        sidecar.append(
            {
                "case_index": index,
                "prompt": normalizer._flatten_text(fields["prompt"]).strip(),
                "target_new": normalizer._flatten_text(fields["target_new"]).strip(),
                "scope_training_negatives": rng.sample(pool, negatives_per_case),
            }
        )
    validate_scope_sidecar(records, sidecar)
    return sidecar


class EditingRecordNormalizer:
    """Normalize probes identically in runtime loading and offline auditing."""

    def _format_model_prompt(self, prompt: str) -> str:
        return str(prompt).strip()

    def _normalize_prompt_list(self, value: Any) -> List[str]:
        from know_surgery.data_contract import rephrase_prompts

        return rephrase_prompts(value)

    def _flatten_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            for item in value:
                flattened = self._flatten_text(item).strip()
                if flattened:
                    return flattened
            return ""
        if isinstance(value, dict):
            for key in ("str", "text", "name", "answer", "target", "value", "label"):
                if key in value:
                    flattened = self._flatten_text(value[key]).strip()
                    if flattened:
                        return flattened
            return ""
        return str(value)

    def _expected_from_item(
        self, item: Dict[str, Any], default_target: Any = None
    ) -> str:
        for key in (
            "ground_truth",
            "expected",
            "answer",
            "target",
            "target_new",
            "target_true",
        ):
            if key in item:
                value = self._flatten_text(item[key]).strip()
                if value:
                    return value
        return self._flatten_text(default_target).strip()

    def _make_eval_item(self, prompt: Any, target: Any) -> Optional[Dict[str, str]]:
        prompt_text = self._flatten_text(prompt).strip()
        target_text = self._flatten_text(target).strip()
        if not prompt_text or not target_text:
            return None
        return {
            "prompt": self._format_model_prompt(prompt_text),
            "ground_truth": target_text,
        }

    def _coerce_eval_entries(
        self, payload: Any, default_target: Any = None
    ) -> List[Dict[str, str]]:
        entries: List[Dict[str, str]] = []

        if payload is None:
            return entries

        if isinstance(payload, str):
            item = self._make_eval_item(payload, default_target)
            return [item] if item else []

        if isinstance(payload, dict):
            if any(key in payload for key in ("prompt", "question")) and not isinstance(
                payload.get("prompt", payload.get("question")), list
            ):
                item = self._make_eval_item(
                    payload.get("prompt", payload.get("question")),
                    self._expected_from_item(payload, default_target),
                )
                return [item] if item else []

            prompts = payload.get("prompt", payload.get("question"))
            targets = payload.get(
                "ground_truth",
                payload.get(
                    "expected",
                    payload.get("answer", payload.get("target", default_target)),
                ),
            )
            if prompts is not None:
                if isinstance(prompts, str):
                    prompts = [prompts]
                if isinstance(targets, str):
                    targets = [targets] * len(prompts)
                elif not isinstance(targets, list):
                    targets = [targets] * len(prompts)
                if len(prompts) != len(targets):
                    raise ValueError("Probe prompt and target counts differ.")
                for prompt, target in zip(prompts, targets):
                    item = self._make_eval_item(prompt, target)
                    if item:
                        entries.append(item)
                return entries

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    entry = self._make_eval_item(
                        item.get("prompt", item.get("question")),
                        self._expected_from_item(item, default_target),
                    )
                else:
                    entry = self._make_eval_item(item, default_target)
                if entry:
                    entries.append(entry)

        return entries

    def _normalize_eval_groups(
        self, raw_inputs: Any, default_group: str
    ) -> Optional[Dict[str, List[Dict[str, str]]]]:
        if raw_inputs is None:
            return None

        grouped: Dict[str, List[Dict[str, str]]] = {}
        if isinstance(raw_inputs, dict) and not any(
            key in raw_inputs
            for key in (
                "prompt",
                "question",
                "ground_truth",
                "expected",
                "answer",
                "target",
            )
        ):
            for group, payload in raw_inputs.items():
                entries = self._coerce_eval_entries(payload)
                if entries:
                    grouped[str(group)] = entries
            return grouped or None

        entries = self._coerce_eval_entries(raw_inputs)
        if entries:
            grouped[default_group] = entries
        return grouped or None
