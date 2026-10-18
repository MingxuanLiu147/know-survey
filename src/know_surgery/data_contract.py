"""Lossless, dependency-free knowledge records shared by CLI and upload adapters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


OPERATIONS = {"unlearning", "inject", "edit"}
KNOWLEDGE_TYPES = {
    "fact",
    "event",
    "procedural",
    "rule",
    "free_text",
    "concept",
    "multimodal",
}
PRESERVED_FIELDS = {
    "source",
    "context",
    "timestamp",
    "conditions",
    "steps",
    "images",
    "image",
    "image_rephrase",
    "rephrase",
    "rephrase_prompts",
    "generalization",
    "locality",
    "locality_inputs",
    "portability",
    "portability_inputs",
    "multimodal_locality_inputs",
    "provenance",
    "training_rephrases",
    "dataset_role",
    "goal",
    "failure_conditions",
    "priority",
    "constraints",
    "scope_training_prompts",
    "scope_training_negatives",
    "import_provenance",
}


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    return next((record[k] for k in keys if record.get(k) not in (None, "")), None)


def normalize_record(
    record: Mapping[str, Any],
    operation: str,
    *,
    knowledge_type: str | None = None,
    index: int = 0,
    role: str | None = None,
) -> dict[str, Any]:
    """Validate canonical targets and preserve source fields without inventing probes."""
    if not isinstance(record, Mapping):
        raise ValueError("Each knowledge record must be a JSON object.")
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")
    if record.get("operation", operation) != operation:
        raise ValueError("Record operation does not match the requested route.")
    kind = knowledge_type or record.get("knowledge_type", "fact")
    if kind not in KNOWLEDGE_TYPES:
        raise ValueError(f"Unsupported knowledge_type: {kind}")
    sample = deepcopy(dict(record))
    sample.update(
        id=str(record.get("id") or f"sample-{index:06d}"),
        operation=operation,
        knowledge_type=kind,
        prompt=_first(record, "prompt", "instruction", "question", "input", "text"),
    )
    if operation in {"inject", "edit"}:
        sample["target_new"] = _first(
            record, "target_new", "output", "answer", "response"
        )
    if operation == "unlearning":
        sample["target_old"] = _first(record, "target_old", "old_answer", "answer")
    elif operation == "edit":
        old = _first(record, "target_old", "old_answer", "target_true")
        if old is not None:
            sample["target_old"] = old
    required = ["prompt", "target_old" if operation == "unlearning" else "target_new"]
    for key in required:
        if not isinstance(sample.get(key), str) or not sample[key].strip():
            raise ValueError(f"Missing or non-text required field: {key}")
    selected_role = role or record.get("dataset_role")
    roles = {"unlearning": {"forget", "retain"}, "inject": {"train"}, "edit": {"edit"}}
    if selected_role is not None:
        if role and record.get("dataset_role", role) != role:
            raise ValueError("Record dataset role conflicts with the supplied role.")
        if selected_role not in roles[operation]:
            raise ValueError(f"Invalid dataset role {selected_role!r} for {operation}")
        sample["dataset_role"] = selected_role
    for key in ("steps", "images"):
        if key in sample and not isinstance(sample[key], list):
            raise ValueError(f"{key} must be a list when supplied.")
    if operation == "edit":
        for dest, aliases in {
            "rephrase_prompts": ("rephrase_prompts", "rephrase", "generalization"),
            "locality_inputs": ("locality_inputs", "locality"),
            "portability_inputs": ("portability_inputs", "portability"),
        }.items():
            value = _first(sample, *aliases)
            if value is not None:
                sample[dest] = (
                    rephrase_prompts(value)
                    if dest == "rephrase_prompts"
                    else side_effect_probes(value, dest.removesuffix("_inputs"))
                )
    return sample


def _probe_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return int(bool(value.strip()))
    if isinstance(value, list):
        return sum(_probe_count(item) for item in value)
    if isinstance(value, dict):
        if "prompt" in value or "question" in value:
            prompt = value.get("prompt", value.get("question"))
            return len(prompt) if isinstance(prompt, list) else int(bool(prompt))
        return sum(_probe_count(v) for v in value.values())
    raise ValueError("Evaluation probes must contain text, lists, or objects.")


def rephrase_prompts(value: Any) -> list[str]:
    """Extract actual questions from structured rephrases, preserving original metadata."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [prompt for item in value for prompt in rephrase_prompts(item)]
    if isinstance(value, dict):
        prompt = value.get("prompt", value.get("question"))
        if prompt is not None:
            return rephrase_prompts(prompt)
        raise ValueError("A structured rephrase requires a prompt or question field.")
    raise ValueError("Rephrases must be text or objects containing a question.")


def side_effect_probes(value: Any, group: str) -> dict[str, list[dict[str, str]]]:
    """Normalize scalar/vector QA probes and reject silently unevaluable inputs."""
    if value is None or value == [] or value == {}:
        return {}
    if isinstance(value, dict) and not any(k in value for k in ("prompt", "question")):
        groups = {}
        for name, entries in value.items():
            normalized = side_effect_probes(entries, str(name))
            for key, items in normalized.items():
                groups.setdefault(key, []).extend(items)
        return groups
    values = value if isinstance(value, list) else [value]
    entries = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(f"{group} probes require prompt and ground_truth.")
        prompts = item.get("prompt", item.get("question"))
        answers = _first(
            item,
            "ground_truth",
            "expected",
            "answer",
            "target",
            "target_new",
            "target_true",
        )
        prompts = prompts if isinstance(prompts, list) else [prompts]
        answers = answers if isinstance(answers, list) else [answers] * len(prompts)
        if len(prompts) != len(answers):
            raise ValueError(f"{group} prompt and answer counts differ.")
        for prompt, answer in zip(prompts, answers):
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not isinstance(answer, str)
                or not answer.strip()
            ):
                raise ValueError(
                    f"{group} probes require nonempty prompt and ground_truth text."
                )
            entries.append({"prompt": prompt, "ground_truth": answer})
    return {group: entries} if entries else {}


def build_cases(
    records: list[dict[str, Any]], operation: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt records to existing dataset handlers; report actual supplied probe counts."""
    if not records:
        raise ValueError("Cannot build cases from empty input.")
    cases = []
    ids = set()
    counts = {"rephrase": 0, "locality": 0, "portability": 0}
    role_counts: dict[str, int] = {}
    scopes: dict[str, set[tuple[str, str]]] = {"forget": set(), "retain": set()}
    for index, record in enumerate(records):
        case = normalize_record(record, operation, index=index)
        identity = (case.get("dataset_role"), case["id"])
        if identity in ids:
            raise ValueError(f"Duplicate record id within a role: {case['id']}")
        ids.add(identity)
        role = case.get("dataset_role", "edit" if operation == "edit" else "train")
        if operation == "unlearning" and role not in {"forget", "retain"}:
            raise ValueError("Unlearning cases require explicit forget/retain roles.")
        case["dataset_role"] = role
        role_counts[role] = role_counts.get(role, 0) + 1
        if operation == "inject":
            case.update(instruction=case["prompt"], output=case["target_new"])
        elif operation == "unlearning":
            case.update(question=case["prompt"], answer=case["target_old"])
            scopes[role].add((case["prompt"].strip(), case["target_old"].strip()))
        else:
            for dest, aliases in {
                "rephrase_prompts": ("rephrase_prompts", "rephrase", "generalization"),
                "locality_inputs": ("locality_inputs", "locality"),
                "portability_inputs": ("portability_inputs", "portability"),
            }.items():
                value = _first(case, *aliases)
                if value is not None:
                    case[dest] = (
                        rephrase_prompts(value)
                        if dest == "rephrase_prompts"
                        else deepcopy(value)
                    )
        for label, keys in {
            "rephrase": ("rephrase_prompts", "rephrase", "generalization"),
            "locality": ("locality_inputs", "locality"),
            "portability": ("portability_inputs", "portability"),
        }.items():
            counts[label] += _probe_count(_first(case, *keys))
        cases.append(case)
    if operation == "unlearning" and set(role_counts) != {"forget", "retain"}:
        raise ValueError("Unlearning requires nonempty forget and retain cases.")
    if operation == "unlearning" and scopes["forget"] & scopes["retain"]:
        raise ValueError(
            "Forget and retain scopes contain identical prompt/target pairs."
        )
    return cases, {
        "records": len(cases),
        "role_records": role_counts,
        "probe_counts": counts,
        "missing_probes": [key for key, count in counts.items() if count == 0],
        "adapter": {
            "edit": "EditingDataset",
            "inject": "AlpacaDataset",
            "unlearning": "QADataset",
        }[operation],
        "semantic_chunking": False,
    }
