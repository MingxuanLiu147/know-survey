#!/usr/bin/env python3
"""Prepare independent IKE demonstrations, SERAC auxiliaries and WISE negatives.

No models or downloads. Training rephrases/locality are copied only from an
explicit auxiliary source; ALL held-out rows are reserved before selection.
Raw CounterFact, zsRE and canonical editing dictionaries are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompts(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [text for item in value for text in prompts(item)]
    if isinstance(value, dict):
        if "prompt" in value:
            return prompts(value["prompt"])
        return [text for item in value.values() for text in prompts(item)]
    return []


def questions(row, include_training=True):
    fields = [
        "prompt",
        "src",
        "rephrase",
        "rephrase_prompt",
        "rephrase_prompts",
        "paraphrase_prompts",
        "loc",
        "locality",
        "locality_inputs",
        "locality_prompt",
        "neighborhood_prompts",
        "portability",
        "portability_inputs",
    ]
    if include_training:
        fields += ["scope_training_prompts", "scope_training_negatives"]
    result = {normalized(text) for key in fields for text in prompts(row.get(key))}
    if "requested_rewrite" in row:
        rewrite = row["requested_rewrite"]
        result.add(normalized(rewrite["prompt"].format(rewrite["subject"])))
    return result


def validate_disjoint(training, heldout):
    """Reject exact normalized prompt or subject overlap, including nested probes."""
    train_questions = set().union(*(questions(row) for row in training))
    heldout_questions = set().union(*(questions(row) for row in heldout))
    train_subjects = {
        normalized(row["subject"]) for row in training if row.get("subject")
    }
    heldout_subjects = {
        normalized(
            row.get("subject", row.get("requested_rewrite", {}).get("subject", ""))
        )
        for row in heldout
    } - {""}
    train_ids = {str(row["id"]) for row in training if row.get("id") is not None}
    heldout_ids = {str(row["id"]) for row in heldout if row.get("id") is not None}
    if (
        train_questions & heldout_questions
        or train_subjects & heldout_subjects
        or train_ids & heldout_ids
    ):
        raise ValueError("Auxiliary training overlaps held-out prompts or subjects")
    return {
        "training_prompt_count": len(train_questions),
        "heldout_prompt_count": len(heldout_questions),
        "prompt_overlap": 0,
        "subject_overlap": 0,
        "source_id_overlap": 0,
        "normalization": "NFKC + casefold + whitespace",
        "limitation": "Exact overlap checks do not establish semantic independence",
    }


def _answer(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return _answer(value[0])
    if isinstance(value, dict):
        return _answer(value.get("str", value.get("ground_truth")))
    return ""


def _probes(value):
    if isinstance(value, list):
        return [probe for item in value for probe in _probes(item)]
    if isinstance(value, dict):
        if isinstance(value.get("prompt"), str):
            answer = _answer(value.get("ground_truth", value.get("target")))
            return (
                [{"prompt": value["prompt"], "ground_truth": answer}] if answer else []
            )
        return [probe for item in value.values() for probe in _probes(item)]
    return []


def canonical(row, index, source):
    """Adapt observed labels only; missing paraphrases/answers stay missing."""
    rewrite = row.get("requested_rewrite")
    subject = row.get("subject", "")
    if rewrite:
        subject = rewrite["subject"]
        prompt = rewrite["prompt"].format(subject)
        target = _answer(rewrite.get("target_new"))
        old = _answer(rewrite.get("target_true"))
        rephrase = prompts(row.get("paraphrase_prompts"))
        locality = (
            [
                {"prompt": text, "ground_truth": old}
                for text in prompts(row.get("neighborhood_prompts"))
            ]
            if old
            else []
        )
    else:
        prompt = row.get("prompt", row.get("src", ""))
        target = _answer(row.get("target_new", row.get("alt")))
        old = _answer(
            row.get("target_old", row.get("ground_truth", row.get("answers")))
        )
        rephrase = prompts(
            row.get(
                "scope_training_prompts",
                row.get(
                    "rephrase_prompts", row.get("rephrase_prompt", row.get("rephrase"))
                ),
            )
        )
        locality = _probes(row.get("locality_inputs", row.get("locality")))
        if not locality and row.get("loc") and row.get("loc_ans"):
            locality = [{"prompt": row["loc"], "ground_truth": _answer(row["loc_ans"])}]
    if not all(
        isinstance(value, str) and value.strip() for value in (prompt, target, subject)
    ):
        raise ValueError("missing_prompt_target_or_subject")
    result = {
        "id": str(row.get("id", row.get("case_id", f"{source}:{index}"))),
        "prompt": prompt,
        "subject": subject,
        "target_new": target,
        "target_old": old or None,
        "rephrase_prompts": rephrase,
        "locality_inputs": locality,
        "portability_inputs": row.get("portability_inputs", row.get("portability")),
        "source_index": index,
    }
    return result


def prepare(
    auxiliary,
    heldout,
    *,
    train_count=64,
    validation_count=16,
    negative_count=2,
    heldout_count=2,
):
    if min(train_count, validation_count, negative_count, heldout_count) < 1:
        raise ValueError("All split counts must be positive")
    reserved = set().union(*(questions(row) for row in heldout))
    reserved_subjects = {
        normalized(
            row.get("subject", row.get("requested_rewrite", {}).get("subject", ""))
        )
        for row in heldout
    } - {""}
    evaluation, rejected = [], []
    for index, row in enumerate(heldout):
        try:
            case = canonical(row, index, "heldout")
            if not case["rephrase_prompts"] or not case["locality_inputs"]:
                raise ValueError("missing_heldout_side_effect_probes")
        except ValueError as error:
            rejected.append({"split": "heldout", "index": index, "reason": str(error)})
            continue
        evaluation.append(case)
        if len(evaluation) == heldout_count:
            break
    selected, used, used_subjects = [], set(), set()
    for index, row in enumerate(auxiliary):
        try:
            case = canonical(row, index, "auxiliary")
            question_set = questions(case)
            subject = normalized(case["subject"])
            if (
                question_set & (reserved | used)
                or subject in reserved_subjects | used_subjects
            ):
                raise ValueError("prompt_or_subject_overlap")
            if not case["rephrase_prompts"] or not case["locality_inputs"]:
                raise ValueError("missing_auxiliary_rephrase_or_labeled_locality")
            positives = {
                normalized(case["prompt"]),
                *map(normalized, case["rephrase_prompts"]),
            }
            negatives = {
                normalized(probe["prompt"]) for probe in case["locality_inputs"]
            }
            if positives & negatives or normalized(
                case["rephrase_prompts"][0]
            ) == normalized(case["prompt"]):
                raise ValueError("positive_negative_or_copy_update_overlap")
        except ValueError as error:
            rejected.append(
                {"split": "auxiliary", "index": index, "reason": str(error)}
            )
            continue
        selected.append(case)
        used.update(question_set)
        used_subjects.add(subject)
        if len(selected) == train_count + validation_count + negative_count:
            break
    if (
        len(selected) != train_count + validation_count + negative_count
        or len(evaluation) != heldout_count
    ):
        raise ValueError(
            "Insufficient independent labeled records for requested splits"
        )
    train = selected[:train_count]
    validation = selected[train_count : train_count + validation_count]
    negative_rows = selected[train_count + validation_count :]
    for case in evaluation:
        case["scope_training_negatives"] = [row["prompt"] for row in negative_rows]

    def auxiliary_rows(rows):
        return [
            {
                "id": row["id"],
                "subject": row["subject"],
                "prompt": row["prompt"],
                "target_new": row["target_new"],
                "scope_training_prompts": row["rephrase_prompts"],
                "scope_training_negatives": [
                    probe["prompt"] for probe in row["locality_inputs"]
                ],
            }
            for row in rows
        ]

    demonstrations = []
    for row in train:
        probe = row["locality_inputs"][0]
        demonstrations.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "target_new": row["target_new"],
                "rephrase_prompt": row["rephrase_prompts"][0],
                "rephrase_prompts": row["rephrase_prompts"],
                "locality_prompt": probe["prompt"],
                "locality_prompts": [
                    p["prompt"]
                    for p in row["locality_inputs"]
                    if p["ground_truth"] == probe["ground_truth"]
                ],
                "locality_ground_truth": probe["ground_truth"],
            }
        )
    splits = {
        "ike_demonstrations.json": demonstrations,
        "serac_train.json": auxiliary_rows(train),
        "serac_validation.json": auxiliary_rows(validation),
        "heldout_edit.json": evaluation,
        "wise_negative_sources.json": negative_rows,
    }
    check = validate_disjoint(selected, heldout)
    return splits, {
        "validation": check,
        "rejected": rejected,
        "selection": {
            "auxiliary": [row["source_index"] for row in selected],
            "heldout": [row["source_index"] for row in evaluation],
        },
        "reserved_heldout_records": len(heldout),
        "wise_negatives": "dedicated auxiliary records excluded from both SERAC train and validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--negative-count", type=int, default=2)
    parser.add_argument("--heldout-count", type=int, default=2)
    args = parser.parse_args()
    sources = {
        key: json.loads(path.read_text(encoding="utf-8"))
        for key, path in (("auxiliary", args.auxiliary), ("heldout", args.heldout))
    }
    if any(not isinstance(rows, list) or not rows for rows in sources.values()):
        parser.error("Both source files must contain non-empty JSON lists")
    splits, manifest = prepare(
        **sources,
        train_count=args.train_count,
        validation_count=args.validation_count,
        negative_count=args.negative_count,
        heldout_count=args.heldout_count,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest["sources"] = {
        key: {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(sources[key]),
        }
        for key, path in (("auxiliary", args.auxiliary), ("heldout", args.heldout))
    }
    manifest["outputs"] = {}
    for name, rows in splits.items():
        path = args.output / name
        path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest["outputs"][name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(rows),
        }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"outputs": manifest["outputs"], "validation": manifest["validation"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
