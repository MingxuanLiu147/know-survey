#!/usr/bin/env python3
"""Prepare distinct workbench examples from existing local dataset files.

Inputs are KnowEdit CounterFact test JSON, tasksource/ruletaker train/test
Parquet (or JSON), and locuslab/TOFU forget10/retain90 JSON or JSONL.
This script never downloads data. The generated manifest is a local run record;
do not publish it or the extracted dataset files with the source package.
"""

import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Local input file not found: {value}")
    return path


def records(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        text = path.read_text(encoding="utf-8")
        if text.lstrip().startswith("["):
            yield from json.loads(text)
        else:
            yield from (json.loads(line) for line in text.splitlines() if line.strip())


def normalized(value):
    return " ".join(str(value).split()).casefold()


def has_text(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(has_text(item) for item in value)
    return False


def edit_record(row, _index):
    if not all(
        has_text(row.get(key))
        for key in ("subject", "prompt", "target_new", "ground_truth", "rephrase")
    ):
        return None
    locality = row.get("locality") or {}
    if not any(
        has_text(probe.get("prompt")) and has_text(probe.get("ground_truth"))
        for probes in locality.values()
        for probe in probes
    ):
        return None
    return row


def rule_record(row, index, split):
    if not all(has_text(row.get(key)) for key in ("context", "question")):
        return None
    if row.get("label") not in {"entailment", "not entailment"}:
        return None
    return {
        "instruction": f"Context:\n{row['context']}\n\nQuestion:\n"
        f"Is the following statement entailed given the context?\n{row['question']}\n\n"
        "Answer with one of: entailment, not entailment.",
        "input": "",
        "output": row["label"],
        "answer": row["label"],
        "source_dataset": "ruletaker",
        "source_dataset_id": "tasksource/ruletaker",
        "source_split": split,
        "source_config": row.get("config"),
        "row_index": index,
    }


def qa_record(row, _index):
    return (
        row if all(has_text(row.get(key)) for key in ("question", "answer")) else None
    )


def select(path, count, convert, identity, excluded=None):
    selected, indices, seen = [], [], set()
    skipped = {"invalid": 0, "duplicate": 0, "overlap": 0}
    for index, row in enumerate(records(path)):
        result = convert(row, index)
        if result is None:
            skipped["invalid"] += 1
            continue
        key = identity(row)
        if excluded and excluded(row):
            skipped["overlap"] += 1
            continue
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        selected.append(result)
        indices.append(index)
        if len(selected) == count:
            return selected, indices, skipped
    raise ValueError(
        f"{path.name}: need {count} distinct valid records, found {len(selected)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "counterfact-test",
        "ruletaker-train",
        "ruletaker-test",
        "tofu-forget",
        "tofu-retain",
    ):
        parser.add_argument(f"--{name}", type=local_path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=Path("data/workbench"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    output = args.output_root.expanduser().resolve()

    def rule_key(row):
        return normalized(row["context"]), normalized(row["question"])

    def question_key(row):
        return normalized(row["question"])

    edit = select(
        args.counterfact_test,
        args.count,
        edit_record,
        lambda row: normalized(row["prompt"]),
    )
    train = select(
        args.ruletaker_train,
        args.count,
        lambda row, i: rule_record(row, i, "train"),
        rule_key,
    )
    train_indices = set(train[1])
    train_contexts = {
        normalized(row["context"])
        for i, row in enumerate(records(args.ruletaker_train))
        if i in train_indices
    }
    test = select(
        args.ruletaker_test,
        args.count,
        lambda row, i: rule_record(row, i, "test"),
        rule_key,
        excluded=lambda row: normalized(row["context"]) in train_contexts,
    )
    forget = select(args.tofu_forget, args.count, qa_record, question_key)
    forget_questions = {question_key(row) for row in forget[0]}
    retain = select(
        args.tofu_retain,
        args.count,
        qa_record,
        question_key,
        excluded=lambda row: question_key(row) in forget_questions,
    )
    # Preserve official split identity; exact question checks do not infer authors.
    jobs = [
        (
            "edit/counterfact.json",
            args.counterfact_test,
            "zjunlp/KnowEdit",
            "counterfact/test",
            None,
            edit,
        ),
        (
            "inject/ruletaker_train.json",
            args.ruletaker_train,
            "tasksource/ruletaker",
            "train",
            "apache-2.0",
            train,
        ),
        (
            "inject/ruletaker_eval.json",
            args.ruletaker_test,
            "tasksource/ruletaker",
            "test",
            "apache-2.0",
            test,
        ),
        (
            "unlearning/tofu_forget.json",
            args.tofu_forget,
            "locuslab/TOFU",
            "forget10/train",
            "mit",
            forget,
        ),
        (
            "unlearning/tofu_retain.json",
            args.tofu_retain,
            "locuslab/TOFU",
            "retain90/train",
            "mit",
            retain,
        ),
    ]
    pending, manifest = (
        {},
        {"selection": "first distinct valid records in source order", "files": []},
    )
    for name, source, dataset, split, license_id, (rows, indices, skipped) in jobs:
        content = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        pending[output / name] = content
        manifest["files"].append(
            {
                "output": name,
                "records": len(rows),
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "source": str(source),
                "source_sha256": file_hash(source),
                "source_dataset": dataset,
                "source_split": split,
                "source_indices": indices,
                "skipped": skipped,
                "license": license_id,
            }
        )
    pending[output / "manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for path, content in pending.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace different existing output: {path}")
    for path, content in pending.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {"records_per_file": args.count, "files": list(name for name, *_ in jobs)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
