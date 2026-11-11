"""Audit zsRE inputs and saved runs without loading a model or using a GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from know_surgery.edit_data import (
    EditingRecordNormalizer,
    file_sha256,
    read_records,
    zsre_fields,
)

METRICS = ("reliability", "generalization", "locality", "portability")


def audit_dataset(path: Path, expected_count=None, expected_sha256=None) -> dict:
    """Count usable probes with the same normalization rules as ZSREDataset."""
    records = read_records(path)
    normalizer = EditingRecordNormalizer()
    coverage, probes, prompts = Counter(), Counter(), Counter()
    errors, warnings, identities = [], [], []
    for index, row in enumerate(records):
        fields = zsre_fields(row)
        prompt, target, subject = (
            normalizer._flatten_text(fields[key]).strip()
            for key in ("prompt", "target_new", "subject")
        )
        if not prompt or not target:
            errors.append(f"record {index}: missing prompt or target_new")
        if not subject:
            warnings.append(f"record {index}: no explicit subject for rank editors")
        prompts[prompt] += 1
        identities.append({"prompt": prompt, "target_new": target, "subject": subject})
        rephrases = normalizer._normalize_prompt_list(fields["rephrase_prompts"])
        coverage["generalization"] += bool(rephrases)
        probes["generalization"] += len(rephrases)
        coverage["reliability"] += bool(prompt and target)
        for metric in ("locality", "portability"):
            try:
                groups = (
                    normalizer._normalize_eval_groups(
                        fields[metric + "_inputs"], metric
                    )
                    or {}
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"record {index} {metric}: {exc}")
                groups = {}
            coverage[metric] += bool(groups)
            for group, entries in groups.items():
                probes[f"{metric}.{group}"] += len(entries)
    digest = file_sha256(path)
    if expected_count is not None and len(records) != expected_count:
        errors.append(f"sample_count={len(records)}, expected={expected_count}")
    if expected_sha256 and digest != expected_sha256:
        errors.append(f"dataset SHA256 differs from expected={expected_sha256}")
    duplicates = sum(count - 1 for count in prompts.values() if count > 1)
    if duplicates:
        warnings.append(
            f"{duplicates} duplicate prompt occurrences; preserve records and inspect targets"
        )
    for metric in METRICS:
        if coverage[metric] < len(records):
            warnings.append(
                f"{metric}: usable probes in {coverage[metric]}/{len(records)} cases; report coverage"
            )
    return {
        "path": str(path),
        "sha256": digest,
        "count": len(records),
        "coverage": dict(coverage),
        "probe_counts": dict(probes),
        "duplicate_prompt_occurrences": duplicates,
        "case_identities": identities,
        "errors": errors,
        "warnings": warnings,
    }


def _score(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def audit_run(
    method: str, path: Path, expected_count: int, dataset: dict | None
) -> dict:
    """Recompute case-macro metrics and distinguish execution from edit efficacy."""
    errors, warnings, scores = [], [], {}
    result = {
        "method": method,
        "path": str(path),
        "errors": errors,
        "warnings": warnings,
    }
    if path.is_dir():
        path = path / "independent_results.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        errors.append(f"cannot read run: {exc}")
        return result
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        errors.append("run must contain a cases array")
        return result
    cases = payload["cases"]
    if any(not isinstance(case, dict) for case in cases):
        errors.append("cases must contain objects")
        return result
    indices = [case.get("case_index") for case in cases]
    if indices != list(range(expected_count)):
        errors.append("case indices must cover the expected ordered cases exactly once")
    executed = sum(case.get("success") is True for case in cases)
    if payload.get("requested_count") != expected_count or len(cases) != expected_count:
        errors.append("requested_count/case_count differs from the pinned dataset")
    if payload.get("successful_cases") != executed:
        errors.append("successful_cases disagrees with execution records")
    if executed != expected_count or payload.get("success") is not True:
        errors.append(f"incomplete execution: {executed}/{expected_count}")
    aggregate = payload.get("aggregate")
    aggregate = aggregate.get("metrics") if isinstance(aggregate, dict) else None
    if not isinstance(aggregate, dict):
        errors.append("aggregate.metrics must be an object")
        aggregate = {}
    for metric in METRICS:
        values = []
        for index, case in enumerate(cases):
            summary = case.get("summary")
            if summary is not None and not isinstance(summary, dict):
                errors.append(f"case {index}: summary must be an object")
            value = summary.get(metric) if isinstance(summary, dict) else None
            if value is None:
                continue
            if not _score(value):
                errors.append(f"{metric} case {index}: invalid score {value!r}")
            elif case.get("success") is not True:
                errors.append(f"{metric} case {index}: failed execution has a score")
            else:
                values.append(float(value))
        mean = sum(values) / len(values) if values else None
        scores[metric] = {
            "mean": mean,
            "available_cases": len(values),
            "missing_cases": len(cases) - len(values),
        }
        entry = aggregate.get(metric)
        if not isinstance(entry, dict):
            errors.append(f"{metric}: aggregate entry missing")
        else:
            for key in ("available_cases", "missing_cases"):
                if entry.get(key) != scores[metric][key]:
                    errors.append(f"{metric}: aggregate {key} disagrees")
            reported = entry.get("mean")
            if (mean is None and reported is not None) or (
                mean is not None
                and (not _score(reported) or abs(mean - reported) > 1e-8)
            ):
                errors.append(f"{metric}: aggregate mean disagrees")
        if dataset and len(values) != dataset["coverage"].get(metric, 0):
            errors.append(
                f"{metric}: evaluated coverage differs from usable source cases"
            )
    manifest_path = path.parent / "input_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            hashes = {
                item.get("sha256")
                for item in manifest.get("datasets", [])
                if isinstance(item, dict)
            }
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"invalid input manifest: {exc}")
            hashes = set()
        if dataset and dataset["sha256"] not in hashes:
            errors.append("input manifest does not bind the audited dataset SHA256")
    else:
        errors.append("input_manifest.json missing; dataset identity is unverified")
    preview_path = path.parent / "edit_requests_preview.json"
    if dataset and preview_path.is_file():
        try:
            previews = json.loads(preview_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"invalid request preview: {exc}")
            previews = None
        if not isinstance(previews, list) or len(previews) != dataset["count"]:
            errors.append("request preview does not cover the source cases")
        else:
            for index, (source, preview) in enumerate(
                zip(dataset["case_identities"], previews)
            ):
                if (
                    not isinstance(preview, dict)
                    or any(
                        preview.get(key) != source[key]
                        for key in ("subject", "target_new")
                    )
                    or source["prompt"] not in str(preview.get("prompt", ""))
                ):
                    errors.append(f"request {index}: source identity/order differs")
    else:
        warnings.append(
            "complete request preview unavailable; inspect per-case input identities"
        )
    config_path = path.parent / "resolved_config.yaml"
    if config_path.is_file():
        import yaml

        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(config, dict):
                raise ValueError("configuration must be an object")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"invalid saved configuration: {exc}")
            config = {}
        trainer = config.get("trainer") or {}
        params = trainer.get("method_args") or {}
        result["configuration"] = {
            "handler": trainer.get("handler"),
            "method_args": params,
        }
        if params.get("mom2_adjustment") is False:
            warnings.append(
                "covariance adjustment disabled; label the actual algorithm variant"
            )
        if params.get("retrieval_backend") == "token_overlap":
            warnings.append(
                "lexical IKE retrieval; not the semantic demonstration protocol"
            )
        if "serac" in method.lower() and not params.get("archive"):
            warnings.append(
                "no trained SERAC archive; inspect and label the historical implementation"
            )
        if "alphaedit" in method.lower() and params.get("mom2_n_samples", 0) <= 200:
            warnings.append(
                "reduced AlphaEdit statistics; report corpus, sample budget and projection rank"
            )
    else:
        errors.append("resolved_config.yaml missing")
    result.update(
        executed_cases=executed, requested_cases=expected_count, metrics=scores
    )
    return result


def main(argv=None) -> int:
    """Write an audit report; strict mode fails on integrity errors, not missing optional probes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1301)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--run", action="append", default=[], metavar="METHOD=PATH")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    dataset = audit_dataset(args.dataset, args.expected_count, args.expected_sha256)
    runs = []
    for item in args.run:
        if "=" not in item:
            parser.error("--run must be METHOD=PATH")
        method, path = item.split("=", 1)
        runs.append(audit_run(method, Path(path), args.expected_count, dataset))
    report = {
        "schema_version": "1.0",
        "dataset": {k: v for k, v in dataset.items() if k != "case_identities"},
        "runs": runs,
    }
    report["valid"] = not dataset["errors"] and not any(run["errors"] for run in runs)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 2 if args.strict and not report["valid"] else 0
