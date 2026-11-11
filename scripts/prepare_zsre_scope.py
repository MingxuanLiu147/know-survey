#!/usr/bin/env python3
"""Prepare independent WISE training negatives without modifying benchmark probes."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from know_surgery.edit_data import build_scope_sidecar, file_sha256, read_records  # noqa: E402


def main():
    """Save a separate, source-bound scope-training file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negatives-per-case", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Choose a new output path for the sidecar and manifest.")
    sidecar = build_scope_sidecar(
        read_records(args.dataset),
        read_records(args.training_data),
        seed=args.seed,
        negatives_per_case=args.negatives_per_case,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(sidecar, stream, ensure_ascii=False, indent=2)
    manifest = {
        "dataset_sha256": file_sha256(args.dataset),
        "training_sha256": file_sha256(args.training_data),
        "sidecar_sha256": file_sha256(args.output),
        "seed": args.seed,
        "cases": len(sidecar),
        "negatives_per_case": args.negatives_per_case,
    }
    with args.output.with_suffix(".manifest.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(manifest, stream, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
