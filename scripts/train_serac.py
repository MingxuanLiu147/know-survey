#!/usr/bin/env python3
"""Train SERAC's scope encoder and counterfactual LM on a disjoint auxiliary split.

Input is a JSON list of explicit training records: prompt, target_new,
scope_training_prompts and scope_training_negatives. Evaluation probe fields
are rejected by the editor; use a separate held-out data file for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trainer.edit.serac import SERACEditor  # noqa: E402
from prepare_memory_editor_data import validate_disjoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Frozen base causal LM")
    parser.add_argument("--scope-model", required=True, help="Pretrained text encoder")
    parser.add_argument(
        "--counterfactual-model",
        required=True,
        help="Trainable causal LM with the base tokenizer vocabulary",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--heldout",
        type=Path,
        required=True,
        help="Reserved edit/evaluation JSON; never used for gradients",
    )
    parser.add_argument(
        "--validation", type=Path, help="Independent auxiliary validation JSON"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check explicit data and split separation without loading models",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; never overwrite an archive",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    rows = json.loads(args.data.read_text())
    if not isinstance(rows, list) or not rows:
        parser.error(
            "--data must contain a non-empty list of explicit training records"
        )
    rows = rows[: args.max_records]
    rows = [dict(row, **SERACEditor._record(row)) for row in rows]
    heldout = json.loads(args.heldout.read_text(encoding="utf-8"))
    if not isinstance(heldout, list) or not heldout:
        parser.error("--heldout must contain a non-empty JSON list")
    split_check = validate_disjoint(rows, heldout)
    validation = None
    if args.validation:
        validation = json.loads(args.validation.read_text(encoding="utf-8"))
        if not isinstance(validation, list) or not validation:
            parser.error("--validation must contain a non-empty auxiliary JSON list")
        validation = [dict(row, **SERACEditor._record(row)) for row in validation]
        validate_disjoint(rows, validation)
        validate_disjoint(validation, heldout)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "split_check": split_check,
                    "training_records": len(rows),
                    "validation_records": len(validation or []),
                },
                indent=2,
            )
        )
        return
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    manifest = {
        "source": str(args.data.resolve()),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "used_samples": len(rows),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "scope": "auxiliary training, not a benchmark result",
        "heldout_sha256": hashlib.sha256(args.heldout.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest()
        if args.validation
        else None,
        "split_check": split_check,
    }
    (args.output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output / "source_hashes.json").write_text(
        json.dumps(
            {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("src/trainer/edit/serac.py", "scripts/train_serac.py")
            },
            indent=2,
        )
        + "\n"
    )
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=args.local_files_only
    ).to(args.device)
    editor = SERACEditor(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=str(args.output / "trainer"),
            report_to=[],
            bf16=False,
            seed=args.seed,
        ),
        scope_model_name_or_path=args.scope_model,
        counterfactual_model_name_or_path=args.counterfactual_model,
        aux_device=args.device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )
    try:
        history = editor.train_serac(
            rows,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            source=args.data.name,
        )
        validation_metrics = (
            editor.validate_auxiliary(validation) if validation else None
        )
        archive = args.output / "serac_auxiliary.pt"
        editor.save_archive(archive)
        summary = {
            "status": "trained",
            "variant": editor.variant,
            "history": history,
            "archive": archive.name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "input_manifest": manifest,
            "auxiliary_validation": validation_metrics,
        }
        (args.output / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(f"Trained SERAC auxiliaries: {archive}")
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2)
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
