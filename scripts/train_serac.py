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
import math
import shlex
import sys
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trainer.edit.serac import SERACEditor, _vocabulary_identity  # noqa: E402
from prepare_memory_editor_data import validate_disjoint  # noqa: E402
from know_surgery.edit_data import file_sha256  # noqa: E402


def validate_model_assets(args):
    """Check local model metadata and tokenizers without loading large weights."""
    tokenizers, configs = {}, {}
    for name, source in (
        ("base", args.model),
        ("scope", args.scope_model),
        ("counterfactual", args.counterfactual_model),
    ):
        # Preflight must never download resources or fall back to a remote ID.
        configs[name] = AutoConfig.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
        tokenizers[name] = AutoTokenizer.from_pretrained(
            source, local_files_only=True, trust_remote_code=False
        )
    for name in ("base", "counterfactual"):
        tokenizer = tokenizers[name]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError(f"SERAC {name} tokenizer needs pad_token or eos_token")
        if configs[name].vocab_size <= max(tokenizer.get_vocab().values()):
            raise ValueError(f"SERAC {name} tokenizer IDs exceed its model vocabulary")
    if _vocabulary_identity(tokenizers["base"]) != _vocabulary_identity(
        tokenizers["counterfactual"]
    ):
        raise ValueError(
            "SERAC token-level benchmark requires matching base/counterfactual tokenizers; "
            "select a counterfactual model from a compatible tokenizer family"
        )
    scope = tokenizers["scope"]
    if scope.pad_token_id is None or scope.sep_token_id is None:
        raise ValueError("SERAC scope tokenizer needs pad_token and sep_token")
    if configs["scope"].hidden_size % args.distance_heads:
        raise ValueError("SERAC scope hidden size must be divisible by distance_heads")
    return {
        "status": "validated_metadata_and_tokenizers",
        "weights_loaded": False,
        "matching_base_counterfactual_tokenizers": True,
        "model_types": {name: config.model_type for name, config in configs.items()},
        "model_vocab_sizes": {
            name: config.vocab_size for name, config in configs.items()
        },
        "soft_weighting_supported": configs["base"].vocab_size
        == configs["counterfactual"].vocab_size,
        "vocabulary_sha256": _vocabulary_identity(tokenizers["base"]),
    }


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
        "--check-model-assets",
        action="store_true",
        help="Also validate offline config/tokenizer compatibility without loading weights",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; never overwrite an archive",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--base-device",
        default="cpu",
        help="Device for the frozen identity-only base (never used by auxiliary gradients)",
    )
    parser.add_argument(
        "--base-dtype",
        choices=("bfloat16", "float32", "float16"),
        default="bfloat16",
        help="Must match the benchmark base precision for archive identity validation",
    )
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--scope-threshold", type=float, default=0.5)
    parser.add_argument("--distance-heads", type=int, default=1)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume a full auxiliary training checkpoint into this new output directory",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="Absolute optimizer-step limit for a bounded recovery check; no trained archive before completion",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Use existing model assets only (always enabled; prepare downloads separately)",
    )
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.lr <= 0
        or not math.isfinite(args.lr)
    ):
        parser.error("--epochs, --batch-size and --lr must be positive and finite")
    SERACEditor._validate_config(
        args.scope_threshold,
        args.distance_heads,
        args.distance_scale,
        args.max_length,
        False,
    )
    if args.checkpoint_every < 1 or (
        args.stop_after_steps is not None and args.stop_after_steps < 1
    ):
        parser.error("--checkpoint-every and --stop-after-steps must be positive")
    if (
        args.resume_from_checkpoint is not None
        and not args.resume_from_checkpoint.is_file()
    ):
        parser.error("--resume-from-checkpoint must name an existing file")
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
    configuration = {
        "base_model": args.model,
        "scope_model": args.scope_model,
        "counterfactual_model": args.counterfactual_model,
        "base_device": args.base_device,
        "base_dtype": args.base_dtype,
        "auxiliary_device": args.device,
        "auxiliary_dtype": "float32",
        "scope_threshold": args.scope_threshold,
        "distance_heads": args.distance_heads,
        "distance_scale": args.distance_scale,
        "soft_weighting": False,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "training_records": len(rows),
        "optimizer_steps": args.epochs
        * ((len(rows) + args.batch_size - 1) // args.batch_size),
        "checkpoint_every": args.checkpoint_every,
        "resume_from_checkpoint": str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint
        else None,
        "stop_after_steps": args.stop_after_steps,
    }
    assets = validate_model_assets(args) if args.check_model_assets else None
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "split_check": split_check,
                    "training_records": len(rows),
                    "validation_records": len(validation or []),
                    "resolved_configuration": configuration,
                    "model_assets": assets,
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
        "resolved_configuration": configuration,
        "model_assets": assets,
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
    try:
        # Catch incompatibility before allocating either large model.
        if assets is None:
            assets = validate_model_assets(args)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=args.local_files_only, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=getattr(torch, args.base_dtype),
            local_files_only=args.local_files_only,
            trust_remote_code=False,
        )
        editor = SERACEditor(
            model=model,
            tokenizer=tokenizer,
            args=TrainingArguments(
                output_dir=str(args.output / "trainer"),
                report_to=[],
                bf16=False,
                seed=args.seed,
                use_cpu=True,
            ),
            scope_model_name_or_path=args.scope_model,
            counterfactual_model_name_or_path=args.counterfactual_model,
            aux_device=args.device,
            max_length=args.max_length,
            scope_threshold=args.scope_threshold,
            distance_heads=args.distance_heads,
            distance_scale=args.distance_scale,
            local_files_only=args.local_files_only,
        )
        # HF Trainer otherwise moves the frozen 7B base to the accelerator even
        # though train_serac only uses the two independent auxiliary networks.
        model.to(args.base_device).requires_grad_(False).eval()
        history = editor.train_serac(
            rows,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            source=args.data.name,
            checkpoint_path=args.output / "training_checkpoint.pt",
            checkpoint_every=args.checkpoint_every,
            resume_from_checkpoint=args.resume_from_checkpoint,
            stop_after_steps=args.stop_after_steps,
        )
        if editor.training_metadata["status"] != "trained":
            summary = {
                "status": "incomplete",
                "variant": editor.variant,
                "training_progress": editor.training_metadata,
                "input_manifest": manifest,
                "model_assets": assets,
            }
            (args.output / "training_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
            print(
                f"SERAC auxiliary checkpoint saved at step {editor.training_metadata['optimizer_steps']}; training incomplete"
            )
            return
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
            "archive_sha256": file_sha256(archive),
            "input_manifest": manifest,
            "auxiliary_validation": validation_metrics,
            "model_assets": assets,
            "training_checkpoint": "training_checkpoint.pt",
            "resume_start_optimizer_step": editor.training_metadata[
                "resume_start_optimizer_step"
            ],
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
