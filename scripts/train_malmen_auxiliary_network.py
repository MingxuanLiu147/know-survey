#!/usr/bin/env python3
"""Pretrain a native compact MALMEN auxiliary network.

The resulting checkpoint is intentionally not compatible with EasyEdit's
official MALMEN archive schema.  A small GPT-2 example is suitable for local
connectivity validation:

  PYTHONPATH=src python scripts/train_malmen_auxiliary_network.py \
    --model openai-community/gpt2 \
    --data data/edit/zsre/zsre_mend_train.json \
    --output saves/edit/malmen/gpt2_zsre.pt \
    --layers 5 --rank 16 --max-edits 4 --epochs 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_mend_edit_network import _load_requests  # noqa: E402
from trainer.edit.malmen import MALMENEditor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a checkpoint-gated compact native MALMEN network."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--data", required=True, help="JSON edit-training samples")
    parser.add_argument("--output", required=True, help="Native checkpoint path")
    parser.add_argument("--max-edits", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--edit-lr", type=float, default=1e-4)
    parser.add_argument("--meta-lr", type=float, default=1e-4)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, nargs="+", default=[5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output = Path(args.output)
    training_args = TrainingArguments(
        output_dir=str(output.parent / "malmen_train_scratch"),
        per_device_train_batch_size=1,
        report_to=[],
        disable_tqdm=True,
    )
    editor = MALMENEditor(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        layers=args.layers,
        edit_lr=args.edit_lr,
        meta_lr=args.meta_lr,
        rank=args.rank,
        n_blocks=args.n_blocks,
        batch_size=args.batch_size,
        n_edits=args.max_edits,
        lambda_reg=args.lambda_reg,
        require_pretrained=True,
    )
    requests = _load_requests(Path(args.data), args.max_edits)
    history = editor.train_auxiliary_network(
        requests,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.meta_lr,
    )
    editor.save_auxiliary_network(output)
    data_path = Path(args.data)
    checkpoint_payload = output.read_bytes()
    checkpoint_metadata = torch.load(output, map_location="cpu", weights_only=True)
    data_payload = data_path.read_bytes()
    evidence = {
        "schema_version": 1,
        "status": "success",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": "compact_native_malmen",
        "claim_boundary": (
            "Checkpoint-training smoke; not an EasyEdit-compatible or official "
            "MALMEN benchmark reproduction."
        ),
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "model": {
            "path": args.model,
            "model_type": getattr(model.config, "model_type", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        },
        "input": {
            "path": str(data_path),
            "sha256": hashlib.sha256(data_payload).hexdigest(),
            "size_bytes": len(data_payload),
            "loaded_requests": len(requests),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "rank": args.rank,
            "n_blocks": args.n_blocks,
            "edit_lr": args.edit_lr,
            "meta_lr": args.meta_lr,
            "lambda_reg": args.lambda_reg,
            "layers": args.layers,
            "history": history,
        },
        "checkpoint": {
            "path": str(output),
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "size_bytes": len(checkpoint_payload),
            "base_model_identity": checkpoint_metadata["base_model_identity"],
        },
    }
    evidence_path = output.with_suffix(output.suffix + ".training.json")
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote compact native MALMEN auxiliary checkpoint to {output}")
    print(f"Wrote training evidence to {evidence_path}")


if __name__ == "__main__":
    main()
