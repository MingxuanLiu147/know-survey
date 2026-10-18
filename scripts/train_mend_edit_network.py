#!/usr/bin/env python3
"""Compact MEND/InstructEdit edit-network pretraining script.

Example:
  PYTHONPATH=src python scripts/train_mend_edit_network.py \\
    --model Qwen/Qwen2.5-0.5B \\
    --data data/edit/zsre/zsre_mend_train.json \\
    --output saves/edit/mend_networks/qwen25_05b.pt \\
    --max-edits 8 --epochs 2
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
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trainer.edit.base import EditRequest  # noqa: E402
from trainer.edit.instructedit import InstructEditEditor  # noqa: E402
from trainer.edit.mend import MENDEditor  # noqa: E402


def _text(value) -> str:
    if isinstance(value, dict):
        value = value.get("str", value.get("text", value.get("answer", "")))
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _load_requests(path: Path, max_edits: int) -> list[EditRequest]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    requests = []
    for item in raw:
        rewrite = item.get("requested_rewrite")
        if isinstance(rewrite, dict):
            subject = _text(rewrite.get("subject"))
            prompt_template = _text(rewrite.get("prompt"))
            try:
                prompt = prompt_template.format(subject)
            except (IndexError, KeyError, ValueError):
                prompt = prompt_template
            target_new = _text(rewrite.get("target_new"))
            target_old = _text(rewrite.get("target_true", rewrite.get("target_old")))
            rephrases = item.get("paraphrase_prompts") or None
            neighborhoods = item.get("neighborhood_prompts") or []
            locality = None
            if neighborhoods and target_old:
                locality = {
                    "neighborhood": [
                        {"prompt": _text(value), "ground_truth": target_old}
                        for value in neighborhoods[:2]
                        if _text(value)
                    ]
                }
        else:
            prompt = _text(item.get("prompt") or item.get("src"))
            subject = _text(item.get("subject") or item.get("subject_entity"))
            rephrases = item.get("rephrase_prompts", item.get("rephrase"))
            if isinstance(rephrases, str):
                rephrases = [rephrases]
            answers = item.get("answers")
            answers_val = _text(answers)
            target_new = _text(
                item.get("target_new")
                or item.get("alt")
                or answers_val
                or item.get("output")
            )
            target_old = _text(
                item.get("target_old") or item.get("pred") or answers_val
            )
            locality = None
            if item.get("loc") and item.get("loc_ans"):
                locality = {
                    "relation_specificity": [
                        {
                            "prompt": _text(item["loc"]),
                            "ground_truth": _text(item["loc_ans"]),
                        }
                    ]
                }

        if not prompt or not target_new:
            continue
        requests.append(
            EditRequest(
                prompt=prompt,
                subject=subject or prompt,
                target_new=target_new,
                target_old=target_old or None,
                rephrase_prompts=rephrases,
                locality_inputs=locality,
            )
        )
        if len(requests) >= max_edits:
            break
    if not requests:
        raise ValueError(f"No edit requests loaded from {path}")
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a compact native MEND or InstructEdit edit network."
    )
    parser.add_argument("--editor", choices=("mend", "instructedit"), default="mend")
    parser.add_argument(
        "--task-type",
        choices=("counterfact", "zsre", "wikirecent", "convsent", "default"),
        default="default",
        help="Instruction task used by --editor instructedit",
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--data", required=True, help="JSON list of edit samples")
    parser.add_argument("--output", required=True, help="Checkpoint output path")
    parser.add_argument("--max-edits", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--edit-lr", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, nargs="+", default=[5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Dummy TrainingArguments-compatible kwargs via EditTrainer/FinetuneTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(Path(args.output).parent / "mend_train_scratch"),
        per_device_train_batch_size=1,
        report_to=[],
        disable_tqdm=True,
    )

    editor_class = MENDEditor if args.editor == "mend" else InstructEditEditor
    editor_kwargs = {}
    if args.editor == "instructedit":
        editor_kwargs["default_task_type"] = args.task_type
        editor_kwargs["instruction_index"] = 0
    editor = editor_class(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        layers=args.layers,
        edit_lr=args.edit_lr,
        n_hidden=args.n_hidden,
        rank=args.rank,
        require_pretrained=False,
        **editor_kwargs,
    )

    requests = _load_requests(Path(args.data), args.max_edits)
    train_kwargs = {
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    if args.editor == "instructedit":
        train_kwargs["task_type"] = args.task_type
    history = editor.train_edit_network(requests, **train_kwargs)
    output = Path(args.output)
    editor.save_edit_network(output)
    data_path = Path(args.data)
    checkpoint_payload = output.read_bytes()
    checkpoint_metadata = torch.load(output, map_location="cpu", weights_only=True)
    data_payload = data_path.read_bytes()
    evidence = {
        "schema_version": 1,
        "status": "success",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation": f"compact_native_{args.editor}",
        "claim_boundary": (
            "Checkpoint-training smoke; not an EasyEdit-compatible or official "
            f"{args.editor} benchmark reproduction."
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
            "editor": args.editor,
            "task_type": args.task_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "rank": args.rank,
            "n_hidden": args.n_hidden,
            "edit_lr": args.edit_lr,
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
    print(f"Wrote compact {args.editor} edit network to {args.output}")
    print(f"Wrote training evidence to {evidence_path}")


if __name__ == "__main__":
    main()
