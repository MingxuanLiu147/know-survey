#!/usr/bin/env python3
"""Train a factor-MEND or legacy compact MEND/InstructEdit edit network.

The factor path requires explicit matrices, independent validation and a full
training checkpoint. Use --preview before reviewing its resource/configuration
budget. A calibration run cannot export a formal editor archive.

Legacy compact example:
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
import subprocess
import sys
import time
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
from trainer.edit.mend_official import MENDFactorEditor  # noqa: E402


class _FactorRunProfile:
    """Record bounded training events without changing optimizer/RNG state."""

    def __init__(self, path: Path, cuda_devices: list[int], batch_size: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cuda_devices = cuda_devices
        self.batch_size = batch_size
        self.started = time.perf_counter()
        self.step_started = None
        self.records = []
        self.phase = "training"
        self.phase_peaks = {}
        self.overall_peaks = {}

    def __call__(self, event: str, details: dict):
        for device in self.cuda_devices:
            torch.cuda.synchronize(device)
        now = time.perf_counter()
        row = {
            "event": event,
            "elapsed_seconds": now - self.started,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            **details,
        }
        if event == "step_started":
            self.step_started = now
        elif event == "optimizer_completed" and self.step_started is not None:
            elapsed = now - self.step_started
            row.update(
                optimizer_step_seconds=elapsed,
                samples_per_second=self.batch_size / elapsed,
            )
        if event != "example_completed":
            row["per_device"] = []
            for device in self.cuda_devices:
                free, total = torch.cuda.mem_get_info(device)
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                previous = self.overall_peaks.get(device, {})
                self.overall_peaks[device] = {
                    "max_memory_allocated": max(
                        allocated, previous.get("max_memory_allocated", 0)
                    ),
                    "max_memory_reserved": max(
                        reserved, previous.get("max_memory_reserved", 0)
                    ),
                }
                row["per_device"].append(
                    {
                        "device": device,
                        "name": torch.cuda.get_device_name(device),
                        "total_bytes": total,
                        "whole_device_used_bytes": total - free,
                        "memory_allocated": torch.cuda.memory_allocated(device),
                        "memory_reserved": torch.cuda.memory_reserved(device),
                        **self.overall_peaks[device],
                        "phase_max_memory_allocated": allocated,
                        "phase_max_memory_reserved": reserved,
                    }
                )
            if self.cuda_devices:
                try:
                    process = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=gpu_uuid,pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    row["compute_process_memory"] = process.stdout.strip().splitlines()
                except (OSError, subprocess.SubprocessError) as error:
                    row["process_memory_query_error"] = str(error)
        if event in {"validation_started", "validation_completed"}:
            self.phase_peaks[self.phase] = row["per_device"]
        if event == "validation_started":
            for device in self.cuda_devices:
                torch.cuda.reset_peak_memory_stats(device)
            self.phase = "validation"
        self.records.append(row)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"runtime_event": event, **details}), flush=True)


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
            locality = item.get("locality_inputs", item.get("locality"))
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
        description="Train factor MEND or a legacy compact MEND/InstructEdit network."
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--n-hidden", type=int, default=None)
    parser.add_argument("--edit-lr", type=float, default=1e-4)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--implementation", choices=("compact", "factor"), default="compact"
    )
    parser.add_argument(
        "--training-purpose", choices=("calibration", "training"), default="training"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), default="float32"
    )
    parser.add_argument(
        "--validation-data", help="Independent auxiliary validation JSON"
    )
    parser.add_argument("--max-validation-edits", type=int, default=128)
    parser.add_argument(
        "--exclude-data",
        action="append",
        default=[],
        help="Held-out test JSON; reject edit/rephrase overlap",
    )
    parser.add_argument(
        "--inner-params", nargs="+", help="Exact factor-MEND weight names"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Auxiliary optimizer steps, not examples",
    )
    parser.add_argument("--network-lr", type=float, default=1e-6)
    parser.add_argument("--lr-lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--cedit", type=float, default=0.1)
    parser.add_argument("--cloc", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--model-device-map",
        help="Reviewed JSON mapping module prefixes to explicit devices",
    )
    parser.add_argument(
        "--checkpoint-path", help="Full optimizer/RNG/progress training checkpoint"
    )
    parser.add_argument("--resume-from")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        help="Bounded recovery check; incomplete runs cannot export an editor",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Validate data and print exact settings without loading weights",
    )
    args = parser.parse_args()
    factor = args.implementation == "factor"
    if args.rank is None:
        args.rank = 1920 if factor else 64
    if args.n_hidden is None:
        args.n_hidden = 1 if factor else 128
    if args.layers is None:
        args.layers = [25, 26, 27] if factor else [5]
    if args.batch_size is None:
        args.batch_size = 10 if factor else 2

    if args.implementation == "factor":
        return _train_factor(args)
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


def _train_factor(args):
    """Train the original factor mechanism; keep the compact CLI backward compatible."""
    if args.dtype == "float16":
        raise ValueError(
            "Factor MEND supports FP32 or BF16 base weights; FP16 requires an unimplemented gradient-scaling protocol"
        )
    if args.editor != "mend":
        raise ValueError("Factor training currently supports MEND only")
    if not args.validation_data or not args.inner_params or not args.checkpoint_path:
        raise ValueError(
            "Factor MEND requires --validation-data, --inner-params, and --checkpoint-path"
        )
    if min(args.max_edits, args.max_validation_edits) < 1:
        raise ValueError("Factor MEND split limits must be positive")
    train = _load_requests(Path(args.data), args.max_edits)
    validation = _load_requests(Path(args.validation_data), args.max_validation_edits)
    # Check source dictionaries before EditRequest conversion, which deliberately
    # does not carry source IDs. Preserve every nested probe in the leakage check.
    from scripts.prepare_memory_editor_data import validate_disjoint

    def source_rows(path):
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("data")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("MEND auxiliary sources must be lists of source records")
        return [
            {**row, "subject": row.get("subject", row.get("subject_entity", ""))}
            if "requested_rewrite" not in row
            else row
            for row in rows
        ]

    source_train, source_validation = (
        source_rows(args.data),
        source_rows(args.validation_data),
    )
    disjoint_checks = {
        "train_vs_validation": validate_disjoint(source_train, source_validation)
    }
    for index, path in enumerate(args.exclude_data):
        excluded = source_rows(path)
        disjoint_checks[f"train_and_validation_vs_excluded_{index}"] = (
            validate_disjoint(source_train + source_validation, excluded)
        )
    for request in [*train, *validation]:
        if not request.rephrase_prompts or not request.locality_inputs:
            raise ValueError(
                "Every factor-MEND auxiliary example needs rephrase and locality probes"
            )
    data = {}
    for label, path in [
        ("train", args.data),
        ("validation", args.validation_data),
        *[
            (f"excluded_{index}", value)
            for index, value in enumerate(args.exclude_data)
        ],
    ]:
        path = Path(path)
        data[label] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    mapping = None
    if args.model_device_map:
        map_path = Path(args.model_device_map)
        mapping = json.loads(map_path.read_text())
        if (
            not isinstance(mapping, dict)
            or not mapping
            or any(
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, str))
                or value in ("disk", "auto")
                for key, value in mapping.items()
            )
        ):
            raise ValueError(
                "MEND device map must explicitly name devices without disk offload"
            )
        data["device_map"] = {
            "path": str(map_path),
            "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        }
    settings = vars(args).copy()
    preview = {
        "implementation": "factor_mend",
        "training_examples": len(train),
        "validation_examples": len(validation),
        "inputs": data,
        "disjoint_checks": disjoint_checks,
        "parameters": settings,
        "resolved_device_map": mapping,
        "status": "configured_requires_real_training_and_validation",
        "checkpoint_compatible_with": "know_surgery_factor_mend_v1",
        "claim_boundary": "No pretrained Qwen network is implied by a configuration preview.",
    }
    if args.preview:
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"device_map": mapping} if mapping is not None else {}
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        local_files_only=True,
        attn_implementation="eager",
        **load_kwargs,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not load_kwargs:
        model.to(device)
    from transformers import TrainingArguments

    editor = MENDFactorEditor(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=str(Path(args.output).parent / "factor_mend_scratch"),
            use_cpu=device.type == "cpu",
            report_to=[],
            disable_tqdm=True,
        ),
        layers=args.layers,
        inner_params=args.inner_params,
        rank=args.rank,
        n_hidden=args.n_hidden,
        edit_lr=args.edit_lr,
        cedit=args.cedit,
        cloc=args.cloc,
        max_length=args.max_length,
        require_pretrained=True,
    )
    cuda_devices = sorted(
        {p.device.index for p in model.parameters() if p.device.type == "cuda"}
    )
    for cuda_device in cuda_devices:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    profile = _FactorRunProfile(
        Path(args.output).parent / "runtime_profile.jsonl",
        cuda_devices,
        args.batch_size,
    )
    profile("training_started", {})
    started = time.perf_counter()
    try:
        history = editor.train_edit_network(
            train,
            validation_requests=validation,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.network_lr,
            lr_lr=args.lr_lr,
            grad_clip=args.grad_clip,
            seed=args.seed,
            checkpoint_path=args.checkpoint_path,
            resume_from=args.resume_from,
            save_every=args.save_every,
            stop_after_steps=args.stop_after_steps,
            data_provenance={"inputs": data, "disjoint_checks": disjoint_checks},
            purpose=args.training_purpose,
            event_callback=profile,
        )
    except Exception as error:
        profile(
            "training_failed", {"error": str(error), "error_type": type(error).__name__}
        )
        raise
    profile("training_completed", {})
    output = Path(args.output)
    exported = bool(history["completed"] and args.training_purpose == "training")
    if exported:
        editor.save_edit_network(output)
    elapsed = time.perf_counter() - started
    evidence = {
        **preview,
        "status": "trained_unqualified"
        if exported
        else (
            "calibration_completed"
            if history["completed"]
            else "training_checkpoint_only"
        ),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": history,
        "wall_time_seconds": elapsed,
        "runtime_profile": str(profile.path),
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "memory": {
            "max_memory_allocated": profile.overall_peaks[torch.cuda.current_device()][
                "max_memory_allocated"
            ]
            if torch.cuda.is_available()
            else None,
            "max_memory_reserved": profile.overall_peaks[torch.cuda.current_device()][
                "max_memory_reserved"
            ]
            if torch.cuda.is_available()
            else None,
            "per_device": [
                {
                    "device": index,
                    **profile.overall_peaks[index],
                }
                for index in cuda_devices
            ],
            "by_phase": profile.phase_peaks,
        },
        "checkpoint": {
            "training": args.checkpoint_path,
            "inference": str(output) if exported else None,
            "sha256": json.loads(
                output.with_name(output.name + ".metadata.json").read_text()
            )["checkpoint_sha256"]
            if exported
            else None,
        },
        "claim_boundary": "A trained auxiliary network is not a complete benchmark result or an accuracy qualification.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(output.suffix + ".training.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": evidence["status"], "checkpoint": evidence["checkpoint"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
