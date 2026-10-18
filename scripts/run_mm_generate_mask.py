"""Generate MMUnlearner vision, language, and combined SFR-on masks.

This launcher mirrors the official MMUnlearner preserve sets for both
MLLMU-Bench and CLEAR while emitting checkpoint-bound artifacts that the
Know-Surgery trainer validates before the first optimizer step.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(repo_root))
    parser.add_argument("--benchmark", choices=["mllmu", "clear"], required=True)
    parser.add_argument(
        "--model-family",
        choices=["llava", "qwen2vl", "qwen2_5_vl"],
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype", choices=["float16", "bfloat16", "float32"], default=None
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--forget-ratio", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mask-max-samples", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--vision-modules", nargs="+", default=None)
    parser.add_argument("--language-modules", nargs="+", default=None)
    parser.add_argument(
        "--parameter-name-tokens",
        nargs="+",
        default=["proj", "fc", "linear", "mlp", "qkv"],
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-path", required=True)
    return parser.parse_args()


def _build_data_config(args, data_dir: Path):
    common = {
        "benchmark": args.benchmark,
        "data_dir": str(data_dir),
        "forget_split_ratio": args.forget_ratio,
        "batch_size": args.batch_size,
        "mask_batch_size": args.batch_size,
        "mask_ans_only": args.benchmark == "mllmu",
        "mask_max_samples": (
            args.mask_max_samples if args.mask_max_samples > 0 else None
        ),
    }
    if args.benchmark == "clear":
        common.update(
            {
                "full_dir": str(data_dir / "full"),
                "full_tofu_dir": str(data_dir / "full+tofu"),
            }
        )
    else:
        common.update(
            {
                "full_set_parquet": str(
                    data_dir / "Full_Set" / "train-00000-of-00001.parquet"
                ),
                "retain_set_parquet": str(
                    data_dir / "Retain_Set" / "train-00000-of-00001.parquet"
                ),
            }
        )
    return OmegaConf.create(common)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "src").is_dir():
        raise FileNotFoundError(f"Invalid repository root: {repo_root}")
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root / "src"))

    from data.clear_dataset import get_clear_mask_data
    from data.multimodal import get_mm_mask_data
    from model.multimodal import get_mm_model
    from trainer.unlearn.sfron import (
        UPSTREAM_COMMIT,
        generate_saliency_mask,
        merge_mask_artifacts,
        model_identity,
        resolve_mm_module_selectors,
    )

    print(
        json.dumps(
            {
                "launcher_cuda_available": torch.cuda.is_available(),
                "launcher_cuda_device_count": torch.cuda.device_count(),
            },
            ensure_ascii=False,
        )
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in run_mm_generate_mask.py launcher")

    data_dir = Path(
        args.data_dir
        or repo_root
        / "data"
        / ("CLEAR" if args.benchmark == "clear" else "MLLMU-Bench")
    ).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{args.benchmark} data directory not found: {data_dir}"
        )

    dtype = args.torch_dtype or (
        "float16" if args.model_family == "llava" else "bfloat16"
    )
    processor_path = args.processor_path or args.model_path
    model_cfg = OmegaConf.create(
        {
            "model_family": args.model_family,
            "model_path": args.model_path,
            "processor_path": processor_path,
            "torch_dtype": dtype,
            "attn_implementation": args.attn_implementation,
            "local_files_only": not args.allow_download,
            "lora": {"enabled": False},
        }
    )
    data_cfg = _build_data_config(args, data_dir)

    model, processor = get_mm_model(model_cfg)
    model = model.to(args.device)
    default_vision, default_language = resolve_mm_module_selectors(
        model, args.model_family
    )
    vision_modules = args.vision_modules or default_vision
    language_modules = args.language_modules or default_language

    loaders = (
        get_clear_mask_data(data_cfg, processor)
        if args.benchmark == "clear"
        else get_mm_mask_data(data_cfg, processor)
    )
    required_loaders = {"forget", "vision_preserve", "language_preserve"}
    missing_loaders = required_loaders - loaders.keys()
    if missing_loaders:
        raise RuntimeError(
            f"Mask data adapter omitted loaders: {sorted(missing_loaders)}"
        )

    max_batches = args.max_batches if args.max_batches > 0 else None
    save_path = Path(args.save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    identity = model_identity(
        model,
        model_family=args.model_family,
        checkpoint_id=args.model_path,
    )
    common_metadata = {
        **identity,
        "benchmark": args.benchmark,
        "forget_ratio": args.forget_ratio,
        "upstream_repository": "https://github.com/Z1zs/MMUnlearner",
        "upstream_commit": UPSTREAM_COMMIT,
    }

    vision_artifact = generate_saliency_mask(
        model=model,
        forget_loader=loaders["forget"],
        preserve_loader=loaders["vision_preserve"],
        modules=vision_modules,
        threshold=args.threshold,
        save_path=save_path / "vision",
        max_batches=max_batches,
        parameter_name_tokens=args.parameter_name_tokens,
        metadata={**common_metadata, "mask_scope": "vision"},
        reuse_cache=not args.no_cache,
        return_artifact=True,
    )
    language_artifact = generate_saliency_mask(
        model=model,
        forget_loader=loaders["forget"],
        preserve_loader=loaders["language_preserve"],
        modules=language_modules,
        threshold=args.threshold,
        save_path=save_path / "language",
        max_batches=max_batches,
        parameter_name_tokens=args.parameter_name_tokens,
        metadata={**common_metadata, "mask_scope": "language"},
        reuse_cache=not args.no_cache,
        return_artifact=True,
    )
    both_metadata = {
        **common_metadata,
        "mask_scope": "both",
        "threshold": args.threshold,
        "max_batches": max_batches,
        "module_selectors": [*vision_modules, *language_modules],
        "parameter_name_tokens": list(args.parameter_name_tokens),
    }
    both_artifact = merge_mask_artifacts(
        vision_artifact, language_artifact, metadata=both_metadata
    )

    artifacts = {
        "vision": vision_artifact,
        "language": language_artifact,
        "both": both_artifact,
    }
    for scope, artifact in artifacts.items():
        torch.save(artifact, save_path / f"{args.benchmark}_{scope}_mask.pt")

    manifest = {
        "status": "completed",
        "benchmark": args.benchmark,
        "model_family": args.model_family,
        "checkpoint_id": identity["checkpoint_id"],
        "forget_ratio": args.forget_ratio,
        "threshold": args.threshold,
        "max_batches": max_batches,
        "vision_modules": vision_modules,
        "language_modules": language_modules,
        "artifacts": {
            scope: {
                "path": str(save_path / f"{args.benchmark}_{scope}_mask.pt"),
                "parameter_count": artifact["metadata"]["selected_parameter_count"],
                "sparsity": artifact["metadata"]["sparsity"],
            }
            for scope, artifact in artifacts.items()
        },
        "upstream_commit": UPSTREAM_COMMIT,
    }
    (save_path / "mask_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
