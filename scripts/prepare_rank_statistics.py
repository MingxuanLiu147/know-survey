#!/usr/bin/env python3
"""Prepare one auditable, resumable rank-editor C0 through canonical layer_stats.

This is a synchronous worker. It does not submit jobs, poll other processes,
refresh credentials or start a monitoring service.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trainer.edit.checkpoint_identity import stream_model_persistent_state_sha256
from trainer.edit.utils.layer_stats import StatisticsInterrupted, layer_stats
from trainer.edit.utils.statistics_context import statistics_execution_identity


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--data-path", required=True)
    result.add_argument("--stats-dir", required=True)
    result.add_argument("--layer", required=True, type=int)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--profile-jsonl", required=True, type=Path)
    result.add_argument("--sample-size", type=int, default=100_000)
    result.add_argument("--sample-seed", type=int, default=1)
    result.add_argument("--dataset-config", default="20200501.en")
    result.add_argument("--max-length", type=int, default=32768)
    result.add_argument("--batch-tokens", type=int, default=32768)
    result.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--checkpoint-every-batches", type=int, default=100)
    result.add_argument("--stop-after-batches", type=int)
    result.add_argument("--force-recompute", action="store_true")
    return result


def main():
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    began = time.perf_counter()
    args.profile_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "loading_model",
        "arguments": vars(args).copy(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "monitoring_watchers": 0,
    }
    manifest["arguments"] = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in manifest["arguments"].items()
    }
    write_json(args.manifest, manifest)
    model = None
    original_state = None
    original_flags = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=getattr(torch, args.dtype),
            attn_implementation="eager",
            device_map={"": "cpu"},
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model.eval()
        original_state = stream_model_persistent_state_sha256(model)
        original_flags = [
            (p.device, p.dtype, p.requires_grad) for p in model.parameters()
        ]
        manifest.update(
            source_model=original_state,
            model_load_and_identity_seconds=time.perf_counter() - began,
            execution=statistics_execution_identity(
                model,
                attention_policy="qwen2_eager_query_chunked",
                query_chunk_size=256,
                placement_policy="qwen2_prefix",
                execution_backend=torch.device(args.device).type,
            ),
            status="collecting",
        )
        if args.device.startswith("cuda"):
            torch.cuda.set_device(args.device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(args.device)
            properties = torch.cuda.get_device_properties(args.device)
            manifest["gpu"] = {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
            }
        write_json(args.manifest, manifest)
        last_event_time = time.perf_counter()
        last_count = 0

        def progress(event):
            nonlocal last_event_time, last_count
            now = time.perf_counter()
            elapsed = now - last_event_time
            enriched = {
                **event,
                "elapsed_seconds": now - began,
                "interval_seconds": elapsed,
                "interval_tokens_per_second": None
                if event.get("cache_hit")
                else (
                    event.get("batch_tokens", 0) / event["batch_compute_seconds"]
                    if event.get("batch_compute_seconds")
                    else None
                ),
            }
            if args.device.startswith("cuda"):
                enriched.update(
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(
                        args.device
                    ),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(
                        args.device
                    ),
                )
            with args.profile_jsonl.open("a") as stream:
                stream.write(json.dumps(enriched) + "\n")
            last_event_time, last_count = now, event["count"]
            if event["status"] == "complete" or event["completed_batches"] % 100 == 0:
                print(json.dumps(enriched), flush=True)

        result = layer_stats(
            model,
            tokenizer,
            f"model.layers.{args.layer}.mlp.down_proj",
            args.stats_dir,
            ds_name="wikipedia",
            dataset_config=args.dataset_config,
            data_path=args.data_path,
            sample_size=args.sample_size,
            sample_policy="upstream_random",
            sample_seed=args.sample_seed,
            max_length=args.max_length,
            batch_tokens=args.batch_tokens,
            precision="float32",
            attention_policy="qwen2_eager_query_chunked",
            query_chunk_size=256,
            placement_policy="qwen2_prefix",
            execution_device=args.device,
            execution_backend=torch.device(args.device).type,
            checkpoint_every_batches=args.checkpoint_every_batches,
            stop_after_batches=args.stop_after_batches,
            force_recompute=args.force_recompute,
            progress_callback=progress,
        )
        manifest.update(
            status="complete",
            count=int(result["count"]),
            shape=list(result["mom2"].shape),
            result_device=str(result["mom2"].device),
            finite=bool(torch.isfinite(result["mom2"]).all()),
        )
    except StatisticsInterrupted as exc:
        manifest.update(status="interrupted_resumable", error=str(exc))
    except BaseException as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if model is not None and original_state is not None:
            restored = stream_model_persistent_state_sha256(model)
            manifest.update(
                restored_model=restored,
                persistent_state_restored=restored == original_state,
                parameter_placement_dtype_grad_restored=original_flags
                == [(p.device, p.dtype, p.requires_grad) for p in model.parameters()],
                attention_backend_restored=model.config._attn_implementation == "eager",
                meta_tensor_count=sum(
                    p.device.type == "meta" for p in model.parameters()
                ),
            )
        if args.device.startswith("cuda") and torch.cuda.is_initialized():
            manifest.update(
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
            )
        manifest.update(
            wall_seconds=time.perf_counter() - began,
            cpu_rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            ended_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        restoration_checks = (
            "persistent_state_restored",
            "parameter_placement_dtype_grad_restored",
            "attention_backend_restored",
        )
        if (
            model is not None
            and original_state is not None
            and not all(manifest.get(k) is True for k in restoration_checks)
        ):
            manifest.update(
                status="failed",
                error_type="RestorationFailure",
                error="Model restoration verification failed",
            )
        write_json(args.manifest, manifest)
        print(
            json.dumps({"status": manifest["status"], "manifest": str(args.manifest)}),
            flush=True,
        )
    if manifest["status"] == "failed":
        raise RuntimeError(manifest.get("error", "Statistics preparation failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
