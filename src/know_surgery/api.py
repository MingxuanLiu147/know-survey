"""Small synchronous Python interface to the existing surgery CLI."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4


def _arguments(config: Mapping[str, Any], evaluate: bool) -> list[str]:
    import yaml

    from know_surgery.resources import config_dir

    allowed = {
        "operation",
        "model",
        "dataset",
        "method",
        "experiment",
        "modality",
        "overrides",
        "output_dir",
    }
    if unknown := set(config) - allowed:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    operation = config.get("operation")
    if operation not in {"inject", "edit", "unlearning"}:
        raise ValueError("operation must be inject, edit, or unlearning.")
    modality = config.get("modality", "text")
    if modality != "text":
        raise ValueError(
            "The Python API currently supports text recipes. "
            "Use the existing CLI for multimodal updates."
        )

    preset_overrides = []
    experiment = config.get("experiment")
    if experiment:
        if any(key in config for key in ("model", "dataset", "method")):
            raise ValueError(
                "With experiment, select model/data/trainer through overrides; "
                "do not mix an explicit experiment with shorthand selectors."
            )
    else:
        registry = yaml.safe_load(
            (config_dir() / "api/presets.yaml").read_text(encoding="utf-8")
        )
        selected = {
            "operation": operation,
            "model": registry["model_aliases"].get(
                config.get("model"), config.get("model")
            ),
            "dataset": registry["dataset_aliases"].get(
                config.get("dataset"), config.get("dataset")
            ),
            "method": config.get("method"),
        }
        preset = next(
            (
                entry
                for entry in registry["presets"]
                if all(entry[key] == value for key, value in selected.items())
            ),
            None,
        )
        if preset is None:
            raise ValueError(
                "No preset matches these selectors. Use operation + experiment "
                "and Hydra overrides for another registered configuration."
            )
        experiment = preset["experiment"]
        preset_overrides = list(preset.get("overrides", []))

    overrides = config.get("overrides", [])
    if not isinstance(overrides, (list, tuple)) or any(
        not isinstance(item, str) or "=" not in item or item.startswith("--")
        for item in overrides
    ):
        raise TypeError("overrides must be a sequence of Hydra key=value strings.")
    if any(
        item.lstrip("+~").split("=", 1)[0] in {"operation", "modality", "experiment"}
        for item in overrides
    ):
        raise ValueError("Set operation, modality, and experiment in config directly.")

    args = [
        f"operation={operation}",
        f"modality={modality}",
        f"experiment={experiment}",
        f"task_name=api_{operation}_{uuid4().hex[:12]}",
        f"paths.work_dir={json.dumps(str(Path.cwd()), ensure_ascii=False)}",
        *preset_overrides,
        *overrides,
        f"++trainer.args.do_eval={str(evaluate).lower()}",
    ]
    if not evaluate:
        args.extend(
            [
                "++trainer.args.eval_on_start=false",
                "++trainer.args.eval_strategy=no",
            ]
        )
    if config.get("output_dir") is not None:
        args.append(
            "paths.output_dir="
            + json.dumps(str(Path(config["output_dir"]).resolve()), ensure_ascii=False)
        )
    return args


def _check_evaluation(cfg, evaluate: bool, *, check_files: bool) -> None:
    if not evaluate:
        if cfg.get("edit_type") == "independent":
            raise ValueError("Independent editing requires evaluate=True.")
        return
    if not cfg.get("eval"):
        raise ValueError("evaluate=True requires a configured evaluator.")
    if cfg.operation != "inject" or cfg.modality != "text":
        return
    for node in cfg.eval.get("inject_metrics", {}).values():
        field = {
            "InjectAccuracyEvaluator": "eval_data_path",
            "InjectRetentionEvaluator": "benchmark_data_path",
        }.get(node.get("handler"))
        if field is None:
            continue
        if node.get(field.removesuffix("_path")):
            continue
        value = node.get(field)
        if not value:
            raise ValueError(f"{node.handler} requires {field} before training.")
        if check_files and not Path(value).is_file():
            raise FileNotFoundError(f"Evaluation input is missing: {value}")


def _collect_metrics(output_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Preserve native summaries, including missing values and metric metadata."""
    paths = sorted(output_dir.glob("*_SUMMARY.json"))
    state_file = output_dir / "trainer_state.json"
    step = 0
    if state_file.is_file():
        step = json.loads(state_file.read_text(encoding="utf-8")).get("global_step")
        if type(step) is not int or step < 0:
            raise ValueError("trainer_state.json has an invalid global_step.")
    paths.extend(
        sorted((output_dir / f"checkpoint-{step}/evals").glob("*_SUMMARY.json"))
    )
    metrics, sources = {}, []
    for path in paths:
        if not path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError(f"Summary points outside the run directory: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a summary object: {path}")
        key = path.relative_to(output_dir).as_posix().removesuffix("_SUMMARY.json")
        metrics[key] = payload
        sources.append(str(path))
    return metrics, sources


def run_surgery(
    config: Mapping[str, Any],
    *,
    evaluate: bool = True,
    preview: bool = False,
    reviewed_config: str | None = None,
) -> dict[str, Any]:
    """Configure, execute, and evaluate a knowledge update with the existing CLI.

    Args:
        config: Either operation/model/dataset/method selectors from
            configs/api/presets.yaml, or operation/experiment for a registered
            text recipe. Optional fields: modality (text), output_dir, and overrides (a list
            of Hydra key=value strings). TOFU and TOFU:forget10 select the
            TOFU-trained checkpoint with forget10 and retain90.
        evaluate: Run the recipe's native evaluators after updating. False also
            disables evaluation callbacks and returns an empty metrics mapping.
        preview: Compose and return the configuration without loading a model,
            starting a subprocess, or creating an output directory.
        reviewed_config: The preflight review_hash explicitly checked by the user.
            Required for execution; changing parameters or inputs invalidates it.

    Returns:
        status, operation, output_dir, resolved_config, command, metrics, and
        summary_files. Metrics are grouped by their native summary source; null
        values and protocol metadata are preserved. No model object is returned.

    Raises:
        ValueError: Invalid selectors, missing evaluator settings, or an existing
            nonempty output directory. Explicit experiment selection avoids
            guessing configurations for unsupported shorthand combinations.
        FileNotFoundError: Required injection evaluation files are absent.
        subprocess.CalledProcessError: The existing CLI reports a failed run.
        RuntimeError: Evaluation produced no summaries, or Hydra is already in
            an initialization context. Run the API outside that context.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    if type(evaluate) is not bool or type(preview) is not bool:
        raise TypeError("evaluate and preview must be bool values.")

    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    from surgery import _compose_config
    from know_surgery.preflight import (
        inspect_configuration,
        prepare_config,
        require_review,
    )

    if GlobalHydra.instance().is_initialized():
        raise RuntimeError(
            "Call run_surgery outside an active Hydra initialization context."
        )
    args = _arguments(config, evaluate)
    prefix = [sys.executable, "-m", "know_surgery"]
    cfg, _ = _compose_config(args, command_prefix=prefix)
    prepare_config(cfg)
    preflight = inspect_configuration(cfg)
    _check_evaluation(cfg, evaluate, check_files=not preview)
    output_dir = Path(cfg.trainer.args.output_dir).resolve()
    if output_dir != Path(cfg.paths.output_dir).resolve():
        raise ValueError(
            "Use config['output_dir'] to keep training and evaluation outputs together."
        )
    result = {
        "status": "preview" if preview else "running",
        "operation": cfg.operation,
        "output_dir": str(output_dir),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "command": prefix + args,
        "metrics": {},
        "summary_files": [],
        "preflight": preflight,
    }
    if preview:
        return result
    require_review(preflight, reviewed_config)
    result["command"].extend(["--reviewed-config", reviewed_config])
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(
            "Output directory is not empty; choose a new output_dir for this run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    record = output_dir / "python_api.json"
    with record.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    try:
        subprocess.run(result["command"], check=True)
        trace_file = output_dir / "run_trace.json"
        if trace_file.is_file():
            trace = json.loads(trace_file.read_text(encoding="utf-8"))
            if trace.get("status") in {"failed", "error"}:
                raise RuntimeError(f"Surgery failed; inspect {trace_file}.")
        if evaluate:
            result["metrics"], result["summary_files"] = _collect_metrics(output_dir)
            if not result["metrics"]:
                raise RuntimeError(
                    "Evaluation produced no summary files; inspect the run output."
                )
        result["status"] = "completed"
    except BaseException:
        result["status"] = "failed"
        raise
    finally:
        record.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return result
