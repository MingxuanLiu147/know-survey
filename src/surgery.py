"""Unified CLI for unlearning, injection, and editing operations."""

from __future__ import annotations

import shlex
import sys
from know_surgery.resources import config_dir


def _usage() -> str:
    return (
        "Usage: know-surgery --config-name=surgery.yaml "
        "operation=unlearning|inject|edit modality=text|multimodal "
        "[Hydra overrides] [--preflight | --reviewed-config HASH]\n"
        "Review layers, method settings and inputs with --preflight before running.\n"
        "\n"
        "Operations:\n"
        "  unlearning\n"
        "  inject\n"
        "  edit\n"
        "\n"
        "Multimodal support:\n"
        "  operation=inject modality=multimodal\n"
        "  operation=unlearning modality=multimodal\n"
        "  operation=edit modality=multimodal\n"
        "\nInstalled workflow commands:\n"
        "  know-surgery agent --help\n"
        "  know-surgery export-skills --output <source-checkout>\n"
    )


def _split_args(argv: list[str]) -> tuple[list[str], list[str], str | None]:
    route_overrides: dict[str, str] = {}
    forwarded: list[str] = []
    cfg_request = None
    config_name = "surgery.yaml"

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"-h", "--help"}:
            print(_usage())
            raise SystemExit(0)
        if arg in {"--config-name", "-cn"}:
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            config_name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--cfg":
            if i + 1 >= len(argv):
                raise ValueError("--cfg requires a value")
            cfg_request = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--cfg="):
            cfg_request = arg.split("=", 1)[1]
            i += 1
            continue

        override = arg.lstrip("+")
        if override.startswith(("operation=", "modality=")):
            key = override.split("=", 1)[0]
            route_overrides[key] = override
        else:
            forwarded.append(arg)
        i += 1

    if config_name not in {"surgery", "surgery.yaml"}:
        raise ValueError("surgery.py only accepts --config-name=surgery.yaml")
    return list(route_overrides.values()), forwarded, cfg_request


def _compose_config(argv: list[str], *, command_prefix: list[str] | None = None):
    route_overrides, forwarded, cfg_request = _split_args(argv)

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import open_dict

    config_path = config_dir()

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_path), job_name="surgery", version_base=None
    ):
        surgery_cfg = compose(config_name="surgery.yaml", overrides=route_overrides)
        operation = surgery_cfg.operation
        modality = str(surgery_cfg.get("modality", "text"))
        if modality == "text":
            routes = surgery_cfg.operations
        elif modality == "multimodal":
            routes = surgery_cfg.multimodal_operations
        else:
            raise ValueError(
                f"Unknown modality `{modality}`. Expected one of: text, multimodal"
            )

        route = routes.get(operation, None)
        if route is None:
            valid = ", ".join(routes.keys())
            raise ValueError(
                f"Operation `{operation}` is unavailable for modality `{modality}`. "
                f"Expected one of: {valid}"
            )

        cfg = compose(config_name=route.config_name, overrides=forwarded)
        with open_dict(cfg):
            cfg.operation = operation
            cfg.mode = route.mode
            cfg.modality = modality
            cfg.entrypoint = route.get("entrypoint", "train")
            cfg.invocation_command = shlex.join(
                [*(command_prefix or ["python", "src/surgery.py"]), *argv]
            )
            # Dataset YAML resources belong to the package; data and outputs
            # remain relative to the user's working directory/overrides.
            if "paths" in cfg and not any(
                arg.lstrip("+").startswith("paths.datasets=") for arg in forwarded
            ):
                cfg.paths.datasets = str(config_path / "data" / "datasets")
        return cfg, cfg_request


def main(
    argv: list[str] | None = None, *, command_prefix: list[str] | None = None
) -> None:
    import argparse
    import json
    from pathlib import Path

    from know_surgery.preflight import (
        inspect_configuration,
        prepare_config,
        require_review,
    )

    original_argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reviewed-config")
    review, forwarded = parser.parse_known_args(original_argv)
    cfg, cfg_request = _compose_config(forwarded, command_prefix=command_prefix)
    if cfg_request is not None:
        from omegaconf import OmegaConf

        if cfg_request != "job":
            raise ValueError("surgery.py currently supports only `--cfg job`.")
        print(OmegaConf.to_yaml(cfg, resolve=False))
        return

    prepare_config(cfg)
    report = inspect_configuration(cfg)
    if review.preflight:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["valid"]:
            raise SystemExit(2)
        return
    require_review(report, review.reviewed_config)
    from omegaconf import OmegaConf, open_dict

    with open_dict(cfg):
        cfg.reviewed_config_hash = report["review_hash"]
        cfg.invocation_command = shlex.join(
            [*(command_prefix or ["python", "src/surgery.py"]), *original_argv]
        )
    output = OmegaConf.select(cfg, "paths.output_dir", default=None) or cfg.get(
        "output_dir"
    )
    if output:
        receipt = Path(output) / "configuration_review.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if previous.get("review_hash") != report["review_hash"]:
                raise ValueError(
                    "Output directory belongs to another reviewed configuration. / 请为变更后的配置选择新输出目录。"
                )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if cfg.entrypoint == "mm_train":
        from mm_train import main as route_main
    elif cfg.entrypoint == "mm_edit":
        from mm_edit import main as route_main
    else:
        from train import main as route_main

    route_entrypoint = getattr(route_main, "__wrapped__", route_main)
    route_entrypoint(cfg)


if __name__ == "__main__":
    main()
