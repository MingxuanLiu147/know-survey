"""Small, stateless checks shared by workbench and configured Agent execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException


def request_digest(request: dict[str, Any]) -> str:
    """Identify a retry without including its caller-supplied request ID."""
    payload = {key: value for key, value in request.items() if key != "request_id"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def composition_identity(resolved_config: str, root: Path) -> dict[str, Any]:
    """Record configuration and local input identities at composition time.

    This detects changes before launch; it does not freeze external files or
    promise isolation from edits made while a model is running.
    """
    composed = OmegaConf.create(resolved_config)
    # --cfg job includes runtime-only Hydra metadata. Hash it as composed rather
    # than resolving unrelated fields outside an active Hydra runtime.
    config = OmegaConf.to_container(composed, resolve=False)
    inputs: dict[str, Any] = {}
    warnings: list[str] = []
    file_keys = {
        "data_path",
        "data_files",
        "edit_data_path",
        "eval_data_path",
        "benchmark_data_path",
        "train_path",
        "retain_path",
        "forget_path",
    }

    def inspect(value, *, selected=False, location=()):
        if isinstance(value, dict):
            for key, item in value.items():
                inspect(
                    item,
                    selected=selected or key in file_keys,
                    location=(*location, str(key)),
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, selected=selected, location=(*location, str(index)))
        elif selected and isinstance(value, str):
            if "${" in value:
                try:
                    value = OmegaConf.select(composed, ".".join(location))
                except OmegaConfBaseException:
                    warnings.append(
                        f"Input identity is unavailable until runtime: {'.'.join(location)}"
                    )
                    return
            if not isinstance(value, str):
                return
            path = (root / Path(value).expanduser()).resolve()
            if not path.is_file():
                return
            stat = path.stat()
            row = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if stat.st_size <= 64 * 1024 * 1024:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                row["sha256"] = digest.hexdigest()
            inputs[str(path)] = row

    inspect(config)
    payload = {"configuration": config, "local_inputs": inputs}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "preview_hash": digest,
        "input_checks": inputs,
        "input_check_warnings": warnings,
    }
