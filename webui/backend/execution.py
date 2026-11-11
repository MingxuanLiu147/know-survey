"""Small, stateless checks shared by workbench and configured Agent execution."""

from __future__ import annotations

import hashlib
import json
import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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

    def remote_dataset(node):
        source = str(node.get("path") or "") if isinstance(node, dict) else ""
        return bool(source) and not (
            source
            in {
                "json",
                "csv",
                "parquet",
                "text",
                "arrow",
                "imagefolder",
                "audiofolder",
                "webdataset",
            }
            or source.startswith(("/", "./", "../", "~/"))
            or source.endswith(".py")
            or (root / source).exists()
        )

    def inspect(value, *, selected=False, location=(), remote=False):
        if isinstance(value, dict):
            for key, item in value.items():
                inspect(
                    item,
                    selected=selected or key in file_keys,
                    location=(*location, str(key)),
                    remote=remote_dataset(item) if key == "hf_args" else remote,
                )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(
                    item,
                    selected=selected,
                    location=(*location, str(index)),
                    remote=remote,
                )
        elif selected and isinstance(value, str):
            if value == "???" or not value.strip():
                raise ValueError(
                    f"Required input {'.'.join(location)} is not configured. / "
                    "必需的数据输入尚未填写，请选择数据文件或先上传数据。"
                )
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
            if urlsplit(value).scheme or (
                remote and not value.startswith(("/", "./", "../", "~/"))
            ):
                return
            path = (root / Path(value).expanduser()).resolve()
            if glob.has_magic(str(path)) and not path.exists():
                matches = sorted(glob.glob(str(path), recursive=True))
                if not matches:
                    raise ValueError(
                        f"Local input pattern {'.'.join(location)} matches no files. / "
                        "本地数据路径未匹配到文件，请检查输入路径。"
                    )
                for match in matches:
                    inspect(match, selected=True, location=location)
                return
            if not path.exists():
                raise ValueError(
                    f"Local input {'.'.join(location)} does not exist. / "
                    "本地数据输入不存在，请检查文件路径或先上传数据。"
                )
            stat = path.stat()
            row = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if path.is_dir():
                row["kind"] = "directory"
                warnings.append(
                    f"Directory contents are not frozen by preview: {'.'.join(location)}"
                )
            elif stat.st_size <= 64 * 1024 * 1024:
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
