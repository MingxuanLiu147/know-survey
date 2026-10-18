"""Versioned, fail-closed loading for editor runtime and continuation state.

Model weights alone are insufficient for memory and adapter editors. A saved
checkpoint carries a manifest binding the runtime configuration and state file
to the exact checkpoint weights and tokenizer. Evaluation uses this same loader.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

from trainer.edit.checkpoint_identity import (
    build_base_model_identity,
    validate_base_model_identity,
)
from trainer.edit.utils.cache_identity import identity_digest, tokenizer_identity
from trainer.edit.utils.nethook import replace_module

MANIFEST_NAME = "edit_artifact.json"
_SCHEMA_VERSION = 1
_STATE_HANDLERS = {
    "IKEEditor": ("ike_edit_state.json", "save_edit_state", "load_edit_state"),
    "SERACEditor": ("serac_archive.pt", "save_archive", "load_archive"),
    "SERACCompactEditor": ("serac_compact_archive.pt", "save_archive", "load_archive"),
    "WISEEditor": ("wise_adapter.pt", "save_adapter", "load_adapter"),
    "WISEMergeEditor": ("wise_merge_adapter.pt", "save_adapter", "load_adapter"),
    "GRACEEditor": ("grace_adapter.pt", "save_adapter", "load_adapter"),
    "AlphaEditEditor": ("alphaedit_state.pt", "save_editor_state", "load_editor_state"),
    "AnyEditEditor": ("anyedit_state.pt", "save_editor_state", "load_editor_state"),
}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_config(editor, method_args):
    """Persist constructor semantics, including defaults absent from Hydra."""
    config = {}
    for cls in reversed(type(editor).__mro__):
        if cls.__module__.startswith("trainer.edit."):
            for name, parameter in inspect.signature(cls.__init__).parameters.items():
                if name in {
                    "self",
                    "scope_model",
                    "scope_tokenizer",
                    "counterfactual_model",
                    "counterfactual_tokenizer",
                } or parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue
                if hasattr(editor, name):
                    config[name] = getattr(editor, name)
                elif parameter.default is not inspect.Parameter.empty:
                    config[name] = parameter.default
    config.update(method_args or {})
    for name in config:
        if hasattr(editor, name):
            config[name] = getattr(editor, name)
    if hasattr(editor, "target_layer") and "inner_params" in config:
        config["inner_params"] = editor.target_layer
    # Runtime artifacts are self-contained; historical source files must not be
    # loaded a second time by the constructor before the verified state loader.
    for name in (
        "archive",
        "memory_path",
        "demonstration_path",
        "scope_model_name_or_path",
        "counterfactual_model_name_or_path",
        "aux_device",
    ):
        if name in config:
            config[name] = None

    def encode_resource(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Unsupported editor configuration value: {type(value).__name__}"
        )

    return json.loads(json.dumps(config, default=encode_resource))


def _saved_model_identity(editor):
    if getattr(editor, "_adapter_installed", False):
        adapter = editor._get_adapter()
        base_layer = copy.deepcopy(editor._original_layer).to(
            adapter.layer.weight.device
        )
        replace_module(editor.model, editor.target_layer, base_layer)
        try:
            return build_base_model_identity(editor.model)
        finally:
            replace_module(editor.model, editor.target_layer, adapter)
    return build_base_model_identity(editor.model)


def save_edit_artifacts(editor, output_dir, method_args=None):
    """Save state and manifest after ``editor.save_model(output_dir)``.

    Returns the manifest path, or None for editors whose entire inference state
    is the saved model weights. No absolute resource paths are added implicitly.
    """
    method = type(editor).__name__
    if method not in _STATE_HANDLERS:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    filename, save_method, _ = _STATE_HANDLERS[method]
    state_path = directory / filename
    getattr(editor, save_method)(state_path)
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "method": method,
        "variant": getattr(editor, "variant", "native_adapted"),
        "method_args": _runtime_config(editor, method_args),
        "checkpoint_identity": _saved_model_identity(editor),
        "tokenizer_sha256": identity_digest(tokenizer_identity(editor.tokenizer)),
        "state": {"file": filename, "sha256": _file_digest(state_path)},
    }
    destination = directory / MANIFEST_NAME
    temporary = directory / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    return destination


def read_edit_artifact(directory):
    """Inspect a checkpoint and reject orphaned or corrupt runtime state."""
    directory = Path(directory)
    path = directory / MANIFEST_NAME
    if not path.is_file():
        if any((directory / row[0]).exists() for row in _STATE_HANDLERS.values()):
            raise ValueError(
                "Editing state exists without edit_artifact.json; regenerate the "
                "checkpoint with the versioned artifact exporter before evaluation"
            )
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("Unsupported editing artifact schema_version")
    method = manifest.get("method")
    if method not in _STATE_HANDLERS:
        raise ValueError(f"Unsupported editing artifact method: {method!r}")
    state = manifest.get("state", {})
    expected_name = _STATE_HANDLERS[method][0]
    if not isinstance(state, dict) or state.get("file") != expected_name:
        raise ValueError("Editing artifact state filename does not match its method")
    state_path = directory / expected_name
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Required editing runtime state is missing: {state_path}"
        )
    if _file_digest(state_path) != state.get("sha256"):
        raise ValueError("Editing artifact state checksum mismatch")
    if not isinstance(manifest.get("method_args"), dict):
        raise ValueError("Editing artifact is missing method_args")
    return manifest


def load_edit_artifacts(model, tokenizer, directory, *, manifest=None):
    """Restore and return an editor bound to a verified saved checkpoint.

    Inference discovers prompt editors through ``model._edit_runtime`` and
    adapter editors through installed modules. The returned editor also exposes
    the same ``edit`` API for continued editing.
    """
    manifest = manifest if manifest is not None else read_edit_artifact(directory)
    if manifest is None:
        return None
    validate_base_model_identity(
        model, manifest.get("checkpoint_identity"), checkpoint_label="Editing artifact"
    )
    if identity_digest(tokenizer_identity(tokenizer)) != manifest.get(
        "tokenizer_sha256"
    ):
        raise ValueError("Editing artifact tokenizer identity is incompatible")
    from trainer import TRAINER_REGISTRY
    from transformers import TrainingArguments

    method = manifest["method"]
    if method in {"SERACEditor", "SERACCompactEditor"}:
        import torch

        state = torch.load(
            Path(directory) / _STATE_HANDLERS[method][0],
            map_location="cpu",
            weights_only=True,
        )
        if any(
            manifest["method_args"].get(name) != value
            for name, value in state.get("config", {}).items()
        ):
            raise ValueError("SERAC artifact manifest and scope configuration disagree")
    editor_cls = TRAINER_REGISTRY[method]
    expected_variant = getattr(editor_cls, "variant", "native_adapted")
    method_args = dict(manifest["method_args"])
    if method == "IKEEditor":
        protocol = method_args.get("protocol", "edit_memory")
        expected_variant = {
            "edit_memory": "edit_memory_icl_v2",
            "copy_update_retain": "independent_copy_update_retain_v1",
            "paper_copy_update_retain": "paper_copy_update_retain_v1",
        }.get(protocol)
        if expected_variant is None:
            raise ValueError("Unknown IKE artifact demonstration protocol")
        if protocol != "edit_memory":
            state_path = Path(directory) / _STATE_HANDLERS[method][0]
            if _file_digest(state_path) != manifest.get("state", {}).get("sha256"):
                raise ValueError("Editing artifact state checksum mismatch")
            # read_edit_artifact verified this embedded corpus before any
            # constructor reads it; do not reuse the source training path.
            method_args["demonstration_path"] = None
            method_args["memory_path"] = str(state_path)
    if manifest.get("variant") != expected_variant:
        raise ValueError("Editing artifact method variant is incompatible")
    editor = editor_cls(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(output_dir=str(directory), report_to=[], bf16=False),
        **method_args,
    )
    filename, _, load_method = _STATE_HANDLERS[method]
    getattr(editor, load_method)(Path(directory) / filename)
    # Keep continuation state reachable without marking rank editors as prompt
    # transformers (EditSuite only consumes the separate _edit_runtime field).
    model._edit_state_owner = editor
    return editor


__all__ = ["load_edit_artifacts", "read_edit_artifact", "save_edit_artifacts"]
