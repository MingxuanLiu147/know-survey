"""Safe operation-aware upload preview and reviewed config registration."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from webui.utils.extensions import (
    ExtensionInputError,
    canonicalize_records,
    default_field_mapping,
    parse_data_file,
    parse_field_mapping,
    validate_identifier,
)

from .catalog import ROOT, CatalogError, RepositoryCatalog
from .schemas import ComposeRequest, RegisterUploadRequest


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_ID = re.compile(r"^[a-f0-9]{16,64}$")
SUPPORTED_OPERATIONS = {"unlearning", "inject", "edit"}


class UploadError(ValueError):
    """Raised when an upload draft cannot be previewed or registered."""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class UploadStore:
    """Persist reviewed normalized data without importing uploaded code."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.upload_root = self.root / "artifacts" / "webui" / "uploads"
        self.upload_root.mkdir(parents=True, exist_ok=True)

    def preview(
        self,
        *,
        data: bytes,
        filename: str,
        mapping: str | dict[str, str] | None,
        knowledge_type: str,
        format_type: str,
        operation: str = "inject",
        retain_data: bytes | None = None,
        retain_filename: str | None = None,
    ) -> dict[str, Any]:
        if not data:
            raise UploadError("Uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError("Upload exceeds the 50 MiB limit.")
        if operation not in SUPPORTED_OPERATIONS:
            raise UploadError("Unsupported custom upload operation.")
        if operation == "unlearning" and not retain_data:
            raise UploadError("Unlearning requires both forget and retain files.")
        if retain_data and len(retain_data) > MAX_UPLOAD_BYTES:
            raise UploadError("Retain upload exceeds the 50 MiB limit.")
        try:
            field_mapping = (
                parse_field_mapping(operation, mapping)
                if mapping
                else default_field_mapping(operation, format_type)
            )
            if operation == "unlearning":
                inputs = {
                    "forget": (data, filename),
                    "retain": (
                        retain_data or b"",
                        retain_filename or "retain.jsonl",
                    ),
                }
            else:
                role = "train" if operation == "inject" else "edit"
                inputs = {role: (data, filename)}
            normalized_by_role = {
                role: canonicalize_records(
                    operation,
                    parse_data_file(raw, filename=source_name),
                    field_mapping,
                    knowledge_type=knowledge_type,
                    format_type=format_type,
                    role=role,
                )
                for role, (raw, source_name) in inputs.items()
            }
        except ExtensionInputError as exc:
            raise UploadError(str(exc)) from exc
        upload_id = uuid.uuid4().hex
        directory = self.upload_root / upload_id
        directory.mkdir(parents=True, exist_ok=False)
        normalized_paths: dict[str, str] = {}
        for role, normalized in normalized_by_role.items():
            filename_suffix = "" if len(normalized_by_role) == 1 else f"_{role}"
            normalized_path = directory / f"normalized{filename_suffix}.jsonl"
            normalized_path.write_text(
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n" for item in normalized
                ),
                encoding="utf-8",
            )
            normalized_paths[role] = normalized_path.relative_to(self.root).as_posix()
        source_sha256s = {
            role: hashlib.sha256(raw).hexdigest()
            for role, (raw, _source_name) in inputs.items()
        }
        combined_digest = hashlib.sha256(
            b"\0".join(raw for raw, _source_name in inputs.values())
        ).hexdigest()
        preview_rows = [
            {**item, "dataset_role": role}
            for role, normalized in normalized_by_role.items()
            for item in normalized[:3]
        ]
        metadata = {
            "schema_version": "1.0",
            "upload_id": upload_id,
            "status": "previewed",
            "operation": operation,
            "knowledge_type": knowledge_type,
            "format_type": format_type,
            "source_filename": Path(filename).name,
            "source_filenames": {
                role: Path(source_name).name
                for role, (_raw, source_name) in inputs.items()
            },
            "source_sha256": combined_digest,
            "source_sha256s": source_sha256s,
            "records": sum(len(items) for items in normalized_by_role.values()),
            "role_records": {
                role: len(items) for role, items in normalized_by_role.items()
            },
            "field_mapping": field_mapping,
            "normalized_path": next(iter(normalized_paths.values())),
            "normalized_paths": normalized_paths,
            "evidence_level": "configured",
        }
        _write_json(directory / "manifest.json", metadata)
        return {**metadata, "preview": preview_rows}

    def _manifest(
        self,
        upload_id: str,
        *,
        expected_operation: str | None = None,
    ) -> dict[str, Any]:
        if not UPLOAD_ID.fullmatch(upload_id):
            raise UploadError("Invalid upload id.")
        path = self.upload_root / upload_id / "manifest.json"
        if not path.is_file():
            raise UploadError("Upload draft was not found.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UploadError("Upload manifest is invalid.") from exc
        operation = payload.get("operation")
        if operation not in SUPPORTED_OPERATIONS:
            raise UploadError("Upload draft has an unsupported operation.")
        if expected_operation is not None and operation != expected_operation:
            raise UploadError("Upload operation does not match the registration route.")
        return payload

    def register(
        self,
        request: RegisterUploadRequest,
        catalog: RepositoryCatalog,
        *,
        operation: str | None = None,
    ) -> dict[str, Any]:
        manifest = self._manifest(
            request.upload_id,
            expected_operation=operation,
        )
        operation = str(manifest["operation"])
        try:
            dataset_id = validate_identifier(
                request.dataset_id,
                field="dataset id",
            )
        except ExtensionInputError as exc:
            raise UploadError(str(exc)) from exc
        snapshot = catalog.snapshot()
        text_models = {
            item["id"] for item in snapshot["models"] if item["modality"] == "text"
        }
        operation_trainers = {item["id"] for item in snapshot["trainers"][operation]}
        if request.model not in text_models:
            raise UploadError("Selected model is not registered as a text model.")
        if request.trainer not in operation_trainers:
            raise UploadError(f"Selected {operation} method is not registered.")
        if request.evaluation not in set(snapshot["evaluations"]):
            raise UploadError("Selected evaluation is not registered.")
        if operation == "unlearning":
            return self._register_unlearning(
                request=request,
                catalog=catalog,
                manifest=manifest,
                dataset_id=dataset_id,
            )

        config_name = f"webui_{dataset_id}_{operation}"
        config_path = (
            self.root / "configs" / "data" / "datasets" / f"{config_name}.yaml"
        )
        experiment_id = f"{operation}/webui/{dataset_id}"
        experiment_path = self.root / "configs" / "experiment" / f"{experiment_id}.yaml"
        if config_path.exists() or experiment_path.exists():
            raise UploadError("This dataset id already exists; choose a new stable id.")

        root_interpolation = "$" + "{paths.root_dir}"
        data_path = (
            f"{root_interpolation}/artifacts/webui/uploads/"
            f"{request.upload_id}/normalized.jsonl"
        )
        role = "train" if operation == "inject" else "edit"
        dataset_key = f"WebUI_{dataset_id}_{role}"
        if operation == "inject":
            dataset_args = {
                "data_path": data_path,
                "instruction_key": "prompt",
                "input_key": "input",
                "output_key": "target_new",
                "max_length": 2048,
            }
            dataset_handler = "AlpacaDataset"
        else:
            dataset_args = {
                "data_path": data_path,
                "prompt_key": "prompt",
                "subject_key": "subject",
                "target_new_key": "target_new",
                "target_old_key": "target_old",
                "max_length": 2048,
                "apply_chat_template": False,
            }
            dataset_handler = "EditingDataset"
        dataset_config = {
            dataset_key: {
                "handler": dataset_handler,
                "args": dataset_args,
            }
        }
        defaults: list[dict[str, str]] = [
            {"override /model": request.model},
            {"override /trainer": request.trainer},
            {"override /data": operation},
            {f"override /data/datasets@data.{role}": config_name},
        ]
        if request.trainer == "inject/LoReFT":
            defaults.append({"override /collator": "DataCollatorForReFTDataset"})
        defaults.append({"override /eval": request.evaluation})
        experiment_config: dict[str, Any] = {
            "defaults": defaults,
            "task_name": f"webui_{dataset_id}_{operation}",
        }
        if operation == "inject":
            experiment_config["trainer"] = {
                "args": {
                    "seed": request.seed,
                    "max_steps": 1,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "do_eval": False,
                    "save_strategy": "no",
                    "report_to": "none",
                }
            }
        else:
            experiment_config.update(
                {
                    "edit_type": "single",
                    "max_edits": 1,
                    "save_edited_model": False,
                }
            )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                dataset_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        experiment_path.write_text(
            "# @package _global_\n"
            "# Generated from a reviewed WebUI upload; compose before execution.\n"
            + yaml.safe_dump(
                experiment_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        manifest.update(
            {
                "status": "registered",
                "dataset_id": config_name,
                "dataset_config": config_path.relative_to(self.root).as_posix(),
                "experiment": experiment_id,
                "experiment_config": experiment_path.relative_to(self.root).as_posix(),
            }
        )
        _write_json(
            self.upload_root / request.upload_id / "manifest.json",
            manifest,
        )
        compose_request = ComposeRequest(
            operation=operation,
            experiment=experiment_id,
            task_name=f"webui_{dataset_id}_{operation}_quick_run",
            seed=request.seed,
            smoke=True,
        )
        try:
            command = catalog.command_preview(compose_request)
        except CatalogError as exc:
            raise UploadError(
                f"Registered files failed catalog validation: {exc}"
            ) from exc
        return {
            **manifest,
            "command": command,
            "compose_request": compose_request.model_dump(),
        }

    def _register_unlearning(
        self,
        *,
        request: RegisterUploadRequest,
        catalog: RepositoryCatalog,
        manifest: dict[str, Any],
        dataset_id: str,
    ) -> dict[str, Any]:
        config_names = {
            role: f"webui_{dataset_id}_{role}" for role in ("forget", "retain")
        }
        config_paths = {
            role: self.root / "configs" / "data" / "datasets" / f"{config_name}.yaml"
            for role, config_name in config_names.items()
        }
        experiment_id = f"unlearn/webui/{dataset_id}"
        experiment_path = self.root / "configs" / "experiment" / f"{experiment_id}.yaml"
        if (
            any(path.exists() for path in config_paths.values())
            or experiment_path.exists()
        ):
            raise UploadError("This dataset id already exists; choose a new stable id.")

        normalized_paths = manifest.get("normalized_paths")
        if not isinstance(normalized_paths, dict) or set(normalized_paths) != {
            "forget",
            "retain",
        }:
            raise UploadError("Unlearning upload is missing normalized data roles.")
        root_interpolation = "$" + "{paths.root_dir}"
        for role, config_path in config_paths.items():
            data_path = f"{root_interpolation}/{normalized_paths[role]}"
            dataset_config = {
                f"WebUI_{dataset_id}_{role}": {
                    "handler": "QADataset",
                    "args": {
                        "hf_args": {
                            "path": "json",
                            "data_files": data_path,
                            "split": "train",
                        },
                        "question_key": "prompt",
                        "answer_key": "target_old",
                        "max_length": 2048,
                    },
                }
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(
                    dataset_config,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )

        experiment_config = {
            "defaults": [
                {"override /model": request.model},
                {"override /trainer": request.trainer},
                {"override /data": "unlearn"},
                {
                    "override /data/datasets@data.forget": config_names["forget"],
                },
                {
                    "override /data/datasets@data.retain": config_names["retain"],
                },
                {"override /eval": request.evaluation},
            ],
            "mode": "unlearn",
            "task_name": f"webui_{dataset_id}_unlearning",
            "data": {"anchor": "forget"},
            "trainer": {
                "args": {
                    "seed": request.seed,
                    "max_steps": 1,
                    "num_train_epochs": 1,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "do_eval": False,
                    "save_strategy": "no",
                    "report_to": "none",
                }
            },
        }
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_path.write_text(
            "# @package _global_\n"
            "# Generated from reviewed forget/retain WebUI uploads.\n"
            + yaml.safe_dump(
                experiment_config,
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        manifest.update(
            {
                "status": "registered",
                "dataset_id": f"webui_{dataset_id}_unlearning",
                "dataset_config": config_paths["forget"]
                .relative_to(self.root)
                .as_posix(),
                "dataset_configs": {
                    role: path.relative_to(self.root).as_posix()
                    for role, path in config_paths.items()
                },
                "experiment": experiment_id,
                "experiment_config": experiment_path.relative_to(self.root).as_posix(),
            }
        )
        _write_json(
            self.upload_root / request.upload_id / "manifest.json",
            manifest,
        )
        compose_request = ComposeRequest(
            operation="unlearning",
            experiment=experiment_id,
            task_name=f"webui_{dataset_id}_unlearning_quick_run",
            seed=request.seed,
            smoke=True,
        )
        try:
            command = catalog.command_preview(compose_request)
        except CatalogError as exc:
            raise UploadError(
                f"Registered files failed catalog validation: {exc}"
            ) from exc
        return {
            **manifest,
            "command": command,
            "compose_request": compose_request.model_dump(),
        }
