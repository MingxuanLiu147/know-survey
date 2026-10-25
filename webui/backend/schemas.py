"""Typed API contracts shared by the backend endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Operation = Literal["unlearning", "inject", "edit"]
Modality = Literal["text", "multimodal"]
Provider = Literal["local", "openai", "deepseek"]
Language = Literal["en", "zh"]


class PlanRequest(BaseModel):
    intent: str = Field(min_length=3, max_length=4000)
    provider: Provider = "local"
    language: Language = "en"
    operation: Operation | None = None
    modality: Modality | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    methods: list[str] | None = None
    num_cases: int | None = Field(default=None, ge=1, le=100_000)


class AgentRequest(BaseModel):
    modality: Modality | None = None
    intent: str = Field(min_length=3, max_length=4000)
    provider: Literal["openai", "deepseek"]
    execute: bool = False
    cuda: str | None = Field(default=None, pattern=r"^[0-9]+(?:,[0-9]+)*$")
    max_steps: int = Field(default=12, ge=1, le=30)
    max_cases: int = Field(default=2, ge=1, le=100_000)
    timeout: int = Field(default=600, ge=1, le=3600)
    configuration: ReviewedConfiguration | None = None
    preview_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_configuration_fields(cls, value):
        if isinstance(value, dict) and value.get("configuration") is not None:
            unknown = set(value) - set(cls.model_fields)
            if unknown:
                raise ValueError(
                    f"Unknown configuration-session fields: {sorted(unknown)}"
                )
        return value

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.configuration is not None:
            if self.modality not in (None, self.configuration.modality):
                raise ValueError("Session modality differs from the configuration.")
            if "max_cases" in self.model_fields_set:
                raise ValueError("Set case limits in configuration, not max_cases.")
            if self.cuda is not None:
                raise ValueError(
                    "Configuration sessions use the server device environment."
                )
        elif self.preview_hash is not None:
            raise ValueError("preview_hash requires configuration.")
        return self


class ComposeRequest(BaseModel):
    operation: Operation
    modality: Modality = "text"
    recipe_id: str | None = None
    experiment: str | None = None
    model: str | None = None
    trainer: str | None = None
    datasets: dict[str, str] = Field(default_factory=dict)
    evaluation: str | None = None
    task_name: str = Field(default="webui_quick_run", min_length=1, max_length=96)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    smoke: bool = True
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_name")
    @classmethod
    def validate_task_name(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("Use letters, digits, dot, underscore, or hyphen.")
        return value


class RunRequest(ComposeRequest):
    gpu: int | None = Field(default=None, ge=0, le=127)
    request_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$"
    )
    preview_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class ReviewedConfiguration(ComposeRequest):
    """Exact workbench parameters accepted by a configuration-bound session."""

    model_config = ConfigDict(extra="forbid")


class RegisterUploadRequest(BaseModel):
    modality: Modality = "text"
    upload_id: str = Field(min_length=8, max_length=64)
    dataset_id: str = Field(min_length=1, max_length=48)
    model: str = "gpt2"
    trainer: str = "inject/LoRA"
    evaluation: str | None = "inject"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class CompareRequest(BaseModel):
    result_ids: list[str] = Field(min_length=2, max_length=12)
