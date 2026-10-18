"""Typed API contracts shared by the backend endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Operation = Literal["unlearning", "inject", "edit"]
Provider = Literal["local", "openai", "deepseek"]
Language = Literal["en", "zh"]


class PlanRequest(BaseModel):
    intent: str = Field(min_length=3, max_length=4000)
    provider: Provider = "local"
    language: Language = "en"
    operation: Operation | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    methods: list[str] | None = None
    num_cases: int | None = Field(default=None, ge=1, le=100_000)


class AgentRequest(BaseModel):
    intent: str = Field(min_length=3, max_length=4000)
    provider: Literal["openai", "deepseek"]
    execute: bool = False
    cuda: str | None = Field(default=None, pattern=r"^[0-9]+(?:,[0-9]+)*$")
    max_steps: int = Field(default=12, ge=1, le=30)
    max_cases: int = Field(default=2, ge=1, le=100_000)
    timeout: int = Field(default=600, ge=1, le=3600)


class ComposeRequest(BaseModel):
    operation: Operation
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


class RegisterUploadRequest(BaseModel):
    upload_id: str = Field(min_length=8, max_length=64)
    dataset_id: str = Field(min_length=1, max_length=48)
    model: str = "gpt2"
    trainer: str = "inject/LoRA"
    evaluation: str = "inject"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)


class CompareRequest(BaseModel):
    result_ids: list[str] = Field(min_length=2, max_length=12)
