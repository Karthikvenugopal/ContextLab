"""Validated runtime and experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PolicyName = Literal["full-history", "bounded-tool-output", "retrieval", "compaction"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictModel):
    endpoint: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer: str | None = None
    api_key: str = "not-required"
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=120, gt=0)
    retries: int = Field(default=2, ge=0, le=10)


class ExecutionLimits(StrictModel):
    max_steps: int = Field(default=20, gt=0)
    max_model_requests: int = Field(default=20, gt=0)
    max_tool_calls: int = Field(default=40, gt=0)
    max_wall_seconds: float = Field(default=900, gt=0)
    max_generated_tokens: int = Field(default=20_000, gt=0)
    command_timeout_seconds: float = Field(default=120, gt=0)
    command_memory_mb: int = Field(default=4096, gt=0)
    command_output_bytes: int = Field(default=2_000_000, gt=0)


class BudgetConfig(StrictModel):
    context_window: int = Field(default=32_768, gt=0)
    reserved_output_tokens: int = Field(default=2_048, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)

    @model_validator(mode="after")
    def usable_window_is_positive(self) -> BudgetConfig:
        if self.reserved_output_tokens + self.safety_margin_tokens >= self.context_window:
            raise ValueError("reserved output and safety margin exhaust context window")
        return self


class PolicyConfig(StrictModel):
    name: PolicyName = "full-history"
    per_tool_tokens: int = Field(default=1200, gt=0)
    total_tool_tokens: int = Field(default=6000, gt=0)
    retrieval_tokens: int = Field(default=3000, gt=0)
    compaction_threshold: float = Field(default=0.75, gt=0, le=1)
    compaction_every_steps: int | None = Field(default=None, gt=0)
    deterministic_compaction: bool = False


class AgentConfig(StrictModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    allowed_commands: list[str] = Field(default_factory=lambda: ["pytest", "python", "ruff"])
    redact_content: bool = False


def load_agent_config(path: Path) -> AgentConfig:
    """Load strict YAML configuration."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AgentConfig.model_validate(data)
