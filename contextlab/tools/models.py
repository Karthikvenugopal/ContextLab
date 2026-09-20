"""Validated tool protocol shared by model and executor."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: Literal["list_files", "read_file", "search", "write_file", "run_command", "retrieve"]
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    content: str
    ok: bool
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
