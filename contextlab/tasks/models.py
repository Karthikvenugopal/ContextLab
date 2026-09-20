"""Strict benchmark task manifest schema."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TaskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: int = Field(default=20, gt=0)
    tool_calls: int = Field(default=40, gt=0)
    wall_seconds: float = Field(default=600, gt=0)


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Literal["localized_bug", "multi_file_bug", "feature", "refactor", "exploration"]
    repository: Path
    revision: str
    description: str
    setup_commands: list[list[str]] = Field(default_factory=list)
    visible_validation: list[str] = Field(default_factory=lambda: ["pytest", "-q"])
    evaluation_tests: Path
    success_criteria: str
    limits: TaskLimits = Field(default_factory=TaskLimits)


def load_task(path: Path) -> TaskDefinition:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = TaskDefinition.model_validate(data)
    base = path.parent
    task.repository = (base / task.repository).resolve()
    task.evaluation_tests = (base / task.evaluation_tests).resolve()
    return task
