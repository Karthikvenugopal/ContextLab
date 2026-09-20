"""Validated controlled experiment matrix configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contextlab.config import ModelConfig, PolicyName


class ContextBudgetVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["short", "medium", "long"]
    context_window: int = Field(gt=256)
    reserved_output_tokens: int = Field(default=256, gt=0)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_id: str
    tasks: list[Path]
    policies: list[PolicyName] = Field(
        default_factory=lambda: [
            "full-history",
            "bounded-tool-output",
            "retrieval",
            "compaction",
        ]
    )
    budgets: list[ContextBudgetVariant]
    trials: int = Field(default=1, gt=0)
    seeds: list[int] = Field(default_factory=lambda: [0])
    order: Literal["rotate", "randomize", "fixed"] = "rotate"
    model: ModelConfig = Field(default_factory=ModelConfig)
    output_directory: Path = Path("results")
    mock: bool = False

    @model_validator(mode="after")
    def enough_seeds(self) -> ExperimentConfig:
        if len(self.seeds) < self.trials:
            raise ValueError("provide at least one seed per trial")
        if len(set(self.policies)) != len(self.policies):
            raise ValueError("policies must be unique")
        return self


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(raw)
    base = path.parent
    config.tasks = [(base / task).resolve() for task in config.tasks]
    config.output_directory = (base / config.output_directory).resolve()
    return config
