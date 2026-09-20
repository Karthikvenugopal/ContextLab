from pathlib import Path

import pytest
from pydantic import ValidationError

from contextlab.config import AgentConfig, BudgetConfig, load_agent_config


def test_defaults_are_valid() -> None:
    assert AgentConfig().policy.name == "full-history"


def test_budget_rejects_impossible_reservation() -> None:
    with pytest.raises(ValidationError):
        BudgetConfig(context_window=100, reserved_output_tokens=100)


def test_load_config() -> None:
    loaded = load_agent_config(Path("configs/agent.yaml"))
    assert loaded.model.model.startswith("Qwen/")
