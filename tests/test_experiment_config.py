from pathlib import Path

import pytest
from pydantic import ValidationError

from contextlab.benchmarking.config import ExperimentConfig, load_experiment_config


def test_loads_full_control_matrix() -> None:
    config = load_experiment_config(Path("configs/benchmark.yaml"))
    assert len(config.policies) == 4
    assert {budget.name for budget in config.budgets} == {"short", "medium", "long"}
    assert config.tasks[0].is_absolute()


def test_requires_seed_for_every_trial() -> None:
    with pytest.raises(ValidationError, match="seed"):
        ExperimentConfig(
            experiment_id="bad",
            tasks=[Path("task.yaml")],
            budgets=[{"name": "short", "context_window": 1000}],
            trials=2,
            seeds=[1],
        )
