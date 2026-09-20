from pathlib import Path

import pytest

from contextlab.benchmarking.config import ContextBudgetVariant, ExperimentConfig
from contextlab.benchmarking.runner import ExperimentRunner


@pytest.mark.asyncio
async def test_controlled_runner_uses_clean_identical_baselines(tmp_path: Path) -> None:
    config = ExperimentConfig(
        experiment_id="test-experiment",
        tasks=[Path("benchmarks/tasks/localized_bug.yaml").resolve()],
        policies=["full-history", "bounded-tool-output"],
        budgets=[ContextBudgetVariant(name="short", context_window=2048)],
        seeds=[11],
        mock=True,
        output_directory=tmp_path,
    )
    output = await ExperimentRunner(config).run()
    records = [__import__("json").loads(path.read_text()) for path in (output / "runs").glob("*.json")]
    assert len(records) == 2
    assert len({record["baseline_revision"] for record in records}) == 1
    assert all(record["official_evaluation"]["success"] for record in records)
    assert {record["policy_id"] for record in records} == {"full-history", "bounded-tool-output"}
