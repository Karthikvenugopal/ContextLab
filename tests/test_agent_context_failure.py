from pathlib import Path

import pytest

from contextlab.agent.models import AgentState, EventKind
from contextlab.agent.runtime import CodingAgent
from contextlab.config import AgentConfig, BudgetConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.policies.full_history import FullHistoryPolicy
from contextlab.tools.execution import MutationTools
from contextlab.tools.repository import RepositoryTools
from contextlab.tools.workspace import RepositoryWorkspace


class NeverCalledInference:
    async def complete(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("overflow must occur before inference")


@pytest.mark.asyncio
async def test_agent_records_structured_context_failure(tmp_path: Path) -> None:
    config = AgentConfig(
        budget=BudgetConfig(context_window=80, reserved_output_tokens=20, safety_margin_tokens=10)
    )
    workspace = RepositoryWorkspace(tmp_path)
    agent = CodingAgent(
        config=config,
        inference=NeverCalledInference(),  # type: ignore[arg-type]
        policy=FullHistoryPolicy(),
        repository_tools=RepositoryTools(workspace),
        mutation_tools=MutationTools(workspace, allowed_commands=[]),
        budget=TokenBudget(config.budget),
    )
    state = AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="full-history",
        system_instruction="s" * 400,
        task_instruction="task",
        objective="done",
    )
    result = await agent.run(state)
    assert result.failure == "context_window"
    assert result.events[-1].kind == EventKind.CONTEXT_FAILURE
