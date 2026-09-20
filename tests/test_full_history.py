import pytest

from contextlab.agent.models import AgentState, Message
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.policies.full_history import FullHistoryPolicy


@pytest.mark.asyncio
async def test_full_history_preserves_order_and_content() -> None:
    state = AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="full-history",
        system_instruction="s",
        task_instruction="t",
        objective="o",
        canonical_messages=[
            Message(role="system", content="rules"),
            Message(role="user", content="task"),
            Message(role="tool", content="large observation", name="read_file"),
        ],
    )
    prepared = await FullHistoryPolicy().prepare_context(state, TokenBudget(BudgetConfig()))
    assert [item.content for item in prepared.messages] == ["rules", "task", "large observation"]
    assert prepared.audit.removed_message_indices == []
