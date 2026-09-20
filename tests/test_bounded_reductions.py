import pytest

from contextlab.agent.models import AgentState, Message
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.policies.bounded import BoundedToolOutputPolicy


def state_with_tool(message: Message) -> AgentState:
    return AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="bounded-tool-output",
        system_instruction="rules",
        task_instruction="task",
        objective="done",
        canonical_messages=[
            Message(role="system", content="rules"),
            Message(role="user", content="task"),
            message,
        ],
    )


@pytest.mark.asyncio
async def test_error_focused_reduction_preserves_failure() -> None:
    original = "noise\n" * 100 + "FAILED test_math.py::test_add - AssertionError: 3 != 4\n" + "more\n" * 100
    state = state_with_tool(
        Message(role="tool", name="run_command", tool_call_id="c1", content=original)
    )
    policy = BoundedToolOutputPolicy(per_tool_tokens=40, total_tool_tokens=40)
    prepared = await policy.prepare_context(state, TokenBudget(BudgetConfig()))
    assert "AssertionError" in prepared.messages[-1].content
    assert policy.reductions[0]["method"] == "error-focused"
    assert state.canonical_messages[-1].content == original


@pytest.mark.asyncio
async def test_global_tool_budget_omits_later_outputs_with_metadata() -> None:
    state = state_with_tool(Message(role="tool", name="read_file", content="a" * 1000))
    state.canonical_messages.append(Message(role="tool", name="read_file", content="b" * 1000))
    policy = BoundedToolOutputPolicy(per_tool_tokens=50, total_tool_tokens=50)
    prepared = await policy.prepare_context(state, TokenBudget(BudgetConfig()))
    assert "omitted" in prepared.messages[-1].content
    assert policy.reductions[-1]["discarded_tokens"] > 0
