import pytest

from contextlab.agent.models import AgentState, Message
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import ContextOverflow, TokenBudget
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
    original = (
        "noise\n" * 100
        + "FAILED test_math.py::test_add - AssertionError: 3 != 4\n"
        + "more\n" * 100
    )
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


@pytest.mark.asyncio
async def test_policy_never_reduces_system_or_task_messages() -> None:
    state = state_with_tool(Message(role="tool", name="read_file", content="x" * 2000))
    policy = BoundedToolOutputPolicy(per_tool_tokens=10, total_tool_tokens=10)
    prepared = await policy.prepare_context(state, TokenBudget(BudgetConfig()))
    assert prepared.messages[0].content == "rules"
    assert prepared.messages[1].content == "task"


@pytest.mark.asyncio
async def test_non_tool_overflow_is_not_silently_truncated() -> None:
    state = state_with_tool(Message(role="tool", name="read_file", content="short"))
    state.canonical_messages[1] = Message(role="user", content="task" * 500)
    policy = BoundedToolOutputPolicy(per_tool_tokens=10, total_tool_tokens=10)
    budget = TokenBudget(
        BudgetConfig(context_window=100, reserved_output_tokens=20, safety_margin_tokens=10)
    )
    with pytest.raises(ContextOverflow):
        await policy.prepare_context(state, budget)


@pytest.mark.asyncio
async def test_bounded_policy_does_not_implicitly_retrieve() -> None:
    state = state_with_tool(Message(role="tool", name="read_file", content="x" * 1000))
    result = await BoundedToolOutputPolicy().recover(
        "forgotten", state, TokenBudget(BudgetConfig())
    )
    assert result.items == []
