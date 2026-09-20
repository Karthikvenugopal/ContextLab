import pytest

from contextlab.agent.models import AgentState, Message
from contextlab.agent.models import AgentEvent, EventKind
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.policies.compaction import CompactionPolicy, CompactionTriggers


@pytest.mark.asyncio
async def test_deterministic_compaction_preserves_structured_state() -> None:
    messages = [Message(role="system", content="rules"), Message(role="user", content="fix parser")]
    messages.extend(Message(role="tool", name="read_file", content=f"finding {i}") for i in range(8))
    state = AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="compaction",
        system_instruction="rules",
        task_instruction="fix parser",
        objective="tests pass",
        canonical_messages=messages,
        files_modified={"src/parser.py"},
        step=3,
    )
    policy = CompactionPolicy(
        triggers=CompactionTriggers(every_steps=1, utilization_threshold=None, tool_output_tokens=None)
    )
    prepared = await policy.prepare_context(state, TokenBudget(BudgetConfig()))
    assert "Files modified: src/parser.py" in prepared.messages[2].content
    assert prepared.audit.compressed_message_indices
    assert len(state.canonical_messages) == 10


@pytest.mark.asyncio
async def test_tracks_file_revisits_after_compaction() -> None:
    state = AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="compaction",
        system_instruction="s",
        task_instruction="t",
        objective="o",
        step=2,
        canonical_messages=[
            Message(role="system", content="s"),
            Message(role="user", content="t"),
            *[Message(role="tool", name="read_file", content="old") for _ in range(5)],
        ],
    )
    old_event = AgentEvent(
        kind=EventKind.TOOL_RESULT,
        step=1,
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="compaction",
        payload={"result": {"call_id": "old", "metadata": {"path": "src/a.py"}}},
    )
    state.events.append(old_event)
    policy = CompactionPolicy(
        triggers=CompactionTriggers(every_steps=1, utilization_threshold=None, tool_output_tokens=None)
    )
    await policy.prepare_context(state, TokenBudget(BudgetConfig()))
    revisit = old_event.model_copy(
        update={"id": "new", "step": 3, "payload": {"result": {"call_id": "new", "metadata": {"path": "src/a.py"}}}}
    )
    await policy.observe(revisit, state)
    assert policy.revisits[0]["path"] == "src/a.py"
