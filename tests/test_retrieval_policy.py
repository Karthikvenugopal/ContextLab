import hashlib

import pytest

from contextlab.agent.models import AgentEvent, AgentState, EventKind, Message
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.policies.retrieval import RetrievalContextPolicy
from contextlab.retrieval.index import Document


def document(source_id: str, content: str, path: str) -> Document:
    return Document(
        source_id=source_id,
        content=content,
        source_type="repository",
        path=path,
        start_line=1,
        end_line=3,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def state() -> AgentState:
    return AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="retrieval",
        system_instruction="rules",
        task_instruction="fix parser",
        objective="handle empty request headers",
        canonical_messages=[
            Message(role="system", content="rules"),
            Message(role="user", content="fix parser"),
            Message(role="assistant", content="old thought"),
            Message(role="tool", name="read_file", content="old output"),
            Message(role="assistant", content="need EmptyHeader behavior"),
        ],
    )


@pytest.mark.asyncio
async def test_recovers_repository_and_historical_sources_within_budget() -> None:
    policy = RetrievalContextPolicy(
        [document("repo:parser", "EmptyHeader is raised by parse_headers", "src/parser.py")],
        retrieval_tokens=30,
        active_messages=1,
    )
    event = AgentEvent(
        kind=EventKind.TOOL_RESULT,
        step=1,
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="retrieval",
        payload={
            "result": {
                "call_id": "c1",
                "name": "search",
                "content": "empty request header validation lives in validator.py",
            }
        },
    )
    current = state()
    await policy.observe(event, current)
    recovered = await policy.recover("empty header", current, TokenBudget(BudgetConfig()))
    assert recovered.tokens <= 30
    assert {item.metadata["source_type"] for item in recovered.items} == {
        "repository",
        "observation",
    }
    assert all(item.metadata["content_hash"] for item in recovered.items)


@pytest.mark.asyncio
async def test_preparation_drops_old_active_messages_and_tracks_repeat_queries() -> None:
    policy = RetrievalContextPolicy(
        [document("repo:parser", "EmptyHeader behavior", "src/parser.py")], active_messages=1
    )
    current = state()
    budget = TokenBudget(BudgetConfig())
    first = await policy.prepare_context(current, budget)
    await policy.prepare_context(current, budget)
    assert 2 in first.audit.removed_message_indices
    assert first.audit.recovered_source_ids == ["repo:parser"]
    assert policy.retrieval_events[-1]["repeat_number"] == 2
