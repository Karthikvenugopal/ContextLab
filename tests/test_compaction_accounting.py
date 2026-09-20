import pytest

from contextlab.agent.models import AgentState, Message
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import TokenBudget
from contextlab.context.policies.compaction import (
    CompactionPolicy,
    CompactionTriggers,
    ModelCompactor,
)
from contextlab.inference.accounting import InferenceTotals
from contextlab.inference.client import InferenceResponse, Usage


class FakeInference:
    async def complete(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse:
        assert request_kind == "compaction"
        return InferenceResponse(
            content="structured compact summary",
            model="mock",
            request_kind=request_kind,
            usage=Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            latency_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_compaction_request_is_auxiliary_inference_workload() -> None:
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
            *[Message(role="tool", name="read_file", content=f"finding {i}") for i in range(6)],
        ],
    )
    policy = CompactionPolicy(
        compactor=ModelCompactor(FakeInference()),  # type: ignore[arg-type]
        triggers=CompactionTriggers(every_steps=1, utilization_threshold=None, tool_output_tokens=None),
    )
    await policy.prepare_context(state, TokenBudget(BudgetConfig()))
    record = policy.drain_new_records()[0]
    totals = InferenceTotals()
    totals.record(record["inference_response"])  # type: ignore[arg-type]
    assert totals.auxiliary_tokens == 120
    assert totals.total_tokens == 120
    assert record["input_tokens"] == 100
