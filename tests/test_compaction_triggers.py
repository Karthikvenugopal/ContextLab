from contextlab.agent.models import AgentState
from contextlab.context.policies.compaction import CompactionTriggers


def state(step: int) -> AgentState:
    return AgentState(
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="compaction",
        system_instruction="s",
        task_instruction="t",
        objective="o",
        step=step,
    )


def test_all_compaction_trigger_modes_report_reasons() -> None:
    triggers = CompactionTriggers(
        token_threshold=500,
        utilization_threshold=0.5,
        every_steps=3,
        tool_output_tokens=100,
    )
    reasons = triggers.reasons(
        state=state(5),
        prompt_tokens=600,
        usable_tokens=1000,
        accumulated_tool_tokens=101,
        last_compaction_step=1,
    )
    assert set(reasons) == {"token_threshold", "utilization", "agent_steps", "tool_output_volume"}


def test_trigger_does_not_fire_below_thresholds() -> None:
    assert (
        CompactionTriggers(every_steps=10).reasons(
            state=state(1),
            prompt_tokens=1,
            usable_tokens=1000,
            accumulated_tool_tokens=1,
            last_compaction_step=0,
        )
        == []
    )
