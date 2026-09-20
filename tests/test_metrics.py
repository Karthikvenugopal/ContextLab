from contextlab.agent.models import AgentEvent, EventKind
from contextlab.instrumentation.metrics import metrics_from_events


def event(kind: EventKind, step: int, **payload: object) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        step=step,
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="p",
        payload=payload,
    )


def test_aggregates_primary_and_context_overhead_metrics() -> None:
    metrics = metrics_from_events(
        [
            event(EventKind.INFERENCE_RESPONSE, 0, prompt_tokens=10, generated_tokens=2, latency_seconds=0.2),
            event(EventKind.INFERENCE_RESPONSE, 1, prompt_tokens=20, generated_tokens=3, latency_seconds=0.3),
            event(EventKind.CONTEXT_PREPARED, 1, utilization=0.75),
            event(EventKind.COMPACTION, 1, input_tokens=30, output_tokens=5, latency_seconds=0.1),
            event(EventKind.RETRIEVAL, 1, tokens=7, latency_seconds=0.02),
        ],
        official_success=True,
        test_pass_rate=1.0,
    )
    assert metrics.total_prompt_tokens == 30
    assert metrics.total_generated_tokens == 5
    assert metrics.compaction_input_tokens + metrics.compaction_output_tokens == 35
    assert metrics.peak_context_utilization == 0.75
