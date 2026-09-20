from contextlab.agent.models import AgentEvent, EventKind
from contextlab.analysis.context_behavior import analyze_context_behavior


def event(kind: EventKind, **payload: object) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        step=1,
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="p",
        payload=payload,
    )


def test_summarizes_retention_recovery_and_compaction() -> None:
    audit = {
        "retained_message_indices": [0, 1],
        "removed_message_indices": [2],
        "compressed_message_indices": [2],
        "recovered_source_ids": ["repo:a"],
        "metadata": {
            "reductions": [
                {"original_tokens": 100, "retained_tokens": 20, "discarded_tokens": 80}
            ]
        },
    }
    behavior = analyze_context_behavior(
        [
            event(EventKind.CONTEXT_PREPARED, audit=audit),
            event(EventKind.RETRIEVAL, query="parser", tokens=12, source_ids=["repo:a"]),
            event(EventKind.RETRIEVAL, query="parser", tokens=12, source_ids=["repo:a"]),
            event(EventKind.COMPACTION, input_tokens=50, output_tokens=10),
        ]
    )
    assert behavior.discarded_tool_tokens == 80
    assert behavior.repeated_retrievals == 1
    assert behavior.recovered_sources["repo:a"] == 3
    assert behavior.compaction_input_tokens + behavior.compaction_output_tokens == 60
