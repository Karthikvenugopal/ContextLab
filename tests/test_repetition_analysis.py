from contextlab.agent.models import AgentEvent, EventKind
from contextlab.analysis.repetition import analyze_repeated_access


def event(kind: EventKind, step: int, payload: dict) -> AgentEvent:
    return AgentEvent(
        kind=kind,
        step=step,
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="retrieval",
        payload=payload,
    )


def test_counts_repeats_without_overclaiming_cause() -> None:
    read = lambda call: {  # noqa: E731
        "result": {"name": "read_file", "call_id": call, "metadata": {"path": "src/a.py"}}
    }
    result = analyze_repeated_access(
        [event(EventKind.TOOL_RESULT, 0, read("a")), event(EventKind.TOOL_RESULT, 2, read("b"))]
    )
    assert result.repeated_file_reads == 1
    assert result.evidence[0].attribution == "unattributed"


def test_marks_only_supported_context_removal_association() -> None:
    search = lambda call: {  # noqa: E731
        "result": {"name": "search", "call_id": call, "metadata": {"query": "Parser"}}
    }
    result = analyze_repeated_access(
        [
            event(EventKind.TOOL_RESULT, 0, search("a")),
            event(EventKind.CONTEXT_PREPARED, 1, {"audit": {"removed_message_indices": [2]}}),
            event(EventKind.TOOL_RESULT, 2, search("b")),
        ]
    )
    assert result.repeated_searches == 1
    assert result.evidence[0].context_loss_supported
