from contextlab.agent.models import AgentEvent, EventKind
from contextlab.context.recovery import ObservationStore


def tool_event(content: str, call_id: str = "c1") -> AgentEvent:
    return AgentEvent(
        kind=EventKind.TOOL_RESULT,
        step=3,
        experiment_id="e",
        run_id="r",
        task_id="t",
        policy_id="retrieval",
        payload={
            "result": {
                "call_id": call_id,
                "name": "read_file",
                "content": content,
                "metadata": {"path": "src/parser.py"},
            }
        },
    )


def test_stores_and_recovers_complete_historical_observations() -> None:
    store = ObservationStore()
    document = store.add_event(tool_event("Parser raises InvalidHeader on empty input"))
    assert document and document.path == "src/parser.py"
    assert store.search("InvalidHeader parser")[0][0].content.startswith("Parser raises")


def test_deduplicates_same_tool_call_and_content() -> None:
    store = ObservationStore()
    store.add_event(tool_event("same"))
    store.add_event(tool_event("same"))
    assert len(store.documents) == 1
