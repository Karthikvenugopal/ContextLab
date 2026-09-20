import json
from pathlib import Path

from contextlab.agent.models import AgentEvent, EventKind
from contextlab.instrumentation.events import EventLogger, load_events


def make_event() -> AgentEvent:
    return AgentEvent(
        kind=EventKind.STATUS,
        step=2,
        experiment_id="exp",
        run_id="run",
        task_id="task",
        policy_id="full-history",
        payload={"content": "private source", "status": "running"},
    )


def test_round_trips_jsonl_events(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    EventLogger(path).write(make_event())
    assert load_events(path)[0].step == 2


def test_redacts_content_but_preserves_metadata(tmp_path: Path) -> None:
    path = tmp_path / "redacted.jsonl"
    EventLogger(path, redact=True).write(make_event())
    record = json.loads(path.read_text())
    assert record["payload"]["content"]["redacted"] is True
    assert record["payload"]["status"] == "running"
