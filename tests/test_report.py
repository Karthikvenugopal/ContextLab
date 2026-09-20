import json
from pathlib import Path

from contextlab.agent.models import AgentEvent, EventKind
from contextlab.analysis.report import generate_report
from contextlab.instrumentation.events import EventLogger


def test_writes_json_csv_and_honest_markdown_report(tmp_path: Path) -> None:
    (tmp_path / "runs").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({"experiment_id": "demo", "mock": True}))
    trace = tmp_path / "traces" / "r.jsonl"
    EventLogger(trace).write(
        AgentEvent(
            kind=EventKind.STATUS,
            step=1,
            experiment_id="demo",
            run_id="r",
            task_id="t",
            policy_id="full-history",
            payload={"status": "completed"},
        )
    )
    run = {
        "task_id": "t",
        "budget_name": "short",
        "policy_id": "full-history",
        "official_evaluation": {"success": True, "pass_rate": 1.0},
        "inference": {"total_tokens": 10},
        "wall_seconds": 0.5,
        "tool_calls": 2,
        "trace": "traces/r.jsonl",
    }
    (tmp_path / "runs" / "r.json").write_text(json.dumps(run))
    report = generate_report(tmp_path, charts=False)
    assert (tmp_path / "aggregate.json").exists()
    assert (tmp_path / "aggregate.csv").exists()
    assert "must not be interpreted as model performance" in report.read_text()
