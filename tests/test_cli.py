from pathlib import Path

from typer.testing import CliRunner

from contextlab.cli.app import app


runner = CliRunner()


def test_tasks_list_exposes_fixture_suite() -> None:
    result = runner.invoke(app, ["tasks", "list"])
    assert result.exit_code == 0
    assert "localized-divide-zero" in result.stdout
    assert "multi_file_bug" in result.stdout


def test_trace_summarize_reports_empty_trace(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("")
    result = runner.invoke(app, ["trace", "summarize", "--input", str(trace)])
    assert result.exit_code == 0
    assert '"inference_requests": 0' in result.stdout


def test_doctor_reports_endpoint_failure_without_failing_python_check() -> None:
    result = runner.invoke(app, ["doctor", "--endpoint", "http://127.0.0.1:1/v1"])
    assert result.exit_code == 0
    assert '"endpoint_available": false' in result.stdout
