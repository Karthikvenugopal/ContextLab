from contextlab.evaluation import evaluate_task
from contextlab.tasks import load_task


def test_official_tests_fail_unmodified_buggy_fixture() -> None:
    task = load_task(__import__("pathlib").Path("benchmarks/tasks/localized_bug.yaml"))
    result = evaluate_task(task, task.repository)
    assert not result.success
    assert result.failed == 1
    assert result.passed == 1
    assert result.pass_rate == 0.5
