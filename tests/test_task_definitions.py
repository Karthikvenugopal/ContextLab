from pathlib import Path

from contextlab.tasks import load_task


def test_benchmark_suite_covers_required_categories() -> None:
    tasks = [load_task(path) for path in Path("benchmarks/tasks").glob("*.yaml")]
    assert {task.category for task in tasks} == {
        "localized_bug",
        "multi_file_bug",
        "feature",
        "refactor",
        "exploration",
    }
    assert all(task.revision == "baseline" for task in tasks)
