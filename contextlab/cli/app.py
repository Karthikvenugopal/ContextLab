"""Typer CLI for agents, controlled benchmarks, traces, and diagnostics."""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Annotated

import httpx
import typer

from contextlab import __version__
from contextlab.analysis import analyze_context_behavior, analyze_repeated_access
from contextlab.analysis.report import generate_report
from contextlab.benchmarking.config import (
    ContextBudgetVariant,
    ExperimentConfig,
    load_experiment_config,
)
from contextlab.benchmarking.runner import ExperimentRunner
from contextlab.config import load_agent_config
from contextlab.instrumentation.events import load_events
from contextlab.instrumentation.metrics import metrics_from_events
from contextlab.tasks import load_task

app = typer.Typer(help="Context management experiments for autonomous coding agents.")
tasks_app = typer.Typer(help="Inspect coding-task manifests.")
agent_app = typer.Typer(help="Run a coding agent on one task.")
benchmark_app = typer.Typer(help="Run and report controlled experiments.")
trace_app = typer.Typer(help="Inspect structured run traces.")
app.add_typer(tasks_app, name="tasks")
app.add_typer(agent_app, name="agent")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(trace_app, name="trace")


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def doctor(
    endpoint: Annotated[
        str, typer.Option(help="OpenAI-compatible /v1 endpoint.")
    ] = "http://127.0.0.1:8000/v1",
) -> None:
    """Diagnose Python, container, GPU, and inference endpoint availability."""
    diagnostics: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_supported": sys.version_info >= (3, 11),
        "platform": platform.platform(),
        "docker": shutil.which("docker") is not None,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
        "endpoint": endpoint,
    }
    try:
        response = httpx.get(endpoint.rstrip("/") + "/models", timeout=2)
        diagnostics["endpoint_status"] = response.status_code
        diagnostics["endpoint_available"] = response.is_success
    except httpx.HTTPError as error:
        diagnostics["endpoint_available"] = False
        diagnostics["endpoint_error"] = str(error)
    typer.echo(json.dumps(diagnostics, indent=2))
    if not diagnostics["python_supported"]:
        raise typer.Exit(1)


@tasks_app.command("list")
def list_tasks(
    directory: Annotated[Path, typer.Option(help="Directory containing task YAML files.")] = Path(
        "benchmarks/tasks"
    ),
) -> None:
    """List strict task manifests and their categories."""
    for path in sorted(directory.glob("*.yaml")):
        task = load_task(path)
        typer.echo(f"{task.id}\t{task.category}\t{path}")


@agent_app.command("run")
def agent_run(
    task: Annotated[Path, typer.Option(exists=True, help="Task manifest.")],
    policy: Annotated[str, typer.Option(help="Context policy identifier.")],
    config: Annotated[Path, typer.Option(exists=True, help="Agent YAML configuration.")],
    seed: Annotated[int, typer.Option()] = 0,
    context_window: Annotated[int | None, typer.Option()] = None,
    endpoint: Annotated[str | None, typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = Path("results"),
    mock: Annotated[bool, typer.Option(help="Use deterministic CPU mock inference.")] = False,
) -> None:
    """Run one task-policy pair in a clean workspace and evaluate it."""
    agent_config = load_agent_config(config)
    if endpoint:
        agent_config.model.endpoint = endpoint
    window = context_window or agent_config.budget.context_window
    experiment = ExperimentConfig(
        experiment_id=f"single-{load_task(task).id}-{policy}",
        tasks=[task.resolve()],
        policies=[policy],
        budgets=[
            ContextBudgetVariant(
                name="medium",
                context_window=window,
                reserved_output_tokens=agent_config.budget.reserved_output_tokens,
            )
        ],
        seeds=[seed],
        model=agent_config.model,
        output_directory=output.resolve(),
        mock=mock,
    )
    result = asyncio.run(ExperimentRunner(experiment).run())
    typer.echo(str(result))


@benchmark_app.command("run")
def benchmark_run(
    config: Annotated[Path, typer.Option(exists=True)],
) -> None:
    """Execute every controlled comparison in an experiment matrix."""
    result = asyncio.run(ExperimentRunner(load_experiment_config(config)).run())
    typer.echo(str(result))


@benchmark_app.command("report")
def benchmark_report(
    results: Annotated[Path, typer.Option(exists=True)],
    no_charts: Annotated[bool, typer.Option(help="Skip optional matplotlib charts.")] = False,
) -> None:
    """Aggregate actual runs and generate CSV, JSON, Markdown, and charts."""
    typer.echo(str(generate_report(results, charts=not no_charts)))


@trace_app.command("summarize")
def trace_summarize(
    input: Annotated[Path, typer.Option(exists=True)],  # noqa: A002
) -> None:
    """Summarize metrics and context behavior from one JSONL trace."""
    events = load_events(input)
    summary = {
        "metrics": metrics_from_events(events).model_dump(),
        "repeated_access": analyze_repeated_access(events).model_dump(),
        "context_behavior": analyze_context_behavior(events).model_dump(),
    }
    typer.echo(json.dumps(summary, indent=2, default=dict))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
