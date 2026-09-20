"""Generate machine-readable aggregates, Markdown, and comparison charts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from contextlab.analysis.aggregate import aggregate_experiment
from contextlab.instrumentation.events import load_events
from contextlab.instrumentation.metrics import metrics_from_events


def generate_report(root: Path, *, charts: bool = True) -> Path:
    rows = aggregate_experiment(root)
    aggregate_json = root / "aggregate.json"
    aggregate_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    aggregate_csv = root / "aggregate.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with aggregate_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    detailed = _detailed_runs(root)
    if charts:
        _generate_charts(root / "charts", rows, detailed)
    config = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    mode = "CPU deterministic mock" if config.get("mock") else "real inference"
    report = root / "report.md"
    lines = [
        f"# ContextLab experiment: {config.get('experiment_id', root.name)}",
        "",
        f"Execution mode: **{mode}**. Sample sizes and failures are reported per condition.",
        "Mock results validate orchestration and must not be interpreted as model performance." if config.get("mock") else "Results reflect the configured inference endpoint and hardware.",
        "",
        "| Task | Budget | Policy | n | Success | Failed | Success rate | Median tokens | Median latency (s) | Median tool calls |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task_id} | {budget_name} | {policy_id} | {sample_size} | "
            "{successful_runs} | {failed_runs} | {success_rate:.3f} | "
            "{total_tokens_median:.1f} | {wall_seconds_median:.3f} | {tool_calls_median:.1f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation boundaries",
            "",
            "These tables are observations from collected run artifacts. A shorter prompt can still produce higher total workload when it causes extra model requests, retrieval, compaction, or repeated repository exploration. Causal claims require repeated real-model trials and uncertainty analysis; ContextLab does not infer causality from a repeated tool call alone.",
            "",
            "Successful and failed runs remain separate in `aggregate.json`; token medians for successful runs are reported independently when available.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _detailed_runs(root: Path) -> list[dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        events = load_events(root / run["trace"])
        metrics = metrics_from_events(events)
        detailed.append(run | {"metrics": metrics.model_dump()})
    return detailed


def _generate_charts(
    directory: Path, rows: list[dict[str, Any]], detailed: list[dict[str, Any]]
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("chart generation requires `pip install contextlab[analysis]`") from error
    directory.mkdir(parents=True, exist_ok=True)
    policies = sorted({row["policy_id"] for row in rows})

    def bar(filename: str, title: str, ylabel: str, values: dict[str, float]) -> None:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(policies, [values.get(policy, 0.0) for policy in policies])
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=20)
        figure.tight_layout()
        figure.savefig(directory / filename, dpi=140)
        plt.close(figure)

    def average(key: str, nested: str | None = None) -> dict[str, float]:
        output: dict[str, float] = {}
        for policy in policies:
            selected = [item for item in detailed if item["policy_id"] == policy]
            values = [
                float(item[nested][key] if nested else item[key])
                for item in selected
            ]
            output[policy] = sum(values) / len(values) if values else 0.0
        return output

    bar("task-success.png", "Official task success", "success rate", {
        policy: sum(float(row["success_rate"]) * int(row["sample_size"]) for row in rows if row["policy_id"] == policy)
        / max(1, sum(int(row["sample_size"]) for row in rows if row["policy_id"] == policy))
        for policy in policies
    })
    bar("total-inference-tokens.png", "Total inference workload", "tokens", average("total_tokens", "inference"))
    bar("task-latency.png", "End-to-end task latency", "seconds", average("wall_seconds"))
    bar("repository-tool-calls.png", "Repository tool calls", "calls", average("tool_calls"))
    bar("retrieval-overhead.png", "Retrieved context volume", "tokens", average("retrieval_tokens", "metrics"))
    bar("compaction-overhead.png", "Compaction inference input", "tokens", average("compaction_input_tokens", "metrics"))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for item in detailed:
        growth = item["metrics"]["prompt_growth_by_step"]
        if growth:
            steps = sorted((int(step), value) for step, value in growth.items())
            axis.plot([step for step, _ in steps], [value for _, value in steps], alpha=0.6, label=item["policy_id"])
    axis.set(title="Prompt-token growth by agent step", xlabel="agent step", ylabel="prompt tokens")
    if detailed:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=False))
        axis.legend(unique.values(), unique.keys())
    figure.tight_layout()
    figure.savefig(directory / "prompt-token-growth.png", dpi=140)
    plt.close(figure)
