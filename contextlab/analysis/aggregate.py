"""Descriptive statistics for controlled policy comparisons."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from contextlab.instrumentation.events import load_events
from contextlab.instrumentation.metrics import metrics_from_events


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0}
    return {
        "median": statistics.median(values),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
    }


def aggregate_experiment(root: Path) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((root / "runs").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        trace = load_events(root / run["trace"])
        evaluation = run["official_evaluation"]
        metrics = metrics_from_events(
            trace,
            official_success=evaluation["success"],
            test_pass_rate=evaluation["pass_rate"],
        )
        run["derived_metrics"] = metrics.model_dump()
        grouped[(run["task_id"], run["budget_name"], run["policy_id"])].append(run)
    rows: list[dict[str, Any]] = []
    for (task_id, budget_name, policy_id), runs in sorted(grouped.items()):
        successes = [run for run in runs if run["official_evaluation"]["success"]]
        token_values = [float(run["inference"]["total_tokens"]) for run in runs]
        latency_values = [float(run["wall_seconds"]) for run in runs]
        tool_values = [float(run["tool_calls"]) for run in runs]
        row: dict[str, Any] = {
            "task_id": task_id,
            "budget_name": budget_name,
            "policy_id": policy_id,
            "sample_size": len(runs),
            "successful_runs": len(successes),
            "failed_runs": len(runs) - len(successes),
            "success_rate": len(successes) / len(runs),
            "mean_test_pass_rate": statistics.fmean(
                run["official_evaluation"]["pass_rate"] for run in runs
            ),
        }
        for prefix, values in (
            ("total_tokens", token_values),
            ("wall_seconds", latency_values),
            ("tool_calls", tool_values),
        ):
            row.update({f"{prefix}_{key}": value for key, value in describe(values).items()})
        if successes:
            row["successful_total_tokens_median"] = statistics.median(
                float(run["inference"]["total_tokens"]) for run in successes
            )
        else:
            row["successful_total_tokens_median"] = None
        rows.append(row)
    return rows
