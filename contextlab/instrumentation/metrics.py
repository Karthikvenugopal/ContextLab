"""Request-, tool-, context-, and task-level trace metrics."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from contextlab.agent.models import AgentEvent, EventKind


class RunMetrics(BaseModel):
    official_success: bool = False
    test_pass_rate: float = 0.0
    context_window_failures: int = 0
    execution_failures: int = 0
    timeouts: int = 0
    total_prompt_tokens: int = 0
    total_generated_tokens: int = 0
    inference_requests: int = 0
    peak_prompt_tokens: int = 0
    peak_context_utilization: float = 0.0
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0
    retrieval_tokens: int = 0
    inference_latency_seconds: float = 0.0
    tool_latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    compaction_latency_seconds: float = 0.0
    repository_tool_calls: int = 0
    prompt_growth_by_step: dict[int, int] = Field(default_factory=dict)
    time_to_first_token_seconds: list[float] = Field(default_factory=list)


def metrics_from_events(
    events: list[AgentEvent], *, official_success: bool = False, test_pass_rate: float = 0.0
) -> RunMetrics:
    metrics = RunMetrics(official_success=official_success, test_pass_rate=test_pass_rate)
    prompt_by_step: defaultdict[int, int] = defaultdict(int)
    for event in events:
        payload = event.payload
        if event.kind == EventKind.INFERENCE_RESPONSE:
            prompt = int(payload.get("prompt_tokens", 0))
            generated = int(payload.get("generated_tokens", 0))
            metrics.total_prompt_tokens += prompt
            metrics.total_generated_tokens += generated
            metrics.inference_requests += 1
            metrics.peak_prompt_tokens = max(metrics.peak_prompt_tokens, prompt)
            metrics.inference_latency_seconds += float(payload.get("latency_seconds", 0))
            ttft = payload.get("time_to_first_token_seconds")
            if ttft is not None:
                metrics.time_to_first_token_seconds.append(float(ttft))
            prompt_by_step[event.step] = prompt
        elif event.kind == EventKind.CONTEXT_PREPARED:
            metrics.peak_context_utilization = max(
                metrics.peak_context_utilization, float(payload.get("utilization", 0))
            )
        elif event.kind == EventKind.CONTEXT_FAILURE:
            metrics.context_window_failures += 1
        elif event.kind == EventKind.TOOL_RESULT:
            result = payload.get("result", {})
            metrics.repository_tool_calls += 1
            if isinstance(result, dict):
                metrics.tool_latency_seconds += float(result.get("duration_seconds", 0))
        elif event.kind == EventKind.RETRIEVAL:
            metrics.retrieval_tokens += int(payload.get("tokens", 0))
            metrics.retrieval_latency_seconds += float(payload.get("latency_seconds", 0))
        elif event.kind == EventKind.COMPACTION:
            metrics.compaction_input_tokens += int(payload.get("input_tokens", 0))
            metrics.compaction_output_tokens += int(payload.get("output_tokens", 0))
            metrics.compaction_latency_seconds += float(payload.get("latency_seconds", 0))
        elif event.kind == EventKind.STATUS and payload.get("status") == "failed":
            metrics.execution_failures += 1
            if payload.get("reason") == "wall_clock":
                metrics.timeouts += 1
    metrics.prompt_growth_by_step = dict(prompt_by_step)
    return metrics
