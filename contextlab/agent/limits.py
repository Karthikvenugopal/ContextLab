"""Central enforcement for bounded autonomous execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from contextlab.config import ExecutionLimits


class LimitExceeded(RuntimeError):
    def __init__(self, limit: str) -> None:
        super().__init__(f"execution limit exceeded: {limit}")
        self.limit = limit


@dataclass
class LimitTracker:
    limits: ExecutionLimits
    started: float = field(default_factory=time.monotonic)
    model_requests: int = 0
    tool_calls: int = 0
    generated_tokens: int = 0

    def before_model(self) -> None:
        self.check_time()
        if self.model_requests >= self.limits.max_model_requests:
            raise LimitExceeded("model_requests")
        self.model_requests += 1

    def after_model(self, generated_tokens: int) -> None:
        self.generated_tokens += generated_tokens
        if self.generated_tokens > self.limits.max_generated_tokens:
            raise LimitExceeded("generated_tokens")

    def before_tool(self) -> None:
        self.check_time()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise LimitExceeded("tool_calls")
        self.tool_calls += 1

    def check_time(self) -> None:
        if time.monotonic() - self.started > self.limits.max_wall_seconds:
            raise LimitExceeded("wall_clock")
