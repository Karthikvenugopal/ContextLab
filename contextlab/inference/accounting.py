"""Separate client estimates from server-reported token usage."""

from pydantic import BaseModel

from contextlab.inference.client import InferenceResponse


class InferenceTotals(BaseModel):
    requests: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    agent_tokens: int = 0
    auxiliary_tokens: int = 0
    latency_seconds: float = 0.0

    def record(self, response: InferenceResponse) -> None:
        self.requests += 1
        self.prompt_tokens += response.usage.prompt_tokens
        self.generated_tokens += response.usage.completion_tokens
        if response.request_kind == "agent":
            self.agent_tokens += response.usage.total_tokens
        else:
            self.auxiliary_tokens += response.usage.total_tokens
        self.latency_seconds += response.latency_seconds

    @property
    def total_tokens(self) -> int:
        return self.agent_tokens + self.auxiliary_tokens
