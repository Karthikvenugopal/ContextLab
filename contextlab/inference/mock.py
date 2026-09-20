"""Deterministic CPU-only inference backends for integration tests and demos."""

from __future__ import annotations

from collections import deque

from contextlab.agent.models import Message
from contextlab.inference.client import InferenceError, InferenceResponse, Usage


class ScriptedInference:
    """Returns model decisions in order while reporting realistic accounting fields."""

    def __init__(self, responses: list[str], *, model: str = "contextlab-mock") -> None:
        self.responses = deque(responses)
        self.model = model
        self.requests: list[dict[str, object]] = []

    async def complete(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse:
        if not self.responses:
            raise InferenceError("scripted inference exhausted")
        content = self.responses.popleft()
        prompt_tokens = sum(max(1, len(item.content.encode()) // 4) for item in messages)
        generated_tokens = max(1, len(content.encode()) // 4)
        self.requests.append(
            {"request_kind": request_kind, "seed": seed, "prompt_tokens": prompt_tokens}
        )
        return InferenceResponse(
            content=content,
            model=self.model,
            request_kind=request_kind,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=generated_tokens,
                total_tokens=prompt_tokens + generated_tokens,
            ),
        )

    async def stream(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse:
        return await self.complete(messages, request_kind=request_kind, seed=seed)
