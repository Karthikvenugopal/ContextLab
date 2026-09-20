"""Minimal asynchronous OpenAI-compatible chat client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from contextlab.agent.models import Message
from contextlab.config import ModelConfig


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class InferenceResponse(BaseModel):
    content: str
    usage: Usage = Field(default_factory=Usage)
    model: str
    latency_seconds: float = 0.0
    time_to_first_token_seconds: float | None = None
    inter_token_intervals_seconds: list[float] = Field(default_factory=list)
    request_kind: str = "agent"


class InferenceClient(Protocol):
    async def complete(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse: ...

    async def stream(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse: ...


class InferenceError(RuntimeError):
    """An inference request failed after bounded retries."""


class OpenAIClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.endpoint.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout_seconds,
        )

    async def complete(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "seed": seed,
            "stream": False,
        }
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.config.retries + 1):
            try:
                response = await self._client.post("chat/completions", json=payload)
                response.raise_for_status()
                body = response.json()
                return InferenceResponse(
                    content=body["choices"][0]["message"]["content"],
                    usage=Usage.model_validate(body.get("usage", {})),
                    model=body.get("model", self.config.model),
                    request_kind=request_kind,
                    latency_seconds=time.perf_counter() - started,
                )
            except (httpx.HTTPError, KeyError, ValueError) as error:
                last_error = error
                if attempt < self.config.retries:
                    await asyncio.sleep(0.1 * (2**attempt))
        raise InferenceError(f"inference failed: {last_error}")

    async def stream(
        self, messages: list[Message], *, request_kind: str = "agent", seed: int = 0
    ) -> InferenceResponse:
        """Consume OpenAI SSE chunks while preserving first-token timing."""
        payload = {
            "model": self.config.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first_token: float | None = None
        previous_chunk_at: float | None = None
        inter_token_intervals: list[float] = []
        pieces: list[str] = []
        usage = Usage()
        try:
            async with self._client.stream("POST", "chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    item = json.loads(line[6:])
                    if item.get("usage"):
                        usage = Usage.model_validate(item["usage"])
                    choices = item.get("choices", [])
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if content:
                        arrived_at = time.perf_counter()
                        if first_token is None:
                            first_token = arrived_at - started
                        elif previous_chunk_at is not None:
                            inter_token_intervals.append(arrived_at - previous_chunk_at)
                        previous_chunk_at = arrived_at
                        pieces.append(content)
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise InferenceError(f"streaming inference failed: {error}") from error
        return InferenceResponse(
            content="".join(pieces),
            usage=usage,
            model=self.config.model,
            latency_seconds=time.perf_counter() - started,
            time_to_first_token_seconds=first_token,
            inter_token_intervals_seconds=inter_token_intervals,
            request_kind=request_kind,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
