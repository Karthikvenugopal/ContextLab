"""Central token budget manager used identically by every policy."""

from __future__ import annotations

import math
from typing import Protocol

from pydantic import BaseModel

from contextlab.agent.models import Message
from contextlab.config import BudgetConfig


class Encoder(Protocol):
    def encode(self, text: str, **kwargs: object) -> list[int]: ...


class ContextOverflow(RuntimeError):
    def __init__(self, estimated: int, usable: int) -> None:
        super().__init__(f"context requires {estimated} tokens but usable budget is {usable}")
        self.estimated = estimated
        self.usable = usable


class TokenCountRecord(BaseModel):
    client_estimate: int
    server_reported: int | None = None
    tokenizer: str
    model: str


class TokenCounter:
    """Tokenizer adapter with an explicit conservative CPU-only fallback."""

    def __init__(self, encoder: Encoder | None = None, *, name: str = "utf8-byte-fallback") -> None:
        self.encoder = encoder
        self.name = name

    def count(self, text: str) -> int:
        if self.encoder is not None:
            return len(self.encoder.encode(text, add_special_tokens=False))
        # Four UTF-8 bytes/token is common for English code; ceil avoids returning zero.
        return max(1, math.ceil(len(text.encode("utf-8")) / 4)) if text else 0


class TokenBudget:
    def __init__(
        self,
        config: BudgetConfig,
        *,
        counter: TokenCounter | None = None,
        model: str = "unknown",
    ) -> None:
        self.config = config
        self.counter = counter or TokenCounter()
        self.model = model

    @property
    def usable_prompt_tokens(self) -> int:
        return (
            self.config.context_window
            - self.config.reserved_output_tokens
            - self.config.safety_margin_tokens
        )

    def count_messages(self, messages: list[Message]) -> int:
        # Explicit role/name overhead approximates common chat templates.
        return sum(4 + self.counter.count(message.content) for message in messages) + 3

    def count_text(self, text: str) -> int:
        return self.counter.count(text)

    def ensure_fits(self, messages: list[Message]) -> int:
        estimated = self.count_messages(messages)
        if estimated > self.usable_prompt_tokens:
            raise ContextOverflow(estimated, self.usable_prompt_tokens)
        return estimated

    def record(self, messages: list[Message], server_reported: int | None = None) -> TokenCountRecord:
        return TokenCountRecord(
            client_estimate=self.count_messages(messages),
            server_reported=server_reported,
            tokenizer=self.counter.name,
            model=self.model,
        )
