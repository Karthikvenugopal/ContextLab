"""Common context policy contract and audit structures."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from contextlab.agent.models import AgentEvent, AgentState, Message


class ContextAudit(BaseModel):
    retained_message_indices: list[int] = Field(default_factory=list)
    removed_message_indices: list[int] = Field(default_factory=list)
    compressed_message_indices: list[int] = Field(default_factory=list)
    recovered_source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreparedContext(BaseModel):
    messages: list[Message]
    estimated_tokens: int
    usable_tokens: int
    audit: ContextAudit = Field(default_factory=ContextAudit)


class RecoveryItem(BaseModel):
    source_id: str
    content: str
    score: float
    tokens: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecoveryResult(BaseModel):
    query: str
    items: list[RecoveryItem] = Field(default_factory=list)
    tokens: int = 0
    latency_seconds: float = 0.0


class TokenBudgetLike(Protocol):
    @property
    def usable_prompt_tokens(self) -> int: ...

    def count_messages(self, messages: list[Message]) -> int: ...


class ContextPolicy(Protocol):
    policy_id: str

    async def prepare_context(
        self, state: AgentState, budget: TokenBudgetLike
    ) -> PreparedContext: ...

    async def observe(self, event: AgentEvent, state: AgentState) -> None: ...

    async def recover(
        self, query: str, state: AgentState, budget: TokenBudgetLike
    ) -> RecoveryResult: ...


class BaseContextPolicy:
    policy_id = "base"

    async def observe(self, event: AgentEvent, state: AgentState) -> None:
        del event, state

    async def recover(
        self, query: str, state: AgentState, budget: TokenBudgetLike
    ) -> RecoveryResult:
        del state, budget
        return RecoveryResult(query=query)
