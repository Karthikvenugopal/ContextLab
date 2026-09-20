"""Canonical agent state and immutable execution events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventKind(str, Enum):
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT_PREPARED = "context_prepared"
    CONTEXT_FAILURE = "context_failure"
    COMPACTION = "compaction"
    RETRIEVAL = "retrieval"
    STATUS = "status"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: EventKind
    step: int
    experiment_id: str
    run_id: str
    task_id: str
    policy_id: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    experiment_id: str
    run_id: str
    task_id: str
    policy_id: str
    system_instruction: str
    task_instruction: str
    objective: str
    step: int = 0
    canonical_messages: list[Message] = Field(default_factory=list)
    events: list[AgentEvent] = Field(default_factory=list)
    files_modified: set[str] = Field(default_factory=set)
    completed: bool = False
    failure: str | None = None

    def emit(self, kind: EventKind, **payload: Any) -> AgentEvent:
        event = AgentEvent(
            kind=kind,
            step=self.step,
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            task_id=self.task_id,
            policy_id=self.policy_id,
            payload=payload,
        )
        self.events.append(event)
        return event
