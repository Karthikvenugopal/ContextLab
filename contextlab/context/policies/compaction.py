"""Structured context compaction with configurable trigger rules."""

from __future__ import annotations

import time
from typing import Protocol

from pydantic import BaseModel, Field

from contextlab.agent.models import AgentEvent, AgentState, EventKind, Message
from contextlab.context.base import (
    BaseContextPolicy,
    ContextAudit,
    PreparedContext,
    TokenBudgetLike,
)
from contextlab.context.budgeting import ContextOverflow
from contextlab.inference.client import InferenceClient, InferenceResponse


class CompactionTriggers(BaseModel):
    token_threshold: int | None = Field(default=None, gt=0)
    utilization_threshold: float | None = Field(default=0.75, gt=0, le=1)
    every_steps: int | None = Field(default=None, gt=0)
    tool_output_tokens: int | None = Field(default=8000, gt=0)

    def reasons(
        self,
        *,
        state: AgentState,
        prompt_tokens: int,
        usable_tokens: int,
        accumulated_tool_tokens: int,
        last_compaction_step: int,
    ) -> list[str]:
        reasons: list[str] = []
        if self.token_threshold is not None and prompt_tokens >= self.token_threshold:
            reasons.append("token_threshold")
        if (
            self.utilization_threshold is not None
            and prompt_tokens / max(1, usable_tokens) >= self.utilization_threshold
        ):
            reasons.append("utilization")
        if self.every_steps is not None and state.step - last_compaction_step >= self.every_steps:
            reasons.append("agent_steps")
        if (
            self.tool_output_tokens is not None
            and accumulated_tool_tokens >= self.tool_output_tokens
        ):
            reasons.append("tool_output_volume")
        return reasons


class CompactionResult(BaseModel):
    summary: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    inference_response: InferenceResponse | None = None


class Compactor(Protocol):
    async def compact(self, messages: list[Message], state: AgentState) -> CompactionResult: ...


class DeterministicCompactor:
    async def compact(self, messages: list[Message], state: AgentState) -> CompactionResult:
        started = time.perf_counter()
        findings: list[str] = []
        unresolved: list[str] = []
        tests: list[str] = []
        for message in messages:
            text = " ".join(message.content.split())
            if not text:
                continue
            excerpt = text[:240]
            if message.role == "tool" and message.name == "run_command":
                tests.append(excerpt)
            elif any(word in text.lower() for word in ("error", "failed", "todo", "unresolved")):
                unresolved.append(excerpt)
            else:
                findings.append(excerpt)
        summary = "\n".join(
            [
                "# Compacted execution context",
                f"Original task: {state.task_instruction}",
                f"Current objective: {state.objective}",
                f"Files modified: {', '.join(sorted(state.files_modified)) or 'none'}",
                "Repository findings: " + (" | ".join(findings[-8:]) or "none"),
                "Unresolved errors: " + (" | ".join(unresolved[-5:]) or "none"),
                "Test results: " + (" | ".join(tests[-5:]) or "none"),
                "Completed actions: represented by the canonical trace; continue from "
                "recent messages.",
                "Remaining work: satisfy the original task and validate with approved tests.",
            ]
        )
        return CompactionResult(
            summary=summary,
            input_tokens=sum(max(1, len(item.content.encode()) // 4) for item in messages),
            output_tokens=max(1, len(summary.encode()) // 4),
            latency_seconds=time.perf_counter() - started,
        )


class ModelCompactor:
    def __init__(self, inference: InferenceClient) -> None:
        self.inference = inference

    async def compact(self, messages: list[Message], state: AgentState) -> CompactionResult:
        started = time.perf_counter()
        prompt = Message(
            role="user",
            content=(
                "Create a structured coding-session summary preserving task requirements, current "
                "objective, repository findings, files modified, unresolved errors, tests, "
                "completed "
                "actions, and remaining work.\n\n"
                + "\n\n".join(f"{item.role}/{item.name or ''}: {item.content}" for item in messages)
            ),
        )
        response = await self.inference.complete([prompt], request_kind="compaction")
        return CompactionResult(
            summary=response.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            latency_seconds=time.perf_counter() - started,
            inference_response=response,
        )


class CompactionPolicy(BaseContextPolicy):
    policy_id = "compaction"

    def __init__(
        self,
        *,
        compactor: Compactor | None = None,
        triggers: CompactionTriggers | None = None,
        recent_messages: int = 4,
    ) -> None:
        self.compactor = compactor or DeterministicCompactor()
        self.triggers = triggers or CompactionTriggers()
        self.recent_messages = recent_messages
        self.summary: str | None = None
        self.compacted_until = 2
        self.last_compaction_step = 0
        self.records: list[dict[str, object]] = []
        self._reported_records = 0
        self.precompaction_paths: set[str] = set()
        self.revisits: list[dict[str, object]] = []

    def drain_new_records(self) -> list[dict[str, object]]:
        records = self.records[self._reported_records :]
        self._reported_records = len(self.records)
        return records

    async def observe(self, event: AgentEvent, state: AgentState) -> None:
        del state
        if event.kind != EventKind.TOOL_RESULT or not self.records:
            return
        result = event.payload.get("result", {})
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        path = metadata.get("path") if isinstance(metadata, dict) else None
        if path and str(path) in self.precompaction_paths:
            self.revisits.append(
                {"path": str(path), "step": event.step, "tool_call_id": result.get("call_id")}
            )

    async def prepare_context(self, state: AgentState, budget: TokenBudgetLike) -> PreparedContext:
        canonical_tokens = budget.count_messages(state.canonical_messages)
        tool_tokens = sum(
            max(1, len(message.content.encode()) // 4)
            for message in state.canonical_messages[self.compacted_until :]
            if message.role == "tool"
        )
        reasons = self.triggers.reasons(
            state=state,
            prompt_tokens=canonical_tokens,
            usable_tokens=budget.usable_prompt_tokens,
            accumulated_tool_tokens=tool_tokens,
            last_compaction_step=self.last_compaction_step,
        )
        compactable_end = max(
            self.compacted_until, len(state.canonical_messages) - self.recent_messages
        )
        compressed: list[int] = []
        if reasons and compactable_end > self.compacted_until:
            compressed = list(range(self.compacted_until, compactable_end))
            for event in state.events:
                if event.kind != EventKind.TOOL_RESULT:
                    continue
                result_payload = event.payload.get("result", {})
                metadata = (
                    result_payload.get("metadata", {}) if isinstance(result_payload, dict) else {}
                )
                if isinstance(metadata, dict) and metadata.get("path"):
                    self.precompaction_paths.add(str(metadata["path"]))
            compact_input = state.canonical_messages[self.compacted_until : compactable_end]
            if self.summary:
                compact_input = [Message(role="system", content=self.summary), *compact_input]
            result = await self.compactor.compact(compact_input, state)
            self.summary = result.summary
            self.compacted_until = compactable_end
            self.last_compaction_step = state.step
            self.records.append(
                {
                    "step": state.step,
                    "reasons": reasons,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_seconds": result.latency_seconds,
                    "inference_response": result.inference_response,
                }
            )
        retained_indices = [
            *range(min(2, len(state.canonical_messages))),
            *range(self.compacted_until, len(state.canonical_messages)),
        ]
        messages = [
            state.canonical_messages[index].model_copy(deep=True) for index in retained_indices
        ]
        if self.summary:
            messages.insert(
                2 if len(messages) >= 2 else len(messages),
                Message(role="system", content=self.summary),
            )
        estimated = budget.count_messages(messages)
        if estimated > budget.usable_prompt_tokens:
            raise ContextOverflow(estimated, budget.usable_prompt_tokens)
        return PreparedContext(
            messages=messages,
            estimated_tokens=estimated,
            usable_tokens=budget.usable_prompt_tokens,
            audit=ContextAudit(
                retained_message_indices=retained_indices,
                removed_message_indices=list(range(2, self.compacted_until)),
                compressed_message_indices=compressed,
                metadata={
                    "compaction_count": len(self.records),
                    "reasons": reasons,
                    "revisits": list(self.revisits),
                },
            ),
        )
