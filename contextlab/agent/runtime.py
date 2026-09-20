"""Policy-independent iterative coding-agent loop."""

from __future__ import annotations

import time
from typing import Any, Protocol

from contextlab.agent.models import AgentState, EventKind, Message
from contextlab.agent.protocol import TOOL_PROTOCOL, MalformedDecision, parse_decision
from contextlab.config import AgentConfig
from contextlab.inference.accounting import InferenceTotals
from contextlab.inference.client import InferenceClient
from contextlab.tools.execution import MutationTools
from contextlab.tools.models import ToolCall, ToolResult
from contextlab.tools.repository import RepositoryTools


class Prepared(Protocol):
    messages: list[Message]
    estimated_tokens: int


class Policy(Protocol):
    async def prepare_context(self, state: AgentState, budget: Any) -> Prepared: ...
    async def observe(self, event: Any, state: AgentState) -> None: ...


class CodingAgent:
    def __init__(
        self,
        *,
        config: AgentConfig,
        inference: InferenceClient,
        policy: Policy,
        repository_tools: RepositoryTools,
        mutation_tools: MutationTools,
        budget: Any,
    ) -> None:
        self.config = config
        self.inference = inference
        self.policy = policy
        self.repository_tools = repository_tools
        self.mutation_tools = mutation_tools
        self.budget = budget
        self.inference_totals = InferenceTotals()
        self.tool_calls = 0

    async def run(self, state: AgentState, *, seed: int = 0) -> AgentState:
        state.canonical_messages.extend(
            [
                Message(role="system", content=f"{state.system_instruction}\n\n{TOOL_PROTOCOL}"),
                Message(role="user", content=state.task_instruction),
            ]
        )
        started = time.monotonic()
        for step in range(self.config.limits.max_steps):
            state.step = step
            prepared = await self.policy.prepare_context(state, self.budget)
            event = state.emit(EventKind.CONTEXT_PREPARED, estimated_tokens=prepared.estimated_tokens)
            await self.policy.observe(event, state)
            response = await self.inference.complete(prepared.messages, seed=seed)
            self.inference_totals.record(response)
            state.emit(
                EventKind.INFERENCE_RESPONSE,
                prompt_tokens=response.usage.prompt_tokens,
                generated_tokens=response.usage.completion_tokens,
                latency_seconds=response.latency_seconds,
                request_kind=response.request_kind,
            )
            state.canonical_messages.append(Message(role="assistant", content=response.content))
            try:
                decision = parse_decision(response.content)
            except MalformedDecision as error:
                state.canonical_messages.append(Message(role="user", content=str(error)))
                continue
            if decision.action == "finish":
                state.completed = True
                state.emit(EventKind.STATUS, status="completed", summary=decision.summary)
                return state
            assert decision.tool is not None
            result = self._execute(decision.tool)
            self.tool_calls += 1
            tool_event = state.emit(EventKind.TOOL_RESULT, result=result.model_dump())
            await self.policy.observe(tool_event, state)
            state.canonical_messages.append(
                Message(role="tool", name=result.name, tool_call_id=result.call_id, content=result.content)
            )
            if result.name == "write_file" and result.ok and "path" in result.metadata:
                state.files_modified.add(str(result.metadata["path"]))
            if time.monotonic() - started > self.config.limits.max_wall_seconds:
                break
        state.failure = "execution_limit"
        state.emit(EventKind.STATUS, status="failed", reason=state.failure)
        return state

    def _execute(self, call: ToolCall) -> ToolResult:
        dispatch = {
            "list_files": self.repository_tools.list_files,
            "read_file": self.repository_tools.read_file,
            "search": self.repository_tools.search,
            "write_file": self.mutation_tools.write_file,
            "run_command": self.mutation_tools.run_command,
        }
        handler = dispatch.get(call.name)
        if handler is None:
            return ToolResult(call_id=call.id, name=call.name, content="tool unavailable", ok=False)
        return handler(call)
