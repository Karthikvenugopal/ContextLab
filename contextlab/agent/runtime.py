"""Policy-independent iterative coding-agent loop."""

from __future__ import annotations

from typing import Any, Protocol

from contextlab.agent.limits import LimitExceeded, LimitTracker
from contextlab.agent.models import AgentState, EventKind, Message
from contextlab.agent.protocol import TOOL_PROTOCOL, MalformedDecision, parse_decision
from contextlab.config import AgentConfig
from contextlab.context.budgeting import ContextOverflow
from contextlab.inference.accounting import InferenceTotals
from contextlab.inference.client import InferenceClient
from contextlab.tools.execution import MutationTools
from contextlab.tools.models import ToolCall, ToolResult
from contextlab.tools.repository import RepositoryTools


class Prepared(Protocol):
    messages: list[Message]
    estimated_tokens: int
    usable_tokens: int
    audit: Any


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
        tracker = LimitTracker(self.config.limits)
        try:
            for step in range(self.config.limits.max_steps):
                state.step = step
                prepared = await self.policy.prepare_context(state, self.budget)
                drain = getattr(self.policy, "drain_new_records", None)
                if callable(drain):
                    for record in drain():
                        auxiliary = record.get("inference_response")
                        if auxiliary is not None:
                            self.inference_totals.record(auxiliary)
                        state.emit(
                            EventKind.COMPACTION,
                            input_tokens=record["input_tokens"],
                            output_tokens=record["output_tokens"],
                            latency_seconds=record["latency_seconds"],
                            reasons=record["reasons"],
                            auxiliary_inference=auxiliary is not None,
                        )
                retrieval_query = prepared.audit.metadata.get("retrieval_query")
                if retrieval_query is not None:
                    state.emit(
                        EventKind.RETRIEVAL,
                        query=retrieval_query,
                        tokens=prepared.audit.metadata.get("retrieval_tokens", 0),
                        latency_seconds=prepared.audit.metadata.get(
                            "retrieval_latency_seconds", 0.0
                        ),
                        source_ids=prepared.audit.recovered_source_ids,
                        explicit=False,
                    )
                event = state.emit(
                    EventKind.CONTEXT_PREPARED,
                    estimated_tokens=prepared.estimated_tokens,
                    usable_tokens=prepared.usable_tokens,
                    utilization=prepared.estimated_tokens / max(1, prepared.usable_tokens),
                    audit=prepared.audit.model_dump(),
                )
                await self.policy.observe(event, state)
                tracker.before_model()
                response = await self.inference.complete(prepared.messages, seed=seed)
                tracker.after_model(response.usage.completion_tokens)
                self.inference_totals.record(response)
                state.emit(
                    EventKind.INFERENCE_RESPONSE,
                    prompt_tokens=response.usage.prompt_tokens,
                    generated_tokens=response.usage.completion_tokens,
                    latency_seconds=response.latency_seconds,
                    time_to_first_token_seconds=response.time_to_first_token_seconds,
                    inter_token_intervals_seconds=response.inter_token_intervals_seconds,
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
                tracker.before_tool()
                result = await self._execute(decision.tool, state)
                self.tool_calls += 1
                tool_event = state.emit(EventKind.TOOL_RESULT, result=result.model_dump())
                await self.policy.observe(tool_event, state)
                state.canonical_messages.append(
                    Message(
                        role="tool",
                        name=result.name,
                        tool_call_id=result.call_id,
                        content=result.content,
                    )
                )
                if result.name == "write_file" and result.ok and "path" in result.metadata:
                    state.files_modified.add(str(result.metadata["path"]))
        except ContextOverflow as error:
            state.failure = "context_window"
            state.emit(
                EventKind.CONTEXT_FAILURE,
                reason=state.failure,
                estimated_tokens=error.estimated,
                usable_tokens=error.usable,
            )
            return state
        except LimitExceeded as error:
            state.failure = error.limit
            state.emit(EventKind.STATUS, status="failed", reason=error.limit)
            return state
        state.failure = "steps"
        state.emit(EventKind.STATUS, status="failed", reason=state.failure)
        return state

    async def _execute(self, call: ToolCall, state: AgentState) -> ToolResult:
        if call.name == "retrieve":
            query = str(call.arguments.get("query", "")).strip()
            if not query:
                return ToolResult(
                    call_id=call.id, name=call.name, content="query is required", ok=False
                )
            recovered = await self.policy.recover(query, state, self.budget)  # type: ignore[attr-defined]
            content = "\n\n".join(
                f"[source={item.source_id} score={item.score:.4f}]\n{item.content}"
                for item in recovered.items
            )
            state.emit(
                EventKind.RETRIEVAL,
                query=query,
                tokens=recovered.tokens,
                latency_seconds=recovered.latency_seconds,
                source_ids=[item.source_id for item in recovered.items],
                explicit=True,
            )
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=content or "no relevant context found",
                ok=True,
                duration_seconds=recovered.latency_seconds,
                metadata={"tokens": recovered.tokens, "sources": len(recovered.items)},
            )
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
