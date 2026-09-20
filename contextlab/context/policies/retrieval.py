"""Retrieval policy with separate repository and observation indexes."""

from __future__ import annotations

import time
from collections import Counter

from contextlab.agent.models import AgentEvent, AgentState, Message
from contextlab.context.base import (
    BaseContextPolicy,
    ContextAudit,
    PreparedContext,
    RecoveryItem,
    RecoveryResult,
    TokenBudgetLike,
)
from contextlab.context.budgeting import ContextOverflow
from contextlab.context.recovery import ObservationStore
from contextlab.retrieval.bm25 import BM25Index
from contextlab.retrieval.index import Document


class RetrievalContextPolicy(BaseContextPolicy):
    policy_id = "retrieval"

    def __init__(
        self,
        repository_documents: list[Document],
        *,
        retrieval_tokens: int = 3000,
        active_messages: int = 6,
    ) -> None:
        self.repository_documents = repository_documents
        self.repository_index = BM25Index(repository_documents)
        self.observations = ObservationStore()
        self.retrieval_tokens = retrieval_tokens
        self.active_messages = active_messages
        self.query_counts: Counter[str] = Counter()
        self.retrieval_events: list[dict[str, object]] = []

    async def observe(self, event: AgentEvent, state: AgentState) -> None:
        del state
        self.observations.add_event(event)

    async def recover(
        self, query: str, state: AgentState, budget: TokenBudgetLike
    ) -> RecoveryResult:
        del state
        started = time.perf_counter()
        self.query_counts[query] += 1
        candidates = [
            *self.repository_index.search(query, limit=12),
            *self.observations.search(query, limit=12),
        ]
        candidates.sort(key=lambda item: (-item[1], item[0].source_id))
        cap = min(self.retrieval_tokens, budget.usable_prompt_tokens)
        used = 0
        items: list[RecoveryItem] = []
        for document, score in candidates:
            tokens = max(1, len(document.content.encode()) // 4)
            if tokens > cap - used:
                continue
            items.append(
                RecoveryItem(
                    source_id=document.source_id,
                    content=document.content,
                    score=score,
                    tokens=tokens,
                    metadata={
                        "source_type": document.source_type,
                        "path": document.path,
                        "start_line": document.start_line,
                        "end_line": document.end_line,
                        "content_hash": document.content_hash,
                    },
                )
            )
            used += tokens
            if used >= cap:
                break
        latency = time.perf_counter() - started
        result = RecoveryResult(query=query, items=items, tokens=used, latency_seconds=latency)
        self.retrieval_events.append(
            {
                "query": query,
                "latency_seconds": latency,
                "tokens": used,
                "sources": [item.source_id for item in items],
                "repeat_number": self.query_counts[query],
            }
        )
        return result

    async def prepare_context(self, state: AgentState, budget: TokenBudgetLike) -> PreparedContext:
        count = len(state.canonical_messages)
        mandatory = list(range(min(2, count)))
        recent_start = max(2, count - self.active_messages)
        selected = sorted(set([*mandatory, *range(recent_start, count)]))
        query = self._automatic_query(state)
        recovered = await self.recover(query, state, budget)
        messages = [state.canonical_messages[index].model_copy(deep=True) for index in selected]
        if recovered.items:
            sources = "\n\n".join(
                f"[source={item.source_id} metadata={item.metadata}]\n{item.content}"
                for item in recovered.items
            )
            messages.insert(
                2 if len(messages) >= 2 else len(messages), Message(role="system", content=sources)
            )
        estimated = budget.count_messages(messages)
        if estimated > budget.usable_prompt_tokens:
            raise ContextOverflow(estimated, budget.usable_prompt_tokens)
        return PreparedContext(
            messages=messages,
            estimated_tokens=estimated,
            usable_tokens=budget.usable_prompt_tokens,
            audit=ContextAudit(
                retained_message_indices=selected,
                removed_message_indices=[index for index in range(count) if index not in selected],
                recovered_source_ids=[item.source_id for item in recovered.items],
                metadata={
                    "retrieval_query": query,
                    "retrieval_tokens": recovered.tokens,
                    "retrieval_latency_seconds": recovered.latency_seconds,
                },
            ),
        )

    @staticmethod
    def _automatic_query(state: AgentState) -> str:
        recent = " ".join(message.content[-500:] for message in state.canonical_messages[-2:])
        return f"{state.objective} {recent}".strip()
