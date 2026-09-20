"""Summarize retention, reduction, recovery, and compaction behavior."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from contextlab.agent.models import AgentEvent, EventKind


class ContextBehavior(BaseModel):
    preparations: int = 0
    retained_message_instances: int = 0
    removed_message_instances: int = 0
    compressed_message_instances: int = 0
    original_tool_tokens: int = 0
    retained_tool_tokens: int = 0
    discarded_tool_tokens: int = 0
    retrieval_queries: int = 0
    retrieval_tokens: int = 0
    repeated_retrievals: int = 0
    recovered_sources: Counter[str] = Field(default_factory=Counter)
    compactions: int = 0
    compaction_input_tokens: int = 0
    compaction_output_tokens: int = 0


def analyze_context_behavior(events: list[AgentEvent]) -> ContextBehavior:
    behavior = ContextBehavior()
    queries: Counter[str] = Counter()
    for event in events:
        payload = event.payload
        if event.kind == EventKind.CONTEXT_PREPARED:
            behavior.preparations += 1
            audit = payload.get("audit", {})
            if not isinstance(audit, dict):
                continue
            behavior.retained_message_instances += len(audit.get("retained_message_indices", []))
            behavior.removed_message_instances += len(audit.get("removed_message_indices", []))
            behavior.compressed_message_instances += len(audit.get("compressed_message_indices", []))
            behavior.recovered_sources.update(audit.get("recovered_source_ids", []))
            metadata = audit.get("metadata", {})
            reductions = metadata.get("reductions", []) if isinstance(metadata, dict) else []
            for reduction in reductions:
                if not isinstance(reduction, dict):
                    continue
                behavior.original_tool_tokens += int(reduction.get("original_tokens", 0))
                behavior.retained_tool_tokens += int(reduction.get("retained_tokens", 0))
                behavior.discarded_tool_tokens += int(reduction.get("discarded_tokens", 0))
        elif event.kind == EventKind.RETRIEVAL:
            behavior.retrieval_queries += 1
            behavior.retrieval_tokens += int(payload.get("tokens", 0))
            query = str(payload.get("query", ""))
            queries[query] += 1
            if queries[query] > 1:
                behavior.repeated_retrievals += 1
            behavior.recovered_sources.update(payload.get("source_ids", []))
        elif event.kind == EventKind.COMPACTION:
            behavior.compactions += 1
            behavior.compaction_input_tokens += int(payload.get("input_tokens", 0))
            behavior.compaction_output_tokens += int(payload.get("output_tokens", 0))
    return behavior
