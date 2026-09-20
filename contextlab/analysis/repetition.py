"""Evidence-based repeated repository interaction analysis."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from contextlab.agent.models import AgentEvent, EventKind


class RepeatEvidence(BaseModel):
    kind: str
    key: str
    first_step: int
    repeated_step: int
    first_call_id: str | None = None
    repeated_call_id: str | None = None
    context_loss_supported: bool = False
    attribution: str = "unattributed"


class RepeatedAccess(BaseModel):
    repeated_file_reads: int = 0
    repeated_searches: int = 0
    evidence: list[RepeatEvidence] = Field(default_factory=list)


def analyze_repeated_access(events: list[AgentEvent]) -> RepeatedAccess:
    seen: dict[tuple[str, str], tuple[int, str | None]] = {}
    removed_before_step: defaultdict[int, bool] = defaultdict(bool)
    result = RepeatedAccess()
    for event in events:
        if event.kind == EventKind.CONTEXT_PREPARED:
            audit = event.payload.get("audit", {})
            removed = audit.get("removed_message_indices", []) if isinstance(audit, dict) else []
            removed_before_step[event.step] = bool(removed)
            continue
        if event.kind != EventKind.TOOL_RESULT:
            continue
        tool_result = event.payload.get("result", {})
        if not isinstance(tool_result, dict):
            continue
        name = str(tool_result.get("name", ""))
        metadata = tool_result.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        if name == "read_file" and metadata.get("path"):
            kind, key = "file_read", str(metadata["path"])
        elif name == "search" and metadata.get("query"):
            kind, key = "search", str(metadata["query"]).casefold()
        else:
            continue
        signature = (kind, key)
        if signature in seen:
            first_step, first_id = seen[signature]
            supported = any(
                removed_before_step[step] for step in range(first_step + 1, event.step + 1)
            )
            evidence = RepeatEvidence(
                kind=kind,
                key=key,
                first_step=first_step,
                repeated_step=event.step,
                first_call_id=first_id,
                repeated_call_id=str(tool_result.get("call_id"))
                if tool_result.get("call_id")
                else None,
                context_loss_supported=supported,
                attribution="context-removal-associated" if supported else "unattributed",
            )
            result.evidence.append(evidence)
            if kind == "file_read":
                result.repeated_file_reads += 1
            else:
                result.repeated_searches += 1
        else:
            seen[signature] = (
                event.step,
                str(tool_result.get("call_id")) if tool_result.get("call_id") else None,
            )
    return result
