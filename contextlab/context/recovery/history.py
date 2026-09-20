"""Immutable external store for complete historical observations."""

from __future__ import annotations

import hashlib

from contextlab.agent.models import AgentEvent, EventKind
from contextlab.retrieval.bm25 import BM25Index
from contextlab.retrieval.index import Document


class ObservationStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._ids: set[str] = set()

    def add_event(self, event: AgentEvent) -> Document | None:
        if event.kind != EventKind.TOOL_RESULT:
            return None
        result = event.payload.get("result", {})
        content = str(result.get("content", ""))
        if not content:
            return None
        digest = hashlib.sha256(content.encode()).hexdigest()
        call_id = str(result.get("call_id", event.id))
        source_id = f"observation:{call_id}:{digest[:12]}"
        if source_id in self._ids:
            return None
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        document = Document(
            source_id=source_id,
            content=content,
            source_type="observation",
            path=str(metadata.get("path")) if metadata.get("path") else None,
            content_hash=digest,
            metadata={
                "tool": str(result.get("name", "unknown")),
                "step": str(event.step),
                "run_id": event.run_id,
            },
        )
        self.documents.append(document)
        self._ids.add(source_id)
        return document

    def search(self, query: str, *, limit: int = 10) -> list[tuple[Document, float]]:
        return BM25Index(self.documents).search(query, limit=limit, source_type="observation")
