"""Optional extension point for embedding-backed retrieval."""

from __future__ import annotations

from typing import Protocol

from contextlab.retrieval.index import Document


class DenseRetriever(Protocol):
    """Adapters may use sentence-transformers/FAISS without burdening CPU installs."""

    def add(self, documents: list[Document]) -> None: ...

    def search(self, query: str, *, limit: int = 10) -> list[tuple[Document, float]]: ...
