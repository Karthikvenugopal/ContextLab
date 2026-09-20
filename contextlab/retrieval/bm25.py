"""Dependency-free BM25 ranking for CPU-only experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from contextlab.retrieval.index import Document


TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


class BM25Index:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(item.content) for item in self.documents]
        self.frequencies = [Counter(items) for items in self.tokens]
        self.avg_length = sum(map(len, self.tokens)) / len(self.tokens) if self.tokens else 0.0
        document_frequency: Counter[str] = Counter()
        for items in self.tokens:
            document_frequency.update(set(items))
        total = len(self.documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source_type: str | None = None,
        path_prefix: str | None = None,
    ) -> list[tuple[Document, float]]:
        query_terms = tokenize(query)
        ranked: list[tuple[Document, float]] = []
        for document, terms, frequencies in zip(
            self.documents, self.tokens, self.frequencies, strict=True
        ):
            if source_type is not None and document.source_type != source_type:
                continue
            if path_prefix is not None and not (document.path or "").startswith(path_prefix):
                continue
            score = 0.0
            length = len(terms)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                normalizer = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(1.0, self.avg_length)
                )
                score += self.idf.get(term, 0.0) * (frequency * (self.k1 + 1)) / normalizer
            if score > 0:
                ranked.append((document, score))
        ranked.sort(key=lambda item: (-item[1], item[0].source_id))
        return ranked[:limit]
