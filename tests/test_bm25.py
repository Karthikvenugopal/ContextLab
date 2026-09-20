import hashlib

from contextlab.retrieval.bm25 import BM25Index
from contextlab.retrieval.index import Document


def document(source_id: str, content: str, path: str) -> Document:
    return Document(
        source_id=source_id,
        content=content,
        source_type="repository",
        path=path,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )


def test_bm25_ranks_rare_relevant_terms_first() -> None:
    index = BM25Index(
        [
            document("a", "generic helper parses input", "src/a.py"),
            document("b", "RareParser RareParser handles malformed payload", "src/parser.py"),
        ]
    )
    ranked = index.search("RareParser malformed")
    assert ranked[0][0].source_id == "b"
    assert ranked[0][1] > 0


def test_bm25_supports_metadata_filters() -> None:
    index = BM25Index(
        [document("a", "shared token", "src/a.py"), document("b", "shared token", "tests/b.py")]
    )
    assert [item.path for item, _ in index.search("shared", path_prefix="src/")] == ["src/a.py"]
