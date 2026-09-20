from pathlib import Path

from contextlab.retrieval.index import RepositoryIndexer


def test_chunks_source_with_line_metadata_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("\n".join(f"line {i}" for i in range(12)))
    documents = RepositoryIndexer(chunk_lines=5, overlap_lines=1).index(tmp_path)
    assert len(documents) == 3
    assert documents[0].start_line == 1 and documents[0].end_line == 5
    assert documents[1].start_line == 5
    assert len(documents[0].content_hash) == 64


def test_excludes_evaluation_only_artifacts(tmp_path: Path) -> None:
    hidden = tmp_path / ".contextlab-evaluation"
    hidden.mkdir()
    (hidden / "secret.py").write_text("REFERENCE_ANSWER = 42")
    (tmp_path / "visible.py").write_text("TODO = True")
    documents = RepositoryIndexer().index(tmp_path)
    assert [item.path for item in documents] == ["visible.py"]
