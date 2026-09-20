"""Source-aware repository discovery and line-based chunking."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}
DENIED_PARTS = {".git", ".contextlab-evaluation", "hidden_tests", "reference_patches"}


class Document(BaseModel):
    source_id: str
    content: str
    source_type: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    content_hash: str
    metadata: dict[str, str] = Field(default_factory=dict)


class RepositoryIndexer:
    def __init__(self, *, chunk_lines: int = 80, overlap_lines: int = 10) -> None:
        if chunk_lines <= overlap_lines or overlap_lines < 0:
            raise ValueError("chunk_lines must exceed non-negative overlap")
        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines

    def discover(self, root: Path) -> list[Path]:
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in DEFAULT_EXTENSIONS
            and not DENIED_PARTS.intersection(path.relative_to(root).parts)
            and path.stat().st_size <= 1_000_000
        )

    def index(self, root: Path) -> list[Document]:
        documents: list[Document] = []
        for path in self.discover(root):
            relative = str(path.relative_to(root))
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not lines:
                continue
            step = self.chunk_lines - self.overlap_lines
            for offset in range(0, len(lines), step):
                chunk = lines[offset : offset + self.chunk_lines]
                if not chunk:
                    break
                content = "\n".join(chunk)
                digest = hashlib.sha256(content.encode()).hexdigest()
                start, end = offset + 1, offset + len(chunk)
                documents.append(
                    Document(
                        source_id=f"repo:{relative}:{start}-{end}:{digest[:12]}",
                        content=content,
                        source_type="repository",
                        path=relative,
                        start_line=start,
                        end_line=end,
                        content_hash=digest,
                    )
                )
                if end == len(lines):
                    break
        return documents
