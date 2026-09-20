"""Isolated, path-safe repository workspaces."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class WorkspaceError(RuntimeError):
    pass


class RepositoryWorkspace:
    def __init__(self, root: Path, *, owned: bool = False) -> None:
        self.root = root.resolve()
        self.owned = owned

    @classmethod
    def from_revision(cls, repository: Path, revision: str = "HEAD") -> RepositoryWorkspace:
        target = Path(tempfile.mkdtemp(prefix="contextlab-workspace-"))
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-hardlinks",
                    str(repository.resolve()),
                    str(target),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "checkout", "--quiet", revision],
                cwd=target,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            shutil.rmtree(target, ignore_errors=True)
            raise WorkspaceError(f"unable to create workspace at {revision}: {error}") from error
        return cls(target, owned=True)

    def resolve(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise WorkspaceError("tool paths must be non-empty and relative")
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise WorkspaceError(f"path escapes workspace: {relative}")
        return candidate

    def close(self) -> None:
        if self.owned:
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> RepositoryWorkspace:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
