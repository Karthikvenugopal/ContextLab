from pathlib import Path

import pytest

from contextlab.tools.workspace import RepositoryWorkspace, WorkspaceError


def test_safe_path_resolution(tmp_path: Path) -> None:
    workspace = RepositoryWorkspace(tmp_path)
    assert workspace.resolve("src/app.py") == tmp_path / "src/app.py"
    with pytest.raises(WorkspaceError):
        workspace.resolve("../secret")


def test_workspace_clones_requested_revision(tmp_path: Path) -> None:
    import subprocess

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "value.txt").write_text("baseline", encoding="utf-8")
    subprocess.run(["git", "add", "value.txt"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=source, check=True)
    with RepositoryWorkspace.from_revision(source) as workspace:
        assert workspace.resolve("value.txt").read_text() == "baseline"
