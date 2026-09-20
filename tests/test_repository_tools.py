from pathlib import Path

from contextlab.tools.models import ToolCall
from contextlab.tools.repository import RepositoryTools
from contextlab.tools.workspace import RepositoryWorkspace


def test_repository_inspection_and_search(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def answer():\n    return 42\n")
    tools = RepositoryTools(RepositoryWorkspace(tmp_path))
    listing = tools.list_files(ToolCall(name="list_files"))
    reading = tools.read_file(ToolCall(name="read_file", arguments={"path": "app.py"}))
    search = tools.search(ToolCall(name="search", arguments={"query": "answer"}))
    assert listing.content == "app.py"
    assert "2:     return 42" in reading.content
    assert "app.py:1" in search.content
