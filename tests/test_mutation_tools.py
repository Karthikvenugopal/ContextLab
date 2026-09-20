from pathlib import Path

from contextlab.tools.execution import MutationTools
from contextlab.tools.models import ToolCall
from contextlab.tools.workspace import RepositoryWorkspace


def test_writes_files_and_runs_allowlisted_command(tmp_path: Path) -> None:
    tools = MutationTools(RepositoryWorkspace(tmp_path), allowed_commands=["python"])
    written = tools.write_file(
        ToolCall(name="write_file", arguments={"path": "pkg/value.py", "content": "VALUE = 7\n"})
    )
    result = tools.run_command(
        ToolCall(name="run_command", arguments={"argv": ["python", "-c", "print('ok')"]})
    )
    assert written.ok and (tmp_path / "pkg/value.py").exists()
    assert result.ok and result.content.strip() == "ok"


def test_rejects_non_allowlisted_and_escaping_writes(tmp_path: Path) -> None:
    tools = MutationTools(RepositoryWorkspace(tmp_path), allowed_commands=["pytest"])
    assert not tools.run_command(ToolCall(name="run_command", arguments={"argv": ["sh", "-c", "x"]})).ok
    assert not tools.write_file(
        ToolCall(name="write_file", arguments={"path": "../outside", "content": "no"})
    ).ok
