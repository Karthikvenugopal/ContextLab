"""Safe file modification and allowlisted command execution."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from contextlab.tools.models import ToolCall, ToolResult
from contextlab.tools.workspace import RepositoryWorkspace, WorkspaceError


class MutationTools:
    def __init__(
        self,
        workspace: RepositoryWorkspace,
        *,
        allowed_commands: list[str],
        command_timeout: float = 120,
    ) -> None:
        self.workspace = workspace
        self.allowed_commands = set(allowed_commands)
        self.command_timeout = command_timeout

    def write_file(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            path = self.workspace.resolve(str(call.arguments["path"]))
            content = str(call.arguments["content"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"wrote {len(content.encode())} bytes to {path.relative_to(self.workspace.root)}",
                ok=True,
                duration_seconds=time.perf_counter() - started,
                metadata={"path": str(path.relative_to(self.workspace.root)), "bytes": len(content.encode())},
            )
        except (KeyError, OSError, WorkspaceError) as error:
            return self._error(call, error, started)

    def run_command(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            raw = call.arguments["argv"]
            argv = shlex.split(raw) if isinstance(raw, str) else [str(item) for item in raw]
            if not argv or Path(argv[0]).name not in self.allowed_commands:
                raise PermissionError(f"command is not allowlisted: {argv[0] if argv else '<empty>'}")
            cwd = self.workspace.resolve(str(call.arguments.get("cwd", ".")))
            if not cwd.is_dir():
                raise ValueError("command cwd is not a directory")
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(self.workspace.root)},
                check=False,
            )
            output = completed.stdout + completed.stderr
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=output,
                ok=completed.returncode == 0,
                duration_seconds=time.perf_counter() - started,
                metadata={"argv": argv, "exit_code": completed.returncode},
            )
        except (KeyError, OSError, ValueError, PermissionError, subprocess.TimeoutExpired) as error:
            return self._error(call, error, started)

    @staticmethod
    def _error(call: ToolCall, error: Exception, started: float) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=call.name,
            content=f"{type(error).__name__}: {error}",
            ok=False,
            duration_seconds=time.perf_counter() - started,
        )
