"""Bounded repository inspection tools without shell interpolation."""

from __future__ import annotations

import re
import time

from contextlab.tools.models import ToolCall, ToolResult
from contextlab.tools.workspace import RepositoryWorkspace, WorkspaceError


class RepositoryTools:
    def __init__(self, workspace: RepositoryWorkspace, *, max_read_bytes: int = 200_000) -> None:
        self.workspace = workspace
        self.max_read_bytes = max_read_bytes

    def list_files(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        relative = str(call.arguments.get("path", "."))
        try:
            root = self.workspace.resolve(relative)
            files = sorted(
                str(path.relative_to(self.workspace.root))
                for path in root.rglob("*")
                if path.is_file() and ".git" not in path.parts
            )
            content = "\n".join(files[:2000])
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=content,
                ok=True,
                duration_seconds=time.perf_counter() - started,
                metadata={"total_files": len(files), "truncated": len(files) > 2000},
            )
        except (OSError, WorkspaceError) as error:
            return self._error(call, error, started)

    def read_file(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            path = self.workspace.resolve(str(call.arguments["path"]))
            raw = path.read_bytes()
            if len(raw) > self.max_read_bytes:
                raise ValueError(f"file exceeds {self.max_read_bytes} byte read limit")
            lines = raw.decode("utf-8", errors="replace").splitlines()
            start = max(1, int(call.arguments.get("start_line", 1)))
            end = min(len(lines), int(call.arguments.get("end_line", len(lines))))
            numbered = "\n".join(
                f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
            )
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=numbered,
                ok=True,
                duration_seconds=time.perf_counter() - started,
                metadata={
                    "path": str(path.relative_to(self.workspace.root)),
                    "start": start,
                    "end": end,
                },
            )
        except (KeyError, OSError, ValueError, WorkspaceError) as error:
            return self._error(call, error, started)

    def search(self, call: ToolCall) -> ToolResult:
        started = time.perf_counter()
        try:
            query = str(call.arguments["query"])
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            matches: list[str] = []
            for path in self.workspace.root.rglob("*"):
                if (
                    not path.is_file()
                    or ".git" in path.parts
                    or path.stat().st_size > self.max_read_bytes
                ):
                    continue
                for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line):
                        matches.append(f"{path.relative_to(self.workspace.root)}:{number}:{line}")
                    if len(matches) >= 500:
                        break
                if len(matches) >= 500:
                    break
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content="\n".join(matches),
                ok=True,
                duration_seconds=time.perf_counter() - started,
                metadata={
                    "matches": len(matches),
                    "truncated": len(matches) >= 500,
                    "query": query,
                },
            )
        except (KeyError, OSError, re.error) as error:
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
