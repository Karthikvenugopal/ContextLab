"""Append-only JSONL traces with optional content redaction."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from contextlab.agent.models import AgentEvent


SENSITIVE_KEYS = {"content", "prompt", "response", "task_instruction"}


def _redact(value: Any, key: str | None = None) -> Any:
    if key in SENSITIVE_KEYS and isinstance(value, str):
        digest = hashlib.sha256(value.encode()).hexdigest()
        return {"redacted": True, "sha256": digest, "characters": len(value)}
    if isinstance(value, dict):
        return {item_key: _redact(item, item_key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class EventLogger:
    def __init__(self, path: Path, *, redact: bool = False) -> None:
        self.path = path
        self.redact = redact
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: AgentEvent) -> None:
        record = event.model_dump(mode="json")
        if self.redact:
            record = _redact(record)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def write_all(self, events: list[AgentEvent]) -> None:
        for event in events:
            self.write(event)


def load_events(path: Path) -> list[AgentEvent]:
    with path.open(encoding="utf-8") as stream:
        return [AgentEvent.model_validate_json(line) for line in stream if line.strip()]
