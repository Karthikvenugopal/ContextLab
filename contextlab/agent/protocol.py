"""Validated structured-output protocol for model-directed actions."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from contextlab.tools.models import ToolCall


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["tool", "finish"]
    tool: ToolCall | None = None
    summary: str = ""


class MalformedDecision(ValueError):
    pass


def parse_decision(content: str) -> AgentDecision:
    """Parse one JSON object, accepting a single Markdown JSON fence."""
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        decision = AgentDecision.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError) as error:
        raise MalformedDecision(f"invalid agent decision: {error}") from error
    if decision.action == "tool" and decision.tool is None:
        raise MalformedDecision("tool action requires a tool call")
    if decision.action == "finish" and decision.tool is not None:
        raise MalformedDecision("finish action cannot contain a tool call")
    return decision


TOOL_PROTOCOL = """Respond with exactly one JSON object.
To use a tool: {"action":"tool","tool":{"name":"read_file","arguments":{"path":"..."}}}
To finish: {"action":"finish","summary":"what changed and validation performed"}
Available tools: list_files(path), read_file(path,start_line,end_line), search(query),
write_file(path,content), run_command(argv,cwd), retrieve(query)."""
