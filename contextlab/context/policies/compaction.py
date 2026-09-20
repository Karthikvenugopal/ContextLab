"""Structured context compaction with configurable trigger rules."""

from __future__ import annotations

from pydantic import BaseModel, Field

from contextlab.agent.models import AgentState


class CompactionTriggers(BaseModel):
    token_threshold: int | None = Field(default=None, gt=0)
    utilization_threshold: float | None = Field(default=0.75, gt=0, le=1)
    every_steps: int | None = Field(default=None, gt=0)
    tool_output_tokens: int | None = Field(default=8000, gt=0)

    def reasons(
        self,
        *,
        state: AgentState,
        prompt_tokens: int,
        usable_tokens: int,
        accumulated_tool_tokens: int,
        last_compaction_step: int,
    ) -> list[str]:
        reasons: list[str] = []
        if self.token_threshold is not None and prompt_tokens >= self.token_threshold:
            reasons.append("token_threshold")
        if (
            self.utilization_threshold is not None
            and prompt_tokens / max(1, usable_tokens) >= self.utilization_threshold
        ):
            reasons.append("utilization")
        if self.every_steps is not None and state.step - last_compaction_step >= self.every_steps:
            reasons.append("agent_steps")
        if self.tool_output_tokens is not None and accumulated_tool_tokens >= self.tool_output_tokens:
            reasons.append("tool_output_volume")
        return reasons
