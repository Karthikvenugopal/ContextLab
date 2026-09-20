"""Policy that bounds tool-output text while retaining full canonical history."""

from __future__ import annotations

from contextlab.agent.models import AgentState, Message
from contextlab.context.base import BaseContextPolicy, ContextAudit, PreparedContext, TokenBudgetLike
from contextlab.context.budgeting import ContextOverflow


class BoundedToolOutputPolicy(BaseContextPolicy):
    policy_id = "bounded-tool-output"

    def __init__(self, *, per_tool_tokens: int = 1200, total_tool_tokens: int = 6000) -> None:
        self.per_tool_tokens = per_tool_tokens
        self.total_tool_tokens = total_tool_tokens
        self.reductions: list[dict[str, object]] = []

    async def prepare_context(
        self, state: AgentState, budget: TokenBudgetLike
    ) -> PreparedContext:
        messages: list[Message] = []
        removed: list[int] = []
        retained: list[int] = []
        remaining_tool_tokens = self.total_tool_tokens
        self.reductions = []
        for index, original in enumerate(state.canonical_messages):
            if original.role != "tool":
                messages.append(original.model_copy(deep=True))
                retained.append(index)
                continue
            original_tokens = max(1, len(original.content.encode()) // 4)
            allowance = min(self.per_tool_tokens, remaining_tool_tokens)
            if original_tokens <= allowance:
                content = original.content
                retained_tokens = original_tokens
                method = "none"
                retained.append(index)
            elif allowance > 0:
                content = self._head_tail(original.content, allowance)
                retained_tokens = max(1, len(content.encode()) // 4)
                method = "head-tail"
                removed.append(index)
            else:
                content = f"[tool output omitted; call_id={original.tool_call_id or 'unknown'}]"
                retained_tokens = max(1, len(content.encode()) // 4)
                method = "omitted-global-budget"
                removed.append(index)
            remaining_tool_tokens = max(0, remaining_tool_tokens - retained_tokens)
            messages.append(original.model_copy(update={"content": content}))
            self.reductions.append(
                {
                    "tool_call_id": original.tool_call_id,
                    "tool": original.name,
                    "original_tokens": original_tokens,
                    "retained_tokens": retained_tokens,
                    "discarded_tokens": max(0, original_tokens - retained_tokens),
                    "method": method,
                }
            )
        estimated = budget.count_messages(messages)
        if estimated > budget.usable_prompt_tokens:
            raise ContextOverflow(estimated, budget.usable_prompt_tokens)
        return PreparedContext(
            messages=messages,
            estimated_tokens=estimated,
            usable_tokens=budget.usable_prompt_tokens,
            audit=ContextAudit(
                retained_message_indices=retained,
                removed_message_indices=removed,
                metadata={"reductions": self.reductions},
            ),
        )

    @staticmethod
    def _head_tail(content: str, token_limit: int) -> str:
        character_limit = max(40, token_limit * 4)
        marker = "\n...[bounded tool output]...\n"
        remaining = max(2, character_limit - len(marker))
        head = remaining // 2
        return content[:head] + marker + content[-(remaining - head) :]
