"""Baseline policy retaining every canonical message without mutation."""

from contextlab.agent.models import AgentState
from contextlab.context.base import BaseContextPolicy, ContextAudit, PreparedContext, TokenBudgetLike


class FullHistoryPolicy(BaseContextPolicy):
    policy_id = "full-history"

    async def prepare_context(
        self, state: AgentState, budget: TokenBudgetLike
    ) -> PreparedContext:
        messages = [message.model_copy(deep=True) for message in state.canonical_messages]
        estimated = budget.count_messages(messages)
        if estimated > budget.usable_prompt_tokens:
            from contextlab.context.budgeting import ContextOverflow

            raise ContextOverflow(estimated, budget.usable_prompt_tokens)
        return PreparedContext(
            messages=messages,
            estimated_tokens=estimated,
            usable_tokens=budget.usable_prompt_tokens,
            audit=ContextAudit(
                retained_message_indices=list(range(len(messages))),
                metadata={"utilization": estimated / budget.usable_prompt_tokens},
            ),
        )
