import pytest

from contextlab.agent.models import Message
from contextlab.config import BudgetConfig
from contextlab.context.budgeting import ContextOverflow, TokenBudget, TokenCounter


def test_budget_accounts_for_reservations_and_overflow() -> None:
    budget = TokenBudget(
        BudgetConfig(context_window=100, reserved_output_tokens=20, safety_margin_tokens=10),
        counter=TokenCounter(name="test"),
    )
    assert budget.usable_prompt_tokens == 70
    with pytest.raises(ContextOverflow) as caught:
        budget.ensure_fits([Message(role="user", content="x" * 400)])
    assert caught.value.estimated > 70


def test_client_and_server_counts_remain_separate() -> None:
    budget = TokenBudget(BudgetConfig(), model="mock")
    record = budget.record([Message(role="user", content="hello")], server_reported=99)
    assert record.server_reported == 99
    assert record.client_estimate != record.server_reported
