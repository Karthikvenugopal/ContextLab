from contextlab.context.base import BaseContextPolicy, ContextPolicy, PreparedContext


def test_base_policy_satisfies_runtime_contract_shape() -> None:
    assert hasattr(BaseContextPolicy(), "prepare_context") is False
    assert PreparedContext(messages=[], estimated_tokens=0, usable_tokens=10).usable_tokens == 10
    assert ContextPolicy is not None
