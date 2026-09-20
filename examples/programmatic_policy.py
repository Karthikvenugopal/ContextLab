"""Construct a policy directly for custom agent integrations."""

from pathlib import Path

from contextlab.config import PolicyConfig
from contextlab.context.factory import create_policy

repository = Path("benchmarks/fixtures/calculator")
policy = create_policy(
    PolicyConfig(name="retrieval", retrieval_tokens=1500),
    repository_root=repository,
)
print(policy.policy_id)
