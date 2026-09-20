"""Construct policies from shared configuration without changing the agent loop."""

from pathlib import Path

from contextlab.config import PolicyConfig
from contextlab.context.base import ContextPolicy
from contextlab.context.policies import (
    BoundedToolOutputPolicy,
    CompactionPolicy,
    FullHistoryPolicy,
    RetrievalContextPolicy,
)
from contextlab.context.policies.compaction import (
    CompactionTriggers,
    Compactor,
    DeterministicCompactor,
    ModelCompactor,
)
from contextlab.inference.client import InferenceClient
from contextlab.retrieval.index import RepositoryIndexer


def create_policy(
    config: PolicyConfig, *, repository_root: Path, inference: InferenceClient | None = None
) -> ContextPolicy:
    if config.name == "full-history":
        return FullHistoryPolicy()
    if config.name == "bounded-tool-output":
        return BoundedToolOutputPolicy(
            per_tool_tokens=config.per_tool_tokens,
            total_tool_tokens=config.total_tool_tokens,
        )
    if config.name == "retrieval":
        return RetrievalContextPolicy(
            RepositoryIndexer().index(repository_root), retrieval_tokens=config.retrieval_tokens
        )
    if config.name == "compaction":
        if config.deterministic_compaction:
            compactor: Compactor = DeterministicCompactor()
        elif inference is not None:
            compactor = ModelCompactor(inference)
        else:
            raise ValueError("model-based compaction requires an inference client")
        return CompactionPolicy(
            compactor=compactor,
            triggers=CompactionTriggers(
                utilization_threshold=config.compaction_threshold,
                every_steps=config.compaction_every_steps,
            ),
        )
    raise ValueError(f"unknown policy: {config.name}")
