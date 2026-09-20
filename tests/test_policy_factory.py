from pathlib import Path

import pytest

from contextlab.config import PolicyConfig
from contextlab.context.factory import create_policy


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("full-history", "FullHistoryPolicy"),
        ("bounded-tool-output", "BoundedToolOutputPolicy"),
        ("retrieval", "RetrievalContextPolicy"),
        ("compaction", "CompactionPolicy"),
    ],
)
def test_factory_selects_all_interchangeable_policies(
    tmp_path: Path, name: str, expected: str
) -> None:
    config = PolicyConfig(name=name, deterministic_compaction=True)  # type: ignore[arg-type]
    assert type(create_policy(config, repository_root=tmp_path)).__name__ == expected
