import pytest

from contextlab.agent.limits import LimitExceeded, LimitTracker
from contextlab.config import ExecutionLimits


def test_model_and_tool_limits_are_explicit() -> None:
    tracker = LimitTracker(ExecutionLimits(max_model_requests=1, max_tool_calls=1))
    tracker.before_model()
    tracker.before_tool()
    with pytest.raises(LimitExceeded, match="model_requests"):
        tracker.before_model()
    with pytest.raises(LimitExceeded, match="tool_calls"):
        tracker.before_tool()
