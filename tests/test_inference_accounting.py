from contextlab.inference.accounting import InferenceTotals
from contextlab.inference.client import InferenceResponse, Usage


def test_auxiliary_tokens_are_included_in_total() -> None:
    totals = InferenceTotals()
    totals.record(
        InferenceResponse(
            content="summary",
            model="mock",
            request_kind="compaction",
            usage=Usage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
        )
    )
    assert totals.auxiliary_tokens == 25
    assert totals.total_tokens == 25
