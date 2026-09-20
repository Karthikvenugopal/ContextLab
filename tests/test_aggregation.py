from contextlab.analysis.aggregate import describe, percentile


def test_percentiles_and_median_are_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 100.0]
    stats = describe(values)
    assert stats["median"] == 2.5
    assert stats["p25"] == percentile(values, 0.25)
    assert stats["p95"] > stats["p75"]


def test_empty_description_is_explicit_zero() -> None:
    assert describe([]) == {"median": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0}
