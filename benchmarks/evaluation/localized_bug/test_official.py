import pytest
from calculator import divide


def test_zero_numerator_is_valid():
    assert divide(0, 5) == 0


def test_zero_denominator_is_rejected():
    with pytest.raises(ZeroDivisionError):
        divide(5, 0)
