import pytest
from settings import load_timeout


def test_contextlab_timeout_takes_precedence(monkeypatch):
    monkeypatch.setenv("APP_TIMEOUT", "20")
    monkeypatch.setenv("CONTEXTLAB_TIMEOUT", "7")
    assert load_timeout() == 7


def test_invalid_value_has_clear_error(monkeypatch):
    monkeypatch.setenv("CONTEXTLAB_TIMEOUT", "never")
    with pytest.raises(ValueError, match="CONTEXTLAB_TIMEOUT"):
        load_timeout()
