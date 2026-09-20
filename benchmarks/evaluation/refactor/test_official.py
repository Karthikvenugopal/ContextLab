from catalog import Catalog


def test_instances_do_not_share_state():
    first, second = Catalog(), Catalog()
    first.add("apple")
    assert first.names() == ["apple"]
    assert second.names() == []
