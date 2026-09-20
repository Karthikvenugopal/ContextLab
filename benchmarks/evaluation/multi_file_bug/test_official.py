from pricing import total


def test_tax_is_applied_to_subtotal():
    assert total(100, "CA") == 110
    assert total(100, "OR") == 100
    assert total(200, "unknown") == 210
