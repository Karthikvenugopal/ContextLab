from tax import tax_rate


def total(subtotal: float, region: str) -> float:
    return subtotal + tax_rate(region)
