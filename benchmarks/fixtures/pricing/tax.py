def tax_rate(region: str) -> float:
    return {"CA": 0.10, "OR": 0.0}.get(region, 0.05)
