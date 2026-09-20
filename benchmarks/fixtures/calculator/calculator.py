def divide(a: float, b: float) -> float:
    """Divide a by b and reject a zero denominator."""
    if a == 0:
        raise ZeroDivisionError("cannot divide by zero")
    return a / b
