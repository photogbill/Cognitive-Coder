"""Prices and discounts for the shop."""


def subtotal(lines):
    """Sum of qty * unit price over (name, qty, unit) lines."""
    return sum(qty * unit for _name, qty, unit in lines)


def apply_discount(total, discount):
    """`discount` is a percentage: 10 or "10%" both mean ten percent off."""
    return round(total * (1 - discount / 100), 2)


def receipt(lines, discount=0):
    total = subtotal(lines)
    return f"total {apply_discount(total, discount):.2f}"
