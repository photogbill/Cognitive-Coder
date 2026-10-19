_PAIRS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
          (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"),
          (4, "IV"), (1, "I")]


def to_roman(n):
    if not isinstance(n, int) or not 1 <= n <= 3999:
        raise ValueError(f"{n!r} is not in 1..3999")
    out = []
    for value, letters in _PAIRS:
        while n >= value:
            out.append(letters)
            n -= value
    return "".join(out)


def from_roman(s):
    if not isinstance(s, str) or not s:
        raise ValueError("empty numeral")
    text = s.upper()
    total, i = 0, 0
    for value, letters in _PAIRS:
        while text.startswith(letters, i):
            total += value
            i += len(letters)
    if i != len(text) or not 1 <= total <= 3999 or to_roman(total) != text:
        raise ValueError(f"{s!r} is not a valid numeral")
    return total
