"""Small text and number helpers used all over the project."""


def slugify(text):
    """Lower-case, words joined by dashes."""
    import re
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return text.strip("-")


def title_case(text):
    """Capitalise words, except small ones after the first."""
    small = {"a", "an", "the", "of", "and", "or", "in"}
    words = text.split()
    out = []
    for i, w in enumerate(words):
        out.append(w.lower() if i and w.lower() in small else w.capitalize())
    return " ".join(out)


def truncate(text, width):
    """Cut to `width` characters, ending in an ellipsis."""
    if width < 4:
        raise ValueError("width must be at least 4")
    if len(text) <= width:
        return text
    return text[:width - 3].rstrip() + "..."


def count_vowels(text):
    """How many vowels."""
    return sum(1 for c in text.lower() if c in "aeiou")


def is_palindrome(text):
    """Ignoring case, spaces and punctuation."""
    letters = [c.lower() for c in text if c.isalnum()]
    return letters == letters[::-1]


def wrap(text, width):
    """Greedy word wrap into lines of at most `width`."""
    words, lines, line = text.split(), [], ""
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        lines.append(line)
    return lines


def initials(text):
    """First letter of each word."""
    return "".join(w[0].upper() for w in text.split() if w)


def dedupe_spaces(text):
    """Collapse runs of whitespace."""
    return " ".join(text.split())


def rot13(text):
    """The ROT13 cipher."""
    import codecs
    return codecs.encode(text, "rot13")


def word_count(text):
    """Words separated by whitespace."""
    return len(text.split())


def format_bytes(n):
    """1536 -> '1.5 KB'."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024


def format_seconds(total):
    """125 -> '2:05', 3725 -> '1:02:05'."""
    minutes, seconds = divmod(int(total), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def parse_bool(text):
    """yes/no/true/false/on/off/1/0."""
    value = text.strip().lower()
    if value in ("yes", "y", "true", "1", "on"):
        return True
    if value in ("no", "n", "false", "0", "off"):
        return False
    raise ValueError(f"not a yes/no value: {text!r}")


def parse_duration(text):
    """Minutes in a duration such as '90m' or '45m'."""
    text = text.strip().lower()
    if not text.endswith("m") or not text[:-1].isdigit():
        raise ValueError(f"not a duration: {text!r}")
    return int(text[:-1])


def parse_ratio(text):
    """'16:9' -> 1.777..."""
    left, _, right = text.partition(":")
    if not right:
        raise ValueError(f"not a ratio: {text!r}")
    return float(left) / float(right)


def clamp(value, lo, hi):
    """`value` held between `lo` and `hi`."""
    return max(lo, min(hi, value))


def lerp(a, b, t):
    """Linear interpolation."""
    return a + (b - a) * t


def mean(values):
    """The arithmetic mean."""
    if not values:
        raise ValueError("mean of nothing")
    return sum(values) / len(values)


def median(values):
    """The middle value."""
    if not values:
        raise ValueError("median of nothing")
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def chunks(items, size):
    """Split a list into pieces of `size`."""
    if size < 1:
        raise ValueError("size must be at least 1")
    return [items[i:i + size] for i in range(0, len(items), size)]


def flatten(items):
    """Nested lists into one list."""
    out = []
    for x in items:
        if isinstance(x, (list, tuple)):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out


def unique(items):
    """In first-seen order."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


if __name__ == "__main__":
    print(parse_duration("90m"), format_seconds(3725))
