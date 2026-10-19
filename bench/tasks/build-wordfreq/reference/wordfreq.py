import re
import sys
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def top_words(text, n):
    if n < 0:
        raise ValueError("n must not be negative")
    counts = Counter(_WORD.findall(text.lower()))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]


def main(argv):
    path = argv[1]
    n = int(argv[2]) if len(argv) > 2 else 10
    with open(path, encoding="utf-8") as fh:
        for word, count in top_words(fh.read(), n):
            print(f"{word} {count}")


if __name__ == "__main__":
    main(sys.argv)
