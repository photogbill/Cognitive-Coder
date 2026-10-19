# SPDX-License-Identifier: Apache-2.0
"""Which files matter most: PageRank over the codemap's reference graph.

WHY THIS EXISTS, 2026-10-19
---------------------------
The architecture block in every cached prefix listed the FIRST 60 FILES
ALPHABETICALLY and then "… and N more". On a project of 200 files a model
was shown `a*` to `f*` and nothing of the code everything else calls.
Debug & Repair's locate step offered the model 80 files in alphabetical
order; the audit chose which files a model reads by the scanners' findings
and then by size.

Aider's repo map answers the same question with PageRank over a graph of
files, an edge from each file to the files whose names it uses, and a
"personalization" toward the files and identifiers the current request
mentions. The codemap already holds that graph — the `edges` table binds
every call and import to the symbol it reaches — so this is the ranking,
not a second index.

TWO RANKINGS, ONE GRAPH:

  * **The global rank** orders the architecture block. It depends only on
    the graph, so it changes only when the code does, and the block is
    snapshotted per epoch anyway: byte-identical within an epoch (M52).
    Files beyond the cap are the LEAST referenced, not the latest in the
    alphabet; the block is still printed in path order, so a reader (and
    the cache) sees a stable list.
  * **The personalized rank** — toward the names and paths a piece of text
    mentions (an error, a description, an audit focus) — orders the files a
    model is OFFERED when it must choose. It is never in the prefix.

Pure Python and deterministic (ties break by path): no networkx, because a
package with no required dependencies does not take one for forty lines.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re

DAMPING = 0.85
ITERATIONS = 60
TOLERANCE = 1e-10

#: File names that are where a program starts: a reader starts there too.
ENTRY_NAMES = ("main.py", "__main__.py", "app.py", "cli.py", "run.py",
               "main.c", "main.cpp", "main.rs", "main.go", "index.js",
               "main.js", "main.ts", "index.ts", "program.cs", "main.gd")


def pagerank(nodes: Iterable[str], edges: Mapping[tuple[str, str], float],
             personalization: Mapping[str, float] | None = None, *,
             damping: float = DAMPING, iterations: int = ITERATIONS
             ) -> dict[str, float]:
    """{node: score}, summing to 1. ``edges`` maps (from, to) to a weight.

    Dangling nodes (no outgoing edge) give their rank back through the
    personalization vector, as networkx does — so a project of files that
    import nothing still ranks by what was mentioned, not uniformly.
    """
    order = sorted(set(nodes) | {a for a, _ in edges} | {b for _, b in edges})
    if not order:
        return {}
    n = len(order)
    pos = {name: i for i, name in enumerate(order)}
    out_weight = [0.0] * n
    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (a, b), w in edges.items():
        if a == b or w <= 0:
            continue
        ia, ib = pos[a], pos[b]
        out_weight[ia] += w
        incoming[ib].append((ia, w))
    if personalization:
        raw = [max(0.0, float(personalization.get(name, 0.0)))
               for name in order]
        total = sum(raw)
        p = [x / total for x in raw] if total > 0 else [1.0 / n] * n
    else:
        p = [1.0 / n] * n
    rank = list(p)
    for _ in range(max(1, iterations)):
        dangling = sum(rank[i] for i in range(n) if out_weight[i] == 0)
        new = [0.0] * n
        for i in range(n):
            s = 0.0
            for j, w in incoming[i]:
                s += rank[j] * w / out_weight[j]
            new[i] = damping * (s + dangling * p[i]) + (1 - damping) * p[i]
        delta = sum(abs(a - b) for a, b in zip(new, rank, strict=True))
        rank = new
        if delta < TOLERANCE:
            break
    return dict(zip(order, rank, strict=True))


def ordered(scores: Mapping[str, float]) -> list[str]:
    """Best first; ties by path, so the order is the same every time."""
    return sorted(scores, key=lambda k: (-round(scores[k], 12), k))


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def mentions(text: str) -> set[str]:
    """The identifiers a piece of text names (three characters or more)."""
    return set(_WORD.findall(text or ""))


def personalization(files: Iterable[str], defined: Mapping[str, Iterable[str]],
                    text: str = "") -> dict[str, float]:
    """A weight per file: entry points a little, and — from ``text`` — the
    files it names by path or stem, and the files that define a name it
    uses, more for a name few files define (Aider's rule: a name ten files
    define says little about which one is meant)."""
    files = list(files)
    words = mentions(text)
    lowered = (text or "").lower().replace("\\", "/")
    weight = dict.fromkeys(files, 0.0)
    for f in files:
        base = f.replace("\\", "/").rsplit("/", 1)[-1]
        if base.lower() in ENTRY_NAMES:
            weight[f] += 1.0
        if lowered:
            stem = base.rsplit(".", 1)[0].lower()
            # whole names only: "src/scenario.py" is not "io.py"
            if re.search(rf"(?<![\w.-]){re.escape(f.lower())}(?![\w])",
                         lowered) or re.search(
                    rf"(?<![\w.-]){re.escape(base.lower())}(?![\w])",
                    lowered):
                weight[f] += 10.0
            elif len(stem) >= 4 and re.search(
                    rf"(?<![\w]){re.escape(stem)}(?![\w])", lowered):
                weight[f] += 5.0
    if words:
        owners: dict[str, list[str]] = {}
        for f, names in defined.items():
            for name in names:
                short = name.rsplit(".", 1)[-1]
                if short in words:
                    owners.setdefault(short, []).append(f)
        for name, where in owners.items():
            share = 8.0 / len(set(where))
            if name.startswith("_"):
                share *= 0.3
            for f in set(where):
                if f in weight:
                    weight[f] += share
    if not any(weight.values()):
        return {}
    return {f: w + 0.01 for f, w in weight.items()}
