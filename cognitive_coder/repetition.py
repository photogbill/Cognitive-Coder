# SPDX-License-Identifier: Apache-2.0
"""Is a reply stuck repeating itself? Answered while it is still being written.

Bill, 2026-10-24: a tab "that shows any code being written as it's being
written … so it can be stopped if the user notices that it's stuck in a loop
or repeating itself."

THE FAILURE. A 24B–40B model at temperature 0.15 can fall into a loop: the
same lines, word for word, until it reaches the token ceiling. At 10 tokens
a second an 8,192-token ceiling is thirteen minutes, and what comes back is
not a file anyone can use. Nothing could see it happening: the loop was
shown a reply only once the reply was finished.

WHAT COUNTS, AND WHAT DOES NOT. Only VERBATIM repetition at the END of the
text — the reply's last stretch is one unit (a line, a block, a sentence)
copied out again and again, and still going. Four rules keep real code from
being mistaken for a loop:

  * **It must be the tail.** A block repeated in the middle of a file and
    then left behind is a pattern the model finished, not a loop it is in.
  * **The unit must say something.** It needs four different letters. A
    `# ------` rule, a row of zeros in a tile map, a run of `}` lines are
    layout or data, not a model that has lost its place — a 40 × 40 grid
    of identical rows is perfectly periodic, and this rule lets it through.
  * **Three copies.** Two copies of a function is a model that wrote it
    twice: a different failure, caught after the fact. Three is a loop.
  * **Real length.** The repeated part must be long enough that acting on
    it saves more than it risks: `watch` (show it) from 400 characters,
    `loop` (a host may cut the reply) from 2,000. A false cut costs a whole
    attempt; a late one costs only the seconds between the two thresholds,
    so the defaults favour patience.

HOW. The last 32 characters of the reply are the probe. If the tail is
periodic with period p, the probe also appears exactly p characters
earlier — so each earlier occurrence of the probe names a candidate period,
and nothing else needs trying. For each, the run is the longest common
suffix of the text and the text shifted by p, found by galloping and halving
with slice comparisons (C speed, not a Python loop per character). The
longest run wins, which picks the true period over its multiples and over a
coincidence. A probe that is mostly blank or punctuation (deep indentation,
a comment rule) recurs everywhere and says nothing, so that check is skipped;
the next one, a few tokens later, has a different tail.

Stdlib only, no I/O, no clock. `find` is the whole idea; `Watch` is the
small stateful wrapper a host feeds a stream into.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Characters of repetition before a host should SHOW it (amber).
WATCH_CHARS = 400
#: Characters of repetition before a host may CUT the reply (red).
LOOP_CHARS = 2000
#: Copies of the unit, the first included, before either level applies.
MIN_COPIES = 3.0
#: Different letters the repeating unit must contain to count at all.
MIN_LETTERS = 4
#: The longest unit looked for, in characters (a block of ~60 lines).
MAX_PERIOD = 3000
#: How much of the end of the reply is examined.
WINDOW = 12000
#: The probe: this many characters at the very end of the reply.
PROBE = 32
#: Candidate periods examined per check, at most.
MAX_CANDIDATES = 48
#: A probe needs this many different non-blank characters to be worth
#: searching for.
PROBE_VARIETY = 5


@dataclass(frozen=True)
class Repetition:
    """The reply's tail, repeating. Offsets are into the whole reply.

    ``text[:keep]`` is the reply with ONE copy of the unit left in — what
    is worth keeping when the rest is cut. ``text[keep:end]`` is the part
    that only repeats, which is what a host highlights.
    """
    start: int          # where the repeating run begins (its first copy)
    keep: int           # where the redundant copies begin
    end: int            # the reply's length when this was found
    period: int         # characters in one copy of the unit
    unit: str           # one copy, as it first appeared
    level: str          # "watch" | "loop"

    @property
    def repeated(self) -> int:
        """Characters that are only copies of earlier text."""
        return max(0, self.end - self.keep)

    @property
    def copies(self) -> float:
        """How many times the unit appears in the run, the first included."""
        return (self.end - self.start) / self.period if self.period else 0.0

    @property
    def looping(self) -> bool:
        return self.level == "loop"

    def describe(self) -> str:
        """One phrase an operator reads: "the same 3 lines, 12 times over"."""
        times = int(self.copies)
        lines = self.unit.count("\n")
        if lines == 1:
            what = "the same line"
        elif lines > 1:
            what = f"the same {lines} lines"
        elif len(self.unit.split()) > 1:
            what = "the same phrase"
        else:
            what = "the same word"
        return f"{what}, {times} times over"

    def to_dict(self) -> dict:
        """Plain data for a host's UI or log."""
        return {"start": self.start, "keep": self.keep, "end": self.end,
                "period": self.period, "copies": round(self.copies, 1),
                "repeated": self.repeated, "level": self.level,
                "describe": self.describe(), "unit": self.unit[:400]}


def find(text: str, *, watch_chars: int = WATCH_CHARS,
         loop_chars: int = LOOP_CHARS, min_copies: float = MIN_COPIES,
         min_letters: int = MIN_LETTERS, max_period: int = MAX_PERIOD,
         window: int = WINDOW) -> Repetition | None:
    """The repetition the text ENDS in, or None.

    None also when there is repetition that does not count (see the module
    docstring): too short, fewer than three copies, a unit with nothing to
    say, or a tail too featureless to probe right now.
    """
    n = len(text or "")
    if n < 2 * PROBE:
        return None
    base = max(0, n - window)
    tail = text[base:]
    m = len(tail)
    probe = tail[-PROBE:]
    if len({c for c in probe if not c.isspace()}) < PROBE_VARIETY:
        return None
    best_run = best_p = 0
    end = m - 1                 # an occurrence must end before the last char
    tried = 0
    while tried < MAX_CANDIDATES:
        i = tail.rfind(probe, 0, end)
        if i < 0:
            break
        end = i + PROBE - 1     # the next one starts further left
        p = (m - PROBE) - i
        if p > max_period:
            break
        if best_p and p % best_p == 0:
            continue            # a multiple of a period already measured
        tried += 1
        run = _shifted_suffix(tail, p)
        if run > best_run:      # a tie keeps the smaller period, found first
            best_run, best_p = run, p
    if not best_p or best_run < watch_chars:
        return None
    if (best_run + best_p) / best_p < min_copies:
        return None
    keep = n - best_run
    unit = text[keep - best_p:keep]
    if len({c.lower() for c in unit if c.isalpha()}) < min_letters:
        return None
    return Repetition(start=n - best_run - best_p, keep=keep, end=n,
                      period=best_p, unit=unit,
                      level="loop" if best_run >= loop_chars else "watch")


def _shifted_suffix(s: str, p: int) -> int:
    """The longest L with ``s[-L:] == s[-L-p:-p]`` — how far back the text
    stays equal to itself shifted by ``p``. Gallop, then halve."""
    m = len(s)
    limit = m - p
    if limit <= 0:
        return 0

    def same(k: int) -> bool:
        return s[m - k:] == s[m - k - p:m - p]

    lo = min(PROBE, limit)
    if not same(lo):
        return 0
    hi = 0
    while lo < limit:
        k = min(lo * 2, limit)
        if same(k):
            lo = k
        else:
            hi = k
            break
    if not hi:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if same(mid):
            lo = mid
        else:
            hi = mid
    return lo


class Watch:
    """`find`, for a stream. Feed it chunks; it checks every ``every``
    characters, so its cost does not grow with the token rate.

    ``found`` is the latest answer (None, a ``watch``, or a ``loop``);
    ``feed`` says when that answer changed, so a host redraws only then.
    """

    def __init__(self, *, every: int = 256, **thresholds) -> None:
        self.every = max(1, int(every))
        self._kw = thresholds
        self._parts: list[str] = []
        self._len = 0
        self._checked = 0
        self.found: Repetition | None = None

    def __len__(self) -> int:
        return self._len

    @property
    def text(self) -> str:
        if len(self._parts) > 1:
            self._parts = ["".join(self._parts)]
        return self._parts[0] if self._parts else ""

    @property
    def looping(self) -> bool:
        return self.found is not None and self.found.looping

    def feed(self, chunk: str) -> bool:
        """Add text. True when a check ran and there is something to show
        (a repetition, or one that has just ended)."""
        if not chunk:
            return False
        self._parts.append(chunk)
        self._len += len(chunk)
        if self._len - self._checked < self.every:
            return False
        self._checked = self._len
        before = self.found
        self.found = find(self.text, **self._kw)
        return before is not None or self.found is not None
