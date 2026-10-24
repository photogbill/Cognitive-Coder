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

GOING IN CIRCLES (2026-10-24). Bill, the same day, with the Live tab
showing it: *"i think something isn't working right, and the new live
screen let me see."* A repair against a test that contradicted itself: the
model wrote 18,000 characters of "Wait, … No. … Actually, … I'll stick to
the standard … I'll change it to abs(diff)? No." — deciding and undeciding,
and never the file. Nothing above sees that: no stretch of it repeats word
for word. `circling` does, from three signals over the last
`CIRCLE_WINDOW` characters of PROSE (fenced code is set aside — a file full
of `"No."` strings is data, not doubt):

  * **reversals** — sentences that begin "Wait", "No.", "Actually", "Hmm",
    "Or maybe", "Let me re-read", "Let's look at … again", and phrases such
    as "I'll stick to", "I'll keep it as is", "on second thought", "that
    doesn't make sense". Counted, per thousand characters, and by KIND: a
    FAQ answering "No." thirty times is one kind of marker, not doubt;
  * **repeated sentences** — the same sentence (20+ characters, inline code
    normalised) said again in the window;
  * **no file** — characters since the last fenced block long enough to be
    the file (`FILE_LINES` lines); a block being written that long is
    progress, and nothing is flagged while it grows.

`watch` from `CIRCLE_WATCH_SINCE` characters without a file, with
`CIRCLE_WATCH_REVERSALS` reversals of two kinds or more at
`CIRCLE_WATCH_DENSITY` per thousand; `loop` (a host may cut) needs all of:
`CIRCLE_CUT_SINCE` characters without a file, `CIRCLE_CUT_REVERSALS`
reversals of three kinds or more at `CIRCLE_CUT_DENSITY` per thousand, and
`CIRCLE_CUT_REPEATS` sentences said again. Streamed through a `Watch`
(read for doubt every `CIRCLE_EVERY` characters), Bill's reply goes amber
at 5,200 characters and is cut at 10,400 of its 17,620; careful reasoning
that corrects itself twice in ten thousand characters, a linear
walk-through, a test-by-test check and a FAQ of "No." answers never reach
amber (tests/test_repetition.py). The result is a `Repetition` of
``kind="dithering"`` with nothing to trim: ``keep`` is the whole reply.

Stdlib only, no I/O, no clock. `find` is the whole idea; `Watch` is the
small stateful wrapper a host feeds a stream into.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

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

# -- going in circles (2026-10-24) -------------------------------------------
#: How much of the end of the reply is read for doubt.
CIRCLE_WINDOW = 6000
#: Prose in that window before anything is judged at all.
CIRCLE_MIN_PROSE = 2500
#: How far back fences are paired. Further back, only their number is
#: counted (at C speed), to know whether the scan starts inside one.
CIRCLE_SCAN = 40000
#: A fenced block of this many lines is the file being written: progress.
FILE_LINES = 25
#: amber: characters without a file, reversals, per-thousand density, kinds
CIRCLE_WATCH_SINCE = 3000
CIRCLE_WATCH_REVERSALS = 8
CIRCLE_WATCH_DENSITY = 2.0
CIRCLE_WATCH_KINDS = 2
#: red (a host may cut): all of these at once
CIRCLE_CUT_SINCE = 10000
CIRCLE_CUT_REVERSALS = 20
CIRCLE_CUT_DENSITY = 3.0
CIRCLE_CUT_KINDS = 3
CIRCLE_CUT_REPEATS = 2
#: The words a host uses when it cuts for this, and the engine looks for in
#: `Completion.error` to add its nudge to the next attempt.
CIRCLES_REASON = "it was going in circles"
#: A stream is read for doubt at most this often (characters): it moves
#: slowly, and the sentence pass is the costly part of a check.
CIRCLE_EVERY = 1024
#: What the next attempt is told after such a cut.
CIRCLES_NUDGE = ("Your last reply went back and forth without deciding. "
                 "Decide once: fix what the tests that are right require; "
                 "list any assertion you believe is wrong as SUSPECT TEST "
                 "lines; then write the file.")


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
    #: "verbatim" (a unit copied out again and again) or "dithering" (going
    #: in circles: no unit, nothing to trim — ``keep == end``)
    kind: str = "verbatim"
    reversals: int = 0          # dithering: reversal markers in the window
    repeats: int = 0            # dithering: sentences said again
    since_file: int = 0         # dithering: characters without a file
    markers: tuple = ()         # dithering: the commonest marker kinds

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

    @property
    def dithering(self) -> bool:
        return self.kind == "dithering"

    def describe(self) -> str:
        """One phrase an operator reads: "the same 3 lines, 12 times over"."""
        if self.dithering:
            said = ", ".join(self.markers[:3]) or "wait, no, actually"
            return (f"going in circles — {self.reversals} reversals "
                    f"({said} …)"
                    + (f" and {self.repeats} sentence"
                       f"{'' if self.repeats == 1 else 's'} said again"
                       if self.repeats else "")
                    + f", {self.since_file:,} characters without the file")
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
        out = {"start": self.start, "keep": self.keep, "end": self.end,
               "period": self.period, "copies": round(self.copies, 1),
               "repeated": self.repeated, "level": self.level,
               "describe": self.describe(), "unit": self.unit[:400],
               "kind": self.kind}
        if self.dithering:
            out.update(reversals=self.reversals, repeats=self.repeats,
                       since_file=self.since_file,
                       markers=list(self.markers))
        return out


def find(text: str, *, watch_chars: int = WATCH_CHARS,
         loop_chars: int = LOOP_CHARS, min_copies: float = MIN_COPIES,
         min_letters: int = MIN_LETTERS, max_period: int = MAX_PERIOD,
         window: int = WINDOW, circles: bool = True) -> Repetition | None:
    """The repetition the text ENDS in, or None.

    None also when there is repetition that does not count (see the module
    docstring): too short, fewer than three copies, a unit with nothing to
    say, or a tail too featureless to probe right now.

    Verbatim repetition first; then, with ``circles`` (the default), a reply
    going in circles (`circling`).
    """
    found = _verbatim(text, watch_chars=watch_chars, loop_chars=loop_chars,
                      min_copies=min_copies, min_letters=min_letters,
                      max_period=max_period, window=window)
    if found is None and circles:
        found = circling(text)
    return found


def _verbatim(text: str, *, watch_chars: int, loop_chars: int,
              min_copies: float, min_letters: int, max_period: int,
              window: int) -> Repetition | None:
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


# -- going in circles --------------------------------------------------------

_FENCE_LINE = re.compile(r"^[ \t]*```", re.M)
_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])[ \t]+|\n+")
_LEAD = re.compile(r"^[\s>*#\-\u2013\u2014\u2022\d.)(]*")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
#: Markers that count only at the START of a sentence ("No." mid-sentence
#: is a list item or a number, not a change of mind).
_STARTS = (
    ("wait", re.compile(r"^(?:but\s+|oh\s+|hold\s+on,?\s+)?wait\b", re.I)),
    ("no", re.compile(r"^no(?:[.,!:;]|$)", re.I)),
    ("actually", re.compile(r"^actually\b", re.I)),
    ("hmm", re.compile(r"^hm+\b", re.I)),
    ("maybe", re.compile(r"^(?:or\s+maybe|maybe\s+not|or\s+perhaps)\b",
                         re.I)),
    ("re-read", re.compile(
        r"^let(?:\s+me|'?s|\s+us)\s+(?:re-?(?:read|check|examine|consider|"
        r"think|look)|reconsider|look\s+(?:at\s+)?\S.*\bagain\b|"
        r"check\s+\S.*\bagain\b)", re.I)),
)
#: One cheap pass that every reversal marker also matches (and more): fewer
#: hits than the amber threshold and the sentence pass is not needed.
_ROUGH = re.compile(
    r"\b(?:wait|no[.,!:;]|actually|hm+|maybe|perhaps|re-?(?:read|check|"
    r"examine|consider|think|look)|reconsider|again|stick|as\s+(?:it\s+)?is|"
    r"second\s+thought|contradictory|make\s+sense|fails)", re.I)
#: Markers that count anywhere in a sentence.
_ANYWHERE = (
    ("stick to", re.compile(
        r"\bi(?:'ll|\s+will)\s+(?:just\s+)?stick\s+(?:to|with)\b", re.I)),
    ("keep it as is", re.compile(
        r"\bi(?:'ll|\s+will)\s+(?:just\s+)?(?:keep|leave)\s+(?:it|this|"
        r"that|the\s+code)\s+as\s+(?:is|it\s+is)\b", re.I)),
    ("on second thought", re.compile(r"\bon\s+second\s+thought\b", re.I)),
    ("contradiction", re.compile(
        r"\b(?:this|that|it)\s+(?:is|seems)\s+contradictory\b|"
        r"\bdoesn'?t\s+make\s+sense\b", re.I)),
    ("if it fails", re.compile(r"\bif\s+it\s+fails,?\s+i(?:'ll|\s+will)\b",
                               re.I)),
)


def _fences(text: str, ticks_before: int | None = None
            ) -> list[tuple[int, int, int, bool]]:
    """(start, end, lines, still open) for each fenced block in the last
    `CIRCLE_SCAN` characters, paired in order; the last one may still be
    being written. A block that began before the scan is taken as starting
    where the scan does — a stream of megabytes is never re-read whole."""
    lo = max(0, len(text) - CIRCLE_SCAN)
    pos = [m.start() for m in _FENCE_LINE.finditer(text, lo)]
    if lo:
        ticks = text.count("```", 0, lo) if ticks_before is None \
            else ticks_before
        if ticks % 2:
            pos.insert(0, lo)
    out = []
    for i in range(0, len(pos), 2):
        s = pos[i]
        if i + 1 < len(pos):
            e = pos[i + 1]
            eol = text.find("\n", e)
            eol = len(text) if eol < 0 else eol
            out.append((s, eol, text.count("\n", s, e), False))
        else:
            out.append((s, len(text), text.count("\n", s), True))
    return out


def circling(text: str, *, window: int = CIRCLE_WINDOW,
             ticks_before: int | None = None) -> Repetition | None:
    """A reply going in circles — deciding and undeciding, and no file —
    or None. See the module docstring for the three signals and why each
    threshold is where it is.

    ``ticks_before``: how many ``` occur before the last `CIRCLE_SCAN`
    characters, when the caller keeps count (a `Watch` does), so a long
    stream is not counted from the start at every check."""
    n = len(text or "")
    if n < CIRCLE_WATCH_SINCE:
        return None
    blocks = _fences(text, ticks_before)
    since = n
    for s, e, lines, still_open in blocks:
        if lines >= FILE_LINES:
            since = 0 if still_open else n - e
    if since < CIRCLE_WATCH_SINCE:
        return None
    base = max(0, n - window)
    parts, cur = [], base
    for s, e, _lines, _open in blocks:
        if e <= base:
            continue
        if s > cur:
            parts.append(text[cur:s])
        cur = max(cur, e)
    if cur < n:
        parts.append(text[cur:n])
    prose = "\n".join(parts)
    if len(prose) < CIRCLE_MIN_PROSE:
        return None
    rough = 0
    for _m in _ROUGH.finditer(prose):
        rough += 1
        if rough >= CIRCLE_WATCH_REVERSALS:
            break
    if rough < CIRCLE_WATCH_REVERSALS:
        return None
    kinds: Counter = Counter()
    said: Counter = Counter()
    for raw in _SENTENCE_SPLIT.split(prose):
        sentence = _LEAD.sub("", raw).strip()
        if not sentence:
            continue
        for kind, rx in _STARTS:
            if rx.search(sentence):
                kinds[kind] += 1
                break
        for kind, rx in _ANYWHERE:
            hits = len(rx.findall(sentence))
            if hits:
                kinds[kind] += hits
        norm = " ".join(_INLINE_CODE.sub("`x`", sentence.lower()).split())
        if len(norm) >= 20:
            said[norm] += 1
    reversals = sum(kinds.values())
    density = reversals / max(1.0, len(prose) / 1000.0)
    repeats = sum(k - 1 for k in said.values() if k > 1)
    if (since >= CIRCLE_CUT_SINCE and reversals >= CIRCLE_CUT_REVERSALS
            and density >= CIRCLE_CUT_DENSITY
            and len(kinds) >= CIRCLE_CUT_KINDS
            and repeats >= CIRCLE_CUT_REPEATS):
        level = "loop"
    elif (reversals >= CIRCLE_WATCH_REVERSALS
          and density >= CIRCLE_WATCH_DENSITY
          and len(kinds) >= CIRCLE_WATCH_KINDS):
        level = "watch"
    else:
        return None
    return Repetition(start=base, keep=n, end=n, period=0, unit="",
                      level=level, kind="dithering", reversals=reversals,
                      repeats=repeats, since_file=since,
                      markers=tuple(k for k, _ in kinds.most_common(4)))


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

    def __init__(self, *, every: int = 256, circles: bool = True,
                 circle_every: int = CIRCLE_EVERY, **thresholds) -> None:
        self.every = max(1, int(every))
        self._kw = thresholds
        self._parts: list[str] = []
        self._len = 0
        self._checked = 0
        self.found: Repetition | None = None
        #: going in circles is read less often than verbatim repetition,
        #: and its last answer stands between readings
        self._circles = bool(circles)
        self._circle_every = max(self.every, int(circle_every))
        self._circled = 0
        self._circle: Repetition | None = None
        #: ``` counted in the text before the circle scan, kept up to date
        #: a stretch at a time
        self._ticks = 0
        self._ticked = 0

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
        text = self.text
        found = find(text, circles=False, **self._kw)
        if self._circles:
            if self._len - self._circled >= self._circle_every:
                self._circled = self._len
                lo = max(0, self._len - CIRCLE_SCAN)
                if lo > self._ticked:
                    # from two back: a ``` split by the last boundary was
                    # not whole then, and is counted now
                    self._ticks += text.count("```", max(0, self._ticked - 2),
                                              lo)
                    self._ticked = lo
                self._circle = circling(text, ticks_before=self._ticks)
            if found is None:
                found = self._circle
        self.found = found
        return before is not None or self.found is not None
