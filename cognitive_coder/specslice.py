# SPDX-License-Identifier: Apache-2.0
"""The part of the request that concerns ONE file, cut by rule.

WHY THIS EXISTS, 2026-10-16
---------------------------
The Star Trek spec was 15,070 characters. Every diffusion prompt repeated
its first 2,000 (`canvas.stub_text`, `canvas.repair_text`) and so did every
whole-file repair (`personas.REPAIR_REQUEST_CHARS`). The formulas the
build was judged against — `atan2(x2 - x1, -(y2 - y1))` at character
7,363, the `+30°` sector offset at 7,904, `overflow * 0.25` at 8,717, the
subsystem keys, the skirmish positions — all sat past the cut. The model
writing `bearing_between` had the sentence "Navigational bearing in [0.0,
360.0) from (x1, y1) to (x2, y2) using math.atan2(x2 - x1, -(y2 - y1))" in
the request and never in its prompt. It wrote the formula from memory,
y-up, and was then repaired three times against tests that were also
written y-up.

A request this size fits a prompt; a 135,000-character design document
does not, and a repair prompt also carries the file, the test, the
interfaces and the errors. So the request is SLICED per file, by rule:

  * the opening section (the title and what the program is) is always kept;
  * sections that name the file — its path, its basename, its dotted
    module, its stem as a word, the file it tests or is tested by, or a
    name it declares — are kept first;
  * sections stating hard requirements (MUST / NEVER / …) come next; they
    are also pinned in every prompt's prefix by `constraints.py`, so here
    they only need to fit;
  * what does not fit is NAMED at the end ("not repeated here: …"), so the
    model knows the request goes on and the operator can see what the file
    was built without.

A request under the budget is passed whole. Sections are markdown headings
when the request has them, paragraphs when it does not. Nothing here is
paraphrased: every kept character is the operator's own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

#: Budgets, in characters. A first attempt sees more of the request than a
#: repair does: the repair prompt also carries the file as it stands, the
#: test and the errors, which are the point of a repair.
FIRST_CHARS = 16_000
REPAIR_CHARS = 8_000
#: The opening section is kept whole up to this; a long preamble is cut
#: here and said to be cut.
INTRO_CHARS = 1_500
#: A condensed section keeps only its lines that name the file, each cut
#: here. Long enough for a bullet of a file manifest.
LINE_CHARS = 400
#: Reserved for the closing note that names the sections left out.
NOTE_CHARS = 400

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_REQUIREMENT = re.compile(
    r"\b(MUST(\s+NOT)?|SHALL(\s+NOT)?|NEVER|ALWAYS|DO\s+NOT|DON'T|REQUIRED)\b")


@dataclass(frozen=True)
class Section:
    heading: str          # "" for the preamble before the first heading
    text: str             # heading line included
    index: int            # document order


@dataclass(frozen=True)
class Slice:
    """What a prompt is given, and what it was not."""
    text: str
    whole: bool
    kept: tuple[str, ...] = ()        # headings kept (whole or condensed)
    condensed: tuple[str, ...] = ()   # kept only as the lines naming the file
    left_out: tuple[str, ...] = ()
    chars: int = 0

    def describe(self) -> str:
        """One line for BUILD_LOG.txt."""
        if self.whole:
            return f"the whole request ({self.chars:,} characters)"
        bits = [f"{self.chars:,} of the request's characters"]
        if self.kept:
            bits.append(f"kept: {', '.join(_short(h) for h in self.kept)}")
        if self.condensed:
            bits.append("only the lines naming the file from: "
                        + ", ".join(_short(h) for h in self.condensed))
        if self.left_out:
            bits.append("left out: "
                        + ", ".join(_short(h) for h in self.left_out))
        return "; ".join(bits)


def _short(heading: str, n: int = 48) -> str:
    h = " ".join(heading.split())
    return h if len(h) <= n else h[:n - 1] + "…"


# --------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------

def sections(request: str) -> list[Section]:
    """Markdown sections (heading + body), or paragraphs for plain text.

    Fenced code is never split on: a `#` inside a fence is a comment, not
    a heading.
    """
    text = (request or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: list[Section] = []
    current: list[str] = []
    heading = ""
    fenced = False
    any_heading = False
    for line in lines:
        if line.strip().startswith(("```", "~~~")):
            fenced = not fenced
            current.append(line)
            continue
        m = None if fenced else _HEADING.match(line)
        if m is not None:
            any_heading = True
            if current and "".join(current).strip():
                out.append(Section(heading, "\n".join(current), len(out)))
            elif current and not out:
                pass                     # blank preamble: nothing to keep
            current = [line]
            heading = m.group(2).strip()
            continue
        current.append(line)
    if current and "".join(current).strip():
        out.append(Section(heading, "\n".join(current), len(out)))
    if any_heading:
        return out
    # Plain text: paragraphs, each its own section with its first words as
    # the "heading" the log can name.
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [Section(_short(p.strip().split("\n")[0], 60), p, i)
            for i, p in enumerate(paras)]


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def file_tokens(path: str) -> list[tuple[re.Pattern, int]]:
    """How a request names a file, with a weight for each spelling.

    `src/physics.py` is named as that path, as `physics.py`, as
    `src.physics`, and as the word `physics`. The bare stem is a whole
    word and counts least: `main`, `test` and `utils` name half the files
    in any project, so a very short or generic stem is not used at all.
    """
    rel = str(path or "").replace("\\", "/").strip("/")
    if not rel:
        return []
    base = rel.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0] if "." in base else base
    dotted = (rel.rsplit(".", 1)[0] if "." in base else rel).replace("/", ".")
    out: list[tuple[re.Pattern, int]] = [
        (re.compile(re.escape(rel)), 3),
        (re.compile(re.escape(rel.replace("/", "\\"))), 3),
    ]
    if base != rel:
        out.append((re.compile(r"(?<![\w/])" + re.escape(base) + r"(?!\w)"), 3))
    if "." in dotted:
        out.append((re.compile(r"(?<![\w.])" + re.escape(dotted) + r"(?![\w])"), 3))
    if len(stem) >= 4 and stem not in _GENERIC_STEMS:
        out.append((re.compile(r"(?<![\w])" + re.escape(stem) + r"(?![\w])",
                               re.I), 1))
    return out


_GENERIC_STEMS = frozenset({"main", "test", "tests", "init", "util", "utils",
                            "core", "base", "app", "lib", "common", "index",
                            "setup", "config", "types", "model", "models"})

#: Anything shaped like a path with an extension: how a file manifest's
#: lines are recognised (`* src/models.py — shared dataclasses`).
_PATHLIKE = re.compile(r"(?<![\w/.\\-])(?:[\w.-]+[/\\])+[\w.-]+\.\w{1,4}")


def is_manifest(section: Section) -> bool:
    """A section that is mostly a LIST OF FILES — the request's manifest.

    Kept condensed (only the lines naming this file) whatever its score:
    the other nine bullets describe other files, and the 1,500 characters
    they cost were, on the Star Trek build, exactly the room the file's
    own section needed.
    """
    lines = [ln for ln in section.text.split("\n")[1:] if ln.strip()]
    if len(lines) < 3:
        return False
    named = {m.group(0) for ln in lines for m in _PATHLIKE.finditer(ln)}
    with_path = sum(1 for ln in lines if _PATHLIKE.search(ln))
    return len(named) >= 3 and with_path * 2 >= len(lines)


def score(section: Section, path: str, *, related: Sequence[str] = (),
          names: Sequence[str] = ()) -> int:
    """How much this section concerns the file. 0 means not at all."""
    text = section.text
    total = 0
    # A section HEADED with the file's name is the file's own section and
    # outranks everything, including the hard requirements (which are in
    # every prompt's prefix anyway and must not crowd it out).
    first = text.strip().split("\n", 1)[0]
    if _HEADING.match(first) and any(
            w >= 2 and p.search(first) for p, w in file_tokens(path)):
        total += 8
    # A test's slice is mostly about the module it tests: the section
    # headed with THAT file's name is the one a test author needs.
    if _HEADING.match(first) and any(
            w >= 2 and p.search(first)
            for other in related for p, w in file_tokens(other)):
        total += 6
    for pattern, weight in file_tokens(path):
        hits = len(pattern.findall(text))
        if hits:
            total += weight * min(hits, 3)
    for other in related:
        for pattern, weight in file_tokens(other):
            hits = len(pattern.findall(text))
            if hits:
                total += max(1, weight - 1) * min(hits, 2)
    for name in names:
        if len(name) < 4:
            continue
        if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text):
            total += 1
    if _REQUIREMENT.search(text):
        total += 1
    return total


# --------------------------------------------------------------------------
# the slice
# --------------------------------------------------------------------------

def slice_for(request: str, path: str, *, budget: int = REPAIR_CHARS,
              related: Sequence[str] = (), names: Sequence[str] = ()
              ) -> Slice:
    """The request as one file's prompt should carry it."""
    text = (request or "").strip()
    if len(text) <= budget:
        return Slice(text=text, whole=True, chars=len(text))
    parts = sections(text)
    if not parts:
        return Slice(text=text[:budget], whole=False, chars=budget)

    scored = {s.index: score(s, path, related=related, names=names)
              for s in parts}
    # Room for the closing note that names what was left out, so the
    # slice as sent stays inside the budget it was asked for.
    budget = max(budget - NOTE_CHARS, budget // 2)
    # The opening section is the title and what the program is: kept
    # first, cut if it is a long preamble.
    intro = parts[0]
    intro_text = intro.text.strip()
    if len(intro_text) > INTRO_CHARS:
        intro_text = intro_text[:INTRO_CHARS].rstrip() + "\n… (cut here)"
    chosen: dict[int, str] = {intro.index: intro_text}
    kept: list[str] = []
    condensed: list[str] = []
    used = len(intro_text)
    order = sorted(parts[1:], key=lambda s: (-scored[s.index], s.index))
    tokens = file_tokens(path) + [t for r in related for t in file_tokens(r)]
    for s in order:
        if scored[s.index] <= 0:
            continue
        body = s.text.strip()
        if is_manifest(s):
            # A list of files: only the lines naming this one, whatever
            # the score — the other bullets describe other files, and the
            # room they cost is what the file's own section needs.
            short = _condense(s, tokens)
            if short and used + len(short) + 2 <= budget:
                chosen[s.index] = short
                condensed.append(s.heading or _short(body.split("\n")[0], 60))
                used += len(short) + 2
            continue
        if used + len(body) + 2 <= budget:
            chosen[s.index] = body
            kept.append(s.heading or _short(body.split("\n")[0], 60))
            used += len(body) + 2
            continue
        if scored[s.index] < 3:
            continue               # named only in passing: not worth a cut
        short = _condense(s, tokens)
        if short and used + len(short) + 2 <= budget:
            chosen[s.index] = short
            condensed.append(s.heading or _short(body.split("\n")[0], 60))
            used += len(short) + 2
    left_out = [s.heading or _short(s.text.strip().split("\n")[0], 60)
                for s in parts[1:] if s.index not in chosen]
    pieces = [chosen[i] for i in sorted(chosen)]
    if left_out:
        pieces.append(f"(The request continues; not repeated here: "
                      f"{'; '.join(_short(h) for h in left_out)}. Its hard "
                      f"requirements are pinned above.)")
    out = "\n\n".join(pieces)
    return Slice(text=out, whole=False, kept=tuple(kept),
                 condensed=tuple(condensed), left_out=tuple(left_out),
                 chars=len(out))


def _condense(section: Section, tokens: Sequence[tuple[re.Pattern, int]]
              ) -> str:
    """The heading and only the lines that name the file."""
    lines = section.text.strip().split("\n")
    head = lines[0] if _HEADING.match(lines[0]) else ""
    keep: list[str] = []
    for line in (lines[1:] if head else lines):
        if any(p.search(line) for p, w in tokens if w >= 2):
            cut = line.strip()
            if len(cut) > LINE_CHARS:
                cut = cut[:LINE_CHARS - 1].rstrip() + "…"
            keep.append(cut)
    if not keep:
        return ""
    body = "\n".join(keep)
    return f"{head}\n(only the lines naming this file:)\n{body}" if head \
        else body
