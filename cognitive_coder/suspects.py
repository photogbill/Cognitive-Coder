# SPDX-License-Identifier: Apache-2.0
"""An assertion the model believes is wrong, said once at the top of its
reply — and what the loop does with it.

Bill, 2026-10-24, watching ATK's Live tab: *"i think something isn't
working right, and the new live screen let me see."* A repair of
`src/physics.py` against `tests/test_physics.py` — a test this engine had
written — went back and forth for 17,620 characters and never wrote the
file. Two of the test's assertions were wrong (one asserted 90.0 under its
own comment saying -90; one expected a torpedo moving 50 units to land
within 35 of a target 10 away) and the rest were right, with real bugs in
the module behind them. The prompt offered one way out: "If you are certain
the test itself is wrong, return this file unchanged". All or nothing — for
a file that needed both a fix and a refusal, there was nowhere to stand.

So the repair prompt now asks for both at once: fix the file for every test
that is right, and name each assertion believed wrong on a line of its own
at the top of the reply,

    SUSPECT TEST: test_shortest_angle_diff — asserts 90.0 under its own
    comment saying the difference is -90

decided once. This module reads those lines out of the reply (so they never
reach the file), and answers the two questions the loop then asks: which
tests are still failing, and are they all ones the model named. When they
are, another repair of the module cannot help; the suspects go to the judge
(adjudicate.py) with the model's reasons, and to the operator.

Stdlib only, no I/O.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re

#: `SUSPECT TEST: name — why` (also `SUSPECT CALLER:`), alone on its line,
#: bulleted, bolded, commented or quoted as a model may write it.
_DASHES = "–—"
_LINE = re.compile(
    r"^[ \t]*(?:[-*>#]+[ \t]*|//[ \t]*)?\**[ \t]*SUSPECT[ _](?P<what>TEST|"
    r"CALLER)\**[ \t]*[:\-" + _DASHES + r"]+[ \t]*\**[ \t]*`?(?P<name>[^\s`"
    + _DASHES + r":]+?(?::\d+)?)`?\**[ \t]*(?:(?:[" + _DASHES
    + r":]|--?|\bbecause\b)[ \t]*(?P<why>.*?))?[ \t]*$", re.I | re.M)
#: unittest's `FAIL: test_x (pkg.mod.Class.test_x)` and pytest's
#: `FAILED path::Class::test_x - …` / `path::test_x FAILED`
_UNITTEST = re.compile(r"^(?:FAIL|ERROR):\s+(?P<name>[\w.]+)\s+\((?P<id>[^)]*)\)",
                       re.M)
_PYTEST = re.compile(r"^(?:FAILED|ERROR)\s+\S*?::(?P<name>[\w\[\]-]+)"
                     r"(?:\s|$)|::(?P<name2>\w+)\s+(?:FAILED|ERROR)\b", re.M)


@dataclass(frozen=True)
class Suspect:
    test: str                  # the test's name as the model wrote it
    why: str = ""
    what: str = "test"         # "test" | "caller"

    @property
    def key(self) -> str:
        """The bare test function name: `TestX.test_y` and
        `tests/test_a.py::test_y` both name `test_y`."""
        return short_name(self.test)

    def line(self) -> str:
        return f"{self.test} — {self.why}" if self.why else self.test


def short_name(name: str) -> str:
    n = str(name or "").strip().strip("`'\"()")
    n = n.split("::")[-1]
    n = n.split(".")[-1]
    n = n.split("[")[0]
    return n.strip()


def split(text: str) -> tuple[str, tuple[Suspect, ...]]:
    """(the reply without its SUSPECT lines, the suspects, in order and
    once each). A reply without any comes back unchanged."""
    if not text or "suspect" not in text.lower():
        return text, ()
    found: list[Suspect] = []
    seen: set = set()

    def take(m: re.Match) -> str:
        name = (m.group("name") or "").strip().rstrip(".,;")
        if not name:
            return m.group(0)
        s = Suspect(test=name, why=(m.group("why") or "").strip(),
                    what=m.group("what").lower())
        if (s.what, s.key) not in seen:
            seen.add((s.what, s.key))
            found.append(s)
        return ""

    out = _LINE.sub(take, text)
    if not found:
        return text, ()
    # the lines are gone; do not leave the blank lines they stood on at the
    # very top of the reply
    return out.lstrip("\n"), tuple(found)


def failing_tests(outputs: Iterable[str]) -> list[str]:
    """The test functions a unittest / pytest run reports as failed or in
    error, by bare name, in order and once each. A module that failed to
    import (`unittest.loader._FailedTest`) is reported as `<import>`, which
    no suspect can cover."""
    out: list[str] = []
    for text in outputs:
        for m in _UNITTEST.finditer(text or ""):
            name = m.group("name").rsplit(".", 1)[-1]
            if "_FailedTest" in (m.group("id") or ""):
                name = "<import>"
            if name not in out:
                out.append(name)
        for m in _PYTEST.finditer(text or ""):
            name = short_name(m.group("name") or m.group("name2") or "")
            if name and name not in out:
                out.append(name)
    return out


def covers(suspects: Sequence[Suspect], failing: Sequence[str]) -> bool:
    """Every failing test is one the model named (and something failed)."""
    named = {s.key for s in suspects if s.what == "test"}
    return bool(failing) and bool(named) and all(f in named for f in failing)


def words(suspects: Sequence[Suspect]) -> str:
    """`test_a (why); test_b (why)` for a sentence an operator reads."""
    return "; ".join(f"{s.test} — {s.why}" if s.why else s.test
                     for s in suspects)
