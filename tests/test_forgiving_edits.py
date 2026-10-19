# SPDX-License-Identifier: Apache-2.0
"""A forgiving anchor, strict about where it lands (patcher.py).

Aider measured nine times more failed edits with its flexible matching
turned off. These are its fallbacks — indentation, trailing spaces, blank
edge lines, `...` pieces — held to this module's rule that a match is
whole words (here: whole lines) and unique, plus the feedback that makes
a retry land: every match's line when there are several, the lines the
model probably meant when there are none, "already applied" when it is.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.patcher import Patcher, forgiving_replace  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
)
from cognitive_coder.types import Edit  # noqa: E402

FILE = ("class Car:\n"
        "    def update(self, dt):\n"
        "        self.speed += dt\n"
        "        return self.speed\n"
        "\n"
        "    def stop(self):\n"
        "        self.speed = 0\n")


def test_an_exact_anchor_is_used_as_it_always_was():
    got, how, why = forgiving_replace(FILE, "self.speed = 0",
                                      "self.speed = 0.0")
    assert got is not None and how == "" and why == ""
    assert "self.speed = 0.0" in got


def test_an_anchor_written_at_the_wrong_indentation_still_lands():
    old = "def update(self, dt):\n    self.speed += dt\n"
    new = "def update(self, dt):\n    self.speed += 2 * dt\n"
    got, how, _ = forgiving_replace(FILE, old, new)
    assert got is not None, "the model dropped the class indentation"
    assert "indentation" in how
    assert "        self.speed += 2 * dt\n" in got
    assert "    def update(self, dt):\n" in got


def test_trailing_spaces_in_the_anchor_do_not_matter():
    got, how, _ = forgiving_replace(
        FILE, "        self.speed = 0   \n", "        self.speed = -1\n")
    assert got is not None and "trailing" in how
    assert "self.speed = -1" in got


def test_dots_stand_for_the_code_between_two_pieces():
    old = "    def update(self, dt):\n...\n        return self.speed\n"
    new = ("    def update(self, dt):\n...\n"
           "        return round(self.speed, 2)\n")
    got, how, _ = forgiving_replace(FILE, old, new)
    assert got is not None and "piece" in how, how
    assert "round(self.speed, 2)" in got
    assert "self.speed += dt" in got          # what the dots stood for


def test_several_matches_are_named_by_line_and_refused():
    text = "x = 1\nprint(x)\nx = 1\n"
    got, _how, why = forgiving_replace(text, "x = 1", "x = 2")
    assert got is None
    assert "lines 1, 3" in why


def test_a_missing_anchor_offers_the_lines_it_probably_meant():
    got, _how, why = forgiving_replace(
        FILE, "    def update(self, dt):\n        self.speed += dt * 1\n",
        "x")
    assert got is None
    assert "Did you mean these lines, from line 2?" in why
    assert "self.speed += dt" in why


def test_an_edit_already_applied_is_said_to_be():
    got, _how, why = forgiving_replace(FILE, "self.speed = 99",
                                       "self.speed = 0")
    assert got is None and "applied already" in why


def test_a_fallback_never_lands_inside_a_name():
    """The whole-word rule survives: fallbacks match WHOLE lines."""
    text = "max_x = 10\n"
    got, _how, _why = forgiving_replace(text, "x = 10", "x = 20")
    assert got is None


def test_the_patcher_says_how_it_matched(tmp_path):
    fs = MemoryFileSystem()
    fs.write("car.py", FILE)
    patcher = Patcher(fs, MemoryStorage(str(tmp_path)), AutoApprove(),
                      RecordingEvents())
    with patcher.begin("t1") as tx:
        res = tx.apply([Edit(path="car.py", kind="replace",
                             old="def stop(self):\n    self.speed = 0\n",
                             new="def stop(self):\n    self.speed = 0.0\n")])
    assert res[0].ok, res[0].reason
    assert "indentation" in res[0].reason
    assert "        self.speed = 0.0\n" in fs.read("car.py")
