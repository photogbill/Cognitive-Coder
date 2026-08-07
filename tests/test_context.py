# SPDX-License-Identifier: Apache-2.0
"""§6.6 — what the model gets to see, and what it is told it did not.

Pure functions, no model, no filesystem. The outline extractors are here
because they are the layer everything else reasons on top of: an outline that
reports the wrong line number sends every subsequent edit to the wrong place,
and nothing downstream can notice.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import context as ctx  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem  # noqa: E402

PY_SOURCE = '''"""A module."""
import csv


def load(path: str) -> list:
    """Load rows."""
    return []


class Stats:
    """Holds statistics."""

    def mean(self, xs):
        """The mean."""
        return sum(xs) / len(xs)

    def _private(self):
        return None


async def fetch(url, *args, **kw):
    """Fetch it."""
    return None
'''


def test_python_symbols_are_exact_with_spans_and_docstrings():
    syms = {s.name: s for s in ctx.symbols(PY_SOURCE, "python")}
    assert syms["load"].line == 5
    assert syms["load"].signature == "def load(path: str) -> list"
    assert syms["load"].docstring == "Load rows."
    assert not syms["load"].approximate
    assert syms["Stats"].kind == "class"
    assert syms["Stats.mean"].kind == "method"
    assert syms["Stats.mean"].parent == "Stats"
    assert "*args" in syms["fetch"].signature
    assert "**kw" in syms["fetch"].signature
    assert syms["load"].end_line > syms["load"].line


def test_a_file_that_does_not_parse_still_yields_an_outline():
    """An outline is most wanted exactly when the file is broken."""
    broken = "def one():\n    return 1\n\ndef two(:\n    pass\n"
    syms = ctx.symbols(broken, "python")
    assert [s.name for s in syms] == ["one", "two"]
    assert all(s.approximate for s in syms), "the fallback must say so"


@pytest.mark.parametrize("lang,source,name,line", [
    ("javascript", "// header\n\nexport function alpha(a) {\n  return a;\n}\n",
     "alpha", 3),
    ("rust", "// x\n\npub fn parse(s: &str) -> u32 {\n    0\n}\n",
     "parse", 3),
    ("go", "package main\n\nfunc Handle(w int) {\n}\n", "Handle", 3),
    ("ruby", "# c\n\ndef run(a)\n  a\nend\n", "run", 3),
    ("gdscript", "extends Node\n\nfunc _ready() -> void:\n\tpass\n",
     "_ready", 3),
])
def test_line_numbers_are_right_for_every_regex_language(lang, source, name,
                                                         line):
    """`^\\s*` matches newlines in multiline mode, so an outline anchored
    with it reports every symbol one line early — and the error is invisible
    until somebody opens the file at the reported line."""
    syms = {s.name: s for s in ctx.symbols(source, lang)}
    assert name in syms, list(syms)
    assert syms[name].line == line


def test_regex_outlines_declare_that_they_are_approximate():
    out = ctx.outline("int add(int a, int b) {\n    return a + b;\n}\n", "c",
                      "u.c")
    assert "pattern-matched, not parsed" in out


def test_a_python_outline_makes_no_such_claim():
    out = ctx.outline(PY_SOURCE, "python", "m.py")
    assert "pattern-matched" not in out
    assert "def load(path: str) -> list" in out


def test_an_interface_is_signatures_without_bodies_or_privates():
    """F9 — a dependency costs ~30 tokens as an interface, ~800 as a file."""
    surface = ctx.interface(PY_SOURCE, "python", "m.py")
    assert "def load(path: str) -> list" in surface
    assert "Load rows." in surface
    assert "return []" not in surface
    assert "_private" not in surface
    assert len(surface) < len(PY_SOURCE)


def test_a_slice_carries_line_numbers_so_an_edit_can_be_located():
    text = "\n".join(f"line {n}" for n in range(1, 40))
    out = ctx.slice_around(text, 20, before=2, after=2)
    assert "   18 | line 18" in out
    assert "   22 | line 22" in out
    assert "line 10" not in out


def test_a_symbol_body_can_be_pulled_out_whole():
    body = ctx.symbol_body(PY_SOURCE, "load", "python")
    assert "def load" in body
    assert "return []" in body
    assert "class Stats" not in body


def test_context_keeps_essential_pieces_even_when_low_priority():
    out = ctx.build_context([
        ctx.Piece("bulky", "x" * 5000, priority=1),
        ctx.Piece("the task", "do the thing", priority=9, essential=True)],
        600)
    assert "do the thing" in out
    assert "bulky" in out.split("NOT INCLUDED")[1]


def test_the_omissions_block_names_every_dropped_piece():
    out = ctx.build_context([("A", "a" * 300, 1), ("B", "b" * 300, 2),
                             ("C", "c" * 300, 3)], 350)
    tail = out.split("NOT INCLUDED")[1]
    assert "B" in tail and "C" in tail


def test_a_budget_can_use_the_ports_token_counter():
    calls = []

    def count(text):
        calls.append(text)
        return len(text) // 10

    ctx.build_context([("A", "a" * 200, 1)], 5000, count_tokens=count)
    assert calls, "the supplied counter was ignored"


def test_the_project_map_skips_noise_and_says_how_much_it_showed():
    fs = MemoryFileSystem({
        "src/a.py": b"x = 1\n",
        "src/b.py": b"y = 2\n",
        ".venv/lib/junk.py": b"nope\n",
        "node_modules/pkg/index.js": b"nope\n",
        ".cc_snapshots/0001-t/files/src/a.py": b"old\n",
        "README.md": b"not code\n",
    })
    out = ctx.project_map(fs)
    assert "src/a.py" in out
    assert ".venv" not in out
    assert "node_modules" not in out
    assert ".cc_snapshots" not in out
    assert "README.md" not in out       # not a code file
    assert "2 code file(s)" in out


def test_relevant_files_rank_by_name_then_body_overlap():
    fs = MemoryFileSystem({
        "src/parser.py": b"def parse(line):\n    return line\n",
        "src/unrelated.py": b"def other():\n    return 1\n",
        "src/uses_parser.py": b"from src.parser import parse\n",
    })
    ranked = ctx.relevant_files(fs, "fix the parser")
    # Both files that mention the parser rank; the one that merely defines
    # `other()` does not. The heuristic is deliberately dumb — no embeddings,
    # no index to keep fresh — and getting the right two or three files in
    # front of the model is all it is for, so asserting an exact ORDER would
    # be asserting more than the function promises.
    assert "src/parser.py" in ranked[:2]
    assert "src/unrelated.py" not in ranked


def test_relevant_files_with_no_usable_query_returns_nothing():
    fs = MemoryFileSystem({"a.py": b"x = 1\n"})
    assert ctx.relevant_files(fs, "") == []
    assert ctx.relevant_files(fs, "a i") == []      # nothing 3 chars or more


def test_canonical_form_ignores_comments_and_whitespace():
    """M34 — or a reformat registers as a change and defeats the detector."""
    from cognitive_coder import textio
    a = "def f():\n    # a comment\n    return 1\n"
    b = "def f():\n\n        return 1\n\n# trailing note\n"
    assert textio.canonical(a) == textio.canonical(b)


def test_canonical_form_still_sees_a_real_change():
    from cognitive_coder import textio
    assert textio.canonical("return 1\n") != textio.canonical("return 2\n")


def test_the_budget_survives_a_port_that_raises():
    """A model port that throws must not take the budget calculation down."""
    class Broken:
        def capabilities(self):
            raise RuntimeError("the engine is not loaded")

    budget = ctx.measure_budget(Broken())
    assert budget.prompt_tokens > 0
    assert "could not be asked" in budget.note
