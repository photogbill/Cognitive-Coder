# SPDX-License-Identifier: Apache-2.0
"""What an independent review of the 2026-10-19 change found — each a test
that fails on the code as it was first written.

The splice and its ordering, a method that overwrote a function, the
guard switched off by ordinary wording and stalled by ordinary refactors,
refusals that piled up and pushed the real error out, a "did you mean"
search that took minutes, an unplaceable reply that ended a task, the
fresh start and narrowing fighting over attempt 3, a replay that always
"diverged" on a debug task.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import _host, _loop  # noqa: E402
from test_narrow import BIG, FIX, SEED, _task, _user  # noqa: E402

from cognitive_coder import bench, keep, personas  # noqa: E402
from cognitive_coder.codemap import rank  # noqa: E402
from cognitive_coder.diagnostics import where_block  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.patcher import _similar, forgiving_replace  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.types import Diagnostic, Task  # noqa: E402

TODO = '''class TodoList:
    def __init__(self):
        self.items = []

    def add(self, text):
        self.items.append(text)

    def summary(self):
        return f"{len(self.items)} items"
'''


def _fence(code: str) -> str:
    return f"```python\n{code}```"


# --------------------------------------------------------------------------
# the guard: ordinary wording does not switch it off …
# --------------------------------------------------------------------------

def test_a_request_to_add_remove_does_not_let_other_code_go():
    reply = TODO.replace('    def summary(self):\n        return f"{len(self'
                         '.items)} items"\n', "    def remove(self, i):\n"
                         "        return self.items.pop(i)\n")
    f = keep.check(TODO, reply, "python", path="todo.py",
                   request="Add a method `remove(index)` that deletes an "
                           "item. Keep everything else exactly as it is.")
    assert f.refused and f.dropped == ("TodoList.summary",)


def test_a_name_merely_mentioned_is_still_kept():
    reply = TODO.replace("    def add(self, text):\n        self.items."
                         "append(text)\n\n", "")
    f = keep.check(TODO, reply, "python", path="todo.py",
                   request="Make the summary say 'tasks', and add a guard "
                           "for empty text.")
    assert f.refused and "TodoList.add" in f.dropped


def test_what_one_sentence_asks_to_remove_may_go_and_is_said():
    reply = TODO.replace('    def summary(self):\n        return f"{len(self'
                         '.items)} items"\n', "")
    f = keep.check(TODO, reply, "python", path="todo.py",
                   request="The summary is not used. Remove summary.")
    assert not f.refused and f.allowed == ("TodoList.summary",)


# --------------------------------------------------------------------------
# … and ordinary refactors do not stall it
# --------------------------------------------------------------------------

FUNCS = ("def add(a, b):\n    return a + b\n\n\n"
         "def sub(a, b):\n    return a - b\n")


def test_functions_turned_into_methods_are_moved_not_lost():
    reply = ("class Calculator:\n    def add(self, a, b):\n"
             "        return a + b\n\n    def sub(self, a, b):\n"
             "        return a - b\n")
    f = keep.check(FUNCS, reply, "python", path="calc.py",
                   request="Turn these functions into a Calculator class.",
                   used_elsewhere=lambda n: "")
    assert not f.refused, f.describe()
    assert {m[0] for m in f.moved} == {"add", "sub"}


def test_functions_moved_to_another_file_and_imported_are_moved():
    reply = "from mathops import add, sub\n\n\nprint(add(1, sub(3, 2)))\n"
    f = keep.check(FUNCS, reply, "python", path="calc.py",
                   request="Move the arithmetic into mathops.py.")
    assert not f.refused and len(f.moved) == 2


def test_ordinary_comments_and_docstrings_are_not_placeholders():
    for line in ("# Other functions",
                 "# Keep the original implementation for compatibility",
                 "# Try int first... fall back to float for other inputs"):
        assert not keep.placeholders("", line, "python"), line
    doc = ('def f():\n    """Notes:\n\n    * the rest of the code path handles '
           'floats\n    """\n    return 1\n')
    assert not keep.placeholders("", doc, "python")


def test_a_dropped_setter_is_a_dropped_definition():
    src = ("class Temp:\n    @property\n    def c(self):\n        return 1\n\n"
           "    @c.setter\n    def c(self, v):\n        self._c = v\n")
    reply = src[:src.index("    @c.setter")]
    f = keep.check(src, reply, "python", path="t.py", request="tidy")
    assert f.refused and f.dropped == ("Temp.c@setter",)


def test_a_file_with_a_byte_order_mark_is_still_guarded():
    src = "﻿def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    f = keep.check(src, "def a():\n    return 1\n", "python", path="m.py",
                   request="faster")
    assert f.refused and f.dropped == ("b",)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

SRC = ("def first():\n    return 1\n\n\ndef second():\n    return 2\n\n\n"
       "def buggy(x):\n    return x + undefined\n")
DROP = _fence("def buggy(x):\n    return x + 1\n")


def test_refusals_do_not_pile_up_or_push_the_real_error_out(tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    seed = (Diagnostic(file="m.py", line=10, severity="error",
                       message="NameError: name 'undefined' is not defined"),)
    host = _host(tmp_path, [DROP] * 4)
    loop = _loop(host, attempts=4, narrow="off", mark_errors=False)
    out = loop.run_task(Task(id="t", path="m.py", purpose="m",
                             lang="python"), request="fix it", seed=seed)
    counts = [len(a.diagnostics) for a in out.attempts]
    assert max(counts) == 2, counts            # one refusal + the error
    last = _user(host, len(host.llm.prompts) - 1)
    assert "name 'undefined' is not defined" in last
    assert last.count("Your reply left out") == 1
    # and the refusal points at no line of the file
    assert all(d.line == 0 for a in out.attempts for d in a.diagnostics
               if d.code == "dropped-code")


def test_an_unplaceable_reply_when_nothing_else_fits_is_a_refusal(tmp_path):
    """Review #7: with a large conventions block the whole file could not be
    shown either, and the task stopped as "the model returned nothing"."""
    (tmp_path / "car.py").write_text(BIG)
    llm = ScriptedLLM(["```python\nprint('hello')\n```", FIX],
                      supports_tools=False, context_tokens=5600)
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=2, max_tokens=512)
    loop.prompts.conventions = "Use type hints everywhere. " * 444
    out = loop.run_task(_task(), request="make the car work", seed=SEED)
    assert out.ok, out.stopped_because
    assert "could not be placed" in out.attempts[0].note


def test_the_fresh_start_is_a_fresh_start_and_narrowing_resumes_after(
        tmp_path):
    src = ("def area(w, h):\n    return w * hh\n\n\n"
           "def perimeter(w, h):\n    return 2 * (w + h)\n\n\n"
           "print(area(2, 3))\n")
    (tmp_path / "m.py").write_text(src)
    wrong = src.replace("hh", "hhh")
    replies = [_fence(wrong), _fence(wrong.replace("2 * (w + h)",
                                                   "(w + h) * 2")),
               _fence(wrong.replace("hhh", "hhhh")),
               _fence("def area(w, h):\n    return w * h\n")]
    host = _host(tmp_path, replies)
    seed = (Diagnostic(file="m.py", line=2, severity="error",
                       message="NameError: name 'hh' is not defined"),)
    events = host.events
    out = _loop(host, attempts=4).run_task(
        Task(id="t", path="m.py", purpose="geometry", lang="python"),
        seed=seed)
    assert out.ok, out.stopped_because
    third, fourth = _user(host, 2), _user(host, 3)
    assert "was written before and did not work" in third
    assert personas.CONTRACT_SYMBOLS not in third
    assert personas.CONTRACT_SYMBOLS in fourth
    said = " ".join(e[1] for e in events.events)
    assert "attempt 3 starts fresh" in said


def test_a_second_round_still_knows_the_operators_file(tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    host = _host(tmp_path, [_fence(SRC.replace("undefined", "2"))] * 2)
    loop = _loop(host, attempts=1)
    task = Task(id="t", path="m.py", purpose="m", lang="python")
    loop.run_task(task, request="fix")
    loop.run_task(task, request="fix")
    assert loop._here_before["m.py"] is True
    assert loop._original["m.py"] == SRC


def test_every_call_is_journaled_even_two_in_one_attempt(tmp_path):
    (tmp_path / "car.py").write_text(CAR_LONG)
    replies = ["```python\nprint('hello')\n```",
               _fence(CAR_LONG.replace("accel", "2.0"))]
    host = _host(tmp_path, replies)
    loop = _loop(host, attempts=1)
    loop.journal = Journal(host.fs, "t")
    loop.run_task(_task(), seed=SEED)
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    assert len(gens) == len(host.llm.prompts) == 2


CAR_LONG = BIG


def test_the_enriched_status_is_said_again(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return g()\n")
    host = _host(tmp_path, [_fence("def f():\n    return 1\n")])
    seed = (Diagnostic(file="m.py", line=2, severity="error",
                       message="NameError: name 'g' is not defined"),)
    _loop(host, attempts=1).run_task(
        Task(id="t", path="m.py", purpose="f", lang="python"), seed=seed)
    assert any("is enriched" in e[1] for e in host.events.events)


# --------------------------------------------------------------------------
# smaller ones
# --------------------------------------------------------------------------

def test_a_did_you_mean_search_is_fast_on_a_big_file():
    big = "".join(f"line_{i} = {i}\n" for i in range(9000))
    anchor = "".join(f"other_{i} = {i}\n" for i in range(40))
    t0 = time.monotonic()
    _similar(big, anchor)
    forgiving_replace(big, anchor, "x = 1\n")
    assert time.monotonic() - t0 < 2.0


def test_an_anchor_that_matches_twice_after_spacing_is_said_to():
    text = "def a():\n    x = 1\n\n\ndef b():\n    x = 1\n"
    got, _how, why = forgiving_replace(text, "x = 1  \n", "x = 2\n")
    assert got is None and "more than once" in why


def test_the_where_block_counts_lines_as_python_does():
    code = "def a():\n    return 1\n\x0c\ndef b():\n    return undefined_name\n"
    block = where_block(code, (Diagnostic(file="m.py", line=5,
                                          severity="error", message="x"),),
                        "m.py")
    assert "█ 5 |     return undefined_name" in block


def test_a_path_inside_another_name_is_not_a_mention():
    files = ["src/io.py", "src/scenario.py"]
    pers = rank.personalization(files, {}, 'File "src/scenario.py", line 3')
    assert pers["src/scenario.py"] > pers["src/io.py"]


def test_a_replay_of_a_debug_task_matches_its_recording(tmp_path):
    """The traceback names the task's folder; it was the run's own, whose
    name has the time in it, so every replay "diverged" at call one."""
    suite = tmp_path / "suite"
    task = suite / "dbg"
    (task / "start").mkdir(parents=True)
    (task / "hidden").mkdir()
    (task / "start" / "m.py").write_text("def f():\n    return g()\n\n\n"
                                         "print(f())\n")
    (task / "hidden" / "test_m.py").write_text(
        "import unittest\nfrom m import f\n\n\nclass T(unittest.TestCase):\n"
        "    def test_f(self):\n        self.assertEqual(f(), 1)\n")
    (task / "bench.json").write_text(json.dumps({
        "kind": "debug", "debug": {
            "error_text": 'Traceback (most recent call last):\n  File '
                          '"{PROJECT}/m.py", line 2, in f\n    return g()\n'
                          "NameError: name 'g' is not defined\n",
            "description": "it crashes"}}))
    (task / "task.md").write_text("it crashes")
    fixed = _fence("def f():\n    return 1\n\n\nprint(f())\n")

    def live(_t, folder):
        return bench.RecordingLLM(ScriptedLLM([fixed] * 3,
                                              supports_tools=False),
                                  folder / bench.REPLIES)
    first = bench.run_suite(live, suite=suite, out=tmp_path / "runs",
                            label="rec", say=lambda _s: None)
    assert first.rows[0]["ok"], first.rows[0]

    def replay(_t, _folder):
        return bench.ReplayLLM(first.path / "dbg" / bench.REPLIES)
    again = bench.run_suite(replay, suite=suite, out=tmp_path / "runs",
                            label="replay", say=lambda _s: None)
    assert again.rows[0]["ok"]
    assert again.rows[0]["replay"]["diverged_at"] == 0


def test_a_narrowed_reply_that_changes_nothing_is_the_unchanged_file(
        tmp_path):
    """The escape hatch: "if you are certain the test is wrong, return the
    file unchanged". A narrowed reply doing exactly that became a refusal
    telling the model to change code (review #18)."""
    (tmp_path / "car.py").write_text(BIG)
    same = ("```python\ndef update(self, dt):\n"
            "    self.speed += dt * accel\n    return self.speed\n```")
    host = _host(tmp_path, [same])
    out = _loop(host, attempts=2).run_task(_task(), seed=SEED, baseline=BIG)
    assert not out.ok
    assert "returned the file unchanged" in out.attempts[0].note


def test_a_partial_reply_to_a_cpp_file_with_macros_is_not_syntax_checked():
    from cognitive_coder.codemap import parse_treesitter
    if not parse_treesitter.available("c"):
        return
    code = ("#ifdef _WIN32\nint start(void) {\n#else\nint start(int argc) {\n"
            "#endif\n    return 0;\n}\n")
    assert parse_treesitter.syntax_errors(code, "c") is None
    assert parse_treesitter.syntax_errors(
        "EXPORT int add(int a, int b) {\n    return a + b;\n}\n", "c") == []
