# SPDX-License-Identifier: Apache-2.0
"""Diffusion mode (canvas.py): holes instead of whole files.

What is held here, with a scripted model that fills canvases:

* WHERE a hole goes — the body of the function the error is in; a window
  around the error when the body is long or the file does not parse; nothing
  when no error names a line of the file; the stub bodies of a pinned file;
* HOW a filled hole comes back — padding and a run-on cut away, the file
  outside the hole byte for byte unchanged (D6), the old lines refused (D3),
  a hole that does not parse refused;
* the LOOP: a repair rewrites the hole and asks no whole file; every
  candidate failing falls back to the whole file; `diffusion="off"` never
  touches the canvas; a pinned file is written body by body after one notes
  call, its signatures held; a failure that names no line of the file is
  repaired in the body the model was least sure of.
"""

from __future__ import annotations

import ast
import re

from cognitive_coder import (
    AutoApprove,
    CanvasRequest,
    CanvasResult,
    Diagnostic,
    Host,
    LocalFileSystem,
    MemoryStorage,
    ModelCapabilities,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
)
from cognitive_coder import canvas as cv
from cognitive_coder.interfaces import PINNED_TEXT
from cognitive_coder.loop import _same_file

CALC = '''def add(a: int, b: int) -> int:
    """The sum."""
    total = a + c
    return total


def mul(a: int, b: int) -> int:
    """The product."""
    return a * b
'''


def _err(line, file="src/calc.py", msg="NameError: name 'c' is not defined"):
    return Diagnostic(file=file, line=line, severity="error", message=msg,
                      tool="python")


# --------------------------------------------------------------------------
# where the hole goes
# --------------------------------------------------------------------------

def test_the_hole_is_the_body_of_the_function_the_error_is_in():
    h = cv.repair_hole(CALC, "python", "src/calc.py", [_err(3)], _same_file)
    assert (h.kind, h.name, h.start, h.end) == ("body", "add", 2, 4)
    assert h.original == "    total = a + c\n    return total\n"
    assert h.head.startswith("def add(") and h.head.endswith('"""The sum."""\n')
    assert h.tail.startswith("\n\ndef mul(")
    assert h.stop_line == "def mul(a: int, b: int) -> int:"
    assert h.indent == 4 and h.line == 3 and h.label == "lines 3-4"


def test_a_long_body_gets_a_window_around_the_error():
    body = "".join(f"    x{i} = {i}\n" for i in range(60))
    code = f"def big():\n    \"\"\"Long.\"\"\"\n{body}    return x0 + y\n"
    h = cv.repair_hole(code, "python", "big.py", [_err(40, file="big.py")],
                       _same_file)
    assert h.kind == "window" and h.start <= 39 < h.end
    assert h.end - h.start <= cv.WINDOW_BEFORE + cv.WINDOW_AFTER + 1
    assert h.start >= 2, "the window stays inside the body"


def test_a_file_that_does_not_parse_gets_a_window():
    code = "def f(:\n    return 1\n\n\nx = 2\n"
    h = cv.repair_hole(code, "python", "f.py",
                       [_err(1, file="f.py", msg="SyntaxError")], _same_file)
    assert h.kind == "window" and h.start == 0


def test_no_located_error_no_hole():
    elsewhere = _err(9, file="tests/test_calc.py", msg="AssertionError")
    assert cv.repair_hole(CALC, "python", "src/calc.py", [elsewhere],
                          _same_file) is None
    warning = Diagnostic(file="src/calc.py", line=3, severity="warning",
                         message="unused")
    assert cv.repair_hole(CALC, "python", "src/calc.py", [warning],
                          _same_file) is None


def test_a_hole_too_big_for_a_canvas_is_refused():
    body = "".join(f"    value_number_{i} = compute_something({i})\n"
                   for i in range(30))
    code = f"def big():\n{body}"
    h = cv.repair_hole(code, "python", "b.py", [_err(5, file="b.py")],
                       _same_file, count_tokens=lambda t: len(t))
    assert h is None, "one character a token: nothing fits; no hole"


STUB = f'''# {PINNED_TEXT}
"""Geometry."""
from dataclasses import dataclass


@dataclass
class Box:
    w: float
    h: float

    def area(self) -> float:
        """Width times height."""
        raise NotImplementedError


def total(boxes: list) -> float:
    """The summed area."""
    raise NotImplementedError(
        "geom.total is not written yet")


def keep(x: int) -> int:
    return x
'''


def test_stub_holes_are_the_raise_bodies_in_file_order():
    holes = cv.stub_holes(STUB, "src/geom.py")
    assert [h.name for h in holes] == ["Box.area", "total"]
    area, total = holes
    assert area.original.strip() == "raise NotImplementedError"
    assert area.head.lstrip().startswith("def area(")
    assert '"""Width times height."""' in area.head
    assert total.end - total.start == 2, "a multi-line raise is all of it"
    assert area.indent == 8 and total.indent == 4


# --------------------------------------------------------------------------
# what comes back
# --------------------------------------------------------------------------

def _hole():
    return cv.repair_hole(CALC, "python", "src/calc.py", [_err(3)],
                          _same_file)


def test_padding_is_stripped_and_a_copied_tail_dropped():
    h = _hole()
    fill = cv.clean_fill("    total = a + b\n    return total\n\n\n\n\n", h,
                         "tail")
    assert fill == "    total = a + b\n    return total\n"
    copied = "    return a + b\n\n\ndef mul(a: int, b: int) -> int:\n"
    assert cv.clean_fill(copied, h, "tail") == "    return a + b\n"


def test_an_open_hole_that_runs_on_is_cut_where_the_body_ends():
    h = _hole()
    runon = ("    return a + b\n\n\ndef mul(a: int, b: int) -> int:\n"
             "    return 0\n")
    assert cv.clean_fill(runon, h, "open", lambda p: True) == \
        "    return a + b\n"
    assert cv.judge(CALC, h, "open", 1.0, _result(runon), "python").fill == \
        "    return a + b\n"
    # no parser to check the cut: nothing is cut (such holes hold a tail)
    assert "def mul" in cv.clean_fill(runon, h, "open")


def test_outside_the_hole_the_file_is_byte_for_byte_unchanged():
    h = _hole()
    new = cv.splice(CALC, h, "    return a + b\n")
    assert new.startswith('def add(a: int, b: int) -> int:\n    """The '
                          'sum."""\n    return a + b\n\n\ndef mul(')
    assert new.split("def mul")[1] == CALC.split("def mul")[1]
    assert new.count("\n") == CALC.count("\n") - 1


def _result(text, conf=0.9, ok=True):
    lines = text.count("\n") + 1
    return CanvasResult(text=text, finish_reason="stop" if ok else "error",
                        error="" if ok else "boom",
                        line_confidence=(conf,) * lines,
                        line_flips=(0,) * lines, steps=6, seed=1)


def test_the_old_lines_and_a_broken_hole_are_refused():
    h = _hole()
    same = cv.judge(CALC, h, "tail", 0.5,
                    _result("    total = a + c\n    return  total\n"),
                    "python")
    assert not same.usable and "same lines" in same.problem
    broken = cv.judge(CALC, h, "tail", 0.5,
                      _result("    total = (a +\n"), "python")
    assert not broken.usable and "does not parse" in broken.problem
    failed = cv.judge(CALC, h, "open", 1.0, _result("", ok=False), "python")
    assert not failed.usable and failed.problem == "boom"


def test_the_usable_candidate_the_model_was_surest_of_is_kept():
    h = _hole()
    cands = [cv.judge(CALC, h, "tail", 0.5, _result("    total = (\n", 0.99),
                      "python"),
             cv.judge(CALC, h, "open", 1.0, _result("    return a + b\n",
                                                    0.6), "python"),
             cv.judge(CALC, h, "tail", 1.0, _result("    return b + a\n",
                                                    0.8), "python")]
    assert cv.choose(cands) == 2
    assert cv.choose(cands[:1]) is None


def test_the_view_marks_the_hole_and_the_contract_says_where_the_reply_starts():
    h = _hole()
    view = cv.marked_view(CALC, h)
    assert "    # >>> REWRITE LINES 3-4" in view
    assert ("    total = a + c\n    return total\n"
            "    # <<< END OF THE LINES TO REWRITE") in view
    stub = cv.stub_holes(STUB)[0]
    assert "THE BODY OF `Box.area` GOES HERE" in cv.marked_view(STUB, stub)
    assert "ENDS there" in cv.contract(h, "open")
    assert "follow the region" in cv.contract(h, "tail")
    assert "already written" in cv.hole_block(h)


def test_the_candidate_schedule():
    h = _hole()
    plan = cv.requests_for(h, lambda style: (), 3, seed=10, attempt=2)
    styles = [(s, n) for s, n, _r in plan]
    assert styles == [("tail", 0.5), ("open", 1.0), ("tail", 1.0)]
    warm = plan[0][2]
    assert warm.seed_text == h.original and warm.tail == h.tail
    assert len({r.seed for _s, _n, r in plan}) == 3, "a seed each"
    assert plan[1][2].tail == "" and plan[1][2].seed_text == ""


def test_the_readout_is_realigned_to_the_cleaned_lines():
    h = _hole()
    raw = "\n    total = a + b\n\n    return total\n\n\n"
    res = CanvasResult(text=raw, line_confidence=(1, .5, 1, .7, 1, 1, 1),
                       line_flips=(0, 2, 0, 1, 0, 0, 0))
    c = cv.judge(CALC, h, "tail", 0.5, res, "python")
    assert c.fill == "    total = a + b\n\n    return total\n"
    assert c.confidence == (.5, 1, .7) and c.flips == (2, 0, 1)
    assert c.least == .5 and abs(c.mean - .6) < 1e-9, "blank lines ignored"
    # lines past the first canvas were never read out: unknown, not certain
    conf, _f = cv.align("a\nb\nc\nd", "a\nb\nc\nd\n", (0.3, 0.4), ())
    assert conf[:2] == (0.3, 0.4) and all(x != x for x in conf[2:])


def test_least_sure_skips_what_was_tried_and_what_is_gone():
    conf = {"add": 0.3, "mul": 0.8, "gone": 0.1}
    assert cv.least_sure(conf, CALC, set()) == "add"
    assert cv.least_sure(conf, CALC, {"add"}) == "mul"
    assert cv.least_sure(conf, CALC, {"add", "mul"}) == ""


# --------------------------------------------------------------------------
# the loop, with a model that fills canvases
# --------------------------------------------------------------------------

class CanvasLLM(ScriptedLLM):
    """ScriptedLLM that also fills canvases, from a script or a function."""

    def __init__(self, replies, fills, **kw):
        super().__init__(replies, supports_tools=False, **kw)
        self.fills = fills
        self.requests: list[CanvasRequest] = []

    def capabilities(self) -> ModelCapabilities:
        caps = super().capabilities()
        return ModelCapabilities(
            name=caps.name, family="gemma", context_tokens=caps.context_tokens,
            supports_grammar=False, supports_canvas=True)

    def canvas_fill(self, request, *, cancel=None):
        self.requests.append(request)
        fill = (self.fills(request) if callable(self.fills)
                else self.fills.pop(0))
        if isinstance(fill, CanvasResult):
            return fill
        return _result(fill, conf=0.85)


def _host(tmp_path, llm):
    return Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


PLAN = "src/calc.py — arithmetic helpers\ntests/test_calc.py — its tests\n"
CALC_REPLY = f"```python\n{CALC}```"
TEST_REPLY = '''```python
import unittest

from src.calc import add, mul


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mul(self):
        self.assertEqual(mul(2, 3), 6)


if __name__ == "__main__":
    unittest.main()
```'''
FIXED = CALC.replace("    total = a + c\n    return total\n",
                     "    total = a + b\n    return total\n")


def _run(tmp_path, llm, **cfg):
    host = _host(tmp_path, llm)
    session = Session(host, config=SessionConfig(
        attempts=3, pin_interfaces=False, review_after_build=False, **cfg))
    session.run("arithmetic helpers, with tests")
    return host, session


def _canvas_events(host):
    return [d for _k, _m, d in host.events.of("status")
            if isinstance(d, dict) and "canvas" in d]


def test_a_repair_rewrites_the_hole_and_asks_for_no_whole_file(tmp_path):
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY],
                    ["    total = a + b\n    return total\n\n\n\n",
                     "    total = (\n"])
    host, session = _run(tmp_path, llm)
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].ok, final["src/calc.py"].stopped_because
    assert len(llm.prompts) == 3, "plan, calc, test — the repair was a hole"
    assert len(llm.requests) == 2
    first = llm.requests[0]
    assert first.head.endswith('"""The sum."""\n')
    assert first.seed_text == "    total = a + c\n    return total\n"
    assert first.noise == 0.5 and first.tail.startswith("\n\ndef mul")
    text = "\n".join(m.content for m in first.messages)
    assert "# >>> REWRITE LINES 3-4" in text and "[THE HOLE" in text
    assert "name 'c' is not defined" in text, "the error is in the prompt"
    assert "Return the complete contents of the file" not in text
    kept = [d["canvas"] for d in _canvas_events(host)
            if "line_confidence" in d["canvas"]]
    assert kept and kept[0]["first_line"] == 3 and kept[0]["style"] == "tail"
    assert kept[0]["lines"] == ["    total = a + b", "    return total"]
    assert len(kept[0]["line_confidence"]) == 2, "aligned to the kept lines"
    journal = "\n".join(p.read_text() for p in
                        (tmp_path / ".cc_journal").rglob("*.jsonl"))
    assert '"canvas"' in journal and '"chosen": 0' in journal


def test_no_usable_hole_falls_back_to_the_whole_file(tmp_path):
    bad = CanvasResult(finish_reason="error", error="does not fit")
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, f"```python\n{FIXED}```"],
                    [bad, bad])
    host, session = _run(tmp_path, llm)
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    assert len(llm.prompts) == 4, "the whole-file repair was asked"
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("no hole came back usable" in m for m in warnings), warnings


def test_off_means_the_canvas_is_never_touched(tmp_path):
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, f"```python\n{FIXED}```"],
                    [])
    host, session = _run(tmp_path, llm, diffusion="off")
    assert llm.requests == []
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    assert any("diffusion mode is off" in m
               for _k, m, _d in host.events.of("status"))


GEOM_PLAN = ("src/geom.py — boxes and their areas\n"
             "src/report.py — prints the total area of two boxes\n")
INTERFACES = '''```python
# file: src/geom.py
"""Boxes."""
from dataclasses import dataclass


@dataclass
class Box:
    w: float
    h: float

    def area(self) -> float:
        """Width times height."""
        return 0.0


def total(boxes: list) -> float:
    """The summed area of the boxes."""
    return 0.0
```

```python
# file: src/report.py
"""Prints the total."""
from src.geom import Box, total


def main() -> int:
    """Print the total area of a 2x3 and a 1x1 box."""
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```'''
BODIES = {
    "Box.area": "        return self.w * self.h\n",
    "total": "    return sum(b.area() for b in boxes)\n\n\ndef stray():\n",
    "main": "    print(total([Box(2, 3), Box(1, 1)]))\n    return 0\n",
}


def _by_name(request):
    text = request.messages[-1].content
    name = re.search(r"\[THE HOLE — the body of `([^`]+)`\]", text).group(1)
    return BODIES[name]


def test_a_pinned_file_is_written_body_by_body_signatures_held(tmp_path):
    llm = CanvasLLM([GEOM_PLAN, INTERFACES, "area: w*h. total: sum.",
                     "main: print the total."], _by_name)
    host = _host(tmp_path, llm)
    session = Session(host, config=SessionConfig(
        attempts=2, review_after_build=False))
    session.run("boxes and a report")
    geom = (tmp_path / "src" / "geom.py").read_text()
    report = (tmp_path / "src" / "report.py").read_text()
    assert PINNED_TEXT not in geom and PINNED_TEXT not in report
    assert "return self.w * self.h" in geom
    assert "return sum(b.area() for b in boxes)" in geom
    assert "def stray" not in geom, "the run-on was cut at the dedent"
    assert "def total(boxes: list) -> float:" in geom
    assert '"""The summed area of the boxes."""' in geom
    ast.parse(geom)
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/geom.py"].ok and final["src/report.py"].ok
    # plan, interfaces, and one notes call per file: no whole file asked
    assert len(llm.prompts) == 4
    names = [re.search(r"body of `([^`]+)`", r.messages[-1].content).group(1)
             for r in llm.requests]
    assert names == ["Box.area", "total", "main"]
    notes = "\n".join(m.content for m in llm.requests[0].messages)
    assert "[IMPLEMENTATION NOTES" in notes and "area: w*h" in notes
    assert llm.requests[0].max_canvases == cv.OPEN_CANVASES_STUB


def test_a_failure_that_names_no_line_is_repaired_where_the_model_doubted(
        tmp_path):
    from cognitive_coder import Loop, LoopConfig, Task
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    calc = FIXED.replace("return a * b", "return a * b + 1")
    (tmp_path / "src" / "calc.py").write_text(calc)
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_calc.py").write_text(
        TEST_REPLY.split("```python\n")[1].split("```")[0])
    llm = CanvasLLM([], ["    return a * b\n", "    return b * a\n"])
    host = _host(tmp_path, llm)
    loop = Loop(host, config=LoopConfig(attempts=1))
    loop.doubt["src/calc.py"] = {"add": 0.9, "mul": 0.2}
    seed = (Diagnostic(file="tests/test_calc.py", line=11, severity="failure",
                       message="AssertionError: 7 != 6", tool="unittest"),)
    out = loop.run_task(Task(id="t1", path="src/calc.py",
                             purpose="arithmetic helpers",
                             test_path="tests/test_calc.py", lang="python"),
                        seed=seed)
    assert out.ok, out.stopped_because
    assert "return a * b\n" in (tmp_path / "src" / "calc.py").read_text()
    assert llm.requests[0].head.startswith("def mul(")
    statuses = [m for _k, m, _d in host.events.of("status")]
    assert any("least sure" in m for m in statuses), statuses


def test_the_same_hole_failing_the_same_way_twice_goes_whole_file(tmp_path):
    """A fix inside `add` that keeps failing with the same error says the
    fault is not in `add`: the next attempt is a whole-file repair."""
    still = "    total = a + c + 0\n    return total\n"
    again = "    total = (a + c)\n    return total\n"
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, f"```python\n{FIXED}```"],
                    [still, still, again, again])
    host, session = _run(tmp_path, llm)
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    assert len(llm.prompts) == 4, "the third attempt asked for the file"
    assert any("the fault is not in those lines" in m
               for _k, m, _d in host.events.of("status"))


def test_a_crlf_file_stays_crlf():
    crlf = CALC.replace("\n", "\r\n")
    h = cv.repair_hole(crlf, "python", "src/calc.py", [_err(3)], _same_file)
    new = cv.splice(crlf, h, "    return a + b\n")
    assert "\n" not in new.replace("\r\n", ""), "no bare LF introduced"
    assert new.split("def mul")[1] == crlf.split("def mul")[1]


# -- regressions from the review of 2026-10-10 -------------------------------

def test_an_inner_closing_brace_is_not_the_end_of_a_js_window():
    js = ("function f(a) {\n  if (a) {\n    return 1;\n  }\n"
          "  return x.y;\n}\n\nfunction g() {\n  return 2;\n}\n"
          + "".join(f"function h{i}() {{ return {i}; }}\n" for i in range(8)))
    h = cv.repair_hole(js, "javascript", "f.js",
                       [_err(5, file="f.js", msg="TypeError")], _same_file)
    plan = cv.requests_for(h, lambda st: (), 4, lang="javascript")
    assert {st for st, _n, _r in plan} == {"tail"}, "outside Python: held"
    fill = "  if (a) {\n    return 1;\n  }\n  return 0;\n"
    assert cv.clean_fill(fill, h, "tail") == fill


def test_the_next_line_must_match_with_its_indentation_and_what_follows():
    code = ("def f(x):\n    if x:\n        y = 1\n        return y\n"
            "    return None\n\n\nprint(f(1))\n")
    h = cv.repair_hole(code, "python", "f.py",
                       [_err(3, file="f.py", msg="boom")], _same_file)
    assert h.kind == "body" and h.after[0] == "print(f(1))"
    fill = "    if x:\n        return None\n    return None\n"
    c = cv.judge(code, h, "open", 1.0, _result(fill), "python")
    assert c.fill == fill, "an inner `return None` is not the next line"


def test_a_column_zero_comment_does_not_end_a_body():
    code = "def f(x):\n    y = x + 1\n    return z\n\n\nprint(f(1))\n"
    h = cv.repair_hole(code, "python", "f.py",
                       [_err(3, file="f.py", msg="NameError")], _same_file)
    fill = "    y = x + 2\n# tweak\n    y *= 3\n    return y\n"
    c = cv.judge(code, h, "open", 1.0, _result(fill), "python")
    assert c.usable and c.fill == fill
    assert "    y *= 3\n    return y\n" in c.code


def test_a_column_zero_line_inside_a_string_does_not_end_a_body():
    code = "def f():\n    return q\n\n\nprint(f())\n"
    h = cv.repair_hole(code, "python", "f.py",
                       [_err(2, file="f.py", msg="NameError")], _same_file)
    fill = '    s = """first\nsecond at column zero\n"""\n    return s\n'
    c = cv.judge(code, h, "open", 1.0, _result(fill), "python")
    assert c.usable and "second at column zero" in c.code
    assert "    return s\n" in c.code


def test_a_form_feed_does_not_move_the_hole():
    code = ("def a():\n    return 0\n\f\ndef b():\n    x = 1\n"
            "    return x + q\n")
    h = cv.repair_hole(code, "python", "m.py",
                       [_err(6, file="m.py", msg="NameError")], _same_file)
    assert h.name == "b" and h.original == "    x = 1\n    return x + q\n"
    new = cv.splice(code, h, "    return 1\n")
    assert "def b():\n    return 1\n" in new and new.count("def ") == 2


def test_a_one_line_def_is_never_a_body_hole():
    code = "def a():\n    return 0\n\n\ndef f(): return 1/0\n\n\nx = 2\n"
    h = cv.repair_hole(code, "python", "m.py",
                       [_err(5, file="m.py", msg="ZeroDivisionError")],
                       _same_file)
    assert h.kind == "window", "the signature is not a hole on its own"
    assert cv.function_hole(code, "m.py", "f") is None
    assert cv.stub_holes("def g(): raise NotImplementedError\n") == []


def test_a_partly_copied_tail_is_dropped():
    code = "".join(f"b{i} = {i}\n" for i in range(1, 20))
    h = cv.repair_hole(code, "python", "m.py",
                       [_err(6, file="m.py", msg="boom")], _same_file)
    first_tail = h.tail.split("\n")[0]
    fill = f"x = 1\n{first_tail}\n"
    assert cv.clean_fill(fill, h, "tail") == "x = 1\n"


def test_a_hole_that_never_came_back_is_not_counted_as_tried(tmp_path):
    bad = CanvasResult(finish_reason="error", error="server busy")
    still = CALC_REPLY.replace("a + c", "a + c + 0")   # a change; same error
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, still],
                    [bad, bad, "    total = a + b\n    return total\n", bad])
    host, session = _run(tmp_path, llm)
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    assert len(llm.requests) == 4, "attempt 3 was a hole again"
    assert not any("the fault is not in those lines" in m
                   for _k, m, _d in host.events.of("status"))


def test_a_located_error_that_does_not_fit_is_not_sent_to_doubt(tmp_path):
    from cognitive_coder import Loop, LoopConfig, Task

    class Tight(CanvasLLM):
        def count_tokens(self, text):
            return len(text or "") * 50          # nothing fits a canvas

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(CALC)
    llm = Tight([f"```python\n{FIXED}```"], [], context_tokens=10**9)
    host = _host(tmp_path, llm)
    loop = Loop(host, config=LoopConfig(attempts=1))
    loop.doubt["src/calc.py"] = {"mul": 0.1}
    out = loop.run_task(Task(id="t1", path="src/calc.py", purpose="math",
                             lang="python"), seed=(_err(3),))
    assert llm.requests == [], "no doubt hole for a located error"
    assert len(llm.prompts) == 1 and out.ok


# --------------------------------------------------------------------------
# BUILD_LOG.txt — every diffusion decision, in words (Bill, 2026-10-10:
# comprehensive logging is a rule for anything in development)
# --------------------------------------------------------------------------

def _build_log(tmp_path) -> str:
    return (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")


def test_the_build_log_says_which_hole_why_every_canvas_and_how_sure(
        tmp_path):
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY],
                    ["    total = a + b\n    return total\n\n\n\n",
                     "    total = (\n"])
    _run(tmp_path, llm)
    log = _build_log(tmp_path)
    assert "diffusion=on (2 canvas(es) per repair, notes on)" in log
    assert re.search(r"diffusion src/calc\.py — attempt 1: rewrite lines "
                     r"3-4 \(`add`\) as a hole — kept canvas 1 of 2", log), log
    assert "why this hole: its error is at line 3" in log
    assert re.search(r"canvas 1: tail, noise 0\.5, seed 1, 6 step\(s\).*"
                     r"usable: 85% sure on average", log)
    assert re.search(r"canvas 2: open, noise 1, .*REJECTED: ", log)
    cut = log.split("the lines cut out", 1)[1]
    assert "    total = a + c" in cut.splitlines()[1]
    kept = log.split("the hole as kept", 1)[1].splitlines()
    assert "`?` under 50%, `~` under 85%" in kept[0]
    assert kept[1].strip() == "85%   |     total = a + b"
    assert "canvas 2 as it came back (rejected)" in log \
        or "canvas 2's raw text (rejected)" in log
    # the attempt's code is logged as coming from the canvases, not from a
    # temperature it never sampled at
    gen = log.rsplit("generate src/calc.py — attempt 1", 1)[-1]
    assert gen.splitlines()[1].strip().startswith(
        "via diffusion canvases (each one is logged above):")


def test_the_build_log_says_every_time_diffusion_steps_aside(tmp_path):
    bad = CanvasResult(finish_reason="error", error="does not fit")
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, f"```python\n{FIXED}```"],
                    [bad, bad])
    _run(tmp_path, llm)
    log = _build_log(tmp_path)
    assert ("rewrite lines 3-4 (`add`) as a hole — none of 2 canvas(es) "
            "usable — this attempt writes the whole file") in log
    assert log.count("REJECTED: does not fit") == 2
    # the whole-file attempt that followed is an ordinary generation
    gen = log.rsplit("generate src/calc.py — attempt 1", 1)[-1]
    assert gen.splitlines()[1].strip().startswith("temperature=")


def test_the_build_log_and_journal_name_a_repeated_hole(tmp_path):
    still = "    total = a + c + 0\n    return total\n"
    again = "    total = (a + c)\n    return total\n"
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, f"```python\n{FIXED}```"],
                    [still, still, again, again])
    _run(tmp_path, llm)
    log = _build_log(tmp_path)
    assert re.search(r"no hole this attempt — lines 3-4 was rewritten "
                     r"already and the same error came back — the fault is "
                     r"not in those lines; the whole file is written "
                     r"instead", log), log
    journal = "\n".join(p.read_text() for p in
                        (tmp_path / ".cc_journal").rglob("*.jsonl"))
    assert '"kind": "skip"' in journal


def test_the_build_log_holds_the_notes_and_each_pinned_body(tmp_path):
    llm = CanvasLLM([GEOM_PLAN, INTERFACES, "area: w*h. total: sum.",
                     "main: print the total."], _by_name)
    host = _host(tmp_path, llm)
    Session(host, config=SessionConfig(
        attempts=2, review_after_build=False)).run("boxes and a report")
    log = _build_log(tmp_path)
    assert "implementation notes planned for 2 bodies before any was " \
           "written" in log
    notes = log.split("implementation notes planned for 2 bodies", 1)[1]
    assert "area: w*h. total: sum." in notes
    assert re.search(r"write lines? [\d-]+ \(`Box\.area`\) as a hole — kept "
                     r"canvas 1 of 1", log), log
    assert "why this hole: `Box.area` is a pinned stub" in log
    assert "2 of 2 pinned bodies written as holes" in log
    assert "1 of 1 pinned body written as holes" in log


def test_the_build_log_says_when_diffusion_mode_is_off_or_unavailable(
        tmp_path):
    (tmp_path / "off").mkdir()
    (tmp_path / "plain").mkdir()
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY, f"```python\n{FIXED}```"],
                    [])
    _run(tmp_path / "off", llm, diffusion="off")
    assert "diffusion=off (switched off for this session)" in \
        _build_log(tmp_path / "off")
    plain = ScriptedLLM([PLAN, CALC_REPLY, TEST_REPLY,
                         f"```python\n{FIXED}```"], supports_tools=False)
    _run(tmp_path / "plain", plain)
    log = _build_log(tmp_path / "plain")
    assert "diffusion=unavailable (the model fills no canvas)" in log
    assert "diffusion src/" not in log


def test_doubt_rows_mark_the_lines_to_distrust():
    hole = _hole()
    cand = cv.Candidate("tail", 0.5, _result("x"), "a = 1\n\nb = 2\nc = 3\n"
                        "d = 4\n", "code", "",
                        confidence=(0.9, float("nan"), 0.6, 0.3,
                                    float("nan")))
    assert hole is not None
    assert cv.doubt_rows(cand) == [" 90%   | a = 1",
                                   "       | ",
                                   " 60% ~ | b = 2",
                                   " 30% ? | c = 3",
                                   "  --   | d = 4"]
    assert "REJECTED" not in cand.describe()
    bad = cv.Candidate("open", 1.0, CanvasResult(finish_reason="error",
                                                 error="boom"),
                       None, "", "boom")
    assert bad.describe().endswith("REJECTED: boom")
    assert "seed none" in bad.describe()


def test_the_build_log_names_a_doubted_body_that_did_not_fit(tmp_path):
    from cognitive_coder import Loop, LoopConfig, Task
    from cognitive_coder.journal import SessionLog

    class Tight(CanvasLLM):
        def count_tokens(self, text):
            return len(text or "") * 50          # nothing fits a canvas

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(CALC)
    llm = Tight([f"```python\n{FIXED}```"], [], context_tokens=10**9)
    host = _host(tmp_path, llm)
    loop = Loop(host, config=LoopConfig(attempts=1))
    loop.log = SessionLog(host.fs, "t")
    loop.doubt["src/calc.py"] = {"mul": 0.1}
    seed = (Diagnostic(file="tests/test_calc.py", line=11, severity="failure",
                       message="AssertionError: 7 != 6", tool="unittest"),)
    loop.run_task(Task(id="t1", path="src/calc.py", purpose="math",
                       lang="python"), seed=seed)
    log = _build_log(tmp_path)
    assert ("the body the model was least sure of (`mul`) could not be cut "
            "out as a hole that fits one canvas") in log, log
    assert "every body the model doubted has been tried" not in log


def test_a_usable_canvas_that_was_not_chosen_is_not_called_rejected(
        tmp_path):
    def fills(request):
        return (_result("    total = a + b\n    return total\n", conf=0.95)
                if request.noise == 0.5 else
                _result("    return a + b\n", conf=0.6))
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY], fills)
    _run(tmp_path, llm)
    log = _build_log(tmp_path)
    assert "canvas 2 as it came back (usable, not chosen)" in log
    assert "(rejected)" not in log


def test_the_config_line_survives_a_bad_candidate_count(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY],
                    ["    total = a + b\n    return total\n"] * 4)
    _run(tmp_path / "a", llm, diffusion_candidates=None)
    assert "diffusion=on (1 canvas(es) per repair" in _build_log(
        tmp_path / "a")
    llm = CanvasLLM([PLAN, CALC_REPLY, TEST_REPLY],
                    ["    total = a + b\n    return total\n"] * 9)
    _run(tmp_path / "b", llm, diffusion_candidates=9)
    assert (f"diffusion=on ({len(cv.REPAIR_SCHEDULE)} canvas(es) per "
            f"repair") in _build_log(tmp_path / "b")
