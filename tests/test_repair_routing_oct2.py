# SPDX-License-Identifier: Apache-2.0
"""Repair routing, as the Oct 2, 2026 racing build needed it.

Devstral-Small-2-24B, the pseudo-3D racing spec, 23 minutes, four files
failed. Two of the failures were the routing's, not the model's:

  * tests/test_math3d.py did not parse (a stray fence on its last line).
    The syntax check recorded the error's file as "<unknown>", so the loop
    could not see it was the TEST that was broken: it reported "It is
    src/math3d.py that has to change", and later rewrote src/physics.py
    twice for the same fault in tests/test_physics.py.
  * src/main.py died in src/math3d.py; math3d.py was repaired; main.py then
    died in src/render.py. The re-check said "it still fails" and stopped,
    and render.py was never repaired.

Each test here is one of those, end to end, through a real session.
"""

from __future__ import annotations

from cognitive_coder import (
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
)


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Devstral-Small-2-24B",
                                context_tokens=16384),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _writes(host):
    """The prompts that asked the model to WRITE something (not review)."""
    def last_user(p):
        return [m for m in p if m.role == "user"][-1].content
    return [p for p in host.llm.prompts
            if not last_user(p).startswith("[TASK]\nReview ")]


def _warnings(host):
    return [m for _k, m, _d in host.events.of("warning")]


def _statuses(host):
    return [m for _k, m, _d in host.events.of("status")]


# ---------------------------------------------------------------------------
# 1. a test that does not parse is the TEST's fault
# ---------------------------------------------------------------------------

CALC_PLAN = ("src/calc.py — adds two numbers\n"
             "tests/test_calc.py — tests for calc\n")

CALC = '''```python
def add(a: int, b: int) -> int:
    """The sum."""
    return a + b
```'''

TEST_CALC_BROKEN = '''```python
import unittest

from src.calc import add


class TestCalc(unittest.TestCase):
    def test_add(self)
        self.assertEqual(add(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
```'''

TEST_CALC = TEST_CALC_BROKEN.replace("def test_add(self)\n",
                                     "def test_add(self):\n")


def test_a_test_that_does_not_parse_is_repaired_not_its_module(tmp_path):
    host = _host(tmp_path, [CALC_PLAN, CALC, TEST_CALC_BROKEN, TEST_CALC])
    session = Session(host, config=SessionConfig(attempts=3))
    session.run("a function that adds two numbers, with a test")

    final = {o.path: o for o in session._final_outcomes()}
    assert final["tests/test_calc.py"].ok, \
        final["tests/test_calc.py"].stopped_because
    assert final["src/calc.py"].ok

    # the syntax error was located IN the test file, by name
    test_runs = [o for o in session.outcomes
                 if o.path == "tests/test_calc.py"]
    first = test_runs[0].attempts[0].diagnostics[0]
    assert first.file.replace("\\", "/").endswith("tests/test_calc.py")
    assert first.file != "<unknown>"
    assert first.tool == "python-ast"

    # ...so the module was never blamed for it, nor rewritten
    assert not any("has to change" in m for m in _warnings(host)), \
        _warnings(host)
    assert len(_writes(host)) == 4      # plan, calc, test, test again
    assert "return a + b" in (tmp_path / "src" / "calc.py").read_text()


# ---------------------------------------------------------------------------
# 2. a module whose OWN test file does not parse is not rewritten for it
# ---------------------------------------------------------------------------

ALPHA_PLAN = "src/alpha.py — returns a greeting\n"

ALPHA = '''```python
def greet() -> str:
    """A greeting."""
    return "hi"
```'''

#: Already in the folder before the build — the "improve what is here" case.
#: Its last line is the Oct 2 stray fence.
EXISTING_BROKEN_TEST = '''import unittest

from src.alpha import greet


class TestAlpha(unittest.TestCase):
    def test_greet(self):
        self.assertEqual(greet(), "hi")


if __name__ == "__main__":
    unittest.main()
```
'''


def test_a_module_is_not_rewritten_for_its_test_s_syntax_error(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_alpha.py").write_text(EXISTING_BROKEN_TEST)
    host = _host(tmp_path, [ALPHA_PLAN, ALPHA, ALPHA, ALPHA])
    session = Session(host, config=SessionConfig(attempts=3))
    session.run("a greeting function")

    alpha = next(o for o in session.outcomes if o.path == "src/alpha.py")
    assert "the error is in tests/test_alpha.py" in alpha.stopped_because, \
        alpha.stopped_because
    assert "rewriting src/alpha.py cannot fix it" in alpha.stopped_because
    # one attempt, not three spent producing the same correct file
    assert len(alpha.attempts) == 1
    assert len(_writes(host)) == 2      # plan, alpha
    # the plan does not own the test, so it is reported, not rewritten
    assert any("which is not a file this plan owns" in m
               for m in _warnings(host)), _warnings(host)
    assert (tmp_path / "tests" / "test_alpha.py").read_text() == \
        EXISTING_BROKEN_TEST


def test_a_module_is_still_retried_for_a_test_that_wants_more_of_it():
    """The new rule is narrow. A test raising "cannot import name" CAN be
    fixed by the module (give it the name), so the test is not the culprit;
    neither is a failed assertion. Only a test that does not parse is."""
    from cognitive_coder.loop import Loop
    from cognitive_coder.types import Diagnostic, Task

    class _Store:
        def files(self):
            return [{"path": "src/alpha.py"},
                    {"path": "tests/test_alpha.py"}]

    class _Map:
        store = _Store()

    loop = Loop.__new__(Loop)
    loop.codemap = _Map()
    task = Task(id="t1", path="src/alpha.py", purpose="a greeting")
    where = "D:\\proj\\tests\\test_alpha.py"

    def culprit(message, code=None):
        return loop._culprit_elsewhere(task, [Diagnostic(
            file=where, line=3, severity="exception", message=message,
            code=code)])

    assert culprit("SyntaxError: invalid syntax") == "tests/test_alpha.py"
    assert culprit("IndentationError: unexpected indent") == \
        "tests/test_alpha.py"
    assert culprit("expected ':'", code="syntax") == "tests/test_alpha.py"
    assert culprit("ImportError: cannot import name 'shout' from "
                   "'src.alpha'") == ""
    assert culprit("AssertionError: 'hi' != 'hello'") == ""


# ---------------------------------------------------------------------------
# 3. a failure that moves to a THIRD file after a repair is followed
# ---------------------------------------------------------------------------

HOP_PLAN = ("src/alpha.py — builds the car's state\n"
            "src/beta.py — describes a state in one line\n"
            "src/main.py — the entry point: builds a state and prints it\n")

ALPHA_WRONG = '''```python
class State:
    """The car."""

    def __init__(self) -> None:
        self.speed = 0.0


def make() -> State:
    """A new car."""
    return State().x
```'''

ALPHA_FIXED = ALPHA_WRONG.replace("return State().x", "return State()")

BETA_WRONG = '''```python
def describe(state) -> str:
    """One line about the car."""
    return f"speed {state.x}"
```'''

BETA_FIXED = BETA_WRONG.replace("state.x", "state.speed")

MAIN = '''```python
from src.alpha import make
from src.beta import describe


def main() -> int:
    print(describe(make()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```'''


def test_a_failure_that_moves_to_a_third_file_is_followed(tmp_path):
    host = _host(tmp_path, [HOP_PLAN, ALPHA_WRONG, BETA_WRONG, MAIN,
                            ALPHA_FIXED, BETA_FIXED])
    session = Session(host, config=SessionConfig(attempts=3))
    session.run("a car state, a describer, and a main that prints it")

    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/main.py"].ok, final["src/main.py"].stopped_because
    assert "state.speed" in (tmp_path / "src" / "beta.py").read_text()
    assert "return State()\n" in (tmp_path / "src" / "alpha.py").read_text()

    mains = [o for o in session.outcomes if o.path == "src/main.py"]
    assert len(mains) == 3, [o.stopped_because for o in mains]
    assert "the error is in src/alpha.py" in mains[0].stopped_because
    assert "the error is now in src/beta.py" in mains[1].stopped_because
    statuses = _statuses(host)
    assert any("src/alpha.py will be repaired against src/main.py" in m
               for m in statuses), statuses
    assert any("src/beta.py will be repaired against src/main.py" in m
               for m in statuses), statuses


def test_a_file_blamed_again_after_its_repair_is_not_repaired_twice(tmp_path):
    """The chain is bounded: each file gets one repair per session."""
    beta_ok = BETA_FIXED
    host = _host(tmp_path, [HOP_PLAN, ALPHA_WRONG, beta_ok, MAIN,
                            ALPHA_WRONG])        # the "repair" changes nothing
    session = Session(host, config=SessionConfig(attempts=3))
    session.run("a car state, a describer, and a main that prints it")

    final = {o.path: o for o in session._final_outcomes()}
    assert not final["src/main.py"].ok
    assert any("src/main.py still fails inside src/alpha.py, which has "
               "already had its one repair" in m for m in _warnings(host)), \
        _warnings(host)
    # plan, alpha, beta, main, alpha's one repair — and nothing more
    assert len(_writes(host)) == 5
