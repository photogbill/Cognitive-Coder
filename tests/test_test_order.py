# SPDX-License-Identifier: Apache-2.0
"""A module's test is built beside it — before it, when the request says so.

Oct 2, 2026: the racing spec says "The engine must implement tests before
writing the module bodies". The PLAN was math3d, physics, track, render,
main, test_math3d, test_physics — every test after every module, because a
test depends only on its module and the topological sort kept the
proposed order. Four of five modules were "BUILT (not verified)" against
zero tests, and physics.py's bug was found after render.py and main.py had
been built on top of it.

§F2 of the build spec makes the design test-first: write the test from the
task and the interface, run it (it must fail against the stub), then write
the module to pass it. That needs an interface to write the test against,
which the skeleton did not have until it pinned one (see
test_interfaces_pinned.py). So:

  * when the request asks for tests first (or `test_first=True`), each test
    whose module's interface is pinned is written right BEFORE its module,
    and its failure against the stub is what it must do;
  * otherwise each test is written right AFTER its module, so the module is
    checked by its own test before the next module is built on it.
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
from cognitive_coder.planner import Planner

TEST_FIRST_SENTENCE = ("The engine must implement tests before writing the "
                       "module bodies, adhering to the following rules:")


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Qwen3-Coder-30B-A3B-Instruct",
                                context_tokens=32768),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _user_text(prompt) -> str:
    return "\n".join(m.content for m in prompt
                     if m.role == "user" and isinstance(m.content, str))


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


# ---------------------------------------------------------------------------
# what counts as asking for tests first
# ---------------------------------------------------------------------------

def test_the_racing_specs_sentence_asks_for_tests_first():
    assert Planner.wants_test_first_text(TEST_FIRST_SENTENCE)
    for text in ("Use TDD.", "Work test-first.", "write the tests first",
                 "Tests must be written before the implementation.",
                 "test driven, please"):
        assert Planner.wants_test_first_text(text), text
    for text in ("A CLI that prints the mean, with tests.",
                 "tests/test_math3d.py MUST NOT initialize a display",
                 "Write a test asserting that physics.update() caps speed.",
                 "the first test of the season"):
        assert not Planner.wants_test_first_text(text), text


# ---------------------------------------------------------------------------
# 1. by default: each test right after its module
# ---------------------------------------------------------------------------

PLAN = ("src/alpha.py — the first part\n"
        "src/beta.py — the second part\n"
        "src/main.py — the entry point\n"
        "tests/test_alpha.py — tests for alpha\n"
        "tests/test_beta.py — tests for beta\n")


def test_each_test_is_planned_right_after_its_module(tmp_path):
    host = _host(tmp_path, [PLAN])
    session = Session(host, config=SessionConfig(pin_interfaces=False))
    session.start("two parts and a main, with tests")
    order = [t.path for t in session.plan.tasks]
    assert order == ["src/alpha.py", "tests/test_alpha.py", "src/beta.py",
                     "tests/test_beta.py", "src/main.py"], order
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "build order:" in log and "2. tests/test_alpha.py" in log


def test_a_modules_test_runs_before_the_next_module_is_built(tmp_path):
    """The Oct 2 shape in miniature: alpha is wrong, its test says so, and
    alpha is repaired against it BEFORE beta — which uses alpha — is built."""
    alpha_wrong = _fence("def double(x: int) -> int:\n"
                         '    """Twice x."""\n    return x + 1\n')
    alpha_fixed = _fence("def double(x: int) -> int:\n"
                         '    """Twice x."""\n    return x * 2\n')
    test_alpha = _fence("import unittest\n\nfrom src.alpha import double\n\n\n"
                        "class TestAlpha(unittest.TestCase):\n"
                        "    def test_double(self):\n"
                        "        self.assertEqual(double(3), 6)\n")
    beta = _fence("from src.alpha import double\n\n\n"
                  "def quadruple(x: int) -> int:\n"
                  '    """Four times x."""\n    return double(double(x))\n')
    test_beta = _fence("import unittest\n\n"
                       "from src.beta import quadruple\n\n\n"
                       "class TestBeta(unittest.TestCase):\n"
                       "    def test_quadruple(self):\n"
                       "        self.assertEqual(quadruple(2), 8)\n")
    main = _fence("from src.beta import quadruple\n\nprint(quadruple(1))\n")
    host = _host(tmp_path, [PLAN, alpha_wrong, test_alpha, alpha_fixed, beta,
                            test_beta, main])
    session = Session(host, config=SessionConfig(
        attempts=2, pin_interfaces=False, review_after_build=False))
    session.run("two parts and a main, with tests")
    final = {o.path: o for o in session._final_outcomes()}
    assert all(o.ok for o in final.values()), session.report()
    built = [o.path for o in session.outcomes]
    # alpha's repair happened before beta was ever written
    assert built.index("tests/test_alpha.py") < built.index("src/beta.py")
    fix_at = max(i for i, p in enumerate(built) if p == "src/alpha.py")
    assert fix_at < built.index("src/beta.py"), built
    assert final["src/alpha.py"].verified and final["src/beta.py"].verified


# ---------------------------------------------------------------------------
# 2. asked for: each test right before its module, against the pinned stub
# ---------------------------------------------------------------------------

CALC_PLAN = ("src/calc.py — arithmetic helpers\n"
             "tests/test_calc.py — tests for calc\n")

CALC_SKELETON = '''```python
# file: src/calc.py
def add(a: int, b: int) -> int:
    """The sum of a and b."""
    raise NotImplementedError
```'''

TEST_CALC_WRONG_NAME = _fence(
    "import unittest\n\nfrom src.calc import plus\n\n\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        self.assertEqual(plus(2, 3), 5)\n")

TEST_CALC = _fence(
    "import unittest\n\nfrom src.calc import add\n\n\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        # 2 + 3 is 5: the request's own definition of a sum\n"
    "        self.assertEqual(add(2, 3), 5)\n")

CALC = _fence("def add(a: int, b: int) -> int:\n"
              '    """The sum of a and b."""\n    return a + b\n')


def test_tests_first_when_the_request_says_so(tmp_path):
    request = f"Arithmetic helpers. {TEST_FIRST_SENTENCE} tests/test_calc.py"
    host = _host(tmp_path, [CALC_PLAN, CALC_SKELETON, TEST_CALC_WRONG_NAME,
                            TEST_CALC, CALC])
    session = Session(host, config=SessionConfig(attempts=3,
                                                 review_after_build=False))
    session.run(request)

    order = [t.path for t in session.plan.tasks]
    assert order == ["tests/test_calc.py", "src/calc.py"], order
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].ok and final["src/calc.py"].verified, \
        session.report()
    # the test, written first, is verified once the module passes it
    assert final["tests/test_calc.py"].ok
    assert final["tests/test_calc.py"].verified

    first = [o for o in session.outcomes if o.path == "tests/test_calc.py"][0]
    # its OWN fault (a name the stub does not define) was repaired; its
    # failure against the stub was accepted, with the reason
    assert len(first.attempts) == 2
    assert any("written before src/calc.py" in c for c in first.caveats)
    assert not first.verified

    prompts = [_user_text(p) for p in host.llm.prompts]
    test_prompt = next(p for p in prompts
                       if "Write the complete contents of "
                          "`tests/test_calc.py`" in p)
    assert "[THE MODULE UNDER TEST — src/calc.py, interface only]" in \
        test_prompt
    assert "def add(a: int, b: int) -> int:" in test_prompt
    calc_prompt = next(p for p in prompts
                       if "Write the complete contents of `src/calc.py`" in p)
    assert "[THE TEST THIS FILE MUST PASS — tests/test_calc.py]" in \
        calc_prompt
    assert "self.assertEqual(add(2, 3), 5)" in calc_prompt
    assert "[THE INTERFACE PINNED FOR THIS FILE — src/calc.py]" in \
        calc_prompt
    # the module was never "repaired against" a test that ran before it
    statuses = [m for _k, m, _d in host.events.of("status")]
    assert not any("will be repaired against" in m for m in statuses)
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "1. tests/test_calc.py  (written first)" in log


def test_tests_first_needs_a_pinned_interface(tmp_path):
    request = f"Arithmetic helpers. {TEST_FIRST_SENTENCE}"
    host = _host(tmp_path, [CALC_PLAN, CALC, TEST_CALC])
    session = Session(host, config=SessionConfig(
        attempts=1, pin_interfaces=False, review_after_build=False))
    session.run(request)
    order = [t.path for t in session.plan.tasks]
    assert order == ["src/calc.py", "tests/test_calc.py"], order
    assert any("no module's interface could be pinned" in c
               for c in session.plan.caveats), session.plan.caveats
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].verified


def test_test_first_can_be_turned_off(tmp_path):
    request = f"Arithmetic helpers. {TEST_FIRST_SENTENCE}"
    host = _host(tmp_path, [CALC_PLAN, CALC_SKELETON, CALC, TEST_CALC])
    session = Session(host, config=SessionConfig(
        attempts=1, test_first=False, review_after_build=False))
    session.start(request)
    assert [t.path for t in session.plan.tasks] == \
        ["src/calc.py", "tests/test_calc.py"]
