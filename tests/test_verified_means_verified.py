# SPDX-License-Identifier: Apache-2.0
"""C4, held to: "verified" means a test ran and passed — and nothing else.

Three ways a build was reported green without being verified, all found by
the review of 2026-09-29:

  * the planner told the engineer "Its tests live in tests/test_alpha.py and
    must pass" about a file nothing would ever write;
  * when a correct test failed against a wrong module, the repairer was
    pointed at the TEST, and the "fixed" test asserted the bug:
    `add` returning `a - b`, and `assertEqual(add(1, 1), 0)`, both committed
    green. The tester persona's own words describe it — a test bent to the
    code "manufactures confidence";
  * a test file that existed but collected zero tests sealed its module as
    verified.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import (  # noqa: E402
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
from cognitive_coder.planner import Planner  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem  # noqa: E402

WRONG = "```python\ndef add(a, b):\n    return a - b\n```"
RIGHT = "```python\ndef add(a, b):\n    return a + b\n```"
GOOD_TEST = ("```python\nimport unittest\n\nfrom src.calc import add\n\n\n"
             "class T(unittest.TestCase):\n"
             "    def test_add(self):\n"
             "        self.assertEqual(add(1, 1), 2)\n```")
BENT_TEST = GOOD_TEST.replace("add(1, 1), 2)", "add(1, 1), 0)")
CALC_REQUEST = ("Build src/calc.py with add(a, b). Tests: tests/test_calc.py "
                "must cover add.")


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _session(host, **config):
    config.setdefault("review_after_build", False)
    return Session(host, config=SessionConfig(**config))


def _session_end(session):
    return [e for e in session.journal.events()
            if e["event"] == "session_end"][-1]["data"]


def _tails(host):
    return ["\n".join(m.content for m in prompt if m.role == "user")
            for prompt in host.llm.prompts]


def _body(fenced):
    return fenced.split("\n", 1)[1].rsplit("```", 1)[0]


# --------------------------------------------------------------------------
# (a) a test file is either planned, or not promised
# --------------------------------------------------------------------------

def test_an_unplanned_test_file_is_not_promised_to_the_engineer(tmp_path):
    """No test task, no test file on disk — so no sentence telling the
    engineer that tests exist and must pass."""
    host = _host(tmp_path, ["src/alpha.py — add two numbers\n", RIGHT])
    session = _session(host, attempts=1)
    session.run("a module that adds two numbers")
    assert session.plan.tasks[0].test_path == ""
    assert "tests/test_alpha.py" not in _tails(host)[1]


def test_pairing_adds_the_tester_task_after_its_module():
    host = Host(llm=ScriptedLLM(["src/alpha.py — add two numbers\n"],
                                supports_tools=False),
                fs=MemoryFileSystem(), events=RecordingEvents())
    planner = Planner(host, pair_tests=True)
    plan = planner.derive_order(planner.plan("add two numbers"))
    assert [(t.path, t.persona) for t in plan.tasks] == [
        ("src/alpha.py", "engineer"), ("tests/test_alpha.py", "tester")]
    assert plan.tasks[0].test_path == "tests/test_alpha.py"


def test_a_pairing_that_does_not_fit_is_a_caveat():
    host = Host(llm=ScriptedLLM(["src/a.py — a\nsrc/b.py — b\n"],
                                supports_tools=False),
                fs=MemoryFileSystem(), events=RecordingEvents())
    plan = Planner(host, pair_tests=True, max_files=3).plan("two modules")
    assert [t.path for t in plan.tasks] == [
        "src/a.py", "src/b.py", "tests/test_a.py"]
    assert any("src/b.py" in c and "no test" in c for c in plan.caveats), \
        plan.caveats


# --------------------------------------------------------------------------
# (b) a test that fails against its module is never bent to fit it
# --------------------------------------------------------------------------

def test_a_failing_test_is_not_rewritten_to_agree_with_the_module(tmp_path):
    """The reviewer's r06, with the repair pass also failing: the session
    ends not-ok and the test on disk is the test the tester wrote."""
    host = _host(tmp_path, ["src/calc.py — add two numbers\n",
                            WRONG, GOOD_TEST, BENT_TEST, WRONG])
    session = _session(host, attempts=1)
    session.run(CALC_REQUEST)
    test_now = (tmp_path / "tests" / "test_calc.py").read_text()
    assert "add(1, 1), 2)" in test_now, test_now
    assert not any("`tests/test_calc.py` does not work yet" in t
                   for t in _tails(host)), "the repairer was aimed at the test"
    assert not _session_end(session)["ok"]
    stop = next(o for o in session.outcomes
                if o.path == "tests/test_calc.py").stopped_because
    assert "src/calc.py" in stop and "not rewritten" in stop, stop


def test_the_module_is_repaired_against_the_test(tmp_path):
    """The right direction: the module is fixed to satisfy the test, the
    repair prompt carries the failing test output, and the test is then
    re-verified rather than trusted."""
    host = _host(tmp_path, ["src/calc.py — add two numbers\n",
                            WRONG, GOOD_TEST, RIGHT])
    session = _session(host, attempts=1)
    session.run(CALC_REQUEST)
    repair = _tails(host)[3]
    assert "`src/calc.py` does not work yet" in repair, repair
    assert "0 != 2" in repair, "the failing test output is missing"
    assert "assertEqual(add(1, 1), 2)" in repair, "the test is missing"
    assert "a + b" in (tmp_path / "src" / "calc.py").read_text()
    assert "add(1, 1), 2)" in (tmp_path / "tests" / "test_calc.py").read_text()
    end = _session_end(session)
    assert end["ok"], (end, session.report())
    assert sorted(end["files"]) == ["src/calc.py", "tests/test_calc.py"]


def test_the_reviewers_r05_scenario_with_pairing_ends_not_ok(tmp_path):
    """r05: a wrong `add` was sealed as verified because no test existed.
    With pairing on, the test exists, fails, and is not bent."""
    host = _host(tmp_path, ["src/alpha.py — add two numbers\n",
                            WRONG.replace("calc", "alpha"),
                            GOOD_TEST.replace("src.calc", "src.alpha"),
                            WRONG])
    session = _session(host, attempts=1, pair_tests=True)
    session.run("a module that adds two numbers")
    test_now = (tmp_path / "tests" / "test_alpha.py").read_text()
    assert test_now.strip() == \
        _body(GOOD_TEST.replace("src.calc", "src.alpha")).strip()
    assert not _session_end(session)["ok"]


def test_a_broken_test_file_is_still_repaired_as_a_test(tmp_path):
    """The test's OWN mistakes — an import of a name that does not exist —
    are the test's to fix. Only a disagreement about behaviour is not."""
    broken = GOOD_TEST.replace("import add", "import addition")
    host = _host(tmp_path, ["src/calc.py — add two numbers\n",
                            RIGHT, broken, GOOD_TEST])
    session = _session(host, attempts=2)
    session.run(CALC_REQUEST)
    assert _session_end(session)["ok"], session.report()


# --------------------------------------------------------------------------
# (c) zero tests collected is not a pass
# --------------------------------------------------------------------------

def test_a_test_file_outside_a_package_is_run_directly_and_verifies(
        tmp_path):
    """The case that USED to collect nothing: tests/ is not a package, so
    whole-project discovery walked past it — "Ran 0 tests", exit 0. The
    test phase is now scoped to the task's own test file (runner's
    `test_path`), which runs it directly, so a correct module verifies."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(_body(GOOD_TEST))
    host = _host(tmp_path, ["src/calc.py — add two numbers\n", RIGHT])
    session = _session(host, attempts=1)
    session.run("add two numbers")
    record = [r for r in session.history() if r.task_id == "t1"][-1]
    assert record.state == "committed" and record.verified, record
    assert _session_end(session)["ok"], session.report()


def test_an_existing_test_file_that_ran_nothing_does_not_verify(tmp_path):
    """A test file that exists but defines no tests: the scoped run reports
    "Ran 0 tests". The file exists; the module is NOT verified."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(
        "import unittest\n\nfrom src.calc import add  # noqa: F401\n")
    host = _host(tmp_path, ["src/calc.py — add two numbers\n", RIGHT])
    session = _session(host, attempts=1)
    session.run("add two numbers")
    record = [r for r in session.history() if r.task_id == "t1"][-1]
    assert record.state == "committed" and not record.verified
    outcome = session.outcomes[0]
    assert any("NOT verified" in c for c in outcome.caveats), outcome.caveats
    end = _session_end(session)
    assert end["unverified"] == ["src/calc.py"]
    assert not end["ok"]
    assert "NOT verified" in session.report()


def test_the_summary_counts_files_with_no_tests(tmp_path):
    host = _host(tmp_path, ["src/alpha.py — the first thing\n"
                            "src/beta.py — the second thing\n",
                            # the interface skeleton: two pinned stubs
                            "```python\n# file: src/alpha.py\nA: int = 1\n"
                            "```\n```python\n# file: src/beta.py\nB: int = 2"
                            "\n```",
                            "```python\nA = 1\n```", "```python\nB = 2\n```"])
    session = _session(host, attempts=1)
    session.run("two modules")
    end = _session_end(session)
    assert end["ok"]
    assert sorted(end["untested"]) == ["src/alpha.py", "src/beta.py"]
    assert "2 of 2 file(s) have no tests" in session.report()
