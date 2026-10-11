# SPDX-License-Identifier: Apache-2.0
"""The mutation check (mutation.py) and the pinned requirements
(constraints.py), as pure rules.

Mutation: the sites (comparisons, + - *, and/or, returned constants), none
under `if __name__ == "__main__"`, none on string concatenation; mutants
spread through the file and capped; each one a real, parseable change; the
check run in a COPY — the control must pass first, a weak test lets a
mutant survive, a strong one kills it, the project's own file is never
touched and the copy is removed.

Constraints: the requirement sentences taken word for word (capitals and
lower case), code blocks skipped, duplicates once, long ones cut and said
so, a cap with the rest counted; the block lands in every prompt's cached
prefix, after the conventions.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from cognitive_coder import LocalFileSystem, SubprocessExec, runner
from cognitive_coder import constraints as cons
from cognitive_coder import mutation as mut
from cognitive_coder.personas import ENGINEER, PromptBuilder

CODE = '''def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def ok(flag, other):
    return flag and other


def greet(name):
    return "hi " + name


if __name__ == "__main__":
    print(clamp(5, 0, 3) == 3)
'''


def test_the_sites_and_the_mutants():
    ms = mut.mutants(CODE, 50)
    whats = [m.what for m in ms]
    assert "`<` → `>=`" in whats and "`>` → `<=`" in whats
    assert "`and` → `or`" in whats
    assert not any("+" in w for w in whats), "string + left alone"
    assert all(m.line < 16 for m in ms), "nothing under the main guard"
    for m in ms:
        ast.parse(m.code)
        assert m.code != CODE
    few = mut.mutants(CODE, 2)
    assert len(few) == 2 and few[0].line < few[1].line, "spread, in order"
    assert mut.mutants("def broken(:\n", 5) == []
    assert [m.what for m in mut.mutants("def f():\n    return True\n")] \
        == ["`return True` → `return False`"]


def _project(tmp_path: Path, test_body: str) -> LocalFileSystem:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(
        "def bigger(a, b):\n    if a > b:\n        return a\n    return b\n")
    (tmp_path / "tests" / "test_calc.py").write_text(
        "import unittest\nfrom src.calc import bigger\n\n\n"
        "class T(unittest.TestCase):\n    def test_it(self):\n"
        f"        {test_body}\n")
    return LocalFileSystem(str(tmp_path))


def _check(fs):
    def run(**kw):
        return runner.run_tests("python", stem="calc", **kw)
    return mut.check(fs=fs, ex=SubprocessExec(), make_fs=LocalFileSystem,
                     run_tests=run, path="src/calc.py",
                     test_path="tests/test_calc.py",
                     code=fs.read("src/calc.py"), timeout=30)


def test_a_weak_test_lets_mutants_survive(tmp_path):
    fs = _project(tmp_path, "self.assertIsNotNone(bigger(2, 1))")
    before = fs.read("src/calc.py")
    report = _check(fs)
    assert report.tried, report.note
    assert report.survivors, "a test that checks nothing kills nothing"
    assert "may not check" in report.line()
    assert fs.read("src/calc.py") == before, "the project is untouched"
    assert not list((tmp_path / ".cc_state" / "mutants").rglob("*.py"))


def test_a_strong_test_kills_them(tmp_path):
    fs = _project(tmp_path, "self.assertEqual(bigger(2, 1), 2); "
                            "self.assertEqual(bigger(1, 2), 2)")
    report = _check(fs)
    assert report.tried and not report.survivors, report.line()
    assert report.line().endswith(
        f"caught {len(report.tried)} of {len(report.tried)} deliberate "
        f"bug{'' if len(report.tried) == 1 else 's'}")


def test_a_failing_control_is_not_judged(tmp_path):
    fs = _project(tmp_path, "self.assertEqual(bigger(2, 1), 99)")
    report = _check(fs)
    assert report.tried == []
    assert "do not pass in a copy" in report.note


def test_constraints_word_for_word():
    text = ("# Spec\n- The game MUST run offline.\n"
            "- Use pygame. Never use global variables.\n"
            "The ball should bounce. Do not add dependencies.\n"
            "```\nx = 1  # must not count\n```\n"
            "it must not crash. It must not crash.\n"
            "Plugins are not allowed. " + "You must keep " + "x" * 400
            + ".\n")
    found = cons.extract(text)
    assert found.items[:4] == ["The game MUST run offline.",
                               "Never use global variables.",
                               "Do not add dependencies.",
                               "it must not crash."]
    assert "Plugins are not allowed." in found.items
    assert found.items[-1].endswith("… (cut here)")
    assert not any("must not count" in c for c in found.items)
    assert cons.extract("Make a calculator.").block() == ""
    many = cons.extract("\n".join(f"- Rule {i} must hold." for i in
                                  range(20)))
    assert len(many.items) == cons.MAX_CONSTRAINTS and many.more == 5
    assert "and 5 more in the request itself" in many.block()


def test_the_block_sits_in_every_prompt_prefix():
    pb = PromptBuilder(conventions="Use four spaces.")
    pb.constraints = cons.extract("It MUST run offline.").block()
    prefix = pb.prefix_for(ENGINEER)
    assert prefix.index("[PROJECT CONVENTIONS]") \
        < prefix.index("[HARD REQUIREMENTS") \
        < prefix.index("It MUST run offline.")


# --------------------------------------------------------------------------
# in a session
# --------------------------------------------------------------------------

from cognitive_coder import (  # noqa: E402
    AutoApprove, Host, MemoryStorage, RecordingEvents, ScriptedLLM, Session,
    SessionConfig)

PLAN = "src/calc.py — arithmetic helpers\ntests/test_calc.py — its tests\n"
CALC = ("def bigger(a: int, b: int) -> int:\n"
        "    \"\"\"The larger of two numbers.\"\"\"\n"
        "    if a > b:\n        return a\n    return b\n")
WEAK_TEST = ("import unittest\n\nfrom src.calc import bigger\n\n\n"
             "class TestCalc(unittest.TestCase):\n"
             "    def test_bigger(self):\n"
             "        self.assertIsNotNone(bigger(2, 1))\n\n\n"
             "if __name__ == \"__main__\":\n    unittest.main()\n")


def _session(tmp_path, replies, **cfg):
    llm = ScriptedLLM(replies, supports_tools=False)
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    return Session(host, config=SessionConfig(
        attempts=2, pin_interfaces=False, review_after_build=False,
        **cfg)), host, llm


def test_a_build_with_a_weak_test_is_warned_about(tmp_path):
    session, host, llm = _session(
        tmp_path, [PLAN, f"```python\n{CALC}```",
                   f"```python\n{WEAK_TEST}```"])
    session.run("a helper that returns the bigger of two numbers, with "
                "tests")
    assert session.mutation, "the verified module was checked"
    rep = session.mutation[0]
    assert rep.path == "src/calc.py" and rep.survivors, rep.line()
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("deliberate bug" in m and "may not check" in m
               for m in warnings), warnings
    assert "[mutation]  src/calc.py: tests/test_calc.py caught" in \
        session.report()
    log = (tmp_path / "BUILD_LOG.txt").read_text()
    assert "MUTATION CHECK" in log and "NOT CAUGHT" in log
    assert (tmp_path / "src" / "calc.py").read_text() == CALC


def test_the_check_can_be_switched_off(tmp_path):
    session, host, llm = _session(
        tmp_path, [PLAN, f"```python\n{CALC}```",
                   f"```python\n{WEAK_TEST}```"], mutation_check=False)
    session.run("a helper that returns the bigger of two numbers, with "
                "tests")
    assert session.mutation == []
    assert "MUTATION CHECK" not in (tmp_path / "BUILD_LOG.txt").read_text()


def test_the_requests_rules_are_in_every_prompt(tmp_path):
    session, host, llm = _session(
        tmp_path, [PLAN, f"```python\n{CALC}```",
                   f"```python\n{WEAK_TEST}```"], mutation_check=False)
    session.run("A helper that returns the bigger of two numbers. It MUST "
                "not import anything. Do not print.")
    assert len(llm.prompts) >= 3
    for prompt in llm.prompts:
        text = "\n".join(m.content for m in prompt)
        assert "[HARD REQUIREMENTS" in text
        assert "It MUST not import anything." in text
        assert "Do not print." in text
    log = (tmp_path / "BUILD_LOG.txt").read_text()
    assert "HARD REQUIREMENTS — from the request" in log
    assert "[pinned]    2 hard requirement(s)" in session.report()


def test_a_debug_session_keeps_the_specs_rules(tmp_path):
    from cognitive_coder import DebugReport
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(CALC.replace("a > b",
                                                           "a > c"))
    (tmp_path / "BUILD_SPEC.md").write_text("# Spec\nIt must never use "
                                            "global variables.\n")
    session, host, llm = _session(tmp_path, [f"```python\n{CALC}```"])
    session.debug(DebugReport(
        error_text="NameError: name 'c' is not defined",
        files=("src/calc.py",)))
    text = "\n".join(m.content for m in llm.prompts[0])
    assert "It must never use global variables." in text
