# SPDX-License-Identifier: Apache-2.0
"""The values at the failure (testrun.py, 2026-10-27).

The harness-owned runner prints, under every failing test, what the
variables were in the frame that failed and in the test; the runner
attaches them to the diagnostic; the feedback shows them. The same
runner, the same arguments, the same output otherwise — proved here
against real subprocesses.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from cognitive_coder import diagnostics, runner, testrun
from cognitive_coder.ports import LocalFileSystem, SubprocessExec
from cognitive_coder.types import Diagnostic

PHYSICS = '''\
import math


def bearing_between(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    heading = math.degrees(math.atan2(dy, dx)) % 360.0
    return heading


def divide(a, b):
    ratio = a / b
    return ratio
'''

UNITTEST = '''\
import unittest

from src.physics import bearing_between, divide


class Bearing(unittest.TestCase):
    def test_north_is_zero(self):
        expected = 90.0
        self.assertEqual(bearing_between(0, 0, 1, 1), expected)

    def test_divide_by_zero(self):
        divisor = 0
        divide(10, divisor)

    def test_passes(self):
        self.assertEqual(divide(4, 2), 2)


if __name__ == "__main__":
    unittest.main()
'''

PYTEST = '''\
from src.physics import bearing_between


def test_north_is_zero():
    expected = 90.0
    assert bearing_between(0, 0, 1, 1) == expected
'''


def _project(tmp_path: Path, test_text: str) -> LocalFileSystem:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "physics.py").write_text(PHYSICS)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_physics.py").write_text(test_text)
    return LocalFileSystem(str(tmp_path))


def test_rewrite_argv_only_touches_the_two_runners():
    assert testrun.rewrite_argv(["python", "-m", "unittest", "discover",
                                 "-s", "tests"], "/p/.cc_state/r.py") == \
        ["python", "/p/.cc_state/r.py", "unittest", "discover", "-s",
         "tests"]
    assert testrun.rewrite_argv(["python", "-m", "pytest", "-q", "x.py"],
                                "r.py") == ["python", "r.py", "pytest",
                                            "-q", "x.py"]
    assert testrun.rewrite_argv(["python", "main.py"], "r.py") == \
        ["python", "main.py"]
    assert testrun.rewrite_argv(["cargo", "test"], "r.py") == ["cargo",
                                                               "test"]


def test_the_script_is_written_once_and_rewritten_when_stale(tmp_path):
    fs = LocalFileSystem(str(tmp_path))
    assert testrun.ensure_script(fs) == testrun.SCRIPT_REL
    written = tmp_path / ".cc_state" / "cc_testrun.py"
    assert written.exists()
    written.write_text("# stale\n")
    assert testrun.ensure_script(fs) == testrun.SCRIPT_REL
    assert written.read_text() == testrun.SCRIPT


def test_parse_values_reads_the_blocks_back():
    out = ("Traceback (most recent call last):\n"
           '  File "tests/test_x.py", line 9, in test_a\n'
           "AssertionError: 45.0 != 90.0\n"
           "[VALUES AT THE FAILURE] src/physics.py:7 in bearing_between\n"
           "    x1 = 0\n"
           "    dy = 1\n"
           "[VALUES AT THE FAILURE] tests/test_x.py:9 in test_a\n"
           "    expected = 90.0\n"
           "\n"
           "Ran 1 test in 0.001s\n")
    blocks = testrun.parse_values(out)
    assert [(b.file, b.line, b.function) for b in blocks] == [
        ("src/physics.py", 7, "bearing_between"),
        ("tests/test_x.py", 9, "test_a")]
    assert blocks[0].rows == ("    x1 = 0", "    dy = 1")


def test_attach_puts_each_block_on_its_own_diagnostic():
    blocks = testrun.parse_values(
        "[VALUES AT THE FAILURE] src/physics.py:7 in bearing_between\n"
        "    x1 = 0\n"
        "[VALUES AT THE FAILURE] tests/test_x.py:9 in test_a\n"
        "    expected = 90.0\n")
    located = Diagnostic(file="src/physics.py", line=7, severity="error",
                         message="ZeroDivisionError")
    unlocated = Diagnostic(severity="failure", message="AssertionError")
    out = testrun.attach([located, unlocated], blocks)
    assert "x1 = 0" in out[0].values and "expected" not in out[0].values
    # the unlocated failure takes what is left
    assert "expected = 90.0" in out[1].values


def test_feedback_shows_the_values_under_the_excerpt():
    d = Diagnostic(file="src/physics.py", line=7, severity="error",
                   message="AssertionError: 45.0 != 90.0",
                   source_excerpt=">>    7 | return heading",
                   values="values at the failure (src/physics.py:7 in "
                          "bearing_between):\n    x1 = 0")
    text = diagnostics.feedback([d])
    assert text.index("return heading") < text.index("x1 = 0")
    assert "values at the failure" in text


@pytest.mark.slow
def test_unittest_failures_carry_the_values_of_both_frames(tmp_path):
    fs = _project(tmp_path, UNITTEST)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              path="src/physics.py",
                              test_path="tests/test_physics.py")
    assert not result.ok
    output = result.output
    # the same output as the plain runner, plus the blocks
    assert "Ran 3 tests" in output
    assert "FAILED (failures=1, errors=1)" in output
    assert output.count(testrun.MARK) >= 3
    # the error raised inside the module carries the module's locals
    # (its arguments first) and then the test's …
    zero = next(d for d in result.diagnostics
                if "test_divide_by_zero" in d.message)
    assert "a = 10" in zero.values and "b = 0" in zero.values
    assert "divisor = 0" in zero.values
    assert zero.values.index("b = 0") < zero.values.index("divisor = 0")
    # … and the assertion, raised in the test, carries the test's frame:
    # the expected value it compared against
    assertion = next(d for d in result.diagnostics
                     if "45.0 != 90.0" in d.message)
    assert "expected = 90.0" in assertion.values
    # the plain diagnostics are what they always were
    assert assertion.file.endswith("test_physics.py") and assertion.line == 9


@pytest.mark.slow
def test_pytest_failures_carry_the_values_as_a_section(tmp_path):
    pytest.importorskip("pytest")
    fs = _project(tmp_path, PYTEST)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              path="src/physics.py",
                              test_path="tests/test_physics.py")
    assert not result.ok
    assert "FAILED tests/test_physics.py::test_north_is_zero" in \
        result.output
    assert testrun.MARK in result.output
    assert any("expected = 90.0" in d.values for d in result.diagnostics)


@pytest.mark.slow
def test_values_off_runs_the_plain_command(tmp_path):
    fs = _project(tmp_path, UNITTEST)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              path="src/physics.py",
                              test_path="tests/test_physics.py",
                              values=False)
    assert not result.ok
    assert testrun.MARK not in result.output
    assert not (tmp_path / ".cc_state" / "cc_testrun.py").exists()


@pytest.mark.slow
def test_a_passing_run_is_unchanged_and_the_script_is_not_a_test(tmp_path):
    fs = _project(tmp_path, "import unittest\n\n\nclass T(unittest."
                            "TestCase):\n    def test_ok(self):\n"
                            "        self.assertTrue(True)\n")
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              path="src/physics.py",
                              test_path="tests/test_physics.py")
    assert result.ok, result.output
    assert "Ran 1 test" in result.output


def test_the_script_is_stdlib_only_and_parses():
    import ast
    tree = ast.parse(testrun.SCRIPT)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"os", "sys", "types", "unittest", "pytest"}


def test_the_script_reports_a_bad_mode(tmp_path):
    import subprocess
    fs = LocalFileSystem(str(tmp_path))
    testrun.ensure_script(fs)
    proc = subprocess.run([sys.executable, str(tmp_path / ".cc_state" /
                                               "cc_testrun.py"), "nope"],
                          capture_output=True, text=True, cwd=str(tmp_path))
    assert proc.returncode == 2 and "usage" in proc.stderr
