# SPDX-License-Identifier: Apache-2.0
"""Verification is proportional to evidence, and blame goes where it belongs.

Two builds of the racing-game spec (Aug 8 session 2, Oct 1) died the same
way: a `tests/test_physics.py` left behind by an EARLIER session failed to
import, `unittest discover` picked it up, and every module of the new build
"failed" on an error in a file it did not own. The model regenerated its
correct file, byte for byte, and stagnation stopped the build.

A third pattern sealed files as DONE: a test file that collected zero tests
— once because the extractor had written a hallucinated "existing file" in
place of the test — passed, because nothing contradicted it.

These tests pin: a test file must collect tests; a failure that does not
name a module is not that module's failure; a failure the session saw
BEFORE the build began is never charged to the build; and the outcome says
"verified" only when the file's own tests ran and passed.
"""

from __future__ import annotations

import pytest

from cognitive_coder import runner
from cognitive_coder.ports import LocalFileSystem, SubprocessExec
from cognitive_coder.types import Diagnostic, TaskOutcome


@pytest.fixture
def workspace(tmp_path):
    return LocalFileSystem(str(tmp_path)), SubprocessExec()


MATH3D = ('from typing import NamedTuple\n\n\n'
          'class ProjectedSegment(NamedTuple):\n'
          '    x: float\n    y: float\n    width: float\n\n\n'
          'def project(z: float) -> float:\n    return 1.0 / z\n')

PHYSICS = ('class CarPhysics:\n    def __init__(self):\n'
           '        self.speed = 0.0\n')

#: Left behind by an earlier session: imports a name the new plan renamed.
STALE_TEST_PHYSICS = ('import unittest\n\nfrom src.physics import CarState\n\n\n'
                      'class T(unittest.TestCase):\n'
                      '    def test_speed(self):\n'
                      '        self.assertEqual(CarState().speed, 0.0)\n')

#: Looks like a test file, defines nothing unittest can find. This is the
#: literal shape of Aug 8's "DONE tests/test_math3d.py" — the extractor had
#: kept the model's imagined copy of math3d.py instead of the test.
NOT_A_TEST = ('class ProjectedSegment:\n'
              '    def __init__(self, x, y, depth, curve, color_pattern):\n'
              '        self.x = x\n')

REAL_TEST_MATH3D = ('import unittest\n\nfrom src.math3d import project\n\n\n'
                    'class T(unittest.TestCase):\n'
                    '    def test_project(self):\n'
                    '        self.assertEqual(project(2.0), 0.5)\n')


def _racing_project(fs):
    fs.write("src/__init__.py", "")
    fs.write("tests/__init__.py", "")
    fs.write("src/math3d.py", MATH3D)
    fs.write("src/physics.py", PHYSICS)


# --------------------------------------------------------------------------
# a test file must collect tests
# --------------------------------------------------------------------------

def test_a_test_file_that_collects_nothing_is_a_failure_with_a_fixable_message(
        workspace):
    fs, ex = workspace
    _racing_project(fs)
    fs.write("tests/test_math3d.py", NOT_A_TEST)
    result = runner.verify(NOT_A_TEST, "python", fs=fs, ex=ex,
                           stem="test_math3d", path="tests/test_math3d.py")
    assert not result.ok, result.summary()
    assert result.diagnostics, "the model must be told what to fix"
    d = result.diagnostics[0]
    assert d.code == "no-tests-collected"
    assert d.file == "tests/test_math3d.py"
    assert "unittest.TestCase" in d.message
    assert "not the module it tests" in d.message


def test_a_test_file_that_collects_tests_still_passes(workspace):
    fs, ex = workspace
    _racing_project(fs)
    fs.write("tests/test_math3d.py", REAL_TEST_MATH3D)
    result = runner.verify(REAL_TEST_MATH3D, "python", fs=fs, ex=ex,
                           stem="test_math3d", path="tests/test_math3d.py")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert not any("ZERO" in c for c in result.caveats)


def test_a_test_task_is_scoped_to_itself_so_a_stale_neighbour_cannot_fail_it(
        workspace):
    """The test task has no `test_path`: the file IS the test. Its verdict
    comes from that file alone; the stale neighbour becomes a caveat."""
    fs, ex = workspace
    _racing_project(fs)
    fs.write("tests/test_physics.py", STALE_TEST_PHYSICS)
    fs.write("tests/test_math3d.py", REAL_TEST_MATH3D)
    result = runner.verify(REAL_TEST_MATH3D, "python", fs=fs, ex=ex,
                           stem="test_math3d", path="tests/test_math3d.py")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert not result.diagnostics
    assert any("test_physics" in c for c in result.caveats), result.caveats


# --------------------------------------------------------------------------
# a module is not charged with another file's failure
# --------------------------------------------------------------------------

def test_a_module_without_its_own_test_is_not_failed_by_a_stale_test_elsewhere(
        workspace):
    """Oct 1: `src/math3d.py` "failed" twice on
    `ImportError: cannot import name 'CarState' from 'src.physics'`."""
    fs, ex = workspace
    _racing_project(fs)
    fs.write("tests/test_physics.py", STALE_TEST_PHYSICS)
    result = runner.verify(MATH3D, "python", fs=fs, ex=ex, stem="math3d",
                           path="src/math3d.py",
                           test_path="tests/test_math3d.py")   # not written yet
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert not result.diagnostics, result.diagnostics
    note = " ".join(result.caveats)
    assert "test_physics" in note and "not this file's errors" in note


def test_a_failure_that_names_the_module_is_still_the_modules_failure(
        workspace):
    """The same stale test, verifying `src/physics.py`: the error names
    `src.physics`, so it IS this file's to answer — unless the session's
    baseline says it was failing before the build (next test)."""
    fs, ex = workspace
    _racing_project(fs)
    fs.write("tests/test_physics.py", STALE_TEST_PHYSICS)
    result = runner.verify(PHYSICS, "python", fs=fs, ex=ex, stem="physics",
                           path="src/physics.py",
                           test_path="tests/test_physics_new.py")
    assert not result.ok
    assert any("CarState" in d.message for d in result.diagnostics)


def test_a_failure_the_baseline_already_knew_is_never_charged_to_the_build(
        workspace):
    fs, ex = workspace
    _racing_project(fs)
    fs.write("tests/test_physics.py", STALE_TEST_PHYSICS)
    result = runner.verify(PHYSICS, "python", fs=fs, ex=ex, stem="physics",
                           path="src/physics.py",
                           test_path="tests/test_physics_new.py",
                           known_failing=("tests.test_physics",))
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert not result.diagnostics
    assert any("test_physics" in c for c in result.caveats)


def test_an_unattributed_failure_stays_a_failure(workspace):
    """Honest default: when the log names no other test module, the file
    under verification keeps the failure rather than being waved through."""
    fs, ex = workspace
    _racing_project(fs)
    diags = (Diagnostic(file="", message="something broke", severity="error"),)
    mine, others = runner._attribute_failures(
        "Traceback (most recent call last):\n  boom\nFAILED (errors=1)\n",
        "src/math3d.py", diags)
    assert not mine and not others     # → run_tests keeps ok=False


# --------------------------------------------------------------------------
# reading the log
# --------------------------------------------------------------------------

def test_failing_test_modules_reads_unittest_and_pytest_logs():
    log = (
        "ERROR: tests.test_physics (unittest.loader._FailedTest.tests.test_physics)\n"
        "FAIL: test_add (tests.test_calc.C.test_add)\n"
        "FAIL: test_scale (test_math3d.TestMath3D.test_scale)\n"
        "FAILED tests/test_render.py::test_draw - AssertionError\n"
        "FAIL: test_add (tests.test_calc.C.test_add)\n")
    assert runner.failing_test_modules(log) == [
        "tests.test_physics", "tests.test_calc", "test_math3d",
        "tests/test_render.py"]


def test_is_test_path_knows_the_usual_shapes():
    yes = ["tests/test_math3d.py", "test_x.py", "src/x_test.go",
           "spec/thing.spec.ts", "__tests__/a.js", "test/a.py"]
    no = ["src/math3d.py", "main.py", "src/testing_tools.py", "test.py"]
    assert all(runner.is_test_path(p) for p in yes)
    assert not any(runner.is_test_path(p) for p in no), [
        p for p in no if runner.is_test_path(p)]


# --------------------------------------------------------------------------
# the outcome says what it can honestly say
# --------------------------------------------------------------------------

def test_an_outcome_distinguishes_verified_from_merely_built():
    built = TaskOutcome(task_id="t1", path="src/track.py", ok=True)
    verified = TaskOutcome(task_id="t2", path="src/physics.py", ok=True,
                           verified=True)
    failed = TaskOutcome(task_id="t3", path="src/main.py", ok=False,
                         stopped_because="gave up")
    assert built.label == "BUILT (not verified)"
    assert verified.label == "VERIFIED"
    assert failed.label == "FAILED"
    assert "built, not verified" in built.summary()
    assert "verified in" in verified.summary()
    assert "DONE" not in built.summary() + verified.summary()
