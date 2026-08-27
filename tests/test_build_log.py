# SPDX-License-Identifier: Apache-2.0
"""A build that explains itself, in a file you can hand to someone.

Bill: *"Can you add the logging and have it automatically output to a file in
the code project folder? Then I can just give you that folder each time
instead of lots of copy and paste and screenshots."*

That is the requirement, and it sets the bar: **the folder alone has to be
enough.** Not the folder plus a screenshot of the console, not the folder plus
"it said something about zero tests" — the folder.

WHAT THE JSONL COULD NOT DO. It is a provenance record: what was produced, by
which model, from which prompt, and whether verification passed. Right for
defending a change; wrong for diagnosing an engine. For every file of every
build it recorded

    "verify": {"run": "ok", "test": "ok", "caveats": [...]}

— a VERDICT, with the evidence thrown away. The evidence was two lines:

    Ran 0 tests in 0.000s
    OK

and had they been written down, the bug that made four green builds
meaningless would have been obvious the same afternoon.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.journal import SessionLog                 # noqa: E402
from cognitive_coder.types import (                            # noqa: E402
    PhaseResult, ProcResult, RunResult)


class _FS:
    """The FileSystemPort's byte interface, in memory."""

    def __init__(self):
        self.files: dict[str, bytes] = {}

    def exists(self, path):
        return path in self.files

    def read_bytes(self, path):
        return self.files[path]

    def write_bytes(self, path, data):
        self.files[path] = data


def _zero_tests() -> RunResult:
    """The exact result that hid the bug, reconstructed."""
    return RunResult(ok=True, lang="python", phases=(
        PhaseResult(
            name="test",
            argv=("python", "-m", "unittest", "discover", "-s", ".",
                  "-p", "test_*.py", "-v"),
            proc=ProcResult(exit_code=0, stdout="",
                            stderr="Ran 0 tests in 0.000s\n\nOK"),
            ok=True),
    ), caveats=("the test command succeeded but ran ZERO tests",))


def _log(fs=None, sid="cc-test-0001"):
    return SessionLog(fs if fs is not None else _FS(), sid)


# ==========================================================================
# where it lands
# ==========================================================================

def test_it_is_a_findable_file_in_the_project():
    """`.cc_journal/cc-20260808-134722-a23ed6.log` is technically in the
    project and practically invisible. A diagnostic nobody can find is a
    diagnostic nobody uses."""
    fs = _FS()
    log = _log(fs)
    log.start("build a thing", "some-model", {"attempts": 4})
    assert "BUILD_LOG.txt" in fs.files, list(fs.files)
    assert SessionLog.FILENAME == "BUILD_LOG.txt"


def test_it_says_what_else_is_in_the_folder():
    """The point is handing over ONE folder, so the file names the rest."""
    fs = _FS()
    _log(fs).start("build a thing", "m", {})
    text = fs.files["BUILD_LOG.txt"].decode()
    for companion in ("BUILD_SPEC.md", "Recommendation.md", ".cc_journal",
                      ".cc_snapshots"):
        assert companion in text, companion


def test_a_second_build_appends_rather_than_replacing():
    """Comparing this run against the last one is most of what it is for."""
    fs = _FS()
    _log(fs, "session-one").start("first request", "m", {})
    _log(fs, "session-two").start("second request", "m", {})
    text = fs.files["BUILD_LOG.txt"].decode()
    assert text.count("SESSION ") == 2, text.count("SESSION ")
    assert "first request" in text and "second request" in text
    #: The preamble is written once, not once per run.
    assert text.count("COGNITIVE CODER — BUILD LOG") == 1


# ==========================================================================
# what it keeps
# ==========================================================================

def test_the_command_and_its_output_are_verbatim():
    """The whole reason the file exists."""
    fs = _FS()
    log = _log(fs)
    log.start("r", "m", {})
    log.phases("tests/test_physics.py", 1, _zero_tests())
    text = fs.files["BUILD_LOG.txt"].decode()

    assert "$ python -m unittest discover -s . -p test_*.py -v" in text
    assert "Ran 0 tests in 0.000s" in text, "the evidence, not the verdict"
    assert "OK" in text
    assert "CAVEAT: the test command succeeded but ran ZERO tests" in text


def test_the_generated_code_is_kept_with_its_numbers():
    fs = _FS()
    log = _log(fs)
    log.start("r", "m", {})
    log.generation("src/math3d.py", 2, temperature=0.15, seed=11,
                   tokens_in=1466, tokens_out=855,
                   prompt_ms=18656, decode_ms=110000,
                   text="def project(x, y, z):\n    return x / z, y / z\n")
    text = fs.files["BUILD_LOG.txt"].decode()
    assert "generate src/math3d.py — attempt 2" in text
    assert "temperature=0.15 seed=11" in text
    assert "prefill=18656ms decode=110000ms" in text
    #: Prefill and decode were conflated once; the rate is the number that
    #: makes either of them meaningful.
    assert "7.8 tok/s" in text, text
    assert "return x / z, y / z" in text


def test_a_skipped_task_is_in_the_file_too():
    fs = _FS()
    log = _log(fs)
    log.start("r", "m", {})
    log.event("SKIPPED src/main.py",
              "needs src/render.py, which did not build")
    assert "SKIPPED src/main.py" in fs.files["BUILD_LOG.txt"].decode()


def test_truncation_says_how_much_it_dropped():
    """A silently shortened log is the same failure as a silently shortened
    prompt: the reader cannot tell 'nothing more' from 'more, withheld'."""
    fs = _FS()
    log = _log(fs)
    log.start("r", "m", {})
    log.block("output", "x" * 9000, limit=100)
    text = fs.files["BUILD_LOG.txt"].decode()
    assert "more characters not" in text, text[-300:]
    assert "8,900" in text, "it must say HOW much"


# ==========================================================================
# it must never be the thing that breaks a build
# ==========================================================================

def test_a_filesystem_that_refuses_costs_nothing():
    class _Refuses(_FS):
        def write_bytes(self, path, data):
            raise PermissionError(path)

    log = _log(_Refuses())
    log.start("r", "m", {})                     # must not raise
    log.phases("x.py", 1, _zero_tests())
    log.generation("x.py", 1, temperature=0.1, seed=None, tokens_in=1,
                   tokens_out=1, prompt_ms=1, decode_ms=1, text="x")
    log.close("done")


@pytest.mark.parametrize("bad", [None, 0, [], {}])
def test_odd_results_do_not_raise(bad):
    """`phases()` is handed whatever the runner produced, and a build must
    survive a shape it did not expect."""
    log = _log()
    log.start("r", "m", {})
    log.phases("x.py", 1, bad)


def test_the_session_and_loop_are_wired_to_it():
    root = Path(__file__).resolve().parent.parent / "cognitive_coder"
    session = (root / "session.py").read_text(encoding="utf-8")
    loop = (root / "loop.py").read_text(encoding="utf-8")

    assert "SessionLog" in session
    assert "self.log.start(" in session
    assert "self.loop.log = self.log" in session, "the loop needs it too"
    assert "self.log.rule(\"PLAN\")" in session
    #: The verify phases are logged from the loop, which is where they happen.
    assert "_log_verify" in loop
    assert "log.generation(" in loop
