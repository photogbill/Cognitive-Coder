# SPDX-License-Identifier: Apache-2.0
"""Debug & Repair (debugrepair.py): the operator's own run, fed back.

Held here, against a real project folder and a scripted model:

* a pasted traceback is located in THIS project's file (the deepest project
  frame, made relative), repaired through the ordinary loop, and recorded:
  the diff, `.cc_state/debug/<session>.md`, a history line, BUILD_LOG.txt;
* with a command, the error is captured by running it, and the same command
  run again CONFIRMS the fix — "fixed" only when it now runs clean;
* a command that runs clean, with nothing else said, is nothing to repair;
* a description alone is located by ONE model call naming files from the
  project's list (an invented name is dropped); `confirm_targets` stops
  there with the proposal and changes nothing; the confirmed files repair;
* a repair that changes nothing is said to have changed nothing;
* the same error after a repair is "still failing", bounded by `rounds`;
* the pure pieces: the command as an argument list, a path made relative
  (Windows too), both ends of a long error kept, a program still running
  at the timeout without an error counted as clean.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cognitive_coder import (
    AutoApprove,
    DebugReport,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
    debug_history,
)
from cognitive_coder import debugrepair as dr
from cognitive_coder.types import Diagnostic, ProcResult

CALC = '''def add(a: int, b: int) -> int:
    """The sum."""
    total = a + c
    return total


def mul(a: int, b: int) -> int:
    """The product."""
    return a * b
'''
FIXED = CALC.replace("a + c", "a + b")
TEST = '''import unittest

from src.calc import add, mul


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mul(self):
        self.assertEqual(mul(2, 3), 6)


if __name__ == "__main__":
    unittest.main()
'''
MAIN = "from src.calc import add\n\nprint(add(2, 3))\n"


def _project(tmp_path: Path, calc: str = CALC) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(calc)
    (tmp_path / "tests" / "test_calc.py").write_text(TEST)
    (tmp_path / "main.py").write_text(MAIN)
    return tmp_path


def _session(tmp_path: Path, replies: list[str]) -> tuple[Session, Host]:
    llm = ScriptedLLM(replies, supports_tools=False)
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    return Session(host, config=SessionConfig(
        attempts=2, review_after_build=False)), host


def _traceback(root: Path) -> str:
    return (f'Traceback (most recent call last):\n'
            f'  File "{root / "main.py"}", line 3, in <module>\n'
            f'    print(add(2, 3))\n'
            f'  File "{root / "src" / "calc.py"}", line 3, in add\n'
            f'    total = a + c\n'
            f"NameError: name 'c' is not defined\n")


def _cmd() -> str:
    return f'"{sys.executable}" main.py'


def test_a_pasted_traceback_is_located_repaired_and_recorded(tmp_path):
    root = _project(tmp_path)
    session, host = _session(root, [f"```python\n{FIXED}```"])
    result = session.debug(DebugReport(error_text=_traceback(root)))
    assert result.state == "repaired", result.verdict
    assert "run it yourself to confirm" in result.verdict
    assert (root / "src" / "calc.py").read_text() == FIXED
    assert list(result.diffs) == ["src/calc.py"]
    assert "+    total = a + b" in result.diffs["src/calc.py"]
    rnd = result.rounds[0]
    assert rnd.targets == ["src/calc.py"]
    assert rnd.how == "the error names src/calc.py:3"
    report = (root / result.report_path).read_text()
    assert "## Changes to `src/calc.py`" in report
    rows = debug_history(host.fs)
    assert rows[-1]["state"] == "repaired"
    assert rows[-1]["changed"] == ["src/calc.py"]
    log = (root / "BUILD_LOG.txt").read_text()
    assert "DEBUG & REPAIR" in log and "DEBUG round 1" in log
    assert "DEBUG REPAIRED src/calc.py" in log or "DEBUG " in log
    assert "changes to src/calc.py" in log


def test_a_command_is_run_to_capture_and_again_to_confirm(tmp_path):
    root = _project(tmp_path)
    session, host = _session(root, [f"```python\n{FIXED}```"])
    result = session.debug(DebugReport(command=_cmd(), run_timeout=30))
    assert result.state == "fixed", (result.verdict, result.last_output)
    assert result.rounds[0].check == "no error — it exited normally"
    assert "5" in result.last_output
    log = (root / "BUILD_LOG.txt").read_text()
    assert "the run, before any repair" in log
    assert "NameError" in log
    assert "the run after round 1" in log


def test_a_command_that_runs_clean_is_nothing_to_repair(tmp_path):
    root = _project(tmp_path, calc=FIXED)
    session, host = _session(root, [])
    result = session.debug(DebugReport(command=_cmd(), run_timeout=30))
    assert result.state == "nothing-to-repair"
    assert "showed no error" in result.verdict
    assert result.diffs == {}


def test_a_description_is_located_by_the_model_and_confirmed_first(tmp_path):
    root = _project(tmp_path)
    locate = json.dumps({"files": ["src/calc.py", "src/invented.py"],
                         "why": "add() computes the total"})
    session, host = _session(root, [locate])
    report = DebugReport(description="adding 2 and 3 crashes",
                         confirm_targets=True)
    result = session.debug(report)
    assert result.state == "located"
    assert result.targets == ["src/calc.py"], "an invented name is dropped"
    assert result.why == "add() computes the total"
    assert (root / "src" / "calc.py").read_text() == CALC, "nothing changed"
    # the operator said yes: the named file is repaired
    session2, _ = _session(root, [f"```python\n{FIXED}```"])
    result = session2.debug(DebugReport(
        description="adding 2 and 3 crashes", files=("src/calc.py",)))
    assert result.state == "repaired", result.verdict
    assert result.rounds[0].how == "named by the operator"
    assert len(debug_history(host.fs)) == 2


def test_a_repair_that_changes_nothing_says_so(tmp_path):
    root = _project(tmp_path)
    session, _ = _session(root, [f"```python\n{CALC}```"] * 3)
    result = session.debug(DebugReport(error_text=_traceback(root)))
    assert result.state == "no-change", result.verdict
    assert result.diffs == {}


def test_a_failed_repair_is_put_back_and_said(tmp_path):
    root = _project(tmp_path)
    still = CALC.replace("a + c", "a + c + 0")       # fails its own tests
    session, _ = _session(root, [f"```python\n{still}```"] * 6)
    result = session.debug(DebugReport(command=_cmd(), rounds=2,
                                       run_timeout=30))
    assert result.state == "no-change", result.verdict
    assert "put back as it was" in result.verdict
    assert (root / "src" / "calc.py").read_text() == CALC, \
        "the operator's file never holds the model's failed attempt"


def test_progress_then_nothing_more_is_still_failing(tmp_path):
    root = _project(tmp_path)
    broken_main = MAIN + "print(undefined_name)\n"
    (root / "main.py").write_text(broken_main)
    session, _ = _session(root, [f"```python\n{FIXED}```",
                                 f"```python\n{broken_main}```"] + [
        f"```python\n{broken_main}```"] * 4)
    result = session.debug(DebugReport(command=_cmd(), rounds=3,
                                       run_timeout=30))
    assert result.state == "still-failing", result.verdict
    assert result.rounds[0].check.startswith("a different error")
    assert result.rounds[1].targets == ["main.py"]
    assert "round 2 could not change anything more" in result.verdict
    assert list(result.diffs) == ["src/calc.py"], "round 1's fix is kept"


def test_a_failing_test_points_at_the_code_it_tests(tmp_path):
    root = _project(tmp_path, calc=CALC.replace("a + c", "a - b"))
    tb = (f'Traceback (most recent call last):\n'
          f'  File "{root / "tests" / "test_calc.py"}", line 8, in '
          f'test_add\n'
          f'    self.assertEqual(add(2, 3), 5)\n'
          f'AssertionError: -1 != 5\n')
    session, _ = _session(root, [f"```python\n{FIXED}```"])
    result = session.debug(DebugReport(error_text=tb))
    assert result.rounds[0].targets == ["src/calc.py"], result.rounds[0].how
    assert "a test — so src/calc.py" in result.rounds[0].how
    assert (root / "tests" / "test_calc.py").read_text() == TEST, \
        "the test is never bent to the code"
    assert result.state == "repaired", result.verdict


def test_a_command_that_cannot_start_or_uses_a_shell_is_said(tmp_path):
    root = _project(tmp_path)
    session, _ = _session(root, [])
    result = session.debug(DebugReport(command="no-such-program-xyz run"))
    assert result.state == "command-failed", result.verdict
    assert "could not be started here" in result.verdict
    assert (root / "src" / "calc.py").read_text() == CALC
    session2, _ = _session(root, [])
    result = session2.debug(DebugReport(command="cd src && python x.py"))
    assert result.state == "command-failed"
    assert "shell feature" in result.verdict


def test_a_clean_run_before_the_repair_cannot_confirm_it(tmp_path):
    root = _project(tmp_path, calc=FIXED)
    (root / "src" / "calc.py").write_text(FIXED.replace("a * b", "a * b + 1"))
    locate = json.dumps({"files": ["src/calc.py"], "why": "mul is there"})
    session, _ = _session(root, [locate, f"```python\n{FIXED}```"])
    result = session.debug(DebugReport(command=_cmd(), run_timeout=30,
                                       description="mul is off by one"))
    assert result.state == "repaired", result.verdict
    assert "cannot confirm the fix" in result.verdict
    assert "fixed" not in result.verdict.split(" — ")[0]


def test_a_program_that_wants_keyboard_input_is_not_judged_fixed(tmp_path):
    root = _project(tmp_path)
    (root / "main.py").write_text("from src.calc import add\n"
                                  "n = int(input('n? '))\nprint(add(n, 1))\n")
    session, _ = _session(root, [f"```python\n{FIXED}```"])
    result = session.debug(DebugReport(command=_cmd(), run_timeout=30,
                                       files=("src/calc.py",)))
    assert result.state == "repaired", result.verdict
    assert "keyboard input" in result.verdict


def test_a_game_still_running_at_the_timeout_counts_as_clean(tmp_path):
    root = _project(tmp_path)
    (root / "src" / "loop.py").write_text("import tkinter  # a GUI\n")
    (root / "main.py").write_text(
        "import time\nfrom src.calc import add\nprint(add(2, 3))\n"
        "while True:\n    time.sleep(0.1)\n")
    session, _ = _session(root, [f"```python\n{FIXED}```"])
    result = session.debug(DebugReport(command=_cmd(), run_timeout=5))
    assert result.state == "fixed", (result.verdict, result.last_output)
    assert "still running after 5 s" in result.rounds[0].check


def test_the_locate_step_sees_top_level_files_not_the_venv(tmp_path):
    root = _project(tmp_path)
    (root / "game.py").write_text("print('hi')\n")
    venv = root / ".venv" / "Lib" / "site-packages" / "aaa"
    venv.mkdir(parents=True)
    (venv / "mod.py").write_text("x = 1\n")
    session, _ = _session(root, [])
    listed = dr._listed(session)
    assert "game.py" in listed and "main.py" in listed
    assert not any(p.startswith(".venv") for p in listed)


def test_a_failure_before_the_run_is_a_result_not_an_exception(tmp_path):
    from cognitive_coder.errors import CognitiveCoderError
    root = _project(tmp_path)
    session, host = _session(root, [])

    def refuse(_caps):
        raise CognitiveCoderError("This model is not a coding model.")

    session._check_model = refuse
    result = session.debug(DebugReport(error_text=_traceback(root)))
    assert result.state == "error"
    assert "not a coding model" in result.verdict
    assert debug_history(host.fs)[-1]["state"] == "error"


def test_nothing_given_is_said():
    result = dr.run(None, DebugReport())
    assert result.state == "nothing-to-repair"
    assert "paste the error" in result.verdict


def test_the_pure_pieces():
    assert dr.split_command('python "my game/main.py" --fast') == \
        ["python", "my game/main.py", "--fast"]
    assert dr.split_command("  ") == []

    class FS:
        def exists(self, p):
            return p in ("src/calc.py",)

    assert dr.relative("D:\\Proj\\src\\calc.py", "d:\\proj", FS()) == \
        "src/calc.py"
    assert dr.relative("/p/src/calc.py", "/p", FS()) == "src/calc.py"
    assert dr.relative("/usr/lib/python3/json.py", "/p", FS()) == ""
    assert dr.relative("src/calc.py", "/p", FS()) == "src/calc.py"
    long = "A" * 5000 + "TAIL"
    clipped = dr.clip(long, 100)
    assert clipped.startswith("A") and clipped.endswith("TAIL")
    assert "characters left out" in clipped
    assert dr.command_problem("python main.py > out.txt")
    assert dr.command_problem('python -c "print(1 < 2)"') == ""

    class FS2:
        def exists(self, p):
            return p == "game.py"

        def root(self):
            return "/p"

    class Reads:
        def __init__(self, loop):
            self.loop = loop

        def __call__(self, p):
            return ""

        def built_to_run(self, script):
            return self.loop

    hung = ProcResult(exit_code=-1, timed_out=True)
    run = dr.judge("[still running]", hung, "/p", FS2(), "game.py",
                   Reads(True))
    assert run.kind == "running" and run.clean()
    run = dr.judge("[still running]", hung, "/p", FS2(), "game.py",
                   Reads(False))
    assert run.kind == "hung" and not run.clean()
    run = dr.judge("could not run: [WinError 2]",
                   ProcResult(exit_code=-1, stderr="could not run: x"),
                   "/p", FS2(), "", Reads(False))
    assert run.kind == "not-started"
    assert dr.judge("ok", ProcResult(exit_code=0), "/p", FS2(), "",
                    Reads(False)).kind == "clean"
    a = dr._error_keys({"x.py": [Diagnostic(
        file="x.py", line=3, message="KeyError at 0x7f00ab")]}, [])
    b = dr._error_keys({"x.py": [Diagnostic(
        file="x.py", line=4, message="KeyError at 0x7f99cd")]}, [])
    assert a == b, "a moved line or a new address is the same error"
