# SPDX-License-Identifier: Apache-2.0
"""Review of the 2026-10-27 change, loop/verification half — reproducers.

Every test here FAILS against the code as reviewed; its docstring says
what is wrong and what the correct behaviour is. One test per defect.
Helpers come from tests/test_loop.py and tests/test_resample.py.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import _host, _loop  # noqa: E402
from test_resample import (  # noqa: E402
    RIGHT,
    TEST,
    WRONG_BOTH,
    WRONG_ONE,
    _task,
)

from cognitive_coder import typecheck  # noqa: E402
from cognitive_coder.journal import Journal, SessionLog  # noqa: E402
from cognitive_coder.loop import _differing_definitions  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    SubprocessExec,
)
from cognitive_coder.types import Diagnostic, ProcResult  # noqa: E402


def _project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(TEST)


def _resample_event(loop):
    return [e for e in loop.journal.events()
            if e.get("event") == "resample"]


# --------------------------------------------------------------------------
# SAMPLE AGAIN BEFORE REPAIRING — the disagreement map
# --------------------------------------------------------------------------

def test_a_cosmetic_difference_is_not_a_disagreement():
    """`_differing_definitions` compares whitespace-split text, so a
    comment, a docstring word or `g(x,1)` vs `g(x, 1)` counts as the two
    samples DISAGREEING about the definition. The loop's own identity
    (`_code_identity`, the AST) says the code is the same. A definition
    whose AST did not change is not one the model was unsure of."""
    a = "def f(x):\n    return x * 2  # double\n\n\ndef g(x):\n    return x * 3\n"
    b = "def f(x):\n    return x * 2  # twice\n\n\ndef g(x):\n    return x * 3\n"
    assert _differing_definitions(a, b) == []
    s1 = "def f(x):\n    return g(x,1)\n"
    s2 = "def f(x):\n    return g(x, 1)\n"
    assert _differing_definitions(s1, s2) == []


def test_a_sample_skipped_as_the_same_code_does_not_pollute_the_doubt_map(
        tmp_path):
    """A sample that differs from attempt 1 only in a comment is skipped as
    "the same code as attempt 1" — and STILL counted in `names_changed`,
    because `differs_in` is computed before the identity check. The
    journal then says the samples disagreed in `double`, the doubt map
    ranks `double` least sure (0.0) and the repair prompt carries a
    [WHERE THE SAMPLES DISAGREED] block pointing at a definition no sample
    disagreed on. Correct: `disagreed == []`, no doubt, no block."""
    _project(tmp_path)
    same_with_comment = WRONG_ONE.replace(
        "return x * 2\n\n\ndef triple", "return x * 2  # twice\n\n\ndef triple")
    assert same_with_comment != WRONG_ONE
    host = _host(tmp_path, [WRONG_ONE, same_with_comment, RIGHT])
    loop = _loop(host, attempts=2, resample=1, narrow="off", type_gate=())
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok, out.stopped_because
    ev = _resample_event(loop)
    assert ev[0]["data"]["candidates"][0]["skipped"] == \
        "the same code as attempt 1"
    assert ev[0]["data"]["disagreed"] == []
    assert loop.doubt.get("calc.py", {}).get("double", 1.0) == 1.0
    repair = [m.content for m in host.llm.prompts[2] if m.role == "user"][0]
    assert "[WHERE THE SAMPLES DISAGREED]" not in repair


def test_a_disagreement_in_a_class_body_reaches_the_record(tmp_path):
    """The samples disagree in a CLASS body — a dataclass field, a class
    constant — and in no function. `_differing_definitions` reports the
    class; `_resample` then keeps only names that are functions or
    methods (`present`), so the journal says `disagreed: []`, the doubt
    map is empty and the repair is never told. A class's fields are
    exactly the cross-file surface this change is about. Correct: the
    class is named."""
    test = ("import unittest\n\nfrom calc import double\n\n\n"
            "class T(unittest.TestCase):\n    def test_double(self):\n"
            "        self.assertEqual(double(2), 4)\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(test)

    def reply(limit: int) -> str:
        return (f"```python\nclass Config:\n    LIMIT = {limit}\n\n\n"
                f"def double(x):\n    return x * Config.LIMIT\n```")
    host = _host(tmp_path, [reply(1), reply(3), reply(2)])
    loop = _loop(host, attempts=2, resample=1, narrow="off", type_gate=())
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="double")
    assert out.ok, out.stopped_because
    ev = _resample_event(loop)
    assert ev[0]["data"]["candidates"][0]["differs_in"] == ["Config"]
    assert ev[0]["data"]["disagreed"] == ["Config"], ev[0]["data"]


def test_a_narrowed_first_attempt_is_not_resampled_as_a_whole_file(
        tmp_path):
    """With `narrow="always"` (what a profile with edit_format "symbols"
    sets, or the operator) attempt 1 on an existing file is a NARROWED
    edit: `sent` asks for only the definitions that change and the reply
    is put back by name. `_resample_wanted` excludes a canvas and a FIM
    attempt 1 but not a narrowed one, so `_resample` re-sends the
    narrowed prompt and treats the reply — a fragment — as the WHOLE
    file. Here the second narrowed reply is the CORRECT `double`: put
    back by name it passes; judged as a whole file it lacks `triple`,
    fails both tests and is thrown away, `triple` is recorded as a
    definition the samples "disagreed" on, and attempt 2 is paid for.
    Correct: no resample of a narrowed attempt 1 (two attempts, no
    `resample` event), or the sample placed by name and kept (one)."""
    _project(tmp_path)
    (tmp_path / "calc.py").write_text(
        "def double(x):\n    return x + x + 1\n\n\n"
        "def triple(x):\n    return x * 3\n")
    wrong = "```python\ndef double(x):\n    return x + 1\n```"
    fixed = "```python\ndef double(x):\n    return x * 2\n```"
    host = _host(tmp_path, [wrong, fixed, fixed])
    loop = _loop(host, attempts=2, resample=1, narrow="always",
                 keep_existing=False, type_gate=())
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="fix double")
    assert out.ok, out.stopped_because
    ev = _resample_event(loop)
    if ev:
        data = ev[0]["data"]
        assert "triple" not in data["disagreed"], data
        assert data["kept"] == 1 and len(out.attempts) == 1, data
    else:
        assert len(out.attempts) == 2


# --------------------------------------------------------------------------
# SAMPLE AGAIN BEFORE REPAIRING — what the records say
# --------------------------------------------------------------------------

def test_the_build_log_shows_the_kept_samples_verify_output(tmp_path):
    """BUILD_LOG.txt: attempt 1's `verify` block is written BEFORE the
    resample, so it holds the FIRST reply's test output (two failures);
    the kept sample's verify is never logged (`_resample` calls `_verify`
    but not `_log_verify`), and neither reply's code is logged beside
    its own verdict. Attempt 1's `generate` block then shows the kept
    sample's code. A reader sees code that fails one test under a verify
    that failed two, and the kept sample's output nowhere. Correct: the
    kept sample's verify output (`FAILED (failures=1)`) is in the log
    under attempt 1."""
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_BOTH, WRONG_ONE, RIGHT])
    loop = _loop(host, attempts=2, resample=1, narrow="off", type_gate=())
    loop.journal = Journal(host.fs, "t")
    loop.log = SessionLog(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok, out.stopped_because
    text = (tmp_path / "BUILD_LOG.txt").read_text()
    first = text.index("verify calc.py — attempt 1")
    second = text.index("verify calc.py — attempt 2")
    attempt1 = text[first:second]
    assert "FAILED (failures=1)" in attempt1, attempt1


def test_the_resample_record_counts_the_samples_actually_drawn(tmp_path):
    """`resample=3`: the first extra sample passes, so the loop stops
    after one decode. The journal's `resample` event still says
    `samples: 3`, the BUILD_LOG headline and the status say "sampled the
    same prompt 3 more time(s)", and an extra sample's `generate` label
    would read "resample k of 3". Two replies were paid for; the record
    claims four. Correct: the record says how many were drawn (1)."""
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_BOTH, RIGHT])
    loop = _loop(host, attempts=2, resample=3, narrow="off", type_gate=())
    loop.journal = Journal(host.fs, "t")
    loop.log = SessionLog(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok and len(host.llm.prompts) == 2
    ev = _resample_event(loop)
    assert ev[0]["data"]["samples"] == 1, ev[0]["data"]
    log = (tmp_path / "BUILD_LOG.txt").read_text()
    assert "3 more time(s)" not in log


# --------------------------------------------------------------------------
# THE TYPE CHECKER'S GATE
# --------------------------------------------------------------------------

class _Exec:
    """An ExecPort whose checker answers with what we say."""

    def __init__(self, tool: str = "", stdout: str = "", stderr: str = "",
                 exit_code: int = 0, timed_out: bool = False) -> None:
        self.tool, self.stdout, self.stderr = tool, stdout, stderr
        self.exit_code, self.timed_out = exit_code, timed_out

    def run(self, argv, *, cwd, timeout, stdin="", env=None):
        return ProcResult(exit_code=self.exit_code, stdout=self.stdout,
                          stderr=self.stderr, timed_out=self.timed_out)

    def which(self, binary):
        return f"/usr/bin/{binary}" if binary == self.tool else None


class _FS:
    def root(self):
        return "/project"


def test_a_checker_that_did_not_run_is_not_a_clean_bill():
    """`typecheck.check` never reads the exit code. An ExecPort that could
    not start the tool (`exit_code=-1`, `could not run: …` on stderr —
    what SubprocessExec returns for an OSError), a mypy that died with a
    fatal message carrying no line (`Duplicate module named`, exit 2) or
    with an INTERNAL ERROR, all come back `ran=True` with zero findings
    and no note: the Evaluate tab lists the tool under `tools_ran`, the
    loop journals `findings: 0`, and the operator is told the file was
    checked. Correct: `ran` is False and the note says why — the stated
    cost of an optional tool."""
    res = typecheck.check(_Exec("mypy", exit_code=-1,
                                stderr="could not run: [Errno 13] "
                                       "Permission denied"),
                          _FS(), ["src/a.py"])
    assert not res.ran and res.note, res
    res = typecheck.check(_Exec("pyright", exit_code=-1,
                                stderr="could not run: [WinError 2] The "
                                       "system cannot find the file"),
                          _FS(), ["src/a.py"])
    assert not res.ran and res.note, res
    res = typecheck.check(_Exec("mypy", exit_code=2, stdout=(
        'src/a.py: error: Duplicate module named "a" (also at '
        '"./b/a.py")\n')), _FS(), ["src/a.py"])
    assert not res.ran and res.note, res


@pytest.mark.slow
def test_the_gate_does_not_fail_legal_attribute_assignment(tmp_path):
    """`attr-defined` / `reportAttributeAccessIssue` are gated as "a
    certain error at runtime". Both fire on `s.score = 5` followed by
    `s.score` when `Ship` does not declare `score` — ordinary Python that
    runs (an instance takes new attributes), and exactly what a small
    model writes from `main.py` into another file's class. The attempt
    passes its tests and is failed by the gate; the repair is told to fix
    a certain error in code that works. Correct: no gated finding for
    an attribute the file itself assigns."""
    ex = SubprocessExec()
    if not typecheck.which(ex)[0]:
        pytest.skip("no type checker installed")
    code = ("class Ship:\n    def __init__(self):\n        self.x = 0\n\n\n"
            "def main():\n    s = Ship()\n    s.score = 5\n"
            "    return s.score\n")
    (tmp_path / "ext.py").write_text(code)
    res = typecheck.check(ex, LocalFileSystem(str(tmp_path)), ["ext.py"],
                          root=str(tmp_path))
    assert res.ran, res.note
    gated = typecheck.gate(res.diagnostics, path="ext.py")
    assert gated == [], [d.one_line() for d in gated]


def test_mypy_call_arg_fact_names_the_class_the_error_names():
    """mypy says `Unexpected keyword argument "dy" for "move" of "Ship"`.
    enrich's new pattern keeps `move` and drops `of "Ship"`, so the fact
    is looked up by bare name and comes back as the FIRST `move` in the
    project — `Enemy.move(self, dx, dy)`, a signature that HAS `dy`,
    presented as the certain truth under an error saying it has not.
    The traceback shape carried `Ship.move`; the checker's must too."""
    from cognitive_coder import enrich
    from cognitive_coder.codemap import CodeMap
    from cognitive_coder.ports import MemoryFileSystem
    fs = MemoryFileSystem({k: v.encode() for k, v in {
        "src/__init__.py": "",
        "src/enemy.py": "class Enemy:\n    def move(self, dx, dy):\n"
                        "        pass\n",
        "src/ship.py": "class Ship:\n    def move(self, dx):\n"
                       "        pass\n"}.items()})
    cm = CodeMap(fs, MemoryStorage())
    cm.index_project()
    d = Diagnostic(file="src/uses.py", line=6, severity="error",
                   code="call-arg", tool="typecheck:mypy",
                   message='Unexpected keyword argument "dy" for "move" of '
                           '"Ship" (found by the type checker in the file '
                           'as written)')
    out = enrich.facts_for([d], codemap=cm, fs=fs)
    assert "src/enemy.py" not in out, out
    assert "def move(self, dx)" in out, out


class _TimingOutExec:
    """A real ExecPort whose mypy never finishes."""

    def __init__(self) -> None:
        self._real = SubprocessExec()
        self.checks = 0

    def which(self, binary):
        if binary == "mypy":
            return "/fake/mypy"
        if binary in typecheck.TOOLS:
            return None
        return self._real.which(binary)

    def run(self, argv, *, cwd, timeout, stdin="", env=None):
        if argv and argv[0] == "/fake/mypy":
            self.checks += 1
            return ProcResult(exit_code=-1, timed_out=True)
        return self._real.run(argv, cwd=cwd, timeout=timeout, stdin=stdin,
                              env=env)


def test_a_checker_that_timed_out_is_not_retried_in_silence(tmp_path):
    """`_type_gate` says the first note of ANY kind once
    (`_said_typecheck`) but switches the checker off only for absence.
    A checker that timed out (120s, not configurable) is therefore run
    again on every attempt of every file for the rest of the session,
    and from the second time on nothing is said: two minutes per
    attempt, unexplained. Correct: a timed-out checker is switched off
    with its cost said, or every timeout is said."""
    _project(tmp_path)
    ex = _TimingOutExec()
    host = Host(llm=ScriptedLLM([WRONG_ONE, RIGHT], supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=ex,
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=2, resample=0, narrow="off",
                 type_gate=typecheck.GATE_CODES)
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok, out.stopped_because
    notes = [m for _k, m, _d in host.events.of("warning")
             if "did not finish" in m]
    assert ex.checks == 1 or len(notes) == ex.checks, (ex.checks, notes)


def test_a_loop_with_the_default_config_does_not_run_a_checker_on_doubles(
        tmp_path):
    """`LoopConfig.resample` defaults to 0 ("the session sets 1 for a
    real model port and 0 for the scripted doubles"); `LoopConfig.
    type_gate` defaults to the full GATE_CODES, and `Host.exec` defaults
    to the real SubprocessExec. So every Loop built directly — every
    test in tests/test_loop.py, test_narrow.py, test_resample.py … — runs
    whatever checker is on PATH on every attempt (about a second each
    with pyright), and its verdict depends on what the machine has
    installed. Correct: the two defaults agree — off until the session
    or the host asks."""
    class Counting(SubprocessExec):
        asked = 0

        def which(self, binary):
            if binary in typecheck.TOOLS:
                Counting.asked += 1
            return super().which(binary)
    _project(tmp_path)
    host = Host(llm=ScriptedLLM([RIGHT], supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=Counting(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=1, narrow="off")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok
    assert Counting.asked == 0, "a scripted double was type-checked"
