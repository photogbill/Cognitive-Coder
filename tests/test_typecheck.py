# SPDX-License-Identifier: Apache-2.0
"""The type checker's gate in the build loop (typecheck.py, 2026-10-27).

A wrong attribute name, a call with the wrong arguments, an undefined name
— found at write time in the file as written, in the project, whether or
not a test reaches the line. The real mypy where it is installed; a fake
checker for the loop's own behaviour, so the suite needs none.
"""

from __future__ import annotations

import json
import shutil

import pytest

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
    typecheck,
)
from cognitive_coder.types import Diagnostic, ProcResult

SHIP = '''\
from dataclasses import dataclass


@dataclass
class Ship:
    x: float
    y: float

    def move(self, dx: float) -> None:
        self.x += dx
'''

USES = '''\
from src.ship import Ship


def go() -> float:
    s = Ship(x=1.0, y=2.0)
    s.move(dx=1.0, dy=2.0)
    return s.heading + unknown_name
'''


class _Exec:
    """An ExecPort whose checker answers with what we say."""

    def __init__(self, tool: str = "", stdout: str = "", stderr: str = "",
                 exit_code: int = 0, timed_out: bool = False) -> None:
        self.tool, self.stdout, self.stderr = tool, stdout, stderr
        self.exit_code, self.timed_out = exit_code, timed_out
        self.calls: list[list[str]] = []

    def run(self, argv, *, cwd, timeout, stdin="", env=None):
        self.calls.append(list(argv))
        return ProcResult(exit_code=self.exit_code, stdout=self.stdout,
                          stderr=self.stderr, timed_out=self.timed_out)

    def which(self, binary):
        return f"/usr/bin/{binary}" if binary == self.tool else None


class _FS:
    def root(self):
        return "/project"


def test_which_prefers_the_pyright_family_then_mypy():
    assert typecheck.which(_Exec("mypy")) == ("mypy", "/usr/bin/mypy")
    assert typecheck.which(_Exec("pyright")) == ("pyright",
                                                 "/usr/bin/pyright")
    assert typecheck.which(_Exec("")) == ("", "")


def test_no_checker_is_a_note_with_what_it_costs():
    res = typecheck.check(_Exec(""), _FS(), ["src/a.py"])
    assert not res.ran and "no type checker installed" in res.note
    assert "offline" in res.note


def test_mypy_output_is_read_with_its_codes():
    out = ('/project/src/uses.py:6:5: error: Unexpected keyword argument '
           '"dy" for "move" of "Ship"  [call-arg]\n'
           '/project/src/uses.py:7:12: error: "Ship" has no attribute '
           '"heading"  [attr-defined]\n'
           '/project/src/uses.py:7:24: error: Name "unknown_name" is not '
           'defined  [name-defined]\n'
           '/project/src/uses.py:7:5: error: Returning Any from function '
           'declared to return "float"  [no-any-return]\n'
           '/project/src/other.py:2:1: error: "Ship" has no attribute "z"'
           '  [attr-defined]\n')
    res = typecheck.check(_Exec("mypy", stdout=out), _FS(), ["src/uses.py"])
    assert res.ran and res.tool == "mypy"
    assert [d.code for d in res.diagnostics] == [
        "call-arg", "attr-defined", "name-defined", "no-any-return",
        "attr-defined"]
    assert res.diagnostics[0].file == "src/uses.py"
    assert res.diagnostics[0].line == 6 and res.diagnostics[0].col == 5
    gated = typecheck.gate(res.diagnostics, path="src/uses.py")
    assert [d.code for d in gated] == ["call-arg", "attr-defined",
                                       "name-defined"]
    assert all(d.file == "src/uses.py" and d.severity == "error"
               for d in gated)
    assert "certain error at runtime" in gated[0].message


def test_mypy_missing_from_the_project_environment_is_said_not_gated():
    res = typecheck.check(_Exec("mypy", stderr="No module named mypy",
                                exit_code=1), _FS(), ["src/a.py"])
    assert not res.ran and "project's environment" in res.note


def test_pyright_json_is_read_with_its_rules():
    data = {"generalDiagnostics": [
        {"file": "/project/src/uses.py", "severity": "error",
         "message": 'Cannot access attribute "heading" for class "Ship"\n'
                    '  Attribute "heading" is unknown',
         "range": {"start": {"line": 6, "character": 11}},
         "rule": "reportAttributeAccessIssue"},
        {"file": "/project/src/uses.py", "severity": "warning",
         "message": "Import \"numpy\" could not be resolved",
         "range": {"start": {"line": 0, "character": 0}},
         "rule": "reportMissingImports"}]}
    res = typecheck.check(_Exec("pyright", stdout=json.dumps(data)), _FS(),
                          ["src/uses.py"])
    assert res.ran and res.tool == "pyright"
    assert res.diagnostics[0].line == 7 and res.diagnostics[0].col == 12
    assert res.diagnostics[0].message == \
        'Cannot access attribute "heading" for class "Ship"'
    gated = typecheck.gate(res.diagnostics, path="src/uses.py")
    assert [d.code for d in gated] == ["reportAttributeAccessIssue"]


def test_pyright_that_prints_no_json_is_a_note():
    res = typecheck.check(_Exec("pyright", stdout="boom", exit_code=2),
                          _FS(), ["src/a.py"])
    assert not res.ran and "did not return JSON" in res.note


def test_a_timed_out_checker_is_a_note():
    res = typecheck.check(_Exec("mypy", timed_out=True), _FS(), ["a.py"])
    assert not res.ran and "did not finish" in res.note


def test_the_gate_keeps_only_certain_codes_in_the_file():
    diags = [Diagnostic(file="src/a.py", line=3, code="attr-defined",
                        message="m", tool="typecheck:mypy"),
             Diagnostic(file="src/a.py", line=4, code="arg-type",
                        message="m", tool="typecheck:mypy"),
             Diagnostic(file="src/b.py", line=1, code="attr-defined",
                        message="m", tool="typecheck:mypy"),
             Diagnostic(file="src/a.py", line=0, code="attr-defined",
                        message="unlocated", tool="typecheck:mypy")]
    assert [d.line for d in typecheck.gate(diags, path="src/a.py")] == [3]
    assert typecheck.gate(diags, codes=(), path="src/a.py") == []


# --------------------------------------------------------------------------
# the real checker, where it is installed
# --------------------------------------------------------------------------

class _OnlyTool(SubprocessExec):
    """The real exec, admitting to one checker only."""

    def __init__(self, tool: str) -> None:
        super().__init__()
        self._tool = tool

    def which(self, binary):
        if binary in typecheck.TOOLS and binary != self._tool:
            return None
        return super().which(binary)


def _real_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "ship.py").write_text(SHIP)
    (tmp_path / "src" / "uses.py").write_text(USES)
    return LocalFileSystem(str(tmp_path))


#: The same three bugs, as each family names them.
_ATTR = {"attr-defined", "reportAttributeAccessIssue"}
_CALL = {"call-arg", "reportCallIssue"}
_NAME = {"name-defined", "reportUndefinedVariable"}


@pytest.mark.slow
@pytest.mark.parametrize("tool", typecheck.TOOLS)
def test_a_real_checker_finds_the_three_certain_errors(tmp_path, tool):
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} not installed")
    fs = _real_project(tmp_path)
    res = typecheck.check(_OnlyTool(tool), fs, ["src/uses.py"],
                          root=str(tmp_path))
    assert res.ran, res.note
    assert res.tool == tool
    codes = {d.code for d in typecheck.gate(res.diagnostics,
                                            path="src/uses.py")}
    assert codes & _ATTR and codes & _CALL and codes & _NAME, codes
    # the sibling it imports is checked silently: no finding names it
    assert not any(d.file.endswith("ship.py") for d in res.diagnostics)
    if tool == "mypy":
        # the cache lands in the engine's own state directory
        assert (tmp_path / ".cc_state" / ".mypy_cache").exists()


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

PLAN = "src/alpha.py — add two numbers\n"
OK_FILE = "```python\ndef add(a, b):\n    return a + b\n```"


class _GatedExec(SubprocessExec):
    """A real exec whose `mypy` is a fake that reports one attr-defined
    error on the first call and nothing after."""

    def __init__(self) -> None:
        super().__init__()
        self.checks = 0

    def which(self, binary):
        if binary == "mypy":
            return "/fake/mypy"
        if binary in typecheck.TOOLS:
            return None
        return super().which(binary)

    def run(self, argv, *, cwd, timeout, stdin="", env=None):
        if argv and argv[0] == "/fake/mypy":
            self.checks += 1
            path = argv[-1].replace("\\", "/")
            if self.checks == 1:
                return ProcResult(exit_code=1, stdout=(
                    f'{path}:2:12: error: "int" has no attribute "nope"  '
                    f'[attr-defined]\n'))
            return ProcResult(exit_code=0, stdout="")
        return super().run(argv, cwd=cwd, timeout=timeout, stdin=stdin,
                           env=env)


def _host(tmp_path, replies, ex):
    return Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=ex,
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def test_the_gate_fails_the_attempt_with_a_located_error_then_passes(
        tmp_path):
    ex = _GatedExec()
    host = _host(tmp_path, [PLAN, OK_FILE, OK_FILE], ex)
    session = Session(host, config=SessionConfig(
        attempts=2, adjudicate=False, review_after_build=False,
        lint_gate=(), type_gate=typecheck.GATE_CODES, mutation_check=False))
    outcomes = session.run("add two numbers")
    out = [o for o in outcomes if o.path == "src/alpha.py"][-1]
    assert out.ok and len(out.attempts) == 2, [a.note for a in out.attempts]
    first = out.attempts[0]
    gated = [d for d in first.diagnostics if d.code == "attr-defined"]
    assert gated and gated[0].file == "src/alpha.py" and gated[0].line == 2
    assert any("type checker found" in c for c in first.diagnostics[0].message
               or "") or any("type checker" in w for _k, w, _d in
                             host.events.of("warning"))
    events = [e for e in session.journal.events() if e["event"] == "typecheck"]
    assert events and events[0]["data"]["tool"] == "mypy"
    assert events[0]["data"]["gated"][0]["code"] == "attr-defined"
    assert ex.checks == 2


def test_the_gate_is_off_for_scripted_doubles_unless_asked(tmp_path):
    ex = _GatedExec()
    host = _host(tmp_path, [PLAN, OK_FILE], ex)
    session = Session(host, config=SessionConfig(
        attempts=1, adjudicate=False, review_after_build=False,
        lint_gate=(), mutation_check=False))
    outcomes = session.run("add two numbers")
    assert outcomes[-1].ok
    assert ex.checks == 0
    assert session.loop.config.type_gate == ()


def test_an_absent_checker_is_said_once_and_not_asked_again(tmp_path):
    class NoChecker(SubprocessExec):
        def __init__(self):
            super().__init__()
            self.asked = 0

        def which(self, binary):
            if binary in typecheck.TOOLS:
                self.asked += 1
                return None
            return super().which(binary)
    ex = NoChecker()
    host = _host(tmp_path, [PLAN, OK_FILE], ex)
    session = Session(host, config=SessionConfig(
        attempts=1, adjudicate=False, review_after_build=False,
        lint_gate=(), type_gate=typecheck.GATE_CODES, mutation_check=False))
    session.run("add two numbers")
    notes = [m for _k, m, _d in host.events.of("warning")
             if "no type checker installed" in m]
    assert len(notes) == 1
    assert session.loop._typecheck_off
