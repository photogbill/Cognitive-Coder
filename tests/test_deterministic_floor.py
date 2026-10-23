# SPDX-License-Identifier: Apache-2.0
"""The deterministic floor — 2026-10-22.

`format_code` and `lint_code` were defined, tested and exported to ATK, and
called by nothing. The operator's two complaints from the 10-20 builds —
indentation, mismatched names — were detected by ruff and thrown away.
Now the reply is formatted before it is written, and a lint finding that is
a certain bug (an undefined name, a redefinition, a syntax error) joins the
verify verdict: green tests with a NameError the tests never reached are
not green.

Needs ruff on PATH; skipped without it, and that skip is itself the
degraded mode the warning names.
"""

from __future__ import annotations

import shutil

import pytest

from cognitive_coder import (AutoApprove, Host, LocalFileSystem,
                             MemoryStorage, RecordingEvents, ScriptedLLM,
                             Session, SessionConfig, SubprocessExec)

pytestmark = pytest.mark.skipif(shutil.which("ruff") is None,
                                reason="ruff not installed")

PLAN = "src/alpha.py — add two numbers\n"
#: parses, formats badly, and `helper` calls a name that exists nowhere;
#: no test file, so the run alone would pass and the file would be "ok"
UNDEFINED = ("```python\ndef add(a,b):\n    return a+b\n\n\n"
             "def helper():\n    return nowhere(1)\n```")
UGLY = "```python\ndef add(a,b):\n  return   a+b\n```"


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def test_an_undefined_name_the_tests_never_reach_is_not_verified(tmp_path):
    host = _host(tmp_path, [PLAN, UNDEFINED])
    session = Session(host, config=SessionConfig(
        attempts=1, adjudicate=False, review_after_build=False))
    outcomes = session.run("add two numbers")
    out = [o for o in outcomes if o.path == "src/alpha.py"][-1]
    assert not out.ok, "a NameError in the file is not a verified file"
    assert any("linter found" in c for c in out.caveats), out.caveats
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("F821" in m and "nowhere" in m for m in warnings), warnings
    # the next attempt would be shown the linter's exact line
    diag = [d for d in out.attempts[-1].diagnostics if d.code == "F821"]
    assert diag and diag[0].line == 6 and diag[0].file == "src/alpha.py"


def test_the_reply_is_formatted_before_it_is_written(tmp_path):
    host = _host(tmp_path, [PLAN, UGLY])
    session = Session(host, config=SessionConfig(
        attempts=1, adjudicate=False, review_after_build=False))
    session.run("add two numbers")
    text = (tmp_path / "src" / "alpha.py").read_text()
    assert text == "def add(a, b):\n    return a + b\n", repr(text)
    fixes = [e for e in session.journal.events() if e["event"] == "autofix"]
    assert fixes and "formatted the file" in fixes[-1]["data"]["fixes"]


def test_the_floor_can_be_switched_off(tmp_path):
    host = _host(tmp_path, [PLAN, UGLY])
    session = Session(host, config=SessionConfig(
        attempts=1, adjudicate=False, review_after_build=False,
        format_before_write=False, lint_gate=()))
    session.run("add two numbers")
    text = (tmp_path / "src" / "alpha.py").read_text()
    assert "return   a+b" in text or "return a+b" in text   # ruff --fix may touch spacing; the formatter did not run
