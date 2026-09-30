# SPDX-License-Identifier: Apache-2.0
"""Per-tool fixer, linter and formatter commands.

A language's lint, format and fix slots each accept several tools, but they
used to share ONE argv template. `flake8 file` and `black -q file` are right;
`ruff file` and `ruff -q file` are rejected by every ruff since 0.5
("unrecognized subcommand"), and ruff is the first Python lint tool found in
any environment with the dev extras. So Python linting and formatting had been
silently doing nothing. Go had the same shape of bug in the fix slot: its
template is `{fmt} -w` (gofmt), but the fixer was chosen from the LINT tools
first, so it ran `go -w file`.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

import pytest

from cognitive_coder import langs, runner
from cognitive_coder.ports import LocalFileSystem, SubprocessExec


class _Which:
    """An ExecPort whose `which` answers from a fixed table."""

    def __init__(self, table: dict[str, str]):
        self.table = table

    def which(self, name: str) -> str | None:
        return self.table.get(name)


def _real_ruff() -> str:
    found = shutil.which("ruff")
    if found:
        return found
    beside = Path(sys.executable).parent / "ruff"
    return str(beside) if beside.exists() else ""


class _VenvExec(SubprocessExec):
    """The real ExecPort, able to see the venv's own ruff even when the
    venv's bin directory is not on PATH (pytest run as .venv/bin/python)."""

    def __init__(self, ruff: str, hide: tuple[str, ...] = ()):
        super().__init__()
        self._ruff = ruff
        self._hide = hide

    def which(self, binary: str) -> str | None:
        if binary in self._hide:
            return None
        if binary == "ruff":
            return self._ruff
        return super().which(binary)


# --------------------------------------------------------------------------
# the lookup itself — no toolchain needed
# --------------------------------------------------------------------------

def test_ruff_gets_its_own_subcommands():
    py = langs.get("python")
    assert py.cmd_for("lint", "/x/ruff")[:2] == ["{lint}", "check"]
    assert py.cmd_for("fmt", "C:\\tools\\ruff.exe")[:2] == ["{fmt}",
                                                            "format"]
    assert py.cmd_for("fix", "ruff")[:3] == ["{lint}", "check", "--fix"]


def test_other_tools_keep_the_generic_template():
    py = langs.get("python")
    assert py.cmd_for("lint", "/usr/bin/flake8") == py.lint_cmd
    assert py.cmd_for("fmt", "black") == py.fmt_cmd


def test_python_has_no_fixer_without_ruff():
    """flake8 and pyflakes have no --fix; running `flake8 check --fix`
    would be an error dressed up as a fix."""
    py = langs.get("python")
    assert py.fix_command(_Which({"flake8": "/bin/flake8",
                                  "black": "/bin/black"})) == ("", [])


def test_go_fixes_with_gofmt_not_go():
    go = langs.get("go")
    tool, cmd = go.fix_command(_Which({"go": "/go/bin/go",
                                       "gofmt": "/go/bin/gofmt"}))
    assert tool == "/go/bin/gofmt"
    assert langs.render(cmd, fmt=tool, lint=tool, src="m.go")[:2] == \
        ["/go/bin/gofmt", "-w"]


def test_javascript_still_fixes_with_eslint():
    js = langs.get("javascript")
    tool, cmd = js.fix_command(_Which({"eslint": "/n/eslint"}))
    assert tool == "/n/eslint" and "--fix" in cmd


# --------------------------------------------------------------------------
# the real ruff
# --------------------------------------------------------------------------

UNUSED = "import os\n\n\ndef f(a):\n    return a\n"


@pytest.fixture
def ruff():
    found = _real_ruff()
    if not found:
        pytest.skip("ruff is not installed, so there is nothing to run")
    return found


@pytest.mark.toolchain
def test_autofix_with_real_ruff_removes_an_unused_import(ruff, tmp_path):
    fixed, done = runner.autofix(UNUSED, "python",
                                 fs=LocalFileSystem(str(tmp_path)),
                                 ex=_VenvExec(ruff))
    assert "import os" not in fixed
    assert "def f(a):" in fixed
    assert done and "ruff" in done[-1]


@pytest.mark.toolchain
def test_lint_with_real_ruff_reports_the_finding(ruff, tmp_path):
    diags, note = runner.lint_code(UNUSED, "python",
                                   fs=LocalFileSystem(str(tmp_path)),
                                   ex=_VenvExec(ruff), stem="mod")
    assert note == ""
    hits = [d for d in diags if "F401" in d.message or "os" in d.message]
    assert hits, [d.one_line() for d in diags]
    assert hits[0].line == 1


@pytest.mark.toolchain
def test_format_with_real_ruff_when_black_is_absent(ruff, tmp_path):
    text, note = runner.format_code("x=1\n", "python",
                                    fs=LocalFileSystem(str(tmp_path)),
                                    ex=_VenvExec(ruff, hide=("black",)))
    assert text == "x = 1\n", (text, note)
