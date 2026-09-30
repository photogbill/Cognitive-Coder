# SPDX-License-Identifier: Apache-2.0
"""M22 and M4 — phases that name themselves, and an honest definition of done.

The headline test here is a DISCRIMINATION test, and §6.4 states it as the
module's acceptance criterion: a deliberately broken C file must fail in
`build`, and a C file that compiles and then divides by zero must fail in
`run`. **If those two are indistinguishable, the module is wrong** no matter
what else it does — because a loop that conflates them hands a small model a
compiler error while asking it to fix a logic bug.

Toolchain-conditional throughout: a machine without a C compiler skips with a
printed note rather than failing (§9). Nobody's CI should go red because it
lacks Rust.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import langs, runner  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    LocalFileSystem,
    MemoryFileSystem,
    SubprocessExec,
)


@pytest.fixture
def workspace(tmp_path):
    return LocalFileSystem(str(tmp_path)), SubprocessExec()


def _need(ex, *binaries):
    if not any(ex.which(b) for b in binaries):
        pytest.skip(f"none of {binaries} is installed on this machine")


# --------------------------------------------------------------------------
# the discrimination test (§6.4's acceptance criterion)
# --------------------------------------------------------------------------

@pytest.mark.toolchain
def test_a_broken_c_file_fails_in_the_BUILD_phase(workspace):
    fs, ex = workspace
    _need(ex, "gcc", "clang", "cc")
    code = '#include <stdio.h>\nint main(void){ printf("%d\\n", x); return 0; }\n'
    result = runner.build_and_run(code, "c", fs=fs, ex=ex, stem="broken")
    assert result.failed_phase == "build", result.summary()
    assert not result.ok
    assert result.diagnostics, "a build failure with no diagnostics is useless"
    assert result.diagnostics[0].line > 0


@pytest.mark.toolchain
def test_a_c_file_that_crashes_at_runtime_fails_in_the_RUN_phase(workspace):
    fs, ex = workspace
    _need(ex, "gcc", "clang", "cc")
    code = ('#include <stdio.h>\nint main(void){ int a=1,b=0; '
            'printf("%d\\n", a/b); return 0; }\n')
    result = runner.build_and_run(code, "c", fs=fs, ex=ex, stem="crash")
    assert result.built, "it should have compiled cleanly"
    assert result.failed_phase == "run", result.summary()


@pytest.mark.toolchain
def test_a_working_c_file_succeeds(workspace):
    fs, ex = workspace
    _need(ex, "gcc", "clang", "cc")
    code = '#include <stdio.h>\nint main(void){ printf("hello\\n"); return 0; }\n'
    result = runner.build_and_run(code, "c", fs=fs, ex=ex, stem="good")
    assert result.ok, result.summary()
    assert "hello" in result.phases[-1].proc.stdout


# --------------------------------------------------------------------------
# guard and syntax are their own phases
# --------------------------------------------------------------------------

def test_a_guard_refusal_is_attributed_to_the_guard_phase(workspace):
    fs, ex = workspace
    result = runner.build_and_run("import os\nos.system('ls')\n", "python",
                                  fs=fs, ex=ex, stem="bad")
    assert result.failed_phase == "guard"
    assert "process spawning" in result.blocked


def test_a_python_syntax_error_is_caught_without_a_subprocess(workspace):
    """`ast.parse` is free and exact — and it is a PRE-check, never done."""
    fs, ex = workspace
    result = runner.build_and_run("def f(:\n    pass\n", "python", fs=fs,
                                  ex=ex, stem="syn")
    assert result.failed_phase == "syntax"
    assert not result.ok


def test_a_compiled_language_does_not_get_a_separate_syntax_phase(workspace):
    """The build IS the syntax check; running the compiler twice would both
    waste seconds and misattribute the failure (M22)."""
    fs, ex = workspace
    _need(ex, "gcc", "clang", "cc")
    result = runner.build_and_run(
        "int main(void){ return zzz; }\n", "c", fs=fs, ex=ex, stem="x")
    assert [p.name for p in result.phases] == ["build"]


# --------------------------------------------------------------------------
# C4: done means built AND tested (M4)
# --------------------------------------------------------------------------

def test_a_test_run_that_collected_zero_tests_is_not_reported_as_success(
        workspace):
    """The most dangerous green there is."""
    fs, ex = workspace
    result = runner.verify("def f():\n    return 1\n", "python", fs=fs, ex=ex,
                           stem="m", path="m.py")
    assert result.ok
    assert any("ZERO tests" in c for c in result.caveats), result.caveats


def test_zero_test_detection_recognises_each_runner():
    for output, label in [
            ("Ran 0 tests in 0.000s\n\nOK\n", "unittest"),
            ("no tests ran in 0.01s", "pytest"),
            ("collected 0 items", "pytest collection"),
            ("ok  \tmyapp\t[no test files]", "go"),
            ("running 0 tests", "rust")]:
        assert runner.zero_tests(output), label
    assert not runner.zero_tests("Ran 4 tests in 0.01s\n\nOK\n")


def test_a_language_with_no_test_runner_says_so_rather_than_passing(workspace):
    """C4 — the absence of tests is STATED, never counted as success."""
    fs, ex = workspace
    result = runner.run_tests("lua", fs=fs, ex=ex)
    assert not result.ok
    assert "no test runner is configured" in result.blocked
    assert "weaker evidence" in result.blocked


# --------------------------------------------------------------------------
# the environment (§6.4)
# --------------------------------------------------------------------------

def test_the_project_root_is_on_pythonpath():
    """Without it, no multi-file Python project can ever verify."""
    env = runner.scrubbed_env("/work", "/project")
    assert env["PYTHONPATH"] == "/project"


def test_the_environment_keeps_what_toolchains_need_to_start(monkeypatch):
    """Go on Windows fails without LOCALAPPDATA (its build cache lives
    there); a toolchain linked against a private libdir needs
    LD_LIBRARY_PATH. Neither is a credential."""
    for name in ("LOCALAPPDATA", "APPDATA", "ProgramData",
                 "LD_LIBRARY_PATH", "SystemDrive"):
        monkeypatch.setenv(name, f"/x/{name}")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "nope")
    env = runner.scrubbed_env("/work")
    for name in ("LOCALAPPDATA", "APPDATA", "ProgramData",
                 "LD_LIBRARY_PATH", "SystemDrive"):
        assert env.get(name) == f"/x/{name}", name
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_python_runs_on_the_engines_own_interpreter():
    """The registry's note says "the interpreter running this engine can
    always run Python" — but it probed PATH, and PATH `python` may be a
    different version (3.12+ turns a zero-test run into exit 5) or absent."""
    assert langs.get("python").which_run(SubprocessExec()) == sys.executable


class _PathOnlyPython(SubprocessExec):
    """Hides the engine's interpreter, so PATH's `python3.12` is used."""

    def which(self, binary):
        if binary == sys.executable:
            return None
        if binary in ("python", "python3", "py"):
            return super().which("python3.12")
        return super().which(binary)


@pytest.mark.toolchain
def test_unittests_exit_5_for_no_tests_is_the_zero_tests_caveat(tmp_path):
    """Python 3.12+ `unittest` exits 5 when it ran nothing. That is the
    zero-tests case — stated as a caveat — not a failed test phase."""
    ex = _PathOnlyPython()
    if not ex.which("python"):
        pytest.skip("python3.12 is not installed on this machine")
    fs = LocalFileSystem(str(tmp_path))
    fs.write("m.py", "def f():\n    return 1\n")
    result = runner.run_tests("python", fs=fs, ex=ex, stem="m", path="m.py")
    assert result.phases[0].proc.exit_code == 5, result.output
    assert result.ok, result.output
    assert any("ZERO tests" in c for c in result.caveats)


@pytest.mark.parametrize("code,lang", [
    ("import time\n# TODO: port the UI to tkinter later\nwhile True:\n"
     "    time.sleep(1)\n", "python"),
    ('"""A pygame-free rewrite."""\nwhile True:\n    pass\n', "python"),
    ("bwxyz = 1\nwhile True:\n    pass\n", "python"),
    ("const expression = 1;\nwhile (true) {}\n", "javascript"),
    ("// uses express later\nwhile (true) {}\n", "javascript"),
])
def test_a_main_loop_word_in_a_comment_or_name_is_not_a_main_loop(code,
                                                                  lang):
    """Observed: `# port to tkinter later` made a deadlocked program
    "ok — still running", and `wx` matched inside `bwxyz`."""
    assert not runner.has_main_loop(code, lang)


@pytest.mark.parametrize("code,lang", [
    ("import tkinter as tk\nroot = tk.Tk()\nroot.mainloop()\n", "python"),
    ("from PyQt5.QtWidgets import QApplication\n", "python"),
    ("import pygame\n", "python"),
    ("const express = require('express');\n", "javascript"),
    ("import express from 'express';\n", "javascript"),
    ('import "github.com/hajimehoshi/ebiten/v2"\n', "go"),
    ("use bevy::prelude::*;\n", "rust"),
])
def test_a_real_main_loop_import_is_recognised(code, lang):
    assert runner.has_main_loop(code, lang)


def test_a_deadlock_with_a_gui_word_in_a_comment_is_a_failure(workspace):
    fs, ex = workspace
    code = ("import time\n# TODO: port the UI to tkinter later\n"
            "while True:\n    time.sleep(1)\n")
    result = runner.build_and_run(code, "python", fs=fs, ex=ex, stem="hang",
                                  timeout=1.0)
    assert not result.ok and result.failed_phase == "run"


def test_the_environment_is_scrubbed_but_still_usable():
    env = runner.scrubbed_env("/work")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["NO_COLOR"] == "1"
    assert env["TEMP"] == "/work"
    assert "PATH" in env


# --------------------------------------------------------------------------
# GDScript's honesty requirements (§6.1a, M40)
# --------------------------------------------------------------------------

def test_a_headless_godot_test_touching_the_scene_tree_earns_its_caveat():
    """M40 — never unqualified success when the evidence is weaker."""
    test_source = ('extends GutTest\n\nfunc test_move():\n'
                   '\tvar n = get_tree().get_root()\n\tassert_true(true)\n')
    caveat = langs.headless_caveat_for(test_source)
    assert caveat
    assert "verify in the editor" in caveat.lower()


def test_a_pure_gdscript_test_earns_no_caveat():
    """A caveat on everything is a caveat nobody reads."""
    assert langs.headless_caveat_for(
        'extends GutTest\n\nfunc test_add():\n\tassert_eq(2 + 2, 4)\n') == ""


def test_godot_test_runner_detection_names_what_it_found():
    fs = MemoryFileSystem({"addons/gut/gut_cmdln.gd": b"",
                           "test/test_x.gd": b""})
    argv, note = langs.godot_test_cmd(fs, "godot")
    assert note == "GUT"
    assert "gut_cmdln.gd" in " ".join(argv)
    assert "--fixed-fps" in argv, "frame deltas must be deterministic"

    fs2 = MemoryFileSystem({"addons/gdUnit4/bin/GdUnitCmdTool.gd": b""})
    _argv2, note2 = langs.godot_test_cmd(fs2, "godot")
    assert note2 == "gdUnit4"

    argv3, note3 = langs.godot_test_cmd(MemoryFileSystem({}), "godot")
    assert argv3 == []
    assert "neither GUT nor gdUnit4" in note3


def test_res_paths_are_translated_at_the_boundary():
    """The one real trap in GDScript support (§6.1a)."""
    assert langs.to_os_path("res://scripts/player.gd") == "scripts/player.gd"
    assert langs.to_os_path("scripts/player.gd") == "scripts/player.gd"
    assert langs.to_res_path("scripts/player.gd") == "res://scripts/player.gd"


# --------------------------------------------------------------------------
# degradation, never crashing (C7, M6)
# --------------------------------------------------------------------------

def test_a_missing_toolchain_produces_a_sentence_not_an_exception(workspace):
    fs, ex = workspace
    lang = langs.get("rust")
    if ex.which("rustc"):
        pytest.skip("rustc is installed, so there is nothing to degrade")
    result = runner.build_and_run("fn main() {}\n", "rust", fs=fs, ex=ex)
    assert not result.ok
    assert "rustc" in result.blocked
    assert lang.install_hint.split(" ")[0] in result.blocked


def test_a_missing_linter_reports_what_its_absence_costs(workspace):
    fs, ex = workspace
    diags, note = runner.lint_code("x = 1\n", "python", fs=fs, ex=ex)
    if note:
        assert "no linter installed" in note
        assert "will only surface" in note


def test_an_unreachable_workspace_is_attributed_to_the_workspace():
    """Not to the code — otherwise the model is sent to fix something fine."""
    fs = MemoryFileSystem()             # root is /project, which is not real
    result = runner.build_and_run("print('hi')\n", "python", fs=fs,
                                  ex=SubprocessExec(), stem="m")
    assert not result.ok
    assert "not on a real disk" in result.blocked
    assert "Editing, outlining and the codemap all work" in result.blocked


# --------------------------------------------------------------------------
# tests that actually RUN (C4) — real toolchains, real output
# --------------------------------------------------------------------------
#
# Observed: Rust's test command was `rustc --test -o {out} {src}`, which
# BUILDS the harness and exits 0 — a `#[test]` asserting false verified ok.
# JavaScript's was `node --test {dir}`, and Node >= 21 reads the directory
# as a file name ("Could not find"), so JavaScript could never verify.

_RUST_FAILING = ('fn main() { println!("hello"); }\n'
                 '#[cfg(test)]\nmod t {\n    #[test]\n'
                 '    fn boom() { assert!(false, "must fail"); }\n}\n')


@pytest.mark.toolchain
def test_a_failing_rust_test_makes_verify_fail(workspace):
    fs, ex = workspace
    _need(ex, "rustc")
    result = runner.verify(_RUST_FAILING, "rust", fs=fs, ex=ex, stem="main",
                           path="main.rs")
    assert not result.ok, result.summary()
    assert result.failed_phase == "test", result.summary()
    assert "boom" in result.output and "FAILED" in result.output


@pytest.mark.toolchain
def test_the_rust_scaffold_verifies_and_its_test_really_ran(workspace):
    fs, ex = workspace
    _need(ex, "rustc")
    code = langs.scaffold_for("rust", "t")
    result = runner.verify(code, "rust", fs=fs, ex=ex, stem="main",
                           path="main.rs")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert "running 1 test" in result.phase("test").output
    assert not any("ZERO" in c for c in result.caveats), result.caveats


@pytest.mark.toolchain
def test_a_rust_file_with_no_tests_is_not_counted_as_tested(workspace):
    fs, ex = workspace
    _need(ex, "rustc")
    result = runner.verify('fn main() { println!("hi"); }\n', "rust", fs=fs,
                           ex=ex, stem="main", path="main.rs")
    assert result.ok
    assert any("ZERO tests" in c for c in result.caveats), result.caveats


@pytest.mark.toolchain
def test_build_artefacts_do_not_land_at_the_project_root(workspace):
    """`<root>/main.bin` beside the operator's sources is litter at best
    and a clobbered file at worst; artefacts live in `.cc_state/build/`."""
    fs, ex = workspace
    _need(ex, "rustc")
    runner.verify(langs.scaffold_for("rust", "t"), "rust", fs=fs, ex=ex,
                  stem="main", path="main.rs")
    assert [p for p in fs.list("*") if not p.startswith(".cc_state/")] \
        == ["main.rs"], fs.list("*")


def _js_project(fs, test_rel="main.test.js", test_body=None):
    fs.write("main.js", langs.scaffold_for("javascript", "t"))
    body = test_body if test_body is not None else \
        langs.test_scaffold_for("javascript", "t")
    if test_rel.count("/"):
        body = body.replace('"./main.js"', '"../main.js"')
    fs.write(test_rel, body)
    return fs.read("main.js")


@pytest.mark.toolchain
def test_the_engines_own_js_scaffold_and_test_verify_on_node(workspace):
    fs, ex = workspace
    _need(ex, "node")
    code = _js_project(fs)
    result = runner.verify(code, "javascript", fs=fs, ex=ex, stem="main",
                           path="main.js")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert result.phase("test") is not None
    assert "# pass 1" in result.phase("test").output
    assert not any("ZERO" in c or "nothing was tested" in c
                   for c in result.caveats), result.caveats


@pytest.mark.toolchain
def test_a_failing_js_test_makes_verify_fail(workspace):
    fs, ex = workspace
    _need(ex, "node")
    code = _js_project(fs, "tests/test_main.js", langs.test_scaffold_for(
        "javascript", "t").replace("main(), 0", "main(), 1"))
    result = runner.verify(code, "javascript", fs=fs, ex=ex, stem="main",
                           path="main.js")
    assert not result.ok and result.failed_phase == "test", result.summary()


@pytest.mark.toolchain
def test_a_js_test_file_with_no_tests_earns_the_zero_tests_caveat(
        workspace):
    """Node reports a test file with no `test()` calls as ONE passing test
    named after the file. That is the zero-test green in disguise."""
    fs, ex = workspace
    _need(ex, "node")
    code = _js_project(fs, test_body='import test from "node:test";\n')
    result = runner.verify(code, "javascript", fs=fs, ex=ex, stem="main",
                           path="main.js")
    assert any("ZERO tests" in c for c in result.caveats), result.caveats


@pytest.mark.toolchain
def test_a_js_project_with_no_test_files_says_nothing_was_tested(
        workspace):
    fs, ex = workspace
    _need(ex, "node")
    fs.write("main.js", langs.scaffold_for("javascript", "t"))
    result = runner.verify(fs.read("main.js"), "javascript", fs=fs, ex=ex,
                           stem="main", path="main.js")
    assert result.ok
    assert any("nothing was tested" in c and "test file" in c
               for c in result.caveats), result.caveats


# --------------------------------------------------------------------------
# the test phase belongs to the TASK (C4, M39)
# --------------------------------------------------------------------------
#
# Observed: `unittest discover` ran the whole project, so one failing test
# anywhere was attributed to every later file — its failure fed back as
# THAT file's diagnostic — and blocked the rest of the build.

_GREET = 'def greet(name):\n    return f"hello {name}"\n'
_GREET_TEST = ('import unittest\n\nfrom src.greet import greet\n\n\n'
               'class T(unittest.TestCase):\n    def test_greet(self):\n'
               '        self.assertEqual(greet("a"), "hello a")\n')
_CALC_TEST_FAILING = ('import unittest\n\n\nclass C(unittest.TestCase):\n'
                      '    def test_add(self):\n'
                      '        self.assertEqual(2 + 2, 5)\n')


def _two_task_project(fs):
    # `tests/__init__.py`, so the WHOLE-suite command finds tests/ at all:
    # `unittest discover -s <root>` does not descend into a folder that is
    # not a package. The scoped run needs no such help.
    fs.write("tests/__init__.py", "")
    fs.write("src/greet.py", _GREET)
    fs.write("tests/test_greet.py", _GREET_TEST)
    fs.write("src/calc.py", "def add(a, b):\n    return a + b\n")
    fs.write("tests/test_calc.py", _CALC_TEST_FAILING)


def test_another_tasks_failing_test_is_a_caveat_not_a_diagnostic(
        workspace):
    fs, ex = workspace
    _two_task_project(fs)
    result = runner.verify(_GREET, "python", fs=fs, ex=ex, stem="greet",
                           path="src/greet.py",
                           test_path="tests/test_greet.py")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert not result.diagnostics, result.diagnostics
    other = [c for c in result.caveats if "test_calc.py" in c]
    assert other, result.caveats
    assert "tests/test_greet.py" in other[0]


def test_the_tasks_own_failing_test_still_fails_it(workspace):
    fs, ex = workspace
    _two_task_project(fs)
    fs.write("tests/test_greet.py", _GREET_TEST.replace('"hello a"', '"x"'))
    result = runner.verify(_GREET, "python", fs=fs, ex=ex, stem="greet",
                           path="src/greet.py",
                           test_path="tests/test_greet.py")
    assert not result.ok and result.failed_phase == "test"
    assert "test_greet" in result.output
    # Scoped: the other task's failure is not in THIS run's output.
    assert "test_add" not in result.phase("test").output


def test_a_test_path_that_does_not_exist_yet_falls_back_to_the_suite(
        workspace):
    fs, ex = workspace
    fs.write("m.py", "def f():\n    return 1\n")
    result = runner.verify("def f():\n    return 1\n", "python", fs=fs,
                           ex=ex, stem="m", path="m.py",
                           test_path="tests/test_m.py")
    assert result.ok
    assert any("ZERO tests" in c for c in result.caveats), result.caveats


@pytest.mark.toolchain
def test_a_js_task_runs_only_its_own_test_file(workspace):
    fs, ex = workspace
    _need(ex, "node")
    code = _js_project(fs, "tests/test_main.js")
    fs.write("tests/test_other.js",
             'import test from "node:test";\nimport assert from '
             '"node:assert";\ntest("other", () => { assert.ok(false); });\n')
    result = runner.verify(code, "javascript", fs=fs, ex=ex, stem="main",
                           path="main.js", test_path="tests/test_main.js")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert any("test_other.js" in c for c in result.caveats), result.caveats


def test_the_loop_passes_the_tasks_test_path_to_verify(tmp_path):
    """The one edit in loop.py: `_verify` hands `task.test_path` down."""
    from cognitive_coder.loop import Loop, LoopConfig
    from cognitive_coder.ports import AutoApprove, Host, MemoryStorage
    from cognitive_coder.types import Task
    host = Host(fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                approval=AutoApprove())
    _two_task_project(host.fs)
    loop = Loop(host, config=LoopConfig())
    task = Task(id="greet", path="src/greet.py", purpose="greet",
                test_path="tests/test_greet.py", lang="python")
    result = loop._verify(task, "python")
    assert result.ok, f"{result.summary()}\n{result.output}"
    assert any("test_calc.py" in c for c in result.caveats), result.caveats


# --------------------------------------------------------------------------
# side channels: format, lint and autofix must never write a project path
# --------------------------------------------------------------------------
#
# Observed: with the library default `DenyAll`, a task for `main.py` had the
# operator's `main.py` overwritten by `autofix` with unapproved model output,
# BEFORE the patcher asked anyone — and `src/util.py` clobbered a root
# `util.py`. The approval gate is the product; these tests are its side
# doors.

class _RecordingDeny:
    """An ApprovalPort that refuses and remembers being asked."""

    def __init__(self):
        self.asked = []

    def approve_diff(self, summary, diff):
        self.asked.append(summary)
        return False

    def approve_remote(self, provider, bytes_out, estimate):
        return False


class _FakeFixer:
    """An ExecPort whose linters/formatters rewrite the file they are handed.

    A fake rather than a mock (§9): it really edits the file on disk, the way
    `ruff --fix` would, so the test sees where the runner put the candidate.
    """

    def __init__(self, tools=("ruff", "black", "clang-format")):
        self.tools = tools
        self.calls = []

    def which(self, binary):
        return f"/fake/bin/{binary}" if binary in self.tools else None

    def run(self, argv, *, cwd, timeout, stdin="", env=None):
        from cognitive_coder.types import ProcResult
        self.calls.append((list(argv), cwd))
        src = argv[-1]
        with open(src, encoding="utf-8") as fh:
            text = fh.read()
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(text.replace("import os\n", ""))
        return ProcResult(exit_code=1,
                          stdout=f"{src}:1:8: F401 `os` imported but unused\n")


def _snapshot(fs):
    return {p: fs.read_bytes(p) for p in fs.list("*")}


def test_autofix_never_writes_a_project_path(tmp_path):
    from cognitive_coder import patcher
    from cognitive_coder.ports import MemoryStorage
    from cognitive_coder.types import Edit
    fs, ex = LocalFileSystem(str(tmp_path)), _FakeFixer()
    fs.write("main.py", "x = 1\n")
    fs.write("util.py", "ROOT = True\n")
    fs.write("src/util.py", "SRC = True\n")
    before = _snapshot(fs)
    approval = _RecordingDeny()

    fixed, done = runner.autofix("import os\nx = 2\n", "python", fs=fs,
                                 ex=ex, stem="main")
    runner.autofix("import os\nSRC = 2\n", "python", fs=fs, ex=ex,
                   stem="util")

    assert fixed == "x = 2\n" and done, "the fixer's work was lost"
    assert _snapshot(fs) == before, "autofix wrote into the project"
    assert all(".cc_state/scratch/" in argv[-1].replace("\\", "/")
               for argv, _cwd in ex.calls), ex.calls
    # And the ONLY way the candidate reaches main.py is the patcher's gate.
    p = patcher.Patcher(fs, MemoryStorage(), approval)
    tx = p.begin("t")
    result = tx.apply([Edit(path="main.py", kind="whole", new=fixed)])[0]
    assert not result.ok and approval.asked == ["edit main.py (task t)"]
    assert fs.read("main.py") == "x = 1\n"


def test_format_code_never_writes_a_project_path(tmp_path):
    fs, ex = LocalFileSystem(str(tmp_path)), _FakeFixer()
    fs.write("main.py", "x = 1\n")
    before = _snapshot(fs)
    text, _note = runner.format_code("import os\ny = 3\n", "python", fs=fs,
                                     ex=ex, stem="main")
    assert text == "y = 3\n"
    assert _snapshot(fs) == before, "format_code wrote into the project"


def test_lint_code_never_writes_a_project_path_or_names_the_scratch(
        tmp_path):
    fs, ex = LocalFileSystem(str(tmp_path)), _FakeFixer()
    fs.write("main.py", "x = 1\n")
    before = _snapshot(fs)
    diags, _note = runner.lint_code("import os\ny = 3\n", "python", fs=fs,
                                    ex=ex, stem="main")
    assert _snapshot(fs) == before, "lint_code wrote into the project"
    assert diags
    # The model is shown the file it is working on, never the engine's own
    # scratch directory — a path it would then try to import from.
    text = " ".join(d.file + " " + d.message for d in diags)
    assert ".cc_state" not in text and str(tmp_path) not in text, text
    assert "main.py" in text


@pytest.mark.toolchain
def test_a_real_ruff_autofix_leaves_the_project_untouched(tmp_path):
    ex = SubprocessExec()
    _need(ex, "ruff")
    fs = LocalFileSystem(str(tmp_path))
    fs.write("main.py", "x = 1\n")
    before = _snapshot(fs)
    fixed, done = runner.autofix("import os\nx = 2\n", "python", fs=fs,
                                 ex=ex, stem="main")
    assert "import os" not in fixed and done
    # Including ruff's own `.ruff_cache/`, which is a write to the operator's
    # tree that nobody approved either.
    assert _snapshot(fs) == before


@pytest.mark.toolchain
@pytest.mark.parametrize("lang_id", sorted(langs.ids()))
def test_every_available_language_scaffold_builds_and_runs(lang_id, tmp_path):
    """§6.1's acceptance: every scaffold actually works, where the
    toolchain exists. Skipped with a note where it does not."""
    ex = SubprocessExec()
    lang = langs.get(lang_id)
    if not lang.available(ex):
        pytest.skip(f"{lang.label}: no toolchain on this machine")
    if lang_id in ("csharp", "gdscript", "sql", "batch", "powershell"):
        pytest.skip(f"{lang.label}: needs a project or a host-specific setup")
    fs = LocalFileSystem(str(tmp_path / lang_id))
    stem = lang.entry if lang_id == "java" else "main"
    code = langs.scaffold_for(lang_id, "scaffold check", stem)
    if not code:
        pytest.skip(f"{lang.label}: no scaffold defined")
    result = runner.build_and_run(code, lang_id, fs=fs, ex=ex, stem=stem)
    assert result.ok, f"{lang.label}: {result.summary()}\n{result.output}"
    assert "hello from" in result.phases[-1].proc.stdout.lower()
