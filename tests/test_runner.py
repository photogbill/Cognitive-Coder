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
