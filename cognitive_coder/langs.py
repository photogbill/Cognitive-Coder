# SPDX-License-Identifier: Apache-2.0
"""Language registry — everything language-specific is a lookup in this file.

Ported from ATK's `atk/core/langs.py`, which was written against the same
principles. The three rules it was built on survive verbatim, because each of
them was learned the hard way:

  1. **Build and run are separate phases.** A compiler error and a runtime
     crash are different problems with different fixes, and a small model
     handed "it didn't work" fixes neither. Keeping the phases apart is what
     lets `runner.py` say WHICH phase failed (M22).

  2. **Every language ships a runnable scaffold WITH a test hook.** A small
     model asked to invent project structure invents it wrong; asked to fill a
     body into a file that already compiles, it does well. This is one of the
     highest-value decisions in the whole design.

  3. **Commands are argument LISTS with placeholders, never strings.** A path
     with a space in it (`C:\\Users\\Bill Smith\\…`) breaks string commands in
     a way that looks like a compiler bug. Lists never quote-fail.

What this port adds over ATK's version:

  * **Toolchain probing goes through `ExecPort.which`** (C2) rather than
    `shutil.which`. The installer's detection record is informational only —
    a compiler installed the week after install day must simply work (§6.1).
  * **GDScript is first class** (§6.1a), not an outline-only afterthought:
    syntax check, run, GUT/gdUnit4 test detection, and the headless caveat.
  * **`cascades`** marks the languages where one missing brace produces forty
    errors. `diagnostics.py` feeds back only the first error for those (F7),
    with more source context instead — a model handed error #23 of 40 will
    earnestly try to fix error #23.
  * **`syntax_cmd`** — a cheap pre-check that is not a completion signal.
    C4/M4 are explicit that parsing is a PRE-check and never means done.

Placeholders in command templates:
  ``{src}``  the source file        ``{out}``   the built artefact
  ``{dir}``  the working directory  ``{stem}``  the source filename, no suffix
  ``{build}`` ``{run}`` ``{fmt}`` ``{lint}``  the resolved tool path
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Lang:
    """One language, and everything the engine needs to know about it."""

    id: str
    label: str
    ext: str                       # primary source extension, with the dot
    exts: tuple[str, ...] = ()     # every extension that belongs to it
    comment: str = "//"
    block_comment: tuple = ()      # (open, close) when the language has one
    compiled: bool = False
    # binaries: the FIRST one found wins, so the list is preference order
    build_tools: tuple[str, ...] = ()
    run_tools: tuple[str, ...] = ()
    build_cmd: list = field(default_factory=list)
    run_cmd: list = field(default_factory=list)
    test_cmd: list = field(default_factory=list)
    syntax_cmd: list = field(default_factory=list)   # cheap pre-check
    fmt_tools: tuple[str, ...] = ()
    fmt_cmd: list = field(default_factory=list)
    lint_tools: tuple[str, ...] = ()
    lint_cmd: list = field(default_factory=list)
    fix_cmd: list = field(default_factory=list)      # `--fix`-able rules (F1)
    entry: str = "main"            # conventional entry filename stem
    scaffold: str = ""             # a file that compiles and runs as-is
    test_scaffold: str = ""        # a test file the loop can run
    stub_style: str = "raise"      # raise | todo | empty — how stubs are made
    cascades: bool = False         # one error produces many (F7)
    build_timeout: float = 60.0
    run_timeout: float = 15.0
    test_timeout: float = 120.0
    notes: str = ""                # what an operator should know
    install_hint: str = ""

    @property
    def needs_build(self) -> bool:
        return bool(self.build_cmd)

    @property
    def feedback_cap(self) -> int:
        """How many diagnostics to feed back (F7).

        One for cascading languages, three otherwise. The cap is not a
        stylistic preference: the second through fortieth errors from a
        missing semicolon in C++ are noise, and a model's attention is the
        scarce resource in this whole engine.
        """
        return 1 if self.cascades else 3

    # -- toolchain probing, through the Port (C2) -------------------------
    def which_build(self, ex: Any) -> str:
        for t in self.build_tools:
            found = ex.which(t)
            if found:
                return found
        return ""

    def which_run(self, ex: Any) -> str:
        if not self.run_tools:            # compiled: the artefact IS the exe
            return "-"
        for t in self.run_tools:
            found = ex.which(t)
            if found:
                return found
        return ""

    def which_tool(self, ex: Any, tools: tuple[str, ...]) -> str:
        for t in tools:
            found = ex.which(t)
            if found:
                return found
        return ""

    def available(self, ex: Any) -> bool:
        """Is at least one usable toolchain present RIGHT NOW?"""
        return bool(self.which_run(ex)
                    and (not self.build_tools or self.which_build(ex)))

    def missing_note(self, ex: Any) -> str:
        """A sentence naming what is missing and what it costs (C6, C7)."""
        if self.build_tools and not self.which_build(ex):
            return (f"{self.label} needs one of "
                    f"{', '.join(self.build_tools)} and none is installed. "
                    f"{self.install_hint}").strip()
        if self.run_tools and not self.which_run(ex):
            return (f"{self.label} needs one of {', '.join(self.run_tools)} "
                    f"on PATH. {self.install_hint}").strip()
        return ""


LANGS: dict[str, Lang] = {}


def _add(lang: Lang) -> Lang:
    LANGS[lang.id] = lang
    return lang


# ---------------------------------------------------------------------------
# The registry — ordered by how likely each is to be wanted, not
# alphabetically. The first entries of a host's dropdown should be the common
# cases.
# ---------------------------------------------------------------------------

_add(Lang(
    id="python", label="Python", ext=".py", exts=(".py", ".pyw"),
    comment="#", run_tools=("python", "python3", "py"),
    run_cmd=["{run}", "{src}"],
    test_cmd=["{run}", "-m", "unittest", "discover", "-s", "{dir}", "-p",
              "test_*.py", "-v"],
    fmt_tools=("black", "ruff"), fmt_cmd=["{fmt}", "-q", "{src}"],
    lint_tools=("ruff", "pyflakes", "flake8"), lint_cmd=["{lint}", "{src}"],
    fix_cmd=["{lint}", "check", "--fix", "--quiet", "{src}"],
    stub_style="raise",
    notes="The safe default: the interpreter running this engine can always "
          "run Python, so no toolchain detection can fail.",
    scaffold='''"""{title}"""


def main() -> int:
    print("hello from {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    test_scaffold='''import unittest

from main import main


class TestMain(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
'''))

_add(Lang(
    id="c", label="C", ext=".c", exts=(".c", ".h"), compiled=True,
    block_comment=("/*", "*/"), cascades=True,
    build_tools=("gcc", "clang", "cc", "cl"),
    build_cmd=["{build}", "-std=c17", "-Wall", "-Wextra", "-g",
               "{src}", "-o", "{out}"],
    syntax_cmd=["{build}", "-fsyntax-only", "{src}"],
    run_cmd=["{out}"],
    fmt_tools=("clang-format",), fmt_cmd=["{fmt}", "-i", "{src}"],
    lint_tools=("cppcheck",),
    lint_cmd=["{lint}", "--enable=warning,style", "--quiet", "{src}"],
    notes="-Wall -Wextra is on deliberately: for a small model the warnings "
          "are the most useful feedback the toolchain produces.",
    install_hint="MinGW-w64 (w64devkit is a single portable folder) or LLVM.",
    scaffold='''/* {title} */
#include <stdio.h>

int main(void) {{
    printf("hello from {title}\\n");
    return 0;
}}
''',
    test_scaffold='''/* Minimal assert-based tests — no framework to install. */
#include <assert.h>
#include <stdio.h>

int add(int a, int b);

int main(void) {{
    assert(add(2, 2) == 4);
    printf("all tests passed\\n");
    return 0;
}}
'''))

_add(Lang(
    id="cpp", label="C++", ext=".cpp",
    exts=(".cpp", ".cc", ".cxx", ".hpp", ".hh"), compiled=True,
    block_comment=("/*", "*/"), cascades=True,
    build_tools=("g++", "clang++", "c++", "cl"),
    build_cmd=["{build}", "-std=c++20", "-Wall", "-Wextra", "-g",
               "{src}", "-o", "{out}"],
    syntax_cmd=["{build}", "-fsyntax-only", "{src}"],
    run_cmd=["{out}"],
    fmt_tools=("clang-format",), fmt_cmd=["{fmt}", "-i", "{src}"],
    lint_tools=("cppcheck",),
    lint_cmd=["{lint}", "--enable=warning,style", "--quiet", "{src}"],
    notes="Template errors cascade worse than any other language here, which "
          "is why the feedback cap is one.",
    install_hint="MinGW-w64 (w64devkit), LLVM, or MSVC Build Tools.",
    scaffold='''// {title}
#include <iostream>

int main() {{
    std::cout << "hello from {title}" << std::endl;
    return 0;
}}
''',
    test_scaffold='''#include <cassert>
#include <iostream>

int add(int a, int b);

int main() {{
    assert(add(2, 2) == 4);
    std::cout << "all tests passed" << std::endl;
    return 0;
}}
'''))

_add(Lang(
    id="rust", label="Rust", ext=".rs", exts=(".rs",), compiled=True,
    cascades=True, build_tools=("rustc",),
    build_cmd=["{build}", "--edition", "2021", "-o", "{out}", "{src}"],
    run_cmd=["{out}"],
    test_cmd=["{build}", "--edition", "2021", "--test", "-o", "{out}",
              "{src}"],
    fmt_tools=("rustfmt",), fmt_cmd=["{fmt}", "{src}"],
    lint_tools=("clippy-driver",), lint_cmd=["{lint}", "{src}"],
    notes="Single-file rustc, not cargo — cargo wants a network for the "
          "registry and this machine may not have one. Crates outside std "
          "are therefore unavailable, and the prompt says so.",
    install_hint="rustup (the toolchain itself is offline once installed).",
    scaffold='''// {title}
fn main() {{
    println!("hello from {title}");
}}

#[cfg(test)]
mod tests {{
    #[test]
    fn it_works() {{
        assert_eq!(2 + 2, 4);
    }}
}}
'''))

_add(Lang(
    id="java", label="Java", ext=".java", exts=(".java",), compiled=True,
    cascades=True, build_tools=("javac",), run_tools=("java",),
    build_cmd=["{build}", "-d", "{dir}", "{src}"],
    run_cmd=["{run}", "-cp", "{dir}", "{stem}"],
    entry="Main",
    notes="The public class name must match the filename — Java is the one "
          "language where a rename breaks the build, so the scaffold and the "
          "runner both derive the class from the file's stem.",
    install_hint="A JDK (Temurin/Adoptium is the usual portable choice).",
    scaffold='''// {title}
public class {stem} {{
    public static void main(String[] args) {{
        System.out.println("hello from {title}");
    }}
}}
'''))

_add(Lang(
    id="go", label="Go", ext=".go", exts=(".go",), compiled=True,
    build_tools=("go",),
    build_cmd=["{build}", "build", "-o", "{out}", "{src}"],
    syntax_cmd=["{build}", "vet", "{src}"],
    run_cmd=["{out}"],
    test_cmd=["{build}", "test", "./..."],
    fmt_tools=("gofmt",), fmt_cmd=["{fmt}", "-w", "{src}"],
    fix_cmd=["{fmt}", "-w", "{src}"],
    lint_tools=("go",), lint_cmd=["{lint}", "vet", "{src}"],
    install_hint="The official Go distribution; it needs no network offline.",
    scaffold='''// {title}
package main

import "fmt"

func main() {{
    fmt.Println("hello from {title}")
}}
'''))

_add(Lang(
    id="csharp", label="C#", ext=".cs", exts=(".cs",), compiled=True,
    build_tools=("dotnet", "csc"), run_tools=("dotnet",),
    build_cmd=["{build}", "build"],
    run_cmd=["{run}", "run", "--project", "{dir}"],
    test_cmd=["{build}", "test"],
    notes="Needs a project file, so C# works in PROJECT mode rather than the "
          "single-file scratchpad. `dotnet new console` first.",
    install_hint="The .NET SDK.",
    scaffold='''// {title}
Console.WriteLine("hello from {title}");
'''))

_add(Lang(
    id="javascript", label="JavaScript (Node)", ext=".js",
    exts=(".js", ".mjs", ".cjs"),
    run_tools=("node",), run_cmd=["{run}", "{src}"],
    syntax_cmd=["{run}", "--check", "{src}"],
    test_cmd=["{run}", "--test", "{dir}"],
    fmt_tools=("prettier",), fmt_cmd=["{fmt}", "--write", "{src}"],
    lint_tools=("eslint",), lint_cmd=["{lint}", "{src}"],
    fix_cmd=["{lint}", "--fix", "{src}"],
    notes="Node's built-in test runner (--test) is used, so no npm install "
          "and no network.",
    install_hint="Node.js LTS.",
    scaffold='''// {title}
function main() {{
    console.log("hello from {title}");
    return 0;
}}

main();
export {{ main }};
''',
    test_scaffold='''import test from "node:test";
import assert from "node:assert";
import {{ main }} from "./main.js";

test("main runs", () => {{
    assert.strictEqual(main(), 0);
}});
'''))

_add(Lang(
    id="typescript", label="TypeScript", ext=".ts", exts=(".ts", ".tsx"),
    compiled=True, build_tools=("tsc",), run_tools=("node",),
    build_cmd=["{build}", "--outDir", "{dir}", "--target", "es2022",
               "--module", "es2022", "{src}"],
    syntax_cmd=["{build}", "--noEmit", "{src}"],
    run_cmd=["{run}", "{dir}/{stem}.js"],
    fmt_tools=("prettier",), fmt_cmd=["{fmt}", "--write", "{src}"],
    lint_tools=("eslint",), lint_cmd=["{lint}", "{src}"],
    notes="Type errors are exactly the feedback a small model needs, which "
          "makes TypeScript one of the better languages to pair with one.",
    install_hint="npm install -g typescript (needs a network once).",
    scaffold='''// {title}
export function main(): number {{
    console.log("hello from {title}");
    return 0;
}}

main();
'''))

_add(Lang(
    id="gdscript", label="GDScript (Godot 4)", ext=".gd", exts=(".gd",),
    comment="#", run_tools=("godot", "godot4", "Godot_v4"),
    # --check-only is the GDScript equivalent of ast.parse: a real pre-check,
    # and a cheap one. It is NOT a completion signal (C4/M4).
    syntax_cmd=["{run}", "--headless", "--check-only", "--script", "{src}"],
    run_cmd=["{run}", "--headless", "--script", "{src}"],
    # The default test command is GUT; `godot_test_cmd()` below picks
    # gdUnit4 instead when the project's addons/ says so.
    test_cmd=["{run}", "--headless", "--fixed-fps", "60", "-s",
              "addons/gut/gut_cmdln.gd", "-gdir=res://test", "-gexit"],
    stub_style="todo",
    build_timeout=30.0, run_timeout=60.0, test_timeout=120.0,
    notes="Godot uses res:// paths; the patcher works in OS paths and "
          "translates at the boundary — res:// must never reach the "
          "FileSystemPort. Without a `godot` binary this degrades to "
          "outline-and-edit only: no syntax check, no run, no tests.",
    install_hint="The Godot 4 editor binary on PATH (a single executable).",
    scaffold='''# {title}
extends Node


func _ready() -> void:
	print("hello from {title}")


func add(a: int, b: int) -> int:
	return a + b
''',
    test_scaffold='''extends GutTest


func test_add() -> void:
	var script := load("res://main.gd").new()
	assert_eq(script.add(2, 2), 4, "add should sum its arguments")
'''))

_add(Lang(
    id="bash", label="Bash / sh", ext=".sh", exts=(".sh", ".bash"),
    comment="#", run_tools=("bash", "sh"), run_cmd=["{run}", "{src}"],
    syntax_cmd=["{run}", "-n", "{src}"],
    lint_tools=("shellcheck",), lint_cmd=["{lint}", "{src}"],
    notes="On Windows this needs Git Bash or WSL. shellcheck, if present, is "
          "unusually good feedback — it catches the quoting mistakes small "
          "models make constantly.",
    scaffold='''#!/usr/bin/env bash
# {title}
set -euo pipefail

main() {{
    echo "hello from {title}"
}}

main "$@"
'''))

_add(Lang(
    id="powershell", label="PowerShell", ext=".ps1", exts=(".ps1", ".psm1"),
    comment="#", run_tools=("pwsh", "powershell"),
    run_cmd=["{run}", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             "{src}"],
    notes="Always present on Windows. Useful for the automation a Windows "
          "host itself needs.",
    scaffold='''# {title}
function Main {{
    Write-Host "hello from {title}"
}}

Main
'''))

_add(Lang(
    id="lua", label="Lua", ext=".lua", exts=(".lua",),
    comment="--", run_tools=("lua", "luajit"), run_cmd=["{run}", "{src}"],
    syntax_cmd=["{run}", "-e", "loadfile('{src}')"],
    install_hint="A single small binary — the easiest toolchain here.",
    scaffold='''-- {title}
local function main()
    print("hello from {title}")
    return 0
end

return main()
'''))

_add(Lang(
    id="ruby", label="Ruby", ext=".rb", exts=(".rb",),
    comment="#", run_tools=("ruby",), run_cmd=["{run}", "{src}"],
    syntax_cmd=["{run}", "-c", "{src}"],
    test_cmd=["{run}", "-Itest", "{src}"],
    lint_tools=("rubocop",), lint_cmd=["{lint}", "{src}"],
    fix_cmd=["{lint}", "-a", "{src}"],
    scaffold='''# {title}
def main
  puts "hello from {title}"
  0
end

main if __FILE__ == $PROGRAM_NAME
'''))

_add(Lang(
    id="sql", label="SQL (SQLite)", ext=".sql", exts=(".sql",),
    comment="--", run_tools=("sqlite3",),
    run_cmd=["{run}", "-batch", "{dir}/scratch.db"],
    notes="Runs against a scratch database inside the workspace, never "
          "against the host's own state. Statements are piped in on stdin.",
    scaffold='''-- {title}
CREATE TABLE IF NOT EXISTS demo (id INTEGER PRIMARY KEY, name TEXT);
INSERT INTO demo (name) VALUES ('hello from {title}');
SELECT * FROM demo;
'''))

_add(Lang(
    id="zig", label="Zig", ext=".zig", exts=(".zig",), compiled=True,
    build_tools=("zig",),
    build_cmd=["{build}", "build-exe", "-femit-bin={out}", "{src}"],
    syntax_cmd=["{build}", "ast-check", "{src}"],
    run_cmd=["{out}"],
    test_cmd=["{build}", "test", "{src}"],
    fmt_tools=("zig",), fmt_cmd=["{fmt}", "fmt", "{src}"],
    fix_cmd=["{fmt}", "fmt", "{src}"],
    notes="One portable download that also works as a C/C++ compiler — the "
          "best value if you only want to install one toolchain.",
    scaffold='''// {title}
const std = @import("std");

pub fn main() !void {{
    std.debug.print("hello from {title}\\n", .{{}});
}}

test "it works" {{
    try std.testing.expect(2 + 2 == 4);
}}
'''))

_add(Lang(
    id="batch", label="Batch (.bat)", ext=".bat", exts=(".bat", ".cmd"),
    comment="REM", run_tools=("cmd",),
    run_cmd=["{run}", "/c", "{src}"],
    notes="Here because this project's own installer is batch, and batch has "
          "traps — the `echo (text)` block-terminating bug among them — that "
          "are worth being able to test in isolation.",
    scaffold='''@echo off
REM {title}
echo hello from {title}
exit /b 0
'''))


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------

def get(lang_id: str) -> Lang | None:
    return LANGS.get((lang_id or "").lower())


def ids() -> list[str]:
    return list(LANGS)


def labels() -> list[tuple[str, str]]:
    """(id, label) in dropdown order."""
    return [(k, v.label) for k, v in LANGS.items()]


def for_extension(path: Any) -> Lang | None:
    """Which language owns this file? Used when a project file is opened."""
    text = str(path).lower()
    suffix = text[text.rfind("."):] if "." in text else ""
    for lang in LANGS.values():
        if suffix == lang.ext or suffix in lang.exts:
            return lang
    return None


def id_for_path(path: Any) -> str:
    lang = for_extension(path)
    return lang.id if lang else ""


def available_ids(ex: Any) -> list[str]:
    """Only the languages whose toolchain is actually installed right now."""
    return [k for k, v in LANGS.items() if v.available(ex)]


def scaffold_for(lang_id: str, title: str = "scratch",
                 stem: str = "main") -> str:
    """A file that builds and runs as-is, ready to be edited.

    Small models are far better at filling a skeleton than at inventing
    structure, so nothing here ever starts one from an empty buffer.
    """
    lang = get(lang_id)
    if not lang or not lang.scaffold:
        return ""
    return lang.scaffold.format(title=title, stem=stem)


def test_scaffold_for(lang_id: str, title: str = "scratch",
                      stem: str = "main") -> str:
    lang = get(lang_id)
    if not lang or not lang.test_scaffold:
        return ""
    return lang.test_scaffold.format(title=title, stem=stem)


def render(cmd: list, *, build: str = "", run: str = "", fmt: str = "",
           lint: str = "", src: str = "", out: str = "", dirpath: str = "",
           stem: str = "") -> list[str]:
    """Fill a command template. Returns a real argv list, never a string."""
    subs = {"build": build, "run": run, "fmt": fmt, "lint": lint,
            "src": src, "out": out, "dir": dirpath, "stem": stem}
    return [str(part).format(**subs) for part in cmd]


# ---------------------------------------------------------------------------
# GDScript specifics (§6.1a) — the parts that don't fit a generic registry
# ---------------------------------------------------------------------------

#: Anything in a GDScript test that behaves differently without a viewport.
#: A headless pass on code touching these is NOT unqualified success (M40).
_HEADLESS_SENSITIVE = (
    "get_viewport", "_physics_process", "RenderingServer", "get_tree(",
    "await ", "Timer", "_process(", "get_window", "DisplayServer",
    "Viewport", "CanvasItem", "move_and_slide", "PhysicsServer",
)

HEADLESS_CAVEAT = (
    "passed headlessly; this test touches the scene tree, physics or "
    "rendering, which behaves differently without a viewport. Verify in the "
    "editor before believing it.")


def godot_test_cmd(fs: Any, run_tool: str) -> tuple[list[str], str]:
    """Which Godot test runner this project uses, detected from `addons/`.

    Returns (argv, note). Detection rather than configuration, because a
    project that has GUT installed has already answered the question and
    asking again is one more thing to get wrong.

    If neither framework is present, the caller falls back to running the
    script — and the note says so, because "we ran it" and "the tests passed"
    are different claims (A.3.2).
    """
    try:
        addons = set(fs.list("addons/*"))
        addons |= set(fs.list("addons/**/*"))
    except Exception:                                    # noqa: BLE001
        addons = set()
    joined = " ".join(addons)
    if "gut" in joined:
        return ([run_tool, "--headless", "--fixed-fps", "60", "-s",
                 "addons/gut/gut_cmdln.gd", "-gdir=res://test", "-gexit"],
                "GUT")
    if "gdunit" in joined.lower():
        return ([run_tool, "--headless", "--fixed-fps", "60", "-s",
                 "addons/gdUnit4/bin/GdUnitCmdTool.gd", "-a", "test"],
                "gdUnit4")
    return ([], "neither GUT nor gdUnit4 is installed in addons/, so there "
                "is no test framework to run — the loop will verify by "
                "running the script instead, which is weaker evidence")


def headless_caveat_for(source: str) -> str:
    """The caveat this test earns, or "" when a headless pass is honest.

    Called on the TEST source, not the implementation: the question is
    whether the evidence is weaker than it looks, and that depends on what
    the test exercises.
    """
    hits = [tok for tok in _HEADLESS_SENSITIVE if tok in (source or "")]
    if not hits:
        return ""
    return HEADLESS_CAVEAT


def to_os_path(res_path: str) -> str:
    """`res://scripts/player.gd` → `scripts/player.gd`.

    The one real trap in GDScript support: the patcher and every Port work in
    OS-relative paths, and a `res://` prefix reaching `FileSystemPort` becomes
    a directory literally named `res:` on Windows. Translate at the boundary,
    every time.
    """
    text = str(res_path or "")
    for prefix in ("res://", "user://"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def to_res_path(os_path: str) -> str:
    """The inverse, for anything handed back to Godot itself."""
    text = str(os_path or "").replace("\\", "/").lstrip("/")
    return text if text.startswith(("res://", "user://")) else f"res://{text}"
