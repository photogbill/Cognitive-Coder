# SPDX-License-Identifier: Apache-2.0
"""Build, run, test, format and lint — with phases that name themselves.

WHY PHASES ARE KEPT APART. "It didn't work" is not a fixable error. A build
failure and a runtime crash have different causes, different fixes and
different feedback, and a loop that conflates them hands a small model a
compiler error while asking it to fix a logic bug. So every result names the
phase that failed: `guard`, `syntax`, `build`, `run` or `test` (M22).

**The acceptance test for this module is a discrimination test**, and it is
worth stating because it is the whole point: a deliberately broken C file must
return `failed_phase == "build"`, and a C file that compiles cleanly and then
divides by zero must return `failed_phase == "run"`. If those two are
indistinguishable, this module is wrong no matter what else it does.

WHAT IS AND ISN'T ENFORCED HERE

  * A hard wall-clock timeout on every phase, with the whole process tree
    killed (M16) — that guarantee belongs to `ExecPort`, and Godot is why it
    exists.
  * **A scrubbed environment**: no inherited API keys, no proxies, nothing
    that could leak into a compiler's telemetry or a test's HTTP client.
    `HTTP_PROXY`/`HTTPS_PROXY` are scrubbed too — a proxy variable is a
    network path, and C3 does not have exceptions.
  * cwd pinned to the workspace; output truncated with the cap stated.
  * The static screen runs BEFORE the compiler. Compiling generated code is
    itself a small risk, and more practically: refusing in a millisecond
    beats waiting sixty seconds for a build.

C4 lives here too: **`ok` means the build succeeded AND the tests ran.** Where
a language or project genuinely has neither, that is stated in `caveats` and
never quietly counted as success (M4). Parsing is a pre-check. It is never a
completion signal.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import replace
import os
import re
from typing import Any

from . import diagnostics, guard, langs
from .types import Diagnostic, PhaseResult, ProcResult, RunResult, Timeouts

# Environment handed to every child process. Deliberately minimal: whatever is
# in the operator's environment — tokens, proxies, licence servers — has no
# business in a build of generated code.
#
# Kept: what a toolchain needs merely to START. Go on Windows fails without
# LOCALAPPDATA (its build cache lives there), .NET and npm read APPDATA and
# ProgramData, a compiler linked against a private libdir needs
# LD_LIBRARY_PATH, and Windows tools look for SystemDrive. None of them is
# a credential or a network path.
_KEEP_FROM_ENV = ("PATH", "SystemRoot", "windir", "COMSPEC", "HOME",
                  "USERPROFILE", "LANG", "LC_ALL", "PATHEXT", "NUMBER_OF_"
                  "PROCESSORS", "PROCESSOR_ARCHITECTURE", "LOCALAPPDATA",
                  "APPDATA", "ProgramData", "LD_LIBRARY_PATH", "SystemDrive")

_FORCED_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "NO_COLOR": "1",             # colour codes are noise in a parsed log
    "TERM": "dumb",
    "CARGO_NET_OFFLINE": "true",
    "npm_config_offline": "true",
    "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    "DOTNET_NOLOGO": "1",
    "GODOT_SUPPRESS_UPDATE_CHECK": "1",
    # ruff otherwise drops `.ruff_cache/` into its working directory or the
    # project root — observed as three files appearing in the operator's
    # tree after one autofix, written without anybody approving anything.
    "RUFF_NO_CACHE": "true",
    # Explicitly emptied rather than merely absent, so a child that reads
    # them gets "" instead of inheriting from a parent shell.
    "HTTP_PROXY": "", "HTTPS_PROXY": "", "ALL_PROXY": "",
    "http_proxy": "", "https_proxy": "", "all_proxy": "",
}


def scrubbed_env(workdir: str, project_root: str = "") -> dict[str, str]:
    """The environment a child process gets. Nothing inherited by accident.

    ``project_root`` is put on `PYTHONPATH` (and Node's `NODE_PATH`), and
    that is not a convenience — without it, **no multi-file project can ever
    verify.** `src/cli.py` doing `from src.stats import summarise` fails with
    `ModuleNotFoundError: No module named 'src'` when the interpreter is
    started on the file rather than on the package, and the model is then
    asked to fix an error that has nothing to do with its code. It will try,
    and it will make things worse. The environment is the engine's
    responsibility, so the engine sets it.

    Scrubbing still applies to everything else: no inherited tokens, no
    licence servers, and no proxy variables — a proxy variable is a network
    path, and C3 does not have exceptions.
    """
    env = {k: os.environ.get(k, "") for k in _KEEP_FROM_ENV
           if os.environ.get(k)}
    env.update(_FORCED_ENV)
    env["TEMP"] = env["TMP"] = str(workdir)
    root = str(project_root or workdir or "")
    if root:
        env["PYTHONPATH"] = root
        env["NODE_PATH"] = root
    return env


def _phase(ex: Any, name: str, argv: Sequence[str], *, cwd: str,
           timeout: float, stdin: str = "",
           project_root: str = "") -> PhaseResult:
    """Run one phase through the ExecPort and wrap the result.

    ``timeout`` of 0 (or negative) means **no limit** — the phase waits. The
    ExecPort honours that; see `SubprocessExec.run`.
    """
    argv = [str(a) for a in argv]
    proc = ex.run(argv, cwd=cwd, timeout=timeout, stdin=stdin,
                  env=scrubbed_env(cwd, project_root or cwd))
    note = ""
    if proc.timed_out:
        # Say which knob raises it. An operator reading "exceeded 15s" cannot
        # tell whether that is a limit on the MODEL or on the program it
        # wrote, and the two have opposite remedies.
        note = (f"{name} was still going after {timeout:.0f}s and the whole "
                f"process tree was killed. This is a limit on the program "
                f"being verified, not on generation — raise it in "
                f"Setup › Coder › verification timeouts, or set it to 0 to "
                f"wait indefinitely.")
    return PhaseResult(name=name, argv=tuple(argv), proc=proc,
                       ok=proc.exit_code == 0 and not proc.timed_out,
                       note=note)


# ---------------------------------------------------------------------------
# programs that are SUPPOSED to keep running
# ---------------------------------------------------------------------------
#
# A game, a server and a GUI have one thing in common: they do not exit. The
# run phase launches them, waits, and kills them on the clock — and until now
# recorded that as `ok=False` with "run exceeded 15s and the whole process tree
# was killed".
#
# Which is exactly backwards. For a pygame racing game, exiting within fifteen
# seconds would be the BUG. The window opens, `while running:` spins, and the
# only way that process ends is somebody closing it. The build was correct and
# was marked a failure — then the timeout text went back to the model as a
# diagnostic, and it spent its remaining attempts trying to fix code that was
# already right.
#
# Same shape as the zero-tests bug and the same cost: a verdict that did not
# match what happened. There it was success reported for work not done; here it
# is failure reported for work done properly.
#
# THE DISCRIMINATION HAS TO BE HONEST, because "it timed out" alone cannot tell
# a healthy main loop from a deadlock. So BOTH must hold:
#
#   1. the code actually uses a main-loop framework — an import, not a guess;
#   2. it produced no error output before the clock ran out.
#
# A program that printed a traceback and then hung is still a failure. And what
# this earns is narrow and gets said out loud in a caveat: the program STARTED
# and STAYED UP. Nobody watched it play.

#: What means "this process is designed to outlive its launch", per language:
#: (pattern, view). Imports and loop calls, never heuristics like `while
#: True` — a marker that guesses would eventually excuse a real hang.
#:
#: Matched against CODE: comments and strings are blanked first (`view` is
#: "bare"), except where the marker IS a string — a JS module name, a Go
#: import path ("text": comments blanked only). And with word boundaries.
#: Observed: `# TODO: port the UI to tkinter later` made a deadlocked
#: script "ok — still running", and `wx` matched inside `bwxyz`.
_PY_LOOP_MODULES = (
    "pygame", "pyglet", "arcade", "panda3d", "ursina", "raylib", "pyray",
    "tkinter", "Tkinter", "PySide6", "PySide2", "PyQt6", "PyQt5", "wx",
    "kivy", "dearpygui", "customtkinter", "flask", "fastapi", "uvicorn",
    "gunicorn", "waitress", "aiohttp", r"http\.server", "socketserver",
    "SimpleHTTPServer", "django", "streamlit", "gradio")
_JS_LOOP = (
    (r"(?:\brequire\s*\(\s*|\bfrom\s+)['\"](?:express|fastify|koa|"
     r"socket\.io|@hapi/hapi)['\"]", "text"),
    (r"\bcreateServer\s*\(|\bapp\.listen\s*\(", "bare"))
_MAIN_LOOP_MARKERS: dict[str, tuple[tuple[str, str], ...]] = {
    "python": (
        (r"^[ \t]*(?:from[ \t]+|import[ \t]+(?:[\w.]+[ \t]*(?:as[ \t]+\w+)?"
         r"[ \t]*,[ \t]*)*)(?:" + "|".join(_PY_LOOP_MODULES) + r")\b",
         "bare"),
        (r"\.mainloop\s*\(|\.exec_?\s*\(\s*\)|\bserve_forever\s*\(|"
         r"\brun_forever\s*\(", "bare")),
    "javascript": _JS_LOOP,
    "typescript": _JS_LOOP,
    "go": ((r"\"[^\"\n]*\b(?:ebiten|raylib)\b[^\"\n]*\"", "text"),
           (r"\bListenAndServe\w*\s*\(", "bare")),
    "rust": ((r"\b(?:actix_web|axum|rocket|warp|bevy|ggez|macroquad|"
              r"winit)\b", "bare"),),
    "csharp": ((r"\bApplication\.Run\b|\bMonoGame\b|\bMicrosoft\.Xna\b|"
                r"\bWebApplication\.|\bapp\.Run\s*\(", "bare"),),
    "java": ((r"\bServerSocket\b|\bSpringApplication\.run\b|\bJFrame\b|"
              r"\bApplication\.launch\b", "bare"),),
    "cpp": ((r"\bSDL_Init\b|\bglfwInit\b|\bglutMainLoop\b|"
             r"\bQApplication\b", "bare"),),
    "c": ((r"\bSDL_Init\b|\bglfwInit\b|\bglutMainLoop\b", "bare"),),
}
_LOOP_COMPILED = {k: tuple((re.compile(p, re.M), v) for p, v in rules)
                  for k, rules in _MAIN_LOOP_MARKERS.items()}

#: Text that means the program was already in trouble when the clock ran out.
#: Lowercased substring match against combined output.
_CRASHED = ("traceback (most recent call last)", "segmentation fault",
            "panic:", "fatal error", "unhandled exception",
            "core dumped", "abort trap", "stack overflow",
            "modulenotfounderror", "importerror", "syntaxerror")


def has_main_loop(code: str, lang_id: str) -> bool:
    """Is this a program designed not to exit? Judged on code, not prose."""
    rules = _LOOP_COMPILED.get(lang_id, ())
    if not rules or not code:
        return False
    text, bare = guard.views(code, lang_id)
    return any(p.search(text if v == "text" else bare) for p, v in rules)


def _still_running(phase: PhaseResult, code: str, lang_id: str) -> bool:
    """Did a long-running program simply outlast the clock, healthily?

    Requires the main-loop marker AND clean output. Either alone is not
    enough: a script with no loop that hangs is a deadlock, and a game that
    printed a traceback before hanging is a crash.
    """
    proc = phase.proc
    if proc is None or not proc.timed_out:
        return False
    if not has_main_loop(code, lang_id):
        return False
    return not any(bad in proc.output.lower() for bad in _CRASHED)


def _unreachable_workspace(phase: PhaseResult, cwd: str) -> str:
    """Was the failure "this directory does not exist", not "this code is bad"?

    A host may pair an in-memory `FileSystemPort` with a real `ExecPort` —
    `MemoryFileSystem` plus `SubprocessExec` is the default `Host`, and it is
    a perfectly sensible arrangement for editing and outlining. But nothing
    can be BUILT there, because the files exist only in a dict.

    Left alone, that surfaces as `could not run: [Errno 2] No such file or
    directory: '/project'` attributed to the `run` phase — which reads as
    "your code is broken" and sends the model off to fix code that is fine.
    C6 says an operator-facing failure is a sentence naming what happened and
    what to do; C7 says a missing capability degrades with a stated cost.
    This is both.
    """
    proc = phase.proc
    if proc is None or proc.exit_code != -1:
        return ""
    text = (proc.stderr or "").lower()
    if "no such file or directory" not in text and "cannot find" not in text:
        return ""
    if str(cwd).lower() not in text and "errno 2" not in text:
        return ""
    return (f"the project's files are not on a real disk that commands can "
            f"be run in ({cwd}), so nothing can be built, run or tested. "
            f"Editing, outlining and the codemap all work; verification does "
            f"not. Point the host's FileSystemPort at a real directory to "
            f"turn verification on.")


# ---------------------------------------------------------------------------
# the cheap pre-check
# ---------------------------------------------------------------------------

def syntax_check(code: str, lang_id: str, *, ex: Any = None,
                 cwd: str = "", src_path: str = "") -> PhaseResult | None:
    """A pre-check, never a completion signal (C4, M4).

    Python is checked in-process with `ast.parse` — free, exact, and it needs
    no interpreter subprocess. Other languages use `Lang.syntax_cmd` when the
    toolchain is present; when it is not, this returns None, meaning "not
    checked", which is different from "checked and fine" and is reported as
    such.
    """
    if lang_id == "python":
        try:
            ast.parse(code)
            return PhaseResult(name="syntax", ok=True)
        except SyntaxError as exc:
            diag = Diagnostic(
                file=exc.filename or src_path, line=exc.lineno or 0,
                col=exc.offset, severity="error",
                message=f"{exc.msg}", code="syntax", tool="python-ast")
            return PhaseResult(
                name="syntax", ok=False,
                proc=ProcResult(exit_code=1, stderr=diag.one_line()),
                note="the file does not parse")

    lang = langs.get(lang_id)
    if not lang or not lang.syntax_cmd or ex is None:
        return None
    tool = lang.which_build(ex) or lang.which_run(ex)
    if not tool or tool == "-":
        return None
    argv = langs.render(lang.syntax_cmd, build=tool, run=tool,
                        src=src_path, dirpath=cwd,
                        stem=_stem(src_path))
    return _phase(ex, "syntax", argv, cwd=cwd or ".",
                  timeout=min(30.0, lang.build_timeout))


def _stem(path: str) -> str:
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


# ---------------------------------------------------------------------------
# build and run
# ---------------------------------------------------------------------------

def build_and_run(code: str, lang_id: str, *, fs: Any, ex: Any,
                  stem: str = "main", workdir: str = "",
                  project_mode: bool = False, stdin: str = "",
                  skip_guard: bool = False, path: str = "",
                  timeout: float | None = None,
                  build_timeout: float | None = None) -> RunResult:
    """Write, screen, build and run one file. Never raises on user-code error.

    ``stem`` matters for Java, where the public class must match the filename
    — the scaffold and the runner both derive the class from it, so a rename
    cannot silently break the build.

    ``path`` is the file's REAL location in the project. Pass it whenever the
    file belongs to the project rather than to a scratchpad: without it this
    writes `<stem>.<ext>` at the project root, which litters the tree with
    copies and — worse — verifies a file that is not the one the imports
    resolve against. `src/stats.py` verified as `/stats.py` is a different
    module with different neighbours, and it can pass while the real one
    fails.
    """
    lang = langs.get(lang_id)
    if lang is None:
        return RunResult(False, lang_id,
                         blocked=f"unknown language {lang_id!r}")

    findings = () if skip_guard else tuple(
        guard.scan(code, lang_id, project_mode))
    stop = guard.blocked(findings)
    if stop:
        return RunResult(False, lang_id, blocked=stop,
                         warnings=guard.advisory(findings))

    root = workdir or fs.root()
    src_rel = path or f"{stem}{lang.ext}"
    # Only write when the content differs. The loop has usually already
    # written this file through the patcher's transaction, and rewriting it
    # here would bypass the snapshot that makes undo possible (§6.5).
    already = fs.exists(src_rel) and fs.read(src_rel) == code
    if not already:
        fs.write(src_rel, code)
    src = _join(root, src_rel)
    out_path = _artefact(fs, root, stem + _EXE)
    phases: list[PhaseResult] = []
    caveats: list[str] = []

    def finish(ok: bool) -> RunResult:
        if not ok and phases:
            # Attribute an unreachable workspace to the WORKSPACE, not to the
            # code. Feeding "No such file or directory" back to the model as
            # a diagnostic sends it off to fix code that is fine.
            unreachable = _unreachable_workspace(phases[-1], root)
            if unreachable:
                return RunResult(ok=False, lang=lang_id, phases=tuple(phases),
                                 blocked=unreachable,
                                 warnings=guard.advisory(findings),
                                 caveats=tuple(caveats))
        text = ("\n".join(p.output for p in phases if not p.ok)
                or "\n".join(p.output for p in phases))
        diags: tuple[Diagnostic, ...] = ()
        if not ok:
            # `root`, so a traceback raised inside a library is located in
            # the project's own deepest frame — one the model can fix.
            diags = tuple(diagnostics.attach_source(
                diagnostics.parse(text, lang_id, root=root), fs,
                sources={src_rel: code}))
        return RunResult(ok=ok, lang=lang_id, phases=tuple(phases),
                         diagnostics=diags,
                         warnings=guard.advisory(findings),
                         caveats=tuple(caveats))

    # --- syntax (cheap, and a pre-check only) ----------------------------
    #
    # Only for languages that DON'T build. For a compiled language the build
    # IS the syntax check, and running the compiler twice would both waste
    # seconds and — worse — attribute a compile error to the `syntax` phase
    # instead of `build`, which is precisely the discrimination §6.4 requires
    # this module to get right (M22). The pre-check earns its place where it
    # is genuinely cheaper than the phase that follows: `ast.parse` for
    # Python, `--check-only` for GDScript, `-n` for bash.
    if not lang.needs_build:
        pre = syntax_check(code, lang_id, ex=ex, cwd=root, src_path=src)
        if pre is not None:
            phases.append(pre)
            if not pre.ok:
                return finish(False)

    # --- build -----------------------------------------------------------
    if lang.needs_build:
        tool = lang.which_build(ex)
        if not tool:
            return RunResult(False, lang_id, phases=tuple(phases),
                             blocked=lang.missing_note(ex))
        argv = langs.render(lang.build_cmd, build=tool, src=src,
                            out=out_path, dirpath=root, stem=stem)
        phases.append(_phase(ex, "build", argv, cwd=root,
                             timeout=Timeouts.resolve(build_timeout,
                                                      lang.build_timeout)))
        if not phases[-1].ok:
            return finish(False)

    # --- run -------------------------------------------------------------
    runner_tool = lang.which_run(ex)
    if not runner_tool:
        return RunResult(False, lang_id, phases=tuple(phases),
                         blocked=lang.missing_note(ex))
    argv = langs.render(lang.run_cmd, run=runner_tool, build=lang.which_build(ex),
                        src=src, out=out_path, dirpath=root, stem=stem)
    # SQL is the odd one: statements are piped in rather than passed as a file.
    piped = code if lang_id == "sql" else stdin
    patience = Timeouts.resolve(timeout, lang.run_timeout)
    phases.append(_phase(ex, "run", argv, cwd=root,
                         timeout=patience, stdin=piped))

    # A game that was still playing when the clock ran out passed. See
    # `_still_running` for why this needs two signals, not one.
    if _still_running(phases[-1], code, lang_id):
        held = phases[-1].proc.duration_s or patience
        phases[-1] = replace(
            phases[-1], ok=True,
            note=(f"still running after {held:.0f}s, which is what this "
                  f"program is supposed to do."))
        caveats.append(
            f"{src_rel} started cleanly and was still running after "
            f"{held:.0f}s, so it was stopped. That is the expected result for "
            f"a program with a main loop and it is NOT a failure — but it "
            f"also means only startup was verified. Nothing here checked that "
            f"it behaves correctly once running; only a test can do that.")

    if lang_id == "gdscript":
        caveats.append("run headlessly; without a viewport, scene-tree, "
                       "physics and rendering behaviour differ")
    return finish(phases[-1].ok)


#: Where compiled artefacts go — the program's binary and a test harness.
#: Not the project root: `<root>/main.bin` was litter beside the operator's
#: sources, and a harness written to `<root>/<stem>` collides with any file
#: or folder of that name. Beside the scratch copies, in the engine's state.
BUILD_DIR = ".cc_state/build"
_EXE = ".exe" if os.name == "nt" else ".bin"


def _artefact(fs: Any, root: str, name: str) -> str:
    """The exec-side path for a build artefact, its folder made to exist.

    Compilers do not create the parent of `-o` (rustc: "couldn't create a
    temp dir"), and the only directory-making primitive a FileSystemPort
    has is writing a file — so the folder is made by writing its
    `.gitignore`, which is also the right thing to leave there. If even
    that fails, the root is the fallback: litter, but a working build.
    """
    marker = f"{BUILD_DIR}/.gitignore"
    try:
        if not fs.exists(marker):
            fs.write(marker, "# build artefacts of programs Cognitive Coder "
                             "verified; safe to delete\n*\n")
        return _join(root, f"{BUILD_DIR}/{name}")
    except Exception:                                    # noqa: BLE001
        return _join(root, name)


def _test_phase(ex: Any, argv: Sequence[str], run_argv: Sequence[str], *,
                cwd: str, timeout: float) -> PhaseResult:
    """The test phase: one command, or build-the-harness THEN run it.

    Two commands, ONE phase named `test`: a harness that does not compile
    is a failed test, and `RunResult.tested` must mean the tests RAN — a
    separate, passing "compile the tests" phase would make it true for a
    harness that was built and never executed, which is the bug this
    exists to fix.
    """
    first = _phase(ex, "test", argv, cwd=cwd, timeout=timeout)
    if not run_argv or not first.ok or first.proc is None:
        return first
    left = timeout
    if timeout and timeout > 0:
        left = max(1.0, timeout - (first.proc.duration_s or 0.0))
    second = _phase(ex, "test", run_argv, cwd=cwd, timeout=left)
    if second.proc is None:
        return second
    a, b = first.proc, second.proc
    joined = ProcResult(
        exit_code=b.exit_code,
        stdout="\n".join(s for s in (a.stdout, b.stdout) if s),
        stderr="\n".join(s for s in (a.stderr, b.stderr) if s),
        duration_s=(a.duration_s or 0.0) + (b.duration_s or 0.0),
        timed_out=b.timed_out, truncated=a.truncated or b.truncated)
    return replace(second, proc=joined)


def run_tests(lang_id: str, *, fs: Any, ex: Any, stem: str = "main",
              workdir: str = "", timeout: float | None = None,
              test_source: str = "", path: str = "",
              test_path: str = "",
              known_failing: Sequence[str] = (),
              whole_suite: bool = False) -> RunResult:
    """Run the language's test command, honestly.

    Two honesty obligations are discharged here:

    * A language with no configured test runner does not silently pass. It
      returns `ok=False` with a `blocked` sentence saying the loop will verify
      by running instead — weaker evidence, named as such (C4, M4).
    * A headless Godot pass on a test that touches the scene tree, physics or
      rendering carries the headless caveat (M40). "The tests passed" may be
      said; "this works" may not.

    ``path`` is the file's real location, as for `build_and_run`; without
    it `{src}` is `<root>/<stem><ext>`, which for `src/lib.rs` is a file
    that does not exist.

    ``whole_suite`` runs everything with no scoping and no attribution —
    the baseline before a build, and the "anything failing elsewhere?"
    pass after a scoped success.

    ``test_path`` is the TASK's own test file (`Task.test_path`, M39). When
    it exists and the language can run one file (`test_one_cmd`, or
    `test_files`), only that file decides the verdict and supplies the
    diagnostics. The whole suite is then run once more, only if this file
    passed, and anything failing elsewhere becomes a caveat naming those
    files — never this file's diagnostics. Before this, one failing test
    anywhere in the project was fed back as the error of every file
    verified after it, and blocked the rest of the build.
    """
    lang = langs.get(lang_id)
    root = workdir or fs.root()
    if lang is None:
        return RunResult(False, lang_id, blocked=f"unknown language {lang_id!r}")

    caveats: list[str] = []
    argv: list[str] = []
    run_argv: list[str] = []
    files: list[str] = []
    test_timeout = Timeouts.resolve(timeout, lang.test_timeout)
    scoped = ""
    # A TEST task has no `test_path` of its own — the file IS the test. It
    # is scoped to itself, so its verdict comes from the file it wrote and a
    # stale neighbour cannot fail it (nor pass for it).
    own_test = (bool(path) and not test_path and not whole_suite
                and is_test_path(path))
    scope_to = "" if whole_suite else (test_path or (path if own_test
                                                      else ""))
    if scope_to and lang_id != "gdscript" and (lang.test_one_cmd
                                               or lang.test_files):
        rel = str(scope_to).replace("\\", "/")
        try:
            scoped = rel if fs.exists(rel) else ""
        except Exception:                                # noqa: BLE001
            scoped = ""

    if lang_id == "gdscript":
        tool = lang.which_run(ex)
        if not tool:
            return RunResult(False, lang_id, blocked=lang.missing_note(ex))
        argv, note = langs.godot_test_cmd(fs, tool)
        if not argv:
            return RunResult(
                False, lang_id,
                blocked=note,
                caveats=("no GDScript test framework detected",))
        caveat = langs.headless_caveat_for(test_source)
        if caveat:
            caveats.append(caveat)
        caveats.append(f"test runner: {note}, headless")
    else:
        if not lang.test_cmd:
            return RunResult(
                False, lang_id,
                blocked=(f"no test runner is configured for {lang.label} — "
                         f"the loop will verify by running the code instead, "
                         f"which is weaker evidence"))
        subs = {"build": lang.which_build(ex), "run": lang.which_run(ex),
                "src": _join(root, path or f"{stem}{lang.ext}"),
                "out": (_artefact(fs, root, f"{stem}.test{_EXE}")
                        if lang.test_run_cmd else _join(root, stem)),
                "dirpath": root, "stem": stem}
        if scoped:
            folder, _, name = scoped.rpartition("/")
            subs.update(test=_join(root, scoped), testname=name,
                        testdir=_join(root, folder) if folder else root)
        cmd = (lang.test_one_cmd if scoped and lang.test_one_cmd
               else lang.test_cmd)
        argv = langs.render(cmd, **subs)
        run_argv = langs.render(lang.test_run_cmd, **subs)
        if any("{" in str(p) or not str(p) for p in argv + run_argv):
            return RunResult(False, lang_id,
                             blocked="the test command could not be resolved "
                                     "— a required toolchain is missing")
        if lang.test_files:
            files = [scoped] if scoped else langs.find_test_files(fs, lang)
            if not files:
                return RunResult(
                    False, lang_id,
                    blocked=(f"no {lang.label} test file was found (looked "
                             f"for {', '.join(lang.test_files[:4])}, … and "
                             f"anything under test/ or tests/) — the loop "
                             f"will verify by running the code instead, "
                             f"which is weaker evidence"))
            # Relative to the cwd, so Node names each file the way the
            # zero-test check below expects.
            argv = argv + files

    phase = _test_phase(ex, argv, run_argv, cwd=root, timeout=test_timeout)
    if (not phase.ok and lang_id == "python" and phase.proc is not None
            and phase.proc.exit_code == 5 and zero_tests(phase.output)):
        # Python 3.12+ `unittest` exits 5 when it ran nothing. That is the
        # zero-test case — said in a caveat below — not a failing test,
        # and treating it as one fed "NO TESTS RAN" to the model as an
        # error in code that was fine.
        phase = replace(phase, ok=True,
                        note="exit status 5: unittest found no tests")
    diags: tuple[Diagnostic, ...] = ()
    if not phase.ok:
        diags = tuple(diagnostics.attach_source(
            diagnostics.parse(phase.output, lang_id, root=root), fs))
    # The file-based signal only means something on a passing run: a
    # test file that failed to LOAD is also listed under its own name.
    empty = zero_tests(phase.output, files if phase.ok else ())
    if empty and own_test and phase.ok:
        # The task WAS the test file, and the runner found no tests in it.
        # That is a failed task, not a caveat: a test file that collects
        # nothing is the model's mistake (wrong names, no TestCase, an
        # "existing file" pasted in place of the test), and it is the one
        # mistake a retry can fix — if it is told. Three builds of one spec
        # sealed such files as DONE; one of them was not even a test.
        return RunResult(
            ok=False, lang=lang_id, phases=(replace(phase, ok=False),),
            diagnostics=(Diagnostic(
                file=path, line=1, severity="error", tool="unittest",
                code="no-tests-collected",
                message=(f"{path} ran, but the test runner collected ZERO "
                         f"tests from it. A test file must define tests the "
                         f"runner can find: for unittest, methods named "
                         f"test_* on a class that subclasses "
                         f"unittest.TestCase, ending with "
                         f"`if __name__ == '__main__': unittest.main()`. "
                         f"Return the complete test file — not the module "
                         f"it tests.")),),
            caveats=tuple(caveats))
    if empty:
        caveats.append(empty)
    if scoped and phase.ok:
        elsewhere = _failing_elsewhere(lang_id, lang, fs=fs, ex=ex,
                                       stem=stem, workdir=workdir,
                                       timeout=timeout, path=path,
                                       mine=scoped)
        if elsewhere:
            caveats.append(elsewhere)
    if (not whole_suite and not scoped and not phase.ok and path
            and not own_test and (lang.test_one_cmd or lang.test_files)):
        # The WHOLE suite decided, because this module's own test does not
        # exist yet. Failures that do not name this module are other
        # files' — a stale test from an earlier session, a sibling's test
        # written against a plan that changed. Charged to this file, they
        # made the model regenerate correct code until stagnation stopped
        # it (Aug 8 and Oct 1, every module of the racing spec).
        #
        # Only for languages whose tests live in separate files: a Rust
        # module's tests are inside it, and a failure there is its own.
        # And only when the log NAMES other test modules — an unattributed
        # failure stays this file's failure, the honest default.
        mine, others = _attribute_failures(phase.output, path, diags,
                                           known_failing)
        if mine:
            diags = tuple(mine)
        elif others:
            which = ", ".join(others[:5])
            caveats.append(
                f"other tests in this project fail ({which}); none of them "
                f"names {path}, so they are not this file's errors and are "
                f"reported here instead.")
            return RunResult(ok=True, lang=lang_id,
                             phases=(replace(phase, ok=True, note=(
                                 "failures were in other files' tests")),),
                             caveats=tuple(caveats))
    return RunResult(ok=phase.ok, lang=lang_id, phases=(phase,),
                     diagnostics=diags, caveats=tuple(caveats))


def is_test_path(path: str) -> bool:
    """`tests/test_x.py`, `x_test.go`, `test/x.spec.ts`: a test by name."""
    rel = str(path or "").replace("\\", "/").lower()
    name = rel.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    folders = rel.split("/")[:-1]
    return (stem.startswith("test_") or stem.endswith("_test")
            or ".test" in name or ".spec" in name
            or any(f in ("test", "tests", "spec", "__tests__")
                   for f in folders))


def _module_tokens(path: str) -> list[str]:
    """The names a traceback or a test log uses for a file: its path, both
    slashes, its dotted module name, and its bare stem when distinctive."""
    rel = str(path or "").replace("\\", "/").strip("/")
    stem_path = rel.rsplit(".", 1)[0] if "." in rel.rsplit("/", 1)[-1] \
        else rel
    dotted = stem_path.replace("/", ".")
    out = [rel, rel.replace("/", "\\"), dotted]
    stem = stem_path.rsplit("/", 1)[-1]
    if len(stem) >= 4 and stem not in ("main", "test", "init", "util",
                                       "utils", "core", "base", "app"):
        out.append(stem)
    return [t for t in dict.fromkeys(out) if t]


def failing_test_modules(log: str) -> list[str]:
    """The test modules a unittest/pytest log blames, in order, once each.

    `ERROR: tests.test_physics (unittest.loader._FailedTest...)`,
    `FAIL: test_add (tests.test_calc.T.test_add)`, pytest's
    `FAILED tests/test_calc.py::test_add`. Used for the pre-build baseline
    and for the "other files" caveat.
    """
    out: list[str] = []
    for m in re.finditer(
            r"^(?:ERROR|FAIL):\s+(?P<head>\S+)\s+\((?P<ident>[\w.]+)\)|"
            r"^FAILED\s+(?P<pytest>[\w./\\-]+\.py)", log or "", flags=re.M):
        if m.group("pytest"):
            name = m.group("pytest")
        else:
            head, ident = m.group("head"), m.group("ident")
            if ident.startswith("unittest.loader._FailedTest"):
                # A test MODULE that failed to import. Python ≤ 3.11:
                # `ERROR: tests.test_x (unittest.loader._FailedTest)` — the
                # module is the head. Python 3.12+:
                # `ERROR: tests.test_x (unittest.loader._FailedTest.tests.test_x)`
                # — either will do; the head is there in both.
                name = head
            else:
                # `FAIL: test_add (tests.test_calc.C.test_add)` → the module
                # is the dotted id minus class and method.
                name = ident.rsplit(".", 2)[0] if ident.count(".") >= 2 \
                    else ident
        if "::" in name:
            name = name.split("::", 1)[0]
        if name not in out:
            out.append(name)
    return out


def _name_variants(name: str) -> list[str]:
    """`tests.test_physics` ⇄ `tests/test_physics.py` ⇄ `test_physics`.

    A log names a test module dotted; a diagnostic names its file with
    slashes; the baseline may hold either. Match all of them.
    """
    n = str(name or "").strip().replace("\\", "/")
    if not n:
        return []
    if n.endswith(".py"):
        path = n
        dotted = n[:-3].strip("/").replace("/", ".")
    elif "/" in n:
        path = n + ".py"
        dotted = n.strip("/").replace("/", ".")
    else:
        dotted = n
        path = n.replace(".", "/") + ".py"
    stem = dotted.rsplit(".", 1)[-1]
    out = [n, dotted, path, path.replace("/", "\\")]
    if len(stem) >= 6 and (stem.startswith("test") or stem.endswith("test")):
        out.append(stem)
    return list(dict.fromkeys(v for v in out if v))


def _attribute_failures(log: str, path: str, diags: Sequence[Diagnostic],
                        known_failing: Sequence[str] = ()
                        ) -> tuple[list[Diagnostic], list[str]]:
    """Split a failing suite's evidence into (this file's, other files').

    A diagnostic is this file's when it is located in it, or when the
    failure block it came from names the file's module — unless that block
    belongs to a test that was ALREADY failing before the build started
    (`known_failing`, the session's baseline): a stale test from an earlier
    session is never this file's problem. Others are listed by the test
    module that produced them, for the caveat.
    """
    tokens = _module_tokens(path)
    # Path-shaped tokens match anywhere; the bare stem only as a whole word,
    # so `physics` does not find itself inside `tests/test_physics.py` and
    # claim a stale test's failure for the module it was written against.
    pattern = re.compile("|".join(
        re.escape(t) if ("/" in t or "\\" in t or "." in t)
        else rf"(?<![\w])(?:{re.escape(t)})(?![\w])" for t in tokens))
    stale = [v for k in known_failing if k for v in _name_variants(k)]
    stale_pattern = re.compile("|".join(re.escape(k) for k in stale)) \
        if stale else None

    def names_me(text: str) -> bool:
        return bool(text) and bool(pattern.search(text))

    def is_stale(text: str) -> bool:
        return bool(stale_pattern and text and stale_pattern.search(text))

    blocks = [b for b in re.split(r"^={20,}\s*$", log or "", flags=re.M)
              if _FAILURE_LINE.search(b)]
    live_blocks = [b for b in blocks if not is_stale(b)]
    mine: list[Diagnostic] = []
    for d in diags:
        where = " ".join(x for x in (d.file, d.message, d.source_excerpt)
                         if x)
        if is_stale(where) and not names_me(d.file):
            continue
        if names_me(d.file) or names_me(d.message) \
                or names_me(d.source_excerpt):
            mine.append(d)
    # unittest/pytest headers name the test module; if a LIVE block names
    # this module, every non-stale diagnostic is this file's to answer.
    if not mine and any(names_me(b) for b in live_blocks):
        mine = [d for d in diags if not is_stale(
            " ".join(x for x in (d.file, d.message, d.source_excerpt)
                     if x))]
    others = failing_test_modules(log)
    return mine, others


# Lines of a test log that report a failure, per runner: unittest, pytest,
# a Python traceback frame, node's TAP, go test, and a Rust panic.
_FAILURE_LINE = re.compile(
    r"^\s*(?:FAIL:|ERROR:|FAILED\b|not ok\b|File \"|location:|--- FAIL|"
    r"thread '.*' panicked)", re.M)


def _failing_elsewhere(lang_id: str, lang: Any, *, fs: Any, ex: Any,
                       stem: str, workdir: str, timeout: float | None,
                       path: str, mine: str) -> str:
    """The caveat for failures OUTSIDE this task's test file, or "".

    Runs the whole suite once; the verdict has already been decided by the
    task's own file. Names the failing files where the log says which they
    are — a module name in `FAIL: test_add (test_calc.C.test_add)`, a path
    in a traceback frame or TAP `location:` — and otherwise quotes the
    first failure line, so the operator can always find it.
    """
    full = run_tests(lang_id, fs=fs, ex=ex, stem=stem, workdir=workdir,
                     timeout=timeout, path=path, whole_suite=True)
    if full.ok or full.blocked or not full.phases:
        return ""
    log = full.phases[0].output
    failing = "\n".join(ln for ln in log.splitlines()
                        if _FAILURE_LINE.match(ln))
    names: list[str] = []
    try:
        candidates = fs.list("*")
    except Exception:                                    # noqa: BLE001
        candidates = []
    exts = set(lang.exts or (lang.ext,))
    for raw in candidates:
        rel = str(raw).replace("\\", "/")
        base = rel.rsplit("/", 1)[-1]
        module, dot, ext = base.rpartition(".")
        if (not dot or f".{ext}" not in exts or rel == mine
                or "test" not in base.lower()
                or any(p.startswith(".") for p in rel.split("/")[:-1])):
            continue
        if rel in failing or re.search(rf"\b{re.escape(module)}\b",
                                       failing):
            names.append(rel)
    first = next((ln.strip() for ln in failing.splitlines()
                  if ln.strip()), "")
    which = (", ".join(names[:5]) if names
             else f"the first failure reads: {first[:120]}")
    return (f"{mine} passes. Other tests in this project fail ({which}); "
            f"they are reported here rather than as this file's errors, "
            f"because they belong to other tasks.")


# A test runner that collected nothing exits 0. That is the most dangerous
# green there is: C4 says "done" means the tests RAN, and a suite of zero
# tests passing is not evidence of anything. Every runner announces its
# count, so this is detectable rather than guessed at.
_EMPTY_RUN = (
    re.compile(r"^Ran 0 tests\b", re.M),                    # unittest
    re.compile(r"\bno tests ran\b", re.I),                  # pytest
    re.compile(r"\bcollected 0 items\b", re.I),             # pytest
    re.compile(r"\bno test files\b", re.I),                 # go
    re.compile(r"^# tests 0\b", re.M),                      # node --test
    re.compile(r"\b0 test(s)? (were )?run\b", re.I),        # GUT / misc
    re.compile(r"\brunning 0 tests\b", re.I),               # rust
)


def zero_tests(output: str, files: Sequence[str] = ()) -> str:
    """The caveat a zero-test run earns, or "" when tests actually ran.

    Separate and public because the loop needs the same judgement: a task
    whose tests "passed" without existing has not been verified, and F2's
    "the test must FAIL first" check depends on telling the two apart.

    ``files`` are the test files handed to a runner that takes them. Node
    reports a file with no `test()` calls in it as ONE passing test named
    after the file (`# Subtest: main.test.js` / `# tests 1`), so "# tests 0"
    never appears; when every file passed shows up that way, nothing ran.
    """
    text = output or ""
    empty = any(pattern.search(text) for pattern in _EMPTY_RUN)
    if not empty and files:
        named = set(re.findall(r"^# Subtest: (.+?)\s*$", text, re.M))
        hollow = [f for f in files
                  if {f, f.replace("/", "\\"),
                      f.rsplit("/", 1)[-1]} & named]
        empty = len(hollow) == len(files)
    if empty:
        return ("the test command succeeded but ran ZERO tests — that is "
                "not evidence the code works, only that nothing "
                "contradicted it")
    return ""


def verify(code: str, lang_id: str, *, fs: Any, ex: Any, stem: str = "main",
           workdir: str = "", project_mode: bool = False,
           test_source: str = "", skip_guard: bool = False,
           path: str = "", timeouts: Timeouts | None = None,
           test_path: str = "",
           known_failing: Sequence[str] = ()) -> RunResult:
    """The C4 definition of done: it builds AND the tests run (M4).

    This is the function the loop calls, and the one place where "done" is
    decided. It refuses to report success on a parse, and where a project
    genuinely has no tests it says so in `caveats` rather than counting the
    absence as a pass.

    ``timeouts`` bounds the generated PROGRAM, never the model. `None` on any
    field takes the language default; `0` waits indefinitely.

    ``test_path`` scopes the test phase to the task's own test file; see
    `run_tests`. Failures elsewhere come back as caveats, not diagnostics.
    ``known_failing`` names the test modules that were already failing
    before the build began (the session's baseline); their failures are
    never charged to the file being verified.
    """
    clocks = timeouts or Timeouts()
    built = build_and_run(code, lang_id, fs=fs, ex=ex, stem=stem,
                          workdir=workdir, project_mode=project_mode,
                          skip_guard=skip_guard, path=path,
                          timeout=clocks.run, build_timeout=clocks.build)
    if not built.ok:
        return built

    tested = run_tests(lang_id, fs=fs, ex=ex, stem=stem, workdir=workdir,
                       test_source=test_source, timeout=clocks.test,
                       path=path, test_path=test_path,
                       known_failing=known_failing)
    if tested.blocked:
        # No test runner is not a pass and not a failure — it is a stated
        # weakness in the evidence. C4 requires saying so out loud.
        return RunResult(
            ok=True, lang=lang_id, phases=built.phases,
            warnings=built.warnings,
            caveats=built.caveats + (
                f"it built and ran, but nothing was tested: {tested.blocked}",))
    return RunResult(ok=tested.ok, lang=lang_id,
                     phases=built.phases + tested.phases,
                     diagnostics=tested.diagnostics,
                     warnings=built.warnings,
                     caveats=built.caveats + tested.caveats)


# ---------------------------------------------------------------------------
# format and lint — deterministic, and therefore free (C5, F1)
# ---------------------------------------------------------------------------

#: Where format, lint and autofix put the candidate they are asked about.
#:
#: NOT the project path. Those three used to write `<stem><ext>` at the
#: project ROOT before the guard and before the transaction, so with the
#: library default `DenyAll` a task for `main.py` overwrote the operator's
#: `main.py` with unapproved model output, and `src/util.py` clobbered a root
#: `util.py`. The only road from a candidate to a project file is the
#: patcher's approval gate (M18); everything else works on a copy.
#:
#: Inside the project, beside the engine's other state, rather than in the
#: system temp directory, for two reasons: it goes through the host's
#: FileSystemPort like every other write (C2), so a host whose ExecPort only
#: sees the project still works; and a formatter run here still finds the
#: project's own `pyproject.toml` / `.clang-format` / `rustfmt.toml` by
#: walking up, so the house style it applies is the operator's, not the
#: tool's default. A temp dir outside would reformat to the wrong line
#: length and hand the model a whole-file diff.
SCRATCH_DIR = ".cc_state/scratch"


def _scratch_rel(stem: str, ext: str) -> str:
    safe = re.sub(r"[\\/:]+", "_", str(stem or "")).strip(".") or "main"
    return f"{SCRATCH_DIR}/{safe}{ext}"


def _scratch_tool(code: str, lang: Any, *, fs: Any, ex: Any, stem: str,
                  workdir: str, name: str, cmd: Sequence[str], tool: str,
                  timeout: float) -> tuple[str | None, PhaseResult | None,
                                           str]:
    """Run one fixer/formatter/linter over a COPY. (text, phase, output).

    ``workdir`` keeps its existing meaning — where the ExecPort sees the
    project root, when that differs from `fs.root()` — so the copy's
    exec-side path is `workdir/.cc_state/scratch/…`. The copy is deleted
    afterwards, along with the build artefact a compiler-driven linter such
    as clippy-driver drops beside it; nothing is left for the codemap, the
    planner or a test runner to mistake for source.

    ``output`` has the scratch path rewritten to the plain file name: the
    model is shown the file it is working on, never the engine's scratch
    directory, which it would otherwise try to import from.
    """
    root = workdir or fs.root()
    rel = _scratch_rel(stem, lang.ext)
    here = rel.rsplit("/", 1)[-1]
    scratch = _join(root, SCRATCH_DIR)
    try:
        fs.write(rel, code)
    except Exception:                                    # noqa: BLE001
        return None, None, ""
    try:
        src = _join(root, rel)
        argv = langs.render(list(cmd), fmt=tool, lint=tool, src=src,
                            dirpath=scratch, stem=here.rsplit(".", 1)[0])
        phase = _phase(ex, name, argv, cwd=scratch, timeout=timeout,
                       project_root=root)
        try:
            after: str | None = fs.read(rel)
        except Exception:                                # noqa: BLE001
            after = None
        output = phase.output
        # Longest spelling first: the relative path is a suffix of the
        # absolute one, and replacing it first leaves `/root/main.py`.
        for spelling in sorted({src, src.replace("\\", "/"), rel,
                                rel.replace("/", "\\")}, key=len,
                               reverse=True):
            output = output.replace(spelling, here)
        return after, phase, output
    finally:
        base = rel[:-len(lang.ext)] if lang.ext else rel
        for leftover in (rel, base, base + ".exe", base + ".pdb"):
            try:
                if fs.exists(leftover):
                    fs.delete(leftover)
            except Exception:                            # noqa: BLE001
                pass


def format_code(code: str, lang_id: str, *, fs: Any, ex: Any,
                stem: str = "main", workdir: str = "") -> tuple[str, str]:
    """Run the language's formatter. Returns (text, note).

    Formatting happens before the model ever sees a file: consistent layout
    means the model spends its attention on logic rather than on guessing the
    house style. A missing formatter is a note, not a failure (C7).

    Code in, code out: the formatter runs on a copy in `SCRATCH_DIR` and no
    project file is touched. Writing the result is the caller's business,
    through the patcher and its approval gate.
    """
    lang = langs.get(lang_id)
    if lang is None or not lang.fmt_cmd:
        return code, ""
    tool = lang.which_tool(ex, lang.fmt_tools)
    if not tool:
        return code, (f"no formatter installed ({', '.join(lang.fmt_tools)}) "
                      f"— layout is left as the model wrote it")
    after, phase, output = _scratch_tool(
        code, lang, fs=fs, ex=ex, stem=stem, workdir=workdir, name="format",
        cmd=lang.cmd_for("fmt", tool), tool=tool, timeout=30.0)
    if after is None or phase is None:
        return code, "the formatter produced no readable output"
    return after, ("" if phase.ok else output[:200])


def autofix(code: str, lang_id: str, *, fs: Any, ex: Any, stem: str = "main",
            workdir: str = "") -> tuple[str, list[str]]:
    """Apply the `--fix`-able rules. Returns (text, list of what was done).

    F1: never ask the model what a rule can answer. Every error fixed
    mechanically is minutes of generation not spent — and models botch
    trivial fixes surprisingly often, usually by rewriting the surrounding
    function while they are in there.

    Every auto-fix is returned so the caller can log it (M35). If the same one
    recurs constantly, the PROMPT needs changing, and the log is how anyone
    finds out.

    Code in, code out: the fixer runs on a copy in `SCRATCH_DIR`. It used to
    run on `<root>/<stem><ext>`, which wrote unapproved model output over the
    operator's file before the patcher had asked anyone (M18).
    """
    lang = langs.get(lang_id)
    if lang is None:
        return code, []
    done: list[str] = []
    text = code

    # Cheap, universal, and correct in every language: a trailing newline.
    if text and not text.endswith("\n"):
        text += "\n"
        done.append("added the missing trailing newline")

    # The fixer is chosen by the slot its template names (Lang.fix_command):
    # Go's `{fmt} -w` must run gofmt, not the first lint tool (`go`).
    tool, fix_cmd = lang.fix_command(ex)
    if tool:
        fixed, _phase_run, _out = _scratch_tool(
            text, lang, fs=fs, ex=ex, stem=stem, workdir=workdir,
            name="autofix", cmd=fix_cmd, tool=tool, timeout=60.0)
        # A fixer that had nothing to offer, or failed, leaves the text
        # as it was; neither is worth a note.
        if fixed is not None and fixed != text:
            text = fixed
            name = str(tool).replace("\\", "/").rsplit("/", 1)[-1]
            done.append(f"ran {name} --fix over the file")
    return text, done


def lint_code(code: str, lang_id: str, *, fs: Any, ex: Any,
              stem: str = "main",
              workdir: str = "") -> tuple[list[Diagnostic], str]:
    """Run whatever linter exists. Returns (diagnostics, note).

    No linter installed is a degraded mode, not a crash (C7, M6) — and the
    note says what the absence costs, so the operator knows which mode they
    are in.

    Lints a copy in `SCRATCH_DIR`; no project file is written. Diagnostics
    name `<stem><ext>`, not the scratch path.
    """
    lang = langs.get(lang_id)
    if lang is None or not lang.lint_cmd:
        return [], ""
    tool = lang.which_tool(ex, lang.lint_tools)
    if not tool:
        return [], (f"no linter installed ({', '.join(lang.lint_tools)}) — "
                    f"style and unused-symbol problems will only surface if "
                    f"they break the build")
    _after, phase, output = _scratch_tool(
        code, lang, fs=fs, ex=ex, stem=stem, workdir=workdir, name="lint",
        cmd=lang.cmd_for("lint", tool), tool=tool, timeout=60.0)
    if phase is None:
        return [], ("the linter could not be run: the scratch copy could "
                    "not be written")
    return diagnostics.parse(output, lang_id), ""


def _join(root: str, rel: str) -> str:
    """Join without importing pathlib semantics into a Port's namespace.

    Deliberately string-level: `root` came from a host's `FileSystemPort` and
    may not be a real local path at all (a MemoryFileSystem's root is
    `/project`). The only consumer of the result is an argv list handed to
    `ExecPort`, whose host decides what a path means.
    """
    if not root:
        return rel
    sep = "\\" if ("\\" in root and "/" not in root) else "/"
    return root.rstrip("/\\") + sep + str(rel).lstrip("/\\")
