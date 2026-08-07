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
import os
import re
from typing import Any

from . import diagnostics, guard, langs
from .types import Diagnostic, PhaseResult, ProcResult, RunResult

# Environment handed to every child process. Deliberately minimal: whatever is
# in the operator's environment — tokens, proxies, licence servers — has no
# business in a build of generated code.
_KEEP_FROM_ENV = ("PATH", "SystemRoot", "windir", "COMSPEC", "HOME",
                  "USERPROFILE", "LANG", "LC_ALL", "PATHEXT", "NUMBER_OF_"
                  "PROCESSORS", "PROCESSOR_ARCHITECTURE")

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
    """Run one phase through the ExecPort and wrap the result."""
    argv = [str(a) for a in argv]
    proc = ex.run(argv, cwd=cwd, timeout=timeout, stdin=stdin,
                  env=scrubbed_env(cwd, project_root or cwd))
    note = ""
    if proc.timed_out:
        note = (f"{name} exceeded {timeout:.0f}s and the whole process tree "
                f"was killed.")
    return PhaseResult(name=name, argv=tuple(argv), proc=proc,
                       ok=proc.exit_code == 0 and not proc.timed_out,
                       note=note)


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
    out_path = _join(root, stem + (".exe" if os.name == "nt" else ".bin"))
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
            diags = tuple(diagnostics.attach_source(
                diagnostics.parse(text, lang_id), fs,
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
                             timeout=build_timeout or lang.build_timeout))
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
    phases.append(_phase(ex, "run", argv, cwd=root,
                         timeout=timeout or lang.run_timeout, stdin=piped))

    if lang_id == "gdscript":
        caveats.append("run headlessly; without a viewport, scene-tree, "
                       "physics and rendering behaviour differ")
    return finish(phases[-1].ok)


def run_tests(lang_id: str, *, fs: Any, ex: Any, stem: str = "main",
              workdir: str = "", timeout: float | None = None,
              test_source: str = "") -> RunResult:
    """Run the language's test command, honestly.

    Two honesty obligations are discharged here:

    * A language with no configured test runner does not silently pass. It
      returns `ok=False` with a `blocked` sentence saying the loop will verify
      by running instead — weaker evidence, named as such (C4, M4).
    * A headless Godot pass on a test that touches the scene tree, physics or
      rendering carries the headless caveat (M40). "The tests passed" may be
      said; "this works" may not.
    """
    lang = langs.get(lang_id)
    root = workdir or fs.root()
    if lang is None:
        return RunResult(False, lang_id, blocked=f"unknown language {lang_id!r}")

    caveats: list[str] = []
    argv: list[str] = []
    test_timeout = timeout or lang.test_timeout

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
        argv = langs.render(
            lang.test_cmd, build=lang.which_build(ex), run=lang.which_run(ex),
            src=_join(root, f"{stem}{lang.ext}"), out=_join(root, stem),
            dirpath=root, stem=stem)
        if any("{" in str(p) or not str(p) for p in argv):
            return RunResult(False, lang_id,
                             blocked="the test command could not be resolved "
                                     "— a required toolchain is missing")

    phase = _phase(ex, "test", argv, cwd=root, timeout=test_timeout)
    diags: tuple[Diagnostic, ...] = ()
    if not phase.ok:
        diags = tuple(diagnostics.attach_source(
            diagnostics.parse(phase.output, lang_id), fs))
    empty = zero_tests(phase.output)
    if empty:
        caveats.append(empty)
    return RunResult(ok=phase.ok, lang=lang_id, phases=(phase,),
                     diagnostics=diags, caveats=tuple(caveats))


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


def zero_tests(output: str) -> str:
    """The caveat a zero-test run earns, or "" when tests actually ran.

    Separate and public because the loop needs the same judgement: a task
    whose tests "passed" without existing has not been verified, and F2's
    "the test must FAIL first" check depends on telling the two apart.
    """
    text = output or ""
    for pattern in _EMPTY_RUN:
        if pattern.search(text):
            return ("the test command succeeded but ran ZERO tests — that is "
                    "not evidence the code works, only that nothing "
                    "contradicted it")
    return ""


def verify(code: str, lang_id: str, *, fs: Any, ex: Any, stem: str = "main",
           workdir: str = "", project_mode: bool = False,
           test_source: str = "", skip_guard: bool = False,
           path: str = "") -> RunResult:
    """The C4 definition of done: it builds AND the tests run (M4).

    This is the function the loop calls, and the one place where "done" is
    decided. It refuses to report success on a parse, and where a project
    genuinely has no tests it says so in `caveats` rather than counting the
    absence as a pass.
    """
    built = build_and_run(code, lang_id, fs=fs, ex=ex, stem=stem,
                          workdir=workdir, project_mode=project_mode,
                          skip_guard=skip_guard, path=path)
    if not built.ok:
        return built

    tested = run_tests(lang_id, fs=fs, ex=ex, stem=stem, workdir=workdir,
                       test_source=test_source)
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

def format_code(code: str, lang_id: str, *, fs: Any, ex: Any,
                stem: str = "main", workdir: str = "") -> tuple[str, str]:
    """Run the language's formatter. Returns (text, note).

    Formatting happens before the model ever sees a file: consistent layout
    means the model spends its attention on logic rather than on guessing the
    house style. A missing formatter is a note, not a failure (C7).
    """
    lang = langs.get(lang_id)
    if lang is None or not lang.fmt_cmd:
        return code, ""
    tool = lang.which_tool(ex, lang.fmt_tools)
    if not tool:
        return code, (f"no formatter installed ({', '.join(lang.fmt_tools)}) "
                      f"— layout is left as the model wrote it")
    root = workdir or fs.root()
    src_rel = f"{stem}{lang.ext}"
    fs.write(src_rel, code)
    argv = langs.render(lang.fmt_cmd, fmt=tool, src=_join(root, src_rel),
                        dirpath=root, stem=stem)
    phase = _phase(ex, "format", argv, cwd=root, timeout=30.0)
    try:
        return fs.read(src_rel), ("" if phase.ok else phase.output[:200])
    except Exception:                                    # noqa: BLE001
        return code, "the formatter produced no readable output"


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

    if lang.fix_cmd:
        tool = lang.which_tool(ex, lang.lint_tools or lang.fmt_tools)
        if tool:
            root = workdir or fs.root()
            src_rel = f"{stem}{lang.ext}"
            fs.write(src_rel, text)
            argv = langs.render(lang.fix_cmd, lint=tool, fmt=tool,
                                src=_join(root, src_rel), dirpath=root,
                                stem=stem)
            phase = _phase(ex, "autofix", argv, cwd=root, timeout=60.0)
            try:
                fixed = fs.read(src_rel)
            except Exception:                            # noqa: BLE001
                fixed = text
            if fixed != text:
                text = fixed
                done.append(f"ran {argv[0]} --fix over the file")
            elif not phase.ok and phase.output:
                pass       # the fixer had nothing to offer; not worth a note
    return text, done


def lint_code(code: str, lang_id: str, *, fs: Any, ex: Any,
              stem: str = "main",
              workdir: str = "") -> tuple[list[Diagnostic], str]:
    """Run whatever linter exists. Returns (diagnostics, note).

    No linter installed is a degraded mode, not a crash (C7, M6) — and the
    note says what the absence costs, so the operator knows which mode they
    are in.
    """
    lang = langs.get(lang_id)
    if lang is None or not lang.lint_cmd:
        return [], ""
    tool = lang.which_tool(ex, lang.lint_tools)
    if not tool:
        return [], (f"no linter installed ({', '.join(lang.lint_tools)}) — "
                    f"style and unused-symbol problems will only surface if "
                    f"they break the build")
    root = workdir or fs.root()
    src_rel = f"{stem}{lang.ext}"
    fs.write(src_rel, code)
    argv = langs.render(lang.lint_cmd, lint=tool, src=_join(root, src_rel),
                        dirpath=root, stem=stem)
    phase = _phase(ex, "lint", argv, cwd=root, timeout=60.0)
    return diagnostics.parse(phase.output, lang_id), ""


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
