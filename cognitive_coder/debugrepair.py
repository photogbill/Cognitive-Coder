# SPDX-License-Identifier: Apache-2.0
"""Debug & Repair: the operator ran it, it failed, here is what they saw.

Bill, 2026-10-10: *"I'm actually fine with not doing a smoke run, if I can
try running it myself on my own and then feed the errors back into
cognitive coder. Perhaps a codebase debug / repair tab that is specific for
that function?"*

A build ends with the engine's own checks. Real use starts after that: the
operator runs the program, it crashes on the third level or prints the
wrong total, and the useful thing to do with that is hand it straight
back. This module takes what they have — any of:

* **the error text**, pasted as it was printed (a traceback, a compiler
  error, a test failure);
* **the command** they ran (`python main.py`), so it can be run again here
  to capture the error, and run again after each repair to CONFIRM the
  error is gone — the one check that answers the question they asked;
* **a description in words** ("the ball goes through the paddle"), for a
  fault that prints nothing;
* **the file(s)**, when they know where it is;

and does, deterministic first, model second, operator last:

1. **Locate.** The error is parsed (`diagnostics.parse`, the same parser
   every build uses) and kept only where it names a file of THIS project —
   the deepest project frame of a traceback, not the library it died in.
   Nothing located (a wrong total prints no traceback) and no file named:
   ONE model call reads the project's file list and outline and names at
   most three files, with why. With `confirm_targets`, the session stops
   there and returns the proposal, so the host can ask before anything is
   rewritten (Guided mode does).
2. **Repair.** Each target goes through the loop's ordinary seeded repair
   (`Loop.run_task(seed=…)`) — the same attempts, verification, patch
   transaction, snapshots and approval gate as a build, so a repair can be
   rejected, rolled back and diffed like any other change. The file as it
   stood is the BASELINE: a "repair" that returns it unchanged stops at once
   as not repaired. Diffusion mode applies: a located error is a hole.
3. **Confirm.** With a command, it is run again (bounded by `run_timeout`;
   a program still running at the end WITHOUT an error — a game, a server —
   counts as "no error within N s", and says so). The same error back means
   that round failed; a different error is progress and drives the next
   round, up to `rounds`. Without a command, the verdict is the files' own
   checks and says that only running it confirms the fix.

4. **Record.** Every round goes to BUILD_LOG.txt in words (what was seen,
   what was located and how, which files and why, each repair's outcome,
   the confirming run's output), to the journal (`debug` events), and to
   `.cc_state/debug/`: one Markdown report per session with each changed
   file's diff, and `history.jsonl`, one line per session, which the host
   shows as the debug history.

WHAT IT NEVER DOES: run anything the operator did not type (the command is
theirs and runs in the project folder, under the host's ExecPort, whose
contract kills the whole tree on timeout); write outside the project; or
touch a file it was not pointed at by the error, the operator or — named
and shown — the locate step.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass, field
import difflib
import json
import os
import re
import shlex
import time
from typing import Any

from . import diagnostics as diag_mod
from . import langs, nonerror, runner
from .errors import BudgetExceeded, Cancelled
from .providers.base import find_json_object
from .types import Diagnostic, Task

DEBUG_DIR = ".cc_state/debug"
HISTORY = f"{DEBUG_DIR}/history.jsonl"
#: How much of the error the prompts and the log carry (both ends kept).
ERROR_CHARS = 4000
#: The locate step names at most this many files; a repair round touches
#: at most this many.
MAX_TARGETS = 3
#: Files shown to the locate step (by name, with their outline).
LOCATE_FILES = 80

LOCATE_CONTRACT = (
    "Reply with ONE JSON object and nothing else:\n"
    '{"files": ["path/as/listed.py"], "why": "one or two sentences"}\n'
    f"At most {MAX_TARGETS} files, most likely first, each EXACTLY as it "
    "appears in the list. Name the file whose code must change, not the "
    "file that merely shows the symptom.")


@dataclass(frozen=True)
class DebugReport:
    """What the operator brings back. Any one field is enough."""
    error_text: str = ""
    command: str = ""
    description: str = ""
    files: tuple = ()
    rounds: int = 3
    #: How long a confirming run may take. A program still running at the
    #: end without an error counts as "no error in that time" — said so.
    run_timeout: float = 60.0
    #: Stop after locating and return the proposal (the host asks first).
    confirm_targets: bool = False

    def empty(self) -> bool:
        return not (self.error_text.strip() or self.command.strip()
                    or self.description.strip() or self.files)


@dataclass
class DebugRound:
    n: int
    seen: str = ""                 # the error this round worked from
    located: list = field(default_factory=list)   # [{file, line, message}]
    targets: list = field(default_factory=list)
    how: str = ""                  # how the targets were chosen
    outcomes: list = field(default_factory=list)  # [{path, label, ...}]
    changed: list = field(default_factory=list)
    check: str = ""                # what the confirming run said
    verdict: str = ""


@dataclass
class DebugResult:
    #: fixed | repaired | still-failing | nothing-to-repair | located |
    #: not-located | no-change | command-failed | stopped | error
    state: str
    verdict: str
    rounds: list = field(default_factory=list)
    #: the locate step's proposal (state "located"), with why
    targets: list = field(default_factory=list)
    why: str = ""
    report_path: str = ""
    diffs: dict = field(default_factory=dict)      # {path: unified diff}
    last_output: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# small, pure pieces
# ---------------------------------------------------------------------------

#: Shell features the command cannot use: it is run as an argument list,
#: never through a shell (the ExecPort contract), so `&&`, a pipe or a
#: redirect would be passed to the program as words.
_SHELL = re.compile(r"&&|\|\||[|<>;&]|^\s*cd\s", re.I)
#: Python asking for keyboard input with nothing to read.
_EOF = re.compile(r"EOFError: EOF when reading a line")
#: Bits of an error message that change between runs without the error
#: changing: object addresses, ids, timings.
_NOISE = re.compile(r"0x[0-9a-fA-F]+|\b\d+(\.\d+)?\s*(ms|s)\b")
_SKIP_DIRS = (".cc_", ".git", "__pycache__", ".venv", "venv", "env",
              "node_modules", "site-packages", ".mypy_cache",
              ".pytest_cache", "build", "dist", ".idea", ".vscode")


def _windows_argv(text: str) -> list[str] | None:
    """Windows' own rules (CommandLineToArgvW), so `--name="Bill Smith"`
    and `"C:\\Program Files\\x.exe"` split exactly as cmd.exe would."""
    try:
        import ctypes
        from ctypes import wintypes
        fn = ctypes.windll.shell32.CommandLineToArgvW
        fn.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.POINTER(wintypes.LPWSTR)
        n = ctypes.c_int(0)
        arr = fn(text, ctypes.byref(n))
        if not arr:
            return None
        try:
            return [arr[i] for i in range(n.value)]
        finally:
            ctypes.windll.kernel32.LocalFree(arr)
    except Exception:                                    # noqa: BLE001
        return None


def split_command(command: str) -> list[str]:
    """The operator's command as an argument list (the ExecPort never takes
    a string). On Windows, Windows' own splitting rules."""
    text = (command or "").strip()
    if not text:
        return []
    if os.name == "nt":
        got = _windows_argv(text)
        if got is not None:
            return got
        parts = shlex.split(text, posix=False)
        return [p[1:-1] if len(p) >= 2 and p[0] == p[-1] == '"' else p
                for p in parts]
    return shlex.split(text)


def command_problem(command: str) -> str:
    """Why this command cannot be run as typed, or ''."""
    text = (command or "").strip()
    if not text:
        return ""
    unquoted = re.sub(r'"[^"]*"|\'[^\']*\'', "", text)
    if _SHELL.search(unquoted):
        return ("it uses a shell feature (&&, a pipe, a redirect, `cd` or "
                "`;`), and it is run directly, not through a shell — type "
                "just the program and its arguments, e.g. `python main.py`")
    try:
        if not split_command(text):
            return "it is empty"
    except ValueError as exc:
        return f"its quotes do not match ({exc})"
    return ""


def clip(text: str, limit: int = ERROR_CHARS) -> str:
    """Both ends of a long output: the cause is often at the top, the
    verdict always at the bottom."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    half = limit // 2
    return (text[:half] + f"\n… {len(text) - limit:,} characters left out "
            f"…\n" + text[-half:])


def relative(path: str, root: str, fs: Any) -> str:
    """A diagnostic's file as a path of THIS project, or '' when it is not
    one (a library, the interpreter, a file that does not exist)."""
    if not path:
        return ""
    p = path.replace("\\", "/")
    r = (root or "").replace("\\", "/").rstrip("/")
    if r and (p.lower().startswith(r.lower() + "/")
              if len(r) > 1 and r[1] == ":" else p.startswith(r + "/")):
        p = p[len(r) + 1:]
    elif p.startswith("/") or (len(p) > 1 and p[1] == ":"):
        return ""
    while p.startswith("./"):
        p = p[2:]
    try:
        return p if fs.exists(p) else ""
    except Exception:                                    # noqa: BLE001
        return ""


def locate(text: str, root: str, fs: Any
           ) -> tuple[dict[str, list[Diagnostic]], list[Diagnostic]]:
    """{project file: its diagnostics} and the diagnostics that name no
    file of this project. Every language's parser is tried — the panel's
    language box says what is being BUILT, not what printed the error."""
    found: dict[str, list[Diagnostic]] = {}
    loose: list[Diagnostic] = []
    for d in diag_mod.parse(text or "", "", root):
        rel = relative(d.file, root, fs)
        if rel:
            found.setdefault(rel, []).append(
                dataclasses.replace(d, file=rel))
        else:
            loose.append(d)
    return found, loose


def _error_keys(found: dict, loose: Sequence[Diagnostic]) -> set:
    """What an error IS, for "the same error again": file and message, not
    the line (a repair moves lines) and not addresses or timings."""
    keys = set()
    for d in [x for ds in found.values() for x in ds] + list(loose):
        if d.code == "unparsed":
            continue
        msg = _NOISE.sub("#", (d.message or "").strip())[:120]
        keys.add((d.file or "", re.sub(r"\s+", " ", msg)))
    return keys


def unified(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}"))


@dataclass
class Run:
    """One run of the operator's command, judged."""
    text: str = ""
    #: not-started | error | clean | running | hung | needs-input
    kind: str = "error"
    found: dict = field(default_factory=dict)
    loose: list = field(default_factory=list)
    timed_out: bool = False

    def clean(self) -> bool:
        return self.kind in ("clean", "running")

    def said(self, timeout: float) -> str:
        return {
            "clean": "no error — it exited normally",
            "running": (f"no error — still running after {timeout:g} s "
                        f"(it is built to keep running)"),
            "hung": (f"no error shown, but it was still running after "
                     f"{timeout:g} s and nothing in it is built to keep "
                     f"running — it may hang"),
            "needs-input": ("it waits for keyboard input, which cannot be "
                            "given here"),
            "not-started": "the command could not be started",
            "error": "it fails",
        }.get(self.kind, self.kind)


def _script_of(argv: Sequence[str], root: str, fs: Any) -> str:
    """The project file the command runs, when one of its words is one."""
    for i, word in enumerate(argv[1:], 1):
        if argv[i - 1] == "-m":
            cand = word.replace(".", "/") + ".py"
            if relative(cand, root, fs):
                return cand
            continue
        rel = relative(word, root, fs)
        if rel and langs.for_extension(rel) is not None:
            return rel
    return ""


def judge(text: str, proc: Any, root: str, fs: Any, script: str,
          read: Any) -> Run:
    """What one run of the command showed."""
    found, loose = locate(text, root, fs)
    run = Run(text=text, found=found, loose=loose,
              timed_out=bool(getattr(proc, "timed_out", False)))
    if proc is None or (getattr(proc, "exit_code", 0) == -1
                        and "could not run" in
                        (getattr(proc, "stderr", "") or "")):
        run.kind = "not-started"
    elif _EOF.search(text or ""):
        run.kind = "needs-input"
    elif found or (getattr(proc, "exit_code", 1) != 0 and not run.timed_out):
        run.kind = "error"
    elif run.timed_out:
        errors = any(d.severity in ("error", "exception", "failure")
                     and d.code != "unparsed" for d in loose)
        if errors:
            run.kind = "error"
        else:
            run.kind = "running" if read.built_to_run(script) else "hung"
    else:
        run.kind = "clean"
    return run


# ---------------------------------------------------------------------------
# the session's side
# ---------------------------------------------------------------------------

def _read(session: Any, path: str) -> str:
    try:
        return session.host.fs.read(path)
    except Exception:                                    # noqa: BLE001
        return ""


def _emit(session: Any, kind: str, message: str, **data: Any) -> None:
    try:
        session.host.emit(kind, message, {"phase": "debug", **data})
    except Exception:                                    # noqa: BLE001
        pass


def _log(session: Any, fn: str, *args: Any, **kw: Any) -> None:
    """BUILD_LOG.txt — a log line must never stop a repair."""
    try:
        getattr(session.log, fn)(*args, **kw)
    except Exception:                                    # noqa: BLE001
        pass


def _journal(session: Any, **fields: Any) -> None:
    try:
        session.journal.log("debug", **fields)
    except Exception:                                    # noqa: BLE001
        pass


def capture(session: Any, command: str, timeout: float) -> Run:
    """Run the operator's command in the project folder, judged. Never
    raises: a command that cannot start is a Run that says so."""
    fs = session.host.fs
    root = fs.root()
    argv = split_command(command)
    if not argv:
        return Run(kind="not-started", text="the command is empty")
    try:
        proc = session.host.exec.run(argv, cwd=root, timeout=float(timeout))
    except Exception as exc:                             # noqa: BLE001
        return Run(kind="not-started",
                   text=f"the command could not be run: {exc}")
    out = proc.output if hasattr(proc, "output") else ""
    tail = (f"\n[still running after {timeout:g} s — stopped there]"
            if getattr(proc, "timed_out", False) else
            f"\n[exit code {proc.exit_code}]")
    return judge((out or "") + tail, proc, root, fs,
                 _script_of(argv, root, fs), _Reader(session))


class _Reader:
    """What `judge` may ask of the project: a file's text, and whether the
    program is BUILT to keep running (a game loop, a GUI, a server) — the
    script itself or, since an entry point often only imports the game,
    any source file of the project (the runner's own markers)."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self, path: str) -> str:
        return _read(self.session, path)

    def built_to_run(self, script: str) -> bool:
        paths = ([script] if script else []) + [
            p for p in _listed(self.session)[:200] if p != script]
        for p in paths:
            lang = langs.id_for_path(p)
            if lang and runner.has_main_loop(self(p), lang):
                return True
        return False


def _listed(session: Any) -> list[str]:
    """The project's own source files: the venv, node_modules, build
    output and the engine's own folders left out BEFORE any cap."""
    fs = session.host.fs
    try:
        every = fs.list("*")
    except Exception:                                    # noqa: BLE001
        return []
    out = []
    for path in sorted(every):
        p = path.replace("\\", "/")
        parts = p.split("/")
        if any(part.startswith(_SKIP_DIRS) for part in parts[:-1]):
            continue
        if langs.for_extension(p) is None:
            continue
        out.append(p)
    return out


def _outline(session: Any, paths: Sequence[str]) -> str:
    rows = []
    store = getattr(getattr(session, "codemap", None), "store", None)
    for p in paths:
        names = []
        try:
            names = [s["name"] for s in store.symbols_in(p)][:12] \
                if store is not None else []
        except Exception:                                # noqa: BLE001
            names = []
        rows.append(f"- {p}" + (f": {', '.join(names)}" if names else ""))
    return "\n".join(rows)


def _likely_order(session: Any, every: Sequence[str], text: str
                  ) -> list[str]:
    """``every`` ordered most likely first by the codemap's ranking toward
    ``text``; files the index does not know keep their order, after."""
    codemap = getattr(session, "codemap", None)
    if codemap is None or not hasattr(codemap, "ranked"):
        return list(every)
    try:
        ranked = codemap.ranked(text)
    except Exception:                                    # noqa: BLE001
        return list(every)
    known = {p.replace("\\", "/"): i for i, p in enumerate(ranked)}
    return sorted(every, key=lambda p: (known.get(p.replace("\\", "/"),
                                                  len(known)), p))


def model_locate(session: Any, report: DebugReport, seen: str
                 ) -> tuple[list[str], str]:
    """One model call: which files most likely hold the fault, and why.
    Source files are offered before tests; a test is never picked when a
    source file was."""
    every = _listed(session)
    # MOST LIKELY FIRST (codemap/rank.py): toward the names and paths the
    # error, the description and the command mention, then by how much the
    # rest of the project uses each file. It was path order, so a project
    # past the cap lost its likeliest files to the alphabet.
    order = _likely_order(session, every,
                          " ".join((report.description, seen,
                                    report.command)))
    files = ([f for f in order if not runner.is_test_path(f)]
             + [f for f in order if runner.is_test_path(f)])[:LOCATE_FILES]
    if not files:
        return [], "the project has no source files to choose from"
    task = ["A program in this project misbehaves. Name the file(s) whose "
            "code must change to fix it — source files, not tests."]
    if report.description.strip():
        task.append(f"What the person running it saw, in their words: "
                    f"{report.description.strip()}")
    if seen.strip():
        task.append("What it printed (no line of it names a file of this "
                    "project):\n" + clip(seen, 2000))
    if report.command.strip():
        task.append(f"How they ran it: {report.command.strip()}")
    try:
        from . import personas
        prompt = session.prompts.build(
            personas.ENGINEER, "\n".join(task), architecture="", epoch=0,
            contract=LOCATE_CONTRACT,
            extra=["[THE PROJECT'S FILES, with what each defines]\n"
                   + _outline(session, files)])
        completion = session.host.llm.complete(
            prompt.messages(), temperature=0.1,
            max_tokens=min(int(session.config.max_tokens or 512), 512),
            cancel=session.cancel_token)
    except (Cancelled, BudgetExceeded):
        raise
    except Exception as exc:                             # noqa: BLE001
        return [], f"the locate call failed: {exc}"
    if getattr(completion, "finish_reason", "") in ("error", "cancelled"):
        return [], (f"the locate call ended "
                    f"({completion.finish_reason}): "
                    f"{getattr(completion, 'error', '') or 'no reason'}")
    obj = find_json_object(completion.text or "", ("files",))
    wanted = obj.get("files") if isinstance(obj, dict) else None
    known = {f.lower(): f for f in files}
    picked = []
    for name in wanted if isinstance(wanted, list) else []:
        real = known.get(str(name).replace("\\", "/").strip().lower())
        if real and real not in picked:
            picked.append(real)
    sources = [p for p in picked if not runner.is_test_path(p)]
    picked = sources or picked
    why = str(obj.get("why", "") if isinstance(obj, dict) else "").strip()
    if not picked:
        return [], ("the model named no file of this project"
                    + (f" (it said: {why})" if why else ""))
    return picked[:MAX_TARGETS], why


def _modules_under_test(session: Any, test: str) -> list[str]:
    """The project files a failing test file is testing: the module whose
    name it carries (`test_calc.py` → `calc.py`), then what it imports."""
    import ast as _ast
    fs = session.host.fs
    every = [f for f in _listed(session) if not runner.is_test_path(f)]
    stem = test.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    bare = re.sub(r"^test_|_test$|\.test$|\.spec$", "", stem)
    out = [f for f in every
           if f.rsplit("/", 1)[-1].rsplit(".", 1)[0] == bare]
    try:
        tree = _ast.parse(_read(session, test))
    except SyntaxError:
        tree = None
    names = []
    for node in _ast.walk(tree) if tree is not None else ():
        if isinstance(node, _ast.ImportFrom) and node.module \
                and not node.level:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, _ast.Import):
            names += [a.name for a in node.names]
    for name in names:
        base = name.replace(".", "/")
        for cand in (f"{base}.py", f"{base}/__init__.py"):
            if cand in every and cand not in out:
                out.append(cand)
    return out[:MAX_TARGETS]


def _task_for(session: Any, path: str, n: int, test: str = "") -> Task:
    """A debug repair task. ATOMIC: a repair that does not pass its checks
    puts the file back as it was — the operator's file is never left
    holding the model's last failed attempt."""
    lang = langs.id_for_path(path) or session.config.lang
    if not test:
        try:
            guess = session.planner.test_path_for(path)
            if guess and guess != path and session.host.fs.exists(guess):
                test = guess
        except Exception:                                # noqa: BLE001
            test = ""
    return Task(id=f"debug{n}-{os.path.basename(path)}", path=path,
                purpose=(f"The existing file {path}, as it stands. Repair "
                         f"it so the reported problem no longer happens; "
                         f"keep everything else as it is."),
                test_path=test, lang=lang, atomic=True)


def _request(report: DebugReport, seen: str) -> str:
    parts = ["DEBUG & REPAIR — the person running this program reports a "
             "problem."]
    if report.command.strip():
        parts.append(f"They ran: {report.command.strip()}")
    if seen.strip():
        parts.append("What it printed:\n" + clip(seen, 2500))
    if report.description.strip():
        parts.append(f"In their words: {report.description.strip()}")
    return "\n\n".join(parts)


def _seed(diags: Sequence[Diagnostic], path: str,
          report: DebugReport) -> tuple:
    seed = list(diags)
    if report.description.strip():
        seed.append(Diagnostic(
            file=path, line=0, severity="failure", tool="operator",
            message=("Reported by the person running it: "
                     + report.description.strip())))
    if not seed:
        seed.append(Diagnostic(
            file=path, line=0, severity="failure", tool="operator",
            message="The program fails when run; this file was named as "
                    "where the fault is."))
    return tuple(seed)


def _targets(session: Any, report: DebugReport, n: int, found: dict,
             seen: str) -> tuple[list, str, str]:
    """[(path, test file, its diagnostics)], how they were chosen, and —
    when the model had to guess — why ("" otherwise)."""
    fs = session.host.fs
    if n == 1 and report.files:
        named = [p.replace("\\", "/") for p in report.files]
        real = [p for p in named if fs.exists(p)]
        gone = [p for p in named if p not in real]
        if real:
            how = "named by the operator" + (
                f" (not in the project: {', '.join(gone)})" if gone else "")
            diags = [d for ds in found.values() for d in ds]
            return [(p, "", [d for d in diags if d.file == p] or diags)
                    for p in real], how, ""
        _log(session, "line", f"  named files not in the project: "
                              f"{', '.join(gone)} — locating instead")
    if found:
        out, bits = [], []
        for path, diags in found.items():
            if runner.is_test_path(path):
                mods = _modules_under_test(session, path)
                for m in mods:
                    if all(m != t for t, _x, _d in out):
                        out.append((m, path, diags))
                bits.append(f"{path}:{diags[0].line} (a test — so "
                            + (", ".join(mods) if mods else "nothing")
                            + ", the code it tests)")
            else:
                if all(path != t for t, _x, _d in out):
                    out.append((path, "", diags))
                bits.append(f"{path}:{diags[0].line}")
        if out:
            return out[:MAX_TARGETS], "the error names " + "; ".join(bits), ""
    targets, why = model_locate(session, report, seen)
    how = "the model, from the file list" + (f": {why}" if why else "")
    return [(t, "", []) for t in targets], how, (why or "a guess")


def _write_record(session: Any, report: DebugReport,
                  result: DebugResult) -> None:
    """`.cc_state/debug/<session>.md` and a line of history.jsonl."""
    fs = session.host.fs
    path = f"{DEBUG_DIR}/{session.id}.md"
    lines = [f"# Debug & Repair — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"**Result:** {result.verdict}", ""]
    if report.command.strip():
        lines += [f"**Command:** `{report.command.strip()}`", ""]
    if report.description.strip():
        lines += [f"**Reported:** {report.description.strip()}", ""]
    for r in result.rounds:
        lines += [f"## Round {r.n}", "", f"- **Located:** {r.how}"]
        for o in r.outcomes:
            lines.append(f"- `{o['path']}`: {o['label']} after "
                         f"{o['attempts']} attempt(s)"
                         + (f" — {o['why']}" if o.get("why") else ""))
        if r.check:
            lines.append(f"- **Run afterwards:** {r.check}")
        lines += [f"- **Verdict:** {r.verdict}", ""]
        if r.seen.strip():
            lines += ["What it worked from:", "", "```",
                      clip(r.seen, 2000), "```", ""]
    for p, d in result.diffs.items():
        lines += [f"## Changes to `{p}`", "", "```diff", d.rstrip(), "```",
                  ""]
    try:
        fs.write(path, "\n".join(lines))
        result.report_path = path
    except Exception:                                    # noqa: BLE001
        result.report_path = ""
    row = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session.id,
           "state": result.state, "verdict": result.verdict,
           "command": report.command.strip(),
           "description": report.description.strip()[:300],
           "error": clip(report.error_text, 300),
           "changed": sorted(result.diffs), "rounds": len(result.rounds),
           "report": result.report_path}
    try:
        from .journal import append_to
        append_to(fs, HISTORY, (json.dumps(row, ensure_ascii=False)
                                + "\n").encode("utf-8"))
    except Exception:                                    # noqa: BLE001
        pass


def history(fs: Any) -> list[dict]:
    """Every debug session in this project, oldest first."""
    try:
        text = fs.read(HISTORY)
    except Exception:                                    # noqa: BLE001
        return []
    out = []
    for line in (text or "").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def run(session: Any, report: DebugReport) -> DebugResult:
    """The whole of Debug & Repair for one report. See the module doc."""
    if report.empty():
        return DebugResult("nothing-to-repair",
                           "nothing was given — paste the error, type the "
                           "command, describe it, or name a file")
    problem = command_problem(report.command)
    timeout = float(report.run_timeout) if report.run_timeout \
        and report.run_timeout > 0 else 60.0
    timeout = max(5.0, timeout)
    result = DebugResult("no-change", "")
    if problem:
        result.state = "command-failed"
        result.verdict = f"the command cannot be run as typed: {problem}"
        _finish(session, report, result, {})
        return result
    _log(session, "rule", "DEBUG & REPAIR")
    if report.command.strip():
        _log(session, "line", f"  command    {report.command.strip()}")
    if report.description.strip():
        _log(session, "block", "in their words", report.description)
    if report.files:
        _log(session, "line", f"  named      {', '.join(report.files)}")
    _log(session, "block", "what they pasted", clip(report.error_text))

    originals: dict[str, str] = {}
    seen = report.error_text or ""
    #: True only when a run HERE showed the problem before any repair:
    #: then, and only then, a clean run afterwards confirms a fix
    confirmable = False
    unconfirmed = ""
    try:
        if report.command.strip():
            _emit(session, "status", f"running `{report.command.strip()}` "
                                     f"to see what it does")
            base = capture(session, report.command, timeout)
            _log(session, "block", "the run, before any repair",
                 clip(base.text))
            if base.kind == "not-started":
                result.state = "command-failed"
                result.verdict = (
                    f"the command could not be started here: "
                    f"{clip(base.text, 300)} — on Windows a script needs "
                    f"its program (`python main.py`, not `main.py`), and "
                    f"`npm` is `npm.cmd`")
                result.last_output = base.text
                _finish(session, report, result, originals)
                return result
            if base.kind == "error":
                confirmable = True
                if not seen.strip():
                    seen = base.text
            elif base.clean() and not (seen.strip()
                                       or report.description.strip()
                                       or report.files):
                result.state = "nothing-to-repair"
                result.verdict = (f"the command ran here and showed no "
                                  f"error ({base.said(timeout)}) — "
                                  f"describe what is wrong, or paste what "
                                  f"you saw")
                result.last_output = base.text
                _finish(session, report, result, originals)
                return result
            else:
                unconfirmed = (f"running it here cannot confirm the fix: "
                               f"before the repair, {base.said(timeout)}")
        else:
            unconfirmed = "no command was given, so nothing ran it"

        # -- not the code's fault (nonerror.py, 2026-10-22) ---------------
        # A relative-import / not-on-path / missing-package traceback is
        # answered with a sentence and a command, before any model call.
        # Four of 26 sessions on one project were these, and every one
        # tried to repair correct imports.
        verdict = nonerror.classify(seen, session.host.fs)
        if verdict is not None and not report.files:
            result.state = "nothing-to-repair"
            result.verdict = verdict.sentence
            result.last_output = seen
            _emit(session, "status", f"not a code error: {verdict.kind}",
                  nonerror=verdict.kind, command=verdict.command)
            _log(session, "event", "DEBUG not a code error",
                 f"{verdict.kind}: {verdict.command or verdict.package}")
            _journal(session, nonerror=verdict.kind,
                     command=verdict.command, package=verdict.package)
            _finish(session, report, result, originals)
            return result

        rounds = max(1, int(report.rounds or 1))
        for n in range(1, rounds + 1):
            if hasattr(session, "_check_budget"):
                session._check_budget()
            if session.cancel_token.is_set():
                raise Cancelled("stopped")
            rnd = DebugRound(n, seen=seen)
            found, loose = locate(seen, session.host.fs.root(),
                                  session.host.fs)
            rnd.located = [{"file": d.file, "line": d.line,
                            "message": d.message[:200]}
                           for ds in found.values() for d in ds]
            plan, how, guessed = _targets(session, report, n, found, seen)
            rnd.how = how
            if guessed and plan and report.confirm_targets:
                result.state = "located"
                result.targets = [p for p, _t, _d in plan]
                result.why = "" if guessed == "a guess" else guessed
                result.verdict = (f"the fault is most likely in "
                                  f"{', '.join(result.targets)} — repair "
                                  f"{'it' if len(plan) == 1 else 'them'}?")
                result.last_output = seen
                _log(session, "event", "DEBUG located (asking first)",
                     f"{', '.join(result.targets)} — {how}")
                break
            if not plan:
                rnd.verdict = f"nothing to repair: {how}"
                result.rounds.append(rnd)
                result.state = "not-located"
                result.verdict = ("the fault could not be placed in a file "
                                  f"— {how}; name the file and try again")
                break
            rnd.targets = [p for p, _t, _d in plan]
            _emit(session, "status",
                  f"debug round {n}: repairing {', '.join(rnd.targets)} — "
                  f"{how}", round=n, targets=rnd.targets)
            _log(session, "event", f"DEBUG round {n}",
                 f"repairing {', '.join(rnd.targets)} — {how}")
            request = _request(report, seen)
            for i, (path, test, diags) in enumerate(plan):
                before = _read(session, path)
                originals.setdefault(path, before)
                task = _task_for(session, path, n * 10 + i, test)
                outcome = session.loop.run_task(
                    task, request=request, seed=_seed(diags, path, report),
                    baseline=before)
                session.outcomes.append(outcome)
                if _read(session, path) != before:
                    rnd.changed.append(path)
                why = getattr(outcome, "stopped_because", "") or ""
                rnd.outcomes.append({"path": path, "label": outcome.label,
                                     "ok": bool(outcome.ok),
                                     "attempts": len(outcome.attempts),
                                     "why": why})
                _log(session, "event", f"DEBUG {outcome.label} {path}",
                     f"{len(outcome.attempts)} attempt(s)"
                     + (f" — {why}" if why else ""))
            if not rnd.changed:
                rnd.verdict = ("nothing was changed — no repair passed its "
                               "checks, and a file whose repair failed is "
                               "put back as it was")
                result.rounds.append(rnd)
                if any(r.changed for r in result.rounds):
                    last = next((r for r in reversed(result.rounds)
                                 if r.check), None)
                    result.state = "still-failing"
                    result.verdict = (
                        f"still failing — round {n} could not change "
                        f"anything more"
                        + (f"; after round {last.n}: {last.check}"
                           if last else ""))
                else:
                    result.state, result.verdict = "no-change", rnd.verdict
                break
            if not confirmable:
                rnd.verdict = ("repaired and its own checks pass — "
                               + unconfirmed + "; run it yourself to "
                               "confirm the problem is gone")
                result.rounds.append(rnd)
                result.state, result.verdict = "repaired", rnd.verdict
                break
            if session.cancel_token.is_set():
                raise Cancelled("stopped")
            _emit(session, "status", f"running `{report.command.strip()}` "
                                     f"again to confirm", round=n)
            again = capture(session, report.command, timeout)
            _log(session, "block", f"the run after round {n}",
                 clip(again.text))
            result.last_output = again.text
            rnd.check = again.said(timeout)
            if again.clean():
                rnd.verdict = "fixed — the command now runs without the error"
                result.rounds.append(rnd)
                result.state, result.verdict = "fixed", rnd.verdict
                break
            if again.kind in ("needs-input", "hung", "not-started"):
                rnd.verdict = (f"the error is gone from what it printed, but "
                               f"{rnd.check} — run it yourself to confirm")
                result.rounds.append(rnd)
                result.state, result.verdict = "repaired", rnd.verdict
                break
            before_keys = _error_keys(found, loose)
            after_keys = _error_keys(again.found, again.loose)
            same = bool(before_keys) and before_keys == after_keys
            rnd.check = ("the same error" if same else
                         "a different error: " + (diag_mod.summarise(
                             [d for ds in again.found.values() for d in ds]
                             + list(again.loose)) or "see the log")[:200])
            rnd.verdict = ("still failing the same way" if same else
                           "progress — a different error now")
            result.rounds.append(rnd)
            result.state = "still-failing"
            result.verdict = (f"still failing after {n} round"
                              f"{'' if n == 1 else 's'}: {rnd.check}")
            seen = again.text
    except (Cancelled, BudgetExceeded) as exc:
        result.state = "stopped"
        result.verdict = (f"stopped ({exc}) — files already repaired and "
                          f"checked are kept")
    _finish(session, report, result, originals)
    return result


def _finish(session: Any, report: DebugReport, result: DebugResult,
            originals: dict) -> None:
    """Diffs from the files as they were, then the record — on EVERY way
    out, so a stop or a question in a later round never loses what an
    earlier round already changed on disk."""
    for path, before in originals.items():
        after = _read(session, path)
        if after != before:
            result.diffs[path] = unified(path, before, after)
    if not result.verdict:
        result.verdict = "nothing was changed"
    if result.diffs and result.state == "located":
        result.verdict += (" (an earlier round already changed "
                           + ", ".join(sorted(result.diffs)) + ")")
    _write_record(session, report, result)
    _journal(session, state=result.state, verdict=result.verdict,
             rounds=len(result.rounds), changed=sorted(result.diffs),
             targets=result.targets, report=result.report_path)
    _log(session, "rule", "DEBUG & REPAIR — RESULT")
    _log(session, "line", f"  {result.verdict}")
    for p in sorted(result.diffs):
        _log(session, "block", f"changes to {p}", result.diffs[p])
    if result.report_path:
        _log(session, "line", f"  report     {result.report_path}")
    _emit(session, "status", f"debug & repair: {result.verdict}",
          state=result.state, report=result.report_path,
          changed=sorted(result.diffs))
