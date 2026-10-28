# SPDX-License-Identifier: Apache-2.0
"""A type checker's certain errors, in the build loop's verdict.

WHY THIS EXISTS, 2026-10-27
---------------------------
The errors a small model makes most are CROSS-FILE: a tuple read as an
object (`'tuple' object has no attribute 'x'`), a keyword the constructor
does not take, one argument too many, a name that lives in another module.
Every one of them is what a type checker reports at write time, with the
file and the line, BEFORE anything runs — and whether or not a test ever
reaches the line, which is the lint gate's own argument (an undefined name
the tests happen not to reach is still an undefined name). The Evaluate &
Bug Hunt tab ran pyright or mypy (bughunt.py); the build loop never did,
so a build found these the slow way, by a traceback, one per attempt.

WHAT IS GATED
-------------
Only the codes that are CERTAIN bugs — the same discipline as the lint
gate's F821 (`LoopConfig.lint_gate`): an attribute that does not exist, a
call with the wrong arguments, an undefined name. Not `arg-type`, not
`union-attr`, not a missing stub for pygame: a checker reading a small
model's half-annotated code is right about those often, not always, and a
gate that is sometimes wrong teaches the loop to repair code that works.

WHICH CHECKER
-------------
`basedpyright`, `pyright` or `mypy`, whichever the host's ExecPort finds
first, each read in its own format. The absence of all three is a note
once per session, with what it costs — and, because this engine runs on
disconnected machines, which one installs offline: mypy is pure Python and
basedpyright ships its own Node in the wheel; the `pyright` pip package
downloads Node on first run and is the wrong choice for an air-gapped
install. bughunt.py reads through here too, so the Evaluate tab and the
loop agree about what a checker said.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from . import runner
from .types import Diagnostic

#: Preference order. The pyright family is read as JSON and is fast; mypy
#: is pure Python and installs anywhere.
#: basedpyright ships its own Node in the wheel; mypy is pure Python; the
#: `pyright` pip package downloads Node into a user folder on first run
#: — last, for a tool that must leave the user profile alone.
TOOLS = ("basedpyright", "mypy", "pyright")

#: The codes a finding must carry to fail an attempt. mypy's error codes
#: and pyright's rule names, side by side; `LoopConfig.type_gate`.
GATE_CODES: tuple[str, ...] = (
    # mypy
    "attr-defined", "call-arg", "name-defined",
    # pyright / basedpyright
    "reportAttributeAccessIssue", "reportCallIssue", "reportUndefinedVariable",
)

#: mypy's arguments: check the bodies of unannotated functions too (a
#: small model's code is half-annotated and the gate is about bodies),
#: never report a library without stubs, follow the project's own imports
#: silently (their types are used; their errors are their own task's).
MYPY_ARGS = ("--no-error-summary", "--show-error-codes",
             "--show-column-numbers", "--ignore-missing-imports",
             "--follow-imports=silent", "--check-untyped-defs",
             "--no-color-output", "--no-pretty")
MYPY_CACHE = ".cc_state/.mypy_cache"

#: What the absence costs, said once per session (C7).
ABSENCE_NOTE = ("no type checker installed (mypy, basedpyright or pyright) "
                "— a wrong attribute name, a call with the wrong arguments "
                "or an undefined name in a line no test reaches will only "
                "show when a run reaches it. `pip install mypy` in the "
                "project's environment adds the check; mypy and basedpyright "
                "install offline, the pyright package does not")

_MYPY_LINE = re.compile(
    r"^(?P<file>[^\n:]+?):(?P<line>\d+)(?::(?P<col>\d+))?: "
    r"(?P<sev>error|warning|note): (?P<msg>.*?)(?:\s+\[(?P<code>[\w-]+)\])?$",
    re.M)

_PYRIGHT_SEVERITY = {"error": "error", "warning": "warning",
                     "information": "note"}


@dataclass(frozen=True)
class CheckResult:
    """What one run of the checker said."""
    tool: str = ""                       # "" = none ran
    diagnostics: tuple[Diagnostic, ...] = ()
    note: str = ""                       # why it did not run, or ""
    ok: bool = False                     # the tool ran and answered
    timed_out: bool = False              # it ran out of time (not retried)

    @property
    def ran(self) -> bool:
        return self.ok


def which(ex: Any) -> tuple[str, str]:
    """(tool name, path) of the first checker the host can run, or ("", "")."""
    for name in TOOLS:
        try:
            found = ex.which(name)
        except Exception:                                # noqa: BLE001
            found = None
        if found:
            return name, str(found)
    return "", ""


def check(ex: Any, fs: Any, files: Sequence[str], *, root: str = "",
          timeout: float = 120.0) -> CheckResult:
    """Run the checker over ``files`` (project-relative) and read it.

    Never raises. A checker that is installed but cannot run (the
    project's interpreter lacks the module, the output is not what the
    tool prints) comes back as a note, not a finding.
    """
    paths = [str(f).replace("\\", "/") for f in files if str(f).strip()]
    if not paths:
        return CheckResult()
    root = root or _root_of(fs)
    name, tool = which(ex)
    if not name:
        return CheckResult(note=ABSENCE_NOTE)
    argv = _argv(name, tool, paths, root)
    try:
        proc = ex.run(argv, cwd=root, timeout=timeout,
                      env=runner.scrubbed_env(root, root))
    except Exception as exc:                             # noqa: BLE001
        return CheckResult(tool=name, note=f"{name} could not be run: "
                                           f"{_plain(exc)}")
    output = (getattr(proc, "stdout", "") or "")
    err = (getattr(proc, "stderr", "") or "")
    code = int(getattr(proc, "exit_code", 0) or 0)
    if getattr(proc, "timed_out", False):
        return CheckResult(tool=name, note=f"{name} did not finish within "
                                           f"{timeout:.0f}s", timed_out=True)
    # THE EXIT CODE IS READ (the 2026-10-27 review): a checker that could
    # not start (-1 from the ExecPort), a mypy fatal (2 with no finding
    # lines), a pyright configuration or internal error (2, 3, 4) used to
    # pass as "ran, found nothing" — a clean bill nobody issued
    if code < 0:
        tail = (err or output).strip().splitlines()[-1:] or ["no output"]
        return CheckResult(tool=name, note=f"{name} could not be run "
                                           f"({tail[0][:160]})")
    if "No module named" in err and name == "mypy":
        return CheckResult(tool=name, note=(
            "mypy is installed for the engine but not in the project's "
            "environment, where the check has to run — `pip install "
            "mypy` there adds it"))
    if name == "mypy":
        diags = _read_mypy(output, root, paths)
        if code == 2 and not any(d.line for d in diags):
            tail = (output or err).strip().splitlines()[-1:] or ["no output"]
            return CheckResult(tool=name, note=f"mypy could not check the "
                                               f"file ({tail[0][:160]})")
        return CheckResult(name, tuple(_dynamic_attrs_dropped(
            diags, fs, root, paths)), ok=True)
    try:
        data = json.loads(output or "{}")
    except ValueError:
        tail = (err or output).strip().splitlines()[-1:] or ["no output"]
        return CheckResult(tool=name, note=f"{name} did not return JSON "
                                           f"({tail[0][:120]})")
    if code in (2, 3, 4) and not (data.get("generalDiagnostics") or []):
        # pyright's own fatal, configuration or argument errors
        tail = (err or output).strip().splitlines()[-1:] or ["no output"]
        return CheckResult(tool=name, note=f"{name} could not check the "
                                           f"file (exit {code}: "
                                           f"{tail[0][:160]})")
    return CheckResult(name, tuple(_dynamic_attrs_dropped(
        _read_pyright(data, root, paths, name), fs, root, paths)), ok=True)


_ATTR_IN_MESSAGE = re.compile(
    r'has no attribute "(?P<attr>\w+)"|attribute "(?P<attr2>\w+)"')
_CANNOT_ASSIGN = re.compile(r"^Cannot assign to attribute")


def _dynamic_attrs_dropped(diags: Sequence[Diagnostic], fs: Any, root: str,
                           paths: Sequence[str]) -> list[Diagnostic]:
    """An attribute the file ASSIGNS somewhere (`s.score = 5`, then
    `s.score`) is legal Python whatever the class declares; a checker
    calls both the write and the later reads an error, and the gate
    would fail working code and send the repair after a class in another
    file (the 2026-10-27 review). Such findings are dropped — only an
    attribute the file never assigns stays a certain error."""
    assigned: dict[str, set[str]] = {}
    for p in paths:
        try:
            text = fs.read(p)
        except Exception:                                # noqa: BLE001
            text = ""
        assigned[p] = set(re.findall(
            r"\.\s*(\w+)\s*(?:=(?!=)|\+=|-=|\*=|/=|//=|%=|\|=|&=|\^=|"
            r"<<=|>>=|\*\*=)", text or ""))
    out: list[Diagnostic] = []
    for d in diags:
        if d.code in ("attr-defined", "reportAttributeAccessIssue"):
            if _CANNOT_ASSIGN.match(d.message or ""):
                continue
            m = _ATTR_IN_MESSAGE.search(d.message or "")
            attr = (m.group("attr") or m.group("attr2")) if m else ""
            names = set()
            for p, found in assigned.items():
                if _same_file(d.file, p):
                    names = found
                    break
            if attr and attr in names:
                continue
        out.append(d)
    return out


def gate(diags: Sequence[Diagnostic], codes: Sequence[str] = GATE_CODES,
         path: str = "") -> list[Diagnostic]:
    """The findings that fail an attempt: in ``path`` (when given), with a
    gated code, as located errors the model can act on."""
    wanted = set(codes or ())
    out: list[Diagnostic] = []
    for d in diags:
        if d.code not in wanted or not d.line:
            continue
        if path and not _same_file(d.file, path):
            continue
        out.append(Diagnostic(
            file=path or d.file, line=d.line, col=d.col, severity="error",
            code=d.code, tool=d.tool,
            message=f"{d.message} (found by the type checker in the file "
                    f"as written: a certain error at runtime, whether or "
                    f"not a test reaches it)"))
    return out


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------

def _argv(name: str, tool: str, paths: Sequence[str],
          root: str) -> list[str]:
    full = [runner._join(root, p) for p in paths]
    if name == "mypy":
        return [tool, *MYPY_ARGS,
                f"--cache-dir={runner._join(root, MYPY_CACHE)}", *full]
    return [tool, "--outputjson", *full]


def _read_mypy(output: str, root: str,
               paths: Sequence[str]) -> list[Diagnostic]:
    out = []
    for m in _MYPY_LINE.finditer(output or ""):
        sev = {"error": "error", "warning": "warning"}.get(
            m.group("sev"), "note")
        file = _rel(m.group("file"), root)
        out.append(Diagnostic(
            file=file, line=int(m.group("line")),
            col=int(m.group("col")) if m.group("col") else None,
            severity=sev, message=m.group("msg").strip(),
            code=m.group("code") or None, tool="typecheck:mypy"))
    return out


def _read_pyright(data: dict, root: str, paths: Sequence[str],
                  name: str) -> list[Diagnostic]:
    out = []
    for d in (data.get("generalDiagnostics") or []):
        sev = _PYRIGHT_SEVERITY.get(str(d.get("severity")), "warning")
        start = (d.get("range") or {}).get("start") or {}
        line = int(start.get("line", -1)) + 1
        col = int(start.get("character", -1)) + 1
        out.append(Diagnostic(
            file=_rel(str(d.get("file", "")), root), line=max(line, 0),
            col=col if col > 0 else None, severity=sev,
            message=str(d.get("message", "")).strip().split("\n")[0],
            code=str(d.get("rule") or "") or None,
            tool=f"typecheck:{name}"))
    return out


def _rel(path: str, root: str) -> str:
    p = path.replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/") + "/"
    return p[len(r):] if r != "/" and p.startswith(r) else p


def _same_file(a: str, b: str) -> bool:
    x = (a or "").replace("\\", "/").lower()
    y = (b or "").replace("\\", "/").lower()
    return bool(x and y) and (x == y or x.endswith("/" + y)
                              or y.endswith("/" + x))


def _root_of(fs: Any) -> str:
    try:
        return str(fs.root())
    except Exception:                                    # noqa: BLE001
        return ""


def _plain(exc: BaseException) -> str:
    text = str(exc).strip()
    return text or type(exc).__name__
