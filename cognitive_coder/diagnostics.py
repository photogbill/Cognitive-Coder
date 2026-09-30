# SPDX-License-Identifier: Apache-2.0
"""Compiler and runtime output → structured errors a small model can act on.

**This is the highest-value module in the engine**, and it is worth saying why
rather than assuming it.

A small model plus a 200-line build log is a bad combination. The log is
mostly noise — include paths, link lines, repeated template expansions — the
useful part is three lines somewhere in the middle, and a small model's
attention lands on whatever is longest rather than whatever is wrong. Hand the
same model *three parsed errors with the offending source quoted* and it fixes
them, because the task has become small and concrete. This module is the
difference between a toy and a tool.

So: parse everything into `types.Diagnostic`, sort real errors ahead of
warnings, quote the source around each one, and cap what goes back.

WHAT'S PARSED
  gcc / clang / cc            file:line:col: error: message
  MSVC (cl)                   file(line,col): error C2065: message
  rustc                       error[E0425]: message  →  --> file:line:col
  javac                       file:line: error: message
  Python                      traceback frames + the final exception line,
                              located in the deepest PROJECT frame
  Node / JS                   the `file:line` header or the first non-
                              `node:` frame + the thrown error; `node
                              --test` TAP failures
  Ruby                        `ruby -c` and `file:line:in 'fn': msg (Exc)`
  ruff / flake8               `file:line:col: CODE msg`, and ruff's
                              `CODE msg` / `--> file:line:col` form
  Go                          file:line:col: message (and `vet: …`)
  TypeScript                  file(line,col): error TS2345: message
  cppcheck / shellcheck       file:line:col: severity: message [id]
  unittest / pytest           FAIL:/ERROR: lines, assertion text, pytest's
                              `path:line: Error` short-summary form
  Godot / GDScript            SCRIPT ERROR: … at: fn (res://path.gd:LINE)
                              and `Parse Error: … at line N` (§6.1a)

Three behaviours that are not obvious and are load-bearing:

  * **rustc's message and location are on separate lines** and must be paired
    IN ORDER. Pairing them any other way attaches the wrong file to the wrong
    error, which is worse than having no location at all.
  * **Python's deepest frame is LAST; JavaScript's is FIRST.** Getting this
    backwards points the model at the entry point instead of the fault.
  * **Unrecognised output is NEVER dropped** (M29). It comes back as one
    diagnostic holding the last meaningful lines. Returning `[]` on a failed
    build is how a loop reports success on broken code — the single worst
    failure this module could have.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import re
from typing import Any

from .types import Diagnostic

# How many diagnostics to hand back by default. More than a handful and a
# small model starts fixing the last one it read instead of the first one that
# matters. Cascading languages get one (F7) — see `langs.Lang.feedback_cap`.
MAX_FEEDBACK = 3

# Source context around each error. Two lines either side is enough to see an
# unbalanced brace or a missing semicolon on the previous line.
CONTEXT_LINES = 2


def _int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# patterns
# ---------------------------------------------------------------------------

_GCC = re.compile(
    r"^(?P<file>[^\s:][^:]*?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<sev>error|warning|note|fatal error):\s*(?P<msg>.+)$", re.M)

_MSVC = re.compile(
    r"^(?P<file>[A-Za-z]?:?[^(\n]+)\((?P<line>\d+)(?:,(?P<col>\d+))?\)\s*:\s*"
    r"(?P<sev>fatal error|error|warning)\s+(?P<code>[A-Z]+\d+)\s*:\s*"
    r"(?P<msg>.+)$", re.M)

_RUSTC_HEAD = re.compile(
    r"^(?P<sev>error|warning)(?:\[(?P<code>E\d+)\])?:\s*(?P<msg>.+)$", re.M)
_RUSTC_LOC = re.compile(
    r"^\s*-->\s*(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+)", re.M)
# rustc's and cargo's closing summaries. They are shaped like errors and
# carry no location, and they USED to sort first: with Rust's feedback cap
# of one, the model was handed "error: aborting due to 1 previous error"
# and never saw E0425.
_RUSTC_TRAILER = re.compile(
    r"^(?:aborting due to\b|could not compile\b|\d+ warnings? emitted\b|"
    r"build failed\b)", re.I)

_JAVAC = re.compile(
    r"^(?P<file>[^\s:][^:]*?):(?P<line>\d+):\s*(?P<sev>error|warning):\s*"
    r"(?P<msg>.+)$", re.M)

_PY_FRAME = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<fn>\S+))?',
    re.M)
_PY_EXC = re.compile(
    r"^(?P<exc>[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt))"
    r"(?::\s*(?P<msg>.*))?$", re.M)

_NODE_FRAME = re.compile(
    r"^\s*at .*?\(?(?P<file>[^\s()]+):(?P<line>\d+):(?P<col>\d+)\)?", re.M)
# The `file:line` header Node prints above a SyntaxError (and above any
# uncaught error it can place). For a SyntaxError it is the ONLY user
# location: every `at` frame beneath it is `node:internal/…`, and taking
# the first frame pointed the model at Node's own module loader.
_NODE_HEADER = re.compile(
    r"^(?P<file>(?:file://)?(?:/|[A-Za-z]:[\\/])[^\n]*?|[^\s:][^\n:]*?"
    r"\.[cm]?[jt]sx?):(?P<line>\d+)[ \t]*$", re.M)
# `node --test` reports failures as TAP. Without this a failing JS test
# came back as the unparsed tail — `# duration_ms 44.66`.
_TAP_NOT_OK = re.compile(r"^(?P<indent>[ \t]*)not ok \d+ - (?P<name>.+?)"
                         r"[ \t]*$", re.M)
_TAP_LOCATION = re.compile(
    r"^[ \t]*location: '(?P<file>.+?):(?P<line>\d+):(?P<col>\d+)'", re.M)
_TAP_STACK = re.compile(
    r"\(?(?P<file>(?:file://)?(?:/|[A-Za-z]:[\\/])[^\s()']+?):"
    r"(?P<line>\d+):(?P<col>\d+)\)?")
_NODE_EXC = re.compile(
    r"^(?P<exc>[A-Z]\w*(?:Error))(?::\s*(?P<msg>.*))?$", re.M)

_CPPCHECK = re.compile(
    r"^(?P<file>[^\s:][^:]*?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<sev>error|warning|style|performance|portability|information):\s*"
    r"(?P<msg>.+?)(?:\s*\[(?P<code>[\w.]+)\])?$", re.M)

# Go prints `file:line:col: message` with NO severity word, so the gcc
# pattern — which requires one — silently misses every Go error. Its own
# pattern, deliberately anchored to a Go-shaped path so it does not swallow
# unrelated colon-separated lines from other toolchains.
_GO = re.compile(
    r"^(?:vet: )?(?P<file>\.{0,2}[\w./\\-]+\.go):(?P<line>\d+):"
    r"(?:(?P<col>\d+):)?\s+(?P<msg>[^\n]+)$", re.M)

_UNITTEST = re.compile(r"^(?P<sev>FAIL|ERROR):\s*(?P<msg>.+)$", re.M)
_ASSERT = re.compile(r"^(?P<exc>AssertionError)(?::\s*(?P<msg>.*))?$", re.M)
# pytest's short test summary: `FAILED tests/test_x.py::test_y - AssertionError: …`
_PYTEST_SUMMARY = re.compile(
    r"^(?P<sev>FAILED|ERROR)\s+(?P<file>[^\s:]+?)::(?P<test>[\w\[\]/.-]+)"
    r"(?:\s+-\s+(?P<msg>.+))?$", re.M)
# pytest's assertion location line: `E       assert 3 == 4`
_PYTEST_E = re.compile(r"^E\s{2,}(?P<msg>.+)$", re.M)
# ... and where it happened, per test section: `____ test_a ____` then
# `test_x.py:2: AssertionError`. The docstring always claimed this form was
# parsed; the summary line alone gave `line=0`.
_PYTEST_SECTION = re.compile(r"^_{3,} (?P<test>\S.*?) _{3,}$", re.M)
_PYTEST_WHERE = re.compile(
    r"^(?P<file>[^\s:][^:\n]*\.py):(?P<line>\d+): (?P<exc>[A-Za-z_][\w.]*)$",
    re.M)

# Ruby. The raise site is the FIRST line (like JavaScript); `from` lines
# follow with the callers. Ruby 3.4 quotes with '…', 3.3 with `…'.
_RUBY_RAISE = re.compile(
    r"^(?P<file>[^\s:][^:\n]*\.rb):(?P<line>\d+):in [`'](?P<fn>[^`'\n]+)'"
    r": (?P<msg>.+?)(?: \((?P<exc>[A-Z]\w*(?:::\w+)*)\))?$", re.M)
_RUBY_FROM = re.compile(
    r"^\s+from (?P<file>[^\s:][^:\n]*\.rb):(?P<line>\d+):in ", re.M)
_RUBY_LINE = re.compile(
    r"^(?:ruby: )?(?P<file>[^\s:][^:\n]*\.rb):(?P<line>\d+): "
    r"(?P<msg>(?!in [`']).+?)(?: \((?P<exc>[A-Z]\w*)\))?$", re.M)

# ruff (full and concise), flake8: a rule code where gcc has a severity.
_LINT = re.compile(
    r"^(?P<file>[^\s:][^:\n]*?\.\w+):(?P<line>\d+):(?P<col>\d+): "
    r"(?P<code>[A-Z]{1,3}\d{2,4}) (?:\[\*\] )?(?P<msg>.+)$", re.M)
_RUFF_FULL = re.compile(
    r"^(?P<code>[A-Z]{1,3}\d{2,4}) (?:\[\*\] )?(?P<msg>.+)\n[ \t]*--> "
    r"(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+)", re.M)
# Rules that are errors rather than style: undefined names and syntax.
_LINT_ERRORS = ("F821", "F822", "F823", "E999")

# Frames that are not the project's: the standard library, installed
# packages, virtualenvs, and runtimes' own internals.
_LIBRARY_PATH = re.compile(
    r"(?i)(?:^|/)(?:site-packages|dist-packages|node_modules|\.?venv|"
    r"lib/python\d[\d.]*|lib64/python\d[\d.]*)(?:/|$)|^<|^node:|"
    r"^internal/")

# Godot 4 (§6.1a). Without these the loop is blind on GDScript, which is the
# whole reason GDScript is first class rather than outline-only.
# `res://player.gd` contains a colon, so the file group must tolerate the
# scheme prefix explicitly — `[^:)]+` silently matches nothing here and the
# diagnostic arrives unlocated, which is the one thing §6.2 must not do.
_GODOT_PATH = r"(?:res://|user://)?[^:)\s]+"
_GODOT_SCRIPT = re.compile(
    r"^[ \t]*(?:SCRIPT ERROR|ERROR):[ \t]*(?P<msg>.+?)[ \t]*$\n"
    # `GDScript::reload` and `Node2D._ready` both appear here, so the
    # function group must allow `::`. Without it the location is dropped and
    # the diagnostic arrives unlocated — which for a parse error is the
    # difference between a fixable report and a shrug.
    r"[ \t]*at:[ \t]*(?:(?P<fn>[\w.:<>]+)[ \t]*)?"
    rf"\((?P<file>{_GODOT_PATH}):(?P<line>\d+)\)", re.M)
_GODOT_SCRIPT_INLINE = re.compile(
    r"^[ \t]*SCRIPT ERROR:[ \t]*(?P<msg>.+?)(?:[ \t]+at:[ \t]*.*?"
    rf"\((?P<file>{_GODOT_PATH}):(?P<line>\d+)\))?[ \t]*$", re.M)
_GODOT_PARSE = re.compile(
    r"^\s*(?:Parse Error|PARSE ERROR):\s*(?P<msg>.+?)"
    r"(?:\s*(?:at line|line)\s*(?P<line>\d+))?\s*$", re.M)
_GODOT_FILE_LINE = re.compile(
    r"(?:res://)?(?P<file>[\w./\\-]+\.gd):(?P<line>\d+)")
# GUT / gdUnit4 failures
_GUT_FAIL = re.compile(
    r"^\s*\[Failed\]:?\s*(?P<msg>.+)$|^\s*FAILED:\s*(?P<msg2>.+)$", re.M)


# ---------------------------------------------------------------------------
# per-family parsers
# ---------------------------------------------------------------------------

def _parse_gcc(text: str, root: str = "") -> list[Diagnostic]:
    out = []
    for m in _GCC.finditer(text):
        sev = m.group("sev").replace("fatal error", "fatal")
        out.append(Diagnostic(
            message=m.group("msg").strip(), file=m.group("file").strip(),
            line=_int(m.group("line")), col=_int(m.group("col")) or None,
            severity=sev, tool="gcc"))
    return out


def _parse_msvc(text: str, root: str = "") -> list[Diagnostic]:
    return [Diagnostic(
        message=m.group("msg").strip(), file=m.group("file").strip(),
        line=_int(m.group("line")), col=_int(m.group("col")) or None,
        severity=m.group("sev").replace("fatal error", "fatal"),
        code=m.group("code") or None,
        tool="tsc" if (m.group("code") or "").startswith("TS") else "msvc")
        for m in _MSVC.finditer(text)]


def _parse_rustc(text: str, root: str = "") -> list[Diagnostic]:
    """rustc puts the message and the location on different lines.

    The location FOLLOWS its message, so heads and locations are paired in
    order and only within the span of one head. Any other pairing attaches the
    wrong file to the wrong error.
    """
    heads = [h for h in _RUSTC_HEAD.finditer(text)
             if not _RUSTC_TRAILER.match(h.group("msg").strip())]
    locs = list(_RUSTC_LOC.finditer(text))
    out = []
    for i, h in enumerate(heads):
        file = ""
        line = col = 0
        nxt = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        for loc in locs:
            if h.end() <= loc.start() < nxt:
                file = loc.group("file").strip()
                line = _int(loc.group("line"))
                col = _int(loc.group("col"))
                break
        out.append(Diagnostic(
            message=h.group("msg").strip(), file=file, line=line,
            col=col or None, severity=h.group("sev"),
            code=h.group("code") or None, tool="rustc"))
    return out


def _in_project(path: str, root: str = "") -> bool:
    """Is this frame's file the project's own, rather than a library's?

    With ``root``, an absolute path must be under it; a relative path is
    taken as relative to it (the runner's cwd). Either way a path through
    site-packages, a virtualenv, `lib/pythonX.Y` or `node:` is a library's.
    Without ``root`` only that second test applies — which is why callers
    that know the root pass it.
    """
    p = (path or "").replace("\\", "/")
    if not p or _LIBRARY_PATH.search(p):
        return False
    r = (root or "").replace("\\", "/").rstrip("/")
    if not r or not (p.startswith("/") or re.match(r"^[A-Za-z]:/", p)):
        return True
    if re.match(r"^[A-Za-z]:", r):                     # Windows: no case
        return p.lower().startswith(r.lower() + "/")
    return p.startswith(r + "/")


def _file_path(raw: str) -> str:
    """`file:///C:/x.js` → `C:/x.js`; `file:///x.js` → `/x.js`."""
    text = (raw or "").strip()
    if text.startswith("file://"):
        text = text[len("file://"):]
        if re.match(r"^/[A-Za-z]:/", text):
            text = text[1:]
    return text


def _parse_python(text: str, root: str = "") -> list[Diagnostic]:
    """The deepest PROJECT frame plus the exception. Not the whole traceback.

    The deepest frame is where it broke; the intermediate frames are how it
    got there, which a model rarely needs and always gets distracted by. In a
    Python traceback the deepest frame is the LAST one — the opposite of a
    JavaScript stack, and getting it backwards points the model at the entry
    point instead of the fault.

    But "deepest" means the deepest frame the model can FIX. An exception
    raised inside the standard library or an installed package used to be
    located at `/usr/lib/python3.12/json/decoder.py:353` — a file the model
    cannot edit and the FileSystemPort cannot quote. The deepest frame in
    the project wins; only when there is none does the last frame stand.
    """
    frames = list(_PY_FRAME.finditer(text))
    excs = [m for m in _PY_EXC.finditer(text)
            if not m.group(0).startswith(" ")]
    if not frames and not excs:
        return []
    exc = excs[-1] if excs else None
    msg = (f"{exc.group('exc')}: {exc.group('msg') or ''}".strip(": ")
           if exc else "failed")
    file = ""
    line = 0
    code = None
    if frames:
        mine = [f for f in frames if _in_project(f.group("file"), root)]
        last = (mine or frames)[-1]
        file = last.group("file")
        line = _int(last.group("line"))
        if last.group("fn"):
            code = f"in {last.group('fn')}"
    return [Diagnostic(message=msg, file=file, line=line,
                       severity="exception", code=code, tool="python")]


def _parse_node(text: str, root: str = "") -> list[Diagnostic]:
    excs = list(_NODE_EXC.finditer(text))
    frames = list(_NODE_FRAME.finditer(text))
    if not excs and not frames:
        return []
    msg = (f"{excs[-1].group('exc')}: {excs[-1].group('msg') or ''}"
           .strip(": ") if excs else "failed")
    file = ""
    line = col = 0
    exc_at = excs[-1].start() if excs else len(text)
    header = [h for h in _NODE_HEADER.finditer(text, 0, exc_at)
              if not h.group("file").startswith("node:")]
    mine = [f for f in frames if _in_project(_file_path(f.group("file")),
                                             root)]
    if header:
        # Node's own `file:line` header — the one user location a
        # SyntaxError has.
        file = _file_path(header[-1].group("file"))
        line = _int(header[-1].group("line"))
    elif mine or frames:
        # The FIRST frame in a JS stack is the deepest — opposite of
        # Python — but never Node's own `node:internal/…`.
        first = (mine or frames)[0]
        file = _file_path(first.group("file"))
        line = _int(first.group("line"))
        col = _int(first.group("col"))
    return [Diagnostic(message=msg, file=file, line=line, col=col or None,
                       severity="exception", tool="node")]


def _parse_node_tap(text: str, root: str = "") -> list[Diagnostic]:
    """`node --test` failures: one per failing test, at the failing line.

    The location is the first project frame of the TAP `stack:` (the
    assertion), else `location:` (the `test(…)` call). A parent reporting
    only that its subtests failed is skipped: its children say why.
    """
    out: list[Diagnostic] = []
    for m in _TAP_NOT_OK.finditer(text):
        end = re.compile(r"^[ \t]*\.\.\.[ \t]*$", re.M).search(text, m.end())
        block = text[m.end():end.start() if end else len(text)]
        if "failureType: 'subtestsFailed'" in block:
            continue
        err = ""
        em = re.search(r"^([ \t]*)error: (?:\|-?\n(?P<body>(?:\1[ \t]+.*\n?"
                       r"|[ \t]*\n)+)|'(?P<one>.*)')", block, re.M)
        if em:
            lines = (em.group("body") or em.group("one") or "").splitlines()
            err = " ".join(ln.strip() for ln in lines if ln.strip())[:200]
        file, line, col = "", 0, 0
        stack = block[block.find("stack:"):] if "stack:" in block else ""
        for sm in _TAP_STACK.finditer(stack):
            path = _file_path(sm.group("file"))
            if _in_project(path, root):
                file, line = path, _int(sm.group("line"))
                col = _int(sm.group("col"))
                break
        if not file:
            lm = _TAP_LOCATION.search(block)
            if lm:
                file = _file_path(lm.group("file"))
                line, col = _int(lm.group("line")), _int(lm.group("col"))
        name = m.group("name").strip()
        out.append(Diagnostic(
            message=f"{name}: {err}" if err else f"{name} failed",
            file=file, line=line, col=col or None, severity="failure",
            code="test", tool="node-test"))
    return out


def _parse_ruby(text: str, root: str = "") -> list[Diagnostic]:
    """Ruby's raise line (first = deepest, like JS) and `ruby -c` errors."""
    out: list[Diagnostic] = []
    raise_at = -1
    rm = _RUBY_RAISE.search(text)
    if rm:
        raise_at = rm.start()
        exc = rm.group("exc")
        msg = rm.group("msg").strip()
        file, line = rm.group("file"), _int(rm.group("line"))
        if not _in_project(file, root):
            for fm in _RUBY_FROM.finditer(text, rm.end()):
                if _in_project(fm.group("file"), root):
                    file, line = fm.group("file"), _int(fm.group("line"))
                    break
        out.append(Diagnostic(
            message=f"{exc}: {msg}" if exc else msg, file=file, line=line,
            severity="exception", code=f"in {rm.group('fn')}", tool="ruby"))
    for m in _RUBY_LINE.finditer(text):
        if m.start() == raise_at:
            continue
        msg = m.group("msg").strip()
        sev = "warning" if msg.startswith("warning:") else "error"
        out.append(Diagnostic(
            message=msg.split(":", 1)[1].strip() if sev == "warning" else msg,
            file=m.group("file"), line=_int(m.group("line")), severity=sev,
            tool="ruby"))
    return out


def _parse_lint(text: str, root: str = "") -> list[Diagnostic]:
    """ruff and flake8 findings: a rule code in place of a severity word."""
    out: list[Diagnostic] = []
    for pattern in (_RUFF_FULL, _LINT):
        for m in pattern.finditer(text):
            code = m.group("code")
            out.append(Diagnostic(
                message=m.group("msg").strip(), file=m.group("file").strip(),
                line=_int(m.group("line")), col=_int(m.group("col")) or None,
                severity="error" if code in _LINT_ERRORS
                or code.startswith("E9") else "warning",
                code=code, tool="lint"))
    return out


def _pytest_where(text: str, root: str = "") -> dict[str, tuple[str, int]]:
    """{test name: (file, line)} from the long traceback's sections.

    The deepest location in the project wins, like a Python traceback: a
    test failing inside `src/calc.py` is located there, not at the call.
    """
    found: dict[str, tuple[str, int]] = {}
    sections = list(_PYTEST_SECTION.finditer(text))
    for i, sec in enumerate(sections):
        stop = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        wheres = list(_PYTEST_WHERE.finditer(text, sec.end(), stop))
        mine = [w for w in wheres if _in_project(w.group("file"), root)]
        if mine or wheres:
            w = (mine or wheres)[-1]
            found[sec.group("test").replace("::", ".")] = (
                w.group("file"), _int(w.group("line")))
    return found


def _parse_tests(text: str, root: str = "") -> list[Diagnostic]:
    out = [Diagnostic(message=m.group("msg").strip(), severity="failure",
                      code="test", tool="unittest")
           for m in _UNITTEST.finditer(text)]
    where = _pytest_where(text, root)
    for m in _PYTEST_SUMMARY.finditer(text):
        test = m.group("test")
        file, line = where.get(test.replace("::", "."),
                               (m.group("file"), 0))
        out.append(Diagnostic(
            message=(m.group("msg") or f"{test} failed").strip(),
            file=file, line=line, severity="failure",
            code=test, tool="pytest"))
    for m in _ASSERT.finditer(text):
        out.append(Diagnostic(
            message=f"AssertionError: {m.group('msg') or ''}".strip(": "),
            severity="failure", code="assert", tool="unittest"))
    for m in _PYTEST_E.finditer(text):
        out.append(Diagnostic(message=m.group("msg").strip(),
                              severity="failure", code="assert",
                              tool="pytest"))
    return out


def _parse_javac(text: str, root: str = "") -> list[Diagnostic]:
    """javac omits the column, so the gcc pattern misses these."""
    return [Diagnostic(message=m.group("msg").strip(),
                       file=m.group("file").strip(),
                       line=_int(m.group("line")), severity=m.group("sev"),
                       tool="javac")
            for m in _JAVAC.finditer(text)]


def _parse_go(text: str, root: str = "") -> list[Diagnostic]:
    """Go's compiler and vet output. No severity word, so no gcc match."""
    out = []
    for m in _GO.finditer(text):
        msg = m.group("msg").strip()
        if msg.lower().startswith(("error:", "warning:")):
            msg = msg.split(":", 1)[1].strip()
        out.append(Diagnostic(
            message=msg, file=m.group("file").strip(),
            line=_int(m.group("line")), col=_int(m.group("col")) or None,
            severity="error", tool="go"))
    return out


def _parse_cppcheck(text: str, root: str = "") -> list[Diagnostic]:
    """cppcheck's severities (style, performance, portability) are its own."""
    return [Diagnostic(message=m.group("msg").strip(),
                       file=m.group("file").strip(),
                       line=_int(m.group("line")),
                       col=_int(m.group("col")) or None,
                       severity=m.group("sev"), code=m.group("code") or None,
                       tool="cppcheck")
            for m in _CPPCHECK.finditer(text)]


def _parse_godot(text: str, root: str = "") -> list[Diagnostic]:
    """Godot's two error shapes, plus GUT/gdUnit4 failures (§6.1a).

    Godot prints the message and `at: func (res://path.gd:LINE)` on separate
    lines for runtime script errors, and a one-line `Parse Error:` with the
    line number tacked on the end for parse failures. Both are handled, and
    `res://` is stripped so the path can be opened by a FileSystemPort.
    """
    out: list[Diagnostic] = []
    seen_spans: list[tuple[int, int]] = []

    for m in _GODOT_SCRIPT.finditer(text):
        out.append(Diagnostic(
            message=m.group("msg").strip(),
            file=_strip_res(m.group("file")), line=_int(m.group("line")),
            severity="error",
            code=f"in {m.group('fn')}" if m.group("fn") else None,
            tool="godot"))
        seen_spans.append((m.start(), m.end()))

    for m in _GODOT_SCRIPT_INLINE.finditer(text):
        if any(a <= m.start() < b for a, b in seen_spans):
            continue
        out.append(Diagnostic(
            message=m.group("msg").strip(),
            file=_strip_res(m.group("file") or ""),
            line=_int(m.group("line")), severity="error", tool="godot"))

    for m in _GODOT_PARSE.finditer(text):
        line = _int(m.group("line"))
        file = ""
        # The filename often sits on a nearby line rather than in the match.
        near = text[max(0, m.start() - 300):m.end() + 300]
        fm = _GODOT_FILE_LINE.search(near)
        if fm:
            file = _strip_res(fm.group("file"))
            line = line or _int(fm.group("line"))
        out.append(Diagnostic(message=m.group("msg").strip(), file=file,
                              line=line, severity="error", code="parse",
                              tool="godot"))

    for m in _GUT_FAIL.finditer(text):
        msg = (m.group("msg") or m.group("msg2") or "").strip()
        if msg:
            out.append(Diagnostic(message=msg, severity="failure",
                                  code="test", tool="gut"))
    return out


def _strip_res(path: str) -> str:
    text = (path or "").strip()
    for prefix in ("res://", "user://"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


_FAMILIES: dict[str, tuple] = {
    "c": (_parse_gcc, _parse_msvc, _parse_cppcheck),
    "cpp": (_parse_gcc, _parse_msvc, _parse_cppcheck),
    "rust": (_parse_rustc, _parse_gcc),
    "java": (_parse_javac, _parse_gcc),
    "go": (_parse_go, _parse_gcc),
    "python": (_parse_python, _parse_tests, _parse_lint, _parse_gcc),
    "javascript": (_parse_node, _parse_node_tap, _parse_tests),
    "typescript": (_parse_msvc, _parse_gcc, _parse_node, _parse_node_tap),
    "csharp": (_parse_msvc, _parse_gcc),
    "zig": (_parse_gcc,),
    "bash": (_parse_gcc,),
    "gdscript": (_parse_godot,),
    "ruby": (_parse_ruby, _parse_gcc, _parse_tests),
    "lua": (_parse_gcc,),
}

_ALWAYS = (_parse_gcc, _parse_python, _parse_node, _parse_tests,
           _parse_godot)


def parse(text: str, lang_id: str = "", root: str = "") -> list[Diagnostic]:
    """Every diagnostic found, errors first, deduplicated.

    Parsers are tried in an order suited to the language but ALL the common
    ones run: a Python script that shells out to a compiler produces both
    kinds of output, and a loop that only understood one of them would fix
    half the problem and report the rest as mysterious.

    ``root`` is the project root as the tools saw it. With it, a traceback
    is located in the deepest frame UNDER the root; without it, in the
    deepest frame that is not recognisably a library's.
    """
    if not text or not text.strip():
        return []

    tried = list(_FAMILIES.get((lang_id or "").lower(),
                               (_parse_gcc, _parse_python, _parse_node,
                                _parse_msvc, _parse_rustc, _parse_tests,
                                _parse_lint, _parse_godot)))
    for extra in _ALWAYS:
        if extra not in tried:
            tried.append(extra)

    found: list[Diagnostic] = []
    seen: set = set()
    for fn in tried:
        for d in fn(text, root):
            if not (d.message or "").strip():
                continue
            if d.key() not in seen:
                seen.add(d.key())
                found.append(d)

    # Several parsers legitimately match the same exception line — Python's
    # `ZeroDivisionError` also looks like a Node error — and one of them will
    # have found the file while the other didn't. Keep the LOCATED one: a
    # diagnostic without a location is the same information, minus the part
    # that makes it fixable.
    located = {d.message[:80] for d in found if d.file}
    found = [d for d in found if d.file or d.message[:80] not in located]

    if not found:
        # Nothing matched — but the caller only asked because something
        # FAILED (M29). Returning [] here would report a clean build on a
        # broken one, which is the worst bug this module could have.
        tail = [ln for ln in text.strip().splitlines() if ln.strip()][-4:]
        if tail:
            found = [Diagnostic(message="\n".join(tail), severity="error",
                                code="unparsed", tool="raw")]

    # Unlocated diagnostics go LAST within their rank. An empty file name
    # sorts before every real one as text, so a summary line with no
    # location used to displace the error it summarised.
    found.sort(key=lambda d: (d.rank, not d.file, d.file, d.line))
    return found


def attach_source(diags: Sequence[Diagnostic], fs: Any = None,
                  sources: dict[str, str] | None = None
                  ) -> list[Diagnostic]:
    """Quote the offending lines. This is what makes the feedback usable.

    Reads through the `FileSystemPort` (C2) or from an explicit `sources`
    dict, so this works in tests with no filesystem at all. Diagnostics are
    frozen, so each one is rebuilt rather than mutated.
    """
    cache: dict[str, list[str]] = {}
    if sources:
        cache.update({k: v.splitlines() for k, v in sources.items()})

    out: list[Diagnostic] = []
    for d in diags:
        if not d.file or not d.line or d.source_excerpt:
            out.append(d)
            continue
        key = d.file.replace("\\", "/")
        if key not in cache:
            text = ""
            if fs is not None:
                for candidate in (key, key.split("/")[-1]):
                    try:
                        text = fs.read(candidate)
                        break
                    except Exception:                    # noqa: BLE001
                        continue
            cache[key] = text.splitlines() if text else []
        lines = cache[key]
        if not lines:
            out.append(d)
            continue
        lo = max(0, d.line - 1 - CONTEXT_LINES)
        hi = min(len(lines), d.line + CONTEXT_LINES)
        quoted = "\n".join(
            f"{'>>' if n == d.line - 1 else '  '} {n + 1:>4} | {lines[n]}"
            for n in range(lo, hi))
        out.append(dataclasses.replace(d, source_excerpt=quoted))
    return out


def feedback(diags: Sequence[Diagnostic], max_errors: int = MAX_FEEDBACK,
             extra_context: bool = False) -> str:
    """The string to hand back to the model. Small, specific, quoted.

    Deliberately capped. Handing back everything is the same mistake as
    handing back the raw log — the model's attention is the scarce resource,
    and the first error is usually the cause of the rest.

    ``extra_context`` is for cascading languages (F7): fewer errors, but more
    source around the one that matters, because in C++ the fortieth error is
    a consequence of the first and fixing it is wasted work.
    """
    if not diags:
        return ""
    real = [d for d in diags if d.is_error] or list(diags)
    shown = real[:max(1, max_errors)]
    parts = []
    for i, d in enumerate(shown, 1):
        block = f"{i}. {d.one_line()}"
        if d.source_excerpt:
            block += f"\n{d.source_excerpt}"
        parts.append(block)
    more = len(real) - len(shown)
    if more > 0:
        if extra_context:
            parts.append(
                f"({more} further error{'s' * (more != 1)} followed from "
                f"this one. In this language they usually cascade — fix the "
                f"one above and the rest generally disappear.)")
        else:
            parts.append(
                f"({more} more of the same kind — fix these first; they are "
                f"usually the cause.)")
    return "\n\n".join(parts)


def feedback_for(text: str, lang_id: str = "", fs: Any = None,
                 sources: dict[str, str] | None = None) -> str:
    """Raw toolchain output straight to model-ready feedback, in one call.

    The cap comes from the language (F7): one for cascading languages, three
    otherwise.
    """
    from . import langs  # local: avoids a cycle
    lang = langs.get(lang_id)
    cap = lang.feedback_cap if lang else MAX_FEEDBACK
    root = ""
    if fs is not None:
        try:
            root = str(fs.root())
        except Exception:                                # noqa: BLE001
            root = ""
    diags = attach_source(parse(text, lang_id, root=root), fs, sources)
    return feedback(diags, cap, extra_context=bool(lang and lang.cascades))


def summarise(diags: Sequence[Diagnostic]) -> str:
    """A one-line count for a status bar."""
    if not diags:
        return "no diagnostics"
    errs = sum(1 for d in diags if d.rank == 0)
    fails = sum(1 for d in diags if d.rank == 1)
    warns = sum(1 for d in diags if d.rank == 2)
    bits = []
    if errs:
        bits.append(f"{errs} error{'s' * (errs != 1)}")
    if fails:
        bits.append(f"{fails} failure{'s' * (fails != 1)}")
    if warns:
        bits.append(f"{warns} warning{'s' * (warns != 1)}")
    return " · ".join(bits) or f"{len(diags)} diagnostic(s)"


def first_error(diags: Sequence[Diagnostic]) -> Diagnostic | None:
    for d in diags:
        if d.rank == 0:
            return d
    return diags[0] if diags else None


def signature(diags: Sequence[Diagnostic]) -> tuple:
    """A stable identity for a set of diagnostics.

    Used by the stagnation detector (§6.9, M34) and by regression memory
    (F10). Sorted, because the ORDER two runs report the same two errors in
    is not a difference worth reacting to.
    """
    return tuple(sorted(f"{d.file}:{d.line}:{(d.message or '')[:60]}"
                        for d in diags))
