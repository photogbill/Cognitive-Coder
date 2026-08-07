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
  Python                      traceback frames + the final exception line
  Node / JS                   stack frames + the thrown error
  Go                          file:line:col: message
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
    r"^(?P<file>\.{0,2}[\w./\\-]+\.go):(?P<line>\d+):(?:(?P<col>\d+):)?\s+"
    r"(?P<msg>[^\n]+)$", re.M)

_UNITTEST = re.compile(r"^(?P<sev>FAIL|ERROR):\s*(?P<msg>.+)$", re.M)
_ASSERT = re.compile(r"^(?P<exc>AssertionError)(?::\s*(?P<msg>.*))?$", re.M)
# pytest's short test summary: `FAILED tests/test_x.py::test_y - AssertionError: …`
_PYTEST_SUMMARY = re.compile(
    r"^(?P<sev>FAILED|ERROR)\s+(?P<file>[^\s:]+?)::(?P<test>[\w\[\]/.-]+)"
    r"(?:\s+-\s+(?P<msg>.+))?$", re.M)
# pytest's assertion location line: `E       assert 3 == 4`
_PYTEST_E = re.compile(r"^E\s{2,}(?P<msg>.+)$", re.M)

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

def _parse_gcc(text: str) -> list[Diagnostic]:
    out = []
    for m in _GCC.finditer(text):
        sev = m.group("sev").replace("fatal error", "fatal")
        out.append(Diagnostic(
            message=m.group("msg").strip(), file=m.group("file").strip(),
            line=_int(m.group("line")), col=_int(m.group("col")) or None,
            severity=sev, tool="gcc"))
    return out


def _parse_msvc(text: str) -> list[Diagnostic]:
    return [Diagnostic(
        message=m.group("msg").strip(), file=m.group("file").strip(),
        line=_int(m.group("line")), col=_int(m.group("col")) or None,
        severity=m.group("sev").replace("fatal error", "fatal"),
        code=m.group("code") or None,
        tool="tsc" if (m.group("code") or "").startswith("TS") else "msvc")
        for m in _MSVC.finditer(text)]


def _parse_rustc(text: str) -> list[Diagnostic]:
    """rustc puts the message and the location on different lines.

    The location FOLLOWS its message, so heads and locations are paired in
    order and only within the span of one head. Any other pairing attaches the
    wrong file to the wrong error.
    """
    heads = list(_RUSTC_HEAD.finditer(text))
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


def _parse_python(text: str) -> list[Diagnostic]:
    """The DEEPEST frame plus the exception. Not the whole traceback.

    The deepest frame is where it broke; the intermediate frames are how it
    got there, which a model rarely needs and always gets distracted by. In a
    Python traceback the deepest frame is the LAST one — the opposite of a
    JavaScript stack, and getting it backwards points the model at the entry
    point instead of the fault.
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
        last = frames[-1]
        file = last.group("file")
        line = _int(last.group("line"))
        if last.group("fn"):
            code = f"in {last.group('fn')}"
    return [Diagnostic(message=msg, file=file, line=line,
                       severity="exception", code=code, tool="python")]


def _parse_node(text: str) -> list[Diagnostic]:
    excs = list(_NODE_EXC.finditer(text))
    frames = list(_NODE_FRAME.finditer(text))
    if not excs and not frames:
        return []
    msg = (f"{excs[-1].group('exc')}: {excs[-1].group('msg') or ''}"
           .strip(": ") if excs else "failed")
    file = ""
    line = col = 0
    if frames:
        # The FIRST frame in a JS stack is the deepest — opposite of Python.
        file = frames[0].group("file")
        line = _int(frames[0].group("line"))
        col = _int(frames[0].group("col"))
    return [Diagnostic(message=msg, file=file, line=line, col=col or None,
                       severity="exception", tool="node")]


def _parse_tests(text: str) -> list[Diagnostic]:
    out = [Diagnostic(message=m.group("msg").strip(), severity="failure",
                      code="test", tool="unittest")
           for m in _UNITTEST.finditer(text)]
    for m in _PYTEST_SUMMARY.finditer(text):
        out.append(Diagnostic(
            message=(m.group("msg") or f"{m.group('test')} failed").strip(),
            file=m.group("file"), severity="failure",
            code=m.group("test"), tool="pytest"))
    for m in _ASSERT.finditer(text):
        out.append(Diagnostic(
            message=f"AssertionError: {m.group('msg') or ''}".strip(": "),
            severity="failure", code="assert", tool="unittest"))
    for m in _PYTEST_E.finditer(text):
        out.append(Diagnostic(message=m.group("msg").strip(),
                              severity="failure", code="assert",
                              tool="pytest"))
    return out


def _parse_javac(text: str) -> list[Diagnostic]:
    """javac omits the column, so the gcc pattern misses these."""
    return [Diagnostic(message=m.group("msg").strip(),
                       file=m.group("file").strip(),
                       line=_int(m.group("line")), severity=m.group("sev"),
                       tool="javac")
            for m in _JAVAC.finditer(text)]


def _parse_go(text: str) -> list[Diagnostic]:
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


def _parse_cppcheck(text: str) -> list[Diagnostic]:
    """cppcheck's severities (style, performance, portability) are its own."""
    return [Diagnostic(message=m.group("msg").strip(),
                       file=m.group("file").strip(),
                       line=_int(m.group("line")),
                       col=_int(m.group("col")) or None,
                       severity=m.group("sev"), code=m.group("code") or None,
                       tool="cppcheck")
            for m in _CPPCHECK.finditer(text)]


def _parse_godot(text: str) -> list[Diagnostic]:
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
    "python": (_parse_python, _parse_tests, _parse_gcc),
    "javascript": (_parse_node, _parse_tests),
    "typescript": (_parse_msvc, _parse_gcc, _parse_node),
    "csharp": (_parse_msvc, _parse_gcc),
    "zig": (_parse_gcc,),
    "bash": (_parse_gcc,),
    "gdscript": (_parse_godot,),
    "ruby": (_parse_gcc, _parse_tests),
    "lua": (_parse_gcc,),
}

_ALWAYS = (_parse_gcc, _parse_python, _parse_node, _parse_tests,
           _parse_godot)


def parse(text: str, lang_id: str = "") -> list[Diagnostic]:
    """Every diagnostic found, errors first, deduplicated.

    Parsers are tried in an order suited to the language but ALL the common
    ones run: a Python script that shells out to a compiler produces both
    kinds of output, and a loop that only understood one of them would fix
    half the problem and report the rest as mysterious.
    """
    if not text or not text.strip():
        return []

    tried = list(_FAMILIES.get((lang_id or "").lower(),
                               (_parse_gcc, _parse_python, _parse_node,
                                _parse_msvc, _parse_rustc, _parse_tests,
                                _parse_godot)))
    for extra in _ALWAYS:
        if extra not in tried:
            tried.append(extra)

    found: list[Diagnostic] = []
    seen: set = set()
    for fn in tried:
        for d in fn(text):
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

    found.sort(key=lambda d: (d.rank, d.file, d.line))
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
    diags = attach_source(parse(text, lang_id), fs, sources)
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
