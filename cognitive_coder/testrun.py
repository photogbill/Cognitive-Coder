# SPDX-License-Identifier: Apache-2.0
"""The values at the failure: what the variables WERE when the test failed.

WHY THIS EXISTS, 2026-10-27
---------------------------
`enrich.py` puts the STATIC truth under an error — the real signature, the
fields a class has, where a name lives. The DYNAMIC truth was never shown:
a repair was told `AssertionError: 45.0 != 90.0` and the line it failed on,
and had to work out from the source alone what `x1, y1, x2, y2` had been
and which way the arithmetic went. On the 2026-10-16 Star Trek build ten of
sixteen failing tests were argued back and forth over exactly that. A
human reaches for the debugger first; this is the debugger's one useful
screen, printed under every failure.

HOW
---
The project's tests are run by a small stdlib-only script this module
writes into the project (`.cc_state/cc_testrun.py`, `SCRIPT`), in place of
`python -m unittest …` / `python -m pytest …`. It runs the SAME runner with
the SAME arguments and prints the same output — tracebacks, `Ran N tests`,
`FAILED (failures=1)`, the exit codes, pytest's `FAILED path::test` lines —
so every parser the engine has keeps working unchanged. Under each failure
it adds one block per frame that matters:

    [VALUES AT THE FAILURE] src/physics.py:118 in bearing_between
        x1 = 0
        y1 = 0
        x2 = 1
        y2 = 1
        heading = 45.0

The deepest frame in the PROJECT first (where it broke: its arguments,
then its other locals), then the test's own frame (the inputs and the
expected value), each capped at `MAX_NAMES` names of `MAX_REPR` characters.
Frames inside the interpreter, site-packages, a virtualenv or the script
itself are never shown. `parse_values` reads the blocks back out of the
run's output and `attach` puts each one on the Diagnostic it belongs to
(`Diagnostic.values`), which `diagnostics.feedback` prints under the
excerpt. No model call; by rule; exact.

Nothing here runs a test. The runner (`runner.run_tests`) decides when to
use the script; this module owns the script, the rewrite of the command
and the reading of what it printed.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass
import hashlib
import re
from typing import Any

from .types import Diagnostic

#: Where the script lives in the project: beside the engine's other state,
#: through the FileSystemPort, never under a user profile.
SCRIPT_REL = ".cc_state/cc_testrun.py"

MARK = "[VALUES AT THE FAILURE]"
MAX_NAMES = 12
MAX_REPR = 120

#: THE SCRIPT. Python 3.8+ stdlib only — it runs under the PROJECT's
#: interpreter, which is not the engine's and may have nothing installed
#: but pytest. Keep it that way: an import it cannot satisfy would fail
#: every test run of every project.
SCRIPT = r'''# written by Cognitive Coder (cognitive_coder/testrun.py) — not yours to
# edit; it is rewritten whenever the engine's copy changes. It runs this
# project's tests exactly as `python -m unittest` / `python -m pytest`
# would and prints, under each failure, the values the variables had.
import os
import sys
import types

MARK = "[VALUES AT THE FAILURE]"
MAX_NAMES = %(max_names)d
MAX_REPR = %(max_repr)d
ROOT = os.path.abspath(os.getcwd()).replace("\\", "/")
HERE = os.path.abspath(__file__).replace("\\", "/")
SKIP = ("/site-packages/", "/dist-packages/", "/.venv/", "/venv/",
        "/.cc_state/", "/__pycache__/", "/lib/python", "/node_modules/")
# Windows paths compare without regard to case (`d:` and `D:` are one
# drive); the normalised forms below are lower-cased there only
CASEFOLD = os.name == "nt"
if CASEFOLD:
    ROOT = ROOT.lower()
    HERE = HERE.lower()

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass


def _norm(path):
    try:
        out = os.path.abspath(path).replace("\\", "/")
    except Exception:
        out = str(path).replace("\\", "/")
    return out.lower() if CASEFOLD else out


def _is_project(filename):
    f = _norm(filename)
    if not f.startswith(ROOT + "/") or f == HERE:
        return False
    return not any(s in f for s in SKIP)


def _is_test(filename):
    name = _norm(filename).rsplit("/", 1)[-1]
    stem = name[:-3] if name.endswith(".py") else name
    return stem.startswith("test_") or stem.endswith("_test")


def _repr(value):
    try:
        text = repr(value)
    except Exception as exc:
        text = "<repr failed: %%s>" %% type(exc).__name__
    text = text.replace("\n", "\\n")
    if len(text) > MAX_REPR:
        text = text[:MAX_REPR - 1] + "…"
    return text


_SKIP_TYPES = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
               types.MethodType, type)


def _rows(frame):
    code = frame.f_code
    names = list(code.co_varnames[:code.co_argcount])
    for name in frame.f_locals:
        if name not in names:
            names.append(name)
    out = []
    for name in names:
        if name.startswith(("__", "@")) or name not in frame.f_locals:
            continue
        value = frame.f_locals[name]
        if isinstance(value, _SKIP_TYPES):
            continue
        if name == "self" and _is_testcase(value):
            continue
        out.append("    %%s = %%s" %% (name, _repr(value)))
        if len(out) >= MAX_NAMES:
            break
    return out


def _is_testcase(value):
    try:
        import unittest
        return isinstance(value, unittest.TestCase)
    except Exception:
        return False


def _rel(filename):
    f = _norm(filename)
    shown = str(filename).replace("\\", "/")
    if f.startswith(ROOT + "/"):
        return shown[len(ROOT) + 1:] if len(shown) > len(ROOT) else f
    return shown


def values_block(tb):
    """The block for one traceback, or ''."""
    frames = []
    while tb is not None:
        frames.append((tb.tb_frame, tb.tb_lineno))
        tb = tb.tb_next
    project = [(f, ln) for f, ln in frames if _is_project(f.f_code.co_filename)]
    if not project:
        return ""
    deepest = project[-1]
    chosen = [deepest]
    test_frame = next(((f, ln) for f, ln in project
                       if _is_test(f.f_code.co_filename)), None)
    if test_frame is not None and test_frame[0] is not deepest[0]:
        chosen.append(test_frame)
    blocks = []
    for frame, lineno in chosen:
        rows = _rows(frame)
        if not rows:
            continue
        head = "%%s %%s:%%d in %%s" %% (MARK, _rel(frame.f_code.co_filename),
                                   lineno, frame.f_code.co_name)
        blocks.append("\n".join([head] + rows))
    return "\n".join(blocks)


def _unittest(argv):
    import unittest

    class Result(unittest.TextTestResult):
        def _exc_info_to_string(self, err, test):
            text = super()._exc_info_to_string(err, test)
            try:
                block = values_block(err[2])
            except Exception:
                block = ""
            if block:
                text = text.rstrip("\n") + "\n" + block + "\n"
            return text

    class Runner(unittest.TextTestRunner):
        resultclass = Result

    unittest.main(module=None, argv=["python -m unittest"] + argv,
                  testRunner=Runner)


def _pytest(argv):
    try:
        import pytest
    except ImportError:
        sys.stderr.write("cc_testrun: pytest is not importable here\n")
        sys.exit(2)

    class Plugin:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            outcome = yield
            try:
                rep = outcome.get_result()
                if (rep.when == "call" and rep.failed
                        and call.excinfo is not None):
                    block = values_block(call.excinfo.value.__traceback__)
                    if block:
                        rep.sections.append((MARK, block))
            except Exception:
                pass

    sys.exit(pytest.main(argv, plugins=[Plugin()]))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("unittest", "pytest"):
        sys.stderr.write("usage: cc_testrun.py unittest|pytest ARGS...\n")
        sys.exit(2)
    # `python <script>` puts the SCRIPT's folder first on sys.path, where
    # `python -m unittest` put the project root; the root goes first and
    # the engine's state folder goes nowhere, so a scratch copy under it
    # can never shadow a project module
    sys.path[:] = [p for p in sys.path
                   if not _norm(p or ".").endswith("/.cc_state")]
    sys.path.insert(0, os.getcwd())
    if sys.argv[1] == "unittest":
        _unittest(sys.argv[2:])
    else:
        _pytest(sys.argv[2:])
''' % {"max_names": MAX_NAMES, "max_repr": MAX_REPR}  # noqa: UP031

_SCRIPT_SHA = hashlib.sha256(SCRIPT.encode("utf-8")).hexdigest()[:12]


def ensure_script(fs: Any) -> str:
    """Write the script into the project if it is missing or stale; return
    its project-relative path, or "" when it could not be written."""
    try:
        if fs.exists(SCRIPT_REL) and _same(fs.read(SCRIPT_REL)):
            return SCRIPT_REL
        fs.write(SCRIPT_REL, SCRIPT)
        return SCRIPT_REL
    except Exception:                                    # noqa: BLE001
        return ""


def _same(text: str) -> bool:
    return hashlib.sha256((text or "").replace("\r\n", "\n")
                          .encode("utf-8")).hexdigest()[:12] == _SCRIPT_SHA


def rewrite_argv(argv: Sequence[str], script: str) -> list[str]:
    """`python -m unittest …` → `python <script> unittest …`, and the same
    for pytest; anything else comes back as it was."""
    argv = list(argv)
    if len(argv) >= 3 and argv[1] == "-m" and argv[2] in ("unittest",
                                                           "pytest"):
        return [argv[0], script, argv[2], *argv[3:]]
    return argv


# --------------------------------------------------------------------------
# reading the blocks back
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ValuesBlock:
    file: str
    line: int
    function: str
    rows: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.rows)


_HEAD = re.compile(
    r"^\s*\[VALUES AT THE FAILURE\] (?P<file>.+?):(?P<line>\d+) in "
    r"(?P<fn>\S+)\s*$")
_ROW = re.compile(r"^    (?P<name>[A-Za-z_]\w*) = ")


def parse_values(output: str) -> list[ValuesBlock]:
    """Every block the script printed, in order."""
    out: list[ValuesBlock] = []
    lines = (output or "").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        rows: list[str] = []
        i += 1
        while i < len(lines) and _ROW.match(lines[i]):
            rows.append(lines[i].rstrip())
            i += 1
        if rows:
            out.append(ValuesBlock(file=m.group("file").replace("\\", "/"),
                                   line=int(m.group("line")),
                                   function=m.group("fn"),
                                   rows=tuple(rows)))
    return out


def _same_file(a: str, b: str) -> bool:
    x = (a or "").replace("\\", "/").lower()
    y = (b or "").replace("\\", "/").lower()
    if not x or not y:
        return False
    return x == y or x.endswith("/" + y) or y.endswith("/" + x) \
        or x.rsplit("/", 1)[-1] == y.rsplit("/", 1)[-1]


def attach(diags: Sequence[Diagnostic],
           blocks: Sequence[ValuesBlock]) -> list[Diagnostic]:
    """Each diagnostic with the values of its own frame.

    A located diagnostic takes the block whose file and line it names;
    a located one with no such block, and an unlocated failure (an
    `AssertionError` line, a `FAILED test` line), takes the blocks of the
    nearest failure in order — the deepest frame's and the test's. Each
    block is given out once.
    """
    if not blocks or not diags:
        return list(diags)
    used: set[int] = set()
    out: list[Diagnostic] = []
    for d in diags:
        if d.values:
            out.append(d)
            continue
        chosen: list[int] = []
        if d.file and d.line:
            for i, b in enumerate(blocks):
                if i in used:
                    continue
                if b.line == d.line and _same_file(b.file, d.file):
                    chosen = [i]
                    break
        if not chosen and d.is_error:
            # the deepest-frame block and its test block come in pairs:
            # the first unused block, plus the next one when it is a
            # test's (the pair the script printed for one failure)
            for i, b in enumerate(blocks):
                if i in used:
                    continue
                chosen = [i]
                if i + 1 < len(blocks) and i + 1 not in used \
                        and _looks_like_test(blocks[i + 1].file) \
                        and not _looks_like_test(b.file):
                    chosen.append(i + 1)
                break
        if not chosen:
            out.append(d)
            continue
        used.update(chosen)
        text = "\n".join(_render(blocks[i]) for i in chosen)
        out.append(dataclasses.replace(d, values=text))
    return out


def _looks_like_test(path: str) -> bool:
    name = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = name[:-3] if name.endswith(".py") else name
    return stem.startswith("test_") or stem.endswith("_test")


def _render(b: ValuesBlock) -> str:
    return (f"values at the failure ({b.file}:{b.line} in {b.function}):\n"
            + b.text)
