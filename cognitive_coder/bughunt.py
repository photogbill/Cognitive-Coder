# SPDX-License-Identifier: Apache-2.0
"""Evaluate & Bug Hunt — the deterministic floor, made visible and saved.

Bill (2026-10-20): *"a software debugging tab specifically to hunt for
software bugs … a list of bugs, such as mismatched function names,
indentation errors … THEN I can feed the output into the code repair tab."*

Most of this is tools, not a model, and the order is the point:

  1. **ruff**, every rule (the loop's gate keeps only the certain-bug codes;
     here the operator sees all of them, ranked) and **ruff format**: a file
     the formatter would change is reported, with the first differing line.
  2. **The codemap**: names a file uses that nothing in the project defines.
  3. **keep.py's scan** over the file as it stands: placeholders and the
     model's narration left in the code by an earlier session.
  4. **A type checker** — `pyright` when installed, else `mypy`, else a
     note saying what its absence costs. pygame ships `.pyi` stubs; the
     10-22 `blit` bug was a type error no test reached.
  5. **The model's pass**, last, with 1–4 in its prompt so it names what
     the tools could not; every item it names must be located to an
     existing line or it is listed under "could not locate", never counted.

Output: a `BugHunt` with ranked items, `report()` as BUGHUNT.md, and
`to_debug_reports()` — one `DebugReport` per file, which is the hand-off
to Repair Bill described.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import json
import re
from typing import Any

from . import keep, langs, runner, structured
from . import scope as scope_mod
from .personas import PERSONAS

#: Lint codes that are warnings an operator can act on but not bugs.
_STYLE_PREFIX = ("E", "W", "I", "N", "D", "Q", "COM", "UP", "SIM", "RET",
                 "ANN", "PL")
_PYRIGHT_SEVERITY = {"error": "error", "warning": "warning",
                     "information": "note"}
_MYPY = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+)(?::(?P<col>\d+))?: "
                   r"(?P<sev>error|warning|note): (?P<msg>.*)$", re.M)


@dataclass
class BugItem:
    path: str
    line: int
    message: str
    source: str              # ruff | format | codemap | keep | pyright | mypy | model
    severity: str = "error"  # error | warning | note
    code: str = ""

    @property
    def rank(self) -> int:
        return {"error": 0, "warning": 1, "note": 2}.get(self.severity, 1)

    def one_line(self) -> str:
        code = f" {self.code}" if self.code else ""
        return f"{self.path}:{self.line}: [{self.source}{code}] {self.message}"


@dataclass
class BugHunt:
    scope: str
    files: list[str]
    items: list[BugItem] = field(default_factory=list)
    could_not_locate: list[str] = field(default_factory=list)
    #: With `votes` > 1: bugs the model named in ONE pass only. Shown, not
    #: counted — a plausible-sounding bug that does not land twice on the
    #: same line is the 24B–40B class's signature invention.
    unconfirmed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # tools absent, etc.
    tools_ran: list[str] = field(default_factory=list)
    model_ran: bool = False
    votes: int = 1

    @property
    def errors(self) -> list[BugItem]:
        return [i for i in self.items if i.severity == "error"]

    def by_file(self) -> dict[str, list[BugItem]]:
        out: dict[str, list[BugItem]] = defaultdict(list)
        for i in sorted(self.items, key=lambda i: (i.path, i.rank, i.line)):
            out[i.path].append(i)
        return dict(out)


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------

def _lint(session: Any, path: str, code: str, lang: str,
          hunt: BugHunt) -> None:
    try:
        found, note = runner.lint_code(code, lang, fs=session.host.fs,
                                       ex=session.host.exec,
                                       stem=path.rsplit("/", 1)[-1]
                                       .rsplit(".", 1)[0])
    except Exception as exc:                                 # noqa: BLE001
        hunt.notes.append(f"the linter failed on {path}: {exc}")
        return
    if note and note not in hunt.notes:
        hunt.notes.append(note)
        return
    if "ruff" not in hunt.tools_ran:
        hunt.tools_ran.append("ruff")
    for d in found:
        sev = "error" if d.severity == "error" else "warning"
        if d.code and d.code.startswith(_STYLE_PREFIX) and sev != "error":
            sev = "note"
        hunt.items.append(BugItem(path, d.line, d.message, "ruff", sev,
                                  d.code or ""))


def _format(session: Any, path: str, code: str, lang: str,
            hunt: BugHunt) -> None:
    try:
        formatted, note = runner.format_code(
            code, lang, fs=session.host.fs, ex=session.host.exec,
            stem=path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    except Exception:                                        # noqa: BLE001
        return
    if note and "no formatter" in note:
        if note not in hunt.notes:
            hunt.notes.append(note)
        return
    if "format" not in hunt.tools_ran:
        hunt.tools_ran.append("format")
    if formatted and formatted != code:
        a, b = code.splitlines(), formatted.splitlines()
        first = next((i + 1 for i, (x, y) in enumerate(zip(a, b, strict=False))
                      if x != y),
                     min(len(a), len(b)) + 1)
        hunt.items.append(BugItem(
            path, first, "the formatter would change this file (indentation "
            "or spacing differs from the project's style from this line on)",
            "format", "warning"))


def _codemap(session: Any, path: str, code: str, lang: str,
             hunt: BugHunt) -> None:
    cm = getattr(session, "codemap", None)
    if cm is None:
        return
    try:
        missing = cm.unresolved_in(code, lang)
    except Exception:                                        # noqa: BLE001
        return
    if "codemap" not in hunt.tools_ran:
        hunt.tools_ran.append("codemap")
    for name in missing[:20]:
        line = next((i + 1 for i, ln in enumerate(code.splitlines())
                     if re.search(rf"\b{re.escape(name)}\b", ln)), 0)
        hunt.items.append(BugItem(
            path, line, f"`{name}` is used here and defined nowhere in the "
            f"project — a NameError or ImportError waiting to run",
            "codemap", "error"))


def _keep(path: str, code: str, lang: str, hunt: BugHunt) -> None:
    for line, text in keep.placeholders("", code, lang):
        hunt.items.append(BugItem(
            path, line, f"a placeholder or the model's own commentary left "
            f"in the code: `{text.strip()[:70]}`", "keep", "error"))
    if lang == "python":
        defined = keep.python_defined(code) or {}
        for name in defined:
            copy_of = keep._copy_of(name, defined)
            if copy_of:
                span = defined[name]
                hunt.items.append(BugItem(
                    path, (span.start + 1) if span else 0,
                    f"`{name}` is a suffixed copy of `{copy_of}`; one of "
                    f"them is the bug with a longer name", "keep", "error"))


def _typecheck(session: Any, files: list[str], hunt: BugHunt) -> None:
    ex, fs = session.host.exec, session.host.fs
    root = fs.root()
    tool = ex.which("pyright")
    if tool:
        argv = [tool, "--outputjson"] + [runner._join(root, f) for f in files]
        try:
            proc = ex.run(argv, cwd=root, timeout=300.0,
                          env=runner.scrubbed_env(root, root))
        except Exception as exc:                             # noqa: BLE001
            hunt.notes.append(f"pyright could not be run: {exc}")
            return
        try:
            data = json.loads(proc.stdout or "{}")
        except ValueError:
            hunt.notes.append("pyright did not return JSON")
            return
        hunt.tools_ran.append("pyright")
        for d in data.get("generalDiagnostics", []) or []:
            sev = _PYRIGHT_SEVERITY.get(str(d.get("severity")), "warning")
            file = _rel(str(d.get("file", "")), root)
            line = int(((d.get("range") or {}).get("start") or {})
                       .get("line", -1)) + 1
            hunt.items.append(BugItem(file, max(line, 0),
                                      str(d.get("message", "")).strip(),
                                      "pyright", sev,
                                      str(d.get("rule") or "")))
        return
    tool = ex.which("mypy")
    if tool:
        argv = [tool, "--no-error-summary", "--show-column-numbers",
                "--ignore-missing-imports"] + \
            [runner._join(root, f) for f in files]
        try:
            proc = ex.run(argv, cwd=root, timeout=300.0,
                          env=runner.scrubbed_env(root, root))
        except Exception as exc:                             # noqa: BLE001
            hunt.notes.append(f"mypy could not be run: {exc}")
            return
        hunt.tools_ran.append("mypy")
        for m in _MYPY.finditer(proc.stdout or ""):
            sev = {"error": "error", "warning": "warning"}.get(
                m.group("sev"), "note")
            hunt.items.append(BugItem(_rel(m.group("file"), root),
                                      int(m.group("line")),
                                      m.group("msg").strip(), "mypy", sev))
        return
    hunt.notes.append("no type checker installed (pyright or mypy) — a "
                      "wrong argument order, a tuple used as an object, a "
                      "method called with the wrong count will only show "
                      "when a test or a run reaches it")


def _rel(path: str, root: str) -> str:
    p = path.replace("\\", "/")
    r = root.replace("\\", "/").rstrip("/") + "/"
    return p[len(r):] if p.startswith(r) else p


# --------------------------------------------------------------------------
# the model's pass
# --------------------------------------------------------------------------

_SCHEMA = ('{"bugs": [{"path": "src/x.py", "line": 12, '
           '"message": "one sentence: what is wrong and why it fails"}]}')


def _model_pass(session: Any, files: list[str], hunt: BugHunt,
                max_files: int = 6, votes: int = 1) -> None:
    """The model's pass, last. With `votes` > 1 the pass runs that many
    times at rising temperature and a bug counts only when at least two
    passes place it on the same line (±2); the rest are listed as
    unconfirmed. One pass is the default: honest and cheap."""
    llm = session.host.llm
    try:
        caps = llm.capabilities()
        if not getattr(caps, "loaded", True):
            hunt.notes.append("no model loaded: the model's pass was skipped")
            return
    except Exception:                                        # noqa: BLE001
        return
    shown = []
    for path in files[:max_files]:
        code = _read(session, path)
        if code.strip():
            shown.append(f"### {path}\n" + "\n".join(
                f"{n:>4} | {ln}" for n, ln in
                enumerate(code.splitlines()[:400], 1)))
    if not shown:
        return
    already = "\n".join(f"  - {i.one_line()}" for i in hunt.items[:30])
    task = "\n".join([
        "BUG HUNT. Read the files below and name the bugs the tools did "
        "not: a function called with its arguments in the wrong order, a "
        "value used as the wrong type, a branch that can never run, a "
        "result that is computed and then ignored, an off-by-one, a wrong "
        "comparison. Not style. Not what the tools already found.",
        "",
        ("The tools already found:\n" + already) if already else
        "The tools found nothing.",
        "",
        "Every bug you name must have the path and the exact line number "
        "from the listing. A bug you cannot place on a line is not reported."
        " If you find nothing, return an empty list — say nothing rather "
        "than invent.",
        "",
        "\n\n".join(shown),
    ])
    lines_of = {p: len(_read(session, p).splitlines()) for p in files}
    votes = max(1, int(votes or 1))
    hunt.votes = votes
    temps = [0.2, 0.5, 0.7, 0.8, 0.9][:votes]
    passes: list[list[BugItem]] = []
    for temp in temps:
        got = structured.ask(session, PERSONAS["reviewer"], task,
                             keys=("bugs",), example=_SCHEMA,
                             temperature=temp, max_tokens=1500,
                             architecture=_architecture(session))
        if not got.ok:
            hunt.notes.append(f"the model's pass could not be read: "
                              f"{got.problem}")
            continue
        hunt.model_ran = True
        found: list[BugItem] = []
        rows = got.data.get("bugs") or []
        for row in rows if isinstance(rows, list) else ():
            if not isinstance(row, dict):
                continue
            path = scope_mod._norm(str(row.get("path", "")))
            msg = str(row.get("message", "")).strip()
            try:
                line = int(row.get("line") or 0)
            except (TypeError, ValueError):
                line = 0
            if path in lines_of and 1 <= line <= lines_of[path] and msg:
                found.append(BugItem(path, line, msg, "model", "warning"))
            elif msg:
                note = f"{path or '?'}:{line or '?'} {msg}"
                if note not in hunt.could_not_locate:
                    hunt.could_not_locate.append(note)
        passes.append(found)
    if not passes:
        return
    if len(passes) == 1:
        hunt.items.extend(passes[0])
        return
    # consensus: an item counts when another pass places one within ±2
    # lines of it in the same file; the message kept is the first pass's
    counted: list[BugItem] = []
    for i, found in enumerate(passes):
        for item in found:
            if any(abs(c.line - item.line) <= 2 and c.path == item.path
                   for c in counted):
                continue
            agree = sum(1 for j, other in enumerate(passes) if j != i and
                        any(o.path == item.path and abs(o.line - item.line)
                            <= 2 for o in other))
            if agree >= 1:
                counted.append(BugItem(item.path, item.line, item.message,
                                       "model", "warning",
                                       code=f"{agree + 1}/{len(passes)}"))
            else:
                note = f"{item.path}:{item.line} {item.message}"
                if note not in hunt.unconfirmed:
                    hunt.unconfirmed.append(note)
    hunt.items.extend(counted)


def _architecture(session: Any) -> str:
    try:
        return session.codemap.architecture()
    except Exception:                                        # noqa: BLE001
        return ""


def _read(session: Any, path: str) -> str:
    try:
        return session.host.fs.read(path)
    except Exception:                                        # noqa: BLE001
        return ""


# --------------------------------------------------------------------------
# the hunt
# --------------------------------------------------------------------------

def hunt(session: Any, scope: str | list[str] = "", *,
         model_pass: bool = True, typecheck: bool = True,
         votes: int = 1) -> BugHunt:
    """Tools first, model last; everything located or set aside. `votes`
    > 1 asks the model that many times and keeps what two passes agree
    on (see `_model_pass`)."""
    files = scope_mod.files(session.host.fs, scope)
    label = scope if isinstance(scope, str) else ", ".join(scope)
    out = BugHunt(scope=label or "the project", files=files)
    if not files:
        out.notes.append("the scope names no source files")
        return out
    for path in files:
        code = _read(session, path)
        if not code.strip():
            continue
        lang = langs.id_for_path(path) or "python"
        _lint(session, path, code, lang, out)
        _format(session, path, code, lang, out)
        _codemap(session, path, code, lang, out)
        _keep(path, code, lang, out)
    if typecheck:
        _typecheck(session, [f for f in files
                             if (langs.id_for_path(f) or "python") == "python"],
                   out)
    if model_pass:
        _model_pass(session, [f for f in files if not scope_mod.is_test(f)],
                    out, votes=votes)
    out.items.sort(key=lambda i: (i.rank, i.path, i.line))
    try:
        session.journal.log("bughunt", scope=out.scope, files=len(files),
                            items=len(out.items), errors=len(out.errors),
                            tools=list(out.tools_ran), model=out.model_ran,
                            votes=out.votes,
                            could_not_locate=len(out.could_not_locate),
                            unconfirmed=len(out.unconfirmed))
    except Exception:                                        # noqa: BLE001
        pass
    return out


def report(h: BugHunt) -> str:
    """BUGHUNT.md: ranked, by file, with what ran and what did not."""
    lines = [f"# Bug hunt — {h.scope}", "",
             f"{len(h.files)} file(s) read; {len(h.errors)} error(s), "
             f"{len(h.items) - len(h.errors)} warning(s)/note(s). Tools: "
             f"{', '.join(h.tools_ran) or 'none'}"
             + ((f"; the model's pass ran {h.votes} times, keeping what two "
                 f"agreed on" if h.votes > 1 else "; the model's pass ran")
                if h.model_ran else "; no model pass") + ".", ""]
    if h.notes:
        lines.append("**What did not run, and what it costs:**")
        lines += [f"- {n}" for n in h.notes]
        lines.append("")
    for path, items in h.by_file().items():
        lines.append(f"## {path}")
        for i in items:
            box = "[ ]"
            lines.append(f"- {box} line {i.line} — {i.severity.upper()} "
                         f"({i.source}{' ' + i.code if i.code else ''}): "
                         f"{i.message}")
        lines.append("")
    if h.could_not_locate:
        lines.append("## Could not locate (the model named these without a "
                     "line that exists — not counted)")
        lines += [f"- {c}" for c in h.could_not_locate]
        lines.append("")
    if h.unconfirmed:
        lines.append(f"## Named in one pass of {h.votes} only (not counted "
                     f"— look if you have a reason to)")
        lines += [f"- {c}" for c in h.unconfirmed]
        lines.append("")
    if not h.items:
        lines.append("Nothing found by the tools"
                     + (" or the model" if h.model_ran else "") + ".")
    return "\n".join(lines)


def to_debug_reports(h: BugHunt, *, only_errors: bool = True,
                     paths: list[str] | None = None) -> list:
    """One DebugReport per file: the hand-off to Repair. The error text is
    the items as located lines, which `locate` reads like a traceback."""
    from .debugrepair import DebugReport
    out = []
    for path, items in h.by_file().items():
        if paths is not None and path not in paths:
            continue
        chosen = [i for i in items if (i.severity == "error" or not only_errors)]
        if not chosen:
            continue
        text = "\n".join(f"{i.path}:{i.line}: {i.message}" for i in chosen)
        out.append(DebugReport(
            error_text=text, files=(path,),
            description=f"Bug hunt findings in {path}: "
                        f"{len(chosen)} item(s) to fix, each located above."))
    return out
