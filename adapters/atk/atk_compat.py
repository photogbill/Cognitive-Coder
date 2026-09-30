# SPDX-License-Identifier: Apache-2.0
"""ATK's OLD call surface, on top of Cognitive Coder's new one.

**Why this file exists, and why a plain re-export would not have done.**

§7.3 says each migration step must leave ATK's suite green. A shim that just
does `from cognitive_coder.guard import *` looks like it satisfies that, and
does not: the engine's modules take **Ports** where ATK's took paths, and
several signatures changed as a direct consequence.

    ATK                                    Cognitive Coder
    ────────────────────────────────────   ────────────────────────────────
    Lang.available()                       Lang.available(exec_port)
    langs.available_ids()                  langs.available_ids(exec_port)
    diagnostics.feedback(TEXT, lang, root) diagnostics.feedback(DIAGS, cap)
    diagnostics.attach_source(d, root)     …(d, fs, sources)
    Diagnostic.source                      Diagnostic.source_excerpt
    coderun.build_and_run(code, l, WSPATH) runner.build_and_run(…, fs=, ex=)
    patcher.apply(edits, ROOT)             patcher.Patcher(...).begin(...)
    codectx.project_map(ROOT)              context.project_map(fs)

Every one of those would have failed at ATK's call sites, at runtime, in
whatever code path happened to run first — which is the worst possible way to
discover it. So this module supplies the OLD signatures, implemented against
the new engine, using a local filesystem and a real subprocess runner as the
Ports.

**It is a bridge, not a destination.** Once ATK's own call sites have been
updated to pass Ports, delete this file and the six shims with it. Until
then, it is what makes the migration reversible: `.py.pre-ccoder` on one
side, a working suite on the other.

One behavioural note that is NOT a bug: `Diagnostic` here exposes `.source`
as an alias of `.source_excerpt`, because ATK's tests read `.source`. The new
name is better and the old one is free.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cognitive_coder import context as _context
from cognitive_coder import diagnostics as _diagnostics
from cognitive_coder import guard as _guard
from cognitive_coder import langs as _langs
from cognitive_coder import patcher as _patcher
from cognitive_coder import runner as _runner
from cognitive_coder.filestorage import JsonFileStorage
from cognitive_coder.ports import AutoApprove, LocalFileSystem, SubprocessExec
from cognitive_coder.types import Edit as _Edit
from cognitive_coder.types import GuardFinding as _GuardFinding

# One shared ExecPort. `which` is the only thing it is used for in the langs
# compatibility layer, and building one per call would be wasteful.
_EX = SubprocessExec()


def _fs(root: Any) -> LocalFileSystem:
    return LocalFileSystem(str(root))


# ==========================================================================
# langs — the registry, with the no-argument probing ATK expects
# ==========================================================================

EXE_SUFFIX = ".exe"
render = _langs.render
scaffold_for = _langs.scaffold_for
ids = _langs.ids
labels = _langs.labels


class CompatLang(_langs.Lang):
    """A `Lang` whose probes work with no argument, as ATK's did.

    ATK calls `lang.available()`; the engine's `Lang.available(ex)` takes an
    ExecPort (C2). The shim used to re-export the engine's `get` directly,
    so `get("python").available()` — the exact call the shim's docstring,
    CHANGELOG and CONFORMANCE all promised still worked — raised
    `TypeError: missing 'ex'`, and the no-argument wrapper that did exist
    was under another name nobody called.

    A SUBCLASS, so `isinstance(lang, Lang)` holds in ATK and in the engine,
    and the probes take the ExecPort optionally, so the engine's own calls
    with one still work. Nothing on the engine's class is changed.
    """

    def available(self, ex: Any = None) -> bool:
        return super().available(_EX if ex is None else ex)

    def which_build(self, ex: Any = None) -> str:
        return super().which_build(_EX if ex is None else ex)

    def which_run(self, ex: Any = None) -> str:
        return super().which_run(_EX if ex is None else ex)


def _compat(lang: Any) -> CompatLang | None:
    """The same object's data as a CompatLang (a shallow copy)."""
    if lang is None or isinstance(lang, CompatLang):
        return lang
    twin = copy.copy(lang)
    twin.__class__ = CompatLang
    return twin


class _CompatLangs(Mapping):
    """`LANGS`, live over the engine's registry, yielding CompatLangs."""

    def __getitem__(self, key: str) -> CompatLang:
        return _compat(_langs.LANGS[key])

    def __iter__(self):
        return iter(_langs.LANGS)

    def __len__(self) -> int:
        return len(_langs.LANGS)


LANGS = _CompatLangs()
Lang = _langs.Lang


def get(lang_id: str) -> CompatLang | None:
    return _compat(_langs.get(lang_id))


def for_extension(path: Any) -> CompatLang | None:
    return _compat(_langs.for_extension(path))


def available_ids() -> list[str]:
    """ATK's no-argument form. The engine's takes an ExecPort (C2)."""
    return _langs.available_ids(_EX)


#: Kept for anything written against the earlier shim.
get_compat = get


# ==========================================================================
# diagnostics — text in, feedback out, and `.source` by its old name
# ==========================================================================

MAX_FEEDBACK = _diagnostics.MAX_FEEDBACK
CONTEXT_LINES = _diagnostics.CONTEXT_LINES


@dataclass
class Diagnostic:
    """ATK's mutable Diagnostic, wrapping the engine's frozen one.

    ATK's code assigns to `d.file` and reads `d.source`; the engine's type is
    frozen and calls that field `source_excerpt`. Both are true at once here.
    """
    message: str
    file: str = ""
    line: int = 0
    col: int = 0
    severity: str = "error"
    code: str = ""
    source: str = ""

    @classmethod
    def _from(cls, d: Any) -> Diagnostic:
        return cls(message=d.message, file=d.file, line=d.line,
                   col=d.col or 0, severity=d.severity, code=d.code or "",
                   source=d.source_excerpt)

    def _to(self) -> Any:
        from cognitive_coder.types import Diagnostic as _D
        return _D(file=self.file, line=self.line, col=self.col or None,
                  severity=self.severity, message=self.message,
                  code=self.code or None, source_excerpt=self.source)

    @property
    def rank(self) -> int:
        return self._to().rank

    def where(self) -> str:
        return self._to().where()

    def one_line(self) -> str:
        return self._to().one_line()


def parse(text: str, lang_id: str = "") -> list[Diagnostic]:
    return [Diagnostic._from(d) for d in _diagnostics.parse(text, lang_id)]


def attach_source(diags: list, root: Any = None) -> list[Diagnostic]:
    """ATK passed a ROOT PATH; the engine takes a FileSystemPort."""
    fs = _fs(root) if root else None
    attached = _diagnostics.attach_source([d._to() for d in diags], fs)
    return [Diagnostic._from(d) for d in attached]


def feedback(text: str, lang_id: str = "", root: Any = None,
             max_errors: int = MAX_FEEDBACK) -> str:
    """ATK's signature: RAW TEXT in, model-ready feedback out.

    The engine split this into `parse` → `attach_source` → `feedback` so the
    loop can carry diagnostics forward without re-parsing (D11). ATK's
    one-shot form is reassembled here.
    """
    fs = _fs(root) if root else None
    diags = _diagnostics.attach_source(_diagnostics.parse(text, lang_id), fs)
    return _diagnostics.feedback(diags, max_errors)


def summarise(diags: list) -> str:
    return _diagnostics.summarise([d._to() if isinstance(d, Diagnostic)
                                   else d for d in diags])


def first_error(diags: list) -> Diagnostic | None:
    for d in diags:
        if d.rank == 0:
            return d
    return diags[0] if diags else None


# ==========================================================================
# codeguard → guard. The surface is unchanged; the names are aliased.
# ==========================================================================

BLOCK = _guard.BLOCK
WARN = _guard.WARN
scan = _guard.scan
blocked = _guard.blocked
advisory = _guard.advisory
explain_to_model = _guard.explain_to_model
#: The engine's own type — the one `scan()` returns. It was a fresh dataclass
#: with the same fields, so `isinstance(scan(code)[0], Finding)` was always
#: False in ATK. Same fields, same `one_line()`: nothing else to adapt.
Finding = _GuardFinding


# ==========================================================================
# coderun → runner. Workspace PATH in, RunResult out.
# ==========================================================================

MAX_OUTPUT = 200_000
DEFAULT_TIMEOUT = 15.0
BUILD_TIMEOUT = 60.0


@dataclass
class Phase:
    """ATK's Phase, rebuilt from the engine's PhaseResult."""
    name: str
    cmd: list = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    seconds: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def output(self) -> str:
        joiner = "\n" if self.stdout and self.stderr else ""
        return (self.stdout + joiner + self.stderr).strip()

    @classmethod
    def _from(cls, p: Any) -> Phase:
        proc = p.proc
        return cls(name=p.name, cmd=list(p.argv),
                   stdout=proc.stdout if proc else "",
                   stderr=proc.stderr if proc else "",
                   returncode=(proc.exit_code if proc else
                               (0 if p.ok else 1)),
                   seconds=proc.duration_s if proc else 0.0,
                   timed_out=bool(proc.timed_out) if proc else False)


@dataclass
class RunResult:
    ok: bool
    lang: str
    phases: list = field(default_factory=list)
    blocked: str = ""
    warnings: str = ""
    diagnostics: list = field(default_factory=list)

    @property
    def failed_phase(self) -> str:
        if self.blocked:
            return "guard"
        for p in self.phases:
            if not p.ok:
                return p.name
        return ""

    @property
    def output(self) -> str:
        return "\n".join(p.output for p in self.phases if p.output).strip()

    @property
    def stdout(self) -> str:
        for p in reversed(self.phases):
            if p.name in ("run", "test"):
                return p.stdout
        return ""

    def summary(self) -> str:
        if self.blocked:
            return f"refused before running — {self.blocked}"
        if self.ok:
            secs = sum(p.seconds for p in self.phases)
            return f"ok in {secs:.1f}s"
        phase = self.failed_phase or "run"
        return f"{phase} failed — {summarise(self.diagnostics)}"

    @classmethod
    def _from(cls, r: Any) -> RunResult:
        return cls(ok=r.ok, lang=r.lang,
                   phases=[Phase._from(p) for p in r.phases],
                   blocked=r.blocked, warnings=r.warnings,
                   diagnostics=[Diagnostic._from(d) for d in r.diagnostics])


def build_and_run(code: str, lang_id: str, workspace: Any,
                  stem: str = "main", timeout: float = DEFAULT_TIMEOUT,
                  build_timeout: float = BUILD_TIMEOUT,
                  project_mode: bool = False, stdin_text: str = "",
                  skip_guard: bool = False) -> RunResult:
    """ATK's positional `workspace` path, on the engine's Port-based runner."""
    fs = _fs(workspace)
    return RunResult._from(_runner.build_and_run(
        code, lang_id, fs=fs, ex=_EX, stem=stem, project_mode=project_mode,
        stdin=stdin_text, skip_guard=skip_guard, timeout=timeout,
        build_timeout=build_timeout))


def run_tests(workspace: Any, lang_id: str, stem: str = "main",
              timeout: float = DEFAULT_TIMEOUT) -> RunResult:
    return RunResult._from(_runner.run_tests(
        lang_id, fs=_fs(workspace), ex=_EX, stem=stem, timeout=timeout))


def format_code(code: str, lang_id: str, workspace: Any,
                stem: str = "main") -> tuple[str, str]:
    return _runner.format_code(code, lang_id, fs=_fs(workspace), ex=_EX,
                               stem=stem)


def lint_code(code: str, lang_id: str, workspace: Any,
              stem: str = "main") -> tuple[list, str]:
    diags, note = _runner.lint_code(code, lang_id, fs=_fs(workspace), ex=_EX,
                                    stem=stem)
    return [Diagnostic._from(d) for d in diags], note


# ==========================================================================
# patcher — the old apply/undo, on top of the new transactions
# ==========================================================================

SNAPSHOT_DIR = ".atk_snapshots"
MAX_SNAPSHOTS = 40
Edit = _Edit
parse_edits = _patcher.parse_edits


@dataclass
class EditResult:
    path: str
    ok: bool
    reason: str = ""
    diff: str = ""


@dataclass
class ApplyOutcome:
    ok: bool
    snapshot: str = ""
    results: list = field(default_factory=list)
    diff: str = ""

    @property
    def applied(self) -> list:
        return [r for r in self.results if r.ok]

    @property
    def refused(self) -> list:
        return [r for r in self.results if not r.ok]

    def summary(self) -> str:
        if not self.results:
            return "no edits proposed"
        good, bad = len(self.applied), len(self.refused)
        bits = [f"{good} file{'s' * (good != 1)} changed"]
        if bad:
            bits.append(f"{bad} refused ({self.refused[0].reason})")
        if self.snapshot:
            bits.append("undo available")
        return " · ".join(bits)


def _patcher_for(root: Any) -> Any:
    """One Patcher per root, so sequence numbers stay monotonic per project.

    A fresh Patcher per call would restart the counter at 1 and make the
    transaction log's linearity a lie (M25 rule 2). Its log lives on disk,
    in `<root>/.cc_state/` like the CLI's: in `MemoryStorage` it died with
    ATK's process, so after a restart `undo(root)` said "nothing to undo"
    beside a full `.cc_snapshots/`.
    """
    key = str(Path(root).resolve())
    if key not in _PATCHERS:
        _PATCHERS[key] = _patcher.Patcher(
            _fs(root), JsonFileStorage(str(Path(key) / ".cc_state")),
            AutoApprove())
    return _PATCHERS[key]


_PATCHERS: dict[str, Any] = {}


def apply(edits: list, root: Any, snapshot: bool = True) -> ApplyOutcome:
    """ATK's one-shot apply, as a single committed transaction.

    ATK auto-applied with undo, so `AutoApprove` is correct here — and it is
    the one place in the whole engine where that is true by inheritance
    rather than by a fresh decision.
    """
    if not edits:
        return ApplyOutcome(False)
    p = _patcher_for(root)
    tx = p.begin(task_id="atk-apply", atomic=False)
    results = tx.apply(list(edits))
    record = tx.commit()
    return ApplyOutcome(
        ok=any(r.ok for r in results),
        snapshot=record.snapshot_dir if any(r.ok for r in results) else "",
        results=[EditResult(r.path, r.ok, r.reason, r.diff) for r in results],
        diff=tx.diff)


def preview(edits: list, root: Any) -> str:
    return _patcher_for(root).preview(edits)


def snapshots(root: Any) -> list[str]:
    """Restore points, newest first — from the transaction log."""
    records = _patcher_for(root).history()
    return [f"{r.seq:04d}-{r.task_id}" for r in reversed(records)
            if r.state == "committed"]


def undo(root: Any, stamp: str = "") -> dict:
    """ATK's undo-the-last-apply, expressed as `undo_to`."""
    p = _patcher_for(root)
    committed = [r for r in p.history() if r.state == "committed"]
    if not committed:
        return {"ok": False, "note": "nothing to undo — no snapshots yet"}
    target = committed[-1]
    if stamp:
        for r in committed:
            if stamp in f"{r.seq:04d}-{r.task_id}":
                target = r
                break
    out = p.undo_to(target.seq - 1, confirm=lambda _s: True)
    return {"ok": bool(out.get("ok")),
            "restored": out.get("restored", []),
            "stamp": f"{target.seq:04d}-{target.task_id}",
            "note": out.get("note", "")}


# ==========================================================================
# codectx → context. Root PATH in, everywhere ATK passed one.
# ==========================================================================

DEFAULT_BUDGET = 24_000
Symbol = _context.Symbol
symbols = _context.symbols
outline = _context.outline
slice_around = _context.slice_around
symbol_body = _context.symbol_body


def build_context(pieces: list, budget: int = DEFAULT_BUDGET) -> str:
    return _context.build_context(pieces, budget)


def project_map(root: Any, max_files: int = 200) -> str:
    return _context.project_map(_fs(root), max_files)


def relevant_files(root: Any, query: str, limit: int = 6) -> list[str]:
    return _context.relevant_files(_fs(root), query, limit)
