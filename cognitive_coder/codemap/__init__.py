# SPDX-License-Identifier: Apache-2.0
"""Architectural Code-RAG: what exists, what calls what, and what breaks.

The codemap answers the question a small model gets most wrong: **what is
actually in this project?** D4 — invented imports and APIs, `from utils import
parse_config` where no such thing exists — is the single most common
small-model error in multi-file work, and it is not fixable by asking more
nicely. It is fixable by handing over the real signatures, and by checking
generated names against reality before running anything.

FIVE TOOLS, NATIVE FIRST (§6.7). The target model has trained tool calling, so
mid-generation lookup is a set of real tools with JSON schemas:

    search_codemap(name)         → signatures for a symbol or module
    read_slice(path, start, end) → a region of a file
    list_symbols(path)           → the outline
    run_tests(pattern)           → run a subset, return parsed diagnostics
    apply_patch(path, old, new)  → anchored edit, through the transaction
                                   and approval gate (M18)

Three reasons this beats a text marker: the model was tuned for exactly this
shape; the output is structurally parseable rather than regex-scraped; and a
schema constrains the arguments in a way prose never can.

**The text-marker fallback** exists for hosts whose model reports
`supports_tools=False`. It accepts several syntaxes, caps lookups at three per
generation, and corrects malformed syntax once and only once (M31). It also
**forces epoch-per-write**: without live tools a lagging summary has no safety
net, so it is not allowed to lag. That is noted here so nobody removes the
tools and quietly breaks the guarantee (G.7).

Freshness, precisely (M30): re-index on every write, so the QUERY interface is
never stale. The INJECTED summary may lag by a declared epoch — see `zoom.py`,
which is where that bargain is explained and enforced.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
import re
import sqlite3
from typing import Any

from .. import langs
from ..types import CodemapStats, Diagnostic, Edit, Symbol, ToolSpec
from . import parse_python, parse_regex, parse_treesitter, zoom
from .store import Store

MAX_TEXT_LOOKUPS = 3          # the fallback's hard cap (M31)

#: Directories never indexed AND never readable through `read_slice`. The
#: two used to differ: the index skipped `.git/`, the tool happily returned
#: `.git/config` (a remote URL, perhaps a token) and `.env` to the model —
#: which, with remote mode on, is outbound context.
#: The one list of directories nothing indexes, lists or reads as source.
#: `context.py` extends it rather than keeping copies: it had two of its
#: own, neither with `.cc_state/` — where the scratch copies autofix works
#: on and the compiled test harnesses now live — so a project's own build
#: debris would have been offered to the model as source.
SKIP_DIRS = (".git", "__pycache__", "node_modules", ".venv", "venv",
             "target", "build", "dist", ".cc_snapshots", ".atk_snapshots",
             ".cc_journal", ".cc_state", ".ccoder", ".python", ".tools")

#: `read_slice` returns at most this much, whatever it is asked for. One
#: call returning 400 KB (100 long lines) was observed; that is a context
#: window spent on one answer.
READ_SLICE_LINES = 200
READ_SLICE_BYTES = 16_000


def skipped_path(path: str) -> bool:
    """True for a path under `SKIP_DIRS`, or a `.env` file anywhere.

    By path COMPONENT: the old substring test also skipped `src/rebuild/`
    because it contains "build/".
    """
    parts = [p for p in str(path or "").replace("\\", "/").split("/")
             if p and p != "."]
    if not parts:
        return False
    return (any(p in SKIP_DIRS for p in parts[:-1])
            or parts[-1].startswith(".env"))


class _BadLine(ValueError):
    """A line bound the model gave that is not a line number."""


def _line_arg(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise _BadLine(str(value))
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise _BadLine(str(value)) from exc


_BAD_LINE = ("read_slice needs whole line numbers for start and end — for "
             "example [READ_SLICE: src/app.py, 10, 40]. Nothing was read.")


class CodeMap:
    """Index, query, and the tool surface the model uses mid-generation."""

    def __init__(self, fs: Any, storage: Any, *, events: Any = None,
                 use_treesitter: bool = True,
                 force_epoch_per_write: bool = False) -> None:
        self.fs = fs
        self._events = events
        self.store = Store(storage.sqlite_path("codemap"))
        self.use_treesitter = use_treesitter
        # M31: a model with no tool calling cannot check a lagging summary,
        # so what it is shown must not lag. That used to be met by rebuilding
        # the snapshot on every write — the cached prefix discarded per
        # file. It is now met by the tail: the staleness note carries every
        # changed file's CURRENT line (zoom.staleness_note), so snapshot plus
        # note is always the present state, for every model, at no cache
        # cost. The flag remains (Session sets it from `supports_tools`) and
        # now only changes the note's wording: no "use search_codemap" to a
        # model that cannot.
        self.force_epoch_per_write = force_epoch_per_write
        self._text_lookups = 0
        self._syntax_corrected = False
        self._db_error_said = False

    def close(self) -> None:
        self.store.close()

    # -- indexing ---------------------------------------------------------
    def parser_for(self, lang_id: str) -> tuple[Callable, str]:
        """Which parser, and an honest label for how exact it is (C7)."""
        if lang_id == "python":
            return parse_python.parse, "exact"
        if self.use_treesitter and parse_treesitter.available(lang_id):
            return (lambda text, path: parse_treesitter.parse(
                text, path, lang_id)), "parsed"
        return (lambda text, path: parse_regex.parse(text, path, lang_id)), \
            "approximate"

    def index_file(self, path: str, text: str | None = None,
                   force: bool = False) -> int:
        """Index one file. Called on EVERY write, so the tools stay live."""
        if text is None:
            try:
                text = self.fs.read(path)
            except Exception:                            # noqa: BLE001
                return 0
        lang = langs.id_for_path(path)
        if not lang:
            return 0
        if not force and not self.store.needs_index(path, text):
            return 0
        parse, _how = self.parser_for(lang)
        try:
            symbols, edges, unresolved = parse(text, path)
        except Exception:                                # noqa: BLE001
            # A parser that throws must not stop an index. The file simply
            # has no symbols known, which the resolution rate will show.
            symbols, edges, unresolved = [], [], []
        try:
            self.store.put_file(path, lang, text, symbols, edges, unresolved)
        except sqlite3.OperationalError as exc:
            # Another session holding the lock past busy_timeout, a full
            # disk, a read-only database: the map is a HINT, and a stale
            # hint is survivable. An exception out of every write is not.
            # Said once, not per file, until a write succeeds again.
            try:
                self.store.db.rollback()
            except sqlite3.Error:
                pass
            if not self._db_error_said:
                self._db_error_said = True
                self._emit("warning",
                           f"The code map could not be updated ({exc}), so "
                           f"lookups may describe files as they were. The "
                           f"build itself is unaffected.")
            return 0
        self._db_error_said = False
        return sum(1 for s in symbols if s.kind != "module")

    def index_project(self, *, limit: int = 2000) -> CodemapStats:
        """Index everything indexable. `.git/` is excluded (M27).

        And FORGET what is no longer there: `store.forget` existed and
        nothing called it, so a deleted file stayed indexed — `resolves()`
        vouched for its functions and the architecture block listed it.
        Only when the listing itself succeeded: a listing that failed says
        nothing about what exists.
        """
        n = 0
        try:
            paths = self.fs.list("*")
            listed = True
        except Exception:                                # noqa: BLE001
            paths, listed = [], False
        present: set[str] = set()
        for path in sorted(paths):
            norm = str(path).replace("\\", "/")
            if skipped_path(norm) or not langs.id_for_path(norm):
                continue
            present.add(norm)
            if n >= limit:
                continue
            self.index_file(norm)
            n += 1
        if listed:
            for row in self.store.files():
                if row["path"] not in present:
                    self.store.forget(row["path"])
        stats = self.store.stats()
        self._emit("status", f"codemap: {zoom.stats_line(stats)}")
        return stats

    def reindex_after_write(self, path: str) -> None:
        """The freshness obligation, in one call (M30).

        The SQLite index updates IMMEDIATELY so the tools are never stale.
        The injected text summary updates by EPOCH — that decision belongs to
        `zoom.should_bump_epoch`, and this method does not make it.
        """
        self.index_file(path, force=True)

    # -- queries ----------------------------------------------------------
    def stats(self) -> CodemapStats:
        return self.store.stats()

    def search(self, name: str) -> list[dict]:
        return self.store.find(name)

    def resolves(self, name: str) -> bool:
        return self.store.resolves(name)

    def unresolved_in(self, text: str, lang_id: str) -> list[str]:
        """Names this code calls that do not exist in the project (D4).

        Run after generation and BEFORE running anything: catching an
        invented API here is cheaper than a failed build and far more precise
        — "there is no `parse_config` in this project" is a fixable sentence,
        an ImportError traceback is a puzzle.

        A check that cries wolf is a check somebody turns off, so every
        name the file itself explains is left alone: its own definitions,
        builtins, locals and parameters (for every language, not just
        Python — `document.getElementById` and `fmt.Println` used to be
        reported on nearly every JS and Go file), and names reached through
        an import of the standard library or an installed package
        (`import numpy as np; np.array`, `from collections import
        OrderedDict`). What IS reported: a name that exists nowhere, a name
        imported from a module that is neither the project's, the standard
        library's nor installed — even when a local later rebinds it — and
        a name imported from a project module that does not define it.
        """
        python = lang_id == "python"
        symbols, _e, unresolved = (
            parse_python.parse(text, "<generated>") if python
            else parse_regex.parse(text, "<generated>", lang_id))
        local = {s.name for s in symbols}
        builtins_ = _BUILTINS.get(lang_id, set())

        # local name → (module, imported name or "")
        heads: dict[str, tuple[str, str]] = {}
        if python:
            for module, name, alias in parse_python.import_bindings(text):
                if name:
                    heads[alias or name] = (module, name)
                else:
                    heads[alias or module.split(".")[0]] = (module, "")
            bound = parse_python.bound_names(text)
        else:
            for module in parse_regex.imports_of(text, lang_id):
                tail = re.split(r"[/.:\\]+", str(module).strip("./"))
                for part in (tail[-1], tail[0]) if tail else ():
                    if part:
                        heads.setdefault(part, (str(module), "*"))
            bound = parse_regex.bound_names(text, lang_id)

        out: list[str] = []
        for _src, name, _kind in unresolved:
            raw = str(name)
            head, short = raw.split(".")[0], raw.split(".")[-1]
            if head in heads:
                # Decided by the IMPORT, before the local-binding checks:
                # `from utils import cfg; cfg = cfg or None; cfg.load()` is
                # still a name from a module that does not exist.
                if not self._import_explains(*heads[head], raw):
                    if raw not in out:
                        out.append(raw)
                continue
            if raw in local or short in local or head in local:
                continue      # defined here, or a method on something that is
            if short in builtins_ or head in builtins_:
                continue
            if head in bound:
                continue      # a local, a parameter, or an attribute on one:
                              # unknowable, and not a claim this check is
                              # entitled to make
            if self.store.resolves(raw) or self.store.resolves(short):
                continue
            if raw not in out:
                out.append(raw)
        return out

    def _import_explains(self, module: str, name: str, raw: str) -> bool:
        """Whether an import accounts for a call through `raw`.

        `name` is "" for `import m`, "*" for a non-Python import. Those two
        are accepted unless `m` is a PROJECT module lacking the attribute:
        `import pygame` where pygame is not installed beside this engine is
        still a real dependency, and the run will say so precisely if it
        is missing. A `from m import n` is judged: standard library or
        installed → fine; a project module → `n` must exist in the project;
        anything else is the D4 case, `from utils import parse_config` with
        no `utils` anywhere, and is reported.
        """
        if name == "*":
            return True
        status = self._module_status(module)
        if status == "project":
            wanted = name or raw.split(".")[-1]
            return bool(self.store.resolves(wanted))
        if not name:
            return True
        return status in ("stdlib", "installed")

    def _module_status(self, module: str) -> str:
        """project | stdlib | installed | unknown, for a Python module."""
        import importlib.util
        import sys

        module = str(module or "")
        if module.startswith(".") or self.store._file_for_module(module):
            return "project"
        top = module.split(".")[0]
        if not top.isidentifier():
            return "unknown"
        if top in getattr(sys, "stdlib_module_names", ()):
            return "stdlib"
        try:
            if importlib.util.find_spec(top) is not None:
                return "installed"
        except (ImportError, ValueError):
            pass
        return "unknown"

    def blast_radius(self, symbol: str, depth: int = 2) -> dict:
        return self.store.blast_radius(symbol, depth)

    def architecture(self, target: str = "", max_tokens: int = 4096,
                     count_tokens=None) -> str:
        return zoom.generate_architecture_context(
            self.store, target, max_tokens=max_tokens,
            count_tokens=count_tokens)

    def prefix_block(self, target: str = "") -> str:
        """The stable, cacheable architecture block (G.7.1).

        The epoch's SNAPSHOT (taken by `bump_epoch`), so the bytes change
        only when the epoch does. Rendered live only before the first
        epoch, when there is no snapshot yet and nothing cached to protect.
        """
        snapshot = self.store.architecture_snapshot()
        if snapshot is not None:
            return snapshot
        return zoom.architecture_prefix(self.store, target=target)

    def tail_blocks(self, target: str, *, count_tokens=None,
                    planned: Sequence[str] = ()) -> list[str]:
        """The volatile tail: interfaces, examples, staleness (G.7.1)."""
        return [b for b in (
            zoom.dependency_interfaces(self.store, target,
                                       count_tokens=count_tokens,
                                       planned=planned),
            zoom.similar_examples(self.store, target),
            zoom.staleness_note(
                self.store,
                can_look_up=not self.force_epoch_per_write)) if b]

    def maybe_bump_epoch(self, **why: Any) -> int:
        bump, reason = zoom.should_bump_epoch(self.store, **why)
        if not bump:
            return self.store.epoch
        n = self.store.bump_epoch(reason)
        self._emit("status", f"architecture snapshot rebuilt (epoch {n}) — "
                             f"{reason}")
        return n

    # -- the tool surface (§6.7) ------------------------------------------
    def tool_specs(self, *, allow_patch: bool = True,
                   allow_tests: bool = True) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name="search_codemap",
                description=("Look up a symbol, function, class or module "
                             "anywhere in this project and get its exact "
                             "signature and location. Use this instead of "
                             "guessing at a name."),
                parameters={"type": "object",
                            "properties": {"name": {
                                "type": "string",
                                "description": "The symbol or module name."}},
                            "required": ["name"]}),
            ToolSpec(
                name="read_slice",
                description=("Read a region of a file, with line numbers. "
                             "Prefer this over asking for a whole file."),
                parameters={"type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "start": {"type": "integer"},
                                "end": {"type": "integer"}},
                            "required": ["path"]}),
            ToolSpec(
                name="list_symbols",
                description="List every symbol defined in one file.",
                parameters={"type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"]}),
        ]
        if allow_tests:
            specs.append(ToolSpec(
                name="run_tests",
                description=("Run a subset of the tests and get the parsed "
                             "failures back."),
                parameters={"type": "object",
                            "properties": {"pattern": {"type": "string"}},
                            "required": []}))
        if allow_patch:
            specs.append(ToolSpec(
                name="apply_patch",
                description=("Replace an exact block of text in a file. The "
                             "old text must appear EXACTLY ONCE — include "
                             "enough surrounding lines to be unambiguous."),
                parameters={"type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "old": {"type": "string"},
                                "new": {"type": "string"}},
                            "required": ["path", "old", "new"]}))
        return specs

    def call_tool(self, name: str, arguments: dict, *,
                  patch_sink: Callable[[Edit], str] | None = None,
                  test_runner: Callable[[str], str] | None = None) -> str:
        """Execute one tool call and return the text the model gets back.

        `apply_patch` deliberately does NOT write here. It hands the edit to
        `patch_sink`, which is the loop's current transaction plus the
        approval gate (M18) — tool calling must never be a side door around
        the approval default, so the side door is simply not built.
        """
        args = arguments or {}
        try:
            if name == "search_codemap":
                return self._tool_search(str(args.get("name", "")))
            if name == "read_slice":
                try:
                    start = _line_arg(args.get("start"), 1)
                    end = _line_arg(args.get("end"), 0)
                except _BadLine:
                    return _BAD_LINE
                return self._tool_slice(str(args.get("path", "")), start,
                                        end)
            if name == "list_symbols":
                return self._tool_symbols(str(args.get("path", "")))
            if name == "run_tests":
                if test_runner is None:
                    return ("Tests cannot be run from here in this session.")
                return test_runner(str(args.get("pattern", "")))
            if name == "apply_patch":
                if patch_sink is None:
                    return ("Patches cannot be applied from here in this "
                            "session; return the code instead.")
                return patch_sink(Edit(path=str(args.get("path", "")),
                                       kind="replace",
                                       old=str(args.get("old", "")),
                                       new=str(args.get("new", ""))))
        except Exception as exc:                         # noqa: BLE001
            # A tool that throws must come back as text the model can act on,
            # never as an exception that ends the generation.
            return f"That tool call failed: {exc}"
        return (f"There is no tool called {name!r}. Available: "
                f"search_codemap, read_slice, list_symbols, run_tests, "
                f"apply_patch.")

    def _tool_search(self, name: str) -> str:
        rows = self.store.find(name)
        if not rows:
            near = [r["name"] for r in self.store.find(name.split(".")[-1][:4])
                    ][:5]
            hint = (f" Closest names in the project: {', '.join(near)}."
                    if near else "")
            return (f"There is no `{name}` anywhere in this project.{hint} "
                    f"Do not call it — either use something that exists, or "
                    f"say that it needs to be written.")
        out = [f"{len(rows)} match(es) for `{name}`:"]
        for r in rows:
            approx = "  (pattern-matched, may be imprecise)" \
                if r["approximate"] else ""
            doc = f"\n      {r['docstring']}" if r["docstring"] else ""
            out.append(f"  {r['path']}:{r['line']}  "
                       f"{r['signature'] or r['name']}{approx}{doc}")
        return "\n".join(out)

    def _tool_slice(self, path: str, start: int, end: int) -> str:
        if skipped_path(path):
            return (f"`{path}` is not something this tool will read: it is "
                    f"version-control, build, environment or engine state, "
                    f"not project source. Nothing was read.")
        try:
            text = self.fs.read(path)
        except Exception:                                # noqa: BLE001
            return (f"`{path}` is not in this project. Use list_symbols on a "
                    f"path that is, or search_codemap to find where "
                    f"something lives.")
        lines = text.splitlines()
        start = max(1, start)
        end = min(len(lines), end or (start + 60),
                  start + READ_SLICE_LINES - 1)
        if start > len(lines):
            return f"`{path}` has only {len(lines)} lines."
        out: list[str] = []
        size = 0
        last = start - 1
        for n in range(start, end + 1):
            row = f"{n:>5} | {lines[n - 1]}"
            if size + len(row) > READ_SLICE_BYTES:
                if not out:              # one enormous line: show its start
                    out.append(row[:READ_SLICE_BYTES] + " …(line cut)")
                    last = n
                break
            out.append(row)
            size += len(row) + 1
            last = n
        tail = ""
        if last < len(lines):
            tail = (f"\n… {len(lines) - last} more lines follow; ask for "
                    f"lines {last + 1}-{min(len(lines), last + 60)} next.")
        return (f"{path} lines {start}-{last} of {len(lines)}:\n"
                + "\n".join(out) + tail)

    def _tool_symbols(self, path: str) -> str:
        rows = self.store.symbols_in(path)
        if not rows:
            return (f"Nothing is indexed for `{path}`. Either it has no "
                    f"symbols, or it is not a source file this project "
                    f"indexes.")
        approx = any(r["approximate"] for r in rows)
        head = f"{len(rows)} symbol(s) in {path}" + (
            "  (pattern-matched, not parsed — it can miss unusual "
            "declarations)" if approx else "")
        body = "\n".join(f"  {r['line']:>5}: {r['kind']} "
                         f"{r['signature'] or r['name']}" for r in rows)
        return f"{head}\n{body}"

    # -- the text-marker fallback (M31) -----------------------------------
    def reset_lookups(self) -> None:
        """Called at the start of each generation; the cap is PER one."""
        self._text_lookups = 0
        self._syntax_corrected = False

    def parse_text_lookups(self, text: str) -> list[tuple[str, str]]:
        """Accept several syntaxes, because drift is guaranteed (D10).

        `[SEARCH_CODEMAP: x]`, `SEARCH_CODEMAP(x)`,
        `<search_codemap>x</search_codemap>` all mean the same thing, and a
        model that has drifted between them is not confused about intent.
        """
        # Every form is anchored to the START of a line, and the call form
        # must be UPPER CASE: case-insensitive and unanchored, the pattern
        # took `def list_symbols(path):` in the model's own code for a
        # lookup and spent one of its three on it.
        found: list[tuple[str, str]] = []
        for tool in ("search_codemap", "list_symbols", "read_slice"):
            up = tool.upper()
            for pattern, flags in (
                    (rf"^[ \t]*\[{up}:\s*([^\]\n]+)\]", re.I),
                    (rf"^[ \t]*{up}\(\s*['\"]?([^)'\"\n]+)['\"]?\s*\)",
                     0),
                    (rf"^[ \t]*<{tool}>\s*(.*?)\s*</{tool}>",
                     re.I | re.S)):
                for m in re.finditer(pattern, text or "", flags | re.M):
                    found.append((tool, m.group(1).strip()))
        return found

    def answer_text_lookups(self, text: str) -> str:
        """The fallback's reply, capped at three lookups per generation.

        On exceeding the cap the model is told it has used its lookups and
        must work with what it has — an instruction, not a complaint, for the
        same reason `guard.explain_to_model` is phrased that way.
        """
        calls = self.parse_text_lookups(text)
        if not calls:
            if (self._looks_like_broken_call(text)
                    and not self._syntax_corrected):
                self._syntax_corrected = True
                # Corrected ONCE, and only once (M31). A model told the same
                # thing three times starts reproducing the correction instead
                # of the code.
                return ("To look something up, write exactly: "
                        "[SEARCH_CODEMAP: the_name] on its own line.")
            return ""
        replies: list[str] = []
        for tool, arg in calls:
            if self._text_lookups >= MAX_TEXT_LOOKUPS:
                replies.append(
                    "You have used your lookups for this file. Work with "
                    "what you have; if something you need is genuinely "
                    "missing, say so instead of guessing.")
                break
            self._text_lookups += 1
            if tool == "search_codemap":
                replies.append(self._tool_search(arg))
            elif tool == "list_symbols":
                replies.append(self._tool_symbols(arg))
            else:
                parts = [p.strip() for p in arg.split(",")]
                try:
                    start = _line_arg(parts[1] if len(parts) > 1 else None,
                                      1)
                    end = _line_arg(parts[2] if len(parts) > 2 else None, 0)
                except _BadLine:
                    replies.append(_BAD_LINE)
                    continue
                replies.append(self._tool_slice(parts[0], start, end))
        return "\n\n".join(replies)

    @staticmethod
    def _looks_like_broken_call(text: str) -> bool:
        """A malformed MARKER, not the words: a comment saying "list
        symbols defined here" used to earn the one-time correction."""
        # The tool's NAME (underscored) not used as a function in code —
        # `def list_symbols(` and `db.search_codemap(x)` are code — or a
        # bracket/tag opened around the words.
        return bool(re.search(
            r"(?<![\w.])(?:search_codemap|list_symbols|read_slice)\b"
            r"(?!\s*\()|[\[<][ \t]*(?:search|list|read)[_ ]"
            r"(?:codemap|symbols|slice)\b", text or "", re.I))

    def _emit(self, kind: str, message: str, data: dict | None = None) -> None:
        if self._events is None:
            return
        try:
            self._events.event(kind, message, data)
        except Exception:                                # noqa: BLE001
            pass


# Names that resolve to the language, not to the project. Without this every
# `len()` and `println!` is reported as an invented API, which would make the
# D4 check noise instead of signal.
_BUILTINS: dict[str, set] = {
    #: `import builtins` — NOT `dir(__builtins__)`.
    #:
    #: The old line branched on whether `__builtins__` was a dict and then
    #: called `dir()` on it either way, which is the same expression twice.
    #: That matters: inside an imported module `__builtins__` IS a dict, so
    #: `dir()` returned the dict's own methods — keys, values, items — and
    #: 75 names instead of about 150. Missing from the list were `reversed`,
    #: `round`, and every exception type, so each of those was reported as
    #: "a name this project does not define" on any file that used one.
    #:
    #: Found by a real run flagging `reversed` in a renderer, after the
    #: local-variable false positives had already been fixed. One remaining
    #: wrong name in an otherwise clean report is worse than a noisy one,
    #: because by then the report is being believed.
    "python": set(dir(builtins)) | {"self", "super"},
    "c": {"printf", "malloc", "free", "memcpy", "strlen", "strcmp", "sizeof",
          "fopen", "fclose", "fprintf", "sprintf", "snprintf", "exit"},
    "cpp": {"printf", "std", "cout", "cerr", "endl", "sizeof", "make_unique",
            "make_shared", "move", "size", "push_back", "begin", "end"},
    "rust": {"println", "format", "vec", "Some", "None", "Ok", "Err",
             "String", "Vec", "unwrap", "expect", "into", "from", "new"},
    "go": {"make", "len", "cap", "append", "panic", "recover", "print",
           "println", "new", "copy", "delete"},
    "javascript": {"console", "require", "JSON", "Math", "Object", "Array",
                   "Promise", "String", "Number", "Boolean", "parseInt",
                   "parseFloat", "setTimeout", "fetch", "document",
                   "window", "globalThis", "navigator", "localStorage",
                   "sessionStorage", "Date", "Map", "Set", "WeakMap",
                   "Error", "TypeError", "RegExp", "Symbol", "BigInt",
                   "setInterval", "clearTimeout", "clearInterval", "alert",
                   "confirm", "prompt", "process", "module", "exports",
                   "Buffer", "URL", "Reflect", "Proxy", "Intl",
                   "requestAnimationFrame", "structuredClone"},
    "gdscript": {"print", "printerr", "push_error", "load", "preload",
                 "range", "len", "str", "int", "float", "Vector2", "Vector3",
                 "get_node", "emit_signal", "connect", "is_instance_valid"},
}
_BUILTINS["typescript"] = _BUILTINS["javascript"]

__all__ = ["CodeMap", "Store", "zoom", "parse_python", "parse_regex",
           "parse_treesitter", "Symbol", "Diagnostic", "MAX_TEXT_LOOKUPS"]
