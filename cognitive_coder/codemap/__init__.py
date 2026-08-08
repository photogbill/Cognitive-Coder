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

from collections.abc import Callable
from typing import Any

from .. import langs
from ..types import CodemapStats, Diagnostic, Edit, Symbol, ToolSpec
from . import parse_python, parse_regex, parse_treesitter, zoom
from .store import Store

MAX_TEXT_LOOKUPS = 3          # the fallback's hard cap (M31)


class CodeMap:
    """Index, query, and the tool surface the model uses mid-generation."""

    def __init__(self, fs: Any, storage: Any, *, events: Any = None,
                 use_treesitter: bool = True,
                 force_epoch_per_write: bool = False) -> None:
        self.fs = fs
        self._events = events
        self.store = Store(storage.sqlite_path("codemap"))
        self.use_treesitter = use_treesitter
        # M31 / G.7's honest cost. The injected architecture summary is
        # allowed to lag by an epoch ONLY because the model can always call
        # `search_codemap` and get the live answer. A host whose model has no
        # tool calling has no such safety net, so on that path the summary is
        # not allowed to lag: the epoch is rebuilt on every write, and the
        # slower prompts are accepted. Set from `capabilities().supports_
        # tools` by the Session — noted here so nobody removes the tools and
        # quietly breaks the guarantee.
        self.force_epoch_per_write = force_epoch_per_write
        self._text_lookups = 0
        self._syntax_corrected = False

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
        self.store.put_file(path, lang, text, symbols, edges, unresolved)
        return len(symbols)

    def index_project(self, *, limit: int = 2000) -> CodemapStats:
        """Index everything indexable. `.git/` is excluded (M27)."""
        skip = (".git/", "__pycache__/", "node_modules/", ".venv/", "venv/",
                "target/", "build/", "dist/", ".cc_snapshots/",
                ".atk_snapshots/", ".cc_journal/", ".python/", ".tools/")
        n = 0
        try:
            paths = self.fs.list("*")
        except Exception:                                # noqa: BLE001
            paths = []
        for path in sorted(paths):
            norm = str(path).replace("\\", "/")
            if any(part in norm for part in skip):
                continue
            if not langs.id_for_path(norm):
                continue
            self.index_file(norm)
            n += 1
            if n >= limit:
                break
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
        """
        if lang_id == "python":
            _s, _e, unresolved = parse_python.parse(text, "<generated>")
        else:
            _s, _e, unresolved = parse_regex.parse(text, "<generated>",
                                                   lang_id)
        local = {s.name for s in
                 (parse_python.parse(text, "<generated>")[0]
                  if lang_id == "python"
                  else parse_regex.parse(text, "<generated>", lang_id)[0])}
        # Names reached THROUGH an import are not invented — `csv.reader` in
        # a file that says `import csv` is the standard library doing its
        # job. Flagging it would make this check noise instead of signal,
        # and a check that cries wolf is a check somebody turns off.
        if lang_id == "python":
            imported = set(parse_python.imports_of(text))
        else:
            imported = set(parse_regex.imports_of(text, lang_id))
        import_heads = {str(i).lstrip(".").split(".")[0] for i in imported}
        import_heads |= {str(i).lstrip(".").rsplit(".", 1)[-1]
                         for i in imported}

        # Names BOUND in this file — locals, parameters, loop variables — as
        # opposed to symbols it exports. `screen.fill(...)` where `screen` came
        # from `pygame.display.set_mode()` is an attribute on a runtime object,
        # and no static check can say whether `.fill` exists on it. Reporting
        # it as "a name this project does not define" is both untrue and
        # noisy: it fired on nearly every generated file, in the same sentence
        # as the genuinely missing names, which is how `TrackSegment` and
        # `render_road` went unnoticed until they became ImportErrors.
        bound = (parse_python.bound_names(text) if lang_id == "python"
                 else set())

        out: list[str] = []
        builtins = _BUILTINS.get(lang_id, set())
        for _src, name, _kind in unresolved:
            raw = str(name)
            short = raw.split(".")[-1]
            head = raw.split(".")[0]
            if head in import_heads or raw in imported:
                continue
            if short in local or short in builtins or raw in local:
                continue
            if head in local or head in builtins:
                continue      # a method on something defined here
            if "." in raw and head in bound:
                continue      # attribute on a local object — unknowable, and
                              # not a claim this check is entitled to make
            if self.store.resolves(raw) or self.store.resolves(short):
                continue
            if raw not in out:
                out.append(raw)
        return out

    def blast_radius(self, symbol: str, depth: int = 2) -> dict:
        return self.store.blast_radius(symbol, depth)

    def architecture(self, target: str = "", max_tokens: int = 4096,
                     count_tokens=None) -> str:
        return zoom.generate_architecture_context(
            self.store, target, max_tokens=max_tokens,
            count_tokens=count_tokens)

    def prefix_block(self, target: str = "") -> str:
        """The stable, cacheable architecture block (G.7.1)."""
        return zoom.architecture_prefix(self.store, target=target)

    def tail_blocks(self, target: str, *, count_tokens=None) -> list[str]:
        """The volatile tail: interfaces, examples, staleness (G.7.1)."""
        return [b for b in (
            zoom.dependency_interfaces(self.store, target,
                                       count_tokens=count_tokens),
            zoom.similar_examples(self.store, target),
            zoom.staleness_note(self.store)) if b]

    def maybe_bump_epoch(self, **why: Any) -> int:
        bump, reason = zoom.should_bump_epoch(self.store, **why)
        if not bump and self.force_epoch_per_write \
                and self.store.changed_since_epoch():
            # The text-marker fallback path (M31, G.7's closing paragraph):
            # with no live tools the summary has no safety net, so it is not
            # allowed to lag at all.
            bump = True
            reason = ("this model has no tool calling, so the architecture "
                      "summary is rebuilt on every write rather than being "
                      "allowed to lag")
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
                return self._tool_slice(str(args.get("path", "")),
                                        int(args.get("start", 1) or 1),
                                        int(args.get("end", 0) or 0))
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
        try:
            text = self.fs.read(path)
        except Exception:                                # noqa: BLE001
            return (f"`{path}` is not in this project. Use list_symbols on a "
                    f"path that is, or search_codemap to find where "
                    f"something lives.")
        lines = text.splitlines()
        start = max(1, start)
        end = min(len(lines), end or (start + 60))
        if start > len(lines):
            return f"`{path}` has only {len(lines)} lines."
        body = "\n".join(f"{n:>5} | {lines[n - 1]}"
                         for n in range(start, end + 1))
        tail = ("" if end >= len(lines)
                else f"\n… {len(lines) - end} more lines follow.")
        return f"{path} lines {start}-{end} of {len(lines)}:\n{body}{tail}"

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
        """Called at the start of each generation. The cap is PER generation."""
        self._text_lookups = 0
        self._syntax_corrected = False

    def parse_text_lookups(self, text: str) -> list[tuple[str, str]]:
        """Accept several syntaxes, because drift is guaranteed (D10).

        `[SEARCH_CODEMAP: x]`, `SEARCH_CODEMAP(x)`,
        `<search_codemap>x</search_codemap>` all mean the same thing, and a
        model that has drifted between them is not confused about intent.
        """
        import re
        found: list[tuple[str, str]] = []
        for tool in ("search_codemap", "list_symbols", "read_slice"):
            up = tool.upper()
            for pattern in (rf"\[{up}:\s*([^\]]+)\]",
                            rf"\b{up}\(\s*['\"]?([^)'\"]+)['\"]?\s*\)",
                            rf"<{tool}>\s*(.*?)\s*</{tool}>"):
                for m in re.finditer(pattern, text, re.I | re.S):
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
            if self._looks_like_broken_call(text) and not self._syntax_corrected:
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
                replies.append(self._tool_slice(
                    parts[0], int(parts[1]) if len(parts) > 1 else 1,
                    int(parts[2]) if len(parts) > 2 else 0))
        return "\n\n".join(replies)

    @staticmethod
    def _looks_like_broken_call(text: str) -> bool:
        import re
        return bool(re.search(r"search[_ ]codemap|list[_ ]symbols",
                              text or "", re.I))

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
                   "parseFloat", "setTimeout", "fetch"},
    "gdscript": {"print", "printerr", "push_error", "load", "preload",
                 "range", "len", "str", "int", "float", "Vector2", "Vector3",
                 "get_node", "emit_signal", "connect", "is_instance_valid"},
}
_BUILTINS["typescript"] = _BUILTINS["javascript"]

__all__ = ["CodeMap", "Store", "zoom", "parse_python", "parse_regex",
           "parse_treesitter", "Symbol", "Diagnostic", "MAX_TEXT_LOOKUPS"]
