# SPDX-License-Identifier: Apache-2.0
"""Choosing what the model gets to see — and saying what it didn't.

The temptation with any coding assistant is to paste the whole file, or the
whole folder. With a local model at a real context budget that is not a
trade-off, it is a wall: the file doesn't fit, and if it does, the part that
matters is buried among the parts that don't. **A small model's attention is
the scarce resource in this entire engine**, and this module is how it gets
spent.

WHAT IT DOES, and why each piece earns its place:

  * **Outline before body.** A list of the symbols in a file — every function
    and class with its signature and line — costs a fraction of the file and
    answers most "what exists here" questions outright. Models are far better
    at "here are the 30 functions, now write one" than at reading 2,000 lines
    and inferring the same list.
  * **Slices, not files.** When one function is being changed, send that
    function, its immediate neighbours, and the imports — not the module.
  * **A hard budget, and SAY WHAT WAS CUT.** Every assembled context ends
    with an explicit `NOT INCLUDED` block naming what was dropped (M28).

**The omission notice is not politeness.** A model that doesn't know something
was withheld will invent its contents — confidently, and in a way that looks
like knowledge. This is the cheapest defence against D4 in the whole engine.

THE BUDGET IS MEASURED, NOT CONFIGURED (G.3). `measure_budget()` derives the
usable prompt size from what the LLMPort actually reports:

    usable = context_tokens − reserved_output − reasoning_tax − safety_margin

Taking it from a config file means the day someone loads a model with a
smaller context than the config claims, the failure is a truncated prompt that
looks like the model getting stupid. Asking the port costs one call.

Symbol extraction is exact for Python (`ast`) and regex-based for everything
else. **The regex version is honestly approximate** — it looks for
declarations at the start of a line — and it says so wherever it surfaces
rather than pretending to parse C++.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

from .types import Symbol

# Characters, not tokens, for the fallback path: tokenisers differ per model
# and this has to work with no model loaded at all. ~3.5–4 chars/token is the
# rule of thumb used consistently across this project, and it is STATED
# wherever it is used rather than hidden (M14).
CHARS_PER_TOKEN = 4
DEFAULT_BUDGET_CHARS = 24_000

# What is held back from the prompt so the model has room to answer (G.8).
RESERVED_OUTPUT_TOKENS = 2048
# Reasoning models spend tokens on `<think>` before saying anything useful
# (D13). Budgeting as if they don't is how a prompt overflows on Magistral
# after working fine on Devstral.
REASONING_TAX_TOKENS = 1024
# Nothing is exact — chat templates add tokens, and an estimated count can be
# wrong by a few percent. 5% back beats a hard failure at the boundary.
SAFETY_FRACTION = 0.05


@dataclass(frozen=True)
class Budget:
    """What actually fits, and how confident we are about the number."""
    prompt_tokens: int
    context_tokens: int
    reserved_output: int
    is_estimate: bool
    note: str = ""

    @property
    def prompt_chars(self) -> int:
        return self.prompt_tokens * CHARS_PER_TOKEN

    def declare(self) -> str:
        """One line for the prompt when the count is a guess (M14).

        A model told the budget is approximate can be asked to be concise; a
        model whose prompt is silently truncated cannot do anything at all.
        """
        if not self.is_estimate:
            return ""
        return ("Note: token counts here are estimated, not exact, so this "
                "context may be slightly larger or smaller than it appears.")


def measure_budget(llm: Any, *, reserved_output: int = RESERVED_OUTPUT_TOKENS,
                   reasoning: bool = False) -> Budget:
    """Ask the port what fits. Never take this from config (G.3).

    A "no model loaded" port reports `context_tokens` anyway (its configured
    default), which is the right behaviour: the budget question has an answer
    even when the model question does not.
    """
    try:
        caps = llm.capabilities()
    except Exception:                                    # noqa: BLE001
        return Budget(prompt_tokens=DEFAULT_BUDGET_CHARS // CHARS_PER_TOKEN,
                      context_tokens=0, reserved_output=reserved_output,
                      is_estimate=True,
                      note="the model port could not be asked how much "
                           "context it has, so a conservative default is in "
                           "use")
    total = int(getattr(caps, "context_tokens", 0) or 0)
    if total <= 0:
        total = 8192
    tax = REASONING_TAX_TOKENS if reasoning else 0
    usable = total - reserved_output - tax
    usable -= int(usable * SAFETY_FRACTION)
    return Budget(prompt_tokens=max(512, usable), context_tokens=total,
                  reserved_output=reserved_output,
                  is_estimate=bool(getattr(caps, "token_count_is_estimate",
                                           True)),
                  note=(f"measured from the loaded model: {total} tokens of "
                        f"context, {reserved_output} reserved for the answer"
                        + (f", {tax} for reasoning" if tax else "")))


# ---------------------------------------------------------------------------
# outlines
# ---------------------------------------------------------------------------

# Declaration patterns per language family. **Approximate by design** — a full
# parser for six languages is not worth the maintenance, and an outline that
# is 95% right is infinitely more useful than no outline. Everything produced
# from these is labelled `approximate=True` and the label survives to the
# prompt.
_DECL: dict[str, str] = {
    "c": r"^(?:[\w*\s]+?)\b(?P<name>\w+)\s*\([^;]*\)\s*\{",
    "cpp": r"^(?:template<[^>]*>\s*)?(?:[\w:*&<>,\s]+?)\b(?P<name>[\w:~]+)"
           r"\s*\([^;]*\)\s*(?:const\s*)?\{|^[ \t]*(?:class|struct)\s+(?P<cls>\w+)",
    "rust": r"^[ \t]*(?:pub\s+)?(?:async\s+)?fn\s+(?P<name>\w+)|"
            r"^[ \t]*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(?P<cls>\w+)",
    "java": r"^[ \t]*(?:public|private|protected|static|final|abstract|\s)*"
            r"(?:[\w<>\[\],\s]+\s+)?(?P<name>\w+)\s*\([^)]*\)\s*\{|"
            r"^[ \t]*(?:public\s+)?(?:class|interface|enum)\s+(?P<cls>\w+)",
    "go": r"^func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)|^type\s+(?P<cls>\w+)",
    "javascript": r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)|"
                  r"^[ \t]*(?:export\s+)?class\s+(?P<cls>\w+)|"
                  r"^[ \t]*(?:const|let|var)\s+(?P<name2>\w+)\s*=\s*"
                  r"(?:async\s*)?\(",
    "csharp": r"^[ \t]*(?:public|private|protected|internal|static|\s)*"
              r"(?:[\w<>\[\],\s]+\s+)?(?P<name>\w+)\s*\([^)]*\)\s*\{|"
              r"^[ \t]*(?:public\s+)?(?:class|struct|interface)\s+(?P<cls>\w+)",
    "zig": r"^[ \t]*(?:pub\s+)?fn\s+(?P<name>\w+)",
    "ruby": r"^[ \t]*def\s+(?P<name>[\w.?!]+)|^[ \t]*class\s+(?P<cls>\w+)",
    "lua": r"^[ \t]*(?:local\s+)?function\s+(?P<name>[\w.:]+)",
    "bash": r"^[ \t]*(?:function\s+)?(?P<name>\w+)\s*\(\)\s*\{",
    "powershell": r"^[ \t]*function\s+(?P<name>[\w-]+)",
    # GDScript is indentation-scoped like Python, so end-of-body is derivable
    # rather than guessed (§6.1a).
    "gdscript": r"^[ \t]*(?:@\w+\s+)*(?:static\s+)?func\s+(?P<name>\w+)|"
                r"^[ \t]*class_name\s+(?P<cls>\w+)|^[ \t]*class\s+(?P<cls2>\w+)|"
                r"^[ \t]*signal\s+(?P<name2>\w+)",
}
_DECL["typescript"] = _DECL["javascript"]

_KEYWORDS = {"if", "for", "while", "switch", "return", "catch", "else",
             "do", "try", "match", "elif", "with"}


def symbols(text: str, lang_id: str = "python") -> list[Symbol]:
    """Every top-level definition, with its line. Exact for Python."""
    if lang_id == "python":
        return _python_symbols(text)
    if lang_id == "gdscript":
        return _indent_symbols(text, _DECL["gdscript"])
    pattern = _DECL.get((lang_id or "").lower())
    if not pattern:
        return []
    out: list[Symbol] = []
    for m in re.finditer(pattern, text, re.M):
        groups = m.groupdict()
        name = (groups.get("name") or groups.get("cls")
                or groups.get("name2") or groups.get("cls2") or "")
        if not name or name in _KEYWORDS:
            continue
        kind = "class" if (groups.get("cls") or groups.get("cls2")) else \
            "function"
        line = text[:m.start()].count("\n") + 1
        out.append(Symbol(
            name=name, kind=kind, line=line,
            signature=m.group(0).strip().rstrip("{").strip()[:120],
            approximate=True))
    return out


def _indent_symbols(text: str, pattern: str) -> list[Symbol]:
    """Indentation-scoped languages: end-of-body is derivable, not guessed."""
    lines = text.splitlines()
    found: list[Symbol] = []
    for m in re.finditer(pattern, text, re.M):
        groups = m.groupdict()
        name = (groups.get("name") or groups.get("cls")
                or groups.get("cls2") or groups.get("name2") or "")
        if not name:
            continue
        line = text[:m.start()].count("\n") + 1
        indent = len(lines[line - 1]) - len(lines[line - 1].lstrip())
        end = line
        for n in range(line, len(lines)):
            body = lines[n]
            if body.strip() and (len(body) - len(body.lstrip())) <= indent:
                break
            end = n + 1
        kind = "class" if (groups.get("cls") or groups.get("cls2")) \
            else ("signal" if groups.get("name2") else "function")
        found.append(Symbol(name=name, kind=kind, line=line, end_line=end,
                            signature=lines[line - 1].strip()[:120],
                            approximate=True))
    return found


def _python_symbols(text: str) -> list[Symbol]:
    """Exact, via the codemap's `ast` extractor — ONE implementation.

    This used to be a second, simpler version living here, and the two had
    drifted: this one dropped type annotations and return types, so
    `def load(path: str) -> list` surfaced in an outline as `def load(path)`.
    That is precisely the information F9 exists to hand the model verbatim —
    a signature without its types is the thing a model hallucinates around —
    and having two extractors meant the better one was only used by half the
    engine. There is now one.
    """
    try:
        ast.parse(text)
    except SyntaxError:
        # A file mid-edit doesn't parse — and an outline is most wanted
        # exactly when the file is broken. Falling back beats returning [].
        return [Symbol(name=m.group(1), kind="function",
                       line=text[:m.start()].count("\n") + 1,
                       signature=m.group(0).strip(), approximate=True)
                for m in re.finditer(r"^[ \t]*(?:async\s+)?def\s+(\w+)\s*\(",
                                     text, re.M)]
    from .codemap import parse_python
    symbols_found, _edges, _unresolved = parse_python.parse(text)
    return symbols_found


def outline(text: str, lang_id: str = "python", path: str = "") -> str:
    """The cheap view of a file: what is in it, and where."""
    syms = symbols(text, lang_id)
    header = f"# outline of {path or 'file'} ({len(text.splitlines())} lines)"
    if not syms:
        return header + "\n# (no top-level definitions found)"
    approx = ""
    if any(s.approximate for s in syms):
        approx = ("\n# NOTE: this outline is pattern-matched, not parsed — "
                  "it can miss unusual declarations.")
    return header + approx + "\n" + "\n".join(s.one_line() for s in syms)


def interface(text: str, lang_id: str = "python", path: str = "") -> str:
    """A file's public surface: signatures and one-line docs, no bodies.

    F9, and on the target machine the highest-value item in that appendix: a
    dependency costs ~30 tokens as an interface and ~800 as a file. This is
    what header files have always been for, and it is the reason a five-file
    project fits in context at all.

    Two benefits, both large: the context cost of a dependency becomes small
    and predictable, and the model cannot hallucinate around a signature it
    has been handed verbatim.
    """
    syms = [s for s in symbols(text, lang_id)
            if not s.name.split(".")[-1].startswith("_")]
    if not syms:
        return f"# {path or 'file'}: no public symbols"
    lines = [f"# interface of {path or 'file'} — signatures only, no bodies"]
    for s in syms:
        doc = f"  # {s.docstring}" if s.docstring else ""
        lines.append(f"{s.signature or s.name}{doc}")
    if any(s.approximate for s in syms):
        lines.append("# (pattern-matched, so it may be incomplete)")
    return "\n".join(lines)


def slice_around(text: str, line: int, before: int = 40,
                 after: int = 80) -> str:
    """The region around a line, WITH line numbers so an edit can be located."""
    lines = text.splitlines()
    lo = max(0, line - 1 - before)
    hi = min(len(lines), line + after)
    return "\n".join(f"{n + 1:>5} | {lines[n]}" for n in range(lo, hi))


def symbol_body(text: str, name: str, lang_id: str = "python") -> str:
    """One function or class, whole. The unit an edit usually operates on."""
    for s in symbols(text, lang_id):
        if s.name == name or s.name.endswith("." + name):
            lines = text.splitlines()
            end = s.end_line or _guess_end(lines, s.line)
            return "\n".join(f"{n:>5} | {lines[n - 1]}"
                             for n in range(s.line, min(end, len(lines)) + 1))
    return ""


def _guess_end(lines: Sequence[str], start: int) -> int:
    """Where a brace-delimited body ends. Approximate, and bounded."""
    depth = 0
    seen = False
    for n in range(start - 1, min(len(lines), start + 400)):
        depth += lines[n].count("{") - lines[n].count("}")
        if "{" in lines[n]:
            seen = True
        if seen and depth <= 0:
            return n + 1
    return min(len(lines), start + 60)


# ---------------------------------------------------------------------------
# assembling, under a budget, with the omissions declared
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """One candidate block. Lower `priority` is kept first."""
    label: str
    text: str
    priority: int = 5
    essential: bool = False      # dropped only if it alone exceeds the budget


def build_context(pieces: Sequence[Piece | tuple], budget: int | Budget,
                  *, count_tokens=None) -> str:
    """Assemble under a budget, saying explicitly what was dropped (M28).

    `count_tokens` is `LLMPort.count_tokens` when the caller has a port; the
    character estimate is used otherwise, and either way the assumption is
    declared in the output when it is a guess.
    """
    limit_chars, declare = _limit(budget)
    items = [p if isinstance(p, Piece) else Piece(*p) for p in pieces]
    ordered = sorted(items, key=lambda p: (0 if p.essential else 1,
                                           p.priority))

    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    for piece in ordered:
        block = f"--- {piece.label} ---\n{(piece.text or '').rstrip()}\n"
        cost = (count_tokens(block) * CHARS_PER_TOKEN if count_tokens
                else len(block))
        if used + cost > limit_chars and kept:
            dropped.append(piece.label)
            continue
        kept.append(block)
        used += cost

    out = "\n".join(kept)
    # The omissions block ALWAYS ends the context (M28) — including when
    # nothing was dropped, because "nothing was omitted" is itself
    # information the model can rely on, and a block that appears only
    # sometimes is a block the model learns to ignore.
    out += "\n--- NOT INCLUDED ---\n"
    if dropped:
        out += ("The following were left out to fit the context window: "
                + ", ".join(dropped)
                + ".\nIf you need any of them, say so or look them up "
                  "instead of guessing at their contents.\n")
    else:
        out += "Nothing was left out; this is everything relevant.\n"
    if declare:
        out += declare + "\n"
    return out


def _limit(budget: int | Budget) -> tuple[int, str]:
    if isinstance(budget, Budget):
        return budget.prompt_chars, budget.declare()
    return int(budget), ""


def project_map(fs: Any, max_files: int = 200) -> str:
    """A tree of the code files, with sizes. The "where am I" view.

    Skips the noise every project accumulates (`.git`, build outputs, venvs,
    our own snapshots) because listing them spends context on nothing. And it
    says how many it did not show, rather than quietly stopping (M28's
    principle, applied to a smaller thing).
    """
    from . import langs
    skip = {".git", "__pycache__", "node_modules", "target", "build", "dist",
            ".venv", "venv", "envs", ".cc_snapshots", ".atk_snapshots",
            ".idea", ".vscode", "obj", "bin", ".python", ".tools"}
    rows: list[str] = []
    total = 0
    try:
        paths = fs.list("*")
    except Exception:                                    # noqa: BLE001
        paths = []
    for path in sorted(paths):
        parts = str(path).replace("\\", "/").split("/")
        if any(part in skip for part in parts):
            continue
        if langs.for_extension(path) is None:
            continue
        total += 1
        if len(rows) >= max_files:
            continue
        try:
            n = len(fs.read(path).splitlines())
        except Exception:                                # noqa: BLE001
            n = 0
        rows.append(f"  {path}  ({n} lines)")
    head = f"# {total} code file(s)"
    if total > len(rows):
        head += f" — showing the first {len(rows)}"
    return head + "\n" + "\n".join(rows)


# Words that carry no signal about WHICH file matters. Without this, "fix
# the parser" matches every file containing "the" — and "the" is a substring
# of "other", "there" and "gather", so the ranking fills with noise and the
# two files that actually matter get pushed off the end. A three-character
# minimum is not enough on its own.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "his",
    "her", "was", "are", "not", "but", "can", "you", "all", "any", "how",
    "why", "who", "its", "our", "use", "using", "add", "adds", "make",
    "makes", "please", "should", "would", "could", "need", "needs", "want",
    "wants", "let", "lets", "new", "old", "get", "gets", "set", "sets",
    "when", "then", "than", "some", "more", "most", "also", "just", "only",
    "have", "has", "had", "will", "does", "did", "done", "one", "two",
    "code", "file", "files", "function", "method", "class", "module",
}


def relevant_files(fs: Any, query: str, limit: int = 6) -> list[str]:
    """Files most likely to matter, ranked by simple overlap.

    Deliberately dumb: no embeddings, no index to keep fresh. It scores on
    filename and identifier overlap, which is usually enough to put the right
    two or three files in front of a model, and it costs nothing. The codemap
    (§6.7) is the sharp instrument; this is the one that works before the
    codemap has been built.
    """
    from . import langs
    words = {w.lower() for w in re.findall(r"\w{3,}", query or "")}
    words -= _STOPWORDS
    if not words:
        return []
    skip = {".git", "__pycache__", "node_modules", "target", ".venv", "venv",
            "envs", ".cc_snapshots", "build", "dist"}
    scored: list[tuple[int, str]] = []
    try:
        paths = fs.list("*")
    except Exception:                                    # noqa: BLE001
        return []
    for path in paths:
        parts = str(path).replace("\\", "/").split("/")
        if any(part in skip for part in parts):
            continue
        if langs.for_extension(path) is None:
            continue
        try:
            text = fs.read(path)
        except Exception:                                # noqa: BLE001
            continue
        name = parts[-1].lower()
        name_hits = sum(2 for w in words if w in name)
        body = text.lower()
        body_hits = sum(1 for w in words if w in body)
        if name_hits or body_hits:
            scored.append((name_hits * 3 + body_hits, path))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [name for _, name in scored[:limit]]
