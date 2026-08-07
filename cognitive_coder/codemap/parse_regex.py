# SPDX-License-Identifier: Apache-2.0
"""A ctags-style signature extractor for everything Python's `ast` can't do.

Zero dependencies, and **honestly approximate**. It looks for declarations at
the start of a line; it does not parse. Every symbol it produces carries
`approximate=True`, and that flag survives all the way to the prompt, where
the outline says *"pattern-matched, not parsed — it can miss unusual
declarations."* An outline that is 95% right is far more useful than no
outline; an outline that CLAIMS to be 100% right and isn't is worse than
either.

`parse_treesitter.py` is strictly better for C/C++/Rust/JS and strictly
optional (C7). When `tree_sitter` is importable this module is the fallback;
when it isn't, this is the whole story, and the engine says which mode it is
in rather than leaving the operator to wonder.

One anchoring rule, learned by getting it wrong: **patterns anchor with
`^[ \\t]*`, never `^\\s*`.** In multiline mode `\\s` matches newlines, so
`^\\s*func` happily starts matching on the blank line above and every line
number comes out one too low — a bug that is invisible until someone tries to
open the file at the reported line.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from ..types import Symbol

_KEYWORDS = {"if", "for", "while", "switch", "return", "catch", "else", "do",
             "try", "match", "elif", "with", "case", "defer", "go", "select"}

# (pattern, kind-hint). Named groups: name / cls / name2 / cls2.
_DECL: dict[str, str] = {
    "c": r"^(?:[\w*\s]+?)\b(?P<name>\w+)\s*\([^;]*\)\s*\{",
    "cpp": r"^(?:template<[^>]*>[ \t]*)?(?:[\w:*&<>,\s]+?)\b(?P<name>[\w:~]+)"
           r"\s*\([^;]*\)\s*(?:const\s*)?\{|"
           r"^[ \t]*(?:class|struct)\s+(?P<cls>\w+)",
    "rust": r"^[ \t]*(?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
            r"fn\s+(?P<name>\w+)|"
            r"^[ \t]*(?:pub\s+)?(?:struct|enum|trait|impl)\s+(?P<cls>\w+)",
    "java": r"^[ \t]*(?:public|private|protected|static|final|abstract|[ \t])*"
            r"(?:[\w<>\[\],\s]+\s+)?(?P<name>\w+)\s*\([^)]*\)\s*\{|"
            r"^[ \t]*(?:public\s+)?(?:class|interface|enum)\s+(?P<cls>\w+)",
    "go": r"^func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)|^type\s+(?P<cls>\w+)",
    "javascript": r"^[ \t]*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
                  r"function\s*\*?\s*(?P<name>\w+)|"
                  r"^[ \t]*(?:export\s+)?(?:default\s+)?class\s+(?P<cls>\w+)|"
                  r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+(?P<name2>\w+)"
                  r"\s*=\s*(?:async\s*)?(?:\(|function)",
    "csharp": r"^[ \t]*(?:public|private|protected|internal|static|async|[ \t])*"
              r"(?:[\w<>\[\],\s]+\s+)?(?P<name>\w+)\s*\([^)]*\)\s*\{|"
              r"^[ \t]*(?:public\s+)?(?:class|struct|interface|record)\s+"
              r"(?P<cls>\w+)",
    "zig": r"^[ \t]*(?:pub\s+)?fn\s+(?P<name>\w+)|"
           r"^[ \t]*(?:pub\s+)?const\s+(?P<cls>\w+)\s*=\s*(?:struct|enum|union)",
    "ruby": r"^[ \t]*def\s+(?P<name>[\w.?!]+)|"
            r"^[ \t]*(?:class|module)\s+(?P<cls>\w+)",
    "lua": r"^[ \t]*(?:local\s+)?function\s+(?P<name>[\w.:]+)",
    "bash": r"^[ \t]*(?:function\s+)?(?P<name>\w+)\s*\(\)\s*\{",
    "powershell": r"^[ \t]*function\s+(?P<name>[\w-]+)",
    "sql": r"^[ \t]*CREATE\s+(?:TABLE|VIEW|INDEX|TRIGGER)\s+"
           r"(?:IF\s+NOT\s+EXISTS\s+)?(?P<cls>\w+)",
    # GDScript (§6.1a). Indentation-scoped like Python, so the end of a body
    # is DERIVABLE rather than guessed — which is why it gets its own path
    # below rather than sharing the brace-counting one.
    "gdscript": r"^[ \t]*(?:@\w+(?:\([^)]*\))?[ \t]+)*(?:static[ \t]+)?"
                r"func[ \t]+(?P<name>\w+)|"
                r"^[ \t]*class_name[ \t]+(?P<cls>\w+)|"
                r"^[ \t]*class[ \t]+(?P<cls2>\w+)|"
                r"^[ \t]*signal[ \t]+(?P<name2>\w+)",
}
_DECL["typescript"] = _DECL["javascript"]
_DECL["python"] = (r"^[ \t]*(?:async[ \t]+)?def[ \t]+(?P<name>\w+)|"
                   r"^[ \t]*class[ \t]+(?P<cls>\w+)")

_INDENT_SCOPED = {"python", "gdscript"}

# Call sites, for the (approximate) call graph. Deliberately crude: a bare
# `name(` at a plausible position. It over-reports slightly, which for blast
# radius is the safe direction — a false caller costs one extra file read; a
# missed caller costs a broken build nobody predicted.
_CALL = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*\(")

_IMPORT: dict[str, str] = {
    "c": r'^[ \t]*#\s*include\s*[<"](?P<name>[^">]+)[">]',
    "cpp": r'^[ \t]*#\s*include\s*[<"](?P<name>[^">]+)[">]',
    "rust": r"^[ \t]*(?:pub\s+)?use\s+(?P<name>[\w:]+)",
    "java": r"^[ \t]*import\s+(?:static\s+)?(?P<name>[\w.]+)",
    "go": r'^[ \t]*(?:import\s+)?"(?P<name>[\w./-]+)"',
    "javascript": r"^[ \t]*import\s+.*?from\s+['\"](?P<name>[^'\"]+)['\"]|"
                  r"require\(['\"](?P<name2>[^'\"]+)['\"]\)",
    "csharp": r"^[ \t]*using\s+(?P<name>[\w.]+)\s*;",
    "python": r"^[ \t]*(?:from\s+(?P<name>[\w.]+)\s+import|"
              r"import\s+(?P<name2>[\w.]+))",
    "gdscript": r"^[ \t]*(?:extends[ \t]+(?P<name>[\w.\"'/:]+)|"
                r"(?:const|var)\s+\w+\s*(?::=|=)\s*(?:preload|load)"
                r"\(['\"](?P<name2>[^'\"]+)['\"]\))",
    "lua": r"require\s*\(?['\"](?P<name>[\w.]+)['\"]",
    "ruby": r"^[ \t]*require(?:_relative)?\s+['\"](?P<name>[^'\"]+)['\"]",
}
_IMPORT["typescript"] = _IMPORT["javascript"]


def parse(text: str, path: str = "",
          lang_id: str = "") -> tuple[list[Symbol], list[tuple], list[tuple]]:
    """(symbols, edges, unresolved) — the same contract as `parse_python`."""
    lang = (lang_id or "").lower()
    pattern = _DECL.get(lang)
    module = _module_of(path)
    if not pattern:
        return [], [], []

    lines = text.splitlines()
    symbols: list[Symbol] = []
    for m in re.finditer(pattern, text, re.M):
        groups = m.groupdict()
        name = (groups.get("name") or groups.get("cls")
                or groups.get("name2") or groups.get("cls2") or "")
        if not name or name in _KEYWORDS:
            continue
        line = text[:m.start()].count("\n") + 1
        kind = ("class" if (groups.get("cls") or groups.get("cls2"))
                else ("signal" if groups.get("name2") and lang == "gdscript"
                      else "function"))
        end = (_indent_end(lines, line) if lang in _INDENT_SCOPED
               else _brace_end(lines, line))
        symbols.append(Symbol(
            name=name, kind=kind, line=line, end_line=end,
            signature=(lines[line - 1].strip().rstrip("{").strip()[:160]
                       if line - 1 < len(lines) else name),
            path=path, approximate=True))

    edges: list[tuple] = []
    for name in imports_of(text, lang):
        edges.append((module, name, "imports"))

    # Attribute calls to the enclosing symbol by line span, so blast radius
    # points at a function rather than at a file.
    own = {s.name for s in symbols}
    unresolved: list[tuple] = []
    for m in _CALL.finditer(text):
        called = m.group("name")
        if called in _KEYWORDS or called in ("print", "printf", "return"):
            continue
        line = text[:m.start()].count("\n") + 1
        src = _enclosing(symbols, line) or module
        if called == src:
            continue
        if called in own:
            edges.append((src, called, "calls"))
        else:
            unresolved.append((src, called, "calls"))
    return symbols, edges, unresolved


def imports_of(text: str, lang_id: str = "") -> list[str]:
    pattern = _IMPORT.get((lang_id or "").lower())
    if not pattern:
        return []
    out: list[str] = []
    for m in re.finditer(pattern, text, re.M):
        name = m.groupdict().get("name") or m.groupdict().get("name2")
        if name:
            out.append(name)
    return out


def _enclosing(symbols: Iterable[Symbol], line: int) -> str:
    best = ""
    best_line = -1
    for s in symbols:
        if s.line <= line <= (s.end_line or s.line) and s.line > best_line:
            best, best_line = s.name, s.line
    return best


def _indent_end(lines: list[str], line: int) -> int:
    """End of an indentation-scoped body. Derivable, so derive it."""
    if line - 1 >= len(lines):
        return line
    head = lines[line - 1]
    indent = len(head) - len(head.lstrip())
    end = line
    for n in range(line, len(lines)):
        body = lines[n]
        if body.strip() and (len(body) - len(body.lstrip())) <= indent:
            break
        if body.strip():
            end = n + 1
    return end


def _brace_end(lines: list[str], line: int) -> int:
    """End of a brace-delimited body. Approximate, and BOUNDED.

    Bounded on purpose: an unbalanced brace in a file mid-edit would
    otherwise walk this to the end of the file and report one symbol
    swallowing everything.
    """
    depth = 0
    seen = False
    for n in range(line - 1, min(len(lines), line + 400)):
        depth += lines[n].count("{") - lines[n].count("}")
        if "{" in lines[n]:
            seen = True
        if seen and depth <= 0:
            return n + 1
    return min(len(lines), line + 60)


def _module_of(path: str) -> str:
    p = str(path or "").replace("\\", "/").strip("/")
    return p or "<file>"
