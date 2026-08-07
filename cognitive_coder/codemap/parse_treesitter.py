# SPDX-License-Identifier: Apache-2.0
"""Optional tree-sitter parsing — strictly better than regex, strictly optional.

For C, C++, Rust and JavaScript a real parse beats pattern-matching by a wide
margin: it gets nested classes right, it does not trip over a declaration
inside a string, and the symbol spans are exact rather than brace-counted. So
it is used when `tree_sitter` and a grammar are importable.

And it is **optional** (C7, M6). The import is inside `available()`, never at
module level (M48), so a machine without it gets the regex extractor and a
sentence saying what that costs — approximate outlines for C/C++/Rust/JS —
rather than an ImportError from `import cognitive_coder`.

`degraded_note()` is what the installer summary and `ccoder doctor` print. The
operator must be told which mode they are in; a silently worse outline is the
kind of thing that gets blamed on the model.
"""

from __future__ import annotations

from typing import Any

from ..types import Symbol

# Tree-sitter node types that carry a definition, per language. Keeping this
# as data means adding a language is a dictionary entry, not a code path.
_DEFS: dict[str, dict[str, str]] = {
    "c": {"function_definition": "function", "struct_specifier": "struct",
          "enum_specifier": "enum", "type_definition": "type"},
    "cpp": {"function_definition": "function", "class_specifier": "class",
            "struct_specifier": "struct", "namespace_definition": "namespace"},
    "rust": {"function_item": "function", "struct_item": "struct",
             "enum_item": "enum", "trait_item": "trait", "impl_item": "impl",
             "mod_item": "module"},
    "javascript": {"function_declaration": "function",
                   "class_declaration": "class",
                   "method_definition": "method",
                   "generator_function_declaration": "function"},
    "typescript": {"function_declaration": "function",
                   "class_declaration": "class",
                   "method_definition": "method",
                   "interface_declaration": "interface"},
    "go": {"function_declaration": "function",
           "method_declaration": "method", "type_declaration": "type"},
    "java": {"method_declaration": "method", "class_declaration": "class",
             "interface_declaration": "interface"},
}

_LANG_ALIASES = {"cpp": "cpp", "c": "c", "rust": "rust",
                 "javascript": "javascript", "typescript": "typescript",
                 "go": "go", "java": "java"}

_cache: dict[str, Any] = {}


def available(lang_id: str) -> bool:
    """Is a real parser present for this language right now?

    Cached because the answer cannot change within a process and the import
    is not free. Any failure — no package, no grammar, an API change between
    tree-sitter versions — is False, which routes to the regex extractor.
    """
    key = _LANG_ALIASES.get((lang_id or "").lower(), "")
    if not key:
        return False
    if key in _cache:
        return _cache[key] is not None
    parser = None
    try:                                          # tree-sitter-languages
        from tree_sitter_languages import get_parser  # noqa: PLC0415
        parser = get_parser(key)
    except Exception:                             # noqa: BLE001
        try:                                      # tree-sitter >= 0.22 shape
            import importlib

            import tree_sitter  # noqa: PLC0415
            mod = importlib.import_module(f"tree_sitter_{key}")
            language = tree_sitter.Language(mod.language())
            parser = tree_sitter.Parser(language)
        except Exception:                         # noqa: BLE001
            parser = None
    _cache[key] = parser
    return parser is not None


def degraded_note(lang_id: str = "") -> str:
    """What the absence costs, in one sentence (C7)."""
    if available(lang_id):
        return ""
    return ("tree-sitter is not installed, so outlines for C, C++, Rust and "
            "JavaScript are pattern-matched rather than parsed — they can "
            "miss unusual declarations. Everything else works normally.")


def parse(text: str, path: str = "",
          lang_id: str = "") -> tuple[list[Symbol], list[tuple], list[tuple]]:
    """(symbols, edges, unresolved), or the regex fallback's answer.

    Falls back rather than raising, on purpose: this module's whole reason
    for existing is to be an upgrade when present and invisible when absent.
    """
    key = _LANG_ALIASES.get((lang_id or "").lower(), "")
    if not available(lang_id):
        from .parse_regex import parse as regex_parse
        return regex_parse(text, path, lang_id)

    parser = _cache[key]
    try:
        tree = parser.parse(text.encode("utf-8"))
    except Exception:                                    # noqa: BLE001
        from .parse_regex import parse as regex_parse
        return regex_parse(text, path, lang_id)

    wanted = _DEFS.get(key, {})
    lines = text.splitlines()
    symbols: list[Symbol] = []
    edges: list[tuple] = []
    unresolved: list[tuple] = []

    def walk(node: Any, parent: str = "") -> None:
        kind = wanted.get(node.type, "")
        name = ""
        if kind:
            name = _named_child(node, text)
            if name:
                full = f"{parent}.{name}" if parent else name
                line = node.start_point[0] + 1
                symbols.append(Symbol(
                    name=full, kind=kind, line=line,
                    end_line=node.end_point[0] + 1,
                    signature=(lines[line - 1].strip().rstrip("{").strip()[:160]
                               if line - 1 < len(lines) else full),
                    path=path, parent=parent, approximate=False))
                if parent:
                    edges.append((parent, full, "contains"))
                parent = full
        for child in node.children:
            walk(child, parent)

    walk(tree.root_node)

    own = {s.name for s in symbols} | {s.name.split(".")[-1] for s in symbols}
    for node, caller in _call_sites(tree.root_node, symbols, text):
        if node in own:
            edges.append((caller, node, "calls"))
        else:
            unresolved.append((caller, node, "calls"))
    return symbols, edges, unresolved


def _named_child(node: Any, text: str) -> str:
    for field in ("name", "declarator"):
        try:
            child = node.child_by_field_name(field)
        except Exception:                                # noqa: BLE001
            child = None
        if child is not None:
            raw = text.encode("utf-8")[child.start_byte:child.end_byte]
            name = raw.decode("utf-8", "replace").strip()
            # A C declarator is `foo(int a)`; the name is the head of it.
            name = name.split("(")[0].strip().lstrip("*&")
            if name:
                return name.split()[-1]
    return ""


def _call_sites(root: Any, symbols: list[Symbol], text: str):
    raw = text.encode("utf-8")
    out = []

    def enclosing(line: int) -> str:
        best, best_line = "", -1
        for s in symbols:
            if s.line <= line <= (s.end_line or s.line) and s.line > best_line:
                best, best_line = s.name, s.line
        return best

    def walk(node: Any) -> None:
        if node.type in ("call_expression", "call", "method_invocation"):
            try:
                fn = node.child_by_field_name("function") or node.children[0]
                name = raw[fn.start_byte:fn.end_byte].decode(
                    "utf-8", "replace").strip()
                name = name.split("(")[0].split("::")[-1].split(".")[-1]
                caller = enclosing(node.start_point[0] + 1)
                if name and caller:
                    out.append((name, caller))
            except Exception:                            # noqa: BLE001
                pass
        for child in node.children:
            walk(child)

    walk(root)
    return out
