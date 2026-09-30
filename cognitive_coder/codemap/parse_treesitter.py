# SPDX-License-Identifier: Apache-2.0
"""Optional tree-sitter parsing — better than regex, and strictly optional.

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

#: The function each grammar package exposes, where it is not `language`.
_LANGUAGE_FN = {"typescript": "language_typescript"}

_LABEL = {"c": "C", "cpp": "C++", "rust": "Rust", "javascript": "JavaScript",
          "typescript": "TypeScript", "go": "Go", "java": "Java"}

_cache: dict[str, Any] = {}
#: Why a language has no parser: "missing" (no tree-sitter at all) or
#: "incompatible: …" (tree-sitter imports, but no grammar would load).
_why: dict[str, str] = {}


def _load(key: str) -> tuple[Any, str]:
    """(parser, "") or (None, why). Never raises."""
    try:
        import tree_sitter  # noqa: PLC0415 — optional (M48)
    except Exception:                                    # noqa: BLE001
        return None, "missing"
    errors: list[str] = []
    # The per-language packages (tree-sitter >= 0.22): `Language(ptr)` and
    # `Parser(language)`. Tried FIRST because it is the maintained path.
    try:
        import importlib

        mod = importlib.import_module(f"tree_sitter_{key}")
        language = tree_sitter.Language(
            getattr(mod, _LANGUAGE_FN.get(key, "language"))())
        try:
            return tree_sitter.Parser(language), ""
        except TypeError:                                # tree-sitter 0.21
            parser = tree_sitter.Parser()
            parser.set_language(language)
            return parser, ""
    except Exception as exc:                             # noqa: BLE001
        errors.append(f"tree_sitter_{key}: {type(exc).__name__}")
    # tree-sitter-languages bundles every grammar but was built against the
    # old API: with tree-sitter 0.26 its `get_parser` raises TypeError, which
    # is what made `available("c")` False on a machine with both installed.
    try:
        from tree_sitter_languages import get_parser  # noqa: PLC0415
        return get_parser(key), ""
    except Exception as exc:                             # noqa: BLE001
        errors.append(f"tree_sitter_languages: {type(exc).__name__}")
    return None, "incompatible: " + "; ".join(errors)


def available(lang_id: str) -> bool:
    """Is a real parser present for this language right now?

    Cached because the answer cannot change within a process and the import
    is not free. Any failure — no package, no grammar, an API change between
    tree-sitter versions — is False, which routes to the regex extractor,
    and the reason is kept for `degraded_note`.
    """
    key = _LANG_ALIASES.get((lang_id or "").lower(), "")
    if not key:
        return False
    if key not in _cache:
        _cache[key], _why[key] = _load(key)
    return _cache[key] is not None


def degraded_note(lang_id: str = "") -> str:
    """What the absence costs, in one sentence (C7) — and WHICH absence.

    "Not installed" and "installed but unusable" need different fixes, and
    the note used to say the first when the second was true.
    """
    key = _LANG_ALIASES.get((lang_id or "").lower(), "")
    if not key:
        return _summary_note()
    if available(key):
        return ""
    why = _why.get(key, "")
    if why == "missing":
        return ("tree-sitter is not installed, so outlines for C, C++, "
                "Rust, JavaScript, TypeScript, Go and Java are "
                "pattern-matched rather than parsed — they can miss unusual "
                "declarations. Everything else works normally.")
    label = _LABEL.get(key, "these languages")
    grammar = f"tree-sitter-{key}" if key else "tree-sitter-<language>"
    return (f"tree-sitter is installed, but no {label} grammar could be "
            f"loaded with it ({why.partition(': ')[2] or why}), so {label} "
            f"outlines are pattern-matched rather than parsed. Installing "
            f"{grammar} (for tree-sitter 0.22 or newer) fixes this. "
            f"Everything else works normally.")


def _summary_note() -> str:
    """The note for the installation as a whole, not for one language.

    With no language named, the previous version reported "incompatible"
    whenever tree-sitter imported — including on a machine where every
    grammar loaded — because it never actually tried one. This asks each
    language, and names only the ones that fall back.
    """
    missing = [k for k in _LABEL if not available(k)]
    if not missing:
        return ""
    if all(_why.get(k) == "missing" for k in missing):
        return degraded_note(missing[0])
    names = ", ".join(_LABEL[k] for k in missing)
    return (f"tree-sitter is installed, but no grammar could be loaded for "
            f"{names}, so outlines for those are pattern-matched rather "
            f"than parsed. Installing tree-sitter-<language> (for "
            f"tree-sitter 0.22 or newer) fixes this. Everything else works "
            f"normally.")


def parse(text: str, path: str = "",
          lang_id: str = "") -> tuple[list[Symbol], list[tuple], list[tuple]]:
    """(symbols, edges, unresolved), or the regex fallback's answer.

    Falls back rather than raising, on purpose: this module's whole reason
    for existing is to be an upgrade when present and invisible when absent.
    The WHOLE walk is inside the fallback, not just `parser.parse`: a node
    API that changed between tree-sitter versions fails there, and it used
    to escape as an exception.
    """
    from .parse_regex import parse as regex_parse

    key = _LANG_ALIASES.get((lang_id or "").lower(), "")
    if not available(lang_id):
        return regex_parse(text, path, lang_id)
    try:
        return _parse_tree(_cache[key], text, path, key)
    except Exception:                                    # noqa: BLE001
        return regex_parse(text, path, lang_id)


def _parse_tree(parser: Any, text: str, path: str,
                key: str) -> tuple[list[Symbol], list[tuple], list[tuple]]:
    from .parse_regex import _module_of, imports_of

    raw = text.encode("utf-8")
    tree = parser.parse(raw)
    wanted = _DEFS.get(key, {})
    lines = text.splitlines()
    module = _module_of(path)
    symbols: list[Symbol] = [Symbol(
        name=module, kind="module", line=1, end_line=max(1, len(lines)),
        path=path, approximate=False)]
    edges: list[tuple] = [(module, name, "imports")
                          for name in imports_of(text, key)]
    unresolved: list[tuple] = []

    def walk(node: Any, parent: str = "") -> None:
        kind = wanted.get(node.type, "")
        if kind:
            name = _definition_name(node, raw, key)
            if name:
                full = (name if "." in name or not parent
                        else f"{parent}.{name}")
                line = node.start_point[0] + 1
                symbols.append(Symbol(
                    name=full, kind=kind, line=line,
                    end_line=node.end_point[0] + 1,
                    signature=_signature(node, raw, lines, line, full),
                    path=path, parent=parent, approximate=False))
                if parent:
                    edges.append((parent, full, "contains"))
                parent = full
        for child in node.children:
            walk(child, parent)

    walk(tree.root_node)

    own = {s.name for s in symbols if s.kind != "module"}
    by_short = {s.name.split(".")[-1]: s.name for s in symbols
                if s.kind != "module"}
    for name, caller in _call_sites(tree.root_node, symbols, raw):
        caller = caller or module
        if name in own:
            edges.append((caller, name, "calls"))
        elif name in by_short and by_short[name] != caller:
            edges.append((caller, by_short[name], "calls"))
        else:
            unresolved.append((caller, name, "calls"))
    return symbols, edges, unresolved


def _text(node: Any, raw: bytes) -> str:
    return raw[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _field(node: Any, name: str) -> Any:
    try:
        return node.child_by_field_name(name)
    except Exception:                                    # noqa: BLE001
        return None


def _definition_name(node: Any, raw: bytes, key: str) -> str:
    """The defined name, per grammar where the grammars differ."""
    import re

    if node.type == "impl_item":
        # A Rust impl has no `name`; it has a `type` (and maybe a `trait`).
        # `impl<T> Trait for Foo<T>` is about Foo, and was never found.
        child = _field(node, "type")
        return (re.split(r"[<(\s]", _text(child, raw).strip())[0]
                .split("::")[-1] if child is not None else "")
    if node.type == "type_declaration":
        # Go keeps the name on the inner `type_spec`, not the declaration.
        for child in node.children:
            name = _field(child, "name")
            if name is not None:
                return _text(name, raw).strip()
        return ""
    if node.type == "method_declaration" and key == "go":
        # Named `Receiver.method`, as the regex outline names it.
        name = _field(node, "name")
        recv = _field(node, "receiver")
        ids = re.findall(r"[A-Za-z_]\w*", _text(recv, raw)) if recv else []
        base = _text(name, raw).strip() if name is not None else ""
        return f"{ids[-1]}.{base}" if ids and base else base
    for field in ("name", "declarator"):
        child = _field(node, field)
        if child is not None:
            name = _text(child, raw).strip()
            # A C declarator is `foo(int a)`; the name is the head of it.
            name = name.split("(")[0].strip().lstrip("*&")
            if name:
                return name.split()[-1]
    return ""


def _signature(node: Any, raw: bytes, lines: list[str], line: int,
               fallback: str) -> str:
    """The declaration up to its body, whitespace collapsed, one line.

    The first LINE was used, which for `static int\nhelper(int a, …)` is
    just `static int`.
    """
    body = _field(node, "body")
    if body is not None and body.start_byte > node.start_byte:
        head = raw[node.start_byte:body.start_byte].decode("utf-8",
                                                           "replace")
        sig = " ".join(head.split()).rstrip("{").strip()
        if sig:
            return sig[:160]
    return (lines[line - 1].strip().rstrip("{").strip()[:160]
            if line - 1 < len(lines) else fallback)


def _call_sites(root: Any, symbols: list[Symbol], raw: bytes):
    out = []

    def enclosing(line: int) -> str:
        best, best_line = "", -1
        for s in symbols:
            if s.kind == "module":
                continue
            # `>=`: of two symbols starting on the same line (a one-line
            # class and its method), the later — inner — one is the caller.
            if s.line <= line <= (s.end_line or s.line) \
                    and s.line >= best_line:
                best, best_line = s.name, s.line
        return best

    def walk(node: Any) -> None:
        if node.type in ("call_expression", "call", "method_invocation"):
            # Java's `obj.helper()` keeps the method in field `name`; the
            # first child is the OBJECT, which was being reported as the
            # thing called.
            fn = (_field(node, "name") if node.type == "method_invocation"
                  else None) or _field(node, "function") or node.children[0]
            name = _text(fn, raw).strip()
            name = name.split("(")[0].split("::")[-1].split(".")[-1]
            if name:
                out.append((name, enclosing(node.start_point[0] + 1)))
        for child in node.children:
            walk(child)

    walk(root)
    return out
