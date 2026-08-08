# SPDX-License-Identifier: Apache-2.0
"""Python source → symbols and edges, exactly, using `ast`.

Exact rather than approximate, because Python has a parser in the standard
library and using anything else would be choosing to be wrong for no saving.
Everything this module produces has `approximate=False`, and that flag is what
downstream code uses to decide whether to caveat an outline.

Two things are extracted, and the second is the one that makes the codemap
worth building:

  * **Symbols** — functions, classes, methods, with signatures, docstrings,
    line spans and parents.
  * **Edges** — calls, imports and containment. This is the call graph, and
    it is what answers "if I change this signature, what breaks?" (blast
    radius, §6.7).

**Unresolved calls are reported, not dropped.** A call graph that silently
discards what it could not bind looks complete and is not — and a model told
"nothing calls this" when six things do will happily delete it. Every call
whose target cannot be found in this file's scope is recorded in
`unresolved`, and the resolution rate is a number the operator can see.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

from ..types import Symbol


def parse(text: str, path: str = "") -> tuple[list[Symbol], list[tuple],
                                              list[tuple]]:
    """Return (symbols, edges, unresolved).

    ``edges`` are ``(src_name, dst_name, kind)`` with kind in
    {calls, imports, contains}. ``unresolved`` are ``(src_name, name, kind)``
    — the calls this file makes that could not be bound locally. Binding
    them across files is the store's job, and what it cannot bind stays
    unresolved and counted.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # A file mid-edit does not parse, and the codemap is most wanted
        # exactly then. The regex extractor is the honest fallback; it is
        # labelled approximate, so nothing downstream mistakes it for this.
        from .parse_regex import parse as regex_parse
        return regex_parse(text, path, "python")

    symbols: list[Symbol] = []
    edges: list[tuple] = []
    unresolved: list[tuple] = []
    module = _module_name(path)

    # Imports first: they are what a called name might resolve TO, and D4
    # (invented imports and APIs) is the most common small-model error in
    # multi-file work. Recording them is what lets the loop check a generated
    # import against reality before running anything.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append((module, alias.name, "imports"))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            for alias in node.names:
                target = f"{base}.{alias.name}" if base else alias.name
                edges.append((module, target, "imports"))

    def visit(body: Iterable[ast.AST], parent: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{parent}.{node.name}" if parent else node.name
                symbols.append(Symbol(
                    name=name, kind="method" if parent else "function",
                    line=node.lineno, end_line=node.end_lineno or node.lineno,
                    signature=_signature(node), path=path, parent=parent,
                    docstring=_first_line(ast.get_docstring(node)),
                    approximate=False))
                if parent:
                    edges.append((parent, name, "contains"))
                _calls(node, name, edges, unresolved)
                visit(node.body, name)
            elif isinstance(node, ast.ClassDef):
                name = f"{parent}.{node.name}" if parent else node.name
                symbols.append(Symbol(
                    name=name, kind="class", line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    signature=_class_signature(node), path=path,
                    parent=parent,
                    docstring=_first_line(ast.get_docstring(node)),
                    approximate=False))
                if parent:
                    edges.append((parent, name, "contains"))
                for base in node.bases:
                    base_name = _name_of(base)
                    if base_name:
                        edges.append((name, base_name, "inherits"))
                visit(node.body, name)

    visit(tree.body)

    # Module-level calls belong to the module itself — a script's top-level
    # code is real code, and pretending it has no callers is how a CLI entry
    # point looks dead.
    top = [n for n in tree.body
           if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef))]
    for node in top:
        _calls(node, module, edges, unresolved, recurse_defs=False)

    known = {s.name for s in symbols} | {s.name.split(".")[-1]
                                         for s in symbols}
    resolved: list[tuple] = []
    for src, dst, kind in edges:
        if kind == "calls" and dst not in known:
            unresolved.append((src, dst, "calls"))
        else:
            resolved.append((src, dst, kind))
    return symbols, resolved, unresolved


def _calls(node: ast.AST, src: str, edges: list, unresolved: list,
           recurse_defs: bool = True) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _name_of(child.func)
            if name:
                edges.append((src, name, "calls"))


def _name_of(node: ast.AST) -> str:
    """`foo`, `mod.foo`, `self.foo` → a dotted name; anything else → ""."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _signature(node) -> str:
    args = []
    a = node.args
    for arg in a.posonlyargs:
        args.append(_arg(arg))
    if a.posonlyargs:
        args.append("/")
    for arg in a.args:
        args.append(_arg(arg))
    if a.vararg:
        args.append("*" + _arg(a.vararg))
    elif a.kwonlyargs:
        args.append("*")
    for arg in a.kwonlyargs:
        args.append(_arg(arg))
    if a.kwarg:
        args.append("**" + _arg(a.kwarg))
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {_annotation(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({', '.join(args)}){ret}"


def _arg(arg: ast.arg) -> str:
    return (f"{arg.arg}: {_annotation(arg.annotation)}" if arg.annotation
            else arg.arg)


def _annotation(node) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:                                    # noqa: BLE001
        return "?"


def _class_signature(node: ast.ClassDef) -> str:
    bases = [_name_of(b) for b in node.bases]
    bases = [b for b in bases if b]
    return f"class {node.name}({', '.join(bases)})" if bases \
        else f"class {node.name}"


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n")[0][:200]


def _module_name(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    if p.endswith(".py"):
        p = p[:-3]
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.strip("/").replace("/", ".") or "<module>"


def bound_names(text: str) -> set[str]:
    """Every name BOUND anywhere in this file: locals, params, loop vars.

    Distinct from the symbol table, which holds only functions and classes —
    the things another module can import. This is the wider set of names that
    exist at run time inside the file, and it answers a question the symbol
    table cannot: *is ``screen`` in ``screen.fill(...)`` a real object?*

    WHY IT WAS NEEDED, 2026-08-07
    -----------------------------
    ``unresolved_in`` reports names the project does not define, and it was
    treating the head of every dotted call as such a name. Since a local
    variable is not a symbol, the result was a warning on nearly every
    generated file::

        src/main.py refers to names this project does not define:
        screen.fill, draw_track, clock.tick, clock.get_time

    Of those four, one was real. ``screen`` was ``pygame.display.set_mode()``
    three lines above; ``clock`` was ``pygame.time.Clock()``. A static
    checker cannot know what those objects are, and it was never going to —
    so it should not be claiming they are undefined.

    The cost was not the noise itself but what the noise concealed. In the
    same runs, ``CarState``, ``generate_track``, ``render_road`` and
    ``TrackSegment`` were flagged in the same sentences, in the same format,
    and were all genuinely missing — each one became an ImportError minutes
    later. The signal was there the whole time, filed alongside the chaff.

    A check that cries wolf is a check somebody turns off, which the docstring
    of ``unresolved_in`` already said. This is that principle applied to the
    case it was missing.

    Walrus targets, ``with ... as``, ``except ... as``, comprehension
    variables, and both ordinary and starred assignment are all bindings and
    all included. Attribute and subscript targets are not — ``self.x = 1``
    binds nothing new called ``x``.

    IMPORTS ARE DELIBERATELY EXCLUDED, and the reason is a regression this
    caused on its first outing. ``from utils import parse_config`` does bind
    ``parse_config``, so an earlier version of this function reported it — and
    ``unresolved_in`` then fell silent about a symbol pulled from a module
    that does not exist, which is one of the most valuable things it catches.

    Imports are already tracked separately and more precisely, because
    "the module resolves" and "the symbol resolves" are different questions.
    Folding them in here answers the second with the first. So this function
    means specifically: *names bound by executable statements in this file* —
    the set you need in order to recognise ``screen.fill`` as an attribute on
    a real local object, and nothing wider.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()

    out: set[str] = set()

    def bind(target: ast.AST) -> None:
        for node in ast.walk(target):
            if isinstance(node, ast.Name):
                out.add(node.id)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for tgt in (node.targets if isinstance(node, ast.Assign)
                        else [node.target]):
                bind(tgt)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind(node.target)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                out.add(node.name)
        elif isinstance(node, comprehension_types):
            bind(node.target)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(node.name)
            args = node.args
            for a in (list(args.posonlyargs) + list(args.args)
                      + list(args.kwonlyargs)):
                out.add(a.arg)
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.Lambda):
            for a in (list(node.args.posonlyargs) + list(node.args.args)
                      + list(node.args.kwonlyargs)):
                out.add(a.arg)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            out.update(node.names)
    return out


comprehension_types = (ast.comprehension,)


def imports_of(text: str) -> list[str]:
    """Just the imports — used to DERIVE the dependency order (§4.2).

    Deriving the DAG from the skeleton's imports is deterministic (C5) and
    correct; asking the model to assert it produces valid JSON that is
    architecturally wrong, which poisons every downstream step.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = "." * node.level + base
            out.append(base)
    return [name for name in out if name]
