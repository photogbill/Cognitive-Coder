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
    # The module is a symbol too, so that module-level calls and imports
    # have a source the store can bind. Without it, `if __name__ ==
    # "__main__": main()` produced an edge from a name that existed nowhere,
    # `callers_of("main")` was empty at "100% resolved", and the CLI entry
    # point looked dead. Named by PATH, because a dotted module name
    # collides with the function inside it: main.py's module is "main".
    module = module_symbol_name(path)
    symbols.append(Symbol(
        name=module, kind="module", line=1,
        end_line=max(1, len(text.splitlines())), path=path,
        docstring=_first_line(ast.get_docstring(tree)), approximate=False))

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
                # The class's DATA surface, as symbols: annotated class-level
                # fields (a NamedTuple's or dataclass's constructor arguments)
                # and the instance attributes its methods assign. These are
                # what a caller or a test needs and what the signature list
                # never carried — every build of the racing spec invented
                # `CarState(speed=…)` and `ProjectedSegment(z=…)` because the
                # model was shown `class CarState` and nothing of what it held.
                for sym in _class_members(node, name, path):
                    symbols.append(sym)
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

    # Bind calls LOCALLY only where Python's own scoping says so. The old
    # rule — "the short name exists somewhere in this file" — sent a plain
    # `save()` to a method `Doc.save`, and left `self.save()` for the store
    # to bind by suffix, where it found the first `%.save` in the PROJECT.
    kinds = {s.name: s.kind for s in symbols}
    parents = {s.name: s.parent for s in symbols}
    resolved: list[tuple] = []
    for src, dst, kind in edges:
        if kind != "calls":
            resolved.append((src, dst, kind))
            continue
        target = _local_target(dst, src, kinds, parents)
        if target:
            resolved.append((src, target, "calls"))
        else:
            unresolved.append((src, dst, "calls"))
    return symbols, resolved, unresolved


def enclosing_class(src: str, kinds: dict, parents: dict) -> str:
    """The class whose method `src` is (directly or via nested defs)."""
    scope = parents.get(src, "")
    while scope:
        if kinds.get(scope) == "class":
            return scope
        scope = parents.get(scope, "")
    return ""


def _local_target(called: str, src: str, kinds: dict,
                  parents: dict) -> str:
    """The symbol in THIS file that `called`, made from `src`, refers to.

    `self.x`/`cls.x` → the enclosing class's `x`; a plain name → a def
    nested in the caller's function scopes, then a top-level def; a dotted
    name → only an exact match (`Cls.method`). Anything else is left for
    the store, which knows the other files.
    """
    head, _, rest = called.partition(".")
    if head in ("self", "cls") and rest:
        cls = enclosing_class(src, kinds, parents)
        return f"{cls}.{rest}" if cls and f"{cls}.{rest}" in kinds else ""
    scope = src
    while scope and kinds.get(scope) != "module":
        if kinds.get(scope) != "class" and f"{scope}.{called}" in kinds:
            return f"{scope}.{called}"
        scope = parents.get(scope, "")
    if kinds.get(called) not in (None, "module"):
        return called
    return ""


def _calls(node: ast.AST, src: str, edges: list, unresolved: list,
           recurse_defs: bool = True) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _name_of(child.func)
            if name:
                edges.append((src, name, "calls"))


def _name_of(node: ast.AST) -> str:
    """`foo`, `mod.foo`, `self.foo` → a dotted name; anything else → "".

    An attribute on something that is not a name — `Path(p).read_text`,
    `"".join`, `rows[0].strip` — is "" too. It used to be the bare
    attribute, so `Path(p).read_text()` recorded a call to `read_text`, a
    name nothing defines, and the D4 check reported it as invented.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name_of(node.value)
        return f"{base}.{node.attr}" if base else ""
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


#: Decorators that change how a class is CONSTRUCTED, so they belong in its
#: signature: a reader seeing `@dataclass class Segment` knows the fields are
#: the constructor. Other decorators are noise here.
_CTOR_DECORATORS = {"dataclass", "define", "frozen", "attrs", "s"}


def _class_signature(node: ast.ClassDef) -> str:
    bases = [_name_of(b) for b in node.bases]
    bases = [b for b in bases if b]
    decos = []
    for d in node.decorator_list:
        name = _name_of(d.func if isinstance(d, ast.Call) else d)
        if name and name.split(".")[-1] in _CTOR_DECORATORS:
            decos.append("@" + name.split(".")[-1])
    head = (" ".join(decos) + " ") if decos else ""
    return f"{head}class {node.name}({', '.join(bases)})" if bases \
        else f"{head}class {node.name}"


# ---------------------------------------------------------------------------
# a class's data surface: fields and instance attributes
# ---------------------------------------------------------------------------

#: Literal node → the type name a reader would write. Anything else is left
#: blank rather than guessed: a wrong type in an interface is worse than none.
_LITERAL_TYPES = {bool: "bool", int: "int", float: "float", str: "str",
                  bytes: "bytes"}


def _class_members(node: ast.ClassDef, cls: str, path: str) -> list[Symbol]:
    """`field` symbols for annotated class-level names, `attribute` symbols
    for `self.<name> = …` in the class's methods (``__init__`` first).

    Fields are the constructor arguments of a NamedTuple, a dataclass, a
    TypedDict or a pydantic model, in declaration order. Attributes are the
    state a plain class exposes after construction. Both carry a type when
    the source states one or a literal makes it obvious, and a default when
    it is a simple constant — a test that caps speed at ``max_speed`` needs
    to know it is 10.0.
    """
    out: list[Symbol] = []
    seen: set[str] = set()
    # Annotated class-level fields, in order. `x: float = 0.0`, `x: float`.
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target,
                                                            ast.Name):
            fname = item.target.id
            if fname.startswith("_") or fname in seen:
                continue
            sig = f"{fname}: {_annotation(item.annotation)}"
            default = _literal_text(item.value)
            if default:
                sig += f" = {default}"
            out.append(Symbol(name=f"{cls}.{fname}", kind="field",
                              line=item.lineno, end_line=item.lineno,
                              signature=sig, path=path, parent=cls,
                              approximate=False))
            seen.add(fname)
    # Instance attributes: `self.x = …` inside methods, __init__ first so the
    # constructor's state leads. A parameter's annotation types `self.x = x`.
    methods = [m for m in node.body
               if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    methods.sort(key=lambda m: 0 if m.name == "__init__" else 1)
    for method in methods:
        params = {a.arg: _annotation(a.annotation)
                  for a in (method.args.posonlyargs + method.args.args
                            + method.args.kwonlyargs) if a.annotation}
        self_name = (method.args.args[0].arg if method.args.args
                     else "self")
        for stmt in ast.walk(method):
            targets: list[tuple[ast.AST, ast.AST | None, ast.AST | None]] = []
            if isinstance(stmt, ast.Assign):
                targets = [(t, stmt.value, None) for t in stmt.targets]
            elif isinstance(stmt, ast.AnnAssign):
                targets = [(stmt.target, stmt.value, stmt.annotation)]
            elif isinstance(stmt, ast.AugAssign):
                targets = [(stmt.target, None, None)]
            for target, value, ann in targets:
                if not (isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == self_name):
                    continue
                aname = target.attr
                if aname.startswith("_") or aname in seen:
                    continue
                typ = _annotation(ann) if ann is not None else ""
                if not typ and isinstance(value, ast.Name) \
                        and value.id in params:
                    typ = params[value.id]
                if not typ:
                    typ = _value_type(value)
                sig = f"{aname}: {typ}" if typ else aname
                default = _literal_text(value)
                if default:
                    sig += f" = {default}"
                where = "" if method.name == "__init__" else \
                    f"  (set in {method.name})"
                out.append(Symbol(name=f"{cls}.{aname}", kind="attribute",
                                  line=stmt.lineno, end_line=stmt.lineno,
                                  signature=sig + where, path=path,
                                  parent=cls, approximate=False))
                seen.add(aname)
    return out


def _literal_text(node: ast.AST | None) -> str:
    """`0.0`, `"x"`, `True`, `None` as written; anything else → ""."""
    if isinstance(node, ast.Constant) and (
            node.value is None or type(node.value) in _LITERAL_TYPES):
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Constant) \
            and type(node.operand.value) in (int, float):
        return f"-{node.operand.value!r}"
    return ""


def _value_type(node: ast.AST | None) -> str:
    """The type a value expression obviously has, or ""."""
    if node is None:
        return ""
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "None"
        return _LITERAL_TYPES.get(type(node.value), "")
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand,
                                                    ast.Constant):
        return _LITERAL_TYPES.get(type(node.operand.value), "")
    if isinstance(node, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(node, ast.JoinedStr):
        return "str"
    if isinstance(node, ast.Call):
        callee = _name_of(node.func)
        # `CarState(...)`, `collections.deque(...)` → the class; a lowercase
        # function's return type is not knowable here, so leave it blank.
        last = callee.split(".")[-1] if callee else ""
        if last and last[0].isupper():
            return last
        if last in ("list", "dict", "set", "tuple", "str", "int", "float",
                    "bool", "bytes", "frozenset"):
            return last
    return ""


def _first_line(doc: str | None) -> str:
    return (doc or "").strip().split("\n")[0][:200]


def module_symbol_name(path: str) -> str:
    """The module's own symbol: its normalised path (never an identifier)."""
    p = str(path or "").replace("\\", "/").strip("/")
    return p or "<module>"


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
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.NamedExpr)):
            bind(node.target)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    bind(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                out.add(node.name)
        elif isinstance(node, ast.comprehension):
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
            args = node.args
            for a in (list(args.posonlyargs) + list(args.args)
                      + list(args.kwonlyargs)):
                out.add(a.arg)
            # `lambda *rest, **kw: rest.count(1)` — these were missed, and
            # `rest.count` was reported as a name the project lacks.
            if args.vararg:
                out.add(args.vararg.arg)
            if args.kwarg:
                out.add(args.kwarg.arg)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            # `case ['go', direction]:` binds `direction`; `case [x, *more]`
            # binds `more`.
            if node.name:
                out.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                out.add(node.rest)                       # `**others`
    return out


def import_bindings(text: str) -> list[tuple[str, str, str]]:
    """Every import as (module, name, alias); name is "" for `import m`.

    `imports_of` keeps its list-of-modules shape because the planner
    derives the dependency order from it. The D4 check needs more: which
    LOCAL names an import binds (`np` for `import numpy as np`,
    `OrderedDict` for `from collections import OrderedDict`), and from
    which module, so that it can tell a standard-library name from an
    invented one.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[tuple[str, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, "", a.asname or ""))
        elif isinstance(node, ast.ImportFrom):
            base = "." * (node.level or 0) + (node.module or "")
            for a in node.names:
                out.append((base, a.name, a.asname or ""))
    return out


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
