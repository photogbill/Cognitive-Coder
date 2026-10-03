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
import re

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

    # The module's DATA surface: type aliases and constants. `TrackSegment
    # = Tuple[float, float, int]` used to be invisible — the index held
    # functions and classes only — so on Oct 2, 2026 `render.py` was shown
    # `build_track() -> List[TrackSegment]` and nothing of what a
    # TrackSegment is, guessed an object with `.x/.y/.z`, and died on its
    # first frame: `'tuple' object has no attribute 'x'`.
    taken = {s.name for s in symbols}
    for sym in _module_values(tree, text, path):
        if sym.name not in taken:
            symbols.append(sym)
            taken.add(sym.name)

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
    """`def name(params) -> ret` as written — DEFAULTS INCLUDED.

    Defaults used to be dropped, so `TrackBuilder(segment_count: int = 300)`
    reached every caller as `__init__(self, segment_count: int)`: a
    parameter that may be left out was shown as one that must be passed,
    and a caller told the truth about everything else still had to guess
    which arguments were optional. An interface is the one place where the
    exact text matters more than brevity.
    """
    args = []
    a = node.args
    positional = list(a.posonlyargs) + list(a.args)
    # `defaults` belong to the LAST len(defaults) positional parameters.
    first_default = len(positional) - len(a.defaults)
    defaults = {i: d for i, d in zip(range(first_default, len(positional)),
                                     a.defaults)}
    for i, arg in enumerate(a.posonlyargs):
        args.append(_arg(arg, defaults.get(i)))
    if a.posonlyargs:
        args.append("/")
    for j, arg in enumerate(a.args):
        args.append(_arg(arg, defaults.get(len(a.posonlyargs) + j)))
    if a.vararg:
        args.append("*" + _arg(a.vararg))
    elif a.kwonlyargs:
        args.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        args.append(_arg(arg, default))
    if a.kwarg:
        args.append("**" + _arg(a.kwarg))
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {_annotation(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({', '.join(args)}){ret}"


def _arg(arg: ast.arg, default: ast.AST | None = None) -> str:
    text = (f"{arg.arg}: {_annotation(arg.annotation)}" if arg.annotation
            else arg.arg)
    if default is not None:
        # PEP 8 spacing: `x: int = 3` but `x=3` — as a person writes it.
        sep = " = " if arg.annotation else "="
        text += sep + _default_text(default)
    return text


#: How long a default may be before it is elided. A default is part of the
#: contract when it is a value (`300`, `"red"`, `None`, `(0, 0)`); a long
#: expression is an implementation detail, and the `…` says one was cut.
_MAX_DEFAULT = 60


def _default_text(node: ast.AST | None) -> str:
    """A default value's source, as written, or `…` when it is long."""
    if node is None:
        return ""
    text = _annotation(node)
    return text if len(text) <= _MAX_DEFAULT else "…"


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
            # With its arguments: `@dataclass(frozen=True)` is a different
            # contract from `@dataclass` — its fields cannot be assigned.
            args = _annotation(d)[len(name):] if isinstance(d, ast.Call) \
                else ""
            decos.append("@" + name.split(".")[-1] + args)
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
    is_enum = any(_name_of(b).split(".")[-1] in _ENUM_BASES
                  for b in node.bases)
    # Annotated class-level fields, in order. `x: float = 0.0`, `x: float`.
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target,
                                                            ast.Name):
            fname = item.target.id
            if fname.startswith("_") or fname in seen:
                continue
            sig = f"{fname}: {_annotation(item.annotation)}"
            # The default AS WRITTEN — `field(default_factory=list)` too.
            # Only literals used to be kept, so a field with a factory
            # default read as a required constructor argument.
            default = _default_text(item.value) if item.value is not None \
                else ""
            if default:
                sig += f" = {default}"
            out.append(Symbol(name=f"{cls}.{fname}", kind="field",
                              line=item.lineno, end_line=item.lineno,
                              signature=sig, path=path, parent=cls,
                              approximate=False))
            seen.add(fname)
        elif isinstance(item, ast.Assign) and len(item.targets) == 1 \
                and isinstance(item.targets[0], ast.Name):
            # An Enum's members, or a class constant (`MAX_SPEED = 200.0`):
            # values a caller names as `Cls.NAME`, so they are interface.
            cname = item.targets[0].id
            if cname.startswith("_") or cname in seen:
                continue
            if not (is_enum or _CONST_NAME.match(cname)):
                continue
            value = _default_text(item.value)
            out.append(Symbol(name=f"{cls}.{cname}",
                              kind="member" if is_enum else "constant",
                              line=item.lineno,
                              end_line=item.end_lineno or item.lineno,
                              signature=f"{cname} = {value}", path=path,
                              parent=cls, approximate=False))
            seen.add(cname)
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


# ---------------------------------------------------------------------------
# a module's data surface: type aliases and constants
# ---------------------------------------------------------------------------

#: `ALL_CAPS` — a constant by every Python convention there is.
_CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: Bases that make a class an Enum: its class-level assignments are members.
_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})

#: Calls that BUILD A TYPE: `Point = namedtuple("Point", "x y")`.
_TYPE_FACTORIES = frozenset({
    "NamedTuple", "namedtuple", "TypedDict", "NewType", "TypeVar",
    "ParamSpec", "TypeVarTuple", "Enum", "IntEnum", "StrEnum", "Flag",
    "IntFlag", "make_dataclass"})

#: Subscripted names that make an expression a type, whatever the target is
#: called: `RGB = Tuple[int, int, int]` is an alias, not a constant.
_GENERICS = frozenset({
    "Tuple", "tuple", "List", "list", "Dict", "dict", "Set", "set",
    "FrozenSet", "frozenset", "Optional", "Union", "Callable", "Sequence",
    "Mapping", "MutableMapping", "Iterable", "Iterator", "Literal",
    "Annotated", "Type", "type", "Deque", "deque", "DefaultDict",
    "OrderedDict", "Counter", "ChainMap", "Generator", "Awaitable",
    "Coroutine", "AsyncIterator", "AsyncIterable", "ClassVar", "Final"})

_BUILTIN_TYPES = frozenset({
    "int", "float", "str", "bytes", "bool", "complex", "dict", "list",
    "tuple", "set", "frozenset", "object", "bytearray", "type"})

#: Past this, a constant's value is a table, not a contract, and is shown
#: by its type alone. An alias is never cut: it IS the contract.
_MAX_CONSTANT = 160


def _module_values(tree: ast.Module, text: str, path: str) -> list[Symbol]:
    """`alias` and `constant` symbols for a module's top-level assignments.

    alias:    `Name = <a type>` — `Tuple[float, float, int]`, `dict`,
              `int | None`, `namedtuple(...)`, `TypeVar(...)` — or anything
              annotated `TypeAlias`, or a `type Name = ...` statement.
    constant: `ALL_CAPS = <anything>`.

    Ordinary lowercase module variables (`screen = pygame.display...`) are
    not interface and are left out; underscored names are private.

    The SIGNATURE IS THE SOURCE STATEMENT, as written, so the model is
    handed the definition rather than a description of it. Its comment —
    on the same line, as a string right after it, or on the line above —
    is the docstring: `# (curve_value, color_pattern, z_position)` is the
    only place a tuple's fields are named at all.
    """
    lines = text.splitlines()
    out: list[Symbol] = []
    for idx, node in enumerate(tree.body):
        name, value, ann, kind = "", None, None, ""
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            name, value, ann = node.target.id, node.value, node.annotation
        elif type(node).__name__ == "TypeAlias":         # 3.12: `type X = …`
            name = getattr(getattr(node, "name", None), "id", "")
            value, kind = getattr(node, "value", None), "alias"
        else:
            continue
        if not name or name.startswith("_"):
            continue
        kind = kind or _value_kind(name, value, ann)
        if not kind:
            continue
        sig = _statement_text(node, text, lines)
        if kind == "constant" and len(sig) > _MAX_CONSTANT:
            typ = (_annotation(ann) if ann is not None
                   else _value_type(value)) or "value"
            sig = f"{name}: {typ} = …"
        doc = (_trailing_comment(node, lines)
               or _attribute_docstring(tree.body, idx)
               or _comment_above(node, lines))
        out.append(Symbol(name=name, kind=kind, line=node.lineno,
                          end_line=node.end_lineno or node.lineno,
                          signature=sig, path=path, docstring=doc[:200],
                          approximate=False))
    return out


def _value_kind(name: str, value: ast.AST | None,
                ann: ast.AST | None) -> str:
    """"alias", "constant" or "" for one top-level assignment."""
    if ann is not None and _name_of(ann).split(".")[-1] == "TypeAlias":
        return "alias"
    if isinstance(value, ast.Call) and \
            _name_of(value.func).split(".")[-1] in _TYPE_FACTORIES:
        return "alias"
    if isinstance(value, ast.Subscript) and _subscript_head(value) \
            in _GENERICS:
        return "alias"
    if _CONST_NAME.match(name):
        return "constant"
    if name[:1].isupper() and value is not None and _is_type_expr(value):
        return "alias"
    return ""


def _subscript_head(node: ast.Subscript) -> str:
    return _name_of(node.value).split(".")[-1]


def _is_type_expr(node: ast.AST) -> bool:
    """Does this expression denote a type? Conservative: when unsure, no."""
    if isinstance(node, ast.Name):
        return node.id in _BUILTIN_TYPES or node.id[:1].isupper()
    if isinstance(node, ast.Attribute):
        return node.attr[:1].isupper()
    if isinstance(node, ast.Subscript):
        head = _subscript_head(node)
        return head in _GENERICS or head[:1].isupper()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return all(_is_type_expr(side) or (isinstance(side, ast.Constant)
                                           and side.value is None)
                   for side in (node.left, node.right))
    if isinstance(node, ast.Call):
        return _name_of(node.func).split(".")[-1] in _TYPE_FACTORIES
    return False


def _statement_text(node: ast.AST, text: str, lines: list[str]) -> str:
    """The statement exactly as written, its lines joined into one."""
    try:
        seg = ast.get_source_segment(text, node)
    except Exception:                                    # noqa: BLE001
        seg = None
    if not seg:
        try:
            seg = ast.unparse(node)
        except Exception:                                # noqa: BLE001
            return ""
    joined = ""
    for part in (ln.strip() for ln in seg.splitlines()):
        if not part:
            continue
        # A line break after `(`/`[` or before `)`/`]` becomes nothing, any
        # other one a single space: `(800,\n 600)` reads `(800, 600)`.
        glue = "" if not joined or joined.endswith(("(", "[", "{")) \
            or part.startswith((")", "]", "}")) else " "
        joined += glue + part
    return joined


def _trailing_comment(node: ast.AST, lines: list[str]) -> str:
    """`# …` after the statement on its last line, or ""."""
    end = (getattr(node, "end_lineno", None) or node.lineno) - 1
    col = getattr(node, "end_col_offset", None)
    if not (0 <= end < len(lines)) or col is None:
        return ""
    rest = lines[end][col:].strip()
    return rest[1:].strip() if rest.startswith("#") else ""


def _comment_above(node: ast.AST, lines: list[str]) -> str:
    """A comment line immediately above the statement, or ""."""
    above = node.lineno - 2
    if not (0 <= above < len(lines)):
        return ""
    line = lines[above].strip()
    if not line.startswith("#") or "cc-stub:" in line:
        return ""
    return line.lstrip("#").strip()


def _attribute_docstring(body: list, idx: int) -> str:
    """A string literal right after the assignment (PEP 257's attribute
    docstring), or ""."""
    if idx + 1 >= len(body):
        return ""
    nxt = body[idx + 1]
    if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) \
            and isinstance(nxt.value.value, str):
        return _first_line(nxt.value.value)
    return ""


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
