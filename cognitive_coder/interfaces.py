# SPDX-License-Identifier: Apache-2.0
"""The interface skeleton: §4.2 step 2, the stubs every file is built against.

WHY THIS EXISTS, 2026-10-02
---------------------------
The racing spec on Qwen3-Coder-30B. The skeleton wrote five stubs, and every
one was the same two things: a docstring and ``def main(): raise
NotImplementedError``. So each file was written blind to the others:
``track.py`` made ``TrackSegment = Tuple[float, float, int]`` — plain
tuples — and ``render.py``, written later, read ``seg.x``/``seg.y``/
``seg.z`` from them and died on its first frame with ``'tuple' object has
no attribute 'x'``. Nothing between the two files had ever said what a
segment IS.

§4.2 always said what the skeleton is for: *signatures, imports,
docstrings, raise NotImplementedError — verify the whole skeleton
imports.* `Planner.stub_for` wrote the docstring and the `raise` by rule,
because a rule cannot invent signatures, and the richer skeleton was left
"for its own pass" (`Planner._role_order`). This is that pass:

  1. ONE model call writes a stub for every planned Python module, with all
     of them in view — the shared data types defined once, in the module
     that owns them, and imported by name everywhere else; every public
     function's exact signature.
  2. The reply is SANITISED BY RULE, not trusted (C5): every function body
     becomes `raise NotImplementedError`; top-level code that would run at
     import — `pygame.init()`, a main loop, an `if __name__` block — is
     dropped; only assignments whose calls build types or values are kept.
     So a stub can always be imported, whatever the model wrote.
  3. Each stub is marked PINNED (`PINNED_TEXT`), which every "is this still
     a stub?" check already understands, and which tells the loop to show
     the stub to the file's author as the contract it must keep.

A file whose block is missing, does not parse, or is not Python keeps the
rule stub, and the session says which. Nothing here can make a build worse
than the rule stub did.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
import re

#: Written into a PINNED stub instead of the rule stub's line. It carries the
#: same `cc-stub:` sentinel, so every existing stub check treats the file as
#: unwritten; the words after it are what the loop looks for.
PINNED_TEXT = ("cc-stub: interface pinned by the skeleton; keep these names, "
               "types and signatures when this file is built")

#: Calls an assignment may make at import time in a stub. Building a type,
#: a dataclass field, a number or a container is safe; anything else may
#: start a window, open a file or call a function that is itself a stub.
_SAFE_CALLS = frozenset({
    "NamedTuple", "namedtuple", "TypedDict", "NewType", "TypeVar",
    "ParamSpec", "TypeVarTuple", "Enum", "IntEnum", "StrEnum", "Flag",
    "IntFlag", "make_dataclass", "field", "tuple", "list", "dict", "set",
    "frozenset", "float", "int", "str", "bool", "bytes", "range", "len",
    "min", "max", "abs", "round", "sorted", "deque", "defaultdict",
    "OrderedDict", "Counter", "auto"})

#: Modules whose functions are pure enough to call at import: `math.pi / 3`
#: and `math.radians(60)` are constants a stub may define.
_SAFE_MODULES = frozenset({"math", "typing", "enum", "dataclasses",
                           "collections", "decimal", "fractions"})


def is_pinned(text: str) -> bool:
    """Is this file a stub whose interface the skeleton pinned?"""
    return PINNED_TEXT in (text or "")


def task_text(request: str, modules: Sequence[tuple[str, str]],
              tests: Sequence[str] = (),
              existing: Sequence[tuple[str, str]] = ()) -> str:
    """The prompt body for the one interface call."""
    lines = [
        "Write the INTERFACE SKELETON for this project: one stub per file "
        "listed below, before any file is written. Each file will later be "
        "written by someone who sees only its own stub and the stubs of "
        "the files it uses, so these stubs are the contract between the "
        "files and must agree with each other exactly.",
        "",
        f"REQUEST: {request}",
        "",
        "FILES (write a stub for each, in this order):",
    ]
    lines += [f"  {path} — {purpose}" for path, purpose in modules]
    if tests:
        lines += ["", "Test files that will be written against these stubs "
                      "(do not write them): " + ", ".join(tests)]
    for path, surface in existing:
        lines += ["", f"ALREADY EXISTS, keep its names — {path}:", surface]
    lines += [
        "",
        "In every stub:",
        "- Define each data type that more than one file uses ONCE, in the "
        "module that owns it — a @dataclass or NamedTuple with typed "
        "fields, or a type alias — and import it BY NAME everywhere else "
        "(`from src.track import Segment`). Prefer a dataclass with named "
        "fields to a bare tuple.",
        "- Every public function and method: full type hints, a one-line "
        "docstring, and the body `raise NotImplementedError`.",
        "- Every class: its constructor, and its public attributes with "
        "types (a dataclass's fields, or `self.speed: float = 0.0` lines "
        "in `__init__`).",
        "- Module-level constants other files need, with their values.",
        "- Imports of this project's modules use the paths above "
        "(`src/track.py` is `src.track`). Nothing runs at import: no "
        "window and no main loop at the top level. The entry point "
        "defines `main()` and calls it only under "
        "`if __name__ == \"__main__\":`.",
        "",
        "Example of one block:",
        "```python",
        "# file: src/shapes.py",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass",
        "class Point:",
        '    """A point on the plane."""',
        "    x: float",
        "    y: float",
        "",
        "",
        "def distance(a: Point, b: Point) -> float:",
        '    """Straight-line distance between two points."""',
        "    raise NotImplementedError",
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the reply → one block per planned file
# ---------------------------------------------------------------------------

_BLOCK = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```", re.S)
_FILE_LINE = re.compile(
    r"^\s*(?:#|//)\s*(?:file\s*:\s*)?[`'\"]?(?P<path>[\w./\\-]+\.\w+)[`'\"]?"
    r"\s*$", re.I)
_HEAD_LINE = re.compile(r"(?P<path>[\w./\\-]+\.py)\b")


def split_reply(text: str, wanted: Sequence[str]) -> dict[str, str]:
    """{planned path: its code} from the model's reply.

    A block is matched to a path by its first line (`# file: src/x.py`, or
    just `# src/x.py`), else by the last path named on the line before the
    fence (`### src/x.py`). Paths are compared normalised, and by file name
    when only one planned file has that name. Unmatched blocks are ignored.
    """
    norm = {_norm(p): p for p in wanted}
    by_name: dict[str, list[str]] = {}
    for p in wanted:
        by_name.setdefault(_norm(p).rsplit("/", 1)[-1], []).append(p)

    def match(candidate: str) -> str:
        key = _norm(candidate)
        if key in norm:
            return norm[key]
        hits = by_name.get(key.rsplit("/", 1)[-1], [])
        return hits[0] if len(hits) == 1 else ""

    out: dict[str, str] = {}
    blocks = list(_BLOCK.finditer(text or ""))
    if not blocks and "file:" in (text or ""):
        # No fences at all, but `# file:` markers: split on those.
        chunks = re.split(r"(?m)^(?=\s*#\s*file\s*:)", text)
        for chunk in chunks:
            first, _, body = chunk.partition("\n")
            m = _FILE_LINE.match(first)
            path = match(m.group("path")) if m else ""
            if path and path not in out:
                out[path] = body
        return out
    for m in blocks:
        body = m.group("body")
        first, _, rest = body.lstrip("\n").partition("\n")
        hit = _FILE_LINE.match(first)
        path = match(hit.group("path")) if hit else ""
        info = _HEAD_LINE.findall(m.group("info") or "")
        if path:
            body = rest
        elif info and match(info[-1]):
            path = match(info[-1])          # ```python src/track.py
        else:
            before = (text[:m.start()].rstrip().rsplit("\n", 1) or [""])[-1]
            heads = _HEAD_LINE.findall(before)
            path = match(heads[-1]) if heads else ""
        if path and path not in out:
            out[path] = body
    return out


def _norm(path: str) -> str:
    p = str(path or "").strip().strip("`'\"").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lower()


# ---------------------------------------------------------------------------
# sanitising: whatever the model wrote, a stub that imports
# ---------------------------------------------------------------------------

def sanitise(code: str, purpose: str = "", *,
             entry_point: bool = False) -> tuple[str, list[str]]:
    """(stub text, what was dropped) — or ("", [reason]) if unusable.

    Line-based, so the model's comments survive: the comment on a type
    alias (`# (curve, color, z)`) is often the only place a tuple's fields
    are named. Bodies are replaced by `raise NotImplementedError` — except
    an `__init__`'s `self.x = …` lines, which ARE interface: they say what
    the object holds. Top-level statements that would run real code at
    import are removed. The result must parse, or the file falls back.

    An `if __name__ == "__main__":` block runs nothing at import and says
    how the program starts, so it is kept as `raise SystemExit(main())` —
    and an ENTRY POINT that defines `main()` without one gets one. A stub
    without it taught the body to leave it out: a `main.py` that defines
    `main()` and never calls it runs, exits 0 and starts nothing.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return "", [f"its stub does not parse ({exc.msg}, line {exc.lineno})"]
    lines = code.splitlines()
    #: (first line, last line, replacement lines), 1-based inclusive.
    cuts: list[tuple[int, int, list[str]]] = []
    dropped: list[str] = []
    has_doc = bool(ast.get_docstring(tree))

    def stub_function(fn: ast.AST) -> None:
        body = list(fn.body)
        start = 0
        if body and _is_docstring(body[0]):
            start = 1
        rest = body[start:]
        if not rest:
            cuts.append((fn.end_lineno + 1, fn.end_lineno,
                         [_indent_of(lines, fn) + "    raise "
                          "NotImplementedError"]))
            return
        for st in rest:
            if lines[st.lineno - 1][:st.col_offset].strip():
                # `def f(): return 1`, or `x = 1; y = 2` — a statement that
                # shares its line with something else. Line surgery cannot
                # separate them; the unparse fallback handles it.
                raise _Unsliceable
        indent = " " * rest[0].col_offset
        if fn.name == "__init__":
            keep = [st for st in rest if _assigns_self(st)]
            drop = [st for st in rest if not _assigns_self(st)]
            for st in drop:
                cuts.append((st.lineno, st.end_lineno, []))
            if not keep:
                cuts.append((rest[-1].end_lineno + 1, rest[-1].end_lineno,
                             [indent + "raise NotImplementedError"]))
            return
        cuts.append((rest[0].lineno, rest[-1].end_lineno,
                     [indent + "raise NotImplementedError"]))

    def visit_class(cls: ast.ClassDef) -> None:
        for st in cls.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stub_function(st)
            elif isinstance(st, ast.ClassDef):
                visit_class(st)
            elif _is_docstring(st) or isinstance(st, ast.Pass):
                continue
            elif isinstance(st, (ast.Assign, ast.AnnAssign)) and \
                    _safe_value(getattr(st, "value", None)):
                continue
            else:
                cuts.append((st.lineno, st.end_lineno, []))
                dropped.append(f"`{_first(lines, st)}` in class {cls.name}")

    has_main = any(isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and st.name == "main" for st in tree.body)
    guarded = False
    try:
        for st in tree.body:
            if isinstance(st, (ast.Import, ast.ImportFrom)) or \
                    _is_docstring(st):
                continue
            if _is_main_guard(st):
                guarded = True
                if has_main:
                    cuts.append((st.body[0].lineno, st.end_lineno,
                                 ["    raise SystemExit(main())"]))
                else:
                    cuts.append((st.lineno, st.end_lineno, []))
                continue
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stub_function(st)
            elif isinstance(st, ast.ClassDef):
                visit_class(st)
            elif isinstance(st, (ast.Assign, ast.AnnAssign)) and \
                    _safe_value(getattr(st, "value", None)):
                continue
            elif type(st).__name__ == "TypeAlias":
                continue
            else:
                cuts.append((st.lineno, st.end_lineno, []))
                dropped.append(f"`{_first(lines, st)}`")
        text = _apply(lines, cuts)
        ast.parse(text)
    except (_Unsliceable, SyntaxError):
        text = _unparse_fallback(tree)
        if not text:
            return "", ["its stub could not be reduced to signatures"]
    if has_main and (entry_point or guarded) and \
            not any(_is_main_guard(st) for st in ast.parse(text).body):
        text = text.rstrip("\n") + ('\n\n\nif __name__ == "__main__":\n'
                                    '    raise SystemExit(main())\n')
    body = text.strip("\n")
    if not has_doc and purpose:
        body = f'"""{_one_line(purpose)}"""\n' + body
    if "from __future__ import annotations" not in body:
        # A stub's annotations may name a class defined further down, or one
        # imported under TYPE_CHECKING; evaluated at import, either is a
        # NameError in a file that does nothing yet. Deferred, neither is.
        body = _after_docstring(body)
    return f"# {PINNED_TEXT}\n{body}\n", dropped


class _Unsliceable(Exception):
    """A body shares a line with its `def`; use the unparse fallback."""


def _apply(lines: list[str], cuts: list[tuple[int, int, list[str]]]) -> str:
    out = list(lines)
    # Bottom-up, so earlier line numbers stay valid. An insertion is a cut
    # whose first line is one past its last.
    for first, last, repl in sorted(cuts, key=lambda c: (c[0], c[1]),
                                    reverse=True):
        out[first - 1:last] = repl
    return "\n".join(out)


def _unparse_fallback(tree: ast.Module) -> str:
    """The same reduction on the tree, for sources line surgery cannot cut.
    Comments are lost; signatures, fields and types are not."""
    class Strip(ast.NodeTransformer):
        def visit_FunctionDef(self, node):           # noqa: N802
            doc = node.body[:1] if node.body and \
                _is_docstring(node.body[0]) else []
            if node.name == "__init__":
                keep = [st for st in node.body if _assigns_self(st)]
                node.body = doc + (keep or [_raise()])
            else:
                node.body = doc + [_raise()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef        # noqa: N815

        def visit_ClassDef(self, node):              # noqa: N802
            self.generic_visit(node)
            node.body = [st for st in node.body
                         if isinstance(st, (ast.FunctionDef, ast.ClassDef,
                                            ast.AsyncFunctionDef, ast.Pass))
                         or _is_docstring(st)
                         or (isinstance(st, (ast.Assign, ast.AnnAssign))
                             and _safe_value(getattr(st, "value", None)))
                         ] or [ast.Pass()]
            return node

    tree = Strip().visit(tree)
    tree.body = [st for st in tree.body
                 if isinstance(st, (ast.Import, ast.ImportFrom,
                                    ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))
                 or _is_docstring(st)
                 or (isinstance(st, (ast.Assign, ast.AnnAssign))
                     and _safe_value(getattr(st, "value", None)))]
    try:
        text = ast.unparse(ast.fix_missing_locations(tree))
        ast.parse(text)
        return text
    except Exception:                                    # noqa: BLE001
        return ""


def _is_main_guard(st: ast.AST) -> bool:
    """`if __name__ == "__main__":` (either way round)."""
    if not isinstance(st, ast.If) or not isinstance(st.test, ast.Compare):
        return False
    t = st.test
    if len(t.ops) != 1 or not isinstance(t.ops[0], ast.Eq):
        return False
    sides = [t.left, *t.comparators]
    names = [x.id for x in sides if isinstance(x, ast.Name)]
    consts = [x.value for x in sides if isinstance(x, ast.Constant)]
    return names == ["__name__"] and consts == ["__main__"]


def _raise() -> ast.stmt:
    return ast.Raise(exc=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                     cause=None)


def _is_docstring(st: ast.AST) -> bool:
    return isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
        and isinstance(st.value.value, str)


def _assigns_self(st: ast.AST) -> bool:
    """`self.x = …` / `self.x: T = …` — an attribute the object holds."""
    targets = st.targets if isinstance(st, ast.Assign) else (
        [st.target] if isinstance(st, ast.AnnAssign) else [])
    return bool(targets) and all(
        isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
        and t.value.id == "self" for t in targets)


def _safe_value(node: ast.AST | None) -> bool:
    """May this value be computed when the stub is imported?"""
    if node is None:
        return True
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Lambda):
            continue                     # a lambda's body runs later, if ever
        if isinstance(cur, ast.Call):
            func = cur.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else "")
            base = func.value.id if isinstance(func, ast.Attribute) and \
                isinstance(func.value, ast.Name) else ""
            if name not in _SAFE_CALLS and base not in _SAFE_MODULES:
                return False
        stack.extend(ast.iter_child_nodes(cur))
    return True


def _indent_of(lines: list[str], node: ast.AST) -> str:
    line = lines[node.lineno - 1] if node.lineno - 1 < len(lines) else ""
    return line[:len(line) - len(line.lstrip())]


def _first(lines: list[str], node: ast.AST) -> str:
    text = lines[node.lineno - 1].strip() if node.lineno - 1 < len(lines) \
        else ""
    return text[:50] + ("…" if len(text) > 50 else "")


def _one_line(text: str) -> str:
    return " ".join(str(text).split()).replace('"""', "'''")[:200]


def _after_docstring(body: str) -> str:
    """`body` with the future import placed after its module docstring."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return "from __future__ import annotations\n" + body
    lines = body.split("\n")
    if tree.body and _is_docstring(tree.body[0]):
        end = tree.body[0].end_lineno
        return "\n".join(lines[:end] + ["from __future__ import annotations"]
                         + lines[end:])
    return "from __future__ import annotations\n" + body


# ---------------------------------------------------------------------------
# cross-file agreement
# ---------------------------------------------------------------------------

def defined_names(text: str) -> set[str]:
    """Top-level names a module defines or imports (what `from m import n`
    can find in it)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef)):
            out.add(st.name)
        elif isinstance(st, ast.Assign):
            for t in st.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target,
                                                          ast.Name):
            out.add(st.target.id)
        elif isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                out.add((a.asname or a.name).split(".")[0])
        elif type(st).__name__ == "TypeAlias":
            out.add(getattr(getattr(st, "name", None), "id", ""))
    return out


def declared_names(text: str) -> list[str]:
    """The public names a module DEFINES at top level, in order: functions,
    classes, assigned names. Imports are not declarations."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out: list[str] = []
    for st in tree.body:
        names: list[str] = []
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef,
                           ast.ClassDef)):
            names = [st.name]
        elif isinstance(st, ast.Assign):
            names = [t.id for t in st.targets if isinstance(t, ast.Name)]
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target,
                                                          ast.Name):
            names = [st.target.id]
        for name in names:
            if not name.startswith("_") and name not in out:
                out.append(name)
    return out


def broken_contract(pinned: str, code: str) -> list[str]:
    """Names the pinned interface declares that `code` no longer defines.

    D12 of the build spec: "the skeleton says which functions must exist.
    Missing ones are a specific, checkable error — 'you did not implement
    parse_header' — which the model fixes readily." Other files were
    written against these names; a body that drops or renames one breaks
    them without failing its own check.
    """
    have = defined_names(code)
    if not have and code.strip():
        return []                        # does not parse: the syntax check
    return [n for n in declared_names(pinned) if n not in have]


def missing_imports(stubs: dict[str, str], module_of) -> list[str]:
    """`from <planned module> import n` where that module does not define n.

    `module_of(path)` gives a path's dotted module. Star imports and
    modules outside the plan are not judged.
    """
    by_module = {module_of(p): p for p in stubs}
    names = {p: defined_names(t) for p, t in stubs.items()}
    out: list[str] = []
    for path, text in stubs.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for st in ast.walk(tree):
            if not isinstance(st, ast.ImportFrom) or st.level or \
                    not st.module:
                continue
            target = by_module.get(st.module)
            if not target or target == path:
                continue
            for a in st.names:
                if a.name != "*" and a.name not in names[target]:
                    out.append(f"{path} imports `{a.name}` from {target}, "
                               f"which does not define it")
    return out
