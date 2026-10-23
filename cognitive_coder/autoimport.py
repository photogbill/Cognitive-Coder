# SPDX-License-Identifier: Apache-2.0
"""Missing imports a rule can add: the deterministic floor's second rung.

The single most common certain error in a small model's reply is a name it
uses and never imports — `math.sqrt` with no `import math`, `Path(...)`
with no `from pathlib import Path`, `Ship(...)` defined in
`src/models.py` and imported nowhere. The lint gate (loop.py, 2026-10-22)
turns each into a located F821 that fails the attempt, which is right —
and then spends a model call and a verify on a fix whose answer was
never in doubt.

So, before the file is written: the names the file uses and does not
define are resolved by RULE, in this order, and the import is added:

  1. a standard-library MODULE used as a module (`math.sqrt`, `os.path`)
     — `import math`; only when the name is used with an attribute, so a
     bare `random` (the function) is not turned into the module;
  2. a well-known standard-library NAME (`Path`, `dataclass`, `Optional`,
     `defaultdict`) — `from pathlib import Path`, from a fixed table;
  3. a symbol THIS PROJECT defines in exactly one other file, at module
     level — `from src.models import Ship`. Exactly one: two definitions
     is a question for the model, not a rule.

Everything else is left for the lint gate to report, located. The rule
never guesses: a name that is not in the table and not in the project
is not imported from anywhere.

What is added is returned, logged and journaled (`autofix` fixes): if the
same import is added on every file, the PROMPT is the thing to fix.
"""

from __future__ import annotations

import ast
import builtins
import re
import symtable
import sys
from typing import Any

#: Names from the standard library that a small model writes bare, and
#: where each comes from. Deliberately short: a name here is one whose
#: origin is not in doubt.
FROM_TABLE: dict[str, str] = {
    "Path": "pathlib", "PurePath": "pathlib",
    "dataclass": "dataclasses", "field": "dataclasses", "asdict": "dataclasses",
    "Enum": "enum", "IntEnum": "enum", "auto": "enum", "Flag": "enum",
    "Optional": "typing", "List": "typing", "Dict": "typing", "Tuple": "typing",
    "Set": "typing", "Any": "typing", "Callable": "typing", "Union": "typing",
    "Iterable": "typing", "Iterator": "typing", "Sequence": "typing",
    "Mapping": "typing", "Generator": "typing", "Protocol": "typing",
    "TypedDict": "typing", "NamedTuple": "typing", "Literal": "typing",
    "ClassVar": "typing", "Final": "typing", "TypeVar": "typing",
    "defaultdict": "collections", "Counter": "collections",
    "deque": "collections", "namedtuple": "collections",
    "OrderedDict": "collections",
    "partial": "functools", "lru_cache": "functools", "cache": "functools",
    "reduce": "functools", "wraps": "functools",
    "ABC": "abc", "abstractmethod": "abc",
    "datetime": "datetime", "timedelta": "datetime", "date": "datetime",
    "timezone": "datetime",
    "deepcopy": "copy",
    "Decimal": "decimal", "Fraction": "fractions",
    "contextmanager": "contextlib", "suppress": "contextlib",
    "Thread": "threading", "Lock": "threading", "RLock": "threading",
    "Event": "threading",
    "Queue": "queue",
    "chain": "itertools", "product": "itertools", "permutations": "itertools",
    "combinations": "itertools", "groupby": "itertools", "islice": "itertools",
    "TestCase": "unittest",
    "sleep": "time", "perf_counter": "time",
    "uuid4": "uuid",
    "StringIO": "io", "BytesIO": "io",
    "randint": "random", "choice": "random", "shuffle": "random",
    "uniform": "random", "randrange": "random", "sample": "random",
    "sqrt": "math", "floor": "math", "ceil": "math", "hypot": "math",
    "atan2": "math", "radians": "math", "degrees": "math", "isclose": "math",
    "pi": "math", "inf": "math",
}

#: Standard-library modules a rule must never import on a program's behalf.
_NEVER = frozenset({"this", "antigravity", "test", "__hello__", "__phello__",
                    "idlelib", "lib2to3", "turtledemo", "pydoc_data",
                    "ensurepip", "venv", "distutils", "tkinter", "turtle"})
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) - _NEVER
_BUILTINS = frozenset(dir(builtins)) | {"__file__", "__name__", "__doc__",
                                        "__spec__", "__loader__",
                                        "__package__", "__builtins__",
                                        "__path__", "__annotations__",
                                        "__debug__", "__class__"}
_FUTURE = re.compile(r"^from\s+__future__\s+import\b")


def undefined_names(code: str, filename: str = "<reply>") -> set[str]:
    """Names referenced anywhere in the module that no scope defines and
    that are not builtins. Python's own symbol table, so a name assigned
    in one function and used in another is correctly undefined there."""
    try:
        top = symtable.symtable(code, filename, "exec")
    except SyntaxError:
        return set()
    defined_at_top = {s.get_name() for s in top.get_symbols()
                      if s.is_assigned() or s.is_imported()
                      or s.is_parameter()}
    # `def f` / `class C` / `import x` at module level all count as assigned
    out: set[str] = set()

    def walk(table: Any) -> None:
        for s in table.get_symbols():
            name = s.get_name()
            if name in _BUILTINS or name in defined_at_top:
                continue
            if table is top:
                if s.is_referenced() and not (s.is_assigned()
                                              or s.is_imported()):
                    out.add(name)
            elif s.is_global() and s.is_referenced() and not (
                    s.is_assigned() or s.is_imported()
                    or s.is_parameter()):
                out.add(name)
        for child in table.get_children():
            walk(child)

    walk(top)
    # `from __future__ import annotations` makes every annotation a string
    # the symbol table never reads; the names in them are still names.
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        ann = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ann = [a.annotation for a in node.args.args + node.args.kwonlyargs
                   + node.args.posonlyargs if a.annotation is not None]
            if node.args.vararg and node.args.vararg.annotation:
                ann.append(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                ann.append(node.args.kwarg.annotation)
            if node.returns is not None:
                ann.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            ann = [node.annotation]
        for a in ann:
            for sub in ast.walk(a):
                if isinstance(sub, ast.Name) and sub.id not in _BUILTINS \
                        and sub.id not in defined_at_top:
                    out.add(sub.id)
    return out


def _attribute_bases(code: str) -> dict[str, set[str]]:
    """name → the attributes read on it (`math.pi` → {"math": {"pi"}})."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value,
                                                         ast.Name):
            out.setdefault(node.value.id, set()).add(node.attr)
    return out


def _project_source(name: str, path: str, codemap: Any) -> str:
    """The dotted module of the ONE other project file that defines
    `name` at module level, or "" (none, or more than one)."""
    store = getattr(codemap, "store", None)
    if store is None:
        return ""
    try:
        rows = store.find(name)
    except Exception:                                    # noqa: BLE001
        return ""
    hits = []
    for r in rows:
        if r.get("name") != name or "." in str(r.get("name", "")):
            continue
        if r.get("kind") not in ("class", "function", "constant"):
            continue
        p = str(r.get("path", "")).replace("\\", "/")
        if not p.endswith(".py") or p == path:
            continue
        base = p.rsplit("/", 1)[-1]
        if base.startswith("test_") or "/tests/" in f"/{p}":
            continue
        hits.append(p)
    hits = sorted(set(hits))
    if len(hits) != 1:
        return ""
    p = hits[0]
    p = p[:-len("/__init__.py")] if p.endswith("/__init__.py") else p[:-3]
    return p.replace("/", ".")


def _insert_at(code: str) -> int:
    """The line index after the docstring, the __future__ imports and the
    top import block — where a new import belongs."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    lines = code.splitlines()
    at = 0
    body = tree.body
    i = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant) and isinstance(
            body[0].value.value, str):
        at = body[0].end_lineno or 1
        i = 1
    while i < len(body) and isinstance(body[i], (ast.Import, ast.ImportFrom)):
        at = body[i].end_lineno or at
        i += 1
    # keep a comment block that sits directly under the imports with them
    return min(at, len(lines))


def fix(code: str, path: str = "", *, lang: str = "python",
        codemap: Any = None) -> tuple[str, list[str]]:
    """(code with the imports a rule can add, the lines added)."""
    if lang != "python" or not code.strip():
        return code, []
    missing = undefined_names(code, path or "<reply>")
    if not missing:
        return code, []
    bases = _attribute_bases(code)
    adds: list[str] = []
    for name in sorted(missing):
        attrs = bases.get(name, set())
        if name in ("datetime", "date") and attrs:
            # `datetime.now()` is the CLASS; `datetime.datetime.now()` and
            # `datetime.timedelta(...)` are the MODULE. The attributes read
            # on it say which one the file means.
            if attrs & {"datetime", "date", "timedelta", "timezone", "time",
                        "MINYEAR", "MAXYEAR"}:
                adds.append(f"import {name}")
            else:
                adds.append(f"from {FROM_TABLE[name]} import {name}")
        elif attrs and name in _STDLIB:
            adds.append(f"import {name}")
        elif name in FROM_TABLE and not attrs:
            adds.append(f"from {FROM_TABLE[name]} import {name}")
        else:
            mod = _project_source(name, path, codemap)
            if mod:
                adds.append(f"from {mod} import {name}")
    if not adds:
        return code, []
    adds = sorted(a for a in adds if a.startswith("import ")) + \
        sorted(a for a in adds if a.startswith("from "))
    lines = code.splitlines()
    at = _insert_at(code)
    # one blank line between a docstring and the first import
    block = list(adds)
    if at > 0 and lines[at - 1].strip() and not (
            lines[at - 1].lstrip().startswith(("import ", "from "))):
        block = [""] + block
    if at < len(lines) and lines[at].strip() and not (
            lines[at].lstrip().startswith(("import ", "from "))):
        block = block + [""]
    new = lines[:at] + block + lines[at:]
    out = "\n".join(new) + ("\n" if code.endswith("\n") else "")
    try:
        ast.parse(out)
    except SyntaxError:
        return code, []
    return out, adds
