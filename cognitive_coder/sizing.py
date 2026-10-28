# SPDX-License-Identifier: Apache-2.0
"""Plan for the model's reach: modules and functions sized for a small model.

WHY THIS EXISTS, 2026-10-27
---------------------------
Every mechanism downstream of the plan is sized for SMALL units, and the
plan was not. The planner persona asked for "the smallest set of files",
which a model reads as FEWER files — and so bigger ones — and nothing
capped a file or a function:

  * a narrowed edit (symedit.py) returns one definition; a 96-line
    `apply_orders` is a 96-line reply, and a narrowed edit of it is the
    whole function again;
  * a canvas (canvas.py) holds 256 tokens: on the 2026-10-18 build 10 of
    19 canvas repairs were skipped because the body the error was in did
    not fit one — `impact_shield_sector`, `apply_damage`,
    `compute_ai_orders`, every real function body;
  * a FIM hole (fim.py) and the values block (testrun.py) both work best
    on a body short enough to read whole;
  * the dropped-code guard (keep.py) has every name in the file to protect
    on every reply, and a whole-file first attempt has to fit the reply
    budget — a 400-line module is 4,000 tokens the model must write
    perfectly in one go.

So the rule goes where the sizes are decided — the planner, the architect
and the engineer — in words, with the numbers (`SessionConfig.
max_module_lines`, `max_function_lines`), and the engine MEASURES what came
back: a pinned stub that already declares more than a module of that size
can hold is said at the skeleton, and a verified file over the cap is a
`size` journal event, a line in BUILD_LOG.txt and a caveat in the report —
never a refusal. A working 300-line file is still working; the cap is a
preference the next Improve pass can act on, said where the operator will
read it.

Pure: nothing here talks to a model or a file.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
import re

#: The defaults. 200 lines is about 2,000 tokens — a whole-file first
#: attempt a 24B model writes in one reply with room to spare; 40 lines is
#: a body a narrowed edit, a canvas (~36 lines at 256 tokens) or a FIM hole
#: takes whole. Both are SessionConfig fields; 0 switches the rule off.
DEFAULT_MODULE_LINES = 200
DEFAULT_FUNCTION_LINES = 40

#: A pinned stub declaring more public definitions than this is likely to
#: grow past `max_module_lines` once its bodies are written: the average
#: body under the function cap is ~20 lines, plus the signature and the
#: docstring the stub already carries.
_LINES_PER_DEF = 20


@dataclass(frozen=True)
class DefSize:
    name: str
    lines: int
    line: int             # 1-based first line of the definition


@dataclass(frozen=True)
class Measure:
    """What a file measures, by rule."""
    path: str
    lines: int
    defs: tuple[DefSize, ...] = ()
    max_module: int = DEFAULT_MODULE_LINES
    max_function: int = DEFAULT_FUNCTION_LINES
    parsed: bool = True
    notes: tuple[str, ...] = field(default=())

    @property
    def over_module(self) -> bool:
        return bool(self.max_module) and self.lines > self.max_module

    @property
    def over_functions(self) -> tuple[DefSize, ...]:
        if not self.max_function:
            return ()
        return tuple(d for d in self.defs if d.lines > self.max_function)

    @property
    def over(self) -> bool:
        return self.over_module or bool(self.over_functions)

    def describe(self) -> str:
        """One sentence for BUILD_LOG.txt and the report, or ""."""
        parts = []
        if self.over_module:
            parts.append(f"{self.lines} lines (the rule is "
                         f"{self.max_module})")
        longest = sorted(self.over_functions, key=lambda d: -d.lines)[:3]
        if longest:
            named = ", ".join(f"`{d.name}` ({d.lines} lines)"
                              for d in longest)
            more = len(self.over_functions) - len(longest)
            parts.append(f"{named}{f' and {more} more' if more > 0 else ''} "
                         f"over the {self.max_function}-line rule for a "
                         f"function")
        if not parts:
            return ""
        return (f"{self.path} is larger than this engine is sized for — "
                + "; ".join(parts)
                + ". It works; a split into smaller pieces would make every "
                  "repair, hole and narrowed edit of it cheaper and surer.")

    def summary(self) -> dict:
        return {"path": self.path, "lines": self.lines,
                "max_module": self.max_module,
                "max_function": self.max_function,
                "over_module": self.over_module,
                "over_functions": [{"name": d.name, "lines": d.lines,
                                    "line": d.line}
                                   for d in self.over_functions]}


# --------------------------------------------------------------------------
# measuring
# --------------------------------------------------------------------------

_DEF_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?:pub|static|inline|export|async|public|"
    r"private|protected|fn|func|function|def|void|int|float|double|bool|"
    r"string|String|var|let|const)\s+)+(?P<name>[A-Za-z_]\w*)\s*[(<]", re.M)


def measure(code: str, path: str = "", lang: str = "python", *,
            max_module: int = DEFAULT_MODULE_LINES,
            max_function: int = DEFAULT_FUNCTION_LINES) -> Measure:
    """Lines, and the lines of every function and method.

    Python is measured from the AST (a function's span, decorators to its
    last line; a nested function belongs to its parent). Other languages
    are measured by a pattern — a definition runs to the next definition
    at the same or a shallower indentation — and said to be approximate.
    """
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    n = len(lines)
    if lang == "python":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return Measure(path, n, (), max_module, max_function,
                           parsed=False)
        defs: list[DefSize] = []

        def walk(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef,
                                      ast.AsyncFunctionDef)):
                    first = min([child.lineno,
                                 *[d.lineno for d in child.decorator_list]])
                    last = child.end_lineno or child.lineno
                    defs.append(DefSize(prefix + child.name,
                                        last - first + 1, first))
                elif isinstance(child, ast.ClassDef):
                    walk(child, prefix + child.name + ".")
                elif isinstance(child, ast.stmt):
                    # a definition under `if TYPE_CHECKING:`, `try:` or
                    # `with` at module or class level is still a definition
                    # of that module or class (the 2026-10-27 review)
                    walk(child, prefix)
        walk(tree, "")
        return Measure(path, n, tuple(defs), max_module, max_function)
    found = [(m.start(), m.group("name"), len(m.group("indent")))
             for m in _DEF_RE.finditer(text)]
    defs = []
    starts = [text.count("\n", 0, pos) for pos, _n, _i in found]
    for i, (_pos, name, indent) in enumerate(found):
        end = n
        for j in range(i + 1, len(found)):
            if found[j][2] <= indent:
                end = starts[j]
                break
        defs.append(DefSize(name, max(1, end - starts[i]), starts[i] + 1))
    return Measure(path, n, tuple(defs), max_module, max_function,
                   notes=("approximate: definitions found by pattern",)
                   if defs else ())


# --------------------------------------------------------------------------
# the rule, in words, for the prompts
# --------------------------------------------------------------------------

def rule_text(max_module: int = DEFAULT_MODULE_LINES,
              max_function: int = DEFAULT_FUNCTION_LINES) -> str:
    """The convention every prompt of a session carries (the cached
    prefix), or "" when both caps are off."""
    parts = []
    if max_module:
        parts.append(f"keep every module under about {max_module} lines — "
                     f"when a part of the request needs more, split it "
                     f"into two modules, each with its own test file")
    if max_function:
        parts.append(f"keep every function and method under about "
                     f"{max_function} lines — a longer one is split into "
                     f"helpers, each named for what it does")
    if not parts:
        return ""
    return ("[SIZE] Code here is written and repaired one function at a "
            "time by a small model, so " + "; ".join(parts) + ".")


def planner_rule(max_module: int = DEFAULT_MODULE_LINES,
                 max_function: int = DEFAULT_FUNCTION_LINES) -> str:
    """The planner's sentence: more files when the request is big."""
    if not max_module:
        return ""
    return (f"Size each file for about {max_module} lines at most"
            + (f" (functions under {max_function} lines)" if max_function
               else "")
            + ": a part of the request that needs more is two files, not "
              "one big one. More small files beat fewer large ones here.")


def architect_rule(max_module: int = DEFAULT_MODULE_LINES,
                   max_function: int = DEFAULT_FUNCTION_LINES) -> str:
    """The architect's bullet: the contract decides the sizes."""
    parts = []
    if max_module:
        # The architect writes one stub per PLANNED file and cannot add
        # one (a block for an unplanned path is dropped by `split_reply`,
        # while an import of it would survive) — so it is asked to SAY
        # when a module will not fit, never to split the plan itself
        # (the 2026-10-27 review).
        parts.append(f"a module's bodies, once written, should fit in "
                     f"about {max_module} lines — keep every file the "
                     f"plan lists, and for one that will plainly need "
                     f"more, say so in a one-line comment at the top of "
                     f"its stub (`# SIZE: this module will run long — "
                     f"<what could be split out>`) and keep its helpers "
                     f"small")
    if max_function:
        parts.append(f"no single function or method should need more than "
                     f"about {max_function} lines of body — split a large "
                     f"step into named helpers in the stub, so each body "
                     f"is written and repaired on its own")
    if not parts:
        return ""
    return "Size the contract for a small model: " + "; ".join(parts) + "."


# --------------------------------------------------------------------------
# the skeleton: a stub that will not fit its module
# --------------------------------------------------------------------------

def stub_warning(stub: str, path: str, *,
                 max_module: int = DEFAULT_MODULE_LINES) -> str:
    """"" or one sentence: this pinned stub declares more than a module of
    `max_module` lines is likely to hold once its bodies are written."""
    if not max_module or not (stub or "").strip():
        return ""
    try:
        tree = ast.parse(stub)
    except SyntaxError:
        return ""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and not node.name.startswith("_"):
            count += 1
    stub_lines = len(stub.splitlines())
    likely = stub_lines + count * _LINES_PER_DEF
    if count and likely > max_module:
        return (f"{path}: its pinned interface declares {count} public "
                f"definitions in {stub_lines} lines of stub; with their "
                f"bodies it will likely run to ~{likely} lines, over the "
                f"{max_module}-line rule — consider splitting it before "
                f"the build, or expect its repairs to be slower")
    return ""


def over_caps(measures: Sequence[Measure]) -> list[Measure]:
    return [m for m in measures if m.over]
