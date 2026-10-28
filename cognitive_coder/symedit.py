# SPDX-License-Identifier: Apache-2.0
"""Narrowed edits: a reply that holds only the definitions it changes (F5).

WHY THIS EXISTS, 2026-10-19
---------------------------
F5 of the build specification — "progressive narrowing of the unit of work:
whole file → one class → one function → one line range" — was specified and
never built: `LoopConfig.narrow_on_stagnation` existed and nothing read it.
Every repair and every change to an existing file asked an ordinary model
for the COMPLETE file. That costs three ways on a small local model:

  * **Decode time.** A whole-file reply runs to `max_tokens` (2,048) before
    continuation. At the 3–5 tok/s Devstral ran at on 2026-10-16 that is
    7–11 minutes per attempt; one function is a few hundred tokens.
  * **Collateral damage (D6).** A model rewriting 400 lines to fix one
    touches others — or leaves them out ("# ... rest unchanged").
  * **Large files could not be repaired at all.** A file too large to show
    whole was refused before any model call.

The shape chosen here follows Aider's published results: whole-file replies
are the format small models get right (Qwen2.5-Coder-32B: 99.6 % well-formed
whole, 71.6 % with search/replace blocks), and its unified-diff study's
advice is to have the model rewrite WHOLE FUNCTIONS rather than make line
edits. A narrowed reply is exactly that — whole-file output, at function
size. The model writes code in the one format it is good at; this module
puts it back by NAME, which is how it cannot land in the wrong place.

WHAT THIS MODULE DECIDES, BY RULE (C5) — the model decides nothing here:

  1. WHERE each definition in the file is (`index`): functions, classes,
     methods (`Cls.meth`), simple module-level names, the main guard, the
     imports. A name defined twice (a property and its setter) is
     AMBIGUOUS and is never replaced by name.
  2. WHAT a reply holds (`parse_reply`): its definitions, its imports, and
     anything else — which, unless it is already in the file word for word,
     refuses the WHOLE reply rather than dropping part of a change.
  3. WHERE each piece goes (`apply`): a definition the file has is replaced
     in place, re-indented to where it lives; a method may come inside its
     `class` line or bare; a new definition goes before the main guard; an
     import the file lacks joins the leading import block. The result must
     parse and must not define anything twice, or nothing is applied.
  4. WHAT the model sees of a file too large to show whole
     (`condensed_view`): the definitions to change in full, every other one
     as its header with the body elided — and a replacement whose body is a
     placeholder (`...`, `pass`) where real code was is never spliced.

Python only: the AST is what makes "put it back by name" exact. Every other
language keeps the whole-file path, and the build log says so.

Nothing here talks to a model or a file. `loop.py` does that; this module is
pure so that every rule above is a unit test.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import io
import re
import textwrap
import tokenize

#: Where Python's parser ends a line (the same rule as canvas.py): "\r\n",
#: "\n" or a lone "\r" — never a form feed or a Unicode separator.
_LINE_END = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")

#: What the condensed view puts where a body is not shown. A real
#: statement and a comment, so the view itself parses — and so a model that
#: copies it back produces a body `apply` recognises as a placeholder.
ELIDED = "...  # body not shown here; it stays as it is"

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def lines_of(code: str) -> list[str]:
    """The file's lines, each with its line ending, as the parser counts."""
    return [part for part in _LINE_END.split(code or "") if part]


def _norm(code: str) -> str:
    """One line ending, and no byte-order mark: a file saved by a Windows
    editor starts with U+FEFF, which `ast.parse` refuses — and a guard that
    cannot parse the file is a guard that silently checks nothing."""
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n")
    return text[1:] if text.startswith("\ufeff") else text


def _is_docstring(node: ast.AST) -> bool:
    return (isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str))


def _first_line(node: ast.AST) -> int:
    """1-based first line of a definition, its decorators included."""
    decos = [d.lineno for d in getattr(node, "decorator_list", []) or []]
    return min([node.lineno, *decos])


def def_key(node: ast.AST) -> str:
    """The name a definition is known by here: its own, or — for a property's
    setter or deleter, which share the getter's name — `v@setter`, so the
    three are three definitions, not one AMBIGUOUS name."""
    name = getattr(node, "name", "")
    for deco in getattr(node, "decorator_list", []) or []:
        if (isinstance(deco, ast.Attribute)
                and deco.attr in ("setter", "deleter", "getter")
                and isinstance(deco.value, ast.Name)
                and deco.value.id == name):
            return f"{name}@{deco.attr}"
    return name


def shown_name(name: str) -> str:
    """`Cls.v@setter` as a person reads it: `Cls.v` (setter)."""
    if "@" in name:
        base, _, variant = name.partition("@")
        return f"{base} ({variant})"
    return name


def assign_name(node: ast.AST) -> str:
    """The one plain name a simple assignment binds, else ""."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1 \
            and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return ""


def is_main_guard(node: ast.AST) -> bool:
    """`if __name__ == "__main__":`, either way round."""
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    t = node.test
    if len(t.ops) != 1 or not isinstance(t.ops[0], ast.Eq) \
            or len(t.comparators) != 1:
        return False
    sides = (t.left, t.comparators[0])
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__"
                    for s in sides))


def body_is_stub(node: ast.AST) -> bool:
    """A body that is only a docstring and `...` / `pass` /
    `raise NotImplementedError` — what an elided view invites back."""
    body = list(getattr(node, "body", []) or [])
    if body and _is_docstring(body[0]):
        body = body[1:]
    if not body:
        return True
    for st in body:
        if isinstance(st, ast.Pass):
            continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant) \
                and st.value.value is Ellipsis:
            continue
        if isinstance(st, ast.Raise) and st.exc is not None:
            exc = st.exc.func if isinstance(st.exc, ast.Call) else st.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                continue
        return False
    return True


# --------------------------------------------------------------------------
# where each definition is
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """One definition in a file: 0-based [start, end) lines, decorators in."""
    name: str            # "f", "Cls", "Cls.meth", "LIMIT", "__main__"
    kind: str            # function | class | method | constant | main
    start: int
    end: int
    indent: int          # columns of the def / class / assignment line
    parent: str = ""     # the class a method or nested class belongs to

    @property
    def label(self) -> str:
        if self.end <= self.start + 1:
            return f"line {self.start + 1}"
        return f"lines {self.start + 1}-{self.end}"


@dataclass(frozen=True)
class Index:
    """Every definition of one file, and where new imports would go."""
    spans: dict
    ambiguous: frozenset
    imports: tuple          # ast.dump of every top-level import statement
    import_end: int         # 0-based line new imports go before
    doc_end: int            # 0-based line after the module docstring
    lines: int
    #: names defined INSIDE a top-level `if` / `try` / `with` / loop — a
    #: reply that defines one at the top would shadow it, not replace it
    hidden: frozenset = frozenset()

    def get(self, name: str) -> Span | None:
        return self.spans.get(name)

    def methods_named(self, short: str) -> list[Span]:
        return [s for s in self.spans.values()
                if s.kind == "method" and s.name.rsplit(".", 1)[-1] == short]

    def method_named(self, short: str) -> bool:
        """Some class has a method called ``short`` (ambiguous or not)."""
        return bool(self.methods_named(short)) or any(
            n.rsplit(".", 1)[-1] == short and "." in n
            for n in self.ambiguous)

    def members_of(self, cls: str) -> list[Span]:
        return [s for s in self.spans.values() if s.parent == cls]

    def names(self, kinds: Sequence[str] = ()) -> list[str]:
        return [n for n, s in self.spans.items()
                if not kinds or s.kind in kinds]


def index(code: str) -> Index | None:
    """Where every definition of ``code`` is — or None if it does not parse."""
    text = _norm(code)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    lines = lines_of(text)
    spans: dict[str, Span] = {}
    seen: dict[str, int] = {}

    def add(span: Span) -> None:
        seen[span.name] = seen.get(span.name, 0) + 1
        spans[span.name] = span

    def walk_class(node: ast.ClassDef, qual: str) -> None:
        for child in node.body:
            if isinstance(child, _DEFS):
                add(Span(f"{qual}.{def_key(child)}", "method",
                         _first_line(child) - 1,
                         child.end_lineno or child.lineno, child.col_offset,
                         qual))
            elif isinstance(child, ast.ClassDef):
                sub = f"{qual}.{child.name}"
                add(Span(sub, "class", _first_line(child) - 1,
                         child.end_lineno or child.lineno, child.col_offset,
                         qual))
                walk_class(child, sub)

    body = list(tree.body)
    doc_end = 0
    if body and _is_docstring(body[0]):
        doc_end = body[0].end_lineno or body[0].lineno
    else:
        # a shebang or an encoding line stays the first line of the file
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("#!") or (s.startswith("#") and "coding" in s
                                      and i < 2):
                doc_end = i + 1
            else:
                break
    imports: list[str] = []
    hidden: set[str] = set()
    import_end = doc_end
    defined_yet = False
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.dump(node))
            # only the LEADING import block decides where new imports go:
            # an import late in the file (inside a try, after code) is the
            # file's business, not a place to grow the block
            if not defined_yet:
                import_end = max(import_end, node.end_lineno or node.lineno)
            continue
        if not (_is_docstring(node) and node is body[0]):
            defined_yet = defined_yet or isinstance(
                node, (*_DEFS, ast.ClassDef))
        if isinstance(node, _DEFS):
            add(Span(def_key(node), "function", _first_line(node) - 1,
                     node.end_lineno or node.lineno, node.col_offset))
        elif isinstance(node, ast.ClassDef):
            add(Span(node.name, "class", _first_line(node) - 1,
                     node.end_lineno or node.lineno, node.col_offset))
            walk_class(node, node.name)
        elif assign_name(node):
            add(Span(assign_name(node), "constant", node.lineno - 1,
                     node.end_lineno or node.lineno, node.col_offset))
        elif is_main_guard(node):
            add(Span("__main__", "main", node.lineno - 1,
                     node.end_lineno or node.lineno, node.col_offset))
        elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For,
                               ast.While)):
            hidden.update(_bound_inside(node))
    ambiguous = frozenset(n for n, c in seen.items() if c > 1)
    for name in ambiguous:
        spans.pop(name, None)
    return Index(spans=spans, ambiguous=ambiguous, imports=tuple(imports),
                 import_end=import_end, doc_end=doc_end, lines=len(lines),
                 hidden=frozenset(hidden - set(spans)))


def _bound_inside(node: ast.AST) -> set[str]:
    """Names a compound statement defines anywhere inside it."""
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (*_DEFS, ast.ClassDef)):
            out.add(sub.name)
        elif assign_name(sub):
            out.add(assign_name(sub))
    return out


def enclosing(idx: Index, line: int) -> str:
    """The innermost function or method holding 1-based ``line``, else ""."""
    best: Span | None = None
    for s in idx.spans.values():
        if s.kind not in ("function", "method"):
            continue
        if s.start < line <= s.end and (
                best is None or (s.end - s.start) < (best.end - best.start)):
            best = s
    return best.name if best is not None else ""


# --------------------------------------------------------------------------
# what a reply holds
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """One top-level item of a reply."""
    kind: str            # import | function | class | constant | main | other
    name: str
    text: str            # its whole lines, at column 0, ending in "\n"
    node: ast.AST = field(compare=False, repr=False, default=None)
    #: every line of the reply, so a CHILD of this node can be cut out by
    #: its own line numbers (which count from the reply, not the piece)
    src: tuple = field(compare=False, repr=False, default=())


_FENCED = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)```", re.S)
_PY_TAGS = ("", "python", "py", "python3", "py3")


def _is_contract_example(block: str) -> bool:
    """The output contract's own worked example, echoed back verbatim: not
    a reply, never placed (the 2026-10-27 review)."""
    from . import personas
    return (" ".join(block.split()) == " ".join(
        personas.SYMBOLS_EXAMPLE.split()))


def reply_code(text: str) -> str:
    """The code of a reply: every Python fence, in order, else the text —
    the contract's own example, repeated, left out."""
    body = _norm(text)
    fences = [m.group(2) for m in _FENCED.finditer(body)
              if m.group(1).lower() in _PY_TAGS]
    blocks = [b for b in fences if not _is_contract_example(b)]
    if fences and not blocks:
        return ""           # the example alone is no reply at all
    if blocks:
        return "\n\n".join(b.strip("\n") + "\n" for b in blocks)
    m = re.search(r"```[ \t]*(?:python3?|py3?)?[ \t]*\n", body)
    if m and body.count("```") == 1:
        return body[m.end():]
    return body


def _cut(lines: Sequence[str], node: ast.AST) -> str:
    start = _first_line(node) - 1
    end = node.end_lineno or node.lineno
    return "".join(ln if ln.endswith("\n") else ln + "\n"
                   for ln in lines[start:end])


def parse_reply(text: str) -> tuple[list[Piece], str]:
    """(pieces, problem). ``problem`` is a sentence when nothing can be used."""
    code = reply_code(text)
    if not code.strip():
        return [], "the reply holds no code"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        dedented = textwrap.dedent(code)
        try:
            tree = ast.parse(dedented)
            code = dedented
        except SyntaxError:
            where = f" (line {exc.lineno}: {exc.msg})" if exc.lineno else ""
            return [], f"the reply is not valid Python{where}"
    lines = tuple(lines_of(code))
    pieces: list[Piece] = []
    for node in tree.body:
        src = _cut(lines, node)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pieces.append(Piece("import", "", src, node, lines))
        elif isinstance(node, _DEFS):
            pieces.append(Piece("function", node.name, src, node, lines))
        elif isinstance(node, ast.ClassDef):
            pieces.append(Piece("class", node.name, src, node, lines))
        elif assign_name(node):
            pieces.append(Piece("constant", assign_name(node), src, node,
                                lines))
        elif is_main_guard(node):
            pieces.append(Piece("main", "__main__", src, node, lines))
        elif _is_docstring(node) or isinstance(node, ast.Pass) or (
                isinstance(node, ast.Expr)
                and isinstance(getattr(node, "value", None), ast.Constant)
                and node.value.value is Ellipsis):
            continue                     # a docstring, `pass`, a bare `...`
        else:
            pieces.append(Piece("other", "", src, node, lines))
    if not pieces:
        return [], "the reply holds no definition"
    return pieces, ""


# --------------------------------------------------------------------------
# putting the pieces back
# --------------------------------------------------------------------------

def _string_continuations(text: str) -> set[int]:
    """1-based lines that lie INSIDE a multi-line string (after its first
    line) — re-indenting them would change the string's value."""
    inside: set[int] = set()
    # Python 3.12 tokenizes an f-string as FSTRING_START … FSTRING_END, not
    # one STRING token: without these, a multi-line f-string's lines were
    # re-indented and its value changed
    f_start = getattr(tokenize, "FSTRING_START", None)
    f_end = getattr(tokenize, "FSTRING_END", None)
    opened: list[int] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
                inside.update(range(tok.start[0] + 1, tok.end[0] + 1))
            elif f_start is not None and tok.type == f_start:
                opened.append(tok.start[0])
            elif f_end is not None and tok.type == f_end and opened:
                first = opened.pop()
                if tok.end[0] > first:
                    inside.update(range(first + 1, tok.end[0] + 1))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return inside


def reindent(text: str, from_col: int, lead: str) -> str:
    """``text``, whose definition line sits at column ``from_col``, moved so
    that line starts with ``lead``. Lines inside multi-line strings and
    blank lines are left as they are (a blank line loses its spaces)."""
    keep = _string_continuations(text)
    out = []
    for no, line in enumerate(text.splitlines(keepends=True), 1):
        if no in keep:
            out.append(line)
            continue
        if not line.strip():
            out.append("\n")
            continue
        cut = 0
        rest = line
        while cut < from_col and rest[:1] in (" ", "\t"):
            rest = rest[1:]
            cut += 1
        out.append(lead + rest)
    return "".join(out)


def _node_at(tree: ast.AST, start: int) -> ast.AST | None:
    """The def or class whose first line (decorators in) is 0-based start."""
    for node in ast.walk(tree):
        if isinstance(node, (*_DEFS, ast.ClassDef)) \
                and _first_line(node) - 1 == start:
            return node
    return None


@dataclass
class Applied:
    """What `apply` did — or, with ``problem`` set, why it did nothing."""
    code: str = ""
    replaced: list = field(default_factory=list)      # qualified names
    added: list = field(default_factory=list)         # qualified names
    imports: list = field(default_factory=list)       # import lines added
    skipped: list = field(default_factory=list)       # (name, why)
    problem: str = ""
    spans: dict = field(default_factory=dict)         # name -> old label
    #: every definition in the reply was the file's own, unchanged — the
    #: model's way of saying "nothing here needs to change"
    unchanged: bool = False

    @property
    def ok(self) -> bool:
        return not self.problem

    def describe(self) -> str:
        if self.problem:
            return f"not placed: {self.problem}"
        bits = []
        if self.replaced:
            bits.append("replaced " + ", ".join(
                f"`{shown_name(n)}`" + (f" ({self.spans[n]})"
                                        if self.spans.get(n) else "")
                for n in self.replaced))
        if self.added:
            bits.append("added " + ", ".join(f"`{shown_name(n)}`"
                                             for n in self.added))
        if self.imports:
            bits.append("added " + "; ".join(f"`{i}`" for i in self.imports))
        left = [(n, w) for n, w in self.skipped if w != "unchanged"]
        if left:
            bits.append("left as it was: " + "; ".join(
                f"`{n}` ({why})" for n, why in left))
        return "; ".join(bits) or "nothing changed"


#: Ops at the same line are applied in this order, highest first, so a
#: REPLACE runs before anything is inserted in front of it, and inserts end
#: up top to bottom as: future imports, imports, constants, new definitions.
_REPLACE, _NEW_TOP, _CONST, _IMPORT, _FUTURE = 9, 3, 2, 1, 0


@dataclass(order=True)
class _Op:
    at: int
    order: int
    #: reply order among ops at the same line and order: applied highest
    #: first, so the FIRST of several inserts at one line ends up on top —
    #: without it a helper and the main guard that calls it, or a base class
    #: and its subclass, came out in reverse and failed at import
    seq: int = 0
    end: int = field(compare=False, default=0)
    text: str = field(compare=False, default="")
    kind: str = field(compare=False, default="replace")



def _lead_of(lines: Sequence[str], start: int) -> str:
    line = lines[start] if 0 <= start < len(lines) else ""
    return line[:len(line) - len(line.lstrip(" \t"))]


def _member_lead(lines: Sequence[str], idx: Index, cls: Span) -> str:
    members = sorted(idx.members_of(cls.name), key=lambda s: s.start)
    if members:
        return _lead_of(lines, members[0].start)
    head = _lead_of(lines, cls.start)
    return head + ("\t" if head.startswith("\t") else "    ")


class _Refuse(Exception):
    """One reason the whole reply is not applied."""


def apply(code: str, reply: str, *, targets: Sequence[str] = (),
          allow_new: bool = True) -> Applied:
    """``code`` with the definitions in ``reply`` put back by name.

    ``targets`` are the qualified names the model was asked to change; they
    resolve a bare method (`def update(self, dt):` at column 0) to its
    class. Anything that cannot be placed exactly refuses the WHOLE reply —
    a half-applied change verifies as something nobody asked for.
    """
    text = _norm(code)
    idx = index(text)
    if idx is None:
        return Applied(problem="the file does not parse, so its definitions "
                               "cannot be found by name")
    pieces, problem = parse_reply(reply)
    if problem:
        return Applied(problem=problem)
    lines = lines_of(text)
    tree = ast.parse(text)
    plan = _Plan(idx, lines, tree, allow_new)
    try:
        for piece in pieces:
            plan.take(piece, targets)
    except _Refuse as why:
        return Applied(problem=str(why), skipped=plan.result.skipped)
    result = plan.result
    if not plan.ops:
        if result.skipped and all(w == "unchanged" for _n, w in
                                  result.skipped):
            return Applied(problem="every definition in the reply is the "
                                   "same as in the file — nothing changed",
                           skipped=result.skipped, unchanged=True)
        return Applied(problem="nothing in the reply could be placed" + (
            ": " + "; ".join(f"`{n}` ({w})" for n, w in result.skipped)
            if result.skipped else ""), skipped=result.skipped)
    out = list(lines)
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"
    for op in sorted(plan.ops, reverse=True):
        if op.kind == "replace":
            out[op.at:op.end] = [op.text]
        else:
            out[op.at:op.at] = [op.text]
    new = "".join(out)
    try:
        ast.parse(new)
    except SyntaxError as exc:
        return Applied(problem=f"the file would not parse with the reply in "
                               f"place (line {exc.lineno}: {exc.msg})",
                       skipped=result.skipped)
    from .canvas import duplicate_definitions
    twice = duplicate_definitions(new, text)
    if twice:
        return Applied(problem=f"the file would {twice}",
                       skipped=result.skipped)
    result.code = new
    return result


class _Plan:
    """The ops one reply turns into, gathered before anything is applied."""

    def __init__(self, idx: Index, lines: Sequence[str], tree: ast.AST,
                 allow_new: bool) -> None:
        self.idx = idx
        self.lines = lines
        self.tree = tree
        self.allow_new = allow_new
        self.top = {ast.dump(n) for n in tree.body}
        self.ops: list[_Op] = []
        self.claimed: list[tuple[int, int]] = []
        self.result = Applied()
        self.new_imports: list[str] = []
        self.future: list[str] = []
        #: names the reply ADDS at the top level, so a new constant that
        #: uses one goes after it, not above it at the imports
        self.new_names: set[str] = set()
        self._seq = 0

    def op(self, at: int, order: int, text: str, kind: str,
           end: int = 0) -> None:
        # `seq` counts UP: sorted highest first, the earliest op at a line
        # is applied LAST — and so, inserted in front of the others, ends
        # up on top, in the order the reply gave
        self._seq += 1
        self.ops.append(_Op(at, order, self._seq, end or at, text, kind))

    # -- the one primitive --------------------------------------------------
    def replace(self, span: Span, piece_text: str, from_col: int,
                new_node: ast.AST | None) -> None:
        old_node = _node_at(self.tree, span.start)
        if (new_node is not None and old_node is not None
                and isinstance(new_node, _DEFS)
                and body_is_stub(new_node) and not body_is_stub(old_node)):
            self.result.skipped.append(
                (span.name, "the reply's body was only a placeholder"))
            return
        for a, b in self.claimed:
            if span.start < b and a < span.end:
                self.result.skipped.append(
                    (span.name, "it overlaps another part of the reply"))
                return
        body = reindent(piece_text, from_col, _lead_of(self.lines,
                                                       span.start))
        old = "".join(self.lines[span.start:span.end])
        if _norm(old).rstrip() == body.rstrip():
            self.result.skipped.append((span.name, "unchanged"))
            return
        self.claimed.append((span.start, span.end))
        self.op(span.start, _REPLACE, body, "replace", end=span.end)
        self.result.replaced.append(span.name)
        self.result.spans[span.name] = span.label

    # -- one piece of the reply -----------------------------------------------
    def take(self, piece: Piece, targets: Sequence[str]) -> None:
        idx = self.idx
        if piece.kind == "import":
            if ast.dump(piece.node) in idx.imports:
                return
            if isinstance(piece.node, ast.ImportFrom) \
                    and piece.node.module == "__future__":
                if piece.text not in self.future:
                    self.future.append(piece.text)
                    self.op(idx.doc_end, _FUTURE, piece.text, "insert")
                    self.result.imports.append(piece.text.strip())
                return
            if piece.text not in self.new_imports:
                self.new_imports.append(piece.text)
                self.op(idx.import_end, _IMPORT, piece.text, "insert")
                self.result.imports.append(piece.text.strip())
            return
        if piece.kind == "other":
            if ast.dump(piece.node) in self.top:
                return                       # already in the file as is
            first = piece.text.strip().splitlines()[0][:60]
            raise _Refuse(f"the reply holds a top-level statement that is "
                          f"not a definition (`{first}`); a narrowed reply "
                          f"can only replace or add whole definitions")
        name = def_key(piece.node) if piece.kind in ("function", "class") \
            else piece.name
        if name in idx.hidden and name not in idx.spans:
            raise _Refuse(f"`{name}` is defined inside an `if`, `try` or "
                          f"`with` block in this file; a narrowed reply "
                          f"would add a second one beside it rather than "
                          f"replace it — return the whole file")
        span = idx.get(name)
        if piece.kind == "main":
            if span is not None:
                self.replace(span, piece.text, 0, None)
            elif self.allow_new:
                self.add_top(piece.text, "__main__")
            return
        if piece.kind == "constant":
            if name in idx.ambiguous:
                self.result.skipped.append(
                    (name, "the file assigns it more than once, so it "
                           "cannot be replaced by name"))
                return
            if span is not None and span.kind == "constant":
                self.replace(span, piece.text, 0, None)
            elif span is not None:
                raise _Refuse(f"`{name}` is a {span.kind} in the file and an "
                              f"assignment in the reply")
            elif self.allow_new:
                self.add_constant(piece, name)
            return
        if piece.kind == "function":
            # A METHOD FIRST, when the reply says it is one: `def run(self)`
            # with the error in `App.run` is App's method, even though the
            # file also has a top-level `run(argv)` — which it would
            # otherwise have silently overwritten.
            if self._looks_like_method(piece, name, targets):
                method = self._bare_method(piece, name, targets)
                if method is not None:
                    self.replace(method, piece.text, 0, piece.node)
                    return
            if name in idx.ambiguous:
                self.result.skipped.append(
                    (shown_name(name), "the file defines it more than once, "
                                       "so it cannot be replaced by name"))
                return
            if span is not None and span.kind == "function":
                self.replace(span, piece.text, 0, piece.node)
                return
            if span is not None:
                raise _Refuse(f"`{name}` is a {span.kind} in the file and a "
                              f"function in the reply")
            method = self._bare_method(piece, name, targets)
            if method is not None:
                self.replace(method, piece.text, 0, piece.node)
                return
            if self.allow_new:
                self.add_top(piece.text, name)
            return
        if piece.kind == "class":
            if name in idx.ambiguous:
                self.result.skipped.append(
                    (name, "the file defines it more than once, so it "
                           "cannot be replaced by name"))
                return
            if span is None:
                if self.allow_new:
                    self.add_top(piece.text, name)
                return
            if span.kind != "class":
                raise _Refuse(f"`{name}` is a {span.kind} in the file and a "
                              f"class in the reply")
            self.merge_class(piece.node, piece.src, span)

    def _looks_like_method(self, piece: Piece, name: str,
                           targets: Sequence[str]) -> bool:
        args = getattr(piece.node, "args", None)
        first = args.args[0].arg if args is not None and args.args else ""
        if first in ("self", "cls") and self.idx.method_named(name):
            return True
        named_method = any(t.rsplit(".", 1)[-1] == name and "." in t
                           for t in targets if t)
        return named_method and name not in targets

    def _bare_method(self, piece: Piece, short: str,
                     targets: Sequence[str]) -> Span | None:
        """A top-level `def` in the reply that is really one method."""
        wanted = [self.idx.get(t) for t in targets
                  if t and t.rsplit(".", 1)[-1] == short
                  and self.idx.get(t) is not None
                  and self.idx.get(t).kind == "method"]
        if len(wanted) == 1:
            return wanted[0]
        found = self.idx.methods_named(short)
        args = getattr(piece.node, "args", None)
        first = args.args[0].arg if args is not None and args.args else ""
        if len(found) == 1 and first in ("self", "cls"):
            return found[0]
        if len(found) > 1 and first in ("self", "cls"):
            raise _Refuse(f"`{short}` is a method of more than one class; "
                          f"write it inside its `class` line")
        return None

    def add_top(self, text: str, name: str) -> None:
        main = self.idx.get("__main__")
        if main is not None:
            at = main.start
            block = text.rstrip("\n") + "\n\n\n"
        else:
            at = self.idx.lines
            block = "\n\n" + text.rstrip("\n") + "\n"
        self.op(at, _NEW_TOP, block, "insert")
        self.new_names.add(name)
        self.result.added.append(name)

    def add_constant(self, piece: Piece, name: str) -> None:
        """A new module-level name: after the imports when it is a plain
        value; with the new definitions when it USES a name the file or the
        reply defines — `HANDLERS = {"b": handle_b}` above `handle_b` is a
        NameError at import."""
        used = {n.id for n in ast.walk(piece.node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        defined = {n for n in self.idx.spans if "." not in n} | self.new_names
        if used & defined:
            self.add_top(piece.text, name)
            return
        at = self.idx.import_end
        self.op(at, _CONST, ("\n" if at > 0 else "") + piece.text, "insert")
        self.new_names.add(name)
        self.result.added.append(name)

    def merge_class(self, node: ast.ClassDef, src: Sequence[str],
                    span: Span) -> None:
        """A class in the reply: the whole class, or some of its members —
        and a nested class inside a partial reply is merged the same way,
        never replaced whole."""
        old = _node_at(self.tree, span.start)
        old_body = list(getattr(old, "body", []) or [])

        def members(body: Sequence[ast.stmt]) -> list[ast.stmt]:
            return [st for st in body
                    if not _is_docstring(st) and not isinstance(st, ast.Pass)
                    and not (isinstance(st, ast.Expr)
                             and isinstance(getattr(st, "value", None),
                                            ast.Constant)
                             and st.value.value is Ellipsis)]

        body = members(node.body)
        defs = [st for st in body if isinstance(st, (*_DEFS, ast.ClassDef))]
        others = [st for st in body if st not in defs]
        old_dumps = {ast.dump(st) for st in old_body}
        # COUNTED, and by the key a definition is known by: a property, its
        # setter and its deleter are three, and a reply holding one of them
        # does not hold the whole class
        everything = Counter(
            [def_key(st) for st in old_body
             if isinstance(st, (*_DEFS, ast.ClassDef))]
            + [assign_name(st) for st in old_body if assign_name(st)])
        reply_names = Counter(
            [def_key(d) for d in defs]
            + [assign_name(st) for st in others if assign_name(st)])
        # a method echoed back as its elided header is not a replacement
        echoed = []
        for d in defs:
            existing = self.idx.get(f"{span.name}.{def_key(d)}")
            if not isinstance(d, _DEFS) or existing is None \
                    or not body_is_stub(d):
                continue
            old_m = _node_at(self.tree, existing.start)
            if old_m is not None and not body_is_stub(old_m):
                echoed.append(d)
        whole = bool(everything) and all(
            reply_names[n] >= c for n, c in everything.items()) \
            and not echoed
        if whole or not everything:
            self.replace(span, _cut(src, node), node.col_offset, node)
            return
        # A PARTIAL class: only members are placed, so the class line, its
        # decorators and its docstring must be the file's own — a new base
        # class or `@dataclass` cannot be dropped while "replaced Foo.m" is
        # reported as success
        header = _class_header_change(old, node)
        if header:
            raise _Refuse(f"the reply's `class {span.name}` changes its "
                          f"{header} but leaves out some of its members; "
                          f"return the whole class to change that")
        changed = [st for st in others if ast.dump(st) not in old_dumps]
        if changed:
            seg = ast.get_source_segment("".join(src), changed[0]) or ""
            first = seg.strip().splitlines()[0][:60] if seg.strip() else ""
            raise _Refuse(f"the reply's `class {span.name}` changes "
                          f"class-level lines (`{first}`) but leaves out "
                          f"some of its members; return the whole class to "
                          f"change those")
        added: list[str] = []
        for d in defs:
            key = def_key(d)
            qual = f"{span.name}.{key}"
            if qual in self.idx.ambiguous:
                self.result.skipped.append(
                    (shown_name(qual), "the class defines it more than "
                                       "once"))
                continue
            target = self.idx.get(qual)
            if target is not None and isinstance(d, ast.ClassDef) \
                    and target.kind == "class":
                self.merge_class(d, src, target)
            elif target is not None:
                self.replace(target, _cut(src, d), d.col_offset, d)
            elif self.allow_new:
                lead = _member_lead(self.lines, self.idx, span)
                added.append(reindent(_cut(src, d), d.col_offset, lead))
                self.result.added.append(qual)
        if added:
            block = "".join("\n" + a for a in added)
            self.op(span.end, _NEW_TOP, block, "insert")


def _class_header_change(old: ast.AST | None, new: ast.ClassDef) -> str:
    """What a reply's class line changes — "" when nothing."""
    if old is None:
        return ""
    if [ast.dump(b) for b in old.bases] != [ast.dump(b) for b in new.bases] \
            or [ast.dump(k) for k in old.keywords] != \
            [ast.dump(k) for k in new.keywords]:
        return "base classes"
    if [ast.dump(d) for d in old.decorator_list] != \
            [ast.dump(d) for d in new.decorator_list]:
        return "decorators"
    new_doc = ast.get_docstring(new, clean=True)
    if new_doc is not None and new_doc != ast.get_docstring(old, clean=True):
        return "docstring"
    return ""


# --------------------------------------------------------------------------
# what the model is shown
# --------------------------------------------------------------------------

def _header_end(node: ast.AST) -> int:
    """1-based last line of a def's header, a one-line docstring kept."""
    body = list(getattr(node, "body", []) or [])
    if not body:
        return node.lineno
    first = body[0]
    if _is_docstring(first) and (first.end_lineno or first.lineno) \
            == first.lineno:
        return first.lineno
    return max(node.lineno, first.lineno - 1)


def condensed_view(code: str, targets: Sequence[str] = ()) -> str:
    """Every target (and the classes holding one) in full; imports, short
    statements and a short main guard as they are; every other definition
    as its header and `...`. No line numbers: the model returns definitions
    by name, never by line."""
    text = _norm(code)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = lines_of(text)
    wanted = {t for t in targets if t}
    out: list[str] = []

    def lead(node: ast.AST) -> str:
        return _lead_of(lines, _first_line(node) - 1)

    def full(node: ast.AST) -> None:
        out.append(_cut(lines, node))

    def head(node: ast.AST) -> None:
        start = _first_line(node) - 1
        end = _header_end(node)
        body = "".join(lines[end:node.end_lineno or node.lineno])
        elided = f"{lead(node)}    {ELIDED}\n"
        if len(body) <= len(elided):
            # a body shorter than the line that would stand for it is
            # shown: eliding it would make the view LONGER, not shorter
            full(node)
            return
        out.append("".join(lines[start:end]))
        out.append(elided)

    def holds(qual: str) -> bool:
        return any(w.startswith(qual + ".") for w in wanted)

    def walk_class(node: ast.ClassDef, qual: str) -> None:
        body = list(node.body)
        start = _first_line(node) - 1
        end = node.lineno
        if body and _is_docstring(body[0]):
            end = body[0].end_lineno or body[0].lineno
            body = body[1:]
        elif body:
            end = max(node.lineno, body[0].lineno - 1)
        out.append("".join(lines[start:end]))
        for i, st in enumerate(body):
            if i and isinstance(st, (*_DEFS, ast.ClassDef)):
                out.append("\n")
            name = getattr(st, "name", "")
            sub = f"{qual}.{name}" if name else ""
            if isinstance(st, _DEFS):
                if sub in wanted:
                    full(st)
                else:
                    head(st)
            elif isinstance(st, ast.ClassDef):
                if sub in wanted:
                    full(st)
                elif holds(sub):
                    walk_class(st, sub)
                else:
                    head(st)
            else:
                full(st)

    for node in tree.body:
        if isinstance(node, _DEFS):
            if node.name in wanted:
                full(node)
            else:
                head(node)
        elif isinstance(node, ast.ClassDef):
            if node.name in wanted:
                full(node)
            elif holds(node.name):
                walk_class(node, node.name)
            else:
                head(node)
        else:
            n = (node.end_lineno or node.lineno) - node.lineno + 1
            limit = 12 if is_main_guard(node) else 6
            if n <= limit or (is_main_guard(node) and "__main__" in wanted):
                full(node)
            else:
                out.append(lines[node.lineno - 1])
                out.append(f"    # {ELIDED}\n")
        out.append("\n")
    return "".join(out).rstrip("\n") + "\n"


# --------------------------------------------------------------------------
# which definitions an error points at
# --------------------------------------------------------------------------

def targets_for(code: str, lines: Sequence[int],
                named: Callable[[], Sequence[str]] | None = None,
                limit: int = 3) -> list[str]:
    """The functions holding the error ``lines`` (1-based), innermost, in
    order and without repeats; then, if none, whatever ``named`` offers."""
    idx = index(code)
    if idx is None:
        return []
    out: list[str] = []
    for line in lines:
        name = enclosing(idx, line)
        if name and name not in out:
            out.append(name)
        if len(out) >= limit:
            return out
    if not out and named is not None:
        for name in named() or ():
            if name in idx.spans and name not in out:
                out.append(name)
            if len(out) >= limit:
                break
    return out
