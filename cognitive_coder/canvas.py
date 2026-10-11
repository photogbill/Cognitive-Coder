# SPDX-License-Identifier: Apache-2.0
"""Diffusion mode: write and repair by HOLES, when the model fills a canvas.

WHY THIS EXISTS, 2026-10-10
---------------------------
Appendix G.4 of the build specification says fill-in-the-middle is "PROVEN,
and underused": give the model the code before and after a hole and it
writes only the hole, so it "structurally cannot touch code outside it"
(the cure for D6). Every provider reported `supports_fim`; nothing ever read
it. Repairs asked for the complete corrected file — which is where D3 (the
same file back, byte for byte), D6 (the surrounding function rewritten while
the model was in there) and D11 (the model pattern-matching its own broken
code) live, and why a file too large for the reply budget could not be
repaired at all.

A DIFFUSION model makes the hole better than FIM does. DiffusionGemma writes
a 256-token canvas all at once and refines every position together, so the
host can lay the canvas out: lines HELD fixed on both sides, a FREE region
the model writes, and — for a warm start — the old lines placed in the free
region and partly re-noised, so the model's own uncertainty decides which of
them change. The model sees the code after the hole while it writes the
hole, inside the canvas, not merely in the prompt.

WHAT THIS MODULE DECIDES, BY RULE (C5) — the model decides nothing here:

  1. WHERE the hole goes. A repair's hole is where the diagnostics put the
     error: the body of the function that contains it, or a window of lines
     around it when the body is long or the file does not parse. The model
     never chooses where an edit lands — the same principle as D8 (the model
     never chooses a path), now for anchors.
  2. WHAT the canvas holds: the lines before the hole (head), the lines
     after it (tail), the old lines (for a warm start), and how much noise.
  3. HOW a filled hole is cleaned and spliced back, and whether it can be
     used at all (it must parse, and it must not be the old code again).
  4. For a new file whose interface was PINNED (interfaces.py), the holes
     are the stub bodies — `raise NotImplementedError` — so every signature,
     field and docstring is held and cannot drift.

What it does not decide: whether the result WORKS. That is the loop's, as it
always was — the tests are the judge, never the model's confidence.

Nothing here talks to a model or a file. `loop.py` does that; this module is
pure so that every rule above is a unit test.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .types import CanvasRequest, CanvasResult, Diagnostic, Message

#: Lines held before a hole: enough to anchor the indentation and say which
#: function this is. The whole file is in the prompt as well.
HEAD_LINES = 6
#: Lines held after a hole, for a canvas that holds the far side too.
TAIL_LINES = 4
#: A stub's head is its signature and docstring; past this it is cut to the
#: last lines, which is where the body starts.
STUB_HEAD_LINES = 14
#: A function body longer than this is not rewritten whole: the hole
#: becomes a window around the error line.
BODY_MAX_LINES = 36
WINDOW_BEFORE = 5
WINDOW_AFTER = 6
#: One canvas is 256 tokens. A hole plans to use at most this many of them,
#: by the host's count, keeping room for padding and the end of the reply.
CANVAS_TOKENS = 256
CANVAS_BUDGET = 200
#: An open hole (the far side shown only in the prompt) may run on into
#: further canvases; a stub body often needs more than one.
OPEN_CANVASES_STUB = 3
#: Notes from the planning call, as shown to every body in the file.
NOTES_CHARS = 2500
#: How sure counts as sure, per line — the same bands ATK's doubt map draws
#: (`ccoder_build.DOUBT_SURE` / `DOUBT_UNSURE`), so BUILD_LOG.txt and the
#: console agree about which lines to distrust.
SURE = 0.85
UNSURE = 0.5

#: The candidate SCHEDULE for one repair attempt, in order: (style, noise).
#: "tail" holds the code after the hole on the canvas; "open" lets the hole
#: end where the model ends its reply. 0.5 starts from the old lines with
#: half of them scrambled — the conservative fix; 1.0 is from noise — a
#: rewrite. Which of these works best on real builds is not known yet; the
#: journal's `canvas` events record which one was kept, so it will be.
REPAIR_SCHEDULE: tuple[tuple[str, float], ...] = (
    ("tail", 0.5), ("open", 1.0), ("tail", 1.0), ("open", 1.0))


@dataclass(frozen=True)
class Hole:
    """One region of a file, to be written by the model and nothing else."""
    path: str
    start: int            # 0-based first line of the region
    end: int              # 0-based, one past the last line
    head: str             # held before the region: whole lines
    tail: str             # the lines after the region: whole lines
    original: str         # the region as it stands
    #: the first non-blank lines AFTER the region, indentation kept (up to
    #: three) — how a reply that ran on past the region is recognised
    after: tuple[str, ...]
    indent: int           # columns the region's code sits at
    kind: str             # "body" | "window" | "stub"
    name: str = ""        # the function, for "body" and "stub"
    line: int = 0         # the 1-based line the error named (repairs)
    why: str = ""         # "located" | "doubt": how the hole was chosen

    @property
    def label(self) -> str:
        if self.end <= self.start + 1:
            return f"line {self.start + 1}"
        return f"lines {self.start + 1}-{self.end}"

    @property
    def size(self) -> int:
        return max(0, self.end - self.start)

    @property
    def stop_line(self) -> str:
        """The first non-blank line after the region, stripped."""
        return self.after[0].strip() if self.after else ""


# --------------------------------------------------------------------------
# where the hole goes
# --------------------------------------------------------------------------

#: Where Python's parser ends a line: "\r\n", "\n", or a lone "\r".
#: `str.splitlines` also splits on form feeds, \x1c-\x1e, \x85 and the
#: Unicode line separators, which the parser does NOT count — a form feed
#: before a `def` once made a hole of the wrong lines and a splice that
#: deleted the `def` (review, 2026-10-10).
_LINE_END = re.compile(r"(?<=\n)|(?<=\r)(?!\n)")


def _lines(code: str) -> list[str]:
    return [part for part in _LINE_END.split(code) if part]


def _indent(line: str) -> int:
    stripped = line.lstrip(" \t")
    if not stripped.strip():
        return -1
    return len(line) - len(stripped)


def _next_nonblank(lines: Sequence[str], start: int,
                   n: int = 3) -> tuple[str, ...]:
    out = []
    for ln in lines[start:]:
        if ln.strip():
            out.append(ln.rstrip("\r\n").rstrip())
            if len(out) == n:
                break
    return tuple(out)


def _region_indent(lines: Sequence[str], start: int, end: int) -> int:
    widths = [_indent(ln) for ln in lines[start:end]]
    widths = [w for w in widths if w >= 0]
    return min(widths) if widths else 0


def _make(path: str, lines: list[str], start: int, end: int, *, kind: str,
          head_from: int | None = None, name: str = "", line: int = 0,
          why: str = "located", head_lines: int = HEAD_LINES) -> Hole:
    hs = max(0, start - head_lines)
    if head_from is not None:
        hs = max(hs, head_from)
    return Hole(path=path, start=start, end=end,
                head="".join(lines[hs:start]),
                tail="".join(lines[end:end + TAIL_LINES]),
                original="".join(lines[start:end]),
                after=_next_nonblank(lines, end),
                indent=_region_indent(lines, start, end),
                kind=kind, name=name, line=line, why=why)


def _functions(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """(qualified name, node) for every function and method, outermost
    first; functions nested inside functions are part of their parent."""
    out: list[tuple[str, ast.AST]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((prefix + child.name, child))
            elif isinstance(child, ast.ClassDef):
                walk(child, prefix + child.name + ".")
    walk(tree, "")
    return out


def _body_start(node: ast.AST) -> int | None:
    """1-based line of the first statement after the docstring, or None
    when the body is nothing but a docstring."""
    body = list(getattr(node, "body", []) or [])
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return body[0].lineno if body else None


def _def_line(node: ast.AST) -> int:
    decos = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min([node.lineno, *decos])


def located_error(diagnostics: Sequence[Diagnostic], path: str,
                  same_file: Callable[[str, str], bool]) -> Diagnostic | None:
    """The first real error that names a line of THIS file."""
    for d in diagnostics:
        if d.is_error and d.line and d.line > 0 and same_file(d.file, path):
            return d
    return None


def repair_hole(code: str, lang: str, path: str,
                diagnostics: Sequence[Diagnostic],
                same_file: Callable[[str, str], bool],
                count_tokens: Callable[[str], int] | None = None
                ) -> Hole | None:
    """The hole for a repair, from the first located error — or None."""
    err = located_error(diagnostics, path, same_file)
    if err is None or not code.strip():
        return None
    lines = _lines(code)
    if not lines:
        return None
    idx = min(err.line, len(lines)) - 1
    lo, hi = 0, len(lines)
    if lang == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        if tree is not None:
            inner = None
            for name, node in _functions(tree):
                if node.lineno <= err.line <= (node.end_lineno or 0):
                    inner = (name, node)      # outermost first; keep going
                    for sub_name, sub in _functions(node):
                        if sub.lineno <= err.line <= (sub.end_lineno or 0):
                            inner = (f"{name}.{sub_name}", sub)
            if inner is not None:
                name, node = inner
                first = _body_start(node)
                last = node.end_lineno or node.lineno
                # `first > node.lineno`: a one-line `def f(): return x` has
                # its body ON the def line — that hole would be the
                # signature, so it gets a window instead
                if (first is not None and first > node.lineno
                        and first <= err.line <= last):
                    start, end = first - 1, last
                    hole = _make(path, lines, start, end, kind="body",
                                 head_from=_def_line(node) - 1, name=name,
                                 line=err.line)
                    if (end - start <= BODY_MAX_LINES
                            and _fits(hole, count_tokens)):
                        return hole
                    lo, hi = first - 1, last         # a window in this body
    start = max(lo, idx - WINDOW_BEFORE)
    end = min(hi, idx + WINDOW_AFTER + 1)
    for before, after in ((WINDOW_BEFORE, WINDOW_AFTER), (2, 3), (1, 1)):
        start = max(lo, idx - before)
        end = min(hi, idx + after + 1)
        hole = _make(path, lines, start, end, kind="window", line=err.line)
        if _fits(hole, count_tokens):
            return hole
    return None


def function_hole(code: str, path: str, name: str,
                  why: str = "doubt") -> Hole | None:
    """The body of the function called `name` (qualified: `Cls.meth`)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = _lines(code)
    for qual, node in _functions(tree):
        if qual != name:
            continue
        first = _body_start(node)
        if first is None or first <= node.lineno:      # a one-line def
            return None
        return _make(path, lines, first - 1, node.end_lineno or node.lineno,
                     kind="body", head_from=_def_line(node) - 1, name=name,
                     why=why)
    return None


def _is_stub_body(node: ast.AST) -> ast.stmt | None:
    """The `raise NotImplementedError` that is the whole body, or None."""
    body = list(getattr(node, "body", []) or [])
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return None
    exc = body[0].exc
    target = exc.func if isinstance(exc, ast.Call) else exc
    if isinstance(target, ast.Name) and target.id == "NotImplementedError":
        return body[0]
    return None


def stub_holes(code: str, path: str = "") -> list[Hole]:
    """Every stub body in a pinned file, in file order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lines = _lines(code)
    holes = []
    for name, node in _functions(tree):
        raise_stmt = _is_stub_body(node)
        if raise_stmt is None or raise_stmt.lineno <= node.lineno:
            continue                      # not a stub, or a one-line def
        start = raise_stmt.lineno - 1
        end = raise_stmt.end_lineno or raise_stmt.lineno
        holes.append(_make(path, lines, start, end, kind="stub",
                           head_from=_def_line(node) - 1, name=name,
                           head_lines=STUB_HEAD_LINES))
    holes.sort(key=lambda h: h.start)
    return holes


def _fits(hole: Hole, count_tokens: Callable[[str], int] | None) -> bool:
    if count_tokens is None:
        # ~3.5 characters a token for code, deliberately pessimistic
        n = (len(hole.head) + int(len(hole.original) * 1.3)
             + len(hole.tail)) / 3.5
        return n + 16 <= CANVAS_BUDGET
    try:
        n = (count_tokens(hole.head) + int(count_tokens(hole.original) * 1.3)
             + count_tokens(hole.tail))
    except Exception:                                   # noqa: BLE001
        return False
    return n + 16 <= CANVAS_BUDGET


# --------------------------------------------------------------------------
# what the model is shown
# --------------------------------------------------------------------------

def marked_view(code: str, hole: Hole, comment: str = "#") -> str:
    """The whole file, with the region between markers.

    A repair shows the old lines between the markers — the model needs to
    know what the code was for. A stub shows nothing there: it is a body
    still to be written.
    """
    lines = _lines(code)
    pad = " " * hole.indent
    before = "".join(lines[:hole.start])
    after = "".join(lines[hole.end:])
    if before and not before.endswith("\n"):
        before += "\n"
    if hole.kind == "stub":
        middle = (f"{pad}{comment} >>> THE BODY OF `{hole.name}` GOES HERE "
                  f"(not written yet)\n{pad}{comment} <<<\n")
    else:
        old = hole.original if hole.original.endswith("\n") \
            else hole.original + "\n"
        middle = (f"{pad}{comment} >>> REWRITE {hole.label.upper()} (the "
                  f"old lines are shown; your lines replace them)\n"
                  f"{old}{pad}{comment} <<< END OF THE LINES TO REWRITE\n")
    return before + middle + after


def repair_text(path: str, purpose: str, request: str = "",
                autofixes: Sequence[str] = (), limit: int = 2000) -> str:
    """The [TASK] for a hole repair. Constant for a file, so a prompt cache
    keeps everything before the parts that change."""
    lines = [f"The file `{path}` does not work yet. Its purpose: {purpose}"]
    if request.strip():
        text = request.strip()
        if len(text) > limit:
            text = (text[:limit].rstrip() + f"… (the request continues; "
                    f"only its first {limit:,} characters are repeated here)")
        lines += ["", f"It is part of this request: {text}"]
    lines += ["",
              "Rewrite ONLY the marked lines so that the errors reported "
              "below are fixed. Everything outside the markers is already "
              "in place and stays exactly as it is — your reply cannot "
              "change it. Keep every name the rest of the file and the "
              "other files use."]
    if autofixes:
        lines += ["", "These were already fixed for you automatically, so "
                      "do not undo them:"]
        lines += [f"  - {f}" for f in autofixes]
    return "\n".join(lines)


def stub_text(path: str, purpose: str, request: str = "",
              limit: int = 2000) -> str:
    """The [TASK] for writing a pinned file's bodies, one hole at a time.
    The same for every body in the file, for the same reason as above."""
    lines = [f"`{path}` is being written one function body at a time. Its "
             f"purpose: {purpose}"]
    if request.strip():
        text = request.strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "… (the request continues)"
        lines += ["", f"It is part of this request: {text}"]
    lines += ["",
              "The signatures, types and docstrings are pinned: other files "
              "are written against them. Write the ONE body marked in the "
              "file below, so that it does what its docstring and the "
              "request say, using only names this file, the files it "
              "imports and the standard library provide."]
    return "\n".join(lines)


def notes_text(path: str, names: Sequence[str]) -> str:
    """The [TASK] for the planning call made before a file's bodies."""
    listed = ", ".join(f"`{n}`" for n in names)
    return (f"Before the bodies of `{path}` are written, plan them. For each "
            f"of {listed}: its name, then one to three plain sentences on "
            f"how it works — the steps, the edge cases, which other names it "
            f"calls. Plain notes for the person who will write the code; no "
            f"code.")


NOTES_CONTRACT = ("OUTPUT CONTRACT\nPlain notes, one short paragraph per "
                  "function, each starting with its name. No code, no "
                  "headings, nothing else.")


def hole_block(hole: Hole) -> str:
    """[THE HOLE]: what this one request is for, and the lines it starts
    with — shown, because the reply really does start with them."""
    head = hole.head if hole.head.endswith("\n") or not hole.head \
        else hole.head + "\n"
    what = (f"the body of `{hole.name}`" if hole.kind in ("stub", "body")
            and hole.name else f"{hole.label} of `{hole.path}`")
    return (f"[THE HOLE — {what}]\nYour reply begins with these lines, "
            f"already written, exactly as they stand:\n{head}")


def contract(hole: Hole, style: str) -> str:
    n_head = hole.head.count("\n") or (1 if hole.head else 0)
    if hole.kind == "stub":
        return ("OUTPUT CONTRACT\nCode only. Your reply starts with the "
                f"{n_head} line(s) shown under THE HOLE, then the body of "
                f"`{hole.name}` — and it ends with that body. Nothing after "
                f"it: not the next function, not a test, no prose, no "
                f"fences.")
    if style == "tail" and hole.tail:
        n_tail = hole.tail.count("\n") or 1
        return ("OUTPUT CONTRACT\nCode only. Your reply starts with the "
                f"{n_head} line(s) shown under THE HOLE, then the new lines "
                f"for the marked region, then the {n_tail} line(s) that "
                f"follow the region in the file, exactly as they stand. "
                f"No prose, no fences.")
    return ("OUTPUT CONTRACT\nCode only. Your reply starts with the "
            f"{n_head} line(s) shown under THE HOLE, then the new lines for "
            f"the marked region — and it ENDS there. Do not repeat the code "
            f"that follows the region. No prose, no fences.")


def requests_for(hole: Hole, messages_for: Callable[[str], Sequence[Message]],
                 n: int, *, seed: int | None = None, attempt: int = 1,
                 count_tokens: Callable[[str], int] | None = None,
                 lang: str = "python"
                 ) -> list[tuple[str, float, CanvasRequest]]:
    """The candidate canvases for one repair attempt: (style, noise, req).

    Outside Python every candidate holds the far side ("tail"): an open
    hole's run-on can only be recognised and cut safely where a parser can
    check the cut, and the engine parses only Python before the build."""
    out = []
    region_tokens = (count_tokens(hole.original) if count_tokens
                     else int(len(hole.original) / 3.5))
    slack = 12 + region_tokens // 3
    for i, (style, noise) in enumerate(REPAIR_SCHEDULE[:max(1, n)]):
        if lang != "python" and hole.tail.strip():
            style = "tail"
        if style == "tail" and not hole.tail.strip():
            style = "open"                     # nothing after it to hold
        s = None if seed is None else seed + 7919 * i + 104729 * attempt
        msgs = tuple(messages_for(style))
        if style == "tail":
            req = CanvasRequest(messages=msgs, head=hole.head,
                                seed_text=hole.original, tail=hole.tail,
                                noise=noise, slack=slack, seed=s)
        else:
            req = CanvasRequest(messages=msgs, head=hole.head,
                                seed_text=hole.original if noise < 1 else "",
                                noise=noise, seed=s)
        out.append((style, noise, req))
    return out


# --------------------------------------------------------------------------
# what comes back
# --------------------------------------------------------------------------

def clean_fill(text: str, hole: Hole, style: str,
               verify: Callable[[list[str]], bool] | None = None
               ) -> str | None:
    """The filled region, cleaned for splicing — or None if there is none.

    What goes: blank padding at the end (a held tail made the region a fixed
    size); a copy of the held tail's first lines that the model wrote
    itself; and a reply that ran on past the region. Nothing else changes —
    no line is rewritten, no whitespace inside a line, no blank line in the
    middle (it may be inside a string).

    A RUN-ON is cut only where `verify` (the caller: "this prefix, spliced,
    still parses") agrees, at the first of these that it accepts:

      * for a function body: a line that leaves the body — dedented below
        it, not a comment (a `# note` at column 0 is still inside the
        function), and not inside a string (a cut there would not parse);
      * for an open hole: the file's next lines, with their indentation,
        appearing in order — not merely the same text (`}` and `return
        None` occur inside regions too).

    Without `verify` (a language the engine does not parse before the build)
    there is no run-on cut at all; such holes always hold their far side.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")
    key = [ln.rstrip() for ln in lines]

    if verify is not None:
        cuts: list[int] = []
        if hole.kind in ("stub", "body"):
            for i, ln in enumerate(lines):
                bare = ln.strip()
                if (bare and not bare.startswith("#")
                        and _indent(ln) < hole.indent):
                    cuts.append(i)
        if style != "tail" and hole.after:
            need = list(hole.after[:2])
            nonblank = [i for i, k in enumerate(key) if k.strip()]
            for pos, i in enumerate(nonblank):
                if i == 0 or key[i] != need[0]:
                    continue
                following = [key[j] for j in nonblank[pos + 1:pos + len(need)]]
                if following == need[1:] or (len(nonblank) - pos < len(need)
                                             and following == need[1:len(
                                                 following) + 1]):
                    cuts.append(i)
        for i in sorted(set(cuts)):
            if verify(lines[:i]):
                lines, key = lines[:i], key[:i]
                break

    if style == "tail" and hole.tail.strip():
        tail = [t.rstrip() for t in
                hole.tail.replace("\r\n", "\n").split("\n") if t.strip()]
        while lines and not key[-1].strip():
            lines.pop()
            key.pop()
        nonblank = [i for i, k in enumerate(key) if k.strip()]
        # the longest run of the region's LAST lines that is the tail's
        # FIRST lines: the model wrote them, and the held copy follows
        for k in range(min(len(tail), len(nonblank)), 0, -1):
            if [key[i] for i in nonblank[-k:]] == tail[:k]:
                lines, key = lines[:nonblank[-k]], key[:nonblank[-k]]
                break

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None
    return "\n".join(lines) + "\n"


def _edge_blanks(region: str) -> tuple[str, str]:
    lines = region.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    lead = 0
    while lead < len(lines) and not lines[lead].strip():
        lead += 1
    trail = 0
    while trail < len(lines) - lead and not lines[len(lines) - 1 - trail].strip():
        trail += 1
    return "\n" * lead, "\n" * trail


def splice(code: str, hole: Hole, fill: str) -> str:
    """The file with the region replaced by `fill`. The region's own blank
    lines at either edge are kept, so the file's spacing does not move."""
    lines = _lines(code)
    lead, trail = _edge_blanks(hole.original)
    # a CRLF file stays CRLF: the model's lines come back with "\n"
    near = lines[max(0, hole.start - 1):hole.end + 1]
    if any(ln.endswith("\r\n") for ln in near):
        fill, lead, trail = (x.replace("\n", "\r\n")
                             for x in (fill, lead, trail))
    out = "".join(lines[:hole.start])
    if out and not out.endswith("\n"):
        out += "\n"
    out += lead + fill + trail + "".join(lines[hole.end:])
    return out


def _same_code(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s).strip()          # noqa: E731
    return norm(a) == norm(b)


def prescreen(code: str, lang: str) -> str:
    """'' when the spliced file can go to the loop; otherwise why not."""
    if lang == "python":
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"does not parse (line {exc.lineno}: {exc.msg})"
    return ""


def align(raw: str, fill: str, confidence: Sequence[float],
          flips: Sequence[int]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """The host's per-line readout, re-aligned to the CLEANED fill.

    `clean_fill` only drops lines (padding, a run-on, a duplicated tail,
    extra blank lines) — it never rewrites one — so each kept line is found
    in order among the raw ones. A line not found (none should be) is
    counted as the least certain line of the hole, never as certain.
    """
    raw_lines = [ln.rstrip() for ln in
                 (raw or "").replace("\r\n", "\n").split("\n")]
    floor = min([c for c in confidence if c == c], default=1.0)
    out_c: list[float] = []
    out_f: list[int] = []
    j = 0
    for ln in (fill or "").rstrip("\n").split("\n"):
        k = j
        while k < len(raw_lines) and raw_lines[k] != ln:
            k += 1
        if k < len(raw_lines):
            # past the readout (an open reply's later canvases): unknown —
            # NaN, which `mean`/`least` skip; never "certain"
            out_c.append(float(confidence[k]) if k < len(confidence)
                         else float("nan"))
            out_f.append(int(flips[k]) if k < len(flips) else 0)
            j = k + 1
        else:
            out_c.append(1.0 if not ln.strip() else floor)
            out_f.append(0)
    return tuple(out_c), tuple(out_f)


@dataclass(frozen=True)
class Candidate:
    style: str
    noise: float
    result: CanvasResult
    fill: str | None
    code: str
    problem: str          # '' = usable
    #: per line of `fill`: how sure the model was, and how contested
    confidence: tuple[float, ...] = ()
    flips: tuple[int, ...] = ()

    @property
    def usable(self) -> bool:
        return not self.problem

    @property
    def mean(self) -> float:
        """Mean confidence over the fill's non-blank lines."""
        lines = (self.fill or "").rstrip("\n").split("\n")
        vals = [c for c, ln in zip(self.confidence, lines)
                if ln.strip() and c == c]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def least(self) -> float:
        """The least certain non-blank line of the fill."""
        lines = (self.fill or "").rstrip("\n").split("\n")
        vals = [c for c, ln in zip(self.confidence, lines)
                if ln.strip() and c == c]
        return min(vals) if vals else 1.0

    def summary(self) -> dict:
        return {"style": self.style, "noise": self.noise,
                "seed": self.result.seed, "ok": self.usable,
                "problem": self.problem,
                "confidence": round(self.mean, 3),
                "steps": self.result.steps,
                "prompt_ms": self.result.prompt_ms,
                "decode_ms": self.result.decode_ms,
                "cached_tokens": self.result.cached_tokens}

    def describe(self) -> str:
        """One line for BUILD_LOG.txt: how this canvas was asked for, what
        it cost, and whether it was usable — or exactly why not."""
        r = self.result
        seed = "none" if r.seed is None else r.seed
        head = (f"{self.style}, noise {self.noise:g}, seed {seed}, "
                f"{r.steps} step(s), prefill {r.prompt_ms} ms "
                f"({r.cached_tokens} prompt token(s) from cache), decode "
                f"{r.decode_ms} ms")
        if self.usable:
            return (f"{head} — usable: {self.mean:.0%} sure on average, "
                    f"least sure line {self.least:.0%}")
        return f"{head} — REJECTED: {self.problem}"


def judge(code: str, hole: Hole, style: str, noise: float,
          result: CanvasResult, lang: str) -> Candidate:
    """Clean, splice and screen one filled canvas."""
    if not result.ok:
        return Candidate(style, noise, result, None, "",
                         result.error or f"the canvas ended "
                         f"({result.finish_reason})")
    verify = None
    if lang == "python":
        def verify(prefix: list[str]) -> bool:
            if not any(ln.strip() for ln in prefix):
                return False
            text = "\n".join(prefix).rstrip("\n") + "\n"
            return prescreen(splice(code, hole, text), lang) == ""
    fill = clean_fill(result.text, hole, style, verify)
    if fill is None:
        return Candidate(style, noise, result, None, "", "the hole came "
                         "back empty")
    if hole.kind != "stub" and _same_code(fill, hole.original):
        return Candidate(style, noise, result, fill, "", "the same lines "
                         "came back")
    new = splice(code, hole, fill)
    conf, flips = align(result.text, fill, result.line_confidence,
                        result.line_flips)
    return Candidate(style, noise, result, fill, new, prescreen(new, lang),
                     conf, flips)


def choose(cands: Sequence[Candidate]) -> int | None:
    """The usable candidate the model was surest of (the earlier on a tie)."""
    best, best_c = None, -1.0
    for i, c in enumerate(cands):
        if c.usable and c.mean > best_c + 1e-9:
            best, best_c = i, c.mean
    return best


def least_sure(confidence: dict[str, float], code: str,
               tried: set[str]) -> str:
    """For a failure that names no line of this file: the function whose
    body the model was least sure of when it wrote it, not tried yet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    present = {name for name, _n in _functions(tree)}
    ranked = sorted((c, n) for n, c in confidence.items()
                    if n in present and n not in tried)
    return ranked[0][1] if ranked else ""


def doubt_html_lines(cand: Candidate) -> list[str]:
    """The fill's lines, for a host that draws the doubt map."""
    return (cand.fill or "").rstrip("\n").split("\n")


#: The legend printed above `doubt_rows` in BUILD_LOG.txt.
DOUBT_LEGEND = (f"how sure the model was of each line: `?` under "
                f"{UNSURE:.0%}, `~` under {SURE:.0%}, `--` not read back")


def doubt_rows(cand: Candidate) -> list[str]:
    """The kept fill as text rows, each with how sure the model was of it.

    The console colours these; a log file cannot, so the band is a mark a
    reader can search for: `?` (under UNSURE) is where to look first when
    the next test fails. A line past the readout says `--` — unknown is
    not the same as sure.
    """
    rows = []
    for i, text in enumerate(doubt_html_lines(cand)):
        conf = cand.confidence[i] if i < len(cand.confidence) else None
        if not text.strip():
            rows.append(f"{'':4}   | {text}")
            continue
        if conf is None or conf != conf:
            rows.append(f"  --   | {text}")
            continue
        mark = "?" if conf < UNSURE else "~" if conf < SURE else " "
        rows.append(f"{conf:4.0%} {mark} | {text}")
    return rows
