# SPDX-License-Identifier: Apache-2.0
"""What a change to an existing file must keep.

WHY THIS EXISTS, 2026-10-19
---------------------------
A change to a file that already has real work in it asks for "the complete
updated file". The pinned-interface check (D12) holds a body to the names
its stub declared — but only for a file the skeleton pinned in THIS
session. A hand-written project being improved (`ccoder audit` → `build`,
Debug & Repair on the operator's own code) had no such check, so a small
model that answered with

    def update(self, dt):
        ...  # rest of the method unchanged

or that simply left `draw_hud` out, had its reply written to disk; on a
project with no tests nothing failed, and the code was gone. Aider counts
these "lazy comments" in its benchmark and prompts against them; OpenHands
avoids them by editing with replace-this-text. This module is the check
that makes the whole-file path safe instead.

THE RULES, BY RULE (C5):

  * **A definition the file had stays** — function, class, method, a
    module-level UPPER_CASE constant, the `if __name__ == "__main__":`
    block — unless the request names it (the operator asked for it to go,
    or to be renamed).
  * **A rename is allowed** when the new name's body is recognisably the
    old one and the old name is used nowhere any more: not in the reply,
    not in the rest of the project (the caller supplies that check).
  * **A placeholder is never code.** A comment line that says the code is
    elsewhere (`# ... rest unchanged`, `// existing code here`), or a body
    that became `...` / `pass` where real code was, is refused, located at
    its line. No request makes a placeholder right.
  * **A request that asks for a rewrite** ("rewrite", "from scratch",
    "refactor", "remove", "delete", …) turns a missing definition from a
    refusal into a warning — the operator may mean it. Placeholders are
    still refused.

  * **A planted placeholder is never protected, and never accepted**
    (2026-10-22, from the journals). A debug session on main.py was handed
    `def run_headless_demo_fixed(): pass` beside the real
    `run_headless_demo`, plus the model's own narration as comments ("I
    will provide the clean, corrected version"). The next two sessions
    were then REFUSED for dropping `run_headless_demo_fixed` — the guard
    defending a stub the model had planted. So: a new function whose body
    is only `pass` / `...` / a docstring is refused on arrival; a new name
    that is an existing name plus `_fixed` / `_new` / `_v2` / `_2` /
    `_corrected` / `_updated` / `_old` is a copy, refused; a definition
    the FILE already holds with a placeholder body is not a loss when a
    reply drops it (nothing real went); and a comment line that is the
    model talking about its own reply is a placeholder.

The check runs BEFORE the reply is written (the loop's guard position), so
the operator's file is still on disk, whole, for the next attempt to see.

Python is checked through its AST; other languages through the codemap's
parsers, which are pattern-matched there and say so.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import difflib
import re

from . import symedit
from .types import Diagnostic

#: Comment openers per family; anything else is matched as "#" or "//".
_COMMENT = r"(?:#|//|--|;|/\*|\*|<!--|')"

#: A comment line that says code is elsewhere. Matched on a WHOLE line that
#: is only a comment, so `x = 1  # ... and more` is never a placeholder; a
#: sentence that merely mentions code ("the existing code moved here from
#: utils.py", "# Keep the original implementation for compatibility",
#: "# Other functions") is not one either — a placeholder is short, and
#: ENDS on the word that says the code is not here.
_LEAD = (r"(?:[(\[]\s*)?(?:(?:the|all|any|your|keep(?:\s+the)?|"
         r"plus(?:\s+the)?|and(?:\s+the)?|insert(?:\s+the)?)\s+)?")
_END = r"\s*[.)\]]*\s*(?:\*/|-->)?\s*$"
#: What says "it is not here", after a phrase naming the code.
_NOT_HERE = (r"(?:\s+(?:is|are|remains?|stays?|goes|left|kept|was))?"
             r"(?:\s+(?:unchanged|the same|as before|as is|as it was|here|"
             r"omitted|not shown|below|above|follows|untouched))")
_PLACEHOLDER = re.compile(
    rf"^\s*{_COMMENT}+\s*{_LEAD}"
    r"(?:"
    # an ellipsis with a word that says "the code is not here"
    r"(?:\.\.\.|…|\[\.\.\.\]|\(\.\.\.\)).*\b(?:rest|existing|unchanged|"
    r"remaining|same|previous|other|omitted|elided|as before|keep|"
    r"implementation|code|methods?|functions?)\b"
    r"|(?:rest|remainder) of (?:the )?(?:code|file|class|module|function|"
    r"method|methods|functions|implementation|program|script|logic)"
    r"(?:" + _NOT_HERE + r")?" + _END +
    r"|(?:existing|previous|original|unchanged|remaining|same)\s+"
    r"(?:code|methods?|functions?|logic|implementation|body|bodies)"
    + _NOT_HERE + _END +
    r"|\[(?:existing|previous|original|unchanged|remaining|same)\s+"
    r"(?:code|methods?|functions?|logic|implementation)\]" + _END +
    r"|(?:omitted|elided|not shown|truncated) for (?:brevity|space|"
    r"clarity)" + _END +
    r"|same as (?:before|above|the original)" + _END +
    r"|(?:code|methods?|functions?|implementation) (?:omitted|elided|"
    r"not shown)(?: for (?:brevity|space|clarity))?" + _END +
    r")", re.I)

#: A comment line that is the model narrating its reply rather than
#: commenting the code: first-person intent, talk of "the user" / "the
#: provided snippet" / "the prompt", or "re-defining" something. Matched on a
#: WHOLE comment line. A real comment that says "we" about the program
#: ("# we retry twice") does not match: the openers below are the ones that
#: only a reply-writer uses.
_NARRATION = re.compile(
    rf"^\s*{_COMMENT}+\s*(?:"
    r"I(?:'ll| will| am going to| have| am| re-| note)\b"
    r"|(?:Note|NOTE):?\s+(?:the user|the prompt|the provided|the given|"
    r"the original snippet|the snippet)\b"
    r"|(?:The|the) (?:user|prompt|provided snippet|given snippet|original "
    r"snippet|snippet) (?:provided|used|had|has|asked|wants|said|gave)\b"
    r"|(?:Re-?defin|Redefin)\w*\b"
    r"|(?:This is a safety check|Helper to fix|Here is the|Below is the)\b"
    r")", re.I)

#: A name that is another definition's name with a "this one is the fixed
#: copy" suffix. The model fixing `f` by adding `f_fixed` beside it is not
#: a fix; it is the bug with a longer name.
_COPY_SUFFIX = re.compile(
    r"^(?P<base>.+?)(?:_|)(?:fixed|new|v2|v3|2|3|corrected|updated|old|"
    r"orig|original|backup|copy|alt|revised|patched)$", re.I)

#: In ONE sentence with a definition's name, these say the operator means
#: it to go, move, or change shape — so its absence is a warning, not a
#: refusal. Anywhere else in a long request they say nothing about it:
#: "add a `remove()` method" removes nothing.
_INTENT = re.compile(
    r"\b(?:remov\w*|delet\w*|drop(?:s|ped|ping)?|get rid of|take\s+\S+\s+"
    r"out|take out|rename\w*|replac\w*|merg\w*|split\w*|mov(?:e|es|ed|ing)"
    r"|extract\w*|convert\w*|turn\w*\b.*\binto|wrap\w*|inlin\w*|"
    r"combin\w*|consolidat\w*|refactor\w*|rewrit\w*|restructur\w*|"
    r"reorgani[sz]\w*|simplif\w*|deprecat\w*|retir\w*)\b", re.I)

#: A sentence that asks for the FILE to be rewritten — with the file named,
#: or "the whole file / this file / from scratch".
_REWRITE = re.compile(
    r"\b(?:rewrit\w*|from scratch|start over|restructur\w*|refactor\w*|"
    r"reorgani[sz]\w*)\b", re.I)

#: How alike two bodies must be for a missing name and a new one to be
#: the same definition renamed.
RENAME_SIMILARITY = 0.6


def _sentences(text: str) -> list[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+|\n\s*\n|\n\s*[-*•]\s+",
                                text or "") if p.strip()]


def intended(name: str, request: str) -> bool:
    """Does one sentence of the request name ``name`` with an intent to
    remove, rename, move or reshape it?"""
    for sentence in _sentences(request):
        if mentioned(name, sentence) and _INTENT.search(sentence):
            return True
    return False


def rewrite_asked(request: str, path: str) -> bool:
    """Does one sentence ask for this FILE to be rewritten?"""
    base = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    for sentence in _sentences(request):
        if not _REWRITE.search(sentence):
            continue
        low = sentence.lower()
        if (base and base.lower() in low) or (len(stem) >= 3 and re.search(
                rf"(?<![\w]){re.escape(stem.lower())}(?![\w])", low)) \
                or re.search(r"from scratch|start over|(?:the|this) (?:whole "
                             r"|entire )?(?:file|module|program|project|code)",
                             low):
            return True
    return False


@dataclass(frozen=True)
class Finding:
    """What a reply would lose. Empty tuples everywhere: nothing."""
    path: str
    dropped: tuple = ()          # names the reply leaves out
    #: left out, but the request asks for exactly that (a warning)
    allowed: tuple = ()
    #: left out, but the reply moved them (imports them, or defines them
    #: under another class) — (name, where it went)
    moved: tuple = ()
    renamed: tuple = ()          # (old, new) allowed renames
    still_used: tuple = ()       # (old, new, where) renames still in use
    placeholders: tuple = ()     # (1-based line, the line) in the reply
    stubbed: tuple = ()          # names whose body became a placeholder
    #: NEW functions the reply adds with a placeholder body (2026-10-22)
    planted: tuple = ()
    #: (new name, the existing name it is a suffixed copy of)
    duplicated: tuple = ()
    approximate: bool = False    # names came from a pattern-matched parser
    warn_only: bool = False      # the request asks for a rewrite

    @property
    def refused(self) -> bool:
        return bool(self.placeholders or self.stubbed or self.planted
                    or self.duplicated
                    or ((self.dropped or self.still_used)
                        and not self.warn_only))

    @property
    def any(self) -> bool:
        return self.refused or bool(self.dropped or self.allowed
                                    or self.moved)

    def describe(self) -> str:
        bits = []
        if self.dropped:
            bits.append("leaves out " + _names(self.dropped))
        if self.still_used:
            bits.append("renames " + ", ".join(
                f"`{o}` to `{n}` while {w} still uses `{o}`"
                for o, n, w in self.still_used[:3]))
        if self.stubbed:
            bits.append("turns the body of " + _names(self.stubbed)
                        + " into a placeholder")
        if self.planted:
            bits.append("adds " + _names(self.planted)
                        + " with an empty body")
        if self.duplicated:
            bits.append("adds " + ", ".join(
                f"`{n}` as a copy of `{o}`" for n, o in self.duplicated[:3])
                + " instead of fixing the original")
        if self.placeholders:
            line, text = self.placeholders[0]
            more = len(self.placeholders) - 1
            bits.append(f"has a placeholder at line {line} (`{text.strip()}`)"
                        + (f" and {more} more" if more else ""))
        return "; ".join(bits)

    def diagnostic(self) -> Diagnostic:
        """The located error the next attempt is shown."""
        # line 0: the placeholder's line is a line of the REPLY, which was
        # never written — as a line of the file it would point the next
        # attempt (and the narrowed edit's targets) at the wrong code
        parts = []
        if self.dropped:
            parts.append(
                f"Your reply left out {_names(self.dropped)}, which "
                f"{self.path} defined; the request does not ask for "
                f"{'it' if len(self.dropped) == 1 else 'them'} to go.")
        if self.still_used:
            o, n, w = self.still_used[0]
            parts.append(f"It renames `{o}` to `{n}`, but {w} still uses "
                         f"`{o}`.")
        if self.stubbed:
            parts.append(f"It replaces the body of {_names(self.stubbed)} "
                         f"with a placeholder where real code was.")
        if self.planted:
            parts.append(f"It adds {_names(self.planted)} with an empty "
                         f"body; a new function must be complete or not "
                         f"added at all.")
        if self.duplicated:
            n, o = self.duplicated[0]
            parts.append(f"It adds `{n}` beside `{o}` instead of fixing "
                         f"`{o}` in place; there must be one `{o}`.")
        if self.placeholders:
            ln, text = self.placeholders[0]
            parts.append(f"Line {ln} of your reply is a placeholder or "
                         f"commentary (`{text.strip()}`) where code "
                         f"belongs — nothing fills it in for you, and the "
                         f"file is not the place to explain the reply.")
        parts.append("Return the complete file: every function, class and "
                     "method the request does not mention stays exactly as "
                     "it was, written out in full.")
        if self.approximate:
            parts.append("(The names were found by pattern-matching, which "
                         "can be imprecise for this language.)")
        return Diagnostic(file=self.path, line=0, severity="error",
                          code="dropped-code", tool="keep",
                          message=" ".join(parts))


def _names(names) -> str:
    shown = ", ".join(f"`{symedit.shown_name(n)}`" for n in list(names)[:6])
    more = len(names) - 6
    return shown + (f" and {more} more" if more > 0 else "")


# --------------------------------------------------------------------------
# what a file defines
# --------------------------------------------------------------------------

def _is_constant(name: str) -> bool:
    letters = [c for c in name if c.isalpha()]
    return bool(letters) and name.upper() == name


def python_defined(code: str) -> dict[str, symedit.Span] | None:
    """{qualified name: span} of what a Python file defines, or None if it
    does not parse. Constants count only when UPPER_CASE: a lower-case
    module variable is often an implementation detail a change may fold
    away; a constant is something other code reads."""
    idx = symedit.index(code)
    if idx is None:
        return None
    out = {}
    for name, span in idx.spans.items():
        if span.kind == "constant" and not _is_constant(name):
            continue
        out[name] = span
    for name in idx.ambiguous:            # a property and its setter: keep
        out.setdefault(name, None)        # the NAME; no span to compare
    return out


def other_defined(code: str, lang: str, path: str) -> set[str]:
    """Names a non-Python file defines, by the codemap's own parsers."""
    try:
        from .codemap import parse_treesitter
        symbols, _e, _u = parse_treesitter.parse(code, path or "x", lang)
    except Exception:                                    # noqa: BLE001
        return set()
    out = set()
    for s in symbols:
        if s.kind in ("module", "file", "import", "variable", "field",
                      "attribute", "parameter"):
            continue
        out.add(f"{s.parent}.{s.name}" if s.parent else s.name)
    return out


# --------------------------------------------------------------------------
# the check
# --------------------------------------------------------------------------

def mentioned(name: str, text: str) -> bool:
    """Does ``text`` name ``name`` (its short name, as a whole word)?"""
    if not text:
        return False
    short = name.rsplit(".", 1)[-1].split("@", 1)[0]
    if short == "__main__":
        return bool(re.search(r"__main__|entry point|main guard", text, re.I))
    return bool(re.search(rf"(?<![\w]){re.escape(short)}(?![\w])", text))


def _used(name: str, code: str) -> bool:
    short = name.rsplit(".", 1)[-1].split("@", 1)[0]
    return bool(re.search(rf"(?<![\w]){re.escape(short)}(?![\w])", code))


def _called(name: str, code: str) -> bool:
    """Is ``name`` still CALLED bare in ``code`` (not as `x.name`, not on
    its own `def` line)? What a move into a class leaves broken."""
    short = name.rsplit(".", 1)[-1].split("@", 1)[0]
    for line in (code or "").splitlines():
        if re.match(rf"\s*(?:async\s+)?def\s+{re.escape(short)}\b", line):
            continue
        if re.search(rf"(?<![\w.]){re.escape(short)}\s*\(", line):
            return True
    return False


def placeholders(before: str, after: str,
                 lang: str = "") -> list[tuple[int, str]]:
    """Placeholder comment lines in ``after`` that ``before`` did not have.
    In Python, lines inside a string (a docstring's `* the rest of the code
    path…` bullet) are not comments and are not looked at."""
    had = {ln.strip() for ln in (before or "").splitlines()}
    in_strings: set[int] = set()
    if lang == "python":
        in_strings = symedit._string_continuations(symedit._norm(after))
    out = []
    for no, line in enumerate(symedit._norm(after).split("\n"), 1):
        if no in in_strings or line.strip() in had:
            continue
        if lang == "python" and not line.lstrip().startswith("#"):
            continue
        if _PLACEHOLDER.match(line) or _NARRATION.match(line):
            out.append((no, line))
    return out


def _copy_of(name: str, old: dict) -> str:
    """The existing definition ``name`` is a suffixed copy of, or ""."""
    leaf = name.rsplit(".", 1)[-1]
    prefix = name[:len(name) - len(leaf)]
    m = _COPY_SUFFIX.match(leaf)
    if not m:
        return ""
    base = m.group("base").rstrip("_")
    if not base or base == leaf:
        return ""
    candidate = prefix + base
    return candidate if candidate in old else ""


def _planted(before: str, after: str, old: dict, new: dict) -> list[str]:
    """New functions / methods in the reply whose body is a placeholder.

    A class with a `pass` body is a legitimate idiom (`class E(Exception):
    pass`), so only functions and methods count."""
    try:
        new_tree = ast.parse(symedit._norm(after))
    except SyntaxError:
        return []
    out = []
    for name, span in new.items():
        if span is None or name in old:
            continue
        if span.kind not in ("function", "method"):
            continue
        node = symedit._node_at(new_tree, span.start)
        if node is not None and symedit.body_is_stub(node):
            out.append(name)
    return out


def _hollow(before: str, old: dict) -> set[str]:
    """Definitions the FILE holds with a placeholder body — dropping one
    loses nothing real, so it is never a refusal."""
    try:
        tree = ast.parse(symedit._norm(before))
    except SyntaxError:
        return set()
    out = set()
    for name, span in old.items():
        if span is None or span.kind not in ("function", "method"):
            continue
        node = symedit._node_at(tree, span.start)
        if node is not None and symedit.body_is_stub(node):
            out.add(name)
    return out


def _def_counts(code: str) -> Counter:
    """How many times each qualified definition occurs (overloads count)."""
    try:
        tree = ast.parse(symedit._norm(code))
    except SyntaxError:
        return Counter()
    counts: Counter = Counter()

    def walk(body, prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts[prefix + symedit.def_key(node)] += 1
            elif isinstance(node, ast.ClassDef):
                counts[prefix + node.name] += 1
                walk(node.body, prefix + node.name + ".")
    walk(tree.body, "")
    return counts


def _imported_names(code: str) -> dict[str, str]:
    """{name: module} for every name ``code`` imports."""
    out: dict[str, str] = {}
    try:
        tree = ast.parse(symedit._norm(code))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                out[a.asname or a.name] = node.module or "."
        elif isinstance(node, ast.Import):
            for a in node.names:
                out[(a.asname or a.name).split(".")[0]] = a.name
    return out


def check(before: str, after: str, lang: str, *, path: str,
          request: str = "",
          used_elsewhere: Callable[[str], str] | None = None) -> Finding:
    """What ``after`` (the reply) loses from ``before`` (the file).

    ``request`` is the REQUEST only — never a traceback: a name a traceback
    runs through is not a name anyone asked to remove. ``used_elsewhere(name)``
    returns the path of another project file that still uses ``name``, or
    "" — it decides whether a rename or a move is safe.
    """
    found_placeholders = tuple(placeholders(before, after, lang))
    warn_only = rewrite_asked(request, path)
    if lang == "python":
        old = python_defined(before)
        new = python_defined(after)
        if old is None or new is None:
            # a reply that does not parse is the syntax check's to report
            return Finding(path=path, placeholders=found_placeholders)
        hollow = _hollow(before, old)
        missing = [n for n in old if n not in new and n not in hollow]
        planted = tuple(_planted(before, after, old, new))
        duplicated = tuple((n, _copy_of(n, old)) for n in new
                           if n not in old and _copy_of(n, old))
        # COUNTED: of a getter and two overloads, a reply that keeps one of
        # each name has still lost definitions
        before_n, after_n = _def_counts(before), _def_counts(after)
        missing += [n for n, c in before_n.items()
                    if n in new and after_n.get(n, 0) < c
                    and n not in missing]
        # a method moved, unchanged in name, is not a loss: `Car.update`
        # still exists if the reply defines `update` in a renamed class —
        # it is the CLASS that is missing, and that is reported once
        missing = [n for n in missing
                   if not any(n.startswith(m + ".") for m in missing
                              if m != n)]
        allowed = tuple(n for n in missing if intended(n, request))
        missing = [n for n in missing if n not in allowed]
        stubbed = tuple(n for n in _stubbed(before, after, old, new)
                        if not intended(n, request))
        moved, missing = _moves(missing, old, new, before, after, request,
                                used_elsewhere)
        renamed, still_used, dropped = _renames(
            before, after, missing, old, new, after, used_elsewhere,
            request)
        return Finding(path=path, dropped=tuple(dropped), allowed=allowed,
                       moved=tuple(moved),
                       renamed=tuple(renamed), still_used=tuple(still_used),
                       placeholders=found_placeholders, stubbed=stubbed,
                       planted=planted, duplicated=duplicated,
                       warn_only=warn_only)
    old_names = other_defined(before, lang, path)
    new_names = other_defined(after, lang, path)
    if not old_names or not new_names:
        return Finding(path=path, placeholders=found_placeholders,
                       approximate=True)
    gone = [n for n in sorted(old_names - new_names) if not _used(n, after)]
    allowed = tuple(n for n in gone if intended(n, request))
    dropped = [n for n in gone if n not in allowed]
    # pattern-matched names miss and mis-split; one missing name in another
    # language is not enough to refuse a reply on
    if len(dropped) < 2:
        dropped = []
    return Finding(path=path, dropped=tuple(dropped), allowed=allowed,
                   placeholders=found_placeholders, approximate=True,
                   warn_only=warn_only)


def _moves(missing: list[str], old: dict, new: dict, before: str,
           after: str, request: str,
           used_elsewhere: Callable[[str], str] | None
           ) -> tuple[list, list]:
    """(moved, still missing). A name is MOVED when the reply imports it
    (it went to another file), or defines the same name under another
    class (a function turned into a method) — and nothing still calls it
    bare, unless the request asks for the move."""
    imported = _imported_names(after)
    moved, rest = [], []
    for name in missing:
        short = name.rsplit(".", 1)[-1].split("@", 1)[0]
        if "." not in name and short in imported:
            moved.append((name, f"imported from {imported[short]}"))
            continue
        span = old.get(name)
        homes = [n for n in new if n not in old and new[n] is not None
                 and n.rsplit(".", 1)[-1] == name.rsplit(".", 1)[-1]
                 and span is not None and new[n].kind in ("function",
                                                          "method")
                 and span.kind in ("function", "method")]
        if homes:
            still = _called(name, after) and "." not in name
            if not still and used_elsewhere is not None and \
                    not intended(name, request):
                try:
                    still = bool(used_elsewhere(name))
                except Exception:                        # noqa: BLE001
                    still = False
            if not still or intended(name, request):
                moved.append((name, f"now `{homes[0]}`"))
                continue
        rest.append(name)
    return moved, rest


def _stubbed(before: str, after: str, old: dict, new: dict) -> list[str]:
    """Names whose real body became a placeholder in the reply."""
    try:
        old_tree = ast.parse(symedit._norm(before))
        new_tree = ast.parse(symedit._norm(after))
    except SyntaxError:
        return []
    out = []
    for name, span in new.items():
        if span is None or old.get(name) is None:
            continue
        if span.kind not in ("function", "method"):
            continue
        n_old = symedit._node_at(old_tree, old[name].start)
        n_new = symedit._node_at(new_tree, span.start)
        if n_old is None or n_new is None:
            continue
        if symedit.body_is_stub(n_new) and not symedit.body_is_stub(n_old):
            out.append(name)
    return out


def _body_text(code: str, span: symedit.Span | None) -> str:
    if span is None:
        return ""
    lines = symedit.lines_of(symedit._norm(code))
    body = lines[span.start + 1:span.end]
    return "".join(ln.strip() + "\n" for ln in body if ln.strip())


def _arity(code: str, span: symedit.Span | None) -> int:
    try:
        node = symedit._node_at(ast.parse(symedit._norm(code)), span.start)
    except (SyntaxError, AttributeError):
        return -1
    args = getattr(node, "args", None)
    if args is None:
        return -1
    return len(args.args) + len(args.kwonlyargs) + bool(args.vararg) + \
        bool(args.kwarg)


def _renames(before: str, after: str, missing: list[str], old: dict,
             new: dict, reply: str,
             used_elsewhere: Callable[[str], str] | None,
             request: str = ""
             ) -> tuple[list, list, list]:
    """(allowed renames, renames still in use, really dropped).

    A rename is a NEW name with the old one's body: alike enough (more alike
    for a short body — "return self.items…" is half of many methods), the
    same number of parameters, and not a name the request asks to ADD — a
    requested `remove()` is the addition, not `summary()` renamed."""
    fresh = [n for n in new if n not in old and new[n] is not None
             and not mentioned(n, request)]
    renamed, still, dropped = [], [], []
    for name in missing:
        span = old.get(name)
        if span is None or span.kind not in ("function", "method", "class"):
            dropped.append(name)
            continue
        parent = name.rsplit(".", 1)[0] if "." in name else ""
        body = _body_text(before, span)
        need = RENAME_SIMILARITY if len(body) >= 120 else 0.85
        arity = _arity(before, span)
        best, score = "", 0.0
        for cand in fresh:
            c_span = new[cand]
            if c_span.kind != span.kind:
                continue
            c_parent = cand.rsplit(".", 1)[0] if "." in cand else ""
            if c_parent != parent:
                continue
            if span.kind != "class" and _arity(after, c_span) != arity:
                continue
            ratio = difflib.SequenceMatcher(
                None, body, _body_text(after, c_span)).ratio()
            if ratio > score:
                best, score = cand, ratio
        if not best or score < need:
            dropped.append(name)
            continue
        where = ""
        if _used(name, reply):
            where = "the reply itself"
        elif used_elsewhere is not None:
            try:
                where = used_elsewhere(name) or ""
            except Exception:                            # noqa: BLE001
                where = ""
        if where:
            still.append((name, best, where))
        else:
            renamed.append((name, best))
            fresh.remove(best)
    return renamed, still, dropped
