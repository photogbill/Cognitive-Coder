# SPDX-License-Identifier: Apache-2.0
"""What a specification says that a build will faithfully get wrong.

WHY THIS EXISTS, 2026-10-22
---------------------------
BUILD_SPEC.md line 89 said `Reserve_energy: float = 100.0`. The interface
stage pinned the capital R faithfully; one model then silently lowercased
it and diverged from the spec and from `test_physics.py` for the rest of
the build. Nobody decided which spelling was meant — a model did, twice,
differently.

`lint(spec)` is deterministic and runs BEFORE the plan. It reports, with
the line, things in a spec that will be carried into code exactly as
written and are almost certainly not what the author meant:

  * a backticked Python name that breaks PEP 8 — a capitalised field or
    function (`Reserve_energy`), camelCase (`warpHeat`) beside snake_case
    elsewhere, a lower-case class — with the spelling it probably meant;
  * the same identifier spelled two ways in backticks (`shield_sectors` and
    `shield_sector`, `warp_heat` and `warpHeat`);
  * a numeric constant given two different values (`SUBTICKS_PER_TURN = 10`
    on one line, `= 12` on another).

It never changes the spec and never guesses silently: a finding is a
sentence the operator confirms or fixes. Ambiguity that needs reading
("does bleed-through apply before or after rotation?") is the judge's and
the model's business, not this file's.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

_TICK = re.compile(r"`([^`\n]{1,80})`")
#: `name: type = value`, `name(`, `name =`, or a bare identifier
_IDENT = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:[:(=]|$)")
_CLASSISH = re.compile(r"\b(?:class|dataclass|enum)\s+`?([A-Za-z_]\w*)")
_ASSIGN = re.compile(r"`?\b([A-Z][A-Z0-9_]{2,})\b`?\s*(?:=|is|of|:)\s*`?"
                     r"(-?\d+(?:\.\d+)?)`?")
_SNAKE = re.compile(r"^[a-z_][a-z0-9_]*$")
_UPPER = re.compile(r"^[A-Z][A-Z0-9_]*$")
_PASCAL = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_CAMEL = re.compile(r"^[a-z]+(?:[A-Z][a-z0-9]*)+$")
_DUNDER = re.compile(r"^__\w+__$")
#: words that are not identifiers even in backticks
_NOT_NAMES = {"true", "false", "none", "null", "and", "or", "not", "if",
              "else", "for", "in", "return", "import", "from", "def",
              "class", "self", "int", "float", "str", "bool", "list",
              "dict", "tuple", "set", "python", "pygame", "pytest", "src",
              "tests", "main", "py"}


@dataclass(frozen=True)
class SpecFinding:
    line: int            # 1-based line of the spec
    kind: str            # case | spelling | constant
    text: str            # the offending spelling
    sentence: str        # what to confirm or fix
    suggestion: str = ""


def _snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    s = re.sub(r"__+", "_", s)
    return s.lower()


def _names_by_line(spec: str) -> list[tuple[int, str]]:
    out = []
    for no, line in enumerate(spec.splitlines(), 1):
        for tick in _TICK.findall(line):
            m = _IDENT.match(tick)
            if not m:
                continue
            name = m.group(1)
            if name.lower() in _NOT_NAMES or _DUNDER.match(name) \
                    or len(name) < 3:
                continue
            out.append((no, name))
    return out


def lint(spec: str) -> list[SpecFinding]:
    """Findings, in line order. Empty when the spec is consistent."""
    spec = spec or ""
    lines = spec.splitlines()
    classes = {m.group(1) for m in _CLASSISH.finditer(spec)}
    names = _names_by_line(spec)
    findings: list[SpecFinding] = []
    seen_case: set[str] = set()

    # 1. case: a field/function that is Capitalised_like_this, or camelCase
    #    in a spec that is otherwise snake_case
    styles = Counter()
    for _no, n in names:
        if _SNAKE.match(n):
            styles["snake"] += 1
        elif _CAMEL.match(n):
            styles["camel"] += 1
    house = "camel" if styles["camel"] > styles["snake"] else "snake"
    for no, n in names:
        if n in classes or _UPPER.match(n) or _PASCAL.match(n) and \
                n in classes:
            continue
        key = (no, n)
        if key in seen_case:
            continue
        if re.match(r"^[A-Z][a-z0-9]*_[A-Za-z0-9_]+$", n):
            # `Reserve_energy`: a capital letter then snake_case — not a
            # class (no class says so), not a constant, not camelCase
            seen_case.add(key)
            findings.append(SpecFinding(
                no, "case", n,
                f"`{n}` is spelled with a capital letter and an underscore; "
                f"a field or function is snake_case in Python. The build "
                f"will keep the capital exactly as written, and a model may "
                f"'correct' it later. Did you mean `{_snake(n)}`?",
                suggestion=_snake(n)))
        elif house == "snake" and _CAMEL.match(n):
            seen_case.add(key)
            findings.append(SpecFinding(
                no, "case", n,
                f"`{n}` is camelCase in a spec whose other names are "
                f"snake_case; the build will use it as written. Did you "
                f"mean `{_snake(n)}`?",
                suggestion=_snake(n)))

    # 2. spelling: the same identifier in two spellings
    by_key: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for no, n in names:
        key = _snake(n).replace("_", "")
        by_key[key][n].append(no)
    for key, forms in by_key.items():
        if len(forms) < 2:
            continue
        # a class and its instance field may legitimately share letters
        # (`Ship` / `ship`); only flag forms that differ in more than case
        # of the first letter, or that differ by an underscore
        ranked = sorted(forms.items(), key=lambda kv: -len(kv[1]))
        main, _ = ranked[0]
        for other, where in ranked[1:]:
            if other in classes or main in classes:
                continue
            if other.lower() == main.lower() and other[0].isupper() != \
                    main[0].isupper() and other[1:] == main[1:]:
                continue
            findings.append(SpecFinding(
                where[0], "spelling", other,
                f"`{other}` (line {where[0]}) and `{main}` (line "
                f"{forms[main][0]}) look like the same name spelled two "
                f"ways; the build will create both. Which is it?",
                suggestion=main))

    # 3. constants: one UPPER_CASE name, two values
    values: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list))
    for no, line in enumerate(lines, 1):
        for m in _ASSIGN.finditer(line):
            values[m.group(1)][m.group(2)].append(no)
    for const, vals in values.items():
        if len(vals) < 2:
            continue
        items = sorted(vals.items(), key=lambda kv: kv[1][0])
        (v1, l1), (v2, l2) = items[0], items[1]
        findings.append(SpecFinding(
            l2[0], "constant", const,
            f"`{const}` is {v1} on line {l1[0]} and {v2} on line {l2[0]}; "
            f"the build will pin whichever it reads last. Which is it?",
            suggestion=v1))

    findings.sort(key=lambda f: (f.line, f.kind))
    return findings


def report(findings: list[SpecFinding]) -> str:
    """The findings as the operator reads them; "" when there are none."""
    if not findings:
        return ""
    out = [f"The specification has {len(findings)} thing(s) a build will "
           f"carry into code exactly as written:"]
    for f in findings:
        out.append(f"  line {f.line}: {f.sentence}")
    out.append("Fix the spec, or confirm each one; nothing is guessed.")
    return "\n".join(out)
