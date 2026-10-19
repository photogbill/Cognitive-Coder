# SPDX-License-Identifier: Apache-2.0
"""Spec vectors: the examples a request states outright, as tests no model
can rewrite.

WHY THIS EXISTS, 2026-10-16
---------------------------
Sixteen test failures on the Star Trek build; ten were the TEST's fault.
Both models wrote `bearing_between(0, 0, 1, 1) == 45` from a y-up habit
against a spec that says north is `(0, -1)`. And where the tests and the
code agreed, they were sometimes both wrong: the spec's bank ids count
from 1 (`fi 1-2`), the code indexed `weapons[bank_id]` from 0, and the test
asserted the same — two models agreeing is not verification.

A specification often states its examples outright:

    Implement `parse_bank_spec(spec: str) -> tuple[int, ...]` supporting
    single ints (`"1"` -> `(1,)`), ranges (`"1-3"` -> `(1, 2, 3)`), and
    comma lists (`"1,3"` -> `(1, 3)`).
    `SUBTICKS_PER_TURN: int = 10`

Those are tests already written — by the operator, in the request. This
module lifts them BY RULE (C5, never a model's reading), writes them to a
test file the harness owns, and the runner runs that file as part of the
module's verification, before the model's own tests. A vector failing is
the module's failure, with the request's sentence as the message. A
model-written test that contradicts a vector is wrong by definition.

What is lifted:

  * `` `literal` -> `literal` `` (also `→`, `=>`, `returns`) in a sentence
    that names a function as `` `name(…)` ``: `name(literal) == literal`.
    Both sides must be Python literals; a tuple on the left with as many
    items as the signature has parameters is the argument list;
  * `` `NAME: type = literal` `` or `` `NAME = literal` `` with NAME in
    UPPER_CASE: `module.NAME == literal`.

Each vector is attributed to the file whose section it sits in (the
nearest heading naming a path), else to a path named in the same sentence.
Nothing else is guessed. For the Star Trek spec this yields seven
assertions — the seven that cannot be wrong.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
import re

from . import specslice

#: The first line of every file this module writes, so nothing — the
#: planner, the skeleton, a stub check — mistakes it for a model's test.
SENTINEL = ("cc-spec-vectors: written by Cognitive Coder from the request's "
            "own examples, by rule; not a model's test and not yours to "
            "change")
#: Where the vectors of `src/parser.py` go: `tests/test_spec_vectors_parser.py`.
FILE_STEM = "test_spec_vectors_"

_BACKTICK = re.compile(r"`([^`\n]+)`")
_ARROW = re.compile(r"`(?P<a>[^`\n]+)`\s*(?:->|→|=>|returns)\s*`(?P<b>[^`\n]+)`")
_SIGNATURE = re.compile(r"`(?P<name>[A-Za-z_]\w*)\((?P<params>[^`()]*)\)[^`]*`")
_CONSTANT = re.compile(
    r"`(?P<name>[A-Z][A-Z0-9_]{2,})\s*(?::\s*(?P<type>[^=`]+?))?\s*=\s*"
    r"(?P<value>[^`]+)`")
_PATH = re.compile(r"(?<![\w/.\\-])((?:[\w.-]+[/\\])+[\w.-]+\.py)(?![\w])")


@dataclass(frozen=True)
class Vector:
    """One example the request states, attributed to a file."""
    path: str              # the module it tests
    kind: str              # "call" | "constant"
    name: str              # the function or constant
    args: str              # the call's argument list as source ("" for a constant)
    expected: str          # the expected value as source
    quote: str             # the request's sentence, word for word

    @property
    def call(self) -> str:
        return f"{self.name}({self.args})" if self.kind == "call" else self.name

    def describe(self) -> str:
        return f"{self.path}: {self.call} == {self.expected}"


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def _literal(text: str) -> bool:
    try:
        ast.literal_eval(text.strip())
        return True
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False


def _sentences(text: str) -> list[str]:
    """Sentences, with code spans intact and fences skipped."""
    out: list[str] = []
    fenced = False
    for raw in (text or "").splitlines():
        if raw.strip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or not raw.strip():
            continue
        line = re.sub(r"^\s*(?:[-*•+]|\d+[.)])\s+", "", raw).strip()
        line = line.lstrip("#> ").strip()
        out.append(line)
    return out


def _params(sig: str) -> list[str]:
    """Parameter names of `a: int, b: str = ""` — "" when unparseable."""
    try:
        tree = ast.parse(f"def _f({sig}): pass")
    except SyntaxError:
        return []
    fn = tree.body[0]
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args
             if a.arg not in ("self", "cls")]
    return names


def extract(request: str, paths: Sequence[str] = ()) -> list[Vector]:
    """Every vector the request states, attributed to a file.

    ``paths`` are the plan's files; a vector attributed to a file that is
    not planned is dropped (it tests nothing this build writes).
    """
    planned = {p.replace("\\", "/").lower(): p for p in paths}
    out: list[Vector] = []
    seen: set[tuple[str, str, str]] = set()
    for section in specslice.sections(request):
        heading_path = _path_in(section.heading)
        for sentence in _sentences(section.text):
            path = _path_in(sentence) or heading_path
            if not path:
                continue
            if planned and path.lower() not in planned:
                continue
            path = planned.get(path.lower(), path)
            for v in _vectors_in(sentence, path):
                key = (v.path, v.call, v.expected)
                if key in seen:
                    continue
                seen.add(key)
                out.append(v)
    return out


def _path_in(text: str) -> str:
    m = _PATH.search(text or "")
    return m.group(1).replace("\\", "/") if m else ""


def _vectors_in(sentence: str, path: str) -> list[Vector]:
    out: list[Vector] = []
    quote = sentence.strip()
    # constants: `NAME: int = 10`
    for m in _CONSTANT.finditer(sentence):
        value = m.group("value").strip()
        if _literal(value):
            out.append(Vector(path=path, kind="constant", name=m.group("name"),
                              args="", expected=value, quote=quote))
    # calls: a named function, then `literal` -> `literal` pairs
    sig = _SIGNATURE.search(sentence)
    if sig is None:
        return out
    name = sig.group("name")
    params = _params(sig.group("params"))
    for m in _ARROW.finditer(sentence):
        a, b = m.group("a").strip(), m.group("b").strip()
        if not (_literal(a) and _literal(b)):
            continue
        args = a
        try:
            value = ast.literal_eval(a)
        except Exception:                                # noqa: BLE001
            continue
        if (isinstance(value, tuple) and len(params) > 1
                and len(value) == len(params)):
            args = a.strip()[1:-1]           # the tuple IS the argument list
        out.append(Vector(path=path, kind="call", name=name, args=args,
                          expected=b, quote=quote))
    return out


# --------------------------------------------------------------------------
# the test file
# --------------------------------------------------------------------------

def test_path_for(module_path: str, test_dir: str = "tests") -> str:
    stem = module_path.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.endswith(".py") else stem
    return f"{test_dir}/{FILE_STEM}{stem}.py"


def _module_of(path: str) -> str:
    rel = path.replace("\\", "/").strip("/")
    rel = rel[:-3] if rel.endswith(".py") else rel
    return rel.replace("/", ".")


def test_file(module_path: str, vectors: Sequence[Vector]) -> str:
    """A unittest file for one module's vectors. Floats compare to six
    places (a formula's rounding is not a disagreement with the request);
    everything else exactly."""
    module = _module_of(module_path)
    lines = [
        f"# {SENTINEL}",
        f'"""Examples the request states outright for `{module_path}`, as tests.',
        "",
        "Each test quotes the sentence it comes from. A failure here means the",
        "module disagrees with the request's own example; the test is not",
        'the thing to change."""',
        "import unittest",
        "",
        f"import {module} as _m",
        "",
        "",
        "def _close(a, b, places=6):",
        '    """Equal, with floats compared to `places`."""',
        "    if isinstance(a, float) or isinstance(b, float):",
        "        try:",
        "            return round(abs(float(a) - float(b)), places) == 0",
        "        except (TypeError, ValueError):",
        "            return False",
        "    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):",
        "        return (type(a) is type(b) and len(a) == len(b)",
        "                and all(_close(x, y, places) for x, y in zip(a, b)))",
        "    if isinstance(a, dict) and isinstance(b, dict):",
        "        return (a.keys() == b.keys()",
        "                and all(_close(a[k], b[k], places) for k in a))",
        "    return a == b",
        "",
        "",
        "class SpecVectors(unittest.TestCase):",
        f'    """The request\'s own examples for `{module_path}`."""',
    ]
    for i, v in enumerate(vectors, 1):
        safe = re.sub(r"\W+", "_", v.name).strip("_") or "vector"
        # `repr` picks the quoting: the request's sentence and the call
        # both contain quotes of their own
        said = repr("the request says: " + " ".join(v.quote.split()))
        call = repr(v.call)
        lines += [
            "",
            f"    def test_{i:02d}_{safe}(self):",
            f"        {repr(_short(v.quote))}",
            f"        got = _m.{v.call}",
            f"        want = {v.expected}",
            f"        self.assertTrue(_close(got, want), {said}",
            f"                        + '\\n' + {call} + ' gave ' + repr(got)",
            "                        + ', not ' + repr(want))",
        ]
    lines += ["", "", 'if __name__ == "__main__":', "    unittest.main()", ""]
    return "\n".join(lines)


def _short(text: str, n: int = 110) -> str:
    t = " ".join(text.split()).replace('"""', "'''")
    return t if len(t) <= n else t[:n - 1] + "…"


def is_vector_file(text: str) -> bool:
    return SENTINEL in (text or "")


def by_module(vectors: Sequence[Vector]) -> dict[str, list[Vector]]:
    out: dict[str, list[Vector]] = {}
    for v in vectors:
        out.setdefault(v.path, []).append(v)
    return out
