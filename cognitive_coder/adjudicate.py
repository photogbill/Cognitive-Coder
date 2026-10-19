# SPDX-License-Identifier: Apache-2.0
"""The judge: when a test and the module it tests disagree, which one does
the REQUEST support? Ruled by a quotation, and the quotation is checked.

WHY THIS EXISTS, 2026-10-16
---------------------------
The loop's rule was: a test that fails against its module is never
rewritten to agree with it — the module must change (`loop._test_disagrees
_with_code`). The rule is right against the failure it was written for: a
repairer pointed at a test bent `assert add(1, 1) == 2` into `== 0` to
match `a - b`. And it was wrong ten times out of sixteen on the Star Trek
build, where the TESTS assumed a y-up world (`bearing_between(0, 0, 1, 1)
== 45`) against a request that says north is `(0, -1)`. Devstral, told
"the code must change", returned the file unchanged — the honest answer —
and the build had no path forward but "fix by hand".

So before a module is blamed, ONE short question is asked of the model:
which does the request support, the module or the test — and quote the
request's words that decide it. The reply is JSON. **The quote is
checked**: it has to appear in the request (whitespace and quote marks
normalised). No quote, or a quote that is not there, and the verdict is
`unclear`, which is the old conservative rule. A model that cannot point
at the sentence does not get to overrule the test.

What a verdict does (the loop and the session):

  * `code`    → the module is repaired against the test, as before;
  * `test`    → the TEST is repaired, with the quote as its diagnostic;
  * `unclear` → as `code`, and the operator is told it was unclear.

One question per (test file, failure signature). Every verdict, quote and
"why" goes to BUILD_LOG.txt and the journal (`adjudicate` events).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from .types import Diagnostic, Message

VERDICTS = ("code", "test", "unclear")
#: A quote shorter than this proves nothing ("MUST", "0.0"): treated as
#: no quote.
MIN_QUOTE_CHARS = 12
#: The judge's reply budget. A verdict is three fields; a model that
#: writes an essay is cut here and the JSON is dug out of what it wrote.
MAX_TOKENS = 400
#: How much of each side the judge is shown. The module and the test are
#: shown whole up to this; past it, the parts the failure names.
SIDE_CHARS = 12_000

CONTRACT = ("OUTPUT CONTRACT\nReturn one JSON object and nothing else, in "
            "this exact shape:\n"
            '{"verdict": "code" | "test" | "unclear", '
            '"quote": "<the request\'s own words that decide it, copied '
            'exactly — or empty>", "why": "<one sentence>"}\n'
            "No fence, no prose before or after it.")


@dataclass(frozen=True)
class Verdict:
    verdict: str                  # "code" | "test" | "unclear"
    quote: str = ""
    why: str = ""
    raw: str = ""
    #: the quote was found in the request (so the verdict stands)
    checked: bool = False
    #: what the verdict was before the quote check downgraded it
    claimed: str = ""

    @property
    def test_is_wrong(self) -> bool:
        return self.verdict == "test" and self.checked

    def describe(self) -> str:
        head = {"code": "the MODULE is wrong",
                "test": "the TEST is wrong",
                "unclear": "unclear"}.get(self.verdict, self.verdict)
        bits = [head]
        if self.claimed and self.claimed != self.verdict:
            bits.append(f"(the model said {self.claimed!r}, but its quote "
                        f"is not in the request, so the verdict does not "
                        f"stand)")
        if self.quote and self.checked:
            bits.append(f'— the request says: "{_short(self.quote, 160)}"')
        elif self.quote:
            bits.append(f'— it quoted "{_short(self.quote, 100)}", which '
                        f"the request does not say")
        if self.why:
            bits.append(f"— {_short(self.why, 200)}")
        return " ".join(bits)


def _short(text: str, n: int) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[:n - 1] + "…"


# --------------------------------------------------------------------------
# the question
# --------------------------------------------------------------------------

def task_text(*, module_path: str, module_code: str, test_path: str,
              test_code: str, failure: str, request: str,
              vectors: str = "") -> str:
    """The [TASK] of the one question."""
    lines = [
        f"A test and the module it tests disagree. `{test_path}` fails "
        f"against `{module_path}`. Decide which one the REQUEST supports: "
        f"the module (the test asserts something the request does not "
        f"say, or says differently) or the test (the module does not do "
        f"what the request says). If the request does not settle it, say "
        f"so.",
        "",
        "Rules: judge by the request's words, not by what seems usual — "
        "a coordinate convention, a numbering from 1, a formula stated "
        "outright all outrank habit. Quote the request's sentence that "
        "decides it, copied exactly; a verdict without a quote from the "
        "request does not stand.",
    ]
    if vectors:
        lines += ["", vectors]
    lines += [
        "",
        f"[THE REQUEST, as far as it concerns {module_path}]",
        request.strip(),
        "",
        f"[THE MODULE — {module_path}]",
        _cut(module_code, SIDE_CHARS),
        "",
        f"[THE TEST — {test_path}]",
        _cut(test_code, SIDE_CHARS),
        "",
        "[THE FAILURE]",
        failure.strip(),
    ]
    return "\n".join(lines)


def _cut(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n].rstrip() + f"\n… ({len(text) - n:,} more characters not shown)"


def failure_text(diagnostics: Sequence[Diagnostic], limit: int = 6) -> str:
    """The failing tests' lines, with the asserting line where known."""
    out: list[str] = []
    for d in diagnostics:
        if not d.is_error:
            continue
        where = d.where()
        head = f"{where}: " if where else ""
        out.append(f"{head}{d.message}" + (f" [{d.code}]" if d.code and
                                            d.code.startswith("in ") else ""))
        for line in (d.source_excerpt or "").splitlines():
            if line.startswith(">>"):
                out.append(f"    {line}")
        if len(out) >= limit * 2:
            break
    return "\n".join(out) or "(no located failure)"


def vectors_note(module_path: str, passed: bool | None, count: int = 0
                 ) -> str:
    """What the judge is told about the request's own examples."""
    if passed is None:
        return ""
    if passed:
        return (f"NOTE: `{module_path}` passes every example the request "
                f"states outright ({count or 'its'} spec vector"
                f"{'' if count == 1 else 's'}). A test that contradicts "
                f"one of those examples is wrong.")
    return (f"NOTE: `{module_path}` FAILS an example the request states "
            f"outright (a spec vector). The module is wrong at least there.")


# --------------------------------------------------------------------------
# the answer
# --------------------------------------------------------------------------

_JSON = re.compile(r"\{.*\}", re.S)


def parse(reply: str) -> Verdict:
    """The verdict in a reply, tolerant of fences and prose around it."""
    text = reply or ""
    m = _JSON.search(text)
    data: Any = None
    if m:
        chunk = m.group(0)
        for candidate in (chunk, _tidy(chunk)):
            try:
                data = json.loads(candidate)
                break
            except (ValueError, TypeError):
                continue
    if not isinstance(data, dict):
        # a bare word answer still counts, with no quote to check
        word = text.strip().strip("`'\"").lower()
        verdict = word if word in VERDICTS else "unclear"
        return Verdict(verdict=verdict, raw=text, claimed=verdict)
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        verdict = "unclear"
    return Verdict(verdict=verdict, quote=str(data.get("quote") or ""),
                   why=str(data.get("why") or ""), raw=text,
                   claimed=verdict)


def _tidy(chunk: str) -> str:
    """Near-JSON a small model writes: single quotes, a trailing comma."""
    t = re.sub(r",\s*([}\]])", r"\1", chunk)
    if "'" in t and '"' not in t:
        t = t.replace("'", '"')
    return t


def _norm(text: str) -> str:
    t = str(text or "")
    t = t.replace("“", '"').replace("”", '"').replace("’", "'").replace(
        "‘", "'").replace("`", "")
    t = re.sub(r"[\s ]+", " ", t).strip().lower()
    return t


def check(verdict: Verdict, request: str) -> Verdict:
    """The quote has to be in the request, or the verdict is `unclear`.

    `unclear` and `code` keep their verdict either way: they change
    nothing the old rule would not have done. Only `test` — the verdict
    that overrules a test — needs the proof.
    """
    quote = _norm(verdict.quote)
    present = (len(quote) >= MIN_QUOTE_CHARS and quote in _norm(request))
    if verdict.verdict != "test":
        return Verdict(verdict=verdict.verdict, quote=verdict.quote,
                       why=verdict.why, raw=verdict.raw, checked=present,
                       claimed=verdict.claimed)
    if present:
        return Verdict(verdict="test", quote=verdict.quote, why=verdict.why,
                       raw=verdict.raw, checked=True, claimed="test")
    return Verdict(verdict="unclear", quote=verdict.quote, why=verdict.why,
                   raw=verdict.raw, checked=False, claimed="test")


def ask(llm: Any, messages: Sequence[Message], request: str, *,
        cancel: Any = None, max_tokens: int = MAX_TOKENS) -> Verdict:
    """One completion, parsed and checked. Never raises on the model's
    account: an error or an empty reply is `unclear`."""
    try:
        completion = llm.complete(messages, temperature=0.0,
                                  max_tokens=max_tokens, cancel=cancel)
    except Exception as exc:                              # noqa: BLE001
        return Verdict(verdict="unclear", why=f"the judge could not be "
                                              f"asked ({exc})")
    if completion.finish_reason == "cancelled":
        return Verdict(verdict="unclear", why="cancelled")
    if completion.finish_reason == "error" or not (completion.text or "").strip():
        return Verdict(verdict="unclear",
                       why=completion.error or "the judge returned nothing")
    from .personas import strip_think
    return check(parse(strip_think(completion.text)), request)


# --------------------------------------------------------------------------
# what a `test` verdict becomes
# --------------------------------------------------------------------------

def diagnostic_for_test(verdict: Verdict, test_path: str,
                        failing: Sequence[Diagnostic]) -> Diagnostic:
    """The diagnostic a test is repaired against when the judge ruled it
    wrong: the request's words, and the assertion that contradicts them."""
    first = next((d for d in failing if d.is_error
                  and d.file and d.file.replace("\\", "/").endswith(
                      test_path.replace("\\", "/").rsplit("/", 1)[-1])),
                 None) or next((d for d in failing if d.is_error), None)
    line = first.line if first is not None else 0
    excerpt = first.source_excerpt if first is not None else ""
    what = (first.message if first is not None else "the failing assertion")
    return Diagnostic(
        file=test_path, line=line, severity="error", code="judge",
        tool="adjudicate", source_excerpt=excerpt,
        message=(f'The request says: "{_short(verdict.quote, 300)}". This '
                 f"test asserts something else ({_short(what, 160)}). The "
                 f"module follows the request; change the assertion so it "
                 f"asserts what the request says. Do not change the module."
                 + (f" ({_short(verdict.why, 160)})" if verdict.why else "")))
