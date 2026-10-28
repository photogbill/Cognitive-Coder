# SPDX-License-Identifier: Apache-2.0
"""FIM mode: holes for an ordinary code model (fill-in-the-middle).

WHY THIS EXISTS, 2026-10-27
---------------------------
The canvas (canvas.py) is the engine's best repair: the hole the error is
in is rewritten and nothing else can change. It needs a diffusion model.
Appendix G.4 of the build specification says fill-in-the-middle is "PROVEN,
and underused" for the same reason — a model given the code before and
after a hole writes only the hole and "structurally cannot touch code
outside it" — and every provider reported `supports_fim`, and canvas.py's
own docstring recorded that nothing ever read it. Qwen3-Coder supports FIM
in every version; so do Qwen2.5-Coder, DeepSeek-Coder, StarCoder, CodeGemma
and Codestral. For all of them the repair of a located error was a whole
file (2,500 tokens, the dropped-code guard watching) or a narrowed edit
(one definition, a fence, an instruction a small model can misread).

A FIM hole is narrower than either: ~200 tokens, no fence, no contract to
follow, nothing outside the hole to drop, and — unlike a canvas — no
256-token limit, so a long body is a hole too (10 of 19 canvas repairs on
the 2026-10-18 build were skipped because the body did not fit one).

WHAT THIS MODULE DECIDES, BY RULE — the model decides nothing here:

  1. WHERE the hole goes: the body of the function the located error is
     in (`canvas.repair_hole`, with FIM's larger limits), else the body of
     a function the failing test calls, else the body the engine doubts
     most (`Loop.doubt`, which the resample's disagreement map fills for
     an ordinary model). Never a hole the model chooses.
  2. WHAT the model is shown: the file up to the hole as the PREFIX —
     ending in a short comment, at the hole's indentation, that says what
     the old body did wrong (the error's one line, the values at the
     failure) — and the file after the hole as the SUFFIX. The comment is
     in the prompt only; the file on disk never has it.
  3. HOW a fill is cleaned and spliced (canvas.clean_fill / splice, the
     "open" style: a run-on past the hole is cut where the code dedents
     out of the body), and whether it can be used: it must parse, must
     not be the old lines again, must not define a name the file already
     has.
  4. Whether the MODEL can fill a hole at all: `probe_request` /
     `probe_verdict` — a known hole with a known answer, asked once at
     the model check. A name in a list is not evidence; a filled hole is.

Nothing here talks to a model or a file; loop.py does that.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Any

from . import canvas as cv
from .types import Diagnostic, FimRequest, FimResult

#: A body longer than this is a window around the error line, as for a
#: canvas — but a FIM reply has no canvas to fit, so the limit is the
#: model's attention, not the server's: three times the canvas's.
BODY_MAX_LINES = 120
#: Tokens a fill may run to. A body under BODY_MAX_LINES is well inside.
MAX_TOKENS = 1024
#: The hint comment's parts are cut to this many characters each.
HINT_CHARS = 160
#: Rows of the values block shown in the hint.
HINT_VALUES = 4
#: The second candidate's temperature step above the first.
SECOND_STEP = 0.25

#: The probe: a hole whose answer every code model knows.
PROBE_PREFIX = ('def add(a, b):\n'
                '    """Return the sum of a and b."""\n')
PROBE_SUFFIX = ('\n\n\ndef sub(a, b):\n'
                '    """Return a minus b."""\n'
                '    return a - b\n')
_PROBE_ANSWER = re.compile(r"return\s+\(?\s*(a\s*\+\s*b|b\s*\+\s*a)\s*\)?")


@dataclass(frozen=True)
class Candidate:
    """One filled hole, judged."""
    result: FimResult
    fill: str | None
    code: str
    problem: str          # "" = usable
    temperature: float = 0.0
    seed: int | None = None
    salvaged: bool = False

    @property
    def usable(self) -> bool:
        return not self.problem

    def summary(self) -> dict:
        r = self.result
        return {"temperature": self.temperature, "seed": self.seed,
                "ok": self.usable, "problem": self.problem,
                "finish_reason": r.finish_reason, "tokens_in": r.tokens_in,
                "tokens_out": r.tokens_out, "prompt_ms": r.prompt_ms,
                "decode_ms": r.decode_ms, "cached_tokens": r.cached_tokens,
                "lines": (self.fill or "").count("\n")}

    def describe(self) -> str:
        r = self.result
        seed = "none" if self.seed is None else self.seed
        head = (f"temperature {self.temperature:g}, seed {seed}, "
                f"{r.tokens_out} token(s) out, prefill {r.prompt_ms} ms "
                f"({r.cached_tokens} prompt token(s) from cache), decode "
                f"{r.decode_ms} ms")
        if self.usable:
            return (f"{head} — usable: {(self.fill or '').count(chr(10))} "
                    f"line(s)"
                    + (" (salvaged: the reply ran past the hole and was "
                       "cut where it left it)" if self.salvaged else ""))
        return f"{head} — REJECTED: {self.problem}"


# --------------------------------------------------------------------------
# where the hole goes
# --------------------------------------------------------------------------

def hole_for_repair(code: str, lang: str, path: str,
                    diagnostics: Sequence[Diagnostic],
                    same_file: Callable[[str, str], bool],
                    fits: Callable[[cv.Hole], bool] | None = None
                    ) -> cv.Hole | None:
    """The hole for a located error: the body it is in, up to
    BODY_MAX_LINES, else a window around its line."""
    return cv.repair_hole(code, lang, path, diagnostics, same_file,
                          count_tokens=None, max_body_lines=BODY_MAX_LINES,
                          fits=fits or (lambda _h: True))


def hole_for_function(code: str, path: str, name: str,
                      why: str = "test") -> cv.Hole | None:
    hole = cv.function_hole(code, path, name, why=why)
    if hole is None or hole.size > BODY_MAX_LINES:
        return None
    return hole


# --------------------------------------------------------------------------
# what the model is shown
# --------------------------------------------------------------------------

def split(code: str, hole: cv.Hole) -> tuple[str, str]:
    """(prefix, suffix): the file before the hole and after it, exactly."""
    lines = cv._lines(code)
    before = "".join(lines[:hole.start])
    after = "".join(lines[hole.end:])
    if before and not before.endswith("\n"):
        before += "\n"
    return before, after


def hint(hole: cv.Hole, diagnostics: Sequence[Diagnostic], path: str,
         same_file: Callable[[str, str], bool], comment: str = "#") -> str:
    """The comment that ends the prefix: what went wrong in the old body.

    A FIM model takes no instruction, so the instruction is where such a
    model reads one — a comment right before the code it writes. In the
    prompt only; `split` gives the file as it is."""
    pad = " " * hole.indent
    err = next((d for d in diagnostics
                if d.is_error and same_file(d.file, path)), None)
    if err is None:
        err = next((d for d in diagnostics if d.is_error), None)
    lines: list[str] = []
    what = (f"the body of `{hole.name}`" if hole.kind == "body" and hole.name
            else hole.label)
    if err is not None:
        message = _one_line(err.message)
        lines.append(f"{pad}{comment} FIX: {what} failed: {message}")
        shown = 0
        for row in (err.values or "").splitlines():
            if row.startswith("    ") and shown < HINT_VALUES:
                lines.append(f"{pad}{comment}   {row.strip()[:HINT_CHARS]}")
                shown += 1
    else:
        lines.append(f"{pad}{comment} FIX: rewrite {what} so its tests "
                     f"pass")
    lines.append(f"{pad}{comment} Write the corrected {what} below; the "
                 f"signature above and everything after it stay as they are.")
    return "\n".join(lines) + "\n"


def _one_line(text: str) -> str:
    text = " ".join((text or "").split())
    return text[:HINT_CHARS] + ("…" if len(text) > HINT_CHARS else "")


def requests_for(code: str, hole: cv.Hole, *, hint_text: str, path: str,
                 extra: Sequence[tuple[str, str]] = (), n: int = 2,
                 temperature: float = 0.15, seed: int | None = None,
                 profile: Any = None, max_tokens: int = MAX_TOKENS
                 ) -> list[FimRequest]:
    """The candidate requests for one repair: the first at the attempt's
    sampling, the next warmer with a new seed. Each carries the prefix
    with the hint, the suffix, the context chunks and the profile's token
    order and texts."""
    prefix, suffix = split(code, hole)
    order = str(getattr(profile, "fim", "") or "psm")
    if order not in ("psm", "spm"):
        order = "psm"
    tokens = tuple(getattr(profile, "fim_tokens", ()) or ())
    out = []
    for i in range(max(1, n)):
        # the first fill at the attempt's own sampling — exactly what the
        # attempt's record says — the next warmer, with a seed of its own
        t = round(min(1.0, temperature + SECOND_STEP * i), 2)
        s = seed if i == 0 else (seed or 0) + 7919 * i + 104729 * i
        out.append(FimRequest(prefix=prefix + hint_text, suffix=suffix,
                              path=path, extra=tuple(extra),
                              max_tokens=max_tokens, temperature=t, seed=s,
                              order=order, tokens=tokens))
    return out


# --------------------------------------------------------------------------
# judging a fill
# --------------------------------------------------------------------------

def judge(code: str, hole: cv.Hole, result: FimResult, lang: str, *,
          temperature: float = 0.0, seed: int | None = None) -> Candidate:
    """Clean, splice and screen one fill, as canvas.judge does a canvas:
    a reply that ran past the hole is cut where it left the body, the
    spliced file must parse, the old lines must not come back, and no
    name may be defined twice."""
    ran_on = (result.finish_reason == "length"
              and bool((result.text or "").strip()))
    if not result.ok and not ran_on:
        return Candidate(result, None, "", result.error
                         or f"the fill ended ({result.finish_reason})",
                         temperature, seed)
    verify = None
    if lang == "python":
        def verify(prefix: list[str]) -> bool:
            if not any(ln.strip() for ln in prefix):
                return False
            text = "\n".join(prefix).rstrip("\n") + "\n"
            return cv.prescreen(cv.splice(code, hole, text), lang) == ""
    report: dict = {}
    fill = cv.clean_fill(_strip_markers(result.text or ""), hole, "open",
                         verify, report=report)
    if ran_on and not report.get("cut"):
        return Candidate(result, fill, "", "the fill ran past the end of "
                         "its budget, and no point where it left the hole "
                         "could be found", temperature, seed)
    if fill is None:
        return Candidate(result, None, "", "the fill came back empty",
                         temperature, seed)
    if hole.kind != "stub" and cv._same_code(fill, hole.original):
        return Candidate(result, fill, "", "the same lines came back",
                         temperature, seed)
    new = cv.splice(code, hole, fill)
    problem = cv.prescreen(new, lang)
    if not problem and lang == "python":
        problem = cv.duplicate_definitions(new, code)
    return Candidate(result, fill, new, problem, temperature, seed,
                     salvaged=bool(ran_on))


_TOKEN_ECHO = re.compile(
    r"<\|?(?:fim_(?:prefix|suffix|middle|pad)|file_sep|repo_name|endoftext"
    r"|im_end|im_start)\|?>|\[(?:PREFIX|SUFFIX|MIDDLE)\]|<(?:PRE|SUF|MID)>"
    r"|<｜fim▁(?:begin|hole|end)｜>")


_HINT_ECHO = re.compile(
    r"^[ \t]*(?:#|//|--|;)\s*(?:FIX: |Write the corrected )", re.M)


def _strip_markers(text: str) -> str:
    """A model that writes its own FIM tokens back wrote nothing there —
    and neither did one that echoes the hint comment (the 2026-10-27
    review: the hint was spliced into the file, "FIX: …" and all; it is
    the prompt's, never the file's). The hint's value rows follow its
    first line and are dropped with it."""
    text = _TOKEN_ECHO.sub("", text)
    out: list[str] = []
    dropping = False
    for line in text.split("\n"):
        if _HINT_ECHO.match(line):
            dropping = True
            continue
        if dropping and re.match(r"^[ \t]*(?:#|//|--|;)\s{3}\S", line):
            continue            # a value row under an echoed FIX: line
        dropping = False
        out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

def probe_request(profile: Any = None) -> FimRequest:
    order = str(getattr(profile, "fim", "") or "psm")
    return FimRequest(prefix=PROBE_PREFIX, suffix=PROBE_SUFFIX,
                      path="probe.py", max_tokens=48, temperature=0.0,
                      seed=7, order=order if order in ("psm", "spm")
                      else "psm",
                      tokens=tuple(getattr(profile, "fim_tokens", ()) or ()))


def probe_verdict(result: FimResult) -> tuple[bool, str]:
    """(usable, why): did the model fill the known hole with its answer?"""
    if result.finish_reason == "cancelled":
        return False, "cancelled"
    if result.finish_reason == "error" or (not result.ok and not
                                           (result.text or "").strip()):
        return False, result.error or f"the fill ended ({result.finish_reason})"
    text = _strip_markers(result.text or "")
    # A PREFIX COMPLETION IS NOT A FILL (the 2026-10-27 review). A server
    # that ignores the suffix continues the prefix: it writes the sum —
    # and then runs on into a `def sub` of its own, and hits the token
    # limit, because nothing told it where the hole ends. A model that
    # read the suffix stops at the end of the hole. So a reply that
    # reaches the code after the hole, or ran to the limit on a one-line
    # hole, fails the probe whatever its first line says.
    after = PROBE_SUFFIX.strip().splitlines()[0].strip()      # def sub(a, b):
    if after in text:
        return False, ("the model continued past the hole into the code "
                       "after it — it did not read the suffix, so it is "
                       "completing the prefix, not filling the middle")
    if result.finish_reason == "length":
        return False, ("the model ran to the token limit on a one-line "
                       "hole — it is completing the prefix, not filling "
                       "the middle")
    lines = list(text.replace("\r\n", "\n").split("\n"))
    # keep the body: cut where the reply dedents out of it
    body: list[str] = []
    for ln in lines:
        if ln.strip() and not ln.startswith(" ") and body:
            break
        body.append(ln)
    fill = "\n".join(body).rstrip("\n") + "\n"
    if not fill.strip():
        return False, "the model returned nothing for a known hole"
    try:
        ast.parse(PROBE_PREFIX + fill + PROBE_SUFFIX)
    except SyntaxError as exc:
        return False, (f"the model's fill of a known hole does not parse "
                       f"(line {exc.lineno}: {exc.msg})")
    if not _PROBE_ANSWER.search(fill):
        return False, (f"the model filled a known hole (`add(a, b)`) with "
                       f"something other than its sum: "
                       f"{_one_line(fill)!r}")
    if fill.count("\n") > 6:
        return False, "the model wrote far more than the hole for a known hole"
    return True, "the model filled a known hole correctly"


def describe_state(ok: bool, why: str, candidates: int) -> str:
    if ok:
        return (f"fim=on ({candidates} fill(s) per repair; {why})")
    return f"fim=off ({why})"
