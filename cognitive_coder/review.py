# SPDX-License-Identifier: Apache-2.0
"""Review: deterministic first, then ONE structured model pass, then synthesis.

THE ORDER IS THE DESIGN (§4.3, C5).

FORGE ran Security and Performance personas as a review stage. That survives,
with three corrections that each cost something to get wrong:

  1. **It runs AFTER the code builds and its tests pass**, not instead.
     Reviewing code that doesn't compile spends tokens on a moot point.
  2. **It starts with actual scanners.** `bandit`, `semgrep`, `cppcheck`,
     `gosec` — whatever is installed — plus the built-in secret and
     unsafe-API scan below, which needs nothing installed at all. The model
     is then asked only about what tools cannot see: trust boundaries, logic
     flaws, misuse potential, costs that appear at scale.
  3. **One structured pass, not two sequential ones.** On a local model two
     passes at minutes each buy nothing a schema cannot (G.5). Security and
     performance come back in one schema'd answer.

AND THE HONESTY REQUIREMENT, WHICH IS THE POINT OF THE WHOLE MODULE (M41):

**Two personas that are the same local model are not an adversarial system.**
Presenting same-model self-review as independent scrutiny is the
confident-wrongness failure mode wearing the costume of diligence. Real
adversarial value needs a different model — the host's swap button — or a
genuinely different information set. Where the perspectives came from one
model, the Recommendation Document says so in one line, near the top, where
it cannot be missed.

The Synthesizer produces the Recommendation Document, structure preserved
from FORGE §5.3: Executive Summary · Quality Assessment · Vulnerabilities &
Fixes · a Plain-English Deployment Guide pitched at the reader's stated skill
level.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from . import diagnostics as dx
from . import langs, personas
from .personas import PERSONAS, PromptBuilder, strip_think
from .providers.base import repair_json
from .types import Diagnostic

# A function longer than this is not necessarily wrong, but it is worth a
# look — and it is a fact rather than an opinion, which is why it belongs in
# the deterministic half.
LONG_FUNCTION_LINES = 40
# Nesting past this is where a small model's own code starts becoming hard
# for it to modify later.
DEEP_NESTING = 5


@dataclass(frozen=True)
class Finding:
    """One reviewed problem. `source` says who found it, which matters."""
    severity: str            # "high" | "medium" | "low" | "note"
    category: str            # "security" | "performance" | "quality"
    title: str
    detail: str
    path: str = ""
    line: int = 0
    fix: str = ""
    source: str = "built-in"   # built-in | bandit | semgrep | model | …

    @property
    def rank(self) -> int:
        return {"high": 0, "medium": 1, "low": 2, "note": 3}.get(
            self.severity, 2)

    def one_line(self) -> str:
        where = f" ({self.path}:{self.line})" if self.path and self.line \
            else (f" ({self.path})" if self.path else "")
        return f"[{self.severity}] {self.title}{where} — {self.source}"


@dataclass
class ReviewResult:
    findings: list[Finding] = field(default_factory=list)
    scanners_run: list[str] = field(default_factory=list)
    scanners_absent: list[str] = field(default_factory=list)
    model_reviewed: bool = False
    same_model: bool = True
    model_name: str = ""
    notes: list[str] = field(default_factory=list)

    def of(self, category: str) -> list[Finding]:
        return sorted((f for f in self.findings if f.category == category),
                      key=lambda f: (f.rank, f.path, f.line))

    @property
    def high(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "high"]

    def summary(self) -> str:
        if not self.findings:
            return ("nothing was flagged"
                    + (" by the scanners that are installed"
                       if self.scanners_absent else ""))
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return " · ".join(f"{n} {sev}" for sev, n in
                          sorted(counts.items(),
                                 key=lambda kv: {"high": 0, "medium": 1,
                                                 "low": 2}.get(kv[0], 3)))


# ==========================================================================
# the deterministic half — no dependencies, always runs (C5)
# ==========================================================================

# Secrets. Deliberately conservative patterns: a false positive here costs a
# minute of someone's attention, and a false negative costs a leaked key.
_SECRETS: tuple[tuple[str, str, str], ...] = (
    (r"AKIA[0-9A-Z]{16}", "AWS access key id",
     "move it to an environment variable and rotate the key — it is in the "
     "source history now"),
    (r"(?:aws.{0,20})?(?:secret|private).{0,20}['\"][0-9a-zA-Z/+]{40}['\"]",
     "AWS secret access key", "move it to an environment variable and rotate it"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
     "a private key block", "remove it from the source and rotate the key"),
    (r"\bsk-[A-Za-z0-9]{20,}", "an OpenAI-style API key",
     "read it from the environment instead, and revoke this one"),
    (r"\bsk-ant-[A-Za-z0-9\-_]{20,}", "an Anthropic API key",
     "read it from the environment instead, and revoke this one"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}", "a GitHub token",
     "read it from the environment instead, and revoke this one"),
    (r"\bxox[baprs]-[A-Za-z0-9\-]{10,}", "a Slack token",
     "read it from the environment instead, and revoke this one"),
    (r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token)\s*[:=]\s*"
     r"['\"][^'\"\s]{8,}['\"]", "a hardcoded credential",
     "read it from the environment or a secrets store"),
    (r"(?i)(?:mongodb|postgres(?:ql)?|mysql|redis|amqp)://[^\s'\"]*:[^\s'\"@]+@",
     "a connection string with an embedded password",
     "put the credentials in the environment and build the URL at runtime"),
)

# Names that are obviously placeholders. Flagging `password = "changeme"` as
# a leaked credential is how a scanner trains people to ignore it.
_PLACEHOLDER = re.compile(
    r"(?i)\b(changeme|example|placeholder|your[-_]?key|xxx+|todo|dummy|"
    r"fake|sample|test[-_]?key|redacted|\.\.\.)\b")


def scan_secrets(text: str, path: str = "") -> list[Finding]:
    """Hardcoded credentials. Needs nothing installed."""
    out: list[Finding] = []
    for pattern, what, fix in _SECRETS:
        for m in re.finditer(pattern, text):
            snippet = m.group(0)
            if _PLACEHOLDER.search(snippet):
                continue
            line = text[:m.start()].count("\n") + 1
            out.append(Finding(
                severity="high", category="security",
                title=f"{what} appears in the source",
                detail=f"Found `{_mask(snippet)}` at line {line}.",
                path=path, line=line, fix=fix))
    return out


def _mask(secret: str) -> str:
    """Show enough to find it, not enough to use it.

    M44's spirit: a key must not be reproduced in a report that will be
    pasted into a ticket, an email, or a chat window.
    """
    text = secret.strip()
    if len(text) <= 12:
        return text[:4] + "…"
    return f"{text[:6]}…{text[-2:]} ({len(text)} chars)"


def scan_python(text: str, path: str = "") -> list[Finding]:
    """The checks Python's own parser can make exactly.

    Everything here is a FACT about the code rather than an opinion about it,
    which is why the model is never asked any of these questions.
    """
    out: list[Finding] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        # eval/exec on anything that is not a literal
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("eval", "exec", "compile"):
            arg = node.args[0] if node.args else None
            literal = isinstance(arg, ast.Constant)
            if not literal:
                out.append(Finding(
                    severity="high", category="security",
                    title=f"{node.func.id}() on a value that is not a literal",
                    detail=("Whatever reaches this call is executed. If any "
                            "part of it can come from outside the program, "
                            "that is arbitrary code execution."),
                    path=path, line=node.lineno,
                    fix="use a lookup table, `ast.literal_eval`, or an "
                        "explicit parser instead"))

        # empty exception handlers
        if isinstance(node, ast.ExceptHandler):
            body = [n for n in node.body
                    if not isinstance(n, (ast.Pass, ast.Expr))]
            if not body:
                bare = node.type is None
                out.append(Finding(
                    severity="medium" if bare else "low",
                    category="quality",
                    title=("a bare `except:` that swallows the error" if bare
                           else "an exception handler that does nothing"),
                    detail=("A failure here disappears silently, so the "
                            "first sign of it will be wrong behaviour "
                            "somewhere else."),
                    path=path, line=node.lineno,
                    fix="log it, re-raise it, or state in a comment why "
                        "ignoring it is correct"))

        # path traversal: joining user input into a path without containment
        if isinstance(node, ast.Call) and _is_attr(node.func, "join") \
                and any(_looks_like_input(a) for a in node.args):
            out.append(Finding(
                severity="medium", category="security",
                title="a path is built from input without a containment check",
                detail=("A value containing `..` walks out of the directory "
                        "you meant. This is the same failure the engine's own "
                        "project-root jail exists to prevent."),
                path=path, line=node.lineno,
                fix="resolve the final path and check it is inside the "
                    "directory you intended, then use it"))

        # long functions and deep nesting
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno
            if length > LONG_FUNCTION_LINES:
                out.append(Finding(
                    severity="low", category="quality",
                    title=f"`{node.name}` is {length} lines long",
                    detail=(f"Past about {LONG_FUNCTION_LINES} lines a "
                            f"function becomes hard for a small model to "
                            f"modify without breaking something it did not "
                            f"mean to touch."),
                    path=path, line=node.lineno,
                    fix="split out the part that has its own name"))
            depth = _max_depth(node)
            if depth >= DEEP_NESTING:
                out.append(Finding(
                    severity="low", category="quality",
                    title=f"`{node.name}` nests {depth} levels deep",
                    detail="Each level is another condition to hold in mind.",
                    path=path, line=node.lineno,
                    fix="return early, or extract the inner block"))

    out.extend(_scan_python_performance(tree, text, path))
    return out


def _scan_python_performance(tree: ast.AST, text: str,
                             path: str) -> list[Finding]:
    """Measurable facts about cost, not opinions about speed (§4.3).

    Quadratic patterns and per-iteration allocation are the two that are
    genuinely findable by pattern. Anything subtler is what the model is for.
    """
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        for inner in ast.walk(node):
            # string concatenation in a loop — O(n²) by construction
            if isinstance(inner, ast.AugAssign) and \
                    isinstance(inner.op, ast.Add) and \
                    _probably_string(inner.value):
                out.append(Finding(
                    severity="low", category="performance",
                    title="a string is built by repeated concatenation "
                          "inside a loop",
                    detail=("Each `+=` copies the whole string, so the cost "
                            "grows with the square of the length. It is "
                            "invisible on ten items and painful on ten "
                            "thousand."),
                    path=path, line=inner.lineno,
                    fix="collect the pieces in a list and `''.join(...)` "
                        "once at the end"))
            # a membership test against a list inside a loop
            if isinstance(inner, ast.Compare) and \
                    any(isinstance(op, ast.In) for op in inner.ops) and \
                    any(isinstance(c, (ast.List, ast.Tuple))
                        for c in inner.comparators):
                out.append(Finding(
                    severity="low", category="performance",
                    title="a membership test against a list inside a loop",
                    detail=("`x in [...]` scans the whole list every time; "
                            "`x in {...}` does not."),
                    path=path, line=inner.lineno,
                    fix="use a set for the membership test"))
    return out


def scan_generic(text: str, lang_id: str = "", path: str = "") -> list[Finding]:
    """The language-agnostic half: TODOs left in "finished" code, and the
    unsafe-API patterns the static screen warns about."""
    out: list[Finding] = []
    lang = langs.get(lang_id)
    comment = lang.comment if lang else "#"
    for m in re.finditer(r"(?i)\b(TODO|FIXME|XXX|HACK)\b[:\s]*(.{0,80})",
                         text):
        line = text[:m.start()].count("\n") + 1
        out.append(Finding(
            severity="note", category="quality",
            title=f"{m.group(1).upper()} left in code being reported as done",
            detail=(m.group(2).strip() or "no detail given") + ".",
            path=path, line=line,
            fix=f"finish it, or move it to an issue and delete the "
                f"{comment} marker"))
    return out


def scan_untested(text: str, test_source: str, lang_id: str = "",
                  path: str = "") -> list[Finding]:
    """Public functions with no mention in the tests (§6.10).

    Deliberately crude — a name appearing anywhere in the test file counts.
    A stricter check would produce false alarms on parametrised tests, and a
    review that cries wolf is a review nobody reads to the end of.
    """
    from . import context as ctx
    if not test_source.strip():
        return []
    out: list[Finding] = []
    for sym in ctx.symbols(text, lang_id or "python"):
        short = sym.name.split(".")[-1]
        if short.startswith("_") or sym.kind not in ("function", "method"):
            continue
        if re.search(rf"\b{re.escape(short)}\b", test_source):
            continue
        out.append(Finding(
            severity="low", category="quality",
            title=f"`{sym.name}` is public and is not mentioned in the tests",
            detail="Nothing would notice if it stopped working.",
            path=path, line=sym.line,
            fix=f"add a test that calls `{short}` with a real input"))
    return out


# ==========================================================================
# external scanners — used when present, named when absent (C7)
# ==========================================================================

#: name → (argv template, languages, what its absence costs)
SCANNERS: dict[str, tuple[list[str], tuple[str, ...], str]] = {
    "bandit": (["{tool}", "-f", "json", "-q", "{src}"], ("python",),
               "Python security scanning is limited to the built-in checks"),
    "semgrep": (["{tool}", "--json", "--quiet", "--config=auto", "{src}"],
                (), "cross-language rule scanning is unavailable"),
    "cppcheck": (["{tool}", "--enable=warning,performance,portability",
                  "--quiet", "--template={file}:{line}:{severity}:{message}",
                  "{src}"], ("c", "cpp"),
                 "C and C++ static analysis is unavailable"),
    "gosec": (["{tool}", "-fmt=json", "-quiet", "{src}"], ("go",),
              "Go security scanning is unavailable"),
    "shellcheck": (["{tool}", "-f", "json", "{src}"], ("bash",),
                   "shell script analysis is unavailable"),
}


def run_scanners(path: str, lang_id: str, *, fs: Any, ex: Any,
                 workdir: str = "") -> tuple[list[Finding], list[str],
                                             list[str]]:
    """Run whatever is installed. Returns (findings, ran, absent)."""
    import json

    root = workdir or fs.root()
    findings: list[Finding] = []
    ran: list[str] = []
    absent: list[str] = []

    for name, (template, langs_for, cost) in SCANNERS.items():
        if langs_for and lang_id not in langs_for:
            continue
        tool = ex.which(name)
        if not tool:
            absent.append(f"{name} — {cost}")
            continue
        argv = [p.format(tool=tool, src=_join(root, path)) for p in template]
        try:
            proc = ex.run(argv, cwd=root, timeout=120)
        except Exception:                                # noqa: BLE001
            absent.append(f"{name} — it is installed but would not run")
            continue
        ran.append(name)
        text = proc.output
        if name == "cppcheck":
            findings.extend(_from_cppcheck(text, path))
            continue
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            # A scanner that produced nothing parseable has still told us
            # something, and dropping it silently is M29's mistake in a
            # different module.
            if proc.exit_code not in (0, 1) and text.strip():
                findings.append(Finding(
                    severity="note", category="quality",
                    title=f"{name} ran but its output could not be read",
                    detail=text.strip()[-300:], path=path, source=name))
            continue
        findings.extend(_from_json(name, data, path))
    return findings, ran, absent


def _from_json(name: str, data: Any, path: str) -> list[Finding]:
    out: list[Finding] = []
    rows: Sequence[Any] = ()
    if name == "bandit":
        rows = data.get("results", []) if isinstance(data, dict) else ()
        for r in rows:
            sev = str(r.get("issue_severity", "MEDIUM")).lower()
            out.append(Finding(
                severity={"high": "high", "medium": "medium"}.get(sev, "low"),
                category="security", title=str(r.get("issue_text", ""))[:120],
                detail=f"{r.get('test_id', '')}: "
                       f"{r.get('issue_text', '')}".strip(": "),
                path=str(r.get("filename", path)),
                line=int(r.get("line_number", 0) or 0),
                fix=str(r.get("more_info", "")), source="bandit"))
    elif name == "semgrep":
        rows = data.get("results", []) if isinstance(data, dict) else ()
        for r in rows:
            extra = r.get("extra", {}) or {}
            sev = str(extra.get("severity", "WARNING")).lower()
            out.append(Finding(
                severity={"error": "high", "warning": "medium"}.get(sev,
                                                                    "low"),
                category="security", title=str(extra.get("message", ""))[:120],
                detail=str(extra.get("message", "")),
                path=str(r.get("path", path)),
                line=int((r.get("start", {}) or {}).get("line", 0) or 0),
                fix=str((extra.get("metadata", {}) or {}).get("fix", "")),
                source="semgrep"))
    elif name == "gosec":
        rows = data.get("Issues", []) if isinstance(data, dict) else ()
        for r in rows:
            sev = str(r.get("severity", "MEDIUM")).lower()
            out.append(Finding(
                severity={"high": "high", "medium": "medium"}.get(sev, "low"),
                category="security", title=str(r.get("details", ""))[:120],
                detail=str(r.get("details", "")),
                path=str(r.get("file", path)),
                line=int(str(r.get("line", "0")).split("-")[0] or 0),
                source="gosec"))
    elif name == "shellcheck" and isinstance(data, list):
        for r in data:
            out.append(Finding(
                severity={"error": "high", "warning": "medium"}.get(
                    str(r.get("level", "")).lower(), "low"),
                category="quality", title=str(r.get("message", ""))[:120],
                detail=f"SC{r.get('code', '')}: {r.get('message', '')}",
                path=str(r.get("file", path)),
                line=int(r.get("line", 0) or 0), source="shellcheck"))
    return out


def _from_cppcheck(text: str, path: str) -> list[Finding]:
    out: list[Finding] = []
    for d in dx.parse(text, "cpp"):
        if d.code == "unparsed":
            continue
        out.append(Finding(
            severity={"error": "high", "warning": "medium"}.get(d.severity,
                                                                "low"),
            category="performance" if d.severity == "performance"
            else "security" if d.severity == "error" else "quality",
            title=d.message[:120], detail=d.message,
            path=d.file or path, line=d.line, source="cppcheck"))
    return out


# ==========================================================================
# the model pass — ONE call, schema'd, covering both perspectives (G.5)
# ==========================================================================

REVIEW_SCHEMA = """{
  "security": [
    {"severity": "high|medium|low",
     "title": "one line",
     "detail": "what could go wrong, and under what circumstances",
     "line": 0,
     "fix": "what to do instead"}
  ],
  "performance": [
    {"severity": "high|medium|low",
     "title": "one line",
     "detail": "what costs what, and at what scale it starts to matter",
     "line": 0,
     "fix": "what to do instead"}
  ],
  "overall": "one or two sentences"
}"""

# GBNF is used where the provider supports it. Constrained decoding beats
# repairing near-JSON afterwards, every time (D9).
REVIEW_GRAMMAR = r'''
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws     ::= [ \t\n]*
'''


def model_review(code: str, path: str, *, llm: Any, prompts: PromptBuilder,
                 deterministic: Sequence[Finding] = (),
                 lang_id: str = "python",
                 temperature: float = 0.35) -> tuple[list[Finding], str, bool]:
    """One call, both perspectives, schema'd. Returns (findings, overall, ok).

    The model is told what the scanners ALREADY found and asked not to repeat
    it. That is not politeness — it is the whole reason this stage is worth
    running. A model that spends its answer restating `bandit`'s output has
    added nothing, and the questions only it can answer go unasked.
    """
    caps = _capabilities(llm)
    if not caps.loaded:
        return [], "", False

    already = "\n".join(f"  - {f.one_line()}" for f in deterministic[:12])
    task = "\n".join([
        f"Review `{path}`. It already builds and its tests pass, so this is "
        f"not about whether it works.",
        "",
        "Answer only the questions TOOLS CANNOT: where trust boundaries sit, "
        "which inputs are attacker-controlled, what happens at a thousand "
        "times the expected scale, and what could be MISUSED even though it "
        "is correct.",
        "",
        ("Automated scanners already found the following. Do not repeat "
         f"them:\n{already}" if already else
         "Automated scanners found nothing, which is not the same as there "
         "being nothing."),
        "",
        "THE CODE:",
        _numbered(code),
    ])

    prompt = prompts.build(
        PERSONAS["reviewer"], task,
        contract=("OUTPUT CONTRACT\nReturn one JSON object in exactly this "
                  f"shape and nothing else:\n{REVIEW_SCHEMA}\n"
                  "Use an empty list where you have nothing to report. An "
                  "empty list stated confidently is more useful than a "
                  "manufactured concern."))

    try:
        completion = llm.complete(
            prompt.messages(), temperature=temperature, max_tokens=1400,
            grammar=REVIEW_GRAMMAR if caps.supports_grammar else None)
    except Exception:                                    # noqa: BLE001
        return [], "", False

    text = strip_think(completion.text)
    data, repaired = repair_json(text)
    if not data:
        return [], "", False

    findings: list[Finding] = []
    for category in ("security", "performance"):
        for row in data.get(category, []) or []:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            findings.append(Finding(
                severity=str(row.get("severity", "low")).lower(),
                category=category, title=title[:160],
                detail=str(row.get("detail", "")),
                path=path, line=int(row.get("line", 0) or 0),
                fix=str(row.get("fix", "")), source="model"))
    overall = str(data.get("overall", "")).strip()
    if repaired:
        overall += ("  (The model's answer needed repairing before it could "
                    "be read, which usually means grammar-constrained "
                    "decoding is available and switched off.)")
    return findings, overall, True


def _numbered(code: str, limit: int = 400) -> str:
    lines = code.splitlines()[:limit]
    body = "\n".join(f"{n:>4} | {line}" for n, line in enumerate(lines, 1))
    if len(code.splitlines()) > limit:
        body += f"\n… {len(code.splitlines()) - limit} further lines not shown."
    return body


# ==========================================================================
# the whole stage
# ==========================================================================

def review(code: str, path: str, *, lang_id: str = "python",
           fs: Any = None, ex: Any = None, llm: Any = None,
           prompts: PromptBuilder | None = None, test_source: str = "",
           use_model: bool = True, workdir: str = "") -> ReviewResult:
    """Deterministic → scanners → one model pass. In that order (§6.10)."""
    result = ReviewResult()

    result.findings.extend(scan_secrets(code, path))
    result.findings.extend(scan_generic(code, lang_id, path))
    if lang_id == "python":
        result.findings.extend(scan_python(code, path))
    if test_source:
        result.findings.extend(scan_untested(code, test_source, lang_id, path))

    if fs is not None and ex is not None:
        found, ran, absent = run_scanners(path, lang_id, fs=fs, ex=ex,
                                          workdir=workdir)
        result.findings.extend(found)
        result.scanners_run = ran
        result.scanners_absent = absent

    if use_model and llm is not None:
        caps = _capabilities(llm)
        result.model_name = caps.name
        model_findings, overall, ok = model_review(
            code, path, llm=llm, prompts=prompts or PromptBuilder(),
            deterministic=list(result.findings), lang_id=lang_id)
        result.model_reviewed = ok
        if ok:
            result.findings.extend(model_findings)
            if overall:
                result.notes.append(overall)
        else:
            result.notes.append(
                "The model review did not produce a usable answer, so only "
                "the deterministic checks below ran.")
        # M41: two personas from one model are one opinion expressed twice.
        result.same_model = True
    return result


def _capabilities(llm: Any):
    from .types import ModelCapabilities
    try:
        return llm.capabilities()
    except Exception:                                    # noqa: BLE001
        return ModelCapabilities(name="", family="unknown", context_tokens=0)


# ==========================================================================
# the Synthesizer — the Recommendation Document (FORGE §5.3)
# ==========================================================================

def recommendation_document(result: ReviewResult, *, request: str = "",
                            files: Sequence[str] = (),
                            skill_level: str = "intermediate",
                            build_summary: str = "",
                            caveats: Sequence[str] = ()) -> str:
    """The deliverable. Four sections, and the honesty line near the top.

    Pitched at `skill_level` because a deployment guide that assumes
    knowledge the reader does not have is a guide they cannot follow, and one
    that explains what they already know is one they stop reading.
    """
    skill = skill_level if skill_level in personas.SKILL_LEVELS \
        else "intermediate"
    lines: list[str] = ["# Recommendation Document", ""]

    if request:
        lines += [f"**What was asked for:** {request}", ""]
    if files:
        lines += [f"**Files reviewed:** {', '.join(files)}", ""]

    # M41 — one line, near the top, where it cannot be missed.
    if result.model_reviewed and result.same_model:
        lines += [
            "> **These perspectives are not independent.** The security and "
            "performance findings below came from the same model"
            + (f" (`{result.model_name}`)" if result.model_name else "")
            + ", so they are one reviewer's opinion expressed twice rather "
              "than two reviewers agreeing. Treat agreement between them as "
              "no evidence at all.",
            ""]

    # -- 1. Executive Summary -------------------------------------------
    lines += ["## Executive summary", ""]
    high = result.high
    if high:
        lines.append(
            f"**{len(high)} thing{'s' * (len(high) != 1)} to deal with "
            f"before this goes anywhere:**")
        lines.append("")
        for f in high:
            lines.append(f"- {f.title}"
                         + (f" — `{f.path}:{f.line}`" if f.line else ""))
        lines.append("")
    else:
        lines += ["Nothing was found that should stop this being used.", ""]

    lines.append(f"Overall: {result.summary()}.")
    if build_summary:
        lines.append(f"Verification: {build_summary}")
    for note in result.notes:
        lines.append(note)
    if caveats:
        lines.append("")
        lines.append("**Caveats on the evidence:**")
        for c in caveats:
            lines.append(f"- {c}")
    lines.append("")

    # -- 2. Quality assessment ------------------------------------------
    lines += ["## Quality assessment", ""]
    quality = result.of("quality")
    if quality:
        for f in quality:
            lines.append(f"- **{f.title}**"
                         + (f" (`{f.path}:{f.line}`)" if f.line else ""))
            if f.detail:
                lines.append(f"  {f.detail}")
            if f.fix:
                lines.append(f"  *Suggested:* {f.fix}")
    else:
        lines.append("No structural problems were flagged.")
    lines.append("")
    lines += _scanner_coverage(result)

    # -- 3. Vulnerabilities & fixes -------------------------------------
    lines += ["## Vulnerabilities and fixes", ""]
    security = result.of("security")
    performance = result.of("performance")
    if not security:
        lines += ["No security findings.", ""]
    for f in security:
        lines.append(f"### {f.title}")
        lines.append("")
        lines.append(f"*{f.severity}* · found by {f.source}"
                     + (f" · `{f.path}:{f.line}`" if f.line else ""))
        lines.append("")
        if f.detail:
            lines.append(f.detail)
            lines.append("")
        if f.fix:
            lines.append(f"**Fix:** {f.fix}")
            lines.append("")

    if performance:
        lines += ["### Performance", ""]
        for f in performance:
            lines.append(f"- **{f.title}**"
                         + (f" (`{f.path}:{f.line}`)" if f.line else ""))
            if f.detail:
                lines.append(f"  {f.detail}")
            if f.fix:
                lines.append(f"  *Suggested:* {f.fix}")
        lines.append("")

    # -- 4. Plain-English deployment guide -------------------------------
    lines += ["## Deployment guide", ""]
    lines += _deployment_guide(result, skill, files)
    return "\n".join(lines)


def _scanner_coverage(result: ReviewResult) -> list[str]:
    """What was and was not checked. C7's obligation, in the document.

    "Nothing was flagged" means something very different depending on which
    scanners were installed, and the reader deserves to know which sentence
    they are reading.
    """
    lines: list[str] = []
    if result.scanners_run:
        lines.append(f"*Scanners run:* {', '.join(result.scanners_run)}.")
    if result.scanners_absent:
        lines.append("")
        lines.append("*Not checked, because these are not installed:*")
        for note in result.scanners_absent:
            lines.append(f"- {note}")
    if not result.scanners_run and not result.scanners_absent:
        lines.append("*Only the built-in checks ran; no external scanner was "
                     "available.*")
    lines.append("")
    return lines


_DEPLOY_BY_SKILL = {
    "novice": [
        "1. **Back up what you have.** Copy the folder somewhere else "
        "first. Everything below is reversible, but a copy costs nothing.",
        "2. **Read the summary above.** Anything marked *high* should be "
        "fixed before anyone else uses this. The rest can wait.",
        "3. **Run the tests yourself once.** Do not take anyone's word for "
        "it, including this document's.",
        "4. **Try it on a small, real example** — not the test data. Test "
        "data is chosen to pass.",
        "5. **Keep the secrets out.** If anything above mentioned a key or "
        "a password in the source, that key should be treated as public "
        "from now on and replaced.",
    ],
    "intermediate": [
        "1. Fix anything marked *high* first; the rest can be scheduled.",
        "2. Run the full suite, not just the tests for the changed files — "
        "the blast radius of a signature change is wider than it looks.",
        "3. Check the caveats above before trusting a green run. A suite "
        "that collected zero tests passes.",
        "4. Move any credential named above into the environment and "
        "rotate it; it is in the file history now regardless of what the "
        "current file says.",
        "5. Deploy behind whatever gate you normally use, and watch the "
        "first real workload rather than the first synthetic one.",
    ],
    "senior": [
        "1. High-severity findings first; everything else is backlog.",
        "2. Verify the trust-boundary claims yourself — the model's "
        "reasoning about attacker-controlled input is the least reliable "
        "part of this document and the part most worth checking.",
        "3. Rotate anything flagged as a credential.",
        "4. The performance findings are pattern-matched, not measured. "
        "Profile before acting on any of them.",
    ],
}


def _deployment_guide(result: ReviewResult, skill: str,
                      files: Sequence[str]) -> list[str]:
    lines = list(_DEPLOY_BY_SKILL[skill])
    if any(f.category == "security" and f.severity == "high"
           for f in result.findings):
        lines.insert(0, "**Do not deploy this yet.** There is at least one "
                        "high-severity security finding above.")
        lines.insert(1, "")
    lines.append("")
    lines.append("*This document reports what was checked and by what. It is "
                 "not a guarantee that nothing else is wrong — no review, "
                 "human or otherwise, can be that.*")
    return lines


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _is_attr(node: Any, name: str) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == name


def _looks_like_input(node: Any) -> bool:
    """A crude "did this come from outside" test.

    Crude on purpose: the alternative is real taint analysis, which is a
    different project. Over-reporting slightly is the safe direction for a
    *low*-severity note that a human then reads.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and re.search(
                r"(?i)request|input|arg|param|user|query|payload|body|"
                r"filename|path_from|untrusted", sub.id):
            return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id == "input":
            return True
    return False


def _probably_string(node: Any) -> bool:
    """Is this expression a string?

    Recurses through `BinOp`, because `out += str(row) + "\\n"` is the form
    this actually appears in far more often than a bare literal — and a check
    that only catches `out += "!"` catches the toy case and misses the real
    one. Also recognises the string-producing builtins and `.format`/`.join`,
    since a small model reaches for those constantly.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.JoinedStr):                  # an f-string
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _probably_string(node.left) or _probably_string(node.right)
    if isinstance(node, ast.Call):
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id in ("str", "repr", "format",
                                                  "chr"):
            return True
        if isinstance(fn, ast.Attribute) and fn.attr in ("format", "join",
                                                         "strip", "upper",
                                                         "lower", "replace",
                                                         "decode"):
            return True
    return False


def _max_depth(node: Any, depth: int = 0) -> int:
    nesting = (ast.If, ast.For, ast.While, ast.With, ast.Try)
    best = depth
    for child in ast.iter_child_nodes(node):
        step = depth + 1 if isinstance(child, nesting) else depth
        best = max(best, _max_depth(child, step))
    return best


def _join(root: str, rel: str) -> str:
    if not root:
        return rel
    sep = "\\" if ("\\" in root and "/" not in root) else "/"
    return root.rstrip("/\\") + sep + str(rel).lstrip("/\\")


def as_diagnostics(result: ReviewResult) -> list[Diagnostic]:
    """Findings as Diagnostics, so a host can render them in one list."""
    return [Diagnostic(
        file=f.path, line=f.line,
        severity={"high": "error", "medium": "warning"}.get(f.severity,
                                                            "note"),
        message=f.title, code=f.category, tool=f.source,
        source_excerpt=f.detail) for f in sorted(
            result.findings, key=lambda f: (f.rank, f.path, f.line))]
