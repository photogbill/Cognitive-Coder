# SPDX-License-Identifier: Apache-2.0
"""Look at a folder and say what is wrong with it, and what could be better.

`ccoder build` makes something from a request. This is the other half of
the loop: point it at a project that already exists — one it built, or one
somebody wrote by hand — and it reports what it finds, ranked, with file
and line, and writes a plan `ccoder build --spec` can carry out. Run it
again afterwards and it says what was fixed, what remains and what is new.

    ccoder audit game/                      # report + plan
    ccoder audit game/ --focus "collisions feel wrong"
    ccoder build -p game/ --spec game/.cc_state/audit/plan.md

The same order as everything else in this engine: deterministic first,
model second, human last.

  1. INVENTORY — which files are source, in which language, and what was
     left out and why (skip list, size cap, file cap). Said, never silent.
  2. DETERMINISTIC — the review stage's scanners on every file, names the
     project does not define, functions nothing calls, modules no test
     touches.
  3. VERIFICATION — does it parse or compile, and do its TESTS run as it
     stands. The program itself is never run: an audit must not start
     somebody's game, server or deletion script to find out what it does.
  4. MODEL PASS — one bounded completion per file for the files most worth
     reading, with a byte-identical cached prefix across all of them (M52),
     read as the LAST object carrying `findings` (a reasoning model restates
     the schema before answering). Its findings are labelled as the
     model's and never merged into the tools' as facts.
  5. REPORT AND PLAN — the document leads with what was NOT reviewed; the
     plan lists the findings as work items on the files they concern.

WHAT IT NEVER DOES. It writes nothing outside `.cc_state/` — the report,
plan and memory under `.cc_state/audit/`, the codemap index beside them —
never a source file, and it never asks `approve_diff` anything, because it
has nothing to approve. Changing code is `build`'s job, through the
transaction, the snapshot and the approval gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import fnmatch
import hashlib
import json
import re
import time
from typing import Any

from . import diagnostics as dx
from . import langs, runner
from . import review as review_mod
from .codemap import SKIP_DIRS, CodeMap
from .errors import Cancelled
from .personas import PERSONAS, PromptBuilder, split_think
from .planner import _looks_like_test
from .ports import NeverCancelled
from .providers.base import find_json_object

AUDIT_DIR = ".cc_state/audit"
REPORT_PATH = f"{AUDIT_DIR}/report.md"
PLAN_PATH = f"{AUDIT_DIR}/plan.md"
MEMORY_PATH = f"{AUDIT_DIR}/last.json"
JOURNAL_DIR = f"{AUDIT_DIR}/journal"

SEVERITIES = ("high", "medium", "low")
KINDS = ("bug", "risk", "smell", "test", "doc")
_RANK = {"high": 0, "medium": 1, "low": 2, "note": 3}

#: Names a project reaches without calling them in its own code: entry
#: points, test hooks, framework callbacks. Never reported as unused.
_ENTRY_NAMES = frozenset({
    "main", "run", "app", "cli", "setup", "teardown", "setUp", "tearDown",
    "setUpClass", "tearDownClass", "handler", "lambda_handler", "index",
    "application", "create_app", "_ready", "_process", "_physics_process",
    "_init", "_input", "_draw", "update", "draw", "render"})

AUDIT_CONTRACT = (
    'Answer with ONE JSON object and nothing after it:\n'
    '{"findings": [{"line": 12, "severity": "high|medium|low", '
    '"kind": "bug|risk|smell|test|doc", "title": "one line", '
    '"detail": "why it matters", "change": "what to change"}]}\n'
    'Use {"findings": []} if you looked and found nothing worth changing. '
    'Do not repeat what the tools already found.')


@dataclass(frozen=True)
class AuditConfig:
    """What to look at, and how hard."""

    #: Glob filters on project-relative paths; empty means every source file.
    paths: tuple[str, ...] = ()
    #: What the operator wants looked at in particular — "the collision
    #: code", "make it faster". Passed to the model pass and into the plan.
    focus: str = ""
    max_files: int = 60
    model_pass: bool = True
    max_model_files: int = 12
    run_tests: bool = True
    test_timeout: float = 300.0
    max_file_bytes: int = 200_000
    reply_tokens: int = 2048


@dataclass(frozen=True)
class AuditFinding:
    """One thing found. `source` says who found it, because that matters."""

    path: str
    line: int
    severity: str               # high | medium | low | note
    kind: str                   # bug | risk | smell | test | doc | …
    title: str
    detail: str = ""
    change: str = ""
    source: str = "tools"       # tools | <scanner name> | model

    @property
    def rank(self) -> int:
        return _RANK.get(self.severity, 2)

    @property
    def by_model(self) -> bool:
        return self.source == "model"

    def fingerprint(self) -> str:
        """Stable across runs: file, kind and title — NOT the line.

        A finding does not become a new one because an edit above it moved
        it down six lines.
        """
        title = re.sub(r"\s+", " ", self.title.lower()).strip()
        title = re.sub(r"\d+", "#", title)
        raw = f"{self.path}|{self.kind}|{title}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def one_line(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.severity}] {self.title} — `{where}`"


@dataclass(frozen=True)
class AuditReport:
    """Everything the audit found, and everything it could not look at."""

    root: str
    files: tuple[tuple[str, str], ...] = ()          # (path, language)
    skipped: tuple[tuple[str, str], ...] = ()        # (path, reason)
    findings: tuple[AuditFinding, ...] = ()
    verification: tuple[str, ...] = ()               # one sentence each
    tests_ran: bool | None = None
    not_checked: tuple[str, ...] = ()                # languages, with why
    scanners_run: tuple[str, ...] = ()
    scanners_absent: tuple[str, ...] = ()
    scanners_failed: tuple[str, ...] = ()
    model_name: str = ""
    model_files: tuple[str, ...] = ()
    model_notes: tuple[str, ...] = ()
    model_skipped: str = ""
    focus: str = ""
    since_last: str = ""
    seconds: float = 0.0

    @property
    def high(self) -> list[AuditFinding]:
        return [f for f in self.findings if f.severity == "high"]

    def document(self) -> str:
        return _document(self)

    def to_spec(self) -> str:
        return _spec(self)


# --------------------------------------------------------------------------
# 1. inventory
# --------------------------------------------------------------------------

def _inventory(fs: Any, config: AuditConfig
               ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """([(path, language, text)], [(path, why it was left out)])."""
    kept: list[tuple[str, str, str]] = []
    skipped: list[tuple[str, str]] = []
    try:
        paths = sorted(str(p).replace("\\", "/") for p in fs.list("*"))
    except Exception as exc:                             # noqa: BLE001
        return [], [(".", f"the folder could not be listed ({exc})")]
    for path in paths:
        parts = path.split("/")
        if any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
            continue
        lang = langs.id_for_path(path)
        if not lang:
            continue
        if config.paths and not any(fnmatch.fnmatch(path, g)
                                    for g in config.paths):
            continue
        try:
            raw = fs.read_bytes(path)
        except Exception:                                # noqa: BLE001
            skipped.append((path, "it could not be read"))
            continue
        if len(raw) > config.max_file_bytes:
            skipped.append((path, f"it is {len(raw) // 1024:,} KB, over "
                                  f"the {config.max_file_bytes // 1024:,} KB "
                                  f"limit for one file"))
            continue
        if b"\x00" in raw[:4096]:
            continue                                     # binary
        kept.append((path, lang, raw.decode("utf-8", errors="replace")))
    if len(kept) > config.max_files:
        # Sources before tests, then the biggest: the most code for the
        # attention spent. What falls off is NAMED in the report.
        kept.sort(key=lambda f: (_looks_like_test(f[0]), -len(f[2])))
        for path, _lang, _text in kept[config.max_files:]:
            skipped.append((path, f"only {config.max_files} files are "
                                  f"reviewed per audit (--max-files)"))
        kept = sorted(kept[:config.max_files])
    return kept, skipped


# --------------------------------------------------------------------------
# 2. deterministic
# --------------------------------------------------------------------------

def _from_review(f: Any) -> AuditFinding:
    kind = {"security": "risk", "performance": "smell",
            "quality": "smell"}.get(f.category, f.category or "smell")
    return AuditFinding(path=f.path, line=f.line or 0, severity=f.severity,
                        kind=kind, title=f.title, detail=f.detail,
                        change=f.fix, source=f.source or "tools")


def _words(texts: Sequence[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        counts.update(re.findall(r"[A-Za-z_]\w*", text))
    return counts


def _possibly_unused(codemap: CodeMap, path: str, text: str,
                     words: Counter) -> list[AuditFinding]:
    """Functions and classes nothing reaches. Said as "possibly", always.

    Two independent signals must agree: the call graph has no caller, AND
    the name appears nowhere else in the project's text. The second catches
    what the graph cannot see — a callback passed by name, a string in
    `__all__`, a route table. A decorated definition is skipped: a
    decorator usually IS the caller (a route, a fixture, a command).
    """
    if _looks_like_test(path):
        return []
    lines = text.splitlines()
    out = []
    for sym in codemap.store.symbols_in(path):
        if sym.get("kind") not in ("function", "method", "class"):
            continue
        short = str(sym["name"]).rsplit(".", 1)[-1]
        if (short.startswith("__") or short.startswith("test")
                or short in _ENTRY_NAMES):
            continue
        line = int(sym.get("line") or 0)
        before = lines[line - 2].strip() if 1 < line <= len(lines) else ""
        if before.startswith("@"):
            continue
        if words.get(short, 0) > 1:
            continue
        try:
            if codemap.store.callers_of(str(sym["name"]), depth=1):
                continue
        except Exception:                                # noqa: BLE001
            continue
        out.append(AuditFinding(
            path=path, line=line, severity="low", kind="smell",
            title=f"`{short}` is possibly unused",
            detail="Nothing in the project calls it or mentions its name. "
                   "It may be reached some way the tools cannot see — by "
                   "reflection, a plugin loader, or from outside the "
                   "project.",
            change=f"Remove `{short}` if it is really unused, or add the "
                   f"call or test that uses it."))
    return out


def _deterministic(host: Any, codemap: CodeMap,
                   files: Sequence[tuple[str, str, str]]
                   ) -> tuple[list[AuditFinding], dict]:
    """The tools' findings, and what the scanners did."""
    found: list[AuditFinding] = []
    scanners = {"run": [], "absent": [], "failed": []}
    tests = [(p, t) for p, _l, t in files if _looks_like_test(p)]
    test_text = "\n".join(t for _p, t in tests)
    words = _words(t for _p, _l, t in files)
    for path, lang, text in files:
        result = review_mod.review(
            text, path, lang_id=lang, fs=host.fs, ex=host.exec,
            test_source=(test_text if tests and not _looks_like_test(path)
                         else ""),
            use_model=False)
        found.extend(_from_review(f) for f in result.findings)
        for key, names in (("run", result.scanners_run),
                           ("absent", result.scanners_absent),
                           ("failed", result.scanners_failed)):
            for name in names:
                if name not in scanners[key]:
                    scanners[key].append(name)
        try:
            missing = codemap.unresolved_in(text, lang)
        except Exception:                                # noqa: BLE001
            missing = []
        if missing:
            names = ", ".join(f"`{m}`" for m in missing[:6])
            first_use = re.search(rf"\b{re.escape(missing[0])}\b", text)
            found.append(AuditFinding(
                path=path,
                line=(text.count("\n", 0, first_use.start()) + 1
                      if first_use else 0),
                severity="medium", kind="bug",
                title=f"refers to names this project does not define: "
                      f"{names}",
                detail="Each will fail when that line runs, unless it is "
                       "defined somewhere the tools cannot see.",
                change="Define or import each name, or remove the use."))
        found.extend(_possibly_unused(codemap, path, text, words))
    found.extend(_untested_modules(files, tests))
    return _without_repeats(found), scanners


_NOT_IN_TESTS = re.compile(r"^`(\w+)` is public and is not mentioned in "
                           r"the tests")


def _without_repeats(found: list[AuditFinding]) -> list[AuditFinding]:
    """Drop the per-function "not in the tests" lines already said.

    A function flagged as possibly unused, or in a module no test mentions
    at all, also collected "`f` is public and is not mentioned in the
    tests" — three lines about one fact, and on a real project that noise
    buried the findings that mattered.
    """
    unused = {(f.path, f.title.split("`")[1]) for f in found
              if f.title.endswith("is possibly unused")}
    bare = {f.path for f in found
            if f.title == "no test file mentions this module"}
    out = []
    for f in found:
        m = _NOT_IN_TESTS.match(f.title)
        if m and (f.path in bare or (f.path, m.group(1)) in unused):
            continue
        out.append(f)
    return out


def _untested_modules(files: Sequence[tuple[str, str, str]],
                      tests: Sequence[tuple[str, str]]
                      ) -> list[AuditFinding]:
    sources = [(p, lang) for p, lang, _t in files
               if not _looks_like_test(p)]
    if not sources:
        return []
    if not tests:
        return [AuditFinding(
            path=".", line=0, severity="medium", kind="test",
            title="the project has no tests",
            detail="Nothing checks that any of it works, so no change — "
                   "including one this audit suggests — can be verified.",
            change="Add a test file for the most important module first.")]
    test_words = _words(t for _p, t in tests)
    out = []
    for path, _lang in sources:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem in ("__init__", "__main__", "main", "setup", "conftest"):
            continue
        if test_words.get(stem, 0):
            continue
        out.append(AuditFinding(
            path=path, line=0, severity="low", kind="test",
            title="no test file mentions this module",
            detail="A change here has nothing to catch it.",
            change=f"Add tests for `{path}`."))
    return out


def _rel(path: str, root: str) -> str:
    """A diagnostic's path, relative to the project, with forward slashes.

    Tools print absolute paths. Written into the plan as-is, an absolute
    path is not recognised as a project file by `spec.py`, and a bare
    `test_x.py:10:` in a quoted message is read as a REQUIRED TEST at the
    project root — which `build` would then create.
    """
    p = str(path or "").replace("\\", "/")
    r = str(root or "").replace("\\", "/").rstrip("/")
    if r and p.startswith(r + "/"):
        p = p[len(r) + 1:]
    return p.lstrip("./") or "."


def _said(d: Any) -> str:
    """A diagnostic as a sentence without its path — see `_rel`."""
    text = (d.message or "").strip()
    return f"{text} [{d.code}]" if d.code and d.code != "syntax" else text


# --------------------------------------------------------------------------
# 3. verification — parse, compile, TESTS. Never the program.
# --------------------------------------------------------------------------

def _verify(host: Any, files: Sequence[tuple[str, str, str]],
            config: AuditConfig) -> tuple[list[AuditFinding], list[str],
                                          bool | None, list[str]]:
    """(findings, sentences, did any test run?, languages not checked)."""
    found: list[AuditFinding] = []
    said: list[str] = []
    not_checked: list[str] = []
    root = host.fs.root()
    for path, lang, text in files:
        try:
            phase = runner.syntax_check(text, lang, ex=host.exec, cwd=root,
                                        src_path=path)
        except Exception:                                # noqa: BLE001
            phase = None
        if phase is None:
            label = (langs.get(lang).label if langs.get(lang) else lang)
            note = f"{label} — no toolchain here to check it"
            if note not in not_checked:
                not_checked.append(note)
            continue
        if not phase.ok:
            parsed = dx.parse(phase.output, lang, root=root)
            first = parsed[0] if parsed else None
            found.append(AuditFinding(
                path=path, line=(first.line if first else 0),
                severity="high", kind="bug",
                title="does not parse or compile",
                detail=(_said(first) if first else
                        (phase.output or "")[-300:]),
                change="Fix the syntax error first; nothing else in this "
                       "file can be verified until it compiles."))
    broken = {f.path for f in found}
    if broken:
        said.append(f"{len(broken)} file(s) do not parse or compile.")
    else:
        said.append("Every file that could be checked parses or compiles.")

    if not config.run_tests:
        said.append("Tests were not run (--no-tests).")
        return found, said, None, not_checked
    ran_any: bool | None = None
    by_lang = sorted({lang for p, lang, _t in files if _looks_like_test(p)})
    if not by_lang:
        said.append("No test files were found, so no tests were run.")
        return found, said, False, not_checked
    for lang in by_lang:
        label = langs.get(lang).label if langs.get(lang) else lang
        try:
            result = runner.run_tests(lang, fs=host.fs, ex=host.exec,
                                      timeout=config.test_timeout)
        except Exception as exc:                         # noqa: BLE001
            said.append(f"{label}: the tests could not be started ({exc}).")
            continue
        output = "\n".join(p.output for p in result.phases)
        empty = runner.zero_tests(output) if result.ok else ""
        if not result.phases:
            said.append(f"{label}: there is no test command available "
                        f"here, so the tests were not run.")
        elif empty:
            said.append(f"{label}: {empty}")
            ran_any = ran_any or False
        elif result.ok:
            said.append(f"{label}: the tests ran and passed.")
            ran_any = True
        else:
            ran_any = True
            first = result.diagnostics[0] if result.diagnostics else None
            where = _rel(first.file, root) if first and first.file \
                else "."
            said.append(f"{label}: the tests FAIL"
                        + (f" — `{where}:{first.line}`: {_said(first)}"
                           if first else "."))
            found.append(AuditFinding(
                path=where, line=(first.line if first else 0),
                severity="high", kind="bug",
                title=f"the {label} test suite fails",
                detail=(_said(first) if first else output[-400:]),
                change="Make the failing test pass, or fix the test if it "
                       "is the test that is wrong."))
        for caveat in result.caveats:
            if caveat and caveat not in said:
                said.append(f"{label}: {caveat}")
    return found, said, ran_any, not_checked


# --------------------------------------------------------------------------
# 4. the model pass
# --------------------------------------------------------------------------

def _worth_reading(files: Sequence[tuple[str, str, str]],
                   found: Sequence[AuditFinding], limit: int
                   ) -> list[tuple[str, str, str]]:
    """The files most worth a model's attention, best first.

    Where the tools already found something, a second reader is most
    likely to find its cause or its neighbours; then the biggest sources.
    Tests come last: they are read for what they check, not reviewed.
    """
    weight: Counter = Counter()
    for f in found:
        weight[f.path] += {"high": 6, "medium": 3, "low": 1}.get(
            f.severity, 0)
    ranked = sorted(files, key=lambda f: (_looks_like_test(f[0]),
                                          -weight[f[0]], -len(f[2])))
    return ranked[:max(0, limit)]


def _numbered(text: str, max_chars: int) -> tuple[str, int, int]:
    """(numbered text, lines shown, lines in all) — cut at a line."""
    lines = text.splitlines()
    out: list[str] = []
    used = 0
    for i, line in enumerate(lines, 1):
        row = f"{i:>5}| {line}"
        if used + len(row) + 1 > max_chars and out:
            return "\n".join(out), i - 1, len(lines)
        out.append(row)
        used += len(row) + 1
    return "\n".join(out), len(lines), len(lines)


def _coerce(row: Any, path: str) -> AuditFinding | None:
    """One model finding, normalised — or None for one with no title."""
    if not isinstance(row, dict):
        return None
    title = str(row.get("title") or "").strip()
    if not title:
        return None
    severity = str(row.get("severity") or "").strip().lower()
    # Fail CLOSED: a severity the contract did not offer ("critical",
    # "HIGH!", "high|medium|low" echoed from the schema) is taken as high,
    # so it cannot hide below the line a reader stops at.
    if severity not in SEVERITIES:
        severity = "high"
    kind = str(row.get("kind") or "").strip().lower()
    if kind not in KINDS:
        kind = "risk"
    digits = re.search(r"\d+", str(row.get("line") or ""))
    return AuditFinding(path=path, line=int(digits.group()) if digits else 0,
                        severity=severity, kind=kind, title=title[:200],
                        detail=str(row.get("detail") or "")[:1200],
                        change=str(row.get("change") or "")[:1200],
                        source="model")


def _model_pass(host: Any, codemap: CodeMap,
                files: Sequence[tuple[str, str, str]],
                found: Sequence[AuditFinding], config: AuditConfig,
                cancel: Any, log: Any
                ) -> tuple[list[AuditFinding], list[str], list[str], str,
                           str]:
    """(findings, notes, files read, why it was skipped, model name)."""
    if not config.model_pass:
        return [], [], [], "the model pass was turned off (--no-model)", ""
    try:
        caps = host.llm.capabilities()
    except Exception as exc:                             # noqa: BLE001
        return [], [], [], f"the model could not be asked what is loaded " \
                           f"({exc})", ""
    if not caps.loaded:
        return [], [], [], ("no model is loaded, so only the tools' "
                            "findings are here"), ""
    chosen = _worth_reading(files, found, config.max_model_files)
    if not chosen:
        return [], [], [], "there were no files to read", caps.name

    from . import skills as skills_mod
    main_lang = Counter(lang for _p, lang, _t in files).most_common(1)[0][0]
    try:
        conventions = skills_mod.load_skills(host.fs, lang=main_lang).block()
    except Exception:                                    # noqa: BLE001
        conventions = ""
    prompts = PromptBuilder(conventions=conventions)
    persona = PERSONAS["reviewer"]
    # ONE architecture string for the whole audit: the prefix must be
    # byte-identical across every call (M52), and nothing is written
    # between them, so there is nothing that could make it change.
    try:
        architecture = codemap.prefix_block()
    except Exception:                                    # noqa: BLE001
        architecture = ""
    # Room for the file: the context, less the reply, less a generous
    # allowance for the instructions, findings and contract around it.
    context = int(caps.context_tokens or 8192)
    room_tokens = max(512, context - config.reply_tokens - 2500)
    max_chars = room_tokens * 3

    by_path: dict[str, list[AuditFinding]] = {}
    for f in found:
        by_path.setdefault(f.path, []).append(f)

    out: list[AuditFinding] = []
    notes: list[str] = []
    read: list[str] = []
    for path, lang, text in chosen:
        if cancel.is_set():
            raise Cancelled("the audit's model pass")
        body, shown, total = _numbered(text, max_chars)
        known = by_path.get(path, [])
        task = [f"Review `{path}` ({lang}) as part of an audit of an "
                f"existing project. Look for bugs, risks, missing tests, "
                f"unclear code and documentation gaps — what a careful "
                f"senior reviewer would want changed, most important "
                f"first."]
        if config.focus:
            task.append(f"The owner asked you to look especially at: "
                        f"{config.focus}")
        if shown < total:
            task.append(f"Only lines 1-{shown} of {total} fit; review "
                        f"those and do not guess about the rest.")
        extra = [f"[{path}]\n{body}"]
        if known:
            extra.append("[ALREADY FOUND BY THE TOOLS — do not repeat]\n"
                         + "\n".join(f"- {f.one_line()}" for f in known))
        prompt = prompts.build(persona, "\n".join(task),
                               architecture=architecture, epoch=0,
                               contract=AUDIT_CONTRACT, extra=extra)
        completion = host.llm.complete(
            prompt.messages(), temperature=0.1,
            max_tokens=config.reply_tokens, cancel=cancel)
        read.append(path)
        if shown < total:
            notes.append(f"`{path}`: the model saw lines 1-{shown} of "
                         f"{total}; the rest was not reviewed by it.")
        if completion.finish_reason == "error":
            notes.append(f"`{path}`: the model call failed — "
                         f"{completion.error or 'no reason was given'}.")
            continue
        _think, answer = split_think(completion.text or "")
        obj = find_json_object(answer or completion.text or "",
                               keys=("findings",))
        rows = obj.get("findings") if isinstance(obj, dict) else None
        if not isinstance(rows, list):
            notes.append(f"`{path}`: the model's reply had no readable "
                         f"findings, so this file was NOT reviewed by it "
                         f"(which is different from it finding nothing).")
            continue
        kept = [f for f in (_coerce(r, path) for r in rows) if f]
        if not kept:
            notes.append(f"`{path}`: the model looked and found nothing "
                         f"it would change.")
        out.extend(kept)
        log(phase="model", path=path, findings=len(kept),
            finish_reason=completion.finish_reason,
            tokens_in=completion.tokens_in, tokens_out=completion.tokens_out)
    return out, notes, read, "", caps.name


# --------------------------------------------------------------------------
# 5. memory, report, plan
# --------------------------------------------------------------------------

def _since_last(fs: Any, found: Sequence[AuditFinding]) -> str:
    """What changed since the last audit — for the TOOLS' findings.

    The model's findings are not compared: two runs of the same model on
    the same file word things differently, and a comparison of wordings
    would report churn as progress.
    """
    now = {f.fingerprint(): f for f in found if not f.by_model}
    try:
        before = json.loads(fs.read(MEMORY_PATH))
        prev = {row["fp"] for row in before.get("findings", [])}
        when = str(before.get("when", "the last audit"))
    except Exception:                                    # noqa: BLE001
        prev, when = None, ""
    try:
        fs.write(MEMORY_PATH, json.dumps({
            "when": time.strftime("%Y-%m-%d %H:%M"),
            "findings": [{"fp": fp, "path": f.path, "title": f.title}
                         for fp, f in sorted(now.items())]}, indent=1))
    except Exception:                                    # noqa: BLE001
        pass
    if prev is None:
        return ""
    gone = len(prev - now.keys())
    kept = len(prev & now.keys())
    new = len(now.keys() - prev)
    return (f"Since the audit of {when}: {gone} of the tools' findings are "
            f"gone, {kept} remain, and {new} are new.")


def _entry(f: AuditFinding) -> str:
    where = f"`{f.path}:{f.line}`" if f.line else f"`{f.path}`"
    text = f"- **{f.severity}** {where} — {f.title}."
    if f.detail:
        text += f" {f.detail.strip()}"
    if f.change:
        text += f" *Change:* {f.change.strip()}"
    if f.source not in ("tools", "model", "built-in"):
        text += f" ({f.source})"
    return text


def _document(r: AuditReport) -> str:
    name = r.root.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "."
    tools = sorted((f for f in r.findings if not f.by_model),
                   key=lambda f: (f.rank, f.path, f.line))
    model = sorted((f for f in r.findings if f.by_model),
                   key=lambda f: (f.rank, f.path, f.line))
    lines = [f"# Audit — {name}", ""]
    # WHAT WAS NOT LOOKED AT comes first. A report read as "this is
    # everything wrong" when a third of the project was never opened is
    # worse than no report.
    gaps = []
    if r.skipped:
        gaps.append(f"{len(r.skipped)} file(s) were not reviewed (listed "
                    f"at the end)")
    if r.model_skipped:
        gaps.append(f"no model read the code: {r.model_skipped}")
    if r.not_checked:
        gaps.append("not compiled or parsed: " + "; ".join(r.not_checked))
    if r.scanners_failed:
        gaps.append("scanners that did not produce a result: "
                    + "; ".join(r.scanners_failed))
    if gaps:
        lines += ["**Not covered:** " + "; ".join(gaps) + ".", ""]
    if r.model_name and model:
        lines += [f"> The model findings below are one model's opinion "
                  f"(`{r.model_name}`), and it was shown the tools' "
                  f"findings first — where the two agree, that is not two "
                  f"independent confirmations.", ""]
    if r.since_last:
        lines += [r.since_last, ""]
    if r.focus:
        lines += [f"Asked to look especially at: {r.focus}", ""]
    counts = Counter(f.severity for f in r.findings)
    lines += [f"{len(r.files)} file(s) reviewed; {len(r.findings)} "
              f"finding(s): {counts.get('high', 0)} high, "
              f"{counts.get('medium', 0)} medium, {counts.get('low', 0)} "
              f"low — {len(tools)} from the tools, {len(model)} from the "
              f"model.", ""]
    lines += ["## Does it work as it stands?", ""]
    lines += [f"- {s}" for s in r.verification] or ["- Not checked."]
    lines.append("")
    lines += ["## Found by the tools", ""]
    lines += [_entry(f) for f in tools] or ["Nothing."]
    lines.append("")
    if r.model_files or r.model_skipped:
        lines += ["## Found by the model"
                  + (f" (`{r.model_name}`)" if r.model_name else ""), ""]
        if r.model_skipped:
            lines.append(f"Not run: {r.model_skipped}.")
        else:
            lines += [_entry(f) for f in model] or [
                "It found nothing it would change in the files it read."]
            lines += ["", "Files it read: "
                      + ", ".join(f"`{p}`" for p in r.model_files) + "."]
        lines += [f"- {n}" for n in r.model_notes]
        lines.append("")
    if r.skipped or r.scanners_absent:
        lines += ["## Not reviewed", ""]
        lines += [f"- `{p}` — {why}" for p, why in r.skipped]
        lines += [f"- scanner not available: {s}" for s in r.scanners_absent]
        lines.append("")
    actionable = [f for f in r.findings if f.severity in ("high", "medium")]
    lines += ["## Next", ""]
    if actionable:
        lines += [f"The {len(actionable)} high and medium finding(s) are "
                  f"written as a build plan in `{PLAN_PATH}`. To act on "
                  f"them — every change is shown for approval and can be "
                  f"undone:", "",
                  "```", f"ccoder build -p {r.root} --spec "
                  f"{r.root.rstrip('/')}/{PLAN_PATH}", "```", ""]
    else:
        lines += ["Nothing high or medium to act on; the plan is empty.",
                  ""]
    return "\n".join(lines)


def _spec(r: AuditReport) -> str:
    """The findings as a build request `ccoder build --spec` can carry out.

    Only high and medium findings: a plan padded with "possibly unused"
    items spends a small model's attention on the least important work.
    The existing tests are NOT named as paths — a named test file becomes
    a task, and the planner would rewrite a test that should only be run.
    """
    name = r.root.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or "."
    act = sorted((f for f in r.findings
                  if f.severity in ("high", "medium")),
                 key=lambda f: (f.path, f.rank, f.line))
    lines = [f"# Improve {name}: fix what the audit found", "",
             "This project already exists. Change ONLY the files named "
             "below, and in each file only what its items ask. Keep every "
             "other line, name and behaviour exactly as it is. The "
             "project's existing tests must still pass.", ""]
    if r.focus:
        lines += [f"The owner's focus: {r.focus}", ""]
    project_wide = [f for f in act if f.path in (".", "")]
    per_file: dict[str, list[AuditFinding]] = {}
    for f in act:
        if f not in project_wide:
            per_file.setdefault(f.path, []).append(f)
    for n, (path, items) in enumerate(sorted(per_file.items()), 1):
        lines += [f"## {n}. {path}", ""]
        for f in items:
            where = f"line {f.line}: " if f.line else ""
            lines.append(f"- {where}{f.title} ({f.severity}, {f.kind}"
                         + (", from the model review" if f.by_model else "")
                         + ")." + (f" {f.detail.strip()}" if f.detail
                                   else "")
                         + (f" Change: {f.change.strip()}" if f.change
                            else ""))
        lines.append("")
    if project_wide:
        lines += ["## Project-wide", ""]
        lines += [f"- {f.title}. {f.change}".rstrip() for f in project_wide]
        lines.append("")
    if not per_file and not project_wide:
        lines += ["Nothing high or medium was found, so there is nothing "
                  "to change.", ""]
    low = sum(1 for f in r.findings if f.severity == "low")
    if low:
        lines += [f"({low} low-severity finding(s) are in the audit report "
                  f"and deliberately left out of this plan.)", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def audit_project(host: Any, *, config: AuditConfig | None = None,
                  cancel: Any = None) -> AuditReport:
    """Audit the project at `host.fs.root()`. Writes only `.cc_state/`."""
    from .journal import Journal

    config = config or AuditConfig()
    cancel = cancel or NeverCancelled()
    t0 = time.monotonic()
    sid = time.strftime("audit-%Y%m%d-%H%M%S")
    journal = Journal(host.fs, sid, events=host.events,
                      directory=JOURNAL_DIR)

    def log(**data: Any) -> None:
        try:
            journal.log("audit", **data)
        except Exception:                                # noqa: BLE001
            pass

    def say(message: str) -> None:
        host.emit("status", message, {"phase": "audit"})

    root = host.fs.root()
    files, skipped = _inventory(host.fs, config)
    log(phase="inventory", files=len(files), skipped=len(skipped))
    if not files:
        report = AuditReport(
            root=root, skipped=tuple(skipped), focus=config.focus,
            verification=("There are no source files here in a language "
                          "this engine knows, so there was nothing to "
                          "audit.",),
            model_skipped="there was no source to read",
            seconds=time.monotonic() - t0)
        _write(host.fs, report)
        return report
    say(f"audit: {len(files)} source file(s) to review")

    codemap = CodeMap(host.fs, host.storage, events=host.events)
    try:
        codemap.index_project()
    except Exception as exc:                             # noqa: BLE001
        say(f"audit: the code index could not be built ({exc}); names and "
            f"callers will not be checked")
    if cancel.is_set():
        raise Cancelled("the audit")
    say("audit: running the checks that need no model")
    found, scanners = _deterministic(host, codemap, files)
    log(phase="deterministic", findings=len(found))

    if cancel.is_set():
        raise Cancelled("the audit")
    say("audit: checking that it compiles and that its tests run")
    checked, said, ran, not_checked = _verify(host, files, config)
    found += checked
    log(phase="verify", findings=len(checked), tests_ran=ran)

    say("audit: asking the model to read the files most worth reading")
    by_model, notes, read, skipped_why, model = _model_pass(
        host, codemap, files, found, config, cancel, log)
    found += by_model

    report = AuditReport(
        root=root, files=tuple((p, lang) for p, lang, _t in files),
        skipped=tuple(skipped), findings=tuple(found),
        verification=tuple(said), tests_ran=ran,
        not_checked=tuple(not_checked),
        scanners_run=tuple(scanners["run"]),
        scanners_absent=tuple(scanners["absent"]),
        scanners_failed=tuple(scanners["failed"]),
        model_name=model, model_files=tuple(read),
        model_notes=tuple(notes), model_skipped=skipped_why,
        focus=config.focus, since_last=_since_last(host.fs, found),
        seconds=time.monotonic() - t0)
    _write(host.fs, report)
    log(phase="done", findings=len(found), high=len(report.high),
        seconds=round(report.seconds, 1))
    return report


def _write(fs: Any, report: AuditReport) -> None:
    for path, text in ((REPORT_PATH, report.document()),
                       (PLAN_PATH, report.to_spec())):
        try:
            fs.write(path, text)
        except Exception:                                # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# the command line — registered by cli.py's SUBCOMMANDS table
# --------------------------------------------------------------------------

def add_cli(sub: Any, common: Any = None) -> None:
    a = sub.add_parser(
        "audit", parents=[common] if common is not None else [],
        help="review an existing project and write a report and a plan",
        description="Look at a project folder and report what is wrong or "
                    "could be better, ranked, with file and line. Writes a "
                    "plan that `ccoder build --spec` can carry out. Never "
                    "runs the program and never changes a source file.")
    a.add_argument("folder", nargs="?", default="",
                   help="the project folder (default: --project, or here)")
    a.add_argument("--focus", default="",
                   help='what to look at especially, e.g. "the collisions"')
    a.add_argument("--no-model", action="store_true",
                   help="tools only; no model reads the code")
    a.add_argument("--no-tests", action="store_true",
                   help="do not run the project's tests")
    a.add_argument("--max-files", type=int, default=60)
    a.add_argument("--max-model-files", type=int, default=12)
    a.add_argument("--only", action="append", default=[], metavar="GLOB",
                   help="review only paths matching this (repeatable)")


def run_cli(args: Any) -> int:
    import sys

    from . import cli
    from .errors import CognitiveCoderError
    from .ports import DenyAll, NullLLM
    from .providers import make_provider

    if getattr(args, "folder", ""):
        args.project = args.folder
    if getattr(args, "remote", ""):
        print("The audit runs on a local model only, for now; drop "
              "--remote. Nothing was done.", file=sys.stderr)
        return 2
    if args.max_files < 1 or args.max_model_files < 0:
        print("--max-files must be at least 1 and --max-model-files at "
              "least 0. Nothing was done.", file=sys.stderr)
        return 2
    root = cli._project_root(args)
    if root is None:
        return 2
    # DenyAll: an audit has nothing to approve. If anything ever asked,
    # the answer would be no — which is the guarantee, stated in code.
    host = cli._host(root, args, approval=DenyAll())
    host.llm = NullLLM()
    if not args.no_model:
        problem = cli.url_problem(args.url)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        try:
            host.llm = make_provider("openai_compatible",
                                     base_url=args.url, model=args.model)
        except CognitiveCoderError as exc:
            print(f"{exc} The audit continues with the tools only.",
                  file=sys.stderr)
            host.llm = NullLLM()
    config = AuditConfig(paths=tuple(args.only), focus=args.focus,
                         model_pass=not args.no_model,
                         max_files=args.max_files,
                         max_model_files=args.max_model_files,
                         run_tests=not args.no_tests)
    try:
        report = audit_project(host, config=config)
    except Cancelled:
        print("The audit was stopped.", file=sys.stderr)
        return 1
    except CognitiveCoderError as exc:
        print(f"The audit could not finish: {exc}", file=sys.stderr)
        return 1
    print(report.document())
    print(f"(Written to {root}/{REPORT_PATH} and {root}/{PLAN_PATH}.)")
    return 0
