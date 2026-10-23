# SPDX-License-Identifier: Apache-2.0
"""Red Team — an attacker who has to prove it.

Bill (2026-10-22): *"otherwise they may use code that instantly exposes
them."* Its own tab, a conversation, a file or a folder in scope, creative
control given. The design rule that makes it honest:

    A FINDING WITHOUT A REPRODUCER IS A GUESS.

The attacker does not opine. For every finding it produces a **reproducer
test** — a unittest module written under `tests/redteam/` that asserts the
SAFE behaviour and therefore FAILS against the current code — and the
engine runs it. A test that fails is a demonstrated finding; a test that
passes (or does not run) means the attacker could not demonstrate it, and
it is listed as such: shown, never counted. After a repair, the same test
passing is the acceptance — the loop's own rule.

Order of work, deterministic first:

  1. **Scanners.** `guard.scan` (secrets, dangerous calls — a screen
     against accidents, nothing more, as guard.py says), then `bandit` and
     `semgrep` through `review.run_scanners` when they are installed, with
     the same cost-of-absence note as every optional tool.
  2. **The threat model.** The attacker reads the scope and writes it as a
     structured list: trust boundaries (every input from outside — argv,
     files, sockets, env, the user), assets (what the code can reach —
     files, network, subprocess, credentials), and what would expose the
     operator. The operator sees this BEFORE any attack runs.
  3. **Attacks.** One reproducer per threat, run with the project's own
     test runner in the sandbox, never on the operator's files directly.
  4. **The report** — REDTEAM.md: scope, threat model, demonstrated
     findings with severity and reproducer, "could not demonstrate" set
     aside. Checked findings become Repair tasks (`to_repairs`): the
     reproducer's command is the DebugReport's command, so Debug & Repair
     runs it before and after and confirms the fix the ordinary way.
  5. **The conversation** (`converse`): the operator can argue — "that
     input never reaches there". The attacker demonstrates or withdraws,
     and a withdrawal is recorded as one.

What it refuses: anything that is not the operator's own code in the
sandbox. The persona says so; `run` only ever reads the scope's files and
writes under `tests/redteam/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from . import guard, langs, review, runner, structured
from . import scope as scope_mod
from .personas import PERSONAS, strip_think
from .types import Message

REPORT = "REDTEAM.md"
TEST_DIR = "tests/redteam"
_THREAT_SCHEMA = (
    '{"threats": [{"id": "t1", "boundary": "where untrusted input enters '
    '(argv, file, socket, env, user)", "asset": "what it reaches (file '
    'system, network, subprocess, secrets, memory)", "exposure": "one '
    'sentence: how the operator is exposed", "path": "src/x.py", '
    '"severity": "high|medium|low"}]}')
_ATTACK_SCHEMA = (
    '{"title": "one line", "severity": "high|medium|low", "path": '
    '"src/x.py", "line": 12, "detail": "what the input does and why", '
    '"test": "a complete unittest module as a string: it imports the code '
    'from the project root (from src.x import f), feeds the attacking '
    'input, and ASSERTS THE SAFE BEHAVIOUR — so it FAILS now and passes '
    'once fixed; class T(unittest.TestCase); no network; no files outside '
    'a tempfile"}')
_SEVERITY = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Threat:
    id: str
    boundary: str
    asset: str
    exposure: str
    path: str = ""
    severity: str = "medium"


@dataclass
class RedFinding:
    id: str
    title: str
    severity: str
    path: str
    line: int
    detail: str
    source: str                      # guard | bandit | semgrep | attacker
    reproducer: str = ""             # tests/redteam/test_rt_<id>.py
    command: str = ""                # how the reproducer is run
    demonstrated: bool | None = None # None: a scanner finding (not attacked)
    output: str = ""                 # the reproducer's failing output
    withdrawn: str = ""              # why, when the attacker withdrew it

    @property
    def counted(self) -> bool:
        return not self.withdrawn and (self.demonstrated is None
                                       or self.demonstrated)

    def one_line(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        state = ("demonstrated" if self.demonstrated else
                 "could not demonstrate" if self.demonstrated is False
                 else "scanner")
        return f"[{self.severity}] {self.title} ({where}) — {self.source}, {state}"


@dataclass
class RedTeam:
    scope: str
    files: list[str]
    threats: list[Threat] = field(default_factory=list)
    findings: list[RedFinding] = field(default_factory=list)
    scanners_ran: list[str] = field(default_factory=list)
    scanners_absent: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)

    @property
    def demonstrated(self) -> list[RedFinding]:
        return [f for f in self.findings if f.counted and f.demonstrated]

    @property
    def undemonstrated(self) -> list[RedFinding]:
        return [f for f in self.findings if f.demonstrated is False
                and not f.withdrawn]

    @property
    def scanner_findings(self) -> list[RedFinding]:
        return [f for f in self.findings if f.demonstrated is None
                and not f.withdrawn]


# --------------------------------------------------------------------------
# 1. scanners
# --------------------------------------------------------------------------

def scan(session: Any, rt: RedTeam) -> None:
    fs, ex = session.host.fs, session.host.exec
    n = 0
    for path in rt.files:
        try:
            code = fs.read(path)
        except Exception:                                    # noqa: BLE001
            continue
        lang = langs.id_for_path(path) or "python"
        for g in guard.scan(code, lang, project_mode=True):
            n += 1
            rt.findings.append(RedFinding(
                id=f"g{n}", title=g.reason, path=path, line=g.line,
                severity="high" if g.severity == "block" else "medium",
                detail=f"matched: {g.match}", source="guard"))
        failed: list[str] = []
        try:
            found, ran, absent = review.run_scanners(
                path, lang, fs=fs, ex=ex, failed=failed)
        except Exception as exc:                             # noqa: BLE001
            rt.notes.append(f"scanners failed on {path}: {exc}")
            continue
        for name in ran:
            if name not in rt.scanners_ran:
                rt.scanners_ran.append(name)
        for note in absent:
            if note not in rt.scanners_absent:
                rt.scanners_absent.append(note)
        for note in failed:
            if note not in rt.notes:
                rt.notes.append(note)
        for f in found:
            n += 1
            rt.findings.append(RedFinding(
                id=f"s{n}", title=f.title, path=f.path or path, line=f.line,
                severity=f.severity if f.severity in _SEVERITY else "medium",
                detail=f.detail, source=f.source))
    if "guard" not in rt.scanners_ran:
        rt.scanners_ran.append("guard")


# --------------------------------------------------------------------------
# 2. the threat model
# --------------------------------------------------------------------------

def _listing(session: Any, files: list[str], max_chars: int = 14000) -> str:
    parts, budget = [], max_chars
    for path in files:
        try:
            code = session.host.fs.read(path)
        except Exception:                                    # noqa: BLE001
            continue
        numbered = "\n".join(f"{n:>4} | {ln}" for n, ln in
                             enumerate(code.splitlines(), 1))
        if len(numbered) > budget:
            numbered = numbered[:budget] + "\n… (truncated)"
        budget -= len(numbered)
        parts.append(f"### {path}\n{numbered}")
        if budget <= 0:
            break
    return "\n\n".join(parts)


def threat_model(session: Any, rt: RedTeam, *, max_threats: int = 8
                 ) -> list[Threat]:
    files = [f for f in rt.files if not scope_mod.is_test(f)]
    task = "\n".join([
        f"THREAT MODEL for {rt.scope}. Read the code and list, as "
        "structured rows, where it trusts something it did not make: every "
        "trust boundary (argv, files it reads, sockets, environment, the "
        "user's typed input, data from another process), the assets each "
        "one can reach (file system, network, subprocess, credentials, "
        "memory), and how that would EXPOSE the person running it — a path "
        "that escapes the project, an input that reaches a shell, a secret "
        "written to a log, a default that listens on every interface, an "
        "unsafe deserialization, an eval. Name the file. Rank by severity. "
        "A code base with no boundaries is a valid answer: an empty list.",
        "",
        _listing(session, files),
    ])
    got = structured.ask(session, PERSONAS["attacker"], task,
                         keys=("threats",), example=_THREAT_SCHEMA,
                         temperature=0.3, max_tokens=1600)
    if not got.ok:
        rt.notes.append(f"the threat model could not be read: {got.problem}")
    rows = (got.data or {}).get("threats") or []
    known = set(rt.files)
    out: list[Threat] = []
    for n, row in enumerate(rows if isinstance(rows, list) else (), 1):
        if not isinstance(row, dict):
            continue
        path = scope_mod._norm(str(row.get("path", "")))
        sev = str(row.get("severity", "medium")).strip().lower()
        out.append(Threat(
            id=f"t{n}", boundary=str(row.get("boundary", "")).strip(),
            asset=str(row.get("asset", "")).strip(),
            exposure=str(row.get("exposure", "")).strip(),
            path=path if path in known else "",
            severity=sev if sev in _SEVERITY else "medium"))
        if len(out) >= max_threats:
            break
    out.sort(key=lambda t: _SEVERITY.get(t.severity, 1))
    rt.threats = out
    try:
        session.journal.log("redteam_threats", scope=rt.scope,
                            threats=len(out))
    except Exception:                                        # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------------------
# 3. attacks — a reproducer each, run
# --------------------------------------------------------------------------

def _test_path(threat_id: str) -> str:
    return f"{TEST_DIR}/test_rt_{threat_id}.py"


def _run_reproducer(session: Any, path: str) -> tuple[bool, str]:
    """(failed_against_current_code, output). Runs ONLY this file."""
    try:
        result = runner.run_tests("python", fs=session.host.fs,
                                  ex=session.host.exec, test_path=path,
                                  path=path, timeout=120.0)
    except Exception as exc:                                 # noqa: BLE001
        return False, f"the reproducer could not be run: {exc}"
    output = "\n".join((p.proc.stdout + "\n" + p.proc.stderr)
                       if p.proc is not None else "" for p in result.phases)
    if runner.zero_tests(output):
        return False, "the reproducer collected no tests"
    return (not result.ok), output.strip()[-2500:]


def attack(session: Any, rt: RedTeam, threats: list[Threat] | None = None
           ) -> list[RedFinding]:
    """One reproducer per threat; a finding counts only if it fails now."""
    fs = session.host.fs
    out: list[RedFinding] = []
    for threat in (threats if threats is not None else rt.threats):
        files = [threat.path] if threat.path else \
            [f for f in rt.files if not scope_mod.is_test(f)][:3]
        task = "\n".join([
            f"ATTACK, in the sandbox, the code below. The threat: at "
            f"{threat.boundary or 'an input'} reaching "
            f"{threat.asset or 'an asset'} — {threat.exposure}.",
            "",
            "Write ONE reproducer: a complete Python unittest module that "
            "imports the code from the project root, feeds the attacking "
            "input, and asserts the SAFE behaviour, so the test FAILS "
            "against the code as it is and passes once it is fixed. No "
            "network. No file outside a tempfile.TemporaryDirectory. If "
            "the threat cannot be demonstrated, say so in `detail` and "
            "return an empty `test`.",
            "",
            _listing(session, files, max_chars=9000),
        ])
        try:
            got = structured.ask(session, PERSONAS["attacker"], task,
                                 keys=("title", "test"),
                                 example=_ATTACK_SCHEMA, temperature=0.4,
                                 max_tokens=2400)
        except Exception as exc:                             # noqa: BLE001
            rt.notes.append(f"{threat.id}: the attacker could not be asked: "
                            f"{exc}")
            continue
        if not got.ok:
            rt.notes.append(f"{threat.id}: the attacker's reply could not "
                            f"be read: {got.problem}")
        data = got.data or {}
        title = str(data.get("title", "")).strip() or threat.exposure[:80]
        test = str(data.get("test", "") or "")
        sev = str(data.get("severity", threat.severity)).strip().lower()
        try:
            line = int(data.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        finding = RedFinding(
            id=threat.id, title=title,
            severity=sev if sev in _SEVERITY else threat.severity,
            path=scope_mod._norm(str(data.get("path", "") or threat.path)),
            line=line, detail=str(data.get("detail", "")).strip(),
            source="attacker", demonstrated=False)
        if test.strip() and "unittest" in test:
            if guard.blocked(guard.scan(test, "python", project_mode=True)):
                finding.detail += " (the reproducer itself was refused by " \
                                  "the guard: it reached outside the sandbox)"
            else:
                path = _test_path(threat.id)
                if not fs.exists(f"{TEST_DIR}/__init__.py"):
                    fs.write(f"{TEST_DIR}/__init__.py", "")
                fs.write(path, test if test.endswith("\n") else test + "\n")
                failed, output = _run_reproducer(session, path)
                finding.reproducer = path
                finding.command = (f"python -m unittest "
                                   f"{path[:-3].replace('/', '.')}")
                finding.demonstrated = failed
                finding.output = output
        out.append(finding)
        rt.findings.append(finding)
    try:
        session.journal.log("redteam_attack", scope=rt.scope,
                            attacked=len(out),
                            demonstrated=sum(1 for f in out if f.demonstrated))
    except Exception:                                        # noqa: BLE001
        pass
    return out


# --------------------------------------------------------------------------
# 4. run and report
# --------------------------------------------------------------------------

def run(session: Any, scope: str | list[str] = "", *, model: bool = True,
        progress: Any = None) -> RedTeam:
    files = scope_mod.files(session.host.fs, scope)
    label = scope if isinstance(scope, str) else ", ".join(scope)
    rt = RedTeam(scope=label or "the project", files=files)
    if not files:
        rt.notes.append("the scope names no source files")
        return rt
    if progress:
        progress("red team: scanners")
    scan(session, rt)
    if model:
        try:
            loaded = getattr(session.host.llm.capabilities(), "loaded", True)
        except Exception:                                    # noqa: BLE001
            loaded = False
        if loaded:
            if progress:
                progress("red team: threat model")
            threat_model(session, rt)
            if progress:
                progress(f"red team: {len(rt.threats)} attack(s)")
            attack(session, rt)
        else:
            rt.notes.append("no model loaded: scanners only, no threat "
                            "model and no attacks")
    rt.findings.sort(key=lambda f: (0 if f.counted else 1,
                                    _SEVERITY.get(f.severity, 1), f.path))
    return rt


def report(rt: RedTeam) -> str:
    lines = [f"# Red team — {rt.scope}", "",
             f"{len(rt.files)} file(s) in scope. Scanners: "
             f"{', '.join(rt.scanners_ran) or 'none'}. "
             f"{len(rt.demonstrated)} demonstrated finding(s), "
             f"{len(rt.scanner_findings)} from scanners, "
             f"{len(rt.undemonstrated)} the attacker could not demonstrate.",
             "",
             "A finding without a reproducer is a guess. Every attacker "
             "finding below either has a test that FAILS against the code "
             "as it stands, or is listed under *could not demonstrate* and "
             "is not counted.", ""]
    if rt.scanners_absent or rt.notes:
        lines.append("**What did not run, and what it costs:**")
        lines += [f"- {n}" for n in rt.scanners_absent + rt.notes]
        lines.append("")
    if rt.threats:
        lines.append("## Threat model")
        for t in rt.threats:
            lines.append(f"- **{t.id}** [{t.severity}] {t.boundary} → "
                         f"{t.asset}: {t.exposure}"
                         + (f" ({t.path})" if t.path else ""))
        lines.append("")
    if rt.demonstrated:
        lines.append("## Demonstrated")
        for f in rt.demonstrated:
            lines.append(f"- [ ] **{f.id}** [{f.severity}] {f.title} — "
                         f"{f.path}" + (f":{f.line}" if f.line else ""))
            if f.detail:
                lines.append(f"      {f.detail}")
            lines.append(f"      reproducer: `{f.reproducer}` "
                         f"(`{f.command}`)")
        lines.append("")
    if rt.scanner_findings:
        lines.append("## From the scanners (not attacked; a screen, not a "
                     "verdict)")
        for f in rt.scanner_findings:
            lines.append(f"- [ ] **{f.id}** [{f.severity}] {f.title} — "
                         f"{f.path}" + (f":{f.line}" if f.line else "")
                         + f" ({f.source})")
        lines.append("")
    if rt.undemonstrated:
        lines.append("## Could not demonstrate (shown, not counted)")
        for f in rt.undemonstrated:
            lines.append(f"- {f.id} {f.title}" + (f" — {f.detail}"
                                                   if f.detail else ""))
        lines.append("")
    withdrawn = [f for f in rt.findings if f.withdrawn]
    if withdrawn:
        lines.append("## Withdrawn after discussion")
        lines += [f"- {f.id} {f.title}: {f.withdrawn}" for f in withdrawn]
        lines.append("")
    if not rt.findings:
        lines.append("Nothing found.")
    return "\n".join(lines)


def to_repairs(rt: RedTeam, ids: list[str] | None = None) -> list:
    """Checked findings as DebugReports. A demonstrated finding carries its
    reproducer's COMMAND, so Debug & Repair runs it before and after and
    confirms the fix; a scanner finding carries the location."""
    from .debugrepair import DebugReport
    out = []
    for f in rt.findings:
        if not f.counted or (ids is not None and f.id not in ids):
            continue
        if f.demonstrated:
            out.append(DebugReport(
                error_text=f.output, command=f.command, files=(f.path,),
                description=f"RED TEAM finding {f.id}: {f.title}. "
                            f"{f.detail} The reproducer {f.reproducer} "
                            f"must pass once this is fixed; do not change "
                            f"the reproducer."))
        else:
            out.append(DebugReport(
                error_text=f"{f.path}:{f.line}: {f.title} — {f.detail}",
                files=(f.path,),
                description=f"RED TEAM scanner finding {f.id} ({f.source}): "
                            f"{f.title}."))
    return out


# --------------------------------------------------------------------------
# 5. the conversation
# --------------------------------------------------------------------------

def converse(session: Any, rt: RedTeam, text: str, *,
             max_tokens: int = 1000) -> str:
    """Argue with the attacker. It demonstrates or withdraws; a withdrawal
    names the finding id and is recorded."""
    rt.messages.append(Message(role="user", content=text))
    summary = "\n".join(f.one_line() for f in rt.findings[:20]) or "(none)"
    task = "\n".join([
        f"RED TEAM conversation about {rt.scope}. The findings so far:",
        summary,
        "",
        "The operator says:", text,
        "",
        "Answer as the attacker who must prove things. If the operator "
        "shows a finding cannot happen, withdraw it: write exactly "
        "`WITHDRAW <id>: <why>` on its own line. If you still hold it, say "
        "what input would demonstrate it. Never claim what you cannot show.",
    ])
    prompt = session.prompts.build(PERSONAS["attacker"], task, contract="")
    completion = session.host.llm.complete(prompt.messages(),
                                           temperature=0.4,
                                           max_tokens=max_tokens)
    reply = strip_think(completion.text).strip()
    for m in re.finditer(r"^WITHDRAW\s+([A-Za-z]\d+)\s*:\s*(.+)$", reply,
                         re.M):
        for f in rt.findings:
            if f.id == m.group(1) and not f.withdrawn:
                f.withdrawn = m.group(2).strip()
    rt.messages.append(Message(role="assistant", content=reply))
    return reply


# --------------------------------------------------------------------------
# 6. serialization — so a UI can hold the state between worker runs
# --------------------------------------------------------------------------

def to_json(rt: RedTeam) -> str:
    """The whole red team as plain data. The GUI holds this between the
    initial run and each `converse` turn, because every worker run builds a
    fresh session and the findings (and their withdrawals) must survive."""
    return json.dumps({
        "scope": rt.scope, "files": rt.files,
        "scanners_ran": rt.scanners_ran, "scanners_absent": rt.scanners_absent,
        "notes": rt.notes,
        "threats": [t.__dict__ for t in rt.threats],
        "findings": [f.__dict__ for f in rt.findings],
        "messages": [{"role": m.role, "content": m.content}
                     for m in rt.messages],
    }, indent=1)


def from_json(text: str) -> RedTeam:
    data = json.loads(text)
    rt = RedTeam(scope=str(data.get("scope", "")),
                 files=list(data.get("files") or []),
                 scanners_ran=list(data.get("scanners_ran") or []),
                 scanners_absent=list(data.get("scanners_absent") or []),
                 notes=list(data.get("notes") or []))
    for row in data.get("threats") or []:
        if isinstance(row, dict):
            rt.threats.append(Threat(**{k: row.get(k) for k in
                                        ("id", "boundary", "asset", "exposure",
                                         "path", "severity") if k in row}))
    for row in data.get("findings") or []:
        if isinstance(row, dict):
            rt.findings.append(RedFinding(**{
                k: row.get(k) for k in
                ("id", "title", "severity", "path", "line", "detail", "source",
                 "reproducer", "command", "demonstrated", "output", "withdrawn")
                if k in row}))
    for row in data.get("messages") or []:
        if isinstance(row, dict):
            rt.messages.append(Message(role=str(row.get("role", "user")),
                                       content=str(row.get("content", ""))))
    return rt
