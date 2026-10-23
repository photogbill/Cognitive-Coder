# SPDX-License-Identifier: Apache-2.0
"""What a project's journals say went wrong — read, ranked, named.

WHY THIS EXISTS, 2026-10-22
---------------------------
Five failed builds and twenty-six debug sessions were read by hand from
`.cc_journal/*.jsonl` and `.cc_state/debug/history.jsonl`, and the reading
found that most of the minutes went to the harness, not the models: a
second attempt that returned the file byte-identical (ten minutes each at
3–5 tok/s), canvas repairs that never ran, a guard defending a placeholder
the model had planted, a green headline over a build no test had checked.
The README's own meta-lesson is "improve the loop, not the prompt"; this is
the loop reading itself.

`analyse(fs)` reads every journal under the project and returns a Report:
per session, the model, the wall time, how much of it was model time, the
per-file verify outcomes, and the FINDINGS — the patterns below, each with
the minutes it cost and the evidence (session, file, attempts). `render`
prints it. Deterministic: no model is asked.

THE PATTERNS, each from a real journal:

  * **same-size second attempt** — attempt 2 produced the same number of
    output tokens as attempt 1 for the same file, in a build phase. On
    the 10-18 build this was byte-identical output: a whole-file repair a
    chat model answers by returning the file. (Cost: attempt 2's decode.)
  * **canvas skipped** — a diffusion repair attempt whose canvas event was
    `skip`: the hole did not fit, or the fill came back as the old lines.
  * **located, not repaired** — a Debug & Repair session that ended at
    `located` with zero rounds: the engine asked and nobody answered.
  * **not a code error** — a debug session whose pasted text was a
    relative-import / not-on-path traceback (nonerror.py now answers it).
  * **no tests ran** — a session that ended `ok` with every file untested.
  * **model time** — the share of wall time spent in prompt + decode, and
    the slowest tok/s seen; a model under 8 tok/s is a RAM-spilled model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import datetime as _dt
import json
import re
from typing import Any

_REL_IMPORT = re.compile(r"attempted relative import|No module named 'src'")


@dataclass
class Finding:
    kind: str
    minutes: float
    evidence: str
    fix: str


@dataclass
class SessionSummary:
    session: str
    mode: str                      # build | debug
    models: list[str]
    minutes: float
    model_minutes: float
    gens: int
    tokens_out: int
    slowest_tok_s: float | None
    verify: dict[str, list[str]] = field(default_factory=dict)
    ok: bool | None = None
    failed: list[str] = field(default_factory=list)
    untested: list[str] = field(default_factory=list)
    debug_state: str = ""


@dataclass
class Report:
    sessions: list[SessionSummary]
    findings: list[Finding]
    debug_states: Counter = field(default_factory=Counter)

    @property
    def minutes_lost(self) -> float:
        return round(sum(f.minutes for f in self.findings), 1)


def _ts(s: str) -> _dt.datetime | None:
    try:
        return _dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def _load(fs: Any, path: str) -> list[dict]:
    out = []
    try:
        text = fs.read(path)
    except Exception:                                     # noqa: BLE001
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _summarise(name: str, ev: list[dict]) -> SessionSummary:
    models = []
    for e in ev:
        m = e.get("model")
        if m and m not in models:
            models.append(m)
    t0, t1 = _ts(ev[0].get("t", "")), _ts(ev[-1].get("t", ""))
    minutes = ((t1 - t0).total_seconds() / 60) if t0 and t1 else 0.0
    gens = [e for e in ev if e.get("event") in ("generate", "canvas")]
    model_ms = sum(float(e.get("decode_ms") or 0)
                   + float(e.get("prompt_ms") or 0) for e in gens)
    toks = sum(int(e.get("tokens_out") or 0) for e in gens)
    slowest = None
    for e in gens:
        d = float(e.get("decode_ms") or 0)
        o = int(e.get("tokens_out") or 0)
        if d > 2000 and o > 50:
            r = o / (d / 1000.0)
            slowest = r if slowest is None else min(slowest, r)
    mode = "debug" if any(e.get("event") == "debug" for e in ev) else "build"
    verify: dict[str, list[str]] = defaultdict(list)
    for e in ev:
        if e.get("event") == "verify":
            v = e.get("verify") or {}
            verify[str(e.get("task"))].append("OK" if v.get("ok") else "FAIL")
    end = next((e for e in reversed(ev)
                if e.get("event") == "session_end"), None)
    d = (end or {}).get("data") or {}
    debug_state = ""
    for e in ev:
        if e.get("event") == "debug":
            debug_state = str((e.get("data") or {}).get("state") or "")
    return SessionSummary(
        session=name, mode=mode, models=models, minutes=round(minutes, 1),
        model_minutes=round(model_ms / 60000.0, 1), gens=len(gens),
        tokens_out=toks,
        slowest_tok_s=round(slowest, 1) if slowest else None,
        verify=dict(verify), ok=d.get("ok"), failed=list(d.get("failed") or []),
        untested=list(d.get("untested") or []), debug_state=debug_state)


def _findings(name: str, ev: list[dict], summary: SessionSummary
              ) -> list[Finding]:
    out: list[Finding] = []
    diffusion = any(e.get("event") == "canvas" for e in ev)
    # 1. same-size second attempt (a chat model returning the file)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for e in ev:
        if e.get("event") == "generate" and e.get("task"):
            by_task[str(e["task"])].append(e)
    for task, gens in by_task.items():
        runs = []
        for e in gens:
            try:
                a = int(e.get("attempt") or 0)
            except ValueError:
                a = 0
            runs.append((a, int(e.get("tokens_out") or 0),
                         float(e.get("decode_ms") or 0)))
        for i in range(1, len(runs)):
            a0, o0, _d0 = runs[i - 1]
            a1, o1, d1 = runs[i]
            if a1 == a0 + 1 and a1 == 2 and o0 and o0 == o1:
                out.append(Finding(
                    "same-size second attempt", round(d1 / 60000.0, 1),
                    f"{name}: {task} attempt 1 → 2 both {o1} tokens",
                    ("a diffusion model's second attempt should have been "
                     "a canvas repair of the located hole — see [canvas "
                     "skipped] for why it was not") if diffusion else
                    ("the second attempt of a chat model is a narrowed "
                     "edit of the located definition, never the whole "
                     "file with the errors (narrow_min_lines=40)")))
    # 2. canvas skipped
    skips = [e for e in ev if e.get("event") == "canvas"
             and (e.get("data") or {}).get("kind") == "skip"]
    repairs = [e for e in ev if e.get("event") == "canvas"
               and (e.get("data") or {}).get("kind") in ("skip", "repair")]
    if skips:
        reasons = Counter(str((e.get("data") or {}).get("reason", ""))[:60]
                          for e in skips)
        why = "; ".join(f"{n}× {r}" for r, n in reasons.most_common(2))
        out.append(Finding(
            "canvas skipped", 0.0,
            f"{name}: {len(skips)} of {len(repairs)} canvas repairs "
            f"skipped — {why}",
            "multi-canvas holes for bodies over 256 tokens; never show "
            "the hole's old lines in the prompt"))
    # 3. no tests ran, headline ok
    if summary.mode == "build" and summary.ok and summary.untested:
        src = [p for p in summary.untested if "test" not in p.lower()]
        if src and all(p in summary.untested for p in src):
            out.append(Finding(
                "no tests ran", 0.0,
                f"{name}: ended ok with {len(src)} file(s) untested",
                "a session with tests present and none run is not ok "
                "(session_end.all_untested, 2026-10-22)"))
    # 4/5. debug sessions
    for e in ev:
        if e.get("event") != "debug":
            continue
        d = e.get("data") or {}
        if d.get("state") == "located" and int(d.get("rounds") or 0) == 0:
            out.append(Finding(
                "located, not repaired", summary.minutes,
                f"{name}: asked 'repair {', '.join(d.get('targets') or [])}?'"
                f" and stopped",
                "repair the located files by default; ask only when the "
                "operator set ask-first"))
    pasted = ""
    for e in ev:
        if e.get("event") == "session_start":
            pasted = json.dumps(e.get("data") or {})
            break
    if summary.mode == "debug" and _REL_IMPORT.search(pasted):
        out.append(Finding(
            "not a code error", summary.minutes,
            f"{name}: the pasted traceback was a relative-import / "
            f"not-on-path error",
            "answered with the right command before any model call "
            "(nonerror.py)"))
    # 6. model time
    # (a canvas's tokens/second is not a decode rate: diffusion is exempt)
    if summary.slowest_tok_s is not None and summary.slowest_tok_s < 8 \
            and summary.mode == "build" and not diffusion:
        out.append(Finding(
            "slow model", round(summary.model_minutes, 1),
            f"{name}: slowest decode {summary.slowest_tok_s} tok/s "
            f"({summary.model_minutes} of {summary.minutes} min in the "
            f"model)",
            "a model under 8 tok/s has spilled into system RAM: a smaller "
            "quantization or context, or narrowed edits so each attempt "
            "is 200 tokens, not 2,500"))
    return out


def analyse(fs: Any, journal_dir: str = ".cc_journal",
            debug_history: str = ".cc_state/debug/history.jsonl") -> Report:
    """Read every journal under the project; rank what cost minutes."""
    try:
        paths = sorted(fs.list(f"{journal_dir}/*.jsonl"))
    except Exception:                                     # noqa: BLE001
        paths = []
    sessions: list[SessionSummary] = []
    findings: list[Finding] = []
    for p in paths:
        ev = _load(fs, str(p))
        if not ev:
            continue
        name = str(p).replace("\\", "/").rsplit("/", 1)[-1]
        name = name[:-len(".jsonl")] if name.endswith(".jsonl") else name
        s = _summarise(name, ev)
        sessions.append(s)
        findings.extend(_findings(name, ev, s))
    states: Counter = Counter()
    for row in _load(fs, debug_history):
        states[str(row.get("state") or "")] += 1
    findings.sort(key=lambda f: -f.minutes)
    return Report(sessions=sessions, findings=findings, debug_states=states)


def render(report: Report) -> str:
    lines = []
    builds = [s for s in report.sessions if s.mode == "build"]
    debugs = [s for s in report.sessions if s.mode == "debug"]
    lines.append(f"POSTMORTEM — {len(builds)} build session(s), "
                 f"{len(debugs)} debug session(s)")
    for s in builds:
        share = (f"{round(100 * s.model_minutes / s.minutes)} %"
                 if s.minutes else "—")
        verified = sum(1 for v in s.verify.values() if v and v[-1] == "OK")
        lines.append(
            f"  {s.session}: {', '.join(m[:40] for m in s.models) or '?'}; "
            f"{s.minutes} min, {share} in the model, {s.tokens_out} tokens "
            f"out; {verified} of {len(s.verify)} file(s) verified"
            + (f"; slowest {s.slowest_tok_s} tok/s" if s.slowest_tok_s
               else "")
            + ("; ENDED OK WITH NOTHING TESTED" if s.ok and s.untested
               and len(s.untested) >= len(s.verify) else ""))
    if report.debug_states:
        lines.append("  debug sessions by outcome: " + ", ".join(
            f"{n} {k}" for k, n in report.debug_states.most_common()))
    if not report.findings:
        lines.append("No patterns found in the journals.")
        return "\n".join(lines)
    lines.append(f"\nFINDINGS, ranked by the minutes they cost "
                 f"({report.minutes_lost} min accounted for):")
    kinds = Counter(f.kind for f in report.findings)
    shown: Counter = Counter()
    for f in report.findings:
        shown[f.kind] += 1
        if shown[f.kind] > 3:
            continue
        lines.append(f"  [{f.kind}] {f.minutes} min — {f.evidence}")
        lines.append(f"      fix: {f.fix}")
    for kind, n in kinds.items():
        if n > 3:
            lines.append(f"  … and {n - 3} more [{kind}]")
    return "\n".join(lines)
