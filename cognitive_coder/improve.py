# SPDX-License-Identifier: Apache-2.0
"""Improve — a conversation that ends in a change list the engine executes.

Bill (2026-10-20): *"a chat tab specifically for going back and forth with
a model regarding possible improvements … it could then structure the
instructions optimally once they are agreed upon."* The 26 Debug & Repair
sessions of that day were this conversation held in a box built for
tracebacks. This is the box built for it.

Three operations, in order:

  1. `converse` — free text both ways, with a persona (consultant, domain
     expert, mad scientist: `personas.py`) and the scope's outline in
     front of the model. Nothing is written to disk.
  2. `structure` — when the operator says "that's it", the transcript is
     turned into `ChangeItem`s: title, files, what must still hold, how to
     verify, risk. A shape a small model fills well when given it. Each is
     a checkbox on the host's side.
  3. `apply` — checked items run ONE AT A TIME through the ordinary repair
     loop (`session.debug` with the item as the description and its files
     named): narrowed edits, `keep.py`, verify, a snapshot each. The
     ledger (IMPROVEMENTS.md) gets a row per item with its outcome, and
     every later item's description carries the rows before it — the
     cross-file consequences Bill asked to be carried forward.

Evolutionary mode is `converse` with the mad scientist proposing and
`apply` with the engineer applying, in a loop; the ledger is the genome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any

from . import scope as scope_mod
from . import structured
from .personas import PERSONAS, strip_think
from .types import Message

LEDGER = "IMPROVEMENTS.md"
_SCHEMA = ('{"changes": [{"title": "imperative, one line", '
           '"files": ["src/x.py"], "keep": "what must still hold after", '
           '"verify": "the test to add or the one that must still pass", '
           '"risk": "low|medium|high", "detail": "two or three sentences '
           'a careful engineer needs to carry it out"}]}')


@dataclass
class ChangeItem:
    id: str
    title: str
    files: list[str]
    keep: str = ""
    verify: str = ""
    risk: str = "medium"
    detail: str = ""
    checked: bool = True

    def as_request(self, ledger_rows: str = "") -> str:
        parts = [f"IMPROVEMENT: {self.title}", ""]
        if self.detail:
            parts += [self.detail, ""]
        if self.keep:
            parts += [f"What must still hold afterwards: {self.keep}", ""]
        if self.verify:
            parts += [f"How it is verified: {self.verify}", ""]
        parts.append("Change only what this improvement needs. Every "
                     "function, class and method the improvement does not "
                     "mention stays exactly as it is.")
        if ledger_rows:
            parts += ["", "Changes already made in this series (keep them "
                          "consistent with yours):", ledger_rows]
        return "\n".join(parts)


@dataclass
class Conversation:
    scope: str = ""
    persona: str = "consultant"
    messages: list[Message] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def transcript(self) -> str:
        out = []
        for m in self.messages:
            who = "Operator" if m.role == "user" else "Model"
            out.append(f"{who}: {m.content}")
        return "\n\n".join(out)


# --------------------------------------------------------------------------
# 1. converse
# --------------------------------------------------------------------------

def _outline(session: Any, scope: str, max_chars: int = 6000) -> str:
    """The scope as the model should see it: the architecture block for
    the project, the files in full when the scope is small."""
    files = scope_mod.files(session.host.fs, scope)
    parts = []
    try:
        parts.append(session.codemap.architecture())
    except Exception:                                        # noqa: BLE001
        pass
    budget = max_chars
    for path in files[:8]:
        try:
            code = session.host.fs.read(path)
        except Exception:                                    # noqa: BLE001
            continue
        if len(code) > budget:
            code = code[:budget] + "\n… (truncated)"
        budget -= len(code)
        parts.append(f"### {path}\n```\n{code}\n```")
        if budget <= 0:
            break
    if len(files) > 8:
        parts.append(f"… and {len(files) - 8} more file(s) in scope: "
                     + ", ".join(files[8:20]))
    return "\n\n".join(parts)


def converse(session: Any, convo: Conversation, text: str, *,
             max_tokens: int = 1200) -> str:
    """One turn. Appends the operator's text and the reply; returns the reply."""
    persona = PERSONAS.get(convo.persona) or PERSONAS["consultant"]
    convo.messages.append(Message(role="user", content=text))
    history = convo.messages[:-1][-12:]
    task = "\n".join([
        f"IMPROVE — a conversation about {convo.scope or 'this project'}.",
        "Read the code below and talk with the operator about what could "
        "be better: features, structure, robustness, what an expert would "
        "expect. Ask when a goal is unclear. Do not write code; the agreed "
        "changes are structured afterwards.",
        "",
        "[THE CODE IN SCOPE]",
        _outline(session, convo.scope),
        "",
        "[THE CONVERSATION SO FAR]",
        "\n\n".join(f"{'Operator' if m.role == 'user' else 'You'}: "
                    f"{m.content}" for m in history) or "(this is the start)",
        "",
        f"Operator: {text}",
    ])
    prompt = session.prompts.build(persona, task, contract="")
    completion = session.host.llm.complete(
        prompt.messages(), temperature=persona.temperature,
        max_tokens=max_tokens)
    reply = strip_think(completion.text).strip()
    convo.messages.append(Message(role="assistant", content=reply))
    try:
        session.journal.log("improve", turn=len(convo.messages) // 2,
                            persona=persona.id, scope=convo.scope,
                            chars=len(reply))
    except Exception:                                        # noqa: BLE001
        pass
    return reply


# --------------------------------------------------------------------------
# 2. structure
# --------------------------------------------------------------------------

def structure(session: Any, convo: Conversation, *,
              max_items: int = 12) -> list[ChangeItem]:
    """The agreed conversation as a change list. Empty when nothing was
    agreed or the reply could not be read."""
    files = scope_mod.files(session.host.fs, convo.scope)
    task = "\n".join([
        "Turn this conversation into the list of changes that were AGREED "
        "— not every idea mentioned, only what the operator accepted or "
        "asked for. One item per change a careful engineer could carry out "
        "on its own. Name the files from the list below; never invent a "
        "path.",
        "",
        "[FILES IN SCOPE]",
        "\n".join(files[:60]) or "(none listed)",
        "",
        "[THE CONVERSATION]",
        convo.transcript(),
    ])
    got = structured.ask(
        session, PERSONAS["structurer"], task, keys=("changes",),
        example=_SCHEMA + "\nAn empty list is the right answer when nothing "
                          "was agreed.",
        temperature=0.1, max_tokens=2000)
    rows = (got.data or {}).get("changes") or []
    known = set(files)
    items: list[ChangeItem] = []
    for n, row in enumerate(rows if isinstance(rows, list) else (), 1):
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        if not title:
            continue
        paths = [scope_mod._norm(str(p)) for p in (row.get("files") or [])
                 if isinstance(p, str)]
        paths = [p for p in paths if p in known] or paths[:3]
        risk = str(row.get("risk", "medium")).strip().lower()
        items.append(ChangeItem(
            id=f"c{n}", title=title, files=paths,
            keep=str(row.get("keep", "")).strip(),
            verify=str(row.get("verify", "")).strip(),
            risk=risk if risk in ("low", "medium", "high") else "medium",
            detail=str(row.get("detail", "")).strip()))
        if len(items) >= max_items:
            break
    try:
        session.journal.log("improve_structure", items=len(items),
                            scope=convo.scope)
    except Exception:                                        # noqa: BLE001
        pass
    return items


# --------------------------------------------------------------------------
# 3. apply
# --------------------------------------------------------------------------

def _ledger_rows(session: Any, ledger: str) -> str:
    try:
        text = session.host.fs.read(ledger)
    except Exception:                                        # noqa: BLE001
        return ""
    rows = [ln for ln in text.splitlines() if ln.startswith("- ")]
    return "\n".join(rows[-12:])


def _ledger_append(session: Any, ledger: str, item: ChangeItem,
                   state: str, verdict: str, changed: list[str]) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    row = (f"- {stamp} **{item.title}** ({', '.join(item.files) or '—'}) "
           f"→ {state}"
           + (f"; changed {', '.join(changed)}" if changed else "")
           + (f"; keep: {item.keep}" if item.keep else "")
           + (f"; {verdict[:160]}" if verdict and state != "repaired" else ""))
    try:
        old = session.host.fs.read(ledger) if session.host.fs.exists(ledger) \
            else ""
    except Exception:                                        # noqa: BLE001
        old = ""
    head = ("# Improvements\n\nEvery change applied by the Improve tab, in "
            "order, with what it had to keep. Later items are shown these "
            "rows so the series stays consistent.\n\n")
    body = old if old.startswith("# Improvements") else head + old
    session.host.fs.write(ledger, body.rstrip("\n") + "\n" + row + "\n")


def apply(session: Any, items: list[ChangeItem], *, ledger: str = LEDGER,
          rounds: int = 3, progress: Any = None) -> list[tuple[ChangeItem, Any]]:
    """Checked items, one at a time, through the repair loop. Returns
    (item, DebugResult) pairs. A failed item does not stop the series:
    its row says so, and the next item is told."""
    from .debugrepair import DebugReport
    out = []
    for item in items:
        if not item.checked:
            continue
        if progress:
            progress(f"improve: {item.title}")
        report = DebugReport(
            description=item.as_request(_ledger_rows(session, ledger)),
            files=tuple(item.files), rounds=rounds)
        try:
            result = session.debug(report)
        except Exception as exc:                             # noqa: BLE001
            class _Failed:
                state = "error"
                verdict = f"{type(exc).__name__}: {exc}"
                diffs: dict = {}
            result = _Failed()
        changed = list(getattr(result, "diffs", {}) or {})
        _ledger_append(session, ledger, item, result.state, result.verdict,
                       changed)
        try:
            session.journal.log("improve_apply", item=item.id,
                                title=item.title, state=result.state,
                                changed=changed)
        except Exception:                                    # noqa: BLE001
            pass
        out.append((item, result))
    return out


def items_to_json(items: list[ChangeItem]) -> str:
    return json.dumps([item.__dict__ for item in items], indent=1)


def items_from_json(text: str) -> list[ChangeItem]:
    try:
        rows = json.loads(text)
    except ValueError:
        return []
    out = []
    for r in rows if isinstance(rows, list) else ():
        if isinstance(r, dict) and r.get("title"):
            out.append(ChangeItem(
                id=str(r.get("id", f"c{len(out) + 1}")),
                title=str(r["title"]), files=list(r.get("files") or []),
                keep=str(r.get("keep", "")), verify=str(r.get("verify", "")),
                risk=str(r.get("risk", "medium")),
                detail=str(r.get("detail", "")),
                checked=bool(r.get("checked", True))))
    return out
