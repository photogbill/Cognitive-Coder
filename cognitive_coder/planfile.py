# SPDX-License-Identifier: Apache-2.0
"""Working memory on disk: the plan the MODEL re-reads.

WHY THIS EXISTS, 2026-11-05 (docs/NEXT_PLAN_20261105.md §1)
----------------------------------------------------------
The journal (journal.py) is a complete record of a build — append-only,
hashed, resumable. It is written FOR THE HOST. The small model never reads
it, so a 7B–24B coder loses the thread the moment its context fills: it
forgets what the task was for, which phase it is in, and what it learned
three files ago. The one good idea taken from OthmanAdi/planning-with-files
(MIT; the pattern, no code) is the fix: *context window = RAM, filesystem =
disk; anything important is written to disk and read back before a
decision.* It matters MORE for a small local model than for a frontier one.

THREE FILES, beside the project, in ``<project>/.ccoder/plan/``:

  task_plan.md   Goal · Next step · Current phase · one ``### Phase N`` per
                 group of files with exactly one ``**Status:**`` line
                 (pending / in_progress / complete / failed). Read by the
                 model (as the header of every prompt) and the operator.
  findings.md    What was learned about THIS codebase: an API shape, a
                 build quirk, a failing test's real cause, an operator
                 correction. The last N ride in the header.
  progress.md    Per step: attempt, model, verify result, one line of the
                 model's reasoning. The operator's, and the model's on
                 resume only.

WHY THE ENGINE WRITES THEM, NOT THE MODEL (§1e). A model allowed to
free-write `task_plan.md` corrupts its own status markers within an hour.
The model PROPOSES (a `note` — a journal event); the engine RENDERS. The
journal stays the single source of truth; the plan files are a VIEW of it
the model can read. A malformed plan file is regenerated, never repaired
by hand and never trusted — `validate()` says what is wrong with one.

Phases are the plan's dependency depth: phase 1 is every file that
depends on nothing, phase 2 every file whose dependencies are all in
phase 1, and so on — the order the build actually runs in. A phase's
Status is computed from its files, never written by anyone.

The HEADER (§1b) is bounded — `SessionConfig.plan_header_tokens`, default
600 — because it is paid for by the context like everything else. It opens
the volatile tail of every generation, repair and review prompt
(personas.PromptBuilder.header), never the cached prefix, because it
changes per task.

Pure except for the three writes; nothing here talks to a model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
import posixpath
import re
from typing import Any

from .journal import now_iso

#: Where the three files live, relative to the project root. Beside the
#: deployed skills (`.ccoder/skills`), never under a user profile.
PLAN_DIR = ".ccoder/plan"
TASK_PLAN = "task_plan.md"
FINDINGS = "findings.md"
PROGRESS = "progress.md"

STATUSES = ("pending", "in_progress", "complete", "failed")

#: How many findings the header carries, newest last, before the token
#: budget trims them.
HEADER_FINDINGS = 6
#: Chars per token for the header's budget when the port has no counter.
CHARS_PER_TOKEN = 4

_PHASE_HEAD = re.compile(r"^### Phase (\d+)\b", re.M)
_STATUS_LINE = re.compile(r"^\*\*Status:\*\*\s*(\S+)\s*$", re.M)


@dataclass(frozen=True)
class PhaseView:
    number: int
    label: str
    status: str
    files: tuple[tuple[str, str, str], ...] = ()   # (path, purpose, state)


@dataclass
class PlanFiles:
    """The renderer and the one owner of the three files (§1e)."""

    fs: Any
    journal: Any = None
    directory: str = ".cc_journal"      # unused: kept for the session's call
    plan_dir: str = PLAN_DIR
    #: Findings this session proposed, newest last: (time, source, task, text).
    findings: list = field(default_factory=list)
    #: Progress lines this session wrote, newest last.
    progress: list = field(default_factory=list)
    #: The last rendered views, for a host's Plan pane.
    goal: str = ""
    next_step: str = ""
    current_phase: str = ""
    phases: tuple = ()
    #: Write failures, counted and said once — never a crash (C6).
    _failed: int = field(default=0, repr=False)
    _said: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    def path(self, name: str) -> str:
        return posixpath.join(self.plan_dir, name)

    def _write(self, name: str, text: str) -> bool:
        try:
            self.fs.write(self.path(name), text)
            return True
        except Exception:                                # noqa: BLE001
            self._failed += 1
            if not self._said and self.journal is not None:
                self._said = True
                try:
                    self.journal.log(
                        "error",
                        sentence=(f"the plan files under {self.plan_dir} "
                                  f"could not be written; the build goes "
                                  f"on without them"))
                except Exception:                        # noqa: BLE001
                    pass
            return False

    def read(self, name: str) -> str:
        try:
            if self.fs.exists(self.path(name)):
                return self.fs.read(self.path(name))
        except Exception:                                # noqa: BLE001
            pass
        return ""

    # ------------------------------------------------------------------
    # findings and progress — the model proposes, the engine renders
    # ------------------------------------------------------------------
    def note(self, text: str, *, task: str = "", source: str = "engine",
             journal: bool = True) -> None:
        """One finding about this codebase. Journaled as a `note` event
        (the record), then rendered into findings.md (the view)."""
        text = " ".join(str(text or "").split())
        if not text:
            return
        for _, _, t, old in self.findings:
            if old == text and t == task:
                return                              # said already
        when = now_iso()
        self.findings.append((when, source, task, text))
        if journal and self.journal is not None:
            try:
                self.journal.log("note", task=task, source=source, text=text)
            except Exception:                            # noqa: BLE001
                pass
        self._write(FINDINGS, self.render_findings())

    def step(self, text: str) -> None:
        """One line of progress.md: what just happened."""
        text = " ".join(str(text or "").split())
        if not text:
            return
        self.progress.append((now_iso(), text))
        self._write(PROGRESS, self.render_progress())

    def restore(self, events: Iterable[dict]) -> None:
        """On resume: the findings and progress of earlier runs, from the
        journal — so the files are regenerated from the record, never
        trusted as found on disk."""
        for row in events:
            kind = row.get("event") or row.get("kind")
            data = row.get("data") or {}
            if kind == "note":
                text = str(data.get("text", "") or "")
                if text:
                    self.findings.append((str(row.get("t", "")),
                                          str(data.get("source", "engine")),
                                          str(row.get("task", "") or
                                              data.get("task", "")),
                                          text))
            elif kind == "verify":
                task = str(row.get("task", "") or "")
                v = row.get("verify") or {}
                if task:
                    ok = v.get("ok")
                    self.progress.append((
                        str(row.get("t", "")),
                        f"{task}: {'verified' if v.get('verified') else 'ok' if ok else 'failed'}"
                        f" after {v.get('attempts', '?')} attempt(s)"
                        + (f" — {data.get('stopped_because')}"
                           if data.get("stopped_because") else "")))
        # the files are a VIEW of the record: regenerated, whatever was
        # on disk (an edit by hand, a stale copy, garbage)
        self._write(FINDINGS, self.render_findings())
        self._write(PROGRESS, self.render_progress())

    # ------------------------------------------------------------------
    # the plan — rendered from the session's live plan and outcomes
    # ------------------------------------------------------------------
    def render(self, plan: Any, *, outcomes: Sequence[Any] = (),
               where: str = "", goal: str = "") -> str:
        """Render task_plan.md from the plan (tasks, statuses, dependency
        order) and write it. Returns the text."""
        tasks = list(getattr(plan, "tasks", ()) or ())
        self.goal = _one_line(goal or getattr(plan, "request", "") or "", 220)
        phases = _phases_of(tasks, outcomes)
        self.phases = tuple(phases)
        nxt = _next_task(tasks)
        if nxt is not None:
            self.next_step = f"{nxt.path} — {_one_line(nxt.purpose, 120)}"
        elif where:
            self.next_step = where
        elif tasks and all(t.status in ("done", "skipped", "failed")
                           for t in tasks):
            self.next_step = "nothing left to write — the review, then done"
        else:
            self.next_step = "waiting for a file to become ready"
        current = next((p for p in phases if p.status == "in_progress"),
                       None) or next((p for p in phases
                                      if p.status == "pending"), None)
        self.current_phase = (f"Phase {current.number} — {current.label}"
                              if current else
                              ("done" if phases else "not planned yet"))
        lines = ["# Task plan", "",
                 f"**Goal:** {self.goal or '(not stated)'}",
                 f"**Next step:** {self.next_step}",
                 f"**Current phase:** {self.current_phase}", ""]
        for p in phases:
            lines.append(f"### Phase {p.number} — {p.label}")
            lines.append(f"**Status:** {p.status}")
            for path, purpose, state in p.files:
                box = "x" if state.startswith("done") else " "
                lines.append(f"- [{box}] {path} — {purpose}"
                             + (f" ({state})" if state else ""))
            lines.append("")
        lines.append(f"_rendered by the engine from the journal, "
                     f"{now_iso()}; edits here are overwritten — correct "
                     f"the build, not this file_")
        text = "\n".join(lines) + "\n"
        self._write(TASK_PLAN, text)
        return text

    def render_findings(self) -> str:
        lines = ["# Findings — what this build learned about this codebase",
                 ""]
        if not self.findings:
            lines.append("_nothing yet_")
        for when, source, task, text in self.findings:
            lines.append(f"- {when} [{source}]"
                         + (f" {task}:" if task else "") + f" {text}")
        return "\n".join(lines) + "\n"

    def render_progress(self) -> str:
        lines = ["# Progress", ""]
        if not self.progress:
            lines.append("_nothing yet_")
        for when, text in self.progress[-400:]:
            lines.append(f"- {when} {text}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # the header (§1b): bounded, and paid for like everything else
    # ------------------------------------------------------------------
    def header(self, *, budget_tokens: int = 600,
               count_tokens: Callable[[str], int] | None = None) -> str:
        """Goal · next step · current phase · the last findings, within
        the budget. "" when there is no plan yet."""
        if not (self.goal or self.next_step):
            return ""
        count = count_tokens or (lambda s: max(1, len(s) // CHARS_PER_TOKEN))
        head = ["[WHERE YOU ARE]",
                f"Goal: {self.goal}",
                f"Next step: {self.next_step}",
                f"Current phase: {self.current_phase}"]
        base = "\n".join(head)
        if count(base) > budget_tokens:
            # even the three lines do not fit: shorten the goal
            room = max(60, budget_tokens * CHARS_PER_TOKEN - 160)
            head[1] = f"Goal: {self.goal[:room]}…"
            base = "\n".join(head)
        recent = [f"- {text}" for _, _, _, text in
                  self.findings[-HEADER_FINDINGS:]]
        kept: list[str] = []
        for line in reversed(recent):                  # newest first…
            trial = base + "\nWhat this build has learned so far:\n" + \
                "\n".join([line] + kept)
            if count(trial) > budget_tokens:
                break
            kept.insert(0, line)                       # …kept in order
        if kept:
            base += "\nWhat this build has learned so far:\n" + "\n".join(kept)
        return base


# ----------------------------------------------------------------------
# the validator: what a well-formed task_plan.md looks like
# ----------------------------------------------------------------------

def validate(text: str) -> list[str]:
    """Problems with a task_plan.md, in words; [] when it is well formed.

    The rules are the ones a model would break if it were allowed to write
    the file: one Status line per phase, a known status word, the three
    header lines, phases numbered in order."""
    problems: list[str] = []
    if not text or not text.strip():
        return ["the file is empty"]
    for key in ("**Goal:**", "**Next step:**", "**Current phase:**"):
        if key not in text:
            problems.append(f"the {key.strip('*: ')} line is missing")
    heads = list(_PHASE_HEAD.finditer(text))
    if not heads:
        problems.append("no `### Phase N` section")
        return problems
    numbers = [int(m.group(1)) for m in heads]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"phases are numbered {numbers}, not 1..{len(numbers)}")
    for i, m in enumerate(heads):
        start = m.end()
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[start:end]
        statuses = _STATUS_LINE.findall(body)
        if len(statuses) != 1:
            problems.append(f"Phase {m.group(1)} has {len(statuses)} "
                            f"`**Status:**` line(s); exactly one is required")
        for st in statuses:
            if st not in STATUSES:
                problems.append(f"Phase {m.group(1)}: status {st!r} is not "
                                f"one of {'/'.join(STATUSES)}")
        if not re.search(r"^- \[[ x]\] ", body, re.M):
            problems.append(f"Phase {m.group(1)} lists no files")
    return problems


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _one_line(text: str, limit: int) -> str:
    out = " ".join(str(text or "").split())
    return out if len(out) <= limit else out[:limit - 1].rstrip() + "…"


def _next_task(tasks: Sequence[Any]) -> Any:
    active = next((t for t in tasks if t.status == "active"), None)
    if active is not None:
        return active
    done = {t.id for t in tasks if t.status == "done"}
    for t in tasks:
        if t.status == "pending" and all(d in done for d in t.depends_on):
            return t
    return next((t for t in tasks if t.status == "pending"), None)


def _phases_of(tasks: Sequence[Any], outcomes: Sequence[Any]
               ) -> list[PhaseView]:
    """Group the plan's tasks by dependency depth, in build order."""
    if not tasks:
        return []
    by_id = {t.id: t for t in tasks}
    depth: dict[str, int] = {}

    def d(tid: str, seen: tuple = ()) -> int:
        if tid in depth:
            return depth[tid]
        t = by_id.get(tid)
        if t is None or tid in seen:
            return 0
        deps = [x for x in t.depends_on if x in by_id]
        val = 1 + max((d(x, seen + (tid,)) for x in deps), default=0)
        depth[tid] = val
        return val

    for t in tasks:
        d(t.id)
    final = {}
    for o in outcomes:
        final[getattr(o, "path", "")] = o
    groups: dict[int, list[Any]] = {}
    for t in tasks:
        groups.setdefault(depth.get(t.id, 1), []).append(t)
    out: list[PhaseView] = []
    for n, level in enumerate(sorted(groups), start=1):
        members = groups[level]
        files = []
        states = []
        for t in members:
            o = final.get(t.path)
            if t.status == "done":
                state = ("done, verified" if o is not None and
                         getattr(o, "verified", False) else "done")
                if o is not None and not getattr(o, "ok", True):
                    state = "failed"
            elif t.status == "active":
                state = "in progress"
            elif t.status == "failed":
                state = "failed"
            elif t.status == "skipped":
                state = "skipped"
            else:
                state = ""
            if t.attempts:
                state = (state + f", {t.attempts} attempt(s)").strip(", ")
            states.append(state)
            files.append((t.path, _one_line(t.purpose, 100), state))
        statuses = [t.status for t in members]
        if all(s == "done" for s in statuses) and not any(
                st.startswith("failed") for st in states):
            status = "complete"
        elif any(s == "active" for s in statuses) or (
                any(s == "done" for s in statuses)
                and any(s == "pending" for s in statuses)):
            status = "in_progress"
        elif any(s == "failed" for s in statuses) or any(
                st.startswith("failed") for st in states):
            status = ("failed" if not any(s in ("pending", "active")
                                          for s in statuses)
                      else "in_progress")
        else:
            status = "pending"
        label = _label_for(members)
        out.append(PhaseView(number=n, label=label, status=status,
                             files=tuple(files)))
    return out


def _label_for(members: Sequence[Any]) -> str:
    names = [posixpath.basename(t.path) for t in members]
    tests = sum(1 for t in members if "test" in t.path.lower())
    if tests and tests == len(members):
        return "tests"
    shown = ", ".join(names[:3]) + (f" +{len(names) - 3}" if len(names) > 3
                                    else "")
    return shown or "files"
