# SPDX-License-Identifier: Apache-2.0
"""Which builds can be continued, and how each one was left.

Bill, 2026-10-23: *"What if an operator stops the coding, but wants to go
back to where he left off?"*

The engine could already do it — `Session.resume` rebuilds a session from
its journal (§6.13) — but only for someone who knew the session's id and
typed it at a command line. A Stop button with no way back reads, to the
person who pressed it, as "everything since lunch is gone". This module is
the half that was missing: **reading a project's journals and saying, in
one sentence each, which builds are unfinished and how they ended**, so a
host can put a Continue button on the screen and mean it.

Everything here is READ from the journal — the durable record — never from
an in-memory object, for the same reason resume is (§6.13): the thing that
would have held the state is exactly what a crash destroys. Nothing here
imports a model, a runner or the patcher; a host may call it on any thread,
on a folder whose build is long over, at startup before any model is loaded.

How a build ended (`BuildState.ended`), judged from its LAST run — a build
continued twice has three runs in one journal, and the question is always
about the latest:

  stopped    the operator pressed Stop (a `cancel` event)
  budget     a remote token or spend budget ran out
  error      the run hit an engine error and said so
  crashed    no `session_end` after the last `session_start`: the host
             closed, crashed, or the machine lost power mid-build
  finished   it ran to its end; some files may still have failed
  abandoned  the operator chose Discard — kept on disk, never offered again
  preview    planned and stopped by design; nothing was built from it

`continuable` is true when there is work left AND the build was not set
aside: a stopped build with every file already verified (Stop during the
review) has nothing to continue.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any

from . import journal as journal_mod

JOURNAL_DIR = ".cc_journal"

#: How a build ended, in the words a Guided reader gets.
ENDED_WORDS = {
    "stopped": "you pressed Stop",
    "budget": "the remote budget ran out",
    "error": "it stopped on an error",
    "crashed": "it stopped unexpectedly — the program closed or the "
               "computer went down mid-build",
    "finished": "it finished, with parts still failing",
    "abandoned": "set aside",
    "preview": "a preview — nothing was built",
    "empty": "nothing was planned",
}


@dataclass(frozen=True)
class BuildState:
    """One build's journal, summarised. Plain data: safe to hand to a GUI."""

    session_id: str
    request: str = ""
    lang: str = ""
    model: str = ""
    started: str = ""            # ISO time of the first row
    last: str = ""               # ISO time of the last row — sorts newest-first
    runs: int = 0                # 1 + the number of times it was continued
    files: tuple[str, ...] = ()  # the plan, in the order it was journaled
    done: tuple[str, ...] = ()
    verified: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    ended: str = "empty"
    last_file: str = ""          # the last file verified in the latest run
    #: {path: sha256} as the latest run left them, or None when unknown —
    #: the latest run crashed before writing them, or the journal predates
    #: them. Continue uses it to tell whether anything was edited by hand.
    hashes: dict[str, str] | None = None
    project: str = ""            # the folder, when found by `scan`
    reason: str = ""             # why it was set aside, when it was

    # -- derived ---------------------------------------------------------
    @property
    def remaining(self) -> tuple[str, ...]:
        done = set(self.done)
        return tuple(p for p in self.files if p not in done)

    @property
    def continuable(self) -> bool:
        if self.ended in ("abandoned", "preview", "empty"):
            return False
        return bool(self.files) and bool(self.remaining)

    @property
    def interrupted(self) -> bool:
        """Ended before its end — the case a Continue button is FOR. A build
        that ran to the end with failures is continuable too, but it is
        the operator's report to read, not an interruption to undo."""
        return self.ended in ("stopped", "budget", "error", "crashed")

    @property
    def title(self) -> str:
        """The request's first real line, short enough for a banner."""
        for line in (self.request or "").splitlines():
            line = re.sub(r"^\s*(?:#+|[-*>]|\d+[.)])\s*", "", line).strip()
            line = re.sub(r"<!--.*?-->", "", line).strip()
            if line:
                return line if len(line) <= 90 else line[:87].rstrip() + "…"
        return "(no request recorded)"

    @property
    def folder(self) -> str:
        return Path(self.project).name if self.project else ""

    # -- words -----------------------------------------------------------
    def when(self) -> str:
        """'2026-10-23 14:02' (UTC as journaled), or ''."""
        stamp = self.last or self.started
        m = re.match(r"(\d{4}-\d\d-\d\d)T(\d\d:\d\d)", stamp or "")
        return f"{m.group(1)} {m.group(2)} UTC" if m else ""

    def sentence(self, mode: str = "guided") -> str:
        """One line: how it ended and how far it got.

        Guided: parts, plain words, what Continue will do.
        Engineer: the counts, the last file, the session id.
        """
        n = len(self.files)
        done = len(self.done)
        left = len(self.remaining)
        failed = len(self.failed)
        if mode == "guided":
            how = ENDED_WORDS.get(self.ended, self.ended)
            bits = [f"{done} of {n} part{'s' * (n != 1)} done"]
            if failed:
                bits.append(f"{failed} failed")
            tail = (f"Continue picks up the {left} left."
                    if self.continuable and left else "")
            return (f"Stopped — {how}. " if self.interrupted
                    else f"{how[:1].upper()}{how[1:]}. ") \
                + "; ".join(bits) + "." + (f" {tail}" if tail else "")
        parts = [self.ended, f"{done}/{n} done"]
        if self.verified:
            parts.append(f"{len(self.verified)} verified")
        if failed:
            parts.append(f"{failed} failed")
        if self.last_file:
            parts.append(f"last: {self.last_file}")
        if self.runs > 1:
            parts.append(f"run {self.runs}")
        if self.model:
            parts.append(self.model)
        parts.append(self.session_id)
        return " · ".join(parts)

    def as_dict(self) -> dict:
        """Plain data for a host that crosses threads or writes JSON."""
        out = dataclasses.asdict(self)
        out.update(remaining=list(self.remaining),
                   continuable=self.continuable,
                   interrupted=self.interrupted, title=self.title,
                   when=self.when(), folder=self.folder,
                   # worded here, once, so a host never re-words them
                   sentence_guided=self.sentence("guided"),
                   sentence_engineer=self.sentence("engineer"))
        for key in ("files", "done", "verified", "failed"):
            out[key] = list(out[key])
        return out


# --------------------------------------------------------------------------
# reading one journal
# --------------------------------------------------------------------------

def _model_of(row: dict) -> str:
    data = row.get("data") or {}
    return str(row.get("model") or data.get("model") or "")


def state_of(fs: Any, session_id: str,
             directory: str = JOURNAL_DIR) -> BuildState:
    """Summarise one session's journal. Never raises for a bad journal —
    an unreadable one summarises as `empty`, which is never continuable."""
    rows = list(journal_mod.read_jsonl(fs, f"{directory}/{session_id}.jsonl"))
    if not rows:
        return BuildState(session_id=session_id)
    core = journal_mod.resume_state(fs, session_id, directory)
    plan = core.get("plan") or {}
    files = tuple(str(p) for p in plan.get("files") or ())

    starts = [i for i, r in enumerate(rows) if r.get("event") == "session_start"]
    last_start = starts[-1] if starts else 0
    latest = rows[last_start:]
    events = [r.get("event") for r in latest]

    model = ""
    for r in rows:
        model = _model_of(r) or model

    if any(r.get("event") == "preview" for r in rows):  # Session.preview()
        ended = "preview"
    elif "abandoned" in events:
        ended = "abandoned"
    elif "session_end" not in events:
        ended = "crashed"
    elif "cancel" in events:
        ended = "stopped"
    elif any(r.get("event") == "budget"
             and (r.get("data") or {}).get("exhausted") for r in latest):
        ended = "budget"
    elif "error" in events or any(
            r.get("event") == "session_end"
            and (r.get("data") or {}).get("error") for r in latest):
        ended = "error"
    elif not files:
        ended = "empty"
    else:
        ended = "finished"

    reason = ""
    for r in latest:
        if r.get("event") == "abandoned":
            reason = str((r.get("data") or {}).get("reason") or "")

    # The hashes are only trustworthy from a run that ENDED: a crashed run's
    # files may have moved after the previous run's session_end wrote them.
    hashes: dict[str, str] | None = None
    if "session_end" in events:
        for r in latest:
            if r.get("event") == "session_end":
                got = (r.get("data") or {}).get("hashes")
                hashes = dict(got) if isinstance(got, dict) else None

    last_file = ""
    for r in latest:
        if r.get("event") == "verify" and r.get("task"):
            v = r.get("verify") or {}
            if v.get("ok") or v.get("test") == "ok":
                last_file = str(r["task"])

    # Counted against the PLAN: a repair can verify a file the plan never
    # named (an operator's module a test blamed), and "8 of 7 done" is the
    # kind of line that makes every other number on the screen suspect.
    planned = set(files)

    def in_plan(paths: Iterable[str]) -> tuple[str, ...]:
        paths = tuple(str(p) for p in paths or ())
        return tuple(p for p in paths if p in planned) if files else paths

    return BuildState(
        session_id=session_id,
        request=str(core.get("request") or ""),
        lang=str(core.get("lang") or ""),
        model=model,
        started=str(rows[0].get("t") or ""),
        last=str(rows[-1].get("t") or ""),
        runs=max(1, len(starts)),
        files=files,
        done=in_plan(core.get("done")),
        verified=in_plan(core.get("verified")),
        failed=in_plan(core.get("failed")),
        ended=ended,
        last_file=last_file,
        hashes=hashes,
        reason=reason)


def states(fs: Any, directory: str = JOURNAL_DIR) -> list[BuildState]:
    """Every session in this project, newest first."""
    out = [state_of(fs, sid, directory)
           for sid in journal_mod.sessions(fs, directory)]
    return sorted(out, key=lambda s: (s.last, s.session_id), reverse=True)


def unfinished(fs: Any, directory: str = JOURNAL_DIR) -> list[BuildState]:
    """The builds in this project a Continue button may offer, newest first.

    Interrupted builds (stopped, crashed, out of budget, error) come before
    builds that ran to the end with failures: the first is what Continue is
    for, the second is a report the operator has already been shown.
    """
    found = [s for s in states(fs, directory) if s.continuable]
    return sorted(found, key=lambda s: (s.interrupted, s.last, s.session_id),
                  reverse=True)


def abandon(fs: Any, session_id: str, directory: str = JOURNAL_DIR, *,
            reason: str = "") -> None:
    """Never offer this build again. Appends an `abandoned` event — the
    journal is the record and is never erased (append-only, §6.5); the
    files the build wrote are untouched. `ccoder resume <id>` can still
    pick it up by name."""
    journal_mod.Journal(fs, session_id, directory=directory).log(
        "abandoned", reason=reason or "the operator set it aside")


# --------------------------------------------------------------------------
# hashes — did anyone edit the files between the stop and the continue?
# --------------------------------------------------------------------------

def file_hashes(fs: Any, paths: Iterable[str]) -> dict[str, str]:
    """{path: sha256 of its bytes} for the paths that exist. Never raises."""
    out: dict[str, str] = {}
    for path in paths:
        try:
            if not fs.exists(path):
                continue
            data = fs.read_bytes(path)
        except Exception:                                # noqa: BLE001
            continue
        out[path] = hashlib.sha256(data).hexdigest()
    return out


def changed_since(then: dict[str, str] | None, now: dict[str, str]
                  ) -> list[str] | None:
    """Paths whose content differs from `then` (edited, created, deleted),
    or None when `then` is unknown — which callers must treat as "anything
    may have changed", never as "nothing did"."""
    if then is None:
        return None
    paths = sorted(set(then) | set(now))
    return [p for p in paths if then.get(p) != now.get(p)]


# --------------------------------------------------------------------------
# across projects — for a host whose builds each get their own folder
# --------------------------------------------------------------------------

def scan(root: str | os.PathLike, *, limit: int = 40,
         directory: str = JOURNAL_DIR,
         exclude: Sequence[str] = ()) -> list[BuildState]:
    """Unfinished builds in every project folder directly under `root`.

    ATK gives every build its own folder under `code_projects\\`; the banner
    that says "you have an unfinished build" has to look in all of them.
    Only the `limit` folders whose journals changed most recently are read,
    so a root with two hundred old projects costs what twenty recent ones
    do. `exclude` names session ids not to report (the one running now).

    Read-only: nothing under `root` is created or changed. (LocalFileSystem
    is NOT used here, because constructing one makes its root directory.)
    """
    base = Path(root)
    try:
        entries = [e for e in os.scandir(base) if e.is_dir()]
    except OSError:
        return []
    candidates: list[tuple[float, Path]] = []
    for entry in entries:
        jdir = Path(entry.path) / directory
        try:
            if jdir.is_dir():
                candidates.append((jdir.stat().st_mtime, Path(entry.path)))
        except OSError:
            continue
    candidates.sort(reverse=True)
    out: list[BuildState] = []
    skip = set(exclude)
    for _mtime, folder in candidates[:max(0, limit)]:
        fs = _ReadOnlyFS(folder)
        for state in unfinished(fs, directory):
            if state.session_id in skip:
                continue
            out.append(dataclasses.replace(state, project=str(folder)))
    return sorted(out, key=lambda s: (s.interrupted, s.last, s.session_id),
                  reverse=True)


@dataclass
class _ReadOnlyFS:
    """The reads `state_of` needs, from a real folder. Not a Port — it
    cannot write — which is exactly what a scan should be unable to do."""

    root: Path

    def _path(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self._path(path).exists()

    def read(self, path: str) -> str:
        return self._path(path).read_text(encoding="utf-8", errors="replace")

    def read_bytes(self, path: str) -> bytes:
        return self._path(path).read_bytes()

    def list(self, pattern: str) -> list[str]:
        return [p.relative_to(self.root).as_posix()
                for p in sorted(self.root.glob(pattern)) if p.is_file()]
