# SPDX-License-Identifier: Apache-2.0
"""Provenance — the append-only record that makes this tool defensible.

C8 says provenance is not optional, and this module is why that is more than a
slogan. **Every artefact records** which provider and model produced it, the
prompt hash, the attempt number, the verification outcome, and the timestamp
(M7). For anyone who has to defend a change — and in the environments this
engine is built for, someone does — that record is decisive. Cline does not do
this; it is one of the few axes on which a small local model beats a frontier
one outright, because the claim being made is about traceability rather than
cleverness.

Format: **append-only JSONL, one event per line**, per session.

    {"t":"2026-08-06T09:14:02Z","event":"generate","task":"src/parser.py",
     "attempt":2,"provider":"local","model":"devstral-small-2-24b-q4_k_m",
     "prompt_sha256":"…","temperature":0.15,"seed":11,
     "tokens_in":3182,"tokens_out":880,"prompt_ms":2900,
     "verify":{"build":"ok","test":"failed","diagnostics":2}}

Three properties earn their keep:

  * **`prompt_ms` is required** (M55). Prompt-processing time per call is the
    ONLY signal that the prefix cache broke (G.7.5). A jump from 3 s to 90 s
    means the prompt prefix changed when it should not have, and nobody will
    notice any other way — a broken cache is silent, it just makes everything
    slowly worse.
  * **Tracebacks live here, not in the operator's face** (C6). `error` events
    carry the full detail; the sentence goes to the EventPort.
  * **Resume is derived from the journal plus the codemap**, not from an
    in-memory object (§6.13) — so it survives a crash, not just a pause. That
    is the difference between "you can pause it" and "your afternoon is not
    gone".

Append-only is load-bearing rather than stylistic: a rollback is a NEW event,
never an erasure (§6.5 rule 4). The history of what was tried is as much a
part of the record as the history of what stuck.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import hashlib
import json
import time
from typing import Any

from .types import Completion, JournalEvent


def now_iso() -> str:
    """UTC, second resolution, unambiguous.

    Deliberately NOT used for ordering anything — the transaction log uses
    sequence numbers precisely because timestamps sort wrongly across a clock
    change (M25 rule 2). This is for humans reading the record afterwards.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def prompt_hash(messages: Sequence[Any] | str) -> str:
    """A stable SHA-256 of a prompt, for provenance and cache debugging.

    Hashing the rendered messages rather than the object means two runs that
    produce the same bytes produce the same hash — which is exactly what the
    prefix-stability test needs (M52), and the same identity the journal
    records for C8.
    """
    if isinstance(messages, str):
        payload = messages
    else:
        payload = "\n".join(
            f"{getattr(m, 'role', '?')}:{getattr(m, 'content', str(m))}"
            for m in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Journal:
    """Append-only JSONL through the FileSystemPort (C2).

    Writes are read-modify-append rather than held open: a host may be a GUI
    that crashes, and a journal that only exists in a file handle is a journal
    that isn't there when it is needed. The cost is real and accepted — this
    is not a hot path, and durability is the entire point.
    """

    def __init__(self, fs: Any, session_id: str, *, events: Any = None,
                 directory: str = ".cc_journal") -> None:
        self.fs = fs
        self.session_id = session_id
        self.dir = directory
        self.path = f"{directory}/{session_id}.jsonl"
        self._events = events
        self._buffer: list[JournalEvent] = []

    # -- writing ----------------------------------------------------------
    def write(self, event: JournalEvent) -> JournalEvent:
        self._buffer.append(event)
        line = event.to_json() + "\n"
        try:
            prior = (self.fs.read_bytes(self.path)
                     if self.fs.exists(self.path) else b"")
            self.fs.write_bytes(self.path, prior + line.encode("utf-8"))
        except Exception:                                # noqa: BLE001
            # A journal that cannot be written must not take the build down
            # with it. The in-memory buffer still holds the record, and the
            # operator is told once rather than on every event.
            if self._events is not None and len(self._buffer) == 1:
                try:
                    self._events.event(
                        "warning",
                        "The session journal could not be written, so this "
                        "run will not leave a provenance record on disk. "
                        "Everything else continues normally.")
                except Exception:                        # noqa: BLE001
                    pass
        return event

    def log(self, event: str, **fields: Any) -> JournalEvent:
        """Record one event. Unknown fields go into `data`."""
        known = {"task", "attempt", "provider", "model", "prompt_sha256",
                 "temperature", "seed", "tokens_in", "tokens_out",
                 "prompt_ms", "verify"}
        head = {k: v for k, v in fields.items() if k in known}
        rest = {k: v for k, v in fields.items() if k not in known}
        return self.write(JournalEvent(
            t=now_iso(), event=event, session=self.session_id,
            data=_jsonable(rest), **head))

    def generation(self, *, task: str, attempt: int, provider: str,
                   completion: Completion, prompt: Sequence[Any] | str,
                   temperature: float, seed: int | None = None,
                   verify: dict | None = None, **extra: Any) -> JournalEvent:
        """The C8 event: everything needed to reconstruct what happened.

        `prompt_ms` comes off the Completion where the provider reported it —
        which every provider in this repo does, because without it the prefix
        cache fails silently (M55, G.7.5).
        """
        return self.log(
            "generate", task=task, attempt=attempt, provider=provider,
            model=completion.model, prompt_sha256=prompt_hash(prompt),
            temperature=temperature, seed=seed,
            tokens_in=completion.tokens_in, tokens_out=completion.tokens_out,
            prompt_ms=completion.prompt_ms,
            verify=verify or {},
            finish_reason=completion.finish_reason, **extra)

    def error(self, sentence: str, detail: str = "", **fields: Any
              ) -> JournalEvent:
        """The traceback goes HERE; the sentence goes to the operator (C6)."""
        return self.log("error", sentence=sentence, detail=detail, **fields)

    # -- reading ----------------------------------------------------------
    def events(self) -> list[dict]:
        """Every event in this session's journal, oldest first."""
        return list(read_jsonl(self.fs, self.path))

    def replay(self) -> list[JournalEvent]:
        return [_to_event(row) for row in self.events()]

    def last(self, event: str) -> dict | None:
        for row in reversed(self.events()):
            if row.get("event") == event:
                return row
        return None

    # -- what the journal is FOR, beyond the audit trail ------------------
    def stats(self) -> dict:
        """The numbers that make tuning an empirical question (G.9, F8).

        Recorded so that "what context size works best on this codebase" and
        "how long a file can this model hold coherently" become queries
        rather than arguments. Nothing else in this design improves with use;
        this is what makes that possible.
        """
        rows = self.events()
        gens = [r for r in rows if r.get("event") == "generate"]
        prompt_times = [int(r.get("prompt_ms", 0)) for r in gens
                        if r.get("prompt_ms")]
        first_try = sum(1 for r in gens if int(r.get("attempt", 1)) == 1
                        and (r.get("verify") or {}).get("test") == "ok")
        return {
            "events": len(rows),
            "generations": len(gens),
            "continuations": sum(1 for r in rows
                                 if r.get("event") == "continuation"),
            "repairs": sum(1 for r in gens if int(r.get("attempt", 1)) > 1),
            "tokens_in": sum(int(r.get("tokens_in", 0)) for r in gens),
            "tokens_out": sum(int(r.get("tokens_out", 0)) for r in gens),
            "models": sorted({r.get("model", "") for r in gens if
                              r.get("model")}),
            "first_attempt_successes": first_try,
            "prompt_ms_median": (sorted(prompt_times)[len(prompt_times) // 2]
                                 if prompt_times else 0),
            "prompt_ms_max": max(prompt_times) if prompt_times else 0,
            # Remote is decided by what `capabilities().is_remote` SAID at
            # the time, recorded per call — never inferred from a provider
            # name. A name-based guess gets this wrong in both directions:
            # it calls a local wrapper remote because the class is named
            # oddly, and it would call a remote endpoint local because
            # somebody named it "local". C3 deserves better than a string
            # match, and the false alarm is as damaging as the false calm:
            # an operator who sees a spurious REMOTE stops trusting the one
            # that matters.
            "remote_calls": sum(1 for r in gens
                                if (r.get("data") or {}).get("remote")),
            "redactions": sum(int((r.get("data") or {}).get("redactions", 0))
                              for r in rows),
        }

    def cache_health(self) -> str:
        """One honest sentence about the prefix cache (G.7.5).

        A broken cache is silent and makes everything slowly worse, so the
        only defence is looking at the number. A 10× spread between the median
        and the maximum means the prefix changed when it should not have.
        """
        s = self.stats()
        med, mx = s["prompt_ms_median"], s["prompt_ms_max"]
        if not med:
            return "no prompt timings recorded yet."
        if mx > med * 10 and mx > 5000:
            return (f"prompt processing spiked to {mx} ms against a median of "
                    f"{med} ms — the cached prompt prefix was probably "
                    f"invalidated. Something varying (a timestamp, an id, a "
                    f"reordered block) may have got into the stable part of "
                    f"the prompt.")
        return (f"prompt processing is steady: median {med} ms, worst "
                f"{mx} ms — the prefix cache looks healthy.")

    def summary(self) -> str:
        """The closing line of a session, in the shape of Appendix E."""
        s = self.stats()
        files = len({r.get("task") for r in self.events()
                     if r.get("event") == "patch" and r.get("task")})
        bits = [f"{s['events']} events", f"{files} files",
                f"{s['generations']} generations"]
        if s["continuations"]:
            bits.append(f"{s['continuations']} continuation"
                        f"{'s' * (s['continuations'] != 1)}")
        if s["repairs"]:
            bits.append(f"{s['repairs']} repair"
                        f"{'s' * (s['repairs'] != 1)}")
        line = "  ".join(bits)
        network = ("no network calls" if not s["remote_calls"]
                   else f"{s['remote_calls']} REMOTE calls")
        return (f"{line}\n{'all local' if not s['remote_calls'] else 'REMOTE'}"
                f" · {network} · {s['redactions']} redactions")


# ---------------------------------------------------------------------------
# reading journals that this process did not write (resume, §6.13)
# ---------------------------------------------------------------------------

def read_jsonl(fs: Any, path: str) -> Iterable[dict]:
    """Every parseable line. A corrupt tail does not lose the whole file.

    A journal truncated mid-line by a crash is the NORMAL way this file ends,
    not an exceptional one — which is precisely the moment resume needs to
    read it. Skipping the broken tail and keeping everything before it is the
    behaviour that makes crash recovery work.
    """
    try:
        raw = fs.read(path)
    except Exception:                                    # noqa: BLE001
        return []
    out: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def sessions(fs: Any, directory: str = ".cc_journal") -> list[str]:
    """Session ids with a journal on disk, newest last."""
    try:
        paths = fs.list(f"{directory}/*.jsonl")
    except Exception:                                    # noqa: BLE001
        return []
    return sorted(p.replace("\\", "/").rsplit("/", 1)[-1][:-6]
                  for p in paths)


def resume_state(fs: Any, session_id: str,
                 directory: str = ".cc_journal") -> dict:
    """What a crashed session had achieved — read from the journal alone.

    Derived from durable state rather than an in-memory object, which is what
    makes this survive a crash rather than merely a pause (§6.13). Everything
    here is a fact the journal recorded, not an inference about what probably
    happened.
    """
    rows = list(read_jsonl(fs, f"{directory}/{session_id}.jsonl"))
    done: list[str] = []
    failed: list[str] = []
    attempts: dict[str, int] = {}
    plan: dict | None = None
    request = ""
    for row in rows:
        event = row.get("event")
        task = row.get("task") or ""
        if event == "session_start":
            request = (row.get("data") or {}).get("request", "") or request
        elif event == "plan":
            data = row.get("data") or {}
            # A `plan` event is written twice for different reasons: once by
            # the planner, carrying the FILE LIST, and again after every task
            # by `replan`, carrying only what remains. Taking the last one
            # loses the file list entirely — and resume then finds nothing to
            # resume, while looking from the outside like a session that had
            # no plan. So: the first event that carries files wins, and later
            # ones may only refine it.
            if data.get("files") and not (plan or {}).get("files"):
                plan = dict(data)
            elif plan is not None and data.get("remaining"):
                plan = {**plan, "remaining": data["remaining"]}
        elif event == "generate" and task:
            attempts[task] = max(attempts.get(task, 0),
                                 int(row.get("attempt", 1)))
        elif event == "verify" and task:
            verdict = (row.get("verify") or {})
            if verdict.get("ok") or verdict.get("test") == "ok":
                if task not in done:
                    done.append(task)
            elif task not in failed:
                failed.append(task)
    return {"session": session_id, "request": request, "plan": plan,
            "done": done, "failed": [t for t in failed if t not in done],
            "attempts": attempts, "events": len(rows),
            "complete": bool(rows and rows[-1].get("event") == "session_end")}


def _to_event(row: dict) -> JournalEvent:
    known = {"t", "event", "session", "task", "attempt", "provider", "model",
             "prompt_sha256", "temperature", "seed", "tokens_in",
             "tokens_out", "prompt_ms", "verify", "data"}
    head = {k: row.get(k) for k in known if k in row}
    head.setdefault("t", "")
    head.setdefault("event", "unknown")
    extra = {k: v for k, v in row.items() if k not in known}
    data = dict(head.pop("data", {}) or {})
    data.update(extra)
    return JournalEvent(data=data, **head)


def _jsonable(obj: Any) -> Any:
    """Anything → something `json.dumps` will accept, without raising.

    A journal that refuses to record an event because a value was awkward is
    worse than one that records `"<Diagnostic …>"`. Provenance is the
    priority; fidelity of exotic types is not.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return _jsonable(dataclasses.asdict(obj))
    return str(obj)
