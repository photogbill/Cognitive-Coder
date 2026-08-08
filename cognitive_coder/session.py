# SPDX-License-Identifier: Apache-2.0
"""Orchestration, provenance and resume — the object a host drives.

`Session` is the whole engine behind one interface: `start(request, profile)`,
`step()`, `run()`, `resume(id)`, `cancel()`. A host builds one, hands it the
Ports, and renders the events. Everything else in this package is machinery
underneath it.

It owns five things and one bargain:

  * **the plan** — skeleton-first (§6.8), re-derived rather than trusted
  * **the loop** — generate → verify → repair (§6.9)
  * **the codemap lifecycle** — index on every write; epochs on the rules of
    G.7, never on a whim
  * **the journal** — every artefact, provenanced (C8)
  * **the wall-clock budget** — because an unattended loop can spend a night
    achieving nothing (F11), and a clean stop that leaves resumable state is
    the difference between "paused" and "your afternoon is gone"

**`LLMPort.capabilities()` is re-read at every task boundary** (§0.1, M10).
The host — never this engine — decides which model is loaded, and it may
change one between calls. A change is an **epoch boundary**: the KV cache and
the prompt-prefix state died with the old model, so the cached prefix is
rebuilt and the journal records the model per call (which it does anyway, C8).
The core contains no swap logic and never asks for one.

**Resume is derived from the journal plus the codemap, not from an in-memory
object** (§6.13). That is what makes it survive a crash rather than merely a
pause: the object that would have held the state is exactly the thing a crash
destroys.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import uuid

from . import journal as journal_mod
from . import personas
from .codemap import CodeMap
from .errors import BudgetExceeded, Cancelled, NoModelLoadedError
from .journal import Journal
from .loop import Loop, LoopConfig
from .patcher import Patcher
from .planner import Planner, _looks_like_test
from .ports import Cancel, Host
from .providers import RemoteGate
from .redact import Budget
from .types import ModelCapabilities, Plan, Task, TaskOutcome


@dataclass
class SessionConfig:
    """Everything tunable, with Appendix G.8's starting values.

    G.8 is explicit that these are **a starting point to measure from, not a
    recommendation to hardcode**. The journal records what was used, so the
    next value is evidence rather than argument (G.9).
    """
    lang: str = "python"
    attempts: int = 4
    temperature: float = 0.15            # generation
    plan_temperature: float = 0.35       # planning / review
    max_tokens: int = 2048               # reserved output, G.8
    seed: int | None = None
    project_mode: bool = True
    use_tools: bool = True
    autofix: bool = True
    skeleton_first: bool = True
    wall_clock_s: float = 0.0            # 0 ⇒ no ceiling (F11)
    per_task_s: float = 0.0
    review_after_build: bool = True      # §4.3: after, never instead
    recommendation_path: str = "Recommendation.md"
    # Remote budgets (M42.4). Zero means no ceiling — which is the right
    # default for a LOCAL session, where the only cost is time and F11's
    # wall-clock budget already covers that.
    max_remote_tokens: int = 0
    max_remote_spend: float = 0.0
    #: Plan size ceiling. Twelve suits a request typed in a sentence and is
    #: arbitrary for a four-section design document, which is the case the
    #: --spec flag exists to serve.
    max_files: int = 12
    conventions: str = ""
    model_system_prompt: str = ""        # the model's OWN shipped prompt
    journal_dir: str = ".cc_journal"


class Session:
    """One request, from plan to verified files, with a record of all of it."""

    def __init__(self, host: Host, *, config: SessionConfig | None = None,
                 session_id: str = "") -> None:
        self.host = host
        self.config = config or SessionConfig()
        self.id = session_id or _new_id()
        self.cancel_token = Cancel()

        self.journal = Journal(host.fs, self.id, events=host.events,
                               directory=self.config.journal_dir)
        # One gate per session, never global and never persisted: a gate
        # that survives a restart is a gate that turns itself on while
        # nobody is looking (C3).
        self.gate = RemoteGate(host.events, host.approval)
        self.budget = Budget(max_tokens=self.config.max_remote_tokens,
                             max_spend=self.config.max_remote_spend)
        self.codemap = CodeMap(host.fs, host.storage, events=host.events)
        self.patcher = Patcher(host.fs, host.storage, host.approval,
                               host.events)
        self.prompts = personas.PromptBuilder(
            model_system_prompt=self.config.model_system_prompt,
            conventions=self.config.conventions)
        self.planner = Planner(host, codemap=self.codemap,
                               journal=self.journal, prompts=self.prompts,
                               lang=self.config.lang,
                               max_files=self.config.max_files)
        self.loop = Loop(
            host, codemap=self.codemap, patcher=self.patcher,
            journal=self.journal, prompts=self.prompts,
            cancel=self.cancel_token,
            config=LoopConfig(
                attempts=self.config.attempts,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens, seed=self.config.seed,
                project_mode=self.config.project_mode,
                use_tools=self.config.use_tools, autofix=self.config.autofix,
                wall_clock_s=self.config.per_task_s))

        self.plan: Plan | None = None
        self.outcomes: list[TaskOutcome] = []
        self.profile: dict = {}
        self.last_review: Any = None
        self._model: str = ""
        self._started = 0.0
        self._finished = False

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self, request: str, profile: dict | None = None) -> Plan:
        """Index, plan, write the skeleton, derive the order.

        `profile` is the questionnaire's answers as a plain dict (Appendix
        C). **The wizard belongs to the host, not the core** — a CLI and a
        web host will supply the same dict by other means — so an empty
        profile must work, and does.
        """
        self._started = time.monotonic()
        self.profile = dict(profile or {})
        caps = self._capabilities(boundary="session start")
        self.journal.log("session_start", request=request,
                         model=caps.name, session=self.id,
                         profile=dict(profile or {}),
                         config={"attempts": self.config.attempts,
                                 "temperature": self.config.temperature,
                                 "max_tokens": self.config.max_tokens,
                                 "lang": self.config.lang})
        self._git_warning()

        stats = self.codemap.index_project()
        self.journal.log("codemap", files=stats.files, symbols=stats.symbols,
                         edges=stats.edges, unresolved=stats.unresolved,
                         resolution=round(stats.resolution_rate, 3))
        self.codemap.maybe_bump_epoch(operator_asked=True)

        self.plan = self.planner.plan(request, profile)
        if self.config.skeleton_first:
            #: ORDER FIRST, THEN STUBS. This line is the whole fix for a
            #: deadlock that made ordering a no-op on every project: the
            #: stubs' imports are written from `depends_on`, so `depends_on`
            #: has to exist before the stubs do. Ordering afterwards read
            #: import-free stubs, learned nothing, and returned the model's
            #: arbitrary order — which is how a run once began with main.py
            #: and ended three failed attempts later.
            self.plan = self.planner.derive_order(self.plan)
            result = self.planner.skeleton(self.plan)
            if not result["ok"]:
                # An architecturally wrong skeleton is caught HERE, in
                # seconds, which is the entire point of §4.2. It is a
                # warning rather than a stop: the operator may still want
                # the files, and the loop will find out the hard way.
                self.host.emit("warning",
                               f"The skeleton has a problem: {result['note']} "
                               f"— the plan may be wrong. Continuing, but "
                               f"watch the first file.",
                               {"phase": "skeleton"})
            self.plan = self.planner.derive_order(self.plan)
        return self.plan

    def preview(self, request: str, profile: dict | None = None) -> dict:
        """Plan, and stop. What would be built, before anything is built.

        WHY THIS IS WORTH A ROUND TRIP
        ------------------------------
        Planning costs one small completion. A build costs twenty minutes of
        a local model's time, and the two questions most worth asking are
        both answerable after the first of those and before the second:

          * **is this the right set of files, in the right order?**
          * **does the plan actually cover what the request asked for?**

        The second is not hypothetical. A specification arrived with a section
        headed "Testing Requirements (Strict)" naming two test files; the
        planner proposed five source files and no tests; and the run went to
        completion reporting, truthfully and about ten times, that the test
        command had run zero tests. Every fact needed to catch that existed
        one completion in. Nobody was shown them together.

        So this returns `tests_required` beside `tests_planned` — the two
        numbers whose disagreement was invisible — along with the build order
        and the context cost. `Planner._ensure_required_tests` now repairs
        that particular gap automatically, and this exists so the operator can
        still SEE it happen rather than trusting that it did.

        Returns plain data, not a Plan, because the caller may be a GUI in
        another thread and handing live objects across that boundary is how a
        SQLite handle ends up on the wrong thread.
        """
        from . import spec as spec_mod

        described = spec_mod.from_text(request)
        self.plan = self.planner.plan(request, profile)
        if self.config.skeleton_first:
            self.plan = self.planner.derive_order(self.plan)

        caps = self._capabilities(boundary="preview")
        planned = [t.path for t in self.plan.tasks]
        tests_planned = [p for p in planned if _looks_like_test(p)]
        return {
            "title": described.title,
            "files": planned,
            "purposes": {t.path: t.purpose for t in self.plan.tasks},
            "tests_required": list(described.required_tests),
            "tests_planned": tests_planned,
            #: Named in the request and NOT in the plan. Should be empty now
            #: that the planner repairs it; if it is ever non-empty the repair
            #: has regressed, and the operator finds out here rather than in
            #: an hour's worth of green output that proved nothing.
            "tests_missing": [t for t in described.required_tests
                              if t not in planned],
            "files_named_in_request": list(described.mentioned_paths),
            "caveats": list(self.plan.caveats),
            "warnings": list(described.warnings),
            "approx_tokens": described.approx_tokens,
            "context_tokens": caps.context_tokens,
            "model": caps.name,
        }

    def step(self) -> TaskOutcome | None:
        """Do the next ready task. None when there is nothing left.

        Capabilities are re-read here, at the task boundary, because this is
        exactly where a host's model-swap button gets pressed (§0.1).
        """
        if self.plan is None:
            raise RuntimeError("start() before step()")
        self._check_budget()
        task = self.plan.next_ready()
        if task is None:
            return None

        caps = self._capabilities(boundary=f"task {task.path}")
        if not caps.loaded:
            raise NoModelLoadedError(
                "capabilities() reports no model at a task boundary")

        self.plan = self.plan.replace(task.with_status("active"))
        outcome = self.loop.run_task(task, request=self.plan.request)
        self.outcomes.append(outcome)
        # M31: a model without tool calling gets a summary that may not lag,
        # because it cannot look anything up to correct one that does. Read
        # from capabilities at the task boundary, so a mid-session model swap
        # to a tool-less model tightens the rule immediately.
        self.codemap.force_epoch_per_write = not caps.supports_tools

        self.plan = self.plan.replace(
            task.with_status("done" if outcome.ok else "failed",
                             attempts=len(outcome.attempts)))
        self.journal.log("verify", task=task.path,
                         verify={"ok": outcome.ok,
                                 "attempts": len(outcome.attempts),
                                 "caveats": list(outcome.caveats)},
                         stopped_because=outcome.stopped_because)

        # A replan is an epoch boundary (G.7.2), and re-planning after each
        # file is what stops a plan being wrong by file five (§6.8).
        self.plan = self.planner.replan(self.plan,
                                        reason=f"after {task.path}")
        self.codemap.maybe_bump_epoch(target=task.path)
        return outcome

    def run(self, request: str = "", profile: dict | None = None
            ) -> list[TaskOutcome]:
        """Plan and build everything. The one-call path for a CLI.

        Cancellation and budget exhaustion both leave resumable state and
        end with a session_end event — a stop is a finished session with an
        honest ending, not an absence of one.
        """
        if request:
            self.start(request, profile)
        try:
            while True:
                outcome = self.step()
                if outcome is None:
                    break
        except Cancelled as exc:
            self.journal.log("cancel", sentence=str(exc))
            self.host.emit("status", str(exc))
        except BudgetExceeded as exc:
            self.journal.log("budget", sentence=str(exc), exhausted=True)
            self.host.emit("budget", str(exc))
        else:
            # §4.3: the review runs AFTER the code builds and its tests pass,
            # not instead — and not at all if the session was cancelled or
            # ran out of budget, because a review of half-finished work
            # reads as authoritative and is not.
            if self.config.review_after_build and any(o.ok
                                                      for o in self.outcomes):
                try:
                    self.review()
                except Cancelled:
                    self.journal.log("cancel", where="review")
        finally:
            self.finish()
        return self.outcomes

    def review(self, *, use_model: bool = True) -> str:
        """The review stage, AFTER everything builds and its tests pass (§4.3).

        Reviewing code that does not compile spends tokens on a moot point,
        so this refuses to run on a session that did not finish cleanly — and
        says so rather than producing a document that looks authoritative
        about code nobody has verified.
        """
        from . import review as review_mod

        done = [o for o in self.outcomes if o.ok]
        if not done:
            self.host.emit("warning",
                           "Nothing verified, so there is nothing to review. "
                           "A review of code that does not build is a review "
                           "of a moot point.")
            return ""

        merged = review_mod.ReviewResult()
        for outcome in done:
            self._check_cancel()
            try:
                code = self.host.fs.read(outcome.path)
            except Exception:                            # noqa: BLE001
                continue
            task = self.plan.task(outcome.task_id) if self.plan else None
            test_source = ""
            if task and task.test_path:
                try:
                    test_source = self.host.fs.read(task.test_path)
                except Exception:                        # noqa: BLE001
                    test_source = ""
            one = review_mod.review(
                code, outcome.path,
                lang_id=(task.lang if task else self.config.lang),
                fs=self.host.fs, ex=self.host.exec,
                llm=self.host.llm if use_model else None,
                prompts=self.prompts, test_source=test_source,
                use_model=use_model)
            merged.findings.extend(one.findings)
            merged.notes.extend(one.notes)
            merged.model_reviewed |= one.model_reviewed
            merged.model_name = one.model_name or merged.model_name
            for name in one.scanners_run:
                if name not in merged.scanners_run:
                    merged.scanners_run.append(name)
            for name in one.scanners_absent:
                if name not in merged.scanners_absent:
                    merged.scanners_absent.append(name)

        self.journal.log("review", findings=len(merged.findings),
                         high=len(merged.high),
                         scanners=merged.scanners_run,
                         model_reviewed=merged.model_reviewed,
                         same_model=merged.same_model)
        for finding in merged.high:
            self.host.emit("diagnostic", finding.one_line(),
                           {"category": finding.category,
                            "severity": finding.severity,
                            "path": finding.path, "line": finding.line})

        document = review_mod.recommendation_document(
            merged, request=self.plan.request if self.plan else "",
            files=[o.path for o in done],
            skill_level=str(self.profile.get("skill_level",
                                             "intermediate")),
            build_summary=f"{len(done)} of {len(self.outcomes)} file(s) "
                          f"built and their tests ran",
            #: The reviewer only ever sees committed files, so without this it
            #: cannot tell a clean build from a collapsed one — and reports the
            #: second as the first.
            unfinished=[o.path for o in self.outcomes if not o.ok],
            caveats=sorted({c for o in done for c in o.caveats}))
        self.host.fs.write(self.config.recommendation_path, document)
        self.host.emit("status",
                       f"review: {merged.summary()} — written to "
                       f"{self.config.recommendation_path}",
                       {"path": self.config.recommendation_path})
        self.last_review = merged
        return document

    def finish(self) -> str:
        if self._finished:
            return self.report()
        self._finished = True
        stats = self.codemap.stats()
        self.journal.log(
            "session_end",
            ok=all(o.ok for o in self.outcomes) and bool(self.outcomes),
            files=[o.path for o in self.outcomes if o.ok],
            failed=[o.path for o in self.outcomes if not o.ok],
            seconds=round(time.monotonic() - self._started, 1),
            codemap=stats.one_line(),
            remote=self.gate.active,
            bytes_out=self.gate.bytes_out,
            redactions=self.gate.redactions,
            budget=self.budget.as_dict() if self.gate.active else {})
        if self.gate.active:
            self.host.emit(
                "remote",
                f"Remote mode was on this session: {self.gate.bytes_out:,} "
                f"bytes sent, {self.gate.redactions} secret(s) redacted "
                f"first.",
                {"enabled": False, "bytes_out": self.gate.bytes_out,
                 "redactions": self.gate.redactions})
        self.host.emit("status", self.journal.summary())
        return self.report()

    def _check_cancel(self) -> None:
        if self.cancel_token.is_set():
            raise Cancelled("the review")

    def enable_remote(self, provider: str, *, reason: str = "") -> None:
        """Turn on ONE remote provider, for THIS session, deliberately (M42).

        There is no configuration file that does this and no environment
        variable that does it. A host calls this because a person asked, and
        the banner goes up for as long as it is true.
        """
        self.gate.enable(provider, reason=reason)

    def remote_provider(self, name: str, **kwargs: Any) -> Any:
        """Build a remote provider bound to THIS session's gate and budget."""
        from .providers import make_provider
        return make_provider(name, gate=self.gate, budget=self.budget,
                             events=self.host.events, journal=self.journal,
                             **kwargs)

    def cancel(self) -> None:
        """Stop at the next phase boundary. Thread-safe (§5.2).

        The ONE method a host may call from another thread. Everything else
        in this class assumes single-threaded use, and the token is what
        makes the exception safe.
        """
        self.cancel_token.set()

    # ------------------------------------------------------------------
    # resume (§6.13)
    # ------------------------------------------------------------------
    @classmethod
    def resume(cls, host: Host, session_id: str, *,
               config: SessionConfig | None = None) -> Session:
        """Rebuild a session from its journal. Survives a crash, not a pause.

        Nothing here reads an in-memory object, because the object is what a
        crash destroys. The journal says which tasks verified; the codemap
        says what is on disk; between them the remaining work is a fact
        rather than a guess.
        """
        session = cls(host, config=config, session_id=session_id)
        state = journal_mod.resume_state(
            host.fs, session_id, (config or SessionConfig()).journal_dir)
        if not state["events"]:
            raise FileNotFoundError(
                f"There is no journal for session {session_id}, so there is "
                f"nothing to resume. Start a new session instead.")
        session.codemap.index_project()

        plan_data = state.get("plan") or {}
        files = list(plan_data.get("files") or [])
        if files:
            done = set(state["done"])
            tasks = []
            for i, path in enumerate(files):
                status = "done" if path in done else "pending"
                tasks.append(Task(
                    id=f"t{i + 1}", path=path,
                    purpose=f"(resumed) part of: {state['request']}",
                    test_path=session.planner.test_path_for(path),
                    lang=session.config.lang, status=status,
                    attempts=state["attempts"].get(path, 0)))
            session.plan = session.planner.derive_order(
                Plan(request=state["request"], tasks=tuple(tasks)))
        session.journal.log("session_start", resumed_from=session_id,
                            request=state["request"],
                            done=state["done"], remaining=[
                                t.path for t in (session.plan.tasks
                                                 if session.plan else ())
                                if t.status == "pending"])
        host.emit("status",
                  f"Resumed session {session_id}: {len(state['done'])} file(s) "
                  f"already verified, "
                  f"{len(files) - len(state['done'])} to go.")
        # A resumed session starts a new epoch: whatever prefix was cached
        # died with the process that held it.
        session.codemap.maybe_bump_epoch(operator_asked=True)
        return session

    @staticmethod
    def previous_sessions(host: Host,
                          directory: str = ".cc_journal") -> list[str]:
        return journal_mod.sessions(host.fs, directory)

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def report(self) -> str:
        """What happened, in the shape of Appendix E, honestly.

        Caveats are surfaced, not buried: a headless Godot pass and a suite
        of zero tests both LOOK like success and are not, and C4 says so out
        loud rather than in a footnote.
        """
        lines: list[str] = []
        if self.plan:
            lines.append(f"[plan]      {len(self.plan.tasks)} files")
            for t in self.plan.tasks:
                lines.append(f"              {t.path}   — {t.purpose}")
        for i, o in enumerate(self.outcomes, 1):
            mark = "→ committed" if o.ok else "→ NOT finished"
            lines.append(f"[build {i}/{len(self.outcomes)}] {o.path}")
            for a in o.attempts:
                bits = [f"   attempt {a.n}"]
                if a.continued:
                    bits.append("continued after truncation")
                if a.autofixes:
                    bits.append(f"auto-fixed: {'; '.join(a.autofixes)}")
                bits.append(a.note or "")
                lines.append("  ".join(b for b in bits if b))
            if not o.ok and o.stopped_because:
                lines.append(f"   stopped: {o.stopped_because}")
            for caveat in o.caveats:
                lines.append(f"   CAVEAT: {caveat}")
            lines.append(f"   {mark}")
        lines.append(f"[codemap]   {self.codemap.stats().one_line()}")
        lines.append(f"[journal]   {self.journal.summary()}")
        lines.append(f"            {self.journal.cache_health()}")
        return "\n".join(lines)

    def history(self) -> list:
        """What was done to the project, as the patcher's linear log."""
        return self.patcher.history()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _capabilities(self, *, boundary: str) -> ModelCapabilities:
        """Re-read at every task boundary; a change is an epoch (M10, M13)."""
        try:
            caps = self.host.llm.capabilities()
        except Exception as exc:                         # noqa: BLE001
            self.journal.error("The model port could not be asked what is "
                               "loaded.", str(exc))
            return ModelCapabilities(name="", family="unknown",
                                     context_tokens=0)
        if caps.name != self._model:
            if self._model:
                # The model changed under us. The KV cache and any
                # prompt-prefix state died with the old model, so the cached
                # prefix is rebuilt — that is the whole of the core's
                # involvement in a swap (§0.1 consequence 2).
                self.host.emit(
                    "warning",
                    f"The loaded model changed from {self._model or 'none'} "
                    f"to {caps.name or 'none'}. The cached prompt prefix has "
                    f"been rebuilt; the next call will be slower.",
                    {"was": self._model, "now": caps.name})
                self.codemap.maybe_bump_epoch(model_changed=True)
            self.journal.log("epoch", model=caps.name, boundary=boundary,
                             context_tokens=caps.context_tokens,
                             supports_tools=caps.supports_tools,
                             is_remote=caps.is_remote)
            self._model = caps.name
        if caps.is_remote:
            self.host.emit("remote",
                           "REMOTE MODE — data leaves this machine.",
                           {"model": caps.name, "enabled": True})
        return caps

    def _check_budget(self) -> None:
        """F11: budget the SESSION, not just the call.

        Local generation is slow and a complex multi-file task can run for
        hours. The stop is clean and leaves resumable state, and it reports
        what was achieved — "it stopped" without "and here is what you got"
        is the unhelpful half of the message.
        """
        if not self.config.wall_clock_s:
            return
        spent = time.monotonic() - self._started
        if spent < self.config.wall_clock_s:
            remaining = self.config.wall_clock_s - spent
            if remaining < self.config.wall_clock_s * 0.25:
                self.host.emit(
                    "budget",
                    f"{remaining / 60:.0f} minutes of the session budget "
                    f"left; {sum(1 for o in self.outcomes if o.ok)} file(s) "
                    f"finished so far.",
                    {"remaining_s": round(remaining)})
            return
        done = ", ".join(o.path for o in self.outcomes if o.ok) or "nothing"
        raise BudgetExceeded("wall-clock",
                             f"{self.config.wall_clock_s / 60:.0f} minutes",
                             done)

    def _git_warning(self) -> None:
        """Say once if the project is a git repo with uncommitted work (§6.5b).

        Do not refuse; do not commit for them. The engine never runs git
        (M27) — this looks for the directory, nothing more.
        """
        try:
            if not self.host.fs.exists(".git"):
                return
        except Exception:                                # noqa: BLE001
            return
        self.host.emit(
            "warning",
            "This project is a git repository. This engine never runs git "
            "and keeps its own snapshots, so your history, stash and index "
            "are untouched — but you may want a clean working tree before "
            "letting it write.",
            {"git": True})


def _new_id() -> str:
    """A session id. Never used in a prompt — see G.7.1.

    Stated here because it is exactly the kind of value that ends up in a
    prompt preamble by accident, and one varying token at position 40
    silently discards 30k tokens of cached work.
    """
    return f"cc-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
