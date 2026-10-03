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

import dataclasses
from dataclasses import dataclass
import time
from typing import Any
import uuid

from . import journal as journal_mod
from . import langs, models, personas, runner
from . import skills as skills_mod
from .codemap import CodeMap
from .errors import (
    BudgetExceeded,
    Cancelled,
    CognitiveCoderError,
    NoModelLoadedError,
    NotACodingModelError,
)
from .journal import Journal, SessionLog
from .loop import Loop, LoopConfig
from .patcher import Patcher
from .planner import Planner, _looks_like_test, is_entry_point as _is_entry_point
from .ports import Cancel, Host
from .providers import RemoteGate
from .redact import Budget
from .types import ModelCapabilities, Plan, Task, TaskOutcome, Timeouts


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
    #: Pair every source file with a planned tester task (M39). Off by
    #: default only because it adds a generation per file and the committed
    #: golden trace scripts its replies for the unpaired plan — see
    #: `Planner.pair_tests`. Without it, a file with no test file is "built
    #: and ran", and the session summary counts how many are in that state.
    pair_tests: bool = False
    conventions: str = ""
    #: Deployed skills (F3). Discovered once, at Session construction, from
    #: `skills_dir` — never re-read mid-session, because the prefix must not
    #: change mid-session (M52). See `cognitive_coder/skills.py`.
    use_skills: bool = True
    skills_dir: str = skills_mod.SKILLS_DIR
    model_system_prompt: str = ""        # the model's OWN shipped prompt
    journal_dir: str = ".cc_journal"
    #: Build with a model whose name says it is NOT a coding model (a
    #: roleplay or fiction merge). Off: such a model is refused at start, in
    #: one sentence, instead of burning twenty minutes. Models the engine
    #: merely does not recognise are allowed either way, under a warning.
    allow_any_model: bool = False
    #: HOW LONG TO WAIT ON THE GENERATED PROGRAM. Not on the model — nothing
    #: in this config bounds generation, by design (see `Timeouts`). `None`
    #: takes the language default; `0` waits indefinitely.
    build_timeout: float | None = None
    run_timeout: float | None = None
    test_timeout: float | None = None


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
        #: The readable half. The JSONL defends a change; this explains a
        #: behaviour — see SessionLog for the line that made the difference.
        # `events` so a build log that stops being writable says so. Without
        # it the log's write failures were counted and never reported —
        # BUILD_LOG.txt is the file an operator reads, so its silence is
        # the one silence that matters.
        self.log = SessionLog(host.fs, self.id,
                              directory=self.config.journal_dir,
                              events=host.events)
        # One gate per session, never global and never persisted: a gate
        # that survives a restart is a gate that turns itself on while
        # nobody is looking (C3).
        self.gate = RemoteGate(host.events, host.approval)
        self.budget = Budget(max_tokens=self.config.max_remote_tokens,
                             max_spend=self.config.max_remote_spend)
        self.codemap = CodeMap(host.fs, host.storage, events=host.events)
        self.patcher = Patcher(host.fs, host.storage, host.approval,
                               host.events)
        # Deployed skills join the config's conventions in the CACHED
        # prefix. Discovery happens HERE, once — a skill edited mid-session
        # takes effect next session, exactly like an epoch (skills.py).
        self.skill_load = (
            skills_mod.load_skills(host.fs, lang=self.config.lang,
                                   directory=self.config.skills_dir,
                                   events=host.events)
            if self.config.use_skills else skills_mod.SkillLoad())
        conventions = "\n\n".join(
            part for part in (self.config.conventions,
                              self.skill_load.block()) if part)
        self.prompts = personas.PromptBuilder(
            model_system_prompt=self.config.model_system_prompt,
            conventions=conventions)
        self.planner = Planner(host, codemap=self.codemap,
                               journal=self.journal, prompts=self.prompts,
                               lang=self.config.lang,
                               max_files=self.config.max_files,
                               patcher=self.patcher,
                               pair_tests=self.config.pair_tests)
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
                wall_clock_s=self.config.per_task_s,
                timeouts=Timeouts(build=self.config.build_timeout,
                                  run=self.config.run_timeout,
                                  test=self.config.test_timeout)))
        #: The readable log is the Loop's too — it is where the verify
        #: phases happen, and their output is the whole point of the file.
        self.loop.log = self.log

        self.plan: Plan | None = None
        self.outcomes: list[TaskOutcome] = []
        #: Modules to repair against a test that failed on them:
        #: (module task, test task, the test's failing diagnostics). One
        #: pass per module per session — a second would be a loop.
        self._repairs: list[tuple[Task, Task, tuple]] = []
        self._repaired: set[str] = set()
        #: What the session was doing, for the sentence a failure earns.
        self._where = ""
        self.profile: dict = {}
        self.last_review: Any = None
        self._model: str = ""
        self._context: int = 0
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
        self._where = "planning"
        self.profile = dict(profile or {})
        caps = self._capabilities(boundary="session start")
        earlier_sessions = self._count_earlier_sessions()
        self.journal.log("session_start", request=request,
                         model=caps.name, session=self.id,
                         profile=dict(profile or {}),
                         config={"attempts": self.config.attempts,
                                 "temperature": self.config.temperature,
                                 "max_tokens": self.config.max_tokens,
                                 "lang": self.config.lang})
        # Right after the start record, so a refused session leaves a
        # journal that says it started and why it stopped.
        self._check_model(caps)
        if self.skill_load.skills or self.skill_load.skipped:
            # C8: which guidance shaped this session, at which revision —
            # and equally which guidance did NOT load, so a session that
            # ignored a rule can be told apart from one that never saw it.
            self.journal.log("skills",
                             active=self.skill_load.provenance(),
                             skipped=[{"path": p, "reason": r}
                                      for p, r in self.skill_load.skipped])
        self.log.start(request, caps.name,
                       {"attempts": self.config.attempts,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "max_files": self.config.max_files,
                        "lang": self.config.lang})
        self._git_warning()

        stats = self.codemap.index_project()
        self.journal.log("codemap", files=stats.files, symbols=stats.symbols,
                         edges=stats.edges, unresolved=stats.unresolved,
                         resolution=round(stats.resolution_rate, 3))
        self.codemap.maybe_bump_epoch(operator_asked=True)
        self._previous_build_warning(earlier_sessions)
        self._baseline_existing_tests()

        self.plan = self.planner.plan(request, profile)
        self.log.rule("PLAN")
        for i, t in enumerate(self.plan.tasks, 1):
            self.log.line(f"  {i}. {t.path}  — {t.purpose[:60]}")
        for c in self.plan.caveats:
            self.log.line(f"  CAVEAT: {c}")
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
            if result.get("kept"):
                # Visible in the plan, not only in an event that scrolled
                # past: the operator should be able to see that the engine
                # declined to stub over their file, and that the file's task
                # will still change it — through a transaction they can undo.
                kept = ", ".join(result["kept"])
                self.plan = dataclasses.replace(
                    self.plan, caveats=self.plan.caveats + (
                        f"{kept} already had real work in it and was kept "
                        f"as found — no stub was written over it; its task "
                        f"will change it in place, as an undoable "
                        f"transaction",))
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
            # The epoch snapshot was taken at start, BEFORE the skeleton
            # existed. The planner indexes each stub as it writes it, but
            # the cached prefix serves the snapshot — so the first file was
            # generated against a project with no architecture in it, which
            # is the one thing skeleton-first exists to provide. A new plan
            # is an epoch boundary (G.7.2); this is that boundary.
            self.codemap.maybe_bump_epoch(replanned=True)
        self._emit_plan(revised=False)
        self._preinstall(request)
        return self.plan

    def _preinstall(self, request: str) -> None:
        """Install the packages the REQUEST names before the first file.

        Oct 1, 2026: "using pygame" was the spec's first line; the engine
        learned it at minute 13, when `render.py` failed to import it, and
        the host's install question then waited for a person. Only a host
        that can install (an ExecPort with `ensure_packages`) does this,
        only for Python, and only for packages the operator's own text
        names — `packages.packages_in` — which is also what lets a host
        approve them without asking. Nothing here fails the build: an
        install the host declines or cannot do shows up later, at the
        import, exactly as before.
        """
        ensure = getattr(self.host.exec, "ensure_packages", None)
        if not callable(ensure) or self.config.lang != "python":
            return
        from . import packages
        wanted = packages.packages_in(request)
        if not wanted:
            return
        names = ", ".join(pip for _imp, pip in wanted)
        self.host.emit("status",
                       f"the request names {names}; making sure "
                       f"{'it is' if len(wanted) == 1 else 'they are'} "
                       f"installed in the project's environment before "
                       f"the first file", {"phase": "preinstall",
                                           "packages": [p for _i, p in wanted]})
        try:
            outcome = ensure(wanted)
        except Exception as exc:                         # noqa: BLE001
            self.host.emit("warning",
                           f"could not pre-install {names}: {exc} — the "
                           f"build goes on; a missing one will show at its "
                           f"import", {"phase": "preinstall"})
            return
        if outcome:
            self.log.event("PREINSTALL", str(outcome))

    def _emit_plan(self, *, revised: bool) -> None:
        """The plan as DATA, for hosts that draw a task board (event `plan`).

        The console line "plan: 7 file(s) proposed" is for reading; a host
        that wants to show one row per file, with its purpose in plain words
        and its state as it changes, needs the tasks themselves. Emitted
        once after planning and again whenever a re-plan changes the set of
        files, so a board never shows a file the plan dropped.
        """
        if self.plan is None:
            return
        tasks = [{"id": t.id, "path": t.path, "purpose": t.purpose,
                  "test_path": t.test_path, "persona": t.persona,
                  "depends_on": list(t.depends_on), "status": t.status,
                  "attempts": t.attempts}
                 for t in self.plan.tasks]
        self.host.emit("plan",
                       f"plan: {len(tasks)} file(s)"
                       + (" (revised)" if revised else ""),
                       {"tasks": tasks, "revised": revised,
                        "request": self.plan.request,
                        "caveats": list(self.plan.caveats)})

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

        #: Marked FIRST, so the journal this writes is never mistaken for a
        #: session that was building: `previous_sessions()` skips it and
        #: `resume()` refuses it.
        self.journal.log("preview", request=request,
                         note="planned only; nothing was built")
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
        if self._repairs:
            return self._repair_step()
        task = self.plan.next_ready()
        if task is None:
            return None

        caps = self._capabilities(boundary=f"task {task.path}")
        if not caps.loaded:
            raise NoModelLoadedError(
                "capabilities() reports no model at a task boundary")

        self.plan = self.plan.replace(task.with_status("active"))
        self._where = task.path
        covers = self._covered_module(task)
        outcome = self.loop.run_task(task, request=self.plan.request,
                                     covers=covers.path if covers else "",
                                     planned=self._planned_paths(task))
        self.outcomes.append(outcome)
        self._credit_module(covers, outcome)
        blamed = self.loop.blamed.pop(task.path, None)
        if blamed:
            self._queue_repair(task, blamed, covers)
        # M31: a model without tool calling gets a summary that may not lag,
        # because it cannot look anything up to correct one that does. Read
        # from capabilities at the task boundary, so a mid-session model swap
        # to a tool-less model tightens the rule immediately.
        self.codemap.force_epoch_per_write = not caps.supports_tools

        self.plan = self.plan.replace(
            task.with_status("done" if outcome.ok else "failed",
                             attempts=len(outcome.attempts)))
        self.log.event(
            f"{outcome.label} {task.path}",
            f"{len(outcome.attempts)} attempt(s)"
            + (f" — {outcome.stopped_because}" if outcome.stopped_because
               else ""))
        self.journal.log("verify", task=task.path,
                         verify={"ok": outcome.ok,
                                 "verified": outcome.verified,
                                 "attempts": len(outcome.attempts),
                                 "caveats": list(outcome.caveats)},
                         stopped_because=outcome.stopped_because)

        # Re-planning after each file is what stops a plan being wrong by
        # file five (§6.8). A REVISED plan — files added or dropped — is an
        # epoch boundary (G.7.2); a plan whose statuses or order moved is
        # not, because the architecture summary lists neither.
        #
        # This used to be `maybe_bump_epoch(target=task.path)`: the file just
        # written had of course changed since the snapshot, so the snapshot
        # was rebuilt after EVERY file and the model re-read its whole
        # cached prefix at the start of every task. The target rule is for
        # the file about to be worked on, and the tail now carries every
        # changed file's current line, so no per-file rebuild is needed; the
        # G.7.2 threshold still rebuilds once enough has changed.
        before = {t.path for t in self.plan.tasks}
        self.plan = self.planner.replan(self.plan,
                                        reason=f"after {task.path}")
        changed = {t.path for t in self.plan.tasks} != before
        self.codemap.maybe_bump_epoch(replanned=changed)
        if changed:
            self._emit_plan(revised=True)
        return outcome

    def _queue_repair(self, task: Task, blamed: tuple,
                      covers: Task | None) -> None:
        """Queue the repair of the file ``task`` blames, if it may be.

        Two shapes, one mechanism. A TEST that disagreed with its module
        and was not rewritten: the module is repaired against the test's
        failing output, then the test runs again. A MODULE that ran and
        died inside another file of this project (`Loop._culprit_elsewhere`)
        — or inside a test file that does not parse — that file is
        repaired against the failure, then this module runs again.

        Each file is repaired at most once per session, whoever blames it.
        That is also what bounds a chain of them: Oct 2, 2026, main.py
        died in math3d.py, then (math3d.py repaired) in render.py. A second
        hop is queued like the first; a file blamed again after its one
        repair is reported, not repaired again.
        """
        target = covers if covers is not None else self._task_at(
            str(blamed[0] or ""))
        if target is not None and target.path != task.path \
                and target.path not in self._repaired:
            self._repaired.add(target.path)
            self._repairs.append((target, task, tuple(blamed[1])))
            self.host.emit(
                "status",
                f"{target.path} will be repaired against {task.path}, "
                f"which {'it fails' if covers is not None else 'fails inside it'}",
                {"task": target.path, "test": task.path,
                 "because": "test" if covers is not None else "caller"})
        elif target is not None and target.path in self._repaired \
                and target.path != task.path:
            self.host.emit(
                "warning",
                f"{task.path} still fails inside {target.path}, which has "
                f"already had its one repair this session; fix "
                f"{target.path} by hand and build again",
                {"task": task.path, "culprit": target.path})
        elif target is None and blamed[0]:
            self.host.emit(
                "warning",
                f"{task.path} fails inside {blamed[0]}, which is not a "
                f"file this plan owns, so it is left alone; fix "
                f"{blamed[0]} by hand and build again",
                {"task": task.path, "culprit": str(blamed[0])})

    def _repair_step(self) -> TaskOutcome:
        """Repair a file against what it failed, then run that again.

        The fix task has the culprit's path and purpose and the blaming
        task's path as its `test_path`; its first attempt is a REPAIR
        seeded with the failing output, and its prompt carries the blamer
        itself — a test as the specification, or a caller as the use that
        must work. If the fix verifies, the blamer is verified again
        without generating anything, and that verdict is what the session
        reports for it.
        """
        module, test, failing = self._repairs.pop(0)
        self._where = f"the repair of {module.path}"
        caps = self._capabilities(boundary=f"repair of {module.path}")
        if not caps.loaded:
            raise NoModelLoadedError(
                "capabilities() reports no model at a task boundary")
        blamer_is_test = (test.persona == "tester"
                          or _looks_like_test(test.path))
        # A test blamer becomes the fix's `test_path` (the specification it
        # answers to, and what verifies it). A caller blamer does NOT: a
        # module is not a test file, and naming it as one made the loop
        # report "src/main.py exists, but the test run collected none of
        # its tests". It travels as `caller` instead.
        fix = Task(id=f"{module.id}-fix", path=module.path,
                   purpose=module.purpose,
                   test_path=test.path if blamer_is_test else module.test_path,
                   persona=module.persona, lang=module.lang, atomic=False)
        outcome = self.loop.run_task(fix, request=self.plan.request,
                                     seed=failing,
                                     planned=self._planned_paths(module),
                                     caller="" if blamer_is_test else test.path)
        self._record(fix, outcome, label="REPAIRED" if outcome.ok
                     else "NOT REPAIRED")
        if not outcome.ok:
            return outcome
        again = self.loop.reverify(
            test, because=f"{module.path} was repaired against it")
        self._record(test, again, label=again.label)
        # The failure may now be raised in a THIRD file (`Loop.reverify`
        # names it). Queue that repair the same way; `_queue_repair`
        # refuses any file that has had its one repair already.
        moved = self.loop.blamed.pop(test.path, None)
        if moved and not again.ok:
            self._queue_repair(test, moved, None)
        if blamer_is_test:
            # Only a TEST passing says anything about the module's
            # correctness; a caller that now runs says the caller runs.
            self._credit_module(module, again)
        self.plan = self.plan.replace(test.with_status(
            "done" if again.ok else "failed"))
        return again

    def _task_at(self, path: str) -> Task | None:
        """The plan's task for a path, or None when the plan does not own
        that file (a pre-existing module the session may not rewrite)."""
        if self.plan is None or not path:
            return None
        want = path.replace("\\", "/")
        for t in self.plan.tasks:
            if t.path.replace("\\", "/") == want:
                return t
        return None

    def _record(self, task: Task, outcome: TaskOutcome, *,
                label: str) -> None:
        self.outcomes.append(outcome)
        self.log.event(f"{label} {task.path}",
                       f"{len(outcome.attempts)} attempt(s)"
                       + (f" — {outcome.stopped_because}"
                          if outcome.stopped_because else ""))
        self.journal.log("verify", task=task.path,
                         verify={"ok": outcome.ok,
                                 "attempts": len(outcome.attempts),
                                 "caveats": list(outcome.caveats)},
                         stopped_because=outcome.stopped_because)

    def _check_model(self, caps: ModelCapabilities) -> None:
        """Refuse a model that is not for code; warn about one that is
        unknown or short of context. Before anything is planned."""
        verdict = models.judge(caps.name, caps.context_tokens)
        self.journal.log("model_check", model=caps.name,
                         verdict=verdict.verdict,
                         context_tokens=caps.context_tokens,
                         context_ok=verdict.context_ok,
                         allowed=verdict.verdict != "not_coding"
                         or self.config.allow_any_model)
        if verdict.refuse and not self.config.allow_any_model:
            self.host.emit("warning", verdict.reason,
                           {"phase": "model_check", "model": caps.name,
                            "verdict": verdict.verdict, "refused": True})
            raise NotACodingModelError(verdict.reason)
        if verdict.refuse:
            self.host.emit("warning",
                           verdict.reason + " Continuing because "
                           "allow_any_model is set.",
                           {"phase": "model_check", "model": caps.name,
                            "verdict": verdict.verdict, "refused": False})
        elif verdict.verdict == "unknown":
            self.host.emit("warning", verdict.reason,
                           {"phase": "model_check", "model": caps.name,
                            "verdict": verdict.verdict})
        if not verdict.context_ok:
            self.host.emit("warning", verdict.context_note,
                           {"phase": "model_check", "model": caps.name,
                            "context_tokens": caps.context_tokens})

    def _count_earlier_sessions(self) -> int:
        """How many builds the folder's log already records. Read BEFORE
        this session writes its own header into the same file."""
        try:
            if not self.host.fs.exists(journal_mod.SessionLog.FILENAME):
                return 0
            text = self.host.fs.read(journal_mod.SessionLog.FILENAME) or ""
        except Exception:                                    # noqa: BLE001
            return 0
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        return text.count("\nSESSION ")

    def _previous_build_warning(self, earlier: int) -> None:
        """Say so when this folder already holds an earlier build.

        The tests it left are baselined (`_baseline_existing_tests`); the
        files it left are kept or stubbed over by the planner's own rules.
        What neither does is tell the operator the plain fact: this is a
        rebuild on top of a build, and a clean folder gives a clean answer.
        """
        if earlier <= 0:
            return
        self.host.emit(
            "warning",
            f"This folder already holds {earlier} earlier build "
            f"session{'s' * (earlier != 1)} (see {journal_mod.SessionLog.FILENAME}). "
            f"Files and tests left by them are in play: tests that already "
            f"fail are not charged to this build, and files with real work "
            f"are kept rather than stubbed over. For a clean result, build "
            f"into an empty folder.",
            {"phase": "previous_build", "sessions": earlier})
        self.log.line(f"  NOTE: {earlier} earlier build session(s) in this "
                      f"folder — building on top of them")

    def _baseline_existing_tests(self) -> None:
        """Run the tests that are ALREADY in the folder, before building.

        A project folder is rarely empty: a previous session left its tests
        behind, or the operator's own are there. Whatever fails before a
        line is written is not this build's doing, and it must never be
        charged to the files this build writes — on Aug 8 and Oct 1 a stale
        `tests/test_physics.py` failed every module of the next build in
        turn, each regenerated correctly, each stopped by stagnation.

        The failing test modules go to the loop as `known_failing`; the
        operator is told, by name, in one sentence.
        """
        lang_id = self.config.lang or "python"
        lang = langs.get(lang_id)
        if lang is None or not lang.test_cmd:
            return
        existing = self._existing_test_files(lang)
        if not existing:
            return
        self._where = "checking the tests already in the folder"
        try:
            before = runner.run_tests(lang_id, fs=self.host.fs,
                                      ex=self.host.exec,
                                      timeout=self.config.test_timeout,
                                      whole_suite=True)
        except Exception as exc:                             # noqa: BLE001
            self.host.emit("warning",
                           f"the tests already in the folder could not be "
                           f"run before the build ({_plain_cause(exc)}); "
                           f"failures in them may be charged to new files",
                           {"phase": "baseline"})
            return
        if before.blocked or before.ok or not before.phases:
            self.journal.log("baseline", existing=list(existing),
                             failing=[], ok=bool(before.ok))
            return
        failing = runner.failing_test_modules(before.phases[0].output)
        if not failing:
            failing = [str(p).replace("\\", "/") for p in existing]
        self.loop.config.known_failing = tuple(failing)
        self.journal.log("baseline", existing=list(existing),
                         failing=list(failing), ok=False)
        names = ", ".join(failing[:6]) + (" …" if len(failing) > 6 else "")
        self.host.emit(
            "warning",
            f"{len(failing)} test module(s) were already failing before "
            f"this build started: {names}. They were left by an earlier "
            f"session or written by hand; their failures will be reported "
            f"as caveats, never as errors in the files this build writes. "
            f"For a clean result, build into an empty folder.",
            {"phase": "baseline", "failing": list(failing)})
        self.log.line(f"  BASELINE: {len(failing)} test module(s) already "
                      f"failing before the build: {names}")

    def _existing_test_files(self, lang: Any) -> list[str]:
        """Test files already on disk for this language, dot-dirs skipped."""
        try:
            paths = self.host.fs.list("*")
        except Exception:                                    # noqa: BLE001
            return []
        exts = set(lang.exts or (lang.ext,))
        out: list[str] = []
        for raw in paths:
            rel = str(raw).replace("\\", "/")
            parts = rel.split("/")
            if any(p.startswith(".") for p in parts[:-1]):
                continue
            name = parts[-1]
            ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
            if ext in exts and runner.is_test_path(rel):
                out.append(rel)
        return sorted(out)

    def _credit_module(self, module: Task | None, test_outcome: TaskOutcome
                       ) -> None:
        """A test that VERIFIED clears the "not verified" mark on the module
        it covers.

        The plan writes modules before their tests, so every module is
        "built, not verified" at its own task's end — its test did not exist
        yet. When the test is written and its tests run and pass, the module
        has been verified after all, and the session's summary must say so
        rather than repeat a verdict that was true an hour ago.
        """
        if module is None or not test_outcome.verified:
            return
        self.loop.unverified.discard(module.path)
        # The module's final outcome carries the verdict too, so a host that
        # renders outcomes (not only the summary) shows the same thing.
        last = next((o for o in reversed(self.outcomes)
                     if o.path == module.path), None)
        if last is not None and last.ok and not last.verified:
            self.outcomes.append(dataclasses.replace(
                last, verified=True,
                caveats=last.caveats + (
                    f"verified by {test_outcome.path}, written after it",)))
        self.host.emit("status",
                       f"{module.path}: now verified — {test_outcome.path} "
                       f"ran its tests against it and they passed",
                       {"task": module.path, "verified": True,
                        "by": test_outcome.path})

    def _planned_paths(self, task: Task) -> tuple[str, ...]:
        """The PATHS of the plan's tasks that `task` depends on.

        `depends_on` holds task ids; the codemap's interface block matches
        paths. Resolving here is what lets a file's first attempt see the
        real signatures of what it imports (F9). A test task always gets the
        module it covers, even when the plan's edge is missing.
        """
        if self.plan is None:
            return ()
        by_id = {t.id: t.path for t in self.plan.tasks}
        paths = [by_id[d] for d in task.depends_on
                 if d in by_id and by_id[d] != task.path]
        covers = self._covered_module(task)
        if covers is not None and covers.path not in paths \
                and covers.path != task.path:
            paths.append(covers.path)
        # WHAT ALREADY EXISTS, for a module. The plan's edges come from the
        # stubs' imports, and a stub imports nothing (only the entry point
        # does, by the role rule) — so on 2026-10-01 `render.py` was written
        # with no interface of `physics`, `track` or `math3d` in front of
        # it and invented `player_state.x` and `segment.width` against a
        # class with neither. Every module already built in the project is
        # appended AFTER the real dependencies, so the interface budget
        # serves those first and the rest as far as it reaches. This is
        # visibility, not a dependency: no import is written, no order
        # changes, and `math3d` is not made to depend on `track` by being
        # shown it (the planner's docstring explains why that would be
        # wrong). The header the block carries says "use these names if
        # you call them", which is all it means.
        if not _looks_like_test(task.path) and task.persona != "tester":
            for other in self._built_siblings(task):
                if other not in paths:
                    paths.append(other)
        return tuple(paths)

    #: How many already-built siblings a module is shown beyond its planned
    #: dependencies. The interface block's own token budget is the real
    #: limit; this keeps the "not shown" list short on a large project.
    SIBLINGS_SHOWN = 10

    def _built_siblings(self, task: Task) -> list[str]:
        """Paths of the plan's modules with a REAL body, not stubs, not
        tests, not entry points (nothing imports an entry point)."""
        if self.plan is None:
            return []
        from .loop import _is_our_stub
        out: list[str] = []
        for other in self.plan.tasks:
            if other.id == task.id or other.path == task.path:
                continue
            if _looks_like_test(other.path) or other.persona == "tester":
                continue
            if other.status not in ("done", "failed"):
                continue
            if _is_entry_point(other):
                continue
            try:
                text = self.host.fs.read(other.path)
            except Exception:                            # noqa: BLE001
                continue
            if not text.strip() or _is_our_stub(text):
                continue
            out.append(other.path)
            if len(out) >= self.SIBLINGS_SHOWN:
                break
        return out

    def _covered_module(self, task: Task) -> Task | None:
        """For a test task, the planned module it tests; else None."""
        if self.plan is None or not (task.persona == "tester"
                                     or _looks_like_test(task.path)):
            return None
        for other in self.plan.tasks:
            if other.id != task.id and other.test_path == task.path:
                return other
        stem = task.path.replace("\\", "/").rsplit("/", 1)[-1]
        stem = stem.rsplit(".", 1)[0].removeprefix("test_")
        stem = stem.removesuffix("_test")
        for other in self.plan.tasks:
            name = other.path.replace("\\", "/").rsplit("/", 1)[-1]
            if other.id != task.id and not _looks_like_test(other.path) \
                    and name.rsplit(".", 1)[0] == stem:
                return other
        return None

    def _final_outcomes(self) -> list[TaskOutcome]:
        """The LAST outcome per file, in the order files were first built.

        A module repaired against its test, and a test verified again
        afterwards, each have two outcomes; the later one is the verdict.
        """
        final: dict[str, TaskOutcome] = {}
        for o in self.outcomes:
            final[o.path] = o
        return list(final.values())

    def run(self, request: str = "", profile: dict | None = None
            ) -> list[TaskOutcome]:
        """Plan and build everything. The one-call path for a CLI.

        Cancellation and budget exhaustion both leave resumable state and
        end with a session_end event — a stop is a finished session with an
        honest ending, not an absence of one.

        ANYTHING ELSE THAT GOES WRONG REACHES THE HOST AS A SENTENCE (C6).
        A Port can raise what it likes — a child's output that is not
        UTF-8 escaped from here as a UnicodeDecodeError traceback, and
        `CognitiveCoderError.wrap`, built for exactly this, was never
        called. Now the traceback goes to the journal, an `error` event
        carries the sentence and the journal's path (docs/PORTS.md), the
        session still ends with `session_end`, and what is raised is a
        CognitiveCoderError whose text is the sentence.
        """
        try:
            self._run(request, profile)
        except CognitiveCoderError as exc:
            self._report_failure(exc)
            raise
        except Exception as exc:                          # noqa: BLE001
            wrapped = CognitiveCoderError.wrap(
                exc, self._unexpected_sentence(exc))
            self._report_failure(wrapped)
            raise wrapped from None
        finally:
            self.finish()
        return self.outcomes

    def _run(self, request: str, profile: dict | None) -> None:
        try:
            if request:
                self.start(request, profile)
            while True:
                outcome = self.step()
                if outcome is None:
                    #: NOTHING READY IS NOT THE SAME AS NOTHING LEFT.
                    #:
                    #: A task whose dependency failed can never become ready,
                    #: so the loop ends and the file is simply never
                    #: mentioned. On a real build that showed as "[build 6/6]"
                    #: against a seven-file plan, with src/main.py absent from
                    #: the log entirely — it was waiting on src/render.py,
                    #: which had failed three attempts earlier.
                    #:
                    #: Not building it is correct. Not saying so is the same
                    #: mistake the installer made: an absence nobody can see.
                    for task, blockers in (self.plan.blocked()
                                           if self.plan else []):
                        self.host.emit(
                            "warning",
                            f"{task.path} was never attempted — it needs "
                            f"{', '.join(blockers)}, which did not build",
                            {"task": task.path, "blocked_by": blockers})
                        self.journal.log("blocked", task=task.path,
                                         blocked_by=blockers)
                        self.log.event(f"SKIPPED {task.path}",
                                       f"needs {', '.join(blockers)}, "
                                       f"which did not build")
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
                self._where = "the review"
                try:
                    self.review()
                except Cancelled:
                    self.journal.log("cancel", where="review")

    def _unexpected_sentence(self, exc: BaseException) -> str:
        """What happened, where, and where the details are — no type names."""
        where = f" while working on {self._where}" if self._where else ""
        return (f"The session stopped unexpectedly{where}: "
                f"{_plain_cause(exc)}. Work already verified has been kept "
                f"and anything half-applied was rolled back. The details are "
                f"in the journal at {self.journal.path}.")

    def _report_failure(self, exc: CognitiveCoderError) -> None:
        """Journal the traceback; hand the host the sentence and a pointer.

        Never raises: this runs on the way out of a failure, and a second
        failure here would replace the sentence with a traceback.
        """
        try:
            self.journal.error(exc.sentence, exc.detail,
                               journal=self.journal.path, where=self._where)
        except Exception:                                 # noqa: BLE001
            pass
        self.host.emit("error", exc.sentence,
                       {"journal": self.journal.path, "where": self._where})

    def review(self, *, use_model: bool = True) -> str:
        """The review stage, AFTER everything builds and its tests pass (§4.3).

        Reviewing code that does not compile spends tokens on a moot point,
        so this refuses to run on a session that did not finish cleanly — and
        says so rather than producing a document that looks authoritative
        about code nobody has verified.
        """
        from . import review as review_mod

        final = self._final_outcomes()
        done = [o for o in final if o.ok]
        if not done:
            self.host.emit("warning",
                           "Nothing verified, so there is nothing to review. "
                           "A review of code that does not build is a review "
                           "of a moot point.")
            return ""

        merged = review_mod.ReviewResult()
        try:
            architecture = self.codemap.prefix_block()
        except Exception:                                # noqa: BLE001
            architecture = ""
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
                use_model=use_model, architecture=architecture)
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
            # A scanner that was installed but timed out or printed
            # something unreadable. Dropped here, it vanished from the
            # merged result and the document said "No security findings"
            # on the strength of a tool that never produced an answer.
            for note in one.scanners_failed:
                if note not in merged.scanners_failed:
                    merged.scanners_failed.append(note)

        # What the tests actually covered. The build line used to read
        # "N of M file(s) built and their tests ran" whatever the tests
        # did, and the all-clear was printed on the strength of it.
        untested = [p for p in self._untested(final) if p in
                    {o.path for o in done}]
        zero = self._unverified(final)
        sources = [o.path for o in done if not _looks_like_test(o.path)]
        no_evidence = sorted(set(untested) | set(zero))
        tests_ran = bool(sources) and len(no_evidence) < len(sources)
        verification = f"{len(done)} of {len(final)} file(s) built"
        if not no_evidence:
            verification += " and their tests ran"
        elif not tests_ran:
            verification += ", and no tests ran for any of them"
        else:
            verification += (f"; {len(no_evidence)} of them have no tests "
                             f"that ran")

        self.journal.log("review", findings=len(merged.findings),
                         high=len(merged.high),
                         scanners=merged.scanners_run,
                         scanners_failed=merged.scanners_failed,
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
            build_summary=verification,
            tests_ran=tests_ran,
            untested=no_evidence if tests_ran else (),
            #: The reviewer only ever sees committed files, so without this it
            #: cannot tell a clean build from a collapsed one — and reports the
            #: second as the first.
            unfinished=[o.path for o in final if not o.ok],
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
        final = self._final_outcomes()
        unverified = self._unverified(final)
        untested = self._untested(final)
        self.journal.log(
            "session_end",
            #: "ok" means VERIFIED. A file whose test file exists and ran
            #: nothing is not, however green its build was.
            ok=bool(final) and all(o.ok for o in final) and not unverified,
            files=[o.path for o in final if o.ok],
            failed=[o.path for o in final if not o.ok],
            unverified=unverified,
            untested=untested,
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

    def _unverified(self, final: list[TaskOutcome]) -> list[str]:
        return [o.path for o in final
                if o.ok and o.path in self.loop.unverified]

    def _untested(self, final: list[TaskOutcome]) -> list[str]:
        """Built-and-ran files with no test file: a weaker claim, counted.

        Not a failure — some files genuinely have no tests — but "verified"
        cannot be said of them, and the summary says how many there are
        rather than letting a green line imply otherwise.
        """
        out = []
        for o in final:
            if not o.ok or _looks_like_test(o.path):
                continue
            task = self.plan.task(o.task_id.removesuffix("-fix")) \
                if self.plan else None
            test_path = task.test_path if task else ""
            if not test_path or not self.host.fs.exists(test_path):
                out.append(o.path)
        return out

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
        directory = (config or SessionConfig()).journal_dir
        state = journal_mod.resume_state(host.fs, session_id, directory)
        if not state["events"]:
            raise FileNotFoundError(
                f"There is no journal for session {session_id}, so there is "
                f"nothing to resume. Start a new session instead.")
        rows = list(journal_mod.read_jsonl(
            host.fs, f"{directory}/{session_id}.jsonl"))
        if _is_preview(rows):
            raise CognitiveCoderError(
                f"Session {session_id} was a preview — it planned and "
                f"stopped, and nothing was built from it, so there is "
                f"nothing to resume. Start a new session with the same "
                f"request instead.")
        session = cls(host, config=config, session_id=session_id)
        session.codemap.index_project()

        plan_data = state.get("plan") or {}
        files = list(plan_data.get("files") or [])
        if files:
            session.plan = session.planner.derive_order(Plan(
                request=state["request"],
                tasks=tuple(session._resumed_tasks(plan_data, state))))
        #: Files the skeleton declined to stub over. Without this the first
        #: replan after a resume would mark the operator's own file "done"
        #: because it has a body — the body that was there all along.
        for row in rows:
            if row.get("event") == "skeleton":
                kept = (row.get("data") or {}).get("kept") or []
                session.planner.kept_as_found = set(kept)
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

    def _resumed_tasks(self, plan_data: dict, state: dict) -> list[Task]:
        """The plan as it was journaled — purpose, persona, language, test
        pairing and atomicity — with the status the journal proves.

        A journal written before the `plan` event carried `tasks` has only
        paths. For those the rest is INFERRED from the path, and the
        inference is the conservative one: a test file is a tester's, the
        language is the extension's, and a test path is kept only when that
        file is planned or on disk.
        """
        files = list(plan_data.get("files") or [])
        recorded = {d.get("path"): d for d in plan_data.get("tasks") or []
                    if isinstance(d, dict)}
        done = set(state["done"])
        planned = {p.replace("\\", "/").lower() for p in files}
        tasks = []
        for i, path in enumerate(files):
            meta = recorded.get(path) or {}
            is_test = _looks_like_test(path)
            if meta:
                test_path = str(meta.get("test_path") or "")
            else:
                guess = "" if is_test else self.planner.test_path_for(path)
                keep = guess and (guess.lower() in planned
                                  or self.host.fs.exists(guess))
                test_path = guess if keep else ""
            tasks.append(Task(
                id=f"t{i + 1}", path=path,
                purpose=str(meta.get("purpose")
                            or f"(resumed) part of: {state['request']}"),
                test_path=test_path,
                persona=str(meta.get("persona")
                            or ("tester" if is_test else "engineer")),
                lang=str(meta.get("lang") or langs.id_for_path(path)
                         or self.config.lang),
                atomic=bool(meta.get("atomic", False)),
                status="done" if path in done else "pending",
                attempts=state["attempts"].get(path, 0)))
        return tasks

    @staticmethod
    def previous_sessions(host: Host,
                          directory: str = ".cc_journal") -> list[str]:
        """Sessions that can be resumed — which excludes previews."""
        out = []
        for session_id in journal_mod.sessions(host.fs, directory):
            rows = journal_mod.read_jsonl(
                host.fs, f"{directory}/{session_id}.jsonl")
            if not _is_preview(rows):
                out.append(session_id)
        return out

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
        final = self._final_outcomes()
        unverified = self._unverified(final)
        if unverified:
            lines.append(f"[verify]    {len(unverified)} file(s) NOT "
                         f"verified — their test file ran no tests: "
                         f"{', '.join(unverified)}")
        built = [o for o in final if o.ok and not _looks_like_test(o.path)]
        untested = self._untested(final)
        if untested:
            lines.append(f"[tests]     {len(untested)} of {len(built)} "
                         f"file(s) have no tests — built and run, not "
                         f"tested: {', '.join(untested)}")
        lines.append(f"[codemap]   {self.codemap.stats().one_line()}")
        lines.append(f"[journal]   {self.journal.summary()}")
        # Every snapshot after the first is a planned full read of the
        # prompt; the first call of the session is counted inside.
        lines.append(f"            "
                     f"{self.journal.cache_health(self.codemap.store.epoch)}")
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
        # The same model reloaded with a different context size is a new
        # KV cache too — the host reloaded it — so it is the same epoch
        # boundary as a swap. Only the name used to count, and a host's
        # "raise n_ctx" button left the old prefix snapshot in force.
        resized = (caps.name == self._model and bool(self._model)
                   and caps.context_tokens != self._context)
        if caps.name != self._model or resized:
            if self._model:
                # The model changed under us. The KV cache and any
                # prompt-prefix state died with the old model, so the cached
                # prefix is rebuilt — that is the whole of the core's
                # involvement in a swap (§0.1 consequence 2).
                what = (f"{caps.name} was reloaded with a "
                        f"{caps.context_tokens:,}-token context (it was "
                        f"{self._context:,})" if resized else
                        f"The loaded model changed from "
                        f"{self._model or 'none'} to {caps.name or 'none'}")
                self.host.emit(
                    "warning",
                    f"{what}. The cached prompt prefix has been rebuilt; "
                    f"the next call will be slower.",
                    {"was": self._model, "now": caps.name,
                     "context_was": self._context,
                     "context_now": caps.context_tokens})
                self.codemap.maybe_bump_epoch(model_changed=True)
            self.journal.log("epoch", model=caps.name, boundary=boundary,
                             context_tokens=caps.context_tokens,
                             supports_tools=caps.supports_tools,
                             is_remote=caps.is_remote)
            self._model = caps.name
            self._context = caps.context_tokens
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


def _plain_cause(exc: BaseException) -> str:
    """An exception, described in words an operator can use (C6)."""
    if isinstance(exc, UnicodeError):
        return ("a program produced output that is not valid text in the "
                "expected encoding")
    if isinstance(exc, MemoryError):
        return "the machine ran out of memory"
    if isinstance(exc, RecursionError):
        return "something was nested too deeply to process"
    if isinstance(exc, OSError):
        why = getattr(exc, "strerror", "") or "no reason was given"
        return f"the operating system refused an operation ({why})"
    return "the engine hit a failure it did not expect"


def _is_preview(rows) -> bool:
    """Was this journal written by `preview()` — a plan nothing was built
    from?"""
    return any(row.get("event") == "preview" for row in rows)


def _new_id() -> str:
    """A session id. Never used in a prompt — see G.7.1.

    Stated here because it is exactly the kind of value that ends up in a
    prompt preamble by accident, and one varying token at position 40
    silently discards 30k tokens of cached work.
    """
    return f"cc-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
