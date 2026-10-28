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

from . import fim as fim_mod
from . import journal as journal_mod
from . import (
    langs,
    models,
    personas,
    profiles,
    resumable,
    runner,
    sizing,
    speclint,
)
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
from .planner import Planner, _looks_like_test
from .planner import is_entry_point as _is_entry_point
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
    #: the deterministic floor (loop.py, 2026-10-22): format before the
    #: write; certain-bug lint codes fail the attempt before it is written
    format_before_write: bool = True
    lint_gate: tuple[str, ...] = ("F821", "F822", "F823", "F811", "F631",
                                  "F632", "F702", "F706", "E999")
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
    #: Ask the model for the INTERFACE SKELETON before any body is written
    #: (§4.2 step 2): real signatures and shared data types for every
    #: planned Python module, one call. Costs one generation; without it
    #: each file is written blind to the others (see `interfaces.py`).
    pin_interfaces: bool = True
    #: Tests before module bodies (F2). None: when the request asks for it.
    #: True/False: always/never. Only for modules whose interface was
    #: pinned; otherwise each test is written right AFTER its module.
    test_first: bool | None = None
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
    #: DIFFUSION MODE (docs/DIFFUSION_MODE.md). "auto": when the loaded model
    #: fills a canvas (DiffusionGemma behind a host that says so), repairs
    #: rewrite the hole the error is in and pinned files are written body by
    #: body with every signature held; "off": the whole-file path always.
    #: Nothing changes for a model that cannot fill a canvas.
    diffusion: str = "auto"
    #: Canvases per repair attempt (the best usable one goes to the tests).
    diffusion_candidates: int = 2
    #: Plan a pinned file's bodies in plain notes before writing them.
    diffusion_notes: bool = True
    #: THE REQUEST'S HARD REQUIREMENTS (constraints.py): the sentences with
    #: MUST / NEVER / DO NOT, word for word, pinned in every prompt's cached
    #: prefix for the whole session.
    pin_constraints: bool = True
    #: THE MUTATION CHECK (mutation.py), after a build: deliberate bugs put
    #: into a COPY of each verified Python module, its tests run against
    #: each; a bug the tests do not notice is reported as a WARNING. Never
    #: changes the project or the verdict.
    mutation_check: bool = True
    mutation_max: int = 5                # mutants per file
    mutation_budget_s: float = 120.0     # for the whole check
    #: SPEC VECTORS (specvectors.py): the examples the request states
    #: outright (`` `"1-3"` -> `(1, 2, 3)` ``, `` `SUBTICKS_PER_TURN: int =
    #: 10` ``) lifted by rule into a test file per module that the harness
    #: owns and the runner runs before the module's own tests. A model's
    #: test that contradicts one is wrong by definition.
    spec_vectors: bool = True
    #: THE JUDGE (adjudicate.py): when a model's test disagrees with the
    #: module it tests, ask which one the request supports — and check the
    #: quote the answer rests on — before blaming the module. Ten of the
    #: sixteen failing tests of the 2026-10-16 build were the test's fault,
    #: and the rule "the code must change" had no way to say so.
    adjudicate: bool = True
    #: HYBRID MODE (hybrid.py, docs/HYBRID.md): with a host that offers a
    #: roster of models (`ports.ModelRoster`), one model writes, the other
    #: repairs, the first finishes — three phases, two swaps. "auto": when
    #: the roster has both a chat and a canvas model; "off": never;
    #: "ar-first" / "diffusion-first": that order, whatever is loaded now.
    hybrid: str = "auto"
    #: Roster names to use for the roles; "" picks by kind.
    hybrid_writer: str = ""
    hybrid_repairer: str = ""
    #: HOW MUCH OF THE REQUEST EACH FILE'S PROMPTS CARRY (specslice.py), in
    #: characters. Whole when it fits; otherwise the file's own section of
    #: the specification, its hard requirements and the lines naming it.
    #: The old fixed cut (2,000 characters, every repair and every
    #: diffusion prompt) dropped every formula of a 15,000-character spec.
    request_chars_first: int = 16_000
    request_chars_repair: int = 8_000
    #: THE SPEED PROBE: one short generation after the model check, and
    #: after every model swap, with its tokens per second and prefill time
    #: written to the log. Devstral ran the whole 2026-10-16 build at 3–5
    #: tok/s — spilled into system RAM — and nothing said so until the
    #: journal was read two and a half hours later. None: on for a real
    #: model port, never for the scripted test doubles (`ScriptedLLM`,
    #: `NullLLM`), whose replies are counted. True/False: always/never.
    speed_probe: bool | None = None
    #: THE DROPPED-CODE GUARD (keep.py): a reply for a file that already
    #: had real code is refused before it is written when it leaves out a
    #: definition the request does not name, or has a placeholder where
    #: code was (`# ... rest unchanged`).
    keep_existing: bool = True
    #: NARROWED EDITS (symedit.py, F5): "auto" — a Python repair returns
    #: only the definitions it changes when the file is over
    #: `narrow_min_lines` lines or after the same attempt twice, and any
    #: repair or change does when the whole file does not fit the context;
    #: "always" — every repair and change of an existing Python file;
    #: "off" — the whole file, always.
    narrow: str = "auto"
    #: 150 → 40 on 2026-10-22, from the journals: shown its whole file plus
    #: the error, Devstral returned the file BYTE-IDENTICAL on attempt 2
    #: in every phase of the 10-18 build (2,696→2,696, 2,482→2,482,
    #: 2,460→2,460 tokens; ai.py at 880 tokens too) — ten minutes per
    #: file-task at 3–5 tok/s for no information. A whole-file second
    #: attempt only earns its tokens on a file too small to narrow.
    narrow_min_lines: int = 40
    #: Each located error's line marked, with the def/class lines around
    #: it, in every repair prompt (diagnostics.where_block).
    mark_errors: bool = True
    #: REGRESSION MEMORY (F10): a repair is told how the same error was
    #: fixed before in this project.
    recall_fixes: bool = True
    #: Opt-in: one short "what is wrong and what to change" call before
    #: each repair, shown to the repair (Aider's architect/editor split).
    diagnose_first: bool = False
    #: PLAN FOR THE MODEL'S REACH (sizing.py, 2026-10-27). The planner is
    #: told to size each file for about `max_module_lines`, the architect
    #: and every prompt to keep a function under `max_function_lines`; a
    #: pinned stub that will plainly outgrow the first is said at the
    #: skeleton, and a verified file over either is a `size` journal event,
    #: a build-log line and a report caveat — never a refusal. 0 is off.
    max_module_lines: int = sizing.DEFAULT_MODULE_LINES
    max_function_lines: int = sizing.DEFAULT_FUNCTION_LINES
    #: THE VALUES AT THE FAILURE (testrun.py, 2026-10-27): the locals of
    #: the frame that failed and of the test, printed by the harness-owned
    #: test runner and shown to every repair under the error.
    capture_values: bool = True
    #: THE TYPE CHECKER'S GATE (typecheck.py, 2026-10-27): the codes that
    #: fail an attempt. None: `typecheck.GATE_CODES` for a real model port
    #: and off for the scripted test doubles (whose suites would otherwise
    #: have to script replies to findings the checker makes about their
    #: fixtures); () switches it off; a tuple names the codes. "A real
    #: model port" means one of the engine's own providers; a host with
    #: its own port (ATK) asks for the gate in its SessionConfig.
    type_gate: tuple[str, ...] | None = None
    #: THE MODEL PROFILE (profiles.py, 2026-10-27): a host may hand the
    #: session one (a dict of ModelProfile's fields); otherwise the
    #: calibrated file for the loaded model, else the built-in table by
    #: name. Its samplers go with every call the port takes them on, its
    #: temperature is the ladder's base, its edit format sets `narrow`
    #: when that is still "auto", and its thinking switch is thrown off
    #: for every parsed reply. `profile_dir` is where calibrated files
    #: are looked for (None: `bench/profiles/` beside the engine).
    profile: dict | None = None
    profile_dir: str | None = None
    #: FIM MODE (fim.py, docs/FIM_MODE.md, 2026-10-27). "auto": when the
    #: host fills in the middle and the probe at the model check finds
    #: the loaded model fills a known hole correctly, a repair rewrites
    #: only the hole its error is in; "on": the same, probing even the
    #: scripted test doubles; "off": never.
    fim: str = "auto"
    fim_candidates: int = 2
    #: SAMPLE AGAIN BEFORE REPAIRING (2026-10-27): extra samples of the
    #: same prompt when a first attempt fails, the best kept, the
    #: definitions they disagree on shown to the repair. None: 1 for one
    #: of the engine's own providers, 0 for any other port (a scripted
    #: double, a host's own port — ATK asks for it in its SessionConfig);
    #: 0 switches it off.
    resample: int | None = None
    resample_max_lines: int = 150
    #: THE INTERVIEW BEFORE THE BUILD (interview.py, 2026-10-27): a host
    #: that has a person to ask sets this and calls `Session.interview`
    #: before `run`, then runs the request the answers were applied to.
    #: The core never asks on its own — a build with nobody there would
    #: stall — so False is the default and headless runs never see it.
    interview: bool = False
    #: CONTINUE (resumable.py, 2026-10-23): before a continued build writes
    #: anything, the files already on disk are checked when anything may
    #: have changed since the last run — a file the operator fixed by hand
    #: that now passes is KEPT and counted, one that broke is REPAIRED from
    #: what is there rather than regenerated, and work a crash left behind
    #: is checked instead of thrown away. Nothing changed (the last run
    #: ended cleanly and every file hashes the same) costs nothing.
    reverify_on_continue: bool = True


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
                              self.skill_load.block(),
                              sizing.rule_text(self.config.max_module_lines,
                                               self.config.max_function_lines))
            if part)
        self.prompts = personas.PromptBuilder(
            model_system_prompt=self.config.model_system_prompt,
            conventions=conventions)
        self.planner = Planner(host, codemap=self.codemap,
                               journal=self.journal, prompts=self.prompts,
                               lang=self.config.lang,
                               max_files=self.config.max_files,
                               patcher=self.patcher,
                               pair_tests=self.config.pair_tests,
                               pin_interfaces=self.config.pin_interfaces,
                               max_tokens=self.config.max_tokens,
                               test_first=self.config.test_first,
                               max_module_lines=self.config.max_module_lines,
                               max_function_lines=self.config.max_function_lines)
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
                format_before_write=self.config.format_before_write,
                lint_gate=tuple(self.config.lint_gate or ()),
                wall_clock_s=self.config.per_task_s,
                timeouts=Timeouts(build=self.config.build_timeout,
                                  run=self.config.run_timeout,
                                  test=self.config.test_timeout),
                canvas=self.config.diffusion,
                canvas_candidates=self.config.diffusion_candidates,
                canvas_notes=self.config.diffusion_notes,
                adjudicate=self.config.adjudicate,
                request_chars_first=self.config.request_chars_first,
                request_chars_repair=self.config.request_chars_repair,
                keep_existing=self.config.keep_existing,
                narrow=self.config.narrow,
                narrow_min_lines=self.config.narrow_min_lines,
                mark_errors=self.config.mark_errors,
                recall_fixes=self.config.recall_fixes,
                diagnose_first=self.config.diagnose_first,
                capture_values=self.config.capture_values,
                type_gate=self._resolve_type_gate(host),
                fim=("off" if str(self.config.fim or "auto").lower() == "off"
                     else "auto"),
                fim_candidates=self.config.fim_candidates,
                resample=self._resolve_resample(host),
                resample_max_lines=self.config.resample_max_lines))
        #: The readable log is the Loop's too — it is where the verify
        #: phases happen, and their output is the whole point of the file.
        self.loop.log = self.log

        self.plan: Plan | None = None
        self.outcomes: list[TaskOutcome] = []
        #: Modules to repair against a test that failed on them:
        #: (module task, test task, the test's failing diagnostics). One
        #: pass per module per session — a second would be a loop.
        self._repairs: list[tuple[Task, Task, tuple]] = []
        #: Tests the judge ruled WRONG after a module repair came back
        #: unchanged: (test task, module task, verdict, failing).
        self._test_repairs: list[tuple[Task, Task, Any, tuple]] = []
        self._repaired: set[str] = set()
        #: What the session was doing, for the sentence a failure earns.
        self._where = ""
        self.profile: dict = {}
        #: What `Planner.skeleton` returned: files, pinned, fallback, note.
        self.skeleton_result: dict = {}
        self.last_review: Any = None
        #: The hard requirements pinned this session (constraints.py).
        self.constraints: Any = None
        #: One MutationReport per checked file (mutation.py).
        self.mutation: list = []
        #: Files over the size rule (sizing.Measure), for the report.
        self.size_findings: list = []
        #: The model profile in force (profiles.ModelProfile), from the
        #: model check on; the loop and the planner read it from here.
        self.model_profile: Any = None
        self._model: str = ""
        self._context: int = 0
        self._started = 0.0
        self._finished = False
        #: The sentence a crash ended the session with, or "". A session
        #: that stopped unexpectedly is never `ok`, whatever its files say:
        #: the 2026-10-16 DiffusionGemma build crashed at 870 s and its
        #: `session_end` read `ok: true` from the two files built before.
        self._failed: str = ""
        #: Hybrid mode's roles (hybrid.Roles) or None; the phase running.
        self._roles: Any = None
        self._phase: Any = None
        #: {path: the diagnostics a re-opened task's first attempt repairs
        #: against} — how a later phase hands a failed file its errors.
        self._seeds: dict[str, tuple] = {}
        #: CONTINUE. `resume()` sets `_resumed` to the journal's summary of
        #: the build being picked up (resumable.BuildState); `run()` then
        #: prepares the continuation once, on the worker thread, instead of
        #: planning. `_verified_paths` is what the board may draw green.
        self._resumed: Any = None
        self._continue_prepared = False
        self._verified_paths: set[str] = set()
        #: THE GRACEFUL STOP. `cancel()` stops at the next safe point, which
        #: can be mid-generation — on a 24B model at 4 tok/s a file four
        #: minutes in is four minutes thrown away. This one lets the file in
        #: hand finish (verified or not) and stops before the next.
        self.stop_token = Cancel()

    # ------------------------------------------------------------------
    # the interview before the build (interview.py)
    # ------------------------------------------------------------------
    def interview(self, request: str, *, max_questions: int = 5) -> Any:
        """The decisions the request leaves open, as questions for the
        operator — one model call, no file written. The host shows them,
        collects answers (`interview.answer`), and runs
        `interview.apply(request)`; the decisions become hard requirements
        pinned in every prompt of the session."""
        from . import interview as interview_mod
        self._where = "the interview"
        caps = self._capabilities(boundary="the interview")
        if self.model_profile is None:
            # the model check has not run yet: the profile is needed so
            # the structured ask gets the right samplers and no thinking
            self.model_profile = profiles.profile_for(
                caps.name, given=self.config.profile,
                directory=self.config.profile_dir)
        result = interview_mod.ask(self, request, max_questions=max_questions,
                                   cancel=self.cancel_token)
        self.log.event("INTERVIEW", interview_mod.summary(result))
        for q in result.questions:
            self.log.line(f"  ? {q.question}  [{' / '.join(q.options)}; "
                          f"default: {q.default}]")
        if result.problem:
            self.log.line(f"  NOTE: {result.problem}")
        self.host.emit("status", interview_mod.summary(result),
                       {"phase": "interview",
                        "questions": [q.as_dict() for q in result.questions],
                        "problem": result.problem})
        return result

    def _resolve_resample(self, host: Host) -> int:
        """`resample` as the loop gets it: the config's number, or one
        extra sample for a real model port and none for the scripted
        doubles — the speed probe's rule, for the same reason."""
        if self.config.resample is not None:
            return max(0, int(self.config.resample))
        return 1 if _real_provider(host.llm) else 0

    def _resolve_type_gate(self, host: Host) -> tuple[str, ...]:
        """`type_gate` as the loop gets it: the config's tuple, or the
        default codes for a real model port and nothing for the scripted
        doubles (the speed probe's rule, for the same reason)."""
        if self.config.type_gate is not None:
            return tuple(self.config.type_gate)
        from . import typecheck
        return tuple(typecheck.GATE_CODES) if _real_provider(host.llm) \
            else ()

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
        self._apply_skill_triggers(request)
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
                        "lang": self.config.lang,
                        "diffusion": self._diffusion_state(caps),
                        "fim": fim_mod.describe_state(
                            self.loop.fim_ok, self.loop.fim_why,
                            self.config.fim_candidates)})
        self._speed_probe(caps, after="the model check")
        caps = self._plan_hybrid(caps)
        self._pin_constraints(request, source="the request")
        self._git_warning()
        self._spec_lint(request)

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
        # BEFORE the skeleton, which is now imported for real: a stub that
        # imports pygame must find pygame, or the check reports an
        # environment matter as a note instead of a fact about the design.
        self._preinstall(request)
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
            self.skeleton_result = dict(result)
            self._log_skeleton(result)
            # The epoch snapshot was taken at start, BEFORE the skeleton
            # existed. The planner indexes each stub as it writes it, but
            # the cached prefix serves the snapshot — so the first file was
            # generated against a project with no architecture in it, which
            # is the one thing skeleton-first exists to provide. A new plan
            # is an epoch boundary (G.7.2); this is that boundary.
            self.codemap.maybe_bump_epoch(replanned=True)
        self._write_spec_vectors(request)
        self._emit_plan(revised=False)
        return self.plan

    def _apply_skill_triggers(self, text: str) -> None:
        """Load the skills whose `triggers:` the request names (skills.py)
        — once, before the first prompt, so the cached prefix is fixed for
        the session from here on exactly as before."""
        if not self.skill_load.held:
            return
        self.skill_load = self.skill_load.for_request(text, self.host.events)
        self.prompts.conventions = "\n\n".join(
            part for part in (self.config.conventions,
                              self.skill_load.block()) if part)
        loaded = [s for s in self.skill_load.skills if s.triggers]
        if loaded:
            self.host.emit("status", "skills loaded for this request: "
                           + ", ".join(f"{s.name} (`{s.triggered_by(text)}`)"
                                       for s in loaded))

    def _write_spec_vectors(self, request: str) -> None:
        """The request's own examples, as tests the harness owns
        (specvectors.py) — written once, after the skeleton, through the
        patcher like any other write, and said in the log."""
        self.loop.spec_tests = {}
        if not self.config.spec_vectors or self.plan is None:
            return
        if (self.config.lang or "python") != "python":
            return
        from . import specvectors
        paths = [t.path for t in self.plan.tasks]
        try:
            vectors = specvectors.extract(request, paths)
        except Exception as exc:                          # noqa: BLE001
            self.log.line(f"  spec vectors  none — the request could not "
                          f"be read for them ({exc})")
            return
        if not vectors:
            self.log.line("  spec vectors  none found (no `literal` -> "
                          "`literal` example beside a named function, and "
                          "no UPPER_CASE constant with a value)")
            return
        layout = self.planner.observe_layout()
        test_dir = layout.get("test_dir") or self.planner.test_dir or "tests"
        grouped = specvectors.by_module(vectors)
        edits: list = []
        files: dict[str, str] = {}
        from .types import Edit
        for module, items in grouped.items():
            test_path = specvectors.test_path_for(module, test_dir)
            files[module] = test_path
            edits.append(Edit(path=test_path, kind="whole",
                              new=specvectors.test_file(module, items),
                              note="spec vectors — the request's own "
                                   "examples as tests"))
        init = f"{test_dir}/__init__.py"
        try:
            if not self.host.fs.exists(init):
                edits.append(Edit(path=init, kind="create", new="",
                                  note="make the test folder a package"))
        except Exception:                                 # noqa: BLE001
            pass
        written = self._write_vector_files(edits)
        if not written:
            self.log.line("  spec vectors  not written — the change was not "
                          "approved")
            return
        self.loop.spec_tests = {m: p for m, p in files.items() if p in written}
        self.log.rule("SPEC VECTORS — examples the request states outright, "
                      "as tests no model may change")
        for module, items in grouped.items():
            if files[module] not in written:
                continue
            self.log.line(f"  {files[module]}  ({len(items)} for {module})")
            for v in items:
                self.log.line(f"    - {v.call} == {v.expected}")
        self.journal.log("constraints", vectors=len(vectors),
                         vector_files=sorted(set(self.loop.spec_tests.values())),
                         items=[v.describe() for v in vectors])
        if self.codemap is not None:
            for path in written:
                try:
                    self.codemap.reindex_after_write(path)
                except Exception:                         # noqa: BLE001
                    pass
        self.host.emit("status",
                       f"{len(vectors)} spec vector"
                       f"{'' if len(vectors) == 1 else 's'} lifted from the "
                       f"request into {len(self.loop.spec_tests)} test "
                       f"file(s) the harness owns; they run before each "
                       f"module's own tests",
                       {"phase": "spec_vectors",
                        "files": sorted(set(self.loop.spec_tests.values())),
                        "count": len(vectors)})

    def _write_vector_files(self, edits: list) -> list[str]:
        """One transaction, one approval. Paths written, or []."""
        if not edits:
            return []
        summary = (f"spec vectors: {len(edits)} test file"
                   f"{'s' * (len(edits) != 1)} from the request's examples")
        try:
            preview = "\n".join(self.patcher.preview([e]) for e in edits)
            approved = bool(self.patcher.approval.approve_diff(summary,
                                                               preview))
        except Exception:                                 # noqa: BLE001
            approved = False
        if not approved:
            return []
        tx = self.patcher.begin("spec-vectors", atomic=False)
        try:
            results = tx.apply(edits, approve=False, summary=summary)
        except Exception:                                 # noqa: BLE001
            tx.rollback("the spec vectors could not be written")
            return []
        tx.commit(verified=False)
        return [r.path for r in results
                if r.ok or "no change" in (r.reason or "")]

    def _log_skeleton(self, result: dict) -> None:
        """What the skeleton pinned, and the order the build will take.

        The order matters now in a way it did not: a test is written beside
        its module — first, when the request asks for that — and the PLAN
        list above is the planner's proposal, not the order of the build.
        """
        pinned = result.get("pinned") or []
        fallback = result.get("fallback") or {}
        reply = self.planner.interface_reply
        if reply is not None:
            # The reply itself, as every generation's is: the snapshot holds
            # the stubs as written, this holds what the model said.
            self.log.generation(
                "the interface skeleton", 1,
                temperature=personas.ARCHITECT.temperature,
                seed=self.config.seed, tokens_in=reply.tokens_in,
                tokens_out=reply.tokens_out, prompt_ms=reply.prompt_ms,
                decode_ms=getattr(reply, "decode_ms", 0), text=reply.text)
        if pinned:
            self.log.event("INTERFACES",
                           f"pinned for {len(pinned)} file(s) before any "
                           f"body was written: {', '.join(pinned)}")
        for path, why in fallback.items():
            self.log.line(f"  NOTE: {path} keeps a plain stub — {why}")
        for note in getattr(self.planner, "import_notes", ()) or ():
            self.log.line(f"  NOTE: {note}")
        for path, note in (getattr(self.planner, "size_notes", {})
                           or {}).items():
            self.log.line(f"  SIZE: {note}")
            self.host.emit("warning", f"skeleton: {note}",
                           {"phase": "skeleton", "size": True, "task": path})
        first = self._tests_first()
        covered = self.planner._covered_modules(self.plan)
        late = [t.path for t in self.plan.tasks
                if t.id in covered and t.path not in first]
        if late and self.planner.wants_test_first(self.plan.request):
            which = ("no module's interface could be pinned, so each test "
                     "is written right after its module instead" if not first
                     else f"{', '.join(late)} "
                          f"{'is' if len(late) == 1 else 'are'} written "
                          f"after {'its module' if len(late) == 1 else 'their modules'}"
                          f", whose interface could not be pinned")
            note = f"the request asks for tests before the module bodies, " \
                   f"but {which}"
            self.plan = dataclasses.replace(
                self.plan, caveats=self.plan.caveats + (note,))
            self.host.emit("warning", note, {"phase": "plan"})
            self.log.line(f"  CAVEAT: {note}")
        order = []
        for i, t in enumerate(self.plan.tasks, 1):
            mark = "  (written first)" if t.path in first else ""
            order.append(f"  {i}. {t.path}{mark}")
        self.log.line("  build order:")
        for line in order:
            self.log.line("  " + line)

    def _tests_first(self) -> set[str]:
        """Paths of the plan's tests that are written before their module."""
        if self.plan is None:
            return set()
        covered = self.planner._covered_modules(self.plan)
        ids = self.planner._tests_written_first(self.plan, covered)
        return {t.path for t in self.plan.tasks if t.id in ids}

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
        data = {"tasks": tasks, "revised": revised,
                "request": self.plan.request,
                "caveats": list(self.plan.caveats)}
        if self._resumed is not None:
            # A CONTINUED build's board starts with files already done, and
            # "done" alone cannot say whether a test proved one: the board
            # drew every one of them grey ("built, not checked") — including
            # the files whose tests had passed — so the first thing an
            # operator saw after pressing Continue was their work demoted.
            for row in tasks:
                row["verified"] = row["path"] in self._verified_paths
            data["continued"] = True
        self.host.emit("plan",
                       f"plan: {len(tasks)} file(s)"
                       + (" (revised)" if revised else "")
                       + (" (continued)" if self._resumed is not None
                          and not revised else ""),
                       data)

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
        if self._test_repairs:
            return self._repair_test_step()
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
        seed = self._seeds.pop(task.path, ())
        outcome = self.loop.run_task(task, request=self.plan.request,
                                     covers=covers.path if covers else "",
                                     planned=self._planned_paths(task),
                                     seed=seed)
        self.outcomes.append(outcome)
        self._credit_module(covers, outcome)
        self._credit_test_written_first(task, outcome)
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
        # THE SIZE RULE is measured on every verified outcome, whichever
        # path verified it. It used to be measured in `_record` only —
        # the repair steps and Continue — so a normal build never measured
        # a file (found by the 2026-10-27 review).
        if outcome.ok:
            self._measure_size(task)

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
        # THE REPAIR ANSWERS FOR ITS RESULT. Oct 2, 2026: render.py was
        # reported REPAIRED against main.py's crash with a byte-identical
        # file — render.py's own check (an import, zero tests) passed, and
        # main.py then failed with the same error. Now an unchanged file
        # stops the repair at once, NOT repaired, and a caller's failure is
        # re-run as the acceptance test. (A test blamer needs no separate
        # acceptance: it is the fix's own test.)
        try:
            before = self.host.fs.read(module.path)
        except Exception:                                # noqa: BLE001
            before = None
        caller = None if blamer_is_test else test
        outcome = self.loop.run_task(fix, request=self.plan.request,
                                     seed=failing,
                                     planned=self._planned_paths(module),
                                     caller=caller.path if caller else "",
                                     accept=caller, baseline=before)
        self._record(fix, outcome, label="REPAIRED" if outcome.ok
                     else "NOT REPAIRED")
        if not outcome.ok:
            # The repair came back unchanged and the judge, quoting the
            # request, agreed the TEST is wrong: the test is repaired next.
            wrong = self.loop.test_wrong.pop(test.path, None)
            if wrong is not None and blamer_is_test:
                _module_path, verdict, failing_diags = wrong
                self._test_repairs.append((test, module, verdict,
                                           tuple(failing_diags)))
                self.host.emit(
                    "status",
                    f"{test.path} will be repaired against the request — "
                    f"the judge ruled it wrong: {verdict.describe()}",
                    {"task": test.path, "module": module.path,
                     "because": "judge"})
            return outcome
        # The acceptance run IS the re-check, against the same files: a
        # caller's second run would cost a second full run timeout.
        again = self.loop.reverify(
            test, because=f"{module.path} was repaired against it",
            result=self.loop.accepted.pop(test.path, None))
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

    def _repair_test_step(self) -> TaskOutcome:
        """Repair a TEST the judge ruled wrong, then credit the module.

        The fix's first attempt is a repair against the judge's diagnostic
        — the request's words and the assertion that contradicts them. A
        test that then passes against the module verifies the module; the
        module's earlier "NOT REPAIRED" was the model declining to break
        correct code, and the record says so.
        """
        from . import adjudicate as adjudicate_mod
        test, module, verdict, failing = self._test_repairs.pop(0)
        self._where = f"the repair of {test.path} (the judge ruled it wrong)"
        caps = self._capabilities(boundary=f"repair of {test.path}")
        if not caps.loaded:
            raise NoModelLoadedError(
                "capabilities() reports no model at a task boundary")
        judged = adjudicate_mod.diagnostic_for_test(verdict, test.path,
                                                    failing)
        fix = Task(id=f"{test.id}-fix", path=test.path, purpose=test.purpose,
                   test_path="", persona="tester", lang=test.lang,
                   atomic=False)
        self.log.event(f"REPAIR {test.path}",
                       f"against the request, not the module — the judge: "
                       f"{verdict.describe()}")
        # the module's author's reasons (SUSPECT TEST lines, 2026-10-24)
        # travel with the judge's: each names one assertion and why
        reasons = tuple(d for d in failing if d.code == "suspect")
        outcome = self.loop.run_task(fix, request=self.plan.request,
                                     covers=module.path,
                                     seed=(judged, *reasons),
                                     planned=self._planned_paths(test))
        self._record(fix, outcome, label="REPAIRED" if outcome.ok
                     else "NOT REPAIRED")
        if outcome.ok:
            caveat = (f"verified by {test.path} after the judge ruled its "
                      f"earlier assertion wrong, quoting the request")
            self.outcomes.append(TaskOutcome(
                task_id=module.id, path=module.path, ok=True,
                result=outcome.result, verified=bool(outcome.verified),
                caveats=(caveat,)))
            if outcome.verified:
                self.loop.unverified.discard(module.path)
            self.host.emit("status",
                           f"{module.path}: {caveat}",
                           {"task": module.path, "verified": outcome.verified,
                            "by": test.path})
            self.plan = self.plan.replace(module.with_status("done"))
        self.plan = self.plan.replace(test.with_status(
            "done" if outcome.ok else "failed"))
        return outcome

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
                                 "verified": bool(outcome.verified),
                                 "attempts": len(outcome.attempts),
                                 "caveats": list(outcome.caveats)},
                         stopped_because=outcome.stopped_because)
        if outcome.ok:
            self._measure_size(task)

    def _measure_size(self, task: Task) -> None:
        """A file that works but is over the size rule (sizing.py): said
        in the log, the journal and the report — not refused."""
        if not (self.config.max_module_lines
                or self.config.max_function_lines):
            return
        try:
            code = self.host.fs.read(task.path)
        except Exception:                                 # noqa: BLE001
            return
        lang = task.lang or self.config.lang
        # a test file is measured for its functions only: its length is
        # one test after another, and the planner does not split tests
        m = sizing.measure(code, task.path, lang,
                           max_module=(0 if runner.is_test_path(task.path)
                                       else self.config.max_module_lines),
                           max_function=self.config.max_function_lines)
        if not m.over:
            return
        self.size_findings.append(m)
        note = m.describe()
        self.log.line(f"  SIZE: {note}")
        self.journal.log("size", task=task.path, **m.summary())
        self.host.emit("status", note,
                       {"task": task.path, "size": m.summary()})

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
            # a swap carries on with the refused model (said as a
            # warning), so what is in force must be THIS model's: the
            # previous model's profile and probe verdict are not (the
            # 2026-10-27 review)
            self._apply_profile(caps)
            self.loop.fim_ok, self.loop.fim_why = False, (
                "the loaded model was refused as not a coding model; "
                "no hole is filled on it")
            raise NotACodingModelError(verdict.reason)
        if verdict.refuse:
            self.host.emit("warning",
                           verdict.reason + " Continuing because "
                           "allow_any_model is set.",
                           {"phase": "model_check", "model": caps.name,
                            "verdict": verdict.verdict, "refused": False})
        elif verdict.verdict == "unknown" and not getattr(
                caps, "supports_canvas", False):
            # A model that fills a canvas is DiffusionGemma by what it DOES,
            # whatever its file is called (Bill's stock file is
            # `Dg_Rc0P1_Patched`); the host knows it from the architecture.
            self.host.emit("warning", verdict.reason,
                           {"phase": "model_check", "model": caps.name,
                            "verdict": verdict.verdict})
        if not verdict.context_ok:
            self.host.emit("warning", verdict.context_note,
                           {"phase": "model_check", "model": caps.name,
                            "context_tokens": caps.context_tokens})
        # DIFFUSION MODE is decided by the model, so it is said here, once,
        # where the operator reads what this build will be.
        canvas = bool(getattr(caps, "supports_canvas", False))
        if canvas and str(self.config.diffusion).lower() != "off":
            self.host.emit("status",
                           "diffusion mode: this model fills a canvas, so "
                           "a repair rewrites only the lines its error is "
                           "in, and a file whose interface was pinned is "
                           "written one body at a time",
                           {"phase": "model_check", "model": caps.name,
                            "diffusion": True})
        elif canvas:
            self.host.emit("status",
                           "diffusion mode is off for this session: whole "
                           "files, as with any model",
                           {"phase": "model_check", "model": caps.name,
                            "diffusion": False})
        self._apply_profile(caps)
        self._probe_fim(caps)

    def _probe_fim(self, caps: ModelCapabilities) -> None:
        """Does the loaded model fill a known hole correctly? One short
        request at the model check (fim.py); its answer, not a name in a
        list, decides whether repairs go by FIM holes this session."""
        self.loop.fim_ok = False
        mode = str(self.config.fim or "auto").lower()
        fill = getattr(self.host.llm, "fim_fill", None)
        why = ""
        if mode == "off":
            why = "switched off for this session"
        elif not callable(fill):
            why = "the host's model port does not fill in the middle"
        elif bool(getattr(caps, "supports_canvas", False)):
            why = "a canvas model fills holes as canvases (diffusion mode)"
        elif not caps.loaded:
            why = "no model is loaded"
        elif mode == "auto":
            from .ports import NullLLM, ScriptedLLM
            if isinstance(self.host.llm, (NullLLM, ScriptedLLM)):
                why = "not probed for a scripted model"
        if why:
            self.loop.fim_why = why
            return
        self._where = "the FIM probe"
        ok = False
        text = ""
        tokens: dict = {}
        try:
            result = fill(fim_mod.probe_request(self.model_profile),
                          cancel=self.cancel_token)
        except Cancelled:
            raise
        except Exception as exc:                          # noqa: BLE001
            why = f"the probe raised ({_plain_cause(exc)})"
        else:
            if result.finish_reason == "cancelled":
                raise Cancelled("the FIM probe")
            ok, why = fim_mod.probe_verdict(result)
            text = (result.text or "")[:200]
            # the probe is a model call like any other: its tokens and
            # timings go to the journal (the 2026-10-27 review)
            tokens = {"tokens_in": int(result.tokens_in or 0),
                      "tokens_out": int(result.tokens_out or 0),
                      "prompt_ms": int(result.prompt_ms or 0),
                      "decode_ms": int(result.decode_ms or 0),
                      "finish_reason": result.finish_reason}
        self.loop.fim_ok, self.loop.fim_why = ok, why
        self.journal.log("fim", kind="probe", model=caps.name, ok=ok,
                         why=why, text=text, **tokens)
        self.log.line(f"  fim       {'on' if ok else 'off'} — {why}")
        self.host.emit("status",
                       ("FIM mode: this model fills in the middle, so a "
                        "repair rewrites only the hole its error is in"
                        if ok else f"FIM mode is off: {why}"),
                       {"phase": "model_check", "model": caps.name,
                        "fim": ok, "why": why})

    def _apply_profile(self, caps: ModelCapabilities) -> None:
        """The profile for the loaded model (profiles.py), in force from
        here: on the loop (samplers, the ladder's base, the thinking
        switch), on the planner, and in the journal and the log."""
        prof = profiles.profile_for(caps.name, given=self.config.profile,
                                    directory=self.config.profile_dir)
        self.model_profile = prof
        self.loop.profile = prof
        self.planner.profile = prof
        # the edit format the model gets right most often, when the
        # operator left the choice to the engine
        if str(self.config.narrow or "auto").lower() == "auto":
            if prof.edit_format == "symbols":
                self.loop.config.narrow = "always"
            elif prof.edit_format == "whole":
                self.loop.config.narrow = "off"
            elif prof.edit_format == "auto":
                self.loop.config.narrow = "auto"
        self.journal.log("profile", model=caps.name, family=prof.family,
                         source=prof.source, temperature=prof.temperature,
                         sampling=prof.sampling(),
                         edit_format=prof.edit_format, fim=prof.fim,
                         thinking=prof.thinking, think_off=prof.think_off)
        try:
            self.log.line(f"  {prof.describe()}")
        except Exception:                                 # noqa: BLE001
            pass
        if prof.sampling() and not profiles.accepts(self.host.llm,
                                                    "sampling"):
            self.host.emit("status",
                           f"the model's profile names samplers "
                           f"({', '.join(prof.sampling())}) this host's "
                           f"model port does not take; its temperature "
                           f"and seed still apply",
                           {"phase": "model_check", "model": caps.name,
                            "sampling": False})

    #: Below this many tokens per second a chat model is almost certainly
    #: running from system RAM, not the card: a 24B Q4 that fits in VRAM
    #: decodes at 15–30 tok/s on a 16 GB card; Devstral offloaded ran at 3–5.
    SLOW_TOK_S = 8.0

    def _speed_probe(self, caps: ModelCapabilities, *, after: str) -> None:
        """One short generation, timed, written down — and a warning when
        the number says the model is spilling into system RAM.

        The whole 2026-10-16 Devstral build ran at 3–5 tok/s: 89 % of its
        two and a half hours was the model decoding, and the first line
        that said so was in a journal read afterwards. One probe at the
        start costs a few seconds and puts the number where the operator
        reads it, when it can still change their mind.
        """
        if self.config.speed_probe is False or not caps.loaded:
            return
        if self.config.speed_probe is None:
            from .ports import NullLLM, ScriptedLLM
            if isinstance(self.host.llm, (NullLLM, ScriptedLLM)):
                return                       # their replies are counted
        from .types import Message
        messages = [Message(role="system",
                            content="You answer with one word."),
                    Message(role="user",
                            content="Reply with the single word READY and "
                                    "nothing else.")]
        self._where = f"the speed probe after {after}"
        t0 = time.monotonic()
        try:
            completion = self.host.llm.complete(
                messages, temperature=0.0, max_tokens=48,
                cancel=self.cancel_token)
        except Cancelled:
            raise
        except Exception as exc:                          # noqa: BLE001
            self.log.line(f"  speed     not measured ({_plain_cause(exc)})")
            return
        if completion.finish_reason == "cancelled":
            raise Cancelled("the speed probe")
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        decode_ms = int(getattr(completion, "decode_ms", 0) or 0)
        prompt_ms = int(completion.prompt_ms or 0)
        tokens_out = int(completion.tokens_out or 0)
        if not tokens_out and completion.text:
            tokens_out = max(1, len(completion.text) // 4)
        if decode_ms <= 0:
            # the port does not separate prefill from decode: the whole
            # call, less prefill where it is known — a ceiling on the rate
            decode_ms = max(1, elapsed_ms - prompt_ms)
        tok_s = tokens_out / (decode_ms / 1000.0) if tokens_out else 0.0
        canvas = bool(getattr(caps, "supports_canvas", False))
        self.journal.log("model_check", model=caps.name, probe=True,
                         after=after, tokens_out=tokens_out,
                         prompt_ms=prompt_ms, decode_ms=decode_ms,
                         elapsed_ms=elapsed_ms, tok_s=round(tok_s, 1),
                         finish_reason=completion.finish_reason,
                         canvas=canvas)
        if canvas:
            # a diffusion model writes a whole canvas at once: tokens per
            # second is not the measure, the time to a canvas is
            self.log.line(f"  speed     one reply ({tokens_out} tokens) in "
                          f"{elapsed_ms / 1000:.1f}s — prefill "
                          f"{prompt_ms} ms, denoising {decode_ms} ms "
                          f"({caps.name}, a canvas model)")
            return
        self.log.line(f"  speed     {tok_s:.1f} tok/s decode, prefill "
                      f"{prompt_ms} ms, {tokens_out} tokens in "
                      f"{elapsed_ms / 1000:.1f}s ({caps.name})")
        if tokens_out and tok_s < self.SLOW_TOK_S:
            note = (f"{caps.name} is generating at {tok_s:.1f} tokens per "
                    f"second, far below what a model that fits in VRAM "
                    f"does; it is probably spilling into system RAM. A "
                    f"smaller quantization, a shorter context, or a "
                    f"quantized KV cache would be several times faster"
                    + (" — and a llama-server started with `--spec-type "
                       "ngram-simple` drafts rewrites from the file it was "
                       "shown at no model cost (docs/SPEED.md)"
                       if getattr(caps, "supports_grammar", False)
                       and not getattr(caps, "is_remote", False) else "")
                    + ". The build goes on at this speed.")
            self.log.line(f"  WARNING   {note}")
            self.host.emit("warning", note,
                           {"phase": "model_check", "model": caps.name,
                            "tok_s": round(tok_s, 1), "slow": True})

    def _diffusion_state(self, caps: Any) -> str:
        """For the build log's config line: whether this build can cut
        holes, and if not, why not."""
        try:
            if not bool(getattr(caps, "supports_canvas", False)):
                return "unavailable (the model fills no canvas)"
            if str(self.config.diffusion).lower() == "off":
                return "off (switched off for this session)"
            from . import canvas as cv
            # what the loop will actually ask for: the schedule caps it
            n = min(max(1, int(self.config.diffusion_candidates or 1)),
                    len(cv.REPAIR_SCHEDULE))
            return (f"on ({n} canvas(es) per repair, notes "
                    f"{'on' if self.config.diffusion_notes else 'off'})")
        except Exception as exc:                         # noqa: BLE001
            # a log line must never stop a session from starting
            return f"unknown ({type(exc).__name__}: {exc})"

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
        # 2026-10-22: when EVERY test module in the folder is already
        # failing, nothing this build writes can be verified by them, and
        # the session must not end `ok`. Two re-runs over one folder did
        # exactly that — all files "untested", headline green.
        modules = {str(p).replace("\\", "/") for p in existing
                   if _looks_like_test(str(p))}
        all_excluded = bool(modules) and len(set(failing)) >= len(modules)
        self._all_tests_baselined = bool(all_excluded)
        self.journal.log("baseline", existing=list(existing),
                         failing=list(failing), ok=False,
                         all_excluded=bool(all_excluded))
        if all_excluded:
            self.host.emit(
                "warning",
                "every test module in the folder was already failing, so "
                "NOTHING this build writes can be checked by them — the "
                "session will not report ok, whatever builds. Repair the "
                "tests first (Debug & Repair on a test file), or build "
                "into an empty folder.",
                {"phase": "baseline", "all_excluded": True})
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

        Unless tests are written first, a module's test is written right
        after it, so the module is "built, not verified" at its own task's
        end — its test did not exist yet. When the test is written and its tests run and pass, the module
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

    def _credit_test_written_first(self, module: Task,
                                   outcome: TaskOutcome) -> None:
        """A module that VERIFIED against a test written before it verifies
        that test too: it was "written, not verified" at its own task's
        end, because nothing it tests existed yet."""
        if not outcome.verified or not module.test_path:
            return
        last = next((o for o in reversed(self.outcomes)
                     if o.path == module.test_path), None)
        if last is None or not last.ok or last.verified:
            return
        if any("PASS against the unwritten stub" in c for c in last.caveats):
            return      # it passed before the module existed: it tests nothing
        if not any("written before" in c for c in last.caveats):
            return      # not a test written first; `_credit_module` has it
        self.loop.unverified.discard(module.test_path)
        self.outcomes.append(dataclasses.replace(
            last, verified=True,
            caveats=last.caveats + (
                f"verified when {module.path} was built against it and "
                f"its tests passed",)))
        self.host.emit("status",
                       f"{module.test_path}: now verified — {module.path} "
                       f"was built to pass it, and it does",
                       {"task": module.test_path, "verified": True,
                        "by": module.path})

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

    def _pin_constraints(self, text: str, *, source: str) -> None:
        """Take the hard requirements out of `text` and pin them in every
        prompt from here on — said in the log either way."""
        if not self.config.pin_constraints:
            self.log.line("  requirements  not pinned (switched off)")
            return
        from . import constraints as constraints_mod
        found = constraints_mod.extract(text or "")
        self.constraints = found
        self.prompts.constraints = found.block()
        # The same sentences reach the skeleton: the ones naming a pinned
        # file go into its docstring (interfaces.with_conventions).
        self.planner.conventions = list(found.items)
        self.journal.log("constraints", source=source, items=found.items,
                         more=found.more)
        if not found.items:
            self.log.line(f"  requirements  none found in {source} (no "
                          f"MUST / NEVER / DO NOT sentence)")
            return
        self.log.rule(f"HARD REQUIREMENTS — from {source}, pinned in "
                      f"every prompt")
        for item in found.items:
            self.log.line(f"  - {item}")
        if found.more:
            self.log.line(f"  … and {found.more} more, not pinned (the "
                          f"first {constraints_mod.MAX_CONSTRAINTS} are)")
        self.host.emit("status",
                       f"{len(found.items)} hard requirement"
                       f"{'' if len(found.items) == 1 else 's'} from "
                       f"{source} pinned in every prompt",
                       {"phase": "constraints", "items": found.items})

    def _mutation_check(self) -> None:
        """Deliberate bugs in a copy of each verified Python module: do
        its tests notice? Survivors are WARNINGS (mutation.py)."""
        from . import mutation
        from .ports import LocalFileSystem
        fs = self.host.fs
        deadline = time.monotonic() + float(self.config.mutation_budget_s
                                            or 0)
        header = False
        for o in self._final_outcomes():
            if (not o.ok or o.path in self.loop.unverified
                    or _looks_like_test(o.path)
                    or langs.id_for_path(o.path) != "python"):
                continue
            task = (self.plan.task(o.task_id.removesuffix("-fix"))
                    if self.plan else None)
            test_path = task.test_path if task else ""
            if not test_path or not fs.exists(test_path):
                continue
            if not header:
                self.log.rule("MUTATION CHECK — do the tests notice a "
                              "deliberate bug?")
                header = True
            if time.monotonic() > deadline:
                self.log.line(f"  {o.path}: not checked — the time for "
                              f"the check ran out")
                continue
            self._where = f"the mutation check of {o.path}"
            stem = o.path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

            def run_tests(_stem: str = stem, **kw: Any) -> Any:
                return runner.run_tests("python", stem=_stem, **kw)

            try:
                rep = mutation.check(
                    fs=fs, ex=self.host.exec, make_fs=LocalFileSystem,
                    run_tests=run_tests, path=o.path, test_path=test_path,
                    code=fs.read(o.path),
                    max_mutants=int(self.config.mutation_max or 0),
                    deadline=deadline, timeout=self.config.test_timeout,
                    cancel=self.cancel_token)
            except Cancelled:
                raise
            except Exception as exc:                      # noqa: BLE001
                rep = mutation.MutationReport(o.path, test_path,
                                              note=f"the check failed: "
                                                   f"{exc}")
            self.mutation.append(rep)
            self.log.event(f"MUTATION {o.path}", rep.line())
            for m, killed in rep.tried:
                self.log.line(f"  line {m.line} {m.what}: "
                              f"{'caught' if killed else 'NOT CAUGHT'}")
            self.journal.log("mutation", task=o.path, test=test_path,
                             tried=len(rep.tried), killed=rep.killed,
                             survived=[{"line": m.line, "change": m.what}
                                       for m in rep.survivors],
                             note=rep.note)
            if rep.survivors:
                self.host.emit("warning", rep.line(),
                               {"phase": "mutation", "task": o.path})
            else:
                self.host.emit("status", rep.line(),
                               {"phase": "mutation", "task": o.path})

    def debug(self, report: Any) -> Any:
        """DEBUG & REPAIR (debugrepair.py): the operator ran the program,
        it failed, and this is what they saw — the error, the command, a
        description, the file. Locate, repair through the ordinary loop,
        run their command again to confirm, record every round.

        A session of its own: started, logged and ended like a build, so
        BUILD_LOG.txt and the journal hold it in order with the builds.
        Returns a `DebugResult` and never raises: a stop, a refused model or
        an unexpected failure is a result ("stopped", "error") with its
        sentence, recorded like any other.
        """
        from . import debugrepair
        self._started = time.monotonic()
        self._where = "debug & repair"
        said = report.description or report.error_text or report.command
        result = None
        self._apply_skill_triggers(" ".join(
            (report.description or "", report.error_text or "",
             report.command or "")))
        try:
            caps = self._capabilities(boundary="session start")
            self.journal.log("session_start", request=f"DEBUG: {said[:500]}",
                             model=caps.name, session=self.id, profile={},
                             config={"attempts": self.config.attempts,
                                     "lang": self.config.lang,
                                     "debug": {"command": report.command,
                                               "files": list(report.files),
                                               "rounds": report.rounds}})
            self._check_model(caps)
            self.log.start("DEBUG & REPAIR\n" + debugrepair.clip(said, 1500),
                           caps.name,
                           {"attempts": self.config.attempts,
                            "temperature": self.config.temperature,
                            "max_tokens": self.config.max_tokens,
                            "lang": self.config.lang,
                            "diffusion": self._diffusion_state(caps)})
            # a repair keeps the requirements the project was built to: the
            # spec ATK saved beside it, when there is one
            try:
                spec = (self.host.fs.read("BUILD_SPEC.md")
                        if self.host.fs.exists("BUILD_SPEC.md") else "")
            except Exception:                             # noqa: BLE001
                spec = ""
            if spec:
                self._pin_constraints(spec, source="BUILD_SPEC.md")
            try:
                stats = self.codemap.index_project()
                self.journal.log("codemap", files=stats.files,
                                 symbols=stats.symbols, edges=stats.edges,
                                 unresolved=stats.unresolved,
                                 resolution=round(stats.resolution_rate, 3))
            except Exception as exc:                      # noqa: BLE001
                self.host.emit("warning", f"the project could not be "
                                          f"indexed ({exc}); debugging "
                                          f"without the code map")
            result = debugrepair.run(self, report)
        except Exception as exc:                          # noqa: BLE001
            # Never raised to the host: a model that is not a coding model,
            # an exhausted budget, a port that failed — each is a RESULT
            # with its sentence, journaled and recorded like any other, so
            # the host's result box never stays at "working on it".
            wrapped = (exc if isinstance(exc, CognitiveCoderError) else
                       CognitiveCoderError.wrap(
                           exc, self._unexpected_sentence(exc)))
            self._report_failure(wrapped)
            result = debugrepair.DebugResult("error", str(wrapped))
            try:
                debugrepair._finish(self, report, result, {})
            except Exception:                             # noqa: BLE001
                pass
        self.journal.log("session_end", ok=result.state in ("fixed",
                                                            "repaired"),
                         files=sorted(result.diffs), failed=[],
                         unverified=[], untested=[],
                         seconds=round(time.monotonic() - self._started, 1),
                         debug=result.state)
        self.log.close(result.verdict)
        self._finished = True
        return result

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
            elif self._resumed is not None and not self._continue_prepared:
                self._prepare_continue(profile)
            while True:
                outcome = self.step()
                if outcome is None:
                    break
                self._stop_if_asked(outcome)
            # Hybrid mode's later phases re-open what failed (and attempt
            # what was blocked on it) with the other model loaded.
            self._hybrid_phases()
            #: NOTHING READY IS NOT THE SAME AS NOTHING LEFT.
            #:
            #: A task whose dependency failed can never become ready, so
            #: the loop ends and the file is simply never mentioned. On a
            #: real build that showed as "[build 6/6]" against a seven-file
            #: plan, with src/main.py absent from the log entirely — it was
            #: waiting on src/render.py, which had failed three attempts
            #: earlier.
            #:
            #: Not building it is correct. Not saying so is the same
            #: mistake the installer made: an absence nobody can see.
            for task, blockers in (self.plan.blocked() if self.plan else []):
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
        except Cancelled as exc:
            after = getattr(exc, "after", "")
            self.journal.log("cancel", sentence=str(exc),
                             **({"after": after} if after else {}))
            self.log.event("STOPPED", str(exc))
            self.host.emit("status", str(exc),
                           {"phase": "stopped", "after": after,
                            "graceful": bool(after)})
        except BudgetExceeded as exc:
            self.journal.log("budget", sentence=str(exc), exhausted=True)
            self.host.emit("budget", str(exc))
        else:
            # §4.3: the review runs AFTER the code builds and its tests pass,
            # not instead — and not at all if the session was cancelled or
            # ran out of budget, because a review of half-finished work
            # reads as authoritative and is not.
            if self.config.mutation_check and any(o.ok
                                                  for o in self.outcomes):
                self._where = "the mutation check"
                try:
                    self._mutation_check()
                except Cancelled:
                    self.journal.log("cancel", where="mutation check")
            if self.config.review_after_build and any(o.ok
                                                      for o in self.outcomes):
                self._where = "the review"
                try:
                    self.review()
                except Cancelled:
                    self.journal.log("cancel", where="review")

    # ------------------------------------------------------------------
    # hybrid mode (hybrid.py, docs/HYBRID.md)
    # ------------------------------------------------------------------
    def _roster(self) -> list:
        """The host's roster, or [] — never raises."""
        port = self.host.llm
        fn = getattr(port, "roster", None)
        if not callable(fn):
            return []
        try:
            return [s for s in (fn() or ()) if getattr(s, "name", "")]
        except Exception as exc:                          # noqa: BLE001
            self.log.line(f"  hybrid    the roster could not be read "
                          f"({exc}); one model")
            return []

    def _plan_hybrid(self, caps: ModelCapabilities) -> ModelCapabilities:
        """Decide the roles, say so, and load the first phase's model if
        it is not the loaded one. Returns the capabilities to go on with."""
        from . import hybrid
        self._roles = None
        mode = str(self.config.hybrid or "auto")
        roster = self._roster()
        if mode.strip().lower() == "off" and not roster:
            return caps
        loaded_kind = ("canvas" if getattr(caps, "supports_canvas", False)
                       else "chat") if caps.loaded else ""
        roles, reason = hybrid.plan_roles(
            mode, roster, caps.name, loaded_kind=loaded_kind,
            writer=self.config.hybrid_writer,
            repairer=self.config.hybrid_repairer)
        if roles is None:
            if roster or mode.strip().lower() != "auto":
                # said only when there was something to decide: a host
                # with no roster gets no line and no journal event
                self.log.line(f"  hybrid    {reason}")
                self.journal.log("phase", hybrid=False, reason=reason,
                                 roster=[s.name for s in roster])
            return caps
        self._roles = roles
        self.log.line(f"  hybrid    {roles.describe()}")
        for ph in roles.phases():
            self.log.line(f"            phase {ph.number} ({ph.name}): "
                          f"{ph.model} — {ph.what}")
        self.journal.log("phase", hybrid=True, order=roles.order,
                         first=roles.first, second=roles.second,
                         roster=[s.name for s in roster], why=roles.why)
        self.host.emit("status", roles.describe(),
                       {"phase": "hybrid", "order": roles.order,
                        "first": roles.first, "second": roles.second})
        first = roles.phases()[0]
        self._phase = first
        if first.model != caps.name:
            if self._swap_to(first.model, why=f"phase 1 ({first.name}) "
                                               f"is {first.model}'s"):
                caps = self._capabilities(boundary="phase 1")
        return caps

    def _swap_to(self, name: str, *, why: str) -> bool:
        """Ask the host to load ``name``. True when it did. Said in the
        log with its duration either way; the core loads nothing itself."""
        load = getattr(self.host.llm, "load_model", None)
        if not callable(load):
            self.log.event("SWAP", "the host cannot load models on request; "
                                   "staying on what is loaded")
            return False
        self._where = f"loading {name}"
        self.log.event("SWAP", f"loading {name} — {why}; the prompt cache "
                               f"starts cold")
        self.host.emit("status", f"loading {name} — {why}",
                       {"phase": "swap", "model": name})
        t0 = time.monotonic()
        try:
            problem = load(name, cancel=self.cancel_token)
        except Exception as exc:                          # noqa: BLE001
            problem = f"the host raised while loading it ({exc})"
        seconds = round(time.monotonic() - t0, 1)
        if self.cancel_token.is_set():
            raise Cancelled(f"loading {name}")
        if problem:
            self.log.event("SWAP", f"{name} was not loaded after {seconds}s "
                                   f"— {problem}; the build goes on with "
                                   f"whatever is loaded")
            self.host.emit("warning",
                           f"could not load {name}: {problem} — the build "
                           f"goes on with the loaded model",
                           {"phase": "swap", "model": name, "ok": False})
            self.journal.log("swap", model=name, ok=False, seconds=seconds,
                             why=why, problem=str(problem)[:300])
            return False
        caps = self._capabilities(boundary=f"swap to {name}")
        self.journal.log("swap", model=name, ok=True, seconds=seconds,
                         why=why, now=caps.name)
        self.log.event("SWAP", f"{caps.name or name} loaded in {seconds}s")
        try:
            self._check_model(caps)
        except NotACodingModelError as exc:
            self.log.line(f"  WARNING   {exc}")
        self._speed_probe(caps, after=f"the swap to {name}")
        self.host.emit("status", f"now running {caps.name or name}",
                       {"phase": "swap", "model": caps.name or name,
                        "ok": True, "seconds": seconds})
        return True

    def _unfinished(self) -> list[Task]:
        """The plan's tasks that failed, and those never attempted because
        a dependency failed — in plan (dependency) order."""
        if self.plan is None:
            return []
        final = {o.path: o for o in self.outcomes}
        out: list[Task] = []
        for task in self.plan.tasks:
            if task.status == "failed":
                out.append(task)
            elif task.status == "pending":
                out.append(task)               # blocked: never attempted
            elif task.status == "done" and task.path in final \
                    and not final[task.path].ok:
                out.append(task)
        return out

    def _hybrid_phases(self) -> None:
        """Phases 2 and 3: swap, re-open what failed, run the step loop
        again. Each phase works in dependency order and ends when nothing
        it can act on is left; a phase with nothing left is skipped."""
        if self._roles is None or self.plan is None:
            return
        phases = self._roles.phases()[1:]
        for phase in phases:
            left = self._unfinished()
            if not left:
                self.log.event(f"PHASE {phase.number} ({phase.name})",
                               "nothing left to do — skipped")
                self.journal.log("phase", number=phase.number,
                                 name=phase.name, model=phase.model,
                                 skipped=True)
                break
            self._phase = phase
            self.log.rule(f"PHASE {phase.number} — {phase.name}: "
                          f"{phase.model}")
            self.log.line(f"  {phase.what}")
            self.log.line(f"  left from the phase before: "
                          f"{', '.join(t.path for t in left)}")
            self.journal.log("phase", number=phase.number, name=phase.name,
                             model=phase.model,
                             left=[t.path for t in left])
            self.host.emit("phase", f"phase {phase.number} ({phase.name}): "
                                    f"{len(left)} file(s) left — "
                                    f"{phase.what}",
                           {"phase": "hybrid", "number": phase.number,
                            "name": phase.name, "model": phase.model,
                            "left": [t.path for t in left]})
            caps = self._capabilities(boundary=f"phase {phase.number}")
            if phase.model != caps.name:
                self._swap_to(phase.model,
                              why=f"phase {phase.number} ({phase.name}) is "
                                  f"{phase.model}'s")
            self._reopen(left)
            fixed_before = sum(1 for o in self._final_outcomes() if o.ok)
            while True:
                outcome = self.step()
                if outcome is None:
                    break
                self._stop_if_asked(outcome)
            final = self._final_outcomes()
            fixed = sum(1 for o in final if o.ok) - fixed_before
            still = [t.path for t in self._unfinished()]
            self.log.event(f"PHASE {phase.number} ({phase.name}) ended",
                           f"{max(0, fixed)} more file(s) ok; "
                           + (f"still failing: {', '.join(still)}" if still
                              else "nothing left failing"))
            self.journal.log("phase", number=phase.number, name=phase.name,
                             model=phase.model, ended=True, fixed=fixed,
                             still=still)

    def _reopen(self, tasks: list) -> None:
        """Put failed tasks back to pending with their last errors as the
        seed of their first attempt; a blocked task simply becomes ready
        once its dependency is done."""
        final = {o.path: o for o in self.outcomes}
        for task in tasks:
            if task.status == "pending":
                continue
            last = final.get(task.path)
            diags: tuple = ()
            if last is not None and last.attempts:
                diags = tuple(last.attempts[-1].diagnostics)
            if diags:
                self._seeds[task.path] = diags
            self._repaired.discard(task.path)
            self.plan = self.plan.replace(task.with_status("pending",
                                                           attempts=0))
        # the loop's per-task memory of what it tried is per run_task;
        # the session's one-repair-per-file rule starts over per phase
        self.loop.blamed.clear()

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
        self._failed = exc.sentence
        try:
            self.journal.error(exc.sentence, exc.detail,
                               journal=self.journal.path, where=self._where)
        except Exception:                                 # noqa: BLE001
            pass
        # BUILD_LOG.txt is the file an operator reads, and it never said. The
        # DiffusionGemma build's log ends mid-attempt as if the operator had
        # walked away; the crash was in the JSONL only.
        try:
            self.log.event("STOPPED UNEXPECTEDLY", exc.sentence)
            if exc.detail:
                self.log.block("traceback", exc.detail)
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
        # 2026-10-22: a session in which NO source file has a test that ran
        # is not ok, however green every file's build was. "untested" is
        # its own state, said as such, so a re-run over a folder whose own
        # tests were all baselined away cannot print a green headline.
        sources = [o.path for o in final
                   if o.ok and not _looks_like_test(o.path)]
        no_evidence = set(untested) | set(unverified)
        evidence = bool(sources) and any(p not in no_evidence
                                         for p in sources)
        all_untested = bool(sources) and not evidence
        # A project that never had tests and whose plan wrote none is still
        # "untested, ok" (some files genuinely have none — the 2026-10-19
        # decision stands). A project that HAS tests — in the plan, or in
        # the folder — and ran none of them is not ok: that is the case
        # where the tests were all baselined away and the headline lied.
        tests_present = (getattr(self, "_all_tests_baselined", False)
                         or any(_looks_like_test(t.path)
                                for t in (self.plan.tasks if self.plan
                                          else ()))
                         or any(_looks_like_test(o.path) for o in final))
        not_ok_untested = all_untested and tests_present
        # What every planned file looked like when this run ended, so a
        # Continue can tell a file the operator edited during the pause
        # from one nobody touched (resumable.changed_since). Hashes, not
        # contents: the snapshots already hold the contents.
        hashes = (resumable.file_hashes(self.host.fs,
                                        [t.path for t in self.plan.tasks])
                  if self.plan else {})
        self.journal.log(
            "session_end",
            hashes=hashes,
            #: "ok" means VERIFIED. A file whose test file exists and ran
            #: nothing is not, however green its build was — and a session
            #: that crashed is not, however green the files before the
            #: crash were. And a session with tests where none ran is not.
            ok=(bool(final) and all(o.ok for o in final) and not unverified
                and not self._failed and not not_ok_untested),
            files=[o.path for o in final if o.ok],
            failed=[o.path for o in final if not o.ok],
            unverified=unverified,
            untested=untested,
            all_untested=all_untested,
            why_not_ok=("no tests ran for any file this session built"
                        if not_ok_untested and bool(final)
                        and all(o.ok for o in final) and not unverified
                        and not self._failed else ""),
            seconds=round(time.monotonic() - self._started, 1),
            codemap=stats.one_line(),
            remote=self.gate.active,
            bytes_out=self.gate.bytes_out,
            redactions=self.gate.redactions,
            budget=self.budget.as_dict() if self.gate.active else {},
            **({"error": self._failed} if self._failed else {}))
        if self.gate.active:
            self.host.emit(
                "remote",
                f"Remote mode was on this session: {self.gate.bytes_out:,} "
                f"bytes sent, {self.gate.redactions} secret(s) redacted "
                f"first.",
                {"enabled": False, "bytes_out": self.gate.bytes_out,
                 "redactions": self.gate.redactions})
        self.host.emit("status", self.journal.summary())
        # The build log's SUMMARY. It closed only Debug & Repair sessions;
        # a build's log just stopped, and a reader could not tell a finished
        # build from one still running.
        if getattr(self.log, "_opened", False):
            try:
                tail = self.journal.summary()
                if self._failed:
                    tail = f"STOPPED UNEXPECTEDLY — {self._failed}\n{tail}"
                self.log.close(tail)
            except Exception:                            # noqa: BLE001
                pass
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

        One of the TWO methods a host may call from another thread (the
        other is `stop_after_current`). Everything else in this class
        assumes single-threaded use, and the tokens are what make the
        exception safe.
        """
        self.cancel_token.set()

    def stop_after_current(self) -> None:
        """Finish the file in hand, then stop. Thread-safe, like `cancel`.

        `cancel()` is right when the build is going wrong. This is for the
        other reason to stop — the operator has to leave, or wants the GPU
        back — when the file being written is fine and four minutes from
        done: it is finished and verified (or not) as normal, and the build
        stops before the next one starts. Nothing is rolled back, because
        nothing was cut short, and Continue picks up at the next file.
        """
        self.stop_token.set()

    def _stop_if_asked(self, outcome: TaskOutcome | None) -> None:
        """Between two tasks: honour `stop_after_current`."""
        if self.stop_token.is_set():
            raise Cancelled(after=getattr(outcome, "path", "") or
                            "the file in hand")

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
            #: THE BASELINE. Tests that were failing before the first run
            #: wrote a line were excused from blame — then a resume forgot,
            #: and the stale test that cost the Aug 8 and Oct 1 builds their
            #: afternoons was charged to the resumed files all over again.
            elif row.get("event") == "baseline":
                data = row.get("data") or {}
                failing = tuple(str(p) for p in data.get("failing") or ())
                if failing:
                    session.loop.config.known_failing = failing
                session._all_tests_baselined = bool(data.get("all_excluded"))
        session._resumed = resumable.state_of(host.fs, session_id, directory)
        session._verified_paths = set(state.get("verified") or ())
        session.journal.log("session_start", resumed_from=session_id,
                            request=state["request"],
                            config={"lang": session.config.lang},
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

    # -- continuing a resumed session (resumable.py) ---------------------
    def _prepare_continue(self, profile: dict | None) -> None:
        """What `start()` is to a new build, for one picked up again.

        `resume()` rebuilt the plan from the journal and stopped there, so a
        resumed session went straight to the step loop with half its setup
        missing: no build-log header (the continued run's lines ran on under
        the last run's summary), no model check, no pinned requirements, no
        spec-vector tests, no hybrid roles, a session clock that started at
        zero — and no `plan` event, so a host's task board stayed empty
        while files were being written. Each of those is restored here, on
        the worker thread, before anything is generated.
        """
        self._continue_prepared = True
        self._started = time.monotonic()
        self._where = "picking the build up again"
        self.profile = dict(profile or {})
        state = self._resumed
        request = self.plan.request if self.plan else state.request
        caps = self._capabilities(boundary="continue")
        self._check_model(caps)
        left = [t.path for t in (self.plan.tasks if self.plan else ())
                if t.status == "pending"]
        how = resumable.ENDED_WORDS.get(state.ended, state.ended)
        said = (f"run {state.runs + 1} of this build — the last one ended "
                f"because {how}; {len(state.done)} of {len(state.files)} "
                f"file(s) were done, {len(left)} left")
        self.log.start(request, caps.name,
                       {"attempts": self.config.attempts,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                        "max_files": self.config.max_files,
                        "lang": self.config.lang,
                        "diffusion": self._diffusion_state(caps)},
                       continued=said)
        self.journal.log("continue", ended=state.ended, run=state.runs + 1,
                         done=list(state.done), remaining=left,
                         previous_model=state.model, model=caps.name)
        if state.model and caps.name and state.model != caps.name:
            # Not a warning: a different model for the rest is often the
            # whole reason for stopping. Said so the record shows it.
            self.log.line(f"  model     this build was started with "
                          f"{state.model}; it continues with {caps.name}")
            self.host.emit("status",
                           f"this build was started with {state.model}; "
                           f"it continues with {caps.name}",
                           {"phase": "continue", "model": caps.name,
                            "previous_model": state.model})
        self._apply_skill_triggers(request)
        caps = self._plan_hybrid(caps)
        self._pin_constraints(request, source="the request")
        self._restore_spec_vectors(request)
        # The board first, then the check: every row appears at once with
        # the state the journal proves, and the re-check updates them live.
        self._emit_plan(revised=False)
        self._reverify_before_continue()
        if self.plan is not None:
            ready = [t.path for t in self.plan.tasks if t.status == "pending"]
            done = [t.path for t in self.plan.tasks if t.status == "done"]
            self.host.emit(
                "status",
                f"Continuing: {len(done)} of {len(self.plan.tasks)} file(s) "
                f"already done, {len(ready)} to go"
                + (f" — next: {ready[0]}" if ready else ""),
                {"phase": "continue", "done": done, "remaining": ready})

    def _restore_spec_vectors(self, request: str) -> None:
        """The request's own examples, as tests (specvectors.py), are pinned
        to their modules for the loop — a mapping that lived only in the
        first run's memory. Recomputed from the request (deterministic), and
        kept only for test files the first run actually wrote."""
        self.loop.spec_tests = {}
        if not self.config.spec_vectors or self.plan is None:
            return
        if (self.config.lang or "python") != "python":
            return
        from . import specvectors
        try:
            vectors = specvectors.extract(request,
                                          [t.path for t in self.plan.tasks])
        except Exception:                                 # noqa: BLE001
            return
        if not vectors:
            return
        layout = self.planner.observe_layout()
        test_dir = layout.get("test_dir") or self.planner.test_dir or "tests"
        mapping = {module: specvectors.test_path_for(module, test_dir)
                   for module in specvectors.by_module(vectors)}
        self.loop.spec_tests = {m: p for m, p in mapping.items()
                                if self.host.fs.exists(p)}
        if self.loop.spec_tests:
            self.log.line(f"  spec vectors  still in force from the first "
                          f"run: "
                          f"{', '.join(sorted(set(self.loop.spec_tests.values())))}")

    def _reverify_before_continue(self) -> None:
        """Check what is on disk before writing anything new.

        Between a Stop and a Continue the operator may have done exactly
        what a stopped build invites: opened the file it was stuck on and
        fixed it. Continue used to trust the journal over the disk — a hand
        fix was ignored and its file regenerated over, and a file broken by
        an edit was still counted "done". Now:

          * the last run ended cleanly and NO file changed (the hashes in
            its session_end match) → nothing is re-run; this costs nothing;
          * otherwise every DONE file is verified again — a change to one
            file can break another's tests — and a pending file that holds
            real work the operator (or a crash) left is verified too:

            done, still passes       → stays done, re-counted this run
            done, now fails          → pending, its first attempt REPAIRS
                                        it from what is there
            pending, passes          → KEPT as it is and counted done
            pending, fails           → pending, repaired from what is
                                        there rather than started over

        A pending file the skeleton kept as found (the operator's own file
        its task is to extend) is only checked when it changed: passing its
        old tests says nothing about whether the task's change was made.
        """
        if self.plan is None or not self.config.reverify_on_continue:
            return
        state = self._resumed
        paths = [t.path for t in self.plan.tasks]
        now = resumable.file_hashes(self.host.fs, paths)
        changed = resumable.changed_since(
            getattr(state, "hashes", None), now)
        done_before = [t for t in self.plan.tasks if t.status == "done"]
        if changed == []:
            self.log.line(f"  re-check  none needed — nothing changed since "
                          f"the last run ended; {len(done_before)} done "
                          f"file(s) taken as they were")
            self.journal.log("reverify", changed=[], checked=[],
                             skipped=True)
            return
        kept = set(getattr(self.planner, "kept_as_found", ()) or ())
        targets: list[Task] = []
        for t in self.plan.tasks:
            if t.path not in now:
                continue
            if t.status == "done":
                targets.append(t)
            elif t.status == "pending":
                edited = changed is not None and t.path in changed
                if t.path in kept and not edited:
                    continue
                if (edited or changed is None) \
                        and self.planner._has_real_work(t.path):
                    targets.append(t)
        if not targets:
            self.journal.log("reverify", changed=changed or [], checked=[],
                             skipped=True)
            return
        self._where = "checking the files already on disk"
        self.log.rule("CONTINUE — the files already on disk, checked before "
                      "anything is written")
        if changed is None:
            self.log.line("  the last run did not end cleanly, so every file "
                          "on disk is checked")
        else:
            self.log.line(f"  changed since the last run: {', '.join(changed)}")
        self.host.emit(
            "status",
            f"checking the {len(targets)} file(s) already on disk before "
            f"writing anything new"
            + (f" — changed since the stop: {', '.join(changed)}"
               if changed else ""),
            {"phase": "reverify", "files": [t.path for t in targets],
             "changed": list(changed or [])})
        still_ok: list[str] = []
        now_failing: list[str] = []
        kept_now: list[str] = []
        repair_from: list[str] = []
        for task in targets:
            was_done = task.status == "done"
            outcome = self.loop.reverify(
                task, because="the build was continued")
            # A failure placed in ANOTHER file is the step loop's to repair
            # when it runs this task, from fresh evidence — not from this.
            self.loop.blamed.pop(task.path, None)
            diags = tuple(outcome.result.diagnostics) if outcome.result \
                else ()
            current = self.plan.task(task.id) or task
            if outcome.ok:
                label = "STILL OK" if was_done else "KEPT"
                self._record(current, outcome, label=label)
                self.plan = self.plan.replace(current.with_status("done"))
                if outcome.verified:
                    self._verified_paths.add(task.path)
                (still_ok if was_done else kept_now).append(task.path)
                if not was_done:
                    self.host.emit(
                        "status",
                        f"{task.path}: kept as it is — it passes its checks "
                        f"now, so nothing will be written over it",
                        {"task": task.path, "ok": True,
                         "verified": outcome.verified, "kept": True})
                continue
            self._record(current, outcome,
                         label="NO LONGER OK" if was_done else "STILL FAILS")
            self._verified_paths.discard(task.path)
            if diags:
                self._seeds[task.path] = diags
            self.plan = self.plan.replace(current.with_status("pending",
                                                              attempts=0))
            (now_failing if was_done else repair_from).append(task.path)
            self.host.emit(
                "warning",
                f"{task.path}: "
                + ("passed before the stop but fails now — something "
                   "changed since; it will be repaired from what is there"
                   if was_done else
                   "has work in it that does not pass yet — its first "
                   "attempt repairs that rather than starting over"),
                {"task": task.path, "phase": "reverify",
                 "was_done": was_done})
        self.journal.log("reverify", changed=list(changed or []),
                         unknown=changed is None,
                         checked=[t.path for t in targets],
                         still_ok=still_ok, now_failing=now_failing,
                         kept=kept_now, repair_from=repair_from)
        self.log.line(f"  re-check  {len(still_ok)} still ok, "
                      f"{len(kept_now)} kept as found, "
                      f"{len(now_failing)} no longer ok, "
                      f"{len(repair_from)} to repair from what is there")

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
        if self.skeleton_result.get("files"):
            pinned = self.skeleton_result.get("pinned") or []
            lines.append(
                f"[skeleton]  {self.skeleton_result.get('note', '')}"
                + (f" — pinned: {', '.join(pinned)}" if pinned else ""))
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
        if untested and built and len(set(untested) | set(unverified)) \
                >= len(built):
            # 2026-10-22: the FIRST thing the report says when nothing ran
            lines.insert(0, "[UNTESTED]  no tests ran for any file this "
                            "session built — nothing here shows the "
                            "program works; this is not an ok result")
        if untested:
            lines.append(f"[tests]     {len(untested)} of {len(built)} "
                         f"file(s) have no tests — built and run, not "
                         f"tested: {', '.join(untested)}")
        for rep in self.mutation:
            lines.append(f"[mutation]  {rep.line()}")
        for m in self.size_findings:
            lines.append(f"[size]      {m.describe()}")
        if self.constraints is not None and self.constraints.items:
            lines.append(f"[pinned]    {len(self.constraints.items)} hard "
                         f"requirement(s) in every prompt")
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

    def _spec_lint(self, request: str) -> None:
        """What the request says that the build will carry into code
        exactly as written (speclint.py, 2026-10-22). A warning and a
        journal row, never a stop: the operator decides, and a model never
        'corrects' a spelling on its own."""
        try:
            findings = speclint.lint(request)
        except Exception:                                    # noqa: BLE001
            return
        if not findings:
            return
        self.journal.log("speclint", findings=[
            {"line": f.line, "kind": f.kind, "text": f.text,
             "suggestion": f.suggestion} for f in findings])
        self.host.emit("warning", speclint.report(findings),
                       {"phase": "speclint",
                        "findings": [(f.line, f.kind, f.text)
                                     for f in findings]})
        try:
            self.log.block("specification lint", speclint.report(findings))
        except Exception:                                    # noqa: BLE001
            pass

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


def _real_provider(llm: Any) -> bool:
    """Is this model port one of the engine's own providers (a model
    server, a GGUF in-process)? The defaults that cost model calls or
    run tools per attempt — extra samples, the type checker's gate — are
    on for those and off for anything else until the host asks: a test
    double scripts one reply per attempt, and a host that wants them
    says so in its SessionConfig (ATK does)."""
    try:
        from .providers.base import ProviderBase
        return isinstance(llm, ProviderBase)
    except Exception:                                     # noqa: BLE001
        return False


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
