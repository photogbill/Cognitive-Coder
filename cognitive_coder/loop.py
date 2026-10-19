# SPDX-License-Identifier: Apache-2.0
"""generate → verify → repair. The engine.

    context → generate ──(tool round-trips)──────────────────┐
       ↑          │                                          │
       │          ▼                                          │
       │  truncation check (finish_reason=="length" ⇒        │
       │                    CONTINUE, don't regenerate)      │
       │          ▼                                          │
       │  guard → syntax pre-check → deterministic pre-fixes │
       │          ▼                                          │
       │       build → run → test                            │
       │          │                                          │
       └── parsed diagnostics (max 3; cascade languages:     │
           first error only) ◄──────────────────────────────┘

THE META-LESSON THIS FILE IS BUILT ON (Appendix D): with a frontier model you
improve results by improving the prompt. With a small model you improve
results by improving the **loop**. Every hour spent on verification, feedback
quality and error localisation is worth ten spent on prompt wording. This
module is where that hour goes.

FIVE BEHAVIOURS THAT ARE NOT OBVIOUS AND ARE ALL LOAD-BEARING:

**1. Truncation is CONTINUED, never regenerated** (D1, M32). A file that ends
mid-function is usually not a model that wrote broken code — it is a model
that ran out of `max_tokens`. `finish_reason == "length"` detects it
structurally; unbalanced delimiters are the backstop. Regenerating pays for
the whole file again and often produces a *different* file.

**2. Failed attempts are NOT accumulated in the context** (D11, M33). Attempt
3's prompt containing attempts 1 and 2 is how a model pattern-matches its own
mistakes and repeats them. The diagnostics carry forward; the broken code does
not.

**3. Deterministic pre-fixes run BEFORE the model sees an error** (F1, M35).
An insertable import, a `--fix`-able lint rule, a formatter pass. Every error
fixed by a rule is minutes of generation not spent — and models botch trivial
fixes surprisingly often, usually by rewriting the surrounding function while
they are in there. Every auto-fix is logged; if the same one recurs
constantly, the *prompt* needs changing, and the log is how anyone finds out.

**4. Stagnation detection hashes code AND diagnostics, and keeps a CYCLE SET**
(M34). Hashing diagnostics alone misses the ping-pong: fix A introduces error
B, fix B reintroduces error A, every attempt has a different diagnostic hash,
and a naive detector concludes progress is being made while the loop runs
forever. A set of every signature seen catches 2-cycles and the 3- and
4-cycles no pairwise comparison finds.

**5. On giving up, it reports THE CYCLE, not just the failure.** *"Attempts 2
and 4 produced the same code and the same two errors; it is alternating
between a missing import and an unused import."* That sentence tells the
operator exactly what is wrong, and it is usually a two-second fix by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from dataclasses import dataclass, field
import difflib
import hashlib
import re
import time
from typing import Any

from . import (
    adjudicate,
    enrich,
    guard,
    langs,
    personas,
    runner,
    specslice,
    textio,
)
from . import canvas as cv
from . import diagnostics as dx
from .context import CHARS_PER_TOKEN, Piece, build_context
from .errors import Cancelled
from .interfaces import broken_contract as _broken_contract
from .interfaces import is_pinned as _is_pinned
from .personas import (
    CONTRACT_FILE,
    PERSONAS,
    Persona,
    PromptBuilder,
    detect_commentary,
    strip_commentary,
    strip_think,
)
from .planner import _is_our_stub, _looks_like_test, strip_stub_sentinel
from .ports import NeverCancelled
from .types import (
    AttemptRecord,
    CanvasRequest,
    Completion,
    Diagnostic,
    Edit,
    Message,
    ModelCapabilities,
    RunResult,
    Task,
    TaskOutcome,
    Timeouts,
)

DEFAULT_ATTEMPTS = 4
MAX_CONTINUATIONS = 3
#: "no seed was passed" — distinct from None, which is a seed of none.
_UNSET = object()
# Consecutive attempts differing by under this fraction of characters are the
# model rearranging whitespace, not making a change (M34, cosmetic churn).
COSMETIC_THRESHOLD = 0.02
# Tool round-trips inside ONE generation. A model that has called eight tools
# has stopped writing code and started browsing.
MAX_TOOL_ROUNDS = 6


@dataclass
class LoopConfig:
    attempts: int = DEFAULT_ATTEMPTS
    temperature: float = 0.15
    max_tokens: int = 2048
    seed: int | None = None
    project_mode: bool = True
    test_first: bool = True          # F2
    use_tools: bool = True           # native tool calling where available
    autofix: bool = True             # F1
    narrow_on_stagnation: bool = True   # F5
    wall_clock_s: float = 0.0        # 0 ⇒ no ceiling (F11 lives in Session)
    #: Clocks on the generated PROGRAM's build/run/test. Nothing here bounds
    #: generation — a model is given as long as it needs.
    timeouts: Timeouts = field(default_factory=Timeouts)
    #: Test modules that were failing BEFORE this build started — the
    #: session's baseline. Their failures are other files' problems and are
    #: reported as caveats, never as the diagnostics of the file under
    #: construction. Without this, a stale `tests/test_physics.py` from an
    #: earlier session failed every module of the next build in turn.
    known_failing: tuple[str, ...] = ()
    #: DIFFUSION MODE (canvas.py, docs/DIFFUSION_MODE.md). "auto": when the
    #: loaded model fills a canvas (`ModelCapabilities.supports_canvas`), a
    #: repair rewrites the HOLE the error is in and a pinned file is written
    #: body by body; "off": never — the whole-file path, exactly as before.
    canvas: str = "auto"
    #: Canvases tried per repair attempt; the one that parses and that the
    #: model was surest of goes to the tests. Each is one short request
    #: against a prompt the server already holds, not a whole file.
    canvas_candidates: int = 2
    #: Before a pinned file's bodies are written, one ordinary call plans
    #: them in plain notes. A canvas starts with code, so a body cannot think
    #: first; the notes are the thinking, written down where every body in
    #: the file can read it.
    canvas_notes: bool = True
    #: THE JUDGE (adjudicate.py): ask which side the request supports
    #: before blaming a module for a test that disagrees with it.
    adjudicate: bool = True
    #: Questions the judge is asked per test file, at most.
    adjudicate_max: int = 3
    #: THE ESCALATION LADDER (docs/HYBRID.md). Attempt 1 is the model's
    #: best shot at the persona's temperature; attempt 2 a repair with the
    #: enriched diagnostics at the same temperature; attempt 3 a FRESH
    #: START — the broken file is not shown — at the next rung with a seed;
    #: attempt 4 and later whole-file repairs at the last rung, a new seed
    #: each. Stagnation ("identical code twice", "the same errors twice")
    #: moves to the next rung instead of stopping the file while a rung is
    #: left. On the 2026-10-16 Devstral build every attempt sampled at
    #: 0.15 with no seed, returned the same file, and stopped. Off: every
    #: attempt at the persona's temperature, and stagnation stops it.
    escalate: bool = True
    #: The rungs: attempt 1–2 at the first (the persona's own temperature
    #: when it is lower), 3 at the second, 4+ at the third.
    temperatures: tuple[float, ...] = (0.15, 0.45, 0.70)
    #: HOW MUCH OF THE REQUEST A FILE'S PROMPTS CARRY (specslice.py), in
    #: characters: whole when it fits, else the file's own section of the
    #: specification first. A repair prompt also carries the file, the test
    #: and the errors, so its budget is the smaller one.
    request_chars_first: int = specslice.FIRST_CHARS
    request_chars_repair: int = specslice.REPAIR_CHARS


@dataclass(frozen=True)
class Sampling:
    """What one attempt samples at — the record the logs print.

    `temperature` and `seed` are what reach `complete()`. `fresh` is the
    third attempt's rule: the broken file is not shown. `canvas` says the
    temperature is a formality (a diffusion model keeps its own), so the
    build log prints the seed and says so.
    """
    attempt: int
    temperature: float
    seed: int | None
    rung: int = 0
    fresh: bool = False
    canvas: bool = False
    reason: str = ""

    def describe(self) -> str:
        """The `generate` line's knobs, honestly."""
        if self.canvas:
            return (f"seed={self.seed} (temperature does not reach a canvas "
                    f"model; it anneals its own)")
        return f"temperature={self.temperature} seed={self.seed}"


@dataclass
class _Signature:
    """One attempt's identity: normalised code AND sorted diagnostics.

    `text` is the normalised code itself, kept unhashed so the cosmetic
    check can measure HOW different two attempts are, not merely whether
    they differ. Built by `_signature`; see there for what "normalised"
    has to mean for each language.
    """
    code: str
    diags: str
    text: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return (self.code, self.diags)


@dataclass
class Loop:
    """Drives one task from empty file to verified code."""

    host: Any
    codemap: Any = None
    patcher: Any = None
    journal: Any = None
    prompts: PromptBuilder = field(default_factory=PromptBuilder)
    config: LoopConfig = field(default_factory=LoopConfig)
    cancel: Any = field(default_factory=NeverCancelled)
    #: Test tasks stopped because the test disagrees with the code it tests:
    #: {test path: (module path, failing diagnostics)}. The Session reads
    #: this to queue the MODULE for a repair against the test.
    blamed: dict = field(default_factory=dict)
    #: Files whose test file exists but whose test run collected nothing.
    #: Never sealed as verified; the session end names them.
    unverified: set = field(default_factory=set)
    #: Set by `_fit_to_context` when the file being repaired cannot be shown
    #: whole: the sentence the attempt stops with. Consumed at once.
    _too_big: str = field(default="", repr=False)
    #: Paths of the current task's planned dependencies (set per run_task).
    _planned: tuple = field(default=(), repr=False)
    #: For a repair routed from another file: the path of the CALLER that
    #: ran and failed inside the file being repaired (set per run_task).
    _caller: str = field(default="", repr=False)
    #: For a task whose file is still its PINNED stub: that interface, shown
    #: on every attempt and enforced on every body (set per run_task).
    _contract: str = field(default="", repr=False)
    #: For a test task: the module it covers (set per run_task).
    _covers: str = field(default="", repr=False)
    #: {path: its last verification} for a repair's ACCEPTANCE run — the
    #: caller whose failure caused the repair, run after the repaired file
    #: passed. The session's `reverify` uses it instead of running the
    #: caller a second time (a game's run is a full timeout).
    accepted: dict = field(default_factory=dict)
    #: DIFFUSION MODE's memory of how sure the model was of each body it
    #: wrote: {path: {function: least certain line, 0..1}}. A failure that
    #: names no line of the file (an assertion in its test) is repaired in
    #: the body the model was least sure of, then the next.
    doubt: dict = field(default_factory=dict)
    _doubt_tried: dict = field(default_factory=dict, repr=False)
    #: {path: {(kind, function or error line, the error)}} — holes tried in
    #: this task. The same hole failing with the same error twice says the
    #: fault is not in the hole (a caller passing the wrong thing, a
    #: definition elsewhere in the file): that attempt is a whole file.
    _holes_tried: dict = field(default_factory=dict, repr=False)
    #: {path: the attempt whose code came from canvases} — so the build
    #: log's `generate` line says "via diffusion canvases" instead of a
    #: temperature that attempt never sampled at.
    _via_canvas: dict = field(default_factory=dict, repr=False)
    #: {(path, "first" | "repair"): the request sliced to that file}
    #: (`specslice`) — cut once per task and logged once when it is a cut.
    _slices: dict = field(default_factory=dict, repr=False)
    #: {test path: (module path, Verdict, the failing diagnostics)} — a
    #: module repair returned its file unchanged and the judge agreed the
    #: TEST is wrong. The session repairs that test next.
    test_wrong: dict = field(default_factory=dict)
    #: {(test path, failure signature): Verdict} — one question each.
    _judged: dict = field(default_factory=dict, repr=False)
    #: {module path: its spec-vector test file} (`specvectors.py`), set by
    #: the session; the runner runs it before the module's own tests.
    spec_tests: dict = field(default_factory=dict)
    #: {module path: did its spec vectors pass at the last verification}
    #: — what the judge is told when a model's test disagrees with it.
    vectors_ok: dict = field(default_factory=dict)
    #: {path: Sampling} — what the current attempt actually samples at, so
    #: the journal and BUILD_LOG.txt print the temperature and seed that
    #: were USED, never the config's (set per attempt by `_generate`).
    _sampling: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # the escalation ladder
    # ------------------------------------------------------------------
    def sampling_for(self, attempt: int, persona: Persona,
                     caps: ModelCapabilities | None = None) -> Sampling:
        """What attempt ``attempt`` samples at, and whether it starts fresh.

        A chat model climbs the temperature rungs with a derived seed from
        the third attempt. A CANVAS model is told nothing about
        temperature — DiffusionGemma's sampler anneals its own, compiled
        into the server — so for it a rung means a new seed and no warm
        start; the record says so rather than printing a temperature that
        did nothing.
        """
        rungs = tuple(self.config.temperatures) or (persona.temperature,)
        base = min(persona.temperature, rungs[0])
        canvas = bool(caps is not None
                      and getattr(caps, "supports_canvas", False))
        if not self.config.escalate or attempt <= 2:
            return Sampling(attempt=attempt,
                            temperature=persona.temperature if attempt <= 2
                            else base,
                            seed=self.config.seed, rung=0, fresh=False,
                            canvas=canvas,
                            reason="the first attempt" if attempt == 1 else
                            "a repair with the errors and the real "
                            "definitions they name")
        rung = min(attempt - 2, len(rungs) - 1)
        seed_base = self.config.seed if self.config.seed is not None else 0
        seed = seed_base + 7919 * attempt
        fresh = attempt == 3
        temperature = rungs[rung]
        if fresh:
            reason = (f"a fresh start — the broken file is not shown; "
                      f"{'a new seed' if canvas else f'temperature {temperature:.2f}, seed {seed}'}")
        else:
            reason = (f"a whole-file repair at "
                      f"{'a new seed' if canvas else f'temperature {temperature:.2f}, seed {seed}'}")
        return Sampling(attempt=attempt, temperature=temperature, seed=seed,
                        rung=rung, fresh=fresh, canvas=canvas, reason=reason)

    # ------------------------------------------------------------------
    # the judge (adjudicate.py)
    # ------------------------------------------------------------------
    def _adjudicate(self, test: Task, module: str,
                    diags: Sequence[Diagnostic], request: str,
                    attempt: int, *, again: bool = False
                    ) -> tuple[Any, bool]:
        """(verdict or None, asked now?) for a test that fails against the
        module it tests. One question per (test, failure signature), at
        most `adjudicate_max` per test; a repeat of a settled signature
        returns the earlier verdict with ``asked=False``, so the caller
        falls back to the old rule rather than looping on a test the judge
        already overruled once."""
        if not self.config.adjudicate or not module:
            return None, False
        key = (test.path, _diag_identity(diags))
        if key in self._judged:
            earlier = self._judged[key]
            # NEW EVIDENCE: a module repair returned its file unchanged —
            # "I am certain the test is wrong" — after an `unclear`. Worth
            # one more question; a settled verdict is not asked again.
            if not (again and earlier.verdict == "unclear"):
                return earlier, False
        asked_before = sum(1 for k in self._judged if k[0] == test.path)
        if asked_before >= max(1, int(self.config.adjudicate_max or 1)):
            return None, False
        module_code = self._read(module)
        test_code = self._read(test.path)
        if not module_code.strip() or not test_code.strip():
            return None, False
        self._check_cancel(f"the judge on {test.path}")
        self._emit("phase", f"{test.path}: asking which side the request "
                            f"supports",
                   {"phase": "adjudicate", "task": test.path,
                    "module": module, "attempt": attempt})
        slice_for_test = self._request_for(test, request, repair=True)
        count = len(self._vector_count(module))
        note = adjudicate.vectors_note(module, self.vectors_ok.get(module),
                                       count)
        body = adjudicate.task_text(
            module_path=module, module_code=module_code,
            test_path=test.path, test_code=test_code,
            failure=adjudicate.failure_text(diags),
            request=slice_for_test, vectors=note)
        prompt = self.prompts.build(
            PERSONAS["judge"], body,
            architecture=(self.codemap.prefix_block(module)
                          if self.codemap is not None else ""),
            epoch=(self.codemap.store.epoch if self.codemap else 0),
            contract=adjudicate.CONTRACT)
        verdict = adjudicate.ask(self.host.llm, prompt.messages(), request,
                                 cancel=self.cancel)
        self._judged[key] = verdict
        self._log_line("")
        self._log_line(f"  judge {test.path} vs {module}: "
                       f"{verdict.describe()}")
        if verdict.raw:
            log = getattr(self, "log", None)
            if log is not None:
                try:
                    log.block("the judge's reply", verdict.raw, limit=1500)
                except Exception:                        # noqa: BLE001
                    pass
        if self.journal is not None:
            try:
                self.journal.log("adjudicate", task=test.path, module=module,
                                 attempt=attempt, verdict=verdict.verdict,
                                 claimed=verdict.claimed,
                                 checked=verdict.checked,
                                 quote=verdict.quote[:400], why=verdict.why[:400],
                                 vectors=self.vectors_ok.get(module))
            except Exception:                            # noqa: BLE001
                pass
        self._emit("status", f"{test.path}: the judge says "
                             f"{verdict.describe()}",
                   {"task": test.path, "module": module,
                    "verdict": verdict.verdict, "checked": verdict.checked,
                    "quote": verdict.quote[:400]})
        return verdict, True

    def _vector_count(self, module: str) -> list[str]:
        """The names of the spec vectors a module has, for the judge's note."""
        path = self.spec_tests.get(module, "")
        if not path:
            return []
        text = self._read(path)
        return re.findall(r"^\s+def (test_\d+_\w+)\(", text, re.M)

    def _next_rung_changes(self, attempt: int, persona: Persona,
                           caps: ModelCapabilities | None) -> str:
        """Why attempt ``attempt + 1`` is a different try, or "" when it
        would be the same sampling again (so stagnation should stop)."""
        if not self.config.escalate:
            return ""
        nxt = self.sampling_for(attempt + 1, persona, caps)
        now = self.sampling_for(attempt, persona, caps)
        if nxt.fresh or nxt.seed != now.seed or \
                nxt.temperature != now.temperature:
            return nxt.reason
        return ""

    # ------------------------------------------------------------------
    # the request, as this file's prompts carry it (specslice.py)
    # ------------------------------------------------------------------
    def _request_for(self, task: Task, request: str, *, repair: bool) -> str:
        """The request sliced to ``task``: whole when it fits the budget,
        else its own section, the hard requirements and the manifest lines
        naming it — said in BUILD_LOG.txt the first time it is a cut."""
        if not request:
            return ""
        key = (task.path, "repair" if repair else "first")
        cached = self._slices.get(key)
        if cached is not None:
            return cached
        related = [p for p in (self._covers, task.test_path, self._caller)
                   if p and p != task.path]
        try:
            from .interfaces import declared_names
            names = declared_names(self._contract) if self._contract else []
        except Exception:                                # noqa: BLE001
            names = []
        budget = (self.config.request_chars_repair if repair
                  else self.config.request_chars_first)
        cut = specslice.slice_for(request, task.path, budget=budget,
                                  related=related, names=names)
        self._slices[key] = cut.text
        if not cut.whole:
            log = getattr(self, "log", None)
            if log is not None:
                try:
                    log.line(f"  request for {task.path} "
                             f"({'repairs' if repair else 'the first attempt'})"
                             f": {cut.describe()}")
                except Exception:                        # noqa: BLE001
                    pass
            if self.journal is not None:
                try:
                    self.journal.log("constraints", task=task.path,
                                     slice="repair" if repair else "first",
                                     chars=cut.chars, kept=list(cut.kept),
                                     condensed=list(cut.condensed),
                                     left_out=list(cut.left_out))
                except Exception:                        # noqa: BLE001
                    pass
        return cut.text

    # ------------------------------------------------------------------
    def run_task(self, task: Task, *, request: str = "", covers: str = "",
                 seed: Sequence[Diagnostic] = (),
                 planned: Sequence[str] = (),
                 caller: str = "", accept: Task | None = None,
                 baseline: str | None = None) -> TaskOutcome:
        """Generate, verify and repair one file until it works or it stops.

        Every phase boundary checks the cancel token (M21). A cancelled task
        leaves resumable state and rolls back any open transaction — that is
        a guarantee made here, not a hope held elsewhere.

        ``covers`` is, for a test task, the module it tests. ``seed`` makes
        attempt 1 a REPAIR against diagnostics that already exist — the
        failing output of a test this file must now pass (see
        `_test_disagrees_with_code`). ``planned`` is the PATHS of the files
        the plan says this one depends on, so their interfaces reach the
        very first attempt. `task.depends_on` holds task IDs (`t3`), and
        handing those to the codemap — which matches paths — meant the
        planned-dependency injection never fired on any build. ``caller``
        is, for a repair the session routed here from another file, the
        path of the file that ran and failed inside this one; its source
        is shown to the repair as the use that must work.

        ``accept`` and ``baseline`` make a REPAIR answer for its result
        (Oct 2, 2026: `render.py` was "REPAIRED" against main.py's crash
        with a byte-identical file, because render.py's own check — an
        import and zero tests — passed). ``baseline`` is the file as it
        stood when the repair began: an attempt that returns it unchanged
        (after normalising) ends the task at once, NOT repaired, without
        running anything. ``accept`` is the task whose failure caused the
        repair: once this file's own check passes, ``accept``'s check runs
        too, and the repair is accepted only if that failure is gone from
        this file. If it is not, its diagnostics drive the next attempt.
        """
        self._planned = tuple(p for p in planned if p and p != task.path)
        self._doubt_tried.pop(task.path, None)
        self._holes_tried.pop(task.path, None)
        self._via_canvas.pop(task.path, None)
        self._caller = caller if caller and caller != task.path else ""
        self._covers = covers if covers and covers != task.path else ""
        lang = task.lang or langs.id_for_path(task.path) or "python"
        persona = PERSONAS.get(task.persona, personas.ENGINEER)
        is_test = task.persona == "tester" or _looks_like_test(task.path)
        #: A test written FIRST (F2): the module it covers is still the
        #: skeleton's stub. Failing against that stub is what such a test
        #: must do; only its own faults are its to fix.
        before_module = bool(is_test and self._covers and _is_our_stub(
            self._read(self._covers)))
        written_first = ""
        if accept is not None:
            self.accepted.pop(accept.path, None)
        #: The interface the skeleton pinned for THIS file, when it is still
        #: the pinned stub: the names the body must keep (D12).
        on_disk = self._read(task.path)
        contract = strip_stub_sentinel(on_disk) \
            if lang == "python" and _is_pinned(on_disk) else ""
        self._contract = contract.strip()
        baseline_id = (_code_identity(baseline, lang)[0]
                       if baseline is not None and baseline.strip() else "")
        caps = self._capabilities()
        attempts: list[AttemptRecord] = []
        seen: set[tuple[str, str]] = set()
        history: list[_Signature] = []
        error_counts: list[int] = []
        code = ""
        result: RunResult | None = None
        stopped = ""
        started = time.monotonic()

        tx = None
        if self.patcher is not None:
            tx = self.patcher.begin(task.id, atomic=task.atomic)

        try:
            for n in range(1, max(1, self.config.attempts) + 1):
                self._check_cancel(f"attempt {n} of {task.path}")
                self._emit("phase", f"{task.path}: attempt {n}",
                           {"phase": "generate", "task": task.path,
                            "attempt": n})

                diag_text = ""
                autofixed: tuple[str, ...] = ()
                last = attempts[-1] if attempts else None
                if last is not None and last.diagnostics:
                    carried = last.diagnostics
                elif last is None or not last.code_sha:
                    carried = tuple(seed)      # nothing newer to go on
                else:
                    carried = ()
                if carried:
                    # M33: the DIAGNOSTICS carry forward, not the broken code.
                    lang_obj = langs.get(lang)
                    diag_text = dx.feedback(
                        carried,
                        lang_obj.feedback_cap if lang_obj else 3,
                        extra_context=bool(lang_obj and lang_obj.cascades))
                    autofixed = last.autofixes if last is not None else ()
                    # Rung 1 of the escalation ladder: change the INPUT.
                    # The error names a symbol; the codemap knows that
                    # symbol; put the truth about it under the error. A
                    # retry at temperature 0.15 with the same prompt is the
                    # same answer — this is what makes the prompt different.
                    facts = enrich.facts_for(carried, codemap=self.codemap,
                                             fs=self.host.fs, lang=lang)
                    if facts:
                        diag_text = f"{diag_text}\n\n{facts}"
                        self._emit("status",
                                   f"{task.path}: attempt {n} is enriched, "
                                   f"not retried — the prompt now carries "
                                   f"the real definitions the last error "
                                   f"named",
                                   {"task": task.path, "attempt": n,
                                    "enriched": True})

                # A REPAIR needs something to repair: code the previous
                # attempt wrote, or errors it produced. After an empty
                # reply there is neither, and attempt 2 used to be told
                # "Fix the errors reported below" with no errors below and
                # no file — by the repairer. That attempt is a first
                # attempt again, with the first attempt's prompt. A seeded
                # task is always a repair: the file exists and the failing
                # test is the error.
                repair = bool(seed) or bool(
                    last is not None and (last.code_sha or last.diagnostics))
                code, completion, continued, sent = self._generate(
                    task, persona, lang, request=request,
                    diagnostics=diag_text, autofixes=autofixed, attempt=n,
                    repair=repair, against_test=bool(seed),
                    located=carried)

                if completion.finish_reason == "cancelled":
                    raise Cancelled(f"generating {task.path}")

                if not code.strip():
                    note = ("the model used every tool round looking things "
                            "up and never wrote the file"
                            if completion.finish_reason == "tool_calls"
                            else "the model returned nothing")
                    attempts.append(AttemptRecord(
                        n=n, finish_reason=completion.finish_reason,
                        note=note))
                    # §6.9: EVERY attempt is journaled. This one cost a
                    # model call and its tokens, and it used to `continue`
                    # without a `generate` event — a journal showing one
                    # attempt where two were paid for.
                    self._journal_attempt(task, n, completion, sent,
                                          {"ok": False, "empty": True,
                                           "note": note})
                    if completion.finish_reason == "error":
                        # The provider says WHICH error when it knows —
                        # HTTP status and the server's own message. The
                        # generic sentence made "llama.cpp was started
                        # without --jinja and refused `tools`" look
                        # exactly like "the server is down".
                        stopped = (completion.error or
                                   "the model could not be reached, or it "
                                   "returned an error")
                        break
                    continue

                # -- deterministic first (C5, F1, M35) -------------------
                self._check_cancel(f"checking {task.path}")
                fixes: list[str] = []
                if self.config.autofix:
                    code, fixes = self._prefix_fixes(code, lang, task)

                # -- a repair that changed nothing is not a repair -------
                if baseline_id and \
                        _code_identity(code, lang)[0] == baseline_id:
                    attempts.append(AttemptRecord(
                        n=n, code_sha=baseline_id,
                        finish_reason=completion.finish_reason,
                        continued=continued,
                        note="returned the file unchanged"))
                    self._journal_attempt(task, n, completion, sent,
                                          {"ok": False, "unchanged": True})
                    stopped = _unchanged_sentence(
                        task.path,
                        self._caller or (accept.path if accept else "")
                        or task.test_path,
                        by_caller=bool(self._caller or accept))
                    # "I am certain the test is wrong" — the one thing an
                    # unchanged file can mean. Ask the judge; if it agrees
                    # with a quote from the request, the session repairs
                    # the TEST next instead of giving up here.
                    if seed and task.test_path and not self._caller \
                            and accept is None:
                        verdict, _asked = self._adjudicate(
                            Task(id=task.id, path=task.test_path,
                                 purpose="", persona="tester",
                                 lang=lang), task.path, tuple(seed),
                            request, n, again=True)
                        if verdict is not None and verdict.test_is_wrong:
                            self.test_wrong[task.test_path] = (
                                task.path, verdict, tuple(seed))
                            stopped = (f"the model returned the file "
                                       f"unchanged, and the judge agrees "
                                       f"the test is wrong — "
                                       f"{verdict.describe()}. "
                                       f"{task.test_path} is repaired next")
                        elif verdict is not None:
                            stopped += f". The judge: {verdict.describe()}"
                    self._emit("warning", f"{task.path}: {stopped}",
                               {"task": task.path, "unchanged": True})
                    break

                findings = guard.scan(code, lang, self.config.project_mode)
                blocked = guard.blocked(findings)
                if blocked:
                    diags = (Diagnostic(file=task.path, severity="error",
                                        message=guard.explain_to_model(findings),
                                        code="guard", tool="guard"),)
                    sig = _signature(code, lang, diags)
                    attempts.append(AttemptRecord(
                        n=n, code_sha=sig.code, diag_sha=sig.diags,
                        diagnostics=diags, autofixes=tuple(fixes),
                        finish_reason=completion.finish_reason,
                        continued=continued, note=f"refused: {blocked}"))
                    self._journal_attempt(task, n, completion, sent,
                                          {"guard": "blocked"})
                    # A refused attempt is still an attempt, and it still
                    # joins the stagnation record. It used to `continue`
                    # before the signature was kept, so a model returning
                    # the same refused file burned every attempt and ended
                    # "gave up after 6 attempts" — never "that was the same
                    # code each time".
                    stopped = self._stagnation(sig, seen, history, attempts,
                                               error_counts, diags)
                    seen.add(sig.pair)
                    history.append(sig)
                    error_counts.append(1)
                    if stopped:
                        stopped += (f" (each time it was refused before it "
                                    f"ran: {blocked})")
                        break
                    continue

                # -- write, then verify (C4, M4) -------------------------
                self._check_cancel(f"writing {task.path}")
                written = self._write(tx, task, code)
                if not written:
                    stopped = ("the change was not approved, so nothing was "
                               "written")
                    break
                # Reindex after EVERY write, not only a passing one (M30).
                # It used to wait for `result.ok`, so after a failed attempt
                # the query tools contradicted `[THE FILE AS IT STANDS]` in
                # the very next prompt: search said `parse` existed nowhere
                # while the quoted file defined it.
                if self.codemap is not None:
                    self.codemap.reindex_after_write(task.path)

                self._check_cancel(f"verifying {task.path}")
                self._emit("phase", f"{task.path}: verifying",
                           {"phase": "verify", "task": task.path,
                            "attempt": n})
                result = self._verify(task, lang)
                self._log_verify(task, n, result)
                if contract:
                    result = self._keep_contract(task, contract, code,
                                                 result)
                if result.ok and accept is not None:
                    # THE REPAIR IS JUDGED BY WHAT CAUSED IT. This file's
                    # own check passing says it imports; it does not say
                    # the caller's failure inside it is gone.
                    result = self._acceptance(task, accept, result, n)
                diags = tuple(result.diagnostics)

                sig = _signature(code, lang, diags)
                record = AttemptRecord(
                    n=n, code_sha=sig.code, diag_sha=sig.diags,
                    diagnostics=diags, autofixes=tuple(fixes),
                    finish_reason=completion.finish_reason,
                    continued=continued,
                    note=result.summary())
                attempts.append(record)
                self._journal_attempt(task, n, completion, sent,
                                      _verify_dict(result))

                if result.blocked:
                    # The ENVIRONMENT stopped this, not the code: no
                    # toolchain, or a workspace commands cannot run in.
                    # Another attempt cannot help, and asking the model to
                    # fix an environment problem makes the code worse while
                    # burning minutes.
                    stopped = result.blocked
                    self._emit("warning",
                               f"{task.path}: {result.blocked}",
                               {"task": task.path})
                    break

                if result.ok:
                    if self.journal is not None:
                        # A `patch` event with its task is what the journal
                        # counts files from, and what a host renders history
                        # from. Emitting it only on success is deliberate:
                        # the transaction log (§6.5) holds every attempted
                        # write, so this is the "what landed" view, not the
                        # "what was tried" one.
                        self.journal.log("patch", task=task.path,
                                         attempt=n, lines=len(
                                             code.splitlines()))
                    # Remember what worked, for next time (F10).
                    self._remember(attempts, diags)
                    break

                # -- a test written first fails against the stub --------
                if before_module and _test_disagrees_with_code(task.path,
                                                               diags):
                    written_first = (
                        f"written before {self._covers}, against its pinned "
                        f"interface; it fails against the stub, as a test "
                        f"written first must, and {self._covers} is built "
                        f"next to pass it")
                    if self.journal is not None:
                        self.journal.log("patch", task=task.path, attempt=n,
                                         lines=len(code.splitlines()),
                                         written_first=True)
                    break

                # -- a test that disagrees with the code is not bent (C4) --
                # … unless the JUDGE, quoting the request, says the test is
                # the one that is wrong (adjudicate.py, docs/HYBRID.md).
                if is_test and _test_disagrees_with_code(task.path, diags):
                    verdict, asked = self._adjudicate(task, covers, diags,
                                                      request, n)
                    if verdict is not None and verdict.test_is_wrong and asked:
                        judged = adjudicate.diagnostic_for_test(
                            verdict, task.path, diags)
                        diags = (judged,) + diags
                        attempts[-1] = dataclasses.replace(
                            attempts[-1], diagnostics=diags,
                            note=f"{attempts[-1].note} — the judge: the test "
                                 f"is wrong")
                        self._emit("status",
                                   f"{task.path}: the judge ruled the TEST "
                                   f"wrong, quoting the request — the test "
                                   f"is repaired next, not {covers}",
                                   {"task": task.path, "covers": covers,
                                    "verdict": "test"})
                        # the ladder below treats this as an ordinary failed
                        # attempt: the next one repairs the test
                    else:
                        stopped = _disagreement_sentence(task.path, covers,
                                                         diags)
                        if verdict is not None:
                            stopped += f". The judge: {verdict.describe()}"
                        self.blamed[task.path] = (covers, diags)
                        self._emit("warning", f"{task.path}: {stopped}",
                                   {"task": task.path, "covers": covers})
                        break

                # -- the error is in ANOTHER file of this project ----------
                # Oct 1, 2026: `main.py` ran and died at `render.py:61`
                # (`'PhysicsState' object has no attribute 'x'`). The loop
                # asked for main.py again, twice, got the same file twice
                # — correctly, there was nothing in main.py to change —
                # and gave up with "the task is probably too large". The
                # file to change was render.py. So: when the innermost
                # project frame of the failure is a different file this
                # project owns, this task stops HERE and names it, and the
                # session repairs that file against this failure, then
                # verifies this one again. Tests keep their own rule above.
                culprit = "" if is_test else self._culprit_elsewhere(
                    task, diags)
                if culprit:
                    stopped = _elsewhere_sentence(task.path, culprit, diags)
                    self.blamed[task.path] = (culprit, diags)
                    self._emit("warning", f"{task.path}: {stopped}",
                               {"task": task.path, "culprit": culprit})
                    break

                # -- stagnation and cycles (M34) -------------------------
                stopped = self._stagnation(sig, seen, history, attempts,
                                           error_counts, diags)
                seen.add(sig.pair)
                history.append(sig)
                error_counts.append(sum(1 for d in diags if d.is_error))
                if stopped:
                    # THE LADDER, NOT THE EXIT. Stagnation means the same
                    # input gave the same output; while a rung is left,
                    # the next attempt is a different input — a fresh
                    # start, a higher temperature, a new seed — and the
                    # reason is said. Only when nothing would change does
                    # the stagnation sentence end the file.
                    change = (self._next_rung_changes(n, persona, caps)
                              if n < max(1, self.config.attempts) else "")
                    if change:
                        self._emit("status",
                                   f"{task.path}: {stopped} — attempt "
                                   f"{n + 1} is {change}",
                                   {"task": task.path, "attempt": n + 1,
                                    "escalate": True})
                        self._log_line(f"  escalation: {stopped} — attempt "
                                       f"{n + 1} is {change}")
                        stopped = ""
                    else:
                        break

                if (self.config.wall_clock_s
                        and time.monotonic() - started
                        > self.config.wall_clock_s):
                    stopped = (f"the time budget for this file "
                               f"({self.config.wall_clock_s:.0f}s) ran out")
                    break

            ok = bool(result and result.ok) or bool(written_first)
            unverified = self._ran_no_tests(task, is_test, result) \
                if result is not None and result.ok else ""
            if unverified:
                self.unverified.add(task.path)
                self._emit("warning", f"{task.path}: {unverified}",
                           {"task": task.path, "verified": False})
            verified = bool(result and result.ok) and not unverified \
                and self._tests_ran(task, is_test, result)
            passes_stub = ""
            if before_module and result is not None and result.ok:
                # F2 step 3: a test that passes against `raise
                # NotImplementedError` exercises nothing of the module.
                verified = False
                passes_stub = (f"its tests PASS against the unwritten stub "
                               f"of {self._covers}, so they do not exercise "
                               f"its behaviour yet")
                self._emit("warning", f"{task.path}: {passes_stub}",
                           {"task": task.path, "verified": False})
            if tx is not None:
                if ok:
                    # committed AND verified: SEALED — unless the tests that
                    # were meant to verify it never ran, or it is a test
                    # written first, which has verified nothing yet.
                    tx.commit(verified=not unverified and not written_first
                              and not passes_stub)
                elif tx.state == "open":
                    # A and B are one change and B failed → both revert. When
                    # the task is not atomic the planner said so, and the
                    # verified work of earlier tasks is untouched either way
                    # (§6.5).
                    if task.atomic:
                        tx.rollback("the task did not verify")
                    else:
                        tx.commit(verified=False)
        except Cancelled:
            if tx is not None and tx.state == "open":
                tx.rollback("cancelled by the operator")
            raise
        except Exception:
            if tx is not None and tx.state == "open":
                tx.rollback("an unexpected failure ended the task")
            raise

        if not stopped and not (result and result.ok) and not written_first:
            stopped = self._give_up_sentence(attempts)

        caveats = tuple(result.caveats) if result else ()
        if unverified:
            caveats += (unverified,)
        if written_first:
            caveats += (written_first,)
        if passes_stub:
            caveats += (passes_stub,)
        outcome = TaskOutcome(
            task_id=task.id, path=task.path,
            ok=ok, attempts=tuple(attempts),
            result=result, stopped_because=stopped, caveats=caveats,
            verified=verified)
        self._emit("status", outcome.summary(),
                   {"task": task.path, "ok": outcome.ok,
                    "verified": outcome.verified})
        return outcome

    def _tests_ran(self, task: Task, is_test: bool,
                   result: RunResult | None) -> bool:
        """Did THIS file's tests run, at least one of them, and pass?

        The definition of verified. A module whose test file does not exist
        is built, not verified; so is one whose test run collected nothing;
        so is a test file that collected nothing (which is also a failure,
        see `runner.run_tests`). "Nothing contradicted it" is `ok`; this is
        the claim a green row is allowed to make.
        """
        if result is None or not result.ok or not result.tested:
            return False
        test_file = task.test_path or (task.path if is_test else "")
        if not test_file or not self._read(test_file):
            return False
        log = "\n".join(p.output for p in result.phases
                        if p.name == "test" and p.ok)
        return not runner.zero_tests(log)

    def reverify(self, task: Task, *, because: str = "",
                 result: RunResult | None = None) -> TaskOutcome:
        """Verify a file again WITHOUT generating anything.

        Used after the module a test covers has been repaired against that
        test: the test's own verdict is the evidence, and asking a model to
        regenerate a test that may now pass would be paying to risk it.
        ``result`` is a verification already run against the files as they
        stand — a repair's acceptance run — and is used as it is.
        """
        lang = task.lang or langs.id_for_path(task.path) or "python"
        is_test = task.persona == "tester" or _looks_like_test(task.path)
        self._check_cancel(f"verifying {task.path} again")
        self._emit("phase", f"{task.path}: verifying again",
                   {"phase": "verify", "task": task.path, "attempt": 0})
        if result is None:
            result = self._verify(task, lang)
            self._log_verify(task, 0, result)
        caveats = tuple(result.caveats)
        if because:
            caveats += (f"verified again after {because}; no new code was "
                        f"generated for it",)
        unverified = self._ran_no_tests(task, is_test, result) \
            if result.ok else ""
        if unverified:
            self.unverified.add(task.path)
            caveats += (unverified,)
        stopped = ""
        if not result.ok:
            first = next((d for d in result.diagnostics if d.is_error), None)
            # THE FAILURE MAY HAVE MOVED. Oct 2, 2026: main.py died in
            # math3d.py; math3d.py was repaired; main.py then died in
            # render.py — a different file, a different bug — and this
            # reported "it still fails" and stopped, so render.py was never
            # touched. When the error is now raised in another project
            # file, name it, exactly as a first failure would. The session
            # decides whether it may still be repaired (each file at most
            # once), so this cannot loop.
            culprit = self._culprit_elsewhere(task, result.diagnostics)
            if culprit:
                self.blamed[task.path] = (culprit, result.diagnostics)
                where = next((d for d in result.diagnostics if d.is_error
                              and d.file and _same_file(d.file, culprit)),
                             None)
                at = (f"{culprit}:{where.line}" if where and where.line
                      else culprit)
                detail = f" — {where.message}" if where and where.message \
                    else ""
                stopped = (f"it still fails after {because or 'the repair'}"
                           f", and the error is now in {at}{detail}")
                self._emit("warning", f"{task.path}: {stopped}",
                           {"task": task.path, "culprit": culprit})
            else:
                stopped = (f"it still fails after {because or 'the repair'}"
                           + (f": {first.one_line()}" if first else ""))
        outcome = TaskOutcome(task_id=task.id, path=task.path, ok=result.ok,
                              result=result, stopped_because=stopped,
                              caveats=caveats,
                              verified=(result.ok and not unverified
                                        and self._tests_ran(task, is_test,
                                                            result)))
        self._emit("status", outcome.summary(),
                   {"task": task.path, "ok": outcome.ok,
                    "verified": outcome.verified})
        return outcome

    def _keep_contract(self, task: Task, contract: str, code: str,
                       result: RunResult) -> RunResult:
        """`result`, failed with a located diagnostic when the body drops a
        name its pinned interface declares (D12)."""
        missing = _broken_contract(contract, code)
        if not missing:
            return result
        names = ", ".join(f"`{n}`" for n in missing[:6])
        diag = Diagnostic(
            file=task.path, line=1, severity="error", code="pinned-interface",
            tool="skeleton",
            message=(f"{task.path} does not define {names}, which the "
                     f"interface pinned for it declares. Other files are "
                     f"written against those names: define each with its "
                     f"pinned signature, as shown under THE INTERFACE "
                     f"PINNED FOR THIS FILE."))
        self._emit("warning", f"{task.path}: the body dropped {names} from "
                              f"its pinned interface",
                   {"task": task.path, "missing": missing})
        return RunResult(ok=False, lang=result.lang,
                         phases=tuple(result.phases),
                         diagnostics=(diag,) + tuple(result.diagnostics),
                         caveats=tuple(result.caveats))

    def _acceptance(self, task: Task, accept: Task, own: RunResult,
                    n: int) -> RunResult:
        """`own` if `accept`'s failure is gone from `task.path`, else a
        failed result carrying `accept`'s diagnostics for the next attempt.

        Gone means: `accept` now passes, or its first error is raised
        somewhere else — the failure has moved on, which `Session` follows
        with `reverify`. Still raised in this file means this repair did
        not do its job, whatever this file's own check says.
        """
        lang = accept.lang or langs.id_for_path(accept.path) or "python"
        self._emit("phase", f"{accept.path}: run again to judge the repair "
                            f"of {task.path}",
                   {"phase": "verify", "task": accept.path, "attempt": 0,
                    "accepts": task.path})
        log = getattr(self, "log", None)
        if log is not None:
            try:
                log.line(f"  acceptance: {accept.path} is run again to judge "
                         f"attempt {n} of the repair of {task.path}")
            except Exception:                            # noqa: BLE001
                pass
        again = self._verify(accept, lang)
        self._log_verify(accept, 0, again)
        self.accepted[accept.path] = again
        if again.ok:
            return own
        first = next((d for d in again.diagnostics if d.is_error), None)
        if first is None or not first.file or \
                not _same_file(first.file, task.path):
            return own
        why = f"{accept.path} still fails inside {task.path}"
        if first.line:
            why += f" at line {first.line}"
        if first.message:
            why += f": {first.message}"
        self._emit("warning",
                   f"{task.path}: its own check passes, but {why} — the "
                   f"repair is not accepted", {"task": task.path,
                                               "accept": accept.path})
        return RunResult(ok=False, phases=tuple(again.phases),
                         diagnostics=tuple(again.diagnostics),
                         caveats=tuple(own.caveats) + (
                             f"its own check passed; {why}",))

    def _ran_no_tests(self, task: Task, is_test: bool,
                      result: RunResult | None) -> str:
        """The not-verified sentence, when a test file exists and ran nothing.

        A green run with ZERO tests collected is the most dangerous green
        there is (runner.zero_tests), and it sealed files as verified: the
        test file was right there, and discovery never reached it. A file
        with no test file at all is a different, weaker claim — "built and
        ran" — and the session summary counts those separately.
        """
        test_file = task.test_path or (task.path if is_test else "")
        if not (result and test_file and self._read(test_file)):
            return ""
        if not runner.zero_tests(result.output):
            return ""
        return (f"{test_file} exists, but the test run collected none of its "
                f"tests, so {task.path} is NOT verified — check that the "
                f"test folder is a package and the test names match what "
                f"the runner discovers")

    # ------------------------------------------------------------------
    # generation, with tool round-trips and continuation
    # ------------------------------------------------------------------
    def _generate(self, task: Task, persona: Persona, lang: str, *,
                  request: str, diagnostics: str, autofixes: Sequence[str],
                  attempt: int, repair: bool | None = None,
                  against_test: bool = False,
                  located: Sequence[Diagnostic] = ()
                  ) -> tuple[str, Completion, bool, list[Message]]:
        """(code, completion, continued, the messages actually sent).

        The messages come back so the journal can hash THEM (C8). Hashing
        the request instead gave every attempt of every task the same
        `prompt_sha256`, which is a provenance field that cannot tell two
        prompts apart.
        """
        if repair is None:
            repair = attempt > 1
        caps = self.host.llm.capabilities()
        # THE LADDER: what this attempt samples at, and whether it is a
        # fresh start. Recorded per task so the journal and the build log
        # print what was used (the 2026-10-16 Devstral log printed the
        # config's 0.15 on every attempt while the host varied it unseen).
        sampling = self.sampling_for(attempt, persona, caps)
        self._sampling[task.path] = sampling
        fresh = bool(repair and sampling.fresh)
        arch = ""
        tail_extra: list[str] = []
        #: The same tail blocks as (label, text), so they can be put back
        #: together under a budget when the whole prompt would not fit.
        tail_pieces: list[tuple[str, str]] = []
        interfaces = examples = staleness = ""
        if self.codemap is not None:
            arch = self.codemap.prefix_block(task.path)
            # Planned dependencies by PATH. The session resolves the plan's
            # task IDs for us; anything in `depends_on` that already looks
            # like a path (a host driving the loop directly) is honoured too.
            planned = tuple(self._planned) + tuple(
                d for d in task.depends_on
                if ("/" in d or "." in d) and d != task.path)
            blocks = self.codemap.tail_blocks(
                task.path, count_tokens=self.host.llm.count_tokens,
                planned=planned)
            for b in blocks:
                if b.startswith("# INTERFACES"):
                    interfaces = b
                elif b.startswith("# HOW THIS CODEBASE"):
                    examples = b
                else:
                    staleness = b

        current = "" if repair else self._read(task.path)
        pinned = self._contract
        if current and _is_our_stub(current):
            current = ""
        # THE REQUEST, SLICED TO THIS FILE (specslice.py). The whole thing
        # when it fits; otherwise the file's own section, the hard
        # requirements and the lines naming it — never the first N
        # characters, which is how every formula of the Star Trek spec
        # fell out of every prompt that was not a first attempt.
        request = self._request_for(task, request, repair=repair)
        if not repair and current.strip():
            # THE FILE ALREADY EXISTS AND IS REAL. The first attempt used to
            # say "Write the complete contents of `x`" and never show `x`,
            # so "improve this" or "extend src/util.py" regenerated a
            # working file blind — and with auto-apply on, replaced it.
            # The model is shown the file and asked for a CHANGE, and the
            # same context check as a repair applies: a file too large to
            # show whole is refused, never rewritten from its first half.
            body = _change_task_text(task, request, lang)
            tail_extra.append(f"[THE FILE AS IT STANDS]\n{current}")
            tail_pieces.append(("THE FILE AS IT STANDS", current))
        elif not repair:
            body = _first_task_text(task, request, lang)
            if pinned:
                # THE CONTRACT THIS FILE WAS PINNED TO. Other files are
                # being written against these exact names and types; a body
                # that renames `Segment` or turns it into a tuple breaks
                # them all, invisibly, until one of them runs.
                label, text = _pinned_piece(task.path, pinned)
                tail_extra.append(f"[{label}]\n{text}")
                tail_pieces.append((label, text))
            test_source = self._read(task.test_path) if (
                task.test_path and _looks_like_test(task.test_path)
                and not (task.persona == "tester"
                         or _looks_like_test(task.path))) else ""
            if test_source.strip():
                # THE TEST EXISTS BEFORE THE MODULE — written first (F2),
                # or already in the folder. "Its tests live in X and must
                # pass" without X in view asked the model to pass a test
                # it could not read.
                rule = ("These tests already exist and are the "
                        "specification: write this file so they pass. They "
                        "are not yours to change.")
                tail_extra.append(
                    f"[THE TEST THIS FILE MUST PASS — {task.test_path}]\n"
                    f"{test_source}\n\n{rule}")
                tail_pieces.append(
                    (f"THE TEST THIS FILE MUST PASS — {task.test_path}",
                     f"{test_source}\n\n{rule}"))
        else:
            # Diagnostics go in the TAIL, not here — tail_for puts them
            # immediately before the output contract, where recency helps
            # most (D7). Repeating them in the task body would spend
            # context restating the same errors twice.
            body = (personas.fresh_start_task(task.path, task.purpose,
                                              request=request, lang=lang)
                    if fresh else
                    personas.repair_task(task.path, task.purpose,
                                         autofixes=autofixes,
                                         request=request))
            persona = PERSONAS["engineer" if fresh else "repairer"]
            existing = "" if fresh else self._read(task.path)
            if existing:
                tail_extra.append(f"[THE FILE AS IT STANDS]\n{existing}")
                tail_pieces.append(("THE FILE AS IT STANDS", existing))
            if fresh:
                self._emit("status",
                           f"{task.path}: attempt {attempt} starts fresh — "
                           f"the broken file is not shown; the interface, "
                           f"the test and the errors are",
                           {"task": task.path, "attempt": attempt,
                            "fresh": True})
            if pinned:
                # Still the contract: a repair that fixes one error by
                # renaming a pinned type has moved the error, not fixed it.
                label, text = _pinned_piece(task.path, pinned)
                tail_extra.append(f"[{label}]\n{text}")
                tail_pieces.append((label, text))
            test_source = self._read(task.test_path) if (
                against_test and task.test_path
                and _looks_like_test(task.test_path)) else ""
            caller_source = self._read(self._caller) if self._caller else ""
            if test_source:
                # The test is the specification this repair answers to. The
                # escape hatch matters: a model that believes the test is
                # wrong should say so by changing nothing, which the
                # stagnation check turns into an honest stop, rather than
                # bend the code into agreement with a mistake.
                tail_extra.append(
                    f"[THE TEST THIS FILE MUST PASS — {task.test_path}]\n"
                    f"{test_source}\n\n"
                    f"The test is the specification. Change this file so "
                    f"the test passes. If you are certain the test itself "
                    f"is wrong, return this file unchanged.")
                tail_pieces.append(
                    (f"THE TEST THIS FILE MUST PASS — {task.test_path}",
                     f"{test_source}\n\nThe test is the specification. "
                     f"Change this file so the test passes. If you are "
                     f"certain the test itself is wrong, return this file "
                     f"unchanged."))
            if caller_source:
                # A CALLER that ran and died inside this file. The error
                # under [ERRORS] is located here; the caller is shown so
                # the repair can see how this file is used, and is not the
                # model's to change in this task.
                rule = (f"`{self._caller}` ran and failed inside this "
                        f"file, at the line the error names. Fix THIS file "
                        f"so that run succeeds. The caller is not yours to "
                        f"change here. Decide from this file's purpose and "
                        f"the real definitions shown under the error "
                        f"whether this file misuses another module (fix "
                        f"the misuse) or lacks something the caller is "
                        f"right to expect (add it). If you are certain the "
                        f"caller itself is wrong, return this file "
                        f"unchanged.")
                tail_extra.append(
                    f"[THE CALLER THAT FAILS INSIDE THIS FILE — "
                    f"{self._caller}]\n{caller_source}\n\n{rule}")
                tail_pieces.append(
                    (f"THE CALLER THAT FAILS INSIDE THIS FILE — "
                     f"{self._caller}", f"{caller_source}\n\n{rule}"))

        # THE MODULE UNDER TEST, for a test written after it — the default
        # order puts each test right after its module, so the tester
        # arrives when the module exists; it
        # was shown the interface — names and signatures — and for pure
        # arithmetic the signature does not carry the formula. On Oct 1,
        # 2026 the tester asserted `x = cx + x/z·(w/2)` to seven places
        # against a projection that scales by field of view; the spec had
        # asked only for scale and Y. The source is shown, with the rule
        # that decides what to assert. A test written FIRST (F2) sees the
        # module's PINNED interface instead, with the rule for that case.
        is_test_task = task.persona == "tester" or _looks_like_test(task.path)
        if is_test_task and self._covers:
            module_source = self._read(self._covers)
            if _is_pinned(module_source):
                # A TEST WRITTEN FIRST (F2): the module is its pinned
                # interface and nothing else. The test is the
                # specification the module will be written to.
                rule = ("The module is not written yet: this is its pinned "
                        "interface, and it will be written next, to pass "
                        "your tests. Call it only through these names and "
                        "signatures. Test the behaviour the request asks "
                        "for; where you assert an exact number, choose "
                        "inputs whose result follows plainly from the "
                        "request, and say in a comment how you got it.")
                stub = strip_stub_sentinel(module_source).strip()
                tail_extra.append(
                    f"[THE MODULE UNDER TEST — {self._covers}, interface "
                    f"only]\n{stub}\n\n{rule}")
                tail_pieces.append(
                    (f"THE MODULE UNDER TEST — {self._covers}, interface "
                     f"only", f"{stub}\n\n{rule}"))
            elif module_source.strip() and not _is_our_stub(module_source):
                rule = ("Test the behaviour the request asks for and what "
                        "this module's docstrings promise; include the edge "
                        "case the request implies. Where a test needs an "
                        "exact number, take it from how this module "
                        "actually computes it — never from a formula or a "
                        "constant of your own that the request does not "
                        "state. If the module plainly contradicts the "
                        "request, assert what the request says and let the "
                        "test fail; that failure is the finding.")
                tail_extra.append(
                    f"[THE MODULE UNDER TEST — {self._covers}]\n"
                    f"{module_source}\n\n{rule}")
                tail_pieces.append(
                    (f"THE MODULE UNDER TEST — {self._covers}",
                     f"{module_source}\n\n{rule}"))

        # -- DIFFUSION MODE: a hole, not the whole file (canvas.py) --------
        # Everything above — the interfaces, the pinned contract, the test,
        # the caller, the enriched errors — is the context of the hole too;
        # only the task, the file view and the output contract differ. Any
        # reason the canvas cannot serve (no located line, a hole that does
        # not fit, no candidate that parses) falls through to the whole-file
        # path below, unchanged.
        if self._canvas_ready(caps):
            via = self._canvas_generate(
                task, persona, lang, repair=repair, located=located,
                pinned=pinned, arch=arch, interfaces=interfaces,
                examples=examples, staleness=staleness,
                diagnostics=diagnostics, tail_extra=tail_extra,
                caps=caps, attempt=attempt, request=request,
                autofixes=autofixes)
            if via is not None:
                return via

        prompt = self.prompts.build(
            persona, body, architecture=arch,
            epoch=(self.codemap.store.epoch if self.codemap else 0),
            interfaces=interfaces, examples=examples, staleness=staleness,
            diagnostics=diagnostics, contract=CONTRACT_FILE,
            extra=tail_extra)
        messages = prompt.messages()
        messages = self._fit_to_context(
            messages, caps, task, attempt,
            rebuild=lambda extra: self.prompts.build(
                persona, body, architecture=arch,
                epoch=(self.codemap.store.epoch if self.codemap else 0),
                staleness=staleness, diagnostics=diagnostics,
                contract=CONTRACT_FILE, extra=extra).messages(),
            pieces=[*tail_pieces,
                    *([("INTERFACES OF WHAT THIS FILE USES", interfaces)]
                      if interfaces else []),
                    *([("HOW THIS CODEBASE DOES THINGS", examples)]
                      if examples else [])])
        if self._too_big:
            # No model call: the answer could only be a truncated file.
            # Returned as an error completion so the attempt is journaled
            # and the loop stops with this sentence, like any provider
            # error (the same path, so the same guarantees).
            why, self._too_big = self._too_big, ""
            return "", Completion(text="", finish_reason="error",
                                  error=why, model=caps.name), False, \
                messages

        tools = ()
        if (self.config.use_tools and caps.supports_tools
                and self.codemap is not None):
            tools = self.codemap.tool_specs(allow_patch=False,
                                            allow_tests=False)
            self.codemap.reset_lookups()

        completion = self._complete(messages, tools=tools,
                                    temperature=sampling.temperature,
                                    seed=sampling.seed)

        # -- tool round-trips are ordinary complete() cycles (§6.9) -----
        rounds = 0
        while (completion.finish_reason == "tool_calls"
               and rounds < MAX_TOOL_ROUNDS):
            self._check_cancel(f"tool call from {task.path}")
            rounds += 1
            messages = list(messages) + [Message(
                role="assistant", content=completion.text,
                tool_calls=completion.tool_calls)]
            for call in completion.tool_calls:
                answer = self.codemap.call_tool(call.name, call.arguments)
                messages.append(Message(role="tool", content=answer,
                                        tool_call_id=call.id))
            completion = self._complete(messages, tools=tools,
                                        temperature=sampling.temperature,
                                        seed=sampling.seed)

        text = strip_think(completion.text)       # D13, M37 — before ANY use

        # -- the text-marker fallback, for models without tools (M31) ---
        if (not tools and self.codemap is not None
                and not caps.supports_tools):
            answer = self.codemap.answer_text_lookups(text)
            if answer:
                messages = list(messages) + [
                    Message(role="assistant", content=text),
                    Message(role="user", content=answer)]
                completion = self._complete(messages,
                                            temperature=sampling.temperature,
                                            seed=sampling.seed)
                text = strip_think(completion.text)

        # -- truncation: CONTINUE, do not regenerate (D1, M32) ----------
        continued = False
        continuations = 0
        while (_is_truncated(completion, text, lang)
               and continuations < MAX_CONTINUATIONS):
            self._check_cancel(f"continuing {task.path}")
            continuations += 1
            continued = True
            tail = "\n".join(text.splitlines()[-8:])
            self._emit("warning",
                       f"{task.path}: the answer hit the length limit; "
                       f"continuing from where it stopped rather than "
                       f"starting again.",
                       {"task": task.path, "continuation": continuations})
            if self.journal is not None:
                self.journal.log("continuation", task=task.path,
                                 attempt=attempt, n=continuations,
                                 lines_so_far=len(text.splitlines()))
            more = self._complete(
                list(messages) + [
                    Message(role="assistant", content=text),
                    Message(role="user", content=personas.continuation_task(
                        tail, len(text.splitlines())))],
                temperature=persona.temperature)
            addition = strip_think(more.text)
            if not addition.strip():
                break
            text = _join_continuation(text, addition)
            completion = Completion(
                text=text, finish_reason=more.finish_reason,
                tokens_in=completion.tokens_in + more.tokens_in,
                tokens_out=completion.tokens_out + more.tokens_out,
                model=more.model or completion.model,
                prompt_ms=more.prompt_ms or completion.prompt_ms)

        # -- commentary detector at the CONSUMING call site (M36) -------
        if detect_commentary(text, lang):
            self._emit("warning",
                       f"{task.path}: the model wrote commentary as well as "
                       f"code; the code has been extracted from it.",
                       {"task": task.path})
            text = strip_commentary(text, lang)

        # A copied stub marker would make a finished file read as a stub.
        code = strip_stub_sentinel(_extract(text, lang, task.path))
        # The prompt that produced the answer: after tool round-trips and
        # the text-lookup fallback, before any continuation (a continuation
        # is derived from this prompt, not a different one).
        return code, completion, continued, list(messages)

    # ------------------------------------------------------------------
    # DIFFUSION MODE (canvas.py)
    # ------------------------------------------------------------------
    def _canvas_ready(self, caps: ModelCapabilities) -> bool:
        """The loaded model fills a canvas, and nothing has switched it off."""
        return (str(self.config.canvas).lower() != "off"
                and bool(getattr(caps, "supports_canvas", False))
                and callable(getattr(self.host.llm, "canvas_fill", None)))

    def _canvas_generate(self, task: Task, persona: Persona, lang: str, *,
                         repair: bool, located: Sequence[Diagnostic],
                         pinned: str, arch: str, interfaces: str,
                         examples: str, staleness: str, diagnostics: str,
                         tail_extra: Sequence[str], caps: ModelCapabilities,
                         attempt: int, request: str,
                         autofixes: Sequence[str]
                         ) -> tuple[str, Completion, bool, list[Message]] | None:
        """(code, completion, False, messages) — or None: use the whole file.

        Two jobs. A REPAIR rewrites the hole its error is in (or, when the
        error names no line of this file, the body the model was least sure
        of). A FIRST attempt at a file whose interface was pinned writes its
        bodies one hole at a time, every signature held.
        """
        code = self._read(task.path)
        context = [x for x in tail_extra
                   if not x.startswith("[THE FILE AS IT STANDS]")]
        epoch = self.codemap.store.epoch if self.codemap else 0
        count = self.host.llm.count_tokens

        def messages(task_text: str, blocks: Sequence[str], contract: str,
                     who: Persona = persona, diags: str = diagnostics
                     ) -> list[Message]:
            return self.prompts.build(
                who, task_text, architecture=arch, epoch=epoch,
                interfaces=interfaces, examples=examples,
                staleness=staleness, diagnostics=diags, contract=contract,
                extra=[*context, *blocks]).messages()

        def fits(msgs: Sequence[Message], canvases: int) -> bool:
            try:
                used = sum(count(m.content) for m in msgs)
            except Exception:                            # noqa: BLE001
                return False
            # the server reserves one canvas MORE than the reply can use
            # (it must fit the prompt, the canvases, and the commit of the
            # last one): `fit = (n_ctx - prompt) / canvas - 1 >= 1`
            return used + cv.CANVAS_TOKENS * (canvases + 1) + 64 \
                <= int(caps.context_tokens or 0)

        if repair:
            hole = cv.repair_hole(code, lang, task.path, located,
                                  same_file=_same_file, count_tokens=count)
            err = cv.located_error(located, task.path, _same_file)
            # A FAILURE THAT NAMES NO LINE HERE — an assertion raised in
            # the test — is located through THE TEST: the functions of
            # this file the failing test calls, the asserting line's own
            # first. Only then DOUBT, the body the model was least sure
            # of. A located error whose hole did not fit is a whole-file
            # repair — not a rewrite of some other function.
            doubted = named = ""
            if hole is None and err is None and lang == "python":
                tried = self._doubt_tried.setdefault(task.path, set())
                test_source = self._read(task.test_path) \
                    if task.test_path else ""
                for candidate in cv.test_named_functions(code, located,
                                                         test_source):
                    if candidate in tried:
                        continue
                    tried.add(candidate)
                    named = candidate
                    hole = cv.function_hole(code, task.path, candidate,
                                            why="test")
                    if hole is not None and not cv._fits(hole, count):
                        hole = None
                    break
            if (hole is None and err is None and lang == "python"
                    and not named and self.doubt.get(task.path)):
                tried = self._doubt_tried.setdefault(task.path, set())
                name = cv.least_sure(self.doubt[task.path], code, tried)
                if name:
                    tried.add(name)
                    doubted = name
                    hole = cv.function_hole(code, task.path, name)
                    if hole is not None and not cv._fits(hole, count):
                        hole = None
            if hole is None:
                if err is not None:
                    why = (f"the error at line {err.line} could not be cut "
                           f"out as a hole that fits one canvas and the "
                           f"model's context")
                elif named:
                    why = (f"the failure names no line of this file; the "
                           f"failing test calls `{named}`, which could not "
                           f"be cut out as a hole that fits one canvas — "
                           f"the next attempt tries the next name")
                elif doubted:
                    why = (f"the failure names no line of this file, and "
                           f"the body the model was least sure of "
                           f"(`{doubted}`) could not be cut out as a hole "
                           f"that fits one canvas — the next attempt tries "
                           f"the next one")
                elif self.doubt.get(task.path):
                    why = ("the failure names no line of this file, and "
                           "every body the model doubted has been tried")
                else:
                    why = "the failure names no line of this file"
                self._canvas_skip(task, attempt, why)
                return None
            # the same FUNCTION (or, for a window, the same error line) with
            # the same error: the region's text changes every attempt, so
            # it cannot be part of the key
            key = (hole.kind, hole.name or str(hole.line),
                   (err.message if err is not None else "doubt"))
            tried = self._holes_tried.setdefault(task.path, set())
            if key in tried:
                self._emit("status",
                           f"{task.path}: {hole.label} was rewritten already "
                           f"and the same error came back — the fault is not "
                           f"in those lines, so this attempt rewrites the "
                           f"whole file",
                           {"task": task.path, "attempt": attempt,
                            "canvas": {"kind": "repeat",
                                       "region": hole.label}})
                self._canvas_skip(task, attempt,
                                  f"{hole.label} was rewritten already and "
                                  f"the same error came back — the fault is "
                                  f"not in those lines")
                return None
            return self._canvas_repair(task, lang, code, hole, messages,
                                       fits, caps, attempt, request,
                                       autofixes, tried=(tried, key))
        if pinned and lang == "python" and _is_pinned(code):
            return self._canvas_bodies(task, persona, lang, code, messages,
                                       fits, caps, attempt, request)
        return None

    def _canvas_repair(self, task: Task, lang: str, code: str, hole: Any,
                       messages: Any, fits: Any, caps: ModelCapabilities,
                       attempt: int, request: str, autofixes: Sequence[str],
                       tried: tuple | None = None
                       ) -> tuple[str, Completion, bool, list[Message]] | None:
        view = cv.marked_view(code, hole, _comment_for(lang))
        task_text = cv.repair_text(task.path, task.purpose, request,
                                   autofixes)
        blocks = [f"[THE FILE AS IT STANDS — {task.path}, the lines to "
                  f"rewrite are marked]\n{view}", cv.hole_block(hole)]

        def for_style(style: str) -> list[Message]:
            return messages(task_text, blocks, cv.contract(hole, style),
                            who=PERSONAS["repairer"])
        # THE LADDER FOR A CANVAS MODEL: from the third attempt, a derived
        # seed and no warm start — every candidate from noise. The old
        # lines are what the model copied back twice already.
        used = self._sampling.get(task.path)
        fresh = bool(used is not None and used.rung > 0)
        plan = cv.requests_for(hole, for_style,
                               max(1, int(self.config.canvas_candidates or 1)),
                               seed=(used.seed if fresh and used is not None
                                     else self.config.seed),
                               attempt=attempt,
                               count_tokens=self.host.llm.count_tokens,
                               lang=lang, fresh=fresh)
        if not plan or not fits(plan[0][2].messages, 1):
            self._canvas_skip(
                task, attempt,
                f"no canvas could be planned for {hole.label}" if not plan
                else f"the prompt for {hole.label} leaves no room for a "
                     f"canvas in the model's context")
            return None
        why = ("its error is there" if hole.why == "located" else
               "the failure names no line of this file, and the failing "
               "test calls this function" if hole.why == "test" else
               "the failure names no line of this file, and the model was "
               "least sure of this body when it wrote it")
        self._emit("status",
                   f"{task.path}: attempt {attempt} rewrites "
                   f"{hole.label}{f' (`{hole.name}`)' if hole.name else ''} "
                   f"as a hole — {why}; {len(plan)} canvas(es)",
                   {"task": task.path, "attempt": attempt,
                    "canvas": {"kind": "repair", "region": hole.label,
                               "why": hole.why}})
        cands: list[cv.Candidate] = []
        for style, noise, req in plan:
            self._check_cancel(f"a hole in {task.path}")
            result = self.host.llm.canvas_fill(req, cancel=self.cancel)
            if result.finish_reason == "cancelled":
                return "", Completion(text="", finish_reason="cancelled",
                                      model=result.model or caps.name), \
                    False, list(req.messages)
            cands.append(cv.judge(code, hole, style, noise, result, lang))
        pick = cv.choose(cands)
        self._journal_canvas(task, attempt, "repair", hole, cands, pick)
        if pick is None:
            reasons = "; ".join(sorted({c.problem for c in cands}))
            self._emit("warning",
                       f"{task.path}: no hole came back usable ({reasons}); "
                       f"this attempt rewrites the whole file instead",
                       {"task": task.path, "attempt": attempt})
            return None
        chosen = cands[pick]
        if tried is not None:
            # recorded only once a hole was KEPT: a hole that never came
            # back usable has not been "rewritten already"
            tried[0].add(tried[1])
        self._via_canvas[task.path] = attempt
        self._emit_canvas(task, attempt, hole, chosen)
        if hole.kind == "body" and hole.name:
            self.doubt.setdefault(task.path, {})[hole.name] = chosen.least
        return chosen.code, self._canvas_completion(cands, chosen, caps), \
            False, list(plan[pick][2].messages)

    def _canvas_bodies(self, task: Task, persona: Persona, lang: str,
                       code: str, messages: Any, fits: Any,
                       caps: ModelCapabilities, attempt: int, request: str
                       ) -> tuple[str, Completion, bool, list[Message]] | None:
        holes = cv.stub_holes(code, task.path)
        if not holes:
            self._canvas_skip(task, attempt,
                              "the file is pinned, but no stub body was "
                              "found to write as a hole")
            return None
        names = [h.name for h in holes]
        task_text = cv.stub_text(task.path, task.purpose, request)
        # Will the first body fit at all — with room for the notes? Asked
        # BEFORE the notes call, which is a whole generation to waste.
        room = ["[IMPLEMENTATION NOTES]\n" + "note " * (cv.NOTES_CHARS // 5)]
        first = messages(task_text, [
            *room, f"[THE FILE BEING WRITTEN — {task.path}]\n"
            f"{cv.marked_view(code, holes[0], _comment_for(lang))}",
            cv.hole_block(holes[0])], cv.contract(holes[0], "open"))
        if not fits(first, cv.OPEN_CANVASES_STUB):
            self._canvas_skip(task, attempt,
                              f"the first body's prompt leaves no room for "
                              f"{cv.OPEN_CANVASES_STUB} canvases in the "
                              f"model's context")
            return None
        notes = ""
        if self.config.canvas_notes:
            notes = self._canvas_notes(task, names, messages, attempt,
                                       request=request)
        notes_block = ([f"[IMPLEMENTATION NOTES — planned before any body "
                        f"was written]\n{notes}"] if notes else [])
        self._emit("status",
                   f"{task.path}: writing {len(names)} pinned bod"
                   f"{'y' if len(names) == 1 else 'ies'} as holes; the "
                   f"interface stays exactly as pinned"
                   f"{' (notes planned first)' if notes else ''}",
                   {"task": task.path, "attempt": attempt,
                    "canvas": {"kind": "bodies", "functions": names}})
        current = code
        written: list[cv.Candidate] = []
        failed: list[str] = []
        sent: list[Message] = []
        for i, name in enumerate(names):
            self._check_cancel(f"the body of {name} in {task.path}")
            hole = next((h for h in cv.stub_holes(current, task.path)
                         if h.name == name), None)
            if hole is None:
                # not skipped silently: it is neither written nor tried,
                # and the summary must name it
                failed.append(name)
                self._readable(task, attempt,
                               f"`{name}` was not written as a hole — its "
                               f"stub could not be found again after the "
                               f"bodies before it were written")
                continue
            view = cv.marked_view(current, hole, _comment_for(lang))
            msgs = messages(task_text, [
                *notes_block,
                f"[THE FILE BEING WRITTEN — {task.path}]\n{view}",
                cv.hole_block(hole)], cv.contract(hole, "open"))
            if not fits(msgs, cv.OPEN_CANVASES_STUB):
                self._canvas_skip(task, attempt,
                                  f"the prompt for `{name}` leaves no room "
                                  f"for {cv.OPEN_CANVASES_STUB} canvases in "
                                  f"the model's context ({len(written)} "
                                  f"bod{'y' if len(written) == 1 else 'ies'}"
                                  f" written before it are discarded)")
                return None
            sent = msgs
            got = None
            tries: list[cv.Candidate] = []
            for k in range(2):         # a second seed when the first fails
                seed = (None if self.config.seed is None
                        else self.config.seed + 7919 * k + 31 * i)
                result = self.host.llm.canvas_fill(CanvasRequest(
                    messages=tuple(msgs), head=hole.head, seed=seed,
                    max_canvases=cv.OPEN_CANVASES_STUB), cancel=self.cancel)
                if result.finish_reason == "cancelled":
                    return "", Completion(text="", finish_reason="cancelled",
                                          model=result.model or caps.name), \
                        False, msgs
                cand = cv.judge(current, hole, "open", 1.0, result, lang)
                tries.append(cand)
                if cand.usable:
                    got = cand
                    break
            self._journal_canvas(task, attempt, "body", hole, tries,
                                 len(tries) - 1 if got else None)
            if got is None:
                failed.append(name)
                continue
            current = got.code
            written.append(got)
            self._emit_canvas(task, attempt, hole, got)
            self.doubt.setdefault(task.path, {})[name] = got.least
        if not written or len(failed) * 2 > len(names):
            self._emit("warning",
                       f"{task.path}: {len(failed)} of {len(names)} bodies "
                       f"did not come back usable as holes; this attempt "
                       f"writes the whole file instead",
                       {"task": task.path, "attempt": attempt})
            self._canvas_skip(task, attempt,
                              f"{len(failed)} of {len(names)} bodies did "
                              f"not come back usable as holes")
            return None
        if failed:
            # Left as `raise NotImplementedError`: the tests fail THERE, at
            # a located line, and the next attempt repairs exactly that hole.
            self._emit("warning",
                       f"{task.path}: {', '.join(f'`{n}`' for n in failed)} "
                       f"not written yet — the tests will say so at its "
                       f"line, and the next attempt fills it",
                       {"task": task.path, "attempt": attempt})
        self._readable(task, attempt,
                       f"{len(written)} of {len(names)} pinned bod"
                       f"{'y' if len(names) == 1 else 'ies'} written as "
                       f"holes" + (f"; left as stubs for the tests to "
                                   f"locate: {', '.join(failed)}"
                                   if failed else ""))
        self._via_canvas[task.path] = attempt
        final = strip_stub_sentinel(current)
        return final, self._canvas_completion(written, written[-1], caps,
                                              text=final), False, sent

    def _canvas_notes(self, task: Task, names: Sequence[str],
                      messages: Any, attempt: int = 0,
                      request: str = "") -> str:
        """One ordinary call: plain notes on how each body works — with the
        file's slice of the request in view, so a formula the request
        states reaches the notes word for word."""
        msgs = messages(cv.notes_text(task.path, names, request), [],
                        cv.NOTES_CONTRACT, who=personas.ENGINEER, diags="")
        try:
            completion = self._complete(msgs,
                                        temperature=personas.ENGINEER.temperature)
        except Exception as exc:                         # noqa: BLE001
            self._readable(task, attempt, "no implementation notes — the "
                           "planning call failed; bodies are written "
                           "without them", details=[f"error: {exc}"])
            return ""
        if completion.finish_reason in ("error", "cancelled"):
            self._readable(task, attempt, f"no implementation notes — the "
                           f"planning call ended "
                           f"({completion.finish_reason}); bodies are "
                           f"written without them")
            return ""
        text = strip_think(completion.text or "").strip()
        self._readable(
            task, attempt,
            f"implementation notes planned for {len(names)} bod"
            f"{'y' if len(names) == 1 else 'ies'} before any was written"
            + ("" if text else " — the model wrote none"),
            details=[f"prefill {completion.prompt_ms} ms, decode "
                     f"{getattr(completion, 'decode_ms', 0)} ms, "
                     f"{len(text):,} characters"
                     + (f" (the first {cv.NOTES_CHARS:,} are shown to "
                        f"every body)" if len(text) > cv.NOTES_CHARS
                        else "")],
            blocks=[("the notes", text)])
        if self.journal is not None:
            self.journal.log("canvas", task=task.path, kind="notes",
                             functions=list(names), chars=len(text),
                             prompt_ms=completion.prompt_ms,
                             model=completion.model)
        return text[:cv.NOTES_CHARS]

    def _canvas_completion(self, cands: Sequence[Any], chosen: Any,
                           caps: ModelCapabilities, text: str = ""
                           ) -> Completion:
        """A Completion for the journal and the attempt record: what the
        canvases cost, summed, and what they produced."""
        results = [c.result for c in cands]
        out = text or chosen.code
        return Completion(
            text=out, finish_reason="stop",
            tokens_in=0, tokens_out=self.host.llm.count_tokens(out)
            if out else 0,
            model=chosen.result.model or caps.name,
            prompt_ms=sum(int(r.prompt_ms or 0) for r in results),
            decode_ms=sum(int(r.decode_ms or 0) for r in results))

    def _journal_canvas(self, task: Task, attempt: int, kind: str, hole: Any,
                        cands: Sequence[Any], pick: int | None) -> None:
        self._log_canvas(task, attempt, kind, hole, cands, pick)
        if self.journal is None:
            return
        self.journal.log("canvas", task=task.path, attempt=attempt,
                         kind=kind, region=hole.label, hole=hole.kind,
                         function=hole.name, line=hole.line, why=hole.why,
                         candidates=[c.summary() for c in cands],
                         chosen=pick)

    def _log_canvas(self, task: Task, attempt: int, kind: str, hole: Any,
                    cands: Sequence[Any], pick: int | None) -> None:
        """Every canvas of one hole, in BUILD_LOG.txt: why this hole, what
        each canvas cost and why any was rejected, and the kept lines with
        how sure the model was of each."""
        name = f" (`{hole.name}`)" if hole.name else ""
        what = "rewrite" if kind == "repair" else "write"
        if pick is not None:
            outcome = f"kept canvas {pick + 1} of {len(cands)}"
        else:
            outcome = (f"none of {len(cands)} canvas(es) usable"
                       + (" — this attempt writes the whole file"
                          if kind == "repair" else ""))
        if hole.kind == "stub":
            why = (f"`{hole.name}` is a pinned stub: its body is written "
                   f"with the signature held")
        elif hole.why == "doubt":
            why = ("the failure names no line of this file, and the model "
                   "was least sure of this body when it wrote it")
        elif hole.why == "test":
            why = ("the failure names no line of this file, and the failing "
                   "test calls this function (the asserting line's own "
                   "call first)")
        else:
            why = f"its error is at line {hole.line}"
        details = [f"why this hole: {why}",
                   *(f"canvas {i}: {c.describe()}"
                     for i, c in enumerate(cands, 1))]
        blocks: list[tuple[str, str]] = []
        if kind == "repair" and hole.original.strip():
            blocks.append(("the lines cut out", hole.original))
        for i, c in enumerate(cands, 1):
            if i - 1 == pick:
                blocks.append((f"the hole as kept — {cv.DOUBT_LEGEND}",
                               "\n".join(cv.doubt_rows(c))))
            elif c.fill:
                blocks.append((f"canvas {i} as it came back "
                               f"({'usable, not chosen' if c.usable else 'rejected'})",
                               c.fill))
            elif (c.result.text or "").strip():
                blocks.append((f"canvas {i}'s raw text (rejected)",
                               c.result.text))
        self._readable(task, attempt,
                       f"{what} {hole.label}{name} as a hole — {outcome}",
                       details=details, blocks=blocks)

    def _canvas_skip(self, task: Task, attempt: int, why: str) -> None:
        """Diffusion mode stepping aside for a whole-file attempt: said in
        both logs, because a silent fallback is indistinguishable from a
        diffusion mode that never ran."""
        self._readable(task, attempt,
                       f"no hole this attempt — {why}; the whole file is "
                       f"written instead")
        if self.journal is not None:
            try:
                self.journal.log("canvas", task=task.path, attempt=attempt,
                                 kind="skip", reason=why)
            except Exception:                            # noqa: BLE001
                pass

    def _readable(self, task: Task, attempt: int, headline: str, *,
                  details: Sequence[str] = (),
                  blocks: Sequence[tuple[str, str]] = ()) -> None:
        """One diffusion decision in BUILD_LOG.txt. Never fails a build."""
        write = getattr(getattr(self, "log", None), "diffusion", None)
        if write is None:
            return
        try:
            write(task.path, attempt, headline, details=details,
                  blocks=blocks)
        except Exception:                                # noqa: BLE001
            pass          # a log must never fail a build

    def _emit_canvas(self, task: Task, attempt: int, hole: Any,
                     chosen: Any) -> None:
        """The kept hole, with how sure the model was of each line — what a
        host draws to colour the code by confidence."""
        r = chosen.result
        lead = cv._edge_blanks(hole.original)[0].count("\n")
        self._emit("status",
                   f"{task.path}: {hole.label}"
                   f"{f' (`{hole.name}`)' if hole.name else ''} filled — "
                   f"{chosen.style} canvas, {r.steps} step(s), the model "
                   f"{chosen.mean:.0%} sure on average",
                   {"task": task.path, "attempt": attempt,
                    "canvas": {"kind": hole.kind, "region": hole.label,
                               "first_line": hole.start + 1 + lead,
                               "function": hole.name,
                               "style": chosen.style, "noise": chosen.noise,
                               "steps": r.steps, "seed": r.seed,
                               "lines": cv.doubt_html_lines(chosen),
                               # None: a line past the readout (unknown)
                               "line_confidence": [
                                   c if c == c else None
                                   for c in chosen.confidence],
                               "line_flips": list(chosen.flips)}})

    def _fit_to_context(self, messages: list[Message],
                        caps: ModelCapabilities, task: Task, attempt: int,
                        *, rebuild, pieces: list[tuple[str, str]]
                        ) -> list[Message]:
        """The prompt, made to fit the model's context — or said not to.

        NOTHING MEASURED THE ASSEMBLED PROMPT. `measure_budget` and
        `build_context` existed and only the ATK adapter called them, so a
        repair of a large file sent `[THE FILE AS IT STANDS]` whole. On a
        local server that is not an error anyone sees: llama.cpp shifts the
        context, the SYSTEM PROMPT is what falls off the front, and the
        model answers a prompt with no instructions in it. On a small model
        that is the likeliest single cause of a repair that "ignores" the
        diagnostics.

        Only the volatile tail is re-assembled — the file as it stands, the
        test, the interfaces, the examples — through `build_context`, which
        cuts the largest essential piece to fit and names what it left out
        (M28). The cached prefix is never touched (M52).
        """
        total = int(getattr(caps, "context_tokens", 0) or 0)
        if total <= 0:
            return messages
        count = self.host.llm.count_tokens
        # The reply is part of the same window; a margin covers the chat
        # template's own tokens, which no count of the text includes.
        margin = max(64, total // 50)
        limit = total - int(self.config.max_tokens) - margin
        used = sum(count(m.content) for m in messages)
        if used <= limit:
            return messages
        movable = sum(count(text) for _label, text in pieces)
        fixed = used - movable
        room = limit - fixed
        whole = [text for label, text in pieces
                 if label == "THE FILE AS IT STANDS"]
        # The file being repaired is the one piece that may NOT be cut: the
        # contract asks for the complete corrected file, and a model shown
        # its first half returns its first half — the rest of the file
        # deleted in a diff that may well still verify. Refuse instead,
        # before a model call is spent on it.
        if whole and count(whole[0]) + 128 > room:
            self._too_big = (
                f"{task.path} is about {count(whole[0]):,} tokens and the "
                f"repair prompt has room for about {max(0, room):,} (a "
                f"{total:,}-token context, less {self.config.max_tokens:,} "
                f"for the reply and {fixed:,} for the instructions and "
                f"errors). A whole-file repair cannot be done at this "
                f"size without showing the model part of the file, and a "
                f"model shown part of a file returns part of a file. Load "
                f"the model with a larger context, split the file, or fix "
                f"the reported error by hand")
            if self.journal is not None:
                self.journal.log("budget", what="context", task=task.path,
                                 attempt=attempt, fitted=False,
                                 refused=True, tokens_before=used,
                                 limit=limit)
            return messages
        if pieces and room > 256:
            tail = build_context(
                [Piece(label, text, priority=i,
                       essential=label.startswith(("THE FILE", "THE TEST",
                                                   "THE INTERFACE")))
                 for i, (label, text) in enumerate(pieces)],
                room * CHARS_PER_TOKEN, count_tokens=count)
            fitted = rebuild([tail])
            now = sum(count(m.content) for m in fitted)
            self._emit("warning",
                       f"{task.path}: the prompt was about {used:,} tokens "
                       f"against {limit:,} available (a {total:,}-token "
                       f"context, less {self.config.max_tokens:,} for the "
                       f"reply), so the file, test and interfaces were cut "
                       f"to fit and the cut is named in the prompt.",
                       {"task": task.path, "tokens_before": used,
                        "tokens_after": now, "limit": limit})
            if self.journal is not None:
                self.journal.log("budget", what="context", task=task.path,
                                 attempt=attempt, fitted=True, tokens_before=used,
                                 tokens_after=now, limit=limit)
            return fitted
        self._emit("warning",
                   f"{task.path}: the prompt is about {used:,} tokens "
                   f"against {limit:,} available, and the part that cannot "
                   f"be cut — instructions, request and diagnostics — is "
                   f"{fixed:,} of them. The server may drop the start of "
                   f"the prompt. A larger context, a smaller reply budget "
                   f"or a shorter request will help.",
                   {"task": task.path, "tokens": used, "limit": limit})
        if self.journal is not None:
            self.journal.log("budget", what="context", task=task.path,
                             attempt=attempt, fitted=False, tokens_before=used, limit=limit)
        return messages

    def _complete(self, messages: Sequence[Message], *, tools=(),
                  temperature: float = 0.15,
                  seed: int | None | object = _UNSET) -> Completion:
        return self.host.llm.complete(
            messages, tools=tools, temperature=temperature,
            max_tokens=self.config.max_tokens,
            seed=self.config.seed if seed is _UNSET else seed,
            cancel=self.cancel)

    # ------------------------------------------------------------------
    # deterministic pre-fixes (F1, M35)
    # ------------------------------------------------------------------
    def _prefix_fixes(self, code: str, lang: str,
                      task: Task) -> tuple[str, list[str]]:
        """Never ask the model what a rule can answer.

        Returns (code, what-was-done). The list is logged and journaled: if
        the same fix recurs on every file, the PROMPT needs changing, and
        this log is the only way anyone finds that out.
        """
        fixed, done = runner.autofix(code, lang, fs=self.host.fs,
                                     ex=self.host.exec,
                                     stem=_stem(task.path))
        if self.codemap is not None:
            missing = self.codemap.unresolved_in(fixed, lang)
            if missing:
                # A FINDING, not a fix — kept out of the fix list so the
                # report does not claim to have repaired something it only
                # noticed. Inserting an import for a name that exists
                # nowhere would turn a clear error into a confusing one (D4);
                # saying so plainly is the useful move.
                self._emit("warning",
                           f"{task.path} refers to names this project does "
                           f"not define: {', '.join(missing[:5])}. They will "
                           f"fail at run time if they are not real.",
                           {"task": task.path, "unresolved": missing[:10]})
                if self.journal is not None:
                    self.journal.log("codemap", task=task.path,
                                     unresolved_names=missing[:10])
        if done and self.journal is not None:
            self.journal.log("autofix", task=task.path, fixes=done)
        return fixed, done

    # ------------------------------------------------------------------
    # whose fault is it
    # ------------------------------------------------------------------
    def _culprit_elsewhere(self, task: Task,
                           diags: Sequence[Diagnostic]) -> str:
        """The OTHER project file where this task's failure is raised, or "".

        The first error diagnostic decides. If it is located in the task's
        own file, the task is at fault and retrying it is right. If it is
        located in a file the codemap indexes — a file of this project, not
        a test, not anything under site-packages or a virtualenv — that
        file is the culprit and retrying this one is waste. Anything else
        (no file, an unknown file, the interpreter itself) is treated as
        this task's problem, as before.
        """
        if self.codemap is None:
            return ""
        first = next((d for d in diags if d.is_error), None)
        if first is None or not first.file:
            return ""
        f = first.file.replace("\\", "/")
        if _same_file(first.file, task.path):
            return ""
        if "site-packages" in f or "/.venv/" in f or "/venv/" in f \
                or "/node_modules/" in f:
            return ""
        try:
            known = [row["path"] for row in self.codemap.store.files()]
        except Exception:                                # noqa: BLE001
            return ""
        for path in known:
            if path == task.path or not _same_file(first.file, path):
                continue
            if not _looks_like_test(path):
                return path
            # A TEST file is normally the authority, never the culprit —
            # except when it does not parse. Oct 2, 2026: src/physics.py
            # failed its run because tests/test_physics.py ended in a stray
            # "```"; the loop rewrote physics.py twice, got the same file
            # twice, and stopped. Nothing in physics.py could fix that.
            if first.code == "syntax" or \
                    _TEST_PARSE_FAULT.search(first.message or ""):
                return path
        return ""

    # ------------------------------------------------------------------
    # stagnation and cycles (M34)
    # ------------------------------------------------------------------
    def _stagnation(self, sig: _Signature, seen: set, history: list,
                    attempts: list, error_counts: list,
                    diags: Sequence[Diagnostic]) -> str:
        """The stop reason, in words the operator can act on, or "".

        Four detectors, because each catches something the others miss:
        identical code, identical diagnostics, ANY repeated signature (which
        catches 3- and 4-cycles), and no forward progress across three
        attempts (which catches slow oscillation that looks like work).
        """
        if history and sig.code == history[-1].code:
            return ("the model produced identical code twice — more attempts "
                    "cannot help. The task is probably too large or too "
                    "vague to fix by retrying; narrow it, or fix the last "
                    "error by hand")

        if history and _cosmetic(history[-1], sig):
            return (f"the model changed under "
                    f"{COSMETIC_THRESHOLD:.0%} of the file between attempts "
                    f"and the errors did not move — it is tinkering, not "
                    f"fixing. The error is probably somewhere it is not "
                    f"looking; fix it by hand or narrow the task")

        if sig.pair in seen:
            # EARLIER attempts only. The current one is named once, at the
            # end; it used to be in both places and the sentence read
            # "attempts 1, 3 and 3".
            where = [a.n for a in attempts[:-1]
                     if (a.code_sha, a.diag_sha) == sig.pair]
            cycle = _describe_cycle(diags)
            return (f"attempts {_and_list(where + [attempts[-1].n])} "
                    f"produced the same code and the same errors — it is "
                    f"going round in a circle{cycle}")

        if history and sig.diags == history[-1].diags:
            return ("two attempts in a row produced exactly the same errors, "
                    "so the model is not learning from the feedback. "
                    "Narrowing the task or widening the context is more "
                    "likely to help than another attempt")

        errs = sum(1 for d in diags if d.is_error)
        if len(error_counts) >= 2 and errs >= max(error_counts[-2:]):
            if len(error_counts) >= 3 and errs >= max(error_counts[-3:]):
                return (f"the error count has not fallen in three attempts "
                        f"(still {errs}) — slow oscillation, not progress")
        return ""

    def _give_up_sentence(self, attempts: Sequence[AttemptRecord]) -> str:
        """What it TRIED and the last real error — never "failed 4 times"."""
        if not attempts:
            return "nothing was generated"
        tried = []
        for a in attempts:
            bits = [f"attempt {a.n}"]
            if a.continued:
                bits.append("continued after truncation")
            if a.autofixes:
                bits.append(f"{len(a.autofixes)} auto-fix"
                            f"{'es' * (len(a.autofixes) != 1)}")
            if a.diagnostics:
                bits.append(f"{sum(1 for d in a.diagnostics if d.is_error)} "
                            f"error(s)")
            tried.append(" — ".join(bits))
        last = attempts[-1]
        detail = ""
        if last.diagnostics:
            detail = (f" The last real error was: "
                      f"{last.diagnostics[0].one_line()}")
        return (f"gave up after {len(attempts)} attempt"
                f"{'s' * (len(attempts) != 1)} ({'; '.join(tried)}).{detail}")

    def _remember(self, attempts: Sequence[AttemptRecord],
                  diags: Sequence[Diagnostic]) -> None:
        """Record the fix that worked, per project (F10).

        Only recorded when a repair actually succeeded — a first-attempt pass
        teaches nothing, and filling the table with non-fixes would make the
        recall useless.
        """
        if self.codemap is None or len(attempts) < 2:
            return
        previous = attempts[-2]
        if not previous.diagnostics:
            return
        signature = "\n".join(dx.signature(previous.diagnostics))
        shape = "; ".join(previous.autofixes) or "model repair"
        try:
            self.codemap.store.remember_fix(_sha(signature), shape)
        except Exception:                                # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # verification and writing
    # ------------------------------------------------------------------
    def _verify(self, task: Task, lang: str) -> RunResult:
        test_source = ""
        if task.test_path:
            test_source = self._read(task.test_path)
        spec_test = self.spec_tests.get(task.path, "")
        result = runner.verify(
            self._read(task.path) or "", lang, fs=self.host.fs,
            ex=self.host.exec, stem=_stem(task.path), path=task.path,
            project_mode=self.config.project_mode, test_source=test_source,
            timeouts=self.config.timeouts, test_path=task.test_path,
            known_failing=self.config.known_failing,
            spec_test_path=spec_test,
            skip_guard=True)      # already screened above; don't pay twice
        if spec_test:
            # True: the vectors ran and passed; False: one failed; None:
            # they never ran (the build failed before them)
            name = spec_test.replace("\\", "/").rsplit("/", 1)[-1]
            ran = any(p.name == "test" and any(name in str(a) for a in p.argv)
                      for p in result.phases)
            failed = any(d.tool == "spec-vector" for d in result.diagnostics)
            self.vectors_ok[task.path] = (not failed) if ran else None
        return result

    def _write(self, tx: Any, task: Task, code: str) -> bool:
        if tx is None:
            self.host.fs.write(task.path, code)
            return True
        existing = self._read(task.path)
        edit = Edit(path=task.path, kind="whole", new=code,
                    note=task.purpose)
        results = tx.apply([edit],
                           summary=f"{task.path} — {task.purpose}")
        if results and results[0].ok:
            return True
        if results and "no change" in (results[0].reason or ""):
            return bool(existing)
        self._emit("warning",
                   f"{task.path} was not written: {results[0].reason}"
                   if results else f"{task.path} was not written",
                   {"task": task.path})
        return False

    def _read(self, path: str) -> str:
        try:
            return self.host.fs.read(path)
        except Exception:                                # noqa: BLE001
            return ""

    def _capabilities(self) -> ModelCapabilities | None:
        try:
            return self.host.llm.capabilities()
        except Exception:                                # noqa: BLE001
            return None

    def _log_line(self, text: str) -> None:
        """One line in BUILD_LOG.txt. Never fails a build."""
        log = getattr(self, "log", None)
        if log is None:
            return
        try:
            log.line(text)
        except Exception:                                # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def _check_cancel(self, where: str) -> None:
        if self.cancel is not None and self.cancel.is_set():
            if self.journal is not None:
                self.journal.log("cancel", where=where)
            raise Cancelled(where)

    def _log_verify(self, task: Task, n: int, result: Any) -> None:
        """Every phase, its command and its output — verbatim.

        The one thing the JSONL never kept. It recorded {"test": "ok"} and
        threw away the two lines that mattered: "Ran 0 tests" and "OK".
        """
        log = getattr(self, "log", None)
        if log is None:
            return
        try:
            log.phases(task.path, n, result)
        except Exception:                                # noqa: BLE001
            pass          # a log must never fail a build

    def _journal_attempt(self, task: Task, n: int, completion: Completion,
                         sent: Sequence[Message], verify: dict) -> None:
        """One `generate` event, hashing the messages actually SENT (C8).

        It hashed `request or task.purpose`, so every attempt of every task
        in a session carried the same `prompt_sha256`. A hash that cannot
        tell a first attempt from a repair is not provenance.
        """
        if self.journal is None:
            return
        try:
            caps = self.host.llm.capabilities()
            remote = bool(caps.is_remote)
        except Exception:                                # noqa: BLE001
            remote = False
        #: The readable log gets the same facts plus the thing the JSONL
        #: deliberately does not keep: the code itself. The snapshots hold
        #: every version, but reading a build means following the attempts
        #: in order, and that is a different job from restoring one.
        canvas = self._via_canvas.get(task.path) == n
        used = self._sampling.get(task.path)
        temperature = (used.temperature if used is not None
                       else self.config.temperature)
        seed = used.seed if used is not None else self.config.seed
        log = getattr(self, "log", None)
        if log is not None:
            try:
                log.generation(
                    task.path, n,
                    temperature=temperature,
                    seed=seed,
                    knobs=(used.describe() if used is not None else ""),
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                    prompt_ms=completion.prompt_ms,
                    decode_ms=getattr(completion, "decode_ms", 0),
                    text=completion.text,
                    via=("diffusion canvases (each one is logged above)"
                         if canvas else ""))
            except Exception:                            # noqa: BLE001
                pass
        self.journal.generation(
            task=task.path, attempt=n,
            provider=getattr(self.host.llm, "name", "host"),
            completion=completion, prompt=list(sent) or task.purpose,
            temperature=temperature, seed=seed,
            verify=verify, remote=remote,
            **({"rung": used.rung, "fresh": used.fresh,
                "escalation": used.reason} if used is not None else {}),
            **({"via": "canvas"} if canvas else {}))

    def _emit(self, kind: str, message: str, data: dict | None = None) -> None:
        self.host.emit(kind, message, data)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _comment_for(lang_id: str) -> str:
    lang = langs.get(lang_id)
    return lang.comment if lang else "#"


def _stem(path: str) -> str:
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


_OPENERS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {v: k for k, v in _OPENERS.items()}
_FENCE_LINE = re.compile(r"^[ \t]*```[\w+#.-]*[ \t]*$")


def _is_truncated(completion: Completion, text: str, lang: str = "") -> bool:
    """`finish_reason == "length"` is the SIGNAL; delimiters are the backstop.

    D1 is explicit that truncation is detected structurally rather than
    inferred. The delimiter check exists for providers that report "stop"
    when they mean "length", which several do.

    THE BACKSTOP HAS TO BE RIGHT, because being wrong is not free. A false
    positive on a complete file costs up to MAX_CONTINUATIONS model calls
    and then appends the model's "The file is complete as written." to a
    correct file, which then fails to parse. The first scanner did that on
    `['\\\\', 'x']` (it read the backslash as escaping the closing quote),
    on `os.path.join("C:\\\\", "x")` for the same reason, and on
    `# note: dict[` because it counted brackets inside comments. So:

      * for Python, the tokenizer decides. `TokenError` means "EOF in
        multi-line statement/string" — that IS truncation — and a clean
        tokenize means the file is whole whatever a bracket count says;
      * for everything else, a backslash escapes the following character
        (including another backslash) and comments are skipped using the
        language's own markers.
    """
    if completion.finish_reason == "length":
        return True
    if not text.strip():
        return False
    body = _unfenced(text)
    if lang == "python":
        verdict = _python_truncated(body)
        if verdict is not None:
            return verdict
    lang_obj = langs.get(lang) if lang else None
    line_comment = lang_obj.comment if lang_obj else "#"
    block = tuple(lang_obj.block_comment) if (
        lang_obj and lang_obj.block_comment) else ()
    return _unbalanced(body, line_comment, block)


def _unfenced(text: str) -> str:
    """The reply without an opening fence line or a closing one.

    The check runs on the RAW reply, and a fenced file that is complete
    must read as complete; the closing fence is what a truncated one lacks.
    """
    lines = text.split("\n")
    if lines and _FENCE_LINE.match(lines[0]):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _FENCE_LINE.match(lines[-1]):
        lines = lines[:-1]
    return "\n".join(lines)


def _python_truncated(code: str) -> bool | None:
    """The tokenizer's verdict, or None when it cannot give one.

    `IndentationError` and the other `SyntaxError`s are real errors in a
    file that may well be complete — the loop's verify step will report
    them properly — so they fall through to the delimiter scan rather than
    being read as truncation.
    """
    import io
    import tokenize
    try:
        for _ in tokenize.generate_tokens(io.StringIO(code).readline):
            pass
    except tokenize.TokenError:
        return True
    except (SyntaxError, ValueError):
        return None
    return False


def _unbalanced(text: str, line_comment: str, block: tuple) -> bool:
    depth = dict.fromkeys(_OPENERS, 0)
    in_str: str | None = None
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2              # the escaped character, whatever it is
                continue
            if ch == in_str:
                in_str = None
            elif ch == "\n" and in_str != "`":
                in_str = None       # an unterminated single-line string
            i += 1
            continue
        if line_comment and text.startswith(line_comment, i):
            end = text.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        if block and text.startswith(block[0], i):
            end = text.find(block[1], i + len(block[0]))
            i = n if end == -1 else end + len(block[1])
            continue
        if ch in "\"'`":
            in_str = ch
        elif ch in _OPENERS:
            depth[ch] += 1
        elif ch in _CLOSERS:
            depth[_CLOSERS[ch]] -= 1
        i += 1
    return any(v > 0 for v in depth.values())


def _join_continuation(head: str, tail: str) -> str:
    """Join without duplicating the overlap the model repeated anyway.

    Models told "do not repeat anything" repeat the last line about a third
    of the time. Detecting the overlap is cheap; a duplicated line in the
    middle of a file is a syntax error that looks like a model failure.

    Three seams the whole-line overlap check could not see, each observed:

      * the cut was MID-LINE and the model re-emitted the full last line,
        so the join read `return tuple(    return tuple(...)`. If the
        head's partial last line is a prefix of the continuation's first
        line, the partial line goes;
      * the model re-opened a code fence at the top of the continuation
        (and sometimes closed one at the end of the head), which put a
        fence mid-file and made `_extract` pick the first fence body — the
        truncated head — as the whole file;
      * the model ignored "do not start again" and re-emitted the file
        from line one. Appending that duplicates every definition; the
        longer copy is the file.
    """
    head_body, opened, closed = _strip_head_fence(head)
    tail_body = _strip_tail_fence(tail)
    if closed:
        # The head ended on a closing fence, so its last code line was a
        # WHOLE line: the cut was at a line boundary, not mid-token, and
        # the "no joiner" rule below does not apply.
        head_body = head_body.rstrip("\n") + "\n"
    joined = _join_bodies(head_body, tail_body)
    if opened and not _FENCE_LINE.match(joined.rstrip("\n").rsplit(
            "\n", 1)[-1]):
        joined = joined.rstrip("\n") + "\n```"
    return (opened + "\n" + joined) if opened else joined


def _strip_head_fence(head: str) -> tuple[str, str, bool]:
    """(body, opening fence line or "", whether a closing fence was cut).

    A trailing fence is dropped: a continuation follows it, so it was the
    model closing a block it had not finished.
    """
    lines = head.split("\n")
    opened = ""
    closed = False
    if lines and _FENCE_LINE.match(lines[0]):
        opened = lines[0]
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and _FENCE_LINE.match(lines[-1]):
        lines = lines[:-1]
        closed = True
    return "\n".join(lines), opened, closed


def _strip_tail_fence(tail: str) -> str:
    lines = tail.lstrip("\n").split("\n")
    if lines and _FENCE_LINE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines)


def _join_bodies(head: str, tail: str) -> str:
    ends_whole = head.endswith("\n")
    head_lines = head.rstrip("\n").split("\n")
    tail_lines = tail.lstrip("\n").split("\n")
    # Re-emitted from the top: the continuation starts with the head's
    # complete lines. The head's LAST line is excluded from the comparison
    # because it is usually the partial one the cut landed in.
    whole = head_lines if ends_whole else head_lines[:-1]
    lead = [ln for ln in whole if ln.strip()][:3]
    if (len(lead) >= 2 and len(tail_lines) >= len(whole)
            and [ln for ln in tail_lines if ln.strip()][:len(lead)] == lead):
        return tail if len(tail) >= len(head) else head
    for overlap in range(min(8, len(head_lines), len(tail_lines)), 0, -1):
        if head_lines[-overlap:] == tail_lines[:overlap]:
            return "\n".join(head_lines + tail_lines[overlap:])
    last = head_lines[-1]
    if (last.strip() and tail_lines and tail_lines[0] != last
            and tail_lines[0].startswith(last)):
        return "\n".join(head_lines[:-1] + tail_lines)
    # No joiner, ever. `finish_reason == "length"` means the model was cut
    # off mid-TOKEN — `    parts = ` — and the continuation resumes at the
    # very next character. Inserting a newline here produces
    # `parts =\nline.split(...)`, which is a syntax error the model will
    # then be blamed for. The leading newlines a model tends to add are
    # stripped for the same reason.
    return head + tail.lstrip("\n")


def _signature(code: str, lang: str,
               diags: Sequence[Diagnostic]) -> _Signature:
    """One attempt's stagnation identity (M34)."""
    digest, text = _code_identity(code, lang)
    return _Signature(code=digest, diags=_diag_identity(diags), text=text)


def _code_identity(code: str, lang: str) -> tuple[str, str]:
    """(hash, normalised text) of one attempt's code.

    For Python the normal form is the AST, unparsed. `textio.canonical`
    collapses LEADING whitespace along with the rest, and in Python leading
    whitespace is semantics: a `return` inside a loop and the same `return`
    after it hashed identically, and a real change was stopped as
    "identical code twice" one attempt before the fix that would have
    passed. The AST ignores what canonical() was for — spacing, blank
    lines, comments — and keeps what it lost.

    A file that does not parse falls back to canonical(), which is exactly
    when stagnation is most likely and a parse is least available. The two
    forms are prefixed so they can never collide with each other.
    """
    if lang == "python":
        try:
            import ast
            text = "ast:" + ast.unparse(ast.parse(code))
            return _sha(text), text
        except (SyntaxError, ValueError, RecursionError):
            pass
    text = "txt:" + textio.canonical(code, _comment_for(lang))
    return _sha(text), text


def _diag_identity(diags: Sequence[Diagnostic]) -> str:
    """Sorted (file, message, offending source) — deliberately NOT the line.

    `diagnostics.signature` keys on the line number, which is right for
    regression memory and wrong here: a model that adds one line above the
    error each attempt moves the SAME NameError one line down every time,
    and with the line in the key that read as progress until the slow-
    oscillation check caught it at attempt 4 instead of attempt 2.
    """
    rows = sorted(f"{d.file}|{(d.message or '')[:80]}|"
                  f"{_offending(d.source_excerpt)}" for d in diags)
    return _sha("\n".join(rows))


def _offending(excerpt: str) -> str:
    """The quoted offending line(s), without the numbers the quote carries.

    `attach_source` marks the offending line `>>  NNNN | text` among its
    context lines. The number and the context both drift when lines are
    added above; the offending text does not.
    """
    return " / ".join(line.split("|", 1)[-1].strip()
                      for line in (excerpt or "").splitlines()
                      if line.startswith(">>"))


def _cosmetic(previous: _Signature, current: _Signature) -> bool:
    """Churn: under COSMETIC_THRESHOLD of the file changed, errors unmoved.

    The first version compared the same two hashes the identical-code check
    had just compared, so it could never be True and the threshold was
    never read. This measures the difference instead — at LINE granularity,
    weighted by characters, because a character-level SequenceMatcher on a
    large file is quadratic and a stagnation check must never be the slow
    part of an attempt. A changed line counts as wholly changed, which errs
    towards "not cosmetic": the safe direction for a check that stops work.
    """
    if previous.diags != current.diags or not previous.text \
            or not current.text:
        return False
    a = previous.text.split("\n")
    b = current.text.split("\n")
    total = sum(map(len, a)) + sum(map(len, b))
    if not total:
        return False
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    same = sum(sum(len(line) for line in a[m.a:m.a + m.size])
               for m in matcher.get_matching_blocks())
    return 2 * same / total >= 1.0 - COSMETIC_THRESHOLD


def _and_list(numbers: Sequence[int]) -> str:
    """1 → "1"; 1, 3 → "1 and 3"; 1, 3, 5 → "1, 3 and 5"."""
    words = [str(n) for n in numbers]
    if len(words) <= 1:
        return "".join(words)
    return f"{', '.join(words[:-1])} and {words[-1]}"


def _describe_cycle(diags: Sequence[Diagnostic]) -> str:
    """Name the two things it is alternating between, if it is two things.

    This is the sentence that turns a wasted twenty minutes into a
    two-second fix by hand, so it is worth the effort of writing it.
    """
    if not diags:
        return ""
    kinds = []
    for d in diags[:2]:
        msg = (d.message or "").lower()
        if "unused" in msg and "import" in msg:
            kinds.append("an unused import")
        elif "not defined" in msg or "undeclared" in msg or "cannot find" in msg:
            kinds.append("a missing definition")
        elif "import" in msg:
            kinds.append("an import problem")
        elif "indent" in msg:
            kinds.append("indentation")
        else:
            kinds.append(f"“{(d.message or '')[:50]}”")
    if len(kinds) >= 2:
        return f". It is alternating between {kinds[0]} and {kinds[1]}"
    return f". The error it keeps producing is {kinds[0]}"


#: A test file's OWN mistakes: it does not parse, or it names something that
#: does not exist. Those are the test's to fix, and fixing them does not
#: change what the test claims about the code.
_TEST_OWN_FAULT = re.compile(
    r"SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError|"
    r"cannot import name|NameError|is not defined", re.I)


#: A test file that cannot even be PARSED. Narrower than _TEST_OWN_FAULT on
#: purpose: "cannot import name X from src.physics", raised in a test, can be
#: fixed by giving src/physics.py an X — but no change to any module can fix
#: a syntax error in the test file.
_TEST_PARSE_FAULT = re.compile(r"SyntaxError|IndentationError|TabError")


def _same_file(diag_file: str, path: str) -> bool:
    f = (diag_file or "").replace("\\", "/")
    p = path.replace("\\", "/")
    return f == p or f.endswith("/" + p) or (
        "/" not in p and f.rsplit("/", 1)[-1] == p)


def _test_disagrees_with_code(test_path: str,
                              diags: Sequence[Diagnostic]) -> bool:
    """Does this failing test run say the CODE is wrong, not the test?

    WHY THIS EXISTS. When a correct test failed against a wrong module, the
    repairer was pointed at the test — and bent it. `add` returned `a - b`,
    the repaired test asserted `add(1, 1) == 0`, and both were committed
    green: the exact thing the tester persona says a bad test does, which is
    "manufacture confidence".

    An assertion failure is located IN the test file, so "the diagnostic
    points somewhere else" cannot be the rule. It is:

      * the test's own breakage (it does not parse, or it imports or names
        something that does not exist) — the test's fault, repair the test;
      * otherwise, an assertion failure, or an error raised in another file
        — the test and the code disagree about BEHAVIOUR, and a disagreement
        is never settled by editing the test until it agrees.
    """
    errors = [d for d in diags if d.is_error]
    for d in errors:
        # `code == "syntax"`: the in-process parse check's message is the
        # bare reason ("invalid syntax", "expected ':'"), with no
        # "SyntaxError" in it for the pattern to find.
        if d.file and _same_file(d.file, test_path) and (
                d.code == "syntax"
                or _TEST_OWN_FAULT.search(d.message or "")):
            return False
    for d in errors:
        message = (d.message or "").lower()
        if "assert" in message or (d.severity or "").lower() == "failure":
            return True
        if d.file and not _same_file(d.file, test_path) and \
                "site-packages" not in d.file:
            return True
    return False


def _elsewhere_sentence(path: str, culprit: str,
                        diags: Sequence[Diagnostic]) -> str:
    first = next((d for d in diags if d.is_error and d.file
                  and _same_file(d.file, culprit)), None)
    where = f"{culprit}:{first.line}" if first and first.line else culprit
    detail = f" — {first.message}" if first and first.message else ""
    return (f"the error is in {where}{detail}, not in {path}; rewriting "
            f"{path} cannot fix it. {culprit} is repaired against this "
            f"failure next, then {path} is checked again")


def _pinned_piece(path: str, pinned: str) -> tuple[str, str]:
    """(label, text) of the pinned-interface block of a prompt."""
    rule = ("This is the interface the skeleton pinned for this file before "
            "any file was written. Other files use it exactly as it stands. "
            "Keep every name, signature, field, type alias and constant "
            "shown here, and replace each `raise NotImplementedError` with "
            "the real body. You may add private helpers.")
    if "__main__" in pinned:
        rule += (" Keep the `if __name__ == \"__main__\":` block: it is how "
                 "the program starts.")
    return (f"THE INTERFACE PINNED FOR THIS FILE — {path}",
            f"{pinned}\n\n{rule}")


def _unchanged_sentence(path: str, blamer: str, *, by_caller: bool) -> str:
    """Why a repair that returned the file unchanged is NOT a repair."""
    whose = (f"{blamer}'s failure" if blamer else "the failure it was "
             "repaired against")
    who = "caller" if by_caller else "test"
    return (f"the model returned the same code — {path} is unchanged "
            f"(identical after normalising), so {whose} cannot have gone "
            f"away. It was told to return the file unchanged only if it is "
            f"certain the {who} itself is wrong: either it is, or the fix "
            f"is one the model cannot see. Read the error and fix "
            f"whichever is wrong by hand")


def _disagreement_sentence(test_path: str, covers: str,
                           diags: Sequence[Diagnostic]) -> str:
    target = f"`{covers}`" if covers else "the code it tests"
    first = next((d for d in diags if d.is_error and d.file), None) or \
        next((d for d in diags if d.is_error), None)
    detail = f" ({first.one_line()})" if first else ""
    return (f"the tests in {test_path} fail against {target}{detail}. The "
            f"test was not rewritten to agree with the code — that would "
            f"make a wrong {covers or 'module'} look verified. It is "
            f"{covers or 'the code under test'} that has to change")


def _first_task_text(task: Task, request: str, lang: str) -> str:
    lang_obj = langs.get(lang)
    label = lang_obj.label if lang_obj else lang
    lines = [f"Write the complete contents of `{task.path}`.", ""]
    if request:
        lines += [f"It is part of this request: {request}", ""]
    lines += [f"Purpose of this file: {task.purpose}",
              f"Language: {label}"]
    if task.test_path:
        lines.append(f"Its tests live in `{task.test_path}` and must pass.")
    if lang_obj and lang_obj.notes:
        lines.append(f"Note for this language: {lang_obj.notes}")
    note = _test_note_for(task, lang_obj)
    if note:
        lines.append(note)
    return "\n".join(lines)


def _test_note_for(task: Task, lang_obj: Any) -> str:
    """The runner's rules, for a TEST file's author only."""
    if lang_obj is None or not lang_obj.test_note:
        return ""
    if task.persona == "tester" or _looks_like_test(task.path):
        return f"How the tests are run: {lang_obj.test_note}"
    return ""


def _change_task_text(task: Task, request: str, lang: str) -> str:
    """The first attempt on a file that already has real work in it."""
    lang_obj = langs.get(lang)
    label = lang_obj.label if lang_obj else lang
    lines = [f"`{task.path}` already exists; its current contents are "
             f"below. Change it as this request asks, and keep everything "
             f"the request does not mention exactly as it is.", ""]
    if request:
        lines += [f"The request: {request}", ""]
    lines += [f"What this file is for: {task.purpose}",
              f"Language: {label}",
              "Return the complete updated file."]
    if task.test_path:
        lines.append(f"Its tests live in `{task.test_path}` and must pass.")
    if lang_obj and lang_obj.notes:
        lines.append(f"Note for this language: {lang_obj.notes}")
    note = _test_note_for(task, lang_obj)
    if note:
        lines.append(note)
    return "\n".join(lines)


_FENCE = re.compile(r"```[\w+#.-]*\n(.*?)```", re.S)


def _extract(text: str, lang_id: str, path: str = "") -> str:
    """Model reply → the file's contents (D5).

    Fence confusion is real: three backticks inside a docstring, a language
    tag that isn't a language, no fence at all, two fences with different
    content. Prefer a fence tagged for this language, then the longest fence,
    then the whole reply — and VALIDATE by parsing where we can, trying the
    next candidate before giving up. Never assume the first fence.
    """
    from . import patcher

    def validates(candidate: str) -> bool:
        if lang_id != "python":
            return bool(candidate.strip())
        try:
            import ast
            ast.parse(candidate)
            return True
        except SyntaxError:
            return False

    return patcher.extract_code(text, lang_id, validator=validates,
                                path=path)


def _verify_dict(result: RunResult) -> dict:
    """The `verify` block of a journal event (C8)."""
    out: dict[str, Any] = {"ok": result.ok}
    for phase in result.phases:
        out[phase.name] = "ok" if phase.ok else "failed"
    if result.diagnostics:
        out["diagnostics"] = len(result.diagnostics)
    if result.caveats:
        out["caveats"] = list(result.caveats)
    return out
