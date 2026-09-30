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
from dataclasses import dataclass, field
import difflib
import hashlib
import re
import time
from typing import Any

from . import diagnostics as dx
from . import guard, langs, personas, runner, textio
from .context import CHARS_PER_TOKEN, Piece, build_context
from .errors import Cancelled
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

    # ------------------------------------------------------------------
    def run_task(self, task: Task, *, request: str = "", covers: str = "",
                 seed: Sequence[Diagnostic] = ()) -> TaskOutcome:
        """Generate, verify and repair one file until it works or it stops.

        Every phase boundary checks the cancel token (M21). A cancelled task
        leaves resumable state and rolls back any open transaction — that is
        a guarantee made here, not a hope held elsewhere.

        ``covers`` is, for a test task, the module it tests. ``seed`` makes
        attempt 1 a REPAIR against diagnostics that already exist — the
        failing output of a test this file must now pass (see
        `_test_disagrees_with_code`).
        """
        lang = task.lang or langs.id_for_path(task.path) or "python"
        persona = PERSONAS.get(task.persona, personas.ENGINEER)
        is_test = task.persona == "tester" or _looks_like_test(task.path)
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
                    repair=repair, against_test=bool(seed))

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

                # -- a test that disagrees with the code is not bent (C4) --
                if is_test and _test_disagrees_with_code(task.path, diags):
                    stopped = _disagreement_sentence(task.path, covers,
                                                     diags)
                    self.blamed[task.path] = (covers, diags)
                    self._emit("warning", f"{task.path}: {stopped}",
                               {"task": task.path, "covers": covers})
                    break

                # -- stagnation and cycles (M34) -------------------------
                stopped = self._stagnation(sig, seen, history, attempts,
                                           error_counts, diags)
                seen.add(sig.pair)
                history.append(sig)
                error_counts.append(sum(1 for d in diags if d.is_error))
                if stopped:
                    break

                if (self.config.wall_clock_s
                        and time.monotonic() - started
                        > self.config.wall_clock_s):
                    stopped = (f"the time budget for this file "
                               f"({self.config.wall_clock_s:.0f}s) ran out")
                    break

            ok = bool(result and result.ok)
            unverified = self._ran_no_tests(task, is_test, result) if ok \
                else ""
            if unverified:
                self.unverified.add(task.path)
                self._emit("warning", f"{task.path}: {unverified}",
                           {"task": task.path, "verified": False})
            if tx is not None:
                if ok:
                    # committed AND verified: SEALED — unless the tests that
                    # were meant to verify it never ran.
                    tx.commit(verified=not unverified)
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

        if not stopped and not (result and result.ok):
            stopped = self._give_up_sentence(attempts)

        caveats = tuple(result.caveats) if result else ()
        if unverified:
            caveats += (unverified,)
        outcome = TaskOutcome(
            task_id=task.id, path=task.path,
            ok=bool(result and result.ok), attempts=tuple(attempts),
            result=result, stopped_because=stopped, caveats=caveats)
        self._emit("status", outcome.summary(),
                   {"task": task.path, "ok": outcome.ok})
        return outcome

    def reverify(self, task: Task, *, because: str = "") -> TaskOutcome:
        """Verify a file again WITHOUT generating anything.

        Used after the module a test covers has been repaired against that
        test: the test's own verdict is the evidence, and asking a model to
        regenerate a test that may now pass would be paying to risk it.
        """
        lang = task.lang or langs.id_for_path(task.path) or "python"
        is_test = task.persona == "tester" or _looks_like_test(task.path)
        self._check_cancel(f"verifying {task.path} again")
        self._emit("phase", f"{task.path}: verifying again",
                   {"phase": "verify", "task": task.path, "attempt": 0})
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
            stopped = (f"it still fails after {because or 'the repair'}"
                       + (f": {first.one_line()}" if first else ""))
        outcome = TaskOutcome(task_id=task.id, path=task.path, ok=result.ok,
                              result=result, stopped_because=stopped,
                              caveats=caveats)
        self._emit("status", outcome.summary(),
                   {"task": task.path, "ok": outcome.ok})
        return outcome

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
                  against_test: bool = False
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
        arch = ""
        tail_extra: list[str] = []
        #: The same tail blocks as (label, text), so they can be put back
        #: together under a budget when the whole prompt would not fit.
        tail_pieces: list[tuple[str, str]] = []
        interfaces = examples = staleness = ""
        if self.codemap is not None:
            arch = self.codemap.prefix_block(task.path)
            blocks = self.codemap.tail_blocks(
                task.path, count_tokens=self.host.llm.count_tokens)
            for b in blocks:
                if b.startswith("# INTERFACES"):
                    interfaces = b
                elif b.startswith("# HOW THIS CODEBASE"):
                    examples = b
                else:
                    staleness = b

        current = "" if repair else self._read(task.path)
        if current and _is_our_stub(current):
            current = ""
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
        else:
            # Diagnostics go in the TAIL, not here — tail_for puts them
            # immediately before the output contract, where recency helps
            # most (D7). Repeating them in the task body would spend
            # context restating the same errors twice.
            body = personas.repair_task(task.path, task.purpose,
                                        autofixes=autofixes,
                                        request=request)
            persona = PERSONAS["repairer"]
            existing = self._read(task.path)
            if existing:
                tail_extra.append(f"[THE FILE AS IT STANDS]\n{existing}")
                tail_pieces.append(("THE FILE AS IT STANDS", existing))
            test_source = self._read(task.test_path) if (
                against_test and task.test_path) else ""
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
                                    temperature=persona.temperature)

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
                                        temperature=persona.temperature)

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
                                            temperature=persona.temperature)
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
        code = strip_stub_sentinel(_extract(text, lang))
        # The prompt that produced the answer: after tool round-trips and
        # the text-lookup fallback, before any continuation (a continuation
        # is derived from this prompt, not a different one).
        return code, completion, continued, list(messages)

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
                       essential=label.startswith(("THE FILE", "THE TEST")))
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
                  temperature: float = 0.15) -> Completion:
        return self.host.llm.complete(
            messages, tools=tools, temperature=temperature,
            max_tokens=self.config.max_tokens, seed=self.config.seed,
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
        return runner.verify(
            self._read(task.path) or "", lang, fs=self.host.fs,
            ex=self.host.exec, stem=_stem(task.path), path=task.path,
            project_mode=self.config.project_mode, test_source=test_source,
            timeouts=self.config.timeouts, test_path=task.test_path,
            skip_guard=True)      # already screened above; don't pay twice

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
        log = getattr(self, "log", None)
        if log is not None:
            try:
                log.generation(
                    task.path, n,
                    temperature=self.config.temperature,
                    seed=self.config.seed,
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                    prompt_ms=completion.prompt_ms,
                    decode_ms=getattr(completion, "decode_ms", 0),
                    text=completion.text)
            except Exception:                            # noqa: BLE001
                pass
        self.journal.generation(
            task=task.path, attempt=n,
            provider=getattr(self.host.llm, "name", "host"),
            completion=completion, prompt=list(sent) or task.purpose,
            temperature=self.config.temperature, seed=self.config.seed,
            verify=verify, remote=remote)

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
        if d.file and _same_file(d.file, test_path) and \
                _TEST_OWN_FAULT.search(d.message or ""):
            return False
    for d in errors:
        message = (d.message or "").lower()
        if "assert" in message or (d.severity or "").lower() == "failure":
            return True
        if d.file and not _same_file(d.file, test_path) and \
                "site-packages" not in d.file:
            return True
    return False


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
    return "\n".join(lines)


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
    return "\n".join(lines)


_FENCE = re.compile(r"```[\w+#.-]*\n(.*?)```", re.S)


def _extract(text: str, lang_id: str) -> str:
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

    return patcher.extract_code(text, lang_id, validator=validates)


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
