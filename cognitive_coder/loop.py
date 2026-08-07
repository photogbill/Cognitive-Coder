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
import hashlib
import re
import time
from typing import Any

from . import diagnostics as dx
from . import guard, langs, personas, runner, textio
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
from .ports import NeverCancelled
from .types import (
    AttemptRecord,
    Completion,
    Diagnostic,
    Edit,
    Message,
    RunResult,
    Task,
    TaskOutcome,
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


@dataclass
class _Signature:
    """One attempt's identity: normalised code AND sorted diagnostics."""
    code: str
    diags: str

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

    # ------------------------------------------------------------------
    def run_task(self, task: Task, *, request: str = "") -> TaskOutcome:
        """Generate, verify and repair one file until it works or it stops.

        Every phase boundary checks the cancel token (M21). A cancelled task
        leaves resumable state and rolls back any open transaction — that is
        a guarantee made here, not a hope held elsewhere.
        """
        lang = task.lang or langs.id_for_path(task.path) or "python"
        persona = PERSONAS.get(task.persona, personas.ENGINEER)
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
                if attempts and attempts[-1].diagnostics:
                    # M33: the DIAGNOSTICS carry forward, not the broken code.
                    lang_obj = langs.get(lang)
                    diag_text = dx.feedback(
                        attempts[-1].diagnostics,
                        lang_obj.feedback_cap if lang_obj else 3,
                        extra_context=bool(lang_obj and lang_obj.cascades))
                    autofixed = attempts[-1].autofixes

                code, completion, continued = self._generate(
                    task, persona, lang, request=request,
                    diagnostics=diag_text, autofixes=autofixed, attempt=n)

                if completion.finish_reason == "cancelled":
                    raise Cancelled(f"generating {task.path}")

                if not code.strip():
                    attempts.append(AttemptRecord(
                        n=n, finish_reason=completion.finish_reason,
                        note="the model returned nothing"))
                    if completion.finish_reason == "error":
                        stopped = ("the model could not be reached, or it "
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
                    attempts.append(AttemptRecord(
                        n=n, code_sha=_sha(textio.canonical(code)),
                        diagnostics=diags, autofixes=tuple(fixes),
                        finish_reason=completion.finish_reason,
                        continued=continued, note=f"refused: {blocked}"))
                    self._journal_attempt(task, n, completion, request,
                                          {"guard": "blocked"})
                    continue

                # -- write, then verify (C4, M4) -------------------------
                self._check_cancel(f"writing {task.path}")
                written = self._write(tx, task, code)
                if not written:
                    stopped = ("the change was not approved, so nothing was "
                               "written")
                    break

                self._check_cancel(f"verifying {task.path}")
                self._emit("phase", f"{task.path}: verifying",
                           {"phase": "verify", "task": task.path,
                            "attempt": n})
                result = self._verify(task, lang)
                diags = tuple(result.diagnostics)

                sig = _Signature(code=_sha(textio.canonical(
                    code, _comment_for(lang))), diags=_sha("\n".join(
                        dx.signature(diags))))
                record = AttemptRecord(
                    n=n, code_sha=sig.code, diag_sha=sig.diags,
                    diagnostics=diags, autofixes=tuple(fixes),
                    finish_reason=completion.finish_reason,
                    continued=continued,
                    note=result.summary())
                attempts.append(record)
                self._journal_attempt(task, n, completion, request,
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
                    if self.codemap is not None:
                        self.codemap.reindex_after_write(task.path)
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
            if tx is not None:
                if ok:
                    tx.commit(verified=True)     # committed AND verified: SEALED
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

        outcome = TaskOutcome(
            task_id=task.id, path=task.path,
            ok=bool(result and result.ok), attempts=tuple(attempts),
            result=result, stopped_because=stopped,
            caveats=tuple(result.caveats) if result else ())
        self._emit("status", outcome.summary(),
                   {"task": task.path, "ok": outcome.ok})
        return outcome

    # ------------------------------------------------------------------
    # generation, with tool round-trips and continuation
    # ------------------------------------------------------------------
    def _generate(self, task: Task, persona: Persona, lang: str, *,
                  request: str, diagnostics: str, autofixes: Sequence[str],
                  attempt: int) -> tuple[str, Completion, bool]:
        caps = self.host.llm.capabilities()
        arch = ""
        tail_extra: list[str] = []
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

        if attempt == 1:
            body = _first_task_text(task, request, lang)
        else:
            # Diagnostics go in the TAIL, not here — tail_for puts them
            # immediately before the output contract, where recency helps
            # most (D7). Repeating them in the task body would spend
            # context restating the same errors twice.
            body = personas.repair_task(task.path, task.purpose,
                                        autofixes=autofixes)
            persona = PERSONAS["repairer"]
            existing = self._read(task.path)
            if existing:
                tail_extra.append(f"[THE FILE AS IT STANDS]\n{existing}")

        prompt = self.prompts.build(
            persona, body, architecture=arch,
            epoch=(self.codemap.store.epoch if self.codemap else 0),
            interfaces=interfaces, examples=examples, staleness=staleness,
            diagnostics=diagnostics, contract=CONTRACT_FILE,
            extra=tail_extra)
        messages = prompt.messages()

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
        while (_is_truncated(completion, text)
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
        if detect_commentary(text):
            self._emit("warning",
                       f"{task.path}: the model wrote commentary as well as "
                       f"code; the code has been extracted from it.",
                       {"task": task.path})
            text = strip_commentary(text, lang)

        code = _extract(text, lang)
        return code, completion, continued

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

        if history and _cosmetic(history[-1], sig, attempts):
            return ("the model only rearranged whitespace between attempts — "
                    "it is not changing anything that matters")

        if sig.pair in seen:
            where = [a.n for a in attempts
                     if (a.code_sha, a.diag_sha) == sig.pair]
            cycle = _describe_cycle(diags)
            return (f"attempts {', '.join(map(str, where))} and "
                    f"{attempts[-1].n} produced the same code and the same "
                    f"errors — it is going round in a circle{cycle}")

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

    def _journal_attempt(self, task: Task, n: int, completion: Completion,
                         request: str, verify: dict) -> None:
        if self.journal is None:
            return
        try:
            caps = self.host.llm.capabilities()
            remote = bool(caps.is_remote)
        except Exception:                                # noqa: BLE001
            remote = False
        self.journal.generation(
            task=task.path, attempt=n,
            provider=getattr(self.host.llm, "name", "host"),
            completion=completion, prompt=request or task.purpose,
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


def _is_truncated(completion: Completion, text: str) -> bool:
    """`finish_reason == "length"` is the SIGNAL; delimiters are the backstop.

    D1 is explicit that truncation is detected structurally rather than
    inferred. The delimiter check exists for providers that report "stop"
    when they mean "length", which several do.
    """
    if completion.finish_reason == "length":
        return True
    if not text.strip():
        return False
    depth = dict.fromkeys(_OPENERS, 0)
    in_str: str | None = None
    prev = ""
    for ch in text:
        if in_str:
            if ch == in_str and prev != "\\":
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch in _OPENERS:
            depth[ch] += 1
        elif ch in _OPENERS.values():
            for opener, closer in _OPENERS.items():
                if ch == closer:
                    depth[opener] -= 1
        prev = ch
    return any(v > 0 for v in depth.values())


def _join_continuation(head: str, tail: str) -> str:
    """Join without duplicating the overlap the model repeated anyway.

    Models told "do not repeat anything" repeat the last line about a third
    of the time. Detecting the overlap is cheap; a duplicated line in the
    middle of a file is a syntax error that looks like a model failure.
    """
    head_lines = head.rstrip("\n").split("\n")
    tail_lines = tail.lstrip("\n").split("\n")
    for overlap in range(min(8, len(head_lines), len(tail_lines)), 0, -1):
        if head_lines[-overlap:] == tail_lines[:overlap]:
            return "\n".join(head_lines + tail_lines[overlap:])
    # No joiner, ever. `finish_reason == "length"` means the model was cut
    # off mid-TOKEN — `    parts = ` — and the continuation resumes at the
    # very next character. Inserting a newline here produces
    # `parts =\nline.split(...)`, which is a syntax error the model will
    # then be blamed for. The leading newlines a model tends to add are
    # stripped for the same reason.
    return head + tail.lstrip("\n")


def _cosmetic(previous: _Signature, current: _Signature,
              attempts: Sequence[AttemptRecord]) -> bool:
    """Whitespace churn looks like change to a naive hash. It isn't (M34).

    `textio.canonical` already collapses whitespace and strips comments
    before hashing, so an identical canonical hash IS the cosmetic case —
    this is the second line of defence for languages where the canonical
    form still differs (indentation-significant ones).
    """
    return previous.code == current.code and previous.diags == current.diags


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
