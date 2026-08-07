# SPDX-License-Identifier: Apache-2.0
"""The shared dataclasses the Ports carry. Half of the public contract (C9).

`ports.py` holds the Protocols; this holds what flows through them. **Freezing
one without the other freezes half a contract** — a host that implements
`LLMPort` is depending on `Message` and `Completion` just as hard as on the
method signatures. So these follow the same semver rule as the Ports: a
breaking change here is a major version (M8).

Everything here is a frozen dataclass. Two reasons, both practical rather than
ideological:

  * A `Message` that a host can mutate after the core has hashed it for the
    journal makes provenance a lie (C8).
  * Frozen types are hashable and safely shareable across the thread boundary
    a GUI host will inevitably introduce.

Where a field is a sequence it is a `tuple`, not a `list`, for the same
reason. `dict` fields (JSON schemas, arbitrary data) are the exception — a
frozen dataclass with a dict field is shallowly immutable, which is the
honest limit of what this can enforce, and it is noted rather than pretended
away.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

# --------------------------------------------------------------------------
# closed vocabularies
# --------------------------------------------------------------------------
# These are tuples rather than enums on purpose: an enum member arriving from
# a host that vendored an older copy of this file compares unequal to ours,
# and the failure is baffling. Plain strings compare by value across copies,
# which is what a two-host contract needs. StrEnum would be prettier and would
# quietly break the vendoring story (§1.2).

#: Valid `Message.role` values.
ROLES = ("system", "user", "assistant", "tool")

#: Valid `Completion.finish_reason` values. "length" is how truncation is
#: DETECTED rather than inferred (D1, M32).
FINISH_REASONS = ("stop", "length", "tool_calls", "cancelled", "error")

#: The closed set of `EventPort.event` kinds (§5.4, M19). New kinds are a
#: minor version; RENAMED kinds are a major one, because hosts render them.
EVENT_KINDS = ("phase", "token", "status", "diagnostic", "patch", "remote",
               "warning", "error", "budget")

#: Journal event names (§6.13). Also public API — a host renders history from
#: these — and under the same semver rule as EVENT_KINDS.
JOURNAL_EVENTS = ("session_start", "session_end", "plan", "skeleton",
                  "generate", "continuation", "guard", "prefix", "verify",
                  "autofix", "patch", "rollback", "codemap", "review",
                  "budget", "cancel", "epoch", "error")


# --------------------------------------------------------------------------
# the LLM contract
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSpec:
    """A tool the core offers the model. JSON-schema parameters."""
    name: str
    description: str
    parameters: dict            # JSON schema object

    def to_openai(self) -> dict:
        """The shape every OpenAI-compatible endpoint wants.

        Lives here rather than in the provider because four providers need the
        identical conversion and a shared bug is better than four different
        ones.
        """
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolCall:
    """A call the model made. Arguments arrive PARSED.

    The provider owns repair of near-JSON and sets ``repaired`` so chronic
    malformation is VISIBLE rather than silently patched over (D9). A model
    that needs its JSON fixed on every call is telling you something — most
    likely that grammar-constrained decoding is available and not switched on
    — and a silent repair hides the message.
    """
    id: str
    name: str
    arguments: dict
    repaired: bool = False


@dataclass(frozen=True)
class Message:
    """One turn.

    ``images`` is optional vision input (raw bytes + media type); hosts
    without vision ignore it and ``capabilities()`` says so.
    ``tool_call_id`` links a role="tool" result to the call it answers.
    """
    role: str
    content: str
    images: tuple[tuple[bytes, str], ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"role must be one of {ROLES}, not {self.role!r}")


@dataclass(frozen=True)
class Completion:
    """What ``complete()`` returns.

    ``finish_reason == "length"`` is how truncation is DETECTED (D1). Do not
    infer truncation from unbalanced braces alone — that is the backstop, not
    the signal.
    """
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""              # journaled per call (C8, §0.1)
    prompt_ms: int = 0           # prompt-processing time; G.7.5, M55

    def __post_init__(self) -> None:
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(
                f"finish_reason must be one of {FINISH_REASONS}, "
                f"not {self.finish_reason!r}")

    @property
    def truncated(self) -> bool:
        """True when the model ran out of room mid-answer (D1).

        The loop's response to this is CONTINUATION, not regeneration (M32) —
        the difference between paying for the tail and paying for the whole
        file again.
        """
        return self.finish_reason == "length"


@dataclass(frozen=True)
class ModelCapabilities:
    """What the CURRENTLY loaded model can do (M13).

    Re-read at every task boundary, because the host may have swapped models
    between calls and the core has no swap logic of its own (§0.1, M10).
    """
    name: str
    family: str                  # "mistral", "llama", … drives chat templating
    context_tokens: int
    supports_tools: bool = False
    supports_grammar: bool = False
    supports_vision: bool = False
    supports_fim: bool = False   # fill-in-the-middle (G.4)
    is_remote: bool = False      # drives the C3 network banner
    token_count_is_estimate: bool = True

    @property
    def loaded(self) -> bool:
        """False means no model is loaded — a normal state, not an error.

        A host reports it by returning capabilities with an empty name rather
        than by raising, so the core can say so plainly instead of catching.
        """
        return bool(self.name)


# --------------------------------------------------------------------------
# execution and diagnosis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False      # True ⇒ the WHOLE process tree was killed
    truncated: bool = False      # output capped; the cap is stated in the text

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """stdout and stderr together, which is what a parser wants."""
        joiner = "\n" if self.stdout and self.stderr else ""
        return (self.stdout + joiner + self.stderr).strip()


@dataclass(frozen=True)
class Diagnostic:
    """One located problem, with the offending source quoted.

    ``source_excerpt`` is what turns a diagnostic from a citation into
    something a small model can act on — it is why §6.2 is called the
    highest-value module in the spec.
    """
    file: str = ""
    line: int = 0
    col: int | None = None
    severity: str = "error"      # "error" | "warning" | "note" | …
    message: str = ""
    code: str | None = None      # e.g. "E0602", "CS1002"
    source_excerpt: str = ""     # the offending lines, quoted
    tool: str = ""               # which parser produced it: "rustc", "pytest"…

    # Ranking, not sorting order for humans: real errors first, style last.
    _RANK = {"error": 0, "fatal": 0, "exception": 0, "failure": 1,
             "warning": 2, "performance": 3, "portability": 3, "style": 3,
             "note": 4, "info": 4}

    @property
    def rank(self) -> int:
        return self._RANK.get((self.severity or "").lower(), 2)

    @property
    def is_error(self) -> bool:
        return self.rank <= 1

    def where(self) -> str:
        """`file:line:col`, with the file basename — paths are noise here."""
        if not self.file:
            return ""
        name = self.file.replace("\\", "/").rsplit("/", 1)[-1]
        if not self.line:
            return name
        loc = f"{name}:{self.line}"
        return f"{loc}:{self.col}" if self.col else loc

    def one_line(self) -> str:
        head = self.where()
        code = f" [{self.code}]" if self.code else ""
        body = f"{self.severity}: {self.message}{code}"
        return f"{head}: {body}" if head else body

    def key(self) -> tuple:
        """Identity for deduplication and for the stagnation hash (§6.9)."""
        return (self.file, self.line, (self.message or "")[:80])


@dataclass(frozen=True)
class GuardFinding:
    """One thing the static screen noticed (§6.3).

    Severity is "block" or "warn". Read `guard.py`'s docstring before drawing
    any conclusion about what a clean scan means — it is a screen against
    ACCIDENTS, and C10/M9 forbid describing it as anything more.
    """
    severity: str                # "block" | "warn"
    reason: str
    match: str
    line: int = 0

    def one_line(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"{self.severity}: {self.reason}{where} — `{self.match}`"


@dataclass(frozen=True)
class PhaseResult:
    """One attributable phase of a run (M22).

    ``name`` is one of guard/syntax/build/run/test/format/lint. "It didn't
    work" is not a fixable error; "the build failed" is.
    """
    name: str
    argv: tuple[str, ...] = ()
    proc: ProcResult | None = None
    ok: bool = False
    note: str = ""

    @property
    def output(self) -> str:
        return self.proc.output if self.proc else ""


@dataclass(frozen=True)
class RunResult:
    """The outcome of a build/run/test cycle, with the failed phase named.

    C4 lives here: ``succeeded`` is build-AND-test, never "it parsed". When a
    project genuinely has no build or no tests, that is stated in
    ``caveats`` rather than counted as success (M4).
    """
    ok: bool
    lang: str = ""
    phases: tuple[PhaseResult, ...] = ()
    blocked: str = ""            # guard refusal reason, if any
    warnings: str = ""
    diagnostics: tuple[Diagnostic, ...] = ()
    caveats: tuple[str, ...] = ()   # e.g. the headless-Godot caveat (M40)

    @property
    def failed_phase(self) -> str:
        if self.blocked:
            return "guard"
        for p in self.phases:
            if not p.ok:
                return p.name
        return ""

    @property
    def built(self) -> bool:
        return any(p.name == "build" and p.ok for p in self.phases)

    @property
    def tested(self) -> bool:
        return any(p.name == "test" and p.ok for p in self.phases)

    def phase(self, name: str) -> PhaseResult | None:
        for p in self.phases:
            if p.name == name:
                return p
        return None

    @property
    def output(self) -> str:
        return "\n".join(p.output for p in self.phases if p.output).strip()

    def summary(self) -> str:
        """One honest sentence. Never "done" when the evidence is weaker."""
        if self.blocked:
            return f"refused before running — {self.blocked}"
        base = ""
        if self.ok:
            secs = sum(p.proc.duration_s for p in self.phases if p.proc)
            base = f"ok in {secs:.1f}s"
        else:
            phase = self.failed_phase or "run"
            errs = sum(1 for d in self.diagnostics if d.is_error)
            base = (f"{phase} failed — {errs} error{'s' * (errs != 1)}"
                    if errs else f"{phase} failed")
        if self.caveats:
            base += " · " + " · ".join(self.caveats)
        return base


# --------------------------------------------------------------------------
# planning (§6.8)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One unit of work in a plan.

    ``path`` is assigned by the PLANNER, never chosen by the model for an
    existing file (D8, M38). ``test_path`` is not optional decoration: F2's
    test-first cycle is only mechanisable if the pairing exists at planning
    time (M39).
    """
    id: str
    path: str
    purpose: str
    test_path: str = ""
    persona: str = "engineer"
    depends_on: tuple[str, ...] = ()
    atomic: bool = False         # flows into the patcher transaction (M25)
    lang: str = ""
    status: str = "pending"      # pending | active | done | failed | skipped
    attempts: int = 0

    def with_status(self, status: str, attempts: int | None = None) -> Task:
        """Frozen types need an explicit way forward; this is it."""
        return Task(id=self.id, path=self.path, purpose=self.purpose,
                    test_path=self.test_path, persona=self.persona,
                    depends_on=self.depends_on, atomic=self.atomic,
                    lang=self.lang, status=status,
                    attempts=self.attempts if attempts is None else attempts)


@dataclass(frozen=True)
class Plan:
    """A file list with purposes, plus the dependency order.

    The order is DERIVED from the skeleton's imports (§4.2), not asserted by
    the model — a deterministic answer to a question tools can answer (C5).
    """
    request: str
    tasks: tuple[Task, ...] = ()
    layout_note: str = ""
    caveats: tuple[str, ...] = ()

    def task(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def next_ready(self) -> Task | None:
        """The first pending task whose dependencies are all done."""
        done = {t.id for t in self.tasks if t.status == "done"}
        for t in self.tasks:
            if t.status == "pending" and set(t.depends_on) <= done:
                return t
        return None

    def replace(self, task: Task) -> Plan:
        return Plan(request=self.request,
                    tasks=tuple(task if t.id == task.id else t
                                for t in self.tasks),
                    layout_note=self.layout_note, caveats=self.caveats)


# --------------------------------------------------------------------------
# edits and transactions (§6.5)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Edit:
    """One proposed change. ``kind`` in {replace, whole, diff, create}."""
    path: str
    kind: str = "replace"
    old: str = ""
    new: str = ""
    note: str = ""


@dataclass(frozen=True)
class EditResult:
    path: str
    ok: bool
    reason: str = ""
    diff: str = ""


@dataclass(frozen=True)
class TransactionRecord:
    """One entry in the linear transaction log (§6.5 rule 5).

    Sequence numbers, not timestamps (M25): timestamps collide within a
    second, sort wrongly across a clock change, and are ambiguous over a DST
    boundary. The log is strictly linear and the numbering proves it.
    """
    seq: int
    task_id: str
    atomic: bool
    files: tuple[str, ...] = ()
    state: str = "open"          # open | committed | rolled_back | aborted
    verified: bool = False       # committed AND verified ⇒ sealed
    snapshot_dir: str = ""
    diff: str = ""
    note: str = ""

    @property
    def sealed(self) -> bool:
        """A sealed transaction can only be undone by explicit `undo_to`."""
        return self.state == "committed" and self.verified


# --------------------------------------------------------------------------
# codemap (§6.7)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str                    # function | class | method | struct | …
    line: int
    end_line: int = 0
    signature: str = ""
    docstring: str = ""
    path: str = ""
    parent: str = ""
    approximate: bool = False    # regex-extracted: say so wherever it surfaces

    def one_line(self) -> str:
        text = f"{self.line:>5}: {self.kind} {self.signature or self.name}"
        return text + ("  ~approx" if self.approximate else "")


@dataclass(frozen=True)
class CodemapStats:
    """Reported rather than assumed — see the `unresolved` table (§6.7).

    A call graph that silently drops what it could not bind looks complete and
    is not. The resolution rate is the honest version of that number.
    """
    files: int = 0
    symbols: int = 0
    edges: int = 0
    unresolved: int = 0
    epoch: int = 0

    @property
    def resolution_rate(self) -> float:
        total = self.edges + self.unresolved
        return 1.0 if not total else self.edges / total

    def one_line(self) -> str:
        return (f"{self.files} files · {self.symbols} symbols · "
                f"{self.edges} edges · {self.unresolved} unresolved "
                f"({self.resolution_rate:.0%} resolved) · epoch {self.epoch}")


# --------------------------------------------------------------------------
# journal (§6.13)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class JournalEvent:
    """One append-only provenance record (C8, M7).

    The required fields for a `generate` event are the whole point of C8:
    provider, model, prompt hash, attempt, verification outcome, timestamp.
    ``prompt_ms`` is required too (M55) — it is the only signal that the
    prefix cache broke (G.7.5), and a broken cache is otherwise silent.
    """
    t: str                       # ISO-8601 UTC
    event: str
    session: str = ""
    task: str = ""
    attempt: int = 0
    provider: str = ""
    model: str = ""
    prompt_sha256: str = ""
    temperature: float | None = None
    seed: int | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    prompt_ms: int = 0
    verify: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """One line of JSONL. Empty fields are dropped — a journal that is
        90% zeros is harder to read, and readability is the point of it."""
        raw = asdict(self)
        keep = {k: v for k, v in raw.items()
                if v not in ("", 0, None, {}, [])}
        keep["t"] = self.t
        keep["event"] = self.event
        return json.dumps(keep, ensure_ascii=False, sort_keys=True,
                          default=str)


# --------------------------------------------------------------------------
# session-level results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AttemptRecord:
    """What one pass of the loop did (§6.9). The give-up message is built
    from these, so "failed after 4 attempts" never has to be the answer."""
    n: int
    code_sha: str = ""
    diag_sha: str = ""
    diagnostics: tuple[Diagnostic, ...] = ()
    finish_reason: str = "stop"
    continued: bool = False
    autofixes: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class TaskOutcome:
    """The end state of one task, in words an operator can act on."""
    task_id: str
    path: str
    ok: bool
    attempts: tuple[AttemptRecord, ...] = ()
    result: RunResult | None = None
    stopped_because: str = ""    # the cycle report, when there is one (M34)
    caveats: tuple[str, ...] = ()

    def summary(self) -> str:
        if self.ok:
            n = len(self.attempts)
            return (f"{self.path}: done in {n} attempt{'s' * (n != 1)}"
                    + (" · " + " · ".join(self.caveats) if self.caveats else ""))
        why = self.stopped_because or "gave up"
        last = ""
        if self.attempts and self.attempts[-1].diagnostics:
            last = f" Last real error: {self.attempts[-1].diagnostics[0].one_line()}"
        return f"{self.path}: {why}.{last}"


def as_dict(obj: Any) -> dict:
    """A dataclass as a plain dict, for `EventPort.event(data=…)`.

    Hosts render events; a dataclass they may not have imported is not
    renderable, and StoragePort values must be JSON-serialisable anyway (M17).
    """
    return asdict(obj) if hasattr(obj, "__dataclass_fields__") else dict(obj)
