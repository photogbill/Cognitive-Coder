# Cognitive Coder — Build Specification

**Version 1.1 · 2026-08-06 · for a fresh implementation session**

*v1.1 merges the v1.0 review corrections (former Appendices G, H, I) into the
sections they amend, so the document reads once, top to bottom, with a single
authority. What changed and why is recorded in Appendix I (corrections
history). The v1.0 file is retained alongside this one.*

---

## 0. How to use this document

You are being handed this by someone who has already thought hard about it. You
have **no prior context** on the systems it must plug into, so everything you
need is here, including an appendix of facts about the host applications.

Read sections 1–3 and section 5 before writing a line. They contain the
decisions that, if you get them wrong, cannot be fixed later without a rewrite.

**This document has one authority: the text in front of you.** There is no
supersession chain and no correction layer. Where a section touches performance
on the target machine, it points at Appendix G, which is the consolidated
treatment.

The facts the whole document is calibrated against:

- **The target model is Devstral Small 2 24B**
  (`mistralai/Devstral-Small-2-24B-Instruct-2512`), already on the operator's
  disk: 256k context, Apache-2.0, **native tool calling**, multimodal,
  recommended temperature 0.15. Appendix G is written against it specifically.
- The machine is **16 GB VRAM + 64 GB system RAM**. Context can be bought with
  RAM; the real currency is seconds, not capability (Appendix G.2).
- **One model is loaded at a time**, and the host — not this engine — decides
  which. See §0.1.
- Where this document reasons about "a 7B", it is describing *the harder case
  this design also survives* — not the target. Appendix D is that reasoning.

**If you read only four things: §2 (the constitution), §5 (the Ports),
Appendix G (the target machine), and §7 (ATK integration).**

### 0.1 The model strategy — SETTLED 2026-08-06

The operator runs one model at a time under the 16 GB ceiling. **Devstral is
the default for everything** — planning, generation, repair, review. Magistral
(Mistral's reasoning model) is a strength for planning and review, and the
operator may choose to plan with it; when that happens, **the swap is a manual
host action** — in ATK, a button that unloads one model and loads the other —
**never something Cognitive Coder initiates or orchestrates.**

Consequences for the engine, all mandatory:

1. The core has **no swap logic**. It asks `LLMPort.capabilities()` what is
   loaded and works with the answer. "No model loaded" is a normal, reportable
   state, not an exception.
2. A model change between calls MUST be treated as an **epoch boundary**
   (§6.7): the KV cache and any prompt-prefix state died with the old model,
   so the cached prefix is rebuilt, and the journal records the model per call
   (which it does anyway, C8).
3. Operator guidance, surfaced in the ATK panel, not enforced in code: **swap
   at most once per session, at a phase boundary** — plan everything, swap,
   build everything. Alternating per-task pays 30–60 s of prompt reprocessing
   per round trip (Appendix G.6 has the arithmetic). Whether Magistral
   planning beats Devstral-only planning is an empirical question the journal
   can answer; do not assume it.

### 0.2 Three rules for reading

- **"MUST" is load-bearing.** Where this document says MUST, a reviewer will
  check it and reject the work if it's absent. Appendix H is the full index of
  them, numbered, for exactly that purpose.
- **Where it says "recommended", you may exercise judgement** — but write down
  what you chose and why, in the module docstring, so the next person can
  disagree with a reason rather than a mystery.
- **Where it says "OPEN QUESTION", stop and ask the project owner.** Don't
  guess. The list is short and deliberate (Appendix B).

The author is Bill (github: photogbill). The module is being published so that
**ParisNeo**, author of LoLLMs, can use it too. That second audience is not a
nice-to-have — it is the reason for most of the architectural constraints in
section 3.

---

## 1. What Cognitive Coder is

**A host-agnostic engine that lets a language model write, build, test, and fix
real code in a real project — designed so it still works when the model is
small and the machine is offline.**

It is a **library first** and an application second. It ships:

- `cognitive_coder/` — a pure-Python package with **no GUI dependency and no
  hard dependency on any host application**.
- Optional adapters: **ATK (PySide6) and a CLI — these two are in scope.**
  A LoLLMs adapter is possible and explicitly NOT part of this build; §8.
- Optional providers: local GGUF via llama-cpp-python, an OpenAI-compatible
  HTTP endpoint (llama.cpp server, Ollama, LM Studio, vLLM), and — strictly
  opt-in — remote APIs.

### 1.1 The honest pitch, including where it loses

Bill asked whether this can be "as good as Cline, or better". The truthful
answer has three parts, and the specification is built around it.

**Where Cognitive Coder can genuinely win:**

| Axis | Why |
|---|---|
| **Works offline, on small models** | Cline assumes a frontier model behind an API. Every design decision here assumes a 7B–24B local model and compensates with scaffolding. |
| **Host-agnostic** | Cline is a VS Code extension. This is a library that embeds in a Qt desktop app, a FastAPI web app, or a terminal. |
| **Audit-grade provenance** | Every line produced is traceable to a model, a prompt, an attempt number and a verification result. Cline does not do this. For anyone who has to defend a change, it's decisive. |
| **Deterministic-first** | Linters, compilers and test runners answer what they can answer; the model is asked only what tools cannot decide. That is cheaper, faster and more trustworthy. |

**Where it will lose, and you should not pretend otherwise:**

- Raw code quality when Cline is driving Claude or Gemini. A local 24B narrows
  the gap; it does not close it.
- Ecosystem polish: Cline has an editor's whole UI, inline decorations, and
  years of interaction design.

**So do not build "Cline but ours".** Build the thing Cline cannot be: a
coding engine that runs on a disconnected laptop in a field office, embeds in
somebody else's application, and can prove what it did.

### 1.2 The name and how it is distributed

`Cognitive Coder`. Import package `cognitive_coder`, CLI entry point `ccoder`.

**It is NOT published to PyPI.** Distribution is: clone the repository, run
`install.bat` on Windows or `install.sh` on Linux. This is a deliberate choice
by the owner and it shapes several things — see §10, which specifies both
installers in full.

Consequences to keep in mind while building:

- The import name must still be distinctive enough not to collide with anything
  a host already has on `sys.path`. `cognitive_coder` is safe (a PyPI search on
  2026-08-06 found nothing using it), and being un-published means it will stay
  that way in practice.
- **`pip install -e .` must still work** from a clone, because that is what the
  installers do internally and what a contributor will run. A valid
  `pyproject.toml` is required even though nothing is uploaded.
- A host that vendors the code (ParisNeo may prefer a git submodule) must be
  able to `sys.path`-insert the repo root and `import cognitive_coder` with no
  install step at all. **Test that path explicitly** — no import-time side
  effects, no reliance on package metadata, no `importlib.metadata.version()`
  calls that fail when uninstalled. Read the version from `version.py`.

---

## 2. The constitution — non-negotiables

These are the principles that must survive every implementation decision. If a
convenience conflicts with one of these, the convenience loses.

### C1. The core imports no GUI and no host

`cognitive_coder/**` MUST NOT import PySide6, Qt, FastAPI, ATK, or LoLLMs.
Not conditionally, not inside functions. The only permitted third-party
imports in the core are ones a host may already have and can be made optional
(see C7).

*Why:* the moment the core imports Qt, LoLLMs cannot use it, and the whole
reason this is a separate module evaporates.

### C2. Everything the host provides comes through a Port

Model inference, file writes, sandboxed execution, storage, progress events —
all of it arrives through a small, typed protocol the host implements. The core
never reaches out to find these things itself.

*Why:* it is the difference between "works in ATK" and "works anywhere". Also
makes the whole engine testable with fakes, which section 9 requires.

### C3. Offline is the default; the network is an explicit, visible choice

No remote call may happen unless the operator has enabled a remote provider
**for this session**, and the UI/CLI must show that remote mode is active
whenever it is. There is no "helpfully falls back to the cloud".

*Why:* one host (ATK) is an air-gapped, zero-telemetry tool used in places
where an outbound connection is a safety problem, not an inconvenience. A
coding module that silently phoned home would break that promise on ATK's
behalf. This is the single most important constraint in the document.

### C4. Nothing is "done" until it builds and the tests run

A generated file that parses is not finished. The engine's definition of
success is: **the build command succeeded AND the test command succeeded** (or
the language/project genuinely has neither, in which case say so explicitly in
the result). `ast.parse()` passing is a *pre-check*, never a completion signal.

### C5. Deterministic first, model second, human last

Where a compiler, linter, formatter, type checker or test runner can answer a
question, it answers it. The model is asked to fix what those tools reported,
or to judge what they cannot. A model's opinion is more fluent and less
checkable than a compiler's.

### C6. Every failure is a sentence, never a traceback

Any error that can reach an operator MUST be a plain sentence naming what
happened and what to do. Internal tracebacks go to the journal, not the face.

### C7. Optional dependencies degrade, never crash

No linter installed? Skip that check and say so in the result. No tokenizer?
Fall back to character budgeting and state the assumption. `tree-sitter` not
available? Use the regex extractor and label the output as approximate.

### C8. Provenance is not optional

Every artefact records: which provider and model produced it, the prompt hash,
the attempt number, the verification outcome, and the timestamp. This is
written to a journal the host can read. It is what makes the tool defensible.

### C9. The API surface is frozen at 1.0 and versioned with semver

Two hosts will depend on this. A breaking change to a Port or to the shared
types the Ports carry (§5) is a major version. See section 10.4.

### C10. The trust model is stated, not implied

Running model-generated code is not an accident of this design — **it is the
product.** C4 requires executing builds and tests, which executes generated
code on the operator's machine. The layers of defence, in order, are:
`guard.py`'s static screen (accidents, not adversaries — §6.3), the host's
`ExecPort` sandboxing policy (the host decides what "sandboxed" means for it),
the scrubbed environment (§6.4), the project-root jail (§6.5), and the
approval gate (§5, `ApprovalPort`). Every one of these must exist; none of
them alone is sufficient; and no documentation may describe the combination as
a security boundary against a hostile model. Say what it is: a screen against
mistakes, operated by a human who stays in charge.

---

## 3. Architecture: ports and adapters

```
        ┌──────────────────────────────────────────────────┐
        │                     HOSTS                        │
        │   ATK (PySide6)   LoLLMs (FastAPI)   ccoder CLI  │
        └───────────────┬──────────────────────────────────┘
                        │ implements Ports
        ┌───────────────▼──────────────────────────────────┐
        │                    PORTS                          │
        │  LLMPort · FileSystemPort · ExecPort ·            │
        │  StoragePort · EventPort · ApprovalPort           │
        └───────────────┬──────────────────────────────────┘
                        │ used by
        ┌───────────────▼──────────────────────────────────┐
        │              COGNITIVE CODER CORE                 │
        │                                                   │
        │  session ── planner ── loop ── verifier           │
        │      │         │        │         │               │
        │  journal    codemap   patcher   diagnostics       │
        │      │         │        │         │               │
        │   context    langs    guard     review            │
        └──────────────────────────────────────────────────┘
```

**Direction of dependency is one-way.** Core knows about Ports. Hosts know
about Core. Core never knows about hosts.

### 3.1 Package layout

```
cognitive-coder/
├── pyproject.toml
├── LICENSE                       # Apache-2.0 (see §10.4)
├── README.md
├── CHANGELOG.md
├── docs/
│   ├── EMBEDDING.md              # how to host it (includes vendoring)
│   ├── PORTS.md                  # the contract, with examples
│   └── PROVIDERS.md
├── examples/
│   └── tiny_host.py              # ~50 lines, Null ports only — proves the
│                                 # embedding story end to end; see §9
├── cognitive_coder/
│   ├── __init__.py               # the PUBLIC API — nothing else is stable
│   ├── version.py
│   ├── ports.py                  # Protocols. The contract. §5
│   ├── types.py                  # the shared dataclasses. Also the contract. §5
│   ├── errors.py
│   ├── langs.py                  # language registry            §6.1
│   ├── diagnostics.py            # compiler/runtime output → structured  §6.2
│   ├── guard.py                  # static screen                §6.3
│   ├── runner.py                 # build / run / test / format / lint  §6.4
│   ├── patcher.py                # transactions, snapshots, undo  §6.5
│   ├── textio.py                 # encoding + EOL detection/preservation §6.5a
│   ├── context.py                # what the model gets to see   §6.6
│   ├── codemap/
│   │   ├── __init__.py
│   │   ├── parse_python.py       # ast-based, exact
│   │   ├── parse_regex.py        # ctags-style fallback
│   │   ├── parse_treesitter.py   # optional, if tree_sitter present
│   │   ├── store.py              # SQLite registry
│   │   └── zoom.py               # semantic zoom + budget       §6.7
│   ├── planner.py                # task decomposition           §6.8
│   ├── loop.py                   # generate → verify → repair   §6.9
│   ├── review.py                 # deterministic + adversarial  §6.10
│   ├── personas.py               # prompts, ToM                 §6.11
│   ├── providers/
│   │   ├── __init__.py           # registry + selection
│   │   ├── base.py
│   │   ├── local_llamacpp.py
│   │   ├── openai_compatible.py  # llama.cpp server, Ollama, LM Studio, vLLM
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   ├── mistral.py
│   │   ├── openrouter.py
│   │   └── openai.py
│   ├── redact.py                 # outbound scrubbing           §6.12
│   ├── session.py                # orchestration + resume       §6.13
│   ├── journal.py                # provenance                   §6.13
│   └── cli.py                    # `ccoder`
├── adapters/
│   ├── atk/                      # PySide6 panel + ATK port impls
│   └── lollms/                   # placeholder README only; see §8
└── tests/
    ├── port_conformance.py       # reusable kit hosts run on THEIR ports §9
    └── fixtures/
        └── worked_session.jsonl  # Appendix E as a golden trace  §9
```

**`adapters/` is deliberately outside the package.** ATK's adapter imports
PySide6. Keeping adapters out of `cognitive_coder/` is what enforces C1
mechanically — a lint rule can assert that no file under `cognitive_coder/`
imports from `adapters/`.

---

## 4. What I changed from the FORGE specification, and why

The original FORGE document is good and most of it survives. These are the
deliberate departures. **Read this section even if you read nothing else in
the review** — it is where the reasoning lives.

### 4.1 It is no longer inside ATK

FORGE placed everything in `atk/core/forge_*.py` and `atk/ui/forge_panel.py`,
importing `atk.core.llm_engine`, `atk.core.sandbox`, `atk.core.debug_loop`,
`atk.config`. That makes it ATK-only, which contradicts the goal of ParisNeo
being able to use it. Every one of those imports becomes a Port.

### 4.2 The DAG-of-files planner is the highest-risk idea in the document

FORGE has the Supervisor emit "a strict Directed Acyclic Graph of executable
files" as JSON, up front, for the whole project. With a frontier model that
works. With a small local model it is the most likely point of total failure.
At 24B the *format* risk is modest — Devstral will emit valid JSON — but the
structural argument holds regardless: **a compiling skeleton catches
ARCHITECTURAL error, which valid JSON does not.** A plan that is valid JSON
and architecturally wrong poisons every downstream step.

**Replacement — skeleton-first, one body at a time:**

1. Ask for a **file list with one-line purposes** (small, cheap, constrained).
2. Generate **stubs only** — signatures, imports, docstrings, `raise
   NotImplementedError`. Verify the whole skeleton *imports/compiles*. This
   catches architectural nonsense in seconds, before any real work.
3. Fill bodies **one file at a time**, verifying after each.
4. Re-plan after each file if the codemap says the shape changed.

Keep the DAG as a *data structure* — dependency order is genuinely useful for
choosing what to build next — but derive it from imports in the skeleton
(deterministic, C5) rather than asking the model to assert it.

Where the model must emit JSON, constrain it: grammar-constrained decoding
where the provider supports it (llama.cpp GBNF, tool schemas), and a lenient
repair parser where it doesn't.

### 4.3 Adversarial debate happens *after* verification, and starts deterministic

FORGE runs Security and Performance personas as a review stage. Keep it — but:

- It runs **after** the code builds and its tests pass, not instead. Reviewing
  code that doesn't compile spends tokens on a moot point.
- The Security reviewer starts with **actual scanners** (`bandit`, `semgrep`,
  `cppcheck`, `gosec` — whatever is installed) and a built-in secret/unsafe-API
  scan. The model is then asked about what tools cannot see: trust boundaries,
  logic flaws, misuse potential.
- The Performance reviewer likewise starts with measurable facts: complexity
  metrics, allocation in hot loops found by pattern, and — where a benchmark
  exists — actual timings.
- On the target machine, **run both model reviews as one structured pass** —
  one call, one schema'd output covering security and performance. Two
  sequential passes at minutes each buys nothing a schema can't (Appendix G.5).
- **Two personas both being the same local model is not an adversarial
  system.** Say so honestly in the output. Real adversarial value requires
  either a different model (the host's swap button, §0.1) or a genuinely
  different information set. Never present same-model self-review as
  independent scrutiny; that is the confident-wrongness failure mode.

### 4.4 The reviewers' output contract (learned the hard way)

A hard-won lesson from ATK: a model asked to "review X and return it improved"
will, if it is small, write a *review* — headed `**Improved Reply:**`,
`**Why?**`, `**Key changes:**` — and the calling code will hand that to the
user as the product.

Therefore, in this codebase, **every prompt whose output is consumed
programmatically MUST end with an explicit output contract** naming the shape
required and the headings forbidden, AND the caller MUST have a detector that
recognises commentary and falls back rather than shipping it. Prompting alone
does not hold. See `personas.py` acceptance criteria (§6.11). This stays even
though Devstral is well behaved: it is cheap insurance, and the failure it
prevents is one already observed in this project's own Athena panel.

### 4.5 CodeMap: kept, and it's the best idea in the addendum

The Module 6 addendum (AST Code-RAG, blast radius, semantic zoom, mid-generation
search) is genuinely excellent and is promoted to a first-class subsystem.
Changes:

- **Blast radius also selects tests.** If a signature changes, don't just queue
  callers for refactor — run the tests that touch those callers *first*. That
  turns a static warning into a fast empirical answer.
- **The token ceiling must be measured in the target model's tokens** where a
  tokenizer is available, and in characters with the ratio stated where it
  isn't. A "max_tokens" slider that actually counts characters silently is a
  lie that overflows contexts.
- **Omissions must be declared.** The injected context ends with an explicit
  list of what was left out. A model that doesn't know something was withheld
  will confidently reference it.
- **Mid-generation lookup is a native tool call**, not a text marker. The
  target model has trained tool calling, so `search_codemap(name)` is a real
  tool with a JSON schema (§6.7). The lenient text-marker parser
  (`[SEARCH_CODEMAP: x]` and variants, capped at 3 per generation) is kept as
  a **fallback only**, for hosts running a model without tool support.

### 4.6 Additions the FORGE spec doesn't have

| Addition | Why |
|---|---|
| **Journal / provenance** (§6.13) | Which model, prompt, attempt and verification produced each artefact. Differentiator, and required by C8. |
| **Resumability** (§6.13) | A twelve-file build that dies at file seven must resume. Long local generations are slow; restarting is brutal. |
| **Cost & latency budget** (§6.12) | API mode needs a hard token/spend ceiling that halts rather than surprises. |
| **Outbound redaction** (§6.12) | If context can leave the machine, secrets must be stripped first. Non-negotiable. |
| **Reproducibility** | Record seed and temperature; support replaying a session's prompts. |
| **Language registry** (§6.1) | FORGE names languages in a wizard but never centralises how to build them. |
| **Structured diagnostics** (§6.2) | The highest-value component; FORGE only feeds back raw tracebacks. |
| **Transactions with sealed history** (§6.5) | FORGE's undo story was a function-call boundary; that ambiguity destroys verified work. |

### 4.7 On Cline and openwork

Bill linked both. Use them as **reference for interaction design, not as
dependencies**: Cline's plan/act separation, its diff-review flow, and its
checkpointing are all good ideas that this spec independently arrives at.
Neither is embeddable here — both are editor/product-shaped, not library-shaped.
There is no license or code to take; take the ideas.

---

## 5. The Ports and the shared types — the contract

`cognitive_coder/ports.py` holds the Protocols; `cognitive_coder/types.py`
holds the dataclasses they carry. **Together they are the public contract
(C9).** The types are as load-bearing as the Protocols — freezing one without
the other freezes half a contract.

Ports are `typing.Protocol` classes: a host implements them structurally,
with no inheritance and no import of this package's base classes required.

### 5.1 The shared types

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """A tool the core offers the model. JSON-schema parameters."""
    name: str
    description: str
    parameters: dict            # JSON schema object


@dataclass(frozen=True)
class ToolCall:
    """A call the model made. Arguments arrive PARSED — the provider owns
    repair of near-JSON and sets `repaired` so chronic malformation is
    visible rather than silently patched over (Appendix D9)."""
    id: str
    name: str
    arguments: dict
    repaired: bool = False


@dataclass(frozen=True)
class Message:
    """One turn. role ∈ {"system", "user", "assistant", "tool"}.
    `images` is optional vision input (raw bytes + media type); hosts
    without vision ignore it and capabilities() says so.
    `tool_call_id` links a role="tool" result to the call it answers."""
    role: str
    content: str
    images: tuple[tuple[bytes, str], ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Completion:
    """What complete() returns. finish_reason ∈
    {"stop", "length", "tool_calls", "cancelled", "error"}.
    "length" is how truncation (D1) is DETECTED, not inferred."""
    text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    tokens_in: int
    tokens_out: int
    model: str                  # journaled per call (C8, §0.1)


@dataclass(frozen=True)
class ModelCapabilities:
    name: str
    family: str                  # "mistral", "llama", … drives chat templating
    context_tokens: int
    supports_tools: bool
    supports_grammar: bool
    supports_vision: bool
    supports_fim: bool           # fill-in-the-middle (Appendix G.4)
    is_remote: bool              # drives the C3 network banner
    token_count_is_estimate: bool


@dataclass(frozen=True)
class ProcResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool              # True ⇒ the WHOLE process tree was killed
    truncated: bool              # output capped; cap size is stated in text


@dataclass(frozen=True)
class Diagnostic:
    file: str
    line: int
    col: int | None
    severity: str                # "error" | "warning" | "note"
    message: str
    code: str | None             # e.g. "E0602", "CS1002"
    source_excerpt: str          # the offending lines, quoted
    tool: str                    # which parser produced it: "rustc", "pytest"…
```

`Plan`, `Task`, `Transaction`, and journal-event shapes are defined in their
owning sections (§6.8, §6.5, §6.13) and also live in `types.py`.

### 5.2 Cancellation

The core is synchronous, and a single generation on the target machine takes
minutes. A GUI host needs a defined way to stop one. The mechanism:

```python
class CancelToken(Protocol):
    def is_set(self) -> bool: ...
```

- `Session` owns a token; `Session.cancel()` sets it (thread-safe — this is
  the ONE thing a host may call from another thread).
- The core checks the token **between phases** (before each build/run/test,
  before each model call, between tool round-trips).
- `LLMPort.complete()`/`stream()` receive the token and SHOULD check it
  between chunks, returning a `Completion` with `finish_reason="cancelled"`.
  A provider that cannot interrupt mid-generation finishes the call and the
  core stops at the next boundary — slower, but defined.
- A cancelled task leaves resumable state (§6.13) and an open transaction is
  rolled back (§6.5). Cancellation is journaled.

### 5.3 The Ports

```python
from typing import Protocol, Iterator, Sequence, Any


class LLMPort(Protocol):
    """Messages in, completion out. Tool calling and vision are part of the
    1.0 contract because the target model has both and the loop is built
    around them (§6.7, §6.9)."""

    def complete(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (),
                 temperature: float = 0.15, max_tokens: int = 2048,
                 stop: Sequence[str] | None = None,
                 grammar: str | None = None,
                 seed: int | None = None,
                 cancel: CancelToken | None = None) -> Completion:
        """Blocking completion. MUST NOT raise on model refusal — return the
        text. If tools are supplied and the model calls one, finish_reason
        is "tool_calls" and the core answers with a role="tool" Message on
        the next call. A host whose model lacks tool support MUST ignore
        `tools` and report supports_tools=False — the core then uses the
        text-marker fallback (§6.7)."""

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        """Token stream for display; the final Completion is available via
        complete()-equivalent semantics. A host without streaming may yield
        one chunk. Checks the cancel token between chunks."""

    def capabilities(self) -> ModelCapabilities:
        """MUST reflect the CURRENTLY loaded model. The host may change
        models between calls (§0.1); the core re-reads this at every task
        boundary and treats a change as an epoch boundary (§6.7)."""

    def count_tokens(self, text: str) -> int:
        """Exact where the host has a tokenizer; a documented estimate
        otherwise. `token_count_is_estimate` says which. Tokenizer
        dependencies (e.g. mistral-common) live in the HOST or provider,
        never in the core (§10.3)."""


class FileSystemPort(Protocol):
    """All file access. Hosts enforce their own jail here.
    Bytes methods are the primitives; the core's textio layer (§6.5a) owns
    encoding and EOL handling ON TOP of them, so snapshots and undo can be
    byte-identical (§9)."""
    def read_bytes(self, path: str) -> bytes: ...
    def write_bytes(self, path: str, content: bytes) -> None:
        """MUST be atomic (write-temp-then-rename) or the host's PORTS.md
        entry must say it is not."""
    def read(self, path: str) -> str: ...      # convenience: UTF-8, errors="replace"
    def write(self, path: str, content: str) -> None: ...
    def exists(self, path: str) -> bool: ...
    def list(self, glob: str) -> list[str]: ...
    def delete(self, path: str) -> None: ...
    def root(self) -> str:
        """The project root. The core resolves every path it touches to a
        real path and refuses anything that escapes this root — including
        via symlinks and `..` (§6.5)."""


class ExecPort(Protocol):
    """Running a command. The host decides what 'sandboxed' means for it (C10)."""
    def run(self, argv: Sequence[str], *, cwd: str, timeout: float,
            stdin: str = "", env: dict | None = None) -> ProcResult:
        """On timeout the ENTIRE process tree MUST be killed — on Windows a
        terminated shell does not take its children with it, and orphaned
        compilers/test runners/Godot instances are a real failure mode.
        `timed_out=True` in the result attests that the tree is dead."""
    def which(self, binary: str) -> str | None: ...


class StoragePort(Protocol):
    """Key-value + a SQLite path. Hosts choose where state lives.
    Values MUST be JSON-serialisable — that is the portability contract."""
    def get(self, key: str, default: Any = None) -> Any: ...
    def set(self, key: str, value: Any) -> None: ...
    def sqlite_path(self, name: str) -> str: ...


class EventPort(Protocol):
    """Progress, logging, and streamed output. Hosts render it."""
    def event(self, kind: str, message: str, data: dict | None = None
              ) -> None: ...


class ApprovalPort(Protocol):
    """Human-in-the-loop. A host may auto-approve; it must say so.
    ALL writes route through approve_diff — including writes the MODEL
    initiates through the apply_patch tool (§6.7). Tool calling MUST NOT
    become a side door around the approval default (§6.5)."""
    def approve_diff(self, summary: str, unified_diff: str) -> bool: ...
    def approve_remote(self, provider: str, bytes_out: int,
                       estimate: str) -> bool: ...
```

### 5.4 The event vocabulary — fixed, because hosts render it

`EventPort.event(kind, …)` kinds are a **closed set at 1.0** (new kinds are a
minor version; renamed kinds are a major one):

| kind | meaning | typical data |
|---|---|---|
| `phase` | loop phase change | `{"phase": "build", "task": "src/x.py", "attempt": 2}` |
| `token` | streamed model output | `{"text": "…"}` |
| `status` | one-line human status | — |
| `diagnostic` | a parsed Diagnostic | the dataclass as dict |
| `patch` | a transaction applied / rolled back | `{"seq": 12, "files": […]}` |
| `remote` | remote mode on/off, bytes out, redaction count | per §6.12 |
| `warning` | degraded mode, headless caveat, estimate in use | — |
| `error` | a sentence per C6 | journal ref for the traceback |
| `budget` | wall-clock / token / spend checkpoint | per §6.12, F11 |

### 5.5 Acceptance criteria for the contract

- Every Port has a **`Null*` implementation** in the same module (`NullLLM`,
  `MemoryFileSystem`, `SubprocessExec`, `MemoryStorage`, `SilentEvents`,
  `AutoApprove`) so the engine is runnable and testable with zero host.
- `MemoryFileSystem` and `SubprocessExec` are what the test suite uses.
- Docstrings state, for each method, **what a host may assume and what it must
  guarantee** — and `tests/port_conformance.py` (§9) makes those guarantees
  executable, so a host can verify its own implementations without reading
  the core.
- `examples/tiny_host.py` drives a trivial session end to end on Null ports.
  It is the README example grown just enough to prove the embedding story,
  and the first thing ParisNeo is handed.

---

## 6. Modules, in build order

Build in this order. Each module lists its purpose, the design decisions that
matter, and acceptance criteria that must be met before moving on.

> **Reuse note.** Six of these modules already exist, written for ATK but
> deliberately Qt-free and dependency-light: `langs.py`, `diagnostics.py`,
> `codeguard.py`, `coderun.py`, `patcher.py`, `codectx.py`. They were built
> against the same principles as this document. **Port them as the starting
> point rather than writing from scratch** — see Appendix A.4 for exactly where
> they are and what has to change. This will save days and, more importantly,
> carries over bug fixes that were found the hard way.

### 6.1 `langs.py` — the language registry

Everything language-specific is a lookup here: extensions, build and run
commands, test/format/lint commands, entry-point conventions, runnable
scaffolds, comment syntax, and whether the language cascades errors (F7).

**Decisions that matter:**

- **Build and run are separate phases.** A compile error and a runtime crash
  are different problems needing different feedback. "It didn't work" is not
  fixable.
- **Commands are argv lists with placeholders, never strings.** A path
  containing a space (`C:\Users\Bill Smith\`) breaks string commands in a way
  that looks like a compiler bug.
- **Every language ships a runnable scaffold with a test hook.** A small model
  asked to invent project structure invents it wrong; asked to fill a body in
  a file that already compiles, it does well. This is one of the highest-value
  decisions in the whole design.
- **Toolchain presence is probed at runtime** via `ExecPort.which()`. The
  installer's detection record (§10.2) is informational — a compiler installed
  the week after install day must simply work.

**Languages (minimum):** Python, C, C++, Rust, Java, Go, C#, JavaScript,
TypeScript, Bash, PowerShell, Lua, Ruby, SQL(SQLite), Zig, Batch, **GDScript**.

#### 6.1a GDScript is first class — SETTLED 2026-08-06

Not an outline-only afterthought. It gets the same treatment as Python:

| Concern | How |
|---|---|
| **Syntax check** | `godot --headless --check-only --script <file>` — a real pre-check, the GDScript equivalent of `ast.parse`. |
| **Run** | `godot --headless --script <file>` for a standalone script; `godot --headless --path <project>` for a scene. |
| **Test** | **GUT** (`godot --headless -s addons/gut/gut_cmdln.gd -gdir=res://test -gexit`) or **gdUnit4** if present. Detect which the project uses from `addons/`; if neither, say so and fall back to running the script. |
| **Diagnostics** | Godot emits `SCRIPT ERROR: ... at: <func> (res://path.gd:LINE)` and `Parse Error: ... at line N`. **Add both patterns to `diagnostics.py`** — without them the loop is blind on this language. |
| **Outline** | Regex patterns in `codemap/parse_regex.py`: `func name(args) -> Type:`, `class_name X`, `extends Y`, `signal s(...)`, `@export var`, `const`, inner `class X:`. Indentation-scoped like Python, so end-of-body is derivable rather than guessed. |
| **Scaffold** | A `Node`-extending script with `_ready()`, a pure function, and a GUT test file that asserts it. |
| **Paths** | Godot uses `res://`. The patcher works in OS paths; **translate at the boundary and never let `res://` reach `FileSystemPort`.** This is the one real trap in GDScript support. |

**Headless Godot is weaker evidence, and must be labelled as such.** A
headless run differs from a real one in ways that produce both false passes
and false failures: no rendered viewport, `SceneTree` behaviour differs,
`_process` deltas are not wall-clock, physics ticks may not advance, and
anything touching `RenderingServer` or viewport textures may silently no-op.
Therefore:

1. **A hard timeout on every Godot invocation** (default 120 s for tests,
   30 s for `--check-only`). The tree-kill guarantee is `ExecPort`'s (§5.3),
   and Godot is the reason it exists — it spawns children, and a script
   awaiting a signal that never fires would otherwise hang the session.
2. **`--fixed-fps N`** for test runs so frame deltas are deterministic.
   A test whose result depends on wall-clock timing is not a test.
3. **Classify the test before trusting it.** Scan for `get_viewport`,
   `_physics_process`, `RenderingServer`, `get_tree()`, `await`, `Timer`. If
   present, the result carries a **headless caveat** in the report:
   *"passed headlessly; this test touches the scene tree / physics /
   rendering, which behaves differently without a viewport. Verify in the
   editor."*
4. **A headless pass on rendering-dependent code is never reported as
   unqualified success.** That is C4 and the honest-failure doctrine (A.3.2):
   the engine may say "the tests passed", but not "this works", when it knows
   the evidence is weaker than it looks.

**Honest constraint to state in the UI:** without a `godot` binary on PATH,
GDScript degrades to outline-and-edit only — no syntax check, no run, no
tests. That is a much weaker mode and the operator must be told which one they
are in (C7). The installer's toolchain detection (§10.2) probes for `godot`
alongside the compilers.

**Acceptance:** for every language whose toolchain is present on the build
machine, `scaffold_for(lang)` produces a file that builds and runs and prints
its greeting. The test asserts this dynamically, skipping absent toolchains
with a printed note.

### 6.2 `diagnostics.py` — the highest-value module

Parse compiler/interpreter/linter/test output into `Diagnostic` (§5.1), quote
the offending source lines, sort real errors first, and cap what goes back to
the model.

**Why it matters more than anything else here:** a small model plus a 200-line
build log fixes nothing — the log is mostly noise and the model's attention
lands on whatever is longest. The same model given *three located errors with
the source quoted* fixes them. This module is the difference between a toy and
a tool.

**Must parse:** gcc/clang, MSVC, rustc (message and `-->` location are on
different lines and must be paired *in order*), javac, Python tracebacks (take
the **deepest** frame plus the exception, not the whole stack), Node/JS stacks
(first frame is deepest — opposite of Python), Go, cppcheck, shellcheck,
unittest/pytest failures, TypeScript, and Godot (§6.1a).

**Critical behaviour:** unrecognised output is **never silently dropped** — it
comes back as one diagnostic holding the last meaningful lines. Returning an
empty list on a failed build makes the loop report success on broken code.

**Acceptance:** a table-driven test with real captured output from each
toolchain; each asserts file, line and message. Plus: given text that matches
nothing, `parse()` returns exactly one diagnostic and never `[]`.

### 6.3 `guard.py` — static screen

Per-language patterns for destructive filesystem operations, process spawning,
network access, registry access and dynamic evaluation, with `block` and `warn`
severities.

**State plainly, in the module docstring, that this is a screen against
ACCIDENTS and not a security boundary** (C10). A determined adversary defeats
a regex; a model that misunderstood the task does not. Anyone reading the code
must leave knowing which one it defends against.

`explain_to_model(findings)` returns an instruction ("use X instead of Y")
rather than a complaint ("that was blocked") — a small model told only that it
failed tends to reword the same code.

One addition: patterns for **version-control commands** (`git commit`,
`git push`, `git reset`, history rewriting) in generated code are `block` by
default. The engine has its own snapshot story (§6.5b); generated code has no
business touching the operator's history.

### 6.4 `runner.py` — build, run, test, format, lint

Phases are attributable: every result names which phase failed (`guard`,
`build`, `run`, `test`). Scrubbed environment — no inherited tokens, API keys,
or proxy variables (`HTTP_PROXY`/`HTTPS_PROXY` included; a proxy variable is a
network path, C3). Per-phase timeouts with tree-kill (§5.3), output truncated
with a stated cap.

**Acceptance:** a deliberately broken C file returns `failed_phase == "build"`
with a located diagnostic; a compiling C file that divides by zero at runtime
returns `failed_phase == "run"`. If those two are indistinguishable, the module
is wrong.

### 6.5 `patcher.py` — edits with a way back

Three edit formats: **anchored replace** (exact old→new), **unified diff**
(applied with context verification), **whole file**.

**The rule that prevents the worst class of damage:** an anchor that matches
more than once is **refused, not guessed**. Ambiguity means the model didn't
give enough context, and picking the first match is how the wrong function gets
edited.

Nothing may be written outside the project root, ever, regardless of mode —
and "outside" is judged on **resolved real paths**: normalise, resolve
symlinks, then check containment under `FileSystemPort.root()`. A `..`
component or a symlink that points out of the tree is a refusal with a plain
sentence, not a write.

**Transactions — explicit, never inferred.** "Undo restores an apply as one
unit" is ambiguous when a loop edits file A (verified good) then file B
(fails): the dangerous reading is the one where rolling back B also reverts A.
Both behaviours are needed, and which applies is *declared*:

- File A verified and committed, B fails → **A stands.** Reverting verified
  work because a later, separate task failed destroys progress the operator
  watched succeed.
- A and B are two halves of one change (a signature and its caller) → **both
  revert.** Leaving A applied is a broken tree that compiles nowhere.

```python
tx = patcher.begin(task_id="add-parser", atomic=True)
tx.apply(edits_for_a)
tx.apply(edits_for_b)
tx.commit()          # or tx.rollback() → reverts EVERYTHING in this tx only
```

Rules:

1. A transaction is opened by the **planner**, per task, and carries `atomic`
   from the plan — a multi-file refactor is atomic; three independent files
   are three transactions.
2. **Sequence numbers, not timestamps.** Snapshot directories are
   `NNNN-<task_id>/` with a monotonic counter. Timestamps collide within a
   second, sort wrongly across a clock change, and are ambiguous over a DST
   boundary. The transaction log is strictly linear and the numbering proves it.
3. **A committed, verified transaction is SEALED.** `rollback()` on a later
   transaction can never touch it. Reaching further back requires an explicit
   `undo_to(seq)` from the operator, which states in plain words how many
   verified transactions it is about to discard.
4. **Rollback is itself journaled** as an event with its own sequence number.
   The history is append-only; undo is a new fact, not an erasure.
5. `patcher.history()` returns the linear log — sequence, task, files,
   verification outcome, and whether it was later rolled back. That is what the
   UI shows and what "what did it just do to my project" is answered from.
6. **Model-initiated edits are not special.** When the model calls the
   `apply_patch` tool (§6.7), the edit enters the *same* transaction and the
   *same* `ApprovalPort` gate as any other edit. Tool calling must never be a
   side door around the approval default.

**Auto-apply vs approval — SETTLED 2026-08-06:**

- The **library default is approval-required.** A new host, or a first run, must
  never silently write to someone's project.
- **Auto-apply is opt-in, and lives behind an advanced setting with an explicit
  warning** naming what it does and that undo is the only safety net. In ATK
  that is Setup → System & Resources → Advanced, consistent with how every
  other consequential toggle there is gated.
- Auto-apply is only safe *because* snapshots and transactional undo exist.
  **Do not remove them to "simplify".** If you ever find yourself deleting the
  snapshot step, delete the auto-apply option first.

#### 6.5a `textio.py` — encoding and line endings, stated once

This is Windows-first software and the model emits `\n`. Without an explicit
policy, anchored edits fail mysteriously on CRLF files, or — worse — a
whole-file write silently converts every line ending in a file the task
touched one function of. The policy:

- On read: detect encoding (UTF-8 with/without BOM; fall back with the
  assumption *stated* in the journal) and dominant EOL style per file.
- **Normalise to `\n` internally.** All anchor matching, diffing, hashing
  (I4 normalisation included) happens on normalised text.
- **On write: restore the file's original encoding, BOM, and EOL style.** New
  files take the project's dominant style, else the platform default.
- The snapshot stores the original **bytes**, so undo is byte-identical by
  construction — which is what the §9 property test asserts.

**Acceptance:** an anchored edit against a CRLF file with a BOM applies
cleanly, changes only the target lines, and undo restores the exact original
bytes.

#### 6.5b Why bespoke snapshots and not git — stated so nobody "fixes" it

Most target projects will be git repos; this engine still ships its own
snapshot/undo machinery, deliberately: it must work on non-repos, it must not
depend on a `git` binary, and it must never touch the operator's history,
stash, or index — a tool that quietly makes commits in someone's repo is
indistinguishable from a mess. The interaction rules:

- If the project is a git repo with uncommitted changes, say so once at
  session start (`warning` event): the operator may want a clean state first.
  Do not refuse; do not commit for them.
- The engine never runs `git`. Generated code that tries is blocked (§6.3).
- `.git/` is excluded from indexing, listing, and patching.

### 6.6 `context.py` — what the model gets to see

Outline before body; slices not files; a hard budget; and an explicit statement
of what was omitted.

- `outline(text, lang)` — every symbol with its signature and line. Exact for
  Python via `ast`; regex elsewhere, labelled as approximate.
- `slice_around(text, line, before, after)` — with line numbers, so an edit can
  be located.
- `build_context(pieces, budget)` — priority-ordered, and the returned string
  ends with a `NOT INCLUDED` block naming what was dropped.

**The omission notice is not politeness.** A model that doesn't know something
was withheld will invent its contents.

The budget itself is **measured at session start, not taken from config** —
see Appendix G.3: usable prompt = `n_ctx` minus reserved output minus the
reasoning tax minus a safety margin.

### 6.7 `codemap/` — architectural Code-RAG

This is the Module 6 addendum, promoted and refined.

**Parsers.** `parse_python.py` uses `ast` (exact). `parse_regex.py` is a
zero-dependency ctags-style signature extractor for everything else, and its
output is **labelled approximate everywhere it surfaces**. `parse_treesitter.py`
is used *if* `tree_sitter` and the relevant grammar are importable — it is
strictly better than regex for C/C++/Rust/JS, and strictly optional (C7).

**Store.** SQLite at a path from `StoragePort.sqlite_path("codemap")`. Schema:

```sql
files(id, path, lang, mtime, hash, indexed_at)
symbols(id, file_id, name, kind, line, end_line, signature, docstring,
        parent_id)
edges(src_symbol_id, dst_symbol_id, kind)   -- calls | imports | contains
unresolved(src_symbol_id, name, kind)        -- calls we couldn't bind
```

`unresolved` matters: a call graph that silently drops what it couldn't resolve
looks complete and isn't. Report the resolution rate.

**Blast radius.** `callers_of(symbol)` transitively, with a depth limit. On a
signature change: queue callers for refactor **and** select the tests that
cover them to run first.

**Semantic zoom.** `generate_architecture_context(target_file, max_tokens)`:

- *Immediate dependencies* (direct imports of the target) → full signatures,
  parameters, docstrings.
- *Distant architecture* → class names and paths only, or omitted.
- Budget enforced with `LLMPort.count_tokens`; when that is an estimate, say so
  in the injected block.
- Ends with the omission list (§6.6).

**Freshness — the rule, restated precisely.** Re-index a file the moment it is
written; store `hash` and `mtime` so a full rescan is cheap and incremental.
The obligation splits in two: **the query interface (the tools below) must
never be stale** — it reads live SQLite; **the injected text summary may lag
deliberately** (it is a cached prompt prefix), provided the model is told it
lags and can always call the tool. The full cache design — epochs, the
zoom/cache split, the staleness note — is Appendix G.7, and it is part of this
module's acceptance.

**Agentic lookup — native tools first.** The target model has trained tool
calling (Appendix G.1), so mid-generation lookup is a set of real tools with
JSON schemas:

```
search_codemap(name)         → signatures for a symbol or module
read_slice(path, start, end) → a region of a file
list_symbols(path)           → the outline
run_tests(pattern)           → run a subset, return parsed diagnostics
apply_patch(path, old, new)  → anchored edit — routes through the
                               transaction + approval gate (§6.5 rule 6)
```

Three reasons this beats a text marker: the model was tuned for exactly this
shape; the output is structurally parseable rather than regex-scraped; and a
schema constrains the arguments in a way prose never can.

**The text-marker fallback**, for hosts whose model lacks tool support
(`supports_tools=False`): lenient parser accepting `[SEARCH_CODEMAP: x]`,
`SEARCH_CODEMAP(x)`, `<search_codemap>x</search_codemap>`; hard cap of 3
lookups per generation; on exceeding the cap, inject "you have used your
lookups, work with what you have"; on a malformed call, respond with the exact
correct syntax once, and only once. **The fallback path also forces
epoch-per-write** (Appendix G.7) — without live tools, a lagging summary has
no safety net, so it is not allowed to lag.

**Acceptance:** index this project's own source and assert (a) every public
function appears, (b) `callers_of` finds a known caller across files, (c) a
signature change flags the right files, (d) the context for a chosen file fits
a 4,096-token budget and names what it dropped.

### 6.8 `planner.py` — skeleton-first decomposition

Per §4.2. `plan(request, host_facts)` returns a `Plan` of `Task`s. A `Task`
carries `path`, `test_path`, `purpose`, `persona`, `depends_on`, `atomic`.
Two points that were implicit in v1.0 and are now explicit:

- **Every implementation task names its test file.** Test-first (F2) is only
  possible if the planner pairs `src/stats.py` with `tests/test_stats.py` at
  planning time; tests are planned artefacts, not afterthoughts.
- **`atomic` flows from the plan into the patcher transaction** (§6.5 rule 1).
  The planner is the only component that knows whether two files are one
  change.

Dependencies are **derived from the generated skeleton's imports**, not
asserted by the model.

Include `replan(plan, journal)` — after each file, if the codemap shows the
shape changed materially, revise the remaining tasks. A plan that cannot change
is a plan that will be wrong by file five. A replan is an epoch boundary
(Appendix G.7).

### 6.9 `loop.py` — the engine

```
context → generate ──(tool round-trips: search_codemap, read_slice…)──┐
   ↑          │                                                        │
   │          ▼                                                        │
   │  truncation check (D1: finish_reason=="length" ⇒ continue,        │
   │                    don't regenerate)                              │
   │          ▼                                                        │
   │  guard → syntax pre-check → deterministic pre-fixes (F1)          │
   │          ▼                                                        │
   │       build → run → test                                          │
   │          │                                                        │
   └── parsed diagnostics (max 3; cascade languages: first error only, │
       F7) ◄───────────────────────────────────────────────────────────┘
```

- Tool round-trips during generation are ordinary `complete()` cycles
  (§5.3), not a halt-parse-resume dance.
- **Deterministic pre-fixes run before the model sees an error** (F1): an
  insertable import, a `--fix`-able lint rule, a formatter pass. Every error
  fixed by a rule is minutes of generation not spent.
- Attempt budget (default 4).
- **Stagnation and cycle detection.** Hashing diagnostics alone misses the
  ping-pong: fix A introduces error B, fix B reintroduces error A, and every
  attempt has a *different* diagnostic hash, so a naive detector concludes
  progress is being made while the loop runs forever. Track a pair:

  ```python
  signature = (sha256(normalised_code), sha256(sorted_diagnostic_keys))
  ```

  - **Identical code twice** → hard stagnation. The model is not changing
    anything; more attempts cannot help. Narrow the unit of work (F5) or stop.
  - **Identical diagnostics twice** → change strategy: widen context, shrink
    the unit, or stop and hand back.
  - **Any signature seen before** → a **cycle**. Keep a set of every signature
    in the task; a repeat is an immediate stop. This catches 2-cycles and
    also the 3- and 4-cycles no pairwise comparison finds.
  - **No forward progress**: error count fails to decrease across three
    attempts, even with all-different signatures. Slow oscillation looks like
    work.
  - **Cosmetic churn**: consecutive attempts differing by under ~2% of
    characters are the model rearranging whitespace. Treat as identical code.

  Normalise before hashing — `textio` normalisation plus comment stripping and
  whitespace collapse — or a reformat registers as a change and defeats the
  whole detector. **On stopping, report the cycle**, not just the failure:
  *"attempts 2 and 4 produced the same code and the same two errors; it is
  alternating between a missing import and an unused import."* That sentence
  tells the operator exactly what is wrong, and it is usually a two-second
  fix by hand.
- Every attempt is journaled (C8). The cancel token is checked at every phase
  boundary (§5.2).
- On give-up, the result states *what it tried* and *the last real error* —
  not "failed after 4 attempts".

### 6.10 `review.py` — deterministic then adversarial

Order: run every available scanner and metric → then **one structured model
review pass** covering security and performance with a schema'd output (§4.3)
→ then the Synthesizer.

Built-in deterministic checks (no dependencies): hardcoded secrets/API keys,
`eval`/`exec` on non-literals, path traversal patterns, empty exception
handlers, TODO/FIXME left in "finished" code, functions over a length
threshold, and missing tests for new public functions.

The Synthesizer produces the Recommendation Document (structure preserved from
FORGE §5.3): Executive Summary, Quality Assessment, Vulnerabilities & Fixes,
Plain-English Deployment Guide tailored to the user's skill level.

**Honesty requirement:** if the security and performance perspectives came
from the same model, the document says so, in one line, near the top.

### 6.11 `personas.py` — prompts and Theory of Mind

Keep FORGE's `[SELF-MODEL] / [USER-MODEL] / [DIRECTIVE]` structure — it is a
good structure and it makes prompts diffable.

Keep the skill levels (Novice / Intermediate / Senior) and the jargon
translation levels. They are genuinely useful and cheap.

**Layer on the model's own system prompt, don't replace it.** Devstral ships
`CHAT_SYSTEM_PROMPT.txt`; the model was tuned with it, and replacing it
wholesale discards that tuning:

```
1. Devstral's shipped system prompt          (its trained behaviour)
2. + project conventions and style           (F3)
3. + [EXECUTION CONSTRAINTS] block            (codemap, per §6.7)
4. + the output contract                      (§4.4 — still required)
```

**Mandatory (§4.4):**

- Every prompt whose output is machine-consumed ends with an **OUTPUT
  CONTRACT**: exactly what to emit, and the headings/preambles forbidden by
  name.
- `personas.detect_commentary(text) -> bool` and `strip_commentary(text) -> str`
  live here, and every call site that consumes model output uses them. The
  detector must require *decoration* (`**Why?**`, `## Rationale`) rather than
  bare words, so a legitimate reply containing "why" is not mangled.
- Prompts must not name the internal machinery they don't want echoed. Models
  repeat prompt vocabulary.
- Reasoning-model output (`<think>` blocks) is stripped before any use (D13);
  ATK's `split_think()` is the reference implementation.

Prompt **ordering** is a performance contract, not a style choice — the
stable-first/volatile-last layout and the byte-identical-prefix requirement
are Appendix G.7, and the prefix-stability test (§9) enforces them.

### 6.12 `providers/` — local and remote

**Local (always available):**
- `local_llamacpp` — via a host-supplied `LLMPort`; usually the host already
  has a model loaded and just wraps it.
- `openai_compatible` — one adapter covers llama.cpp server, Ollama, LM Studio,
  vLLM and LiteLLM. This is the highest-value provider to write: it is local,
  and it is what most self-hosters actually run.

**Remote (opt-in, off by default — C3):** Anthropic, Google Gemini, Mistral,
OpenRouter, OpenAI.

> Note for the implementer: include OpenAI in the provider list for the benefit
> of users who want it. The project owner does not use it and this is a settled
> preference — implement it, don't advocate it, and don't make it a default
> anywhere.

**Every remote provider MUST:**

1. Be **disabled unless explicitly enabled for the session**. No env-var
   auto-detection that quietly turns the network on.
2. Route **every outbound message** through `redact.py` first — the initial
   payload, every subsequent turn, and **every tool-result message in a tool
   round-trip** (a file slice returned to a remote model is outbound context
   like any other). API keys, tokens, private keys, `.env` contents,
   connection strings, and anything matching the host's configured secret
   patterns are replaced with `[REDACTED:kind]`, and the count of redactions
   is reported.
3. Call `ApprovalPort.approve_remote(provider, bytes_out, estimate)` before the
   first call of a session, showing what is about to leave the machine.
4. Enforce a **budget**: max tokens per session and max spend, from config;
   when exceeded, stop and report — never silently continue.
5. Record provider, model, token counts and (where the API reports it) cost, in
   the journal.
6. Surface a persistent "REMOTE MODE — data leaves this machine" indicator via
   `EventPort` for the host to display.

Keys come from the host (`StoragePort`), or from environment variables the
operator sets deliberately. **Keys are never written to the journal, the
codemap, or any log.**

**Acceptance for `redact.py`:** a fixture project seeded with a fake AWS key,
a private key block, a `.env`, and a connection string; every outbound payload
in a scripted remote session is captured and asserted clean, and the redaction
count matches.

### 6.13 `session.py` + `journal.py` — orchestration, provenance, resume

`Session` is the object a host drives: `start(request, profile)`, `step()`,
`resume(id)`, `cancel()` (§5.2). It owns the plan, the loop, the codemap
lifecycle, the epoch state (Appendix G.7), and the journal. It re-reads
`LLMPort.capabilities()` at every task boundary (§0.1).

`journal.py` writes an append-only JSONL per session:

```json
{"t":"2026-08-06T09:14:02Z","event":"generate","task":"src/parser.py",
 "attempt":2,"provider":"local","model":"devstral-small-2-24b-q4_k_m",
 "prompt_sha256":"…","temperature":0.15,"seed":11,
 "tokens_in":3182,"tokens_out":880,"prompt_ms":2900,
 "verify":{"build":"ok","test":"failed","diagnostics":2}}
```

`prompt_ms` — prompt-processing time per call — is required: it is the only
signal that the prefix cache broke (Appendix G.7.5).

**Resume** is derived from the journal plus the codemap, not from an in-memory
object — so it survives a crash, not just a pause.

Journal event kinds and their required fields are enumerated in `types.py`
and documented in PORTS.md; they are public API (a host renders history from
them) and follow the same semver rules as §5.4.

---

## 7. Integration: ATK

**ATK is a real, substantial, working application; the integration must not
disturb it.**

### 7.1 Conflicts to avoid — checked against the current ATK tree

| Risk | Detail | Resolution |
|---|---|---|
| **Module name collisions** | ATK already has `atk/core/sandbox.py`, `debug_loop.py`, `langs.py`, `diagnostics.py`, `codeguard.py`, `coderun.py`, `patcher.py`, `codectx.py`, `pdw_plot.py`, `agc.py`. | Cognitive Coder lives in its own top-level package. The ATK adapter is `atk/core/ccoder_host.py` + `atk/ui/ccoder_panel.py`. **No file named `forge_*` and no new `atk/core/` module that shadows an existing name.** |
| **Duplication with work already done** | The six modules listed in §6 already exist inside ATK and overlap CC's remit almost exactly. Two parallel implementations would diverge within a month. | **Cognitive Coder becomes the owner.** ATK's copies are deleted and `atk/core/*` re-exports from `cognitive_coder` for backwards compatibility. Do this in one commit, with the test suite green before and after. See A.4. |
| **VRAM ceiling & model swapping** | ATK enforces 16 GB; the cognitive core is **mutually exclusive with Whisper**, and only one LLM is loaded at a time. | **Swapping is ATK's, full stop** (§0.1): a button that unloads one model and loads the other, using ATK's existing `vram.py`/`gguf_meta.py` logic. CC requests inference through `LLMPort`, re-reads `capabilities()` at task boundaries, treats a model change as an epoch boundary, and MUST handle "no model loaded" as a normal, reportable state — not an exception. The adapter surfaces ATK's existing "unload Whisper first" guidance. |
| **Zero telemetry** | ATK's core promise. | C3. In the ATK adapter, remote providers are **off, and the enable control sits behind ATK's Setup → System & Resources with an explicit warning**, consistent with how ATK gates every other subsystem. |
| **Sandbox semantics differ** | ATK's `sandbox.py` blacklists `subprocess` *in generated code* — but CC must run compilers *as the host*. Conflating those would make CC unable to build anything. | The ATK `ExecPort` implementation runs build tools with ATK's scrubbed-environment approach; the *generated code* is still screened by `guard.py`. Two different things, kept apart. |
| **SQLite** | ATK has `state_db.py` and a project DB. A second DB file is fine; a second *schema in the same file* is not. | `StoragePort.sqlite_path("codemap")` returns a separate file under ATK's data dir. Never write to ATK's `state.db`. |
| **Settings shape** | ATK settings is a nested dict persisted by the app. | The adapter maps `StoragePort` onto `ctx.settings["ccoder"]`. CC never reads ATK settings directly. |
| **Threading** | ATK runs work on `QRunnable` via `atk/core/workers.py`; the GUI thread must never block. | CC's core is synchronous and must remain so. **Never** call a Port from a thread the host didn't hand you. The adapter wraps `Session.step()` in a worker and marshals `EventPort` calls back to the GUI thread. `Session.cancel()` is the one cross-thread call (§5.2), and it is how ATK's Stop button works. |
| **UI doctrine** | ATK's rule (D4 in its plan): one tab per function, no pop-ups to go hunting for. | The ATK panel is a workspace tab with sub-tabs, not dialogs. The one justified modal is the diff approval — and only if ATK's owner wants approval at all; he has chosen auto-apply with undo. |
| **Detachability** | ATK panes can be popped to another monitor (`atk/ui/detach.py`). | The CC panel's console, diff view and CodeMap tree should be registered as detachable panes. |
| **Test style** | ATK uses no pytest: plain `python tests/test_x.py` scripts with a `check(name, cond)` helper printing PASS/FAIL and a final count. | CC's own suite may use pytest (it's a library, and ParisNeo will expect it). The **ATK adapter's** tests follow ATK's style. Keep them separate. |

### 7.2 What the ATK adapter contains

```
atk/core/ccoder_host.py     # Port implementations bound to ATK
atk/ui/ccoder_panel.py      # workspace panel (replaces the Developer Sandbox
                            # sub-tab's engine; keeps the tab)
```

`ccoder_host.py` implements:

- `LLMPort` → `atk.core.llm_engine.LLMEngine` (`chat_stream`, GBNF grammar
  support, `split_think` for reasoning models, ATK's per-family chat
  templates, and — since ATK installs `mistral-common` — **exact token
  counting**, `token_count_is_estimate=False`).
- `FileSystemPort` → project root or `SANDBOX_DIR` per mode.
- `ExecPort` → subprocess with ATK's scrubbed env and Windows tree-kill
  (Job Objects or `taskkill /T`).
- `StoragePort` → `ctx.settings["ccoder"]` + ATK's `DATA_DIR`.
- `EventPort` → `ctx.set_status` + the panel's console + ATK's Cognitive Flow
  view.
- `ApprovalPort` → auto-approve for diffs (owner's choice), real prompt for
  remote enablement.

**The panel also gets "Attach a screenshot" on any coding task.** Devstral is
multimodal, and the owner habitually debugs by screenshot — the clipped Setup
page, the sliced RF rail, the `(x0.001)` axis were all caught that way and
none would have been caught by reading code. A coding agent that can be handed
a screenshot of a broken UI and the code that renders it is a genuinely
different tool. It routes through `Message.images` (§5.1); hosts without
vision ignore it and say so. Obvious uses: "this dialog looks wrong, here's
the screenshot and the layout code"; an axis label wrong in a plot — exactly
the class of bug invisible in source; a photo of a whiteboard sketch as the
spec for a new panel.

### 7.3 Migration of the six existing modules

Ordered, and each step leaves the suite green:

1. Copy the six modules into `cognitive_coder/`, renaming
   `codeguard→guard`, `coderun→runner`, `codectx→context`.
2. Replace their direct `atk.config` / `atk.core.dsp` imports with Port calls.
3. In ATK, replace each module's body with
   `from cognitive_coder.X import *  # noqa` plus a one-line comment saying
   where it moved and why.
4. Run ATK's full suite. It must stay green — those modules have ~200 checks
   between them.
5. Only then delete the re-export shims, updating ATK's imports.

---

## 8. Other hosts — and the dependency rule that protects ATK

**ATK does not use LoLLMs and must never depend on it.** The owner's reason is
concrete: LoLLMs ships breaking changes without warning, and an offline field
tool cannot inherit somebody else's release cadence.

That is an argument FOR the port architecture, not against it. Note the
direction of dependency:

```
   LoLLMs  ──depends on──►  Cognitive Coder  ◄──depends on──  ATK
                                    │
                             depends on nothing
```

Cognitive Coder never imports LoLLMs. LoLLMs (if ParisNeo wants it) imports
Cognitive Coder. A breaking change on his side cannot reach ATK, because
nothing in ATK's dependency chain points at him. Keep it that way:

- **A CI test MUST assert that no file under `cognitive_coder/**` imports
  `lollms`, `lollms_client`, or any host package.** Same test as C1.
- **Do not build the LoLLMs adapter as part of this project.** It is removed
  from the phase plan. Hand ParisNeo the specification, the port conformance
  kit (§9), and `examples/tiny_host.py`, and let him decide what he wants; he
  knows his own architecture better than a spec written from outside it.
- If he does build one, the natural surfaces are his **MCP plugin** system
  (documented as covering "code execution, file manipulation" — exactly this
  module's shape) and his **binding** system, where a ~100-line `LLMPort`
  implementation over `lollms_client` would give Cognitive Coder every backend
  he already supports: ollama, vllm, llamacpp, gemini, openrouter, exllamav2
  and the rest. Mention this to him as an option; do not build it here.

**What the ports still buy ATK, with no other host in the picture:**

1. **Testability.** `ScriptedLLM` and `MemoryFileSystem` mean the whole engine
   is exercised with no model and no disk. That alone justifies them.
2. **The model-swap hook is trivial** — a different model behind `LLMPort` is
   a different `capabilities()` answer, not a different engine.
3. **ATK's own boundaries stay clean.** The engine cannot accidentally reach
   into `ctx.settings` or write outside the project, because it has no way to.
4. **The gift to ParisNeo costs nothing extra.** The architecture that makes it
   testable is the same one that makes it portable.

## 9. Testing contract

- **Fakes, not mocks.** `MemoryFileSystem`, a `ScriptedLLM` that returns
  canned responses in order, and a `RecordingEvents`. The whole engine must be
  drivable with zero real models and zero network.
- **Golden diagnostics.** Real captured toolchain output committed as fixtures.
- **The worked session is a golden trace, not just prose.** Appendix E is
  committed as `tests/fixtures/worked_session.jsonl` — the expected journal
  event sequence for the reference task on a `ScriptedLLM`. The phase-5
  acceptance diffs the actual journal against it (shapes and ordering, not
  timestamps). A spec example nobody executes drifts; a fixture cannot.
- **The port conformance kit.** `tests/port_conformance.py` is a reusable
  suite any host runs against *its own* Port implementations: write-then-read
  round-trips, atomicity where claimed, tree-kill on timeout, jail
  enforcement, capabilities honesty (`token_count_is_estimate`,
  `supports_tools`). This is how ParisNeo validates an adapter without
  reading the core, and how the docstring guarantees of §5 become executable.
- **Toolchain-conditional tests.** Skip with a printed note when a compiler is
  absent; never fail because a machine lacks Rust.
- **A no-network test that fails if any socket opens** during a local-only
  run. Mechanism, so the test actually catches something: monkeypatch
  `socket.socket` (and `socket.create_connection`) at the start of the run to
  raise, then drive a full scripted session. This is C3's enforcement.
- **A "no host imports" test**: walk `cognitive_coder/**` and assert nothing
  imports PySide6, FastAPI, `atk`, or `lollms`. This is C1's enforcement.
- **The prefix-stability test**: build the prompt twice for the same
  (persona, epoch, target) and assert the prefixes are byte-identical up to
  the cache boundary (Appendix G.7.5). This test is the only thing standing
  between a working cache design and a silently 20×-slower one six months
  from now.
- **Property test on `patcher`**: apply then undo must restore byte-identical
  content, for a corpus of random edits — including CRLF and BOM files
  (§6.5a).
- Target ≥85% line coverage on the core; the parsers and the loop should be
  near 100% because they are where wrongness hides.

---

## 10. Distribution, installers and repo hygiene

**Clone and run an installer. No PyPI.** The installers are therefore part of
the product, not an afterthought — for many users they are the entire first
impression.

### 10.1 The installer contract

Both `install.bat` (Windows) and `install.sh` (Linux) MUST satisfy all of the
following. This contract is lifted from ATK, where it was arrived at painfully,
and it is worth inheriting wholesale.

1. **Fully non-interactive.** No prompts, no "press any key", nothing that
   waits for a human. A user who walks away must come back to a finished
   install, not a question. (In ATK this rule exists because a third-party
   installer's hidden prompt stalled the whole run.)
2. **No manual file extraction, ever.** If something must be downloaded, the
   installer downloads and unpacks it. Never instruct a human to fetch a zip.
3. **Idempotent.** Running it twice is safe and re-fetches only what is
   missing. This is how a user recovers from a partial install.
4. **Nothing global is changed.** A virtual environment inside the clone
   (`.venv/`), no PATH edits, no registry writes, no system packages.
5. **Every optional component degrades.** A failed optional download disables
   one feature and says which; it never aborts the install.
6. **It ends with a summary** listing what landed and what did not, with one
   line per item saying what a missing item costs:

   ```
   ============================================================
    Cognitive Coder — installation summary
   ============================================================
     [OK] core engine          .venv ready, 0 required deps
     [OK] Python toolchain     python 3.11.9
     [OK] C/C++ toolchain      gcc 13.2.0
     [--] Rust toolchain       not found — Rust targets unavailable
     [--] tree-sitter          not installed — C/C++/Rust outlines will be
                               regex-approximate rather than parsed
     [--] remote providers     not installed (offline by default; run
                               install.bat /providers to add them)

     [--] entries are OPTIONAL: each disables one feature, not the tool.
     Re-run the installer to retry only what is missing.
   ```
7. **Exit code is meaningful.** 0 if the core engine is usable, non-zero only
   if it is not. A missing optional toolchain is not a failure.
8. **Windows batch gotcha, stated because it has bitten this author before:**
   inside a parenthesised block, `echo (text)` terminates the block early.
   Avoid parentheses in echoed text, or escape them (`^(`).

### 10.2 What each installer does

```
1. find Python 3.11+. If none is usable → FETCH 3.11 (§10.2a), don't fail
2. create .venv in the clone, from THAT interpreter
3. pip install -e .            (core: zero required runtime dependencies)
4. detect toolchains: python, gcc/clang/cl, rustc, javac, go, node, dotnet,
   zig, lua, ruby, sqlite3, godot — recorded for the summary; langs.py probes
   again at runtime (§6.1), so this record is informational
5. optional, only on an explicit flag:
     /providers  or  --providers   → install remote-provider SDKs
     /treesitter or  --treesitter  → install tree_sitter + grammars
     /dev        or  --dev         → ruff, pytest, coverage
6. run the self-test:  .venv/bin/ccoder doctor
7. print the summary
```

### 10.2a Python: 3.11 is the floor and the fetched version — and what a venv is not

**Settled 2026-08-06, amended in v1.1: the code targets Python 3.11 features;
the installer fetches 3.11 when the machine has nothing suitable; but the
package does NOT cap the interpreter version.** The v1.0 cap
(`<3.12`) contradicted library-first: an embedded library does not control
its host's interpreter, and LoLLMs or a contributor on 3.12/3.13 must be able
to `pip install -e .` — code written for 3.11 runs fine there. So:

- `pyproject.toml` declares `requires-python = ">=3.11"`. No upper bound.
- **CI runs 3.11 only**, on Windows and Linux, and the README says exactly
  that: *tested on 3.11; expected to work on later versions; earlier versions
  are refused at install.*
- The installer still standardises the *bundled* environment on 3.11, so
  every installer-built machine is identical.

What 3.11 unlocks, and should therefore be used without hesitation:
`tomllib` (config with no dependency), `ExceptionGroup` / `except*` (useful
for a loop that runs several verifications), `typing.Self`, `enum.StrEnum`,
and noticeably faster startup — which matters for a CLI invoked repeatedly.

One correction to state plainly, because it is a common and expensive
misunderstanding: **a virtual environment does not give a project its own
Python.** A venv isolates *packages*; the interpreter is whatever created it.
`python -m venv .venv` on a machine whose `python` is 3.9 produces a 3.9 venv,
and Cognitive Coder will fail inside it at the first `match` statement or
`X | None` annotation — with a syntax error, which looks like broken code
rather than a wrong interpreter.

So the installer does this, in order, and **reports which branch it took**:

```
1. probe, in order: py -3.11, python3.11,
                    then bare `python` / `python3` — and CHECK --version,
                    because `python` is 3.9 on more machines than you expect
2. if 3.11+ is found              → build .venv from it.  DONE.
3. if not, fetch a standalone CPython 3.11 into .python/ inside the clone,
   then build .venv from that.  DONE.  (This is the NORMAL path on an old
   machine — not an error, not a warning.)
4. only if the fetch is impossible (no network, no cached copy)
   → fail with a sentence naming what was found and what is needed.
```

**How to fetch, in preference order:**

- **`uv`** (Astral) — a single static binary that can install a specific CPython
  and create the venv: `uv python install 3.11` then `uv venv --python 3.11`.
  It is by far the least code, works identically on Windows and Linux, and is
  fast. Fetch `uv` itself first (single file, no installer), keep it in
  `.tools/`, and never touch the system Python.
- **python-build-standalone** (the relocatable CPython builds `uv` itself uses)
  — download the tarball for the platform and unpack into `.python/`. Use this
  if you would rather not depend on `uv`.
- **Windows embeddable Python** (`python-3.11.x-embed-amd64.zip`) is small
  (~10 MB) and tempting, but it is *stripped*: no `pip`, no `venv`, no
  `tkinter`, and `._pth` path handling that surprises people. Only reach for it
  if the two above are unavailable, and expect to run `get-pip.py`.

**Rules that keep this honest:**

- Everything lands **inside the clone** (`.python/`, `.tools/`, `.venv/`).
  Nothing is installed system-wide, no PATH is modified, and deleting the
  clone removes every trace — the same promise as §10.1 rule 4.
- The fetch respects §10.1 rule 1: it is non-interactive, and rule 5: if it
  fails, the installer says exactly what it tried and what to install by hand.
- `ccoder doctor` prints **which interpreter is in use and where it came
  from** — system, fetched, or vendored. When something behaves oddly six
  months from now, that line is the first question answered.

`ccoder doctor` is a required deliverable, not a nicety: it prints the same
summary on demand, so a user diagnosing a problem three months later has one
command to run.

### 10.3 Dependencies

`pyproject.toml` declares **zero required runtime dependencies** beyond the
standard library. Everything else is an extra:
`[llamacpp]`, `[anthropic]`, `[google]`, `[mistral]`, `[openai]`,
`[treesitter]`, `[dev]`, `[all]`.

A stdlib-only core is what makes this embeddable in a Qt desktop app and a
FastAPI server without an argument about versions. Protect it: a CI check
should fail the build if `cognitive_coder/**` imports anything not in the
standard library at module level.

**Where tokenizers live, stated so nobody breaks the rule:** exact token
counting (e.g. `mistral-common`, which ATK already installs) belongs to the
**host or provider implementing `LLMPort.count_tokens`** — never to the core.
The core calls the port and honours `token_count_is_estimate`; that is the
whole arrangement.

### 10.4 The rest

- **License: Apache-2.0. SETTLED 2026-08-06.** Permissive enough for anyone to
  vendor, and unlike MIT it carries an explicit patent grant — worth having for
  a tool that generates code. Ship `LICENSE` in the first commit and put the
  SPDX identifier (`# SPDX-License-Identifier: Apache-2.0`) at the top of every
  source file, so a vendored copy stays attributable.
- **Semver, and the public API is `cognitive_coder/__init__.py` only** — which
  re-exports the Ports and the shared types of §5; those are the frozen
  surface (C9). Tag releases in git; without PyPI, the tag *is* the release.
  `version.py` is the single source of truth and is read without package
  metadata.
- `CHANGELOG.md` from the first commit.
- CI: ruff, the no-host-imports test, the no-network test, the stdlib-only
  check, the prefix-stability test, and the suite on **Windows and Linux
  both** — Windows matters, ATK is Windows-first, and path handling is where
  cross-platform code dies.
- `README.md` opens with what it is, what it needs, `git clone` + the
  installer, and a ten-line embedding example using the Null ports so it runs
  anywhere. It also carries the meta-lesson (Appendix D): with a frontier
  model you improve results by improving the prompt; with a small model you
  improve results by improving the loop.
- `docs/EMBEDDING.md` must include a **vendoring** section: how to use the repo
  as a git submodule with no install at all.

---

## 11. Phase plan

*Reordered in v1.1: `codemap` now precedes `loop`, because the loop's
acceptance trace (Appendix E) uses codemap injection and re-indexing — the
v1.0 ordering asked phase 4 to demonstrate a phase-5 feature.*

| Phase | Contents | Done when |
|---|---|---|
| **0** | `ports.py`, `types.py`, Null implementations, `tiny_host.py`, repo skeleton, CI | `pytest` green on an empty engine; no-host-imports test passes; tiny_host runs |
| **0b** | `install.bat`, `install.sh`, `ccoder doctor` | A clean clone installs non-interactively on Windows AND Linux and prints an honest summary. Do this EARLY — it is how every other phase gets tested on a fresh machine, and leaving it to the end means discovering install problems when you can least afford them. |
| **1** | `langs`, `diagnostics`, `guard`, `runner` (ported from ATK) | A broken C file yields a located build error; a working one runs |
| **2** | `patcher` (+ transactions, `textio`), `context`, journal | Apply/undo property test passes, including CRLF/BOM corpus |
| **3** | `providers/openai_compatible` + `local` | Same engine drives Ollama and a llama.cpp server |
| **4** | `codemap` (parsers, store, zoom, blast radius, tools) | Indexes itself; context fits a 4k budget and declares omissions |
| **5** | `loop` + `personas` | End-to-end: "write a CSV parser with tests" builds and passes on Devstral Small 2 at the G.8 settings, unattended — and the journal diffs clean against the Appendix E golden trace on a ScriptedLLM |
| **6** | `planner` (skeleton-first) + `session` resume | A 5-file project builds; killed at file 3, resumes correctly |
| **7** | `review` (deterministic + single structured model pass) + Recommendation Document | Finds a planted secret and a planted O(n²) |
| **8** | Remote providers + `redact` + budgets | No-network test still passes with providers present but disabled; redaction fixture asserts clean payloads |
| **9** | ATK adapter + migration of the six modules | ATK's full suite green before and after |

Phases 0–5 are the minimum viable engine. If time runs out, that is the
shippable subset.

---

## Appendix A — facts about ATK a fresh session cannot guess

### A.1 What ATK is

Analyst Toolkit: an offline, zero-telemetry intelligence workbench for Windows
10/11 (PySide6 + llama.cpp + SDR hardware), built for aid workers, NGO
volunteers and medics working where there is no reliable connection and no
reason to trust one. Workspaces: Chat & Cowork, Analysis Suite, Network Link,
Geospatial, RF/Radio, Air Picture, ATHENA, Setup & Config.

### A.2 Hard constraints

- **16 GB VRAM ceiling.** The cognitive core (a ~24B GGUF) and Whisper are
  **mutually exclusive**, and only one LLM is loaded at a time; swapping is a
  manual host action (§0.1).
- **Zero telemetry.** Nothing leaves the machine without explicit action.
- **Portable install.** Drop-an-exe-in-`bin\`; no multi-gigabyte environments.
  `install.bat` is deliberately non-interactive end to end.
- Python 3.11+, PySide6, no Docker.

### A.3 Doctrine the codebase follows (adopt it — it will make review easier)

1. **Deterministic first, model second, human last.**
2. **Honest failure.** UNKNOWN and AMBIGUOUS are real answers. A wrong name
   stops a search; "I don't know" is actionable.
3. **Say what was omitted** — thinned plots, truncated reads, dropped context.
4. **One tab per function, no pop-ups** to go hunting for.
5. **Comments explain WHY**, especially where an obvious approach was rejected.
6. **Tests assert the reasoning, not just the result** — including tests that
   assert honest wording is present in the UI.

### A.4 The six modules to port, and their gotchas

In `ATK/atk/core/`: `langs.py`, `diagnostics.py`, `codeguard.py`, `coderun.py`,
`patcher.py`, `codectx.py`. All Qt-free; only `coderun.py` and `codectx.py`
import anything from ATK (`atk.core.dsp` is *not* among them — check imports at
port time). Hard-won details already baked in, which a rewrite would lose:

- `diagnostics`: rustc's message and location are on separate lines and must be
  paired in order; Python's deepest frame is last, JavaScript's is first;
  unparsed output must never yield an empty list.
- `patcher`: ambiguous anchors are refused, and undo restores a whole apply.
  (v1.1 layers the explicit transaction model of §6.5 on top.)
- `context`: NaN-vs-omitted distinction, and the declared-omissions block.
- `langs`: argv lists not strings; scaffolds that actually build.

### A.5 Related ATK subsystems worth knowing about

- `atk/core/llm_engine.py` — GGUF loading, streaming, **GBNF grammar-constrained
  JSON**, `split_think()` for reasoning models. This is what `LLMPort` binds to.
- `atk/core/workers.py` — `QRunnable` + `submit()`; all long work goes here.
- `atk/core/sandbox.py` — the existing Python-only execution jail. **Do not
  extend it**; CC's `guard.py` + `runner.py` supersede it, and the old jail
  stays for the legacy Developer Sandbox until the panel is replaced.
- `atk/ui/detach.py` — pane detaching; register CC's panes with it.
- `atk/core/vram.py`, `gguf_meta.py` — VRAM budgeting and offline GGUF header
  parsing. This is the machinery behind the host's model-swap button (§0.1).
- `FUTURE_PLANS.md` — the reasoning log for the whole project. Read the entries
  on the coding workbench before starting; they cover ground this spec
  summarises.

---

## Appendix B — open questions for the owner

1. ~~License.~~ **Closed 2026-08-06: Apache-2.0.**
2. ~~Approval mode default.~~ **Closed 2026-08-06:** the library default is
   approval-required; auto-apply is opt-in behind an advanced setting with a
   warning (§6.5).
3. ~~GDScript support level.~~ **Closed 2026-08-06: FIRST CLASS.** See §6.1a.
4. ~~Minimum Python.~~ **Closed 2026-08-06, amended in v1.1:** 3.11 floor, no
   upper cap; the installer fetches 3.11 when the machine hasn't got a usable
   Python. See §10.2a.
5. ~~Does ParisNeo want the MCP plugin or the personality?~~ **Closed
   2026-08-06:** not our call and not our build. He gets the document and
   decides for himself.
6. ~~Second-model strategy.~~ **Closed 2026-08-06 (v1.1):** Devstral is the
   default for everything; Magistral planning is optional, reached only via
   the host's manual swap button, at most once per session. See §0.1.
7. **macOS:** the owner has specified `install.bat` and `install.sh`. `install.sh`
   will mostly work on macOS but the toolchain detection differs (clang not gcc,
   no `.exe` suffixes). Support it, or state plainly in the README that it is
   untested there? Recommend the latter until someone asks — an untested claim
   of support is worse than an honest gap.

---

## Appendix C — the questionnaire matrix (preserved from FORGE)

Retained verbatim in intent; it is good design work and should survive. Group
into four tabs: Core Target & Language; Code Style & Strictness; Mechanics
(game/data/security); Theory of Mind profile.

Implementation note: the answers are a plain dict passed to
`Session.start(request, profile)`. **The wizard belongs to the host, not the
core** — CLI and LoLLMs users will supply the same dict by other means. The
core must work with an empty profile and sensible defaults.

---

## Appendix D — small-model failure modes, and the scaffolding that answers them

This is the most useful thing in the document if you have not built against a
7B–24B model before. Every one of these is common at the small end, none is
fixable by asking the model more nicely, and each has a mechanical answer.
**Build the answers in from the start**; retrofitting them is how a project
stalls at 80%.

*Calibration for the actual 24B target is folded into the table's last column:
a 24B is not stupid — it follows instructions, emits valid JSON, and holds a
250–300 line file coherently. Some of these failures become rare at that size.
Keep the handling anyway where it costs nothing; the engine must also survive
the harder case.*

| # | Failure | What it looks like | The antidote — and its weight at 24B |
|---|---|---|---|
| **D1** | **Truncation at `max_tokens`** | A file that ends mid-function, often mid-line. Frequently mistaken for "the model wrote broken code". | Detect it structurally: `finish_reason == "length"` (§5.1), plus unbalanced braces/parens as a backstop. On detection, **continue** rather than regenerate — re-prompt with the tail and "continue from exactly here, do not repeat". Track it in the journal; a high truncation rate means `max_tokens` is too low for the unit of work. *Fully relevant at 24B.* |
| **D2** | **Commentary instead of product** | Asked to return improved code, returns `**Improved version:**` … `**Changes made:**`. Then the caller ships the commentary. | §4.4. Output contract in the prompt AND a detector at the call site that falls back. Never trust the prompt alone. *Rarer at 24B; keep it — the failure was observed in this project's own Athena panel.* |
| **D3** | **The same fix, forever** | Attempts 2, 3 and 4 produce byte-identical code and therefore identical errors. | The stagnation/cycle detector of §6.9 — code hash AND diagnostic hash, cycle set, forward-progress check. *Fully relevant at 24B.* |
| **D4** | **Invented imports and APIs** | `from utils import parse_config` where no such module or function exists. The single most common small-model error in multi-file work. | This is exactly what `codemap` exists for. Additionally: after generation, resolve every import and called symbol against the codemap **before** running anything, and feed unresolved names back as a specific, located error. Cheaper than a failed build and far more precise. *Reduced but real at 24B.* |
| **D5** | **Fence confusion** | Three backticks inside a docstring; a language tag that isn't a language; no fence at all; two fences with different content. | Lenient extraction: prefer a fence whose tag matches the target language, then the longest fence, then the whole reply. Then **validate by parsing** — if it doesn't parse, try the next candidate before giving up. Never assume the first fence. *Rare at 24B; keep the handling, it costs nothing.* |
| **D6** | **Helpful rewriting of code it wasn't asked to touch** | You asked for one function; it returns the whole file, subtly reformatted, with an unrelated "improvement" that breaks something. | Prefer **anchored edits** over whole-file writes for existing files; use FIM where supported (Appendix G.4) — it structurally cannot touch code outside the hole. When a whole file comes back, diff it and report the blast radius before applying — and if the diff touches symbols outside the task, say so loudly in the journal. *Fully relevant at 24B.* |
| **D7** | **Losing the last instruction** | The final constraint in a long prompt is the one most often ignored. | Put the **output contract last** (recency helps) *and* the hard constraints first (primacy helps). Repeat the single most important one in both positions. Keep prompts short — instruction-following degrades faster with prompt length than with task difficulty. *Relevant at every size.* |
| **D8** | **Confident wrong PATHS** | Writes to `src/main.py` when the project uses `app/main.py`. | Never let the model choose the path for an existing file. The planner assigns paths; the model is told the path. For new files, validate the path against the project's observed layout before accepting it. *Fully relevant at 24B.* |
| **D9** | **JSON that is nearly JSON** | Trailing commas, single quotes, a prose sentence before the object, comments. | Grammar-constrained decoding where supported (llama.cpp GBNF; ATK already does this). Where not: a repair parser that strips prose, fixes quotes and trailing commas, and — critically — **reports that it had to repair** (`ToolCall.repaired`, §5.1), so a chronically malformed model is visible rather than silently patched over. *Rare at 24B with native tools; the repair flag stays.* |
| **D10** | **Tool-call syntax drift** | `[SEARCH_CODEMAP: x]` becomes `SEARCH_CODEMAP(x)` or `<search codemap="x">`. | *Moot on the target — tool calling is native (§6.7). Applies only to the text-marker fallback:* accept several forms, cap invocations, correct the syntax once and only once. |
| **D11** | **Context poisoning by its own bad output** | Attempt 3's prompt contains attempts 1 and 2, so the model pattern-matches its own mistakes and repeats them. | Do **not** accumulate failed attempts in the context. Carry forward the *diagnostics*, not the broken code. One prior attempt maximum, and only when the error is a direct continuation. *Fully relevant at 24B.* |
| **D12** | **Silent unit-of-work overrun** | Asked for a 400-line module; produces 90 lines and stops, having decided that's enough. | Verify against the plan: the skeleton says which functions must exist. Missing ones are a specific, checkable error — "you did not implement `parse_header`" — which the model fixes readily. *Fully relevant at 24B.* |
| **D13** | **Reasoning models leaking `<think>`** | Chain-of-thought appears in the file. | Strip reasoning tags before use. ATK has `split_think()` for this; port the same handling. Never write raw model output to a file without passing it through extraction. *Relevant whenever Magistral is loaded (§0.1).* |

**The meta-lesson, worth stating in the README:** with a frontier model you
improve results by improving the prompt. With a small model you improve results
by improving the *loop*. Every hour spent on verification, feedback quality and
error localisation is worth ten spent on prompt wording.

---

## Appendix E — a worked session, end to end

What "working" looks like, for the implementer to build towards and test
against. Target: Devstral Small 2, no network. **This trace is also committed
as a machine-readable golden fixture** — see §9 — so it is executed, not just
admired.

**Request:** *"A CLI that reads a CSV of sensor readings and prints the mean,
min and max per column, with tests."*

```
[plan]      3 files proposed
              src/readings.py   — load and validate the CSV
              src/stats.py      — mean/min/max per column
              src/cli.py        — argument parsing and output
[skeleton]  stubs written, imports resolved, python -c "import src.cli" OK
              → dependency order derived: readings → stats → cli
[build 1/3] src/readings.py
   attempt 1  generated 74 lines
              guard: clean
              syntax: OK
              build: n/a (interpreted)
              test:  FAILED — 1 error
                     tests/test_readings.py:22: AssertionError:
                     expected 3 rows, got 4 (header counted as data)
   attempt 2  fed back: the located assertion + the 5 lines around it
              generated 76 lines
              test:  OK (4 passed)
   → committed. codemap re-indexed src/readings.py (3 symbols, 2 edges)
[build 2/3] src/stats.py
   attempt 1  codemap injected: readings.load_readings(path) -> list[dict]
              test:  OK (6 passed)
   → committed
[build 3/3] src/cli.py
   attempt 1  truncated at max_tokens (finish_reason=length)
   attempt 1c continued from the tail — no regeneration
              test:  OK (3 passed)
   → committed
[verify]    full suite: 13 passed
[review]    deterministic: no secrets, no eval, no bare except
                           1 function over 40 lines (stats.summarise)
            model (one structured pass):
              security:    no untrusted input paths; CSV parsing uses the
                           stdlib reader, no eval-based conversion
              performance: reads the whole file into memory — fine at this
                           size, named as a limit for large inputs
            NOTE: security and performance perspectives came from the same
                  model, so they are not independent scrutiny.
[document]  Recommendation.md written
[journal]   17 events, 3 files, 4 generations, 1 continuation, 1 repair
            all local · no network calls · 0 redactions (none needed)
```

**Acceptance for phase 5 is precisely this**: that trace, on a local model,
without a human intervening — and the same event sequence, shape-diffed
against the golden fixture, on a `ScriptedLLM` in CI. If it needs
hand-holding, the loop is not finished.

---

## Appendix F — techniques that buy accuracy on local models

Appendix D lists what goes *wrong*. This lists what actively makes a local
model better at complex work. Each carries an honest confidence rating —
**PROVEN** (well established, build it), **LIKELY** (strong reasoning, expect
it to work), **WORTH TRYING** (plausible, measure before relying on it) — and,
where the 24B target changes the calculus, **a verdict for the actual
machine**, folded in from the v1.0 calibration review.

Build F1, F9 and F3 first. Between them they will do more for output quality
than any amount of prompt tuning.

### F1. Deterministic pre-fixes — never ask the model what a rule can answer
**PROVEN — and at 24B also a *speed* feature: every error fixed by a rule is
minutes of generation not spent.** This is C5 taken to its conclusion.

A large fraction of diagnostics have exactly one mechanically correct fix:

| Diagnostic | Deterministic fix |
|---|---|
| `NameError` / undefined symbol that EXISTS in the codemap | insert the import |
| unused import / unused variable | delete it |
| missing `self` on a method | insert it |
| formatting, indentation, line length | run the formatter |
| missing trailing newline, tabs vs spaces | normalise |
| missing semicolon / brace (C-family, unambiguous position) | insert it |
| `--fix`-able lint rules (ruff, eslint, gofmt, rustfmt) | run the fixer |

Apply these **before** the model ever sees the error (§6.9). Every one you fix
mechanically is one the model doesn't get a chance to botch — and models botch
trivial fixes surprisingly often, usually by rewriting the surrounding
function while they're in there.

Log every auto-fix. If the same one recurs constantly, the *prompt* needs
changing, and the log is how you find out.

### F2. Test-first, because a test is a smaller generation than an implementation
**LIKELY**, and the highest-leverage structural change available.

For each unit of work, generate the **test before the implementation**:

1. Model writes the test from the task description and the interface.
2. **Human or deterministic check that the test encodes the right intent** —
   this is the cheap moment to catch a misunderstanding.
3. Run it. It must FAIL (if it passes against a stub, it tests nothing —
   a real and common failure worth asserting explicitly).
4. Model writes the implementation.
5. Test passes ⇒ done, by measurement rather than judgement.

Why it works so well: a test is short, highly patterned, and locally scoped.
And it converts "is this right?" from something requiring judgement into
something requiring execution. Step 3 matters more than it looks: a test that
passes against `raise NotImplementedError` is worse than no test, because it
manufactures false confidence for the rest of the session. The planner's
`test_path` pairing (§6.8) is what makes this mechanical.

### F3. Retrieve examples from THIS codebase, not from the model's priors
**LIKELY — upgraded at 24B: it is good enough to *imitate* style faithfully,
which a 7B is not. Two examples buy real consistency.**

Local models have weak priors on your specific idioms — your error handling
style, your logging, your naming. Before generating, use the codemap to find
**one or two existing functions that are structurally similar** to the one
being written, and include them under a heading like *"this is how this
codebase does this"*.

This is a different use of the codemap from dependency injection and both
should run: dependencies tell it what EXISTS, examples tell it what GOOD LOOKS
LIKE HERE. Similarity can stay dumb: same directory, similar name, similar
signature shape, or calls the same helpers. No embeddings needed.

### F4. Best-of-N, decided by the test suite
**PROVEN in the literature — but DEMOTED on the target machine.** Three
samples at minutes each is a poor trade against better context, and at
Devstral's recommended temperature of 0.15 the samples are near-identical
anyway — 3× the wall clock for almost the same answer. If N is ever used, it
must raise temperature for the extra samples (0.15 / 0.4 / 0.7) or it is pure
waste. Keep it as an option for units that have already failed twice; default
N=1; never routine.

The underlying insight survives: local-model failures are frequently *sampling
accidents* rather than capability gaps, and a re-roll fixes a sampling
accident in one round where an iteration loop can spend four arguing with a
bad sample.

### F5. Progressive narrowing of the unit of work
**LIKELY.**

When a unit fails repeatedly, don't retry the same size — shrink it:

```
whole file  →  one class  →  one function  →  one line range
```

Each narrowing raises the ratio of context-to-output, which is precisely the
ratio local models are sensitive to. Wire it into the stagnation detector
(§6.9): identical diagnostics twice ⇒ narrow, don't retry.

The converse also matters — if a unit succeeds first time repeatedly, the
planner may be splitting too finely and paying overhead for nothing.

### F6. Restate the task before doing it
**LIKELY — DEMOTED to optional at 24B**, which rarely misunderstands a
well-scoped task. **Keep the mechanical half**: before generating, verify that
the functions the task will call exist in the codemap; if the plan names
something that isn't there, stop before generating (this catches D4 before a
single line is written). Drop the prose restatement.

### F7. Cascade-aware error feedback
**PROVEN**, language-specific.

The loop caps feedback at three diagnostics. For cascading languages that is
still too many: one missing brace or semicolon in C++ produces forty errors,
thirty-nine of which are noise, and a model will earnestly try to fix
error #23.

Mark cascade-prone languages in `langs.py` (C, C++, Rust to a degree, Java) and
feed back **only the first error** for those, with more source context around
it instead. Non-cascading languages (Python, Go, JS) keep the cap of three.

### F8. Sized work units, measured rather than assumed
**LIKELY — at 24B the starting target moves up to 250–300 lines.**

Quality degrades sharply past a certain output length. The planner should
target the threshold, and split larger files into multiple passes with
explicit seams (imports and class skeleton first; then one method per pass
into a file that already compiles).

Then *measure it*: record output length against first-attempt success in the
journal. After a few sessions the honest threshold for the operator's specific
model is a query, not a guess. That is the kind of thing an audit journal makes
possible almost for free.

### F9. Interfaces as contracts between files
**PROVEN — and on the target machine, the highest-value item in this
appendix: a dependency costs ~30 tokens as an interface and ~800 as a file.**
It is what header files have always been for.

When file B depends on file A, give B **A's interface**, never A's body:
signatures, types, docstrings, exceptions raised. Generate these as
first-class artefacts (`.pyi`-shaped for Python; for C/C++ the header already
is one).

Two benefits: the context cost of a dependency becomes small and predictable,
and the model cannot hallucinate around a signature it has been handed
verbatim.

### F10. Regression memory — the tool gets better at YOUR codebase, offline
**WORTH TRYING**, and the most interesting idea here.

When a fix succeeds, record the pair: *(normalised diagnostic signature → the
shape of the fix that worked)*. On seeing a matching signature later, offer the
previous fix as a hint — or, where the fix was purely mechanical, apply it
deterministically (F1 grows itself).

Over months this makes the tool measurably better on this specific codebase
**with no training, no network and no data leaving the machine** — which for an
air-gapped tool is a genuinely distinctive property. Nothing else in this
document improves with use.

Keep it small, inspectable and per-project (a table in the codemap database),
and make it clearable. A learned "fix" that is wrong must be as easy to delete
as it was to acquire.

### F11. Wall-clock budget for the whole session
**PROVEN** necessity, not an optimisation.

Local generation is slow. A complex multi-file task can run for hours, and an
unattended loop can spend a night achieving nothing. Budget the session, not
just the call: a wall-clock ceiling, a checkpoint that reports *what has been
achieved so far* (`budget` events, §5.4), and a clean stop that leaves
resumable state (§6.13).

### F12. Working across a model swap
**Reframed in v1.1: the swap belongs to the host (§0.1), so this is no longer
an engine feature.** What survives as engine obligations: treat a model change
as an epoch boundary; journal the model per call; and, if the host wants to
preserve a prefix across a swap-and-return, the KV state save/restore
machinery of Appendix G.6 is the mechanism — offered to the host, never
driven by the core. The operator guidance (one swap per session, at a phase
boundary) lives in the ATK panel text.

---

## Appendix G — performance engineering on the target machine

**Consolidated in v1.1 from the v1.0 Appendices G, H and I.** This is the
single authority on speed, context, caching, and model handling for the
actual hardware: **Devstral Small 2 24B on 16 GB VRAM + 64 GB system RAM.**
Model facts verified against the model card on 2026-08-06.

### G.1 What the model actually is

| Property | Value | What it means for this build |
|---|---|---|
| `mistralai/Devstral-Small-2-24B-Instruct-2512` | 24B dense | Already in ATK's `models\` folder. |
| **Licence** | **Apache-2.0** | Commercial use fine. No licence problem for publishing. |
| **Context** | **256k (262144)** | Capacity is not the constraint; seconds are. See G.2. |
| **Native tool calling** | yes, Mistral tool-call format | The codemap tools (§6.7) are the primary interface, not a bolt-on. |
| **Recommended temperature** | **0.15** (Mistral's own examples) | The generation default (§5.3); 0.3–0.5 for planning/review only. |
| **Multimodal** | yes, accepts images | The screenshot feature, §7.2. |
| **Ships `CHAT_SYSTEM_PROMPT.txt`** | in the repo | The base prompt layer, §6.11. |
| **Tokenizer** | `mistral-common` | Exact token counting via the host (§10.3). |
| Built for | Mistral Vibe, a CLI coding agent | The model is *trained around a loop that looks like this one*. |

That last row is the important one. The shipped system prompt describes an
agent that receives "user prompts, project context, and files", emits
"function calls (shell commands, code edits)", and applies "patches, run
commands, based on user approvals". **Cognitive Coder is close to the shape
this model was tuned for.** Lean into that rather than inventing a different
protocol.

For reference, the nearest alternatives in this hardware tier (verified
2026-08-06): Qwen3-Coder (30B-class MoE, Apache-2.0, best speed-to-quality on
a bigger card) and Codestral 22B (the FIM specialist, but **non-commercial
licence** — fine for personal use, check before shipping). A quantisation
note that outranks model choice: code is more sensitive to quantisation than
prose, because it needs exact syntax and long-range consistency. **If Q5_K_M
of a 24B fits, prefer it over Q4 of a bigger model.**

### G.2 The real constraint: seconds, not capability

A 24B is not stupid. It follows instructions, emits valid JSON, and holds a
250–300 line file coherently. The binding constraints are elsewhere:

**VRAM arithmetic** (40 layers, GQA with 8 KV heads, head dim 128):

```
weights, Q4_K_M              ≈ 14.3 GB
KV cache, fp16               ≈ 160 KB per token
   4k context   ≈ 0.6 GB      →  fits on the GPU alongside the weights
   8k context   ≈ 1.3 GB      →  fits, tightly
  32k context   ≈ 5.2 GB      →  does NOT fit in 16 GB with the weights
```

Three knobs for buying context with the 64 GB of RAM, in the order to try:

1. **Quantise the KV cache** (`type_k=q8_0, type_v=q8_0`). Roughly halves KV
   memory for a small quality cost. Cheapest win by far — 16k context becomes
   affordable on the GPU.
2. **Put the KV cache in system RAM** (`offload_kqv=False` in
   llama-cpp-python; `--no-kv-offload` in llama.cpp). Weights stay on the GPU,
   KV lives in the 64 GB; 64k+ is reachable. **Cost: every token's attention
   crosses PCIe** — generation slows noticeably, prompt processing more.
3. **Fewer GPU layers.** Simplest; slowest per token.

**But the cost that does not go away:** even with 64k available, *filling* it
is expensive. Prompt processing runs at a few hundred tokens/sec, so a 32k
prompt is one to three minutes **before the first output token**, and
generation itself runs at roughly 10–25 tok/s with partial offload — a
300-line file is three to six minutes.

The operating principle, and it governs everything below: **buy context
deliberately, and never pay for it twice.** Every token of context must earn
its place; every attempt is expensive; spend engineering effort on *preparing
one excellent prompt*, not on recovering from cheap bad ones. The codemap,
interface extraction and slice selection are not supporting features — they
are the main event.

### G.3 Measure the real budget at startup; don't take it from config

`n_ctx` is not the usable budget. Compute it and store it on the session:

```
usable_prompt = n_ctx
              − reserved_output_tokens        (what you'll ask it to write)
              − reasoning_tax                 (below)
              − safety_margin (~10%)
```

ATK already has `gguf_meta.py` (offline header parsing) and `vram.py` for
this. Feed the number into `context.build_context()` as the budget. A budget
taken from a config file drifts from reality the moment the operator changes
models.

**The reasoning tax:** when Magistral is the loaded model (§0.1), it spends
1,000–2,000 tokens thinking *before* it writes anything. Subtract that from
the budget explicitly rather than discovering it as mysterious truncation —
and always strip `<think>` blocks before use (D13). This tax is one of the
concrete reasons Devstral, not a reasoning model, is the generation default:
where output is the product, thinking is overhead; where thinking is the
product (planning, review), it can be worth a swap.

### G.4 Fill-in-the-middle for edits, where the model supports it
**PROVEN, and underused.**

Editing an existing function by saying "here is the file, rewrite it" wastes
the whole file in *and* the whole file out — minutes at these speeds — and
invites D6 (helpful rewriting of things you didn't ask about).

Code-specialised models support **FIM**: give the prefix and the suffix, and
the model fills the hole. Cost drops to the size of the hole, and it
structurally cannot touch code outside it.

Detect FIM support from the model's tokens (`<|fim_prefix|>` / `<|fim_suffix|>`
/ `<|fim_middle|>`, names vary by family — surfaced as
`ModelCapabilities.supports_fim`) and use it for edits when present, falling
back to anchored replacement otherwise. `langs.py` already knows where
functions begin and end; that is exactly the prefix/suffix boundary.

### G.5 Spend model calls like they cost minutes — because they do

Consequences threaded through the body, collected here:

- **Review is one structured pass**, not two sequential personas (§4.3,
  §6.10).
- **Best-of-N is demoted** (F4): near-pointless at temperature 0.15 unless the
  extra samples raise temperature, and expensive regardless.
- **Prose restatement is dropped** (F6); the mechanical codemap check stays.
- **Deterministic pre-fixes are a speed feature** (F1), not just a quality
  one.
- **Interfaces-as-contracts (F9) and semantic zoom (§6.7) are the main
  event** — they are what make one excellent prompt affordable.

### G.6 Model swapping: the costs, honestly

The host owns the swap (§0.1). These are the facts the host's button — and
the operator guidance around it — should be built on:

- With 64 GB of RAM, a 14 GB GGUF stays in the OS page cache after first
  load, so a *reload* costs 10–20 seconds, not 60. **But the weights are only
  half the state: the KV cache dies with the model instance.** A naive swap
  back means reprocessing the entire prefix — at 16k tokens and a few hundred
  tokens/sec, that is 30–60 seconds of pure recomputation on *every* return
  trip. This is why phase-alternation loses and one-swap-per-session wins.
- **KV state save/restore exists** and can soften a return trip:
  `llama_state_save_file` / `llama_state_load_file`
  (`save_state()`/`load_state()` in llama-cpp-python; `--prompt-cache` on the
  CLI). Reading a 2–3 GB state file from page-cached RAM is a few seconds
  against 30–60 of reprocessing. **Verify before relying on it:** the state
  is bound to that model with those parameters (an `n_ctx` or quant change
  invalidates it); save/load has been fragile across llama.cpp versions —
  treat a failed restore as normal, fall back to reprocessing, log it, carry
  on. q8_0 KV quantisation halves the file, so do that first regardless.
- **Whether Magistral planning beats Devstral-only planning is measurable,
  not assumable.** Devstral is an instruction model that scores well on
  agentic SWE work; the reasoning model's advantage on planning may not
  survive the cost of getting to it. The journal records first-attempt
  success rates per model — let it answer.
- **If a swap happens, time it at a point where the prefix was going to
  change anyway** (a new target file, an epoch rebuild — G.7). Swapping at a
  moment when the cache would otherwise have been reused pays the cost twice.

### G.7 Prefix caching and codemap freshness — the reconciliation

llama.cpp caches the KV state of a prompt prefix: if the *beginning* of the
prompt is byte-identical to the previous call, those tokens are not
reprocessed. At these speeds that is the difference between 3 seconds and
minutes, on every call. But §6.7 demands a fresh codemap, and every file
write updates the map — taken naively, those two rules say "rebuild the
prefix after every write and never benefit from the cache at all".

**The resolution turns on one distinction: the injected map is a HINT; the
tool call is GROUND TRUTH.** `search_codemap()` queries live SQLite every
time, so the *queryable* interface is never stale, and staleness in the
*injected* text costs at most an extra tool call — never a wrong answer.
Restated rule: **the query interface must never be stale; the cached hint may
lag, provided the model can always check.**

Five mechanisms, all required:

**1. Order the prompt stable-first, volatile-last — and split it at the zoom
boundary, which is already the right seam:**

```
CACHED PREFIX  ── 1. system prompt / persona          (never changes)
                  2. project conventions, style        (never changes)
                  3. DISTANT architecture, low-res     (changes by epoch)
                  ── stable for the whole session, ~10–20k tokens
─────────────── cache boundary ───────────────
VOLATILE TAIL  ── 4. immediate dependencies, high-res (F9 interfaces)
                  5. same-codebase examples (F3)
                  6. THE TASK
                  7. staleness note + last attempt's diagnostics
                  8. output contract                   (short; last for recency)
                  ── a few thousand tokens, reprocessed each call
```

The semantic zoom's split between *distant architecture* and *immediate
dependencies* is exactly the split between slow-changing and fast-changing
content. **The zoom tiering and the cache tiering are the same split** — both
answer "what is unlikely to matter in detail right now". Items 7 and 8
conflict (volatile vs last); resolve by keeping the contract short enough
that repeating it costs little, with the diagnostics immediately before it.
**Never put a timestamp, a session id, or a randomised preamble anywhere in
the prefix** — one varying token at position 40 silently discards 30k tokens
of cached work.

**2. The SQLite index updates immediately; the injected text updates by
epoch.** Re-index on every write (tools stay live). Rebuild the cached prefix
only at an epoch boundary: session start · the plan is revised (§6.8
`replan`) · N files changed since the snapshot (default 5) · a signature
change whose blast radius includes the current target · a model change
(§0.1) · the operator asks.

**3. Declare the lag in the volatile tail**, never in the prefix:

> *Architecture snapshot: epoch 4. Three files have changed since it was
> taken (`parser.py`, `stats.py`, `cli.py`). For anything you are about to
> touch, call `search_codemap` rather than trusting the summary above.*

Putting this in the tail is essential — a note in the prefix would change the
prefix bytes and invalidate the very cache it describes. The tail is
reprocessed anyway, so it is free.

**4. Keep several cached prefixes alive.** With 64 GB of RAM there is no
reason to thrash one slot. Cache per (persona, epoch) and keep the last few;
llama.cpp server exposes multiple slots and `--cache-reuse`, and
llama-cpp-python has `save_state()`/`load_state()`. Switching target files
inside an epoch then costs nothing.

**5. Measure it, because a broken cache is silent.** The journal records
prompt-processing time per call (`prompt_ms`, §6.13). A jump from 3 s to 90 s
means the prefix changed when it should not have — and that is the *only* way
anyone will notice. The CI prefix-stability test (§9) asserts byte-identical
prefixes for the same (persona, epoch, target).

**The honest cost:** within an epoch, the low-resolution architecture summary
can be up to five files out of date. That is acceptable *only* because the
model can always call the tool — so the text-marker fallback path (§6.7),
which has no live tools in the same sense, **forces epoch-per-write** and
accepts the slower prompts. It is noted in the fallback path so nobody
removes the tools and quietly breaks the guarantee.

### G.8 Starting configuration to try first

Not a recommendation to hardcode — a starting point to measure from, recorded
in the journal so the next value is evidence-based:

```
model            Devstral-Small-2-24B-Instruct-2512, Q4_K_M  (or Q5_K_M if it fits)
n_ctx            16384        ← start here, raise once prefix caching is proven
type_k/type_v    q8_0         ← halve the KV cost before touching offload
offload_kqv      True         ← flip to False only if you need >32k
n_gpu_layers     as many as fit with the KV above
temperature      0.15         generation · 0.35 planning/review
reserved output  2048 tokens
work unit target 250–300 lines
tool calling     native (§6.7)
```

### G.9 The tuning decision worth surfacing in the UI

At 16 GB the operator faces a real trade: **fast and blinkered (all layers on
GPU, smaller context) versus slower and better-sighted (KV tricks or offload,
larger context).** For multi-file coding, seeing more usually wins — but not
always, and it depends on the codebase.

Do not guess on their behalf. Expose it, state the trade in one sentence, and
let the journal answer it empirically after a few sessions: record context
size, offload split, tokens/sec and first-attempt success rate, and the right
answer becomes a query rather than an argument.

---

## Appendix H — the MUST index: one numbered conformance checklist

Every binding obligation in this document, collected. A reviewer works down
this list; an implementer turns it into a test plan. Each item cites the
section that explains *why* — the why is not repeated here. If an item on
this list conflicts with body text, the body text wins and this index has a
bug; report it.

**Constitution & architecture**

| # | Obligation | § |
|---|---|---|
| M1 | `cognitive_coder/**` imports no PySide6/Qt/FastAPI/ATK/LoLLMs, ever — enforced by a CI walk of the tree | C1, §8, §9 |
| M2 | Everything host-provided arrives through a Port; the core never locates hosts itself | C2 |
| M3 | No remote call without explicit per-session enablement; remote mode is visibly indicated whenever active | C3, §6.12 |
| M4 | "Done" = build succeeded AND tests ran (or the absence of both is stated); parse success is never completion | C4 |
| M5 | Operator-facing errors are plain sentences; tracebacks go to the journal | C6 |
| M6 | Missing optional dependencies degrade with a stated cost; they never crash | C7 |
| M7 | Every artefact journals provider, model, prompt hash, attempt, verification, timestamp | C8 |
| M8 | Ports AND shared types are frozen at 1.0 under semver; breaking either is a major version | C9, §5, §10.4 |
| M9 | No documentation describes guard + sandbox + jail as a security boundary against a hostile model | C10, §6.3 |
| M10 | The core contains no model-swap logic; capabilities() is re-read at task boundaries; "no model loaded" is a normal state | §0.1 |

**The contract (§5)**

| # | Obligation | § |
|---|---|---|
| M11 | `complete()` never raises on model refusal — it returns the text | §5.3 |
| M12 | A host without tool support ignores `tools` and reports `supports_tools=False`; the core then uses the text fallback | §5.3, §6.7 |
| M13 | `capabilities()` reflects the currently loaded model | §5.3 |
| M14 | `count_tokens` honesty: exact or a flagged estimate; tokenizer deps live in host/provider, never core | §5.3, §10.3 |
| M15 | `write_bytes` is atomic, or the host's PORTS.md entry says it is not | §5.3 |
| M16 | `ExecPort.run` kills the entire process tree on timeout; `timed_out=True` attests it | §5.3, §6.1a |
| M17 | `StoragePort` values are JSON-serialisable | §5.3 |
| M18 | Every write — including model-initiated `apply_patch` tool calls — routes through `ApprovalPort.approve_diff` and the current transaction | §5.3, §6.5, §6.7 |
| M19 | Event kinds are the closed set of §5.4; additions are minor versions, renames major | §5.4 |
| M20 | Every Port ships a Null implementation; the engine runs hostless | §5.5 |
| M21 | Cancellation: token checked at phase boundaries; cancelled work leaves resumable state; open transactions roll back | §5.2 |

**Engine behaviour**

| # | Obligation | § |
|---|---|---|
| M22 | Build/run/test/guard failures are attributed to their phase, distinguishably | §6.4 |
| M23 | An anchor matching more than once is refused, never guessed | §6.5 |
| M24 | No write outside the resolved real-path project root, in any mode | §6.5 |
| M25 | Transactions: planner-declared atomicity; sequence numbers not timestamps; committed+verified = sealed; rollback journaled; history queryable | §6.5 |
| M26 | Undo restores byte-identical content — encoding, BOM and EOL included | §6.5a, §9 |
| M27 | The engine never runs `git`; generated code invoking VCS is blocked; `.git/` excluded from index/patch | §6.3, §6.5b |
| M28 | Injected context ends with the declared-omissions block | §6.6 |
| M29 | Unrecognised toolchain output yields one diagnostic, never `[]` | §6.2 |
| M30 | The codemap query interface is never stale; the injected summary may lag only by declared epoch, with the staleness note in the volatile tail | §6.7, G.7 |
| M31 | The text-marker fallback caps lookups at 3 and forces epoch-per-write | §6.7, G.7 |
| M32 | Truncation (`finish_reason=="length"`) triggers continuation, not regeneration, and is journaled | D1, §6.9 |
| M33 | Failed attempts are not accumulated in context; diagnostics carry forward, not broken code | D11, §6.9 |
| M34 | Stagnation detection hashes normalised code AND diagnostics, keeps a cycle set, and reports the cycle in plain words on stop | §6.9 |
| M35 | Deterministic pre-fixes run before the model sees an error, and every auto-fix is logged | F1, §6.9 |
| M36 | Machine-consumed prompts end with an output contract; every consuming call site runs the commentary detector | §4.4, §6.11 |
| M37 | `<think>` blocks are stripped before any model output is used | D13, §6.11 |
| M38 | Model paths are assigned by the planner for existing files, validated against layout for new ones | D8, §6.8 |
| M39 | Every implementation task carries its test path; a new test must fail before the implementation exists | §6.8, F2 |
| M40 | Headless Godot passes on scene/physics/rendering-touching tests carry the headless caveat; never unqualified success | §6.1a |
| M41 | Same-model review is labelled non-independent, near the top of the document it produces | §4.3, §6.10 |

**Providers, network, distribution**

| # | Obligation | § |
|---|---|---|
| M42 | Remote providers: session-explicit enablement, no env-var auto-on; approval before first call; budgets that halt; persistent REMOTE indicator | §6.12 |
| M43 | Every outbound message — including tool results — passes through `redact.py`; redaction counts are reported | §6.12 |
| M44 | Keys never touch the journal, codemap, or any log | §6.12 |
| M45 | Installers: non-interactive, idempotent, nothing global, degrading options, honest summary, meaningful exit code | §10.1 |
| M46 | The installer verifies interpreter version (`--version`), fetches 3.11 when needed, and reports which branch it took | §10.2a |
| M47 | `requires-python = ">=3.11"` — floor, no cap; CI runs 3.11 on Windows and Linux | §10.2a, §10.4 |
| M48 | Zero required runtime deps; CI fails on any non-stdlib import at module level in the core | §10.3 |
| M49 | `ccoder doctor` exists and prints the install summary plus interpreter provenance on demand | §10.2a |
| M50 | Vendored use (`sys.path`-insert, no install) works: no import-time side effects, no package-metadata reads; version from `version.py` | §1.2 |

**Testing (all of §9 is binding; the load-bearing four)**

| # | Obligation | § |
|---|---|---|
| M51 | The no-network test drives a full scripted session with sockets monkeypatched to raise | §9 |
| M52 | The prefix-stability test asserts byte-identical prefixes per (persona, epoch, target) | §9, G.7 |
| M53 | The Appendix E golden trace is committed as a fixture and shape-diffed in CI | §9, App E |
| M54 | The port conformance kit ships and covers atomicity, tree-kill, jail, and capabilities honesty | §9, §5.5 |
| M55 | The journal records `prompt_ms` per call | §6.13, G.7 |

---

## Appendix I — corrections history (v1.0 → v1.1)

v1.0 was written in layers, with later appendices correcting earlier sections
and a precedence chain to arbitrate. That structure was honest but hostile to
a fresh reader: it required holding five correction layers in mind at once.
v1.1 merges every correction into the section it amends and deletes the
chain. This appendix records what moved and what was *wrong*, because a spec
that hides its own errors teaches the reader to trust the wrong parts. The
v1.0 file is kept beside this one for the full archaeology.

**Corrections carried over from the v1.0 review (Appendix I of that
document), now merged into the body:**

- **Model swapping cost (was H8, corrected by I1).** The claim that a swap
  costs 10–20 s accounted for the weights and forgot that the KV cache dies
  with the model instance — the true return-trip cost is 30–60 s of prompt
  reprocessing. Merged into G.6, and then overtaken by an owner decision
  (below).
- **Snapshot transactions (was I2).** "Undo restores an apply as one unit"
  was ambiguous at the function-call boundary; the explicit transaction model
  with planner-declared atomicity, sequence numbers and sealed history is now
  §6.5.
- **Headless Godot evidence (was I3).** Merged into §6.1a: timeouts,
  tree-kill, `--fixed-fps`, test classification, and the headless caveat.
- **Stagnation cycles (was I4).** Diagnostic-hash-only detection misses
  ping-pong cycles; the code+diagnostics signature pair, cycle set, and
  churn/progress checks are now §6.9.
- **Codemap freshness vs prefix caching (was I5).** The hint/ground-truth
  distinction, the epoch mechanism, and the zoom/cache split are now G.7.

**New in v1.1 (from the 2026-08-06 external review):**

- **The LLMPort contract could not express tool calling, vision, or
  cancellation** — the features the rest of the spec depends on — and it was
  about to be frozen at 1.0. Rewritten as messages-based with `ToolSpec`/
  `ToolCall`/`Completion` and a `CancelToken` (§5.1–5.3). The shared types
  are now explicitly part of the frozen contract (C9).
- **Tool-initiated edits now explicitly route through the approval gate and
  transaction** (§6.5 rule 6, M18). As written in v1.0, native tool calling
  was a side door around the settled approval default.
- **Encoding and line endings were unspecified** on a Windows-first project;
  `textio.py` (§6.5a) now owns detection, internal normalisation, and
  byte-faithful restoration — without which the byte-identical undo property
  test was unsatisfiable.
- **`requires-python = ">=3.11,<3.12"` contradicted library-first**: an
  embedded library does not control its host's interpreter. Now a floor with
  no cap; CI stays 3.11-only (§10.2a).
- **The phase plan asked phase 4 (loop) to demonstrate phase 5 (codemap).**
  Reordered: codemap is now phase 4, loop phase 5, and the Appendix E trace
  is phase 5's acceptance (§11).
- **Process-tree kill promoted from a Godot rule to an `ExecPort` rule**
  (§5.3): orphaned children on timeout are a general Windows failure mode,
  not an engine-specific one.
- **The git stance is now stated** (§6.5b) so nobody "improves" the bespoke
  snapshots into shadow commits: never run git, warn once on a dirty repo,
  exclude `.git/`.
- **Owner decisions folded in:** the model strategy — Devstral for
  everything by default, Magistral planning optional, **swapping is the
  host's manual action, never the engine's** (§0.1); this reframed F12 and
  turned G.6 into guidance for the host's button rather than an engine
  feature. The stray "Ministral" in v1.0 was a typo for Magistral.
- **Smaller fixes:** journal example model name (was `magistral-24b-q4` in a
  spec whose generator is Devstral); tokenizer dependency placement stated
  (§10.3); redaction extended to tool-result messages (§6.12); path
  containment defined on resolved real paths (§6.5); runtime toolchain
  probing outranks the installer's record (§6.1); event vocabulary fixed
  (§5.4); the LoLLMs adapter directory reduced to a placeholder README to
  match §8.
- **New apparatus:** the MUST index (Appendix H), the port conformance kit,
  the golden trace fixture, and `examples/tiny_host.py` (§9, §5.5).

*End of specification.*







