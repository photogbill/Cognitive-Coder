# Cognitive Coder

**A host-agnostic engine that lets a language model write, build, test and fix
real code in a real project — designed so it still works when the model is
small and the machine is offline.**

It is a **library first** and an application second. The core is pure Python
with **no GUI dependency, no host dependency, and zero required runtime
dependencies**. It embeds in a Qt desktop app, a FastAPI server or a
terminal, and it runs on a disconnected laptop in a field office.

```
git clone https://github.com/photogbill/Cognitive-Coder
cd Cognitive-Coder
install.bat          # Windows
./install.sh         # Linux
.venv/bin/ccoder doctor
```

Cognitive Coder is **not published to PyPI**: clone it and run the installer,
or embed it as described below. The git tag is the release; `CHANGELOG.md`
is the record. Apache-2.0.

---

## Contents

- [What it needs](#what-it-needs)
- [Ten lines of embedding](#ten-lines-of-embedding)
- [Integrating it into your own front end](#integrating-it-into-your-own-front-end)
- [The command line](#the-command-line)
- [Where it wins, and where it loses](#where-it-wins-and-where-it-loses)
- [The non-negotiables](#the-non-negotiables)
- [The trust model](#the-trust-model-stated-plainly)
- [What it does](#what-it-does)
- [Languages](#languages)
- [Documentation](#documentation)

---

## What it needs

- **Python 3.11 or later.** Tested on 3.11 on Windows and Linux; earlier
  versions are refused at install. If the machine has nothing suitable, the
  installer fetches 3.11 **into the clone** — nothing is installed
  system-wide, and deleting the clone removes every trace.
- **A model.** Anything with an OpenAI-compatible endpoint — llama.cpp
  server, Ollama, LM Studio, vLLM, LiteLLM — or a GGUF loaded in-process
  through `llama-cpp-python`. The engine is calibrated against
  **Devstral Small 2 24B** at 16 GB of VRAM, but nothing about it is
  specific to that model.
- **Optionally, a toolchain.** `ccoder doctor` prints exactly which languages
  are usable right now and what each missing one costs. Nothing fails
  because you lack Rust; that language is unavailable and it says so.

## Ten lines of embedding

Runs anywhere, as it stands: no model, no network, no host application.

```python
import tempfile

from cognitive_coder import (AutoApprove, Host, LocalFileSystem, ScriptedLLM,
                             Session, SessionConfig)

# A scripted model answers each call in order: the plan, then the file.
replies = ["greet.py — say hello to a named person\n",
           '```python\ndef greet(name):\n    return f"hi {name}"\n```']
host = Host(llm=ScriptedLLM(replies, supports_tools=False),
            fs=LocalFileSystem(tempfile.mkdtemp()), approval=AutoApprove())
session = Session(host, config=SessionConfig(attempts=1))
session.run("a greet function")
print(session.report())
```

Two replies, because a session asks twice: once to **plan**, once per
**file**. `tests/test_cli.py` executes this block, so it stays true.
`examples/tiny_host.py` is the same idea grown just enough to prove the
embedding story end to end — about fifty lines, every one of them a line a
host author writes. Read it first.

---

## Integrating it into your own front end

Cognitive Coder has no UI of its own and wants none. Your application
supplies six small objects — **the Ports** — and gets back a session it can
drive, cancel, render and audit. Nothing you write inherits from anything in
this package: the Ports are `typing.Protocol`s, implemented structurally, so
your code never imports a base class of ours. This section is the whole
contract; `docs/PORTS.md` has every guarantee in full and `docs/EMBEDDING.md`
the long form.

### 1. Get the code into your project

Three ways in, all supported and all tested:

| Route | When | How |
|---|---|---|
| **Vendor it** (git submodule) | You would rather not manage another dependency | `git submodule add https://github.com/photogbill/Cognitive-Coder vendor/cognitive-coder`, then `sys.path.insert(0, "vendor/cognitive-coder")` and `import cognitive_coder`. No install, no package metadata. Pin to a tag. |
| **Editable install** | A Python environment you control | `pip install -e ./Cognitive-Coder` — plus any extras you want: `[llamacpp]`, `[treesitter]`, `[tokenizers]`, `[openai]`, `[anthropic]`, `[google]`, `[mistral]`, `[dev]`, `[all]`. |
| **The installers** | A standalone copy, or a contributor | `install.bat` / `install.sh` build a `.venv` inside the clone, detect toolchains and print an honest summary. |

The vendoring route is protected by tests: importing the package has **no
side effects** (no directory created, no database opened, no config read, no
toolchain probed), the version is read from `version.py` rather than package
metadata, and no module imports anything outside the standard library at
module level.

### 2. Implement the six Ports

| Port | Methods | What it is in your application |
|---|---|---|
| `LLMPort` | `complete(messages, *, tools, temperature, max_tokens, stop, grammar, seed, cancel) → Completion` · `stream(messages, **kw)` · `capabilities() → ModelCapabilities` · `count_tokens(text)` | A thin shim over the model you already have loaded — never a second model. |
| `FileSystemPort` | `read` / `write` / `read_bytes` / `write_bytes` / `exists` / `list(glob)` / `delete` / `root()` | The project folder, jailed: no write lands outside `root()`, judged on resolved real paths. |
| `ExecPort` | `run(argv, *, cwd, timeout, stdin="", env=None) → ProcResult` · `which(binary)` | How compilers, linters and test runners are run. **This is where your sandboxing policy lives.** |
| `StoragePort` | `get(key, default)` · `set(key, value)` · `sqlite_path(name)` | Where the engine keeps the transaction log, its sequence counter and the CodeMap index — per project. |
| `EventPort` | `event(kind, message, data=None)` | Your UI. Every phase, token, diagnostic, patch, warning and budget notice arrives here. |
| `ApprovalPort` | `approve_diff(summary, unified_diff) → bool` · `approve_remote(provider, bytes_out, estimate) → bool` | The human in charge: nothing is written to disk, and nothing leaves the machine, without a yes. |

Ready-made implementations ship for every one, so you only write the Ports
that touch your application: `LocalFileSystem` / `MemoryFileSystem`,
`SubprocessExec`, `JsonFileStorage` / `MemoryStorage`, `SilentEvents` /
`RecordingEvents`, `AutoApprove` / `DenyAll`, and `ScriptedLLM` / `NullLLM`
for development without a model. `Host(...)` fills any Port you leave out
with its Null implementation.

A complete host:

```python
from cognitive_coder import Host, Session, SessionConfig, SubprocessExec

class MyEvents:
    def event(self, kind, message, data=None):
        my_panel.append(f"[{kind}] {message}")          # marshal to your UI thread

class MyApproval:
    def approve_diff(self, summary, unified_diff):
        return my_dialog.confirm(summary, unified_diff)
    def approve_remote(self, provider, bytes_out, estimate):
        return my_dialog.confirm_network(provider, bytes_out, estimate)

host = Host(
    llm=MyLLMPort(my_loaded_model),          # section 3
    fs=MyFileSystem(project_root),           # or LocalFileSystem(project_root)
    exec=SubprocessExec(),                   # or your own sandbox
    storage=MyStorage(),                     # or JsonFileStorage(".cc_state")
    events=MyEvents(),
    approval=MyApproval(),
)

session = Session(host, config=SessionConfig(lang="python"))
outcomes = session.run("add CSV import to the report module")
print(session.report())
```

### 3. Wrap the model you already have

The most common case and the cheapest: your application has a model loaded
and `LLMPort` is a shim over it.

```python
from cognitive_coder import Completion, ModelCapabilities

class MyLLMPort:
    def __init__(self, engine):
        self.engine = engine

    def complete(self, messages, *, tools=(), temperature=0.15,
                 max_tokens=2048, stop=None, grammar=None, seed=None,
                 cancel=None):
        text, t = self.engine.generate(messages, temperature=temperature,
                                       max_tokens=max_tokens, stop=stop,
                                       grammar=grammar, cancel=cancel)
        return Completion(text=text,
                          finish_reason="length" if t.hit_limit else "stop",
                          tokens_in=t.prompt_tokens, tokens_out=t.gen_tokens,
                          model=self.engine.model_name,   # per call — it may change
                          prompt_ms=t.prompt_ms)          # the cache-health signal

    def stream(self, messages, **kw):
        yield from self.engine.stream(messages)

    def capabilities(self):
        return ModelCapabilities(name=self.engine.model_name or "",  # "" = none loaded
                                 family="mistral", context_tokens=self.engine.n_ctx,
                                 supports_tools=self.engine.has_tools,
                                 supports_grammar=True,
                                 token_count_is_estimate=False)

    def count_tokens(self, text):
        return len(self.engine.tokenize(text))
```

The guarantees a host must keep (the full list is in `docs/PORTS.md`):

- **`complete()` never raises on a model refusal** — return the text. A
  refusal is data the loop acts on; an exception is a crash it cannot.
- **`capabilities()` reflects what is loaded right now.** "No model loaded"
  is a normal state, reported as `ModelCapabilities(name="")`, not an
  exception. The engine re-reads it at every task boundary and treats a
  change of model as an epoch boundary (the cached prompt prefix is rebuilt).
- **A model without tool support ignores `tools` and says
  `supports_tools=False`**; the engine then uses its text-marker fallback.
- **`model` is named per call** and **`prompt_ms` is reported where you can
  measure it** — the only signal that the prompt-prefix cache broke, a
  failure that is otherwise silent.

If you have no model of your own to wrap, two providers ship:
`cognitive_coder.providers.OpenAICompatible(base_url, model=...)` for
llama.cpp server, Ollama, LM Studio, vLLM and LiteLLM, and
`cognitive_coder.providers.LocalLlamaCpp(model_path, n_ctx=...)` for an
in-process GGUF (the `llamacpp` extra; pass `llama=` to wrap a `Llama` you
already built). Remote providers (OpenAI, Anthropic, Google, Mistral,
OpenRouter) exist behind `RemoteGate`: nothing leaves the machine unless your
host enables one *for this session* and the person approves the send.

### 4. Render the events

`EventPort.event(kind, message, data)` is your entire UI feed. The vocabulary
is a **closed set**, so a renderer can switch on it without worrying about
tomorrow's addition: `phase`, `token`, `status`, `diagnostic`, `patch`,
`remote`, `warning`, `error`, `budget`, `plan`. A `phase` drives a progress
indicator, `diagnostic` fills a problems list, `patch` carries a unified
diff, `token` streams generated text, and `remote` lights a warning that
stays lit for as long as a remote provider is in use.

### 5. Threading and cancellation

The core is **synchronous and single-threaded**, deliberately: a GUI has its
own concurrency model and the engine does not impose a second one. Run a
session on a worker thread; events arrive on that thread, so marshal them to
your UI thread the way your framework expects.

**`Session.cancel()` is the one method safe to call from any thread.** It
is a `threading.Event` underneath. Cancellation and budget exhaustion both
leave resumable state and end the session with an honest `session_end`
rather than an absence of one; `Session.resume(host, session_id)` rebuilds a
session from its journal after a crash.

```python
class SessionRunner(QRunnable):          # the pattern, not the API
    def run(self):
        self.session.run(self.request)

self.session.cancel()                    # from the UI thread, at any time
```

### 6. What the engine leaves in the project

All of it inside the project root, and all of it yours to delete:

- **`.cc_snapshots/`** — the original bytes of every file a transaction
  touched, with the diff. Undo restores from here, byte-identical.
- **`.cc_journal/`** — one append-only JSONL per session: provenance, and
  the source of truth for resume.
- **`BUILD_LOG.txt`** — the readable account of every session.
- **`Recommendation.md`** — the review written after a build that verified
  (`SessionConfig.recommendation_path` moves it).

Your `StoragePort` decides where the transaction log and CodeMap live; the
`ccoder` CLI uses `JsonFileStorage` in `.cc_state/`. **It never runs git.**
A project with uncommitted changes is mentioned once at session start and
left alone. `session.history()` answers "what did it just do to my project"
from the transaction log.

### 7. Prove the Ports before you ship them

```python
from tests.port_conformance import check_all

report = check_all(fs=MyFileSystem("/project"), exec=MyExec(),
                   llm=MyLLMPort(engine), storage=MyStorage())
print(report.text())
assert report.ok
```

The conformance kit ships as part of the product. It turns every "a host
must guarantee" into an executable assertion — write atomicity, process-tree
kill, the path jail, capabilities honesty — so you do not have to read the
core to know whether your implementation is right.

### 8. Two worked examples

- **ATK (PySide6 desktop).** `adapters/atk/` holds the six Ports and the
  workspace panel ATK runs: `ccoder_host.py` is the Qt-free half (the Ports,
  tested without a `QApplication`), `ccoder_panel.py` the only file that
  imports Qt. Its `LLMPort` wraps the model ATK already has loaded; its
  `StoragePort` keeps one bucket per project; events cross to the GUI thread
  through signals; approvals are dialogs with *Don't apply* as the default.
  `adapters/atk/README.md` explains each choice.
- **A web host (the LoLLMs shape).** `adapters/lollms/README.md` shows the
  outline: an `LLMPort` over the binding the server already has, an
  `EventPort` that pushes to the client over SSE or a websocket, and
  everything else from the ready-made implementations. The core imports no
  web framework either — FastAPI, Starlette and Flask are on the same
  forbidden list as Qt.

### The API surface

Everything a host needs is exported from `cognitive_coder`: the six Port
protocols, `Host`, `Session`, `SessionConfig`, the shared types (`Message`,
`Completion`, `ModelCapabilities`, `ProcResult`, `Diagnostic`, `TaskOutcome`,
…), the ready-made Port implementations, the providers, `Journal`, `CodeMap`,
`Patcher`, `audit_project` and the skills loader. The public API is the Ports
of `ports.py` and the shared types of `types.py`; breaking either is a major
version.

---

## The command line

`ccoder` is the reference host — the same engine behind a terminal:

| Command | What it does |
|---|---|
| `ccoder doctor` | Which Python, which model endpoint, which languages are usable right now and what each missing toolchain costs. |
| `ccoder build -p PROJECT "request"` | Plan and build; `--interview` asks the decisions a request leaves open before planning; `--spec FILE` builds from a written plan; `--remote PROVIDER` enables a remote model for this run only. |
| `ccoder resume SESSION` | Pick a stopped build up where it left off. |
| `ccoder history` | What was changed, by which task, verified or rolled back. |
| `ccoder postmortem` | What a failed session tried, and why each attempt failed. |
| `ccoder skills` | Deployed project skills: `deploy` the starter pack, `new NAME`, or list what a session would load. |
| `ccoder audit PROJECT` | What is wrong and what could be better in an existing project, ranked, with file and line — and a plan `build` can carry out. |
| `ccoder bench` | The regression suite: tasks graded by hidden tests, each run compared with the last; `--calibrate` measures a model's own sampling profile. |

---

## Where it wins, and where it loses

Asked honestly whether this can be "as good as Cline, or better", the truthful
answer has three parts, and the design is built around it.

**Where it wins**

| Axis | Why |
|---|---|
| **Works offline, on small models** | Cline assumes a frontier model behind an API. Every decision here assumes a 7B–24B local model and compensates with scaffolding. |
| **Host-agnostic** | Cline is a VS Code extension. This is a library that embeds in a Qt desktop app, a FastAPI web app or a terminal. |
| **Audit-grade provenance** | Every line produced is traceable to a model, a prompt, an attempt number and a verification result. For anyone who has to defend a change, that is decisive. |
| **Deterministic-first** | Linters, compilers and test runners answer what they can; the model is asked only what tools cannot decide. Cheaper, faster, more trustworthy. |

**Where it loses, and there is no point pretending otherwise**

- Raw code quality when Cline is driving a frontier model. A local 24B
  narrows the gap; it does not close it.
- Ecosystem polish. Cline has an editor's whole UI, inline decorations and
  years of interaction design.

So this is not "Cline but ours". It is the thing Cline cannot be: a coding
engine that runs on a disconnected laptop, embeds in somebody else's
application, and can prove what it did.

**The meta-lesson.** With a frontier model you improve results by improving
the prompt. With a small model you improve results by improving the *loop*.
Every hour spent on verification, feedback quality and error localisation is
worth ten spent on prompt wording — which is why the largest and most
carefully tested module here is the one that parses compiler output, and why
"it didn't work" is never an answer this engine gives.

---

## The non-negotiables

These survive every implementation decision. If a convenience conflicts with
one, the convenience loses.

1. **The core imports no GUI and no host.** Not conditionally, not inside
   functions. Enforced by a test that walks the tree.
2. **Everything the host provides comes through a Port** — model inference,
   file writes, sandboxed execution, storage, events, approval.
3. **Offline is the default; the network is an explicit, visible choice.**
   No remote call happens unless you enabled a remote provider *for this
   session*, and remote mode is shown whenever it is on. There is no
   "helpfully falls back to the cloud".
4. **Nothing is "done" until it builds and the tests run.** A file that
   parses is not finished. A test suite that collected zero tests is not
   evidence.
5. **Deterministic first, model second, human last.**
6. **Every failure is a sentence, never a traceback.**
7. **Optional dependencies degrade, never crash** — and the cost of each
   absence is stated.
8. **Provenance is not optional.** Provider, model, prompt hash, attempt,
   verification outcome, timestamp — for every artefact.
9. **The public API is the Ports and the shared types**, versioned with
   semver.
10. **The trust model is stated, not implied.**

## The trust model, stated plainly

Running model-generated code is not an accident of this design — **it is the
product.** Building and testing generated code means executing it on your
machine. The layers of defence, in order:

- `guard.py`'s static screen — **a screen against accidents, not a security
  boundary.** A determined adversary defeats a regex; a model that
  misunderstood the task does not.
- the host's `ExecPort` sandboxing policy — the host decides what
  "sandboxed" means for it;
- a scrubbed environment: no inherited API keys, no proxy variables;
- the project-root jail: no write lands outside your project, in any mode,
  judged on resolved real paths;
- the approval gate: **the library default is approval-required.**

Every one of these exists; none is sufficient alone; and **none of it is a
defence against a hostile model.** What it is: a screen against mistakes,
operated by a human who stays in charge.

---

## What it does

Every obligation in the specification's conformance list has a test
(`docs/CONFORMANCE.md`); the suite is about 1,400 tests and runs without a
model.

| | |
|---|---|
| **The contract** | `ports.py`, `types.py`, a Null implementation of every Port, the reusable conformance kit |
| **Installers** | `install.bat`, `install.sh`, `ccoder doctor` |
| **Deterministic layer** | 17 languages, compiler-output parsing, the static screen, attributable build/run/test phases |
| **Edits** | transactions with sequence numbers, byte-identical undo, encoding and line-ending preservation |
| **Providers** | any OpenAI-compatible endpoint, in-process GGUF, five remote providers behind a per-session gate with every outbound message redacted |
| **CodeMap** | SQLite symbol index, call graph, blast radius, semantic zoom, five agentic tools |
| **The loop** | generate → verify → repair, truncation continuation, deterministic pre-fixes, cycle detection; a reply that repeats itself or goes in circles is caught while it streams |
| **Planning** | skeleton-first, dependency order derived from imports, replanning after each file, resume from the journal, the interview that pins open decisions before the plan |
| **Review** | deterministic scanners, then one structured model pass, then the Recommendation Document |
| **Narrowed edits** | a repair or change returns only the functions it rewrites, put back by name, so long files stay repairable |
| **The dropped-code guard** | a reply that leaves out code the file had, or writes `# ... rest unchanged`, is refused before it is written |
| **FIM holes** | a model that fills in the middle repairs only the body the error is in — probed at the model check, never trusted by name |
| **Sample before repair** | a failed first attempt is sampled again at a new seed before any repair; where the samples disagree is the repair's doubt map |
| **The values at the failure** | the locals of the frame that failed, printed under every test failure by a stdlib-only runner and shown to the repair |
| **The type checker's gate** | mypy / basedpyright / pyright in the loop: a wrong attribute, a wrong call or an undefined name fails the attempt before a test has to reach it |
| **Model profiles** | the model card's sampling, edit format, FIM tokens and thinking switch, per family; `ccoder bench --calibrate` measures a model's own |
| **The size rule** | modules of ~200 lines and functions of ~40, asked of the planner and measured on what came back |
| **Structured output** | a grammar-backed JSON ask for hosts that need a typed answer |
| **Deployed skills** | project guidance in `.ccoder/skills/*.md` — house style, testing conventions, forbidden patterns — loaded into the cached prompt prefix and recorded by content hash in the journal |
| **Audit** | `ccoder audit`: tools first, then one bounded model read per file; findings ranked with file and line, and a plan `build` can carry out |
| **The regression suite** | `ccoder bench`: tasks graded by hidden tests, each run compared with the last, replay of recorded replies with no model |
| **ATK adapter** | six Port implementations and a workspace panel, tested without Qt |

**One line about the review stage.** On the worked example the model
reported "reads fine to me" while the deterministic scanner found the AWS key
it was sitting on. That is the whole argument for asking tools first.

## Languages

Python, C, C++, Rust, Java, Go, C#, JavaScript, TypeScript, **GDScript**,
Bash, PowerShell, Lua, Ruby, SQL (SQLite), Zig, Batch.

GDScript is first class: syntax checking via `--check-only`, GUT and gdUnit4
detection, Godot's own error formats parsed, and — because it matters — **a
headless pass on rendering-dependent code is never reported as unqualified
success.**

## Documentation

- `docs/PORTS.md` — the contract: each Port, with examples and what a host
  must guarantee
- `docs/EMBEDDING.md` — hosting it, including vendoring as a git submodule,
  threading, and configuring a real machine
- `docs/PROVIDERS.md` — models, endpoints and the offline default
- `docs/AUDIT.md` — auditing an existing project, and the fix loop
- `docs/NARROWED_EDITS.md` — how an existing file is changed
- `docs/FIM_MODE.md` — holes for an ordinary code model
- `docs/BENCH.md` — the regression suite, `ccoder bench`, and replay
- `docs/SPEED.md` — where the minutes go, and speculative decoding on the
  inference server
- `docs/CONFORMANCE.md` — the obligations and the tests that assert them
- `COGNITIVE_CODER_BUILD_SPEC_v1.1.md` — the full specification this
  implements, including the reasoning behind every constraint
- `CHANGELOG.md` — every release, from the first commit

## License

Apache-2.0. Permissive enough for anyone to vendor, and — unlike MIT — it
carries an explicit patent grant, which is worth having for a tool that
generates code.
