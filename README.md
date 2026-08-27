# Cognitive Coder

**A host-agnostic engine that lets a language model write, build, test and fix
real code in a real project — designed so it still works when the model is
small and the machine is offline.**

It is a **library first** and an application second. The core is pure Python
with **no GUI dependency, no host dependency, and zero required runtime
dependencies**. It embeds in a Qt desktop app, a FastAPI server, or a
terminal, and it runs on a disconnected laptop in a field office.

```
git clone https://github.com/photogbill/cognitive-coder
cd cognitive-coder
install.bat          # Windows
./install.sh         # Linux
.venv/bin/ccoder doctor
```

It is **not published to PyPI**. Clone it and run the installer; the
installers are part of the product, not an afterthought.

---

## What it needs

- **Python 3.11 or later.** Tested on 3.11 on Windows and Linux; expected to
  work on later versions; earlier versions are refused at install.
  If your machine has nothing suitable, the installer fetches 3.11 **into the
  clone** — nothing is installed system-wide, and deleting the clone removes
  every trace.
- **A model.** Anything with an OpenAI-compatible endpoint: llama.cpp server,
  Ollama, LM Studio, vLLM, LiteLLM. Or a GGUF loaded in-process through
  `llama-cpp-python`. The engine is calibrated against **Devstral Small 2 24B**
  at 16 GB of VRAM, but nothing about it is specific to that.
- **Optionally, a toolchain.** `ccoder doctor` prints exactly which languages
  are usable right now and what each missing one costs. Nothing fails because
  you lack Rust; that language is simply unavailable and it says so.

## Ten lines of embedding

Runs anywhere. No model, no network, no host application.

```python
from cognitive_coder import (AutoApprove, Host, LocalFileSystem,
                             ScriptedLLM, Session)

host = Host(llm=ScriptedLLM(['```python\ndef greet(n):\n    return f"hi {n}"\n```']),
            fs=LocalFileSystem("/path/to/project"),
            approval=AutoApprove())
session = Session(host)
session.run("a greet function")
print(session.report())
```

`examples/tiny_host.py` is that grown just enough to prove the embedding
story end to end. It is the first thing to read if you are writing a host.

---

## Where it wins, and where it loses

Asked honestly whether this can be "as good as Cline, or better", the truthful
answer has three parts, and the design is built around it.

**Where it genuinely wins**

| Axis | Why |
|---|---|
| **Works offline, on small models** | Cline assumes a frontier model behind an API. Every decision here assumes a 7B–24B local model and compensates with scaffolding. |
| **Host-agnostic** | Cline is a VS Code extension. This is a library that embeds in a Qt desktop app, a FastAPI web app, or a terminal. |
| **Audit-grade provenance** | Every line produced is traceable to a model, a prompt, an attempt number and a verification result. For anyone who has to defend a change, that is decisive. |
| **Deterministic-first** | Linters, compilers and test runners answer what they can answer; the model is asked only what tools cannot decide. Cheaper, faster, more trustworthy. |

**Where it loses, and there is no point pretending otherwise**

- Raw code quality when Cline is driving a frontier model. A local 24B
  narrows the gap; it does not close it.
- Ecosystem polish. Cline has an editor's whole UI, inline decorations, and
  years of interaction design.

So this is not "Cline but ours". It is the thing Cline cannot be: a coding
engine that runs on a disconnected laptop, embeds in somebody else's
application, and can prove what it did.

---

## The meta-lesson

**With a frontier model you improve results by improving the prompt. With a
small model you improve results by improving the *loop*.**

Every hour spent on verification, feedback quality and error localisation is
worth ten spent on prompt wording. That is why the largest and most carefully
tested module here is the one that parses compiler output, and why "it didn't
work" is never an answer this engine gives.

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
4. **Nothing is "done" until it builds and the tests run.** A file that parses
   is not finished. A test suite that collected zero tests is not evidence.
5. **Deterministic first, model second, human last.**
6. **Every failure is a sentence, never a traceback.**
7. **Optional dependencies degrade, never crash** — and the cost of each
   absence is stated.
8. **Provenance is not optional.** Provider, model, prompt hash, attempt,
   verification outcome, timestamp — for every artefact.
9. **The API surface is frozen at 1.0** and versioned with semver.
10. **The trust model is stated, not implied.** See below.

## The trust model, stated plainly

Running model-generated code is not an accident of this design — **it is the
product.** Building and testing generated code means executing it on your
machine. The layers of defence, in order:

- `guard.py`'s static screen — **a screen against ACCIDENTS, not a security
  boundary.** A determined adversary defeats a regex; a model that
  misunderstood the task does not.
- the host's `ExecPort` sandboxing policy — the host decides what "sandboxed"
  means for it;
- a scrubbed environment: no inherited API keys, no proxy variables;
- the project-root jail: no write lands outside your project, in any mode,
  judged on resolved real paths;
- the approval gate: **the library default is approval-required.**

Every one of these exists; none is sufficient alone; and **none of it is a
defence against a hostile model.** What it is: a screen against mistakes,
operated by a human who stays in charge.

---

## What is built

All ten phases of the build specification. Every one of Appendix H's 55
numbered obligations is met; 48 have a test that asserts it
(`docs/CONFORMANCE.md`).

| | |
|---|---|
| **The contract** | `ports.py`, `types.py`, a Null implementation of every Port, the reusable conformance kit |
| **Installers** | `install.bat`, `install.sh`, `ccoder doctor` |
| **Deterministic layer** | 17 languages, compiler-output parsing, the static screen, attributable build/run/test phases |
| **Edits** | transactions with sequence numbers, byte-identical undo, encoding and line-ending preservation |
| **Providers** | any OpenAI-compatible endpoint, in-process GGUF, and the gate that keeps remote off |
| **CodeMap** | SQLite symbol index, call graph, blast radius, semantic zoom, five agentic tools |
| **The loop** | generate → verify → repair, truncation continuation, deterministic pre-fixes, cycle detection |
| **Planning** | skeleton-first, dependency order derived from imports, replanning after each file, resume from the journal |
| **Review** | deterministic scanners, then one structured model pass, then the Recommendation Document — with the non-independence line where it belongs |
| **Remote** | five providers behind a per-session gate, every outbound message redacted including tool results, budgets that halt |
| **ATK** | six Port implementations, a workspace panel, and a dry-run-by-default migration for the six modules this engine grew out of |

**The review stage says something worth knowing about C5.** On the worked
example the model reported "reads fine to me" while the deterministic scanner
found the AWS key it was sitting on. That is the whole argument for asking
tools first, in one line of output.

## Deployed skills

Project guidance that travels with the code. `ccoder skills deploy` seeds
`.ccoder/skills/` with three editable markdown files — house style, testing
conventions, forbidden patterns — and every session loads whatever is there
into the cached prompt prefix as project conventions. The idea is borrowed
from deepseek-cowork's deployed skill templates; the reason it fits HERE is
one that project never had: a small local model needs scaffolding, and
guidance that lives beside the code is versioned, diffed and reviewed like
the code, and present on the disconnected laptop where a settings screen
is not.

```
ccoder skills deploy     # seed the starter pack (never overwrites)
ccoder skills            # what a session would load, and what it skipped
ccoder skills new NAME   # scaffold a new one
ccoder build ... --no-skills   # opt out for one run
```

A skill may scope itself with `lang: python` in its header. Sorted filename
order is priority order; an oversized skill is skipped by name, never
truncated mid-rule. The journal records each active skill's content hash at
session start, so a generated line is traceable to the exact revision of
the guidance that shaped it.

## Languages

Python, C, C++, Rust, Java, Go, C#, JavaScript, TypeScript, **GDScript**,
Bash, PowerShell, Lua, Ruby, SQL (SQLite), Zig, Batch.

GDScript is first class, not an outline-only afterthought: syntax checking via
`--check-only`, GUT and gdUnit4 detection, Godot's own error formats parsed,
and — because it matters — **a headless pass on rendering-dependent code is
never reported as unqualified success.**

## Documentation

- `docs/PORTS.md` — the contract, with examples, and what each Port must
  guarantee
- `docs/EMBEDDING.md` — how to host it, including vendoring as a git submodule
- `docs/PROVIDERS.md` — models, endpoints, and the offline default
- `COGNITIVE_CODER_BUILD_SPEC_v1.1.md` — the full specification this
  implements, including the reasoning behind every constraint

## License

Apache-2.0. Permissive enough for anyone to vendor, and — unlike MIT — it
carries an explicit patent grant, which is worth having for a tool that
generates code.
