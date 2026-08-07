# Embedding Cognitive Coder

Three ways in, in increasing order of commitment. All three are supported and
all three are tested.

## 1. Vendoring — a git submodule, no install at all

The lightest option, and the one to reach for if you would rather not manage
another dependency.

```bash
git submodule add https://github.com/photogbill/cognitive-coder vendor/cognitive-coder
```

```python
import sys
sys.path.insert(0, "vendor/cognitive-coder")
import cognitive_coder            # works. no install, no metadata.
```

This path is protected by tests, because it is easy to break by accident:

- **No import-time side effects.** Importing the package must not create a
  directory, open a database, read a config file or probe a toolchain. A test
  imports it in an empty directory and asserts the directory is still empty.
- **No package-metadata reads.** `importlib.metadata.version(...)` raises for
  a vendored copy. The version comes from `version.py`, and a test walks the
  AST to prove it.
- **Zero required runtime dependencies.** A test walks every module and fails
  on any non-stdlib import at module level.

If you vendor, pin to a tag. Without PyPI, **the git tag is the release.**

## 2. `pip install -e .` from a clone

What both installers do internally, and what a contributor runs.

```bash
git clone https://github.com/photogbill/cognitive-coder
pip install -e ./cognitive-coder
```

Optional extras, each buying exactly one capability:
`[llamacpp]`, `[treesitter]`, `[tokenizers]`, `[dev]`.

## 3. The installers

`install.bat` / `install.sh` build a `.venv` inside the clone, install the
core, detect toolchains, and print an honest summary. Nothing global changes;
deleting the clone removes every trace. If the machine has no usable Python
they fetch 3.11 **into the clone** rather than failing.

---

## Writing a host

A host is six objects. Here is a complete one:

```python
from cognitive_coder import Host, Session, SessionConfig

class MyEvents:
    def event(self, kind, message, data=None):
        my_panel.append(f"[{kind}] {message}")

class MyApproval:
    def approve_diff(self, summary, unified_diff):
        return my_dialog.confirm(summary, unified_diff)

    def approve_remote(self, provider, bytes_out, estimate):
        return my_dialog.confirm_network(provider, bytes_out, estimate)

host = Host(
    llm=MyLLMPort(my_already_loaded_model),   # wrap what you have
    fs=MyFileSystem(project_root),
    exec=MyExec(),                            # or SubprocessExec()
    storage=MyStorage(),
    events=MyEvents(),
    approval=MyApproval(),
)

session = Session(host, config=SessionConfig(lang="python"))
for outcome in session.run("add CSV import to the report module"):
    print(outcome.summary())
print(session.report())
```

Nothing there inherits from anything of ours. That is the point.

**Then run the conformance kit** (`tests/port_conformance.py`) against your
implementations. It covers atomicity, process-tree kill, the jail, and
capabilities honesty — the four things §9 names, and the four most likely to
be subtly wrong.

### Wrapping a model you already have

The most common case, and the cheapest: the host has a model loaded, and
`LLMPort` is a thin shim over it.

```python
class MyLLMPort:
    def __init__(self, engine):
        self.engine = engine

    def complete(self, messages, *, tools=(), temperature=0.15,
                 max_tokens=2048, stop=None, grammar=None, seed=None,
                 cancel=None):
        # MUST NOT raise on a refusal — return the text.
        text, timings = self.engine.generate(
            messages, temperature=temperature, max_tokens=max_tokens,
            stop=stop, grammar=grammar, cancel=cancel)
        return Completion(
            text=text,
            finish_reason="length" if timings.hit_limit else "stop",
            tokens_in=timings.prompt_tokens, tokens_out=timings.gen_tokens,
            model=self.engine.model_name,      # per call: it may have changed
            prompt_ms=timings.prompt_ms)       # the cache-health signal

    def stream(self, messages, **kw):
        yield from self.engine.stream(messages)

    def capabilities(self):
        return ModelCapabilities(
            name=self.engine.model_name or "",     # "" when nothing is loaded
            family="mistral", context_tokens=self.engine.n_ctx,
            supports_tools=self.engine.has_tools,
            supports_grammar=True,
            token_count_is_estimate=False)         # a real tokenizer

    def count_tokens(self, text):
        return len(self.engine.tokenize(text))
```

Two details that are easy to miss and expensive later:

- **`model` per call, not per session.** The host may have swapped models
  between calls, and the journal records which one produced each artefact.
- **`prompt_ms` where you can measure it.** It is the only signal that the
  prompt prefix cache broke, and a broken cache is silent — everything simply
  gets slower, forever.

---

## Threading

The core is **synchronous and single-threaded**, deliberately: a GUI host has
its own concurrency model and the engine should not impose a second one.

Run a session on a worker thread. **`Session.cancel()` is the only method
safe to call from another thread**, and it is safe by construction — the
cancel token is a `threading.Event`. Everything the UI needs to render
arrives through `EventPort`, on the worker thread, so marshal it to your UI
thread the way your framework expects.

```python
# Qt sketch — the pattern, not the API
class SessionRunner(QRunnable):
    def run(self):
        self.session.run(self.request)

# from the UI thread, at any time:
self.session.cancel()
```

---

## What the engine does to your project

Two directories, both inside the project root, both yours to delete:

- **`.cc_snapshots/`** — `NNNN-<task_id>/` per transaction, with the original
  bytes of every file touched and a `MANIFEST.txt` holding the diff. This is
  what undo restores from, which is why undo is byte-identical rather than
  approximately right.
- **`.cc_journal/`** — one append-only JSONL per session. Provenance, and the
  source of truth for resume.

**It never runs git.** If your project is a repository with uncommitted
changes it says so once, at session start, and does nothing about it. A tool
that quietly makes commits in someone's repository is indistinguishable from
a mess.

`ccoder history` — or `session.history()` — answers "what did it just do to
my project" from the transaction log: sequence, task, files, whether it was
verified, whether it was later rolled back.

---

## Configuring for a real machine

The defaults are Appendix G.8's **starting point to measure from**, not a
recommendation to freeze:

```
n_ctx            16384      raise once prefix caching is proven
type_k/type_v    q8_0       halve the KV cost before touching offload
offload_kqv      True       flip to False only if you need >32k
temperature      0.15       generation · 0.35 planning and review
reserved output  2048 tokens
work unit        250–300 lines
```

At 16 GB the operator faces a real trade: **fast and blinkered** (all layers
on GPU, smaller context) versus **slower and better-sighted** (KV tricks or
offload, larger context). For multi-file coding, seeing more usually wins —
but not always, and it depends on the codebase. Expose it, state the trade in
one sentence, and let the journal answer it after a few sessions.
`session.journal.stats()` records context size, token counts, timings and
first-attempt success rate, so the right answer becomes a query rather than
an argument.

## Model swapping is yours, not ours

**The core contains no swap logic.** It asks `capabilities()` what is loaded
and works with the answer. If your host offers a swap button:

- treat a model change as an **epoch boundary** — the engine does this for
  you when it notices, rebuilding the cached prompt prefix, because the KV
  cache died with the old model;
- **guide the operator to swap at most once per session, at a phase
  boundary.** Alternating per task pays 30–60 seconds of prompt reprocessing
  every round trip;
- if you want to preserve a prefix across a swap-and-return,
  `LocalLlamaCpp.save_state()` / `load_state()` is offered for exactly that —
  offered to you, never driven by the core.
