# The Ports — the contract

Six Protocols and the dataclasses they carry. Together they are the public
API, frozen at 1.0 under semver: **breaking either the Ports or the shared
types is a major version.** Freezing one without the other freezes half a
contract, because a host implementing `LLMPort` depends on `Message` and
`Completion` just as hard as on the method signatures.

Ports are `typing.Protocol` classes. **A host implements them structurally**
— no inheritance, no import of anything from this package. That is what lets
this drop into somebody else's application without them taking a dependency
on our base classes.

## Before you read any further

Implement the Ports, then run the conformance kit against your own
implementations:

```python
from tests.port_conformance import check_all

report = check_all(fs=MyFileSystem("/project"), exec=MyExec(),
                   llm=MyLLM(), storage=MyStorage())
print(report.text())
assert report.ok
```

**You should not have to read the core to know whether you got it right.**
The kit is how that stays true — it turns every "a host must guarantee"
below into an executable assertion.

---

## LLMPort

Messages in, completion out. Tool calling and vision are part of the 1.0
contract because the target model has both and the loop is built around them.
A host whose model has neither says so in `capabilities()` and the core takes
its fallback paths — a supported configuration, not a degraded one.

```python
def complete(self, messages, *, tools=(), temperature=0.15, max_tokens=2048,
             stop=None, grammar=None, seed=None, cancel=None) -> Completion
def stream(self, messages, **kw) -> Iterator[str]
def capabilities(self) -> ModelCapabilities
def count_tokens(self, text: str) -> int
```

**A host must guarantee:**

1. **`complete()` MUST NOT raise on a model refusal.** If the model says "I
   won't do that", return it as text. A refusal is data the loop can act on;
   an exception is a crash it cannot.
2. If `tools` is supplied and the model calls one, `finish_reason` is
   `"tool_calls"` and `Completion.tool_calls` carries **parsed** arguments —
   repairing near-JSON if needed, and setting `ToolCall.repaired` when you
   did. A model that needs repairing on every call is telling you something,
   and a silent repair destroys the message.
3. **A host whose model lacks tool support MUST ignore `tools`** and report
   `supports_tools=False`. The core then uses the text-marker fallback rather
   than silently getting nothing back.
4. **`capabilities()` reflects the model loaded RIGHT NOW**, not the
   configured one. The core re-reads it at every task boundary and treats a
   change as a cache-invalidating epoch boundary.
5. **"No model loaded" is a normal state**, reported as
   `ModelCapabilities(name="", …)` — not an exception.
6. `count_tokens` is exact where you have a tokenizer and a documented
   estimate otherwise, and `token_count_is_estimate` says honestly which.
   **Tokenizer dependencies live in your code, never in the core.**
7. `Completion.model` names what actually answered, every time. The journal
   records it per call.
8. `Completion.prompt_ms` — prompt-processing time — where you can measure
   it. It is the only signal that the prompt prefix cache broke, and a broken
   cache is otherwise completely silent: everything just gets slower.

---

## FileSystemPort

All file access. Hosts enforce their own jail here.

```python
def read_bytes(self, path) -> bytes
def write_bytes(self, path, content: bytes) -> None
def read(self, path) -> str            # UTF-8, errors="replace"
def write(self, path, content: str) -> None
def exists(self, path) -> bool
def list(self, glob) -> list[str]
def delete(self, path) -> None
def root(self) -> str
```

**A host must guarantee:**

1. **`write_bytes` is atomic** — write a temp file in the *same directory*,
   then rename — **or your entry here says plainly that it is not.** A
   half-written source file that still parses is the worst possible failure,
   because nothing notices.
2. **Bytes in, bytes out, untouched.** Do not "helpfully" normalise line
   endings inside `write_bytes`. The core's `textio` layer owns encoding and
   EOL handling on top of your primitives, and byte-identical undo depends on
   you not interfering.
3. **`list()` excludes `.git/`.** The engine never runs git and never indexes
   it.
4. **`root()` is a real, absolute path.** The core resolves every path it
   touches to a real path and refuses anything escaping this root — including
   via `..` and via symlinks. You are welcome to check again.

`LocalFileSystem` in `ports.py` is a correct implementation you may copy: the
temp file is in the same directory (rename is only atomic within a
filesystem), and containment is judged on resolved real paths.

---

## ExecPort

Running a command. **The host decides what "sandboxed" means for it.**

```python
def run(self, argv, *, cwd, timeout, stdin="", env=None) -> ProcResult
def which(self, binary) -> str | None
```

**A host must guarantee:**

1. **On timeout the ENTIRE process tree is killed.** On Windows a terminated
   shell does not take its children with it, and orphaned compilers, test
   runners and Godot instances are a real, observed failure — Godot is
   precisely why this clause exists. `timed_out=True` attests the tree is
   dead, and the conformance kit tests it with a process that spawns a child.
2. `argv` is a real argument list, never a string. The core never builds one:
   a path containing a space breaks string commands in a way that looks like
   a compiler bug.

**A host may assume:** the core probes toolchains at runtime through
`which()` rather than trusting an installer's record, so a compiler installed
the week after install day simply works.

---

## StoragePort

```python
def get(self, key, default=None) -> Any
def set(self, key, value) -> None
def sqlite_path(self, name) -> str
```

**A host must guarantee:** values are **JSON-serialisable**. That is the
portability contract — a host storing pickles cannot hand its state to a host
storing JSON, and resume has to survive that. `sqlite_path` returns a stable,
writable path for a given logical name, with its parent directory present.

---

## EventPort

```python
def event(self, kind: str, message: str, data: dict | None = None) -> None
```

`kind` is a **closed set at 1.0** — new kinds are a minor version, renamed
kinds are a major one, because hosts render them:

| kind | meaning | typical data |
|---|---|---|
| `phase` | loop phase change | `{"phase": "build", "task": "src/x.py", "attempt": 2}` |
| `token` | streamed model output | `{"text": "…"}` |
| `status` | one-line human status | — |
| `diagnostic` | a parsed Diagnostic | the dataclass as a dict |
| `patch` | a transaction applied or rolled back | `{"seq": 12, "files": [...]}` |
| `remote` | remote mode on/off, bytes out, redaction count | `{"provider": …, "enabled": true}` |
| `warning` | degraded mode, headless caveat, estimate in use | — |
| `error` | a plain sentence | a journal reference for the traceback |
| `budget` | wall-clock / token / spend checkpoint | `{"remaining_s": 300}` |

**A host must guarantee:** this does not raise, and does not block for long.
It is called from inside the loop; a slow event handler is a slow engine.
`message` is always a plain sentence fit to show a human.

---

## ApprovalPort

Human in the loop. A host may auto-approve; **it must say so.**

```python
def approve_diff(self, summary: str, unified_diff: str) -> bool
def approve_remote(self, provider: str, bytes_out: int, estimate: str) -> bool
```

**A host must guarantee:**

1. **ALL writes route through `approve_diff`** — including writes the *model*
   initiates through the `apply_patch` tool. Tool calling must never become a
   side door around the approval default, so the side door is simply not
   built: the tool hands its edit to the loop's transaction, which calls you.
2. **The library default is approval-required.** A new host, or a first run,
   must never silently write to someone's project. If you auto-apply, tell
   your operator, and keep the snapshot and undo machinery that makes it
   survivable. If you ever find yourself deleting the snapshot step, delete
   the auto-apply option first.
3. `approve_remote` is called before the **first** remote call of a session,
   showing what is about to leave the machine.

---

## Cancellation

```python
class CancelToken(Protocol):
    def is_set(self) -> bool: ...
```

The core is synchronous and one generation on a local model takes minutes, so
a GUI host needs a defined way to stop. `Session.cancel()` is **the one method
a host may call from another thread.** The core checks the token between
phases: before each build, run and test, before each model call, between tool
round-trips.

A cancelled task leaves resumable state, and any open transaction is rolled
back. That is a guarantee of the engine, not a hope. A provider that cannot
interrupt mid-generation finishes the call and the core stops at the next
boundary — slower, but defined.

---

## The Null implementations

Every Port ships one, so the engine runs with zero host:

| | |
|---|---|
| `NullLLM` | answers nothing, honestly — reports no model loaded |
| `ScriptedLLM` | canned replies in order; the whole engine is drivable with it |
| `MemoryFileSystem` | in-memory; what most tests use |
| `LocalFileSystem` | a real, atomic, jailed filesystem you may copy |
| `SubprocessExec` | real execution with a genuine process-tree kill |
| `MemoryStorage` | in-memory, with a JSON round-trip check on every `set` |
| `SilentEvents` / `RecordingEvents` | discard, or remember for assertions |
| `AutoApprove` / `DenyAll` | yes-and-record, or the honest default |

`Host(...)` bundles the six with Null defaults for anything omitted — and
defaults `approval` to `DenyAll`, so a host that forgot to implement approval
gets "nothing was written, because nothing was approved". A bug report rather
than a disaster.
