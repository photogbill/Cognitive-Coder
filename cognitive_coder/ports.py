# SPDX-License-Identifier: Apache-2.0
"""The Ports — everything the host provides — and a Null implementation of each.

This module and `types.py` together are the public contract (C9, M8). Ports are
`typing.Protocol` classes: a host implements them **structurally**, with no
inheritance and no import of anything from this package. That is not stylistic
purity — it is what lets ParisNeo drop this into LoLLMs without taking a
dependency on our base classes, and it is why C2 exists.

Each method's docstring states two things, deliberately separated:

    *A host may assume …*   what the CORE promises about how it will call you.
    *A host must guarantee …*   what YOU promise, which `tests/port_conformance.py`
                                turns into executable assertions (§9, M54).

If you are writing a host: implement the Ports, then run the conformance kit
against your implementations. You should not have to read the core to know
whether you got it right, and the kit is how that stays true.

**Every Port ships a Null implementation in this module** (M20) so the engine
runs hostless: `NullLLM`, `MemoryFileSystem`, `SubprocessExec`,
`MemoryStorage`, `SilentEvents`, `AutoApprove`. The test suite uses
`MemoryFileSystem` and `SubprocessExec` for real; `examples/tiny_host.py`
drives a whole session on the set of them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import fnmatch
import os
from pathlib import Path
import posixpath
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Protocol, runtime_checkable

from .types import Completion, Message, ModelCapabilities, ProcResult, ToolSpec

# How SubprocessExec keeps a child's output, per stream: the first HEAD bytes
# and a rolling window of the last TAIL, with the gap stated. Both ends,
# because the verdict is at the END — a verbose unittest run kept only its
# first 200 KB, dropped `FAIL:` and `Ran N tests`, and the model was handed
# `test_11109 ... ok` as the diagnostic.
OUTPUT_HEAD = 150_000
OUTPUT_TAIL = 50_000
# Kept per stream: 200 KB is far more than anyone reads. Still exported for
# hosts that quote it; it is now the SUM of the two ends above.
MAX_OUTPUT = OUTPUT_HEAD + OUTPUT_TAIL
# Past this many bytes across both streams the whole tree is killed. The cap
# used to run AFTER `communicate()` had buffered everything, so a four-line
# C `for(;;) puts(...)` OOM-killed the engine at 1.1 s — long before any
# timeout, and with `timeout=0` there was no timeout at all.
OUTPUT_CEILING = 8_000_000


# ==========================================================================
# cancellation (§5.2)
# ==========================================================================

@runtime_checkable
class CancelToken(Protocol):
    """The one thing a host may call from another thread.

    The core is synchronous and a single generation on the target machine
    takes minutes, so a GUI host needs a defined way to stop one. The core
    checks this between phases: before each build/run/test, before each model
    call, between tool round-trips (M21).
    """

    def is_set(self) -> bool: ...


class Cancel:
    """A thread-safe `CancelToken`. Hosts may use this or bring their own."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


class NeverCancelled:
    """The do-nothing token, so callers never have to test for None."""

    def is_set(self) -> bool:
        return False


# ==========================================================================
# the Ports
# ==========================================================================

@runtime_checkable
class LLMPort(Protocol):
    """Messages in, completion out.

    Tool calling and vision are part of the 1.0 contract because the target
    model has both and the loop is built around them (§6.7, §6.9). A host
    whose model has neither says so in `capabilities()` and the core takes its
    fallback paths — that is a supported configuration, not a degraded one.
    """

    def complete(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (),
                 temperature: float = 0.15,
                 max_tokens: int = 2048,
                 stop: Sequence[str] | None = None,
                 grammar: str | None = None,
                 seed: int | None = None,
                 cancel: CancelToken | None = None) -> Completion:
        """Blocking completion.

        *A host may assume*: messages are ordered oldest-first and alternate
        sanely; a `role="tool"` message always carries the `tool_call_id` of a
        call the host reported; the core will not call this concurrently on
        one instance.

        *A host must guarantee*:

        1. **This MUST NOT raise on model refusal** (M11). If the model says
           "I won't do that", return it as text. A refusal is data the loop
           can act on; an exception is a crash the loop cannot.
        2. If `tools` is supplied and the model calls one, `finish_reason` is
           `"tool_calls"` and `Completion.tool_calls` is populated with
           **parsed** arguments (repairing near-JSON if needed, and setting
           `ToolCall.repaired` when you did — D9).
        3. A host whose model lacks tool support **MUST ignore `tools`** and
           report `supports_tools=False` (M12); the core then uses the
           text-marker fallback (§6.7) rather than silently getting nothing.
        4. Honour `cancel` if you can; if you cannot interrupt mid-generation,
           finish the call and the core stops at the next boundary. Defined
           and slow beats undefined and fast.
        5. Set `Completion.model` to what actually answered, every time —
           the host may have swapped models since the last call (§0.1) and
           the journal records it per call (C8).
        """
        ...

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        """Token stream, for display only.

        *A host must guarantee*: chunks are text fragments in order; the cancel
        token is checked between them. A host without streaming may yield one
        chunk — that is explicitly fine, the core never depends on granularity.
        """
        ...

    def capabilities(self) -> ModelCapabilities:
        """What is loaded RIGHT NOW (M13).

        *A host must guarantee*: this reflects the currently loaded model, not
        the configured one. The host may change models between calls (§0.1);
        the core re-reads this at every task boundary and treats a change as
        an epoch boundary (§6.7). Return `ModelCapabilities(name="", …)` when
        nothing is loaded — that is a normal, reportable state (M10), not an
        occasion to raise.
        """
        ...

    def count_tokens(self, text: str) -> int:
        """Exact where you have a tokenizer, a documented estimate otherwise.

        *A host must guarantee*: `capabilities().token_count_is_estimate` says
        which of the two this is, honestly (M14). The core budgets context
        against this number and DECLARES the assumption to the model when it
        is an estimate — an undeclared estimate is how a context overflows.

        Tokenizer dependencies (`mistral-common`, `tiktoken`, …) live in the
        HOST or the provider, never in the core (M14, §10.3).
        """
        ...


@runtime_checkable
class FileSystemPort(Protocol):
    """All file access. Hosts enforce their own jail here.

    The bytes methods are the primitives; the core's `textio` layer (§6.5a)
    owns encoding and EOL handling on TOP of them, so snapshots and undo can
    be byte-identical (M26). A host that "helpfully" normalises line endings
    inside `write_bytes` breaks that guarantee — don't.
    """

    def read_bytes(self, path: str) -> bytes: ...

    def write_bytes(self, path: str, content: bytes) -> None:
        """*A host must guarantee*: this is **atomic** — write to a temp file
        in the same directory, then rename — or your PORTS.md entry says
        plainly that it is not (M15). A half-written source file that still
        parses is the worst possible failure here, because it looks fine.

        *And*: replacing an existing file **keeps its permission bits**
        (and a new file gets the umask's mode, not the temp file's 0600).
        A temp-and-rename that forgets this strips the execute bit from
        every script the engine edits. A failure is raised as an exception
        whose message names the file and says nothing was written.
        """
        ...

    def read(self, path: str) -> str:
        """Convenience: UTF-8, `errors="replace"`. Never raises on encoding."""
        ...

    def write(self, path: str, content: str) -> None: ...

    def exists(self, path: str) -> bool: ...

    def list(self, glob: str) -> list[str]:
        """Paths matching a glob, relative to `root()`, `/`-separated.

        *A host must guarantee*: `.git/` is excluded (M27). The engine never
        runs git and never indexes it.
        """
        ...

    def delete(self, path: str) -> None: ...

    def root(self) -> str:
        """The project root.

        *A host must guarantee*: this is a real, absolute path. The core
        resolves every path it touches to a real path and refuses anything
        that escapes this root — including via `..` and via symlinks (M24).
        The core does that check itself; you are welcome to check again.
        """
        ...


@runtime_checkable
class ExecPort(Protocol):
    """Running a command. The host decides what "sandboxed" means for it (C10)."""

    def run(self, argv: Sequence[str], *, cwd: str, timeout: float,
            stdin: str = "", env: dict | None = None) -> ProcResult:
        """Run one command to completion or to the timeout.

        *A host may assume*: `argv` is a real argument list, never a string —
        a path containing a space breaks string commands in a way that looks
        like a compiler bug, so the core never builds one.

        *A host must guarantee*: **on timeout the ENTIRE process tree is
        killed** (M16). On Windows a terminated shell does not take its
        children with it, and orphaned compilers, test runners and Godot
        instances are a real, observed failure mode — Godot is precisely why
        this clause exists. `timed_out=True` in the result attests that the
        tree is dead, and the conformance kit tests it with a process that
        spawns a child.

        *A host must also guarantee*: **`run()` returns and never raises
        on anything the child prints.** Output held in memory is bounded
        (keep both ends — the verdict of a test run is at the end — and set
        `truncated=True` with the gap stated), a child that floods output is
        stopped even when `timeout` is 0, and bytes that are not valid text
        are replaced rather than raised on. The engine runs generated code;
        a generated program that loops on `printf` must not take the engine
        down with it.
        """
        ...

    def which(self, binary: str) -> str | None:
        """Where a tool is, or None.

        *A host may assume*: the core probes toolchains at RUNTIME through
        this (§6.1) rather than trusting an installer's record — a compiler
        installed the week after install day must simply work.
        """
        ...


@runtime_checkable
class StoragePort(Protocol):
    """Key-value state plus a SQLite path. Hosts choose where state lives."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None:
        """*A host must guarantee*: values are JSON-serialisable (M17). That
        is the portability contract — a host storing pickles cannot hand its
        state to a host that stores JSON, and resume has to survive that."""
        ...

    def sqlite_path(self, name: str) -> str:
        """A filesystem path for a SQLite database with this logical name.

        *A host must guarantee*: the parent directory exists and is writable,
        and the same `name` returns the same path for the life of a project.
        """
        ...


@runtime_checkable
class EventPort(Protocol):
    """Progress, logging and streamed output. Hosts render it."""

    def event(self, kind: str, message: str,
              data: dict | None = None) -> None:
        """*A host may assume*: `kind` is from the closed set in
        `types.EVENT_KINDS` (M19) and `message` is a plain sentence fit to
        show a human (C6). `data` is JSON-serialisable.

        *A host must guarantee*: this does not raise and does not block for
        long. It is called from inside the loop; a slow event handler is a
        slow engine.
        """
        ...


@runtime_checkable
class ApprovalPort(Protocol):
    """Human in the loop. A host may auto-approve; it must say so."""

    def approve_diff(self, summary: str, unified_diff: str) -> bool:
        """*A host must guarantee*: ALL writes route through here — including
        writes the MODEL initiates through the `apply_patch` tool (M18).
        Tool calling must never become a side door around the approval
        default (§6.5 rule 6).

        The library default is approval-required. A host that auto-approves
        must tell its operator that it does, and must keep the snapshot and
        undo machinery that makes auto-apply survivable (§6.5).
        """
        ...

    def approve_remote(self, provider: str, bytes_out: int,
                       estimate: str) -> bool:
        """Called before the FIRST remote call of a session (M42).

        *A host must guarantee*: this asks a human, or the host is documented
        as auto-approving outbound network traffic — which for an air-gapped
        host would be a contradiction worth noticing (C3).
        """
        ...


# ==========================================================================
# Null implementations (M20) — the engine must run with zero host
# ==========================================================================

class NullLLM:
    """An LLMPort that answers nothing, honestly.

    Used by `tiny_host.py` and by any test that needs the shape of a model
    without the cost of one. Reports `name=""`, so the core exercises its
    "no model loaded" path (M10) — which is exactly what you want the default
    to rehearse.
    """

    def __init__(self, name: str = "", context_tokens: int = 8192) -> None:
        self._name = name
        self._ctx = context_tokens
        self.calls: list[tuple] = []          # inspectable in tests

    def complete(self, messages: Sequence[Message], **kw) -> Completion:
        self.calls.append((tuple(messages), kw))
        return Completion(
            text="", finish_reason="error", model=self._name,
            tokens_in=sum(self.count_tokens(m.content) for m in messages))

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        yield ""

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name=self._name, family="none", context_tokens=self._ctx,
            supports_tools=False, supports_grammar=False,
            supports_vision=False, supports_fim=False, is_remote=False,
            token_count_is_estimate=True)

    def count_tokens(self, text: str) -> int:
        # ~3.5 characters per token is the rule of thumb this project uses
        # everywhere it has no tokenizer. Stated, not hidden, because
        # capabilities() flags it as an estimate and the core says so in the
        # prompt when it matters (M14).
        return max(1, len(text or "") // 4)


class ScriptedLLM:
    """An LLMPort that returns canned answers in order. Fakes, not mocks (§9).

    The whole engine must be drivable with zero real models and zero network,
    and this is the thing that makes that true. It is in `ports.py` rather
    than in the tests because hosts want it too — it is how you develop a
    panel without a 14 GB model loaded.

    Exhausting the script is a loud failure, not a quiet empty string: a test
    that silently gets "" from the fifth call is a test that passes for the
    wrong reason.
    """

    def __init__(self, replies: Sequence[Any], *,
                 name: str = "scripted", supports_tools: bool = True,
                 context_tokens: int = 16384) -> None:
        self._replies = list(replies)
        self._name = name
        self._tools = supports_tools
        self._ctx = context_tokens
        self.prompts: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message], **kw) -> Completion:
        self.prompts.append(tuple(messages))
        if not self._replies:
            raise AssertionError(
                "ScriptedLLM ran out of replies — the engine asked for more "
                "completions than the script provides. Add the next expected "
                "reply, or fix the loop that is asking again.")
        reply = self._replies.pop(0)
        if isinstance(reply, Completion):
            return reply
        return Completion(text=str(reply), finish_reason="stop",
                          model=self._name,
                          tokens_in=sum(self.count_tokens(m.content)
                                        for m in messages),
                          tokens_out=self.count_tokens(str(reply)))

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        yield self.complete(messages, **kw).text

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name=self._name, family="mistral", context_tokens=self._ctx,
            supports_tools=self._tools, supports_grammar=True,
            supports_vision=False, supports_fim=False, is_remote=False,
            token_count_is_estimate=True)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text or "") // 4)


class MemoryFileSystem:
    """An in-memory FileSystemPort. What the test suite uses (§5.5).

    Atomicity is trivially satisfied (a dict assignment either happened or did
    not), which is honest rather than convenient: the conformance kit's
    atomicity test is aimed at REAL hosts, and this one passes it because it
    genuinely cannot tear.
    """

    def __init__(self, files: dict[str, bytes] | None = None,
                 root: str = "/project") -> None:
        self._root = root
        self.files: dict[str, bytes] = dict(files or {})

    # -- helpers ---------------------------------------------------------
    def _key(self, path: str) -> str:
        """Normalise to a root-relative, `/`-separated key.

        Refusing escapes here as well as in the patcher is belt and braces,
        and M24 says "in any mode" — a jail with one door is not a jail.

        This door used to be painted on: `normpath("/" + p)` turns
        `/../config.py` into `/config.py`, so an escape was CLAMPED into
        the root — `../config.py` overwrote the project's own `config.py`
        and `/etc/passwd` became `etc/passwd`. Escapes and foreign absolute
        paths are now refused.
        """
        p = str(path).replace("\\", "/")
        root = self._root.replace("\\", "/").rstrip("/")
        if root and (p == root or p.startswith(root + "/")):
            p = p[len(root):].lstrip("/")
        elif p.startswith("/") or (len(p) > 1 and p[1] == ":"):
            raise ValueError(
                f"{path!r} is outside the project root ({self._root}); "
                f"nothing was written.")
        p = posixpath.normpath(p) if p else "."
        if p == ".." or p.startswith("../"):
            raise ValueError(
                f"{path!r} escapes the project root ({self._root}); "
                f"nothing was written.")
        return "" if p == "." else p

    # -- the port --------------------------------------------------------
    def read_bytes(self, path: str) -> bytes:
        key = self._key(path)
        if key not in self.files:
            raise FileNotFoundError(f"{path} is not in this project")
        return self.files[key]

    def write_bytes(self, path: str, content: bytes) -> None:
        self.files[self._key(path)] = bytes(content)

    def read(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def write(self, path: str, content: str) -> None:
        self.write_bytes(path, content.encode("utf-8"))

    def exists(self, path: str) -> bool:
        try:
            return self._key(path) in self.files
        except ValueError:
            return False

    def list(self, glob: str) -> list[str]:
        pattern = _norm_glob(glob)
        out = [p for p in sorted(self.files)
               if not _in_git(p) and (
                   fnmatch.fnmatch(p, pattern)
                   or fnmatch.fnmatch(posixpath.basename(p), pattern))]
        return out

    def delete(self, path: str) -> None:
        self.files.pop(self._key(path), None)

    def root(self) -> str:
        return self._root


class LocalFileSystem:
    """A real, atomic FileSystemPort rooted at one directory.

    Offered because every host needs this and writing it correctly is fiddly:
    the temp file must be in the SAME directory as the target (rename is only
    atomic within a filesystem), `.git/` must be excluded from listing (M27),
    and containment must be judged on RESOLVED real paths so a symlink cannot
    walk out (M24).
    """

    def __init__(self, root: str) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        raw = Path(path)
        target = raw if raw.is_absolute() else self._root / raw
        # resolve(strict=False): the file may not exist yet, but its RESOLVED
        # location still has to be inside the root — that is what stops a
        # symlinked directory from being a way out.
        real = target.resolve()
        if real != self._root and self._root not in real.parents:
            raise ValueError(
                f"{path!r} resolves outside the project folder "
                f"({self._root}); nothing was written.")
        return real

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_bytes(self, path: str, content: bytes) -> None:
        """Atomic replace that keeps the target's permission bits.

        `mkstemp` creates 0600, and `os.replace` puts THAT file in the
        target's place — so every edit used to reset the mode: a 755
        `build.sh` lost its execute bit, a group-readable file stopped
        being one, and a 444 file the operator made read-only came back
        writable. An existing target's mode is copied onto the temp file
        before the swap; a new file is created with 0666 and the umask
        applied by the kernel, like any `open()`.
        """
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            keep = stat.S_IMODE(os.stat(target).st_mode)
        except OSError:
            keep = None
        fd, tmp = _create_temp_beside(target)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            if keep is not None:
                os.chmod(tmp, keep)
            try:
                os.replace(tmp, target)      # atomic within a filesystem
            except PermissionError as exc:
                # Windows refuses to replace a file that is open elsewhere
                # or marked read-only; POSIX, a folder we cannot write.
                raise PermissionError(
                    exc.errno,
                    f"{path} could not be replaced ({exc.strerror or exc}): "
                    f"it is open in another program, read-only, or in a "
                    f"folder that cannot be written. Nothing was written; "
                    f"close it or clear the read-only flag and try "
                    f"again.") from exc
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def read(self, path: str) -> str:
        return self.read_bytes(path).decode("utf-8", errors="replace")

    def write(self, path: str, content: str) -> None:
        self.write_bytes(path, content.encode("utf-8"))

    def exists(self, path: str) -> bool:
        try:
            return self._resolve(path).exists()
        except (ValueError, OSError):
            return False

    def list(self, glob: str) -> list[str]:
        out = []
        for p in sorted(self._root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self._root).as_posix()
            if _in_git(rel):
                continue
            pattern = _norm_glob(glob)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name,
                                                                pattern):
                out.append(rel)
        return out

    def delete(self, path: str) -> None:
        target = self._resolve(path)
        if target.is_file():
            target.unlink()

    def root(self) -> str:
        return str(self._root)


class _Capture:
    """One stream's output: a head, a rolling tail, and a byte count.

    Memory is bounded by construction — at most HEAD + 2×TAIL + one chunk
    per stream, however long the child runs — which is the whole point:
    the old path buffered everything and capped afterwards.
    """

    def __init__(self, head: int, tail: int) -> None:
        self.head_limit = head
        self.tail_limit = tail
        self.head = bytearray()
        self.tail = bytearray()
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.head_limit - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail += chunk
            if len(self.tail) > 2 * self.tail_limit:
                del self.tail[:-self.tail_limit]

    @property
    def truncated(self) -> bool:
        return self.total > len(self.head) + min(len(self.tail),
                                                 self.tail_limit)

    def text(self) -> str:
        tail = bytes(self.tail[-self.tail_limit:]) if self.tail else b""
        if not self.truncated:
            return _decode(bytes(self.head) + tail)
        omitted = self.total - len(self.head) - len(tail)
        return (_decode(bytes(self.head))
                + f"\n… {omitted:,} bytes omitted — the first "
                  f"{len(self.head):,} and the last {len(tail):,} are kept …"
                  f"\n" + _decode(tail))


def _decode(raw: bytes) -> str:
    """Child output → text, never raising (the old path raised on 0xFF).

    UTF-8 first, since that is what nearly every toolchain emits; then the
    locale's codec strictly, which is what `text=True` used to mean and what
    a Windows compiler writing cp1252 produces; then UTF-8 with
    replacement. Line endings are normalised the way text-mode pipes did,
    because every parser downstream matches on `\\n`.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        import locale
        enc = (locale.getpreferredencoding(False) or "utf-8")
        text = ""
        if enc.lower().replace("-", "").replace("_", "") != "utf8":
            try:
                text = raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                text = ""
        text = text or raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pump(pipe: Any, sink: _Capture, flood: threading.Event,
          captures: tuple[_Capture, ...], ceiling: int) -> None:
    """Drain one pipe into its capture until EOF. Runs on its own thread.

    Two threads, one per stream, because a child blocked writing stderr
    while we wait on stdout is a deadlock, and `select` does not work on
    Windows pipes. Keeps reading after the ceiling is hit so the child is
    never blocked on a full pipe; the tree is being killed anyway.
    """
    read = getattr(pipe, "read1", pipe.read)
    try:
        while True:
            chunk = read(65536)
            if not chunk:
                break
            sink.add(chunk)
            if sum(c.total for c in captures) > ceiling:
                flood.set()
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


class SubprocessExec:
    """A real ExecPort that kills the whole process tree on timeout (M16).

    The tree-kill is the reason this class exists rather than a two-line
    `subprocess.run` wrapper. On POSIX we `setsid` and signal the process
    GROUP; on Windows we create a job-like process group and fall back to
    `taskkill /T /F`, because a terminated cmd.exe cheerfully leaves its
    children running — which is how an orphaned Godot instance ends up
    holding a file lock nobody can explain an hour later.

    The tree-kill is BEST EFFORT, and it is worth being exact about where
    it stops. A descendant that calls `setsid()` / `start_new_session` (or,
    on Windows, is created outside the job) has left the group and survives
    the kill. When such a process also holds the output pipe open, `run()`
    no longer waits on the pipe for it: it waits a short grace period,
    returns what was captured so far, and says so. The old behaviour —
    wait five more seconds on the pipe, then throw the output away — cost
    the one thing the operator needed. A host that must contain hostile
    code needs an OS sandbox (a job object, a container, a cgroup); this
    class is a screen against accidents, like `guard`.

    Output is read on two threads into a bounded head and a rolling tail,
    decoded without ever raising, and a child that floods past
    `OUTPUT_CEILING` bytes is killed even when `timeout` is 0.
    """

    #: How long, after the child has gone, to wait for its pipes to close.
    #: A descendant still holding them past this is a stray (see above).
    DRAIN_GRACE = 1.5

    def __init__(self, scrub_env: bool = True,
                 output_ceiling: int = OUTPUT_CEILING) -> None:
        self.scrub_env = scrub_env
        # A host whose test suites are legitimately chattier raises this;
        # memory stays bounded either way, since only the two ends are kept.
        self.output_ceiling = max(1, int(output_ceiling))

    def run(self, argv: Sequence[str], *, cwd: str, timeout: float,
            stdin: str = "", env: dict | None = None) -> ProcResult:
        argv = [str(a) for a in argv]
        if env is None and self.scrub_env:
            # `scrub_env` was stored and never read, so `env=None` meant
            # Popen's "inherit everything". The runner always passes an
            # environment; the review stage's scanners did not, and ran
            # with the operator's API keys and proxy variables — the one
            # thing the README's trust model says a child never gets.
            from .runner import scrubbed_env  # noqa: PLC0415 — no cycle
            env = scrubbed_env(cwd, cwd)
        t0 = time.monotonic()
        kwargs: dict = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True          # own process group
        else:
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd,
                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=env, **kwargs)
        except (OSError, ValueError) as exc:
            return ProcResult(exit_code=-1, stderr=f"could not run: {exc}",
                              duration_s=time.monotonic() - t0)

        out = _Capture(OUTPUT_HEAD, OUTPUT_TAIL)
        err = _Capture(OUTPUT_HEAD, OUTPUT_TAIL)
        flood = threading.Event()
        readers = [threading.Thread(target=_pump,
                                    args=(pipe, sink, flood, (out, err),
                                          self.output_ceiling),
                                    daemon=True, name="ccoder-pump")
                   for pipe, sink in ((proc.stdout, out),
                                      (proc.stderr, err))]
        for t in readers:
            t.start()
        if stdin:
            threading.Thread(target=_feed, args=(proc.stdin, stdin),
                             daemon=True, name="ccoder-stdin").start()

        # 0 (or negative) means WAIT. The operator asked for no ceiling on
        # time; the ExecPort is where that has to be honoured, because every
        # phase timeout funnels through here. The OUTPUT ceiling still
        # applies — "wait forever" never meant "buffer forever".
        patience = timeout if timeout and timeout > 0 else None
        deadline = t0 + patience if patience else None
        stopped = ""                        # "" | "timeout" | "flood"
        while True:
            try:
                proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                pass
            if flood.is_set():
                stopped = "flood"
                break
            if deadline is not None and time.monotonic() >= deadline:
                stopped = "timeout"
                break
        if stopped:
            self._kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        # The child is gone; its pipes close unless a descendant still holds
        # them. Wait a little, then stop waiting and say why.
        grace_end = time.monotonic() + self.DRAIN_GRACE
        for t in readers:
            t.join(timeout=max(0.0, grace_end - time.monotonic()))
        stray = any(t.is_alive() for t in readers)
        if stray and not stopped:
            # Returned while descendants still hold its output: those are
            # orphans in the making. Same group, same kill.
            self._kill_tree(proc)
            for t in readers:
                t.join(timeout=0.5)
            stray = any(t.is_alive() for t in readers)

        notes: list[str] = []
        if stopped == "timeout":
            notes.append(
                f"cognitive-coder: this was still running after "
                f"{timeout:.0f}s and the whole process tree was killed. This "
                f"clock is on the PROGRAM, not on the model that wrote it. "
                f"Two ordinary reasons a program never finishes here: it is "
                f"waiting for input, and nothing is typed in; or it has a "
                f"main loop — a game, a server, a window — and is behaving "
                f"correctly.")
        elif stopped == "flood":
            notes.append(
                f"cognitive-coder: this printed more than "
                f"{self.output_ceiling:,} bytes of output, so the whole "
                f"process tree was stopped. A program producing this much "
                f"output is almost always looping; the start and the end of "
                f"what it printed are kept above.")
        if stray:
            notes.append(
                "cognitive-coder: a process this command started detached "
                "itself (a new session) and was still holding the output "
                "open, so it may still be running outside the process tree "
                "that was killed. The output above is what arrived before "
                "that.")
        err_text = err.text()
        if notes:
            err_text = (err_text + "\n" if err_text else "") + "\n".join(
                notes)
        code = proc.returncode
        if stopped or code is None:
            code = -9
        return ProcResult(exit_code=code, stdout=out.text(), stderr=err_text,
                          duration_s=time.monotonic() - t0,
                          timed_out=stopped == "timeout",
                          truncated=out.truncated or err.truncated
                          or stopped == "flood")

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Kill the process AND its descendants. Best effort, in order."""
        if os.name == "posix":
            # The group id is the child's pid (start_new_session), and it is
            # used directly rather than via getpgid: once the child has been
            # reaped getpgid fails, yet its group lives on in any
            # descendant still running — exactly the one to kill.
            for pgid in _pgids(proc):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    return
                except (ProcessLookupError, PermissionError, OSError):
                    continue
        else:
            taskkill = shutil.which("taskkill")
            if taskkill:
                try:
                    subprocess.run([taskkill, "/T", "/F", "/PID",
                                    str(proc.pid)],
                                   capture_output=True, timeout=10)
                    return
                except Exception:                        # noqa: BLE001
                    pass
        try:
            proc.kill()
        except Exception:                                # noqa: BLE001
            pass

    def which(self, binary: str) -> str | None:
        return shutil.which(binary)


def _create_temp_beside(target: Path) -> tuple[int, str]:
    """A new, exclusive temp file in the target's own folder: (fd, path).

    Not `mkstemp`, which forces 0600: `os.open` with 0o666 lets the kernel
    apply the umask, so a NEW file ends up with the mode any other program
    would have given it — without calling `os.umask()`, which is
    process-wide and would race the output-reader threads.
    """
    import secrets
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        tmp = str(target.parent / f".cc-{secrets.token_hex(6)}.tmp")
        try:
            return os.open(tmp, flags, 0o666), tmp
        except FileExistsError:
            continue
    raise FileExistsError(
        f"could not create a temporary file beside {target.name} in "
        f"{target.parent}; nothing was written.")


def _pgids(proc: subprocess.Popen) -> list[int]:
    out = [proc.pid]
    try:
        live = os.getpgid(proc.pid)
        if live != proc.pid:
            out.insert(0, live)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    return out


def _feed(pipe: Any, text: str) -> None:
    """Write stdin on its own thread and close it.

    A child that never reads stdin would otherwise block this write once
    the pipe fills, and with it the whole engine — past the timeout, since
    the clock is checked on the main thread. Encoded as text-mode pipes
    used to encode it (the locale codec, `\\n` → the platform's EOL), so
    the SQL runner and anything else fed on stdin see no change.
    """
    import locale
    enc = locale.getpreferredencoding(False) or "utf-8"
    if os.name == "nt":
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    try:
        pipe.write(text.encode(enc, errors="replace"))
        pipe.flush()
    except (OSError, ValueError):
        pass                    # the child exited or closed stdin: fine
    finally:
        try:
            pipe.close()
        except (OSError, ValueError):
            pass


def _in_git(rel: str) -> bool:
    """Is this path inside a `.git` directory — in any letter case?

    Case-insensitively, because on Windows and macOS `.GIT/` and `.Git/`
    ARE `.git/`, and M27 excludes git's directory, not a spelling of it.
    """
    return any(part.lower() == ".git"
               for part in str(rel).replace("\\", "/").split("/")[:-1])


def _norm_glob(glob: str) -> str:
    """Normalise a glob to a root-relative, `/`-separated pattern.

    Written out rather than done inline with `lstrip("./")`, because that is
    a trap: `lstrip` takes a SET OF CHARACTERS, not a prefix, so
    `".cc_journal/*.jsonl".lstrip("./")` yields `"cc_journal/*.jsonl"` — the
    leading dot is eaten and every dotted directory silently stops matching.
    The journal lives in `.cc_journal/`, so this bug makes resume find no
    previous sessions while looking, from the outside, like there simply
    were none.
    """
    pattern = str(glob or "").replace("\\", "/")
    while pattern.startswith("./"):
        pattern = pattern[2:]
    return pattern.lstrip("/")


class MemoryStorage:
    """An in-memory StoragePort. SQLite databases go to a temp directory.

    JSON round-trips on `set` deliberately: it is the cheapest possible way to
    catch a host storing something unserialisable (M17) at the moment it does
    it, rather than three days later when resume fails.

    The temp directory is made only when a database is first asked for, as a
    private subfolder of ONE process-wide `TemporaryDirectory` that is
    removed at exit. It used to be a fresh `/tmp/ccoder-*` per construction,
    never removed — thousands accumulated on a CI box, most for instances
    that never opened a database. An explicit ``base_dir`` is the host's to
    keep and is never deleted.
    """

    _shared: tempfile.TemporaryDirectory | None = None
    _shared_lock = threading.Lock()

    def __init__(self, base_dir: str | None = None) -> None:
        import json
        self._json = json
        self._data: dict[str, Any] = {}
        self._dir: Path | None = None
        if base_dir:
            self._dir = Path(base_dir)
            self._dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _shared_root(cls) -> Path:
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = tempfile.TemporaryDirectory(
                    prefix="ccoder-", ignore_cleanup_errors=True)
            return Path(cls._shared.name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        try:
            self._json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"StoragePort values must be JSON-serialisable; {key!r} is "
                f"not ({exc}).") from exc
        self._data[key] = value

    def sqlite_path(self, name: str) -> str:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="storage-",
                                              dir=self._shared_root()))
        return str(self._dir / f"{name}.sqlite3")


class SilentEvents:
    """An EventPort that discards. The default when a host does not care."""

    def event(self, kind: str, message: str,
              data: dict | None = None) -> None:
        return None


class RecordingEvents:
    """An EventPort that remembers. What the test suite asserts against (§9)."""

    def __init__(self, echo: bool = False) -> None:
        self.events: list[tuple[str, str, dict]] = []
        self.echo = echo

    def event(self, kind: str, message: str,
              data: dict | None = None) -> None:
        self.events.append((kind, message, dict(data or {})))
        if self.echo:
            print(f"[{kind}] {message}", file=sys.stderr)

    def kinds(self) -> list[str]:
        return [k for k, _, _ in self.events]

    def of(self, kind: str) -> list[tuple[str, str, dict]]:
        return [e for e in self.events if e[0] == kind]


class AutoApprove:
    """An ApprovalPort that says yes — and records that it did.

    The library default is approval-REQUIRED (§6.5); this exists for tests,
    for `tiny_host.py`, and for hosts that have made auto-apply an explicit,
    warned, advanced setting. `approve_remote` defaults to **False** even
    here, because C3 is the one constraint that does not get a convenient
    default: silently approving outbound network traffic is precisely the
    thing an air-gapped host promised would not happen.
    """

    def __init__(self, remote: bool = False) -> None:
        self.remote_ok = remote
        self.diffs: list[tuple[str, str]] = []
        self.remote_asks: list[tuple[str, int, str]] = []

    def approve_diff(self, summary: str, unified_diff: str) -> bool:
        self.diffs.append((summary, unified_diff))
        return True

    def approve_remote(self, provider: str, bytes_out: int,
                       estimate: str) -> bool:
        self.remote_asks.append((provider, bytes_out, estimate))
        return self.remote_ok


class DenyAll:
    """An ApprovalPort that refuses everything — the honest library default.

    A new host, or a first run, must never silently write to someone's
    project. Wiring this in by default means a host that forgot to implement
    approval gets "nothing was written, because nothing was approved", which
    is a bug report rather than a disaster.
    """

    def approve_diff(self, summary: str, unified_diff: str) -> bool:
        return False

    def approve_remote(self, provider: str, bytes_out: int,
                       estimate: str) -> bool:
        return False


# --------------------------------------------------------------------------
# the bundle a Session is handed
# --------------------------------------------------------------------------

class Host:
    """The six Ports, together, with Null defaults for anything omitted.

    Not itself a Port — a convenience so a host, a test, or `tiny_host.py`
    can say `Host(llm=…, fs=…)` and get working defaults for the rest. The
    core takes this, or the individual Ports; both are supported.
    """

    def __init__(self, *, llm: LLMPort | None = None,
                 fs: FileSystemPort | None = None,
                 exec: ExecPort | None = None,          # noqa: A002
                 storage: StoragePort | None = None,
                 events: EventPort | None = None,
                 approval: ApprovalPort | None = None) -> None:
        self.llm: LLMPort = llm or NullLLM()
        self.fs: FileSystemPort = fs or MemoryFileSystem()
        self.exec: ExecPort = exec or SubprocessExec()
        self.storage: StoragePort = storage or MemoryStorage()
        self.events: EventPort = events or SilentEvents()
        # Approval-required is the library default (§6.5). A host that wants
        # auto-apply passes AutoApprove() explicitly, having warned its user.
        self.approval: ApprovalPort = approval or DenyAll()

    def emit(self, kind: str, message: str, data: dict | None = None) -> None:
        """Fire an event without every call site needing a try/except.

        An EventPort that raises must not take the build down with it — the
        host's progress bar is not more important than the operator's code.
        """
        try:
            self.events.event(kind, message, data)
        except Exception:                                # noqa: BLE001
            pass
