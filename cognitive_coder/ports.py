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
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Protocol, runtime_checkable

from .types import Completion, Message, ModelCapabilities, ProcResult, ToolSpec

# Output past this is truncated. 200 KB is far more than anyone reads and
# small enough that a runaway `while(1) printf` cannot exhaust memory.
MAX_OUTPUT = 200_000


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
        """
        p = str(path).replace("\\", "/")
        if p.startswith(self._root.replace("\\", "/")):
            p = p[len(self._root):]
        p = posixpath.normpath("/" + p.lstrip("/"))
        if p.startswith("/.."):
            raise ValueError(f"{path!r} escapes the project root")
        return p.lstrip("/")

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
               if not p.startswith(".git/") and (
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
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(target.parent),
                                   prefix=".cc-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)          # atomic within a filesystem
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
            if rel.startswith(".git/") or "/.git/" in rel:
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


class SubprocessExec:
    """A real ExecPort that kills the whole process tree on timeout (M16).

    The tree-kill is the reason this class exists rather than a two-line
    `subprocess.run` wrapper. On POSIX we `setsid` and signal the process
    GROUP; on Windows we create a job-like process group and fall back to
    `taskkill /T /F`, because a terminated cmd.exe cheerfully leaves its
    children running — which is how an orphaned Godot instance ends up
    holding a file lock nobody can explain an hour later.
    """

    def __init__(self, scrub_env: bool = True) -> None:
        self.scrub_env = scrub_env

    def run(self, argv: Sequence[str], *, cwd: str, timeout: float,
            stdin: str = "", env: dict | None = None) -> ProcResult:
        argv = [str(a) for a in argv]
        t0 = time.monotonic()
        kwargs: dict = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True          # own process group
        else:
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, env=env, **kwargs)
        except (OSError, ValueError) as exc:
            return ProcResult(exit_code=-1, stderr=f"could not run: {exc}",
                              duration_s=time.monotonic() - t0)
        # 0 (or negative) means WAIT — `communicate` reads that as None. The
        # operator asked for no ceiling; the ExecPort is where that has to be
        # honoured, because every phase timeout funnels through here.
        patience = timeout if timeout and timeout > 0 else None
        try:
            out, err = proc.communicate(input=stdin or None, timeout=patience)
            timed_out = False
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:                            # noqa: BLE001
                out, err = "", ""
            err = (err or "") + (
                f"\ncognitive-coder: this was still running after "
                f"{timeout:.0f}s and the whole process tree was killed. This "
                f"clock is on the PROGRAM, not on the model that wrote it. "
                f"Two ordinary reasons a program never finishes here: it is "
                f"waiting for input, and nothing is typed in; or it has a "
                f"main loop — a game, a server, a window — and is behaving "
                f"correctly.")
            timed_out = True
        out, cut_a = _cap(out or "")
        err, cut_b = _cap(err or "")
        return ProcResult(exit_code=proc.returncode if not timed_out else -9,
                          stdout=out, stderr=err,
                          duration_s=time.monotonic() - t0,
                          timed_out=timed_out, truncated=cut_a or cut_b)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """Kill the process AND its descendants. Best effort, in order."""
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
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


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT:
        return text, False
    return (text[:MAX_OUTPUT] + f"\n… truncated at {MAX_OUTPUT:,} characters. "
            f"A program producing this much output is usually looping.", True)


class MemoryStorage:
    """An in-memory StoragePort. SQLite databases go to a temp directory.

    JSON round-trips on `set` deliberately: it is the cheapest possible way to
    catch a host storing something unserialisable (M17) at the moment it does
    it, rather than three days later when resume fails.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        import json
        self._json = json
        self._data: dict[str, Any] = {}
        self._dir = Path(base_dir or tempfile.mkdtemp(prefix="ccoder-"))
        self._dir.mkdir(parents=True, exist_ok=True)

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
