# SPDX-License-Identifier: Apache-2.0
"""ATK's six Port implementations. Destination: `atk/core/ccoder_host.py`.

**This file lives outside `cognitive_coder/` on purpose** (§3.1). It knows
about ATK; the core must not. A CI test walks the core and fails the build if
anything under `cognitive_coder/**` ever imports from here.

It is deliberately **Qt-free**. Everything that needs a widget lives in
`ccoder_panel.py`; this module takes plain callables, so it can be tested
without a QApplication and so `Session` can be driven from a `QRunnable`
without the Ports knowing what thread they are on.

The one thing worth reading before the code: **ATK's `sandbox.py` blacklists
`subprocess` in GENERATED code, but Cognitive Coder must run compilers as the
host.** Conflating those two would make the engine unable to build anything
at all. They are different things and this file keeps them apart —
`ATKExec` runs build tools with ATK's scrubbed environment; the *generated
code* is still screened by `cognitive_coder.guard` before it is compiled.

Installation into ATK, in order, each step leaving the suite green (§7.3):

    1. run `python adapters/atk/migrate.py --atk <ATK> --dry-run` from the
       CC clone and read what it would write
    2. run it with `--apply` — it installs this file as
       `atk/core/ccoder_host.py`, the panel as `atk/ui/ccoder_panel.py` and
       `atk_compat.py` as `atk/core/ccoder_compat.py`, and shims the six
    3. run ATK's full suite
    4. only then update ATK's imports and delete the re-export shims

Nothing here writes to ATK's `state.db`. A second database FILE is fine; a
second schema in the same file is not (§7.1).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import sys
import threading
import time
from typing import Any

# The engine itself imports nothing from ATK, which is why this import is
# one-directional and safe.
from cognitive_coder import (
    Completion,
    Message,
    ModelCapabilities,
    ProcResult,
    Session,
    SessionConfig,
)
from cognitive_coder.ports import Host, SubprocessExec
from cognitive_coder.runner import scrubbed_env

#: Where the adapter records what it cannot say through a port — chiefly a
#: port callable that itself raised. ATK's own logging configuration decides
#: where that lands.
_log = logging.getLogger(__name__)


# ==========================================================================
# LLMPort → atk.core.llm_engine.LLMEngine
# ==========================================================================

class ATKLLM:
    """Binds `LLMPort` to the model ATK already has loaded.

    It does NOT load anything. ATK owns loading, unloading and the 16 GB
    ceiling — the cognitive core and Whisper are mutually exclusive, and the
    swap button belongs to ATK (§0.1, §7.1). This class asks what is loaded
    and works with the answer.

    `events` is an optional `(kind, message)` callable — the host's
    EventPort, in practice — for the things worth saying that are not
    failures of the call: nothing loaded, a tokenizer that broke, ATK's
    think-splitter failing. None of them may raise out of here (M11).
    """

    name = "atk"
    is_remote = False

    def __init__(self, engine: Any, *, n_ctx_default: int = 16384,
                 events: Callable[[str, str], None] | None = None) -> None:
        self.engine = engine
        self._n_ctx_default = n_ctx_default
        self._events = events
        self._tokenizer: Any = None
        #: Which model the tokenizer decision was made for. A swap in ATK's
        #: Setup page changes the answer: a Mistral tokenizer kept after a
        #: switch to Qwen counts the wrong vocabulary, EXACTLY, and says it
        #: is not an estimate.
        self._tokenizer_for: str | None = None
        self._warned: set[str] = set()
        self.last_prompt_ms = 0

    def _say(self, message: str, *, once: str = "") -> None:
        if once:
            if once in self._warned:
                return
            self._warned.add(once)
        if self._events is not None:
            try:
                self._events("warning", message)
            except Exception:                            # noqa: BLE001
                # The EventPort contract says it does not raise; if it does
                # anyway, the build matters more than the message.
                _log.exception("the events callable raised")

    # -- generation -------------------------------------------------------
    def complete(self, messages: Sequence[Message], *, tools: Sequence = (),
                 temperature: float = 0.15, max_tokens: int = 2048,
                 stop: Sequence[str] | None = None,
                 grammar: str | None = None, seed: int | None = None,
                 cancel: Any = None) -> Completion:
        """Blocking completion, drained from ATK's streaming API.

        `LLMEngine` exposes `chat_stream` rather than a blocking call, so the
        stream is drained here — which is also where the cancel token gets
        checked between chunks, giving ATK's Stop button a response time of
        one token rather than one generation.

        **Prefill and decode are timed apart.** The stream's first chunk is
        the boundary: everything before it is prompt processing (`prompt_ms`,
        the number G.7.5 reads to tell whether the prefix cache held), and
        everything after it is generation (`decode_ms`). This adapter used to
        report the whole call as `prompt_ms` and `decode_ms=0` — so on the
        one host the timings were measured on, the cache check read a number
        that tracked output length instead.

        **This never raises on a model refusal** (M11). An engine error comes
        back as `finish_reason="error"` with the sentence on the EventPort,
        because a refusal is data the loop can act on and an exception is a
        crash it cannot.
        """
        if cancel is not None and cancel.is_set():
            return Completion(text="", finish_reason="cancelled",
                              model=self._model_name())

        if not getattr(self.engine, "is_loaded", False):
            # M10: a normal, reportable state — not an exception.
            self._say("No model is loaded in ATK, so nothing was generated. "
                      "Load one in Setup.", once="unloaded")
            return Completion(text="", finish_reason="error", model="")

        # ATK's chat template has no tool role, so a tool result becomes a
        # user turn IN PLACE. It was appended after every other message,
        # which moved the answer to a question below the questions asked
        # after it.
        payload = [{"role": "user", "content": f"[tool result]\n{m.content}"}
                   if m.role == "tool" else
                   {"role": m.role, "content": m.content}
                   for m in messages]

        t0 = time.monotonic()
        first: float | None = None
        chunks: list[str] = []
        cancelled = False
        failed: Exception | None = None
        try:
            for token in self.engine.chat_stream(
                    payload, temperature=temperature, max_tokens=max_tokens):
                if first is None:
                    first = time.monotonic()
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                chunks.append(token)
        except Exception as exc:                         # noqa: BLE001
            # llama-cpp can raise anything from inside a generator; M11
            # says it comes back as data, and the sentence goes out.
            failed = exc
        end = time.monotonic()
        prompt_ms = int(((first if first is not None else end) - t0) * 1000)
        decode_ms = int((end - first) * 1000) if first is not None else 0
        self.last_prompt_ms = prompt_ms
        raw = "".join(chunks)
        if failed is not None:
            self._say(f"ATK's model stopped with an error mid-reply "
                      f"({failed}); the attempt is reported as failed.")
            return Completion(text=raw, finish_reason="error",
                              model=self._model_name(), prompt_ms=prompt_ms,
                              decode_ms=decode_ms)

        answer = self._split_think(raw)
        # Tokens GENERATED, reasoning included: that is what `max_tokens`
        # bounds. D1 wants truncation read from the finish reason, and ATK's
        # stream carries none, so it is inferred from the one number that
        # decides it. It was inferred from characters (len >= 3.2 per token)
        # instead, which called a complete 7,200-character file at 2,048
        # tokens "length" while the same Completion said tokens_out=1,800.
        generated = self.count_tokens(raw) if raw else 0
        finish = ("cancelled" if cancelled else
                  "length" if max_tokens and generated >= max_tokens else
                  "stop")
        return Completion(
            text=answer, finish_reason=finish,
            tokens_in=self.count_tokens(
                "\n".join(m.content for m in messages)),
            tokens_out=generated, model=self._model_name(),
            prompt_ms=prompt_ms, decode_ms=decode_ms)

    def _split_think(self, raw: str) -> str:
        """ATK's `split_think`, when ATK is there; the raw reply otherwise.

        It handles `[THINK]…[/THINK]` (Magistral) and `<think>…</think>`,
        closed or unclosed. The core strips think tags too, but doing it here
        means ATK's richer handling wins and the reasoning stays available
        for the panel (D13, M37). Outside ATK — the CC clone, the tests —
        the import fails and that is expected, not worth a word.
        """
        try:
            from atk.core.llm_engine import split_think
        except ImportError:
            return raw
        try:
            _reasoning, answer = split_think(raw)
        except Exception as exc:                         # noqa: BLE001
            self._say(f"ATK's split_think failed ({exc}); the reply is used "
                      f"as it came and the core strips think tags itself.",
                      once="split_think")
            return raw
        return answer

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        cancel = kw.pop("cancel", None)
        try:
            for token in self.engine.chat_stream(
                    payload, temperature=kw.get("temperature", 0.15),
                    max_tokens=kw.get("max_tokens", 2048)):
                if cancel is not None and cancel.is_set():
                    return
                yield token
        except Exception as exc:                         # noqa: BLE001
            self._say(f"ATK's model stopped with an error mid-stream "
                      f"({exc}).")
            return

    # -- capabilities -----------------------------------------------------
    def capabilities(self) -> ModelCapabilities:
        """What is loaded RIGHT NOW (M13).

        Re-read by the core at every task boundary, which is exactly where
        ATK's swap button gets pressed. An empty name means nothing is
        loaded, and that is a normal state (M10) — it is what the panel shows
        when Whisper has the VRAM.
        """
        loaded = bool(getattr(self.engine, "is_loaded", False))
        meta = dict(getattr(self.engine, "metadata", {}) or {})
        name = meta.get("model_file", "") if loaded else ""
        family = _family(name)
        return ModelCapabilities(
            name=name, family=family,
            context_tokens=int(meta.get("n_ctx", self._n_ctx_default) or
                               self._n_ctx_default),
            # llama.cpp's chat handlers do tools; whether the loaded MODEL
            # was trained for them is the real question, and the family is
            # the best available answer without asking it.
            supports_tools=family in ("mistral", "qwen", "llama"),
            supports_grammar=True,           # GBNF is why ATK uses llama.cpp
            supports_vision=bool(getattr(self.engine, "has_vision", False)),
            supports_fim="devstral" in name.lower() or
                         "codestral" in name.lower(),
            is_remote=False,
            token_count_is_estimate=self._tokenizer_is_estimate())

    def _model_name(self) -> str:
        meta = getattr(self.engine, "metadata", {}) or {}
        return str(meta.get("model_file", "")) if getattr(
            self.engine, "is_loaded", False) else ""

    # -- token counting ---------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """EXACT, because ATK already installs `mistral-common` (§7.2).

        M14 puts the tokenizer in the host, never the core, and this is what
        that buys: the core budgets against a real number instead of paying
        a safety margin of several hundred tokens on every call.
        """
        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text or ""))
            except Exception as exc:                     # noqa: BLE001
                self._say(f"The Mistral tokenizer failed ({exc}); counting "
                          f"with the loaded model's own tokenizer instead.",
                          once="tokenizer")
        try:
            return len(self.engine._llm.tokenize((text or "").encode("utf-8")))
        except Exception:                                # noqa: BLE001
            # No llama.cpp handle (unloaded, or a stub engine): the stated
            # estimate, which capabilities() declares as one (M14).
            return max(1, len(text or "") // 4)

    def _get_tokenizer(self) -> Any:
        model = self._model_name()
        if self._tokenizer_for == model:
            return self._tokenizer
        self._tokenizer_for = model
        self._tokenizer = None
        if _family(model) != "mistral":
            return None
        try:
            from mistral_common.tokens.tokenizers.mistral import (
                MistralTokenizer,
            )
        except ImportError:
            return None             # not installed: the fallback is stated
        try:
            self._tokenizer = MistralTokenizer.v3().instruct_tokenizer
        except Exception as exc:                         # noqa: BLE001
            self._say(f"mistral-common is installed but its tokenizer could "
                      f"not be built ({exc}); token counts fall back.",
                      once="tokenizer-build")
        return self._tokenizer

    def _tokenizer_is_estimate(self) -> bool:
        if self._get_tokenizer() is not None:
            return False
        return not hasattr(getattr(self.engine, "_llm", None), "tokenize")


def _family(name: str) -> str:
    lower = (name or "").lower()
    for key, family in (("devstral", "mistral"), ("magistral", "mistral"),
                        ("mistral", "mistral"), ("codestral", "mistral"),
                        ("qwen", "qwen"), ("llama", "llama"),
                        ("gemma", "gemma"), ("phi", "phi")):
        if key in lower:
            return family
    return "unknown"


# ==========================================================================
# FileSystemPort → the project root, or SANDBOX_DIR in scratchpad mode
# ==========================================================================

class ATKFileSystem:
    """Atomic writes, a real jail, and `.git/` excluded from listing.

    Copied in spirit from `cognitive_coder.ports.LocalFileSystem`, which is
    the reference implementation — the differences are that this one honours
    ATK's SANDBOX_DIR for scratchpad mode and reports through ATK's status
    line when it refuses something.
    """

    def __init__(self, root: str | Path, *,
                 on_refusal: Callable[[str], None] | None = None) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._on_refusal = on_refusal

    def _resolve(self, path: str) -> Path:
        raw = Path(path)
        target = raw if raw.is_absolute() else self._root / raw
        # resolve(strict=False): the file may not exist yet, but its RESOLVED
        # location must still be inside the root — that is what stops a
        # symlinked directory being a way out (M24).
        real = target.resolve()
        if real != self._root and self._root not in real.parents:
            message = (f"Refused to touch {path!r}: it resolves outside the "
                       f"project folder ({self._root}). Nothing was written.")
            if self._on_refusal:
                self._on_refusal(message)
            raise ValueError(message)
        return real

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_bytes(self, path: str, content: bytes) -> None:
        """Atomic (M15): temp file in the SAME directory, then rename.

        The same directory matters — `os.replace` is only atomic within one
        filesystem, and a temp file in %TEMP% is frequently on another
        volume on a Windows machine with a separate data drive.
        """
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # The target's permission bits survive the swap (docs/PORTS.md,
        # FileSystemPort 5): mkstemp makes 0600, and os.replace put THAT in
        # the target's place, so a 755 script lost its execute bit on every
        # edit. A new file gets the ordinary 0666-less-umask default.
        try:
            keep: int | None = stat.S_IMODE(os.stat(target).st_mode)
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
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def append_bytes(self, path: str, data: bytes) -> None:
        """The optional append (docs/PORTS.md): the journal and the build
        log cost what they add, instead of a read-and-rewrite per event."""
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "ab") as fh:
            fh.write(data)

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
        import fnmatch

        pattern = str(glob or "").replace("\\", "/")
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.lstrip("/")
        out: list[str] = []
        for p in sorted(self._root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(self._root).as_posix()
            # M27: the engine never runs git and never indexes it.
            if rel.startswith(".git/") or "/.git/" in rel:
                continue
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


# ==========================================================================
# ExecPort → subprocess with ATK's scrubbed env and a Windows tree-kill
# ==========================================================================

class ATKExec:
    """Runs build tools AS THE HOST. Not the same thing as ATK's sandbox.

    `atk/core/sandbox.py` blacklists `subprocess` in *generated code*, and
    should keep doing so. This class exists because the engine has to invoke
    `gcc` — and conflating the two rules would make it unable to build
    anything (§7.1).

    The generated code is still screened before it gets here, by
    `cognitive_coder.guard`, which is a screen against ACCIDENTS and not a
    security boundary. Nothing in this file should be described as one.
    """

    def __init__(self, *, extra_env: dict | None = None) -> None:
        self._extra = dict(extra_env or {})
        self._exec = SubprocessExec(scrub_env=False)

    def run(self, argv: Sequence[str], *, cwd: str, timeout: float,
            stdin: str = "", env: dict | None = None) -> ProcResult:
        """Run one command; kill the whole tree at the timeout (M16).

        `timeout` 0 (or less) means WAIT, as `types.Timeouts` defines it and
        `ports.SubprocessExec` honours it. It was handed to `communicate`
        as 0, which expires at once: a two-line script came back killed,
        exit -9, "this exceeded 0s" — for an operator who had asked for no
        limit at all.

        No `env` (None or empty) means the SCRUBBED environment, never the
        inherited one. `env or {}` became `env=None` at Popen, which means
        "inherit everything": API keys, tokens and proxy variables reached
        every scanner the review stage runs, and a proxy variable is a
        network path C3 does not allow.
        """
        environment = dict(env) if env else scrubbed_env(cwd, cwd)
        environment.update(self._extra)
        # The running itself is the core's. This class kept its own copy
        # of the capture code, and with it every defect the core has since
        # lost: `communicate()` buffered all output, so a generated program
        # printing in a loop ran ATK out of memory before its timeout; the
        # locale codec raised on a single invalid byte; and the head-only
        # cap cut away the FAIL line of a long test run. One
        # implementation, tested once, is the fix — the policy that is
        # ATK's (its environment, its extra variables) stays here.
        return self._exec.run(argv, cwd=cwd, timeout=timeout, stdin=stdin,
                              env=environment)

    def which(self, binary: str) -> str | None:
        return shutil.which(binary)


def _create_temp_beside(target: Path) -> tuple[int, str]:
    """An exclusive temp file in `target`'s directory, mode 0666 − umask.

    Not `mkstemp`, which always makes 0600: a NEW file written through it
    would be private to its owner, unlike anything `open()` makes. And not
    by reading the umask either — `os.umask` can only be read by setting
    it, and that is process-wide, in a process full of threads.
    """
    import uuid

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    while True:
        tmp = str(target.parent / f".cc-{uuid.uuid4().hex[:12]}.tmp")
        try:
            return os.open(tmp, flags, 0o666), tmp
        except FileExistsError:
            continue


# ==========================================================================
# StoragePort → ctx.settings["ccoder"] + ATK's DATA_DIR
# ==========================================================================

class ATKStorage:
    """Maps onto ATK's settings dict and data directory (§7.1), per project.

    Three rules, all load-bearing:

      * **Never write to ATK's `state.db`.** A second database FILE is fine;
        a second schema in the same file is not. `sqlite_path` returns a
        separate file under ATK's data dir.
      * **The core never reads ATK settings directly.** It has no way to —
        it holds this object and nothing else, which is the point of C2.
      * **One bucket per PROJECT.** The bucket and the SQLite directory were
        keyed by name alone, so project B opened project A's codemap and
        appended to A's transaction log. They are keyed by the first 12 hex
        digits of the SHA-256 of the resolved project root:
        `settings["ccoder"]["projects"][<key>]` and `<DATA_DIR>/ccoder/<key>/`.

    **The worker thread never touches `ctx.settings`.** `set()` was called
    from the build's worker thread and called `save_settings()` every time,
    with failures swallowed: dozens of whole-settings writes per task, racing
    the GUI thread's own, and a failed save invisible. Now `get`/`set` work
    on a private copy under a lock, and `flush()` — which the panel calls on
    the GUI thread after each patch and at the end of a build — writes it
    back and saves once, and SAYS so through `report` when the save fails.
    A crash mid-build therefore loses at most the log rows since the last
    flush; the snapshots and the journal are on disk regardless.
    """

    def __init__(self, ctx: Any, data_dir: str | Path,
                 project_root: str | Path, *, namespace: str = "ccoder",
                 report: Callable[[str], None] | None = None) -> None:
        self._ctx = ctx
        self._namespace = namespace
        self._report = report
        self.root = str(Path(project_root).expanduser().resolve())
        self.key = project_key(self.root)
        self._dir = Path(data_dir) / "ccoder" / self.key
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Read without creating anything: construction must not change
        # ATK's settings, only a flush may.
        bucket = (getattr(ctx, "settings", None) or {}).get(namespace) or {}
        saved = (bucket.get("projects") or {}).get(self.key) or {}
        self._data: dict[str, Any] = json.loads(
            json.dumps(saved.get("state") or {}))
        self._dirty = False

    def _projects(self) -> dict:
        """ATK's settings for every project. GUI thread only."""
        settings = getattr(self._ctx, "settings", None)
        if settings is None:
            self._ctx.settings = settings = {}
        return settings.setdefault(self._namespace, {}).setdefault(
            "projects", {})

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            # A copy: the GUI thread serialises this dict in flush(), and a
            # caller mutating a shared object mid-dump is a RuntimeError.
            return json.loads(json.dumps(self._data[key]))

    def set(self, key: str, value: Any) -> None:
        try:
            text = json.dumps(value)
        except (TypeError, ValueError) as exc:
            # M17. Failing here, at the moment of the mistake, beats failing
            # three days later when ATK cannot persist its settings.
            raise ValueError(
                f"StoragePort values must be JSON-serialisable; {key!r} is "
                f"not ({exc}).") from exc
        with self._lock:
            self._data[key] = json.loads(text)
            self._dirty = True

    def flush(self) -> bool:
        """Write this project's state into ATK's settings and save. GUI only.

        Returns False, after saying why through `report`, when ATK's save
        fails; the state stays dirty and the next flush tries again.
        """
        with self._lock:
            if not self._dirty:
                return True
            snapshot = json.loads(json.dumps(self._data))
            self._dirty = False
        self._projects()[self.key] = {"root": self.root, "state": snapshot}
        save = getattr(self._ctx, "save_settings", None)
        if not callable(save):
            return True
        try:
            save()
        except Exception as exc:                         # noqa: BLE001
            # ATK's save can fail in any way its storage can; whatever the
            # type, the operator is TOLD, which is what swallowing it lost.
            with self._lock:
                self._dirty = True
            _say(self._report,
                 f"Cognitive Coder could not save its history for "
                 f"{self.root} into ATK's settings ({exc}). It is kept in "
                 f"memory and will be saved again after the next change.")
            return False
        return True

    def sqlite_path(self, name: str) -> str:
        return str(self._dir / f"{name}.sqlite3")


def project_key(root: str | Path) -> str:
    """The per-project key: 12 hex digits of SHA-256 of the resolved root."""
    resolved = str(Path(root).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:12]


def _say(report: Callable[[str], None] | None, message: str) -> None:
    """Hand a sentence to a callable that must not take the build down."""
    if report is None:
        _log.warning("%s", message)
        return
    try:
        report(message)
    except Exception:                                    # noqa: BLE001
        _log.exception("the report callable raised; the message was: %s",
                       message)


# ==========================================================================
# EventPort → ctx.set_status + the panel console + the Cognitive Flow view
# ==========================================================================

class ATKEvents:
    """Renders engine events into ATK's three surfaces (§7.2).

    Takes plain callables rather than widgets, so the whole adapter stays
    testable without a QApplication — and so the panel can decide for itself
    how to marshal onto the GUI thread. **This is called from a worker
    thread**, and the panel's callbacks are what make that safe.
    """

    def __init__(self, *, status: Callable[[str], None] | None = None,
                 console: Callable[[str, str], None] | None = None,
                 flow: Callable[[dict], None] | None = None,
                 remote_banner: Callable[[str], None] | None = None) -> None:
        self._status = status
        self._console = console
        self._flow = flow
        self._remote = remote_banner

    def event(self, kind: str, message: str,
              data: dict | None = None) -> None:
        data = data or {}
        try:
            if kind == "token":
                if self._console:
                    self._console("token", message)
                return
            if kind == "remote":
                # M42.6: the indicator stays up for as long as it is true.
                if self._remote:
                    self._remote(message if data.get("enabled") else "")
                if self._console:
                    self._console("remote", message)
                return
            if kind == "phase" and self._flow:
                self._flow({"label": message,
                            "kind": data.get("phase", "phase")})
            if kind in ("status", "error", "warning", "budget") and \
                    self._status:
                self._status(message)
            if self._console:
                self._console(kind, message)
        except Exception:                                # noqa: BLE001
            # An EventPort that raises must not take the build down with it
            # — ATK's progress bar is not more important than the operator's
            # code. But a panel callback that raises is a bug in the panel,
            # and it goes to ATK's log with its traceback rather than away.
            _log.exception("a Cognitive Coder panel callback raised on a %r "
                           "event: %s", kind, message)


# ==========================================================================
# ApprovalPort → a real question, asked on the GUI thread
# ==========================================================================

class ATKApproval:
    """Diffs: auto-apply if the operator chose it, else ask. Remote: ask.

    §6.5's settled position: the LIBRARY default is approval-required, and
    auto-apply is opt-in behind an advanced setting with an explicit warning.
    In ATK that setting is Setup → System & Resources → Advanced, consistent
    with how every other consequential toggle there is gated.

    Auto-apply is only safe BECAUSE snapshots and transactional undo exist.
    If anyone ever finds themselves removing the snapshot step, they should
    remove this class's `auto_apply` option first.

    **`approve_remote` is never auto-approved**, whatever the diff setting
    says. C3 is ATK's core promise and it does not get a convenient default.

    Both questions are called on the build's WORKER thread. The callables
    are expected to marshal to the GUI thread themselves — `AskOnGuiThread`
    below is how the panel does it.
    """

    def __init__(self, *, auto_apply: bool = False,
                 ask_diff: Callable[[str, str], bool] | None = None,
                 ask_remote: Callable[[str, int, str], bool] | None = None
                 ) -> None:
        self.auto_apply = auto_apply
        self._ask_diff = ask_diff
        self._ask_remote = ask_remote
        self.applied: list[str] = []

    def approve_diff(self, summary: str, unified_diff: str) -> bool:
        if self.auto_apply:
            self.applied.append(summary)
            return True
        if self._ask_diff is None:
            return False        # nothing to ask with ⇒ nothing gets written
        return bool(self._ask_diff(summary, unified_diff))

    def approve_remote(self, provider: str, bytes_out: int,
                       estimate: str) -> bool:
        if self._ask_remote is None:
            return False
        return bool(self._ask_remote(provider, bytes_out, estimate))


class AskOnGuiThread:
    """Run a dialog on the GUI thread, from whichever thread is asking.

    The panel built its `QMessageBox` on the WORKER thread — the classic Qt
    crash, on the one path (C3's remote question) that always asks. And with
    no diff callable at all, a fresh install refused every change without a
    word. This is the marshalling, Qt-free so it can be tested:

      * `on_gui_thread()` — is the caller already on the GUI thread? The
        panel passes `QThread.currentThread() is app.thread()`.
      * `post(job)` — run `job` on the GUI thread and return once it has.
        The panel passes a signal's `emit`, connected with
        `Qt.BlockingQueuedConnection`.

    A blocking queued call to one's OWN thread deadlocks, so a caller that
    is already on the GUI thread runs the dialog directly. A dialog that
    raises, or a post that never ran it, is a NO: nothing is approved by
    accident.
    """

    def __init__(self, *, on_gui_thread: Callable[[], bool],
                 post: Callable[[Callable[[], None]], None],
                 report: Callable[[str], None] | None = None) -> None:
        self._on_gui_thread = on_gui_thread
        self._post = post
        self._report = report

    def __call__(self, dialog: Callable[[], bool]) -> bool:
        box: dict[str, Any] = {}

        def job() -> None:
            try:
                box["answer"] = bool(dialog())
            except Exception as exc:                     # noqa: BLE001
                box["error"] = exc

        if self._on_gui_thread():
            job()
        else:
            self._post(job)
        if "error" in box:
            _say(self._report, f"The approval dialog failed "
                               f"({box['error']}), so the answer is no and "
                               f"nothing was approved.")
            return False
        return bool(box.get("answer", False))

    def for_diff(self, dialog: Callable[[str, str], bool]
                 ) -> Callable[[str, str], bool]:
        return lambda summary, diff: self(lambda: dialog(summary, diff))

    def for_remote(self, dialog: Callable[[str, int, str], bool]
                   ) -> Callable[[str, int, str], bool]:
        return lambda provider, size, estimate: self(
            lambda: dialog(provider, size, estimate))


# ==========================================================================
# small, Qt-free pieces of the panel — here so they can be tested
# ==========================================================================

def change_log(session: Any, *, limit: int = 10) -> str:
    """The diffs of the most recent transactions, from the public history.

    Read from `session.history()` and each transaction's snapshot
    MANIFEST, which is where the patcher keeps the diff. The panel used
    to iterate `session.patcher._open` — a private attribute that is one
    Transaction or None, never a list — so every patch event raised
    TypeError, and the codemap and recommendation refreshes after it never
    ran. A transaction that is still open has no manifest yet and simply
    appears on the next refresh.
    """
    parts: list[str] = []
    records = [r for r in session.history()
               if r.state != "rollback_of" and r.snapshot_dir]
    for rec in records[-limit:]:
        try:
            manifest = session.host.fs.read(f"{rec.snapshot_dir}/MANIFEST.txt")
        except (OSError, ValueError):
            continue            # not committed yet, or pruned
        _, _, diff = manifest.partition("--- what changed ---\n")
        if diff.strip():
            parts.append(f"# transaction {rec.seq} · {rec.state}"
                         f"{' · sealed' if rec.sealed else ''} · "
                         f"{', '.join(rec.files)}\n{diff.rstrip()}\n")
    return "\n".join(parts)


def project_root_for(ctx: Any) -> tuple[Path | None, str]:
    """The folder a build may write into, or None and the sentence why not.

    The panel fell back to `Path.cwd()` when ATK had no project set — and
    ATK's working directory is ATK's own checkout, so the generator was
    aimed at ATK's source tree. There is no safe default folder, so there
    is no default: no project, no build. A folder that CONTAINS ATK's own
    package is refused for the same reason.
    """
    raw = getattr(ctx, "project_root", None)
    if not raw:
        return None, ("No project folder is set, so nothing was built. "
                      "Choose one first — Cognitive Coder will not guess, "
                      "because its guess was once ATK's own source tree.")
    root = Path(raw).expanduser().resolve()
    atk = _atk_package_dir()
    if atk is not None and (root == atk or root in atk.parents
                            or atk in root.parents):
        return None, (f"{root} holds ATK's own source ({atk}), so nothing "
                      f"was built. Choose a project folder outside it.")
    return root, ""


def _atk_package_dir() -> Path | None:
    """Where the running ATK's `atk` package is, if it is importable."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("atk")
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin and spec.origin not in ("namespace", "built-in"):
        return Path(spec.origin).resolve().parent
    locations = list(spec.submodule_search_locations or [])
    return Path(locations[0]).resolve() if locations else None


def failure_line(detail: str) -> str:
    """The last non-blank line of a failure's detail, for the console.

    `detail.strip().splitlines()[-1]` raised IndexError on an empty detail,
    inside the very slot that reports failures.
    """
    lines = [ln for ln in (detail or "").splitlines() if ln.strip()]
    return (lines[-1].strip() if lines
            else "no detail was given — see ATK's log")


# ==========================================================================
# putting it together
# ==========================================================================

def build_host(ctx: Any, engine: Any, project_root: str | Path, *,
               data_dir: str | Path | None = None,
               auto_apply: bool = False,
               status: Callable[[str], None] | None = None,
               console: Callable[[str, str], None] | None = None,
               flow: Callable[[dict], None] | None = None,
               remote_banner: Callable[[str], None] | None = None,
               ask_diff: Callable[[str, str], bool] | None = None,
               ask_remote: Callable[[str, int, str], bool] | None = None
               ) -> Host:
    """One call from an ATK panel to a fully-wired engine host."""
    if data_dir is None:
        try:
            from atk.config import DATA_DIR
            data_dir = DATA_DIR
        except ImportError:
            # Outside ATK (the tests, the CC clone): beside the project,
            # never in the working directory.
            data_dir = Path(project_root) / ".atk"
    events = ATKEvents(status=status, console=console, flow=flow,
                       remote_banner=remote_banner)

    def warn(message: str) -> None:
        events.event("warning", message)

    return Host(
        llm=ATKLLM(engine, events=lambda kind, message:
                   events.event(kind, message)),
        fs=ATKFileSystem(project_root, on_refusal=status),
        exec=ATKExec(),
        storage=ATKStorage(ctx, data_dir, project_root, report=warn),
        events=events,
        approval=ATKApproval(auto_apply=auto_apply, ask_diff=ask_diff,
                             ask_remote=ask_remote))


def build_session(host: Host, *, lang: str = "python",
                  conventions: str = "", **config: Any) -> Session:
    """A session configured the way ATK's doctrine wants it.

    The conventions block is where A.3's doctrine reaches the model: comments
    explain WHY, honest failure, say what was omitted. Passing it here rather
    than hardcoding it in the core is the whole point of the Port design —
    ATK's house style is ATK's business.
    """
    return Session(host, config=SessionConfig(
        lang=lang,
        conventions=conventions or ATK_CONVENTIONS,
        **config))


#: ATK's own doctrine (A.3), handed to the model as project conventions.
ATK_CONVENTIONS = """\
This project's rules, which matter more than general good practice:

1. Deterministic first, model second, human last. If a rule, a compiler or a
   test can answer a question, do not write code that asks a model.
2. Honest failure. UNKNOWN and AMBIGUOUS are real answers. A wrong name stops
   a search; "I don't know" is actionable.
3. Say what was omitted — thinned plots, truncated reads, dropped context.
   A reader who does not know something was left out will assume it was not
   there.
4. Comments explain WHY, especially where an obvious approach was rejected.
   What the code does is visible; why it does it that way is not.
5. Errors that reach a person are plain sentences naming what happened and
   what to do. Tracebacks go to the log.
6. This tool is offline and zero-telemetry. Never open a network connection,
   never phone home, never add a dependency that does either.
"""


def preflight(engine: Any, project_root: str | Path) -> list[str]:
    """Problems worth naming before the operator presses Build.

    Every one of these is something that would otherwise surface as a
    confusing failure three minutes in, and each has a specific remedy the
    operator can act on now.
    """
    notes: list[str] = []
    if not getattr(engine, "is_loaded", False):
        notes.append(
            "No model is loaded, so nothing can be generated. Load one in "
            "Setup — and if Whisper has the VRAM, unload it first: they are "
            "mutually exclusive at 16 GB.")
    root = Path(project_root)
    if not root.exists():
        notes.append(f"The project folder {root} does not exist yet.")
    elif (root / ".git").exists():
        notes.append(
            "This project is a git repository. Cognitive Coder never runs "
            "git and keeps its own snapshots in .cc_snapshots/, so your "
            "history, stash and index are untouched — but a clean working "
            "tree makes the diffs easier to read.")
    # ruff's target-version is OUR floor; this runs under ATK's interpreter,
    # which nothing here chose. The check is the point, not dead code.
    if sys.version_info < (3, 11):                       # noqa: UP036
        notes.append(
            f"This interpreter is {sys.version_info.major}."
            f"{sys.version_info.minor}; Cognitive Coder needs 3.11 or later.")
    return notes
