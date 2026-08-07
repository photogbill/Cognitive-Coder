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

    1. copy this file to `atk/core/ccoder_host.py`
    2. copy `ccoder_panel.py` to `atk/ui/ccoder_panel.py`
    3. run `python adapters/atk/migrate.py --dry-run` from the CC clone
    4. run it for real, then ATK's full suite
    5. only then delete the re-export shims

Nothing here writes to ATK's `state.db`. A second database FILE is fine; a
second schema in the same file is not (§7.1).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

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
from cognitive_coder.ports import Host

MAX_OUTPUT = 200_000


# ==========================================================================
# LLMPort → atk.core.llm_engine.LLMEngine
# ==========================================================================

class ATKLLM:
    """Binds `LLMPort` to the model ATK already has loaded.

    It does NOT load anything. ATK owns loading, unloading and the 16 GB
    ceiling — the cognitive core and Whisper are mutually exclusive, and the
    swap button belongs to ATK (§0.1, §7.1). This class asks what is loaded
    and works with the answer.
    """

    name = "atk"
    is_remote = False

    def __init__(self, engine: Any, *,
                 n_ctx_default: int = 16384) -> None:
        self.engine = engine
        self._n_ctx_default = n_ctx_default
        self._tokenizer: Any = None
        self._tokenizer_tried = False
        self.last_prompt_ms = 0

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
            return Completion(text="", finish_reason="error",
                              model="")

        payload = [{"role": m.role, "content": m.content}
                   for m in messages if m.role != "tool"]
        # ATK's chat template has no tool role; a tool result is folded into
        # the user turn rather than dropped, because dropping it makes the
        # model answer a question it was never shown the answer to.
        for m in messages:
            if m.role == "tool":
                payload.append({"role": "user",
                                "content": f"[tool result]\n{m.content}"})

        t0 = time.monotonic()
        chunks: list[str] = []
        cancelled = False
        try:
            for token in self.engine.chat_stream(
                    payload, temperature=temperature, max_tokens=max_tokens):
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                chunks.append(token)
        except Exception:                                # noqa: BLE001
            return Completion(text="".join(chunks), finish_reason="error",
                              model=self._model_name(),
                              prompt_ms=int((time.monotonic() - t0) * 1000))

        raw = "".join(chunks)
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        self.last_prompt_ms = elapsed_ms

        # ATK's `split_think` handles both `[THINK]…[/THINK]` (Magistral) and
        # `<think>…</think>`, closed or unclosed. The core strips think tags
        # too, but doing it here means ATK's own richer handling wins, and
        # the reasoning stays available for the panel to display (D13, M37).
        answer = raw
        try:
            from atk.core.llm_engine import split_think
            _reasoning, answer = split_think(raw)
        except Exception:                                # noqa: BLE001
            pass

        finish = "cancelled" if cancelled else (
            "length" if len(answer) and max_tokens and
            self._looks_truncated(answer, max_tokens) else "stop")
        return Completion(
            text=answer, finish_reason=finish,
            tokens_in=self.count_tokens(
                "\n".join(m.content for m in messages)),
            tokens_out=self.count_tokens(answer),
            model=self._model_name(), prompt_ms=elapsed_ms)

    @staticmethod
    def _looks_truncated(text: str, max_tokens: int) -> bool:
        """ATK's stream does not report a finish reason, so estimate it.

        D1 says truncation is detected STRUCTURALLY, and `finish_reason` is
        the signal. ATK's API does not carry one, so this is the honest
        approximation — and the loop's unbalanced-delimiter backstop catches
        what it misses. Better an approximation that is usually right than a
        `"stop"` that is confidently wrong.
        """
        return len(text) >= max_tokens * 3.2

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
        except Exception:                                # noqa: BLE001
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
            except Exception:                            # noqa: BLE001
                pass
        try:
            return len(self.engine._llm.tokenize((text or "").encode("utf-8")))
        except Exception:                                # noqa: BLE001
            return max(1, len(text or "") // 4)

    def _get_tokenizer(self) -> Any:
        if self._tokenizer_tried:
            return self._tokenizer
        self._tokenizer_tried = True
        family = _family(self._model_name())
        if family != "mistral":
            return None
        try:
            from mistral_common.tokens.tokenizers.mistral import (
                MistralTokenizer)
            self._tokenizer = MistralTokenizer.v3().instruct_tokenizer
        except Exception:                                # noqa: BLE001
            self._tokenizer = None
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
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".cc-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
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

    def run(self, argv: Sequence[str], *, cwd: str, timeout: float,
            stdin: str = "", env: dict | None = None) -> ProcResult:
        argv = [str(a) for a in argv]
        environment = dict(env or {})
        environment.update(self._extra)
        t0 = time.monotonic()

        kwargs: dict[str, Any] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            # A new process GROUP is what makes taskkill /T able to find the
            # children. Without it, a terminated cmd.exe leaves them running.
            kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=environment or None, **kwargs)
        except (OSError, ValueError) as exc:
            return ProcResult(exit_code=-1, stderr=f"could not run: {exc}",
                              duration_s=time.monotonic() - t0)

        try:
            out, err = proc.communicate(input=stdin or None, timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except Exception:                            # noqa: BLE001
                out, err = "", ""
            err = (err or "") + (
                f"\ncognitive-coder: this exceeded {timeout:.0f}s and the "
                f"whole process tree was killed. Godot and MSVC both spawn "
                f"children that outlive their parent, which is why the tree "
                f"and not just the process is killed.")
            timed_out = True

        out, cut_a = _cap(out or "")
        err, cut_b = _cap(err or "")
        return ProcResult(
            exit_code=proc.returncode if not timed_out else -9,
            stdout=out, stderr=err, duration_s=time.monotonic() - t0,
            timed_out=timed_out, truncated=cut_a or cut_b)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        """M16. On Windows a terminated shell does not take its children.

        An orphaned Godot instance holding a file lock is the observed
        failure this exists to prevent, and it is why `timed_out=True` is an
        attestation rather than a note.
        """
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
                    subprocess.run(
                        [taskkill, "/T", "/F", "/PID", str(proc.pid)],
                        capture_output=True, timeout=10,
                        creationflags=getattr(subprocess,
                                              "CREATE_NO_WINDOW", 0))
                    return
                except Exception:                        # noqa: BLE001
                    pass
        try:
            proc.kill()
        except Exception:                                # noqa: BLE001
            pass

    def which(self, binary: str) -> str | None:
        return shutil.which(binary)


def _cap(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT:
        return text, False
    return (text[:MAX_OUTPUT] + f"\n… truncated at {MAX_OUTPUT:,} characters. "
            f"A program producing this much output is usually looping.", True)


# ==========================================================================
# StoragePort → ctx.settings["ccoder"] + ATK's DATA_DIR
# ==========================================================================

class ATKStorage:
    """Maps onto ATK's settings dict and data directory (§7.1).

    Two rules from the conflict table, both load-bearing:

      * **Never write to ATK's `state.db`.** A second database FILE is fine;
        a second schema in the same file is not. `sqlite_path` returns a
        separate file under ATK's data dir.
      * **The core never reads ATK settings directly.** It has no way to —
        it holds this object and nothing else, which is the point of C2.
    """

    def __init__(self, ctx: Any, data_dir: str | Path, *,
                 namespace: str = "ccoder") -> None:
        self._ctx = ctx
        self._namespace = namespace
        self._dir = Path(data_dir) / "ccoder"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _bucket(self) -> dict:
        settings = getattr(self._ctx, "settings", None)
        if settings is None:
            self._ctx.settings = settings = {}
        return settings.setdefault(self._namespace, {})

    def get(self, key: str, default: Any = None) -> Any:
        return self._bucket().get(key, default)

    def set(self, key: str, value: Any) -> None:
        import json
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            # M17. Failing here, at the moment of the mistake, beats failing
            # three days later when ATK cannot persist its settings.
            raise ValueError(
                f"StoragePort values must be JSON-serialisable; {key!r} is "
                f"not ({exc}).") from exc
        self._bucket()[key] = value
        save = getattr(self._ctx, "save_settings", None)
        if callable(save):
            try:
                save()
            except Exception:                            # noqa: BLE001
                pass

    def sqlite_path(self, name: str) -> str:
        return str(self._dir / f"{name}.sqlite3")


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
            # An EventPort that raises must not take the build down with it.
            # ATK's progress bar is not more important than the operator's
            # code.
            pass


# ==========================================================================
# ApprovalPort → auto-apply for diffs, a real prompt for the network
# ==========================================================================

class ATKApproval:
    """The owner has chosen auto-apply with undo. Remote is a different question.

    §6.5's settled position: the LIBRARY default is approval-required, and
    auto-apply is opt-in behind an advanced setting with an explicit warning.
    In ATK that setting is Setup → System & Resources → Advanced, consistent
    with how every other consequential toggle there is gated.

    Auto-apply is only safe BECAUSE snapshots and transactional undo exist.
    If anyone ever finds themselves removing the snapshot step, they should
    remove this class's `auto_apply` default first.

    **`approve_remote` is never auto-approved**, whatever the diff setting
    says. C3 is ATK's core promise and it does not get a convenient default.
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
        except Exception:                                # noqa: BLE001
            data_dir = Path(project_root) / ".atk"
    return Host(
        llm=ATKLLM(engine),
        fs=ATKFileSystem(project_root, on_refusal=status),
        exec=ATKExec(),
        storage=ATKStorage(ctx, data_dir),
        events=ATKEvents(status=status, console=console, flow=flow,
                         remote_banner=remote_banner),
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
    if sys.version_info < (3, 11):
        notes.append(
            f"This interpreter is {sys.version_info.major}."
            f"{sys.version_info.minor}; Cognitive Coder needs 3.11 or later.")
    return notes
