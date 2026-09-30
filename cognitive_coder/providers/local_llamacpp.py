# SPDX-License-Identifier: Apache-2.0
"""A GGUF loaded in-process, via llama-cpp-python.

Usually the host already has a model loaded and simply wraps it — ATK's
`llm_engine.py` is exactly that, and binding `LLMPort` to an existing engine
is both cheaper and correct, because the host owns loading and unloading
(§0.1). This provider is for the case where nothing else has done it: the CLI,
a test rig, or a host that would rather not build its own.

**`llama_cpp` is imported inside the constructor, never at module level.**
That is M48 — a CI check fails the build if the core imports a non-stdlib
module at import time — and it is also C7: a machine without llama-cpp-python
gets a sentence explaining what is missing and what it costs, not an
ImportError from `import cognitive_coder`.

Two behaviours worth knowing about:

  * **`save_state()`/`load_state()` are exposed** (G.7.4, G.6). With 64 GB of
    RAM there is no reason to thrash one KV slot; keeping a prefix per
    (persona, epoch) means switching target files inside an epoch costs
    nothing. The core never drives a model swap — that is the host's button
    (§0.1, M10) — but if the host wants to preserve a prefix across one, this
    is the mechanism it needs.
  * **`prompt_ms` is measured** (M55), by streaming: time to the first token
    is prefill, the rest is decode. `Llama` has no `get_timings()` (the old
    code called it and always fell back to the whole call), so the
    non-streaming fallback reads llama.cpp's perf counters where the binding
    exposes them and otherwise reports `decode_ms=0`, meaning "not split".
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import time
from typing import Any

from ..errors import ConfigurationError
from ..types import Completion, Message, ModelCapabilities, ToolSpec
from .base import (
    ProviderBase,
    StreamAccumulator,
    _as_int,
    _as_ms,
    as_dict,
    error_detail,
    family_for,
    first_choice,
    map_finish,
    messages_to_openai,
    parse_tool_calls,
    supports_fim,
)


class LocalLlamaCpp(ProviderBase):
    """An `LLMPort` over an in-process GGUF."""

    name = "local_llamacpp"
    is_remote = False

    def __init__(self, model_path: str = "", *, llama: Any = None,
                 n_ctx: int = 16384, n_gpu_layers: int = -1,
                 type_k: str = "q8_0", type_v: str = "q8_0",
                 offload_kqv: bool = True, chat_format: str | None = None,
                 verbose: bool = False) -> None:
        """Wrap an existing `Llama` (pass `llama=`) or load one from a path.

        The defaults are Appendix G.8's starting configuration, and they are a
        STARTING POINT to measure from, not a recommendation to freeze:
        16k context, q8_0 KV (halve the KV cost before touching offload), KQV
        offloaded, as many layers on the GPU as fit underneath that. The
        journal records what was used so the next value is evidence-based
        (G.9).
        """
        self._ctx = n_ctx
        self._name = ""
        self.last_prompt_ms = 0
        if llama is not None:
            self.llama = llama
            self._name = str(getattr(llama, "model_path", "") or "wrapped")
            return
        if not model_path:
            raise ConfigurationError(
                "No model file was given. Point this at a .gguf file, or "
                "pass an already-loaded model from the host.")
        try:
            from llama_cpp import Llama  # noqa: PLC0415 — see docstring
        except ImportError as exc:
            raise ConfigurationError(
                "llama-cpp-python is not installed, so a GGUF cannot be "
                "loaded in-process. Either install it (pip install "
                "'cognitive-coder[llamacpp]'), or run a llama.cpp server and "
                "use the openai_compatible provider instead — which is the "
                "more common arrangement anyway.", str(exc)) from exc
        self.llama = Llama(
            model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            type_k=type_k, type_v=type_v, offload_kqv=offload_kqv,
            chat_format=chat_format, verbose=verbose)
        self._name = model_path.replace("\\", "/").rsplit("/", 1)[-1]

    # -- the Port ---------------------------------------------------------
    def complete(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (), temperature: float = 0.15,
                 max_tokens: int = 2048, stop: Sequence[str] | None = None,
                 grammar: str | None = None, seed: int | None = None,
                 cancel: Any = None) -> Completion:
        if cancel is not None and cancel.is_set():
            return self.cancelled(self._name)

        kwargs: dict[str, Any] = {
            "messages": messages_to_openai(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = list(stop)
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = self.tools_payload(tools)
            kwargs["tool_choice"] = "auto"
        if grammar:
            try:
                from llama_cpp import LlamaGrammar  # noqa: PLC0415
                kwargs["grammar"] = LlamaGrammar.from_string(grammar)
            except Exception:                            # noqa: BLE001
                # No grammar support is a degraded mode, not a failure: the
                # repair parser in base.py covers it, and `ToolCall.repaired`
                # makes the cost visible (C7, D9).
                pass

        # Streamed, for the same reason as `openai_compatible`: the time to
        # the first token is prefill, the rest is decode. The old path timed
        # one blocking call and then asked `self.llama.get_timings()` for the
        # split — a method `Llama` does not have — so `prompt_ms` was always
        # the whole call and `decode_ms` was never set.
        t0 = time.monotonic()
        try:
            chunks = self.llama.create_chat_completion(stream=True, **kwargs)
            acc = StreamAccumulator(t0, self._name)
            for chunk in chunks:
                if cancel is not None and cancel.is_set():
                    return self.cancelled(self._name)
                acc.feed(chunk, time.monotonic())
            out = dataclasses.replace(acc.completion(time.monotonic()),
                                      model=self._name)
            self.last_prompt_ms = out.prompt_ms
            return out
        except Exception:                                # noqa: BLE001
            # Some chat handlers cannot stream with tools; the call is made
            # again whole rather than failed. Anything that fails both ways
            # was never about streaming.
            pass

        t0 = time.monotonic()
        try:
            data = self.llama.create_chat_completion(**kwargs)
        except Exception as exc:                         # noqa: BLE001
            detail = error_detail({"error": f"{type(exc).__name__}: {exc}"})
            return self.failed(
                f"llama.cpp could not generate an answer ({detail}). "
                f"Nothing was generated.", model=self._name,
                prompt_ms=int((time.monotonic() - t0) * 1000))
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        choice = first_choice(data)
        msg = choice.get("message")
        if not isinstance(msg, dict):
            return self.failed(
                "llama.cpp answered in a shape that is not a chat "
                "completion (no choices[0].message). Nothing was "
                "generated.", model=self._name, prompt_ms=elapsed_ms)
        usage = as_dict(data.get("usage"))
        calls = parse_tool_calls([c for c in msg.get("tool_calls") or ()
                                  if isinstance(c, dict)])
        prompt_ms, decode_ms = self._perf(elapsed_ms)
        self.last_prompt_ms = prompt_ms
        return Completion(
            text=str(msg.get("content") or ""), tool_calls=calls,
            finish_reason=map_finish(str(choice.get("finish_reason") or ""),
                                     bool(calls)),
            tokens_in=_as_int(usage.get("prompt_tokens")),
            tokens_out=_as_int(usage.get("completion_tokens")),
            model=self._name, prompt_ms=prompt_ms, decode_ms=decode_ms)

    def _perf(self, fallback_ms: int) -> tuple[int, int]:
        """(prefill, decode) ms from llama.cpp's own counters, if reachable.

        llama-cpp-python exposes them only through the low-level binding
        (`llama_perf_context` on current builds, `llama_get_timings` on old
        ones), on the context the `Llama` object holds privately. Where
        neither is reachable the whole call is reported as `prompt_ms` with
        `decode_ms` 0 — which `cache_health()` reads as "cannot tell", the
        honest answer — rather than a guessed split.
        """
        try:
            import llama_cpp  # noqa: PLC0415 — optional, see module doc
            ctx = self.llama._ctx.ctx
            for fn in ("llama_perf_context", "llama_get_timings"):
                if hasattr(llama_cpp, fn):
                    perf = getattr(llama_cpp, fn)(ctx)
                    p = _as_ms(getattr(perf, "t_p_eval_ms", None))
                    d = _as_ms(getattr(perf, "t_eval_ms", None))
                    if p is not None:
                        return p, d or 0
        except Exception:                                # noqa: BLE001
            pass
        return fallback_ms, 0

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        cancel = kw.pop("cancel", None)
        acc = StreamAccumulator(time.monotonic(), self._name)
        try:
            chunks = self.llama.create_chat_completion(
                messages=messages_to_openai(messages), stream=True,
                temperature=kw.get("temperature", 0.15),
                max_tokens=kw.get("max_tokens", 2048))
            for chunk in chunks:
                if cancel is not None and cancel.is_set():
                    return
                before = len(acc.parts)
                acc.feed(chunk, time.monotonic())
                yield from acc.parts[before:]
        except Exception:                                # noqa: BLE001
            return

    def capabilities(self) -> ModelCapabilities:
        ctx = self._ctx
        try:
            ctx = int(self.llama.n_ctx())
        except Exception:                                # noqa: BLE001
            pass
        return ModelCapabilities(
            name=self._name, family=family_for(self._name),
            context_tokens=ctx,
            supports_tools=True,        # llama.cpp's chat handlers do tools
            supports_grammar=True,      # GBNF is the point of llama.cpp
            supports_vision=False,      # true only with a paired mmproj
            supports_fim=supports_fim(self._name),
            is_remote=False,
            token_count_is_estimate=False)   # a real tokenizer is loaded

    def count_tokens(self, text: str) -> int:
        """Exact — there is a tokenizer right here, so use it (M14)."""
        try:
            return len(self.llama.tokenize((text or "").encode("utf-8")))
        except Exception:                                # noqa: BLE001
            from .base import estimate_tokens
            return estimate_tokens(text)

    # -- KV state, offered to the host (G.6, G.7.4) -----------------------
    def save_state(self) -> Any:
        """The KV cache as an opaque object the host may keep.

        Offered, never driven. The core contains no swap logic (M10); if the
        host wants to preserve a prefix across its own model-swap button,
        this is what it needs, and what it does with it is its business.
        """
        try:
            return self.llama.save_state()
        except Exception:                                # noqa: BLE001
            return None

    def load_state(self, state: Any) -> bool:
        try:
            self.llama.load_state(state)
            return True
        except Exception:                                # noqa: BLE001
            return False
