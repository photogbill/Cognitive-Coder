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
  * **`prompt_ms` is measured** (M55). llama-cpp-python reports timings in its
    own way; where it doesn't, the eval time either side of the call is
    measured, because a number that is only approximately right still catches
    a 3-second prefix turning into 90.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import time
from typing import Any

from ..errors import ConfigurationError
from ..types import Completion, Message, ModelCapabilities, ToolSpec
from .base import (
    ProviderBase,
    family_for,
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

        t0 = time.monotonic()
        try:
            data = self.llama.create_chat_completion(**kwargs)
        except Exception:                                # noqa: BLE001
            return Completion(text="", finish_reason="error",
                              model=self._name,
                              prompt_ms=int((time.monotonic() - t0) * 1000))
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        calls = parse_tool_calls(msg.get("tool_calls") or ())
        finish = str(choice.get("finish_reason") or "stop")
        finish = {"length": "length", "tool_calls": "tool_calls"}.get(
            finish, "stop")
        if calls and finish != "length":
            finish = "tool_calls"
        prompt_ms = self._prompt_ms(elapsed_ms)
        self.last_prompt_ms = prompt_ms
        return Completion(
            text=str(msg.get("content") or ""), tool_calls=calls,
            finish_reason=finish,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            model=self._name, prompt_ms=prompt_ms)

    def _prompt_ms(self, fallback_ms: int) -> int:
        """Prompt-processing time, from the library where it exposes it."""
        try:
            timings = self.llama.get_timings()           # newer builds
            value = int(float(timings.get("prompt_ms", 0)))
            if value:
                return value
        except Exception:                                # noqa: BLE001
            pass
        return fallback_ms

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        cancel = kw.pop("cancel", None)
        try:
            chunks = self.llama.create_chat_completion(
                messages=messages_to_openai(messages), stream=True,
                temperature=kw.get("temperature", 0.15),
                max_tokens=kw.get("max_tokens", 2048))
            for chunk in chunks:
                if cancel is not None and cancel.is_set():
                    return
                delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                text = delta.get("content")
                if text:
                    yield text
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
