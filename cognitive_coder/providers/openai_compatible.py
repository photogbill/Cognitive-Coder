# SPDX-License-Identifier: Apache-2.0
"""One adapter for llama.cpp server, Ollama, LM Studio, vLLM and LiteLLM.

**This is the highest-value provider to write**, and the reason is not
technical elegance: it is local, and it is what most self-hosters actually
run. One HTTP shape covers five servers, and the operator does not have to
care which one is behind the URL.

Built on `urllib` from the standard library, not `requests` or `httpx`. That
is M48 — the core has zero required runtime dependencies — and it is also
what lets this be embedded in a Qt app and a FastAPI server without an
argument about which HTTP client version wins.

**A local endpoint is not a remote one.** `http://localhost:8080` and
`http://192.168.1.40:8080` are LAN addresses; talking to them is not "the
network" in the sense C3 cares about, and the class reports `is_remote=False`
for them. A non-private host IS remote, reports itself as such, and the
session's remote gate applies (M42). That distinction is enforced here rather
than trusted to configuration, because getting it wrong on an air-gapped
machine is exactly the failure C3 exists to prevent.

`prompt_ms` is read from the server's timing fields where it reports them
(llama.cpp does), and measured end-to-end where it doesn't. It is required by
M55 because it is the only way anyone notices the prefix cache breaking.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import ipaddress
import json
import socket
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from ..types import Completion, Message, ModelCapabilities, ToolSpec
from .base import (
    ProviderBase,
    estimate_tokens,
    family_for,
    messages_to_openai,
    parse_tool_calls,
    supports_fim,
)

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 900.0          # a 24B generating 800 tokens takes minutes


def is_local_url(url: str) -> bool:
    """True for loopback and private-range hosts. The C3 judgement call.

    Names that are not addresses are resolved once; a name that will not
    resolve is treated as NOT local, because the safe default when you cannot
    tell is the one that makes the engine ask permission.
    """
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    if host in ("localhost", "127.0.0.1", "::1", ""):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            addr = ipaddress.ip_address(socket.gethostbyname(host))
        except (OSError, ValueError):
            return False
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local)


class OpenAICompatible(ProviderBase):
    """An `LLMPort` over any `/v1/chat/completions` endpoint."""

    name = "openai_compatible"

    def __init__(self, base_url: str = DEFAULT_URL, *, model: str = "",
                 api_key: str = "", timeout: float = DEFAULT_TIMEOUT,
                 context_tokens: int = 0, supports_tools: bool | None = None,
                 supports_vision: bool = False,
                 headers: dict | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._key = api_key
        self.timeout = timeout
        self._ctx = context_tokens
        self._tools_flag = supports_tools
        self._vision = supports_vision
        self._headers = dict(headers or {})
        self.is_remote = not is_local_url(self.base_url)
        self._probed: dict | None = None
        self.last_prompt_ms = 0

    # -- HTTP -------------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float | None = None
              ) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        for k, v in self._headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def _get(self, path: str, timeout: float = 10.0) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    # -- the Port ---------------------------------------------------------
    def complete(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (), temperature: float = 0.15,
                 max_tokens: int = 2048, stop: Sequence[str] | None = None,
                 grammar: str | None = None, seed: int | None = None,
                 cancel: Any = None) -> Completion:
        if cancel is not None and cancel.is_set():
            return self.cancelled(self.model)

        payload: dict[str, Any] = {
            "model": self.model or "local",
            "messages": messages_to_openai(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = seed
        if grammar:
            # llama.cpp's GBNF field. Grammar-constrained decoding is
            # strictly better than repairing near-JSON afterwards (D9), so it
            # is used whenever the server accepts it.
            payload["grammar"] = grammar
        caps = self.capabilities()
        if tools and caps.supports_tools:
            payload["tools"] = self.tools_payload(tools)
            payload["tool_choice"] = "auto"
        # M12: a host whose model lacks tool support IGNORES `tools`. Sending
        # them anyway to a server that doesn't understand them is how a
        # perfectly good local setup starts returning 400s.

        t0 = time.monotonic()
        try:
            data = self._post("/v1/chat/completions", payload)
        except urllib.error.HTTPError:
            # The body of an HTTP error is deliberately not read here: it can
            # echo the request, and the request can contain file contents. It
            # belongs in the journal or nowhere, never in a variable that
            # might end up in a log.
            # M11: never raise on a model's refusal — and by extension, hand
            # back a Completion the loop can reason about rather than an
            # exception it must special-case.
            return Completion(
                text="", finish_reason="error", model=self.model,
                prompt_ms=int((time.monotonic() - t0) * 1000))
        except (urllib.error.URLError, OSError, TimeoutError):
            return Completion(text="", finish_reason="error",
                              model=self.model,
                              prompt_ms=int((time.monotonic() - t0) * 1000))
        except (ValueError, TypeError):
            return Completion(text="", finish_reason="error",
                              model=self.model)

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        # llama.cpp reports prompt processing separately, which is the number
        # that actually answers "did the prefix cache hold?" (G.7.5).
        prompt_ms = int(timings.get("prompt_ms") or 0) or elapsed_ms
        self.last_prompt_ms = prompt_ms

        calls = parse_tool_calls(msg.get("tool_calls") or ())
        finish = str(choice.get("finish_reason") or "stop")
        finish = {"tool_calls": "tool_calls", "length": "length",
                  "stop": "stop", "eos": "stop", "content_filter": "stop"
                  }.get(finish, "stop")
        if calls and finish != "length":
            finish = "tool_calls"
        return Completion(
            text=str(msg.get("content") or ""), tool_calls=calls,
            finish_reason=finish,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            model=str(data.get("model") or self.model),
            prompt_ms=prompt_ms)

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        """Server-sent events, for display. Cancellation is checked per chunk."""
        cancel = kw.pop("cancel", None)
        payload: dict[str, Any] = {
            "model": self.model or "local",
            "messages": messages_to_openai(messages),
            "temperature": kw.get("temperature", 0.15),
            "max_tokens": kw.get("max_tokens", 2048),
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for raw in r:
                    if cancel is not None and cancel.is_set():
                        return
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        return
                    try:
                        chunk = json.loads(body)
                    except ValueError:
                        continue
                    delta = ((chunk.get("choices") or [{}])[0]
                             .get("delta") or {})
                    text = delta.get("content")
                    if text:
                        yield text
        except (urllib.error.URLError, OSError, TimeoutError):
            return

    def capabilities(self) -> ModelCapabilities:
        """Probed once from `/v1/models`, then cached for the session.

        M13 says this reflects the CURRENTLY loaded model. For a server that
        is what `/v1/models` reports; `refresh()` re-probes, and the session
        calls it at task boundaries so a swap behind the server is noticed.
        """
        if self._probed is None:
            self._probed = self._probe()
        info = self._probed
        name = self.model or info.get("name", "")
        return ModelCapabilities(
            name=name, family=family_for(name),
            context_tokens=self._ctx or info.get("context", 8192),
            supports_tools=(self._tools_flag if self._tools_flag is not None
                            else info.get("tools", True)),
            supports_grammar=info.get("grammar", False),
            supports_vision=self._vision,
            supports_fim=supports_fim(name),
            is_remote=self.is_remote,
            token_count_is_estimate=True)

    def refresh(self) -> ModelCapabilities:
        self._probed = None
        return self.capabilities()

    def _probe(self) -> dict:
        """Ask the server what it is. Failure is a normal, quiet outcome.

        A server that is not running yet is not an error here — the host may
        be about to start one, and `capabilities().loaded` being False is the
        supported way to say "nothing is loaded" (M10).
        """
        info: dict[str, Any] = {}
        try:
            data = self._get("/v1/models")
            rows = data.get("data") or []
            if rows:
                info["name"] = str(rows[0].get("id") or "")
                meta = rows[0].get("meta") or {}
                if meta.get("n_ctx_train"):
                    info["context"] = int(meta["n_ctx_train"])
        except Exception:                                # noqa: BLE001
            return {"name": self.model, "context": self._ctx or 8192,
                    "tools": self._tools_flag is not False, "grammar": False}
        try:
            # llama.cpp's own endpoint; its presence means GBNF is available,
            # which is worth knowing (D9).
            props = self._get("/props", timeout=5.0)
            info["grammar"] = True
            ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
            if ctx:
                info["context"] = int(ctx)
        except Exception:                                # noqa: BLE001
            pass
        info.setdefault("context", self._ctx or 8192)
        info.setdefault("tools", True)
        return info

    def count_tokens(self, text: str) -> int:
        """llama.cpp's `/tokenize` when present, the stated estimate if not.

        The exact path is worth taking when it is free: budgeting against an
        estimate wastes context on the safety margin, and the margin is
        several hundred tokens on every call.
        """
        try:
            data = self._post("/tokenize", {"content": text or ""},
                              timeout=10.0)
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return len(tokens)
        except Exception:                                # noqa: BLE001
            pass
        return estimate_tokens(text)


def detect(candidates: Sequence[str] = (), *,
           timeout: float = 2.0) -> list[str]:
    """Which of the usual local endpoints are actually up.

    Ollama, LM Studio, llama.cpp and vLLM each have a conventional port. A
    host can offer the operator a list instead of a text box, which removes
    an entire class of "why isn't it working" — and this is a LOCAL probe, so
    C3 is untouched by it.
    """
    urls = list(candidates) or [
        "http://127.0.0.1:8080",     # llama.cpp server
        "http://127.0.0.1:11434",    # Ollama
        "http://127.0.0.1:1234",     # LM Studio
        "http://127.0.0.1:8000",     # vLLM
        "http://127.0.0.1:4000",     # LiteLLM
    ]
    found = []
    for url in urls:
        try:
            req = urllib.request.Request(f"{url.rstrip('/')}/v1/models")
            with urllib.request.urlopen(req, timeout=timeout):
                found.append(url)
        except Exception:                                # noqa: BLE001
            continue
    return found
