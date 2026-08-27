# SPDX-License-Identifier: Apache-2.0
"""Remote providers — Anthropic, Google, Mistral, OpenRouter, OpenAI.

**Read `providers/__init__.py` first.** The gate that governs everything here
was built before these were, deliberately, so that adding a provider could
not accidentally route around it. Nothing in this file can be constructed
without a `RemoteGate` that a person has enabled for this session.

Every one of these does the same six things, which is why they share a base
class rather than being five hand-written HTTP clients (§6.12):

  1. **Refuse unless enabled for this session.** No env-var auto-detection —
     finding a key in the environment is not consent (M42.1).
  2. **Redact every outbound message** — the first, every subsequent turn,
     and every tool result (M43). A file slice returned to a remote model is
     outbound context like any other.
  3. **Ask before the first call**, showing what is about to leave.
  4. **Enforce a budget that HALTS.** Not a warning: a stop (M42.4).
  5. **Journal provider, model, tokens and cost.**
  6. **Emit the persistent REMOTE indicator**, so a host can keep it on
     screen for as long as it is true.

Keys come from the host's `StoragePort` or from an environment variable the
operator set deliberately, and **never touch the journal, the codemap or any
log** (M44).

All five are HTTP over stdlib `urllib`; none needs a vendor SDK. That keeps
M48 intact — the core still has zero required runtime dependencies — and it
means a version bump in somebody's client library cannot break an air-gapped
tool that never calls these anyway.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import time
from typing import Any
import urllib.error
import urllib.request

from .. import redact
from ..errors import BudgetExceeded, ConfigurationError
from ..types import Completion, Message, ModelCapabilities, ToolSpec
from .base import (
    ProviderBase,
    estimate_tokens,
    messages_to_openai,
    parse_tool_calls,
)

#: The one place a generation clock survives, and only because the risk here
#: is the opposite one. A local model that takes an hour costs an hour; a
#: hung METERED call can bill for a socket nobody is reading. The local
#: providers wait as long as it takes (see `openai_compatible.DEFAULT_TIMEOUT`)
#: — this is a hang guard on a paid connection, set well past any real answer.
DEFAULT_TIMEOUT = 1800.0


class RemoteProvider(ProviderBase):
    """What all five share. Not usable on its own."""

    name = "remote"
    is_remote = True
    endpoint = ""
    default_model = ""
    #: (input, output) price per million tokens, for the budget. Indicative
    #: only — prices change, and a wrong number here makes the budget wrong
    #: rather than absent, which is why `max_tokens` is the ceiling to trust.
    price_per_mtok: tuple[float, float] = (0.0, 0.0)

    def __init__(self, *, api_key: str = "", model: str = "",
                 gate: Any = None, budget: redact.Budget | None = None,
                 events: Any = None, journal: Any = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 redact_soft: bool = True,
                 extra_patterns: Sequence[tuple[str, str]] = (),
                 context_tokens: int = 200_000) -> None:
        if gate is None:
            raise ConfigurationError(
                f"{self.name} is a remote provider and no session gate was "
                f"given, so nothing was sent. Remote mode is enabled per "
                f"session, by a person, or not at all.")
        if not gate.allowed(self.name):
            raise ConfigurationError(
                f"{self.name} has not been enabled for this session, so "
                f"nothing was sent. Everything stays on this machine unless "
                f"you turn a remote provider on deliberately.")
        if not api_key:
            raise ConfigurationError(
                f"No API key was given for {self.name}. Keys come from the "
                f"host's settings or from an environment variable you set "
                f"yourself — this engine will not go looking for one.")
        self._key = api_key
        self.model = model or self.default_model
        self.gate = gate
        self.budget = budget or redact.Budget()
        self._events = events
        self._journal = journal
        self.timeout = timeout
        self._redact_soft = redact_soft
        self._extra = tuple(extra_patterns)
        self._ctx = context_tokens
        self.last_report: redact.RedactionReport | None = None

    # -- the shape each provider fills in --------------------------------
    def _headers(self) -> dict[str, str]:                # pragma: no cover
        raise NotImplementedError

    def _payload(self, messages: Sequence[Message], **kw) -> dict:
        raise NotImplementedError                        # pragma: no cover

    def _parse(self, data: dict, elapsed_ms: int) -> Completion:
        raise NotImplementedError                        # pragma: no cover

    # -- the part that is identical for all of them ----------------------
    def complete(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (), temperature: float = 0.15,
                 max_tokens: int = 2048, stop: Sequence[str] | None = None,
                 grammar: str | None = None, seed: int | None = None,
                 cancel: Any = None) -> Completion:
        if cancel is not None and cancel.is_set():
            return self.cancelled(self.model)

        # 1. The budget is checked BEFORE the call, not after. Checking
        #    afterwards means the call that broke the ceiling was still made
        #    and still paid for.
        stop_reason = self.budget.exceeded()
        if stop_reason:
            raise BudgetExceeded(
                "remote", stop_reason,
                f"{self.budget.calls} call(s) to {self.name}")

        # 2. Redact EVERYTHING, including tool results (M43).
        clean, report = redact.redact_messages(
            messages, soft=self._redact_soft, extra=self._extra)
        self.last_report = report
        self.gate.redactions += report.total

        # 3. Approval before the first call of the session, with the size
        #    and the redaction count in front of the person deciding.
        self.gate.check(self.name,
                        bytes_out=redact.outbound_bytes(clean),
                        estimate=redact.describe(clean, report))

        if report.total:
            self._emit("remote",
                       f"{report.summary()} before sending to {self.name}.",
                       {"provider": self.name, "enabled": True,
                        **report.as_dict()})

        payload = self._payload(clean, tools=tools, temperature=temperature,
                                max_tokens=max_tokens, stop=stop, seed=seed)
        t0 = time.monotonic()
        try:
            data = self._post(payload)
        except urllib.error.HTTPError as exc:
            # M11: never raise on a model refusal, and by extension hand back
            # something the loop can reason about. The body is NOT read — it
            # can echo the request, and the request may have contained file
            # contents.
            self._emit("error",
                       f"{self.name} returned HTTP {exc.code}. Nothing was "
                       f"generated; the session can continue locally.",
                       {"provider": self.name, "status": exc.code})
            return Completion(text="", finish_reason="error",
                              model=self.model)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError):
            self._emit("error",
                       f"{self.name} could not be reached. Nothing was "
                       f"generated, and nothing further will be sent unless "
                       f"you try again.",
                       {"provider": self.name})
            return Completion(text="", finish_reason="error",
                              model=self.model)

        completion = self._parse(data, int((time.monotonic() - t0) * 1000))

        # 4 and 5. Record the cost, then journal it — never the key (M44).
        cost = self._cost(completion.tokens_in, completion.tokens_out)
        self.budget.record(tokens_in=completion.tokens_in,
                           tokens_out=completion.tokens_out, cost=cost)
        if self._journal is not None:
            self._journal.log("budget", provider=self.name,
                              model=completion.model,
                              tokens_in=completion.tokens_in,
                              tokens_out=completion.tokens_out,
                              cost=round(cost, 6),
                              redactions=report.total,
                              remaining=self.budget.remaining(),
                              remote=True)
        self._emit("budget",
                   f"{self.name}: {self.budget.tokens:,} tokens used this "
                   f"session; {self.budget.remaining()} left.",
                   self.budget.as_dict())
        return completion

    def _post(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=body, method="POST")
        for key, value in self._headers().items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    def _cost(self, tokens_in: int, tokens_out: int) -> float:
        price_in, price_out = self.price_per_mtok
        return (tokens_in * price_in + tokens_out * price_out) / 1_000_000

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        """One chunk. Streaming a remote provider is a host concern.

        Deliberately not implemented per-provider: five streaming parsers is
        five places for a redaction bug to hide, and the value — watching
        tokens appear from a service that is already fast — is small.
        """
        yield self.complete(messages, **kw).text

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            name=self.model, family=self.name, context_tokens=self._ctx,
            supports_tools=True, supports_grammar=False,
            supports_vision=True, supports_fim=False, is_remote=True,
            token_count_is_estimate=True)

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def _emit(self, kind: str, message: str, data: dict) -> None:
        if self._events is None:
            return
        try:
            self._events.event(kind, message, data)
        except Exception:                                # noqa: BLE001
            pass


# ==========================================================================
# the five
# ==========================================================================

class Anthropic(RemoteProvider):
    name = "anthropic"
    endpoint = "https://api.anthropic.com/v1/messages"
    default_model = "claude-sonnet-4-5"
    price_per_mtok = (3.0, 15.0)

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json",
                "x-api-key": self._key,
                "anthropic-version": "2023-06-01"}

    def _payload(self, messages: Sequence[Message], **kw) -> dict:
        # Anthropic takes the system prompt out of band rather than as a
        # message, which is the one real shape difference among the five.
        system = "\n\n".join(m.content for m in messages
                             if m.role == "system")
        turns = [{"role": "assistant" if m.role == "assistant" else "user",
                  "content": m.content}
                 for m in messages if m.role != "system"]
        payload: dict[str, Any] = {
            "model": self.model, "messages": turns or [
                {"role": "user", "content": system or "."}],
            "max_tokens": kw.get("max_tokens", 2048),
            "temperature": kw.get("temperature", 0.15)}
        if system and turns:
            payload["system"] = system
        tools = kw.get("tools") or ()
        if tools:
            payload["tools"] = [{"name": t.name, "description": t.description,
                                 "input_schema": t.parameters} for t in tools]
        return payload

    def _parse(self, data: dict, elapsed_ms: int) -> Completion:
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if b.get("type") == "text")
        raw_calls = [{"id": b.get("id", ""), "name": b.get("name", ""),
                      "arguments": b.get("input", {})}
                     for b in blocks if b.get("type") == "tool_use"]
        usage = data.get("usage") or {}
        reason = {"max_tokens": "length", "tool_use": "tool_calls",
                  "end_turn": "stop", "stop_sequence": "stop"}.get(
                      str(data.get("stop_reason", "end_turn")), "stop")
        return Completion(
            text=text, tool_calls=parse_tool_calls(raw_calls),
            finish_reason=reason,
            tokens_in=int(usage.get("input_tokens", 0) or 0),
            tokens_out=int(usage.get("output_tokens", 0) or 0),
            model=str(data.get("model", self.model)), prompt_ms=elapsed_ms)


class _OpenAIShaped(RemoteProvider):
    """Everything that speaks `/v1/chat/completions`. Three of the five do."""

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json",
                "authorization": f"Bearer {self._key}"}

    def _payload(self, messages: Sequence[Message], **kw) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_openai(messages),
            "temperature": kw.get("temperature", 0.15),
            "max_tokens": kw.get("max_tokens", 2048)}
        if kw.get("stop"):
            payload["stop"] = list(kw["stop"])
        if kw.get("seed") is not None:
            payload["seed"] = kw["seed"]
        tools = kw.get("tools") or ()
        if tools:
            payload["tools"] = self.tools_payload(tools)
            payload["tool_choice"] = "auto"
        return payload

    def _parse(self, data: dict, elapsed_ms: int) -> Completion:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        calls = parse_tool_calls(message.get("tool_calls") or ())
        reason = {"length": "length", "tool_calls": "tool_calls"}.get(
            str(choice.get("finish_reason", "stop")), "stop")
        if calls and reason != "length":
            reason = "tool_calls"
        return Completion(
            text=str(message.get("content") or ""), tool_calls=calls,
            finish_reason=reason,
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
            model=str(data.get("model", self.model)), prompt_ms=elapsed_ms)


class Mistral(_OpenAIShaped):
    name = "mistral"
    endpoint = "https://api.mistral.ai/v1/chat/completions"
    default_model = "devstral-medium-latest"
    price_per_mtok = (0.4, 2.0)


class OpenRouter(_OpenAIShaped):
    name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    default_model = "mistralai/devstral-medium"
    price_per_mtok = (0.0, 0.0)     # varies per upstream model; see below

    def _headers(self) -> dict[str, str]:
        headers = super()._headers()
        # OpenRouter asks for these so it can attribute traffic. Both are
        # deliberately generic: the project name, not the operator's.
        headers["HTTP-Referer"] = "https://github.com/photogbill/cognitive-coder"
        headers["X-Title"] = "Cognitive Coder"
        return headers

    def _parse(self, data: dict, elapsed_ms: int) -> Completion:
        completion = super()._parse(data, elapsed_ms)
        # OpenRouter reports actual cost, which beats any price table we
        # could keep current. Fold it into the budget directly.
        usage = data.get("usage") or {}
        if usage.get("cost"):
            self.budget.spend += float(usage["cost"])
        return completion


class OpenAI(_OpenAIShaped):
    """Present for the benefit of users who want it.

    A settled preference of the project owner, recorded here so nobody
    "fixes" it: **implement it, do not advocate it, and do not make it a
    default anywhere.** It is not the default provider, not the default in
    the CLI, and not first in any list.
    """
    name = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"
    default_model = "gpt-4.1-mini"
    price_per_mtok = (0.4, 1.6)


class Google(RemoteProvider):
    name = "google"
    endpoint = ("https://generativelanguage.googleapis.com/v1beta/models/"
                "{model}:generateContent")
    default_model = "gemini-2.5-flash"
    price_per_mtok = (0.3, 2.5)

    def _headers(self) -> dict[str, str]:
        return {"content-type": "application/json",
                "x-goog-api-key": self._key}

    def _post(self, payload: dict) -> dict:
        # Google puts the model in the URL rather than the body, so the
        # endpoint is per-call.
        body = json.dumps(payload).encode("utf-8")
        url = self.endpoint.format(model=self.model)
        req = urllib.request.Request(url, data=body, method="POST")
        for key, value in self._headers().items():
            req.add_header(key, value)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    def _payload(self, messages: Sequence[Message], **kw) -> dict:
        system = "\n\n".join(m.content for m in messages
                             if m.role == "system")
        contents = [{"role": "model" if m.role == "assistant" else "user",
                     "parts": [{"text": m.content}]}
                    for m in messages if m.role != "system"]
        payload: dict[str, Any] = {
            "contents": contents or [{"role": "user",
                                      "parts": [{"text": system or "."}]}],
            "generationConfig": {
                "temperature": kw.get("temperature", 0.15),
                "maxOutputTokens": kw.get("max_tokens", 2048)}}
        if system and contents:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    def _parse(self, data: dict, elapsed_ms: int) -> Completion:
        candidates = data.get("candidates") or [{}]
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        text = "".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata") or {}
        reason = {"MAX_TOKENS": "length"}.get(
            str(candidates[0].get("finishReason", "STOP")), "stop")
        return Completion(
            text=text, finish_reason=reason,
            tokens_in=int(usage.get("promptTokenCount", 0) or 0),
            tokens_out=int(usage.get("candidatesTokenCount", 0) or 0),
            model=self.model, prompt_ms=elapsed_ms)


#: Registered by `providers/__init__.py`. Order is alphabetical rather than
#: preferential — a list is an implicit recommendation and this one is not.
REMOTE_PROVIDERS: dict[str, tuple[type, str]] = {
    "anthropic": (Anthropic, "Claude, via api.anthropic.com."),
    "google": (Google, "Gemini, via generativelanguage.googleapis.com."),
    "mistral": (Mistral, "Mistral's hosted models, including Devstral."),
    "openai": (OpenAI, "OpenAI's hosted models."),
    "openrouter": (OpenRouter, "OpenRouter, which fronts many upstreams and "
                               "reports real cost per call."),
}


def key_from(storage: Any, provider: str, *, env: bool = True) -> str:
    """A key from the host's storage, or from an env var set DELIBERATELY.

    The environment is checked only when the caller asks for it, and finding
    a key there still does not enable anything (M42.1) — the gate is a
    separate, explicit act. This function answers "what key would we use if
    we were allowed to", never "are we allowed to".
    """
    import os

    if storage is not None:
        stored = storage.get(f"cognitive_coder.keys.{provider}", "")
        if stored:
            return str(stored)
    if not env:
        return ""
    return os.environ.get({
        "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY"}.get(provider, ""), "")
