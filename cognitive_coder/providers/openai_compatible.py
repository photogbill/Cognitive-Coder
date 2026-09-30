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
(llama.cpp does), and otherwise measured as the time to the first streamed
token, with the rest as `decode_ms`. It is required by M55 because it is the
only way anyone notices the prefix cache breaking.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import ipaddress
import json
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from .. import redact
from ..errors import BudgetExceeded, ConfigurationError
from ..types import Completion, Message, ModelCapabilities, ToolSpec
from .base import (
    ProviderBase,
    StreamAccumulator,
    _as_int,
    _as_ms,
    as_dict,
    error_detail,
    estimate_tokens,
    family_for,
    first_choice,
    http_error_detail,
    map_finish,
    messages_to_openai,
    parse_tool_calls,
    sse_data,
    supports_fim,
)

DEFAULT_URL = "http://127.0.0.1:8080"

#: NO CEILING ON GENERATION. `0` means wait as long as the model takes.
#:
#: Bill: *"if there are any time limits on generation they should be removed.
#: I might at some point use a 70B GGUF Q4_K_M since I can CPU offload
#: heavily, and accept the lengthy generations. The whole thing is that I
#: don't want to lose final result quality for speed reasons."*
#:
#: This was 900s, which is a number that looks generous until you do the
#: arithmetic for the machine it has to serve. A 70B Q4_K_M with most layers
#: on the CPU decodes at roughly 1–2 tok/s. A 4096-token file is then 35 to 70
#: minutes, and the old ceiling would have cut it off at fifteen — mid-file,
#: with nothing to show for the wait.
#:
#: And a truncated generation is the worst possible outcome, not a middling
#: one. It costs the wait AND the result, then hands the repair loop a file
#: that stops mid-function: a syntax error no rewrite can fix, because the
#: problem was never in the code. The attempts drain against it.
#:
#: A hung server is the thing a timeout is supposed to catch, and it still is
#: — but that is what `/health` is for (see `reachable()`), and it does not
#: require guessing how long an honest answer should take.
DEFAULT_TIMEOUT = 0.0

#: An opener with NO proxies, for local endpoints. `urlopen`'s default
#: opener honours `HTTP_PROXY`/`http_proxy` (and, on Windows, the registry
#: proxy) with no loopback bypass unless `NO_PROXY` happens to list it — so on
#: a corporate laptop a request to 127.0.0.1 was observed arriving at the
#: proxy, and the "local" model call left the machine by default. An empty
#: `ProxyHandler` is how urllib spells "connect directly". Building it opens
#: no socket.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open(url_is_local: bool, req: urllib.request.Request,
          timeout: float | None) -> Any:
    """Open `req` directly when local; through the system's proxies if not.

    Remote keeps the environment's proxy on purpose: a network that
    REQUIRES its proxy for the internet must still work once a person has
    turned remote mode on.
    """
    if url_is_local:
        return _DIRECT.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _host_of(url: str) -> str:
    """The lower-cased host of an http(s) URL, or "" if it is not one."""
    try:
        parts = urllib.parse.urlparse(url or "")
        host = parts.hostname or ""
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https"):
        return ""
    return host.lower()


def is_local_url(url: str) -> bool:
    """True for loopback and private-range hosts. The C3 judgement call.

    Decided from the URL's text alone, with NO name resolution. It used to
    call `gethostbyname` for every non-literal host, at construction, before
    any consent — a lookup is already traffic — and the answer was not even
    evidence: a rebinding domain that resolved to 127.0.0.1 at classification
    was called local, and could resolve anywhere by the time of the call.

    So: literal loopback, private and link-local addresses are local, as are
    `localhost` and `*.localhost` (RFC 6761 reserves them for this machine).
    ANY other name is remote — the conservative answer, because it is the
    one that makes the engine ask. A LAN box by name (`mybox.local`) is
    reached by its address instead. A string that is not an http(s) URL is
    not local; the constructor refuses it with a sentence.
    """
    host = _host_of(url)
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    # `::ffff:a.b.c.d` is an IPv4 address in IPv6 clothing; judge the
    # address it maps to, or `::ffff:8.8.8.8` could pass as "private".
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local)


class OpenAICompatible(ProviderBase):
    """An `LLMPort` over any `/v1/chat/completions` endpoint."""

    name = "openai_compatible"

    def __init__(self, base_url: str = DEFAULT_URL, *, model: str = "",
                 api_key: str = "", timeout: float = DEFAULT_TIMEOUT,
                 context_tokens: int = 0, supports_tools: bool | None = None,
                 supports_vision: bool = False,
                 headers: dict | None = None, gate: Any = None,
                 budget: redact.Budget | None = None, events: Any = None,
                 journal: Any = None, redact_soft: bool = True,
                 extra_patterns: Sequence[tuple[str, str]] = ()) -> None:
        if not _host_of(base_url):
            # "api.openai.com" with no scheme parsed with an EMPTY host,
            # and an empty host was classified local — so the most likely
            # typo for a public endpoint was the one that skipped the gate.
            raise ConfigurationError(
                f"{base_url!r} is not a usable endpoint address, so nothing "
                f"was contacted. Give the full URL with its scheme, for "
                f"example http://127.0.0.1:8080 for a local server.")
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
        self._probed_at = 0.0
        self.last_prompt_ms = 0
        #: None until known; False once the server refused a stream and
        #: accepted the same request without one.
        self._stream_ok: bool | None = None
        # The remote half. Unused when the URL is local — 127.0.0.1 is this
        # machine and C3 has nothing to say about it — but REQUIRED when it
        # is not: the observed leak was this class pointed at a public URL,
        # enabled, and sending the raw prompt (an AWS key included) with no
        # approval asked, no redaction and `gate.bytes_out == 0`, because the
        # gate was consulted once at construction and never again.
        self.gate = gate
        self.budget = budget if budget is not None else redact.Budget()
        self._events = events
        self._journal = journal
        self._redact_soft = redact_soft
        self._extra = tuple(extra_patterns)
        self.last_report: redact.RedactionReport | None = None
        #: One numbering for the life of this provider (a session's
        #: conversation). Per call, the same key was
        #: `[REDACTED:aws_key_id_2]` in one request and
        #: `[REDACTED:aws_key_id]` in the next, and the model
        #: treated one credential as two. Held in memory only.
        self._redaction = redact.RedactionState()

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
        wait = self._patience(timeout if timeout else self.timeout)
        with _open(not self.is_remote, req, wait) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def _get(self, path: str, timeout: float = 10.0) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        with _open(not self.is_remote, req, timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    # -- C3: the remote sequence -------------------------------------------
    def _outbound(self, messages: Sequence[Message]) -> list[Message]:
        """What may be sent: budget → redact → approve, when remote.

        The same order as `RemoteProvider.complete`, for the same reasons:
        the budget is checked BEFORE the call (afterwards, the call that
        broke the ceiling was already paid for), redaction happens before
        the approval so the person deciding sees the scrubbed size and the
        count, and `gate.check` raises rather than returning False so a
        caller cannot forget to look. A local URL returns the messages
        untouched — there is nobody to redact them from.
        """
        if not self.is_remote:
            return list(messages)
        if self.gate is None:
            raise ConfigurationError(
                f"{self.name} is pointed at {self.base_url}, which is not a "
                f"local address, and no session gate was given, so nothing "
                f"was sent. Remote mode is enabled per session, by a "
                f"person, or not at all.")
        stop_reason = self.budget.exceeded()
        if stop_reason:
            raise BudgetExceeded(
                "remote", stop_reason,
                f"{self.budget.calls} call(s) to {self.name}")
        clean, report = redact.redact_messages(
            messages, soft=self._redact_soft, extra=self._extra,
            state=self._redaction)
        self.last_report = report
        self.gate.redactions += report.total
        self.gate.check(self.name, bytes_out=redact.outbound_bytes(clean),
                        estimate=redact.describe(clean, report))
        if report.total:
            self._emit("remote",
                       f"{report.summary()} before sending to {self.name}.",
                       {"provider": self.name, "enabled": True,
                        **report.as_dict()})
        return clean

    def _record_remote(self, tokens_in: int, tokens_out: int,
                       model: str) -> None:
        """Count a remote call against the budget and journal it (M42.4).

        Never the key (M44): nothing here has seen it. Cost is recorded as
        zero because an arbitrary OpenAI-compatible endpoint publishes no
        price; the token ceiling is the one to trust, and it is enforced.
        """
        if not self.is_remote:
            return
        self.budget.record(tokens_in=tokens_in, tokens_out=tokens_out)
        report = self.last_report
        if self._journal is not None:
            try:
                self._journal.log(
                    "budget", provider=self.name, model=model,
                    tokens_in=tokens_in, tokens_out=tokens_out, cost=0.0,
                    redactions=report.total if report else 0,
                    remaining=self.budget.remaining(), remote=True)
            except Exception:                            # noqa: BLE001
                pass
        self._emit("budget",
                   f"{self.name}: {self.budget.tokens:,} tokens used this "
                   f"session; {self.budget.remaining()} left.",
                   self.budget.as_dict())

    def _emit(self, kind: str, message: str, data: dict) -> None:
        if self._events is None:
            return
        try:
            self._events.event(kind, message, data)
        except Exception:                                # noqa: BLE001
            pass

    # -- the Port ---------------------------------------------------------
    def _payload(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (), temperature: float = 0.15,
                 max_tokens: int = 2048, stop: Sequence[str] | None = None,
                 grammar: str | None = None, seed: int | None = None,
                 stream: bool = True) -> dict[str, Any]:
        """The request body, identical for `complete()` and `stream()`.

        One builder because there were two, and the display path had
        quietly lost `stop`, `seed`, `grammar` and the configured headers.
        """
        payload: dict[str, Any] = {
            "model": self.model or "local",
            "messages": messages_to_openai(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            # Usage arrives in a final chunk only when asked for; without it
            # a streamed call reports zero tokens.
            payload["stream_options"] = {"include_usage": True}
        if not self.is_remote:
            # llama.cpp keeps the KV cache for the shared prefix only when
            # asked; that is G.7's whole premise. Local only: OpenAI's own
            # API rejects arguments it does not recognise with a 400.
            payload["cache_prompt"] = True
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = seed
        if grammar:
            # llama.cpp's GBNF field. Grammar-constrained decoding is
            # strictly better than repairing near-JSON afterwards (D9), so it
            # is used whenever the server accepts it.
            payload["grammar"] = grammar
        if tools and self.capabilities().supports_tools:
            payload["tools"] = self.tools_payload(tools)
            payload["tool_choice"] = "auto"
        # M12: a host whose model lacks tool support IGNORES `tools`. Sending
        # them anyway to a server that doesn't understand them is how a
        # perfectly good local setup starts returning 400s.
        return payload

    def _request(self, payload: dict) -> urllib.request.Request:
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/json")
        if self._key:
            req.add_header("Authorization", f"Bearer {self._key}")
        for k, v in self._headers.items():
            req.add_header(k, v)
        return req

    def complete(self, messages: Sequence[Message], *,
                 tools: Sequence[ToolSpec] = (), temperature: float = 0.15,
                 max_tokens: int = 2048, stop: Sequence[str] | None = None,
                 grammar: str | None = None, seed: int | None = None,
                 cancel: Any = None) -> Completion:
        """One answer, fetched as a STREAM so its timings mean something.

        It used to be one blocking POST, which made `prompt_ms` the whole
        call and left `decode_ms` at zero — so `cache_health()` withheld its
        verdict for every run and G.7.5's prefix-cache check was dead.
        Streaming separates them (see `StreamAccumulator`) and lets cancel
        be honoured per chunk rather than after the last token. A server
        that refuses streaming is asked once more without it, and not
        asked to stream again until `refresh()`.
        """
        if cancel is not None and cancel.is_set():
            return self.cancelled(self.model)
        messages = self._outbound(messages)
        opts = {"tools": tools, "temperature": temperature,
                "max_tokens": max_tokens, "stop": stop, "grammar": grammar,
                "seed": seed}
        if self._stream_ok is not False:
            out = self._exchange(self._payload(messages, **opts), cancel)
            if out is not None:
                self._stream_ok = True
                return self._finish(out)
        out = self._exchange(self._payload(messages, stream=False, **opts),
                             cancel)
        if out is not None and out.finish_reason != "error":
            # Only now is the refusal known to be about streaming, rather
            # than about the request (a 400 for `tools` fails both ways).
            self._stream_ok = False
        return self._finish(out or self.failed(
            f"The model server at {self.base_url} refused the request. "
            f"Nothing was generated.", model=self.model))

    def _finish(self, completion: Completion) -> Completion:
        self.last_prompt_ms = completion.prompt_ms
        if completion.finish_reason not in ("error", "cancelled"):
            self._record_remote(completion.tokens_in,
                                completion.tokens_out, completion.model)
        return completion

    #: Statuses that can mean "not like THAT" about streaming itself, so a
    #: non-streaming retry is worth one more request. Auth and rate limits
    #: are not among them: asking again differently cannot fix those.
    _STREAM_REFUSED = (400, 404, 405, 415, 422, 501)

    def _exchange(self, payload: dict, cancel: Any) -> Completion | None:
        """POST once; read SSE or a JSON body. None = streaming refused.

        Every failure comes back as a Completion whose `error` is a
        sentence naming what happened — the HTTP status and the server's
        own message where there was one — never an exception (M11).
        """
        where = f"The model server at {self.base_url}"
        t0 = time.monotonic()

        def ms() -> int:
            return int((time.monotonic() - t0) * 1000)

        try:
            with _open(not self.is_remote, self._request(payload),
                       self._patience(self.timeout)) as r:
                ctype = str(r.headers.get("Content-Type") or "").lower()
                if "event-stream" not in ctype:
                    raw = r.read()
                    try:
                        data = json.loads(raw.decode("utf-8", "replace"))
                    except ValueError:
                        kind = ctype.split(";")[0] or "no content type"
                        return self.failed(
                            f"{where} answered with something that is not "
                            f"a chat completion ({kind}). Nothing was "
                            f"generated.", model=self.model, prompt_ms=ms())
                    return self._from_body(data, ms(), where)
                acc = StreamAccumulator(t0, self.model)
                for data in sse_data(r):
                    if cancel is not None and cancel.is_set():
                        return self.cancelled(acc.model)
                    if data.strip() == "[DONE]":
                        break
                    try:
                        acc.feed(json.loads(data), time.monotonic())
                    except ValueError:
                        continue
                    if acc.error is not None:
                        break
                if acc.error is not None:
                    detail = error_detail({"error": acc.error})
                    return self.failed(
                        f"{where} reported an error part-way through its "
                        f"answer" + (f": {detail}" if detail else "")
                        + ". The partial answer was discarded.",
                        model=acc.model, prompt_ms=ms())
                if not acc.events:
                    return self.failed(
                        f"{where} ended its stream without sending an "
                        f"answer. Nothing was generated.",
                        model=self.model, prompt_ms=ms())
                return acc.completion(time.monotonic())
        except urllib.error.HTTPError as exc:
            if payload.get("stream") and exc.code in self._STREAM_REFUSED:
                return None
            detail = http_error_detail(exc)
            hint = ""
            if exc.code == 400 and payload.get("tools"):
                # The observed case: llama.cpp started without --jinja
                # answers a request carrying `tools` with a 400, and the
                # operator was told only that it "could not be reached".
                hint = (" The request offered tools; a llama.cpp server "
                        "started without --jinja rejects them, so restart "
                        "it with --jinja or turn tool use off.")
            return self.failed(
                f"{where} answered HTTP {exc.code}"
                + (f": {detail}" if detail else "") + "."
                + hint + " Nothing was generated.",
                model=self.model, prompt_ms=ms(), status=exc.code)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = error_detail({"error": str(getattr(exc, "reason", exc)
                                                or type(exc).__name__)})
            return self.failed(
                f"{where} could not be reached"
                + (f" ({reason})" if reason else "")
                + ". Nothing was generated; check that it is running.",
                model=self.model, prompt_ms=ms())

    def _from_body(self, data: Any, elapsed_ms: int,
                   where: str = "") -> Completion:
        """A whole (non-streamed) chat completion.

        Without server timings this path cannot separate prefill from
        decode, so it says so: `prompt_ms` is the whole call and `decode_ms`
        stays 0, which `cache_health()` reads as "no verdict possible"
        instead of mistaking a slow answer for a broken cache.
        """
        where = where or f"The model server at {self.base_url}"
        choice = first_choice(data)
        msg = choice.get("message")
        if not isinstance(msg, dict):
            detail = error_detail(data)
            return self.failed(
                f"{where} answered in a shape that is not a chat completion"
                + (f" ({detail})" if detail
                   else " (no choices[0].message)")
                + ". Nothing was generated.",
                model=self.model, prompt_ms=elapsed_ms)
        usage = as_dict(data.get("usage"))
        timings = as_dict(data.get("timings"))
        # llama.cpp reports prompt processing separately, which is the number
        # that actually answers "did the prefix cache hold?" (G.7.5).
        prompt_ms = _as_ms(timings.get("prompt_ms"))
        decode_ms = _as_ms(timings.get("predicted_ms"))
        calls = parse_tool_calls(
            [c for c in msg.get("tool_calls") or () if isinstance(c, dict)])
        return Completion(
            text=str(msg.get("content") or ""), tool_calls=calls,
            finish_reason=map_finish(str(choice.get("finish_reason") or ""),
                                     bool(calls)),
            tokens_in=_as_int(usage.get("prompt_tokens")),
            tokens_out=_as_int(usage.get("completion_tokens")),
            model=str(data.get("model") or self.model),
            prompt_ms=prompt_ms if prompt_ms is not None else elapsed_ms,
            decode_ms=decode_ms or 0)

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        """Server-sent events, for display. Cancel is checked per chunk."""
        cancel = kw.pop("cancel", None)
        messages = self._outbound(messages)
        opts = {k: kw[k] for k in ("tools", "temperature", "max_tokens",
                                   "stop", "grammar", "seed") if k in kw}
        payload = self._payload(messages, **opts)
        acc = StreamAccumulator(time.monotonic(), self.model)
        try:
            with _open(not self.is_remote, self._request(payload),
                       self._patience(self.timeout)) as r:
                for data in sse_data(r):
                    if cancel is not None and cancel.is_set():
                        return
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    before = len(acc.parts)
                    acc.feed(chunk, time.monotonic())
                    yield from acc.parts[before:]
        except (urllib.error.URLError, OSError, TimeoutError):
            return
        finally:
            # Counted even when the stream carried no usage: the call was
            # made, and a budget that only counts calls it can price lets an
            # unpriced endpoint run without a ceiling.
            self._record_remote(_as_int(acc.usage.get("prompt_tokens")),
                                _as_int(acc.usage.get("completion_tokens")),
                                acc.model)

    #: Seconds a probe stays fresh. M13 says capabilities describe the
    #: CURRENTLY loaded model, and the probe used to be cached for the
    #: process lifetime: the docstring said the session calls `refresh()` at
    #: task boundaries, and nothing did, so a model swapped (or a server
    #: restarted with a different context size) behind the URL was never
    #: noticed. The session re-reads `capabilities()` at every task
    #: boundary, so a short TTL turns that into a real re-probe without
    #: the session having to know about it — and within a burst of calls
    #: it costs nothing.
    probe_ttl: float = 15.0

    def capabilities(self) -> ModelCapabilities:
        """From `/v1/models` (and llama.cpp's `/props`), at most `probe_ttl`
        seconds old; `refresh()` re-probes immediately."""
        now = time.monotonic()
        if self._probed is None or now - self._probed_at >= self.probe_ttl:
            self._probed = self._probe()
            self._probed_at = time.monotonic()
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
        """Forget what the server said and ask again, now.

        Also forgets a remembered streaming refusal: a different model or
        a restarted server may well stream.
        """
        self._probed = None
        self._stream_ok = None
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

        Never for a remote endpoint: `/tokenize` carries the raw text (the
        loop hands it whole files), so on a public URL it would be an
        unredacted, unapproved upload beside the gated one.
        """
        if self.is_remote:
            return estimate_tokens(text)
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
        # A candidate that is not local is not probed at all: this runs
        # before any session exists, so there is no gate to ask, and "is it
        # up?" is still a connection to somebody else's machine.
        if not is_local_url(url):
            continue
        try:
            req = urllib.request.Request(f"{url.rstrip('/')}/v1/models")
            with _open(True, req, timeout):
                found.append(url)
        except Exception:                                # noqa: BLE001
            continue
    return found
