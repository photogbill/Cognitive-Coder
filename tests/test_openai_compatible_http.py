# SPDX-License-Identifier: Apache-2.0
"""`OpenAICompatible` against a real (local, fake) HTTP server.

The pure-logic tests in `test_providers.py` cannot see the failures these
catch, because every one of them lives on the wire: what bytes left, which
socket they went to, how long the first token took, and what a server that
answers with the wrong shape does to `complete()`.

The server binds 127.0.0.1 and never anything else. Where a test needs the
provider to BELIEVE its endpoint is remote, it patches the classifier rather
than pointing at a public host, so no test here can leave the machine.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import redact  # noqa: E402
from cognitive_coder.errors import (  # noqa: E402
    BudgetExceeded,
    ConfigurationError,
)
from cognitive_coder.ports import AutoApprove, RecordingEvents  # noqa: E402
from cognitive_coder.providers import RemoteGate, make_provider  # noqa: E402
from cognitive_coder.providers import openai_compatible as oc  # noqa: E402
from cognitive_coder.types import Message  # noqa: E402

SECRET = "AKIAIOSFODNN7EXAMPLE"
_PROXY_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
               "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


class FakeServer:
    """A scriptable `/v1/*` endpoint that records every request."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict, bytes]] = []
        self.post_handler = None      # callable(handler, body) or None
        self.models = [{"id": "served-model",
                        "meta": {"n_ctx_train": 32768}}]
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args):
                pass

            def do_GET(self):
                outer.requests.append(("GET", self.path,
                                       dict(self.headers), b""))
                if self.path.endswith("/v1/models"):
                    self._send(200, json.dumps(
                        {"data": outer.models}).encode())
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(n) if n else b""
                outer.requests.append(("POST", self.path,
                                       dict(self.headers), body))
                if outer.post_handler is not None:
                    outer.post_handler(self, body)
                    return
                self._send(200, json.dumps({
                    "choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1},
                    "model": "answering-model"}).encode())

            def _send(self, code, body, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def posts(self, path: str = "/v1/chat/completions") -> list[bytes]:
        return [r[3] for r in self.requests
                if r[0] == "POST" and r[1].endswith(path)]

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


@pytest.fixture
def no_proxy_env(monkeypatch):
    """This sandbox has proxy variables of its own; none may leak in."""
    for key in _PROXY_VARS:
        monkeypatch.delenv(key, raising=False)


def send_sse(handler, frames, *, before=0.0, between=0.0,
             ctype="text/event-stream"):
    """Write `frames` (bytes) as a stream, flushing each one.

    HTTP/1.0 with no Content-Length: the connection closing ends the body,
    which is how a streaming server that never sends `[DONE]` behaves too.
    """
    time.sleep(before)
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.end_headers()
    handler.wfile.flush()
    for frame in frames:
        handler.wfile.write(frame)
        handler.wfile.flush()
        if between:
            time.sleep(between)


def chunk(content=None, *, finish=None, **extra) -> bytes:
    """One `data:` frame carrying a delta, SSE-framed."""
    delta = {} if content is None else {"content": content}
    body = {"choices": [{"delta": delta, "finish_reason": finish}],
            **extra}
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()


class _Journal:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict]] = []

    def log(self, event, **fields):
        self.rows.append((event, fields))


def _remote(monkeypatch, server, *, approve=True, budget=None, journal=None):
    """An openai_compatible provider that believes its URL is public.

    Only the CLASSIFICATION is faked; every byte still goes to 127.0.0.1.
    """
    monkeypatch.setattr(oc, "is_local_url", lambda url: False)
    approval = AutoApprove(remote=approve)
    events = RecordingEvents()
    gate = RemoteGate(events, approval)
    gate.enable("openai_compatible", reason="the test asked")
    provider = make_provider(
        "openai_compatible", gate=gate, base_url=server.url, model="m",
        supports_tools=False, budget=budget, journal=journal, events=events)
    return provider, gate, approval, events


# --------------------------------------------------------------------------
# item 1 — C3: a remote-classified openai_compatible goes through the gate
# --------------------------------------------------------------------------

def test_remote_openai_compatible_asks_redacts_and_counts(
        monkeypatch, server, no_proxy_env):
    """The observed leak: with the provider enabled and a public URL, the raw
    prompt containing an AWS key left the machine, `approve_remote` was never
    called, and `gate.bytes_out` stayed 0."""
    journal = _Journal()
    provider, gate, approval, _events = _remote(monkeypatch, server,
                                                journal=journal)
    assert provider.is_remote
    out = provider.complete([Message(role="user",
                                     content=f'token = "{SECRET}"')])
    assert out.finish_reason == "stop"
    assert approval.remote_asks, "approve_remote was never called"
    sent = b"".join(server.posts())
    assert sent, "nothing reached the server at all"
    assert SECRET.encode() not in sent, "the raw secret left the machine"
    assert b"REDACTED" in sent
    assert gate.bytes_out > 0
    assert gate.redactions >= 1
    assert any(event == "budget" and f.get("remote")
               for event, f in journal.rows)
    assert provider.budget.calls == 1


def test_remote_openai_compatible_budget_halts_before_the_call(
        monkeypatch, server, no_proxy_env):
    budget = redact.Budget(max_tokens=10)
    budget.record(tokens_in=6, tokens_out=6)
    provider, _gate, approval, _ = _remote(monkeypatch, server,
                                           budget=budget)
    with pytest.raises(BudgetExceeded):
        provider.complete([Message(role="user", content="hi")])
    assert not server.posts(), "the call that broke the ceiling was made"
    assert not approval.remote_asks


def test_remote_openai_compatible_declined_sends_nothing(
        monkeypatch, server, no_proxy_env):
    provider, _gate, approval, _ = _remote(monkeypatch, server,
                                           approve=False)
    with pytest.raises(ConfigurationError):
        provider.complete([Message(role="user", content="hi")])
    assert approval.remote_asks
    assert not server.posts()


def test_remote_openai_compatible_stream_goes_through_the_gate(
        monkeypatch, server, no_proxy_env):
    provider, gate, approval, _ = _remote(monkeypatch, server)
    list(provider.stream([Message(role="user",
                                  content=f'token = "{SECRET}"')]))
    assert approval.remote_asks
    sent = b"".join(server.posts())
    assert sent and SECRET.encode() not in sent
    assert gate.bytes_out > 0


def test_remote_count_tokens_never_posts_the_text(
        monkeypatch, server, no_proxy_env):
    """`/tokenize` carries raw file contents; for a remote endpoint that is
    an unredacted, unapproved upload, so the estimate is used instead."""
    provider, *_ = _remote(monkeypatch, server)
    n = provider.count_tokens(f"x = '{SECRET}'")
    assert n >= 1
    assert not server.posts("/tokenize")


# --------------------------------------------------------------------------
# item 2 — the system proxy is for remote endpoints only
# --------------------------------------------------------------------------

@pytest.fixture
def proxy(monkeypatch):
    """A second fake server standing in for the corporate proxy."""
    p = FakeServer()
    for key in _PROXY_VARS:
        monkeypatch.delenv(key, raising=False)
    for key in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(key, p.url)
    # `urlopen` caches its opener, proxies included, on first use; clearing
    # it reproduces a process that STARTED with the proxy set, which is the
    # real case (the variable comes from the login environment).
    monkeypatch.setattr(urllib.request, "_opener", None)
    yield p
    p.close()


def test_a_local_request_does_not_reach_the_proxy(server, proxy):
    provider = oc.OpenAICompatible(server.url, model="m",
                                   supports_tools=False)
    out = provider.complete([Message(role="user", content="hi")])
    assert out.text == "ok"
    assert server.posts(), "the local server never saw the request"
    assert not proxy.requests, [r[1] for r in proxy.requests]


def test_a_remote_request_still_honours_the_proxy(monkeypatch, server,
                                                  proxy):
    """A corporate network that REQUIRES its proxy for the internet must
    still work once remote mode is on — the bypass is for local only."""
    monkeypatch.setattr(oc, "is_local_url", lambda url: False)
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("openai_compatible")
    provider = make_provider("openai_compatible", gate=gate,
                             base_url=server.url, model="m",
                             supports_tools=False)
    provider.complete([Message(role="user", content="hi")])
    assert any(r[0] == "POST" for r in proxy.requests)


def test_a_local_provider_is_untouched_by_the_gate(server, no_proxy_env):
    """No gate, no approval, no redaction: 127.0.0.1 is this machine."""
    provider = make_provider("openai_compatible", base_url=server.url,
                             model="m", supports_tools=False)
    assert not provider.is_remote
    out = provider.complete([Message(role="user",
                                     content=f'token = "{SECRET}"')])
    assert out.text == "ok"
    assert SECRET.encode() in b"".join(server.posts())


# --------------------------------------------------------------------------
# item 4 — complete() streams, so prefill and decode are measured apart
# --------------------------------------------------------------------------

def _local(server, **kw):
    return oc.OpenAICompatible(server.url, model="m", supports_tools=False,
                               **kw)


def _model_like(handler, body):
    """300 ms of 'prefill', then 700 ms of tokens — like a real model.

    Answers non-streaming requests too, in one piece after the whole
    second, which is what the pre-streaming provider saw and reported as
    a 1000 ms `prompt_ms`.
    """
    req = json.loads(body)
    if not req.get("stream"):
        time.sleep(1.0)
        handler._send(200, json.dumps({
            "choices": [{"message": {"content": "x" * 7},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
        ).encode())
        return
    frames = [chunk("x") for _ in range(7)]
    frames.append(chunk(finish="stop"))
    frames.append(b'data: {"choices":[],"usage":{"prompt_tokens":5,'
                  b'"completion_tokens":7}}\n\n')
    frames.append(b"data: [DONE]\n\n")
    send_sse(handler, frames, before=0.3, between=0.1)


def test_prefill_and_decode_are_measured_apart(server, no_proxy_env):
    """G.7.5's prefix-cache check was dead for every shipped provider:
    `prompt_ms` was the whole call and `decode_ms` was never set, so
    `cache_health()` always withheld its verdict."""
    server.post_handler = _model_like
    out = _local(server).complete([Message(role="user", content="hi")])
    assert out.text == "x" * 7
    assert out.prompt_ms > 0 and out.decode_ms > 0
    assert out.prompt_ms < out.decode_ms, (out.prompt_ms, out.decode_ms)
    assert 200 <= out.prompt_ms <= 700
    assert (out.tokens_in, out.tokens_out) == (5, 7)
    sent = json.loads(server.posts()[-1])
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["cache_prompt"] is True


def test_llamacpp_timings_override_the_wall_clock(server, no_proxy_env):
    """A fully cached prefix is SUB-millisecond; it must stay tiny, not be
    replaced by the wall time because `int(0.6)` is falsy."""
    def handler(h, body):
        send_sse(h, [chunk("a"), chunk("b", finish="stop"),
                     b'data: {"choices":[],"timings":{"prompt_ms":0.6,'
                     b'"predicted_ms":900.0}}\n\n',
                     b"data: [DONE]\n\n"], before=0.2)
    server.post_handler = handler
    out = _local(server).complete([Message(role="user", content="hi")])
    assert out.text == "ab"
    assert out.prompt_ms == 1
    assert out.decode_ms == 900


def test_llamacpp_reports_how_much_of_the_prompt_it_processed(
        server, no_proxy_env):
    """`timings.prompt_n` is the exact answer to "did the prefix cache
    hold?" — 40 of 4,000 tokens processed means 3,960 came from cache."""
    def handler(h, body):
        send_sse(h, [chunk("a", finish="stop"),
                     b'data: {"choices":[],"usage":{"prompt_tokens":4000,'
                     b'"completion_tokens":1},"timings":{"prompt_n":40,'
                     b'"prompt_ms":30.0,"predicted_ms":90.0}}\n\n',
                     b"data: [DONE]\n\n"])
    server.post_handler = handler
    out = _local(server).complete([Message(role="user", content="hi")])
    assert (out.tokens_in, out.prompt_processed) == (4000, 40)


def test_an_openai_shaped_cached_count_is_read_too():
    from cognitive_coder.providers.base import processed_tokens
    usage = {"prompt_tokens": 1000,
             "prompt_tokens_details": {"cached_tokens": 900}}
    assert processed_tokens({}, usage) == 100
    assert processed_tokens({}, {"prompt_tokens": 1000}) == 0


def test_sse_frames_are_assembled_per_the_spec(server, no_proxy_env):
    """Comments, CRLF, and one JSON object split across two `data:` lines
    (legal SSE: the lines join with a newline)."""
    def handler(h, body):
        send_sse(h, [
            b": keep-alive\n\n",
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\r\n\r\n',
            b'data: {"choices":[{"delta":\n',
            b'data: {"content":"lo "}}]}\n\n',
            b"event: ignored\nid: 7\n"
            b'data: {"choices":[{"delta":{"content":"there"}}]}\n\n',
            b"data: [DONE]\n\n"])
    server.post_handler = handler
    out = _local(server).complete([Message(role="user", content="hi")])
    assert out.text == "Hello there"


def test_a_utf8_character_split_across_writes_survives(server,
                                                       no_proxy_env):
    def handler(h, body):
        frame = chunk("w\u00f6rld")
        cut = frame.index("\u00f6".encode()) + 1      # inside the o-umlaut
        send_sse(h, [frame[:cut], frame[cut:], b"data: [DONE]\n\n"],
                 between=0.05)
    server.post_handler = handler
    assert _local(server).complete(
        [Message(role="user", content="hi")]).text == "w\u00f6rld"


def test_nothing_after_done_is_read(server, no_proxy_env):
    def handler(h, body):
        send_sse(h, [chunk("kept"), b"data: [DONE]\n\n", chunk("LOST")])
    server.post_handler = handler
    assert _local(server).complete(
        [Message(role="user", content="hi")]).text == "kept"


def test_streamed_tool_calls_are_reassembled(server, no_proxy_env):
    def handler(h, body):
        def tc(**fn):
            d = {"tool_calls": [{"index": 0, "id": "c1",
                                 "function": fn}]}
            return ("data: " + json.dumps(
                {"choices": [{"delta": d}]}) + "\n\n").encode()
        send_sse(h, [tc(name="read_slice", arguments=""),
                     tc(arguments='{"path": '), tc(arguments='"a.py"}'),
                     chunk(finish="tool_calls"), b"data: [DONE]\n\n"])
    server.post_handler = handler
    provider = oc.OpenAICompatible(server.url, model="m",
                                   supports_tools=True)
    from cognitive_coder.types import ToolSpec
    out = provider.complete(
        [Message(role="user", content="hi")],
        tools=[ToolSpec(name="read_slice", description="d",
                        parameters={"type": "object"})])
    assert out.finish_reason == "tool_calls"
    assert out.tool_calls[0].name == "read_slice"
    assert out.tool_calls[0].arguments == {"path": "a.py"}


def test_a_server_that_refuses_streaming_is_asked_once_without(
        server, no_proxy_env):
    def handler(h, body):
        if json.loads(body).get("stream"):
            h._send(400, b'{"error":{"message":"stream not supported"}}')
        else:
            h._send(200, json.dumps({"choices": [{"message": {
                "content": "plain"}, "finish_reason": "stop"}]}).encode())
    server.post_handler = handler
    provider = _local(server)
    assert provider.complete([Message(role="user", content="a")]).text == \
        "plain"
    assert provider.complete([Message(role="user", content="b")]).text == \
        "plain"
    flags = [json.loads(b).get("stream") for b in server.posts()]
    assert flags == [True, False, False], flags


def test_cancel_is_checked_per_chunk(server, no_proxy_env):
    class After:
        def __init__(self, n):
            self.n = n

        def is_set(self):
            self.n -= 1
            return self.n < 0

    def handler(h, body):
        send_sse(h, [chunk("a")] * 50 + [b"data: [DONE]\n\n"],
                 between=0.01)
    server.post_handler = handler
    out = _local(server).complete([Message(role="user", content="hi")],
                                  cancel=After(3))
    assert out.finish_reason == "cancelled"


def test_the_display_stream_sends_what_complete_sends(server,
                                                      no_proxy_env):
    """`stream()` dropped stop/seed/grammar and the configured headers,
    and split a multi-line data frame into two unparseable halves."""
    def handler(h, body):
        send_sse(h, [b'data: {"choices":[{"delta":\n',
                     b'data: {"content":"joined"}}]}\n\n',
                     b"data: [DONE]\n\n"])
    server.post_handler = handler
    provider = _local(server, headers={"X-Custom": "1"})
    text = "".join(provider.stream([Message(role="user", content="hi")],
                                   stop=["END"], seed=7,
                                   grammar="root ::= x"))
    assert text == "joined"
    sent = json.loads(server.posts()[-1])
    assert {"stop", "seed", "grammar"} <= set(sent)
    assert server.requests[-1][2].get("X-Custom") == "1"


# --------------------------------------------------------------------------
# item 5 — any body shape is a Completion; an HTTP status is a sentence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("body,ctype", [
    (b"[]", "application/json"),
    (b"null", "application/json"),
    (b'{"choices": [null]}', "application/json"),
    (b'{"choices": [{"message": "hi"}]}', "application/json"),
    (b"<html>proxy says hi</html>", "text/html"),
])
def test_a_wrong_shaped_body_is_an_error_completion_not_a_crash(
        server, no_proxy_env, body, ctype):
    """Each of these raised AttributeError straight out of `complete()`."""
    server.post_handler = lambda h, _b: h._send(200, body, ctype)
    out = _local(server).complete([Message(role="user", content="hi")])
    assert out.finish_reason == "error"
    assert out.text == "", "an error sentence in `text` is read as code"
    assert "not a chat completion" in out.error
    assert "Traceback" not in out.error


def test_an_http_error_names_its_status_and_message(server, no_proxy_env):
    """llama.cpp without `--jinja` rejects `tools` with a 400 and says why;
    the operator saw only "could not be reached, or returned an error"."""
    journal = _Journal()
    server.post_handler = lambda h, _b: h._send(400, json.dumps({"error": {
        "message": "tools param requires --jinja flag",
        "type": "invalid_request_error"}}).encode())
    provider = oc.OpenAICompatible(server.url, model="m",
                                   supports_tools=False, journal=journal)
    out = provider.complete([Message(role="user", content="hi")])
    assert out.finish_reason == "error" and out.text == ""
    assert "400" in out.error
    assert "tools param requires --jinja flag" in out.error
    assert any(e == "error" and "400" in f.get("sentence", "")
               for e, f in journal.rows)


def test_an_error_message_is_redacted_and_capped(server, no_proxy_env):
    """A server can echo the request in its error, and the request can
    carry file contents; what reaches a sentence is scrubbed and short."""
    message = f"bad input near key={SECRET} " + "x" * 2000
    server.post_handler = lambda h, _b: h._send(
        500, json.dumps({"error": {"message": message}}).encode())
    out = _local(server).complete([Message(role="user", content="hi")])
    assert "500" in out.error
    assert SECRET not in out.error
    assert len(out.error) < 500


def test_an_unreachable_server_is_a_sentence_too():
    out = oc.OpenAICompatible("http://127.0.0.1:9", model="m",
                              timeout=2.0).complete(
        [Message(role="user", content="hi")])
    assert out.finish_reason == "error"
    assert "could not be reached" in out.error


def test_an_error_inside_the_stream_is_reported(server, no_proxy_env):
    server.post_handler = lambda h, _b: send_sse(h, [
        chunk("par"),
        b'data: {"error": {"message": "context size exceeded"}}\n\n'])
    out = _local(server).complete([Message(role="user", content="hi")])
    assert out.finish_reason == "error"
    assert "context size exceeded" in out.error


def test_remote_parsers_survive_wrong_shapes(monkeypatch):
    from cognitive_coder.providers import remote as R
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    for name in ("anthropic", "google", "mistral"):
        gate.enable(name)
    for body in ([], None, {"choices": [None]}, {"content": [None, "x"]},
                 {"candidates": [None]}, {"choices": [{"message": "hi"}]}):
        for cls in (R.Anthropic, R.Google, R.Mistral):
            monkeypatch.setattr(cls, "_post", lambda self, p, b=body: b)
            out = cls(api_key="k", gate=gate).complete(
                [Message(role="user", content="hi")])
            assert out.finish_reason == "error", (cls.name, body)
            assert out.error


def test_remote_timeout_zero_means_no_limit(server, no_proxy_env):
    """`urlopen(timeout=0)` is a NON-BLOCKING socket, not "no limit": every
    call failed instantly and looked like the service being down."""
    from cognitive_coder.providers import remote as R

    class Local(R._OpenAIShaped):
        name = "mistral"
        endpoint = f"{server.url}/v1/chat/completions"

    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("mistral")
    out = Local(api_key="k", gate=gate, timeout=0).complete(
        [Message(role="user", content="ping")])
    assert out.finish_reason == "stop" and out.text == "ok"


def test_remote_http_error_names_status_and_message(server, no_proxy_env):
    from cognitive_coder.providers import remote as R

    class Local(R._OpenAIShaped):
        name = "mistral"
        endpoint = f"{server.url}/v1/chat/completions"

    server.post_handler = lambda h, _b: h._send(429, json.dumps(
        {"error": {"message": "rate limit reached"}}).encode())
    events = RecordingEvents()
    gate = RemoteGate(events, AutoApprove(remote=True))
    gate.enable("mistral")
    out = Local(api_key="k", gate=gate, events=events).complete(
        [Message(role="user", content="ping")])
    assert out.finish_reason == "error"
    assert "429" in out.error and "rate limit reached" in out.error


def test_local_llamacpp_survives_wrong_shapes():
    from cognitive_coder.providers.local_llamacpp import LocalLlamaCpp

    class Odd:
        model_path = "x.gguf"

        def create_chat_completion(self, **kw):
            if kw.get("stream"):
                raise ValueError("no streaming here")
            return []

    out = LocalLlamaCpp(llama=Odd()).complete(
        [Message(role="user", content="hi")])
    assert out.finish_reason == "error" and out.error


# --------------------------------------------------------------------------
# item 6 — capabilities() follows the server, not the first answer it got
# --------------------------------------------------------------------------

def test_capabilities_notice_a_model_swap_once_stale(server, no_proxy_env):
    """M13: capabilities describe the CURRENTLY loaded model. The probe was
    cached for the process lifetime and nothing ever called `refresh()`, so
    a model swapped behind the server was never noticed."""
    provider = oc.OpenAICompatible(server.url)
    assert provider.capabilities().name == "served-model"
    server.models = [{"id": "SWAPPED", "meta": {"n_ctx_train": 4096}}]
    provider.probe_ttl = 0.0
    caps = provider.capabilities()
    assert caps.name == "SWAPPED"
    assert caps.context_tokens == 4096


def test_capabilities_are_cached_within_the_ttl(server, no_proxy_env):
    provider = oc.OpenAICompatible(server.url)
    provider.capabilities()
    gets = len([r for r in server.requests if r[0] == "GET"])
    for _ in range(5):
        provider.capabilities()
    assert len([r for r in server.requests if r[0] == "GET"]) == gets


def test_refresh_reprobes_now(server, no_proxy_env):
    provider = oc.OpenAICompatible(server.url)
    provider.capabilities()
    server.models = [{"id": "SWAPPED"}]
    assert provider.capabilities().name == "served-model"   # cached
    assert provider.refresh().name == "SWAPPED"
