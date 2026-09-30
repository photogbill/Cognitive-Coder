# SPDX-License-Identifier: Apache-2.0
"""The provider layer's pure logic — where wrongness hides quietly.

None of this needs a model or a network. The JSON repair parser in particular
deserves a hard test: it is the thing standing between "the model's arguments
were nearly right" and a tool call silently doing nothing, and **its second
job is to report that it had to repair at all** (D9). A repair that hides
itself turns a chronically malformed model into a mystery.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.providers import base  # noqa: E402
from cognitive_coder.providers.openai_compatible import (  # noqa: E402
    OpenAICompatible,
    is_local_url,
)
from cognitive_coder.types import Message, ToolSpec  # noqa: E402

# --------------------------------------------------------------------------
# JSON repair (D9)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected,repaired", [
    ('{"name": "load"}', {"name": "load"}, False),
    ('  {"name": "load"}  ', {"name": "load"}, False),
    # trailing comma
    ('{"name": "load",}', {"name": "load"}, True),
    ('{"a": [1, 2,],}', {"a": [1, 2]}, True),
    # single quotes
    ("{'name': 'load'}", {"name": "load"}, True),
    # prose before the object
    ('Sure! Here you go: {"name": "load"}', {"name": "load"}, True),
    # a fence around it
    ('```json\n{"name": "load"}\n```', {"name": "load"}, True),
    # a line comment
    ('{\n  // the symbol\n  "name": "load"\n}', {"name": "load"}, True),
    # prose either side
    ('I think {"name": "load"} is right.', {"name": "load"}, True),
])
def test_near_json_is_repaired_and_the_repair_is_reported(text, expected,
                                                          repaired):
    value, was_repaired = base.repair_json(text)
    assert value == expected
    assert was_repaired is repaired, (
        "a silent repair hides a chronically malformed model")


def test_something_that_was_never_json_is_not_pretended_into_one():
    """"The model didn't answer with arguments" and "it answered badly" are
    different problems, and the caller needs to tell them apart."""
    assert base.repair_json("just some prose") == ({}, False)
    assert base.repair_json("") == ({}, False)
    assert base.repair_json("[1, 2, 3]") == ({}, False)   # not an object


def test_an_apostrophe_inside_a_string_is_not_mangled():
    """Rewriting every `'` turns "it's" into a syntax error — a worse
    outcome than not repairing at all."""
    value, _ = base.repair_json("{'msg': \"it's fine\"}")
    assert value == {"msg": "it's fine"}


def test_nested_objects_survive_the_balance_matcher():
    value, was_repaired = base.repair_json(
        'here: {"a": {"b": {"c": 1}}, "d": [{"e": 2}]} — done')
    assert value == {"a": {"b": {"c": 1}}, "d": [{"e": 2}]}
    assert was_repaired


def test_a_brace_inside_a_string_does_not_end_the_object():
    value, _ = base.repair_json('prefix {"pattern": "a { b }", "n": 1} suffix')
    assert value == {"pattern": "a { b }", "n": 1}


# --------------------------------------------------------------------------
# tool calls
# --------------------------------------------------------------------------

def test_tool_call_arguments_arrive_parsed():
    calls = base.parse_tool_calls([
        {"id": "c1", "function": {"name": "search_codemap",
                                  "arguments": '{"name": "load"}'}}])
    assert calls[0].name == "search_codemap"
    assert calls[0].arguments == {"name": "load"}
    assert not calls[0].repaired


def test_a_repaired_tool_call_carries_the_flag():
    calls = base.parse_tool_calls([
        {"id": "c1", "function": {"name": "read_slice",
                                  "arguments": "{'path': 'a.py',}"}}])
    assert calls[0].arguments == {"path": "a.py"}
    assert calls[0].repaired


def test_a_missing_id_gets_a_deterministic_one():
    calls = base.parse_tool_calls([{"function": {"name": "x",
                                                 "arguments": "{}"}}])
    assert calls[0].id == "call_0"


def test_already_parsed_arguments_are_accepted():
    calls = base.parse_tool_calls([
        {"id": "c", "function": {"name": "x", "arguments": {"a": 1}}}])
    assert calls[0].arguments == {"a": 1}
    assert not calls[0].repaired


# --------------------------------------------------------------------------
# message shaping
# --------------------------------------------------------------------------

def test_messages_convert_to_the_wire_shape():
    out = base.messages_to_openai([
        Message(role="system", content="be helpful"),
        Message(role="user", content="hello")])
    assert out == [{"role": "system", "content": "be helpful"},
                   {"role": "user", "content": "hello"}]


def test_a_tool_result_carries_its_call_id():
    out = base.messages_to_openai([
        Message(role="tool", content="the answer", tool_call_id="c1")])
    assert out[0]["tool_call_id"] == "c1"


def test_images_become_content_parts():
    out = base.messages_to_openai([
        Message(role="user", content="what is this",
                images=((b"\x89PNG\r\n", "image/png"),))])
    parts = out[0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_a_tool_spec_converts_to_the_openai_shape():
    spec = ToolSpec(name="search", description="find it",
                    parameters={"type": "object", "properties": {}})
    wire = spec.to_openai()
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "search"


# --------------------------------------------------------------------------
# model identification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,family", [
    ("Devstral-Small-2-24B-Instruct-2512", "mistral"),
    ("magistral-small-2509", "mistral"),
    ("Qwen2.5-Coder-32B", "qwen"),
    ("Llama-3.3-70B", "llama"),
    ("deepseek-coder-v2", "deepseek"),
    ("something-nobody-has-heard-of", "unknown"),
])
def test_the_chat_template_family_is_guessed_from_the_name(name, family):
    """Guessing is honest here: a wrong guess costs a slightly worse
    template, not a wrong answer — and the alternative is asking the operator
    a question they should not have to answer."""
    assert base.family_for(name) == family


@pytest.mark.parametrize("name,expected", [
    ("codestral-22b", True),
    ("Devstral-Small-2-24B", True),
    ("qwen2.5-coder-7b", True),
    ("llama-3.3-70b-instruct", False),
])
def test_fill_in_the_middle_support_is_recognised(name, expected):
    """G.4 — FIM structurally cannot touch code outside the hole, which is
    the clean answer to a model rewriting what it was not asked to."""
    assert base.supports_fim(name) is expected


def test_the_token_estimate_is_never_zero():
    """A zero would make a budget divide-by-zero or admit infinite context."""
    assert base.estimate_tokens("") >= 1
    assert base.estimate_tokens("a" * 400) > 50


# --------------------------------------------------------------------------
# local versus remote (C3)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url,local", [
    ("http://127.0.0.1:8080", True),
    ("http://localhost:11434", True),
    ("http://[::1]:8080", True),
    ("http://192.168.1.40:8080", True),
    ("http://10.0.0.5:1234", True),
    ("http://172.16.0.9:8000", True),
    ("https://api.anthropic.com", False),
    ("https://openrouter.ai/api/v1", False),
])
def test_local_and_remote_endpoints_are_told_apart(url, local):
    assert is_local_url(url) is local


def test_a_provider_knows_whether_it_is_remote_from_its_url():
    assert not OpenAICompatible("http://127.0.0.1:8080").is_remote
    assert OpenAICompatible("https://api.example.com").is_remote


@pytest.fixture
def no_dns(monkeypatch):
    """Classification must not resolve names — before consent, a lookup is
    already traffic, and its answer is not evidence of locality."""
    import socket

    lookups: list[str] = []

    def answer(host, *args, **kwargs):
        lookups.append(host)
        # What split-horizon DNS and a rebinding domain would say.
        return "127.0.0.1"

    monkeypatch.setattr(socket, "gethostbyname", answer)
    monkeypatch.setattr(socket, "getaddrinfo", answer)
    return lookups


@pytest.mark.parametrize("url,local", [
    ("http://localhost:8080", True),
    ("http://LOCALHOST:8080", True),
    ("http://llm.localhost:8080", True),
    ("http://127.0.0.1:8080", True),
    ("http://[::1]:8080", True),
    ("http://[::ffff:127.0.0.1]:8080", True),
    ("http://169.254.1.1:8080", True),
    ("http://0.0.0.0:8080", True),
    # A NAME is remote, whatever a resolver would say about it: a rebinding
    # domain resolved to 127.0.0.1 at classification and was called local.
    ("http://rebind.example:8080", False),
    ("http://llm.corp.example:8080", False),
    ("http://mybox.local:8080", False),
    ("http://localhost.example.com", False),
    ("http://127.0.0.1@api.openai.com/", False),     # userinfo trick
    ("http://[::ffff:8.8.8.8]:8080", False),         # v4-mapped public
    ("http://100.64.0.1:8080", False),               # CGNAT is not a LAN
    # Scheme-less strings used to parse with an EMPTY host, which counted as
    # local: "api.openai.com" was classified as this machine.
    ("api.openai.com", False),
    ("api.openai.com:443/v1", False),
    ("", False),
    ("not a url", False),
    ("file:///etc/passwd", False),
    ("ftp://127.0.0.1/", False),
])
def test_classification_uses_no_dns(no_dns, url, local):
    assert is_local_url(url) is local
    assert no_dns == [], f"classifying {url!r} resolved {no_dns}"


@pytest.mark.parametrize("url", ["api.openai.com", "", "not a url",
                                 "ftp://127.0.0.1/", "file:///etc/passwd",
                                 "http://"])
def test_an_unusable_url_is_refused_with_a_sentence(url):
    from cognitive_coder.errors import ConfigurationError
    with pytest.raises(ConfigurationError) as exc:
        OpenAICompatible(url)
    text = str(exc.value)
    assert "http://" in text and "Traceback" not in text


def test_a_provider_that_cannot_reach_its_endpoint_says_so_calmly():
    """M11's neighbour: a dead endpoint is a Completion with
    finish_reason="error", not an exception the loop has to special-case."""
    provider = OpenAICompatible("http://127.0.0.1:9", timeout=1.0)
    out = provider.complete([Message(role="user", content="hi")])
    assert out.finish_reason == "error"
    assert out.text == ""


def test_capabilities_of_an_unreachable_endpoint_report_nothing_loaded():
    """M10 — "no model loaded" is a normal state, reported not raised."""
    provider = OpenAICompatible("http://127.0.0.1:9", timeout=1.0)
    caps = provider.capabilities()
    assert not caps.loaded
    assert caps.context_tokens > 0        # a budget still has an answer


# --------------------------------------------------------------------------
# local_llamacpp timings (M55, G.7.5)
# --------------------------------------------------------------------------

class _FakeLlama:
    """Enough of llama-cpp-python's `Llama` to time a call.

    Note what is absent: `get_timings()`. The provider called it, it does
    not exist on `Llama`, and the fallback was the whole call's wall time.
    """
    model_path = "/models/devstral.gguf"

    def __init__(self, stream_refuses_tools=False):
        self.calls: list[dict] = []
        self.stream_refuses_tools = stream_refuses_tools

    def n_ctx(self):
        return 16384

    def create_chat_completion(self, **kw):
        import time
        self.calls.append(kw)
        if not kw.get("stream"):
            time.sleep(0.5)
            return {"choices": [{"message": {"content": "whole"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        if self.stream_refuses_tools and kw.get("tools"):
            raise ValueError("streaming is not supported with tools")

        def gen():
            time.sleep(0.2)
            for piece in ("a", "b", "c", "d", "e"):
                yield {"choices": [{"delta": {"content": piece},
                                    "finish_reason": None}]}
                time.sleep(0.1)
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 5}}
        return gen()


def test_local_llamacpp_measures_prefill_and_decode_apart():
    from cognitive_coder.providers.local_llamacpp import LocalLlamaCpp
    out = LocalLlamaCpp(llama=_FakeLlama()).complete(
        [Message(role="user", content="hi")])
    assert out.text == "abcde"
    assert 0 < out.prompt_ms < out.decode_ms, (out.prompt_ms, out.decode_ms)
    assert (out.tokens_in, out.tokens_out) == (3, 5)


def test_local_llamacpp_falls_back_when_streaming_is_refused():
    from cognitive_coder.providers.local_llamacpp import LocalLlamaCpp
    llama = _FakeLlama(stream_refuses_tools=True)
    out = LocalLlamaCpp(llama=llama).complete(
        [Message(role="user", content="hi")],
        tools=[ToolSpec(name="x", description="d",
                        parameters={"type": "object"})])
    assert out.text == "whole"
    assert [bool(c.get("stream")) for c in llama.calls] == [True, False]


def test_local_llamacpp_cancel_is_checked_per_chunk():
    from cognitive_coder.providers.local_llamacpp import LocalLlamaCpp

    class After:
        def __init__(self):
            self.n = 2

        def is_set(self):
            self.n -= 1
            return self.n < 0

    out = LocalLlamaCpp(llama=_FakeLlama()).complete(
        [Message(role="user", content="hi")], cancel=After())
    assert out.finish_reason == "cancelled"


# --------------------------------------------------------------------------
# JSON repair must not rewrite string contents (item 9)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,key,expected", [
    # The observed corruption: an `apply_patch` argument whose OLD text
    # contained `[1, 2,]` came back as `[1, 2]` and no longer matched.
    ('{"old": "xs = [1, 2,]", "new": "ys = []",}', "old", "xs = [1, 2,]"),
    ('{"old": "f(a, b,)\\n}", "n": 1,}', "old", "f(a, b,)\n}"),
    ('{"url": "http://h//x", "n": 1,}', "url", "http://h//x"),
    ('{"c": "a // not a comment", "n": 1,}', "c", "a // not a comment"),
    ("{'old': 'a = [1,]', 'n': 1,}", "old", "a = [1,]"),
])
def test_repair_leaves_string_contents_alone(text, key, expected):
    value, repaired = base.repair_json(text)
    assert repaired
    assert value[key] == expected


def test_repair_still_strips_comments_and_commas_outside_strings():
    value, repaired = base.repair_json(
        '{\n  "a": [1, 2,], // trailing\n  /* block */ "b": "x",\n}')
    assert value == {"a": [1, 2], "b": "x"} and repaired


def test_find_json_object_takes_the_answer_not_the_echoed_schema():
    """A reasoning model restates the schema it was shown before answering;
    the FIRST balanced object is the example, not the answer."""
    text = ('The schema is {"security": [{"title": "one line", '
            '"severity": "high|medium|low"}], "performance": []}. '
            'Looking at the code... here is my answer:\n'
            '```json\n{"security": [{"title": "SQL injection", '
            '"severity": "high"}], "performance": []}\n```\n'
            'Hope that helps {"note": "x"}')
    found = base.find_json_object(text, keys=("security", "performance"))
    assert found["security"][0]["title"] == "SQL injection"


def test_find_json_object_without_keys_takes_the_last():
    assert base.find_json_object('{"a": 1} then {"a": 2}') == {"a": 2}
    assert base.find_json_object("no json here") == {}
    assert base.find_json_object('{"s": "a } b"} x') == {"s": "a } b"}


def test_find_json_object_an_empty_answer_beats_the_echo():
    text = ('Schema: {"security": [{"title": "one line"}], '
            '"performance": []}\nNothing to report: {}')
    assert base.find_json_object(text, keys=("security",)) == {}
