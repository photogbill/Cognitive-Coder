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
