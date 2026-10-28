# SPDX-License-Identifier: Apache-2.0
"""Model profiles (profiles.py, 2026-10-27).

A profile per model family: its card's samplers go with every call the
port takes them on, its temperature is the ladder's base, its edit format
sets `narrow` when the operator left that to the engine, and its thinking
switch is thrown off for every reply the engine parses — with the text
switch on the prompt and the template switch on the port.
"""

from __future__ import annotations

from cognitive_coder import profiles
from cognitive_coder.personas import ENGINEER
from cognitive_coder.ports import (
    AutoApprove,
    Host,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.providers import openai_compatible as oc
from cognitive_coder.session import Session, SessionConfig
from cognitive_coder.types import Completion, Message, ModelCapabilities


def test_the_table_knows_the_usual_coding_models():
    q = profiles.builtin_for("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
    assert q.family == "qwen3-coder"
    assert q.sampling() == {"top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
    assert q.temperature == 0.7 and q.fim == "psm" and q.thinking == "none"
    d = profiles.builtin_for("mistralai_Devstral-Small-2-24B-Instruct-2512")
    assert d.family == "devstral" and d.temperature == 0.15
    assert d.fim == "unknown"             # probed, never assumed
    assert profiles.builtin_for("some-random-model") is profiles.GENERIC
    assert profiles.builtin_for("") is profiles.GENERIC


def test_qwen3_variants_differ_in_how_they_think():
    switch = profiles.builtin_for("Qwen3-30B-A3B-Q4_K_M.gguf")
    assert switch.thinking == "switch"
    assert switch.text_switch == "/no_think" and switch.template_switch
    always = profiles.builtin_for("Qwen3-30B-A3B-Thinking-2507-Q4.gguf")
    assert always.thinking == "always" and not always.has_switch
    never = profiles.builtin_for("Qwen3-30B-A3B-Instruct-2507-Q4.gguf")
    assert never.thinking == "none"
    # the coder is matched before the plain family
    assert profiles.builtin_for("qwen3-coder-480b").family == "qwen3-coder"


def test_the_ladder_climbs_from_the_profile_temperature():
    assert profiles.rungs_for(profiles.GENERIC) == (0.15, 0.45, 0.70)
    q = profiles.builtin_for("qwen3-coder")
    assert profiles.rungs_for(q) == (0.7, 1.0, 1.0)
    d = profiles.builtin_for("devstral")
    assert profiles.rungs_for(d) == (0.15, 0.45, 0.70)


class _OldPort:
    """A host written to the 1.0 signature: no sampling, no thinking."""

    def complete(self, messages, *, tools=(), temperature=0.15,
                 max_tokens=2048, stop=None, grammar=None, seed=None,
                 cancel=None):
        return Completion(text="")


def test_kwargs_reach_only_a_port_that_takes_them():
    q = profiles.builtin_for("qwen3-coder")
    assert profiles.call_kwargs(_OldPort(), q, thinking=False) == {}
    new = ScriptedLLM([])
    kw = profiles.call_kwargs(new, q, thinking=False)
    assert kw == {"sampling": {"top_p": 0.8, "top_k": 20,
                               "repeat_penalty": 1.05}}
    # thinking is sent only for a model with a switch
    s = profiles.builtin_for("qwen3-30b")
    kw = profiles.call_kwargs(new, s, thinking=False)
    assert kw["thinking"] is False and kw["sampling"]["top_p"] == 0.8
    always = profiles.builtin_for("magistral")
    assert "thinking" not in profiles.call_kwargs(new, always, thinking=False)
    # an explicit sampler overrides the profile's
    kw = profiles.call_kwargs(new, q, sampling={"top_p": 0.5})
    assert kw["sampling"]["top_p"] == 0.5
    assert profiles.call_kwargs(new, None) == {}


def test_the_text_switch_lands_once_on_the_last_user_turn():
    s = profiles.builtin_for("qwen3-30b")
    msgs = [Message(role="system", content="sys"),
            Message(role="user", content="write it")]
    out = profiles.soft_switch(msgs, s, thinking=False)
    assert out[-1].content == "write it\n/no_think"
    assert out[0].content == "sys"
    again = profiles.soft_switch(out, s, thinking=False)
    assert again[-1].content.count("/no_think") == 1
    # not when thinking is wanted, nor for a model without a text switch
    assert profiles.soft_switch(msgs, s, thinking=True)[-1].content == \
        "write it"
    assert profiles.soft_switch(
        msgs, profiles.builtin_for("devstral"), thinking=False
    )[-1].content == "write it"


def test_a_calibrated_profile_is_found_before_the_table(tmp_path):
    q = profiles.builtin_for("qwen3-coder")
    measured = profiles.replace(q, edit_format="symbols",
                                measured={"symbols": {"pass": 0.8}})
    path = profiles.save_calibrated("Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
                                    measured, directory=tmp_path)
    assert path.name == "Qwen3-Coder-30B-A3B.json"
    found = profiles.profile_for("models/Qwen3-Coder-30B-A3B-Q4_K_M.gguf",
                                 directory=tmp_path)
    assert found.source == "calibrated" and found.edit_format == "symbols"
    assert found.measured["symbols"]["pass"] == 0.8
    assert found.sampling() == q.sampling()
    # a host's own dict wins over both
    given = profiles.profile_for("qwen3-coder", given={"temperature": 0.3},
                                 directory=tmp_path)
    assert given.source == "operator" and given.temperature == 0.3


def test_describe_is_one_readable_line():
    text = profiles.builtin_for("qwen3-30b").describe()
    assert text.startswith("profile: Qwen3 (built-in)")
    assert "thinking switch" in text and "top p 0.8" in text


# --------------------------------------------------------------------------
# in a session
# --------------------------------------------------------------------------

PLAN = "greet.py — say hello\n"
FILE = "```python\ndef greet(name):\n    return f'hi {name}'\n```"


def _session(name: str, replies, **cfg) -> tuple[Session, ScriptedLLM]:
    llm = ScriptedLLM(list(replies), supports_tools=False, name=name)
    host = Host(llm=llm, fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    return Session(host, config=SessionConfig(
        attempts=1, review_after_build=False, mutation_check=False,
        adjudicate=False, **cfg)), llm


def test_the_session_applies_the_profile_to_every_call():
    session, llm = _session("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                            [PLAN, FILE])
    session.run("a greet function")
    assert session.model_profile.family == "qwen3-coder"
    assert session.loop.profile is session.model_profile
    events = [e for e in session.journal.events() if e["event"] == "profile"]
    assert events and events[0]["data"]["family"] == "qwen3-coder"
    # the plan call and the file call both carried the card's samplers
    assert all(c.get("sampling", {}).get("top_p") == 0.8 for c in llm.calls)
    # and the file was asked for at the card's temperature, not 0.15
    assert llm.calls[-1]["temperature"] == 0.7


def test_a_switch_model_is_told_not_to_think_into_a_file():
    session, llm = _session("Qwen3-30B-A3B-Q4_K_M.gguf", [PLAN, FILE])
    session.run("a greet function")
    assert all(c.get("thinking") is False for c in llm.calls)
    assert all(p[-1].content.rstrip().endswith("/no_think")
               for p in llm.prompts)
    gen = [e for e in session.journal.events() if e["event"] == "generate"]
    assert gen and gen[-1]["prompt_sha256"]


def test_the_sampling_line_reaches_the_build_log():
    session, llm = _session("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                            [PLAN, FILE])
    session.run("a greet function")
    s = session.loop.sampling_for(1, ENGINEER)
    assert s.temperature == 0.7
    assert "top_p=0.8" in s.describe() and "repeat_penalty=1.05" in s.describe()


def test_the_edit_format_sets_narrow_when_it_was_auto():
    session, _llm = _session("x", [PLAN, FILE], profile={
        "family": "x", "edit_format": "symbols"})
    session.run("a greet function")
    assert session.loop.config.narrow == "always"
    session, _llm = _session("x", [PLAN, FILE], narrow="off", profile={
        "family": "x", "edit_format": "symbols"})
    session.run("a greet function")
    assert session.loop.config.narrow == "off", "the operator's choice stands"


def test_a_generic_model_changes_nothing():
    session, llm = _session("scripted", [PLAN, FILE])
    session.run("a greet function")
    assert session.model_profile is profiles.GENERIC
    assert all("sampling" not in c and "thinking" not in c for c in llm.calls)
    assert llm.calls[-1]["temperature"] == 0.15


# --------------------------------------------------------------------------
# the OpenAI-compatible provider's payload
# --------------------------------------------------------------------------

def _provider(grammar: bool) -> oc.OpenAICompatible:
    p = oc.OpenAICompatible("http://127.0.0.1:1", model="m",
                            supports_tools=False)
    p.capabilities = lambda: ModelCapabilities(           # type: ignore
        name="m", family="qwen", context_tokens=8192,
        supports_grammar=grammar)
    return p


def test_the_payload_carries_the_samplers_llama_cpp_takes():
    p = _provider(grammar=True)
    body = p._payload([Message(role="user", content="x")],
                      sampling={"top_p": 0.8, "top_k": 20,
                                "repeat_penalty": 1.05, "min_p": None},
                      thinking=False)
    assert body["top_p"] == 0.8 and body["top_k"] == 20
    assert body["repeat_penalty"] == 1.05 and "min_p" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_a_server_that_did_not_answer_props_gets_top_p_only():
    p = _provider(grammar=False)
    body = p._payload([Message(role="user", content="x")],
                      sampling={"top_p": 0.8, "top_k": 20}, thinking=None)
    assert body["top_p"] == 0.8 and "top_k" not in body
    assert "chat_template_kwargs" not in body


# --------------------------------------------------------------------------
# a host handing over the wrong settings entry
# --------------------------------------------------------------------------

def test_a_dict_that_names_no_profile_field_does_not_blank_the_table():
    """ATK's settings already have a `ccoder.profile` — the operator's
    skill level, `{"skill_level": "senior"}`. Handed to the session as the
    MODEL profile it used to become a blank "operator" profile: Devstral
    lost its 0.15, Qwen its FIM tokens, silently. A dict that says nothing
    a profile could hold is ignored; the table answers."""
    prof = profiles.profile_for("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                                given={"skill_level": "senior"})
    assert prof.family == "qwen3-coder" and prof.source == "built-in"
    assert prof.fim_tokens
    # a dict that does say something is still the operator's
    prof = profiles.profile_for("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
                                given={"temperature": 0.3})
    assert prof.source == "operator" and prof.temperature == 0.3
