# SPDX-License-Identifier: Apache-2.0
"""The interview before the build (interview.py, 2026-10-27).

One structured call lists the decisions a request leaves open; the
operator answers (or the model's default stands); each decision becomes
a MUST sentence in the request, and so a hard requirement pinned in every
prompt of the build.
"""

from __future__ import annotations

import json

from cognitive_coder import constraints, interview
from cognitive_coder.ports import (
    AutoApprove,
    Host,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.session import Session, SessionConfig

REQUEST = ("A deterministic starship simulation. Ships have a heading in "
           "degrees and move on a 2D grid.")
REPLY = json.dumps({"questions": [
    {"id": "q1", "question": "Which way is north on the grid?",
     "options": ["up: +y is north", "down: north is (0, -1)"],
     "default": "up: +y is north",
     "why": "every bearing and movement formula depends on it"},
    {"id": "q2", "question": "What happens when a heading goes past 360?",
     "options": ["wrap around to 0", "raise an error"],
     "default": "wrap around to 0", "why": "normalize_angle"},
    {"question": "too short?", "options": ["a"], "default": "a"},
    {"id": "q4", "question": "Which way is north on the grid?",
     "options": ["x", "y"], "default": "x"},
]})


def _session(replies):
    llm = ScriptedLLM(list(replies), supports_tools=False, name="m")
    host = Host(llm=llm, fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    return Session(host, config=SessionConfig(interview=True)), llm, host


def test_clean_keeps_usable_questions_with_a_default_among_the_options():
    qs = interview.clean(json.loads(REPLY)["questions"])
    assert [q.id for q in qs] == ["q1", "q2"]            # the dupe and the
    assert qs[0].default == "up: +y is north"             # one-option one go
    assert qs[1].options == ("wrap around to 0", "raise an error")
    # a default that is not an option exactly falls to the nearest, then
    # the first
    [q] = interview.clean([{"question": "Units for distance in the sim?",
                            "options": ["km", "grid units"],
                            "default": "grid"}])
    assert q.default == "grid units"
    [q] = interview.clean([{"question": "Units for distance in the sim?",
                            "options": ["km", "grid units"],
                            "default": "parsecs"}])
    assert q.default == "km"
    assert interview.clean("not a list") == []
    assert interview.clean([{"question": "x?", "options": []}]) == []


def test_the_session_asks_once_and_journals_the_questions():
    session, llm, host = _session([REPLY])
    asked = session.interview(REQUEST)
    assert asked.asked and asked.ok and len(asked.questions) == 2
    assert len(llm.prompts) == 1
    body = [m.content for m in llm.prompts[0] if m.role == "user"][0]
    assert "decisions it leaves OPEN" in body and REQUEST in body
    events = [e for e in session.journal.events() if e["event"] == "interview"]
    assert events and events[0]["data"]["count"] == 2
    assert any("interview: 2 decision(s)" in m
               for _k, m, _d in host.events.of("status"))


def test_answers_become_must_sentences_the_constraints_pin():
    session, _llm, _host = _session([REPLY])
    asked = session.interview(REQUEST)
    interview.answer(asked, {"q1": "down: north is (0, -1)", "2": "",
                             "q9": "ignored"})
    assert asked.answers == {"q1": "down: north is (0, -1)"}
    text = asked.apply(REQUEST)
    assert text.startswith(REQUEST)
    assert interview.HEADING in text
    assert ("- The code MUST follow this decision: Which way is north on the "
            "grid: down: north is (0, -1).") in text
    # the second decision is the model's default, said as such
    assert "wrap around to 0." in text
    found = constraints.extract(text)
    pinned = " ".join(c for c in found.items)
    assert "north is (0, -1)" in pinned and "wrap around to 0" in pinned


def test_a_model_that_finds_nothing_open_costs_one_call():
    session, llm, _host = _session([json.dumps({"questions": []})])
    asked = session.interview(REQUEST)
    assert not asked.ok and "no open decision" in asked.problem
    assert asked.apply(REQUEST) == REQUEST
    assert len(llm.prompts) == 1
    session, _llm, _host = _session(["not json at all", "still not"])
    asked = session.interview(REQUEST)
    assert not asked.ok and asked.problem


def test_the_terminal_interview_reads_numbers_text_and_enter():
    asked = interview.Interview(questions=interview.clean(
        json.loads(REPLY)["questions"]))
    replies = iter(["2", "", "typed answer"])
    lines: list[str] = []
    interview.ask_on_terminal(asked, read=lambda _p: next(replies),
                              write=lines.append)
    assert asked.answers["q1"] == "down: north is (0, -1)"
    assert "q2" not in asked.answers
    assert any("(default)" in ln for ln in lines)
    assert "by the operator" in interview.summary(asked)


def test_round_trip_through_json_for_a_host_that_holds_the_state():
    asked = interview.Interview(questions=interview.clean(
        json.loads(REPLY)["questions"]), answers={"q1": "x"}, asked=True)
    again = interview.Interview.from_json(json.loads(json.dumps(
        asked.to_json())))
    assert again.questions == asked.questions and again.answers == {"q1": "x"}
