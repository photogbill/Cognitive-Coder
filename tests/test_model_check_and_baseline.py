# SPDX-License-Identifier: Apache-2.0
"""Before a build: is this a coding model, and what is already in the folder?

Aug 8, 2026, 7:23 PM: `Slimaki-Tavern-24B` — a roleplay merge loaded for
something else — built the racing spec for nineteen minutes and produced
nothing. Its name was in the log's header. Nobody reads headers; the engine
has to.

Same evening, and again on Oct 1: a `tests/test_physics.py` left by the
earlier session failed to import, and its failure was charged to every new
file in turn. The session now runs what is already there first, and says
what it found.
"""

from __future__ import annotations

import pytest

from cognitive_coder import (
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
    models,
)
from cognitive_coder.errors import NotACodingModelError

PLAN = "src/alpha.py — the first thing\nsrc/beta.py — the second thing\n"
ALPHA = '```python\ndef alpha():\n    """First."""\n    return 1\n```'
BETA = '```python\ndef beta():\n    """Second."""\n    return 2\n```'


def _host(tmp_path, replies, *, name="scripted", context_tokens=16384):
    return Host(llm=ScriptedLLM(replies, supports_tools=False, name=name,
                                context_tokens=context_tokens),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


# --------------------------------------------------------------------------
# the verdict, by name
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf",
    "Codestral-22B-v0.1-Q5_K_M.gguf", "Qwen2.5-Coder-14B-Instruct-Q6_K.gguf",
    "qwen3-coder-30b-a3b", "deepseek-coder-v2-lite", "granite-code-8b",
    "starcoder2-15b", "CodeLlama-13b-Instruct", "phi-4-Q4_K_M.gguf",
])
def test_known_coding_models_are_recognised(name):
    assert models.classify(name) == "coding", name


@pytest.mark.parametrize("name", [
    "Slimaki-Tavern-24B-v1.3.Q4_K_M.gguf", "MythoMax-L2-13B",
    "Llama-3-8B-Uncensored-RP", "Cydonia-22B-v1", "Magnum-v4-27b",
    "some-coder-tavern-merge",            # a coding marker does not redeem it
])
def test_roleplay_and_fiction_models_are_refused_by_name(name):
    assert models.classify(name) == "not_coding", name
    assert models.judge(name).refuse
    assert "not a coding model" in models.judge(name).reason


@pytest.mark.parametrize("name", [
    "Mistral-Small-3.2-24B-Instruct", "Meta-Llama-3.1-8B-Instruct",
    "gemma-3-12b-it", "my-finetune-v7", "",
])
def test_general_and_unknown_models_are_allowed_under_a_warning(name):
    verdict = models.judge(name)
    assert verdict.verdict == "unknown"
    assert not verdict.refuse and verdict.warn


def test_a_small_context_window_is_named_as_a_problem():
    v = models.judge("Devstral-Small-2-24B", context_tokens=8192)
    assert not v.context_ok and "8,192" in v.context_note
    assert models.judge("Devstral-Small-2-24B", context_tokens=32768).context_ok
    assert models.judge("Devstral-Small-2-24B", context_tokens=0).context_ok, (
        "an unreported context size is not a reported small one")


def test_the_short_name_is_the_part_a_person_recognises():
    assert models._short(
        "mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"
    ) == "Devstral-Small-2-24B-Instruct-2512"
    assert models._short("Slimaki-Tavern-24B-v1.3.Q4_K_M.gguf") == \
        "Slimaki-Tavern-24B-v1.3"


# --------------------------------------------------------------------------
# the session acts on it
# --------------------------------------------------------------------------

def test_a_session_refuses_a_roleplay_model_before_planning(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA],
                 name="Slimaki-Tavern-24B-v1.3.Q4_K_M.gguf")
    session = Session(host, config=SessionConfig(attempts=1))
    with pytest.raises(NotACodingModelError) as err:
        session.start("two small modules")
    assert "Slimaki-Tavern" in str(err.value)
    assert not (tmp_path / "src").exists(), "nothing was planned or written"
    kinds = [m for _k, m, _d in host.events.of("warning")]
    assert any("not a coding model" in m for m in kinds)


def test_allow_any_model_turns_the_refusal_into_a_warning(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA],
                 name="Slimaki-Tavern-24B-v1.3.Q4_K_M.gguf")
    session = Session(host, config=SessionConfig(attempts=1,
                                                 allow_any_model=True))
    outcomes = session.run("two small modules")
    assert [o.ok for o in outcomes] == [True, True]
    assert any("Continuing because allow_any_model" in m
               for _k, m, _d in host.events.of("warning"))


def test_an_unrecognised_model_builds_under_a_warning(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA], name="my-finetune-v7")
    session = Session(host, config=SessionConfig(attempts=1))
    outcomes = session.run("two small modules")
    assert [o.ok for o in outcomes] == [True, True]
    assert any("not a model this engine recognises" in m
               for _k, m, _d in host.events.of("warning"))


def test_a_coding_model_with_a_small_context_is_warned_about(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA], name="Devstral-Small-2-24B",
                 context_tokens=4096)
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    assert any("tokens of context" in m
               for _k, m, _d in host.events.of("warning"))


# --------------------------------------------------------------------------
# what is already in the folder
# --------------------------------------------------------------------------

STALE_TEST = ('import unittest\n\nfrom src.gamma import Gamma\n\n\n'
              'class T(unittest.TestCase):\n    def test_g(self):\n'
              '        self.assertTrue(Gamma())\n')


def test_tests_already_failing_in_the_folder_are_baselined_and_named(
        tmp_path):
    """Oct 1: the stale test failed every module of the new build."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_gamma.py").write_text(STALE_TEST)
    host = _host(tmp_path, [PLAN, ALPHA, BETA], name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    outcomes = session.run("two small modules")
    assert [o.ok for o in outcomes] == [True, True], [
        o.summary() for o in outcomes]
    assert "tests.test_gamma" in session.loop.config.known_failing
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("already failing before this build" in m
               and "test_gamma" in m for m in warnings), warnings
    assert "BASELINE" in (tmp_path / "BUILD_LOG.txt").read_text()


def test_a_folder_with_an_earlier_build_is_said_so(tmp_path):
    (tmp_path / "BUILD_LOG.txt").write_text(
        "COGNITIVE CODER — BUILD LOG\n\n====\nSESSION cc-1\n====\n"
        "started 2026-08-08\n\n====\nSESSION cc-2\n====\n")
    host = _host(tmp_path, [PLAN, ALPHA, BETA], name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("2 earlier build sessions" in m for m in warnings), warnings


def test_an_empty_folder_raises_no_such_warnings(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA], name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert not any("earlier build" in m or "already failing" in m
                   for m in warnings), warnings
    assert session.loop.config.known_failing == ()


def test_planned_dependencies_reach_the_loop_as_paths(tmp_path):
    """`depends_on` holds task ids; the codemap wants paths. The session
    translates — and a test task always gets the module it covers."""
    from cognitive_coder.types import Plan, Task
    host = _host(tmp_path, [PLAN, ALPHA, BETA], name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    session.plan = Plan(request="r", tasks=(
        Task(id="t1", path="src/alpha.py", purpose="a",
             test_path="tests/test_alpha.py"),
        Task(id="t2", path="src/beta.py", purpose="b", depends_on=("t1",)),
        Task(id="t3", path="tests/test_alpha.py", purpose="tests",
             persona="tester"),
    ))
    assert session._planned_paths(session.plan.tasks[1]) == ("src/alpha.py",)
    assert session._planned_paths(session.plan.tasks[2]) == ("src/alpha.py",)
    assert session._planned_paths(session.plan.tasks[0]) == ()
