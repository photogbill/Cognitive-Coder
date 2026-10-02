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


def test_a_module_with_no_known_dependencies_is_shown_what_exists(tmp_path):
    """Oct 1, 2026: `render.py` was written with no interface of `physics`,
    `track` or `math3d` in front of it — the stubs import nothing, so the
    plan had no edge to follow — and invented `player_state.x` against a
    class with no such attribute. A module now sees every module already
    built in the project, after its real dependencies; never tests, never
    stubs, never the entry point, and a test task is unaffected."""
    from cognitive_coder.loop import _is_our_stub  # noqa: F401 (documents)
    from cognitive_coder.types import Plan, Task
    host = _host(tmp_path, [PLAN], name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "math3d.py").write_text("def project():\n    return 1\n")
    (tmp_path / "src" / "physics.py").write_text(
        "class PhysicsState:\n    def __init__(self):\n        self.speed = 0\n")
    (tmp_path / "src" / "track.py").write_text(
        "# cc-stub: written by the skeleton; replaced when this file is built\n"
        "def main():\n    raise NotImplementedError\n")
    (tmp_path / "src" / "render.py").write_text("# cc-stub: x\n")
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    (tmp_path / "tests" / "test_math3d.py").write_text("import unittest\n")
    session.plan = Plan(request="r", tasks=(
        Task(id="t1", path="src/math3d.py", purpose="projection", status="done"),
        Task(id="t2", path="src/physics.py", purpose="the car", status="done"),
        Task(id="t3", path="src/track.py", purpose="the track"),     # pending: still a stub
        Task(id="t4", path="src/render.py", purpose="drawing", depends_on=("t1",)),
        Task(id="t5", path="src/main.py", purpose="the game loop", status="done"),
        Task(id="t6", path="tests/test_math3d.py", purpose="tests",
             persona="tester", status="done"),
        Task(id="t7", path="tests/test_physics.py", purpose="tests",
             persona="tester"),
    ))
    render = session.plan.tasks[3]
    # the planned dependency first, then what exists; not the stub, not
    # the entry point, not the tests, not itself
    assert session._planned_paths(render) == ("src/math3d.py", "src/physics.py")
    # a test task gets its module and nothing else
    assert session._planned_paths(session.plan.tasks[6]) == ("src/physics.py",)
    # the first module built sees nothing, because nothing exists yet
    first = Task(id="t0", path="src/zero.py", purpose="z")
    session.plan = Plan(request="r", tasks=(first,) + session.plan.tasks)
    session.plan = Plan(request="r", tasks=tuple(
        t.with_status("pending") if t.id != "t0" else t
        for t in session.plan.tasks))
    assert session._planned_paths(first) == ()


# --------------------------------------------------------------------------
# a test written later verifies the module written earlier
# --------------------------------------------------------------------------

PLAN_WITH_TEST = ("src/alpha.py — the first thing\n"
                  "tests/test_alpha.py — tests for alpha\n")
ALPHA_TEST = ('```python\nimport unittest\n\nfrom src.alpha import alpha\n\n\n'
              'class T(unittest.TestCase):\n    def test_alpha(self):\n'
              '        self.assertEqual(alpha(), 1)\n\n\n'
              'if __name__ == "__main__":\n    unittest.main()\n```')


def test_a_module_built_before_its_test_is_verified_once_the_test_passes(
        tmp_path):
    """Plans write modules first, so every module ends its own task "built,
    not verified". When its test is written and passes, the session must
    say verified — not repeat the verdict from an hour earlier."""
    host = _host(tmp_path, [PLAN_WITH_TEST, ALPHA, ALPHA_TEST],
                 name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    outcomes = session.run("one module and its test")
    by_path = {o.path: o for o in session._final_outcomes()}
    assert by_path["tests/test_alpha.py"].verified, by_path
    assert by_path["src/alpha.py"].verified, (
        "the module's test ran and passed; the module is verified")
    assert "src/alpha.py" not in session._unverified(session._final_outcomes())
    assert any("now verified" in m for _k, m, _d in host.events.of("status"))
    assert all(o.ok for o in outcomes)


def test_a_test_written_after_its_module_sees_the_module(tmp_path):
    """The tester's prompt carries the module's source and the rule that
    decides what to assert — not only the interface."""
    host = _host(tmp_path, [PLAN_WITH_TEST, ALPHA, ALPHA_TEST],
                 name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    session.run("one module and its test")
    def last_user(p):
        return [m for m in p if m.role == "user"][-1].content
    tester_prompts = [p for p in host.llm.prompts if
                      "Write the complete contents of `tests/test_alpha.py`"
                      in last_user(p)]
    assert tester_prompts, "no tester prompt found"
    text = "\n".join(m.content for m in tester_prompts[0]
                     if isinstance(m.content, str))
    assert "THE MODULE UNDER TEST — src/alpha.py" in text
    assert 'def alpha():' in text and '"""First."""' in text
    assert "never from a formula or a constant of your own" in text
    assert "How the tests are run:" in text          # the runner's rules
    # the module's own prompt did not get the block
    module_prompt = next(p for p in host.llm.prompts if
                         "Write the complete contents of `src/alpha.py`"
                         in last_user(p))
    mtext = "\n".join(m.content for m in module_prompt
                      if isinstance(m.content, str))
    assert "THE MODULE UNDER TEST" not in mtext
    assert "How the tests are run:" not in mtext


def test_the_plan_reaches_the_host_as_data(tmp_path):
    """A task board needs the tasks, not the sentence about them."""
    host = _host(tmp_path, [PLAN_WITH_TEST, ALPHA, ALPHA_TEST],
                 name="Devstral-Small-2-24B")
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("one module and its test")
    plans = host.events.of("plan")
    assert plans, host.events.kinds()
    _kind, message, data = plans[-1]
    assert message.startswith("plan: 2 file(s)")
    paths = [t["path"] for t in data["tasks"]]
    assert paths == ["src/alpha.py", "tests/test_alpha.py"]
    assert data["tasks"][0]["purpose"] == "the first thing"
    assert data["tasks"][1]["persona"] == "tester"
    assert data["revised"] is False
