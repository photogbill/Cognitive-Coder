# SPDX-License-Identifier: Apache-2.0
"""An assertion the model believes is wrong, named once — and what follows.

Bill, 2026-10-24, on ATK's Live tab: a repair of src/physics.py against a
test that was partly wrong went back and forth for 17,620 characters. The
repair prompt offered all or nothing ("if you are certain the test itself
is wrong, return this file unchanged"). Now it is per assertion: fix what
the right tests require, name each wrong one on a SUSPECT TEST line. These
tests hold that the lines are read out of the reply and never reach the
file; that the prompt says so; that when the only failures left are the
named ones the judge hears the reasons and the TEST is repaired next — or,
when it cannot be, the build stops naming each one and why; that a later
attempt is told what it already decided; and that a reply cut for going in
circles is followed by one told to decide once.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
    repetition,
    suspects,
)
from cognitive_coder.codemap import CodeMap  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.loop import (  # noqa: E402
    CALLER_RULE,
    TEST_RULE,
    Loop,
    LoopConfig,
)
from cognitive_coder.patcher import Patcher  # noqa: E402
from cognitive_coder.types import Completion, Task  # noqa: E402

# --------------------------------------------------------------------------
# reading the lines
# --------------------------------------------------------------------------


def test_suspect_lines_come_out_of_the_reply_however_they_are_dressed():
    reply = (
        "SUSPECT TEST: test_shortest_angle_diff — asserts 90.0 under its own "
        "comment saying -90\n"
        "- **SUSPECT TEST:** `TestPhysics.test_torpedo_hit` - the torpedo "
        "moves 50 units, past the 35-unit radius\n"
        "SUSPECT CALLER: main.py:12 — passes degrees where radians are "
        "expected\n"
        "SUSPECT TEST: test_shortest_angle_diff — said twice\n\n"
        "```python\ndef f():\n    return 1\n```")
    body, named = suspects.split(reply)
    assert body.startswith("```python") and "SUSPECT" not in body
    assert [(s.what, s.test) for s in named] == [
        ("test", "test_shortest_angle_diff"),
        ("test", "TestPhysics.test_torpedo_hit"),
        ("caller", "main.py:12")]
    assert named[1].key == "test_torpedo_hit"
    assert named[0].why.startswith("asserts 90.0")
    assert named[2].why.startswith("passes degrees")
    # nothing to take: the reply comes back as it was
    plain = "the suspect is in the code\n```python\nx = 1\n```"
    assert suspects.split(plain) == (plain, ())


def test_failing_tests_are_read_from_unittest_and_pytest():
    log = ("FAIL: test_north (tests.test_nav.TestNav.test_north)\n"
           "ERROR: test_distance (tests.test_nav.TestNav.test_distance)\n"
           "FAILED tests/test_nav.py::TestNav::test_east - assert 1 == 2\n"
           "FAIL: test_north (tests.test_nav.TestNav.test_north)\n")
    assert suspects.failing_tests([log]) == ["test_north", "test_distance",
                                             "test_east"]
    broken = "ERROR: tests.test_nav (unittest.loader._FailedTest)\n"
    assert suspects.failing_tests([broken]) == ["<import>"]
    named = suspects.split("SUSPECT TEST: test_north — y-up\nx")[1]
    assert suspects.covers(named, ["test_north"])
    assert not suspects.covers(named, ["test_north", "test_distance"])
    assert not suspects.covers(named, ["<import>"])
    assert not suspects.covers(named, [])


def test_the_rules_are_per_assertion_and_say_decide_once():
    assert "SUSPECT TEST: <test name>" in TEST_RULE
    assert "Decide once" in TEST_RULE
    assert "every test that is right" in TEST_RULE
    assert "unchanged only if every failing test is one you list" in TEST_RULE
    assert "certain the test itself is wrong" not in TEST_RULE
    assert "SUSPECT CALLER" in CALLER_RULE and "Decide once" in CALLER_RULE


# --------------------------------------------------------------------------
# in a build
# --------------------------------------------------------------------------

REQUEST = ("A tiny navigation module. Headings MUST use navigational "
           "bearings where 0.0 is North (0, -1) and 90.0 is East (1, 0): "
           "`bearing(x, y)` returns the bearing of the point (x, y) from "
           "the origin, and `distance(x, y)` its straight-line distance "
           "from the origin. Also write tests/test_nav.py for src/nav.py.")
PLAN = ("src/nav.py — bearing and distance\n"
        "tests/test_nav.py — tests for nav\n")
BEARING = ("import math\n\n\n"
           "def bearing(x: float, y: float) -> float:\n"
           '    """Navigational bearing of (x, y): north is (0, -1)."""\n'
           "    return math.degrees(math.atan2(x, -y)) % 360.0\n\n\n")
NAV_BUGGY = ("```python\n" + BEARING
             + "def distance(x: float, y: float) -> float:\n"
               "    return x + y\n```")
NAV_FIXED = ("```python\n" + BEARING
             + "def distance(x: float, y: float) -> float:\n"
               "    return math.hypot(x, y)\n```")
WHY = "the request says North is (0, -1); the test puts it at (0, 1)"
REPAIR = f"SUSPECT TEST: test_north — {WHY}\n\n{NAV_FIXED}"
#: test_north is wrong (y-up); test_distance is right
TEST_MIXED = ("```python\nimport unittest\n\n"
              "from src.nav import bearing, distance\n\n\n"
              "class TestNav(unittest.TestCase):\n"
              "    def test_north(self):\n"
              "        self.assertAlmostEqual(bearing(0.0, 1.0), 0.0)\n\n"
              "    def test_distance(self):\n"
              "        self.assertAlmostEqual(distance(3.0, 4.0), 5.0)\n\n\n"
              "if __name__ == '__main__':\n    unittest.main()\n```")
TEST_RIGHT = TEST_MIXED.replace("bearing(0.0, 1.0)", "bearing(0.0, -1.0)")
UNCLEAR = '{"verdict": "unclear", "quote": "", "why": "not sure"}'
VERDICT_TEST = json.dumps({
    "verdict": "test",
    "quote": "0.0 is North (0, -1) and 90.0 is East (1, 0)",
    "why": "the test puts north at (0, 1)"})


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Qwen3-Coder-30B-A3B-Instruct",
                                context_tokens=32768),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _session(tmp_path, replies, **cfg):
    host = _host(tmp_path, replies)
    cfg.setdefault("pin_interfaces", False)
    cfg.setdefault("review_after_build", False)
    cfg.setdefault("mutation_check", False)
    cfg.setdefault("attempts", 3)
    return host, Session(host, config=SessionConfig(**cfg))


def _texts(prompt) -> str:
    return "\n".join(m.content for m in prompt if isinstance(m.content, str))


def test_named_suspects_go_to_the_judge_and_the_test_is_repaired(tmp_path):
    """The module repair fixes distance (a real bug) and names test_north;
    only test_north still fails; the judge, hearing the reason, agrees with
    a quote; the TEST is repaired, and both end verified."""
    host, session = _session(tmp_path, [PLAN, NAV_BUGGY, TEST_MIXED,
                                        UNCLEAR,         # on the test's run
                                        REPAIR,          # module: fix + name
                                        VERDICT_TEST,    # judge, with reasons
                                        TEST_RIGHT])     # the test repaired
    session.run(REQUEST)
    final = {o.path: o for o in session._final_outcomes()}
    assert final["tests/test_nav.py"].ok, final["tests/test_nav.py"]
    assert final["src/nav.py"].ok and final["src/nav.py"].verified
    nav = (tmp_path / "src" / "nav.py").read_text(encoding="utf-8")
    assert "math.hypot" in nav and "SUSPECT" not in nav
    # the module repair was asked per assertion
    repair_prompt = _texts(host.llm.prompts[4])
    assert "SUSPECT TEST: <test name>" in repair_prompt
    assert "certain the test itself is wrong" not in repair_prompt
    # the judge heard the model's reason
    judge_prompt = _texts(host.llm.prompts[5])
    assert WHY in judge_prompt and "believes this assertion is wrong" in \
        judge_prompt
    # and the test's repair was seeded with it too
    assert WHY in _texts(host.llm.prompts[6])
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "names as wrong: test_north" in log
    assert "the judge agrees" in log
    assert len(host.llm.prompts) == 7


def test_without_the_judge_the_build_stops_naming_each_one(tmp_path):
    host, session = _session(tmp_path, [PLAN, NAV_BUGGY, TEST_MIXED, REPAIR],
                             adjudicate=False)
    session.run(REQUEST)
    fix = [o for o in session.outcomes if o.path == "src/nav.py"][-1]
    assert not fix.ok
    assert len(fix.attempts) == 1                  # no repair spun on it
    why = fix.stopped_because
    assert "every test still failing is one the model named as wrong" in why
    assert "test_north" in why and WHY in why
    assert "by hand" in why
    assert len(host.llm.prompts) == 4
    # the fix that WAS right stayed
    assert "math.hypot" in (tmp_path / "src" / "nav.py").read_text()
    # the journal keeps the suspects with the attempt
    gens = [r for r in session.loop.journal.events()
            if r.get("event") == "generate" and r.get("task") == "src/nav.py"]
    kept = [g["verify"].get("suspects") for g in gens
            if (g.get("verify") or {}).get("suspects")]
    assert kept and kept[-1][0]["test"] == "test_north"
    assert kept[-1][0]["why"] == WHY
    warnings = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("named as wrong" in m for m in warnings)


def test_a_suspect_with_another_failure_left_repairs_on_and_remembers(
        tmp_path):
    """test_north named, but distance still wrong: the repair goes on, and
    the next attempt is told what it already decided."""
    still_buggy = f"SUSPECT TEST: test_north — {WHY}\n\n{NAV_BUGGY}"
    still_buggy = still_buggy.replace("return x + y", "return x + y + 0")
    host, session = _session(tmp_path, [PLAN, NAV_BUGGY, TEST_MIXED,
                                        still_buggy, REPAIR],
                             adjudicate=False)
    session.run(REQUEST)
    fix = [o for o in session.outcomes if o.path == "src/nav.py"][-1]
    assert len(fix.attempts) == 2
    second = _texts(host.llm.prompts[4])
    assert "WHAT YOU ALREADY DECIDED" in second
    assert f"SUSPECT TEST: test_north — {WHY}" in second
    assert "Do not argue them again" in second
    assert "every test still failing is one the model named" in \
        fix.stopped_because


# --------------------------------------------------------------------------
# after a cut for going in circles
# --------------------------------------------------------------------------

def _loop(host, **config):
    loop = Loop(host, codemap=CodeMap(host.fs, host.storage),
                patcher=Patcher(host.fs, host.storage, host.approval,
                                host.events),
                config=LoopConfig(**config))
    loop.journal = Journal(host.fs, "t")
    return loop


def _circles_cut():
    return Completion(text="Wait, no. Actually …", finish_reason="cut",
                      model="scripted",
                      error=f"{repetition.CIRCLES_REASON} — going in circles "
                            f"— 29 reversals (no, wait, actually …)")


def _repeat_cut():
    return Completion(text="X = 1\n" * 50, finish_reason="cut",
                      model="scripted",
                      error="it was repeating itself: the same line, 50 "
                            "times over")


def test_after_a_cut_for_going_in_circles_the_next_attempt_is_told_to_decide(
        tmp_path):
    host = Host(llm=ScriptedLLM([_circles_cut(), "```python\nX = 1\n```"],
                                supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=2, seed=11)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="a constant",
                                 lang="python"), request="make a constant")
    assert outcome.ok
    second = _texts(host.llm.prompts[1])
    assert repetition.CIRCLES_NUDGE in second
    assert "YOUR LAST REPLY WAS CUT" in second
    gens = [r for r in loop.journal.events() if r.get("event") == "generate"]
    assert gens[1].get("temperature") > gens[0].get("temperature")


def test_a_verbatim_cut_gets_no_such_nudge(tmp_path):
    host = Host(llm=ScriptedLLM([_repeat_cut(), "```python\nX = 1\n```"],
                                supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request="make a constant")
    assert repetition.CIRCLES_NUDGE not in _texts(host.llm.prompts[1])
