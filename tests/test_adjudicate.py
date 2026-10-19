# SPDX-License-Identifier: Apache-2.0
"""The judge: a test that disagrees with its module is ruled by the request.

Ten of the sixteen failing tests of the 2026-10-16 Star Trek build were the
TEST's fault — written y-up against a request that says north is (0, -1)
— and the rule "the code must change" sent the module to be broken. Now
one question is asked, the answer must quote the request, and the quote
is checked (docs/HYBRID.md, adjudicate.py).
"""

from __future__ import annotations

import json

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
    adjudicate,
)
from cognitive_coder.types import Diagnostic

REQUEST = ("A tiny navigation module. Headings MUST use navigational "
           "bearings where 0.0 is North (0, -1) and 90.0 is East (1, 0): "
           "`bearing(x, y)` returns the bearing of the point (x, y) from "
           "the origin. Also write tests/test_nav.py for src/nav.py.")

PLAN = ("src/nav.py — the bearing function\n"
        "tests/test_nav.py — tests for nav\n")

NAV_RIGHT = ("```python\nimport math\n\n\n"
             "def bearing(x: float, y: float) -> float:\n"
             '    """Navigational bearing of (x, y) from the origin: '
             "north is (0, -1).\"\"\"\n"
             "    return math.degrees(math.atan2(x, -y)) % 360.0\n```")

#: y-up: east is right, north is UP — the habit, not the request
TEST_WRONG = ("```python\nimport unittest\n\nfrom src.nav import bearing\n\n\n"
              "class TestNav(unittest.TestCase):\n"
              "    def test_north(self):\n"
              "        self.assertAlmostEqual(bearing(0.0, 1.0), 0.0)\n```")

TEST_RIGHT = ("```python\nimport unittest\n\nfrom src.nav import bearing\n\n\n"
              "class TestNav(unittest.TestCase):\n"
              "    def test_north(self):\n"
              "        # the request: 0.0 is North (0, -1)\n"
              "        self.assertAlmostEqual(bearing(0.0, -1.0), 0.0)\n```")

VERDICT_TEST = json.dumps({
    "verdict": "test",
    "quote": "0.0 is North (0, -1) and 90.0 is East (1, 0)",
    "why": "the test puts north at (0, 1); the request says (0, -1)"})

VERDICT_TEST_BAD_QUOTE = json.dumps({
    "verdict": "test",
    "quote": "north is always up on the screen",
    "why": "that is the usual convention"})

VERDICT_CODE = json.dumps({
    "verdict": "code", "quote": "0.0 is North (0, -1)",
    "why": "the module returns the wrong value"})


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


# --------------------------------------------------------------------------
# the answer and its check
# --------------------------------------------------------------------------

def test_a_quote_the_request_does_not_contain_downgrades_the_verdict():
    v = adjudicate.check(adjudicate.parse(VERDICT_TEST_BAD_QUOTE), REQUEST)
    assert v.verdict == "unclear" and v.claimed == "test" and not v.checked
    assert "does not stand" in v.describe()
    good = adjudicate.check(adjudicate.parse(VERDICT_TEST), REQUEST)
    assert good.verdict == "test" and good.checked and good.test_is_wrong


def test_the_quote_check_forgives_whitespace_quotes_and_backticks():
    reply = json.dumps({"verdict": "test",
                        "quote": "`bearing(x, y)`  returns the bearing of "
                                 "the   point (x, y) from the origin",
                        "why": "so"})
    assert adjudicate.check(adjudicate.parse(reply), REQUEST).checked


def test_a_too_short_quote_proves_nothing():
    reply = json.dumps({"verdict": "test", "quote": "MUST", "why": "so"})
    assert adjudicate.check(adjudicate.parse(reply), REQUEST).verdict == \
        "unclear"


def test_prose_around_the_json_and_near_json_are_read():
    v = adjudicate.parse("Sure. Here is my verdict:\n```json\n"
                         "{'verdict': 'code', 'quote': 'x', 'why': 'y',}\n"
                         "```")
    assert v.verdict == "code" and v.quote == "x"
    assert adjudicate.parse("test").verdict == "test"
    assert adjudicate.parse("I cannot say").verdict == "unclear"


def test_code_and_unclear_verdicts_need_no_quote():
    v = adjudicate.check(adjudicate.parse('{"verdict": "code", "quote": "",'
                                          ' "why": "obvious"}'), REQUEST)
    assert v.verdict == "code" and not v.test_is_wrong


def test_the_diagnostic_for_a_wrong_test_carries_the_request():
    v = adjudicate.check(adjudicate.parse(VERDICT_TEST), REQUEST)
    d = adjudicate.diagnostic_for_test(v, "tests/test_nav.py", [Diagnostic(
        file="tests/test_nav.py", line=8, severity="exception",
        message="AssertionError: 180.0 != 0.0", code="in test_north")])
    assert d.file == "tests/test_nav.py" and d.line == 8
    assert "0.0 is North (0, -1)" in d.message
    assert "Do not change the module" in d.message


# --------------------------------------------------------------------------
# in the loop: the test is repaired, not the module
# --------------------------------------------------------------------------

def test_a_test_the_judge_rules_wrong_is_repaired_and_the_module_stands(
        tmp_path):
    """nav.py is right; test_nav.py is y-up. The judge quotes the request,
    the TEST is repaired, and nav.py is never sent to be broken."""
    host, session = _session(tmp_path, [PLAN, NAV_RIGHT, TEST_WRONG,
                                        VERDICT_TEST, TEST_RIGHT])
    session.run(REQUEST)
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/nav.py"].ok and final["tests/test_nav.py"].ok
    assert final["src/nav.py"].verified
    nav = (tmp_path / "src" / "nav.py").read_text()
    assert "atan2(x, -y)" in nav, "the module was not touched"
    test = (tmp_path / "tests" / "test_nav.py").read_text()
    assert "bearing(0.0, -1.0)" in test
    # the judge's prompt carried the request's slice, both files and the
    # failure; the test's repair prompt carried the request's words
    judge = "\n".join(m.content for m in host.llm.prompts[3])
    assert "[THE REQUEST, as far as it concerns src/nav.py]" in judge
    assert "[THE TEST — tests/test_nav.py]" in judge
    assert "AssertionError" in judge
    assert "JSON" in judge
    repair = "\n".join(m.content for m in host.llm.prompts[4])
    assert "The request says: \"0.0 is North (0, -1)" in repair
    assert "Do not change the module" in repair
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "judge tests/test_nav.py vs src/nav.py: the TEST is wrong" in log
    assert "the request says: \"0.0 is North" in log
    journal = "\n".join(p.read_text() for p in
                        (tmp_path / ".cc_journal").rglob("*.jsonl"))
    assert '"event": "adjudicate"' in journal and '"verdict": "test"' in journal
    assert not any("will be repaired against tests/test_nav.py" in m
                   for _k, m, _d in host.events.events)


def test_a_test_verdict_without_a_real_quote_falls_back_to_the_old_rule(
        tmp_path):
    """The judge says "test" but quotes words the request does not have:
    the verdict does not stand, and the module is repaired against the
    test as before — said in the log."""
    nav_bent = ("```python\nimport math\n\n\n"
                "def bearing(x: float, y: float) -> float:\n"
                '    """bent"""\n'
                "    return math.degrees(math.atan2(x, y)) % 360.0\n```")
    host, session = _session(tmp_path, [PLAN, NAV_RIGHT, TEST_WRONG,
                                        VERDICT_TEST_BAD_QUOTE, nav_bent])
    session.run(REQUEST)
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "the model said 'test', but its quote is not in the request" in log
    assert any("will be repaired against tests/test_nav.py" in m
               for _k, m, _d in host.events.events), \
        [m for _k, m, _d in host.events.events]


def test_a_code_verdict_repairs_the_module_as_before(tmp_path):
    nav_wrong = ("```python\nimport math\n\n\n"
                 "def bearing(x: float, y: float) -> float:\n"
                 '    """wrong"""\n'
                 "    return math.degrees(math.atan2(y, x)) % 360.0\n```")
    host, session = _session(tmp_path, [PLAN, nav_wrong, TEST_RIGHT,
                                        VERDICT_CODE, NAV_RIGHT])
    session.run(REQUEST)
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/nav.py"].ok and final["tests/test_nav.py"].ok
    assert "atan2(x, -y)" in (tmp_path / "src" / "nav.py").read_text()
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "judge tests/test_nav.py vs src/nav.py: the MODULE is wrong" in log


def test_an_unchanged_module_repair_asks_the_judge_and_repairs_the_test(
        tmp_path):
    """The judge is off for the test's own failure (so the module is blamed
    as before), the module repair returns the file unchanged — "I am
    certain the test is wrong" — and THEN the judge is asked, agrees with
    a quote, and the test is repaired."""
    host, session = _session(tmp_path, [PLAN, NAV_RIGHT, TEST_WRONG,
                                        # the judge on the test's failure
                                        '{"verdict": "unclear", "quote": "",'
                                        ' "why": "not sure"}',
                                        # the module repair: unchanged
                                        NAV_RIGHT,
                                        # the judge again, on the unchanged
                                        # return — a new question, since the
                                        # first was unclear with no quote
                                        VERDICT_TEST,
                                        TEST_RIGHT])
    session.run(REQUEST)
    final = {o.path: o for o in session._final_outcomes()}
    assert final["tests/test_nav.py"].ok, final["tests/test_nav.py"].stopped_because
    assert final["src/nav.py"].ok and final["src/nav.py"].verified
    assert "bearing(0.0, -1.0)" in (tmp_path / "tests" / "test_nav.py").read_text()
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "the judge agrees the test is wrong" in log
    assert "REPAIR tests/test_nav.py — against the request, not the module" in log


def test_the_judge_can_be_switched_off(tmp_path):
    host, session = _session(tmp_path, [PLAN, NAV_RIGHT, TEST_WRONG,
                                        NAV_RIGHT], adjudicate=False)
    session.run(REQUEST)
    assert not any("judge" in m for _k, m, _d in host.events.events)
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "judge tests/test_nav.py" not in log
