# SPDX-License-Identifier: Apache-2.0
"""FIM mode (fim.py, docs/FIM_MODE.md, 2026-10-27).

A hole for an ordinary code model: the body the error is in, filled in
the middle, nothing else touched. The pure rules first, then the loop
with a scripted FimLLM, then the probe that decides whether a model gets
the mode at all, then the providers' prompt building.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import _loop  # noqa: E402

from cognitive_coder import canvas as cv  # noqa: E402
from cognitive_coder import fim, profiles  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.providers import local_llamacpp  # noqa: E402
from cognitive_coder.session import Session, SessionConfig  # noqa: E402
from cognitive_coder.types import (  # noqa: E402
    Diagnostic,
    FimRequest,
    FimResult,
    Task,
)


def _same(a: str, b: str) -> bool:
    return a.replace("\\", "/").rsplit("/", 1)[-1] == \
        b.replace("\\", "/").rsplit("/", 1)[-1]


PHYSICS = '''\
"""Ship physics."""
import math


def normalize(deg):
    return deg % 360.0


def bearing_between(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    heading = math.degrees(math.atan2(dy, dx))
    return normalize(heading)


def main():
    print(bearing_between(0, 0, 1, 1))
'''
ERR = (Diagnostic(file="src/physics.py", line=12, severity="error",
                  message="AssertionError: 45.0 != 90.0",
                  values="values at the failure (tests/test_p.py:9 in "
                         "test_north):\n    expected = 90.0"),)
GOOD_FILL = ("    dx, dy = x2 - x1, y2 - y1\n"
             "    heading = math.degrees(math.atan2(dx, -dy))\n"
             "    return normalize(heading)\n")


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------

def test_the_hole_is_the_body_of_the_function_the_error_is_in():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    assert hole is not None and hole.kind == "body"
    assert hole.name == "bearing_between" and hole.why == "located"
    prefix, suffix = fim.split(PHYSICS, hole)
    assert prefix.endswith("def bearing_between(x1, y1, x2, y2):\n")
    assert suffix.startswith("\n\ndef main():")
    assert prefix + hole.original + suffix == PHYSICS


def test_a_long_body_is_a_hole_for_fim_where_the_canvas_gave_a_window():
    body = "".join(f"    x{i} = {i}\n" for i in range(60))
    code = f"def big(a):\n{body}    return x0 / a\n"
    err = (Diagnostic(file="m.py", line=30, severity="error",
                      message="ZeroDivisionError"),)
    assert cv.repair_hole(code, "python", "m.py", err, _same).kind == "window"
    hole = fim.hole_for_repair(code, "python", "m.py", err, _same)
    assert hole.kind == "body" and hole.size == 61


def test_the_hint_names_the_error_and_the_values_at_the_hole_indent():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    text = fim.hint(hole, ERR, "src/physics.py", _same)
    lines = text.splitlines()
    assert lines[0] == ("    # FIX: the body of `bearing_between` failed: "
                        "AssertionError: 45.0 != 90.0")
    assert lines[1] == "    #   expected = 90.0"
    assert "signature above" in lines[-1]
    # in the prompt only: the file's own split has no hint
    assert "FIX:" not in fim.split(PHYSICS, hole)[0]


def test_requests_carry_the_profile_and_warm_up_with_a_new_seed():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    prof = profiles.builtin_for("qwen3-coder")
    reqs = fim.requests_for(PHYSICS, hole, hint_text="    # FIX\n",
                            path="src/physics.py", n=2, temperature=0.15,
                            seed=3, profile=prof,
                            extra=[("INTERFACES.py", "# iface")])
    assert len(reqs) == 2
    # the first fill is at the attempt's OWN sampling — what its record
    # says — the second warmer, with a seed of its own
    assert reqs[0].temperature == 0.15 and reqs[0].seed == 3
    assert reqs[1].temperature == 0.4 and reqs[1].seed != reqs[0].seed
    assert reqs[0].order == "psm" and reqs[0].tokens == prof.fim_tokens
    assert reqs[0].prefix.endswith("    # FIX\n")
    assert reqs[0].extra == (("INTERFACES.py", "# iface"),)
    assert reqs[0].path == "src/physics.py"


def test_judge_accepts_a_fill_that_parses_and_differs():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    cand = fim.judge(PHYSICS, hole, FimResult(text=GOOD_FILL), "python")
    assert cand.usable, cand.problem
    assert "atan2(dx, -dy)" in cand.code
    assert cand.code.count("def bearing_between") == 1
    assert "def main():" in cand.code and "def normalize" in cand.code


def test_judge_rejects_the_old_lines_a_parse_failure_and_a_duplicate():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    same = fim.judge(PHYSICS, hole, FimResult(text=hole.original), "python")
    assert same.problem == "the same lines came back"
    broken = fim.judge(PHYSICS, hole, FimResult(text="    return (\n"),
                       "python")
    assert "does not parse" in broken.problem
    # a fill that starts the NEXT function over is cut where it leaves the
    # body: the duplicate never reaches the file
    dup = fim.judge(PHYSICS, hole, FimResult(
        text="    return 1.0\n\n\ndef normalize(deg):\n    return deg\n"),
        "python")
    assert dup.usable and dup.code.count("def normalize") == 1
    assert "return 1.0" in dup.code and "return deg\n" not in dup.code
    err = fim.judge(PHYSICS, hole, FimResult(finish_reason="error",
                                             error="no tokens"), "python")
    assert err.problem == "no tokens"


def test_judge_cuts_a_fill_that_ran_on_into_the_next_function():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    ran_on = GOOD_FILL + "\n\ndef main():\n    print('again')\n"
    cand = fim.judge(PHYSICS, hole, FimResult(text=ran_on,
                                              finish_reason="length"),
                     "python")
    assert cand.usable, cand.problem
    assert cand.salvaged
    assert cand.code.count("def main") == 1 and "again" not in cand.code


def test_a_model_that_echoes_its_fim_tokens_is_cleaned():
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    cand = fim.judge(PHYSICS, hole, FimResult(
        text=GOOD_FILL + "<|fim_pad|><|endoftext|>"), "python")
    assert cand.usable, cand.problem


def test_the_probe_accepts_the_sum_and_nothing_else():
    ok, why = fim.probe_verdict(FimResult(text="    return a + b\n"))
    assert ok and "correctly" in why
    ok, why = fim.probe_verdict(FimResult(text="    return b + a"))
    assert ok
    ok, why = fim.probe_verdict(FimResult(text="    return a - b\n"))
    assert not ok and "something other than its sum" in why
    ok, why = fim.probe_verdict(FimResult(text="    return (\n"))
    assert not ok and "does not parse" in why
    ok, why = fim.probe_verdict(FimResult(text=""))
    assert not ok and "nothing" in why
    ok, why = fim.probe_verdict(FimResult(finish_reason="error",
                                          error="no FIM tokens"))
    assert not ok and why == "no FIM tokens"
    # a reply that ran on INTO THE CODE AFTER THE HOLE is a prefix
    # completion, not a fill: the model never read the suffix (the
    # 2026-10-27 review) — and so is one that ran to the token limit
    ok, why = fim.probe_verdict(FimResult(
        text="    return a + b\n\n\ndef sub(a, b):\n    return a - b\n",
        finish_reason="length"))
    assert not ok and "did not read the suffix" in why
    ok, why = fim.probe_verdict(FimResult(text="    return a + b\n",
                                          finish_reason="length"))
    assert not ok and "token limit" in why
    # a stray blank line after the sum is still the sum
    ok, why = fim.probe_verdict(FimResult(text="    return a + b\n\n"))
    assert ok


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

class FimScripted(ScriptedLLM):
    """A ScriptedLLM that also fills holes, from a second script."""

    def __init__(self, replies, fills, **kw):
        super().__init__(replies, **kw)
        self._fills = list(fills)
        self.fim_requests: list[FimRequest] = []

    def fim_fill(self, request, *, cancel=None):
        self.fim_requests.append(request)
        if not self._fills:
            raise AssertionError("FimScripted ran out of fills")
        reply = self._fills.pop(0)
        if isinstance(reply, FimResult):
            return reply
        return FimResult(text=str(reply), model=self._name,
                         tokens_in=len(request.prefix) // 4,
                         tokens_out=len(str(reply)) // 4)


def _host(tmp_path, replies, fills):
    return Host(llm=FimScripted(replies, fills, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _task():
    return Task(id="t1", path="src/physics.py", purpose="physics",
                lang="python")


def _write(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "physics.py").write_text(PHYSICS)


def test_a_repair_fills_the_hole_and_touches_nothing_else(tmp_path):
    _write(tmp_path)
    host = _host(tmp_path, [], [GOOD_FILL])
    loop = _loop(host, attempts=2)
    loop.fim_ok = True
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="ship physics", seed=ERR)
    assert out.ok, out.stopped_because
    assert not host.llm.prompts, "no chat call was needed"
    req = host.llm.fim_requests[0]
    assert req.prefix.endswith("stay as they are.\n")
    assert "FIX: the body of `bearing_between`" in req.prefix
    assert req.suffix.startswith("\n\ndef main():")
    after = (tmp_path / "src" / "physics.py").read_text()
    assert "atan2(dx, -dy)" in after and "FIX:" not in after
    assert after.startswith('"""Ship physics."""')
    events = [e for e in loop.journal.events() if e.get("event") == "fim"]
    assert events and events[0]["data"]["kept"] == 0
    assert events[0]["data"]["name"] == "bearing_between"
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    assert gens[-1]["data"].get("via") == "fim"


def test_an_unusable_fill_tries_the_second_then_another_path(tmp_path):
    _write(tmp_path)
    whole = "```python\n" + PHYSICS.replace("atan2(dy, dx)",
                                            "atan2(dx, -dy)") + "```"
    host = _host(tmp_path, [whole], ["    return (\n", "    return (\n"])
    loop = _loop(host, attempts=2)
    loop.fim_ok = True
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="ship physics", seed=ERR)
    assert out.ok, out.stopped_because
    assert len(host.llm.fim_requests) == 2
    assert host.llm.fim_requests[1].temperature > \
        host.llm.fim_requests[0].temperature
    assert host.llm.prompts, "the whole-file path took the attempt"
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("no fill came back usable" in m for m in warnings)


def test_the_same_hole_with_the_same_error_is_not_filled_a_third_time(
        tmp_path):
    """A fill that parses but does not fix it: the next attempt is not
    the same hole again."""
    _write(tmp_path)
    still_wrong = ("    dx, dy = x2 - x1, y2 - y1\n"
                   "    heading = math.degrees(math.atan2(dy, dx)) + 0.0\n"
                   "    return normalize(heading)\n")
    whole = "```python\n" + PHYSICS.replace("atan2(dy, dx)",
                                            "atan2(dx, -dy)") + "```"
    host = _host(tmp_path, [whole], [still_wrong])
    loop = _loop(host, attempts=3)
    loop.fim_ok = True
    loop.journal = Journal(host.fs, "t")
    # the test the module must pass, so the second attempt carries the
    # same located error
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_p.py").write_text(
        "import unittest\nfrom src.physics import bearing_between\n\n\n"
        "class T(unittest.TestCase):\n    def test_north(self):\n"
        "        self.assertEqual(bearing_between(0, 0, 1, 1), 135.0)\n")
    task = Task(id="t1", path="src/physics.py", purpose="physics",
                lang="python", test_path="tests/test_p.py")
    # the seed is the test's own failure, as the session hands it on
    seed = (Diagnostic(file="src/physics.py", line=12, severity="error",
                       message="AssertionError: 45.0 != 135.0"),)
    out = loop.run_task(task, request="ship physics", seed=seed)
    assert out.ok, out.stopped_because
    assert len(host.llm.fim_requests) == 1
    decisions = [e for e in loop.journal.events() if e.get("event") == "fim"]
    assert len(decisions) == 1


def test_fim_is_not_used_without_the_probe_or_when_off(tmp_path):
    _write(tmp_path)
    whole = "```python\n" + PHYSICS.replace("atan2(dy, dx)",
                                            "atan2(dx, -dy)") + "```"
    host = _host(tmp_path, [whole], [GOOD_FILL])
    loop = _loop(host, attempts=1)
    assert not loop.fim_ok
    loop.run_task(_task(), request="ship physics", seed=ERR)
    assert not host.llm.fim_requests
    host = _host(tmp_path, [whole], [GOOD_FILL])
    loop = _loop(host, attempts=1, fim="off")
    loop.fim_ok = True
    loop.run_task(_task(), request="ship physics", seed=ERR)
    assert not host.llm.fim_requests


# --------------------------------------------------------------------------
# the session's probe
# --------------------------------------------------------------------------

PLAN = "greet.py — say hello\n"
FILE = "```python\ndef greet(name):\n    return f'hi {name}'\n```"


def _session(fills, mode="on"):
    llm = FimScripted([PLAN, FILE], fills, supports_tools=False,
                      name="Qwen3-Coder-30B")
    host = Host(llm=llm, fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(
        attempts=1, review_after_build=False, mutation_check=False,
        adjudicate=False, fim=mode))
    return session, llm, host


def test_the_probe_turns_fim_on_only_for_a_correct_fill():
    session, llm, host = _session(["    return a + b\n"])
    session.run("a greet function")
    assert session.loop.fim_ok and "correctly" in session.loop.fim_why
    probe = llm.fim_requests[0]
    assert probe.prefix == fim.PROBE_PREFIX and probe.tokens
    events = [e for e in session.journal.events() if e["event"] == "fim"]
    assert events[0]["data"]["kind"] == "probe" and events[0]["data"]["ok"]
    assert any("FIM mode: this model fills" in m
               for _k, m, _d in host.events.of("status"))


def test_the_probe_turns_fim_off_for_a_wrong_fill_or_an_error():
    session, _llm, host = _session(["    return a - b\n"])
    session.run("a greet function")
    assert not session.loop.fim_ok
    assert "something other than its sum" in session.loop.fim_why
    assert any("FIM mode is off" in m for _k, m, _d in host.events.of("status"))
    session, _llm, _host = _session([FimResult(finish_reason="error",
                                               error="no FIM tokens")])
    session.run("a greet function")
    assert not session.loop.fim_ok and session.loop.fim_why == "no FIM tokens"


def test_auto_does_not_probe_a_scripted_model_and_off_never_asks():
    session, llm, _host = _session(["    return a + b\n"], mode="auto")
    session.run("a greet function")
    assert not llm.fim_requests and not session.loop.fim_ok
    assert "scripted" in session.loop.fim_why
    session, llm, _host = _session(["    return a + b\n"], mode="off")
    session.run("a greet function")
    assert not llm.fim_requests and "switched off" in session.loop.fim_why


# --------------------------------------------------------------------------
# the in-process provider's prompt
# --------------------------------------------------------------------------

class FakeLlama:
    """Enough of llama-cpp-python's `Llama` to build a FIM prompt from the
    profile's token texts: each special marker is one id, plain text is
    one id per character."""

    SPECIALS = {"<|fim_prefix|>": 1, "<|fim_suffix|>": 2, "<|fim_middle|>": 3,
                "[PREFIX]": 4, "[SUFFIX]": 5}

    def __init__(self):
        self.calls = []

    def tokenize(self, text: bytes, add_bos=False, special=False):
        s = text.decode("utf-8")
        if special and s in self.SPECIALS:
            return [self.SPECIALS[s]]
        return [100 + ord(c) for c in s]

    def create_completion(self, **kw):
        self.calls.append(kw)
        return {"choices": [{"text": "    return a + b\n",
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(kw["prompt"]),
                          "completion_tokens": 6}}


def test_the_prompt_is_built_from_the_profile_tokens_in_psm_order():
    llama = FakeLlama()
    req = FimRequest(prefix="ab", suffix="cd", order="psm",
                     tokens=profiles.builtin_for("qwen3-coder").fim_tokens)
    ids, problem = local_llamacpp.fim_prompt_ids(llama, req)
    assert problem == ""
    assert ids == [1, 100 + ord("a"), 100 + ord("b"), 2, 100 + ord("c"),
                   100 + ord("d"), 3]


def test_spm_puts_the_suffix_first_with_no_middle_token():
    llama = FakeLlama()
    req = FimRequest(prefix="ab", suffix="cd", order="spm",
                     tokens=("[PREFIX]", "[SUFFIX]", ""))
    ids, problem = local_llamacpp.fim_prompt_ids(llama, req)
    assert problem == ""
    assert ids == [5, 100 + ord("c"), 100 + ord("d"), 4, 100 + ord("a"),
                   100 + ord("b")]


def test_a_model_without_the_tokens_is_a_sentence():
    llama = FakeLlama()
    req = FimRequest(prefix="ab", suffix="cd", tokens=("<PRE>", "<SUF>",
                                                        "<MID>"))
    ids, problem = local_llamacpp.fim_prompt_ids(llama, req)
    assert not ids and "not a single token" in problem
    ids, problem = local_llamacpp.fim_prompt_ids(
        llama, FimRequest(prefix="a", suffix="b"))
    assert not ids and "names none" in problem


def test_the_provider_fills_through_create_completion():
    provider = local_llamacpp.LocalLlamaCpp.__new__(local_llamacpp.LocalLlamaCpp)
    provider.llama = FakeLlama()
    provider._name = "m.gguf"
    req = FimRequest(prefix="def add(a, b):\n", suffix="\n", seed=4,
                     tokens=profiles.builtin_for("qwen3-coder").fim_tokens)
    result = provider.fim_fill(req)
    assert result.ok and result.text == "    return a + b\n"
    assert provider.llama.calls[0]["seed"] == 4
    assert result.tokens_out == 6
