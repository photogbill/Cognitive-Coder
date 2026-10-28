# SPDX-License-Identifier: Apache-2.0
"""Sample again before repairing (loop.Loop._resample, 2026-10-27).

A failed first attempt is asked again at a new seed before any repair
prompt is built; each sample is verified like the first; the best stays
on disk; the definitions the samples disagree on are the repair's doubt
map and are named in its prompt.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import _host, _loop  # noqa: E402

from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.loop import _differing_definitions  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.session import Session, SessionConfig  # noqa: E402
from cognitive_coder.types import Task  # noqa: E402

TEST = '''\
import unittest

from calc import double, triple


class T(unittest.TestCase):
    def test_double(self):
        self.assertEqual(double(2), 4)

    def test_triple(self):
        self.assertEqual(triple(2), 6)
'''


def _file(double: str, triple: str) -> str:
    return (f"```python\ndef double(x):\n    return {double}\n\n\n"
            f"def triple(x):\n    return {triple}\n```")


WRONG_BOTH = _file("x + x + 1", "x * 2")       # two failing tests
WRONG_ONE = _file("x * 2", "x * 2")            # one failing test
RIGHT = _file("x * 2", "x * 3")                # passes


def _task():
    return Task(id="t1", path="calc.py", purpose="double and triple",
                lang="python", test_path="tests/test_calc.py")


def _project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calc.py").write_text(TEST)


def test_the_best_sample_replaces_a_failed_first_attempt(tmp_path):
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_BOTH, RIGHT])
    loop = _loop(host, attempts=2, resample=1)
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok and out.verified, out.stopped_because
    assert len(out.attempts) == 1, "attempt 1 passed once resampled"
    assert "x * 3" in (tmp_path / "calc.py").read_text()
    assert len(host.llm.prompts) == 2
    # the sample was the same prompt, warmer, seeded
    assert host.llm.prompts[1] == host.llm.prompts[0]
    assert host.llm.calls[1]["temperature"] > host.llm.calls[0]["temperature"]
    assert host.llm.calls[1]["seed"] is not None
    ev = [e for e in loop.journal.events() if e.get("event") == "resample"]
    assert ev and ev[0]["data"]["kept"] == 1
    assert set(ev[0]["data"]["disagreed"]) == {"double", "triple"}
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    # two replies were paid for, two generate events: the kept sample as
    # attempt 1's own, the first reply as the one it replaced
    assert len(gens) == 2
    assert any("replaced by sample 1" in str(g.get("verify", {}))
               for g in gens)
    kept = next(g for g in gens if g.get("verify", {}).get("ok"))
    # attempt 1's record carries the sampling the KEPT reply was made at
    assert kept["temperature"] == host.llm.calls[1]["temperature"]
    assert kept["seed"] == host.llm.calls[1]["seed"]


def test_fewer_errors_wins_and_the_disagreement_reaches_the_repair(tmp_path):
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_BOTH, WRONG_ONE, RIGHT])
    loop = _loop(host, attempts=2, resample=1, narrow="off")
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok, out.stopped_because
    assert len(out.attempts) == 2
    # attempt 1 is recorded as the better sample: one failing test, not two
    first = out.attempts[0]
    assert sum(1 for d in first.diagnostics if d.is_error) < 4
    # the repair prompt names where the samples disagreed
    repair = [m.content for m in host.llm.prompts[2] if m.role == "user"][0]
    assert "[WHERE THE SAMPLES DISAGREED]" in repair
    assert "`double`" in repair
    # … and the doubt map ranks the changed definition least sure
    assert loop.doubt["calc.py"]["double"] < loop.doubt["calc.py"]["triple"]


def test_attempt_1_stays_when_the_sample_is_no_better(tmp_path):
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_ONE, WRONG_BOTH, RIGHT])
    loop = _loop(host, attempts=2, resample=1, narrow="off")
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok, out.stopped_because
    ev = [e for e in loop.journal.events() if e.get("event") == "resample"]
    assert ev[0]["data"]["kept"] == 0
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    assert len(gens) == 3, "attempt 1, its one sample, attempt 2 — once each"
    assert sum("resample 1 of 1" in str(g.get("verify", {}))
               for g in gens) == 1
    # the file on disk after attempt 1 was attempt 1's own reply, so the
    # repair saw it, not the worse sample
    repair = [m.content for m in host.llm.prompts[2] if m.role == "user"][0]
    assert "return x * 2" in repair and "x + x + 1" not in repair


def test_an_identical_sample_is_counted_and_skipped(tmp_path):
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_ONE, WRONG_ONE, RIGHT])
    loop = _loop(host, attempts=2, resample=1, narrow="off")
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok
    ev = [e for e in loop.journal.events() if e.get("event") == "resample"]
    assert ev[0]["data"]["candidates"][0]["skipped"] == \
        "the same code as attempt 1"
    assert ev[0]["data"]["disagreed"] == []


def test_no_resample_when_off_or_for_a_long_file(tmp_path):
    _project(tmp_path)
    host = _host(tmp_path, [WRONG_ONE, RIGHT])
    loop = _loop(host, attempts=2, resample=0, narrow="off")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok and len(out.attempts) == 2
    host = _host(tmp_path, [WRONG_ONE, RIGHT])
    loop = _loop(host, attempts=2, resample=1, resample_max_lines=3,
                 narrow="off")
    out = loop.run_task(_task(), request="double and triple")
    assert out.ok and len(out.attempts) == 2


def test_differing_definitions_by_name():
    a = "def f(x):\n    return 1\n\n\ndef g(x):\n    return 2\n"
    b = "def f(x):\n    return 1\n\n\ndef g(x):\n    return 3\n"
    assert _differing_definitions(a, b) == ["g"]
    c = "def f(x):\n    return 1\n\n\ndef h(x):\n    return 2\n"
    assert _differing_definitions(a, c) == ["g", "h"]
    assert _differing_definitions(a, a) == []
    assert _differing_definitions(a, "def (:\n") == []


def test_the_session_resamples_the_engines_own_providers_only():
    host = Host(llm=ScriptedLLM([], supports_tools=False),
                fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    assert Session(host).loop.config.resample == 0
    assert Session(host, config=SessionConfig(resample=2)) \
        .loop.config.resample == 2
    from cognitive_coder.providers import openai_compatible as oc
    real = oc.OpenAICompatible("http://127.0.0.1:1", model="m",
                               supports_tools=False)
    host = Host(llm=real, fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    assert Session(host).loop.config.resample == 1
    assert Session(host).loop.config.type_gate, "the type gate too"
