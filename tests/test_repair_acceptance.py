# SPDX-License-Identifier: Apache-2.0
"""A cross-file repair is accepted only when its cause is gone.

Oct 2, 2026, 358.9 s into the racing build: `REPAIRED src/render.py — 1
attempt(s)`. Snapshot 0007-t4-fix has an empty "what changed": the model
had returned render.py byte for byte, and the engine accepted it because
render.py passed its OWN check — an import and zero tests. main.py then
failed with the same `'tuple' object has no attribute 'x'` at render.py:24.

Two rules now:

  * a repair that returns the file unchanged (identical after normalising:
    the AST, so a comment or blank line is no change) is reported at once
    as NOT REPAIRED, with the reason, and nothing is run for it;
  * a repair of B triggered by A's failure is judged by A: after B's own
    check passes, A's runs, and while A still fails inside B the repair is
    not accepted — A's error drives B's next attempt.

`pin_interfaces=False` throughout: these replay the Oct 2 conditions, where
each file was written blind to the others.
"""

from __future__ import annotations

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
)

PLAN = ("src/track.py — the track data model\n"
        "src/render.py — draws the visible segments\n"
        "src/main.py — the entry point: builds the track and draws it\n")

TRACK = '''```python
from typing import List, Tuple

TrackSegment = Tuple[float, float, int]  # (curve, color, z_position)


def build_track() -> List[TrackSegment]:
    """Three straight segments."""
    return [(0.0, i % 2, i) for i in range(3)]
```'''

#: render.py as Oct 2 wrote it, in miniature: attributes read off a tuple.
RENDER_WRONG = '''```python
from typing import List

from src.track import TrackSegment


def draw(segments: List[TrackSegment]) -> List[float]:
    """The depth of each segment."""
    return [seg.z for seg in segments]
```'''

#: The same file, re-indented and with a comment: identical once normalised.
RENDER_SAME_AGAIN = '''```python
from typing import List
from src.track import TrackSegment

# Draw the road.
def draw(segments: List[TrackSegment]) -> List[float]:
    """The depth of each segment."""
    return [seg.z for seg in segments]
```'''

#: A real change that does not fix it: still an attribute off a tuple.
RENDER_STILL_WRONG = '''```python
from typing import List

from src.track import TrackSegment


def draw(segments: List[TrackSegment]) -> List[float]:
    """The depth of each segment, nearest first."""
    depths = [seg.z for seg in segments]
    return sorted(depths)
```'''

RENDER_FIXED = '''```python
from typing import List

from src.track import TrackSegment


def draw(segments: List[TrackSegment]) -> List[float]:
    """The depth of each segment, nearest first."""
    return sorted(float(z) for _curve, _color, z in segments)
```'''

MAIN = '''```python
from src.render import draw
from src.track import build_track


def main() -> int:
    print(len(draw(build_track())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```'''


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Qwen3-Coder-30B-A3B-Instruct",
                                context_tokens=32768),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _session(tmp_path, replies):
    host = _host(tmp_path, replies)
    session = Session(host, config=SessionConfig(
        attempts=3, pin_interfaces=False, review_after_build=False))
    return host, session


def _log(tmp_path) -> str:
    return (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")


def test_a_repair_that_returns_the_same_code_is_not_repaired(tmp_path):
    host, session = _session(tmp_path, [PLAN, TRACK, RENDER_WRONG, MAIN,
                                        RENDER_SAME_AGAIN])
    session.run("a track and a renderer")

    fix = [o for o in session.outcomes if o.path == "src/render.py"][-1]
    assert not fix.ok
    assert len(fix.attempts) == 1, "it stops at once, no second attempt"
    assert "the model returned the same code" in fix.stopped_because
    assert "src/main.py's failure cannot have gone away" in \
        fix.stopped_because

    log = _log(tmp_path)
    assert "NOT REPAIRED src/render.py" in log
    assert "] REPAIRED src/render.py" not in log
    # main.py is not "checked again" after a repair that did nothing
    final = {o.path: o for o in session._final_outcomes()}
    assert not final["src/main.py"].ok
    assert "seg.z" in (tmp_path / "src" / "render.py").read_text()
    # plan, track, render, main, the one repair — nothing more was asked
    assert len(host.llm.prompts) == 5


def test_a_repair_is_accepted_only_when_the_callers_failure_is_gone(
        tmp_path):
    host, session = _session(tmp_path, [PLAN, TRACK, RENDER_WRONG, MAIN,
                                        RENDER_STILL_WRONG, RENDER_FIXED])
    session.run("a track and a renderer")

    fix = [o for o in session.outcomes if o.path == "src/render.py"][-1]
    assert fix.ok, fix.stopped_because
    assert len(fix.attempts) == 2, [a.note for a in fix.attempts]
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/main.py"].ok, final["src/main.py"].stopped_because
    assert "for _curve, _color, z in segments" in \
        (tmp_path / "src" / "render.py").read_text()

    # attempt 1 passed render.py's own check and was NOT accepted, because
    # main.py still died inside it — said in so many words
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("src/render.py: its own check passes, but src/main.py still "
               "fails inside src/render.py" in m for m in warnings), warnings
    # and attempt 2 was asked with main.py's error and the alias's truth
    second = host.llm.prompts[-1]
    text = "\n".join(m.content for m in second if isinstance(m.content, str))
    assert "'tuple' object has no attribute 'z'" in text
    assert "TrackSegment = Tuple[float, float, int]" in text
    assert "plain tuple" in text

    log = _log(tmp_path)
    assert "REPAIRED src/render.py — 2 attempt(s)" in log
    # the acceptance run of main.py is in the log, under render's attempt
    assert log.count("verify src/main.py") >= 3


def test_an_unchanged_repair_against_a_test_says_the_test_may_be_wrong(
        tmp_path):
    """The same rule for a module repaired against its failing test: the
    prompt allows "return it unchanged if the test is wrong", and that
    answer is reported as what it is — not as a repair, and not after a
    second identical attempt."""
    plan = "src/calc.py — adds two numbers\ntests/test_calc.py — tests\n"
    calc = '''```python
def add(a: int, b: int) -> int:
    """The sum."""
    return a - b
```'''
    test = '''```python
import unittest

from src.calc import add


class TestCalc(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
```'''
    host = _host(tmp_path, [plan, calc, test, calc])
    session = Session(host, config=SessionConfig(
        attempts=3, pin_interfaces=False, review_after_build=False))
    session.run("add two numbers, with a test")
    fix = [o for o in session.outcomes if o.path == "src/calc.py"][-1]
    assert not fix.ok and len(fix.attempts) == 1
    assert "the model returned the same code" in fix.stopped_because
    assert "certain the test itself is wrong" in fix.stopped_because
    assert len(host.llm.prompts) == 4
