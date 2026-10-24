# SPDX-License-Identifier: Apache-2.0
"""A reply stuck repeating itself — found while it streams, and what the loop
does when a host cuts one (2026-10-24).

Two halves. `repetition.find` must flag a real loop early enough to matter
and NEVER flag ordinary code: a false cut costs a whole attempt, so the
negative tests here matter more than the positive ones — including the
whole of the engine's own source, streamed through a `Watch` seven
characters at a time. Then the loop: a `cut` reply is an attempt, honestly
recorded, that writes nothing, hands its inputs on, and is followed by an
attempt that samples differently (the same seed would loop again).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import repetition  # noqa: E402
from cognitive_coder.codemap import CodeMap  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.loop import Loop, LoopConfig, _is_truncated  # noqa: E402
from cognitive_coder.patcher import Patcher  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.types import FINISH_REASONS, Completion, Task  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BLOCK = ("    if self.enemy.hp <= 0:\n"
         "        self.score += 10\n"
         "        self.spawn_enemy()\n")
HEAD = "import pygame\n\nclass Game:\n    def update(self):\n"


# --------------------------------------------------------------------------
# finding a loop
# --------------------------------------------------------------------------

def test_a_block_repeated_word_for_word_is_a_loop():
    text = HEAD + BLOCK * 40
    found = repetition.find(text)
    assert found is not None and found.looping
    assert found.period == len(BLOCK)
    assert found.describe() == "the same 3 lines, 40 times over"
    # what is worth keeping: everything before, and ONE copy
    assert text[:found.keep].count("spawn_enemy") == 1
    assert text[:found.keep].startswith(HEAD)


def test_a_short_run_is_only_watched():
    found = repetition.find("x = 1\n" + BLOCK * 8)
    assert found is not None and found.level == "watch"
    assert not found.looping


def test_a_thinking_loop_without_newlines_is_found():
    text = ("<think>Let me check the import. "
            + "Wait, the import of pygame.mixer must come first. " * 60)
    found = repetition.find(text)
    assert found is not None and found.looping
    assert found.describe().startswith("the same phrase")


def test_two_copies_of_a_function_are_not_a_loop_but_three_are():
    fn = "".join(f"def f{i}(a, b):\n    return a * {i} + b - {i * 3}\n\n"
                 for i in range(30))
    assert repetition.find("x\n" + fn + fn) is None
    found = repetition.find("x\n" + fn * 3)
    assert found is not None and found.looping


def test_a_loop_the_model_left_behind_is_not_flagged():
    """Only the TAIL counts: repetition the model finished is a pattern."""
    text = ("a\n" + BLOCK * 10
            + "".join(f"value_{i} = compute({i}, 'alpha')\n"
                      for i in range(60)))
    assert repetition.find(text) is None


def test_data_and_layout_are_never_a_loop():
    """Periodic, harmless, and the reason a unit needs four letters."""
    grid = "MAP = [\n" + ("    [" + ", ".join(["0"] * 20) + "],\n") * 80
    rows = "LEVEL = [\n" + ('    "#' + "." * 38 + '#",\n') * 80
    rule = "# " + "-" * 3000
    closers = "x = 1\n" + "}\n" * 1500
    for text in (grid, rows, rule, closers):
        assert repetition.find(text) is None, text[:60]


def test_the_engines_own_source_never_trips_it():
    """Forty thousand lines of real code, streamed as a model would send
    them. Not one `watch`, let alone a cut."""
    files = sorted((ROOT / "cognitive_coder").rglob("*.py"))
    files += sorted((ROOT / "tests").glob("*.py"))
    assert len(files) > 50
    for path in files:
        src = path.read_text(encoding="utf-8", errors="replace")
        w = repetition.Watch()
        for k in range(0, len(src), 7):
            w.feed(src[k:k + 7])
            assert w.found is None, (path.name, w.found.to_dict())


def test_a_watch_sees_it_early_and_cuts_well_before_the_ceiling():
    w = repetition.Watch()
    w.feed("class A:\n    pass\n")
    first_watch = loop_at = None
    for _ in range(400):
        w.feed(BLOCK)
        if w.found is not None and first_watch is None:
            first_watch = len(w)
        if w.looping:
            loop_at = len(w)
            break
    assert first_watch is not None and first_watch < 1200
    # LOOP_CHARS of repetition, plus at most one check interval and a copy
    assert loop_at is not None
    assert loop_at < repetition.LOOP_CHARS + 2 * w.every + 2 * len(BLOCK)


def test_a_featureless_tail_is_skipped_not_misread():
    """Deep indentation recurs everywhere; probing it would try thousands
    of periods. The check is skipped and the next one probes again."""
    assert repetition.find("x = 1\n" + " " * 5000) is None


def test_describe_reads_as_an_operator_would_say_it():
    one_line = repetition.find("z\n" + "print('the enemy moved left')\n" * 90)
    assert one_line is not None
    assert one_line.describe().startswith("the same line, ")
    assert one_line.to_dict()["level"] == "loop"


# --------------------------------------------------------------------------
# a cut reply, in the loop
# --------------------------------------------------------------------------

def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _loop(host, **config):
    loop = Loop(host, codemap=CodeMap(host.fs, host.storage),
                patcher=Patcher(host.fs, host.storage, host.approval,
                                host.events),
                config=LoopConfig(**config))
    loop.journal = Journal(host.fs, "t")
    return loop


def _cut(text="```python\nX = 1\n" + "X = 1\n" * 50):
    return Completion(text=text, finish_reason="cut", model="scripted",
                      error="it was repeating itself: the same line, "
                            "50 times over")


def test_cut_is_a_finish_reason():
    assert "cut" in FINISH_REASONS
    assert Completion(text="", finish_reason="cut").finish_reason == "cut"


def test_a_cut_reply_is_never_continued():
    """A cut ends where the host stopped it; continuing it asks for more
    of the loop. Unbalanced on purpose."""
    assert not _is_truncated(_cut("def f(a, b:\n    return {"),
                             "def f(a, b:\n    return {", "python")


def test_a_cut_reply_writes_nothing_and_the_next_attempt_succeeds(tmp_path):
    host = _host(tmp_path, [_cut(), "```python\nX = 1\n```"])
    loop = _loop(host, attempts=2, seed=11)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="a constant",
                                 lang="python"), request="make a constant")
    assert outcome.ok, outcome
    assert host.fs.read("m.py").count("X = 1") == 1
    gens = [r for r in loop.journal.events() if r.get("event") == "generate"]
    assert [g.get("attempt") for g in gens] == [1, 2]
    assert gens[0]["data"].get("finish_reason") == "cut"
    assert gens[0]["verify"].get("cut") is True
    assert "repeating itself" in gens[0]["verify"].get("note", "")
    # the operator is told, with the attempt and the reason
    warnings = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("attempt 1 was cut" in m for m in warnings), warnings


def test_the_attempt_after_a_cut_samples_differently(tmp_path):
    """The ladder keeps attempt 2 at attempt 1's seed and temperature, and
    those would write the same loop token for token."""
    host = _host(tmp_path, [_cut(), "```python\nX = 1\n```"])
    loop = _loop(host, attempts=2, seed=11)
    loop.run_task(Task(id="t1", path="m.py", purpose="a constant",
                       lang="python"), request="make a constant")
    gens = [r for r in loop.journal.events() if r.get("event") == "generate"]
    first, second = gens[0], gens[1]
    assert second.get("seed") != first.get("seed")
    assert second.get("temperature") > first.get("temperature")
    assert "cut" in (second["data"].get("escalation") or "")


def test_after_a_cut_the_attempt_is_done_again_not_skipped(tmp_path):
    """Nothing was written, so attempt 2 is attempt 1 again — the first
    attempt's prompt, not a repair of nothing."""
    host = _host(tmp_path, [_cut(), "```python\nX = 1\n```"])
    _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request="make a constant")
    first = [m.content for m in host.llm.prompts[0] if m.role == "user"]
    second = [m.content for m in host.llm.prompts[1] if m.role == "user"]
    assert first == second


def test_a_build_of_cuts_gives_up_saying_so(tmp_path):
    host = _host(tmp_path, [_cut(), _cut()])
    outcome = _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request="make a constant")
    assert not outcome.ok
    assert "repeating itself" in outcome.stopped_because, outcome
    assert [a.finish_reason for a in outcome.attempts] == ["cut", "cut"]
