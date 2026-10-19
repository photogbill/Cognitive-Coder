# SPDX-License-Identifier: Apache-2.0
"""Narrowed edits in the loop (F5) — and what came with them: the marked
error lines, regression memory read back, the opt-in diagnosis.

F5 was specified and never wired: `narrow_on_stagnation` was declared and
nothing read it, and a file too large to show whole could not be repaired
at all. Each test here fails on the code before it.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import _host, _loop  # noqa: E402

from cognitive_coder import personas  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.types import Diagnostic, Task  # noqa: E402

FILLER = "".join(f"\n\ndef helper_{i}(x):\n    return x + {i}\n"
                 for i in range(60))
CAR = ('"""A car."""\n\n\nclass Car:\n    def __init__(self):\n'
       '        self.speed = 0.0\n\n    def update(self, dt):\n'
       '        self.speed += dt * accel\n        return self.speed\n'
       + FILLER +
       '\n\ndef main():\n    car = Car()\n    print(car.update(1.0))\n\n\n'
       'if __name__ == "__main__":\n    main()\n')
SEED = (Diagnostic(file="car.py", line=9, severity="error",
                   message="NameError: name 'accel' is not defined"),)
FIX = ("```python\ndef update(self, dt):\n    self.speed += dt * 2.0\n"
       "    return self.speed\n```")


def _user(host, call):
    return [m.content for m in host.llm.prompts[call] if m.role == "user"][0]


def _task():
    return Task(id="t1", path="car.py", purpose="a car", lang="python")


def test_a_long_files_repair_returns_only_the_function(tmp_path):
    (tmp_path / "car.py").write_text(CAR)
    host = _host(tmp_path, [FIX])
    loop = _loop(host, attempts=2)
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(_task(), request="make the car work", seed=SEED)
    assert out.ok, out.stopped_because
    tail = _user(host, 0)
    assert "The error is in `Car.update`" in tail
    assert personas.CONTRACT_SYMBOLS in tail
    after = (tmp_path / "car.py").read_text()
    assert "self.speed += dt * 2.0" in after
    assert after.count("def helper_") == 60          # nothing else touched
    narrow = [e for e in loop.journal.events() if e.get("event") == "narrow"]
    assert narrow and narrow[0]["data"]["placed"] is True
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    assert gens[-1]["data"].get("via") == "narrow"


def test_a_short_files_repair_stays_whole_file(tmp_path):
    src = "def f(x):\n    return x + y\n"
    (tmp_path / "m.py").write_text(src)
    host = _host(tmp_path, ["```python\ndef f(x):\n    return x + 1\n```"])
    seed = (Diagnostic(file="m.py", line=2, severity="error",
                       message="NameError: name 'y' is not defined"),)
    _loop(host, attempts=1).run_task(
        Task(id="t1", path="m.py", purpose="f", lang="python"), seed=seed)
    assert personas.CONTRACT_FILE in _user(host, 0)


def test_narrowing_off_is_the_whole_file_always(tmp_path):
    (tmp_path / "car.py").write_text(CAR)
    host = _host(tmp_path, ["```python\n" + CAR.replace("accel", "2.0")
                            + "```"])
    _loop(host, attempts=1, narrow="off").run_task(_task(), seed=SEED)
    assert personas.CONTRACT_FILE in _user(host, 0)


BIG = CAR.replace(FILLER, "".join(
    f"\n\ndef helper_{i}(x):\n    y = x * {i}\n    if y > 100:\n"
    f"        y = y - 100\n    elif y < -100:\n        y = y + 100\n"
    f"    return x + y + {i}\n" for i in range(60)))


def test_a_file_too_large_to_show_whole_is_repaired_condensed(tmp_path):
    """Refused before any model call, before this — now the target in full
    and every other definition as its header."""
    (tmp_path / "car.py").write_text(BIG)
    host = Host(llm=ScriptedLLM([FIX], supports_tools=False,
                                context_tokens=2600),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=1, max_tokens=512)
    out = loop.run_task(_task(), request="make the car work", seed=SEED)
    assert out.ok, out.stopped_because
    tail = _user(host, 0)
    assert "condensed" in tail
    assert "return x + y + 41" not in tail   # an elided body
    assert "def helper_41(x):" in tail       # its header
    assert "self.speed += dt * accel" in tail  # the target, whole
    after = (tmp_path / "car.py").read_text()
    assert after.count("def helper_") == 60
    assert "return x + y + 41" in after


def test_an_unplaceable_reply_for_a_too_large_file_is_a_refusal(tmp_path):
    (tmp_path / "car.py").write_text(BIG)
    host = Host(llm=ScriptedLLM(["```python\nprint('hello')\n```", FIX],
                                supports_tools=False, context_tokens=2600),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    out = _loop(host, attempts=2, max_tokens=512).run_task(
        _task(), seed=SEED)
    assert out.ok, out.stopped_because
    assert "could not be placed" in out.attempts[0].note
    assert "could not be put into car.py" in _user(host, 1)


def test_the_same_error_twice_narrows_the_next_attempt(tmp_path):
    """F5: identical diagnostics twice ⇒ narrow, don't retry."""
    src = ("def area(w, h):\n    return w * hh\n\n\n"
           "def perimeter(w, h):\n    return 2 * (w + h)\n\n\n"
           "print(area(2, 3))\n")
    (tmp_path / "m.py").write_text(src)
    wrong = src.replace("hh", "hhh")
    replies = ["```python\n" + wrong + "```",
               "```python\n" + wrong.replace("2 * (w + h)", "(w + h) * 2")
               + "```",
               "```python\ndef area(w, h):\n    return w * h\n```"]
    host = _host(tmp_path, replies)
    seed = (Diagnostic(file="m.py", line=2, severity="error",
                       message="NameError: name 'hh' is not defined"),)
    out = _loop(host, attempts=3, escalate=False).run_task(
        Task(id="t1", path="m.py", purpose="geometry", lang="python"),
        seed=seed)
    assert out.ok, out.stopped_because
    assert personas.CONTRACT_SYMBOLS in _user(host, 2)
    assert "def perimeter" in (tmp_path / "m.py").read_text()


def test_the_errors_are_marked_where_they_are(tmp_path):
    src = "class A:\n    def f(self):\n        x = 1\n        return y\n"
    (tmp_path / "m.py").write_text(src)
    host = _host(tmp_path, ["```python\nclass A:\n    def f(self):\n"
                            "        return 1\n```"])
    seed = (Diagnostic(file="m.py", line=4, severity="error",
                       message="NameError: name 'y' is not defined"),)
    _loop(host, attempts=1).run_task(
        Task(id="t1", path="m.py", purpose="a", lang="python"), seed=seed)
    tail = _user(host, 0)
    assert "[WHERE THE ERRORS ARE" in tail
    assert "█ 4 |         return y" in tail
    assert "  1 | class A:" in tail


def test_how_an_error_was_fixed_before_is_offered_next_time(tmp_path):
    """F10 wrote every fix and nothing read one back; and every model fix
    was recorded as the useless shape "model repair"."""
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    replies = ["```python\nVALUE = math.pi\n```",
               "```python\nimport math\n\nVALUE = math.pi\n```",
               "```python\nOTHER = math.tau\n```",
               "```python\nimport math\n\nOTHER = math.tau\n```"]
    host = _host(tmp_path, replies)
    loop = _loop(host, attempts=2, autofix=False)
    loop.run_task(Task(id="t1", path="a.py", purpose="pi", lang="python"))
    loop.run_task(Task(id="t2", path="b.py", purpose="tau", lang="python"))
    tail = _user(host, 3)
    assert "HOW THIS SAME ERROR WAS FIXED BEFORE" in tail
    assert "added 1 import" in tail


def test_the_opt_in_diagnosis_comes_before_the_repair(tmp_path):
    (tmp_path / "m.py").write_text("")
    replies = ["```python\nVALUE = undefined_one\n```",
               "`undefined_one` does not exist; use the literal 3.",
               "```python\nVALUE = 3\n```"]
    host = _host(tmp_path, replies)
    out = _loop(host, attempts=2, diagnose_first=True).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"))
    assert out.ok, out.stopped_because
    assert personas.CONTRACT_DIAGNOSIS in _user(host, 1)
    assert "use the literal 3" in _user(host, 2)
