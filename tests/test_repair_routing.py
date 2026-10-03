# SPDX-License-Identifier: Apache-2.0
"""A failure is repaired where it is raised, not where it was noticed.

Oct 1, 2026, 21:11: `src/main.py` ran and died at `src/render.py:61` —
`'PhysicsState' object has no attribute 'x'`. The loop asked for main.py
again, got the identical file (there was nothing in main.py to change),
and stopped: "the model produced identical code twice — the task is
probably too large or too vague". The task was fine. The bug was one line
in a different file, and the facts that fixed it — PhysicsState's real
attributes — were sitting in the codemap.

Now the task whose run dies in another project file stops at once and
names that file; the session repairs the culprit against the failure,
with the caller shown and the real definitions under the error; then the
caller is run again, and that is its verdict.
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

PLAN = ("src/alpha.py — the car's state and a helper that reads it\n"
        "src/main.py — the entry point: builds a state and prints it\n")

ALPHA_WRONG = '''```python
class State:
    """The car."""

    def __init__(self) -> None:
        self.speed = 0.0


def describe(state: State) -> str:
    """One line about the car."""
    return f"speed {state.x}"
```'''

MAIN = '''```python
from src.alpha import State, describe


def main() -> int:
    print(describe(State()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```'''

ALPHA_FIXED = '''```python
class State:
    """The car."""

    def __init__(self) -> None:
        self.speed = 0.0


def describe(state: State) -> str:
    """One line about the car."""
    return f"speed {state.speed}"
```'''

MAIN_AGAIN = MAIN     # what a retry of main.py would have produced: the same

#: The interface skeleton's reply (§4.2 step 2). It pins `State` WITHOUT an
#: `x`, which is the point: the stub is the contract, and the body that
#: reads `state.x` broke it.
SKELETON = '''```python
# file: src/alpha.py
class State:
    """The car."""

    def __init__(self) -> None:
        self.speed: float = 0.0


def describe(state: State) -> str:
    """One line about the car."""
    raise NotImplementedError
```
```python
# file: src/main.py
from src.alpha import State, describe


def main() -> int:
    """Build a state and print it."""
    raise NotImplementedError
```'''


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Devstral-Small-2-24B",
                                context_tokens=16384),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def test_a_run_that_dies_in_another_file_repairs_that_file(tmp_path):
    host = _host(tmp_path, [PLAN, SKELETON, ALPHA_WRONG, MAIN, ALPHA_FIXED])
    session = Session(host, config=SessionConfig(attempts=3))
    session.run("a state and a main that prints it")

    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/main.py"].ok, final["src/main.py"].stopped_because
    assert "speed 0.0" not in (tmp_path / "src" / "alpha.py").read_text()
    assert "state.speed" in (tmp_path / "src" / "alpha.py").read_text()

    # main.py was NOT retried: one attempt, then the culprit was named
    first_main = next(o for o in session.outcomes if o.path == "src/main.py")
    assert len(first_main.attempts) == 1
    assert "the error is in src/alpha.py" in first_main.stopped_because
    assert "rewriting src/main.py cannot fix it" in first_main.stopped_because

    # the session said what it was doing, in order
    statuses = [m for _k, m, _d in host.events.of("status")]
    assert any("src/alpha.py will be repaired against src/main.py, which "
               "fails inside it" in m for m in statuses), statuses
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("src/main.py: the error is in src/alpha.py" in m
               for m in warnings), warnings
    assert not any("collected none of its tests" in m for m in warnings), (
        "a caller is not a test file", warnings)

    # the model was asked to WRITE exactly five times: plan, the interface
    # skeleton, alpha, main, the repair of alpha — never main again (the
    # review stage's questions come after and are not generations)
    def last_user(p):
        return [m for m in p if m.role == "user"][-1].content
    writes = [p for p in host.llm.prompts
              if not last_user(p).startswith("[TASK]\nReview ")]
    assert len(writes) == 5, [last_user(p)[:60] for p in host.llm.prompts]

    # and the repair prompt showed the caller, as a caller, not as a test
    sent = writes[-1]
    text = "\n".join(m.content for m in sent if isinstance(m.content, str))
    assert "THE CALLER THAT FAILS INSIDE THIS FILE — src/main.py" in text
    assert "THE TEST THIS FILE MUST PASS" not in text
    assert "WHAT ACTUALLY EXISTS" in text          # the facts under the error
    assert "speed" in text


def test_a_failure_in_the_task_s_own_file_is_still_its_own(tmp_path):
    """The rule is narrow: the innermost project frame must be ANOTHER
    file. A module that dies in itself is retried as before."""
    own_bug = '''```python
def main() -> int:
    return undefined_name


if __name__ == "__main__":
    raise SystemExit(main())
```'''
    fixed = '''```python
def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```'''
    host = _host(tmp_path, ["src/main.py — the entry point\n", own_bug,
                            fixed])
    session = Session(host, config=SessionConfig(attempts=3))
    session.run("a main")
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/main.py"].ok
    assert len(final["src/main.py"].attempts) == 2
    assert not any("the error is in" in m
                   for _k, m, _d in host.events.of("warning"))


def test_a_culprit_the_plan_does_not_own_is_named_not_touched(tmp_path):
    """A pre-existing module the plan never listed is not rewritten on
    the strength of a traceback; the operator is told where to look."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "legacy.py").write_text(
        "def helper(x):\n    return x.nope\n", encoding="utf-8")
    main = '''```python
from src.legacy import helper


def main() -> int:
    print(helper(object()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```'''
    host = _host(tmp_path, ["src/main.py — the entry point\n", main, main])
    session = Session(host, config=SessionConfig(attempts=2))
    session.run("use the legacy helper")
    final = {o.path: o for o in session._final_outcomes()}
    assert not final["src/main.py"].ok
    assert (tmp_path / "src" / "legacy.py").read_text() == \
        "def helper(x):\n    return x.nope\n"
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("not a file this plan owns" in m for m in warnings), warnings
