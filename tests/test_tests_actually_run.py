# SPDX-License-Identifier: Apache-2.0
"""The caveat that was true every time, and nobody heard it.

Every build of every project reported, on every file::

    the test command succeeded but ran ZERO tests — that is not evidence the
    code works, only that nothing contradicted it

It was written as a warning about an edge case. It was describing the normal
state of affairs, and it was exactly right: nothing was ever verified.

THE CAUSE is one line of `unittest` behaviour. `discover` will not descend
into a directory that is not an importable package. The generated `tests\\`
folder had no `__init__.py`, so discovery from the project root walked past
it, found nothing, and exited zero.

Measured on a real generated project:

    python -m unittest discover -s . -p "test_*.py"
        Ran 0 tests   OK

    touch tests/__init__.py
    python -m unittest discover -s . -p "test_*.py"
        Ran 14 tests  FAILED (failures=3, errors=7)

Ten real defects that four green builds had been stepping over.

HOW IT HID, which is the part worth keeping. The caveat was honest,
prominent, and printed on every file — and *because* it was on every file it
read as boilerplate rather than as a finding. A warning that is always on is
a warning nobody sees. That is the same shape as the installer that ran fast
and skipped four stages: the evidence was present and unreadable.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.planner import Planner                   # noqa: E402
from cognitive_coder.types import Plan, Task                  # noqa: E402


class _FS:
    def __init__(self):
        self.files: dict[str, str] = {}

    def write(self, path, text):
        self.files[path] = text

    def exists(self, path):
        return path in self.files

    def read(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


class _Host:
    def __init__(self):
        self.fs = _FS()
        self.emitted = []

    def emit(self, kind, message, data=None):
        self.emitted.append((kind, message))


def _planner():
    p = object.__new__(Planner)
    p.lang, p.journal, p.codemap = "python", None, None
    p.host = _Host()
    return p


def _plan(paths):
    return Plan(request="r", tasks=tuple(
        Task(id=f"t{i + 1}", path=p, purpose="x", test_path="",
             persona="engineer", lang="python", atomic=False)
        for i, p in enumerate(paths)))


# ==========================================================================
# the cause, demonstrated against real unittest
# ==========================================================================

def test_discover_skips_a_directory_that_is_not_a_package(tmp_path):
    """Not a claim about our code — a fact about `unittest`.

    Asserted by running it, because this is the whole reason the bug existed
    and a paraphrase of the behaviour is what let it survive.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.py").write_text("def add(a, b):\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_thing.py").write_text(textwrap.dedent("""
        import unittest
        from src.thing import add

        class T(unittest.TestCase):
            def test_it(self):
                self.assertEqual(add(1, 1), 2)
    """).strip() + "\n")

    def discover():
        return subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", ".",
             "-p", "test_*.py"],
            cwd=tmp_path, capture_output=True, text=True, timeout=120)

    before = discover()
    assert "Ran 0 tests" in before.stderr, before.stderr
    #: And it EXITS ZERO. That is what made a silent no-op look like a pass.
    assert before.returncode == 0

    (tmp_path / "tests" / "__init__.py").write_text("")
    after = discover()
    assert "Ran 1 test" in after.stderr, after.stderr
    assert after.returncode != 0, "the failing test must now be seen"


# ==========================================================================
# the fix
# ==========================================================================

def test_the_skeleton_makes_every_python_folder_a_package():
    p = _planner()
    p._make_packages(_plan(["src/math3d.py", "src/physics.py",
                            "tests/test_math3d.py", "tests/test_physics.py"]))
    created = sorted(p.host.fs.files)
    assert "tests/__init__.py" in created, created
    #: src too — a namespace package is enough for the import to work, but
    #: being explicit removes the difference between a layout that happens to
    #: work and one that is meant to.
    assert "src/__init__.py" in created, created


def test_it_does_not_clobber_an_existing_package_file():
    p = _planner()
    p.host.fs.files["tests/__init__.py"] = "# hand-written, keep me\n"
    p._make_packages(_plan(["tests/test_a.py"]))
    assert p.host.fs.files["tests/__init__.py"] == "# hand-written, keep me\n"


def test_a_file_at_the_root_needs_no_package():
    p = _planner()
    p._make_packages(_plan(["main.py"]))
    assert p.host.fs.files == {}, p.host.fs.files


def test_only_python(tmp_path):
    p = _planner()
    p.lang = "rust"
    p._make_packages(_plan(["src/main.rs", "tests/test_main.rs"]))
    assert p.host.fs.files == {}, "__init__.py means nothing outside Python"


def test_a_folder_it_cannot_write_is_not_fatal():
    """The write jail refusing is the jail working, not an error to raise."""
    p = _planner()

    def refuse(path, text):
        raise PermissionError(path)

    p.host.fs.write = refuse
    p._make_packages(_plan(["tests/test_a.py"]))     # must not raise


# ==========================================================================
# the second finding from the same run
# ==========================================================================

def test_a_task_blocked_by_a_failure_is_named():
    """`src/main.py` was never attempted and never mentioned.

    It depended on `src/render.py`, which failed three attempts earlier. A
    dependency that failed never becomes done, so `next_ready` stops
    returning the task and the run simply ends — showing "[build 6/6]"
    against a plan of seven files.

    Not building it is right. Not saying so is the installer's mistake in a
    different costume: an absence nobody can see.
    """
    tasks = [
        Task(id="t1", path="src/render.py", purpose="", test_path="",
             persona="engineer", lang="python", status="failed"),
        Task(id="t2", path="src/main.py", purpose="", test_path="",
             persona="engineer", lang="python", status="pending",
             depends_on=("t1",)),
    ]
    plan = Plan(request="r", tasks=tuple(tasks))

    assert plan.next_ready() is None, "correctly refuses to build it"
    blocked = plan.blocked()
    assert len(blocked) == 1, blocked
    task, blockers = blocked[0]
    assert task.path == "src/main.py"
    assert blockers == ["src/render.py"], blockers


def test_nothing_is_reported_blocked_when_the_build_went_well():
    tasks = [
        Task(id="t1", path="src/a.py", purpose="", test_path="",
             persona="engineer", lang="python", status="done"),
        Task(id="t2", path="src/b.py", purpose="", test_path="",
             persona="engineer", lang="python", status="pending",
             depends_on=("t1",)),
    ]
    plan = Plan(request="r", tasks=tuple(tasks))
    assert plan.blocked() == []
    assert plan.next_ready() is not None


def test_the_session_announces_it():
    src = (Path(__file__).resolve().parent.parent / "cognitive_coder"
           / "session.py").read_text(encoding="utf-8")
    assert "was never attempted" in src
    assert "self.plan.blocked()" in src
    #: Journalled as well as emitted, so it survives the console scrolling.
    assert 'journal.log("blocked"' in src


@pytest.mark.parametrize("phrase", [
    "will not descend into a directory that is not an",
    "a warning that is always on is a warning nobody sees",
])
def test_the_reasoning_is_recorded(phrase):
    """Written down where the next person will be standing when it matters."""
    src = (Path(__file__).resolve().parent.parent / "cognitive_coder"
           / "planner.py").read_text(encoding="utf-8").lower()
    assert phrase.lower() in src, phrase
