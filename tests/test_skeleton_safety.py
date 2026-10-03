# SPDX-License-Identifier: Apache-2.0
"""The skeleton writes to someone's project, so it obeys the same rules as
every other write: snapshotted, approved, undoable — and never over a file
that already has real work in it.

WHY THIS FILE EXISTS. `Planner.skeleton()` called `host.fs.write` for every
non-test task, unconditionally. A plan that named an existing file — "extend
src/util.py" — replaced two hundred lines of hand-written code with a stub,
with no snapshot, no approval and no undo. And under the library default
`DenyAll` the stubs and `src/__init__.py` were written anyway, while the CLI
told the operator "nothing was written".
"""

from __future__ import annotations

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
)
from cognitive_coder.ports import DenyAll  # noqa: E402

REAL = ('"""Hand-written, 200 lines of real work."""\n\n\n'
        'def util(x):\n    return x * 2\n')
PLAN = "src/util.py — extend the utilities\nsrc/new.py — a new thing\n"
REQUEST = "extend src/util.py and add src/new.py"
#: The interface skeleton's reply: two Python modules is a contract to pin,
#: and the kept file is shown to it rather than stubbed (`interfaces.py`).
SKELETON = ('```python\n# file: src/new.py\ndef new() -> int:\n'
            '    """A new thing."""\n    raise NotImplementedError\n```')


def _host(tmp_path, replies, approval=None):
    return Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(),
                approval=approval or AutoApprove())


def _project_files(host):
    """Everything in the project that is not the engine's own record."""
    return sorted(p for p in host.fs.list("**/*")
                  if not p.startswith((".cc_journal/", ".state/")))


def _seed(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.py").write_bytes(REAL.encode("utf-8"))


def test_an_existing_file_with_a_body_is_kept_byte_for_byte(tmp_path):
    _seed(tmp_path)
    host = _host(tmp_path, [PLAN, SKELETON])
    session = Session(host, config=SessionConfig(attempts=1))
    session.start(REQUEST)
    assert (tmp_path / "src" / "util.py").read_bytes() == \
        REAL.encode("utf-8")
    assert "src/util.py" in [t.path for t in session.plan.tasks], \
        "the path must stay in the plan — it is still work to do"
    assert any("kept as found" in c and "src/util.py" in c
               for c in session.plan.caveats), session.plan.caveats


def test_the_skeleton_is_one_snapshotted_transaction(tmp_path):
    _seed(tmp_path)
    host = _host(tmp_path, [PLAN, SKELETON])
    session = Session(host, config=SessionConfig(attempts=1))
    session.start(REQUEST)
    skeleton = [r for r in session.history() if r.task_id == "skeleton"]
    assert len(skeleton) == 1, session.history()
    record = skeleton[0]
    assert record.state == "committed"
    assert not record.verified, "a stub is not verified work"
    assert set(record.files) == {"src/new.py", "src/__init__.py"}
    manifest = tmp_path / record.snapshot_dir / "MANIFEST.txt"
    assert manifest.exists(), "no snapshot was persisted for the skeleton"


def test_the_skeleton_asks_for_approval_once(tmp_path):
    _seed(tmp_path)
    approval = AutoApprove()
    host = _host(tmp_path, [PLAN, SKELETON], approval=approval)
    Session(host, config=SessionConfig(attempts=1)).start(REQUEST)
    asked = [summary for summary, _diff in approval.diffs]
    assert len(asked) == 1, asked
    assert asked[0].startswith("skeleton"), asked
    assert "src/new.py" in approval.diffs[0][1]
    assert "src/util.py" not in approval.diffs[0][1], \
        "the kept file must not appear in the diff being approved"


def test_the_skeleton_can_be_undone(tmp_path):
    host = _host(tmp_path, [PLAN, SKELETON])
    session = Session(host, config=SessionConfig(attempts=1))
    session.start(REQUEST)
    assert (tmp_path / "src" / "new.py").exists()
    seq = [r for r in session.history() if r.task_id == "skeleton"][0].seq
    result = session.patcher.undo_to(seq - 1)
    assert result["ok"], result
    assert not (tmp_path / "src" / "new.py").exists()


def test_under_deny_all_the_skeleton_writes_nothing_and_says_so(tmp_path):
    _seed(tmp_path)
    host = _host(tmp_path, [PLAN, SKELETON], approval=DenyAll())
    before = _project_files(host)
    session = Session(host, config=SessionConfig(attempts=1))
    session.start(REQUEST)
    assert _project_files(host) == before
    said = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("not approved" in m and "nothing was written" in m
               for m in said), said


# --------------------------------------------------------------------------
# item 8: a stub is recognised as a stub in every language
# --------------------------------------------------------------------------

def test_every_file_of_a_javascript_plan_is_generated(tmp_path):
    """`_has_body` knew only the Python stub's `NotImplementedError`, so a
    JS scaffold — a `console.log` body — read as finished work. After the
    first step `replan` marked every remaining task done: a 3-file plan
    built one file and reported nothing wrong."""
    def js(name):
        return (f"```javascript\nfunction {name}() {{ return 1; }}\n"
                f"module.exports = {{ {name} }};\n```")
    plan = "lib/a.js — helpers\nlib/b.js — more helpers\nlib/c.js — app\n"
    host = _host(tmp_path, [plan, js("a"), js("b"), js("c")])
    session = Session(host, config=SessionConfig(
        attempts=1, lang="javascript", review_after_build=False))
    session.run("three js modules")
    assert len(host.llm.prompts) == 4, [t.status for t in session.plan.tasks]
    assert sorted(o.path for o in session.outcomes) == \
        ["lib/a.js", "lib/b.js", "lib/c.js"]


def test_the_stub_marker_uses_the_languages_own_comment():
    from cognitive_coder.planner import STUB_SENTINEL, Planner, _has_body
    from cognitive_coder.ports import MemoryFileSystem
    from cognitive_coder.types import Plan, Task

    planner = Planner(Host(fs=MemoryFileSystem()))
    # Either of the language's own comment forms is right; the block form
    # is used where the language declares one (JavaScript does).
    for lang, path, first in (("javascript", "a.js", ("//", "/*")),
                              ("lua", "a.lua", ("--",)),
                              ("c", "a.c", ("/*",))):
        task = Task(id="t1", path=path, purpose="helpers", lang=lang)
        stub = planner.stub_for(task, Plan(request="r", tasks=(task,)))
        marker = [ln for ln in stub.splitlines() if STUB_SENTINEL in ln]
        assert marker and marker[0].startswith(first), (lang, stub)
        planner.host.fs.write(path, stub)
        assert not _has_body(planner.host.fs, path, lang), (lang, stub)
    # A shebang must stay on line one, or the script stops being one.
    task = Task(id="t1", path="run.sh", purpose="runner", lang="bash")
    stub = planner.stub_for(task, Plan(request="r", tasks=(task,)))
    assert stub.startswith("#!"), stub
    assert STUB_SENTINEL in stub.splitlines()[1]


def test_real_code_and_old_python_stubs_are_told_apart():
    from cognitive_coder.planner import _has_body
    from cognitive_coder.ports import MemoryFileSystem
    fs = MemoryFileSystem()
    fs.write("a.js", "function a() { return 1; }\n")
    assert _has_body(fs, "a.js", "javascript")
    fs.write("old.py", 'def main() -> int:\n    raise NotImplementedError(\n'
                       '        "old.main is not written yet")\n')
    assert not _has_body(fs, "old.py", "python"), \
        "stubs written before the sentinel existed are still stubs"


def test_a_copied_stub_marker_does_not_survive_into_real_code(tmp_path):
    """A model that echoes the stub's marker line into its implementation
    would leave a real file that looks like a stub forever."""
    from cognitive_coder.planner import STUB_SENTINEL
    plan = "src/a.py — one thing\n"
    host = _host(tmp_path, [plan])
    session = Session(host, config=SessionConfig(
        attempts=1, review_after_build=False))
    session.start("one module")
    stub = (tmp_path / "src" / "a.py").read_text()
    marker = next(ln for ln in stub.splitlines() if STUB_SENTINEL in ln)
    host.llm._replies.append(
        f'```python\n{marker}\ndef a():\n    """One."""\n    return 1\n```')
    session.run()
    written = (tmp_path / "src" / "a.py").read_text()
    assert STUB_SENTINEL not in written, written
    assert "def a()" in written


def test_a_kept_file_is_still_built_not_marked_done(tmp_path):
    """Keeping the file must not turn "extend src/util.py" into a no-op:
    `replan` marks a pending task done when its file already has a body,
    and a kept file always has one."""
    _seed(tmp_path)
    new = '```python\ndef new():\n    """New."""\n    return 1\n```'
    extended = ('```python\ndef util(x):\n    """Double."""\n'
                '    return x * 2\n\n\ndef triple(x):\n    """Triple."""\n'
                '    return x * 3\n```')
    # util.py SECOND, so it is still pending when the first replan runs.
    plan = "src/new.py — a new thing\nsrc/util.py — extend the utilities\n"
    host = _host(tmp_path, [plan, SKELETON, new, extended])
    session = Session(host, config=SessionConfig(
        attempts=1, review_after_build=False))
    session.run(REQUEST)
    assert sorted(o.path for o in session.outcomes) == \
        ["src/new.py", "src/util.py"], session.report()
    assert "def triple" in (tmp_path / "src" / "util.py").read_text()
