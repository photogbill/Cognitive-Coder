# SPDX-License-Identifier: Apache-2.0
"""Orchestration, resume, budgets, and the model-swap epoch rule.

The resume test is the one that matters most: **resume is derived from the
journal plus the codemap, not from an in-memory object** (§6.13), so it must
survive a process that died — not merely one that paused. The test therefore
throws the Session away entirely and rebuilds from disk.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

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
from cognitive_coder.errors import (  # noqa: E402
    BudgetExceeded,
    NoModelLoadedError,
)
from cognitive_coder.ports import NullLLM  # noqa: E402
from cognitive_coder.types import ModelCapabilities  # noqa: E402

PLAN = ("src/alpha.py — the first thing\nsrc/beta.py — the second thing\n")
ALPHA = '```python\ndef alpha():\n    """First."""\n    return 1\n```'
BETA = '```python\ndef beta():\n    """Second."""\n    return 2\n```'


def _host(tmp_path, replies, llm=None):
    return Host(llm=llm or ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def test_a_session_plans_then_builds_each_file(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    outcomes = session.run("two small modules")
    assert [o.ok for o in outcomes] == [True, True]
    assert (tmp_path / "src" / "alpha.py").exists()


def test_the_report_reads_like_appendix_E(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    session.run("two small modules")
    report = session.report()
    for marker in ("[plan]", "[build 1/2]", "[codemap]", "[journal]"):
        assert marker in report, report


def test_caveats_are_surfaced_in_the_report_not_buried(tmp_path):
    """C4 — a suite of zero tests LOOKS like success and is not."""
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    session.run("two small modules")
    assert "CAVEAT" in session.report()


# --------------------------------------------------------------------------
# resume (§6.13)
# --------------------------------------------------------------------------

def test_resume_is_derived_from_the_journal_on_disk(tmp_path):
    """It must survive a CRASH, not merely a pause — so the object that
    would have held the state is thrown away before resuming."""
    host = _host(tmp_path, [PLAN, ALPHA])       # runs out after alpha
    session = Session(host, config=SessionConfig(attempts=1))
    session_id = session.id
    with pytest.raises(AssertionError):         # ScriptedLLM runs dry
        session.start("two small modules")
        while session.step():
            pass
    session.finish()
    del session                                  # the process "died"

    revived_host = _host(tmp_path, [BETA])
    revived = Session.resume(revived_host, session_id)
    assert revived.plan is not None
    remaining = [t.path for t in revived.plan.tasks if t.status == "pending"]
    assert "src/alpha.py" not in remaining, "it would redo finished work"
    assert "src/beta.py" in remaining


def _crash_after_first_file(tmp_path, plan, request, first):
    host = _host(tmp_path, [plan, first])
    session = Session(host, config=SessionConfig(attempts=1))
    with pytest.raises(AssertionError):         # ScriptedLLM runs dry
        session.start(request)
        while session.step():
            pass
    return session


def _shape(plan):
    return {t.path: (t.purpose, t.persona, t.lang, t.test_path, t.atomic)
            for t in plan.tasks}


def test_resume_rebuilds_the_plan_it_was_given(tmp_path):
    """The `plan` event recorded paths only, so resume rebuilt every task as
    "(resumed) part of: …", an engineer, in the session's language, paired
    with a test of its own: a test file lost its tester, `beta.js` became
    Python, and a test was told to have tests."""
    plan = ("src/alpha.py — parse the header line of a CSV\n"
            "src/beta.js — render the table in the browser\n")
    request = "Build src/alpha.py and src/beta.js. Tests: tests/test_alpha.py"
    session = _crash_after_first_file(tmp_path, plan, request, ALPHA)
    before = _shape(session.plan)
    assert before["tests/test_alpha.py"][1] == "tester", "fixture changed"
    session_id = session.id
    del session

    revived = Session.resume(_host(tmp_path, []), session_id)
    assert _shape(revived.plan) == before


def test_a_kept_file_is_still_kept_after_resume(tmp_path):
    """Item 7's kept-as-found set must survive a crash, or the first replan
    after resume marks the operator's file "done" because it has a body."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.py").write_text("def util(x):\n    return x\n")
    session = _crash_after_first_file(
        tmp_path, "src/new.py — new\nsrc/util.py — extend it\n",
        "add src/new.py and extend src/util.py", ALPHA)
    session_id = session.id
    del session
    revived = Session.resume(_host(tmp_path, []), session_id)
    assert revived.planner.kept_as_found == {"src/util.py"}


def test_a_preview_is_not_a_session_to_resume(tmp_path):
    """`preview()` plans and stops. It wrote a journal like any session, so
    `previous_sessions()` listed it and `resume()` would "resume" a plan
    that nothing was ever built from."""
    from cognitive_coder.errors import CognitiveCoderError
    host = _host(tmp_path, [PLAN])
    session = Session(host)
    session.preview("two small modules")
    assert session.id not in Session.previous_sessions(host)
    with pytest.raises(CognitiveCoderError) as exc:
        Session.resume(_host(tmp_path, []), session.id)
    assert "preview" in str(exc.value)
    assert "Traceback" not in str(exc.value)


def test_previous_sessions_are_listable(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    session.run("two small modules")
    assert session.id in Session.previous_sessions(host)


def test_resuming_something_that_never_ran_says_so(tmp_path):
    host = _host(tmp_path, [])
    with pytest.raises(FileNotFoundError) as exc:
        Session.resume(host, "cc-does-not-exist")
    assert "nothing to resume" in str(exc.value)


# --------------------------------------------------------------------------
# the model, and the epoch rule (§0.1, M10, M13)
# --------------------------------------------------------------------------

def test_no_model_loaded_is_a_normal_reportable_state(tmp_path):
    """M10 — the host owns loading; "nothing is loaded" is not an exception
    until it stops the work in progress."""
    host = _host(tmp_path, [], llm=NullLLM(name=""))
    session = Session(host, config=SessionConfig(attempts=1,
                                                 skeleton_first=False))
    session.start("anything")
    with pytest.raises(NoModelLoadedError) as exc:
        session.step()
    assert "No model is loaded" in str(exc.value)
    assert "Load a model in the host" in str(exc.value)


def test_a_model_change_is_treated_as_an_epoch_boundary(tmp_path):
    """§0.1 consequence 2 — the KV cache died with the old model, so the
    cached prefix is rebuilt and the change is journaled."""
    class Swapping:
        """A host that swaps models between calls, as ATK's button does."""

        def __init__(self):
            self.name = "devstral-small-2-24b"
            self.replies = [PLAN, ALPHA, BETA]

        def capabilities(self):
            return ModelCapabilities(name=self.name, family="mistral",
                                     context_tokens=16384,
                                     supports_tools=False)

        def complete(self, messages, **kw):
            from cognitive_coder.types import Completion
            text = self.replies.pop(0) if self.replies else ""
            return Completion(text=text, model=self.name)

        def stream(self, messages, **kw):
            yield ""

        def count_tokens(self, text):
            return max(1, len(text or "") // 4)

    llm = Swapping()
    host = _host(tmp_path, [], llm=llm)
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    epoch_before = session.codemap.store.epoch

    llm.name = "magistral-small"          # the operator pressed the button
    session.step()

    assert session.codemap.store.epoch > epoch_before, (
        "a model change must invalidate the cached prompt prefix")
    warnings = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("loaded model changed" in m for m in warnings), warnings
    assert any(r.get("event") == "epoch"
               for r in session.journal.events())


def test_the_core_contains_no_model_swap_logic():
    """M10 — the core asks what is loaded; it never changes it."""
    core = Path(__file__).resolve().parent.parent / "cognitive_coder"
    banned = ("load_model", "unload_model", "switch_model", "swap_model")
    offenders = []
    for path in core.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in banned:
            if f"def {name}" in text or f".{name}(" in text:
                offenders.append(f"{path.name}: {name}")
    assert not offenders, offenders


# --------------------------------------------------------------------------
# budgets (F11) and cancellation (§5.2)
# --------------------------------------------------------------------------

def test_the_wall_clock_budget_stops_cleanly_and_says_what_was_achieved(
        tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1,
                                                 wall_clock_s=0.0001))
    session.start("two small modules")
    import time
    time.sleep(0.01)
    with pytest.raises(BudgetExceeded) as exc:
        session.step()
    assert "budget" in str(exc.value)
    assert "What was finished" in str(exc.value) or "nothing" in str(exc.value)


def test_cancelling_ends_the_session_with_resumable_state(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    session.cancel()
    session.run()
    events = [r.get("event") for r in session.journal.events()]
    assert "cancel" in events
    assert "session_end" in events


def test_a_git_repository_earns_one_warning_and_no_git_command(tmp_path):
    """§6.5b — say so once; do not refuse, do not commit for them."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    warnings = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("never runs git" in m for m in warnings), warnings


class _UndecodableExec(SubprocessExec):
    """An ExecPort whose child printed bytes that are not UTF-8 — a real
    Port failure, and not one of the engine's own error types."""

    def run(self, argv, **kw):
        raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1,
                                 "invalid start byte")


def test_a_port_failure_reaches_the_host_as_a_sentence(tmp_path):
    """C6: every failure that reaches a human is a sentence; the traceback
    goes to the journal. A UnicodeDecodeError from a child's output escaped
    `Session.run` as a raw traceback, and `CognitiveCoderError.wrap` — the
    mechanism built for exactly this — was never called."""
    from cognitive_coder.errors import CognitiveCoderError
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    host.exec = _UndecodableExec()
    session = Session(host, config=SessionConfig(attempts=1))
    with pytest.raises(CognitiveCoderError) as exc:
        session.run("two small modules")
    sentence = str(exc.value)
    assert "Traceback" not in sentence
    assert "UnicodeDecodeError" not in sentence, "a type name is not a " \
        "sentence"
    assert session.journal.path in sentence
    errors = [(m, d) for k, m, d in host.events.events if k == "error"]
    assert errors and errors[-1][0] == sentence
    assert errors[-1][1]["journal"] == session.journal.path
    logged = [e for e in session.journal.events() if e["event"] == "error"]
    assert logged, "the journal has no error event"
    detail = logged[-1]["data"]["detail"]
    assert "Traceback" in detail and "UnicodeDecodeError" in detail
    assert session.journal.events()[-1]["event"] == "session_end"


def test_the_engine_never_shells_out_to_git():
    """M27 — checked against the AST, because it is a promise.

    Looks for `git` appearing as the FIRST element of a list passed to
    `run`/`which`/`Popen` — i.e. as a command. A dictionary key called
    "git" in an event payload is not a command, and a text search cannot
    tell the two apart.
    """
    import ast as _ast
    core = Path(__file__).resolve().parent.parent / "cognitive_coder"
    offenders = []
    for path in core.rglob("*.py"):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, (_ast.List, _ast.Tuple)) and arg.elts:
                    head = arg.elts[0]
                    if isinstance(head, _ast.Constant) and \
                            str(head.value).lower() in ("git", "git.exe"):
                        offenders.append(f"{path.name}:{node.lineno}")
                if isinstance(arg, _ast.Constant) and \
                        str(arg.value).lower() in ("git", "git.exe"):
                    fn = getattr(node.func, "attr", "") or \
                        getattr(node.func, "id", "")
                    if fn in ("which", "run", "Popen", "call",
                              "check_output"):
                        offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_a_context_resize_is_an_epoch_boundary_too(tmp_path):
    """The same model reloaded with a larger n_ctx is a new KV cache: the
    host reloaded it. Only a NAME change counted, so a host's "raise the
    context" button left the old prefix snapshot in force."""
    class Resizing:
        def __init__(self):
            self.ctx = 16384
            self.replies = [PLAN]

        def capabilities(self):
            return ModelCapabilities(name="devstral-small-2-24b",
                                     family="mistral",
                                     context_tokens=self.ctx,
                                     supports_tools=False)

        def complete(self, messages, **kw):
            from cognitive_coder.types import Completion
            text = self.replies.pop(0) if self.replies else ""
            return Completion(text=text, model="devstral-small-2-24b")

        def stream(self, messages, **kw):
            yield ""

        def count_tokens(self, text):
            return max(1, len(text or "") // 4)

    llm = Resizing()
    host = _host(tmp_path, [], llm=llm)
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("two small modules")
    session._capabilities(boundary="first")
    before = session.codemap.store.epoch

    session._capabilities(boundary="unchanged")
    assert session.codemap.store.epoch == before, "no change, no epoch"

    llm.ctx = 32768
    session._capabilities(boundary="resized")
    assert session.codemap.store.epoch > before
    warnings = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("32,768-token context" in m for m in warnings), warnings


def test_a_session_built_on_one_thread_runs_on_another(tmp_path):
    """ATK's panel builds the Session on the GUI thread and runs it on a
    worker. The codemap's SQLite connection belonged to the thread that
    opened it, so every codemap call from the build raised
    ProgrammingError — observed as tool calls answering "That tool call
    failed" and an index that silently stopped."""
    import threading
    host = _host(tmp_path, [PLAN, ALPHA, BETA])
    session = Session(host, config=SessionConfig(attempts=1))
    errors: list[BaseException] = []

    def work():
        try:
            session.run("two small modules")
        except BaseException as exc:                     # noqa: BLE001
            errors.append(exc)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join(120)
    assert not errors, errors
    assert session.codemap.store.files(), "nothing was indexed"
    # And the GUI side reads it afterwards, from the thread that built it.
    assert session.codemap.stats().files >= 1
