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
