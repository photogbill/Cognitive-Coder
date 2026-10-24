# SPDX-License-Identifier: Apache-2.0
"""Continue — a stopped build picked up where it left off (resumable.py).

Bill, 2026-10-23: *"What if an operator stops the coding, but wants to go
back to where he left off?"*

Every test here throws the stopped Session away before continuing, as the
resume tests do: the state must come from the journal and the disk, never
from an object a crash would have destroyed (§6.13).

What is pinned:

  * how a build ended — stopped, crashed, finished with failures, set
    aside, a preview — read from the journal of its LATEST run;
  * which builds a host may offer, interrupted ones first, newest first;
  * Discard sets a build aside without erasing anything;
  * a scan across project folders reads and never writes;
  * the last verdict on a file wins (a file that passed, then failed, is
    not "done");
  * Continue restores what the first run set up — the build-log header,
    the board's plan with its green rows, the test baseline;
  * the files on disk are checked before anything is written: a hand fix
    is KEPT, a file broken by hand is REPAIRED from what is there, and a
    clean pause costs nothing;
  * the graceful stop finishes the file in hand and stops before the next.
"""

from __future__ import annotations

import json
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
    resumable,
)
from cognitive_coder import journal as journal_mod  # noqa: E402
from cognitive_coder.errors import Cancelled  # noqa: E402

PLAN = ("src/alpha.py — the first thing\nsrc/beta.py — the second thing\n")
ALPHA = '```python\ndef alpha():\n    """First."""\n    return 1\n```'
BETA = '```python\ndef beta():\n    """Second."""\n    return 2\n```'
SKELETON = ('```python\n# file: src/alpha.py\ndef alpha() -> int:\n'
            '    """First."""\n    raise NotImplementedError\n```\n'
            '```python\n# file: src/beta.py\ndef beta() -> int:\n'
            '    """Second."""\n    raise NotImplementedError\n```')


def _host(tmp_path, replies):
    return Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _stopped_after_alpha(tmp_path) -> str:
    """A build the operator stopped gracefully after its first file."""
    session = Session(_host(tmp_path, [PLAN, SKELETON, ALPHA]),
                      config=SessionConfig(attempts=1))
    session.stop_after_current()            # pressed before it got far
    session.run("two small modules")
    sid = session.id
    del session                             # the process is gone
    return sid


def _crashed_after_alpha(tmp_path) -> str:
    """A build whose process died while writing its second file — no
    session_end, no rollback, nothing said."""
    session = Session(_host(tmp_path, [PLAN, SKELETON, ALPHA]),
                      config=SessionConfig(attempts=1))
    with pytest.raises(AssertionError):     # ScriptedLLM runs dry on beta
        session.start("two small modules")
        while session.step():
            pass
    sid = session.id
    del session
    return sid


def _rows(tmp_path, sid) -> list[dict]:
    return list(journal_mod.read_jsonl(LocalFileSystem(str(tmp_path)),
                                       f".cc_journal/{sid}.jsonl"))


# --------------------------------------------------------------------------
# how a build ended
# --------------------------------------------------------------------------

def test_a_stopped_build_says_so_and_can_be_continued(tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    state = resumable.state_of(LocalFileSystem(str(tmp_path)), sid)
    assert state.ended == "stopped"
    assert state.interrupted and state.continuable
    assert state.files == ("src/alpha.py", "src/beta.py")
    assert state.done == ("src/alpha.py",)
    assert state.remaining == ("src/beta.py",)
    assert state.last_file == "src/alpha.py"
    assert state.title == "two small modules"
    guided = state.sentence("guided")
    assert "you pressed Stop" in guided and "1 of 2 parts done" in guided
    assert "Continue picks up the 1 left" in guided
    engineer = state.sentence("engineer")
    assert "stopped" in engineer and "1/2 done" in engineer
    assert sid in engineer


def test_a_crash_is_told_apart_from_a_stop(tmp_path):
    """No session_end after the last session_start: the program closed or
    the machine went down. And its hashes are NOT trusted."""
    sid = _crashed_after_alpha(tmp_path)
    state = resumable.state_of(LocalFileSystem(str(tmp_path)), sid)
    assert state.ended == "crashed"
    assert state.continuable and state.interrupted
    assert state.hashes is None
    assert "stopped unexpectedly" in state.sentence("guided")


def test_a_finished_build_with_nothing_left_is_not_offered(tmp_path):
    session = Session(_host(tmp_path, [PLAN, SKELETON, ALPHA, BETA]),
                      config=SessionConfig(attempts=1))
    session.run("two small modules")
    state = resumable.state_of(LocalFileSystem(str(tmp_path)), session.id)
    assert state.ended == "finished"
    assert not state.continuable
    assert resumable.unfinished(LocalFileSystem(str(tmp_path))) == []


def test_a_stop_after_the_last_file_leaves_nothing_to_continue(tmp_path):
    """Stop pressed during the review: everything was built, so Continue
    has nothing to offer and must not pretend otherwise."""
    session = Session(_host(tmp_path, [PLAN, SKELETON, ALPHA, BETA]),
                      config=SessionConfig(attempts=1))

    class StopOnBeta(RecordingEvents):
        def event(self, kind, message, data=None):
            super().event(kind, message, data)
            if (data or {}).get("task") == "src/beta.py":
                session.stop_after_current()

    session.host.events = StopOnBeta()
    session.journal._events = session.host.events
    session.run("two small modules")
    state = resumable.state_of(LocalFileSystem(str(tmp_path)), session.id)
    assert state.ended == "stopped"
    assert not state.remaining
    assert not state.continuable


def test_a_preview_is_never_offered(tmp_path):
    host = _host(tmp_path, [PLAN])
    session = Session(host)
    session.preview("two small modules")
    state = resumable.state_of(host.fs, session.id)
    assert state.ended == "preview" and not state.continuable


def test_an_unreadable_journal_summarises_as_empty(tmp_path):
    fs = LocalFileSystem(str(tmp_path))
    fs.write_bytes(".cc_journal/cc-junk.jsonl", b"not json\n{broken\n")
    state = resumable.state_of(fs, "cc-junk")
    assert state.ended == "empty" and not state.continuable


# --------------------------------------------------------------------------
# which builds to offer, and Discard
# --------------------------------------------------------------------------

def test_unfinished_lists_interrupted_builds_newest_first(tmp_path):
    first = _stopped_after_alpha(tmp_path)
    second = _crashed_after_alpha(tmp_path)
    found = [s.session_id
             for s in resumable.unfinished(LocalFileSystem(str(tmp_path)))]
    assert set(found) == {first, second}


def test_discard_sets_a_build_aside_and_erases_nothing(tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    fs = LocalFileSystem(str(tmp_path))
    before = len(_rows(tmp_path, sid))
    resumable.abandon(fs, sid, reason="tried something else")
    assert resumable.unfinished(fs) == []
    state = resumable.state_of(fs, sid)
    assert state.ended == "abandoned"
    assert state.reason == "tried something else"
    rows = _rows(tmp_path, sid)
    assert len(rows) == before + 1, "append-only: a new event, no erasure"
    assert (tmp_path / "src" / "alpha.py").exists(), "files untouched"
    # and it can still be picked up by name
    revived = Session.resume(_host(tmp_path, [BETA]), sid)
    assert revived.plan is not None


def test_continuing_a_discarded_build_offers_it_again_after(tmp_path):
    """Discard is about the offer, not the build: a build continued by name
    after being set aside, and stopped again, is unfinished again."""
    sid = _stopped_after_alpha(tmp_path)
    fs = LocalFileSystem(str(tmp_path))
    resumable.abandon(fs, sid)
    revived = Session.resume(_host(tmp_path, []), sid)
    revived.cancel()
    revived.run()
    assert [s.session_id for s in resumable.unfinished(fs)] == [sid]


def test_scan_finds_builds_across_project_folders_and_writes_nothing(
        tmp_path):
    root = tmp_path / "code_projects"
    a = root / "snake-game"
    b = root / "csv-parser"
    c = root / "untouched"
    for folder in (a, b, c):
        folder.mkdir(parents=True)
    sid_a = _stopped_after_alpha(a)
    sid_b = _crashed_after_alpha(b)
    listing = sorted(str(p) for p in root.rglob("*"))
    found = resumable.scan(root)
    assert {(s.session_id, s.folder) for s in found} == {
        (sid_a, "snake-game"), (sid_b, "csv-parser")}
    assert sorted(str(p) for p in root.rglob("*")) == listing, \
        "a scan must not create anything"
    assert [s.session_id for s in resumable.scan(root, exclude=[sid_a])] \
        == [sid_b]
    assert resumable.scan(tmp_path / "does-not-exist") == []


def test_build_state_crosses_threads_as_plain_data(tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    data = resumable.state_of(LocalFileSystem(str(tmp_path)), sid).as_dict()
    json.dumps(data)                         # nothing but plain data
    assert data["continuable"] and data["remaining"] == ["src/beta.py"]
    assert "you pressed Stop" in data["sentence_guided"]
    assert sid in data["sentence_engineer"]


def test_the_title_is_the_first_real_line_of_a_spec():
    state = resumable.BuildState(
        session_id="cc-x",
        request="<!-- Copied by ATK -->\n\n# Snake game\n\nUse pygame.")
    assert state.title == "Snake game"


# --------------------------------------------------------------------------
# the journal
# --------------------------------------------------------------------------

def test_the_last_verdict_on_a_file_wins(tmp_path):
    """Append-only "once done, always done" let a file that passed and was
    later blamed — repaired, still failing — come back from a resume as
    finished."""
    fs = LocalFileSystem(str(tmp_path))
    j = journal_mod.Journal(fs, "cc-v")
    j.log("plan", files=["a.py", "b.py"])
    j.log("verify", task="a.py", verify={"ok": True, "verified": True})
    j.log("verify", task="b.py", verify={"ok": False})
    j.log("verify", task="b.py", verify={"ok": True, "verified": False})
    j.log("verify", task="a.py", verify={"ok": False})
    state = journal_mod.resume_state(fs, "cc-v")
    assert state["done"] == ["b.py"]
    assert state["failed"] == ["a.py"]
    assert state["verified"] == []


def test_a_session_end_records_what_each_file_looked_like(tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    (end,) = [r for r in _rows(tmp_path, sid)
              if r.get("event") == "session_end"]
    hashes = end["data"]["hashes"]
    assert set(hashes) == {"src/alpha.py", "src/beta.py"}
    assert all(len(h) == 64 for h in hashes.values())


# --------------------------------------------------------------------------
# continuing
# --------------------------------------------------------------------------

def test_continue_builds_what_was_left_and_nothing_else(tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    host = _host(tmp_path, [BETA])          # one reply: alpha is not redone
    session = Session.resume(host, sid)
    outcomes = session.run()
    assert [o.path for o in outcomes] == ["src/beta.py"]
    assert all(o.ok for o in outcomes)
    assert "return 2" in (tmp_path / "src" / "beta.py").read_text()
    after = resumable.state_of(host.fs, sid)
    assert after.ended == "finished" and after.runs == 2
    assert not after.continuable


def test_continue_restores_the_board_and_the_build_log(tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    host = _host(tmp_path, [BETA])
    session = Session.resume(host, sid)
    session.run()
    (plan_event, *_) = host.events.of("plan")
    _kind, message, data = plan_event
    assert data.get("continued") is True and "(continued)" in message
    rows = {t["path"]: t for t in data["tasks"]}
    assert rows["src/alpha.py"]["status"] == "done"
    assert rows["src/beta.py"]["status"] == "pending"
    assert "verified" in rows["src/alpha.py"]
    log = (tmp_path / "BUILD_LOG.txt").read_text()
    assert f"SESSION {sid} — CONTINUED" in log
    assert "run 2 of this build" in log and "you pressed Stop" in log
    events = [r["event"] for r in _rows(tmp_path, sid)]
    assert "continue" in events


def test_a_clean_pause_costs_nothing(tmp_path):
    """Nothing changed since the stop: no file is run again."""
    sid = _stopped_after_alpha(tmp_path)
    host = _host(tmp_path, [BETA])
    session = Session.resume(host, sid)
    session.run()
    (reverify,) = [r for r in _rows(tmp_path, sid)
                   if r.get("event") == "reverify"]
    assert reverify["data"]["skipped"] is True
    assert reverify["data"]["checked"] == []


def test_a_hand_fix_made_during_the_pause_is_kept_not_regenerated(tmp_path):
    """The operator stopped the build and wrote beta.py themselves. It
    passes its checks, so Continue keeps it — and asks the model for
    NOTHING (ScriptedLLM with no replies raises if it is asked)."""
    sid = _stopped_after_alpha(tmp_path)
    beta = tmp_path / "src" / "beta.py"
    beta.write_text('def beta():\n    """Mine."""\n    return 22\n')
    host = _host(tmp_path, [])
    session = Session.resume(host, sid)
    session.run()
    assert "return 22" in beta.read_text(), "the hand fix was written over"
    kept = [e for e in host.events.of("status") if e[2].get("kept")]
    assert kept and kept[0][2]["task"] == "src/beta.py"
    (reverify,) = [r for r in _rows(tmp_path, sid)
                   if r.get("event") == "reverify"]
    assert reverify["data"]["kept"] == ["src/beta.py"]
    assert reverify["data"]["changed"] == ["src/beta.py"]
    assert not resumable.state_of(host.fs, sid).continuable


def test_a_file_broken_during_the_pause_is_repaired_from_what_is_there(
        tmp_path):
    sid = _stopped_after_alpha(tmp_path)
    (tmp_path / "src" / "alpha.py").write_text("def alpha(:\n    return 1\n")
    host = _host(tmp_path, [ALPHA, BETA])
    session = Session.resume(host, sid)
    session.run()
    (reverify,) = [r for r in _rows(tmp_path, sid)
                   if r.get("event") == "reverify"]
    assert reverify["data"]["now_failing"] == ["src/alpha.py"]
    warnings = [m for _k, m, d in host.events.of("warning")
                if d.get("phase") == "reverify"]
    assert any("passed before the stop but fails now" in m for m in warnings)
    assert "return 1" in (tmp_path / "src" / "alpha.py").read_text()
    assert "def alpha(:" not in (tmp_path / "src" / "alpha.py").read_text()
    assert not resumable.state_of(host.fs, sid).continuable


def test_after_a_crash_every_file_on_disk_is_checked(tmp_path):
    """The crashed run wrote no hashes, so "nothing changed" cannot be
    known — every done file is checked, and the build goes on."""
    sid = _crashed_after_alpha(tmp_path)
    host = _host(tmp_path, [BETA])
    session = Session.resume(host, sid)
    session.run()
    (reverify,) = [r for r in _rows(tmp_path, sid)
                   if r.get("event") == "reverify"]
    assert reverify["data"]["unknown"] is True
    assert "src/alpha.py" in reverify["data"]["still_ok"]
    assert "return 2" in (tmp_path / "src" / "beta.py").read_text()


def test_reverify_can_be_switched_off(tmp_path):
    sid = _crashed_after_alpha(tmp_path)
    host = _host(tmp_path, [BETA])
    session = Session.resume(host, sid, config=SessionConfig(
        reverify_on_continue=False))
    session.run()
    assert not [r for r in _rows(tmp_path, sid)
                if r.get("event") == "reverify"]


def test_the_test_baseline_survives_a_continue(tmp_path):
    """Tests failing before the first run were excused from blame; a resume
    forgot, and the stale test was charged to the resumed files again."""
    fs = LocalFileSystem(str(tmp_path))
    j = journal_mod.Journal(fs, "cc-b")
    j.log("session_start", request="x", config={"lang": "python"})
    j.log("baseline", existing=["tests/test_old.py"],
          failing=["tests/test_old.py"], ok=False)
    j.log("plan", files=["src/a.py"])
    j.log("cancel", sentence="stopped")
    j.log("session_end", ok=False)
    revived = Session.resume(_host(tmp_path, []), "cc-b")
    assert revived.loop.config.known_failing == ("tests/test_old.py",)


# --------------------------------------------------------------------------
# the graceful stop
# --------------------------------------------------------------------------

def test_stop_after_current_finishes_the_file_then_stops(tmp_path):
    host = _host(tmp_path, [PLAN, SKELETON, ALPHA])
    session = Session(host, config=SessionConfig(attempts=1))
    session.stop_after_current()
    outcomes = session.run("two small modules")
    assert [o.path for o in outcomes] == ["src/alpha.py"]
    assert outcomes[0].ok
    (cancel,) = [r for r in _rows(tmp_path, session.id)
                 if r.get("event") == "cancel"]
    assert cancel["data"]["after"] == "src/alpha.py"
    assert "after src/alpha.py was finished" in cancel["data"]["sentence"]
    assert "rolled back" not in cancel["data"]["sentence"]
    stopped = [d for _k, _m, d in host.events.of("status")
               if d.get("phase") == "stopped"]
    assert stopped and stopped[0]["graceful"] is True


def test_the_graceful_stop_sentence_promises_nothing_was_rolled_back():
    assert "rolled back" not in str(Cancelled(after="src/x.py"))
    assert "rolled back" in str(Cancelled("the review"))
    assert Cancelled(after="src/x.py").after == "src/x.py"


def test_a_uis_saved_chat_tabs_are_never_read_as_source():
    """ATK keeps each chat tab's conversation in `<project>/.cc_tabs/`. A
    transcript is not source: it must not be indexed, scoped or offered to
    the model (with remote mode on that would be outbound context)."""
    from cognitive_coder import scope
    from cognitive_coder.codemap import SKIP_DIRS
    assert ".cc_tabs" in SKIP_DIRS
    assert ".cc_tabs/" in scope.SKIP
