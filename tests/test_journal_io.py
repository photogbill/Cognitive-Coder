# SPDX-License-Identifier: Apache-2.0
"""What writing the journal and BUILD_LOG.txt costs, and when it fails.

Observed: every journal event READ the whole file and rewrote it — 500
events of a 172 KB journal cost 43 MB read and 43 MB written — and
`SessionLog.block()` rewrote the file once per LINE. And only the first
write failure warned, so a disk filling mid-session was silent.

`FileSystemPort` has no append primitive, so an optional `append_bytes`
is detected and used; without it the old read-modify-write remains.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import journal as J  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem  # noqa: E402


class CountingFS(MemoryFileSystem):
    """Counts every byte read, written and appended, and every call."""

    def __init__(self, *a, append=True, **k):
        super().__init__(*a, **k)
        self.read_total = self.write_total = 0
        self.calls = 0
        if not append:
            self.append_bytes = None     # an fs without the optional method

    def read_bytes(self, path):
        data = super().read_bytes(path)
        self.read_total += len(data)
        return data

    def write_bytes(self, path, content):
        self.calls += 1
        self.write_total += len(content)
        super().write_bytes(path, content)

    def append_bytes(self, path, content):
        self.calls += 1
        self.write_total += len(content)
        key = self._key(path)
        self.files[key] = self.files.get(key, b"") + bytes(content)


class _Events:
    def __init__(self):
        self.rows = []

    def event(self, *args):
        self.rows.append(args)


def test_journal_appends_cost_what_they_add():
    fs = CountingFS()
    journal = J.Journal(fs, "s")
    for i in range(500):
        journal.log("verify", task=f"src/f{i}.py",
                    verify={"ok": True, "caveats": ["x" * 200]})
    size = len(fs.files[".cc_journal/s.jsonl"])
    assert fs.write_total == size, (fs.write_total, size)
    assert fs.read_total == 0
    assert len(journal.events()) == 500


def test_journal_without_append_still_works():
    fs = CountingFS(append=False)
    journal = J.Journal(fs, "s")
    for i in range(3):
        journal.log("verify", task=f"f{i}")
    assert [r["task"] for r in journal.events()] == ["f0", "f1", "f2"]


def test_a_log_block_is_one_write_not_one_per_line():
    fs = CountingFS()
    log = J.SessionLog(fs, "s")
    log.start("req", "m", {})
    before = fs.calls
    log.block("output", "\n".join(f"line {i}" for i in range(200)))
    assert fs.calls - before == 1
    assert fs.write_total == len(fs.files["BUILD_LOG.txt"])
    assert "line 199" in fs.files["BUILD_LOG.txt"].decode()


def test_start_and_phases_write_once_each():
    from cognitive_coder.types import PhaseResult, ProcResult, RunResult
    fs = CountingFS()
    log = J.SessionLog(fs, "s")
    log.start("a request", "model", {"a": 1})
    assert fs.calls == 1
    result = RunResult(ok=True, phases=(
        PhaseResult(name="test", argv=("pytest",), ok=True,
                    proc=ProcResult(exit_code=0, stdout="1 passed")),))
    log.phases("a.py", 1, result)
    assert fs.calls == 2
    assert "1 passed" in fs.files["BUILD_LOG.txt"].decode()


class FlakyFS(MemoryFileSystem):
    fail = False

    def write_bytes(self, path, content):
        if self.fail:
            raise OSError("disk full")
        super().write_bytes(path, content)


def test_a_journal_that_stops_being_writable_mid_session_warns():
    """The first write succeeded, so the old once-only warning never
    fired: a disk filling mid-session lost the rest of the record in
    silence."""
    fs, events = FlakyFS(), _Events()
    journal = J.Journal(fs, "s", events=events)
    journal.log("session_start")
    fs.fail = True
    for _ in range(3):
        journal.log("verify", task="a")
    warnings = [r for r in events.rows if r[0] == "warning"]
    assert len(warnings) == 1, warnings
    assert "could not be written" in warnings[0][1]


def test_repeated_failures_warn_again_but_rate_limited():
    fs, events = FlakyFS(), _Events()
    fs.fail = True
    journal = J.Journal(fs, "s", events=events)
    for _ in range(J.WARN_EVERY * 3):
        journal.log("verify", task="a")
    warnings = [r for r in events.rows if r[0] == "warning"]
    assert 2 <= len(warnings) <= 4, len(warnings)
    # recovering and failing again warns at once
    fs.fail = False
    journal.log("verify", task="b")
    fs.fail = True
    n = len(events.rows)
    journal.log("verify", task="c")
    assert len(events.rows) == n + 1


def test_the_build_log_warns_too_when_given_events():
    fs, events = FlakyFS(), _Events()
    log = J.SessionLog(fs, "s", events=events)
    log.line("ok")
    fs.fail = True
    log.line("lost")
    assert any(r[0] == "warning" and "BUILD_LOG" in r[1]
               for r in events.rows)


def test_the_session_hands_its_events_to_the_build_log():
    """SessionLog could warn, but Session constructed it without `events`,
    so a BUILD_LOG.txt that stopped being writable was counted and never
    reported — and it is the file an operator actually reads."""
    from cognitive_coder import Host, RecordingEvents, ScriptedLLM, Session
    fs, events = FlakyFS(), RecordingEvents()
    session = Session(Host(llm=ScriptedLLM([]), fs=fs, events=events))
    session.log.line("ok")
    fs.fail = True
    session.log.line("lost")
    assert any(kind == "warning" and "BUILD_LOG" in message
               for kind, message, _data in events.events)
