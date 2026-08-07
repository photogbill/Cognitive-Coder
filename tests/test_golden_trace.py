# SPDX-License-Identifier: Apache-2.0
"""M53 — Appendix E as an executed fixture, not an admired one.

The spec's worked session is committed as `fixtures/worked_session.jsonl` and
**shape-diffed** against a real run on a `ScriptedLLM`: event names and
ordering, not timestamps or durations. The reasoning in §9 is exact — *a spec
example nobody executes drifts; a fixture cannot.*

Shape-diffing rather than byte-diffing is the whole design of this test. The
things that legitimately vary between runs (a timestamp, a token count, a
duration) must not fail it, and the things that must not vary (which events
happen, in what order, against which file) must.

The scripted trace deliberately includes the awkward parts of Appendix E:
a first attempt that fails its tests, a repair that succeeds, and a
generation truncated at `max_tokens` that is CONTINUED rather than
regenerated.
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
)
from cognitive_coder.types import Completion  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "worked_session.jsonl"

# Appendix E's request, verbatim.
REQUEST = ("A CLI that reads a CSV of sensor readings and prints the mean, "
           "min and max per column, with tests.")

REPLIES = [
    # [plan] 3 files proposed
    "src/readings.py — load and validate the CSV\n"
    "src/stats.py — mean/min/max per column\n"
    "src/cli.py — argument parsing and output\n",

    # [build 1/3] src/readings.py, attempt 1 — treats the header as data,
    # via a constant it never defined. Parses cleanly; fails when run, with
    # a LOCATED error, which is the shape of feedback §6.2 exists to
    # produce.
    '```python\nimport csv\n\n\n'
    'HEADER_ROWS = header_row_count\n\n\n'
    'def load_readings(path):\n'
    '    """Load the CSV, returning a list of rows."""\n'
    '    with open(path, newline="") as fh:\n'
    '        return list(csv.reader(fh))[HEADER_ROWS:]\n```',

    # attempt 2 — fed back the located NameError; the header is now handled
    # by DictReader instead of by an invented constant
    '```python\nimport csv\n\n\n'
    'def load_readings(path):\n'
    '    """Load the CSV, returning a list of row dicts."""\n'
    '    with open(path, newline="") as fh:\n'
    '        return list(csv.DictReader(fh))\n```',

    # [build 2/3] src/stats.py — codemap injected the readings signature
    '```python\nfrom src.readings import load_readings\n\n\n'
    'def summarise(path):\n'
    '    """Mean, min and max per column."""\n'
    '    rows = load_readings(path)\n'
    '    return {"rows": len(rows)}\n```',

    # [build 3/3] src/cli.py — TRUNCATED at max_tokens…
    Completion(text='import sys\n\nfrom src.stats import summarise\n\n\n'
                    'def main(argv=None):\n'
                    '    """Parse arguments and print the summary."""\n'
                    '    argv = argv or sys.argv[1:]\n'
                    '    if not argv:\n'
                    '        print("usage: cli.py FILE")\n'
                    '        return 2\n'
                    '    print(summar',
               finish_reason="length", model="scripted-devstral"),
    # …and CONTINUED from the tail, not regenerated (D1, M32)
    Completion(text='ise(argv[0]))\n    return 0\n',
               finish_reason="stop", model="scripted-devstral"),

    # [review] ONE structured pass covering security AND performance (G.5),
    # run after everything builds and its tests pass (§4.3). Appendix E's
    # trace ends here, with the Recommendation Document.
    '{"security": [], "performance": [{"severity": "low", "title": '
    '"the whole file is read into memory", "detail": "fine at this size; a '
    'limit for large inputs", "line": 6, "fix": "stream the rows if the '
    'files grow"}], "overall": "small, clear, and does what it says"}',
    '{"security": [], "performance": [], "overall": "nothing to add"}',
    '{"security": [], "performance": [], "overall": "nothing to add"}',
]


def _run(tmp_path) -> Session:
    host = Host(llm=ScriptedLLM(REPLIES, name="scripted-devstral",
                                supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(attempts=2,
                                                 skeleton_first=True))
    session.run(REQUEST)
    return session


def shape(rows) -> list[dict]:
    """The parts of a journal that MUST be stable, and only those.

    Timestamps, token counts, durations and hashes vary per run by design.
    Event names, ordering, task attribution and attempt numbers do not, and
    those are what a regression would break.
    """
    out = []
    for row in rows:
        entry = {"event": row.get("event")}
        if row.get("task"):
            entry["task"] = row["task"]
        if row.get("attempt"):
            entry["attempt"] = row["attempt"]
        out.append(entry)
    return out


def test_the_worked_session_runs_end_to_end(tmp_path):
    """Appendix E's acceptance: unattended, no human intervening."""
    session = _run(tmp_path)
    assert [o.ok for o in session.outcomes] == [True, True, True], \
        session.report()
    for name in ("src/readings.py", "src/stats.py", "src/cli.py"):
        assert (tmp_path / name).exists(), f"{name} was not written"


def test_the_truncated_file_was_continued_not_regenerated(tmp_path):
    """M32 — the awkward step of Appendix E, and the one most easily lost."""
    session = _run(tmp_path)
    cli = (tmp_path / "src" / "cli.py").read_text()
    assert "print(summarise(argv[0]))" in cli, cli
    assert cli.count("def main") == 1, "it regenerated instead of continuing"
    assert any(r.get("event") == "continuation"
               for r in session.journal.events())


def test_the_repaired_file_took_two_attempts(tmp_path):
    """Appendix E: attempt 1 fails its tests, attempt 2 passes."""
    session = _run(tmp_path)
    readings = next(o for o in session.outcomes
                    if o.path.endswith("readings.py"))
    assert len(readings.attempts) == 2, [a.note for a in readings.attempts]
    assert readings.ok


def test_the_journal_shape_matches_the_committed_golden_trace(tmp_path):
    """M53 — shape-diffed in CI, so the spec example cannot drift.

    Regenerate deliberately with:
        python tests/test_golden_trace.py --write
    and read the diff before committing it. A golden file updated without
    being read is a golden file that proves nothing.
    """
    if not FIXTURE.exists():
        pytest.skip("run `python tests/test_golden_trace.py --write` first")
    session = _run(tmp_path)
    expected = [json.loads(line) for line in
                FIXTURE.read_text(encoding="utf-8").splitlines() if
                line.strip()]
    actual = shape(session.journal.events())

    assert actual == expected, (
        "the journal's event sequence changed.\n"
        f"expected: {json.dumps(expected, indent=1)}\n"
        f"actual:   {json.dumps(actual, indent=1)}")


def test_the_review_stage_runs_and_writes_the_document(tmp_path):
    """Appendix E's last two lines: `[review]` and `[document]`."""
    session = _run(tmp_path)
    assert (tmp_path / "Recommendation.md").exists()
    doc = (tmp_path / "Recommendation.md").read_text(encoding="utf-8")
    assert "not independent" in doc, "M41's line is missing"
    assert any(r.get("event") == "review" for r in session.journal.events())


def test_the_session_reports_no_network_and_no_redactions(tmp_path):
    """The last line of Appendix E: `all local · no network calls`."""
    session = _run(tmp_path)
    summary = session.journal.summary()
    assert "all local" in summary
    assert "no network calls" in summary
    assert "0 redactions" in summary


def test_provenance_is_complete_for_every_generation(tmp_path):
    """C8/M7 — provider, model, prompt hash, attempt, verification, time."""
    session = _run(tmp_path)
    gens = [r for r in session.journal.events()
            if r.get("event") == "generate"]
    assert gens
    for row in gens:
        for field in ("t", "model", "prompt_sha256", "attempt", "provider"):
            assert row.get(field), f"a generate event is missing {field}"
        assert "verify" in row or row.get("data", {}).get("guard")


if __name__ == "__main__":                               # pragma: no cover
    import tempfile

    if "--write" in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            s = _run(Path(tmp))
            FIXTURE.parent.mkdir(parents=True, exist_ok=True)
            FIXTURE.write_text(
                "\n".join(json.dumps(row, sort_keys=True)
                          for row in shape(s.journal.events())) + "\n",
                encoding="utf-8")
            print(f"wrote {FIXTURE} — READ THE DIFF before committing it")
            print(s.report())
