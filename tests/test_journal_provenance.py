# SPDX-License-Identifier: Apache-2.0
"""Provenance fields that were silently lost or crashed the reader (C8).

Each of these was observed: `temperature=0.0` and `seed=0` vanished from the
journal because `0.0 == 0`; a row with an explicit `null` took `stats()`
down with `int(None)`; the event vocabulary drifted from what is actually
written; and two prompts differing only in their tool traffic hashed alike.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import journal as J  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem  # noqa: E402
from cognitive_coder.types import (  # noqa: E402
    JOURNAL_EVENTS,
    JournalEvent,
    Message,
    ToolCall,
)


class _Events:
    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def event(self, *args):
        self.rows.append(args)


def test_greedy_decoding_and_seed_zero_are_recorded():
    """temperature=0.0 is the most reproducible setting there is; dropping
    it makes a greedy run indistinguishable from one that never said."""
    row = json.loads(JournalEvent(t="x", event="generate", temperature=0.0,
                                  seed=0).to_json())
    assert row["temperature"] == 0.0
    assert row["seed"] == 0


def test_unset_and_empty_fields_are_still_dropped():
    row = json.loads(JournalEvent(t="x", event="plan").to_json())
    assert set(row) == {"t", "event"}, row


def test_stats_tolerate_explicit_nulls():
    fs = MemoryFileSystem({".cc_journal/s.jsonl": (
        b'{"t":"x","event":"generate","prompt_ms":null,"tokens_out":5,'
        b'"decode_ms":null,"tokens_in":null,"attempt":null}\n'
        b'{"t":"y","event":"generate","prompt_ms":"12","decode_ms":1000,'
        b'"tokens_out":20,"data":{"redactions":null}}\n')})
    stats = J.Journal(fs, "s").stats()
    assert stats["generations"] == 2
    assert stats["prompt_ms_median"] == 12
    assert stats["tokens_per_s_median"] == 20.0
    assert J.Journal(fs, "s").cache_health()


def test_events_the_core_writes_are_in_the_vocabulary():
    """`session.py` writes "blocked"; a concurrent feature writes "audit"."""
    assert "blocked" in JOURNAL_EVENTS
    assert "audit" in JOURNAL_EVENTS


def test_an_unknown_event_name_warns_once_and_is_still_written():
    fs, events = MemoryFileSystem(), _Events()
    journal = J.Journal(fs, "s", events=events)
    journal.log("blocked", task="a.py")
    assert not events.rows
    journal.log("made_up_event")
    journal.log("made_up_event")
    warnings = [r for r in events.rows if r[0] == "warning"]
    assert len(warnings) == 1 and "made_up_event" in warnings[0][1]
    assert [r["event"] for r in journal.events()] == [
        "blocked", "made_up_event", "made_up_event"]


def test_prompt_hash_sees_tool_traffic_and_images():
    base = [Message(role="user", content="go")]
    call = Message(role="assistant", content="", tool_calls=(
        ToolCall(id="c1", name="read_slice", arguments={"path": "a.py"}),))
    other = Message(role="assistant", content="", tool_calls=(
        ToolCall(id="c1", name="read_slice", arguments={"path": "b.py"}),))
    assert J.prompt_hash(base + [call]) != J.prompt_hash(base + [other])
    r1 = Message(role="tool", content="x", tool_call_id="c1")
    r2 = Message(role="tool", content="x", tool_call_id="c2")
    assert J.prompt_hash([r1]) != J.prompt_hash([r2])
    i1 = Message(role="user", content="see", images=((b"\x89PNG1",
                                                       "image/png"),))
    i2 = Message(role="user", content="see", images=((b"\x89PNG2",
                                                       "image/png"),))
    assert J.prompt_hash([i1]) != J.prompt_hash([i2])


def test_prompt_hash_is_unchanged_for_plain_messages():
    """Existing journals must keep matching: the old identity stands for
    every message that carries no tool traffic or images."""
    msgs = [Message(role="system", content="s"),
            Message(role="user", content="u")]
    assert J.prompt_hash(msgs) == hashlib.sha256(
        b"system:s\nuser:u").hexdigest()
    assert J.prompt_hash("raw") == hashlib.sha256(b"raw").hexdigest()


def _timed(fs, times, decode=500, tokens_in=4000):
    journal = J.Journal(fs, "s")
    for ms in times:
        journal.log("generate", task="a.py", attempt=1, prompt_ms=ms,
                    decode_ms=decode, tokens_out=100, tokens_in=tokens_in)
    return journal


def test_planned_full_reads_are_not_reported_as_a_broken_cache():
    """The first call and one per planned snapshot read the whole prompt.
    The verdict fired on any single slow read, so a healthy run with a
    planned rebuild was told its cache had been invalidated."""
    fs = MemoryFileSystem()
    journal = _timed(fs, [9000, 300, 280, 8800, 310, 290, 300])
    verdict = journal.cache_health(planned_rebuilds=2)
    assert "looks healthy" in verdict and "2 planned full reads" in verdict


def test_more_full_reads_than_planned_is_still_caught():
    fs = MemoryFileSystem()
    journal = _timed(fs, [9000, 300, 8700, 280, 8800, 310, 9100])
    verdict = journal.cache_health(planned_rebuilds=1)
    assert "invalidated" in verdict and "4 prompts" in verdict


def test_a_cache_broken_on_most_calls_is_not_called_steady():
    """With a median of raw times, a cache that failed on most calls made
    the median itself a from-scratch time, and the verdict said steady."""
    fs = MemoryFileSystem()
    journal = _timed(fs, [9000, 8800, 9100, 8700, 300, 8900, 9050])
    assert "invalidated" in journal.cache_health(planned_rebuilds=2)


def test_the_servers_own_count_gives_an_exact_verdict():
    from cognitive_coder.types import Completion
    fs = MemoryFileSystem()
    journal = J.Journal(fs, "s")
    for processed in (4000, 400, 350, 420, 380):
        journal.generation(
            task="a.py", attempt=1, provider="p", prompt="x",
            temperature=0.1,
            completion=Completion(text="x", tokens_in=4000, tokens_out=100,
                                  prompt_ms=300, decode_ms=900,
                                  prompt_processed=processed))
    verdict = journal.cache_health(planned_rebuilds=1)
    assert "72% of prompt tokens came from the cache" in verdict
    assert "working" in verdict
