# SPDX-License-Identifier: Apache-2.0
"""M52 — byte-identical prompt prefixes, per (persona, epoch, target).

**This test is the only thing standing between a working cache design and a
silently 20×-slower one six months from now.**

llama.cpp caches the KV state of a prompt prefix: if the beginning of the
prompt is byte-identical to the previous call, those tokens are not
reprocessed. At local speeds that is the difference between 3 seconds and
minutes, on every call. Break it and nothing fails — everything just gets
slowly, inexplicably worse, and the only visible trace is `prompt_ms` in the
journal (G.7.5, M55).

So the prefix is asserted byte-identical, and the specific ways it usually
gets broken are each given their own test: a timestamp, a session id, a
randomised ordering, a "files changed" note that belongs in the tail.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import personas  # noqa: E402
from cognitive_coder.codemap import (
    CodeMap,  # noqa: E402
    zoom,  # noqa: E402
)
from cognitive_coder.personas import PromptBuilder  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem, MemoryStorage  # noqa: E402


def _codemap():
    files = {
        "src/readings.py": b'"""Load."""\n\n\ndef load_readings(path):\n'
                           b'    """Load rows."""\n    return []\n',
        "src/stats.py": b'"""Stats."""\nfrom src.readings import '
                        b'load_readings\n\n\ndef summarise(path):\n'
                        b'    """Summarise."""\n    return {}\n',
        "src/cli.py": b'"""CLI."""\n\n\ndef main():\n    """Entry."""\n'
                      b'    return 0\n',
    }
    cm = CodeMap(MemoryFileSystem(files), MemoryStorage())
    cm.index_project()
    return cm


def test_the_prefix_is_byte_identical_across_calls():
    """The core assertion: same persona, same epoch, same target, same bytes."""
    cm = _codemap()
    builder = PromptBuilder(model_system_prompt="You are a coding model.",
                            conventions="Comment the why, not the what.")
    arch = cm.prefix_block("src/stats.py")

    first = builder.prefix_for(personas.ENGINEER, architecture=arch,
                               epoch=cm.store.epoch)
    second = builder.prefix_for(personas.ENGINEER, architecture=arch,
                                epoch=cm.store.epoch)
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")


def test_the_prefix_does_not_change_when_the_target_file_changes():
    """G.7.1 — the architecture block is deliberately not target-ordered.

    Sorting the architecture by relevance to the current file would discard
    30k tokens of cached work to save a few hundred, every time the target
    moved. The high-resolution part is where relevance belongs.
    """
    cm = _codemap()
    builder = PromptBuilder(conventions="house style")
    a = builder.prefix_for(personas.ENGINEER,
                           architecture=cm.prefix_block("src/stats.py"))
    b = builder.prefix_for(personas.ENGINEER,
                           architecture=cm.prefix_block("src/cli.py"))
    assert a == b


def test_the_prefix_contains_nothing_that_varies_with_time():
    """The classic cache killer: one varying token at position 40.

    A timestamp, a session id, a random seed, a duration — any of them in
    the prefix silently discards the whole cached prompt on every call.
    """
    cm = _codemap()
    builder = PromptBuilder(model_system_prompt="System.",
                            conventions="Conventions.")
    prefix = builder.prefix_for(personas.ENGINEER,
                                architecture=cm.prefix_block("src/stats.py"))
    # ISO dates, clock times, epoch seconds, uuids and session ids.
    patterns = [
        r"\d{4}-\d{2}-\d{2}",            # a date
        r"\d{2}:\d{2}:\d{2}",            # a clock time
        r"\b1[6-9]\d{8}\b",              # a unix timestamp
        r"\bcc-\d{8}-",                  # our own session id format
        r"[0-9a-f]{8}-[0-9a-f]{4}-",     # a uuid
    ]
    for pattern in patterns:
        assert not re.search(pattern, prefix), (
            f"the cached prefix contains something matching {pattern!r} — "
            f"that invalidates the KV cache on every call")


def test_the_staleness_note_is_in_the_TAIL_and_never_the_prefix():
    """G.7.3 — a note in the prefix would invalidate the cache it describes.

    This is the subtle one, and it is the mistake a careful person makes:
    telling the model the summary is stale seems like it belongs beside the
    summary. It belongs in the tail, which is reprocessed anyway, so putting
    it there is free.
    """
    cm = _codemap()
    cm.store.bump_epoch("test")
    cm.store.note_change("src/stats.py")
    note = zoom.staleness_note(cm.store)
    assert "changed since" in note

    prefix = cm.prefix_block("src/stats.py")
    assert "changed since" not in prefix
    assert note not in prefix
    assert note in cm.tail_blocks("src/stats.py")


def test_the_prefix_changes_when_the_epoch_changes():
    """The one thing that MAY change it — and should, when it does.

    A new epoch means the architecture was deliberately rebuilt, so
    discarding the cache is the correct trade rather than an accident.
    (This test used to assert the prefix changed after `index_file` with NO
    epoch bump — pinning the defect: the "epoch-scoped" block was live.)
    """
    cm = _codemap()
    cm.store.bump_epoch("start")
    before = cm.prefix_block("src/stats.py")
    cm.index_file("src/extra.py", "def added():\n    pass\n", force=True)
    cm.store.bump_epoch("test")
    after = cm.prefix_block("src/stats.py")
    assert before != after, ("a new file should change the architecture "
                            "block at the next epoch — otherwise the model "
                            "never learns it exists")
    assert "src/extra.py" in after


def test_the_prefix_does_not_change_within_an_epoch():
    """G.7.2: the injected architecture updates BY EPOCH. Every patch
    re-indexes its file, and the prefix bytes used to change with it while
    `should_bump_epoch` said no — a cache miss on every write."""
    cm = _codemap()
    cm.store.bump_epoch("start")
    before = cm.prefix_block("src/stats.py")
    cm.index_file("src/extra.py", "def added():\n    pass\n", force=True)
    cm.index_file("src/cli.py", "def main():\n    pass\n\n\n"
                  "def more():\n    pass\n", force=True)
    assert zoom.should_bump_epoch(cm.store) == (False, "")
    assert cm.prefix_block("src/stats.py") == before
    # The query interface stays live all the same (M30).
    assert cm.store.symbols_in("src/extra.py")


def test_epoch_bumps_only_for_the_reasons_G7_lists():
    """A closed list, because bumping costs 30–60 s of reprocessing."""
    cm = _codemap()
    cm.store.bump_epoch("start")            # clears changed_since_epoch
    assert zoom.should_bump_epoch(cm.store) == (False, "")

    assert zoom.should_bump_epoch(cm.store, operator_asked=True)[0]
    assert zoom.should_bump_epoch(cm.store, model_changed=True)[0]
    assert zoom.should_bump_epoch(cm.store, replanned=True)[0]

    for i in range(zoom.EPOCH_FILE_THRESHOLD):
        cm.store.note_change(f"src/f{i}.py")
    bump, why = zoom.should_bump_epoch(cm.store)
    assert bump and "files have changed" in why


def test_the_output_contract_is_last_in_the_tail():
    """D7 — recency helps, so the contract goes last.

    Diagnostics sit immediately before it: items 7 and 8 of G.7.1 conflict,
    and the resolution is to keep the contract short enough that repeating it
    costs little.
    """
    builder = PromptBuilder()
    tail = builder.tail_for("write the file",
                            diagnostics="1. main.py:3: error: nope",
                            contract=personas.CONTRACT_FILE)
    assert tail.rstrip().endswith(personas.CONTRACT_FILE)
    assert tail.index("WHAT WENT WRONG") < tail.index("OUTPUT CONTRACT")


def test_prompt_messages_keep_the_cache_boundary_as_a_message_boundary():
    """The split is a real message split, so per-message caches benefit too."""
    builder = PromptBuilder(model_system_prompt="sys")
    prompt = builder.build(personas.ENGINEER, "do the thing",
                           architecture="# ARCH")
    messages = prompt.messages()
    assert [m.role for m in messages] == ["system", "system", "user"]
    assert messages[1].content == prompt.prefix
    assert messages[2].content == prompt.tail


def test_the_first_file_sees_the_skeleton_in_the_cached_prefix(tmp_path):
    """The epoch snapshot was taken at session start, before the skeleton
    existed, so the first file was generated against a prefix with no
    architecture in it — the one thing skeleton-first is for."""
    from cognitive_coder import (
        AutoApprove,
        Host,
        LocalFileSystem,
        RecordingEvents,
        ScriptedLLM,
        Session,
        SessionConfig,
        SubprocessExec,
    )
    host = Host(llm=ScriptedLLM(["src/alpha.py — parse the input\n"
                                 "src/beta.py — report on it\n"]),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".s")),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(skeleton_first=True))
    session.start("a parser and a report")
    assert (tmp_path / "src" / "beta.py").exists(), "no skeleton written"
    prefix = session.codemap.prefix_block()
    assert "src/beta.py" in prefix, prefix
