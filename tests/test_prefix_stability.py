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
    assert "changed since" in note.lower()

    prefix = cm.prefix_block("src/stats.py")
    assert "changed since" not in prefix.lower()
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


# --------------------------------------------------------------------------
# what a prefix cache can actually reuse (the two design decisions, 09-30)
# --------------------------------------------------------------------------
#
# llama-server and llama-cpp-python keep the previous prompt's KV and reuse
# the longest run of tokens matching the next prompt. These tests measure
# that run directly, on the prompts the engine sends.

def _rendered(messages) -> str:
    return "".join(f"<|{m.role}|>{m.content}" for m in messages)


def _common(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def test_a_persona_switch_keeps_the_conventions_and_architecture_cached():
    """The persona came FIRST in the prefix, so the switch to the repairer
    on every repair — and back to the engineer for the next file — threw
    away the whole cached prompt: about 1% survived, measured."""
    builder = PromptBuilder(conventions="Use type hints everywhere. " * 40)
    architecture = _codemap().prefix_block()
    engineer = builder.build(personas.PERSONAS["engineer"], "write a.py",
                             architecture=architecture).messages()
    repairer = builder.build(personas.PERSONAS["repairer"], "fix a.py",
                             architecture=architecture).messages()
    a, b = _rendered(engineer), _rendered(repairer)
    shared = a[:_common(a, b)]
    assert "Use type hints everywhere." in shared
    assert architecture in shared, "the architecture was not reused"


class _Model:
    """Answers by what it is asked for; records every prompt."""

    FILES = {
        "src/vec.py": "def add(a, b):\n    return (a[0] + b[0], a[1] + b[1])\n",
        "src/physics.py": "from src.vec import add\n\n\n"
                          "def step(p, v):\n    return add(p, v)\n",
        "src/world.py": "from src.physics import step\n\n\n"
                        "def tick(p):\n    return step(p, (0, 1))\n",
    }

    def __init__(self):
        self.prompts = []

    def capabilities(self):
        from cognitive_coder.types import ModelCapabilities
        return ModelCapabilities(name="fake", family="mistral",
                                 context_tokens=32768, supports_tools=True)

    def count_tokens(self, text):
        return max(1, len(text or "") // 4)

    def stream(self, messages, **kw):
        yield self.complete(messages, **kw).text

    def complete(self, messages, **kw):
        from cognitive_coder.types import Completion
        self.prompts.append(list(messages))
        last = messages[-1].content
        if "security" in last and "performance" in last:
            return Completion(text='{"security": [], "performance": [], '
                                   '"overall": "fine"}')
        m = re.search(r"`(src/\w+\.py)`", last)
        if not m:
            return Completion(text="src/vec.py — vector maths\n"
                                   "src/physics.py — uses vec\n"
                                   "src/world.py — uses physics\n")
        body = self.FILES.get(m.group(1), "X = 1\n")
        return Completion(text=f"```python\n{body}```")


def _session(tmp_path):
    from cognitive_coder import (
        AutoApprove,
        Host,
        LocalFileSystem,
        RecordingEvents,
        Session,
        SessionConfig,
        SubprocessExec,
    )
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "__init__.py").write_text("")
    for i in range(10):
        (lib / f"part{i}.py").write_text("".join(
            f"def {n}_{i}(value):\n    return value\n\n\n"
            for n in ("load", "parse", "clamp", "blend", "report")))
    model = _Model()
    host = Host(llm=model, fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".cc_state")),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(attempts=2))
    session.run("a small physics package in src")
    return session, model


def test_the_snapshot_is_not_rebuilt_after_every_file(tmp_path):
    """It was: `maybe_bump_epoch(target=<the file just written>)` always
    fired, so every task began by re-reading the whole cached prefix."""
    session, _model = _session(tmp_path)
    rebuilt = [r for r in session.journal.events()
               if r.get("event") == "epoch"]
    epochs = session.codemap.store.epoch
    # One for the session start and one after the skeleton; three files
    # is under G.7.2's threshold, so nothing after that.
    assert epochs <= 2, (epochs, rebuilt)


def test_the_first_attempt_still_sees_what_was_already_written(tmp_path):
    """Why the rebuild existed: without it the snapshot shows the skeleton,
    and a fresh stub has no calls for the interfaces block to follow. The
    tail now carries each changed file's CURRENT line instead."""
    _session_, model = _session(tmp_path)
    first_world = next(p for p in model.prompts
                       if "`src/world.py`" in p[-1].content)
    tail = first_world[-1].content
    assert "src/physics.py: step" in tail, tail[-1500:]


def test_a_build_reuses_most_of_each_prompt(tmp_path):
    """The number both decisions were made on. On this small build 29% of
    the prompt text was reusable before and 45% after (the rest is each
    file's own tail, which no ordering can share); on a realistic project —
    fifteen library modules and a skill file — it was 37% before and 67%
    after, 46% less prompt processing. The floor sits between the two
    small-build numbers."""
    _session_, model = _session(tmp_path)
    total = reused = 0
    previous = ""
    for prompt in model.prompts:
        text = _rendered(prompt)
        reused += _common(previous, text)
        total += len(text)
        previous = text
    assert reused / total >= 0.40, f"{reused / total:.0%} reused"
