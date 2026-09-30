# SPDX-License-Identifier: Apache-2.0
"""M23, M24, M25, M26 — edits, and the way back.

The property test is the important one: **apply then undo must restore
byte-identical content**, for a corpus of random edits, including CRLF files
and files with a BOM (§9). That guarantee is what makes auto-apply survivable
rather than reckless, and it is the guarantee most likely to be quietly broken
by a well-meaning refactor of the encoding layer.
"""

from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import patcher, textio  # noqa: E402
from cognitive_coder.errors import TransactionError  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    DenyAll,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
)
from cognitive_coder.types import Edit  # noqa: E402

# A corpus that covers the encodings this actually meets in the wild. The
# CRLF and BOM entries are not exotic: this is Windows-first software and the
# model emits `\n`, so every one of these is a file somebody has.
CORPUS = {
    "plain_lf.py": b"def a():\n    return 1\n\n\ndef b():\n    return 2\n"
,
    "crlf.py": b"def a():\r\n    return 1\r\n\r\ndef b():\r\n    return 2\r\n"
,
    "bom_crlf.py": "\ufeffdef a():\r\n    return 1\r\n\r\ndef b():\r\n"
                   "    return 2\r\n".encode("utf-8"),
    "bom_lf.py": "\ufeffdef a():\n    return 1\n".encode("utf-8"),
    "no_trailing_newline.py": b"def a():\n    return 1",
    # `return 1` is what the edit test anchors on; the é keeps the
    # encoding load-bearing. It used to be `return 'héllo'`, so the
    # encoding-preservation test skipped UTF-16 for want of an anchor.
    "utf16.py": "# héllo\ndef a():\n    return 1\n".encode("utf-16"),
    "latin1.py": b"# caf\xe9\ndef a():\n    return 1\n",
    "mixed_eol.py": b"def a():\r\n    return 1\nc = 3\r\n",
}


def _patcher(files=None):
    fs = MemoryFileSystem(dict(files or CORPUS))
    return fs, patcher.Patcher(fs, MemoryStorage(), AutoApprove(),
                               RecordingEvents())


# --------------------------------------------------------------------------
# encoding and line endings (§6.5a, M26)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CORPUS))
def test_decode_encode_round_trips_byte_for_byte(name):
    """The foundation of M26: the shape survives the trip on its own.

    One case genuinely cannot: a file that MIXES line endings has no single
    style to restore, so the minority endings are normalised. C7's rule is
    that a limitation degrades with a STATED cost, so the requirement here is
    byte-identity **or** an assumption saying plainly what will change. A
    silent normalisation would fail this test, and should.
    """
    original = CORPUS[name]
    tf = textio.decode(original)
    if tf.encode() != original:
        assert tf.assumption, (
            f"{name} does not round-trip and says nothing about why — that "
            f"is a silent change to somebody's file")
        assert "line-ending" in tf.assumption
        assert textio.is_mixed_eol(original.decode("utf-8", "replace"))


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_an_edit_preserves_encoding_bom_and_line_endings(name):
    """A whole-file write must not silently convert every line ending.

    That is the failure that turns a one-function change into a
    whole-file diff, and makes review impossible.
    """
    fs, p = _patcher()
    before = fs.files[name]
    tx = p.begin("edit", atomic=False)
    result = tx.apply([Edit(path=name, kind="replace",
                            old="return 1", new="return 42")])
    assert result[0].ok, result[0].reason
    tx.commit()

    after = fs.files[name]
    tf_before, tf_after = textio.decode(before), textio.decode(after)
    assert tf_after.bom == tf_before.bom, "the BOM was added or dropped"
    assert tf_after.encoding == tf_before.encoding, "the encoding changed"
    assert "return 42" in tf_after.text
    raw_before = before.decode(tf_before.encoding, "replace")
    if textio.is_mixed_eol(raw_before):
        # The one case that cannot be byte-faithful: the minority endings
        # become the dominant one. C7 — it is SAID, in the result.
        assert tf_after.eol == tf_before.eol
        assert "line-ending" in result[0].reason, result[0].reason
    else:
        assert tf_after.eol == tf_before.eol, "line endings were rewritten"
        assert after.count(b"\r") == before.count(b"\r"), (
            "line endings were rewritten")


def test_apply_then_undo_restores_byte_identical_content():
    """M26 — the property test, over the whole corpus at once."""
    fs, p = _patcher()
    originals = dict(fs.files)

    tx = p.begin("everything", atomic=True)
    for name in sorted(CORPUS):
        tx.apply([Edit(path=name, kind="whole",
                       new="# replaced entirely\nx = 1\n")])
    tx.rollback()

    for name, original in originals.items():
        assert fs.files[name] == original, (
            f"{name} was not restored byte-for-byte")


def test_random_edits_then_undo_restore_byte_identical_content():
    """The randomised half of the property test (§9)."""
    rng = random.Random(20260806)
    for trial in range(25):
        fs, p = _patcher()
        originals = dict(fs.files)
        name = rng.choice(sorted(CORPUS))
        kind = rng.choice(["whole", "replace"])
        tx = p.begin(f"trial{trial}", atomic=True)
        if kind == "whole":
            tx.apply([Edit(path=name, kind="whole",
                           new="".join(rng.choice("abc \n") for _ in
                                       range(rng.randint(1, 200))))])
        else:
            tx.apply([Edit(path=name, kind="replace", old="return 1",
                           new=f"return {rng.randint(2, 99)}")])
        tx.rollback()
        for path, original in originals.items():
            assert fs.files[path] == original, (
                f"trial {trial}: {path} differs after undo")


# --------------------------------------------------------------------------
# the rule that prevents the worst damage (M23)
# --------------------------------------------------------------------------

def test_an_ambiguous_anchor_is_refused_not_guessed():
    """M23 — picking the first match is how the WRONG function gets edited."""
    fs, p = _patcher({"m.py": b"def a():\n    return 1\n\ndef b():\n"
                              b"    return 1\n"})
    tx = p.begin("ambiguous")
    result = tx.apply([Edit(path="m.py", kind="replace", old="return 1",
                            new="return 2")])[0]
    assert not result.ok
    assert "more than once" in result.reason
    assert "refusing" in result.reason
    assert fs.files["m.py"] == (b"def a():\n    return 1\n\ndef b():\n"
                                b"    return 1\n")


def test_an_anchor_inside_a_longer_name_is_refused():
    """Observed: raw substring search let `x = 1` → `x = 2` apply to
    `max = 10` and produce `max = 20` — applied cleanly, silently wrong."""
    fs, p = _patcher({"m.py": b"max = 10\nprint(max)\n"})
    tx = p.begin("sub")
    result = tx.apply([Edit(path="m.py", kind="replace", old="x = 1",
                            new="x = 2")])[0]
    assert not result.ok
    assert "matches only inside a longer name" in result.reason
    assert fs.files["m.py"] == b"max = 10\nprint(max)\n"


def test_a_whole_word_match_is_used_when_a_partial_one_also_exists():
    """The in-a-name hit is not a candidate, so it does not make the real
    one ambiguous either."""
    fs, p = _patcher({"m.py": b"max = 10\nx = 1\n"})
    tx = p.begin("sub")
    result = tx.apply([Edit(path="m.py", kind="replace", old="x = 1",
                            new="x = 2")])[0]
    assert result.ok, result.reason
    assert fs.files["m.py"] == b"max = 10\nx = 2\n"


def test_preview_names_an_anchor_inside_a_longer_name():
    fs, p = _patcher({"m.py": b"total = 10\n"})
    out = p.preview([Edit(path="m.py", kind="replace", old="tal = 1",
                          new="tal = 2")])
    assert "inside a longer name" in out


def test_a_missing_anchor_says_the_file_may_have_changed():
    """C6 — a sentence naming what happened, and what it probably means."""
    fs, p = _patcher({"m.py": b"x = 1\n"})
    tx = p.begin("missing")
    result = tx.apply([Edit(path="m.py", kind="replace", old="y = 2",
                            new="y = 3")])[0]
    assert not result.ok
    assert "not in the file" in result.reason


def test_editing_a_mixed_eol_file_says_what_else_it_changed():
    """C7: "edited one line of a mostly-CRLF file, rewrote every LF line"
    must be stated. The assumption used to be computed and thrown away."""
    fs = MemoryFileSystem({"m.py": b"a = 1\r\nb = 2\nc = 3\r\n"})
    events = RecordingEvents()
    p = patcher.Patcher(fs, MemoryStorage(), AutoApprove(), events)
    tx = p.begin("mixed")
    result = tx.apply([Edit(path="m.py", kind="replace", old="a = 1",
                            new="a = 9")])[0]
    assert result.ok
    assert "line-ending" in result.reason
    assert any("line-ending" in msg for _k, msg, _d in events.of("warning"))


def test_a_whole_file_edit_of_a_utf16_file_keeps_its_bom():
    fs, p = _patcher({"u.py": "\ufeffx = 1\r\n".encode("utf-16-le")})
    tx = p.begin("u")
    assert tx.apply([Edit(path="u.py", kind="whole", new="x = 2\n")])[0].ok
    assert fs.files["u.py"] == "\ufeffx = 2\r\n".encode("utf-16-le")


def test_an_edit_that_widens_the_encoding_says_so():
    fs, p = _patcher({"l.py": b"# caf\xe9\nx = 1\n"})
    tx = p.begin("l")
    result = tx.apply([Edit(path="l.py", kind="replace", old="x = 1",
                            new="x = '\u2192'")])[0]
    assert result.ok
    assert "UTF-8" in result.reason, result.reason


def test_an_anchor_matching_crlf_text_still_applies():
    """The model emits `\\n`; the file has `\\r\\n`. It must still work.

    Without normalisation this fails mysteriously — the operator watches a
    perfectly good edit be refused for no visible reason.
    """
    fs, p = _patcher({"crlf.py": b"a = 1\r\nb = 2\r\n"})
    tx = p.begin("crlf")
    result = tx.apply([Edit(path="crlf.py", kind="replace",
                            old="a = 1\nb = 2", new="a = 9\nb = 8")])[0]
    assert result.ok, result.reason
    assert fs.files["crlf.py"] == b"a = 9\r\nb = 8\r\n"


# --------------------------------------------------------------------------
# the jail (M24)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "../escaped.py", "../../escaped.py", "/etc/passwd",
    "C:\\Windows\\System32\\evil.py", "src/../../escaped.py",
    ".git/config", ".git/hooks/pre-commit",
])
def test_no_write_lands_outside_the_project_root(path):
    """M24 and M27 — in any mode, including `.git/`."""
    fs, p = _patcher({"keep.py": b"x = 1\n"})
    tx = p.begin("escape")
    result = tx.apply([Edit(path=path, kind="whole", new="pwned")])[0]
    assert not result.ok, f"{path} was WRITTEN"
    assert list(fs.files) == ["keep.py"]


@pytest.mark.parametrize("path", [".GIT/config", ".Git/hooks/pre-commit",
                                  "src/../.GIT/HEAD"])
def test_git_is_refused_in_any_case(path):
    """On Windows and macOS `.GIT` IS `.git`; a case-sensitive check was a
    way into the one directory the engine promises never to touch (M27)."""
    fs, p = _patcher({"keep.py": b"x = 1\n"})
    tx = p.begin("git")
    assert not tx.apply([Edit(path=path, kind="whole", new="x")])[0].ok
    assert list(fs.files) == ["keep.py"]


def test_list_excludes_git_in_any_case(tmp_path):
    from cognitive_coder.ports import LocalFileSystem
    mem = MemoryFileSystem({".GIT/config": b"", "a.py": b"",
                            "sub/.Git/x": b""})
    assert mem.list("*") == ["a.py"]
    (tmp_path / ".Git").mkdir()
    (tmp_path / ".Git" / "config").write_text("x")
    (tmp_path / "a.py").write_text("")
    assert LocalFileSystem(str(tmp_path)).list("*") == ["a.py"]


def test_a_godot_res_path_is_translated_rather_than_taken_literally():
    """§6.1a's one real trap: `res://` must never reach the FileSystemPort."""
    fs, p = _patcher({"scripts/player.gd": b"extends Node\n"})
    tx = p.begin("godot")
    result = tx.apply([Edit(path="res://scripts/player.gd", kind="whole",
                            new="extends Node2D\n")])[0]
    assert result.ok, result.reason
    assert "scripts/player.gd" in fs.files
    assert not any(k.startswith("res:") for k in fs.files)


# --------------------------------------------------------------------------
# transactions (M25)
# --------------------------------------------------------------------------

def test_a_sealed_transaction_survives_a_later_rollback():
    """M25 rule 3 — verified work is not destroyed by a later failure.

    Reverting work the operator watched succeed, because a separate later
    task failed, is the behaviour this whole model exists to prevent.
    """
    fs, p = _patcher({"a.py": b"a = 1\n", "b.py": b"b = 1\n"})
    first = p.begin("task-a", atomic=False)
    first.apply([Edit(path="a.py", kind="whole", new="a = 99\n")])
    first.commit(verified=True)
    assert first.record().sealed

    second = p.begin("task-b", atomic=True)
    second.apply([Edit(path="b.py", kind="whole", new="b = 99\n")])
    second.rollback()

    assert fs.files["a.py"] == b"a = 99\n", "sealed work was destroyed"
    assert fs.files["b.py"] == b"b = 1\n"


def test_an_atomic_transaction_reverts_all_of_its_files():
    """M25 — a signature and its caller are one change; both revert."""
    fs, p = _patcher({"sig.py": b"def f(a):\n    pass\n",
                      "call.py": b"f(1)\n"})
    tx = p.begin("refactor", atomic=True)
    tx.apply([Edit(path="sig.py", kind="whole", new="def f(a, b):\n    pass\n")])
    tx.apply([Edit(path="call.py", kind="whole", new="f(1, 2)\n")])
    tx.rollback()
    assert fs.files["sig.py"] == b"def f(a):\n    pass\n"
    assert fs.files["call.py"] == b"f(1)\n"


def test_rollback_deletes_a_file_the_transaction_created():
    """"Restore" for a file that did not exist means removing it."""
    fs, p = _patcher({"keep.py": b"x = 1\n"})
    tx = p.begin("create", atomic=True)
    tx.apply([Edit(path="new.py", kind="whole", new="y = 2\n")])
    assert "new.py" in fs.files
    tx.rollback()
    assert "new.py" not in fs.files


def test_sequence_numbers_are_monotonic_and_a_rollback_is_appended():
    """M25 rules 2 and 4 — a linear log, and undo is a new fact."""
    fs, p = _patcher({"a.py": b"a = 1\n"})
    tx = p.begin("one")
    tx.apply([Edit(path="a.py", kind="whole", new="a = 2\n")])
    tx.commit(verified=True)
    tx2 = p.begin("two")
    tx2.apply([Edit(path="a.py", kind="whole", new="a = 3\n")])
    tx2.rollback()

    history = p.history()
    seqs = [r.seq for r in history]
    assert seqs == sorted(seqs), "the log is not linear"
    assert len(seqs) == len(set(seqs)), "a sequence number was reused"
    assert any(r.state == "rollback_of" for r in history), (
        "the rollback was not journaled as its own event")


def test_history_is_one_row_per_transaction_plus_rollback_events():
    fs, p = _patcher({"a.py": b"a = 1\n"})
    tx = p.begin("only")
    tx.apply([Edit(path="a.py", kind="whole", new="a = 2\n")])
    tx.commit(verified=True)
    rows = [r for r in p.history() if r.state != "rollback_of"]
    assert len(rows) == 1
    assert rows[0].state == "committed" and rows[0].verified


def test_undo_to_states_how_much_verified_work_it_would_discard():
    """M25 rule 3 — reaching past a seal is deliberately awkward."""
    fs, p = _patcher({"a.py": b"a = 1\n", "b.py": b"b = 1\n"})
    t1 = p.begin("first")
    t1.apply([Edit(path="a.py", kind="whole", new="a = 2\n")])
    t1.commit(verified=True)
    t2 = p.begin("second")
    t2.apply([Edit(path="b.py", kind="whole", new="b = 2\n")])
    t2.commit(verified=True)

    asked = {}

    def confirm(sentence):
        asked["sentence"] = sentence
        return False

    out = p.undo_to(1, confirm=confirm)
    assert not out["ok"]
    assert "verified" in asked["sentence"]
    assert "b.py" in asked["sentence"]
    assert fs.files["b.py"] == b"b = 2\n", "it undid without confirmation"

    out = p.undo_to(1, confirm=lambda s: True)
    assert out["ok"]
    assert fs.files["b.py"] == b"b = 1\n"
    assert fs.files["a.py"] == b"a = 2\n", "it went back too far"


def _commit_three_edits_of_keep(p):
    for i in range(1, 4):
        tx = p.begin(f"t{i}")
        tx.apply([Edit(path="keep.py", kind="whole", new=f"v{i}\n")])
        tx.commit(verified=True)


def test_undo_to_past_a_pruned_snapshot_refuses_rather_than_deleting(
        monkeypatch):
    """Observed: a missing snapshot was read as "created by this
    transaction" and the pre-existing file was DELETED, with ok=True. The
    pruner removes exactly those bytes, so this happened to any project
    more than MAX_SNAPSHOTS transactions old."""
    monkeypatch.setattr(patcher, "MAX_SNAPSHOTS", 1)
    fs, p = _patcher({"keep.py": b"v0\n"})
    _commit_three_edits_of_keep(p)
    asked = []
    out = p.undo_to(1, confirm=lambda s: asked.append(s) or True)
    assert not out["ok"]
    assert fs.files.get("keep.py") == b"v3\n", "the file was changed"
    assert "pruned" in out["note"] and "keep.py" in out["note"]
    assert "Nothing was changed" in out["note"]
    assert not asked, "it asked to confirm an undo it was going to refuse"


def test_undo_to_still_deletes_a_file_the_transaction_CREATED(monkeypatch):
    """The manifest says NEW — so deleting is the correct undo, even when
    the snapshot directory itself has been pruned."""
    monkeypatch.setattr(patcher, "MAX_SNAPSHOTS", 1)
    fs, p = _patcher({"keep.py": b"k\n"})
    t1 = p.begin("base")
    t1.apply([Edit(path="keep.py", kind="whole", new="k2\n")])
    t1.commit(verified=True)
    t2 = p.begin("create")
    t2.apply([Edit(path="made.py", kind="whole", new="m = 1\n")])
    t2.commit(verified=True)
    t3 = p.begin("later")
    t3.apply([Edit(path="other.py", kind="whole", new="o = 1\n")])
    t3.commit(verified=True)
    out = p.undo_to(t1.seq, confirm=lambda s: True)
    assert out["ok"], out["note"]
    assert "made.py" not in fs.files and "other.py" not in fs.files
    assert fs.files["keep.py"] == b"k2\n"


def test_prune_keeps_the_NEWEST_snapshots_past_seq_9999(monkeypatch):
    """`{seq:04d}` sorts "10000" before "9999" as text, so a lexicographic
    prune deleted the newest snapshot once the counter reached five
    digits."""
    monkeypatch.setattr(patcher, "MAX_SNAPSHOTS", 2)
    fs, p = _patcher({"a.py": b"a = 0\n"})
    p.storage.set("cognitive_coder.patcher.seq", 9997)
    seqs = []
    for i in range(1, 5):
        tx = p.begin(f"n{i}")
        tx.apply([Edit(path="a.py", kind="whole", new=f"a = {i}\n")])
        tx.commit(verified=True)
        seqs.append(tx.seq)
    kept = {k.split("/")[1].split("-")[0] for k in fs.files
            if k.startswith(".cc_snapshots/")}
    assert kept == {str(s).zfill(4) for s in seqs[-2:]}, kept


def test_a_second_open_transaction_is_refused():
    fs, p = _patcher({"a.py": b"a = 1\n"})
    p.begin("one")
    with pytest.raises(TransactionError):
        p.begin("two")


def test_an_exception_inside_a_transaction_rolls_it_back():
    """§5.2 — no half-applied state is left behind, ever."""
    fs, p = _patcher({"a.py": b"a = 1\n"})
    with pytest.raises(ValueError), p.begin("boom", atomic=True) as tx:
        tx.apply([Edit(path="a.py", kind="whole", new="a = 2\n")])
        raise ValueError("something went wrong mid-task")
    assert fs.files["a.py"] == b"a = 1\n"


# --------------------------------------------------------------------------
# the approval gate (M18)
# --------------------------------------------------------------------------

def test_nothing_is_written_when_approval_is_declined():
    """The library default is approval-REQUIRED, and it is a real gate."""
    fs = MemoryFileSystem({"a.py": b"a = 1\n"})
    p = patcher.Patcher(fs, MemoryStorage(), DenyAll(), RecordingEvents())
    tx = p.begin("declined")
    result = tx.apply([Edit(path="a.py", kind="whole", new="a = 2\n")])[0]
    assert not result.ok
    assert "not approved" in result.reason
    assert fs.files["a.py"] == b"a = 1\n"


def test_every_write_reaches_the_approval_port_with_a_real_diff():
    """M18 — including model-initiated edits; there is no second path."""
    fs = MemoryFileSystem({"a.py": b"a = 1\n"})
    approval = AutoApprove()
    p = patcher.Patcher(fs, MemoryStorage(), approval, RecordingEvents())
    tx = p.begin("t")
    tx.apply([Edit(path="a.py", kind="whole", new="a = 2\n")])
    assert len(approval.diffs) == 1
    summary, diff = approval.diffs[0]
    assert "a.py" in summary
    assert "-a = 1" in diff and "+a = 2" in diff


# --------------------------------------------------------------------------
# parsing model output (D5)
# --------------------------------------------------------------------------

def test_edits_are_parsed_from_several_formats():
    text = ("Here you go:\n\n"
            "<<<CC-EDIT src/x.py\nold line\n===\nnew line\n>>>CC-END\n\n"
            "```python path=src/y.py\ny = 2\n```\n")
    edits = patcher.parse_edits(text)
    by_path = {e.path: e for e in edits}
    assert by_path["src/x.py"].kind == "replace"
    assert by_path["src/x.py"].old == "old line"
    assert by_path["src/y.py"].kind == "whole"


def test_extract_code_prefers_a_fence_that_parses():
    """D5 — never assume the first fence; validate, then fall through."""
    text = ("```python\ndef broken(:\n```\n\n"
            "```python\ndef good():\n    return 1\n```\n")

    def validates(candidate):
        import ast
        try:
            ast.parse(candidate)
            return True
        except SyntaxError:
            return False

    out = patcher.extract_code(text, "python", validator=validates)
    assert "def good" in out
    assert "broken" not in out


@pytest.mark.parametrize("lang_id,tag,code,command", [
    ("rust", "rust", "fn main() {\n    println!(\"hi\");\n}", "cargo run"),
    ("c", "c", "int main(void) {\n    return 0;\n}", "gcc main.c"),
    ("go", "go", "package main\n\nfunc main() {}", "go run ."),
])
def test_an_untagged_command_fence_is_not_taken_as_the_code(
        lang_id, tag, code, command):
    """Observed: `.get(lang_id, "")` put "" into the alias set for every
    language outside a five-entry map, so an UNTAGGED fence counted as
    tagged for the target — and `main.rs` was written as `cargo run`."""
    text = (f"Build with:\n```\n{command}\n```\n\nHere is the file:\n"
            f"```{tag}\n{code}\n```\n")
    assert patcher.extract_code(text, lang_id) == code


def test_the_longest_untagged_fence_beats_a_fence_for_another_language():
    text = ("Run it:\n```bash\ncargo build --release && ./target/release/"
            "app --verbose --config ./config/app.toml\n```\n"
            "```\nfn main() {}\n```\n"
            "```\nfn main() {\n    let x = 1;\n}\n```\n")
    assert patcher.extract_code(text, "rust") == (
        "fn main() {\n    let x = 1;\n}")


def test_a_fence_tagged_rs_counts_as_rust():
    text = "```\ncargo run\n```\n```rs\nfn main() {}\n```\n"
    assert patcher.extract_code(text, "rust") == "fn main() {}"


def test_preview_changes_nothing():
    fs, p = _patcher({"a.py": b"a = 1\n"})
    diff = p.preview([Edit(path="a.py", kind="whole", new="a = 2\n")])
    assert "+a = 2" in diff
    assert fs.files["a.py"] == b"a = 1\n"


def test_preview_reports_an_ambiguous_anchor_before_anything_is_tried():
    fs, p = _patcher({"m.py": b"x = 1\nx = 1\n"})
    diff = p.preview([Edit(path="m.py", kind="replace", old="x = 1",
                           new="x = 2")])
    assert "ambiguous" in diff
