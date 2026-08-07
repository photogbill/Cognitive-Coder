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
    "utf16.py": "def a():\n    return 'héllo'\n".encode("utf-16"),
    "latin1.py": b"# caf\xe9\ndef a():\n    return 1\n",
    "mixed_eol.py": b"a = 1\r\nb = 2\nc = 3\r\n",
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
    if b"a()" not in textio.decode(before).text.encode("utf-8"):
        pytest.skip("no anchor in this fixture")

    tx = p.begin("edit", atomic=False)
    result = tx.apply([Edit(path=name, kind="replace",
                            old="return 1", new="return 42")])
    if not result[0].ok:
        pytest.skip(f"anchor not applicable: {result[0].reason}")
    tx.commit()

    after = fs.files[name]
    tf_before, tf_after = textio.decode(before), textio.decode(after)
    if textio.is_mixed_eol(tf_before.text) or tf_before.assumption:
        pytest.skip("mixed line endings — the declared-assumption path")
    assert tf_after.eol == tf_before.eol, "line endings were rewritten"
    assert tf_after.bom == tf_before.bom, "the BOM was added or dropped"
    assert tf_after.encoding == tf_before.encoding, "the encoding changed"
    assert "return 42" in tf_after.text


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


def test_a_missing_anchor_says_the_file_may_have_changed():
    """C6 — a sentence naming what happened, and what it probably means."""
    fs, p = _patcher({"m.py": b"x = 1\n"})
    tx = p.begin("missing")
    result = tx.apply([Edit(path="m.py", kind="replace", old="y = 2",
                            new="y = 3")])[0]
    assert not result.ok
    assert "not in the file" in result.reason


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
