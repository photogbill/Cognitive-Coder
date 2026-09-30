# SPDX-License-Identifier: Apache-2.0
"""What the engine and its kit leave behind, and how its errors read.

Three findings from the review, each about something left in a state the
operator did not ask for:

  * `MemoryStorage()` made a `/tmp/ccoder-*` directory on EVERY construction
    and never removed one — thousands accumulated on a CI box, most for
    instances that never opened a database;
  * the conformance kit — which a host runs against its OWN project root —
    left `cc_conformance/`, a symlink to the root's PARENT, and a
    `.git/config` that makes git treat the folder as a broken repository;
  * `CognitiveCoderError.wrap` built subclasses through their own
    constructors, so `PathEscape.wrap(...)` produced "Refused to touch
    'could not load the plan'" and `NoModelLoadedError.wrap` raised
    TypeError — an error path that errors.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cognitive_coder import errors  # noqa: E402
from cognitive_coder.ports import LocalFileSystem, MemoryStorage  # noqa: E402

# --------------------------------------------------------------------------
# MemoryStorage
# --------------------------------------------------------------------------


def test_constructing_memory_storage_creates_no_directory(monkeypatch):
    made = []
    real = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp",
                        lambda *a, **k: made.append(1) or real(*a, **k))
    for _ in range(50):
        MemoryStorage()
    assert not made, f"{len(made)} temp directories for 50 constructions"


def test_memory_storage_databases_are_private_stable_and_shared_root():
    a, b = MemoryStorage(), MemoryStorage()
    pa, pb = a.sqlite_path("codemap"), b.sqlite_path("codemap")
    assert pa != pb, "two instances must not share a database"
    assert a.sqlite_path("codemap") == pa, "the path must be stable"
    # One process-wide parent, cleaned up at exit — not one per instance.
    assert Path(pa).parent.parent == Path(pb).parent.parent
    db = sqlite3.connect(pa)
    db.execute("CREATE TABLE t (x)")
    db.close()


def test_an_explicit_base_dir_is_still_honoured(tmp_path):
    s = MemoryStorage(str(tmp_path / "state"))
    assert Path(s.sqlite_path("x")).parent == tmp_path / "state"


# --------------------------------------------------------------------------
# the conformance kit cleans up after itself
# --------------------------------------------------------------------------

def test_the_filesystem_checks_leave_the_root_as_they_found_it(tmp_path):
    from tests.port_conformance import check_filesystem
    root = tmp_path / "proj"
    root.mkdir()
    (root / "keep.py").write_text("x = 1\n")
    report = check_filesystem(LocalFileSystem(str(root)))
    assert report.ok, report.text()
    assert sorted(os.listdir(root)) == ["keep.py"]
    assert sorted(os.listdir(tmp_path)) == ["proj"], "the parent was touched"


def test_the_kit_never_writes_into_a_real_git_directory(tmp_path):
    from tests.port_conformance import check_filesystem
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n\trepo = mine\n")
    report = check_filesystem(LocalFileSystem(str(tmp_path)))
    assert report.ok, report.text()
    assert os.listdir(tmp_path / ".git") == ["config"]
    assert (tmp_path / ".git" / "config").read_text() == \
        "[core]\n\trepo = mine\n"


# --------------------------------------------------------------------------
# errors.wrap
# --------------------------------------------------------------------------

def _raise_and_wrap(cls, sentence):
    try:
        raise ValueError("boom")
    except ValueError as exc:
        return cls.wrap(exc, sentence)


@pytest.mark.parametrize("cls", [
    errors.CognitiveCoderError, errors.PathEscape, errors.NoModelLoadedError,
    errors.GuardRefusal, errors.Cancelled, errors.BudgetExceeded,
    errors.TransactionError])
def test_wrap_works_for_every_error_type(cls):
    err = _raise_and_wrap(cls, "could not load the plan")
    assert isinstance(err, cls)
    assert str(err) == "Could not load the plan."
    assert "ValueError: boom" in err.detail
    assert "Traceback (most recent call last)" in err.detail
    assert err.__cause__ is not None and str(err.__cause__) == "boom"


def test_wrap_without_a_sentence_still_gives_one():
    err = _raise_and_wrap(errors.CognitiveCoderError, "")
    text = str(err)
    assert text.endswith(".") and "boom" in text
    assert "Traceback" not in text and "ValueError" not in text


def test_wrapping_an_engine_error_keeps_its_own_detail():
    inner = errors.CognitiveCoderError("Inner.", "inner detail line")
    outer = errors.CognitiveCoderError.wrap(inner, "Outer happened")
    assert "inner detail line" in outer.detail


def test_the_kit_leaves_an_in_memory_host_empty():
    from cognitive_coder.ports import MemoryFileSystem
    from tests.port_conformance import check_filesystem
    fs = MemoryFileSystem()
    report = check_filesystem(fs)
    assert report.ok, report.text()
    assert fs.files == {}, sorted(fs.files)


@pytest.mark.parametrize("path", ["../config.py", "a/../../config.py",
                                  "/etc/passwd", "C:/Windows/evil.py",
                                  "/projectX/y.py"])
def test_the_in_memory_filesystem_refuses_escapes_instead_of_clamping(path):
    """`normpath("/" + "../config.py")` is `/config.py`: the escape was
    clamped INTO the root, so `../config.py` silently overwrote the
    project's own `config.py`. The kit caught it on the default Host."""
    from cognitive_coder.ports import MemoryFileSystem
    fs = MemoryFileSystem({"config.py": b"KEEP\n"})
    with pytest.raises(ValueError):
        fs.write_bytes(path, b"pwned")
    assert fs.files == {"config.py": b"KEEP\n"}
    assert not fs.exists(path)


def test_the_in_memory_filesystem_still_accepts_its_own_root():
    from cognitive_coder.ports import MemoryFileSystem
    fs = MemoryFileSystem()
    fs.write("/project/src/a.py", "x = 1\n")
    fs.write("src/../b.py", "y = 1\n")
    assert sorted(fs.files) == ["b.py", "src/a.py"]
