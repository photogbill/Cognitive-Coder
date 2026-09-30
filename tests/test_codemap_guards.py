# SPDX-License-Identifier: Apache-2.0
"""The codemap's tool surface and store, hardened (item 15).

Each case was observed: `read_slice` returned 400 KB for one call and read
`.git/config` and `.env` (the index's skip list was never applied to the
tool); `[READ_SLICE: a.py, ten, twenty]` raised ValueError out of
`answer_text_lookups`; the marker regex took `def list_symbols(path):` in
generated code for a lookup; a deleted file stayed indexed forever; an
older on-disk schema crashed `put_file`; and a second writer's lock
escaped `index_file` as an exception.
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.codemap import CodeMap  # noqa: E402
from cognitive_coder.codemap import store as store_mod  # noqa: E402
from cognitive_coder.codemap.store import Store  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem, MemoryStorage  # noqa: E402


class _Events:
    def __init__(self):
        self.rows = []

    def event(self, *args):
        self.rows.append(args)


def _cm(files, **kw):
    cm = CodeMap(MemoryFileSystem(dict(files)), MemoryStorage(), **kw)
    cm.index_project()
    return cm


# -- read_slice ---------------------------------------------------------------

def test_read_slice_is_capped_in_lines_and_bytes():
    cm = _cm({"a.py": b"x = 1\n" * 1000,
              "big.py": ("x = 1  # " + "y" * 4000 + "\n").encode() * 100})
    out = cm.call_tool("read_slice", {"path": "a.py", "start": -5,
                                      "end": 10 ** 9})
    assert out.count("\n") <= 205
    assert "more lines" in out
    out = cm.call_tool("read_slice", {"path": "big.py", "start": 1,
                                      "end": 100})
    assert len(out) <= 17_000, len(out)
    assert "more" in out.splitlines()[-1]


@pytest.mark.parametrize("path", [".git/config", ".env", ".env.local",
                                  "node_modules/x/index.js",
                                  ".cc_journal/s.jsonl", "sub/.env"])
def test_read_slice_refuses_what_the_index_skips(path):
    cm = _cm({path: b"SECRET=hunter2\n", "a.py": b"x = 1\n"})
    out = cm.call_tool("read_slice", {"path": path})
    assert "hunter2" not in out
    assert "not" in out and "read" in out


def test_malformed_bounds_are_a_sentence():
    cm = _cm({"src/a.py": b"def a():\n    return 1\n"})
    cm.reset_lookups()
    out = cm.answer_text_lookups("[READ_SLICE: src/a.py, ten, twenty]")
    assert "line number" in out
    out = cm.call_tool("read_slice", {"path": "src/a.py", "start": "abc"})
    assert "line number" in out and "invalid literal" not in out


# -- text markers -------------------------------------------------------------

def test_generated_code_is_not_mistaken_for_a_lookup():
    cm = _cm({"src/a.py": b"def a():\n    return 1\n"})
    cm.reset_lookups()
    code = ("def list_symbols(path):\n    return outline(path)\n\n\n"
            "def search_codemap(name):\n    return db.find(name)\n")
    assert cm.parse_text_lookups(code) == []
    assert cm.answer_text_lookups(code) == ""
    cm.reset_lookups()
    comment = "def outline(p):\n    # list symbols defined in the file\n"
    assert cm.answer_text_lookups(comment) == ""


def test_real_markers_still_work():
    cm = _cm({"src/a.py": b"def a():\n    return 1\n"})
    for text in ("[SEARCH_CODEMAP: a]", "  SEARCH_CODEMAP(a)\n",
                 "<search_codemap>a</search_codemap>",
                 "I need this:\n[search_codemap: a]\n"):
        assert cm.parse_text_lookups(text) == [("search_codemap", "a")], text


# -- store lifecycle ----------------------------------------------------------

def test_a_deleted_file_leaves_the_index():
    cm = _cm({"src/old.py": b"def gone():\n    return 1\n",
              "src/use.py": b"from src.old import gone\n\n\n"
                            b"def f():\n    return gone()\n"})
    assert cm.store.callers_of("gone")
    cm.fs.files.pop("src/old.py")
    cm.index_project()
    assert [f["path"] for f in cm.store.files()] == ["src/use.py"]
    assert not cm.resolves("gone")
    # The caller's edge is not silently dropped: it is unresolved again.
    assert "gone" in {u["name"] for u in cm.store.unresolved_names()}


def test_an_older_schema_is_migrated():
    path = os.path.join(tempfile.mkdtemp(), "codemap.sqlite3")
    db = sqlite3.connect(path)
    db.executescript(
        "CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT "
        "NULL, lang TEXT, mtime REAL, hash TEXT, indexed_at REAL);"
        "CREATE TABLE symbols (id INTEGER PRIMARY KEY, file_id INTEGER, "
        "name TEXT, kind TEXT, line INTEGER, end_line INTEGER, signature "
        "TEXT, docstring TEXT, parent_id INTEGER);"
        "CREATE TABLE edges (src_symbol_id INTEGER, dst_symbol_id INTEGER, "
        "kind TEXT);"
        "CREATE TABLE unresolved (src_symbol_id INTEGER, name TEXT, "
        "kind TEXT);")
    db.commit()
    db.close()
    store = Store(path)
    store.put_file("a.py", "python", "def f():\n    pass\n", [], [], [])
    assert store.meta("schema_version") == str(store_mod.SCHEMA_VERSION)
    store.close()


def test_the_store_tolerates_a_second_writer():
    path = os.path.join(tempfile.mkdtemp(), "codemap.sqlite3")
    store = Store(path)
    mode = store.db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    assert store.db.execute("PRAGMA busy_timeout").fetchone()[0] > 0
    store.close()


def test_a_database_error_while_indexing_is_one_sentence():
    events = _Events()
    cm = CodeMap(MemoryFileSystem({"a.py": b"x = 1\n"}), MemoryStorage(),
                 events=events)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    cm.store.put_file = locked
    assert cm.index_file("a.py") == 0
    assert cm.index_file("a.py", force=True) == 0
    said = [r for r in events.rows if "locked" in str(r[1])]
    assert len(said) == 1, events.rows
    assert "Traceback" not in said[0][1]
