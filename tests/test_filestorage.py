# SPDX-License-Identifier: Apache-2.0
"""`JsonFileStorage` — a StoragePort whose state outlives the process.

Why it exists: `ccoder history` always answered "Nothing has been changed"
while `.cc_snapshots/0001-t1/` sat on disk beside it. The CLI used
`MemoryStorage`, and the patcher keeps its sequence counter and its
transaction log in the StoragePort — so both died with the `build` process,
and every later command started from an empty log. Worse than an empty
answer: the next build restarted the counter at 1, so M25's "the numbering
proves the log is linear" was false across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import port_conformance  # noqa: E402

from cognitive_coder import JsonFileStorage, PortError  # noqa: E402
from cognitive_coder.patcher import Patcher  # noqa: E402
from cognitive_coder.ports import AutoApprove, MemoryFileSystem  # noqa: E402
from cognitive_coder.types import Edit  # noqa: E402


def test_it_passes_the_storage_conformance_kit(tmp_path):
    report = port_conformance.check_storage(JsonFileStorage(str(tmp_path)))
    assert report.ok, report.text()


def test_a_value_survives_into_a_second_instance(tmp_path):
    """The whole point: a later process sees what an earlier one wrote."""
    JsonFileStorage(str(tmp_path)).set("cognitive_coder.patcher.seq", 3)
    later = JsonFileStorage(str(tmp_path))
    assert later.get("cognitive_coder.patcher.seq") == 3


def test_a_missing_key_returns_the_default(tmp_path):
    storage = JsonFileStorage(str(tmp_path))
    assert storage.get("absent") is None
    assert storage.get("absent", []) == []


def test_an_unserialisable_value_is_refused_and_nothing_is_written(
        tmp_path):
    storage = JsonFileStorage(str(tmp_path))
    storage.set("k", 1)
    with pytest.raises(ValueError, match="'k' is not"):
        storage.set("k", {"when": object()})
    assert storage.get("k") == 1, "the refused value did not replace it"


def test_values_come_back_as_json_says_they_are(tmp_path):
    storage = JsonFileStorage(str(tmp_path))
    storage.set("t", {"files": ("a.py", "b.py")})
    assert storage.get("t") == {"files": ["a.py", "b.py"]}


def test_each_key_is_one_readable_json_file_and_no_temp_is_left(tmp_path):
    storage = JsonFileStorage(str(tmp_path))
    storage.set("cognitive_coder.patcher.log", [{"seq": 1}])
    storage.set("cognitive_coder.patcher.log", [{"seq": 1}, {"seq": 2}])
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["cognitive_coder.patcher.log.json"]
    assert json.loads((tmp_path / names[0]).read_text("utf-8")) == [
        {"seq": 1}, {"seq": 2}]


@pytest.mark.parametrize("key", ["a/b", "a_b", "../escape", "con", "NUL.x",
                                 "", "x" * 300])
def test_any_key_lands_inside_the_directory_in_its_own_file(tmp_path, key):
    storage = JsonFileStorage(str(tmp_path / "state"))
    storage.set(key, key)
    assert storage.get(key) == key
    for path in (tmp_path / "state").iterdir():
        assert len(path.name) < 160
    assert not (tmp_path / "escape.json").exists()


def test_keys_that_differ_only_in_punctuation_do_not_collide(tmp_path):
    storage = JsonFileStorage(str(tmp_path))
    storage.set("a/b", 1)
    storage.set("a_b", 2)
    storage.set("a:b", 3)
    assert (storage.get("a/b"), storage.get("a_b"), storage.get("a:b")) == (
        1, 2, 3)


def test_a_corrupt_file_is_a_sentence_not_a_silent_default(tmp_path):
    """Returning the default here would restart the patcher's counter at 1
    and silently duplicate sequence numbers — worse than stopping."""
    storage = JsonFileStorage(str(tmp_path))
    storage.set("cognitive_coder.patcher.seq", 4)
    (tmp_path / "cognitive_coder.patcher.seq.json").write_text("{half")
    with pytest.raises(PortError) as err:
        storage.get("cognitive_coder.patcher.seq")
    assert "cognitive_coder.patcher.seq.json" in str(err.value)


def test_sqlite_path_is_in_the_same_directory_and_stable(tmp_path):
    storage = JsonFileStorage(str(tmp_path / "state"))
    path = Path(storage.sqlite_path("codemap"))
    assert path.parent == (tmp_path / "state").resolve()
    assert path.name == "codemap.sqlite3"
    assert storage.sqlite_path("codemap") == str(path)


def test_the_directory_is_made_on_construction(tmp_path):
    JsonFileStorage(str(tmp_path / "deep" / "state"))
    assert (tmp_path / "deep" / "state").is_dir()


def test_the_patchers_sequence_continues_across_processes(tmp_path):
    """M25: one linear log, numbered once, across every run."""
    fs = MemoryFileSystem()
    first = Patcher(fs, JsonFileStorage(str(tmp_path)), AutoApprove())
    tx = first.begin("t1")
    tx.apply([Edit(path="a.py", kind="whole", new="a = 1\n")])
    tx.commit(verified=True)

    later = Patcher(fs, JsonFileStorage(str(tmp_path)), AutoApprove())
    assert [r.seq for r in later.history()] == [1]
    assert later.begin("t2").seq == 2
