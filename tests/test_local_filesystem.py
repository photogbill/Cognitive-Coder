# SPDX-License-Identifier: Apache-2.0
"""LocalFileSystem writes keep the file's permissions (M15 without harm).

Atomic replace is `mkstemp` + `os.replace`, and `mkstemp` creates 0600. So
every edit through the engine used to reset the target's mode: a 755
`build.sh` lost its execute bit (and the next `./build.sh` failed for no
visible reason), a 644 file became unreadable to the group, and a 444 file
the operator had marked read-only came back writable.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import patcher  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    LocalFileSystem,
    MemoryStorage,
)
from cognitive_coder.types import Edit  # noqa: E402

posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX permission bits; Windows has only the read-only flag")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@posix_only
@pytest.mark.parametrize("mode", [0o755, 0o644, 0o640, 0o444])
def test_an_edit_keeps_the_files_mode(tmp_path, mode):
    target = tmp_path / "build.sh"
    target.write_text("#!/bin/sh\necho hi\n")
    os.chmod(target, mode)
    fs = LocalFileSystem(str(tmp_path))
    p = patcher.Patcher(fs, MemoryStorage(str(tmp_path / ".s")),
                        AutoApprove())
    tx = p.begin("mode")
    result = tx.apply([Edit(path="build.sh", kind="replace", old="echo hi",
                            new="echo bye")])[0]
    assert result.ok, result.reason
    assert target.read_text() == "#!/bin/sh\necho bye\n"
    assert _mode(target) == mode, f"{mode:o} became {_mode(target):o}"
    tx.rollback()
    assert _mode(target) == mode, "rollback changed the mode"


@posix_only
def test_a_new_file_gets_the_umask_mode_not_0600(tmp_path):
    old = os.umask(0o022)
    try:
        fs = LocalFileSystem(str(tmp_path))
        fs.write_bytes("new.py", b"x = 1\n")
    finally:
        os.umask(old)
    assert _mode(tmp_path / "new.py") == 0o644


def test_a_failed_replace_names_the_file(tmp_path, monkeypatch):
    """A PermissionError from the replace (Windows: the file is open or
    read-only) must say WHICH file and what to do, as a sentence (C6)."""
    fs = LocalFileSystem(str(tmp_path))
    fs.write_bytes("locked.py", b"x = 1\n")

    def refuse(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError) as exc:
        fs.write_bytes("locked.py", b"x = 2\n")
    text = str(exc.value)
    assert "locked.py" in text and "Nothing was written" in text
    assert (tmp_path / "locked.py").read_bytes() == b"x = 1\n"
    assert [p.name for p in tmp_path.iterdir()] == ["locked.py"], (
        "the temp file was left behind")
