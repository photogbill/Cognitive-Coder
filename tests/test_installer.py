# SPDX-License-Identifier: Apache-2.0
"""The installers' bytes: line endings, syntax, and the uv pin.

`.gitattributes` makes git WRITE `.bat` files with CRLF on checkout. This
checks the same property from the other side, as its header promises: a
file that arrives LF-only — written by an editor on Linux, copied out of an
archive, committed with the attribute bypassed — is caught by the suite.
It has to be: on 2026-08-07 every `.bat` here was LF-only, cmd.exe ran each
as a stream of simple commands, skipped every `call :label` and bracketed
block, and nothing failed, so nothing said so.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
BATCH = sorted(p for pattern in ("*.bat", "*.cmd")
               for p in ROOT.rglob(pattern) if ".venv" not in p.parts)
SHELL = sorted(p for p in ROOT.rglob("*.sh") if ".venv" not in p.parts)


def test_there_is_something_to_check():
    assert {p.name for p in BATCH} >= {"install.bat", "push.bat"}
    assert "install.sh" in {p.name for p in SHELL}


@pytest.mark.parametrize("path", BATCH, ids=lambda p: p.name)
def test_every_batch_line_ends_in_crlf(path):
    data = path.read_bytes()
    bare = [n for n, line in enumerate(data.split(b"\n")[:-1], 1)
            if not line.endswith(b"\r")]
    assert not bare, (f"{path.name}: {len(bare)} line(s) end in LF alone, "
                      f"first at line {bare[0]}. cmd.exe seeks labels by "
                      f"byte offset and will skip what it cannot find.")


@pytest.mark.parametrize("path", SHELL, ids=lambda p: p.name)
def test_every_shell_line_ends_in_lf(path):
    assert b"\r" not in path.read_bytes(), (
        f"{path.name} has a carriage return; sh reads it as part of the "
        f"command and fails with 'not found'.")


@pytest.mark.skipif(shutil.which("sh") is None, reason="no sh here")
def test_install_sh_parses():
    result = subprocess.run(["sh", "-n", str(ROOT / "install.sh")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _pin(text: str, name: str) -> str:
    match = re.search(rf'{name}="?([0-9a-f.]+)"?', text)
    assert match, f"{name} is not set"
    return match.group(1)


def test_both_installers_pin_the_same_uv_and_verify_it():
    sh = (ROOT / "install.sh").read_text(encoding="utf-8")
    bat = (ROOT / "install.bat").read_bytes().decode("utf-8")
    assert _pin(sh, "UV_VERSION") == _pin(bat, "UV_VERSION")
    assert len(_pin(sh, "UV_INSTALLER_SHA256")) == 64
    assert len(_pin(bat, "UV_PS1_SHA256")) == 64
    # Fetched from the pinned release and checked — never piped to a shell.
    # (Code lines only: the comments quote the old pipe on purpose.)
    sh_code = [ln for ln in sh.splitlines()
               if not ln.lstrip().startswith("#")]
    bat_code = [ln for ln in bat.splitlines()
                if not ln.lstrip().upper().startswith("REM")]
    assert not any(re.search(r"\|\s*(ba)?sh\b", ln) for ln in sh_code)
    assert not any(re.search(r"\|\s*iex\b", ln) for ln in bat_code)
    for text in (sh, bat):
        assert "UV_PYTHON_INSTALL_DIR" in text, "the Python lands in .python/"
