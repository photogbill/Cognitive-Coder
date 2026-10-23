# SPDX-License-Identifier: Apache-2.0
"""Tracebacks that are not the code's fault are answered without a model.

2026-10-22: four of 26 Debug & Repair sessions on one project were the
program started from the wrong folder — `attempted relative import with no
known parent package`, `No module named 'src'` — and every one of them
spent model time repairing imports that were right. The answer is the
command that runs it, before anything is located.
"""

from __future__ import annotations

from pathlib import Path

from cognitive_coder import nonerror
from cognitive_coder import (AutoApprove, DebugReport, Host, LocalFileSystem,
                             MemoryStorage, RecordingEvents, ScriptedLLM,
                             Session, SessionConfig, SubprocessExec)


def _project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "models.py").write_text("X = 1\n")
    (tmp_path / "src" / "main.py").write_text(
        "from .models import X\n\nprint(X)\n")
    return tmp_path


def _session(root: Path, replies: list[str]) -> Session:
    llm = ScriptedLLM(replies, supports_tools=False)
    host = Host(llm=llm, fs=LocalFileSystem(str(root)), exec=SubprocessExec(),
                storage=MemoryStorage(str(root / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    return Session(host, config=SessionConfig(attempts=2,
                                              review_after_build=False))


REL = ('D:\\proj\\src>python main.py\n'
       'Traceback (most recent call last):\n'
       '  File "D:\\proj\\src\\main.py", line 1, in <module>\n'
       '    from .models import X\n'
       'ImportError: attempted relative import with no known parent package\n')

NOSRC = ('D:\\proj>python src\\main.py\n'
         'Traceback (most recent call last):\n'
         '  File "D:\\proj\\src\\main.py", line 1, in <module>\n'
         '    from src.models import X\n'
         "ModuleNotFoundError: No module named 'src'\n")

PYGAME = ('Traceback (most recent call last):\n'
          '  File "D:\\proj\\src\\render.py", line 2, in <module>\n'
          '    import pygame\n'
          "ModuleNotFoundError: No module named 'pygame'\n")

REAL = ('Traceback (most recent call last):\n'
        '  File "D:\\proj\\src\\main.py", line 3, in <module>\n'
        '    print(Y)\n'
        "NameError: name 'Y' is not defined\n")


def test_relative_import_names_the_module_command(tmp_path):
    root = _project(tmp_path)
    v = nonerror.classify(REL, LocalFileSystem(str(root)))
    assert v is not None and v.kind == "relative-import"
    assert v.command == "python -m src.main"
    assert "not a code error" in v.sentence
    assert "python -m src.main" in v.sentence
    assert "break the test runner" in v.sentence     # never "fix" the imports


def test_the_projects_own_package_not_on_path(tmp_path):
    root = _project(tmp_path)
    v = nonerror.classify(NOSRC, LocalFileSystem(str(root)))
    assert v is not None and v.kind == "not-on-path"
    assert v.command == "python -m src.main"


def test_a_missing_third_party_package_is_an_install(tmp_path):
    root = _project(tmp_path)
    v = nonerror.classify(PYGAME, LocalFileSystem(str(root)))
    assert v is not None and v.kind == "missing-package"
    assert v.package == "pygame"
    assert "pip install pygame" in v.sentence


def test_a_real_error_is_none(tmp_path):
    root = _project(tmp_path)
    assert nonerror.classify(REAL, LocalFileSystem(str(root))) is None
    assert nonerror.classify("", LocalFileSystem(str(root))) is None


def test_debug_and_repair_answers_without_a_model(tmp_path):
    """The whole path: a pasted non-error ends the session with the
    command, no round runs, no file changes, and the model is never asked
    (a ScriptedLLM with no replies would raise if it were)."""
    root = _project(tmp_path)
    session = _session(root, [])
    before = (root / "src" / "main.py").read_text()
    result = session.debug(DebugReport(error_text=REL))
    assert result.state == "nothing-to-repair"
    assert "python -m src.main" in result.verdict
    assert result.rounds == []
    assert (root / "src" / "main.py").read_text() == before


def test_a_named_file_still_repairs(tmp_path):
    """The operator who NAMES a file has decided it is a code problem; the
    classifier steps aside and the normal path runs (and here stops at the
    model, which has no replies)."""
    root = _project(tmp_path)
    session = _session(root, [])
    result = session.debug(DebugReport(error_text=REL, files=["src/main.py"]))
    assert result.state != "nothing-to-repair"
