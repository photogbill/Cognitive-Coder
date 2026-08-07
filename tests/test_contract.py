# SPDX-License-Identifier: Apache-2.0
"""C1, C9, M48 and M50 — the constraints a CI walk can prove.

These are the tests that stop the architecture eroding. Every one of them
guards something that would still WORK if it broke, and would only be
discovered by the second host, months later, when it is expensive.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "cognitive_coder"

sys.path.insert(0, str(ROOT))

# The imports that would end the project (C1). The moment the core imports
# Qt, LoLLMs cannot use it, and the whole reason this is a separate module
# evaporates.
FORBIDDEN = ("PySide6", "PyQt5", "PyQt6", "shiboken6", "qtpy",
             "fastapi", "starlette", "flask", "atk", "lollms")


def core_files() -> list[Path]:
    return sorted(p for p in CORE.rglob("*.py"))


def imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_core_imports_no_gui_and_no_host():
    """M1 — enforced by a CI walk of the tree, not by good intentions.

    Conditionally, inside a function, in a try block — none of it is allowed
    (C1). This walks the AST rather than grepping, so a lazily-imported Qt
    inside a helper is caught too.
    """
    offenders = []
    for path in core_files():
        for name in imported_names(path):
            if name.lower() in {f.lower() for f in FORBIDDEN}:
                offenders.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not offenders, "\n".join(offenders)


def test_core_never_imports_adapters():
    """`adapters/` is outside the package precisely so this is checkable."""
    offenders = [str(p.relative_to(ROOT)) for p in core_files()
                 if "adapters" in imported_names(p)]
    assert not offenders, offenders


def test_core_has_no_module_level_third_party_imports():
    """M48 — zero required runtime dependencies, checked mechanically.

    A stdlib-only core is what lets this drop into a Qt desktop app and a
    FastAPI server without an argument about versions. Optional dependencies
    are allowed, but only inside a function, where C7 can catch the
    ImportError and degrade with a stated cost.
    """
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    if not stdlib:                        # pragma: no cover — 3.9 and older
        pytest.skip("sys.stdlib_module_names needs Python 3.10+")
    offenders = []
    for path in core_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:            # MODULE level only
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:            # relative: our own package
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name and name not in stdlib and name != "cognitive_coder":
                    offenders.append(
                        f"{path.relative_to(ROOT)} imports {name} at module "
                        f"level")
    assert not offenders, "\n".join(offenders)


def test_vendored_use_works_without_install():
    """M50 — `sys.path`-insert and import, with no install and no metadata.

    ParisNeo may prefer a git submodule. A subprocess with a clean path is
    the only honest way to test this: importing in-process would pass on a
    machine where the package happens to be installed.
    """
    code = (
        f"import sys; sys.path.insert(0, r'{ROOT}');\n"
        "import cognitive_coder as cc;\n"
        "assert cc.__version__, 'no version';\n"
        "assert cc.Session and cc.Host and cc.LLMPort;\n"
        "print('ok', cc.__version__)\n")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(ROOT.parent), env=env)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_import_has_no_side_effects(tmp_path):
    """M50 — importing must not create a directory or open a database.

    Checked by importing in a subprocess whose cwd is an empty directory and
    asserting the directory is still empty afterwards.
    """
    code = (f"import sys; sys.path.insert(0, r'{ROOT}'); "
            f"import cognitive_coder")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == [], (
        f"importing the package created {[p.name for p in tmp_path.iterdir()]}")


def test_version_is_not_read_from_package_metadata():
    """M50 — `importlib.metadata` raises for a vendored copy.

    Checked against the AST rather than the text, because version.py
    EXPLAINS why it does not use importlib.metadata, and a grep cannot tell
    a warning from a violation.
    """
    tree = ast.parse((CORE / "version.py").read_text(encoding="utf-8"))
    banned = {"importlib", "pkg_resources"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    assert not (found & banned), (
        f"version.py imports {found & banned}; a vendored copy has no "
        f"distribution metadata to read")

    # And nowhere else in the core reads it either.
    offenders = []
    for path in core_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr == "version":
                value = getattr(node.value, "attr", "")
                if value == "metadata":
                    offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_every_port_has_a_null_implementation():
    """M20 — the engine must run hostless."""
    from cognitive_coder import ports
    for name in ("NullLLM", "MemoryFileSystem", "SubprocessExec",
                 "MemoryStorage", "SilentEvents", "AutoApprove"):
        assert hasattr(ports, name), f"{name} is missing"


def test_event_kinds_are_a_closed_set():
    """M19 — hosts render these; additions are minor, renames are major."""
    from cognitive_coder.types import EVENT_KINDS
    assert EVENT_KINDS == ("phase", "token", "status", "diagnostic", "patch",
                           "remote", "warning", "error", "budget")


def test_public_api_exports_the_frozen_surface():
    """C9/M8 — the Ports AND the shared types they carry."""
    import cognitive_coder as cc
    for name in ("LLMPort", "FileSystemPort", "ExecPort", "StoragePort",
                 "EventPort", "ApprovalPort", "Message", "Completion",
                 "ModelCapabilities", "ProcResult", "Diagnostic", "ToolSpec",
                 "ToolCall"):
        assert name in cc.__all__, f"{name} is not in the public API"
        assert hasattr(cc, name)


def test_every_source_file_carries_the_licence_identifier():
    """§10.4 — so a vendored copy stays attributable."""
    missing = [str(p.relative_to(ROOT)) for p in core_files()
               if "SPDX-License-Identifier: Apache-2.0"
               not in p.read_text(encoding="utf-8")[:400]]
    assert not missing, missing


def test_guard_is_never_described_as_a_security_boundary():
    """M9 — say what it is: a screen against mistakes.

    The words matter because someone will read this code and decide how much
    to trust it. Overclaiming here is worse than the gap it papers over.
    """
    text = (CORE / "guard.py").read_text(encoding="utf-8").lower()
    assert "not a security boundary" in text
    assert "accident" in text
    for doc in (ROOT / "README.md", ROOT / "docs" / "PORTS.md"):
        if doc.exists():
            body = doc.read_text(encoding="utf-8").lower()
            assert "security boundary" not in body or \
                "not a security boundary" in body, (
                    f"{doc.name} describes the screen as a security boundary")
