# SPDX-License-Identifier: Apache-2.0
"""Reading a build request from a file, and previewing what it produced.

A sentence types fast and plans badly. Everything that decides whether a build
goes well — which modules exist, what each owns, what must not import what,
which tests must be written — is thinking done before the model is asked for
anything, and it does not fit in a shell argument. The first real
specification put through this engine was sixty lines in four numbered
sections, typed into a field that showed one line at a time.

The preview exists for a narrower reason. A specification headed "Testing
Requirements (Strict)" named two test files; the planner proposed five source
files and no tests; the run finished reporting, truthfully and repeatedly,
that the test command had run zero tests. Every fact needed to catch that
existed one small completion in. Nobody was shown them side by side.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import spec as spec_mod                  # noqa: E402
from cognitive_coder.ports import (                           # noqa: E402
    LocalFileSystem,
    MemoryStorage,
    ScriptedLLM,
    SubprocessExec,
)
from cognitive_coder.ports import Host                        # noqa: E402
from cognitive_coder.session import Session, SessionConfig    # noqa: E402

#: Bill's specification, trimmed to the parts that carry structure.
RACING = """Build Specification: Pseudo-3D Racing Game (Pole Position Style)
1. Project Overview
Build a retro, pseudo-3D racing game using pygame, in the style of classic
16-bit arcade racers, at version 1.1 of this document.
2. Architectural Directives
src/math3d.py: Pure logic for raster projecting 3D track segments.
src/physics.py: The player's car state.
src/track.py: The data model representing the track layout.
src/render.py: Pygame-specific drawing functions.
src/main.py: The Pygame event loop and game initialization.
3. Module Specifications
Rule: This module MUST NOT import pygame.
4. Testing Requirements (Strict)
Headless Execution: Test files (tests/test_math3d.py, tests/test_physics.py)
MUST NOT initialize a pygame display or require a window system.
"""

#: The plan Ministral-24B actually produced for that specification: five
#: source files, main.py first, and not one test.
MINISTRAL_PLAN = (
    "src/main.py - Pygame event loop and game initialization\n"
    "src/math3d.py - 3D to 2D projection of track segments\n"
    "src/physics.py - Player car state and physics simulation\n"
    "src/render.py - Pygame-specific rendering of projected segments\n"
    "src/track.py - Track data model with segments and curve definitions\n")


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / "plan.md"
    path.write_text(RACING, encoding="utf-8")
    return path


# ==========================================================================
# loading
# ==========================================================================

def test_a_specification_loads_and_describes_itself(spec_file):
    sp = spec_mod.load(spec_file)
    assert sp.title.startswith("Build Specification: Pseudo-3D Racing Game")
    assert sp.required_tests == ("tests/test_math3d.py",
                                 "tests/test_physics.py")
    assert list(sp.mentioned_paths) == [
        "src/math3d.py", "src/physics.py", "src/track.py",
        "src/render.py", "src/main.py"]
    assert sp.approx_tokens > 50
    assert sp.warnings == ()


def test_prose_that_looks_like_a_path_is_not_one(spec_file):
    """`Pseudo-3D`, `16-bit` and `version 1.1` must not become files.

    A preview listing imaginary files is worse than one listing none: it
    reads as though the engine understood something it did not.
    """
    sp = spec_mod.load(spec_file)
    for junk in ("Pseudo-3D", "16-bit", "1.1", "3D"):
        assert not any(junk in p for p in sp.mentioned_paths), junk


def test_tests_are_reported_separately_from_sources(spec_file):
    sp = spec_mod.load(spec_file)
    assert not set(sp.mentioned_paths) & set(sp.required_tests)


def test_front_matter_is_stripped_and_said_so(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("---\ntitle: x\nauthor: y\n---\n# Real Heading\nBuild it.",
                    encoding="utf-8")
    sp = spec_mod.load(path)
    assert "author: y" not in sp.text
    assert sp.title == "Real Heading"
    assert any("front matter" in w for w in sp.warnings)


def test_a_plain_text_spec_takes_its_first_line_as_the_title(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("Make a CSV summariser\n\nIt should read a file.",
                    encoding="utf-8")
    assert spec_mod.load(path).title == "Make a CSV summariser"


@pytest.mark.parametrize("name,content,fragment", [
    ("missing.md", None, "does not exist"),
    ("empty.md", "", "is empty"),
    ("blank.md", "   \n\n  \n", "nothing in it but whitespace"),
    ("notes.docx", "x", "not a text specification"),
])
def test_a_file_that_cannot_be_used_says_why(tmp_path, name, content,
                                             fragment):
    """Errors are sentences, not tracebacks (C6).

    This runs at the very start of a long operation; a stack trace for a
    mistyped filename costs the whole run.
    """
    path = tmp_path / name
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(spec_mod.SpecError) as exc:
        spec_mod.load(path)
    assert fragment in str(exc.value)


def test_a_folder_is_refused_clearly(tmp_path):
    with pytest.raises(spec_mod.SpecError, match="folder, not a"):
        spec_mod.load(tmp_path)


def test_a_spec_too_large_for_context_is_flagged_not_truncated():
    sp = spec_mod.from_text("x " * 30_000)
    assert any("context" in w for w in sp.warnings)
    # Flagged, never silently shortened: the operator decides.
    assert len(sp.text) > 40_000


def test_undecodable_bytes_are_replaced_and_reported(tmp_path):
    path = tmp_path / "p.md"
    path.write_bytes("Build a thing \xff\xfe with smart quotes".encode("latin-1"))
    sp = spec_mod.load(path)
    assert sp.text
    assert any("could not be decoded" in w for w in sp.warnings)


# ==========================================================================
# preview
# ==========================================================================

def _session(tmp_path, replies):
    host = Host(fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=None, llm=ScriptedLLM(replies))
    return Session(host, config=SessionConfig(skeleton_first=True))


def test_preview_shows_the_build_order_not_the_models_order(tmp_path,
                                                            spec_file):
    """The plan that failed, previewed before it can fail again."""
    sp = spec_mod.load(spec_file)
    pv = _session(tmp_path, [MINISTRAL_PLAN] * 8).preview(sp.text)
    assert pv["files"][-1] == "src/main.py", pv["files"]
    assert pv["files"].index("src/physics.py") < pv["files"].index(
        "src/main.py")


def test_preview_puts_tests_required_beside_tests_planned(tmp_path,
                                                          spec_file):
    """The two numbers whose disagreement went unnoticed for a whole build."""
    sp = spec_mod.load(spec_file)
    pv = _session(tmp_path, [MINISTRAL_PLAN] * 8).preview(sp.text)
    assert len(pv["tests_required"]) == 2
    assert len(pv["tests_planned"]) == 2
    # The planner repaired it; the preview is how anyone can tell it did.
    assert pv["tests_missing"] == []
    assert any("left out" in c for c in pv["caveats"])


def test_preview_reports_the_context_cost(tmp_path, spec_file):
    pv = _session(tmp_path, [MINISTRAL_PLAN] * 8).preview(
        spec_mod.load(spec_file).text)
    assert pv["approx_tokens"] > 0
    assert pv["context_tokens"] >= 0
    assert pv["title"].startswith("Build Specification")


def test_preview_writes_nothing_that_a_build_would_write(tmp_path,
                                                         spec_file):
    """It plans. It must not leave source behind.

    The skeleton step writes stubs, and preview deliberately does not run it —
    ordering is derived directly instead. A preview that scaffolds a project
    is a build with a misleading name.
    """
    sp = spec_mod.load(spec_file)
    _session(tmp_path, [MINISTRAL_PLAN] * 8).preview(sp.text)
    assert not (tmp_path / "src").exists()
    assert list(tmp_path.glob("src/*.py")) == []


def test_a_typed_request_previews_too(tmp_path):
    """--spec is not required; a sentence still works, and says it has no
    tests, which is the honest answer rather than an empty section."""
    pv = _session(tmp_path, ["src/thing.py - does the thing\n"] * 8).preview(
        "build me a thing")
    assert pv["files"] == ["src/thing.py"]
    assert pv["tests_required"] == []
    assert pv["tests_planned"] == []
