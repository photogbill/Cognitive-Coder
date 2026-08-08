# SPDX-License-Identifier: Apache-2.0
"""Four defects found by running this engine against a real specification.

On 2026-08-07 the same pseudo-3D racing specification was built twice with
different local models, and both projects and console logs were kept. Reading
those logs, and then running the generated code, exposed four separate faults.
Each is pinned here against the evidence that found it.

The two runs, because every assertion below is really about them::

    Devstral-24B   plan: math3d -> physics -> track -> render -> main
                   3 of 5 committed, imports clean, crashes on frame one:
                   CarPhysics.update() takes a `curve` argument that main.py
                   never passes.

    Ministral-24B  plan: main -> math3d -> physics -> render -> track
                   3 of 5 committed. render.py and main.py never import at
                   all, because TrackSegment does not exist in track.py.

The gap between them was mostly the order the planner happened to propose,
which is what [1] is about — and the mechanism meant to remove that luck had
been a no-op since it was written.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.codemap import parse_python as pp        # noqa: E402
from cognitive_coder.planner import Planner, _required_tests  # noqa: E402
from cognitive_coder.review import (                          # noqa: E402
    ReviewResult,
    recommendation_document,
)
from cognitive_coder.types import Plan, Task                  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

#: The two plans exactly as the models proposed them.
MINISTRAL = [
    ("src/main.py", "Pygame event loop and game initialization"),
    ("src/math3d.py", "3D to 2D projection of track segments for raster"),
    ("src/physics.py", "Player car state and physics simulation"),
    ("src/render.py", "Pygame-specific rendering of projected track segments"),
    ("src/track.py", "Track data model with segments and curve definitions"),
]
DEVSTRAL = [
    ("src/math3d.py", "pure 3D-to-2D projection logic for track segments"),
    ("src/physics.py", "car state and update logic with centrifugal force"),
    ("src/track.py", "track segment data model with curve and color patterns"),
    ("src/render.py", "pygame-specific drawing using projected coordinates"),
    ("src/main.py", "pygame event loop, input handling, and game init"),
]

SPEC_TESTS = (
    "Testing Requirements (Strict) The engine must implement tests before "
    "writing the module bodies, adhering to the following rules: Headless "
    "Execution: Test files (tests/test_math3d.py, tests/test_physics.py) "
    "MUST NOT initialize a pygame display or require a window system.")


class _FS:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def read(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def exists(self, path):
        return path in self.files

    def write(self, path, text):
        self.files[path] = text


class _Host:
    def __init__(self, fs=None):
        self.fs = fs or _FS()
        self.emitted = []

    def emit(self, kind, message, meta=None):
        self.emitted.append((kind, message))


@pytest.fixture
def planner():
    def _make(files=None):
        p = object.__new__(Planner)
        p.lang, p.journal, p.codemap = "python", None, None
        p.host = _Host(_FS(files))
        return p
    return _make


def _plan(rows):
    return Plan(request="r", tasks=tuple(
        Task(id=f"t{i + 1}", path=path, purpose=purpose, test_path="",
             persona="engineer", lang="python", atomic=False)
        for i, (path, purpose) in enumerate(rows)))


# ==========================================================================
# [1] the ordering step was a no-op on every project ever built
# ==========================================================================
#
# `derive_order` reads imports off disk. It ran only AFTER `skeleton()`, and
# `stub_for` writes a stub's imports from `depends_on` — which `derive_order`
# is what populates. Empty in, empty out: it learned nothing and returned the
# model's order untouched. It failed silently for the best possible reason, in
# that a function returning *an* order looks like it worked.

def test_an_entry_point_is_ordered_last(planner):
    out = [t.path for t in planner().derive_order(_plan(MINISTRAL)).tasks]
    assert out[-1] == "src/main.py"
    assert out.index("src/main.py") > out.index("src/physics.py")


def test_a_correct_plan_is_left_alone(planner):
    out = [t.path for t in planner().derive_order(_plan(DEVSTRAL)).tasks]
    assert out == [path for path, _ in DEVSTRAL]


def test_an_entry_point_is_recognised_by_purpose(planner):
    rows = [("src/zzz.py", "the game loop and startup"),
            ("src/lib.py", "helpers")]
    out = [t.path for t in planner().derive_order(_plan(rows)).tasks]
    assert out[-1] == "src/zzz.py"


def test_real_imports_override_the_heuristic(planner):
    p = planner({"src/main.py": "from src.track import Track\n",
                 "src/track.py": "from src.math3d import project\n",
                 "src/math3d.py": "", "src/physics.py": "",
                 "src/render.py": ""})
    out = [t.path for t in p.derive_order(_plan(MINISTRAL)).tasks]
    assert out.index("src/math3d.py") < out.index("src/track.py")
    assert out.index("src/track.py") < out.index("src/main.py")


def test_a_test_file_follows_the_module_it_covers(planner):
    rows = MINISTRAL + [("tests/test_physics.py", "physics tests")]
    out = [t.path for t in planner().derive_order(_plan(rows)).tasks]
    assert out.index("src/physics.py") < out.index("tests/test_physics.py")


def test_ordering_happens_before_the_skeleton_is_written():
    """The deadlock itself, asserted structurally.

    If the skeleton is built first its stubs carry no imports, and every later
    ordering pass reads them and learns nothing. The sequence is the fix.
    """
    src = (REPO / "cognitive_coder" / "session.py").read_text(encoding="utf-8")
    assert 0 < src.find("derive_order") < src.find("self.planner.skeleton")


# ==========================================================================
# [2] a requirement in the request was silently dropped
# ==========================================================================
#
# The specification had a section headed "Testing Requirements (Strict)"
# naming two test files. Both models proposed five files, all under src/, no
# tests. Nothing checked. Every build step then reported, truthfully, "the
# test command succeeded but ran ZERO tests" — about ten times, describing a
# symptom three steps downstream of its cause. With no tests, verification
# degraded to "it imports", which is how a physics module was committed green
# with an update() signature no caller satisfied.

def test_named_test_files_are_extracted():
    assert _required_tests(SPEC_TESTS) == ["tests/test_math3d.py",
                                           "tests/test_physics.py"]


@pytest.mark.parametrize("text", [
    "src/latest_data.py", "tests/helpers.py", "src/contest_ui.py",
    "protest.py", "src/fastest.py", "manifest.py",
])
def test_ordinary_paths_are_not_mistaken_for_tests(text):
    assert _required_tests(text) == []


@pytest.mark.parametrize("text,want", [
    ("write test_foo.py", ["test_foo.py"]),
    ("src/thing.test.ts", ["src/thing.test.ts"]),
    ("pkg/mod_test.go", ["pkg/mod_test.go"]),
    ("spec/thing_spec.rb", ["spec/thing_spec.rb"]),
])
def test_the_common_conventions_are_recognised(text, want):
    assert _required_tests(text) == want


def test_omitted_tests_are_added_and_announced(planner):
    p = planner()
    out, added = p._ensure_required_tests(SPEC_TESTS,
                                          list(_plan(DEVSTRAL).tasks), {})
    assert added == ["tests/test_math3d.py", "tests/test_physics.py"]
    assert all(t.persona == "tester" for t in out if "test_" in t.path)
    # A silent correction is still a correction nobody can audit.
    assert any("omitted" in message for _kind, message in p.host.emitted)


def test_adding_required_tests_is_idempotent(planner):
    p = planner()
    out, _ = p._ensure_required_tests(SPEC_TESTS,
                                      list(_plan(DEVSTRAL).tasks), {})
    _out2, added2 = p._ensure_required_tests(SPEC_TESTS, out, {})
    assert added2 == []


def test_a_request_naming_no_tests_invents_none(planner):
    _out, added = planner()._ensure_required_tests(
        "just build me a thing", list(_plan(DEVSTRAL).tasks), {})
    assert added == []


# ==========================================================================
# [3] the undefined-name check cried wolf on every file
# ==========================================================================
#
# `unresolved_in` treated the head of every dotted call as a name the project
# should define. A local variable is not a symbol, so `screen.fill` — where
# `screen` came from `pygame.display.set_mode()` two lines above — was
# reported as undefined. The damage was concealment rather than noise: these
# appeared in the same sentence, same format, as CarState, generate_track,
# render_road and TrackSegment, every one of which was genuinely missing.

LOCALS_CODE = (
    "import pygame\n"
    "from src.physics import CarState\n"
    "def main(fps):\n"
    "    screen = pygame.display.set_mode((800, 600))\n"
    "    clock = pygame.time.Clock()\n"
    "    segments = []\n"
    "    for seg in range(10):\n"
    "        segments.append(seg)\n"
    "    with open('x') as fh:\n"
    "        data = fh.read()\n"
    "    try:\n"
    "        pass\n"
    "    except ValueError as exc:\n"
    "        print(exc)\n"
    "    screen.fill((0, 0, 0))\n"
    "    clock.tick(fps)\n")


@pytest.mark.parametrize("name", [
    "screen", "clock", "segments", "seg", "fh", "data", "exc", "fps", "main",
])
def test_locals_and_parameters_are_recognised_as_bound(name):
    assert name in pp.bound_names(LOCALS_CODE)


@pytest.mark.parametrize("code,name", [
    ("if (n := 1):\n    pass\n", "n"),
    ("xs = [q for q in range(3)]\n", "q"),
    ("def f(a, *args, **kw):\n    pass\n", "kw"),
    ("f = lambda lp: lp\n", "lp"),
    ("class C:\n    pass\n", "C"),
    ("a, (b, c) = 1, (2, 3)\n", "c"),
    ("first, *rest = [1, 2, 3]\n", "rest"),
])
def test_every_binding_form_is_covered(code, name):
    assert name in pp.bound_names(code)


def test_an_attribute_target_binds_no_bare_name():
    bound = pp.bound_names("class C:\n    def f(self):\n        self.x = 1\n")
    assert "x" not in bound
    assert "self" in bound


def test_imports_are_not_treated_as_local_bindings():
    """The regression this fix caused on its first outing.

    `from utils import parse_config` does bind the name — so counting imports
    here silenced `unresolved_in` about a symbol pulled from a module that
    does not exist, which is among the most valuable things it catches.
    Imports are tracked separately and more precisely, because "the module
    resolves" and "the symbol resolves" are different questions.
    """
    assert "parse_config" not in pp.bound_names(
        "from utils import parse_config\n")
    assert "pygame" not in pp.bound_names(LOCALS_CODE)


def test_a_syntax_error_yields_nothing_rather_than_raising():
    assert pp.bound_names("def (:\n") == set()


def test_attribute_calls_on_locals_are_suppressed_end_to_end():
    """The eight names that fired on every build, and the ones that mattered.

    Asserted through the same filtering `unresolved_in` performs, on the code
    shape that produced them.
    """
    text = LOCALS_CODE + "    return generate_track()\n"
    bound = pp.bound_names(text)
    _sym, _edges, unresolved = pp.parse(text, "<generated>")
    local = {s.name for s in pp.parse(text, "<generated>")[0]}
    heads = {str(i).lstrip(".").split(".")[0] for i in pp.imports_of(text)}

    kept = []
    for _src, name, _kind in unresolved:
        raw = str(name)
        if raw.split(".")[0] in heads or raw.split(".")[-1] in local:
            continue
        if "." in raw and raw.split(".")[0] in bound:
            continue
        kept.append(raw)

    for noise in ("screen.fill", "clock.tick", "segments.append"):
        assert noise not in kept, f"{noise} is an attribute on a local"
    assert "generate_track" in kept, "a genuinely missing name must survive"


# ==========================================================================
# [4] the summary contradicted its own body, four lines down
# ==========================================================================
#
# The reviewer is handed the files that were COMMITTED. When a build
# collapsed, the failed files were absent from what it read — so the worse the
# run went, the less there was to criticise. A program that crashed on its
# first frame was summarised as "Nothing was found that should stop this being
# used", directly above a Verification line reading "3 of 5 file(s) built".

def _doc(**kw):
    base = dict(request="Pseudo-3D Racing Game",
                files=["src/math3d.py", "src/physics.py", "src/track.py"],
                build_summary="3 of 5 file(s) built and their tests ran")
    base.update(kw)
    return recommendation_document(ReviewResult(), **base)


def test_an_unfinished_build_leads_the_summary():
    doc = _doc(unfinished=["src/render.py", "src/main.py"])
    assert "did not finish" in doc
    assert "src/render.py" in doc and "src/main.py" in doc
    assert doc.index("did not finish") < doc.index("Overall:")


def test_the_all_clear_is_withheld_when_files_are_missing():
    doc = _doc(unfinished=["src/render.py"])
    assert "Nothing was found that should stop this being used" not in doc


def test_a_partial_review_says_it_is_partial():
    doc = _doc(unfinished=["src/render.py"])
    assert "not an assessment of the program as a whole" in doc
    # Absence of findings about a file nobody reviewed is not a clean bill.
    assert "is not evidence" in doc


def test_a_completed_build_still_gets_its_all_clear():
    doc = _doc(unfinished=[])
    assert "Nothing was found that should stop this being used" in doc
    assert "did not finish" not in doc


def test_the_session_passes_failed_files_to_the_reviewer():
    """Without this wiring the reviewer cannot tell a clean build from a
    collapsed one, and reports the second as the first."""
    src = (REPO / "cognitive_coder" / "session.py").read_text(encoding="utf-8")
    assert "unfinished=[o.path for o in self.outcomes if not o.ok]" in src
