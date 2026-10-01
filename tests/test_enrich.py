# SPDX-License-Identifier: Apache-2.0
"""Rung 1 of the escalation ladder: the repair prompt carries the truth.

Every case here is an error from the Aug 8 / Oct 1 builds of the racing
spec, with the fact the model needed and never got.
"""

from __future__ import annotations

import pytest

from cognitive_coder import enrich
from cognitive_coder.codemap import CodeMap
from cognitive_coder.ports import (
    LocalFileSystem,
    MemoryFileSystem,
    MemoryStorage,
)
from cognitive_coder.types import Diagnostic

PHYSICS = '''
class CarPhysics:
    def __init__(self):
        self.speed = 0.0
        self.max_speed = 10.0
        self.horizontal_position = 0.0

    def update(self, dt: float, input_accelerate: bool, input_brake: bool,
               input_steer: float) -> None:
        pass
'''

RENDER = '''
class Renderer:
    def __init__(self, screen_width: int, screen_height: int, fov: float):
        self.screen_width = screen_width

    def render(self, screen, track_segments, car_physics) -> None:
        pass
'''

MATH3D = '''
from typing import NamedTuple


class ProjectedSegment(NamedTuple):
    x: float
    y: float
    width: float
'''

TRACK = '''
class TrackBuilder:
    def build_track(self) -> list:
        return []
'''


@pytest.fixture
def codemap():
    fs = MemoryFileSystem({k: v.encode() for k, v in {
        "src/__init__.py": "", "src/physics.py": PHYSICS,
        "src/render.py": RENDER, "src/math3d.py": MATH3D,
        "src/track.py": TRACK}.items()})
    cm = CodeMap(fs, MemoryStorage())
    cm.index_project()
    return cm


def _diag(message: str, excerpt: str = "") -> Diagnostic:
    return Diagnostic(file="src/main.py", line=1, severity="exception",
                      message=message, source_excerpt=excerpt, tool="python")


def test_import_of_a_name_the_module_lacks_shows_what_it_has(codemap):
    """Aug 8, session 2: `cannot import name 'CarState' from 'src.physics'`."""
    out = enrich.facts_for(
        [_diag("ImportError: cannot import name 'CarState' from "
               "'src.physics' (D:\\\\x\\\\src\\\\physics.py)")],
        codemap=codemap, fs=codemap.fs)
    assert out.startswith(enrich.HEADER)
    assert "`src.physics` (src/physics.py) does not define `CarState`" in out
    assert "Did you mean `CarPhysics`?" in out
    assert "class CarPhysics" in out
    assert "def __init__(self)" in out
    assert "max_speed: float = 10.0" in out


def test_an_unexpected_keyword_shows_the_constructor_and_its_state(codemap):
    """Oct 1: `CarPhysics.__init__() got an unexpected keyword argument
    'max_speed'` — the test guessed the attributes were parameters."""
    out = enrich.facts_for(
        [_diag("TypeError: CarPhysics.__init__() got an unexpected keyword "
               "argument 'max_speed'")],
        codemap=codemap, fs=codemap.fs)
    assert "`CarPhysics.__init__` has no parameter `max_speed`" in out
    assert "def __init__(self)" in out
    assert "# instance attributes:" in out and "max_speed" in out


def test_a_namedtuple_keyword_shows_its_fields(codemap):
    """`ProjectedSegment.__new__() got an unexpected keyword argument 'z'`."""
    out = enrich.facts_for(
        [_diag("TypeError: ProjectedSegment.__new__() got an unexpected "
               "keyword argument 'z'")],
        codemap=codemap, fs=codemap.fs)
    assert "# construct: ProjectedSegment(x, y, width)" in out


def test_a_missing_positional_argument_shows_the_exact_signature(codemap):
    """Oct 1 main.py: `Renderer.render() missing 1 required positional
    argument: 'car_physics'` — the real omission was `screen`."""
    out = enrich.facts_for(
        [_diag("TypeError: Renderer.render() missing 1 required positional "
               "argument: 'car_physics'")],
        codemap=codemap, fs=codemap.fs)
    assert "does not match its signature" in out
    assert "def render(self, screen, track_segments, car_physics) -> None" \
        in out
    assert "src/render.py:" in out


def test_a_missing_attribute_shows_the_class_and_the_nearest_name(codemap):
    """Oct 1 render.py: `car_physics.offset` on a class whose attribute is
    `horizontal_position`."""
    out = enrich.facts_for(
        [_diag("AttributeError: 'CarPhysics' object has no attribute "
               "'offset'")],
        codemap=codemap, fs=codemap.fs)
    assert "`CarPhysics` has no attribute `offset`" in out
    assert "horizontal_position" in out


def test_a_name_error_says_where_the_name_lives(codemap):
    out = enrich.facts_for(
        [_diag("NameError: name 'TrackBuilder' is not defined")],
        codemap=codemap, fs=codemap.fs)
    assert "defined in src/track.py" in out
    assert "from src.track import TrackBuilder" in out


def test_a_name_nothing_defines_is_said_plainly(codemap):
    out = enrich.facts_for(
        [_diag("NameError: name 'generate_track' is not defined")],
        codemap=codemap, fs=codemap.fs)
    assert "Nothing in this project defines `generate_track`" in out


def test_a_missing_module_lists_the_real_ones(codemap):
    out = enrich.facts_for(
        [_diag("ModuleNotFoundError: No module named 'physics'")],
        codemap=codemap, fs=codemap.fs)
    assert "There is no module `physics`" in out
    assert "from src.physics import" in out


def test_errors_that_name_nothing_produce_nothing(codemap):
    out = enrich.facts_for(
        [_diag("ZeroDivisionError: division by zero")],
        codemap=codemap, fs=codemap.fs)
    assert out == ""


def test_only_python_shapes_are_recognised(codemap):
    out = enrich.facts_for(
        [_diag("ImportError: cannot import name 'X' from 'src.physics'")],
        codemap=codemap, fs=codemap.fs, lang="rust")
    assert out == ""


def test_a_file_the_codemap_has_not_indexed_is_read_directly(tmp_path):
    fs = LocalFileSystem(str(tmp_path))
    fs.write("src/physics.py", PHYSICS)
    out = enrich.facts_for(
        [_diag("ImportError: cannot import name 'CarState' from "
               "'src.physics'")],
        codemap=None, fs=fs)
    assert "class CarPhysics" in out
