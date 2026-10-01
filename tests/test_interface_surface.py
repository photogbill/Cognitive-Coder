# SPDX-License-Identifier: Apache-2.0
"""The interface a model is shown must carry what a caller needs.

Three builds of the same racing-game specification (Aug 8 and Oct 1, 2026)
failed the same way: the tests called `CarPhysics(max_speed=…)` on a
constructor that takes nothing, `ProjectedSegment(z=…)` on a NamedTuple whose
fields are `x, y, width`, and `main.py` called `renderer.render(track, car)`
on a method that wants `screen` first. In every case the model had been shown
`class CarPhysics` and the public methods — and nothing of the constructor,
the fields or the attributes, because `interface()` filtered every name
starting with `_`, and `__init__` starts with `_`.

These tests pin the surface: constructors, fields, instance attributes,
members under their class, and the same text from both renderers.
"""

from __future__ import annotations

from cognitive_coder import context as ctx
from cognitive_coder.codemap import parse_python

PHYSICS = '''
class CarPhysics:
    def __init__(self):
        self.speed = 0.0
        self.max_speed = 10.0
        self.horizontal_position = 0.0
        self._scratch = None

    def update(self, dt: float, input_accelerate: bool, input_brake: bool,
               input_steer: float) -> None:
        """Advance one tick."""
        self.steering_angle = input_steer

    def _helper(self):
        return 1
'''

MATH3D = '''
from typing import NamedTuple


class ProjectedSegment(NamedTuple):
    x: float
    y: float
    width: float


def project_segment(segment_x: float, segment_z: float,
                    screen_width: int) -> ProjectedSegment:
    """Project one segment."""
    return ProjectedSegment(0, 0, 0)
'''

RENDER = '''
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """One slice of road."""
    index: int
    curve: float = 0.0
    light: bool = True


class Renderer:
    def __init__(self, screen_width: int, screen_height: int, fov: float):
        self.screen_width = screen_width
        self.colors = [(1, 2, 3)]
        self.car = CarState()

    def render(self, screen, track_segments, car_physics) -> None:
        pass


class Plain:
    def go(self):
        pass


class Child(Plain):
    pass
'''


def test_the_constructor_is_part_of_the_interface():
    """`__init__` is how a thing is made; hiding it as private is what
    produced `CarPhysics(max_speed=…)` three builds running."""
    surface = ctx.interface(PHYSICS, "python", "src/physics.py")
    assert "def __init__(self)" in surface
    assert "_helper" not in surface, "single-underscore names stay private"
    assert "_scratch" not in surface


def test_instance_attributes_are_listed_with_types_and_defaults():
    surface = ctx.interface(PHYSICS, "python", "src/physics.py")
    assert "# instance attributes:" in surface
    assert "speed: float = 0.0" in surface
    assert "max_speed: float = 10.0" in surface
    assert "horizontal_position: float = 0.0" in surface
    # Set outside __init__: typed from the parameter it copies, and marked.
    assert "steering_angle: float  (set in update)" in surface


def test_namedtuple_fields_become_a_constructor_line():
    """`ProjectedSegment(x=0, y=0, z=100, curve=0, color_pattern=[])` was
    invented against a NamedTuple whose only fields are x, y, width."""
    surface = ctx.interface(MATH3D, "python", "src/math3d.py")
    assert ("# fields (constructor arguments, in this order): "
            "x: float, y: float, width: float") in surface
    assert "# construct: ProjectedSegment(x, y, width)" in surface


def test_dataclass_fields_keep_their_defaults_and_the_decorator_shows():
    surface = ctx.interface(RENDER, "python", "src/render.py")
    assert "@dataclass class Segment" in surface
    assert "index: int, curve: float = 0.0, light: bool = True" in surface
    assert "# construct: Segment(index, curve, light)" in surface


def test_members_sit_indented_under_their_class():
    surface = ctx.interface(RENDER, "python", "src/render.py")
    lines = surface.splitlines()
    i = lines.index("class Renderer")
    assert lines[i + 1].startswith("    # instance attributes:")
    assert lines[i + 2] == ("    def __init__(self, screen_width: int, "
                            "screen_height: int, fov: float)")
    assert lines[i + 3] == ("    def render(self, screen, track_segments, "
                            "car_physics) -> None")


def test_attribute_types_come_from_parameters_literals_and_calls():
    surface = ctx.interface(RENDER, "python", "src/render.py")
    assert "screen_width: int" in surface        # from the parameter
    assert "colors: list" in surface             # from the literal
    assert "car: CarState" in surface            # from the constructor call


def test_a_bare_class_says_it_takes_no_arguments_and_a_subclass_claims_nothing():
    surface = ctx.interface(RENDER, "python", "src/render.py")
    assert "# construct: Plain()  — takes no arguments" in surface
    child = surface[surface.index("class Child(Plain)"):]
    assert "construct" not in child, (
        "a subclass inherits a constructor this parser cannot see")


def test_the_codemap_rows_render_the_same_surface(tmp_path):
    """The block the MODEL receives is built from the store's rows, not from
    `interface()`. Both must say the same thing."""
    symbols, _edges, _unresolved = parse_python.parse(PHYSICS,
                                                      "src/physics.py")
    rows = [{"name": s.name, "kind": s.kind, "line": s.line,
             "end_line": s.end_line, "signature": s.signature,
             "docstring": s.docstring, "approximate": s.approximate}
            for s in symbols if s.kind != "module"]
    from_rows, _ = ctx.interface_lines(rows)
    from_source, _ = ctx.interface_lines(
        [s for s in symbols if s.kind != "module"])
    assert from_rows == from_source
    assert any("def __init__(self)" in line for line in from_rows)


def test_fields_and_attributes_are_symbols_the_codemap_can_find():
    symbols, _e, _u = parse_python.parse(MATH3D + PHYSICS, "m.py")
    kinds = {s.name: s.kind for s in symbols}
    assert kinds["ProjectedSegment.width"] == "field"
    assert kinds["CarPhysics.max_speed"] == "attribute"
    assert "CarPhysics._scratch" not in kinds


def test_an_outline_is_unchanged_in_shape_by_the_new_symbols():
    """`outline()` lists definitions; fields and attributes ride along as
    rows but a class's methods are still there with their signatures."""
    out = ctx.outline(PHYSICS, "python", "src/physics.py")
    assert "method def update(self, dt: float" in out
    assert "class class CarPhysics" in out
