# SPDX-License-Identifier: Apache-2.0
"""The skeleton pins the interfaces; every file sees the exact definitions.

Oct 2, 2026, the racing spec on Qwen3-Coder-30B (BUILD_LOG.txt of
specification-pseudo-3d-racing_30b). Every stub the skeleton wrote was a
docstring and `def main(): raise NotImplementedError`, so each file was
written blind to the others. `src/track.py` made

    TrackSegment = Tuple[float, float, int]  # (curve_value, color_pattern, z_position)

and `src/render.py`, written next with `build_track() -> List[TrackSegment]`
in front of it and nothing of what a TrackSegment is, read `seg.x`,
`seg.y`, `seg.z` — `'tuple' object has no attribute 'x'` at render.py:24,
and main.py died on its first frame. The codemap indexed functions and
classes only: a module-level type alias did not exist for it.

Two fixes, each tested here:

  * the codemap indexes type aliases and constants as their exact source
    statement, keeps defaults in signatures, and the interface block shows
    them — so a file sees what it consumes, exactly;
  * the skeleton asks the model ONCE for every module's stub — shared data
    types and public signatures — reduces it by rule to signatures that
    import cleanly, and the loop shows each file its pinned stub as the
    contract it must keep.
"""

from __future__ import annotations

import ast

from cognitive_coder import (
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
)
from cognitive_coder import context, interfaces
from cognitive_coder.codemap import parse_python

#: src/track.py exactly as the Oct 2 build wrote it (the first 5 lines).
TRACK_OCT2 = '''from typing import List, Tuple

# Track segment data model
TrackSegment = Tuple[float, float, int]  # (curve_value, color_pattern, z_position)

class TrackBuilder:
    """Builds a track with segments that define curves and visual patterns."""

    def __init__(self, segment_count: int = 300):
        self.segment_count = segment_count
        self.segments: List[TrackSegment] = []

    def generate_straight(self, length: int = 100) -> None:
        """Add a straight segment to the track."""
        for _ in range(length):
            self.segments.append((0.0, len(self.segments) % 2, len(self.segments)))

def build_track() -> List[TrackSegment]:
    """Build and return a complete track."""
    builder = TrackBuilder()
    builder.generate_straight(3)
    return builder.segments
'''


def _host(tmp_path, replies, **kw):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Qwen3-Coder-30B-A3B-Instruct",
                                context_tokens=32768),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove(), **kw)


def _user_text(prompt) -> str:
    return "\n".join(m.content for m in prompt
                     if m.role == "user" and isinstance(m.content, str))


def _prompt_for(host, path: str):
    """The first prompt that asked the model to write `path`."""
    for p in host.llm.prompts:
        if f"Write the complete contents of `{path}`" in _user_text(p):
            return p
    raise AssertionError(f"no prompt asked for {path}")


def _fence(code: str) -> str:
    return f"```python\n{code}\n```"


# ---------------------------------------------------------------------------
# 1. the codemap sees type aliases and constants, exactly
# ---------------------------------------------------------------------------

def test_a_module_level_type_alias_is_a_symbol_with_its_exact_source():
    symbols, _e, _u = parse_python.parse(TRACK_OCT2, "src/track.py")
    alias = next((s for s in symbols if s.name == "TrackSegment"), None)
    assert alias is not None, "the alias is not indexed at all"
    assert alias.kind == "alias"
    assert alias.signature == "TrackSegment = Tuple[float, float, int]"
    assert alias.docstring == "(curve_value, color_pattern, z_position)"


def test_aliases_and_constants_of_every_common_shape_are_indexed():
    src = '''import math
from typing import Dict, Optional, Tuple, TypeAlias, TypeVar
FOV = math.pi / 3  # sixty degrees
Color = Tuple[int, int, int]
Seg = dict
Speed = float
MaybeInt = int | None
Vec: TypeAlias = "list[float]"
T = TypeVar("T")
RGB = Tuple[int, int, int]
screen = make_screen()
Game = GameClass()
_private = Dict[str, int]
'''
    got = {s.name: (s.kind, s.signature)
           for s in parse_python.parse(src, "m.py")[0]
           if s.kind in ("alias", "constant")}
    assert got["FOV"] == ("constant", "FOV = math.pi / 3")
    assert got["Color"] == ("alias", "Color = Tuple[int, int, int]")
    assert got["Seg"] == ("alias", "Seg = dict")
    assert got["Speed"] == ("alias", "Speed = float")
    assert got["MaybeInt"] == ("alias", "MaybeInt = int | None")
    assert got["Vec"][0] == "alias"
    assert got["T"] == ("alias", 'T = TypeVar("T")')
    assert got["RGB"] == ("alias", "RGB = Tuple[int, int, int]")
    # a plain variable, an instance and a private name are not interface
    assert "screen" not in got and "Game" not in got and "_private" not in got


def test_signatures_keep_their_defaults():
    symbols, _e, _u = parse_python.parse(TRACK_OCT2, "src/track.py")
    sig = {s.name: s.signature for s in symbols}
    # Oct 2 showed `def __init__(self, segment_count: int)` — required.
    assert sig["TrackBuilder.__init__"] == \
        "def __init__(self, segment_count: int = 300)"
    assert sig["TrackBuilder.generate_straight"] == \
        "def generate_straight(self, length: int = 100) -> None"
    src = "def f(a, b=2, *args, c, d: str = 'x', **kw): pass\n"
    assert parse_python.parse(src)[0][1].signature == \
        "def f(a, b=2, *args, c, d: str = 'x', **kw)"


def test_a_dataclass_field_default_is_shown_as_written():
    src = '''from dataclasses import dataclass, field
from enum import Enum

@dataclass
class Car:
    MAX_SPEED = 200.0
    speed: float = 0.0
    tags: list = field(default_factory=list)

class Mode(Enum):
    FAST = 1
    SLOW = 2
'''
    text = context.interface(src, "python", "m.py")
    assert "tags: list = field(default_factory=list)" in text, text
    assert "# class constants: MAX_SPEED = 200.0" in text, text
    assert "# members: FAST = 1, SLOW = 2" in text, text


def test_the_interface_shows_the_alias_before_what_uses_it_and_says_tuple():
    text = context.interface(TRACK_OCT2, "python", "src/track.py")
    lines = text.splitlines()
    alias = next(i for i, ln in enumerate(lines)
                 if ln.startswith("TrackSegment = Tuple[float, float, int]"))
    user = next(i for i, ln in enumerate(lines) if "def build_track" in ln)
    assert alias < user, text
    assert "(curve_value, color_pattern, z_position)" in lines[alias]
    assert "a plain tuple, not a class: no named attributes" in lines[alias]


def test_render_is_shown_what_a_track_segment_is(tmp_path):
    """The Oct 2 order — track built before render — with no pinning: the
    interface block render.py's first attempt gets now carries the alias,
    exactly. (It carried `build_track() -> List[TrackSegment]` only.)"""
    plan = ("src/track.py — the track data model\n"
            "src/render.py — draws the track\n"
            "src/main.py — the entry point\n")
    render = _fence("from src.track import TrackSegment, build_track\n\n\n"
                    "def draw(segments):\n"
                    "    return [seg[0] for seg in segments]\n")
    main = _fence("from src.render import draw\n"
                  "from src.track import build_track\n\n"
                  "print(len(draw(build_track())))\n")
    host = _host(tmp_path, [plan, _fence(TRACK_OCT2), render, main])
    session = Session(host, config=SessionConfig(
        attempts=1, pin_interfaces=False, review_after_build=False))
    session.run("a track and something that draws it")
    text = _user_text(_prompt_for(host, "src/render.py"))
    assert "TrackSegment = Tuple[float, float, int]  # (curve_value, " \
           "color_pattern, z_position)" in text, text
    assert "def __init__(self, segment_count: int = 300)" in text


def test_a_tuple_attribute_error_names_the_projects_tuple_aliases(tmp_path):
    from cognitive_coder import enrich
    from cognitive_coder.codemap import CodeMap
    from cognitive_coder.types import Diagnostic
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "track.py").write_text(TRACK_OCT2, encoding="utf-8")
    host = _host(tmp_path, [])
    cm = CodeMap(host.fs, host.storage)
    cm.index_project()
    facts = enrich.facts_for(
        [Diagnostic(file="src/render.py", line=24, severity="exception",
                    message="AttributeError: 'tuple' object has no "
                            "attribute 'x'")], codemap=cm, fs=host.fs)
    assert "plain tuple" in facts, facts
    assert "TrackSegment = Tuple[float, float, int]" in facts, facts
    assert "(curve_value, color_pattern, z_position)" in facts, facts


# ---------------------------------------------------------------------------
# 2. the skeleton pins the interfaces before any body is written
# ---------------------------------------------------------------------------

PLAN3 = ("src/track.py — the track data model: segments with a curve\n"
         "src/render.py — draws the visible segments\n"
         "src/main.py — the entry point: builds the track and draws it\n")

SKELETON = '''Here are the stubs.

```python
# file: src/track.py
from dataclasses import dataclass
from typing import List

SEGMENT_LENGTH = 200.0


@dataclass
class Segment:
    """One slice of road."""
    z: float
    curve: float = 0.0
    dark: bool = False


def build_track(n: int = 3) -> List[Segment]:
    """Straights and curves."""
    segments = [Segment(z=i * SEGMENT_LENGTH) for i in range(n)]
    return segments
```

```python
# file: src/render.py
from typing import List

from src.track import Segment


def draw(segments: List[Segment]) -> List[float]:
    """The depth of each segment, nearest first."""
    raise NotImplementedError
```

```python
# file: src/main.py
from src.render import draw
from src.track import build_track

print("starting")


def main() -> int:
    """Build the track and draw it."""
    print(len(draw(build_track())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
'''

TRACK = _fence('''from dataclasses import dataclass
from typing import List

SEGMENT_LENGTH = 200.0


@dataclass
class Segment:
    """One slice of road."""
    z: float
    curve: float = 0.0
    dark: bool = False


def build_track(n: int = 3) -> List[Segment]:
    """Straights and curves."""
    return [Segment(z=i * SEGMENT_LENGTH, dark=bool(i % 2)) for i in range(n)]
''')

RENDER = _fence('''from typing import List

from src.track import Segment


def draw(segments: List[Segment]) -> List[float]:
    """The depth of each segment, nearest first."""
    return [seg.z for seg in segments]
''')

MAIN = _fence('''from src.render import draw
from src.track import build_track


def main() -> int:
    """Build the track and draw it."""
    print(len(draw(build_track())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')


def _pinned_session(tmp_path, replies, **config):
    host = _host(tmp_path, replies)
    session = Session(host, config=SessionConfig(
        attempts=2, review_after_build=False, **config))
    return host, session


def test_the_skeleton_writes_the_pinned_interfaces_not_bare_stubs(tmp_path):
    host, session = _pinned_session(tmp_path, [PLAN3, SKELETON])
    session.start("a pseudo-3D road: track data, a renderer, a main loop")
    track = (tmp_path / "src" / "track.py").read_text()
    render = (tmp_path / "src" / "render.py").read_text()
    main = (tmp_path / "src" / "main.py").read_text()
    for text in (track, render, main):
        assert interfaces.PINNED_TEXT in text
        ast.parse(text)
    # the shared type is defined once, with its fields, and imported by name
    assert "class Segment:" in track and "curve: float = 0.0" in track
    assert "from src.track import Segment" in render
    # bodies are stubs — even the one the model wrote out
    assert "raise NotImplementedError" in track
    assert "Segment(z=i * SEGMENT_LENGTH)" not in track
    # nothing runs at import: the top-level print goes; the __main__ guard
    # stays, reduced to the call, because it is how the program starts
    assert 'print("starting")' not in main
    assert "def main() -> int:" in main
    assert 'if __name__ == "__main__":\n    raise SystemExit(main())' in main
    assert "print(len(draw(build_track())))" not in main
    # the order is learned from the stubs' imports
    order = [t.path for t in session.plan.tasks]
    assert order.index("src/track.py") < order.index("src/render.py") \
        < order.index("src/main.py")
    render_task = next(t for t in session.plan.tasks
                       if t.path == "src/render.py")
    track_task = next(t for t in session.plan.tasks
                      if t.path == "src/track.py")
    assert track_task.id in render_task.depends_on
    # and the journal records the call, with provenance
    rows = [r for r in session.journal.events()
            if r.get("event") == "interfaces"]
    assert rows and rows[0].get("prompt_sha256")
    assert rows[0]["data"]["pinned"] == ["src/main.py", "src/render.py",
                                         "src/track.py"]


def test_every_stub_the_skeleton_pins_can_be_imported(tmp_path):
    host, session = _pinned_session(tmp_path, [PLAN3, SKELETON])
    session.start("a pseudo-3D road")
    import subprocess
    import sys
    for mod in ("src.track", "src.render", "src.main"):
        proc = subprocess.run([sys.executable, "-c", f"import {mod}"],
                              cwd=str(tmp_path), capture_output=True,
                              text=True, timeout=60)
        assert proc.returncode == 0, (mod, proc.stderr)
        assert "starting" not in proc.stdout


def test_each_file_is_shown_its_pinned_interface_and_what_it_uses(tmp_path):
    host, session = _pinned_session(
        tmp_path, [PLAN3, SKELETON, TRACK, RENDER, MAIN])
    session.run("a pseudo-3D road")
    final = {o.path: o for o in session._final_outcomes()}
    assert all(o.ok for o in final.values()), session.report()
    # the FIRST file built — written before render exists — already
    # carries the contract render will use
    track_prompt = _user_text(_prompt_for(host, "src/track.py"))
    assert "[THE INTERFACE PINNED FOR THIS FILE — src/track.py]" in \
        track_prompt
    assert "class Segment:" in track_prompt
    assert "Keep every name, signature, field" in track_prompt
    # render sees its own contract AND Segment's exact fields
    render_prompt = _user_text(_prompt_for(host, "src/render.py"))
    assert "[THE INTERFACE PINNED FOR THIS FILE — src/render.py]" in \
        render_prompt
    assert "## src/track.py" in render_prompt
    assert "z: float, curve: float = 0.0, dark: bool = False" in \
        render_prompt
    assert "SEGMENT_LENGTH = 200.0" in render_prompt
    # the sentinel never reaches a built file
    for name in ("track", "render", "main"):
        assert "cc-stub:" not in (tmp_path / "src" / f"{name}.py").read_text()


def test_a_body_that_drops_a_pinned_name_is_sent_back(tmp_path):
    """D12: the skeleton says which names must exist. The Oct 2 shape —
    track.py writing plain tuples instead of the pinned `Segment` — is now
    a located error on track.py itself, before render.py is built on it."""
    tuples = _fence("from typing import List, Tuple\n\n"
                    "SEGMENT_LENGTH = 200.0\n"
                    "TrackSegment = Tuple[float, float, int]\n\n\n"
                    "def build_track(n: int = 3) -> List[TrackSegment]:\n"
                    '    """Straights and curves."""\n'
                    "    return [(0.0, 0, i) for i in range(n)]\n")
    host, session = _pinned_session(
        tmp_path, [PLAN3, SKELETON, tuples, TRACK, RENDER, MAIN])
    session.run("a pseudo-3D road")
    track = [o for o in session.outcomes if o.path == "src/track.py"][0]
    assert track.ok and len(track.attempts) == 2, \
        [a.note for a in track.attempts]
    first = track.attempts[0].diagnostics[0]
    assert first.code == "pinned-interface"
    assert "does not define `Segment`" in first.message
    retry = _user_text(host.llm.prompts[3])
    assert "does not define `Segment`" in retry
    assert "[THE INTERFACE PINNED FOR THIS FILE — src/track.py]" in retry
    assert "class Segment" in (tmp_path / "src" / "track.py").read_text()
    assert all(o.ok for o in session._final_outcomes()), session.report()


def test_a_stub_that_is_missing_or_broken_keeps_the_rule_stub(tmp_path):
    reply = SKELETON.replace(
        "def draw(segments: List[Segment]) -> List[float]:",
        "def draw(segments: List[Segment]) -> List[float]")  # no colon
    reply = reply.replace("# file: src/main.py", "# file: src/other.py")
    host, session = _pinned_session(tmp_path, [PLAN3, reply])
    result_plan = session.start("a pseudo-3D road")
    assert result_plan is not None
    track = (tmp_path / "src" / "track.py").read_text()
    render = (tmp_path / "src" / "render.py").read_text()
    main = (tmp_path / "src" / "main.py").read_text()
    assert interfaces.PINNED_TEXT in track
    assert interfaces.PINNED_TEXT not in render and "def main" in render
    assert interfaces.PINNED_TEXT not in main and "def main" in main
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("src/render.py keeps a plain stub — its stub does not parse"
               in m for m in warnings), warnings
    assert any("src/main.py keeps a plain stub — the reply had no stub for "
               "it" in m for m in warnings), warnings


def test_a_single_file_plan_makes_no_interface_call(tmp_path):
    host, session = _pinned_session(
        tmp_path, ["src/greet.py — says hello\n",
                   _fence("def greet() -> str:\n    return 'hi'\n")])
    session.run("a greeting")
    assert len(host.llm.prompts) == 2           # the plan, the file
    assert not any(r.get("event") == "interfaces"
                   for r in session.journal.events())


def test_pinning_can_be_turned_off(tmp_path):
    host, session = _pinned_session(tmp_path, [PLAN3, TRACK, RENDER, MAIN],
                                    pin_interfaces=False)
    session.run("a pseudo-3D road")
    assert all(o.ok for o in session._final_outcomes()), session.report()
    assert "def main" in (tmp_path / ".cc_snapshots").joinpath(
        "0001-skeleton", "MANIFEST.txt").read_text()


def test_the_skeleton_says_when_its_files_disagree(tmp_path):
    reply = SKELETON.replace("from src.track import Segment",
                             "from src.track import Segment, Lane")
    host, session = _pinned_session(tmp_path, [PLAN3, reply])
    session.start("a pseudo-3D road")
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("src/render.py imports `Lane` from src/track.py, which does "
               "not define it" in m for m in warnings), warnings


def test_an_entry_point_stub_keeps_how_the_program_starts():
    """A stub `main.py` with `def main()` and no `__main__` guard taught the
    body to leave the guard out: a program that runs, exits 0, and starts
    nothing."""
    code = "def main() -> int:\n    \"\"\"Run.\"\"\"\n    return 0\n"
    stub, _ = interfaces.sanitise(code, "the game loop", entry_point=True)
    assert stub.rstrip().endswith('if __name__ == "__main__":\n'
                                  '    raise SystemExit(main())'), stub
    plain, _ = interfaces.sanitise(code, "a helper module")
    assert "__main__" not in plain


def test_the_sanitiser_reduces_any_stub_to_signatures():
    code = '''import pygame
pygame.init()
WIDTH = 800
screen = pygame.display.set_mode((WIDTH, 600))


class Car:
    COLOR = pygame.Color(255, 0, 0)

    def __init__(self, speed: float = 0.0) -> None:
        self.speed: float = speed
        print("made a car")

    def update(self, dt: float) -> None: self.speed += dt


while True:
    break
'''
    stub, dropped = interfaces.sanitise(code, "the car")
    tree = ast.parse(stub)
    assert interfaces.PINNED_TEXT in stub
    assert "pygame.init()" not in stub and "set_mode" not in stub
    assert "while True" not in stub and "made a car" not in stub
    assert "COLOR" not in stub
    assert "WIDTH = 800" in stub
    assert "self.speed: float = speed" in stub      # interface, kept
    assert "def update(self, dt: float) -> None:" in stub
    assert ast.get_docstring(tree) == "the car"
    assert len(dropped) >= 3, dropped
