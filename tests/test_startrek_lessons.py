# SPDX-License-Identifier: Apache-2.0
"""What the two Star Trek builds of 2026-10-16 taught, as tests.

One build ran DiffusionGemma, one ran Devstral; both failed, and the
journals put most of the damage on the harness (docs/HYBRID.md). Each
test here replays one of those faults and asserts it is closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cognitive_coder import constraints, interfaces, langs, specslice
from cognitive_coder.planner import Planner
from cognitive_coder.ports import (
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    SubprocessExec,
)

SPEC = (Path(__file__).parent / "fixtures" / "startrek_spec.md")


@pytest.fixture(scope="module")
def spec() -> str:
    return SPEC.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# hard requirements keep their underscores
# --------------------------------------------------------------------------

def test_identifiers_in_hard_requirements_keep_their_underscores(spec):
    found = constraints.extract(spec)
    joined = "\n".join(found.items)
    assert "tests/test_physics.py" in joined
    assert "testphysics" not in joined
    assert "SUBTICKS_PER_TURN = 10" in joined
    assert "SUBTICKSPERTURN" not in joined
    assert len(found.items) == 12 and found.more == 0


def test_markdown_emphasis_is_still_removed():
    assert constraints._unemphasise("You MUST **never** use *global* state") \
        == "You MUST never use global state"
    assert constraints._unemphasise("`SUBTICKS_PER_TURN = 10` MUST hold") \
        == "SUBTICKS_PER_TURN = 10 MUST hold"
    assert constraints._unemphasise("call `__init__` once") == "call __init__ once"


# --------------------------------------------------------------------------
# the request is sliced to the file, never cut at a character count
# --------------------------------------------------------------------------

def test_a_repair_slice_carries_the_files_own_section(spec):
    cut = specslice.slice_for(spec, "src/physics.py",
                              budget=specslice.REPAIR_CHARS,
                              related=["tests/test_physics.py"])
    assert not cut.whole
    assert "math.atan2(x2 - x1, -(y2 - y1))" in cut.text      # char 7,363
    assert "int(((rel + 30.0) % 360.0) // 60.0)" in cut.text   # char 7,904
    assert "overflow * 0.25" in cut.text
    assert "not repeated here" in cut.text
    assert any("Physics" in h for h in cut.kept)
    assert any("Parser" in h for h in cut.left_out)


def test_the_files_own_section_outranks_the_hard_requirements(spec):
    cut = specslice.slice_for(spec, "src/parser.py", budget=3000)
    assert "parse_bank_spec" in cut.text
    assert any("Parser" in h for h in cut.kept)


def test_a_short_request_is_passed_whole():
    cut = specslice.slice_for("Make a calculator. It MUST add.", "calc.py")
    assert cut.whole and cut.text == "Make a calculator. It MUST add."


def test_a_test_file_is_sliced_with_the_module_it_covers(spec):
    cut = specslice.slice_for(spec, "tests/test_physics.py", budget=8000,
                              related=["src/physics.py"])
    assert "bearing_between" in cut.text


def test_plain_prose_without_headings_is_sliced_by_paragraph():
    text = ("Overview of the thing.\n\n" + "Unrelated filler. " * 300
            + "\n\nThe module src/alpha.py MUST return 7 from seven().\n\n"
            + "More filler. " * 300)
    cut = specslice.slice_for(text, "src/alpha.py", budget=800)
    assert not cut.whole
    assert "MUST return 7" in cut.text
    assert "Overview of the thing." in cut.text


# --------------------------------------------------------------------------
# a dataclass written with defaults before required fields still imports
# --------------------------------------------------------------------------

STUB = '''# cc-stub: pinned
"""models"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Ship:
    """A ship."""
    ship_id: str
    warp_heat: float = 0.0
    cloaked: bool = False
    subsystems: dict[str, float]
    fleet_order: str = "attack"


@dataclass
class Order:
    kind: str
    value: float = 0.0
'''


def test_defaulted_fields_before_required_ones_become_keyword_only():
    fixed, notes = interfaces.fix_dataclass_order(STUB)
    assert "warp_heat: float = field(default=0.0, kw_only=True)" in fixed
    assert "cloaked: bool = field(default=False, kw_only=True)" in fixed
    assert 'fleet_order: str = "attack"' in fixed        # after the last required
    assert "value: float = 0.0" in fixed                  # a valid class is untouched
    assert "from dataclasses import dataclass, field" in fixed
    assert len(notes) == 2 and "keyword-only" in notes[0]
    ns: dict = {}
    exec(compile(fixed, "stub", "exec"), ns)              # it imports
    names = [f.name for f in ns["Ship"].__dataclass_fields__.values()]
    assert names == ["ship_id", "warp_heat", "cloaked", "subsystems",
                     "fleet_order"]                        # order kept


def test_a_valid_dataclass_is_left_alone():
    text = STUB.replace("    warp_heat: float = 0.0\n    cloaked: bool = False\n", "")
    fixed, notes = interfaces.fix_dataclass_order(text)
    assert fixed == text and notes == []


# --------------------------------------------------------------------------
# the request's conventions reach the stub's docstring
# --------------------------------------------------------------------------

def test_sentences_naming_the_file_or_its_names_land_in_its_docstring():
    sentences = [
        "Ship headings MUST use navigational bearings where 0.0 is North (0, -1).",
        "You MUST NEVER call input() in any module.",
        "src/models.py MUST NOT import pygame.",
    ]
    out = interfaces.with_conventions(STUB, "src/models.py", sentences)
    assert interfaces.CONVENTIONS_TITLE in out
    assert "North (0, -1)" in out                 # names `Ship`
    assert "MUST NOT import pygame" in out        # names the path
    assert "input()" not in out                   # names neither
    assert out.startswith("# cc-stub")
    import ast
    ast.parse(out)


def test_a_stub_without_a_docstring_gets_one():
    bare = "# cc-stub: pinned\nfrom __future__ import annotations\n\nX = 1\n"
    out = interfaces.with_conventions(bare, "src/alpha.py",
                                      ["src/alpha.py MUST stay pure."])
    assert out.splitlines()[0].startswith("# cc-stub")
    assert "MUST stay pure" in out
    import ast
    assert ast.get_docstring(ast.parse(out))


# --------------------------------------------------------------------------
# the skeleton is imported, not only parsed
# --------------------------------------------------------------------------

def _planner(tmp_path, replies=()):
    host = Host(llm=ScriptedLLM(list(replies), supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(), events=RecordingEvents(),
                approval=AutoApprove())
    return host, Planner(host, lang="python")


def test_a_skeleton_that_parses_but_cannot_be_imported_is_reported(tmp_path):
    host, planner = _planner(tmp_path)
    host.fs.write("src/__init__.py", "")
    host.fs.write("src/models.py",
                  "from dataclasses import dataclass\n\n@dataclass\n"
                  "class S:\n    a: int = 1\n    b: int\n")
    host.fs.write("src/physics.py", "from src.models import S\n")
    ok, note = planner.verify_skeleton(["src/models.py", "src/physics.py"])
    assert not ok
    assert "does not import" in note and "src/models.py" in note
    assert "TypeError" in note


def test_a_missing_third_party_module_is_a_note_not_a_broken_skeleton(
        tmp_path):
    host, planner = _planner(tmp_path)
    host.fs.write("src/__init__.py", "")
    host.fs.write("src/render.py", "import no_such_package_xyz\n")
    ok, note = planner.verify_skeleton(["src/render.py"])
    assert ok, note
    assert any("no_such_package_xyz" in n for n in planner.import_notes)


def test_a_project_module_imported_by_the_wrong_name_is_a_broken_skeleton(
        tmp_path):
    host, planner = _planner(tmp_path)
    host.fs.write("src/__init__.py", "")
    host.fs.write("src/models.py", "X = 1\n")
    host.fs.write("src/physics.py", "from models import X\n")   # not src.models
    ok, note = planner.verify_skeleton(["src/models.py", "src/physics.py"])
    assert not ok and "src/physics.py" in note


def test_module_names_are_derived_for_the_runner():
    assert langs.module_name("src/physics.py") == "src.physics"


# --------------------------------------------------------------------------
# the speed probe: a model spilling into system RAM is said at the start
# --------------------------------------------------------------------------

class _TimedLLM:
    """A port that reports timings like a real one — 4 tokens in 1 s."""

    name = "timed"

    def __init__(self, tok_s: float, canvas: bool = False) -> None:
        self.tok_s = tok_s
        self.canvas = canvas
        self.probes = 0

    def complete(self, messages, **kw):
        from cognitive_coder.types import Completion
        self.probes += 1
        return Completion(text="READY", finish_reason="stop",
                          model="big-model.gguf", tokens_in=20, tokens_out=4,
                          prompt_ms=500, decode_ms=int(4000 / self.tok_s))

    def stream(self, messages, **kw):
        yield "READY"

    def capabilities(self):
        from cognitive_coder.types import ModelCapabilities
        return ModelCapabilities(name="big-model.gguf", family="mistral",
                                 context_tokens=16384,
                                 supports_canvas=self.canvas)

    def count_tokens(self, text):
        return max(1, len(text or "") // 4)


def _session_with(tmp_path, llm, **cfg):
    from cognitive_coder import Session, SessionConfig
    events = RecordingEvents()
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(), storage=MemoryStorage(),
                events=events, approval=AutoApprove())
    cfg.setdefault("attempts", 1)
    cfg.setdefault("review_after_build", False)
    cfg.setdefault("mutation_check", False)
    return Session(host, config=SessionConfig(**cfg)), events


def test_a_slow_model_is_named_in_the_log_and_a_warning(tmp_path):
    llm = _TimedLLM(tok_s=4.0)
    session, events = _session_with(tmp_path, llm)
    try:
        session.run("hello.py — print hello")
    except Exception:                                    # noqa: BLE001
        pass                   # the fake writes nothing; the probe is the point
    assert llm.probes >= 1
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "speed     4.0 tok/s" in log
    assert "spilling into system RAM" in log
    assert any("tokens per second" in m for k, m, _ in events.events
               if k == "warning")


def test_a_fast_model_gets_its_number_and_no_warning(tmp_path):
    llm = _TimedLLM(tok_s=24.0)
    session, events = _session_with(tmp_path, llm)
    try:
        session.run("hello.py — print hello")
    except Exception:                                    # noqa: BLE001
        pass
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "speed     24." in log and "tok/s" in log
    assert "spilling" not in log


def test_the_scripted_doubles_are_never_probed(tmp_path):
    llm = ScriptedLLM([], supports_tools=False)
    session, _events = _session_with(tmp_path, llm)
    from cognitive_coder import CognitiveCoderError
    with pytest.raises(CognitiveCoderError):   # the script is empty: the PLAN
        session.run("hello.py — print hello")   # call runs it dry, not a probe
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "speed " not in log


def test_a_canvas_model_is_timed_per_reply_not_per_token(tmp_path):
    llm = _TimedLLM(tok_s=4.0, canvas=True)
    session, events = _session_with(tmp_path, llm)
    try:
        session.run("hello.py — print hello")
    except Exception:                                    # noqa: BLE001
        pass
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "a canvas model" in log
    assert "spilling" not in log


# --------------------------------------------------------------------------
# spec vectors: the request's own examples run before the model's tests
# --------------------------------------------------------------------------

VECTOR_REQUEST = (
    "# Bank specs\n\n"
    "## 1. The parser (`src/banks.py`)\n\n"
    "Implement `parse_bank_spec(spec: str) -> tuple[int, ...]` supporting "
    "single ints (`\"1\"` -> `(1,)`), ranges (`\"1-3\"` -> `(1, 2, 3)`), "
    "and comma lists (`\"1,3\"` -> `(1, 3)`). "
    "Define `MAX_BANKS: int = 4` in `src/banks.py` too.\n")

BANKS_PLAN = "src/banks.py — the bank spec parser\n"

BANKS_WRONG = ("```python\nMAX_BANKS: int = 4\n\n\n"
               "def parse_bank_spec(spec: str) -> tuple:\n"
               '    """Parse."""\n'
               "    out = []\n"
               "    for part in spec.split(','):\n"
               "        if '-' in part:\n"
               "            a, b = part.split('-')\n"
               "            out.extend(range(int(a), int(b)))   # off by one\n"
               "        else:\n"
               "            out.append(int(part))\n"
               "    return tuple(out)\n```")

BANKS_RIGHT = BANKS_WRONG.replace("range(int(a), int(b)))   # off by one",
                                  "range(int(a), int(b) + 1))")


def test_spec_vectors_are_lifted_written_and_run_first(tmp_path):
    from cognitive_coder import specvectors
    spec_vectors = specvectors.extract(VECTOR_REQUEST, ["src/banks.py"])
    assert [v.call for v in spec_vectors] == [
        "MAX_BANKS", 'parse_bank_spec("1")', 'parse_bank_spec("1-3")',
        'parse_bank_spec("1,3")']
    llm = ScriptedLLM([BANKS_PLAN, BANKS_WRONG, BANKS_RIGHT],
                      supports_tools=False)
    session, events = _session_with(tmp_path, llm, attempts=2,
                                    pin_interfaces=False)
    session.run(VECTOR_REQUEST)
    vector_file = tmp_path / "tests" / "test_spec_vectors_banks.py"
    assert vector_file.exists()
    text = vector_file.read_text(encoding="utf-8")
    assert specvectors.is_vector_file(text)
    assert 'parse_bank_spec("1-3")' in text and "MAX_BANKS" in text
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/banks.py"].ok, final["src/banks.py"].stopped_because
    assert len(final["src/banks.py"].attempts) == 2
    # attempt 1 failed on the request's own example, with its words
    first = final["src/banks.py"].attempts[0]
    assert any(d.tool == "spec-vector" for d in first.diagnostics)
    assert any("the request says" in d.message for d in first.diagnostics)
    # the repair prompt carried the request's sentence
    repair = "\n".join(m.content for m in llm.prompts[2])
    assert "the request says" in repair and '"1-3"' in repair
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "SPEC VECTORS" in log and 'parse_bank_spec("1-3") == (1, 2, 3)' in log
    assert any("spec vector" in m for k, m, _ in events.events
               if k == "status")


def test_spec_vectors_can_be_switched_off(tmp_path):
    llm = ScriptedLLM([BANKS_PLAN, BANKS_WRONG], supports_tools=False)
    session, _events = _session_with(tmp_path, llm, attempts=1,
                                     pin_interfaces=False, spec_vectors=False)
    session.run(VECTOR_REQUEST)
    assert not (tmp_path / "tests" / "test_spec_vectors_banks.py").exists()
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/banks.py"].ok     # nothing contradicted the wrong code
