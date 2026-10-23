# SPDX-License-Identifier: Apache-2.0
"""The engine behind the four tabs (docs/NEXT_TABS.md): Evaluate & Bug Hunt,
Improve, Red Team, Blueprint. Held against a real project folder and a
scripted model.

The one rule all four share: the tab constrains the EDIT SHAPE, never the
model. So what is pinned here is the shape — a bug is a located line or it
is set aside; an improvement is a change list the operator checked, applied
one at a time through the repair loop with a ledger; a red-team finding
without a reproducer that fails now is not counted; a blueprint is refused
when its contracts are not owned exactly once or its dependencies cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    blueprint,
    bughunt,
    improve,
    redteam,
    scope,
)

CALC = '''def add(a: int, b: int) -> int:
    """The sum."""
    total = a + c
    return total


def mul(a: int, b: int) -> int:
    """The product."""
    return a * b
'''
FIXED = CALC.replace("a + c", "a + b")
TOOL = '''import os


def build(name: str) -> str:
    """The shell line for a greeting."""
    return "echo " + name


def greet(name: str) -> int:
    return os.system(build(name))
'''


def _project(tmp_path: Path, calc: str = CALC) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(calc)
    (tmp_path / "src" / "tool.py").write_text(TOOL)
    (tmp_path / "tests" / "test_calc.py").write_text(
        "import unittest\nfrom src.calc import mul\n\n\n"
        "class T(unittest.TestCase):\n    def test_mul(self):\n"
        "        self.assertEqual(mul(2, 3), 6)\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "junk.py").write_text("x = 1\n")
    return tmp_path


def _session(root: Path, replies: list[str]) -> Session:
    llm = ScriptedLLM(replies, supports_tools=False)
    host = Host(llm=llm, fs=LocalFileSystem(str(root)), exec=SubprocessExec(),
                storage=MemoryStorage(str(root / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    return Session(host, config=SessionConfig(attempts=2,
                                              review_after_build=False))


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------

def test_scope_is_the_project_a_file_or_a_folder_and_skips_the_venv(tmp_path):
    root = _project(tmp_path)
    fs = LocalFileSystem(str(root))
    everything = scope.files(fs, "")
    assert "src/calc.py" in everything and "tests/test_calc.py" in everything
    assert not any(p.startswith(".venv") for p in everything)
    assert scope.files(fs, "src/calc.py") == ["src/calc.py"]
    assert scope.files(fs, "src") == ["src/__init__.py", "src/calc.py",
                                      "src/tool.py"]
    assert scope.files(fs, ["src/tool.py", "nope.py"]) == ["src/tool.py"]
    assert scope.is_test("tests/test_calc.py") and not scope.is_test("src/x.py")


# ---------------------------------------------------------------------------
# 1. Evaluate & Bug Hunt
# ---------------------------------------------------------------------------

def test_the_tools_locate_the_undefined_name_without_a_model(tmp_path):
    root = _project(tmp_path)
    session = _session(root, [])
    h = bughunt.hunt(session, "src/calc.py", model_pass=False,
                     typecheck=False)
    assert h.files == ["src/calc.py"]
    assert not h.model_ran
    located = [i for i in h.items if i.path == "src/calc.py" and i.line == 3]
    assert located, [i.one_line() for i in h.items]
    assert any(i.source == "ruff" and i.severity == "error" for i in located)
    text = bughunt.report(h)
    assert text.startswith("# Bug hunt — src/calc.py")
    assert "## src/calc.py" in text and "line 3" in text


def test_the_model_pass_runs_last_and_an_unplaceable_bug_is_set_aside(
        tmp_path):
    root = _project(tmp_path)
    reply = json.dumps({"bugs": [
        {"path": "src/calc.py", "line": 3,
         "message": "c is never defined; the sum uses the wrong name"},
        {"path": "src/calc.py", "line": 900, "message": "off the end"},
        {"path": "src/invented.py", "line": 1, "message": "no such file"},
    ]})
    session = _session(root, [reply])
    h = bughunt.hunt(session, "src", model_pass=True, typecheck=False)
    assert h.model_ran
    model_items = [i for i in h.items if i.source == "model"]
    assert [(i.path, i.line) for i in model_items] == [("src/calc.py", 3)]
    assert len(h.could_not_locate) == 2
    assert "Could not locate" in bughunt.report(h)


def test_the_hand_off_to_repair_is_one_report_per_file_of_located_lines(
        tmp_path):
    root = _project(tmp_path)
    session = _session(root, [])
    h = bughunt.hunt(session, "src", model_pass=False, typecheck=False)
    reports = bughunt.to_debug_reports(h)
    assert reports, "the undefined name is an error, so there is a report"
    calc = next(r for r in reports if r.files == ("src/calc.py",))
    assert "src/calc.py:3:" in calc.error_text
    assert bughunt.to_debug_reports(h, paths=["src/nothing.py"]) == []


def test_an_empty_scope_is_said_not_hidden(tmp_path):
    root = _project(tmp_path)
    session = _session(root, [])
    h = bughunt.hunt(session, "docs", model_pass=False, typecheck=False)
    assert h.files == [] and h.items == []
    assert any("no source files" in n for n in h.notes)


# ---------------------------------------------------------------------------
# 2. Improve
# ---------------------------------------------------------------------------

def test_converse_keeps_the_turns_and_structure_names_only_real_files(
        tmp_path):
    root = _project(tmp_path)
    agreed = json.dumps({"changes": [
        {"title": "Fix the undefined name in add", "files": ["src/calc.py"],
         "keep": "mul is untouched", "verify": "tests/test_calc.py passes",
         "risk": "low", "detail": "add() refers to c; it should be b."},
        {"title": "Something in a file that does not exist",
         "files": ["src/ghost.py"], "risk": "bananas"},
        {"title": ""},
    ]})
    session = _session(root, ["What is add meant to return?",
                              "Then the fix is the name.", agreed])
    convo = improve.Conversation(scope="src/calc.py")
    r1 = improve.converse(session, convo, "add is wrong")
    r2 = improve.converse(session, convo, "the sum of a and b")
    assert r1 == "What is add meant to return?" and "name" in r2
    assert [m.role for m in convo.messages] == ["user", "assistant",
                                                "user", "assistant"]
    items = improve.structure(session, convo)
    assert [i.title for i in items] == [
        "Fix the undefined name in add",
        "Something in a file that does not exist"]
    assert items[0].files == ["src/calc.py"] and items[0].risk == "low"
    assert items[1].risk == "medium", "an unknown risk word is medium"
    assert items[1].files == ["src/ghost.py"], \
        "an unknown path is kept so the operator can see and uncheck it"
    # round trip for the panel's state
    again = improve.items_from_json(improve.items_to_json(items))
    assert [i.title for i in again] == [i.title for i in items]


def test_apply_runs_each_checked_item_through_repair_and_keeps_a_ledger(
        tmp_path):
    root = _project(tmp_path)
    session = _session(root, [f"```python\n{FIXED}```"])
    items = [
        improve.ChangeItem(id="c1", title="Fix the undefined name in add",
                           files=["src/calc.py"], keep="mul is untouched",
                           verify="tests/test_calc.py", detail="c -> b"),
        improve.ChangeItem(id="c2", title="Not this one", files=["src/tool.py"],
                           checked=False),
    ]
    seen: list[str] = []
    results = improve.apply(session, items, progress=seen.append)
    assert [item.id for item, _ in results] == ["c1"], "unchecked is skipped"
    item, result = results[0]
    assert result.state == "repaired", result.verdict
    assert (root / "src" / "calc.py").read_text() == FIXED
    assert seen == ["improve: Fix the undefined name in add"]
    ledger = (root / improve.LEDGER).read_text()
    assert ledger.startswith("# Improvements")
    assert "**Fix the undefined name in add** (src/calc.py) → repaired" in ledger
    assert "keep: mul is untouched" in ledger
    # the request the model saw constrained the shape
    request = item.as_request("- earlier row")
    assert "Change only what this improvement needs" in request
    assert "earlier row" in request


# ---------------------------------------------------------------------------
# 3. Red Team
# ---------------------------------------------------------------------------

def test_the_scanners_find_the_shell_and_a_finding_needs_a_reproducer(
        tmp_path):
    root = _project(tmp_path)
    threats = json.dumps({"threats": [
        {"id": "t1", "boundary": "the name argument", "asset": "subprocess",
         "exposure": "a name with `;` runs a second command",
         "path": "src/tool.py", "severity": "high"}]})
    reproducer = (
        "import unittest\n\nfrom src.tool import build\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_a_shell_metacharacter_is_refused(self):\n"
        "        with self.assertRaises(ValueError):\n"
        "            build('x; rm -rf /')\n")
    attack = json.dumps({"title": "Shell injection through the name",
                         "severity": "high", "path": "src/tool.py",
                         "line": 6, "detail": "build() concatenates.",
                         "test": reproducer})
    session = _session(root, [threats, attack])
    rt = redteam.run(session, "src/tool.py")
    guard_hits = [f for f in rt.findings if f.source == "guard"]
    assert guard_hits and guard_hits[0].demonstrated is None
    assert "guard" in rt.scanners_ran
    assert [t.id for t in rt.threats] == ["t1"]
    att = next(f for f in rt.findings if f.source == "attacker")
    assert att.demonstrated is True, att.output
    assert att.reproducer == "tests/redteam/test_rt_t1.py"
    assert (root / att.reproducer).exists()
    assert (root / "tests" / "redteam" / "__init__.py").exists()
    assert att.command == "python -m unittest tests.redteam.test_rt_t1"
    assert att in rt.demonstrated
    text = redteam.report(rt)
    assert text.startswith("# Red team") and "Shell injection" in text
    assert att.command in text
    # the hand-off carries the COMMAND, so Repair confirms by running it
    reports = redteam.to_repairs(rt, ids=["t1"])
    assert len(reports) == 1 and reports[0].command == att.command
    assert "do not change the reproducer" in reports[0].description


def test_a_reproducer_that_passes_now_is_not_a_finding(tmp_path):
    root = _project(tmp_path)
    passing = ("import unittest\n\nfrom src.tool import build\n\n\n"
               "class T(unittest.TestCase):\n"
               "    def test_echo(self):\n"
               "        self.assertEqual(build('a'), 'echo a')\n")
    attack = json.dumps({"title": "Claimed", "severity": "high",
                         "path": "src/tool.py", "line": 6, "detail": "x",
                         "test": passing})
    session = _session(root, [attack])
    rt = redteam.RedTeam(scope="src/tool.py", files=["src/tool.py"],
                         threats=[redteam.Threat(id="t1", boundary="name",
                                                 asset="shell", exposure="e",
                                                 path="src/tool.py")])
    out = redteam.attack(session, rt)
    assert out[0].demonstrated is False
    assert out[0] in rt.undemonstrated and out[0] not in rt.demonstrated
    assert redteam.to_repairs(rt) == [], "an undemonstrated finding is a guess"


def test_a_reproducer_that_leaves_the_sandbox_is_refused(tmp_path):
    root = _project(tmp_path)
    escaping = ("import unittest\nimport subprocess\n\n\n"
                "class T(unittest.TestCase):\n"
                "    def test_x(self):\n"
                "        subprocess.run(['ls'])\n")
    attack = json.dumps({"title": "Escapes", "severity": "high",
                         "path": "src/tool.py", "line": 1, "detail": "d",
                         "test": escaping})
    session = _session(root, [attack])
    rt = redteam.RedTeam(scope="src/tool.py", files=["src/tool.py"],
                         threats=[redteam.Threat(id="t1", boundary="b",
                                                 asset="a", exposure="e",
                                                 path="src/tool.py")])
    out = redteam.attack(session, rt)
    assert out[0].demonstrated is False and out[0].reproducer == ""
    assert "refused by the guard" in out[0].detail
    assert not (root / "tests" / "redteam").exists()


def test_the_attacker_withdraws_by_id_in_conversation(tmp_path):
    root = _project(tmp_path)
    session = _session(root, [
        "You are right that the name is validated upstream.\n"
        "WITHDRAW t1: the caller already rejects `;`\n"
        "WITHDRAW t9: no such finding\n"])
    rt = redteam.RedTeam(scope="src/tool.py", files=["src/tool.py"])
    rt.findings.append(redteam.RedFinding(
        id="t1", title="Shell injection", severity="high", path="src/tool.py",
        line=6, detail="d", source="attacker", demonstrated=True,
        command="python -m unittest tests.redteam.test_rt_t1"))
    reply = redteam.converse(session, rt, "the caller validates the name")
    assert "WITHDRAW t1" in reply
    assert rt.findings[0].withdrawn == "the caller already rejects `;`"
    assert rt.findings[0] not in rt.demonstrated
    assert redteam.to_repairs(rt) == []
    assert "withdrawn" in redteam.report(rt).lower()


# ---------------------------------------------------------------------------
# 4. Blueprint
# ---------------------------------------------------------------------------

def _bp(*mods: dict) -> blueprint.Blueprint:
    return blueprint.from_dict({"system": "a game", "modules": list(mods)})


def test_a_blueprint_is_refused_for_shared_contracts_cycles_and_orphans():
    bp = _bp({"id": "a", "files": ["a.py"], "depends": ["b"],
              "contract": [{"name": "X", "signature": "def X()"}]},
             {"id": "b", "files": ["b.py", "a.py"], "depends": ["a"],
              "contract": [{"name": "X", "signature": "def X()"}]},
             {"id": "c", "files": [], "depends": ["ghost"]})
    assert not bp.ok
    joined = "\n".join(bp.problems)
    assert "contract `X` is provided by a, b" in joined
    assert "file `a.py` is owned by a, b" in joined
    assert "dependency cycle" in joined
    assert "depends on `ghost`" in joined
    assert "owns no files" in joined


def test_order_builds_dependencies_first():
    bp = _bp({"id": "ui", "files": ["ui.py"], "depends": ["core", "io"]},
             {"id": "io", "files": ["io.py"], "depends": ["core"]},
             {"id": "core", "files": ["core.py"]})
    assert bp.ok
    assert blueprint.order(bp) == ["core", "io", "ui"]


def test_stubs_are_written_for_unbuilt_dependencies_and_the_brief_is_scoped(
        tmp_path):
    root = _project(tmp_path)
    session = _session(root, [])
    bp = _bp({"id": "core", "files": ["src/core.py", "tests/test_core.py"],
              "contract": [
                  {"name": "Ship", "kind": "class",
                   "signature": "class Ship(hull: float)", "doc": "a hull"},
                  {"name": "damage", "kind": "function",
                   "signature": "def damage(ship: Ship, amount: float) -> Ship"},
                  {"name": "MAX_HULL", "kind": "constant",
                   "signature": "MAX_HULL = 100.0"}]},
             {"id": "ui", "files": ["src/ui.py"], "depends": ["core"],
              "purpose": "draw the ship"})
    ui = bp.module("ui")
    written = blueprint.write_stubs(session, bp, ui)
    assert written == ["src/core.py"]
    stub = (root / "src" / "core.py").read_text()
    assert "CONTRACT STUB for module `core`" in stub
    assert "class Ship:" in stub and "def __init__(self, hull: float)" in stub
    assert "def damage(ship: Ship, amount: float) -> Ship:" in stub
    assert "MAX_HULL = 100.0" in stub
    import ast
    ast.parse(stub)
    # a built dependency is real code and is left alone
    bp.module("core").status = "built"
    (root / "src" / "core.py").write_text("REAL = True\n")
    assert blueprint.write_stubs(session, bp, ui) == []
    assert (root / "src" / "core.py").read_text() == "REAL = True\n"
    text = blueprint.brief(bp, ui, "- core: built")
    assert "module `ui`" in text and "src/ui.py" in text
    assert "CONTRACTS THIS MODULE MAY CALL" in text and "def damage" in text
    assert "- core: built" in text
    assert "draw the ship" in text
    assert "a game" in text and "tests/test_core.py" not in text.split(
        "CONTRACTS THIS MODULE MAY CALL")[0], "its own files only"


def test_plan_then_decompose_carries_the_agreed_boundaries(tmp_path):
    root = _project(tmp_path)
    design = json.dumps({"system": "a ship game", "modules": [
        {"id": "core", "name": "Core", "purpose": "state",
         "files": ["src/core.py"], "contract": [
             {"name": "Ship", "kind": "class", "signature": "class Ship()"}]},
        {"id": "ui", "name": "UI", "purpose": "draw", "files": ["src/ui.py"],
         "depends": ["core"]}]})
    session = _session(root, ["Is the network in scope?", design])
    plan = blueprint.Plan(spec="A ship game. Must work offline.")
    reply = blueprint.converse(session, plan, "let's plan a ship game")
    assert reply == "Is the network in scope?"
    plan.messages.append(blueprint.Message(role="user",
                                           content="No network, ever."))
    bp = blueprint.decompose(session, plan.spec, plan=plan)
    assert bp.ok and [m.id for m in bp.modules] == ["core", "ui"]
    # the architect's prompt carried the agreed conversation
    sent = session.host.llm.prompts[-1]
    text = "\n".join(m.content for m in sent)
    assert "WHAT WAS AGREED IN PLANNING" in text and "No network, ever." in text
    ledger = blueprint.ledger_text(bp)
    assert "core → ui" in ledger and "## ui — UI  [planned]" in ledger
    again = blueprint.from_json(blueprint.to_json(bp))
    assert [m.id for m in again.modules] == ["core", "ui"]


def test_a_decomposition_the_model_did_not_give_is_a_problem_not_a_crash(
        tmp_path):
    root = _project(tmp_path)
    # structured.ask retries once when the first reply is not an object,
    # so two prose replies are needed to exercise the give-up path.
    session = _session(root, ["I would rather not.", "Still no."])
    bp = blueprint.decompose(session, "a thing")
    assert not bp.ok and bp.modules == []
    assert "did not return a decomposition" in bp.problems[0]


def test_run_module_records_the_outcome_in_the_ledger(tmp_path):
    root = _project(tmp_path)
    bp = _bp({"id": "core", "name": "Core", "files": ["src/core.py"],
              "contract": [{"name": "Ship", "kind": "class",
                            "signature": "class Ship()"}]},
             {"id": "ui", "name": "UI", "files": ["src/ui.py"],
              "depends": ["core"]})

    class _Outcome:
        def __init__(self, path, ok):
            self.path, self.ok = path, ok

    class _Fake:
        """A session whose `run` is scripted: the shape is what is pinned."""
        def __init__(self, session, outcomes):
            self.host = session.host
            self.journal = session.journal
            self.outcomes = outcomes
            self.requests: list[str] = []

        def run(self, request):
            self.requests.append(request)
            return self.outcomes

    real = _session(root, [])
    fake = _Fake(real, [_Outcome("src/ui.py", True)])
    ui = bp.module("ui")
    blueprint.run_module(fake, bp, ui)
    assert ui.status == "built"
    assert "stubs written: src/core.py" in ui.note
    assert "module `ui`" in fake.requests[0]
    ledger = (root / blueprint.LEDGER).read_text()
    assert "## ui — UI  [built]" in ledger
    fake2 = _Fake(real, [_Outcome("src/core.py", False)])
    blueprint.run_module(fake2, bp, bp.module("core"))
    assert bp.module("core").status == "failed"
    assert "not finished: src/core.py" in bp.module("core").note
    assert "[failed]" in (root / blueprint.LEDGER).read_text()


# ---------------------------------------------------------------------------
# structured — the grammar + one-retry layer every tab asks through
# ---------------------------------------------------------------------------

def test_structured_retries_once_when_the_first_reply_is_not_the_object(
        tmp_path):
    from cognitive_coder import structured
    from cognitive_coder.personas import PERSONAS
    root = _project(tmp_path)
    session = _session(root, [
        "Sure, here are the bugs you asked about: (prose, no JSON)",
        '{"bugs": [{"path": "src/calc.py", "line": 3, "message": "x"}]}'])
    got = structured.ask(session, PERSONAS["reviewer"], "find bugs",
                         keys=("bugs",), example='{"bugs": []}')
    assert got.ok and got.attempts == 2
    assert got.data["bugs"][0]["line"] == 3
    assert got.grammar is True, "the scripted host advertises grammar support"
    # the retry quoted the problem back to the model
    second = session.host.llm.prompts[-1]
    assert any("not the object asked for" in m.content for m in second)


def test_structured_gives_up_after_two_failures_without_crashing(tmp_path):
    from cognitive_coder import structured
    from cognitive_coder.personas import PERSONAS
    root = _project(tmp_path)
    session = _session(root, ["no", "still no"])
    got = structured.ask(session, PERSONAS["reviewer"], "t",
                         keys=("bugs",), example='{"bugs": []}')
    assert not got.ok and got.attempts == 2
    assert got.data is None and got.problem


def test_structured_takes_the_first_good_reply_without_retrying(tmp_path):
    from cognitive_coder import structured
    from cognitive_coder.personas import PERSONAS
    root = _project(tmp_path)
    session = _session(root, ['{"bugs": []}'])      # empty list is a success
    got = structured.ask(session, PERSONAS["reviewer"], "t",
                         keys=("bugs",), example='{"bugs": []}')
    assert got.ok and got.attempts == 1 and got.data == {"bugs": []}
