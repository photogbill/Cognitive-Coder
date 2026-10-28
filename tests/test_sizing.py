# SPDX-License-Identifier: Apache-2.0
"""Plan for the model's reach (sizing.py, 2026-10-27).

The rule reaches the planner, the architect and every prompt's cached
prefix; a stub that will outgrow its module is said at the skeleton; a
verified file over the rule is measured, said and journaled — and never
refused.
"""

from __future__ import annotations

from cognitive_coder import interfaces, sizing
from cognitive_coder import planner as planner_mod
from cognitive_coder.personas import PromptBuilder
from cognitive_coder.ports import (
    AutoApprove,
    Host,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.session import Session, SessionConfig

SMALL = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"


def _long_function(n: int, name: str = "work") -> str:
    body = "\n".join(f"    x{i} = {i}" for i in range(n - 1))
    return f"def {name}():\n{body}\n    return x0\n"


def test_measure_python_counts_lines_and_every_def():
    code = SMALL + "\n\nclass K:\n    def m(self):\n        return 3\n"
    m = sizing.measure(code, "src/k.py", max_module=200, max_function=40)
    assert m.lines == code.count("\n")
    assert [d.name for d in m.defs] == ["a", "b", "K.m"]
    assert not m.over


def test_a_long_function_is_over_the_function_rule():
    code = _long_function(60)
    m = sizing.measure(code, "src/w.py", max_module=200, max_function=40)
    assert not m.over_module
    assert [d.name for d in m.over_functions] == ["work"]
    assert "`work` (61 lines)" in m.describe()
    assert "It works" in m.describe()


def test_a_long_module_is_over_the_module_rule():
    code = "\n".join(f"X{i} = {i}" for i in range(250)) + "\n"
    m = sizing.measure(code, "src/c.py", max_module=200, max_function=40)
    assert m.over_module and not m.over_functions
    assert "250 lines (the rule is 200)" in m.describe()


def test_zero_switches_a_rule_off():
    code = _long_function(60)
    m = sizing.measure(code, "src/w.py", max_module=0, max_function=0)
    assert not m.over
    assert sizing.rule_text(0, 0) == ""
    assert sizing.planner_rule(0, 0) == ""


def test_a_file_that_does_not_parse_is_measured_as_lines_only():
    m = sizing.measure("def (:\n", "src/bad.py")
    assert not m.parsed and m.lines == 1 and not m.defs


def test_other_languages_are_measured_by_pattern_and_say_so():
    code = ("int main() {\n" + "\n".join("  x++;" for _ in range(50))
            + "\n}\n\nint helper() {\n  return 1;\n}\n")
    m = sizing.measure(code, "main.c", "c", max_function=40)
    assert [d.name for d in m.defs] == ["main", "helper"]
    assert m.over_functions and m.over_functions[0].name == "main"
    assert any("approximate" in n for n in m.notes)


def test_the_rule_reaches_the_cached_prefix_of_every_prompt():
    host = Host(llm=ScriptedLLM([]), fs=MemoryFileSystem(),
                storage=MemoryStorage(), events=RecordingEvents(),
                approval=AutoApprove())
    session = Session(host, config=SessionConfig(max_module_lines=180,
                                                 max_function_lines=35))
    prefix = session.prompts.prefix_for(
        __import__("cognitive_coder").personas.ENGINEER)
    assert "[SIZE]" in prefix
    assert "180 lines" in prefix and "35 lines" in prefix


def test_the_planner_prompt_asks_for_small_files():
    text = planner_mod._plan_prompt("a game", "python", {},
                                    size_rule=sizing.planner_rule(200, 40))
    assert "about 200 lines at most" in text
    assert "More small files beat fewer large ones" in text
    assert "smallest set of files" not in text


def test_the_architect_is_told_the_sizes():
    text = interfaces.task_text("req", [("src/a.py", "a")],
                                size_rule=sizing.architect_rule(200, 40))
    assert "about 200 lines" in text and "about 40 lines" in text


def test_a_stub_that_will_outgrow_its_module_is_said():
    stub = "\n".join(f"def f{i}(x: int) -> int:\n"
                     f'    """f{i}."""\n    raise NotImplementedError\n'
                     for i in range(14))
    note = sizing.stub_warning(stub, "src/big.py", max_module=200)
    assert note.startswith("src/big.py: its pinned interface declares 14")
    assert "consider splitting" in note
    assert sizing.stub_warning(stub, "src/big.py", max_module=0) == ""
    small = "def f(x: int) -> int:\n    raise NotImplementedError\n"
    assert sizing.stub_warning(small, "src/small.py") == ""


def test_a_verified_file_over_the_rule_is_said_and_journaled_not_refused():
    """The session measures a file its loop verified; the finding is a
    journal event, a build-log line and a report line."""
    host = Host(llm=ScriptedLLM([]), fs=MemoryFileSystem(),
                storage=MemoryStorage(), events=RecordingEvents(),
                approval=AutoApprove())
    session = Session(host, config=SessionConfig(max_function_lines=10))
    host.fs.write("src/w.py", _long_function(30))
    from cognitive_coder.types import Task, TaskOutcome
    task = Task(id="t1", path="src/w.py", purpose="work", lang="python")
    session._record(task, TaskOutcome(task_id="t1", path="src/w.py",
                                      ok=True, verified=True),
                    label="VERIFIED")
    assert session.size_findings and session.size_findings[0].over
    assert any(r.get("event") == "size" for r in session.journal.events())
    assert "[size]" in session.report()
    assert session.outcomes[-1].ok


def test_prompt_builder_prefix_is_stable_with_the_rule():
    pb = PromptBuilder(conventions=sizing.rule_text(200, 40))
    from cognitive_coder.personas import ENGINEER
    assert pb.prefix_for(ENGINEER) == pb.prefix_for(ENGINEER)
