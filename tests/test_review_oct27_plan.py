# SPDX-License-Identifier: Apache-2.0
"""Review of the 2026-10-27 change (528f931..HEAD): the size rule, the
interview, the worked examples in the contracts and the bench calibration.

One test per suspected defect. Each FAILS against the code as it stands;
its docstring says what happens and what the correct behaviour is.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
    bench,
    constraints,
    interfaces,
    interview,
    personas,
    profiles,
    sizing,
    symedit,
)
from cognitive_coder import planner as planner_mod  # noqa: E402
from cognitive_coder.personas import PERSONAS, PromptBuilder  # noqa: E402


def _long_function(n: int, name: str = "work") -> str:
    body = "\n".join(f"    x{i} = {i}" for i in range(n - 1))
    return f"def {name}():\n{body}\n    return x0\n"


def _question(text: str, options, default: str) -> dict:
    return {"question": text, "options": list(options), "default": default}


# --------------------------------------------------------------------------
# THE SIZE RULE
# --------------------------------------------------------------------------

def test_size_rule_is_never_measured_on_the_ordinary_build_path(tmp_path):
    """`Session._measure_size` is reached only through `Session._record`,
    which the repair steps and the CONTINUE re-verify call. The ordinary
    path every task takes — `Session.step()` — appends the outcome and
    writes the `verify` event itself and never measures. So a file a
    normal build verified at 61 lines in one function (the rule is 40)
    produces no `size` journal event, no SIZE line and no `[size]` report
    caveat. (tests/test_sizing.py tests the feature by calling `_record`
    directly, which is why it passes.) Correct: a verified file over the
    rule is measured whichever path verified it."""
    plan = "src/alpha.py — the first thing\n"
    reply = "```python\n" + _long_function(61, "alpha") + "```"
    host = Host(llm=ScriptedLLM([plan, reply], supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(
        attempts=1, pin_interfaces=False, mutation_check=False,
        review_after_build=False, max_function_lines=40))
    outcomes = session.run("one module")
    assert [o.ok for o in outcomes] == [True]
    written = (tmp_path / "src" / "alpha.py").read_text()
    assert sizing.measure(written, "src/alpha.py").over, "fixture is wrong"
    assert session.size_findings, "the verified file was never measured"
    assert any(e.get("event") == "size" for e in session.journal.events())
    assert "[size]" in session.report()


def test_planner_persona_still_asks_for_the_smallest_set_of_files():
    """sizing.py's own docstring names the cause of oversized files: the
    planner persona "asked for 'the smallest set of files', which a model
    reads as FEWER files — and so bigger ones". The task text was changed;
    `PLANNER.self_model` was not. The planner's prompt now says both
    "the smallest set of files" ([SELF-MODEL], first) and "More small
    files beat fewer large ones" ([TASK], later), and a small model
    follows the first thing it was told about itself. (test_sizing.py's
    `test_the_planner_prompt_asks_for_small_files` checks the task text
    alone, so it cannot see this.) Correct: no sentence of the planner's
    prompt asks for the fewest files."""
    builder = PromptBuilder(conventions=sizing.rule_text(200, 40))
    task = planner_mod._plan_prompt(
        "a game", "python", {}, size_rule=sizing.planner_rule(200, 40))
    prompt = builder.build(PERSONAS["planner"], task,
                           contract=personas.CONTRACT_LIST)
    full = "\n".join(m.content for m in prompt.messages())
    assert "More small files beat fewer large ones" in full, "fixture"
    assert "smallest set of files" not in full


def test_architect_rule_asks_for_a_split_the_skeleton_throws_away():
    """The architect is told "one stub per file listed below" and, by
    `sizing.architect_rule`, that "a file that would need more is two
    files with a contract between them". It cannot add a file: a block for
    a path the plan does not list is dropped by `interfaces.split_reply`
    ("Unmatched blocks are ignored"), while the kept stub's import of that
    file survives `sanitise` — so following the rule pins a stub that
    imports a module nothing provides, and the skeleton check warns "the
    file split is probably wrong" about a split the engine's own prompt
    asked for. Correct: the architect is told to SAY a module is too
    large (a note the operator reads), never to split the plan's files."""
    rule = sizing.architect_rule(200, 40)
    reply = (
        "```python\n# file: src/engine.py\n"
        "from src.engine_ai import plan_moves\n\n\n"
        "def step(ships: list) -> None:\n"
        '    """Advance one tick."""\n    raise NotImplementedError\n```\n'
        "```python\n# file: src/engine_ai.py\n"
        "def plan_moves(ships: list) -> list:\n"
        '    """The AI\'s orders."""\n    raise NotImplementedError\n```\n')
    blocks = interfaces.split_reply(reply, ["src/engine.py"])
    assert list(blocks) == ["src/engine.py"], "fixture: the split is dropped"
    stub, _dropped = interfaces.sanitise(blocks["src/engine.py"], "engine")
    assert "from src.engine_ai import plan_moves" in stub, "fixture"
    # the rule the architect is given must not ask for what is thrown away
    assert "two files" not in rule, rule


# --------------------------------------------------------------------------
# THE INTERVIEW
# --------------------------------------------------------------------------

def test_interview_answer_is_lost_when_the_question_holds_a_quoted_example():
    """`Interview.decisions_text` writes one line per decision:
    `- The code MUST follow this decision: <question>: <answer>.` and
    relies on constraints.py to pin it. constraints `_SENTENCE` splits at
    `[.!?]` + space + `[A-Z0-9"'(]`, so a question with an abbreviation or
    a parenthesis before a quote, digit or capital — `e.g. 'km/h'`, `up?
    (i.e. …)` — is split in two: the half with MUST is pinned without the
    answer, the half with the answer has no MUST and is dropped. The
    operator's decision is in no prompt. Correct: the pinned sentence
    carries the answer."""
    asked = interview.Interview(questions=interview.clean([
        _question("Which unit should speed use, e.g. 'km/h' or 'm/s'?",
                  ["km/h", "m/s"], "km/h")]))
    interview.answer(asked, {"q1": "m/s"})
    found = constraints.extract(asked.apply("A deterministic sim."))
    assert found.items, "nothing pinned at all"
    assert any("m/s" in c and "MUST" in c for c in found.items), found.items


def test_interview_answer_is_cut_off_when_the_question_is_long():
    """`clean` keeps a question up to 240 characters; the decision line
    adds 36 characters of lead ("The code MUST follow this decision: ")
    and the answer after the question. constraints.py cuts a pinned
    sentence at MAX_CHARS (240) — so for any question over ~200
    characters the pinned sentence ends "… (cut here)" BEFORE the answer,
    and every prompt carries the question with no decision. Correct: the
    pinned text holds the answer (the decision is what matters, the
    question's wording is not)."""
    question = ("When two ships occupy the same cell at the end of a turn, "
                "which one is considered to have arrived first for the "
                "purpose of resolving the collision and the damage that "
                "follows from it, the faster one or the one with the lower "
                "id?")
    assert len(question) <= 240, "fixture: clean() would keep it whole"
    asked = interview.Interview(questions=interview.clean([
        _question(question, ["the faster one", "the lower id"],
                  "the faster one")]))
    interview.answer(asked, {"q1": "the lower id"})
    found = constraints.extract(asked.apply("A deterministic sim."))
    assert any("the lower id" in c for c in found.items), found.items


def test_interview_decisions_are_dropped_when_the_request_has_fifteen_musts():
    """The decisions block is appended at the END of the request, and
    constraints.py pins the first MAX_CONSTRAINTS (15) sentences in
    document order. A written specification — the kind of request that
    needs an interview — easily has fifteen MUST sentences of its own, so
    every confirmed decision lands in "… and N more, not pinned". The
    module docstring promises the opposite ("so the sixth file's repair
    still knows which way is north"). Correct: a decision the operator
    just confirmed is pinned ahead of, or regardless of, the cap."""
    spec = "\n".join(f"The game MUST do thing number {i}." for i in range(15))
    asked = interview.Interview(questions=interview.clean([
        _question("Which way is north on the grid?", ["up", "down"], "up")]))
    interview.answer(asked, {"q1": "down"})
    found = constraints.extract(asked.apply(spec))
    assert any("north" in c for c in found.items), (found.items, found.more)


def test_options_given_as_one_string_are_not_a_question():
    """`clean` iterates `item["options"]` without checking it is a list.
    A model that writes `"options": "up or down"` yields the CHARACTERS
    of the string, so the operator is asked "Which way is north?" with
    options `u` / `p` / `o` / `r` and the default `u` — and `u` is then
    pinned into every prompt as the decision. Correct: options that are
    not a list of strings make the question unusable."""
    got = interview.clean([{"question": "Which way is north on the grid?",
                            "options": "up or down", "default": "up"}])
    assert got == [], [(q.options, q.default) for q in got]


def test_the_schema_echoed_back_is_not_a_question():
    """structured.py names the failure mode: a small model returns "the
    schema itself with the example values still in it". `clean` accepts
    the schema's own example as a question — the operator is asked "one
    plain sentence a non-programmer can answer" with the options "the
    first way" / "the second way", and the default is pinned into every
    prompt as `The code MUST follow this decision: one plain sentence a
    non-programmer can answer: the first way.` Correct: the schema's
    example is dropped like a duplicate."""
    echoed = json.loads(interview._SCHEMA)["questions"]
    got = interview.clean(echoed)
    assert got == [], [q.question for q in got]


def test_ctrl_c_in_the_terminal_interview_does_not_confirm_the_defaults():
    """`ask_on_terminal` catches KeyboardInterrupt from `input()` and
    treats it as Enter: the model's default stands, the loop goes on to
    the next question, and the build starts with every remaining decision
    pinned under "## Decisions confirmed by the operator". Ctrl+C is the
    operator saying stop (the CLI's `ConsoleApproval` reads it as "no",
    `_run` as cancel). Correct: an interrupt leaves the interview, with
    nothing confirmed in the operator's name."""
    asked = interview.Interview(questions=interview.clean([
        _question("Which way is north on the grid?", ["up", "down"], "up"),
        _question("Do ids count from 0 or 1?", ["0", "1"], "0")]))
    calls = []

    def read(prompt: str) -> str:
        calls.append(prompt)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interview.ask_on_terminal(asked, read=read, write=lambda _s: None)
    assert len(calls) == 1, "the second question was asked after Ctrl+C"
    assert asked.answers == {}


# --------------------------------------------------------------------------
# WORKED EXAMPLES IN THE CONTRACTS
# --------------------------------------------------------------------------

def test_an_echoed_symbols_example_overwrites_a_real_beacon():
    """`CONTRACT_SYMBOLS` ends with a worked reply that changes
    `Beacon.bearing_to`. `symedit.reply_code` joins EVERY Python fence of
    a reply, so a model that repeats the example before its own edit (the
    copying the example exists to encourage) hands the parser two fences
    — and in a project that has a `Beacon` with a `bearing_to` (a
    direction-finding tool: the operator's own domain) the example's body
    REPLACES the real method, its convention comment and all, while the
    placement is journaled as a success. Correct: the contract's own
    example, repeated verbatim in a reply, is never placed."""
    example = personas.CONTRACT_SYMBOLS.split("reads, in full:", 1)[1].strip()
    real = (
        '"""Bearings between beacons for the DF tab."""\n'
        "import math\n\n\n"
        "class Beacon:\n"
        "    def __init__(self, x, y):\n"
        "        self.x = x\n        self.y = y\n\n"
        "    def bearing_to(self, other):\n"
        "        # north is (0, -1): bearing clockwise from north\n"
        "        dx, dy = other.x - self.x, other.y - self.y\n"
        "        return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0\n"
        "\n\n"
        "def tally(count, limit):\n"
        "    if count > limit:\n        return True\n    return False\n")
    reply = (example + "\n\n```python\ndef tally(count, limit):\n"
             "    if count >= limit:\n        return True\n    return False\n"
             "```\n")
    applied = symedit.apply(real, reply, targets=["tally"])
    assert applied.ok, applied.problem
    assert "count >= limit" in applied.code
    assert "north is (0, -1)" in applied.code, applied.replaced
    assert "Beacon.bearing_to" not in applied.replaced


# --------------------------------------------------------------------------
# BENCH CALIBRATION
# --------------------------------------------------------------------------

def test_calibration_with_no_evidence_does_not_pin_a_format(tmp_path,
                                                            monkeypatch):
    """When every task of every format errored (the server died, a
    refused connection: `run_task` turns an exception into a row with
    `error`), `measure` reports 0 passed / 0 generations for each format
    and `choose_format` breaks the three-way tie at zero by "the narrowest
    wins" — `symbols`. `calibrate` then writes bench/profiles/<stem>.json
    with `edit_format: "symbols"`, which `profile_for` finds first on every
    later load of that model and the session turns into `narrow="always"`
    — a setting measured by nothing. Correct: a calibration that measured
    no generation writes no format (`auto`) and says so."""
    def no_evidence(make_llm, *, suite, out, only, label, config, say,
                    events, ex):
        rows = [{"task": "t", "ok": False, "error": "ConnectionError: refused",
                 "generations": 0, "hidden": {"passed": 0, "total": 2}}]
        run_dir = tmp_path / "runs" / label
        run_dir.mkdir(parents=True)
        return bench.BenchRun(path=run_dir, rows=rows, problems=[])

    monkeypatch.setattr(bench, "run_suite", no_evidence)
    dead = bench.measure([{"ok": False, "error": "x", "generations": 0,
                           "hidden": {"passed": 0, "total": 2}}])
    assert bench.choose_format({"whole": dead, "symbols": dead,
                                "fim": dead}) == "auto"
    report = bench.calibrate(lambda t, f: None, "Qwen3-Coder-30B-Q4.gguf",
                             suite=tmp_path, out=tmp_path / "runs",
                             say=lambda _s: None,
                             profile_dir=tmp_path / "profiles")
    assert report["chosen"] == "auto"
    found = profiles.profile_for("Qwen3-Coder-30B-Q4.gguf",
                                 directory=tmp_path / "profiles")
    assert found.edit_format == "auto", found.edit_format
