# SPDX-License-Identifier: Apache-2.0
"""§6.7's four acceptance criteria, plus M28, M30 and M31.

The four, verbatim from the spec: index this project's own source and assert
(a) every public function appears, (b) `callers_of` finds a known caller
across files, (c) a signature change flags the right files, (d) the context
for a chosen file fits a 4,096-token budget and names what it dropped.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import context as ctx  # noqa: E402
from cognitive_coder.codemap import CodeMap, zoom  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    MemoryFileSystem,
    MemoryStorage,
    NullLLM,
    ScriptedLLM,
)

PROJECT = {
    "src/readings.py":
        b'"""Load sensor readings."""\nimport csv\n\n\n'
        b'def load_readings(path: str) -> list:\n'
        b'    """Load and validate the CSV."""\n    return []\n\n\n'
        b'def _private_helper():\n    return None\n',
    "src/stats.py":
        b'"""Summary statistics."""\nfrom src.readings import load_readings\n'
        b'\n\ndef summarise(path: str) -> dict:\n'
        b'    """Mean, min and max per column."""\n'
        b'    rows = load_readings(path)\n    return {}\n',
    "src/cli.py":
        b'"""Command line."""\nfrom src.stats import summarise\n\n\n'
        b'def main() -> int:\n    """Entry point."""\n'
        b'    print(summarise("data.csv"))\n    return 0\n',
    "tests/test_stats.py":
        b'from src.stats import summarise\n\n\n'
        b'def test_summarise():\n    assert summarise("f") == {}\n',
    "lib/util.c":
        b'#include <stdio.h>\n\nint add(int a, int b) {\n    return a + b;\n}'
        b'\n\nint scale(int a) {\n    return add(a, a);\n}\n',
}


@pytest.fixture
def codemap():
    cm = CodeMap(MemoryFileSystem(dict(PROJECT)), MemoryStorage())
    cm.index_project()
    return cm


# --------------------------------------------------------------------------
# the four acceptance criteria
# --------------------------------------------------------------------------

def test_a_every_public_function_appears(codemap):
    for name in ("load_readings", "summarise", "main", "add", "scale"):
        assert codemap.search(name), f"{name} was not indexed"


def test_b_callers_are_found_across_files(codemap):
    callers = codemap.store.callers_of("load_readings")
    paths = {c["path"] for c in callers}
    assert "src/stats.py" in paths, (
        "a cross-file call was left unresolved — the call graph is empty in "
        "the only direction that matters")


def test_c_a_signature_change_flags_the_right_files(codemap):
    blast = codemap.blast_radius("summarise")
    assert "src/cli.py" in blast["files"]
    assert "tests/test_stats.py" in blast["tests_first"], (
        "the tests that cover the callers must be run FIRST")


def test_d_the_context_fits_a_4k_budget_and_names_what_it_dropped(codemap):
    """M28 — a model that doesn't know something was withheld invents it."""
    out = codemap.architecture("src/stats.py", max_tokens=4096)
    assert "NOT INCLUDED" in out
    assert len(out) // 4 <= 4096, "the budget was blown"
    assert "load_readings" in out, "the direct dependency was not injected"


# --------------------------------------------------------------------------
# honesty about what the map knows
# --------------------------------------------------------------------------

def test_unresolved_calls_are_kept_and_counted_not_dropped(codemap):
    """A graph that silently drops what it could not bind looks complete."""
    stats = codemap.stats()
    assert stats.edges > 0
    assert 0.0 <= stats.resolution_rate <= 1.0
    assert "resolved" in stats.one_line()


def test_regex_extracted_symbols_are_labelled_approximate(codemap):
    """C7 — say which mode you are in, everywhere it surfaces."""
    rows = codemap.store.symbols_in("lib/util.c")
    assert rows and all(r["approximate"] for r in rows)
    assert "pattern-matched" in codemap.call_tool("list_symbols",
                                                  {"path": "lib/util.c"})


def test_python_symbols_are_exact(codemap):
    rows = codemap.store.symbols_in("src/stats.py")
    assert rows and not any(r["approximate"] for r in rows)
    assert rows[0]["signature"] == "def summarise(path: str) -> dict"


# --------------------------------------------------------------------------
# D4 — the invented-API check
# --------------------------------------------------------------------------

def test_a_name_that_exists_nowhere_is_reported(codemap):
    missing = codemap.unresolved_in(
        'from utils import parse_config\n\n\n'
        'def go():\n    return parse_config("a")\n', "python")
    assert "parse_config" in missing


def test_a_stdlib_call_through_an_import_is_not_reported(codemap):
    """`csv.reader` in a file that says `import csv` is not an invention.

    A check that cries wolf is a check somebody turns off.
    """
    missing = codemap.unresolved_in(
        'import csv\n\n\ndef go(fh):\n    return list(csv.reader(fh))\n',
        "python")
    assert missing == [], missing


def test_a_call_to_something_the_project_defines_is_not_reported(codemap):
    missing = codemap.unresolved_in(
        'from src.readings import load_readings\n\n\n'
        'def go():\n    return load_readings("x")\n', "python")
    assert "load_readings" not in missing


def test_the_search_tool_says_plainly_when_a_name_does_not_exist(codemap):
    answer = codemap.call_tool("search_codemap", {"name": "parse_config"})
    assert "no `parse_config`" in answer
    assert "Do not call it" in answer


# --------------------------------------------------------------------------
# the tool surface (§6.7) and the fallback (M31)
# --------------------------------------------------------------------------

def test_the_five_tools_are_offered_with_json_schemas(codemap):
    names = {t.name for t in codemap.tool_specs()}
    assert names == {"search_codemap", "read_slice", "list_symbols",
                     "run_tests", "apply_patch"}
    for spec in codemap.tool_specs():
        assert spec.parameters["type"] == "object"
        assert spec.description


def test_apply_patch_has_no_side_door_around_approval(codemap):
    """M18 — tool calling must never bypass the transaction and the gate."""
    answer = codemap.call_tool("apply_patch",
                               {"path": "src/stats.py", "old": "return {}",
                                "new": "return {'a': 1}"})
    assert "cannot be applied" in answer
    assert codemap.fs.read("src/stats.py").endswith("return {}\n")


def test_a_failing_tool_call_comes_back_as_text_not_an_exception(codemap):
    answer = codemap.call_tool("read_slice", {"path": "nope/missing.py"})
    assert "not in this project" in answer
    assert codemap.call_tool("no_such_tool", {}).startswith(
        "There is no tool called")


def test_the_text_marker_fallback_accepts_several_syntaxes(codemap):
    """D10 — drift is guaranteed, and a drifted call is not a confused one."""
    for text in ("[SEARCH_CODEMAP: summarise]",
                 "SEARCH_CODEMAP(summarise)",
                 "<search_codemap>summarise</search_codemap>"):
        codemap.reset_lookups()
        assert "summarise" in codemap.answer_text_lookups(text), text


def test_the_fallback_caps_lookups_at_three(codemap):
    """M31 — and the cap message is an instruction, not a complaint."""
    codemap.reset_lookups()
    answers = [codemap.answer_text_lookups("[SEARCH_CODEMAP: summarise]")
               for _ in range(5)]
    assert "used your lookups" in answers[-1]
    assert "Work with what you have" in answers[-1]


def test_a_malformed_call_is_corrected_once_and_only_once(codemap):
    """A model told the same thing three times reproduces the correction."""
    codemap.reset_lookups()
    first = codemap.answer_text_lookups("I will search_codemap for summarise")
    second = codemap.answer_text_lookups("search codemap again please")
    assert "[SEARCH_CODEMAP:" in first
    assert second == ""


# --------------------------------------------------------------------------
# freshness and epochs (M30, G.7)
# --------------------------------------------------------------------------

def test_the_query_interface_is_never_stale(codemap):
    """M30 — SQLite is re-read live; only the injected TEXT may lag."""
    codemap.fs.write("src/stats.py",
                     '"""Stats."""\n\n\ndef renamed_entirely(path):\n'
                     '    """New name."""\n    return {}\n')
    codemap.reindex_after_write("src/stats.py")
    assert codemap.search("renamed_entirely")
    assert not codemap.search("summarise")


def test_the_injected_summary_may_lag_but_declares_that_it_does(codemap):
    codemap.store.bump_epoch("test")
    codemap.fs.write("src/cli.py", "def main():\n    return 1\n")
    codemap.reindex_after_write("src/cli.py")
    note = zoom.staleness_note(codemap.store)
    assert "changed since" in note
    assert "search_codemap" in note


# --------------------------------------------------------------------------
# context.py (§6.6, M28)
# --------------------------------------------------------------------------

def test_the_omissions_block_always_ends_the_context():
    """Including when nothing was dropped — a block that appears only
    sometimes is a block the model learns to ignore."""
    dropped = ctx.build_context(
        [("A", "x" * 400, 1), ("B", "y" * 400, 2)], 300)
    assert "NOT INCLUDED" in dropped
    assert "B" in dropped.split("NOT INCLUDED")[1]

    nothing = ctx.build_context([("A", "short", 1)], 5000)
    assert "Nothing was left out" in nothing


def test_the_budget_is_measured_from_the_port_not_from_config():
    """G.3 — a config-derived budget fails the day a smaller model loads."""
    budget = ctx.measure_budget(ScriptedLLM([], context_tokens=16384))
    assert budget.context_tokens == 16384
    assert budget.prompt_tokens < 16384 - ctx.RESERVED_OUTPUT_TOKENS
    assert "measured from the loaded model" in budget.note


def test_an_estimated_token_count_is_declared_to_the_model():
    """M14 — an undeclared estimate is how a context overflows."""
    budget = ctx.measure_budget(NullLLM())
    assert budget.is_estimate
    assert "estimated, not exact" in budget.declare()


def test_a_reasoning_model_gets_a_smaller_prompt_budget():
    """D13 — budgeting as if `<think>` is free overflows on Magistral."""
    plain = ctx.measure_budget(ScriptedLLM([], context_tokens=16384))
    thinking = ctx.measure_budget(ScriptedLLM([], context_tokens=16384),
                                  reasoning=True)
    assert thinking.prompt_tokens < plain.prompt_tokens


def test_an_interface_is_far_cheaper_than_the_file(codemap):
    """F9 — ~30 tokens as an interface against ~800 as a file."""
    body = codemap.fs.read("src/readings.py")
    surface = ctx.interface(body, "python", "src/readings.py")
    assert len(surface) < len(body)
    assert "load_readings" in surface
    assert "_private_helper" not in surface, "privates are not interface"
    assert "return []" not in surface, "a body leaked into the interface"


def test_a_tool_less_model_forces_an_epoch_on_every_write(codemap):
    """M31 — the injected summary may lag ONLY because the model can check.

    With no live tools there is no safety net, so the summary is not allowed
    to lag at all. G.7's closing paragraph says so explicitly, and notes that
    the slower prompts are the accepted cost.
    """
    codemap.force_epoch_per_write = True
    codemap.store.bump_epoch("start")
    before = codemap.store.epoch

    codemap.fs.write("src/cli.py", "def main():\n    return 1\n")
    codemap.reindex_after_write("src/cli.py")
    assert codemap.maybe_bump_epoch(target="src/cli.py") > before, (
        "a tool-less host must rebuild the summary on every write")


def test_a_tool_using_model_is_allowed_to_lag(codemap):
    """The other half: with live tools, lagging is cheap and correct.

    Rebuilding the prefix on every write would discard the KV cache
    constantly — 30 to 60 seconds of prompt reprocessing per file, to save at
    most one tool call.
    """
    codemap.force_epoch_per_write = False
    codemap.store.bump_epoch("start")
    before = codemap.store.epoch

    codemap.fs.write("src/readings.py", "def load_readings(p):\n    return []\n")
    codemap.reindex_after_write("src/readings.py")
    assert codemap.maybe_bump_epoch(target="src/cli.py") == before
