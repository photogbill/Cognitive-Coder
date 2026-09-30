# SPDX-License-Identifier: Apache-2.0
"""M32–M37 — the five loop behaviours that are not obvious and all matter.

Every test here guards something that would still *appear* to work if it
broke, which is exactly why it needs a test: truncation silently becoming
regeneration, failed attempts silently poisoning the context, a cycle
detector silently failing to detect a 3-cycle. None of these produce an
error message. They produce a tool that is quietly worse.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import personas  # noqa: E402
from cognitive_coder.codemap import CodeMap  # noqa: E402
from cognitive_coder.loop import (  # noqa: E402
    Loop,
    LoopConfig,
    _describe_cycle,
    _extract,
    _is_truncated,
    _join_continuation,
)
from cognitive_coder.patcher import Patcher  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
)
from cognitive_coder.types import (  # noqa: E402
    Completion,
    Diagnostic,
    Message,
    Task,
)


def _host(tmp_path, replies, supports_tools=False):
    return Host(llm=ScriptedLLM(replies, supports_tools=supports_tools),
                fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _loop(host, **config):
    cm = CodeMap(host.fs, host.storage)
    return Loop(host, codemap=cm,
                patcher=Patcher(host.fs, host.storage, host.approval,
                                host.events),
                config=LoopConfig(**config))


# --------------------------------------------------------------------------
# M32 — truncation is CONTINUED, not regenerated (D1)
# --------------------------------------------------------------------------

def test_truncation_is_detected_from_finish_reason_not_guessed():
    """D1 — detect it STRUCTURALLY. Delimiters are the backstop."""
    assert _is_truncated(Completion(text="def f():", finish_reason="length"),
                         "def f():")
    assert not _is_truncated(Completion(text="x = 1\n", finish_reason="stop"),
                             "x = 1\n")


def test_unbalanced_delimiters_are_the_backstop():
    """Several providers report "stop" when they mean "length"."""
    assert _is_truncated(Completion(text="", finish_reason="stop"),
                         "def f(a, b:\n    return {'k': [1, 2")


def test_a_string_containing_a_brace_is_not_mistaken_for_truncation():
    """A false positive here costs a whole extra generation, every time."""
    assert not _is_truncated(
        Completion(text="", finish_reason="stop"),
        'x = "an unmatched { in a string"\ny = 1\n')


@pytest.mark.parametrize("code", [
    # A backslash escapes the NEXT character, including another backslash.
    # `'\\'` is a one-character string, not an unterminated one.
    "paths = ['\\\\', 'x']\n",
    'p = os.path.join("C:\\\\", "x")\n',
    # A bracket inside a comment is prose, not syntax.
    "# note: dict[\nx = {}\n",
    "x = 1  # handles (a\ny = 2\n",
    # An apostrophe in a comment must not open a string that swallows the
    # rest of the file.
    "x = 1  # don't\ny = (1, 2)\nz = 'a'\n",
    # A raw regex with an escaped paren.
    'r = re.compile(r"\\(")\n',
])
def test_a_complete_python_file_is_never_mistaken_for_a_truncated_one(code):
    """Each false positive here costs up to MAX_CONTINUATIONS model calls
    and then APPENDS the model's "The file is complete." to a correct file.
    The reviewer found all four shapes in real output."""
    assert not _is_truncated(Completion(text=code, finish_reason="stop"),
                             code, "python"), code


def test_a_bracket_in_a_line_comment_is_skipped_for_other_languages():
    """The scanner is the only check for languages without a tokenizer, so
    it must know the language's comment marker."""
    js = "// note: dict[\nlet x = {};\n"
    assert not _is_truncated(Completion(text=js, finish_reason="stop"),
                             js, "javascript")
    c = "/* a ( in a block comment */\nint x = 1;\n"
    assert not _is_truncated(Completion(text=c, finish_reason="stop"), c, "c")


def test_a_genuinely_cut_file_is_still_detected():
    """The backstop must keep catching the real thing (D1)."""
    cut = "def f(:\n    return [1,"
    assert _is_truncated(Completion(text=cut, finish_reason="stop"), cut,
                         "python")
    cut_str = 'DOC = """this string\nnever ends\n'
    assert _is_truncated(Completion(text=cut_str, finish_reason="stop"),
                         cut_str, "python")
    js = "function f() {\n  return [1,"
    assert _is_truncated(Completion(text=js, finish_reason="stop"), js,
                         "javascript")


def test_a_fenced_complete_reply_is_not_truncated():
    """The truncation check runs on the raw reply, fences and all."""
    text = "```python\npaths = ['\\\\', 'x']\n```\n"
    assert not _is_truncated(Completion(text=text, finish_reason="stop"),
                             text, "python")


def test_a_truncated_generation_is_continued_and_journaled(tmp_path):
    """M32 — continuation, not regeneration. Regenerating pays twice and
    frequently produces a DIFFERENT file, which is worse than slow."""
    replies = [
        Completion(text='def parse(line):\n    """Split."""\n    parts = ',
                   finish_reason="length", model="scripted"),
        Completion(text='line.split(",")\n    return parts\n',
                   finish_reason="stop", model="scripted"),
    ]
    host = _host(tmp_path, replies)
    from cognitive_coder.journal import Journal
    loop = _loop(host, attempts=1)
    loop.journal = Journal(host.fs, "t")
    outcome = loop.run_task(Task(id="t1", path="p.py", purpose="parse a line",
                                 lang="python"))
    written = host.fs.read("p.py")
    assert "parts = line.split" in written, written
    assert written.count("def parse") == 1, "it regenerated instead"
    assert outcome.attempts[0].continued
    assert any(r.get("event") == "continuation"
               for r in loop.journal.events())


def test_a_repeated_tail_is_not_duplicated_when_continuing():
    """Models told "do not repeat" repeat the last line about a third of the
    time; a duplicated line mid-file is a syntax error that reads as a model
    failure."""
    head = "def f():\n    a = 1\n    b = 2\n"
    tail = "    b = 2\n    return a + b\n"
    joined = _join_continuation(head, tail)
    assert joined.count("b = 2") == 1
    assert joined.endswith("return a + b\n")


def test_a_re_emitted_partial_line_replaces_the_cut_one():
    """Cut MID-LINE, the model re-emits the whole last line. Whole-line
    overlap cannot see that, and the join produced
    `return tuple(    return tuple(...)` — a syntax error blamed on the
    model."""
    head = 'def parse(line):\n    """Split."""\n    return tuple('
    tail = '    return tuple(line.split(","))\n'
    joined = _join_continuation(head, tail)
    assert joined.count("return tuple(") == 1, joined
    assert joined.endswith('    return tuple(line.split(","))\n')


def test_a_re_opened_fence_in_the_continuation_is_stripped():
    """When the continuation re-opens a fence, the joined text has a fence
    mid-file and `_extract` picks the first fence body — the truncated head
    — as the whole file."""
    head = '```python\ndef f():\n    x = ('
    tail = '```python\n1)\n    return x\n```'
    joined = _join_continuation(head, tail)
    assert "(```" not in joined, joined
    assert _extract(joined, "python") == "def f():\n    x = (1)\n    return x"


def test_a_head_that_closed_its_fence_is_reopened_for_the_join():
    """A head cut right after a stray closing fence must not leave the
    continuation outside the fence."""
    head = "```python\ndef f():\n    return [1,\n```"
    tail = "        2]\n```"
    joined = _join_continuation(head, tail)
    assert _extract(joined, "python").strip() == \
        "def f():\n    return [1,\n        2]"


def test_a_continuation_that_restarts_from_the_top_is_taken_whole():
    """Some models ignore "do not start again" and re-emit the file from
    line one. Appending that to the head duplicates every definition;
    the longer complete copy is the file."""
    head = "import os\n\n\ndef f():\n    return os.getcwd("
    tail = "import os\n\n\ndef f():\n    return os.getcwd()\n"
    joined = _join_continuation(head, tail)
    assert joined.count("def f") == 1, joined
    assert joined == tail


# --------------------------------------------------------------------------
# M33 — failed attempts are not accumulated (D11)
# --------------------------------------------------------------------------

def test_the_repair_prompt_carries_diagnostics_and_not_the_broken_code():
    """D11 — attempt 3's prompt containing attempts 1 and 2 is how a model
    pattern-matches its own mistakes and repeats them."""
    text = personas.repair_task(
        "src/x.py", "parse a line",
        "1. src/x.py:4: error: name 'parts' is not defined",
        autofixes=["added the missing trailing newline"])
    assert "not defined" in text
    assert "do not undo" in text.lower()
    # The prompt must not contain a slot for prior attempts at all.
    assert "attempt 1" not in text.lower()
    assert "previous attempt" not in text.lower()


def _journaled_loop(host, **config):
    from cognitive_coder.journal import Journal
    loop = _loop(host, **config)
    loop.journal = Journal(host.fs, "t")
    return loop


def _user_tail(host, call):
    return [m.content for m in host.llm.prompts[call] if m.role == "user"][0]


def test_an_empty_reply_is_journaled_as_an_attempt(tmp_path):
    """§6.9 — every attempt is journaled (C8). An empty reply cost a model
    call and tokens, and `continue`d without a `generate` event, so the
    journal showed one attempt where two were made."""
    host = _host(tmp_path, ["", "```python\nX = 1\n```"])
    loop = _journaled_loop(host, attempts=2)
    loop.run_task(Task(id="t1", path="m.py", purpose="a constant",
                       lang="python"), request="make a constant")
    gens = [r for r in loop.journal.events() if r.get("event") == "generate"]
    assert [g.get("attempt") for g in gens] == [1, 2]
    assert gens[0].get("verify"), "the empty attempt has no verdict"


def test_after_an_empty_reply_the_next_attempt_is_a_first_attempt(tmp_path):
    """Nothing was written and nothing was diagnosed, so there is nothing
    to repair. Attempt 2 was told "Fix the errors reported below" with no
    errors below and no file — by the repairer persona."""
    host = _host(tmp_path, ["", "```python\nX = 1\n```"])
    _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request="make a constant")
    tail = _user_tail(host, 1)
    assert "Write the complete contents of `m.py`" in tail, tail
    assert "Fix the errors" not in tail
    system = "\n".join(m.content for m in host.llm.prompts[1]
                       if m.role == "system")
    assert personas.REPAIRER.self_model not in system


def test_tool_exhaustion_is_named_in_the_attempt_note(tmp_path):
    """A model that spends every tool round browsing returns no file; the
    note should say that, not "the model returned nothing"."""
    from cognitive_coder.loop import MAX_TOOL_ROUNDS
    from cognitive_coder.types import ToolCall
    browse = Completion(
        text="", finish_reason="tool_calls", model="s",
        tool_calls=(ToolCall(id="c1", name="list_symbols",
                             arguments={"path": "m.py"}),))
    host = _host(tmp_path, [browse] * (MAX_TOOL_ROUNDS + 1),
                 supports_tools=True)
    outcome = _journaled_loop(host, attempts=1).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert "tool" in outcome.attempts[0].note, outcome.attempts[0].note


def test_a_repair_prompt_still_carries_the_request(tmp_path):
    """From attempt 2 the repairer saw the file's purpose and the errors,
    and never the request — so "MUST NOT import pygame" was gone exactly
    when the model was rewriting the file."""
    request = "A constant module. It MUST NOT import pygame."
    host = _host(tmp_path, ["```python\nVALUE = undefined_one\n```",
                            "```python\nVALUE = 3\n```"])
    _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request=request)
    tail = _user_tail(host, 1)
    assert "Fix the errors" in tail, "fixture: attempt 2 should be a repair"
    assert "MUST NOT import pygame" in tail, tail


def test_a_long_request_is_capped_in_the_repair_prompt_and_says_so():
    long = "Requirement. " * 400
    text = personas.repair_task("m.py", "x", request=long)
    assert len(text) < len(long)
    assert "Requirement." in text
    assert "…" in text and "2,000" in text


def test_each_attempt_journals_the_hash_of_what_was_sent(tmp_path):
    """`prompt_sha256` hashed the REQUEST, so every attempt of every task
    carried the same value and it could not tell two prompts apart — the
    one job a provenance hash has (C8)."""
    from cognitive_coder.journal import prompt_hash
    host = _host(tmp_path, ["```python\nVALUE = undefined_one\n```",
                            "```python\nVALUE = 3\n```"])
    loop = _journaled_loop(host, attempts=2)
    loop.run_task(Task(id="t1", path="m.py", purpose="a constant",
                       lang="python"), request="make a constant")
    hashes = [r.get("prompt_sha256") for r in loop.journal.events()
              if r.get("event") == "generate"]
    assert len(hashes) == 2 and hashes[0] != hashes[1], hashes
    assert hashes == [prompt_hash(host.llm.prompts[0]),
                      prompt_hash(host.llm.prompts[1])]


def test_the_codemap_sees_a_failed_attempts_file(tmp_path):
    """The codemap was reindexed only when an attempt PASSED, so after a
    failure the query tools contradicted `[THE FILE AS IT STANDS]`: search
    said there was no `parse` anywhere in the project while the file on
    disk — the one quoted in the same prompt — defined it."""
    host = _host(tmp_path, ["```python\ndef parse(line):\n"
                            "    return line.split()\n\n\n"
                            "VALUE = undefined_q\n```"])
    loop = _loop(host, attempts=1)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="x",
                                 lang="python"))
    assert not outcome.ok, "fixture: the attempt must fail"
    assert "parse" in loop.codemap.call_tool("list_symbols",
                                             {"path": "m.py"})
    found = loop.codemap.call_tool("search_codemap", {"name": "parse"})
    assert "m.py" in found, found


def test_only_the_last_attempts_diagnostics_are_fed_back(tmp_path):
    """One prior attempt maximum, and only the diagnostics from it."""
    replies = [
        "```python\nVALUE = undefined_one\n```",
        "```python\nVALUE = undefined_two\n```",
        "```python\nVALUE = 3\n```",
    ]
    host = _host(tmp_path, replies)
    loop = _loop(host, attempts=3)
    loop.run_task(Task(id="t1", path="m.py", purpose="return a number",
                       lang="python"))
    third = host.llm.prompts[-1]
    body = "\n".join(m.content for m in third)
    assert "undefined_one" not in body, (
        "the first failed attempt's code leaked into the third prompt")


# --------------------------------------------------------------------------
# M34 — stagnation and cycles
# --------------------------------------------------------------------------

def test_identical_code_twice_stops_immediately(tmp_path):
    """Hard stagnation: more attempts cannot help.

    The failing code is an undefined NAME rather than an unused import,
    deliberately. `ruff --fix` deletes an unused import — which is F1 working
    exactly as designed — so a test built on one passes or fails depending on
    whether a linter happens to be installed. A test whose result depends on
    the machine is not a test.
    """
    same = "```python\nVALUE = undefined_name_xyz\n```"
    host = _host(tmp_path, [same, same, same, same])
    loop = _loop(host, attempts=4)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="x",
                                 lang="python"))
    assert not outcome.ok
    assert "identical code twice" in outcome.stopped_because
    assert len(outcome.attempts) <= 3, "it kept going after hard stagnation"


def test_a_repeated_signature_is_reported_as_a_cycle(tmp_path):
    """The ping-pong 2-cycle: A→B→A. Every attempt has a DIFFERENT
    diagnostic hash, so a naive detector concludes progress is being made."""
    a = "```python\nVALUE = missing_alpha\n```"
    b = "```python\nVALUE = missing_beta\n```"
    host = _host(tmp_path, [a, b, a, b])
    loop = _loop(host, attempts=4)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="x",
                                 lang="python"))
    assert not outcome.ok
    assert ("circle" in outcome.stopped_because
            or "identical" in outcome.stopped_because), \
        outcome.stopped_because


def test_the_cycle_sentence_names_each_attempt_once(tmp_path):
    """It used to read "attempts 1, 3 and 3" — the current attempt was in
    the list of earlier matches AND appended after it."""
    a = "```python\nVALUE = missing_alpha\n```"
    b = "```python\nVALUE = missing_beta\n```"
    host = _host(tmp_path, [a, b, a, b])
    outcome = _loop(host, attempts=4).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert "attempts 1 and 3 produced" in outcome.stopped_because, \
        outcome.stopped_because
    assert "3 and 3" not in outcome.stopped_because


def test_the_same_error_drifting_down_the_file_is_the_same_error(tmp_path):
    """The model adds a line above the error each time, so the error's LINE
    NUMBER changes every attempt while the error does not. With the line in
    the key, this was caught at attempt 4 by the slow-oscillation check,
    not at attempt 2 by "the same errors twice"."""
    drift = ["```python\n"
             + "\n".join(f"pad{i} = {i}" for i in range(n))
             + "\nVALUE = undefined_q\n```" for n in range(1, 7)]
    host = _host(tmp_path, drift)
    outcome = _loop(host, attempts=6).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python"))
    lines = [a.diagnostics[0].line for a in outcome.attempts]
    assert lines[0] != lines[-1], "the fixture no longer drifts"
    assert len(outcome.attempts) == 2, outcome.stopped_because
    assert "same errors" in outcome.stopped_because


def test_moving_a_return_out_of_a_loop_is_not_identical_code(tmp_path):
    """`textio.canonical` collapses leading whitespace, so in Python a
    return INSIDE a loop and one AFTER it hashed the same, and a real
    change was stopped as "identical code twice" one attempt before the
    fix that would have passed."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "tests" / "test_m.py").write_text(
        "import unittest\nfrom m import first_truthy\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_it(self):\n"
        "        self.assertEqual(first_truthy([0, 2, 3]), 2)\n")
    head = "```python\ndef first_truthy(xs):\n    for x in xs:\n"
    inside = head + ("        if x:\n            pass\n        return x\n"
                     "    return None\n```")
    after = head + ("        if x:\n            pass\n    return x\n"
                    "    return None\n```")
    right = head + ("        if x:\n            return x\n"
                    "    return None\n```")
    host = _host(tmp_path, [inside, after, right])
    outcome = _loop(host, attempts=3).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python",
             test_path="tests/test_m.py"))
    assert outcome.ok, outcome.stopped_because
    assert len(outcome.attempts) == 3


def test_reformatting_python_is_still_identical_code(tmp_path):
    """The AST hash must keep what canonical() was for: a reformat and a
    new comment are not a change."""
    one = "```python\nVALUE = missing_w\n```"
    two = "```python\n\n\nVALUE   =   missing_w   # retry\n\n```"
    host = _host(tmp_path, [one, two, one, two])
    outcome = _loop(host, attempts=4).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert len(outcome.attempts) == 2
    assert "identical code" in outcome.stopped_because


def test_identical_guard_blocked_code_stops_the_loop(tmp_path):
    """A refused attempt `continue`d before the signature was recorded, so
    the same refused file burned every attempt and ended "gave up after 6
    attempts" instead of saying it was the same code each time."""
    blocked = "```python\nimport subprocess\nsubprocess.run(['ls'])\n```"
    host = _host(tmp_path, [blocked] * 6)
    outcome = _loop(host, attempts=6).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert len(outcome.attempts) == 2, outcome.stopped_because
    assert "identical code" in outcome.stopped_because
    assert "refused" in outcome.stopped_because


def test_a_tiny_edit_that_moves_no_error_is_reported_as_churn(tmp_path):
    """COSMETIC_THRESHOLD was declared and never used, and `_cosmetic`
    compared the hashes the identical-code check had just compared, so it
    could never be True. Under 2% of the file changed and the errors did
    not move: that is tinkering, and the sentence should say so."""
    consts = "\n".join(f"LIMIT_{i} = {i * 10}" for i in range(60))
    first = f"```python\n{consts}\nVALUE = undefined_q\n```"
    second = first.replace("LIMIT_7 = 70", "LIMIT_7 = 71")
    host = _host(tmp_path, [first, second, first])
    outcome = _loop(host, attempts=3).run_task(
        Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert len(outcome.attempts) == 2
    assert "under 2%" in outcome.stopped_because, outcome.stopped_because


def test_the_cycle_report_names_what_it_is_alternating_between():
    """The sentence that turns twenty wasted minutes into a two-second fix."""
    diags = (Diagnostic(message="'json' imported but unused", severity="error"),
             Diagnostic(message="name 'json' is not defined",
                        severity="error"))
    text = _describe_cycle(diags)
    assert "alternating between" in text
    assert "unused import" in text
    assert "missing definition" in text


def test_giving_up_reports_what_was_tried_and_the_last_real_error(tmp_path):
    """§6.9 — never "failed after 4 attempts"."""
    host = _host(tmp_path, ["```python\nVALUE = a_xyz\n```",
                            "```python\nVALUE = b_xyz\n```",
                            "```python\nVALUE = c_xyz\n```"])
    loop = _loop(host, attempts=3)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="x",
                                 lang="python"))
    assert not outcome.ok
    summary = outcome.summary()
    assert "attempt 1" in outcome.stopped_because or "circle" in \
        outcome.stopped_because
    assert outcome.stopped_because != f"gave up after {len(outcome.attempts)} attempts"
    assert "m.py" in summary


# --------------------------------------------------------------------------
# M35 — deterministic pre-fixes (F1)
# --------------------------------------------------------------------------

def test_a_missing_trailing_newline_is_fixed_by_rule_not_by_the_model(
        tmp_path):
    host = _host(tmp_path, ["```python\ndef f():\n    return 1```"])
    loop = _loop(host, attempts=1)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="x",
                                 lang="python"))
    assert host.fs.read("m.py").endswith("\n")
    assert any("trailing newline" in f
               for a in outcome.attempts for f in a.autofixes)


def test_autofixes_are_journaled_so_a_recurring_one_is_visible(tmp_path):
    """M35 — if the same fix recurs constantly, the PROMPT needs changing,
    and the log is how anyone finds out."""
    from cognitive_coder.journal import Journal
    host = _host(tmp_path, ["```python\ndef f():\n    return 1```"])
    loop = _loop(host, attempts=1)
    loop.journal = Journal(host.fs, "t")
    loop.run_task(Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert any(r.get("event") == "autofix" for r in loop.journal.events())


# --------------------------------------------------------------------------
# M36 and M37 — the output contract and reasoning tags
# --------------------------------------------------------------------------

def test_commentary_is_detected_only_when_it_is_DECORATED():
    """A detector with false positives is a detector somebody turns off."""
    assert personas.detect_commentary("**Improved version:**\n\ncode here")
    assert personas.detect_commentary("## Rationale\n\nBecause…")
    assert personas.detect_commentary("**Changes made:**\n- a\n- b")
    # A legitimate reply that merely contains the words must survive.
    assert not personas.detect_commentary(
        "def why_this_works():\n    # explanation of the rationale\n"
        "    return 1\n")
    assert not personas.detect_commentary(
        "# This function explains why the changes made here are safe\n")


def test_stripping_commentary_prefers_the_fenced_code():
    text = ("**Improved Reply:**\n\n```python\ndef f():\n    return 1\n```\n\n"
            "**Changes made:** renamed the variable.")
    out = personas.strip_commentary(text, "python")
    assert out.strip() == "def f():\n    return 1"


_BASH_WITH_A_WHY_COMMENT = (
    "#!/usr/bin/env bash\nset -euo pipefail\n"
    "## Why this exists: CI needs a one-shot runner\n"
    "run_all() {\n  pytest -q\n}\nrun_all\n")


def test_a_bash_comment_that_looks_like_a_heading_survives_unchanged():
    """`re.sub` inside the line removed `## Why ` and left the tail as
    CODE: the script gained a command called `this`. A line that begins
    with the language's comment marker is a comment in the file, and a
    comment is harmless where it stands."""
    out = personas.strip_commentary(_BASH_WITH_A_WHY_COMMENT, "bash")
    assert "## Why this exists: CI needs a one-shot runner" in out
    assert "\nthis exists" not in out
    assert out.strip() == _BASH_WITH_A_WHY_COMMENT.strip()
    assert not personas.detect_commentary(_BASH_WITH_A_WHY_COMMENT, "bash")


def test_a_dunder_name_is_not_a_bold_heading():
    """`__` opens markdown bold only when it CLOSES on the same line;
    `__analysis = {}` is an assignment, and stripping its prefix left
    `= {}`."""
    code = "__analysis = {}\n\n\ndef why_(x):\n    return x\n"
    assert not personas.detect_commentary(code, "python")
    assert personas.strip_commentary(code, "python").strip() == code.strip()
    dunder = "__analysis__ = {}\n"
    assert personas.strip_commentary(dunder, "python").strip() == \
        dunder.strip()


def test_a_docstring_is_never_edited():
    """The heading inside a docstring lost `## Notes on`, and the preamble
    rule deleted "The following values were measured:" — a sentence in
    the program, not a sentence about it."""
    code = ('"""Tuning constants.\n\n## Notes on tuning\n'
            'The following values were measured:\n"""\n\nGAIN = 1.5\n')
    out = personas.strip_commentary(code, "python")
    assert "## Notes on tuning" in out
    assert "The following values were measured:" in out
    assert out.strip() == code.strip()


def test_real_commentary_without_a_fence_is_still_removed_whole():
    """The detector's job is unchanged: a decorated heading and a chatty
    preamble go — as WHOLE lines, never leaving a tail behind."""
    reply = ("Sure, here is the file:\n"
             "**Improved version:**\n"
             "def f():\n    return 1\n"
             "**Changes made:** renamed the variable.\n")
    out = personas.strip_commentary(reply, "python")
    assert out.strip() == "def f():\n    return 1", out
    heading = "### Rationale\nx = 1\n"
    assert personas.strip_commentary(heading, "javascript").strip() == \
        "x = 1"


def test_a_preamble_after_the_first_code_line_is_code():
    """Preamble stripping applies BEFORE the code starts, never inside it.
    A heredoc line that reads like a preamble is the program's output."""
    code = ("report() {\n"
            "  cat <<EOF\n"
            "Here is the nightly report:\n"
            "EOF\n"
            "}\n")
    out = personas.strip_commentary("**Rationale**\n" + code, "bash")
    assert "Here is the nightly report:" in out, out
    assert out.strip() == code.strip()


def test_the_loop_leaves_a_legitimate_bash_file_alone(tmp_path):
    """End to end: the file written is the file the model returned."""
    host = _host(tmp_path, [_BASH_WITH_A_WHY_COMMENT])
    loop = _loop(host, attempts=1)
    loop.run_task(Task(id="t1", path="run.sh", purpose="ci runner",
                       lang="bash"))
    written = host.fs.read("run.sh")
    assert "## Why this exists" in written, written
    assert not any("commentary" in m for _k, m, _d in host.events.events)


def test_stripping_never_returns_nothing():
    """Handing back an empty string because a heuristic was keen is worse
    than handing back a reply with a heading in it."""
    assert personas.strip_commentary("**Rationale**").strip()


def test_think_blocks_are_stripped_before_any_use():
    """M37, D13 — reasoning in a source file is a broken file."""
    text = "<think>Let me consider…</think>\ndef f():\n    return 1\n"
    assert personas.strip_think(text) == "def f():\n    return 1"
    reasoning, answer = personas.split_think(text)
    assert "consider" in reasoning
    assert "<think>" not in answer


def test_an_unclosed_think_block_is_treated_as_reasoning():
    """The model was truncated mid-thought; everything after the tag is
    reasoning, not answer. Treating it as answer writes CoT into a file."""
    text = "def f():\n    return 1\n<think>Now let me reconsider the"
    answer = personas.strip_think(text)
    assert "reconsider" not in answer
    assert "def f()" in answer


def test_the_output_contract_names_what_to_produce():
    """§4.4 — and it does not rehearse the forbidden phrases, because models
    repeat prompt vocabulary."""
    assert "complete contents of the file" in personas.CONTRACT_FILE
    assert "Changes made" not in personas.CONTRACT_FILE
    assert "Improved Reply" not in personas.CONTRACT_FILE


def test_a_same_model_review_is_labelled_non_independent():
    """M41 — never present self-review as independent scrutiny."""
    caveat = personas.independence_caveat(True)
    assert "not independent scrutiny" in caveat
    assert personas.independence_caveat(False) == ""


# --------------------------------------------------------------------------
# D5 — fence extraction
# --------------------------------------------------------------------------

def test_code_extraction_survives_fence_confusion():
    assert "def f" in _extract("```python\ndef f():\n    return 1\n```",
                               "python")
    # No fence at all.
    assert "def g" in _extract("def g():\n    return 2\n", "python")
    # A fence tagged with something that is not a language.
    assert "def h" in _extract("```\ndef h():\n    return 3\n```", "python")
    # Two fences: the one that PARSES wins, not the first.
    out = _extract("```python\ndef bad(:\n```\n```python\ndef ok():\n"
                   "    return 1\n```", "python")
    assert "def ok" in out and "bad" not in out


# --------------------------------------------------------------------------
# cancellation (§5.2, M21)
# --------------------------------------------------------------------------

def test_cancelling_rolls_back_an_open_transaction(tmp_path):
    from cognitive_coder.errors import Cancelled
    from cognitive_coder.ports import Cancel

    host = _host(tmp_path, ["```python\nx = 1\n```"])
    host.fs.write("m.py", "original = True\n")
    token = Cancel()
    token.set()
    loop = _loop(host, attempts=1)
    loop.cancel = token
    with pytest.raises(Cancelled):
        loop.run_task(Task(id="t1", path="m.py", purpose="x", lang="python"))
    assert host.fs.read("m.py") == "original = True\n"


def test_a_cancelled_message_is_a_sentence_not_a_traceback():
    from cognitive_coder.errors import Cancelled
    text = str(Cancelled("generating src/x.py"))
    assert "Stopped at your request" in text
    assert "rolled back" in text
    assert "Traceback" not in text


def test_a_provider_error_reaches_the_operator_in_its_own_words(tmp_path):
    """Providers now fill `Completion.error` with the HTTP status and the
    server's message. The loop used to replace it with "could not be
    reached, or it returned an error", so a server refusing `tools` read
    exactly like a server that was down."""
    said = ("the endpoint answered HTTP 400: tools are not supported "
            "unless llama-server is started with --jinja")
    host = _host(tmp_path, [Completion(text="", finish_reason="error",
                                       error=said)])
    outcome = _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request="make a constant")
    assert not outcome.ok
    assert "HTTP 400" in outcome.stopped_because
    assert "--jinja" in outcome.stopped_because


def test_a_provider_error_with_no_detail_keeps_the_general_sentence(
        tmp_path):
    host = _host(tmp_path, [Completion(text="", finish_reason="error")])
    outcome = _loop(host, attempts=2).run_task(
        Task(id="t1", path="m.py", purpose="a constant", lang="python"),
        request="make a constant")
    assert "could not be reached" in outcome.stopped_because


def _big_repair(tmp_path, *, lines, context):
    big = "".join(f"value_{i} = {i}\n" for i in range(lines))
    first = f"```python\n{big}print(undefined_name)\n```"
    llm = ScriptedLLM([first, "```python\nX = 1\n```"],
                      supports_tools=False, context_tokens=context)
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=2, max_tokens=1024)
    outcome = loop.run_task(Task(id="t1", path="m.py", purpose="constants",
                                 lang="python"),
                            request="make some constants")
    return llm, host, outcome


def test_a_repair_prompt_that_fits_is_left_alone(tmp_path):
    llm, host, _outcome = _big_repair(tmp_path, lines=40, context=16384)
    text = "\n".join(m.content for m in llm.prompts[1])
    assert "value_39 = 39" in text and "cut to fit" not in text


def test_a_repair_that_cannot_show_the_whole_file_is_refused(tmp_path):
    """Nothing measured the assembled prompt, so a repair of a large file
    sent `[THE FILE AS IT STANDS]` whole; llama.cpp then shifts the context
    and the SYSTEM PROMPT is what falls off the front. Cutting the file
    instead is worse: the contract asks for the complete file, and a model
    shown half returns half. So the attempt stops, before a model call,
    with a sentence that says what would make it possible."""
    llm, host, outcome = _big_repair(tmp_path, lines=3000, context=4096)
    assert len(llm.prompts) == 1, "a model call was spent on the refusal"
    assert not outcome.ok
    assert "larger context" in outcome.stopped_because
    assert "m.py is about" in outcome.stopped_because


def test_reference_material_is_cut_to_fit_but_the_file_never_is(tmp_path):
    """Interfaces and examples may be cut; the file being repaired may not.
    With a file that fits and references that do not, the prompt is made
    to fit and the cut is named."""
    from cognitive_coder.loop import Loop
    llm = ScriptedLLM([], supports_tools=False, context_tokens=4096)
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    loop = _loop(host, attempts=1, max_tokens=1024)
    small_file = "X = 1\n"
    huge_reference = "def helper():\n    pass\n" * 2000
    sent = []

    def rebuild(extra):
        sent.append(extra)
        return [Message(role="system", content="rules"),
                Message(role="user", content="\n".join(extra))]

    fitted = Loop._fit_to_context(
        loop, [Message(role="system", content="rules"),
               Message(role="user", content=small_file + huge_reference)],
        llm.capabilities(), Task(id="t", path="m.py", purpose="p"), 2,
        rebuild=rebuild,
        pieces=[("THE FILE AS IT STANDS", small_file),
                ("INTERFACES OF WHAT THIS FILE USES", huge_reference)])
    text = "\n".join(m.content for m in fitted)
    assert "X = 1" in text                     # the file, whole
    # The reference that did not fit is NAMED as left out (M28), so the
    # model knows to look it up rather than guess.
    assert "NOT INCLUDED" in text and "INTERFACES OF WHAT" in text
    assert "def helper" not in text
    assert sum(llm.count_tokens(m.content) for m in fitted) <= 4096 - 1024


def test_an_existing_file_is_shown_to_the_model_and_changed(tmp_path):
    """The first attempt said "Write the complete contents of `x`" and
    never showed `x`, so asking to improve or extend a working file
    regenerated it blind — with auto-apply on, replacing it."""
    (tmp_path / "game.py").write_text(
        "SPEED = 3\n\n\ndef tick(x):\n    return x + SPEED\n")
    host = _host(tmp_path, ["```python\nSPEED = 5\n\n\ndef tick(x):\n"
                            "    return x + SPEED\n```"])
    _loop(host, attempts=1).run_task(
        Task(id="t1", path="game.py", purpose="the game loop",
             lang="python"), request="make the game faster")
    first = "\n".join(m.content for m in host.llm.prompts[0])
    assert "already exists" in first
    assert "def tick(x):" in first and "SPEED = 3" in first
    assert "Write the complete contents" not in first


def test_a_skeleton_stub_is_still_written_from_scratch(tmp_path):
    from cognitive_coder.planner import STUB_SENTINEL
    (tmp_path / "m.py").write_text(
        f"# {STUB_SENTINEL} written by the skeleton\n"
        "def f():\n    raise NotImplementedError\n")
    host = _host(tmp_path, ["```python\ndef f():\n    return 1\n```"])
    _loop(host, attempts=1).run_task(
        Task(id="t1", path="m.py", purpose="f", lang="python"),
        request="write f")
    first = "\n".join(m.content for m in host.llm.prompts[0])
    assert "Write the complete contents" in first
    assert "already exists" not in first


def test_an_existing_file_too_big_to_show_is_refused_not_rewritten(
        tmp_path):
    big = "".join(f"value_{i} = {i}\n" for i in range(3000))
    (tmp_path / "m.py").write_text(big)
    llm = ScriptedLLM(["```python\nX = 1\n```"], supports_tools=False,
                      context_tokens=4096)
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())
    outcome = _loop(host, attempts=1, max_tokens=1024).run_task(
        Task(id="t1", path="m.py", purpose="constants", lang="python"),
        request="tidy it")
    assert not llm.prompts, "a model call was spent"
    assert not outcome.ok and "larger context" in outcome.stopped_because
    assert (tmp_path / "m.py").read_text() == big
