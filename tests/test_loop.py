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
from cognitive_coder.types import Completion, Diagnostic, Task  # noqa: E402


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
