# SPDX-License-Identifier: Apache-2.0
"""The worked example in each output contract is itself well-formed.

2026-10-27: `CONTRACT_SYMBOLS` and `CONTRACT_EDIT` each end with one
example of the shape they ask for. An example the engine's own parser
could not place would teach the model a shape the engine refuses, so each
is run through that parser here, exactly as a reply would be.
"""

from __future__ import annotations

from cognitive_coder import patcher, personas, symedit

LEDGER = '''\
import math


class Ledger:
    def __init__(self, entries):
        self.entries = list(entries)

    def rounded_total(self, entries):
        return 0.0

    def count(self):
        return len(self.entries)
'''


def _example_after(contract: str, marker: str) -> str:
    """The example text: everything after the sentence that introduces it."""
    assert marker in contract
    return contract.split(marker, 1)[1].strip()


def test_symbols_example_is_a_placeable_narrowed_reply():
    """The example's SHAPE places. (The example itself, verbatim, is
    deliberately dropped by `symedit.reply_code` — a model that echoes it
    must not have it placed — so the shape is tested under a real name.)"""
    example = _example_after(personas.CONTRACT_SYMBOLS, "reads, in full:")
    assert "ExampleLedger" in example
    shaped = example.replace("ExampleLedger", "Ledger")
    applied = symedit.apply(LEDGER, shaped, targets=["Ledger.rounded_total"])
    assert applied.ok, applied.problem
    assert "math.floor" in applied.code
    # the untouched method is still there, untouched
    assert "def count" in applied.code
    assert "len(self.entries)" in applied.code


def test_the_symbols_example_echoed_verbatim_is_not_placed():
    example = _example_after(personas.CONTRACT_SYMBOLS, "reads, in full:")
    assert symedit.reply_code(example).strip() == ""
    assert symedit.reply_code("```python\nx = 1\n```").strip() == "x = 1"


def test_the_edit_example_echoed_verbatim_is_not_an_edit():
    example = _example_after(personas.CONTRACT_EDIT, "reads, in full:")
    real = example.replace("src/tally.py", "src/mine.py")
    assert patcher.parse_edits(example + "\n" + real) == \
        patcher.parse_edits(real)


def test_edit_example_parses_as_one_cc_edit_block():
    """The shape parses (under a real path — the verbatim example is
    dropped by `parse_edits` on purpose)."""
    example = _example_after(personas.CONTRACT_EDIT, "reads, in full:")
    edits = patcher.parse_edits(example.replace("src/tally.py",
                                                "src/mine.py"))
    assert len(edits) == 1
    assert edits[0].path == "src/mine.py"
    assert "count > limit" in edits[0].old
    assert "count >= limit" in edits[0].new


def test_examples_are_labelled_as_the_shape_not_the_file():
    for contract in (personas.CONTRACT_SYMBOLS, personas.CONTRACT_EDIT):
        assert "not of this file" in contract
