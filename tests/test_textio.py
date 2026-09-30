# SPDX-License-Identifier: Apache-2.0
"""Encodings, BOMs and line endings (§6.5a) — the cases that went wrong.

`test_patcher.py` holds the round-trip property over a corpus. These are the
specific decodes that were WRONG while still round-tripping, which is why
the property test alone never caught them: a file decoded as the wrong
encoding re-encodes to the same bytes, and the damage is in what the model
was shown and what an edit then writes.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import textio  # noqa: E402


def test_ascii_content_utf16le_without_a_bom_is_read_as_utf16():
    """ASCII in UTF-16-LE is also VALID UTF-8 — every other byte a NUL —
    so trying strict UTF-8 first decoded it as `x\\x00 \\x00=…`, with no
    assumption stated, and the model was shown NULs."""
    raw = "x = 1\r\ny = 2\r\n".encode("utf-16-le")
    tf = textio.decode(raw)
    assert tf.encoding == "utf-16-le"
    assert tf.text == "x = 1\ny = 2\n"
    assert tf.eol == textio.CRLF
    assert "\x00" not in tf.text
    assert tf.encode() == raw


def test_ascii_content_utf16be_without_a_bom_is_read_as_utf16():
    raw = "x = 1\n".encode("utf-16-be")
    tf = textio.decode(raw)
    assert tf.encoding == "utf-16-be" and tf.text == "x = 1\n"


def test_a_utf8_file_is_not_mistaken_for_utf16():
    raw = "x = 'héllo'\n".encode()
    tf = textio.decode(raw)
    assert tf.encoding == "utf-8" and not tf.assumption


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be", "utf-32-le",
                                   "utf-8"])
def test_the_bom_is_stripped_on_decode_and_restored_on_encode(codec):
    """U+FEFF is a BOM for every Unicode encoding, not only UTF-8. It used
    to stay in the TEXT for UTF-16/32, so a whole-file edit — new text,
    no U+FEFF — silently dropped the BOM."""
    raw = ("﻿" + "def a():\r\n    return 1\r\n").encode(codec)
    tf = textio.decode(raw)
    assert tf.bom, codec
    assert not tf.text.startswith("﻿"), "the BOM leaked into the text"
    assert tf.encode() == raw, "round trip"
    rewritten = tf.encode("def a():\n    return 2\n")
    assert textio.decode(rewritten).bom, "a whole-file write dropped the BOM"
    assert rewritten == ("﻿" + "def a():\r\n    return 2\r\n"
                         ).encode(codec)


def test_widening_to_utf8_is_stated_not_silent():
    """An edit that adds a character the file's encoding cannot hold is
    written as UTF-8. That changes the file's encoding, so it is said."""
    tf = textio.decode(b"# caf\xe9\nx = 1\n")        # cp1252
    note = tf.encoding_note("# caf\xe9\nx = '→'\n")
    assert "UTF-8" in note and tf.encoding in note
    assert tf.encoding_note("# caf\xe9\nx = 2\n") == ""
