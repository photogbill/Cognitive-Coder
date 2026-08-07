# SPDX-License-Identifier: Apache-2.0
"""Encoding and line endings — the policy, stated once, in one place.

This is Windows-first software and the model emits `\\n`. Without an explicit
policy, two things happen, and the second is worse than the first:

  1. Anchored edits fail mysteriously on CRLF files. The model's anchor has
     `\\n`, the file has `\\r\\n`, the text is "not found", and the operator
     watches a perfectly good edit be refused for no visible reason.
  2. A whole-file write silently converts EVERY line ending in a file the
     task touched one function of. The diff is the whole file. Code review
     becomes impossible, and the blame for a hundred unrelated lines lands on
     whoever ran the tool.

THE POLICY:

  * **On read** — detect the encoding (UTF-8 with or without BOM, UTF-16, and
    a stated fallback) and the file's DOMINANT line-ending style.
  * **Normalise to `\\n` internally.** All anchor matching, diffing, hashing
    and stagnation-detection happen on normalised text. Every component above
    this one may assume `\\n`, always.
  * **On write, restore** the file's original encoding, BOM and EOL style. New
    files take the project's dominant style, else the platform default.
  * **The snapshot stores the original BYTES**, so undo is byte-identical by
    construction rather than by careful reconstruction (M26). That distinction
    is the whole reason this module exists as bytes-in/bytes-out rather than
    as a str convenience layer.

Where a fallback is used, the assumption is *stated* — it goes in the journal
and, where it matters, into the result the operator reads (C7). A file decoded
with `errors="replace"` has had characters changed; that is a fact the person
whose file it is should be told.
"""

from __future__ import annotations

from dataclasses import dataclass

# Byte-order marks, longest first — UTF-32's BOM starts with UTF-16's, so a
# shorter-first check misidentifies it.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xff\xfe", "utf-16-le"),
)

CRLF = "\r\n"
LF = "\n"
CR = "\r"


@dataclass(frozen=True)
class TextFile:
    """A decoded file, plus everything needed to write it back unchanged.

    ``text`` is always `\\n`-normalised. ``encode()`` is the inverse of the
    decode that produced it: same encoding, same BOM, same EOL style. Round
    tripping an unmodified TextFile MUST reproduce the original bytes exactly,
    and `tests/test_textio.py` asserts precisely that over a corpus.
    """
    text: str                    # ALWAYS \n-normalised
    encoding: str = "utf-8"
    bom: bool = False
    eol: str = LF
    assumption: str = ""         # "" when the decode was unambiguous

    @property
    def approximate(self) -> bool:
        """True when characters may have been replaced during decoding."""
        return bool(self.assumption)

    def encode(self, text: str | None = None) -> bytes:
        """Back to bytes, in the file's ORIGINAL shape.

        Pass ``text`` to write different content into the same shape — which
        is what every edit does, and why the shape is carried separately from
        the content.
        """
        body = self.text if text is None else text
        if self.eol != LF:
            body = body.replace(LF, self.eol)
        enc = self.encoding
        if self.bom and enc == "utf-8":
            enc = "utf-8-sig"
        try:
            return body.encode(enc)
        except (UnicodeEncodeError, LookupError):
            # The edit introduced a character the original encoding cannot
            # hold — e.g. an em-dash into a cp1252 file. Widening to UTF-8 is
            # the only non-lossy option, and it is a change worth admitting
            # to rather than performing silently; the caller reports
            # `assumption` when it is set.
            return body.encode("utf-8")

    def with_text(self, text: str) -> TextFile:
        return TextFile(text=text, encoding=self.encoding, bom=self.bom,
                        eol=self.eol, assumption=self.assumption)


def detect_eol(text: str) -> str:
    """The file's DOMINANT line ending. Mixed files pick the majority.

    Majority rather than first-seen: a mostly-CRLF file with three stray LF
    lines is a CRLF file, and treating it as LF would rewrite every other
    line on the next whole-file write.
    """
    crlf = text.count(CRLF)
    lf = text.count(LF) - crlf
    cr = text.count(CR) - crlf
    if crlf >= lf and crlf >= cr and crlf:
        return CRLF
    if cr > lf and cr:
        return CR
    return LF


def is_mixed_eol(text: str) -> bool:
    """Does this file use more than one line ending?

    Worth its own function because the answer changes what can be promised.
    A single-style file round-trips byte-for-byte through decode/encode; a
    MIXED file cannot, because "restore the file's line-ending style" has no
    single answer for it. The minority endings are normalised to the
    dominant one on write, and that is a real change to lines the task never
    touched — so it is declared rather than performed quietly (C7).

    Undo is unaffected: the snapshot stores the original BYTES, so a rollback
    is byte-identical even here (M26). The limitation is confined to writing
    NEW content into a mixed file.
    """
    crlf = text.count(CRLF)
    lf = text.count(LF) - crlf
    cr = text.count(CR) - crlf
    return sum(1 for n in (crlf, lf, cr) if n) > 1


def normalise(text: str) -> str:
    """Every line ending becomes `\\n`. CRLF first, or CR-handling doubles."""
    return text.replace(CRLF, LF).replace(CR, LF)


def _mixed_note(text: str) -> str:
    """The declaration a mixed-line-ending file earns (C7)."""
    if not is_mixed_eol(text):
        return ""
    return ("this file mixes line-ending styles; writing to it will "
            "normalise them all to its dominant style, which changes lines "
            "the task did not touch. Undo still restores the original bytes "
            "exactly.")


def decode(raw: bytes) -> TextFile:
    """Bytes → a TextFile that knows how to become those bytes again.

    Order matters: BOM, then strict UTF-8, then UTF-16 heuristics, then a
    stated fallback. Strict UTF-8 before anything lossy is what makes the
    common case exact.
    """
    if not raw:
        return TextFile(text="", encoding="utf-8", bom=False, eol=LF)

    for bom, enc in _BOMS:
        if raw.startswith(bom):
            try:
                text = raw.decode(enc)
                base = "utf-8" if enc == "utf-8-sig" else enc
                return TextFile(text=normalise(text), encoding=base, bom=True,
                                eol=detect_eol(text),
                                assumption=_mixed_note(text))
            except UnicodeDecodeError:
                break

    try:
        text = raw.decode("utf-8")
        return TextFile(text=normalise(text), encoding="utf-8", bom=False,
                        eol=detect_eol(text), assumption=_mixed_note(text))
    except UnicodeDecodeError:
        pass

    # A high proportion of NUL bytes in a text file means UTF-16 without a
    # BOM, which is common in Windows tooling output.
    if raw.count(b"\x00") > len(raw) // 4:
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
                return TextFile(text=normalise(text), encoding=enc, bom=False,
                                eol=detect_eol(text),
                                assumption=f"decoded as {enc} (no BOM); "
                                           f"if the file is something else, "
                                           f"say so and it will be reread")
            except UnicodeDecodeError:
                continue

    # cp1252 accepts almost any byte, so it is the honest last stop before
    # replacement — and either way the assumption is STATED (C7).
    for enc in ("cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            return TextFile(
                text=normalise(text), encoding=enc, bom=False,
                eol=detect_eol(text),
                assumption=f"not valid UTF-8; decoded as {enc}. Characters "
                           f"outside that set may be wrong.")
        except UnicodeDecodeError:
            continue

    text = raw.decode("utf-8", errors="replace")
    return TextFile(
        text=normalise(text), encoding="utf-8", bom=False,
        eol=detect_eol(text),
        assumption="this file is not valid text in any encoding tried; "
                   "undecodable bytes were replaced, so an edit would "
                   "corrupt them. It is safer to edit this one by hand.")


def read(fs, path: str) -> TextFile:
    """Read through a FileSystemPort and decode (C2)."""
    return decode(fs.read_bytes(path))


def write(fs, path: str, tf: TextFile, text: str | None = None) -> None:
    """Write back in the file's original shape."""
    fs.write_bytes(path, tf.encode(text))


def project_eol(fs, sample: int = 20) -> str:
    """The project's dominant EOL, for files that don't exist yet.

    A new file should look like its neighbours. Sampling twenty is plenty —
    this is a tiebreak, not a census, and reading a whole tree to answer it
    would cost more than the question is worth.
    """
    counts = {CRLF: 0, LF: 0}
    seen = 0
    try:
        paths = fs.list("*")
    except Exception:                                    # noqa: BLE001
        paths = []
    for path in paths:
        if seen >= sample:
            break
        try:
            raw = fs.read_bytes(path)
        except Exception:                                # noqa: BLE001
            continue
        if b"\x00" in raw[:1024]:                        # binary; skip
            continue
        seen += 1
        counts[CRLF if b"\r\n" in raw else LF] += 1
    if not seen:
        import os
        return CRLF if os.name == "nt" else LF
    return CRLF if counts[CRLF] > counts[LF] else LF


def new_file(fs, text: str) -> TextFile:
    """A TextFile for a path that does not exist yet."""
    return TextFile(text=normalise(text), encoding="utf-8", bom=False,
                    eol=project_eol(fs))


# ---------------------------------------------------------------------------
# normalisation for hashing (§6.9, M34)
# ---------------------------------------------------------------------------

def canonical(text: str, comment: str = "#") -> str:
    """Text reduced to what a change actually MEANS, for the stagnation hash.

    Normalise before hashing — line endings, comments, and whitespace — or a
    reformat registers as a change and defeats the whole cycle detector. A
    model that reindents its output and changes nothing else has made no
    progress, and the detector must agree.

    Deliberately crude: it strips line comments and collapses whitespace. It
    does not parse, because it has to work on a file that does not parse —
    which is precisely when stagnation happens.
    """
    out = []
    for line in normalise(text or "").split(LF):
        if comment and comment != "//":
            head = line.split(comment, 1)[0]
        else:
            head = line.split("//", 1)[0]
        head = " ".join(head.split())
        if head:
            out.append(head)
    return LF.join(out)
