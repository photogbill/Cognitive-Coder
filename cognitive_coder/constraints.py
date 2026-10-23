# SPDX-License-Identifier: Apache-2.0
"""The request's hard requirements, pinned into every prompt.

A specification says "the game MUST run offline", "never use global
variables", "do not add dependencies" — once, near the top. By the sixth
file and the third repair, that sentence is thousands of tokens behind the
model, or was never in the prompt at all: the repair prompt carries the
file, the error and the interfaces, not the spec. A small model then adds
`import requests` to fix a bug, and nothing in the prompt said not to.

So the sentences that state a hard requirement are taken out of the request
ONCE, by rule, word for word — never paraphrased (a paraphrase is the
model's reading of the requirement, and the point is the operator's words)
— and pinned in the cached prefix of every prompt for the session, beside
the conventions. Fixed for the session, so they cost one prefill and do not
break the prompt cache (M52).

What counts, by rule (deterministic first, C5): a sentence with MUST, MUST
NOT, SHALL, NEVER, ALWAYS, DO NOT or REQUIRED in capitals, or with "must",
"must not", "never", "do not" / "don't", "may not", "is not allowed" /
"are not allowed" in any case. Code blocks are skipped. At most
`MAX_CONSTRAINTS` sentences, each cut at `MAX_CHARS` and said to be cut;
the rest are counted, not dropped silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

MAX_CONSTRAINTS = 15
MAX_CHARS = 240
TITLE = ("[HARD REQUIREMENTS — from the request, word for word. Every file "
         "and every repair must keep them]")

_STRONG = re.compile(r"\b(MUST(\s+NOT)?|SHALL(\s+NOT)?|NEVER|ALWAYS|"
                     r"DO\s+NOT|DON'T|REQUIRED)\b")
_WEAK = re.compile(r"\b(must(\s+not)?|never|do\s+not|don['’]t|may\s+not|"
                   r"(is|are)\s+not\s+allowed|mustn['’]t)\b", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)]|[a-z][.)])\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
#: Markdown emphasis — `*word*`, `**bold**`, `_word_`, `` `code` `` — whose
#: delimiters sit at WORD BOUNDARIES. The first version matched any run of
#: `*`, `_` or backtick, so an identifier's underscores were "emphasis":
#: `tests/test_physics.py, tests/test_parser.py` was pinned in every prompt
#: as `tests/testphysics.py, tests/testparser.py`, and `SUBTICKS_PER_TURN`
#: as `SUBTICKSPERTURN` (2026-10-16, the Star Trek build). An underscore
#: inside a word is part of the word; emphasis opens after a non-word
#: character and closes before one.
_EMPHASIS = re.compile(r"(?<![\w\\])([*_]{1,3})(?!\s)(.+?)(?<!\s)\1(?![\w])")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")


def _unemphasise(line: str) -> str:
    """Markdown emphasis removed, identifiers left exactly as written.

    Code spans first, and their contents are kept verbatim: `` `__init__` ``
    is the name `__init__`, not bold "init". Emphasis is then removed from
    the prose between them, where `_word_` can only be emphasis.
    """
    out: list[str] = []
    last = 0
    for m in _CODE_SPAN.finditer(line):
        out.append(_strip_emphasis(line[last:m.start()]))
        out.append(m.group(1))
        last = m.end()
    out.append(_strip_emphasis(line[last:]))
    return "".join(out)


def _strip_emphasis(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _EMPHASIS.sub(r"\2", text)
    return text


@dataclass
class Constraints:
    items: list = field(default_factory=list)
    more: int = 0          # found past MAX_CONSTRAINTS

    def block(self) -> str:
        """What goes into the prompt prefix; '' when there is nothing."""
        if not self.items:
            return ""
        lines = [TITLE, *(f"- {c}" for c in self.items)]
        if self.more:
            lines.append(f"- … and {self.more} more in the request itself.")
        return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    fenced = False
    for raw in (text or "").splitlines():
        if raw.strip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced or not raw.strip():
            continue
        line = _BULLET.sub("", raw).strip().lstrip("#> ").strip()
        line = _unemphasise(line)
        out.extend(s.strip() for s in _SENTENCE.split(line) if s.strip())
    return out


def extract(text: str) -> Constraints:
    """The requirement sentences of a request or specification."""
    found = Constraints()
    seen: set[str] = set()
    for sentence in _sentences(text):
        if not (_STRONG.search(sentence) or _WEAK.search(sentence)):
            continue
        key = re.sub(r"\W+", " ", sentence.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        if len(found.items) >= MAX_CONSTRAINTS:
            found.more += 1
            continue
        if len(sentence) > MAX_CHARS:
            sentence = sentence[:MAX_CHARS - 1].rstrip() + "… (cut here)"
        found.items.append(sentence)
    return found
