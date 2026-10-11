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

import re
from dataclasses import dataclass, field

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
        line = re.sub(r"[*_`]{1,3}([^*_`]+)[*_`]{1,3}", r"\1", line)
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
