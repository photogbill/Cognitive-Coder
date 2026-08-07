# SPDX-License-Identifier: Apache-2.0
"""Prompts, Theory of Mind, and the output contract that stops commentary.

THE HARD-WON LESSON THIS MODULE EXISTS FOR (§4.4):

A model asked to "review X and return it improved" will, if it is small, write
a *review* — headed `**Improved Reply:**`, followed by `**Changes made:**` —
and the caller will ship the commentary as if it were the product. This was
observed in this project's own Athena panel. It is not fixed by asking more
politely.

So two things, and **both** are required:

  1. **Every prompt whose output is machine-consumed ends with an OUTPUT
     CONTRACT** naming exactly what to emit and forbidding the headings and
     preambles by name (M36).
  2. **Every call site that consumes model output runs the detector.** Never
     trust the prompt alone. `detect_commentary` and `strip_commentary` live
     here and are used at every consuming call site — that is the second half
     of M36, and the half that actually saves you.

The detector requires **decoration** (`**Why?**`, `## Rationale`) rather than
bare words, so a legitimate reply that happens to contain "why" is not
mangled. A detector with false positives gets switched off, and then you have
no detector.

LAYERING, NOT REPLACING (§6.11). Devstral ships a system prompt it was tuned
with; replacing it wholesale discards that tuning. So:

    1. the model's own shipped system prompt      (its trained behaviour)
    2. + project conventions and style            (F3)
    3. + [EXECUTION CONSTRAINTS] block            (codemap, §6.7)
    4. + the output contract                      (§4.4 — still required)

PROMPT ORDER IS A PERFORMANCE CONTRACT, NOT A STYLE CHOICE (G.7.1). The
stable-first / volatile-last layout is what lets llama.cpp reuse the KV cache
of the prefix: 3 seconds instead of minutes, on every call. `build_prompt`
returns the two halves separately so the caller can assert they are
byte-identical between calls (M52), which is the only thing standing between
a working cache design and a silently 20×-slower one six months from now.

**Never put a timestamp, a session id, or a randomised preamble in the
prefix.** One varying token at position 40 discards 30k tokens of cached work.

Two more rules that look like details and are not:
  * **Prompts must not name the internal machinery they don't want echoed.**
    Models repeat prompt vocabulary. A prompt that says "do not emit a
    Changes Made section" teaches the phrase "Changes Made".
  * **`<think>` blocks are stripped before any use** (D13, M37). Reasoning
    output in a source file is not a style problem, it is a broken file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re

from .types import Message

# --------------------------------------------------------------------------
# reasoning-tag stripping (D13, M37)
# --------------------------------------------------------------------------

_THINK = re.compile(
    r"<(think|thinking|reasoning|thought)>.*?</\1>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(think|thinking|reasoning|thought)>.*", re.S | re.I)


def split_think(text: str) -> tuple[str, str]:
    """(reasoning, answer). The reference implementation is ATK's.

    An UNCLOSED `<think>` is the interesting case: the model was truncated
    mid-thought, and everything after the tag is reasoning, not answer.
    Treating it as answer writes chain-of-thought into a source file.

    **Text with no reasoning tags comes back byte-for-byte.** That is not
    fastidiousness: a truncated generation ends mid-token — `    parts = ` —
    and stripping the trailing space makes the continuation join as
    `parts =line.split(...)`, a syntax error the model then gets blamed for
    (D1, M32). Only text this function actually changed gets tidied.
    """
    raw = text or ""
    if "<think" not in raw.lower() and "<reasoning" not in raw.lower() \
            and "<thought" not in raw.lower():
        return "", raw
    reasoning = "\n".join(m.group(0) for m in _THINK.finditer(raw))
    answer = _THINK.sub("", raw)
    open_tag = _THINK_OPEN.search(answer)
    if open_tag:
        reasoning += "\n" + open_tag.group(0)
        answer = _THINK_OPEN.sub("", answer)
    return reasoning.strip(), answer.strip()


def strip_think(text: str) -> str:
    """Just the answer. Called before model output is used for ANYTHING."""
    return split_think(text)[1]


# --------------------------------------------------------------------------
# commentary detection (§4.4, D2, M36)
# --------------------------------------------------------------------------

# DECORATED headings only. `**Changes made:**` and `## Rationale` are
# commentary; the word "why" in a sentence is not. Requiring the decoration
# is what keeps the false-positive rate at zero, and a detector with false
# positives is a detector somebody switches off.
_COMMENTARY = re.compile(
    # The decoration must be UNAMBIGUOUS. `**Bold:**` and `## Heading` are
    # prose furniture; a single `#` is a Python, shell, Ruby or GDScript
    # comment, and `# explanation of the rationale` is a perfectly good
    # comment in code we asked for. Requiring two hashes removes that entire
    # collision — nobody writes `## rationale` as a code comment — and a
    # detector with false positives is a detector somebody switches off,
    # which leaves you with no detector at all.
    r"^[ \t]*(?:\*\*|__|#{2,4}[ \t]+|\d+\.[ \t]*\*\*)[ \t]*"
    r"(improved\s+(?:reply|version|code|answer)|changes?\s+made|"
    r"what\s+(?:i\s+)?changed|rationale|explanation|reasoning|why[\s?]|"
    r"summary\s+of\s+changes|notes?\s+on|here'?s?\s+(?:the|your)|"
    r"key\s+(?:changes|improvements)|analysis)\b",
    re.I | re.M)

_PREAMBLE = re.compile(
    r"^\s*(?:sure|certainly|of\s+course|here'?s|here\s+is|i'?ll|i\s+will|"
    r"let\s+me|below\s+is|the\s+following)\b[^\n]{0,120}[:.]\s*$",
    re.I | re.M)


def detect_commentary(text: str) -> bool:
    """Did the model write ABOUT the answer instead of writing the answer?"""
    body = strip_think(text or "")
    return bool(_COMMENTARY.search(body))


def strip_commentary(text: str, lang_id: str = "") -> str:
    """The product, with the commentary removed — or the original if unsure.

    Conservative on purpose: if stripping would leave nothing, the original
    is returned. Handing back an empty string because a heuristic was too
    keen is a worse failure than handing back a reply with a heading in it,
    since the caller can still extract code from the latter.
    """
    body = strip_think(text or "")
    fenced = re.findall(r"```[\w+#.-]*\n(.*?)```", body, re.S)
    if fenced:
        best = max(fenced, key=len).strip("\n")
        if best.strip():
            return best
    cleaned = _COMMENTARY.sub("", body)
    cleaned = _PREAMBLE.sub("", cleaned).strip()
    return cleaned or body.strip()


# --------------------------------------------------------------------------
# Theory of Mind (Appendix C, §6.11)
# --------------------------------------------------------------------------

SKILL_LEVELS = ("novice", "intermediate", "senior")

_SKILL_GUIDANCE = {
    "novice": ("Explain anything non-obvious in a short comment on the line "
               "above it. Prefer the clear approach over the clever one. "
               "Avoid idioms that need prior knowledge to read."),
    "intermediate": ("Comment the WHY, not the what. Use the language's "
                     "normal idioms without explaining them."),
    "senior": ("Assume fluency. Comment only where an obvious approach was "
               "rejected and the reason is not visible in the code."),
}

_JARGON = {
    "novice": ("When you must use a technical term, put a plain-English "
               "gloss in brackets the first time."),
    "intermediate": "Use standard terminology without glossing it.",
    "senior": "Use precise technical terminology.",
}


@dataclass(frozen=True)
class Persona:
    """One role, with its self-model and its view of the user.

    FORGE's `[SELF-MODEL] / [USER-MODEL] / [DIRECTIVE]` structure is kept: it
    is a good structure, and — the practical reason — it makes prompts
    DIFFABLE. When output quality changes, being able to diff two prompts and
    see one changed block is worth a great deal.
    """
    id: str
    label: str
    self_model: str
    directive: str
    temperature: float = 0.15

    def block(self, profile: dict | None = None) -> str:
        p = dict(profile or {})
        skill = str(p.get("skill_level", "intermediate")).lower()
        if skill not in SKILL_LEVELS:
            skill = "intermediate"
        lines = [
            "[SELF-MODEL]",
            self.self_model,
            "",
            "[USER-MODEL]",
            f"The person reading this code is {skill}. "
            f"{_SKILL_GUIDANCE[skill]} {_JARGON[skill]}",
        ]
        if p.get("domain"):
            lines.append(f"The project's domain is {p['domain']}.")
        if p.get("constraints"):
            lines.append(f"Standing constraints: {p['constraints']}")
        lines += ["", "[DIRECTIVE]", self.directive]
        return "\n".join(lines)


ENGINEER = Persona(
    id="engineer", label="Engineer",
    self_model=("You write working code in an existing project. You are "
                "precise about names and signatures, and you never invent an "
                "API you have not been shown — if you need something that "
                "does not exist, you look it up or say so."),
    directive=("Write the file you are asked for, completely, so that it "
               "compiles and its tests pass. Use only what the project and "
               "the standard library provide."),
    temperature=0.15)

TESTER = Persona(
    id="tester", label="Test author",
    self_model=("You write tests that would FAIL against an unimplemented "
                "stub and PASS against a correct implementation. A test that "
                "passes against `raise NotImplementedError` tests nothing and "
                "is worse than no test, because it manufactures confidence."),
    directive=("Write the test file for the described behaviour. Test the "
               "contract, not the implementation. Include the edge case that "
               "the description implies but does not state."),
    temperature=0.2)

PLANNER = Persona(
    id="planner", label="Planner",
    self_model=("You break a request into the smallest set of files that "
                "actually delivers it. You do not invent structure the "
                "request does not need, and you pair every implementation "
                "file with its test file."),
    directive=("List the files needed, one line each, as `path — purpose`. "
               "Nothing else."),
    temperature=0.35)

REVIEWER = Persona(
    id="reviewer", label="Reviewer",
    self_model=("You look for what tools cannot see: trust boundaries, logic "
                "flaws, misuse potential, and costs that only appear at "
                "scale. You do not repeat what a linter already said."),
    directive=("Report findings as a structured list. Say plainly when you "
               "find nothing — a clean review stated confidently is more "
               "useful than a manufactured concern."),
    temperature=0.35)

REPAIRER = Persona(
    id="repairer", label="Repairer",
    self_model=("You fix the specific errors you are shown and change "
                "NOTHING else. Rewriting the surrounding function while you "
                "are in there is how a one-line fix becomes a regression."),
    directive=("Fix the reported errors. Return the complete corrected file. "
               "Do not refactor, rename, or reformat anything the errors did "
               "not force you to touch."),
    temperature=0.15)

PERSONAS = {p.id: p for p in (ENGINEER, TESTER, PLANNER, REVIEWER, REPAIRER)}


# --------------------------------------------------------------------------
# output contracts (§4.4, M36)
# --------------------------------------------------------------------------
#
# Written to say what TO produce rather than to list what not to. A prompt
# that says "do not write a Changes Made section" has just taught the model
# the phrase "Changes Made" — models repeat prompt vocabulary, so the
# forbidden things are named once, flatly, at the end, and never rehearsed.

CONTRACT_FILE = (
    "OUTPUT CONTRACT\n"
    "Return the complete contents of the file, and nothing else. Start at the "
    "first line of the file and stop at the last. No prose before it, no "
    "prose after it, no headings, no summary of what you did.")

CONTRACT_EDIT = (
    "OUTPUT CONTRACT\n"
    "Return only the changed block, in this exact form:\n"
    "<<<CC-EDIT path/to/file\n"
    "the exact text to replace\n"
    "===\n"
    "the replacement text\n"
    ">>>CC-END\n"
    "The text to replace must appear EXACTLY ONCE in the file — include "
    "enough surrounding lines to make it unique. Nothing outside the block.")

CONTRACT_LIST = (
    "OUTPUT CONTRACT\n"
    "Return one item per line, in the form `path — purpose`. No numbering, "
    "no headings, no preamble, no closing remarks.")

CONTRACT_JSON = (
    "OUTPUT CONTRACT\n"
    "Return one JSON object and nothing else. No fence, no prose.")


# --------------------------------------------------------------------------
# prompt assembly (G.7.1) — the cache boundary is the point
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    """A prompt split at the cache boundary.

    `prefix` MUST be byte-identical between calls with the same (persona,
    epoch, target) — that is the prefix-stability test (M52), and the reason
    this type exists rather than a plain list of messages.
    """
    prefix: str
    tail: str
    system: str = ""

    def messages(self) -> list[Message]:
        """The two halves, as messages, in cache order.

        The prefix is its own system message so that the boundary is a real
        message boundary — servers that cache per message get the benefit
        for free, and it costs nothing where they don't.
        """
        out = []
        if self.system:
            out.append(Message(role="system", content=self.system))
        if self.prefix:
            out.append(Message(role="system", content=self.prefix))
        out.append(Message(role="user", content=self.tail))
        return out

    @property
    def text(self) -> str:
        return "\n\n".join(p for p in (self.system, self.prefix, self.tail)
                           if p)


@dataclass
class PromptBuilder:
    """Assembles prompts with the stable/volatile split held to.

    `model_system_prompt` is the model's OWN shipped prompt (Devstral ships
    `CHAT_SYSTEM_PROMPT.txt`). It goes first and is not replaced — the model
    was tuned with it, and discarding that tuning to install our own voice is
    a bad trade.
    """
    model_system_prompt: str = ""
    conventions: str = ""
    profile: dict = field(default_factory=dict)

    # -- the stable half --------------------------------------------------
    def prefix_for(self, persona: Persona, *, architecture: str = "",
                   epoch: int = 0) -> str:
        """CACHED PREFIX: persona, conventions, low-resolution architecture.

        Nothing time-varying, nothing task-varying. `epoch` appears only as
        an integer that changes when the architecture is deliberately
        rebuilt — which is the one thing in here that is ALLOWED to change
        the prefix, because when it changes the cache should be discarded.
        """
        parts = [persona.block(self.profile)]
        if self.conventions:
            parts.append("[PROJECT CONVENTIONS]\n" + self.conventions)
        if architecture:
            parts.append("[EXECUTION CONSTRAINTS]\n"
                         "This is the project you are working in. Do not "
                         "invent files, functions or imports that are not "
                         "here.\n\n" + architecture)
        return "\n\n".join(parts)

    # -- the volatile half ------------------------------------------------
    def tail_for(self, task: str, *, interfaces: str = "",
                 examples: str = "", staleness: str = "",
                 diagnostics: str = "", contract: str = CONTRACT_FILE,
                 extra: Sequence[str] = ()) -> str:
        """VOLATILE TAIL, ordered for recency (D7).

        The order is deliberate and load-bearing: hard constraints benefit
        from primacy, the output contract from recency. Items 7 and 8 of
        G.7.1 conflict (diagnostics are volatile, the contract wants to be
        last); the resolution is to keep the contract SHORT so repeating it
        costs little, and put the diagnostics immediately before it.
        """
        parts: list[str] = []
        if interfaces:
            parts.append(interfaces)
        if examples:
            parts.append(examples)
        parts.append("[TASK]\n" + task)
        parts.extend(x for x in extra if x)
        if staleness:
            parts.append(staleness)
        if diagnostics:
            parts.append("[WHAT WENT WRONG LAST TIME]\n" + diagnostics)
        if contract:
            parts.append(contract)
        return "\n\n".join(parts)

    def build(self, persona: Persona, task: str, *, architecture: str = "",
              epoch: int = 0, **tail_kw) -> Prompt:
        return Prompt(
            system=self.model_system_prompt,
            prefix=self.prefix_for(persona, architecture=architecture,
                                   epoch=epoch),
            tail=self.tail_for(task, **tail_kw))


# --------------------------------------------------------------------------
# the repair prompt (D11, M33)
# --------------------------------------------------------------------------

def repair_task(path: str, purpose: str, diagnostics: str = "",
                autofixes: Sequence[str] = ()) -> str:
    """The task text for a repair attempt.

    **The broken code is NOT included** (D11, M33). Attempt 3's prompt
    containing attempts 1 and 2 is how a model pattern-matches its own
    mistakes and repeats them. The DIAGNOSTICS carry forward; the broken code
    does not. The current file is on disk and goes in as context, once, as
    the file — not as "here is what you got wrong twice".

    ``diagnostics`` is included here only when the caller is not going to
    place it in the volatile tail itself. `PromptBuilder.tail_for` puts it
    immediately before the output contract, which is where recency helps most
    (D7) — so the loop passes it there and leaves this empty rather than
    stating the same errors twice. An argument that is silently ignored is a
    bug waiting for someone to trust it.
    """
    lines = [f"The file `{path}` does not work yet. Its purpose: {purpose}",
             "",
             "Fix the errors reported below and return the complete "
             "corrected file. Change nothing the errors did not force you to "
             "change."]
    if diagnostics:
        lines += ["", diagnostics]
    if autofixes:
        lines += ["",
                  "These were already fixed for you automatically, so do not "
                  "undo them:"]
        lines += [f"  - {f}" for f in autofixes]
    return "\n".join(lines)


def continuation_task(tail: str, lines_so_far: int) -> str:
    """The prompt for continuing a truncated generation (D1, M32).

    Continuation, not regeneration. The model ran out of room; it did not get
    it wrong. Re-generating from the start pays for the whole file again and
    frequently produces a DIFFERENT file, which is worse than slow.
    """
    return ("Your previous answer was cut off because it reached the length "
            f"limit, after {lines_so_far} lines. Continue from exactly where "
            "it stopped. Do not repeat anything you have already written, and "
            "do not start again.\n\n"
            "The last lines you wrote were:\n"
            f"{tail}\n\n"
            "Continue from the very next character.")


def guard_task(path: str, explanation: str) -> str:
    """The prompt after a guard refusal. An instruction, not a complaint."""
    return (f"The code you wrote for `{path}` was refused before it ran.\n\n"
            f"{explanation}\n\n"
            f"Return the complete file, rewritten so none of that is needed.")


def independence_caveat(same_model: bool) -> str:
    """The one line a same-model review must carry, near the top (M41).

    **Two personas that are the same local model are not an adversarial
    system.** Presenting same-model self-review as independent scrutiny is
    the confident-wrongness failure mode, dressed as diligence. Real
    adversarial value requires a different model or a genuinely different
    information set.
    """
    if not same_model:
        return ""
    return ("NOTE: the security and performance perspectives below came from "
            "the same model, so they are not independent scrutiny. Treat them "
            "as one reviewer's opinion expressed twice, not as two reviewers "
            "agreeing.")
