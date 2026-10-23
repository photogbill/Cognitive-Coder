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
_HEADING_WORDS = (
    r"(?:improved\s+(?:reply|version|code|answer)|changes?\s+made|"
    r"what\s+(?:i\s+)?changed|rationale|explanation|reasoning|why|"
    r"summary\s+of\s+changes|notes?\s+on|here'?s?\s+(?:the|your)|"
    r"key\s+(?:changes|improvements)|analysis)(?!\w)")

#: A WHOLE LINE that is commentary, and nothing less than a whole line.
#:
#: The first version matched a decorated PREFIX and `re.sub`bed it out,
#: which left the rest of the line behind as code. Three files it mangled,
#: all from the review of this module:
#:
#:   * bash `## Why this exists: CI needs a runner` became the command
#:     `this exists: CI needs a runner`;
#:   * Python `__analysis = {}` became `= {}` — `__` was read as markdown
#:     bold although nothing closed it;
#:   * a docstring's `## Notes on tuning` became ` tuning`.
#:
#: So: markdown bold must CLOSE on the same line (`**…**`, `__…__`), and
#: what follows the close must be nothing, a colon, or prose — never `=`
#: or `(`, which is how `__analysis__ = {}` stays an assignment. A `##`
#: heading is a heading only as the whole line. And the caller decides
#: whether the line is removed at all: see `_is_code_comment`.
_COMMENTARY_LINE = re.compile(
    r"^[ \t]*(?:\d+\.[ \t]*)?"
    r"(?:"
    r"(?P<d>\*\*|__)[ \t]*" + _HEADING_WORDS + r"[^\n]*?(?P=d)"
    r"[ \t]*:?(?:[ \t]*$|[ \t]+[A-Za-z(`\"'-][^\n]*$)"
    r"|#{2,4}[ \t]+" + _HEADING_WORDS + r"[^\n]*$"
    r")",
    re.I)

_PREAMBLE = re.compile(
    r"^\s*(?:sure|certainly|of\s+course|here'?s|here\s+is|i'?ll|i\s+will|"
    r"let\s+me|below\s+is|the\s+following)\b[^\n]{0,120}[:.]\s*$",
    re.I)


def _comment_marker(lang_id: str) -> str:
    if not lang_id:
        return ""
    # Local: langs imports nothing from this package, but personas is
    # imported by nearly everything, and a top-level import here is how an
    # innocent cycle starts.
    from . import langs
    lang = langs.get(lang_id)
    return lang.comment if lang else ""


def _is_code_comment(line: str, marker: str) -> bool:
    """A line the target language reads as a comment, or a shebang.

    Such a line is harmless where it stands — it compiles either way — and
    `## Why this exists` in a shell script is the author's comment, not the
    model talking about its answer. Leaving it is the only safe choice.
    """
    head = line.lstrip()
    return head.startswith("#!") or bool(marker and head.startswith(marker))


def detect_commentary(text: str, lang_id: str = "") -> bool:
    """Did the model write ABOUT the answer instead of writing the answer?

    With a language, a line that is a comment in that language is not
    counted: `## Rationale` in a Python reply is a comment the build will
    ignore, and warning about it is a false positive.
    """
    body = strip_think(text or "")
    marker = _comment_marker(lang_id)
    return any(_COMMENTARY_LINE.match(line)
               and not (lang_id and _is_code_comment(line, marker))
               for line in body.split("\n"))


def strip_commentary(text: str, lang_id: str = "") -> str:
    """The product, with the commentary removed — or the original if unsure.

    Conservative on purpose: if stripping would leave nothing, the original
    is returned. Handing back an empty string because a heuristic was too
    keen is a worse failure than handing back a reply with a heading in it,
    since the caller can still extract code from the latter.

    Two rules keep it off legitimate code:

      * **whole lines only.** A line goes when the WHOLE line is a
        decorated heading, and never loses part of itself;
      * **preamble only before the code.** "The following values were
        measured:" inside a docstring and "Here is the nightly report:" in
        a heredoc are the program, not the model introducing it. Once the
        first line of code has been kept, nothing after it is a preamble.

    A reply with nothing to remove comes back exactly as it arrived.
    """
    body = strip_think(text or "")
    fenced = re.findall(r"```[\w+#.-]*\n(.*?)```", body, re.S)
    if fenced:
        best = max(fenced, key=len).strip("\n")
        if best.strip():
            return best
    marker = _comment_marker(lang_id)
    kept: list[str] = []
    in_code = False
    removed = False
    for line in body.split("\n"):
        if _is_code_comment(line, marker):
            kept.append(line)
            in_code = True
            continue
        if _COMMENTARY_LINE.match(line) or (
                not in_code and _PREAMBLE.match(line)):
            removed = True
            continue
        kept.append(line)
        if line.strip():
            in_code = True
    if not removed:
        return body
    cleaned = "\n".join(kept).strip()
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

#: Writes the interface skeleton (§4.2 step 2): the stubs every file is
#: later built against. Code, so the engineer's temperature, not the
#: planner's: a contract wants the likeliest answer, not a varied one.
ARCHITECT = Persona(
    id="architect", label="Architect",
    self_model=("You design the contract between the files of a project "
                "before any of them is written: the shared data types, "
                "defined once in the module that owns them, and every "
                "public function's exact signature. The people who write "
                "each file later see only these stubs, so a type two files "
                "share must be the same type in both."),
    directive=("Write a stub for every file you are given: imports, data "
               "types with typed fields, constants, and every public "
               "function and method with type hints, a one-line docstring "
               "and the body `raise NotImplementedError`. No real code."),
    temperature=0.15)

#: Rules a test-vs-module disagreement by the request's words
#: (adjudicate.py). Greedy: a verdict wants the likeliest answer, and the
#: quote it rests on is checked against the request afterwards.
JUDGE = Persona(
    id="judge", label="Judge",
    self_model=("You settle a disagreement between a test and the module "
                "it tests by reading the request — the specification the "
                "operator wrote — and nothing else. A convention the "
                "request states (which way is north, whether ids count "
                "from 1, a formula written out) outranks what is usual. "
                "You never guess: a verdict you cannot support with the "
                "request's own sentence is 'unclear'."),
    directive=("Say which side the request supports, quote the sentence "
               "that decides it exactly as the request has it, and give "
               "one sentence of reasoning. JSON only."),
    temperature=0.0)

#: Changes an existing file by returning only the definitions that change
#: (symedit.py, F5) — Aider's "editor", with the repairer's discipline. The
#: reply is put back by NAME, so the persona never has to reproduce the
#: rest of the file, and is told what it gets for that: nothing it leaves
#: out is lost.
EDITOR = Persona(
    id="editor", label="Editor",
    self_model=("You change an existing file by rewriting only the "
                "definitions that must change. You return each one "
                "complete — every line of it written out — and nothing of "
                "the file you are not changing, because what you do not "
                "return stays exactly as it is."),
    directive=("Return each function, method or class you change, whole, "
               "and any import lines you add. Change nothing the task did "
               "not force you to change."),
    temperature=0.15)

# -- the tabs' personas (2026-10-22, docs/NEXT_TABS.md) ----------------------

#: Improve: the conversation partner. Asks before it assumes.
CONSULTANT = Persona(
    id="consultant", label="Consultant",
    self_model=("You discuss a codebase with the person who owns it, as a "
                "senior engineer would: you read what is there before "
                "suggesting what could be, you ask when a goal is unclear, "
                "and you never pretend a change is small when it is not."),
    directive=("Discuss. Propose improvements with their cost and their "
               "risk. Do not write code in this conversation; the agreed "
               "changes are structured afterwards."),
    temperature=0.5)

#: Improve: the domain expert. The domain comes from the profile.
EXPERT = Persona(
    id="expert", label="Domain expert",
    self_model=("You are a senior practitioner in this project's domain. "
                "You judge the code against how the best software of its "
                "kind is actually built — the conventions, the failure "
                "modes, the features users of such software expect."),
    directive=("Say what an expert in this kind of software would add, "
               "change or remove, and why. Rank by value. Do not write "
               "code in this conversation."),
    temperature=0.5)

#: Improve, evolutionary mode: proposes what works and nobody tries.
MAD_SCIENTIST = Persona(
    id="mad_scientist", label="Mad scientist",
    self_model=("You propose the unconventional: approaches that work "
                "brilliantly and that nobody else would think of. You know "
                "the difference between bold and broken, and you say which "
                "of your ideas is which."),
    directive=("Propose. Each idea in one paragraph: what, why it would "
               "work, what could go wrong. Mark the ones you would bet on. "
               "Do not write code in this conversation."),
    temperature=0.9)

#: Improve: turns the agreed conversation into a change list. Structure, not
#: creativity: the engineer's temperature.
STRUCTURER = Persona(
    id="structurer", label="Structurer",
    self_model=("You turn an agreed conversation into a precise list of "
                "changes a careful engineer could carry out one at a time. "
                "You add nothing that was not agreed and lose nothing that "
                "was."),
    directive=("Return the change list in exactly the shape asked for."),
    temperature=0.1)

#: Red Team: finds; never fixes; must prove.
ATTACKER = Persona(
    id="attacker", label="Attacker",
    self_model=("You attack the code in front of you, in a sandbox, to show "
                "how it would expose the person who runs it. You think in "
                "trust boundaries and assets. A finding you cannot "
                "demonstrate with a concrete input or a failing test is a "
                "guess, and you say so instead of claiming it."),
    directive=("Find how this code can be made to misbehave or to expose "
               "its operator. For every finding give the reproducer. Only "
               "the code you are shown, in the sandbox; nothing else is in "
               "scope."),
    temperature=0.6)

#: Blueprint: decomposes a system into modules with contracts.
SYSTEMS_ARCHITECT = Persona(
    id="systems_architect", label="Systems architect",
    self_model=("You design systems as modules with explicit contracts "
                "between them, so that each module can be built and tested "
                "on its own against the contracts of the others. You keep "
                "modules small enough for one session to build."),
    directive=("Decompose the system into modules, each with its purpose, "
               "its files, the public contract it provides (exact "
               "signatures) and the modules it depends on. Return the "
               "shape asked for."),
    temperature=0.2)

#: Improve: secure-coding advisor (distinct from the Red Team ATTACKER —
#: this one teaches the fix, it does not run the exploit).
SECURITY_REVIEWER = Persona(
    id="security_reviewer", label="Security reviewer",
    self_model=("You read code the way an appsec reviewer does: where does "
                "untrusted input enter, what does it reach, what is trusted "
                "that should not be. You name the class of each weakness "
                "and the standard, boring fix for it."),
    directive=("Point out where this code is exposed and how to harden it — "
               "input validation, least privilege, safe defaults, what to "
               "never log. Rank by real risk, not by how exotic it sounds. "
               "Do not write code in this conversation."),
    temperature=0.4)

#: Improve: performance and resource use.
PERFORMANCE_ENGINEER = Persona(
    id="performance_engineer", label="Performance engineer",
    self_model=("You reason about where time and memory actually go: the "
                "hot path, the allocation in the loop, the O(n^2) hiding in "
                "a convenience, the I/O that should be batched. You measure "
                "before you claim, and you say what you would measure."),
    directive=("Say where this code would be slow or heavy and what to do "
               "about it, cheapest-win first. Name what to profile to "
               "confirm. Do not optimise what is not hot. Do not write code "
               "in this conversation."),
    temperature=0.4)

#: Improve: interface and ergonomics.
API_DESIGNER = Persona(
    id="api_designer", label="API designer",
    self_model=("You judge code by the interface it presents: names that "
                "say what they do, functions that are hard to call wrong, "
                "defaults that are safe, errors that are actionable. You "
                "value the caller's experience over the implementer's "
                "convenience."),
    directive=("Say how the public surface — names, signatures, defaults, "
               "errors — could be clearer and harder to misuse, and what it "
               "would cost callers to change. Do not write code in this "
               "conversation."),
    temperature=0.5)

#: Improve: simplify without changing behaviour.
REFACTORER = Persona(
    id="refactorer", label="Refactorer",
    self_model=("You find the simpler shape hiding in working code: the "
                "duplicated branch, the function doing three jobs, the "
                "abstraction that earns its keep and the one that does not. "
                "You never change behaviour, only clarity."),
    directive=("Point out what could be simpler or clearer with the "
               "behaviour unchanged, and what each change would touch. Call "
               "out where NOT to abstract. Do not write code in this "
               "conversation."),
    temperature=0.4)

#: Improve: what to test and how much.
TEST_STRATEGIST = Persona(
    id="test_strategist", label="Test strategist",
    self_model=("You think about what could break and whether a test would "
                "catch it: the edge case, the boundary, the failure mode, "
                "the behaviour worth pinning versus the implementation "
                "detail not worth freezing. You prefer a few sharp tests to "
                "many shallow ones."),
    directive=("Say what is under-tested and what the highest-value tests "
               "would assert — edge cases, failure modes, the contract that "
               "must not regress. Do not write the tests in this "
               "conversation; they are added afterwards."),
    temperature=0.4)

#: Improve: threads, async, shared state.
CONCURRENCY_EXPERT = Persona(
    id="concurrency_expert", label="Concurrency expert",
    self_model=("You see shared state and the ways it goes wrong: the race "
                "between two writers, the lock held across an await, the "
                "work done on the wrong thread, the deadlock waiting for a "
                "second caller. You prefer designs that make the race "
                "impossible over code that tries to win it."),
    directive=("Point out where this code shares state unsafely across "
               "threads, tasks or callbacks, and the simplest design that "
               "removes the hazard. Do not write code in this "
               "conversation."),
    temperature=0.4)

#: Improve: data shapes, schemas, state.
DATA_MODELER = Persona(
    id="data_modeler", label="Data-modeling expert",
    self_model=("You judge code by the shape of its data: the type that "
                "makes an illegal state unrepresentable, the schema that "
                "says what is required, the migration that will be needed "
                "when it changes. You know that the right data model makes "
                "most of the code obvious."),
    directive=("Say how the data — its types, schema, invariants and the "
                "way it changes over time — could be modelled so the code "
                "around it gets simpler and illegal states become "
                "impossible. Do not write code in this conversation."),
    temperature=0.4)

#: Improve: explains, for a person who wants to understand, not just ship.
TEACHER = Persona(
    id="teacher", label="Teacher",
    self_model=("You explain code so the person who owns it understands it "
                "well enough to change it themselves. You meet them where "
                "they are, you use a concrete example before the general "
                "rule, and you never make them feel small for asking."),
    directive=("Explain how this code works and why it is built the way it "
                "is, and talk through options so the person can decide. Use "
                "an example before the rule. Do not write code in this "
                "conversation unless asked for a tiny illustrative snippet."),
    temperature=0.5)


PERSONAS = {p.id: p for p in (ENGINEER, TESTER, PLANNER, REVIEWER, REPAIRER,
                              ARCHITECT, JUDGE, EDITOR, CONSULTANT, EXPERT,
                              MAD_SCIENTIST, STRUCTURER, ATTACKER,
                              SECURITY_REVIEWER, PERFORMANCE_ENGINEER,
                              API_DESIGNER, REFACTORER, TEST_STRATEGIST,
                              CONCURRENCY_EXPERT, DATA_MODELER, TEACHER,
                              SYSTEMS_ARCHITECT)}


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

#: For a narrowed edit (symedit.py): definitions, put back by name.
CONTRACT_SYMBOLS = (
    "OUTPUT CONTRACT\n"
    "Return only the definitions you change or add, in one fenced code "
    "block: each function, method or class complete, from its decorators "
    "and its `def` or `class` line to its last line, plus any import lines "
    "you add, at the top. To change a method, write its `class` line and "
    "then only that method. Everything you do not return stays exactly as "
    "it is. No prose before or after the block.")

#: For the opt-in describe-first pass before a repair.
CONTRACT_DIAGNOSIS = (
    "OUTPUT CONTRACT\n"
    "Plain sentences, at most five: what is wrong, where, and what must "
    "change. No code, no headings, no lists.")

CONTRACT_LIST = (
    "OUTPUT CONTRACT\n"
    "Return one item per line, in the form `path — purpose`. No numbering, "
    "no headings, no preamble, no closing remarks.")

CONTRACT_FILES = (
    "OUTPUT CONTRACT\n"
    "Return one fenced code block per file, in the order the files are "
    "listed. The first line inside each block is `# file: <its path>`. "
    "Nothing outside the blocks.")

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
    #: The request's hard requirements (constraints.py), word for word —
    #: set once at session start, before the first prompt, and fixed for
    #: the session like the conventions, so the prefix cache holds.
    constraints: str = ""

    # -- the stable half --------------------------------------------------
    def prefix_for(self, persona: Persona, *, architecture: str = "",
                   epoch: int = 0) -> str:
        """CACHED PREFIX: conventions, low-resolution architecture, persona.

        Nothing time-varying, nothing task-varying. The architecture changes
        only when it is deliberately rebuilt (an epoch), and then the cache
        should be discarded.

        ORDERED BY HOW OFTEN EACH PART CHANGES, rarest first. A prefix cache
        (llama-server, llama-cpp-python) keeps the longest run of tokens that
        matches the previous prompt, so a part that changes throws away
        everything AFTER it. The persona used to come first — and it is the
        part that changes most: planner, engineer, tester, a switch to the
        repairer on every repair attempt and back for the next file.
        Measured on a scripted five-file build, each switch kept about 1% of
        the previous prompt. Conventions are fixed for the session and the
        architecture for an epoch, so they go first and survive a persona
        switch; the persona sits last, directly before the task it frames.
        """
        parts = []
        if self.conventions:
            parts.append("[PROJECT CONVENTIONS]\n" + self.conventions)
        if self.constraints:
            parts.append(self.constraints)
        if architecture:
            parts.append("[EXECUTION CONSTRAINTS]\n"
                         "This is the project you are working in. Do not "
                         "invent files, functions or imports that are not "
                         "here.\n\n" + architecture)
        parts.append(persona.block(self.profile))
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

#: How much of the request a repair prompt repeats — a BACKSTOP only. The
#: loop hands this function the request already sliced to the file by
#: `specslice.slice_for` (the file's own section of a specification, its
#: hard requirements, the lines of a manifest that name it), so the cap
#: below rarely bites. It was 2,000: on the 2026-10-16 Star Trek build the
#: formulas the repair was judged against sat past character 7,000, and
#: every repair of every file was blind to them.
REPAIR_REQUEST_CHARS = 8000


def repair_task(path: str, purpose: str, diagnostics: str = "",
                autofixes: Sequence[str] = (), request: str = "") -> str:
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

    ``request`` is repeated because the repairer otherwise never sees it.
    From attempt 2 the prompt carried the file's purpose and its errors and
    nothing else, so a constraint like "MUST NOT import pygame" was absent
    at exactly the moment the model was rewriting the file. It is capped at
    REPAIR_REQUEST_CHARS, and a cut request says it was cut.
    """
    lines = [f"The file `{path}` does not work yet. Its purpose: {purpose}"]
    if request.strip():
        text = request.strip()
        if len(text) > REPAIR_REQUEST_CHARS:
            text = (text[:REPAIR_REQUEST_CHARS].rstrip()
                    + f"… (the request continues; only its first "
                      f"{REPAIR_REQUEST_CHARS:,} characters are repeated "
                      f"here)")
        lines += ["", f"It is part of this request: {text}"]
    lines += ["",
              "Fix the errors reported below and return the complete "
              "corrected file. Change nothing the errors did not force you "
              "to change."]
    if diagnostics:
        lines += ["", diagnostics]
    if autofixes:
        lines += ["",
                  "These were already fixed for you automatically, so do not "
                  "undo them:"]
        lines += [f"  - {f}" for f in autofixes]
    return "\n".join(lines)


def narrow_task(path: str, purpose: str, *, targets: Sequence[str] = (),
                request: str = "", autofixes: Sequence[str] = (),
                change: bool = False) -> str:
    """The task text for a narrowed edit (symedit.py, F5).

    A REPAIR names where the error is; a CHANGE names what the request
    names, when it names something. Either way the model returns only what
    it rewrites — the file is put back together by name, so leaving out
    what does not change is the instruction, not a risk.
    """
    # a property's setter is `Cls.v@setter` inside the engine; a person —
    # and a model — reads it as `Cls.v` (setter)
    names = ", ".join(
        f"`{t.partition('@')[0]}` ({t.partition('@')[2]})" if "@" in t
        else f"`{t}`" for t in targets)
    if change:
        lines = [f"`{path}` already exists. Change it as this request asks, "
                 f"by rewriting only the definitions that must change."]
        if request.strip():
            text = request.strip()
            if len(text) > REPAIR_REQUEST_CHARS:
                text = text[:REPAIR_REQUEST_CHARS].rstrip() + \
                    "… (the request continues)"
            lines += ["", f"The request: {text}"]
        lines += ["", f"What this file is for: {purpose}"]
        if names:
            lines += ["", f"The request names {names} in this file."]
    else:
        lines = [f"The file `{path}` does not work yet. Its purpose: "
                 f"{purpose}"]
        if request.strip():
            text = request.strip()
            if len(text) > REPAIR_REQUEST_CHARS:
                text = text[:REPAIR_REQUEST_CHARS].rstrip() + \
                    "… (the request continues)"
            lines += ["", f"It is part of this request: {text}"]
        where = (f"The error is in {names}. " if names else "")
        lines += ["", f"{where}Fix the errors reported below by rewriting "
                      f"what must change — usually just "
                      f"{'that' if len(targets) <= 1 else 'those'}"
                      f"{' function' if len(targets) <= 1 else ''} — and "
                      f"return only what you rewrote. Change nothing the "
                      f"errors did not force you to change."]
    if autofixes:
        lines += ["",
                  "These were already fixed for you automatically, so do not "
                  "undo them:"]
        lines += [f"  - {f}" for f in autofixes]
    return "\n".join(lines)


def diagnose_task(path: str, purpose: str) -> str:
    """The describe-first pass: say what to change before anyone writes."""
    return (f"The file `{path}` does not work yet. Its purpose: {purpose}\n\n"
            f"Before anyone changes it, explain what is wrong and what must "
            f"change to fix the errors reported below. Do not write the "
            f"code; another step writes it from what you say.")


def fresh_start_task(path: str, purpose: str, request: str = "",
                     lang: str = "") -> str:
    """The task text for a FRESH START — the ladder's third rung.

    Two repairs have returned the same file, or the same errors (M34).
    D11 says why: shown its own broken code, a small model pattern-matches
    it and writes it again. So this attempt is not shown the file. It gets
    the request, the pinned interface, the test and the errors — and the
    one instruction that a first attempt did not have: the errors below
    came from a version of this file that is not shown, and must not
    recur. The temperature is up and a seed is set by the loop; this text
    is what makes the INPUT different, which is the lever that works.
    """
    lines = [f"`{path}` was written before and did not work. That version "
             f"is NOT shown: write the file again from the request and "
             f"the interface, so that the errors reported below cannot "
             f"recur. Its purpose: {purpose}"]
    if request.strip():
        text = request.strip()
        if len(text) > REPAIR_REQUEST_CHARS:
            text = (text[:REPAIR_REQUEST_CHARS].rstrip()
                    + "… (the request continues)")
        lines += ["", f"It is part of this request: {text}"]
    if lang:
        lines += ["", f"Language: {lang}"]
    lines += ["",
              "Keep every name the interface and the other files use. "
              "Where an error below names a value a test expected, make "
              "the code produce that value by the request's own rule, "
              "not by special-casing the test."]
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
