# SPDX-License-Identifier: Apache-2.0
"""The interview before the build: the decisions a request leaves open.

WHY THIS EXISTS, 2026-10-27
---------------------------
A small model fills a gap in a request with an invention, and invents
consistently enough that the invention becomes the build's truth. On the
2026-10-16 Star Trek build the test writer assumed a y-up world and the
spec said north is (0, -1); ten of sixteen failing tests followed from that
one unstated decision, and the judge, the suspects and two models' time
were spent arguing it. Eleven of twenty-six Debug & Repair sessions in the
10-20 journals were the operator typing what the request had not said.
`speclint.py` catches spelling; nothing asked the operator the questions.

So, before the plan, ONE structured call: "the decisions this request
leaves open that would change the code — at most five — each with the
answer you would otherwise assume." The host shows them as plain-language
choices with the model's default pre-selected; thirty seconds of the
operator's time. Each answer is appended to the request as a sentence in
the operator's name with MUST in it, which `constraints.py` pins into
every prompt of the session, word for word — so the sixth file's repair
still knows which way is north.

The interview never runs unasked: it needs a person (`SessionConfig.
interview`), it is skipped in headless and bench runs, and a model that
returns nothing usable costs one call (two, when the first reply was
not JSON and structured.py asked once more) and nothing else. Nothing here
writes a file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any

from . import constraints, structured
from .errors import Cancelled
from .personas import PERSONAS

MAX_QUESTIONS = 5
MAX_OPTIONS = 4
HEADING = constraints.DECISIONS_HEADING

_SCHEMA = ('{"questions": [{"id": "q1", "question": "one plain sentence a '
           'non-programmer can answer", "options": ["the first way", "the '
           'second way"], "default": "the first way", "why": "one sentence: '
           'which code it changes"}]}')


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    options: tuple[str, ...]
    default: str
    why: str = ""

    def as_dict(self) -> dict:
        return {"id": self.id, "question": self.question,
                "options": list(self.options), "default": self.default,
                "why": self.why}


@dataclass
class Interview:
    """The questions asked, and the answers once they are given."""
    questions: list[Question] = field(default_factory=list)
    answers: dict[str, str] = field(default_factory=dict)
    problem: str = ""              # why there are no questions, or ""
    asked: bool = False            # a model call was made
    #: The operator chose to build WITHOUT the decisions (closed the
    #: dialog, or nobody was there to answer): nothing is pinned, and the
    #: model's defaults stay what they were — guesses, not rules.
    declined: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.questions)

    def decided(self) -> list[tuple[Question, str]]:
        """Each question with its answer: the operator's, else the
        model's default (the choice it would have made unasked)."""
        return [(q, (self.answers.get(q.id) or q.default).strip())
                for q in self.questions]

    def decisions_text(self) -> str:
        """The block appended to the request. Each line names MUST, so
        the hard-requirement rule (constraints.py) pins it in every
        prompt for the session."""
        if not self.questions or self.declined:
            return ""
        lines = [HEADING]
        for q, answer in self.decided():
            lines.append(f"- The code MUST follow this decision: "
                         f"{_sentence(q.question)} {answer.rstrip('.')}.")
        return "\n".join(lines)

    def apply(self, request: str) -> str:
        """The request with the decisions appended — the request a build
        runs on."""
        block = self.decisions_text()
        if not block:
            return request
        return request.rstrip() + "\n\n" + block + "\n"

    def to_json(self) -> dict:
        return {"questions": [q.as_dict() for q in self.questions],
                "answers": dict(self.answers), "problem": self.problem,
                "asked": self.asked, "declined": self.declined}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Interview:
        qs = [Question(id=str(q.get("id") or f"q{i + 1}"),
                       question=str(q.get("question") or ""),
                       options=tuple(str(o) for o in q.get("options") or ()),
                       default=str(q.get("default") or ""),
                       why=str(q.get("why") or ""))
              for i, q in enumerate(data.get("questions") or [])]
        return cls(questions=qs, answers=dict(data.get("answers") or {}),
                   problem=str(data.get("problem") or ""),
                   asked=bool(data.get("asked")),
                   declined=bool(data.get("declined")))


def _sentence(question: str) -> str:
    """`Which way is north?` → `Which way is north:` — the question as the
    lead of a statement."""
    q = " ".join((question or "").split()).rstrip("?.: ")
    return q + ":"


# --------------------------------------------------------------------------
# asking
# --------------------------------------------------------------------------

def task_text(request: str, max_questions: int = MAX_QUESTIONS) -> str:
    return "\n".join([
        f"Read this request and list the decisions it leaves OPEN that "
        f"would change the code: a coordinate convention, a unit, a "
        f"format, what happens on bad input, where a file lives, which of "
        f"two readings is meant. At most {max_questions}; none if the "
        f"request settles everything. For each: one plain question a "
        f"person who does not program can answer, two to {MAX_OPTIONS} "
        f"short options, the option you would assume if nobody answered "
        f"(`default`, one of the options exactly), and one sentence on "
        f"which code it changes. Do not ask what the request already says.",
        "",
        f"REQUEST: {request}",
    ])


def ask(session: Any, request: str, *,
        max_questions: int = MAX_QUESTIONS, cancel: Any = None) -> Interview:
    """One structured call (structured.py): the open decisions as
    questions. Never raises on the model's account."""
    out = Interview(asked=True)
    try:
        got = structured.ask(session, PERSONAS["interviewer"],
                             task_text(request, max_questions),
                             keys=("questions",), example=_SCHEMA,
                             temperature=0.2, max_tokens=1200,
                             cancel=cancel)
    except Cancelled:
        raise                       # a Stop is the operator's, not a problem
    except Exception as exc:                              # noqa: BLE001
        out.problem = f"the interview could not be asked ({exc})"
        return out
    if not got.ok or not isinstance(got.data, dict):
        out.problem = got.problem or "the model returned nothing usable"
        return out
    out.questions = clean(got.data.get("questions"), max_questions)
    if not out.questions:
        out.problem = ("the model found no open decision in the request"
                       if isinstance(got.data.get("questions"), list)
                       and not got.data.get("questions")
                       else "the model's questions were not usable")
    try:
        session.journal.log("interview", kind="questions",
                            count=len(out.questions), problem=out.problem,
                            questions=[q.as_dict() for q in out.questions])
    except Exception:                                      # noqa: BLE001
        pass
    return out


#: The schema's own placeholder texts: a model that echoes the example
#: back has asked nothing (structured.py names this failure mode).
_SCHEMA_TEXTS = frozenset(t.lower() for t in (
    "one plain sentence a non-programmer can answer", "the first way",
    "the second way", "one sentence: which code it changes"))


def _is_schema_echo(text: str) -> bool:
    return " ".join(str(text or "").split()).strip().lower() in _SCHEMA_TEXTS


def clean(raw: Any, max_questions: int = MAX_QUESTIONS) -> list[Question]:
    """The usable questions out of what the model sent, by rule: a
    question with two or more distinct options and a default among them;
    duplicates dropped; the first `max_questions` kept."""
    if not isinstance(raw, list):
        return []
    out: list[Question] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = " ".join(str(item.get("question") or "").split())
        if len(question) < 8 or _is_schema_echo(question):
            continue
        raw_options = item.get("options")
        if not isinstance(raw_options, (list, tuple)):
            # "options": "up or down" is one string, not a list — iterating
            # it would make the letters the choices (the 2026-10-27 review)
            continue
        options: list[str] = []
        for o in raw_options:
            if not isinstance(o, (str, int, float)):
                continue
            text = " ".join(str(o).split()).strip()
            if text and text.lower() not in {x.lower() for x in options}:
                options.append(text[:160])
        if any(_is_schema_echo(o) for o in options):
            continue
        options = options[:MAX_OPTIONS]
        if len(options) < 2:
            continue
        default = " ".join(str(item.get("default") or "").split()).strip()
        match = next((o for o in options if o.lower() == default.lower()),
                     "")
        if not match:
            match = next((o for o in options
                          if default and default.lower() in o.lower()),
                         options[0])
        key = re.sub(r"\W+", " ", question.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(Question(id=f"q{len(out) + 1}", question=question[:240],
                            options=tuple(options), default=match,
                            why=" ".join(str(item.get("why") or "")
                                         .split())[:240]))
        if len(out) >= max_questions:
            break
    return out


def answer(interview: Interview, answers: Mapping[str, Any]) -> Interview:
    """The operator's answers, by question id or by its number; an answer
    that is not one of the options is kept as typed (it is their call)."""
    for key, value in dict(answers or {}).items():
        text = " ".join(str(value or "").split()).strip()
        if not text:
            continue
        qid = str(key)
        if qid.isdigit():
            qid = f"q{qid}"
        if any(q.id == qid for q in interview.questions):
            interview.answers[qid] = text[:240]
    return interview


def ask_on_terminal(interview: Interview, *, read: Any = input,
                    write: Any = print) -> Interview:
    """The CLI's interview: each question with numbered options, the
    default marked; Enter keeps the default; a number picks; any other
    text is the answer."""
    for i, q in enumerate(interview.questions, 1):
        write(f"\n{i}. {q.question}")
        if q.why:
            write(f"   ({q.why})")
        for j, option in enumerate(q.options, 1):
            mark = " (default)" if option == q.default else ""
            write(f"   {j}) {option}{mark}")
        try:
            reply = str(read("   your answer [Enter = default]: ") or "")
        except EOFError:
            # nobody is there to answer: nothing is confirmed in the
            # operator's name — the decisions are left to the model, said
            interview.declined = True
            write("   (no answer — the open decisions are left to the "
                  "model, none is pinned)")
            return interview
        # KeyboardInterrupt is NOT caught: Ctrl+C is a cancel everywhere
        # else in this CLI, and must not confirm the defaults (the
        # 2026-10-27 review)
        reply = reply.strip()
        if not reply:
            continue
        if reply.isdigit() and 1 <= int(reply) <= len(q.options):
            interview.answers[q.id] = q.options[int(reply) - 1]
        else:
            interview.answers[q.id] = reply[:240]
    return interview


def summary(interview: Interview) -> str:
    """One line for the log and the status bar."""
    if not interview.questions:
        return f"interview: no questions ({interview.problem or 'nothing open'})"
    if interview.declined:
        return (f"interview: {len(interview.questions)} decision(s) left to "
                f"the model — none pinned")
    answered = sum(1 for q in interview.questions if q.id in interview.answers)
    return (f"interview: {len(interview.questions)} decision(s) confirmed, "
            f"{answered} by the operator, "
            f"{len(interview.questions) - answered} by the model's default")
