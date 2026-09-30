# SPDX-License-Identifier: Apache-2.0
"""Errors an operator may see — and the rule that they are sentences (C6).

C6 says: every failure that can reach a human is a plain sentence naming what
happened and what to do. Tracebacks go to the journal, not the face.

That rule is easy to state and easy to erode, so it is mechanised here rather
than left to discipline. Every exception in this module carries:

  * ``sentence`` — what an operator reads. No type names, no tracebacks, no
    "unexpected error occurred". It says what happened and what to do next.
  * ``detail``   — the developer-facing text, including a formatted traceback
    when one exists. This goes to the journal.

``str(exc)`` returns the sentence, so the lazy call site — ``event("error",
str(exc))`` — is also the correct one. That is deliberate: a rule that is
harder to follow than to break will be broken.

Nothing in this module is raised at an operator across a Port boundary without
a matching journal entry; ``Session`` is what pairs them.
"""

from __future__ import annotations

import traceback


class CognitiveCoderError(Exception):
    """Base for everything this package raises deliberately.

    A host may catch this one type and be sure it has caught every failure the
    engine considers its own. Anything else escaping the core is a bug in the
    core, and should be reported as one.
    """

    def __init__(self, sentence: str, detail: str = "") -> None:
        super().__init__(sentence)
        self.sentence = sentence
        self.detail = detail or ""

    def __str__(self) -> str:            # what an operator sees
        return self.sentence

    @classmethod
    def wrap(cls, exc: BaseException, sentence: str = ""
             ) -> CognitiveCoderError:
        """Turn an internal exception into an operator-facing one.

        The traceback is preserved in ``detail`` for the journal — this is the
        mechanism by which C6 loses nothing. A swallowed traceback is a
        different bug from a displayed one, and both are bad. The original is
        also chained as ``__cause__``, so `raise err` shows both.

        Works for EVERY subclass. It used to call ``cls(sentence, detail)``,
        which is the base signature only: `PathEscape.wrap(...)` produced
        "Refused to touch 'could not load the plan'" and
        `NoModelLoadedError.wrap(...)` raised TypeError — an error path that
        errors. Subclass fields keep their class defaults.

        ``sentence`` is made a sentence (capitalised, closed with a full
        stop). Without one, the exception's own message is used — never its
        type name or traceback, which belong in ``detail``.
        """
        detail = "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__))
        if isinstance(exc, CognitiveCoderError) and exc.detail:
            detail = exc.detail.rstrip("\n") + "\n\n" + detail
        err = Exception.__new__(cls)
        CognitiveCoderError.__init__(err, _as_sentence(sentence, exc),
                                     detail)
        err.__cause__ = exc
        return err


def _as_sentence(text: str, exc: BaseException) -> str:
    """A plain sentence: capitalised, ending in . ! or ? (C6)."""
    body = " ".join(str(text or "").split())
    if not body:
        said = " ".join(str(exc).split()) if not isinstance(
            exc, CognitiveCoderError) else exc.sentence
        body = (f"The engine stopped: {said.rstrip('.')}" if said else
                "The engine stopped on an internal error; the details are "
                "in the journal")
    body = body[0].upper() + body[1:]
    return body if body.endswith((".", "!", "?")) else body + "."


class PortError(CognitiveCoderError):
    """A host's Port did something the contract forbids.

    Raised at the CORE's boundary, not the host's, so the sentence names the
    host's obligation rather than the core's internals — the person reading it
    is usually the person who wrote the Port.
    """


class ConfigurationError(CognitiveCoderError):
    """The engine was asked to do something its settings do not allow."""


class NoModelLoadedError(CognitiveCoderError):
    """There is no model to ask.

    Not an exceptional condition in the ordinary sense: the host owns model
    loading and unloading (§0.1), so "no model is loaded" is a NORMAL,
    reportable state that the core must survive gracefully. It is an exception
    only because it stops the work in progress.
    """

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "No model is loaded, so there is nothing to ask. Load a model in "
            "the host application and run this again.", detail)


class GuardRefusal(CognitiveCoderError):
    """The static screen refused to run generated code (§6.3).

    Not a security verdict — see the guard module's docstring, and C10. This
    is the engine declining to run something that looks like an accident.
    """

    findings: tuple = ()            # a wrapped instance keeps defaults

    def __init__(self, reason: str, findings: tuple = ()) -> None:
        super().__init__(
            f"The generated code was refused before it ran: {reason}.")
        self.findings = findings


class PathEscape(CognitiveCoderError):
    """A path resolved outside the project root (M24).

    Deliberately its own type: this is the one failure that must never be
    retried, softened, or handled generically. Every mode, every caller, no
    exceptions.
    """

    path: str = ""
    root: str = ""

    def __init__(self, path: str, root: str) -> None:
        super().__init__(
            f"Refused to touch {path!r}: it resolves outside the project "
            f"folder ({root}). Nothing was written.")
        self.path = path
        self.root = root


class TransactionError(CognitiveCoderError):
    """A transaction was used in a way the model in §6.5 does not permit."""


class Cancelled(CognitiveCoderError):
    """The operator cancelled (§5.2).

    Resumable state has been left behind and any open transaction has been
    rolled back by the time this is raised. That is a guarantee of the raiser,
    not a hope of the catcher.
    """

    def __init__(self, where: str = "") -> None:
        super().__init__(
            f"Stopped at your request{f' during {where}' if where else ''}. "
            f"Work already verified has been kept; anything half-applied was "
            f"rolled back.")


class BudgetExceeded(CognitiveCoderError):
    """A wall-clock, token or spend budget ran out (F11, §6.12).

    Carries what was achieved, because "it stopped" without "and here is what
    you got" is the unhelpful half of the message.
    """

    kind: str = ""
    limit: str = ""

    def __init__(self, kind: str, limit: str, achieved: str = "") -> None:
        got = f" What was finished: {achieved}." if achieved else ""
        super().__init__(
            f"Stopped: the {kind} budget of {limit} ran out.{got}")
        self.kind = kind
        self.limit = limit
