# SPDX-License-Identifier: Apache-2.0
"""One way to ask a model for a JSON object, and get one.

Every tab that asks for structure — the bug hunt's list, Improve's change
list, the red team's threat model and reproducers, Blueprint's modules,
Debug & Repair's locate step — used to do the same three things by hand:
build a prompt with a schema in the contract, call `complete`, and hope
`find_json_object` finds something. For a 24B–40B model that hope is good
on nine calls in ten. The tenth returns prose around the object, a fence
with a trailing comma, two objects, or the schema itself with the example
values still in it — and the caller gets `{}` with no idea why.

This module makes the ask ONE thing, with two defences that cost nothing
when they are not needed:

  1. **A grammar, when the host can enforce one.** llama.cpp's GBNF makes
     "return one JSON object" a property of decoding instead of a request.
     The host reports `supports_grammar` honestly (ATK's port proves it at
     the seam rather than asserting it), so a grammar is passed only where
     it is applied, and the journal records which.
  2. **One retry that quotes the problem.** When the reply does not parse
     — or parses but lacks the keys asked for — the model is told exactly
     that, with the first 300 characters of what it sent, and asked once
     more. Not a loop: a second failure is returned as a failure with the
     reason, because a model that cannot return an object twice will not
     on the fifth try, and the caller's "could not read the reply" note is
     worth more than the tokens.

The grammar is the generic JSON object grammar (the keys are the prompt's
business, not the grammar's): a schema-derived grammar pins the shape but
rejects a model that puts the keys in another order, which is a failure
mode nobody wants to debug at 2 a.m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .personas import strip_think
from .providers.base import find_json_object

#: The generic JSON-object grammar (GBNF). Identical to `review.py`'s; it
#: is repeated here rather than imported so neither module depends on the
#: other's internals for a seven-line constant.
JSON_GRAMMAR = r'''
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" ["\\/bfnrt])* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws     ::= [ \t\n]*
'''


@dataclass
class Structured:
    """What came back: the object (or None), and how it was obtained."""
    data: dict | None
    text: str                  # the final reply, think-stripped
    attempts: int              # 1, or 2 when the retry ran
    grammar: bool              # a grammar was passed to the host
    problem: str = ""          # why `data` is None, when it is

    @property
    def ok(self) -> bool:
        return self.data is not None


def _supports_grammar(llm: Any) -> bool:
    try:
        return bool(getattr(llm.capabilities(), "supports_grammar", False))
    except Exception:                                        # noqa: BLE001
        return False


def _is_empty_object(text: str) -> bool:
    import re
    return any(re.fullmatch(r"\{\s*\}", text[a:b])
               for a, b in _objects(text))


def _objects(text: str):
    from .providers.base import _balanced_objects
    return list(_balanced_objects(text))


def _problem(text: str, keys: tuple[str, ...]) -> str:
    """Why `text` is not the object asked for, in one sentence."""
    stripped = text.strip()
    if not stripped:
        return "the reply was empty"
    obj = find_json_object(stripped, keys=keys)
    if not obj and not _is_empty_object(stripped):
        return "the reply held no JSON object that could be parsed"
    missing = [k for k in keys if k not in obj]
    if missing:
        return (f"the object is missing the key(s) "
                f"{', '.join(repr(k) for k in missing)}")
    return ""


def ask(session: Any, persona: Any, task: str, *, keys: tuple[str, ...],
        example: str, temperature: float = 0.2, max_tokens: int = 1500,
        architecture: str = "", extra: tuple[str, ...] = (),
        retry: bool = True, cancel: Any = None) -> Structured:
    """One structured ask. `keys` are the top-level keys the object must
    have; `example` is the schema shown in the output contract."""
    llm = session.host.llm
    grammar = _supports_grammar(llm)
    contract = (f"OUTPUT CONTRACT\nReturn one JSON object in exactly this "
                f"shape and nothing else — no prose before or after it:\n"
                f"{example}")
    prompt = session.prompts.build(persona, task, architecture=architecture,
                                   contract=contract, extra=list(extra))
    messages = list(prompt.messages())
    kw: dict[str, Any] = {"temperature": temperature,
                          "max_tokens": max_tokens}
    if grammar:
        kw["grammar"] = JSON_GRAMMAR
    if cancel is not None:
        kw["cancel"] = cancel
    # the model profile (profiles.py): its samplers, and thinking OFF —
    # a reasoning model's thinking in front of a grammar-constrained JSON
    # object is a reply that is neither
    from . import profiles
    prof = getattr(session, "model_profile", None)
    kw.update(profiles.call_kwargs(llm, prof, thinking=False))
    messages = profiles.soft_switch(messages, prof, thinking=False)
    completion = llm.complete(messages, **kw)
    if completion.finish_reason == "cancelled":
        # a Stop is a stop, not "the reply was empty" to be asked again
        # (the 2026-10-27 review)
        from .errors import Cancelled
        raise Cancelled("a structured ask")
    text = strip_think(completion.text or "")
    problem = _problem(text, keys)
    attempts = 1
    if problem and retry and completion.finish_reason != "error":
        attempts = 2
        from .types import Message
        shown = text.strip()[:300] or "(nothing)"
        messages = messages + [
            Message(role="assistant", content=text.strip()[:2000]),
            Message(role="user", content=(
                f"That was not the object asked for: {problem}. You sent:\n"
                f"{shown}\n\nReturn ONLY one JSON object with the keys "
                f"{', '.join(keys)}, in the shape shown in the contract. "
                f"No explanation."))]
        kw["temperature"] = min(temperature, 0.1)
        completion = llm.complete(messages, **kw)
        if completion.finish_reason == "cancelled":
            from .errors import Cancelled
            raise Cancelled("a structured ask")
        text = strip_think(completion.text or "")
        problem = _problem(text, keys)
    data = None if problem else find_json_object(text, keys=keys)
    out = Structured(data=data, text=text, attempts=attempts,
                     grammar=grammar, problem=problem)
    try:
        session.journal.log("structured", persona=getattr(persona, "id", ""),
                            keys=list(keys), ok=out.ok, attempts=attempts,
                            grammar=grammar, problem=problem)
    except Exception:                                        # noqa: BLE001
        pass
    return out
