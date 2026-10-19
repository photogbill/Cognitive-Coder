# SPDX-License-Identifier: Apache-2.0
"""Hybrid mode: which model plays which role, and in what order (policy).

Pure decisions, no I/O, no model — so every rule here is a unit test and
the session can print the reasoning before it loads anything. The session
does the loading (`Session._swap_to`) through the host's `ModelRoster`
port; the core still contains no loading code of its own.

THE SHAPE (docs/HYBRID.md): three phases, two swaps.

    ar-first         chat model writes → canvas model repairs holes →
                     chat model finishes with whole-file repairs
    diffusion-first  canvas model writes → chat model judges and repairs →
                     canvas model revisits

Roles are picked BY CAPABILITY — a slot that fills a canvas is a canvas
model, whatever its file is called — never by name. With one kind only,
the later phases are the same model again with the ladder's fresh-start
prompts: still worth a swap-free second pass, and said to be that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .types import ModelSlot

ORDERS = ("ar-first", "diffusion-first")
MODES = ("auto", "off", *ORDERS)


@dataclass(frozen=True)
class Phase:
    number: int
    name: str            # "write" | "repair" | "finish"
    model: str           # the roster name to load for it
    kind: str            # "chat" | "canvas" | ""
    what: str            # one line for the log


@dataclass(frozen=True)
class Roles:
    order: str
    first: str            # the roster name that writes
    second: str           # the roster name that repairs after the first swap
    first_kind: str = ""
    second_kind: str = ""
    why: str = ""

    @property
    def swaps(self) -> bool:
        return self.first != self.second

    def phases(self) -> list[Phase]:
        if self.order == "diffusion-first":
            return [
                Phase(1, "write", self.first, self.first_kind,
                      "the canvas model writes every pinned body as a hole "
                      "and every other file whole"),
                Phase(2, "repair", self.second, self.second_kind,
                      "the chat model judges each disagreement by the "
                      "request and repairs what still fails, whole files, "
                      "in dependency order"),
                Phase(3, "finish", self.first, self.first_kind,
                      "the canvas model revisits what is left, one hole at "
                      "a time, with fresh seeds"),
            ]
        return [
            Phase(1, "write", self.first, self.first_kind,
                  "the chat model plans, pins the skeleton and writes every "
                  "file and test with the full per-file request"),
            Phase(2, "repair", self.second, self.second_kind,
                  "the canvas model rewrites only the hole each remaining "
                  "failure is in, in dependency order"),
            Phase(3, "finish", self.first, self.first_kind,
                  "the chat model repairs what holes could not fix — whole "
                  "files, fresh-start prompts, temperature up"),
        ]

    def describe(self) -> str:
        if not self.swaps:
            return (f"hybrid: one model only ({self.first}); three passes, "
                    f"no swap — {self.why}")
        return (f"hybrid ({self.order}): {self.first} writes, "
                f"{self.second} repairs, {self.first} finishes — two swaps "
                f"— {self.why}")


def plan_roles(mode: str, roster: Sequence[ModelSlot], loaded: str, *,
               loaded_kind: str = "", writer: str = "", repairer: str = ""
               ) -> tuple[Roles | None, str]:
    """(roles, reason) — roles is None when the build runs as one model.

    ``loaded`` is the model the host reports now; ``loaded_kind`` what it
    is ("canvas" or "chat") when the host knows. ``writer`` and
    ``repairer`` name roster slots to use regardless of kind; "" picks by
    kind: the chat model writes (or the canvas model, for diffusion-first),
    the other kind repairs.
    """
    mode = (mode or "auto").strip().lower()
    if mode == "off":
        return None, "hybrid is off: one model, the ordinary loop"
    if mode not in MODES:
        return None, (f"hybrid={mode!r} is not one of {', '.join(MODES)}; "
                      f"running as one model")
    slots = [s for s in roster if s.name]
    if not slots:
        return None, ("the host offers no roster of models, so there is "
                      "nothing to swap to; one model, the ordinary loop")
    by_name = {s.name: s for s in slots}
    kinds = {s.name: (s.kind or ("canvas" if s.name == loaded
                                 and loaded_kind == "canvas" else
                                 "chat" if s.name == loaded and loaded_kind
                                 else s.kind))
             for s in slots}
    chat = [s.name for s in slots if kinds.get(s.name) == "chat"]
    canvas = [s.name for s in slots if kinds.get(s.name) == "canvas"]
    unknown = [s.name for s in slots if kinds.get(s.name) not in ("chat",
                                                                  "canvas")]
    # a named slot wins; else the loaded model of the right kind; else the
    # first of that kind
    def pick(name: str, want: str, avoid: str = "") -> str:
        if name:
            if name in by_name:
                return name
            return ""
        pool = chat if want == "chat" else canvas
        if loaded in pool and loaded != avoid:
            return loaded
        for n in pool:
            if n != avoid:
                return n
        return ""

    order = mode if mode in ORDERS else ("ar-first" if chat else
                                         "diffusion-first")
    if writer and writer not in by_name:
        return None, (f"hybrid_writer={writer!r} is not a roster model "
                      f"({', '.join(by_name)}); running as one model")
    if repairer and repairer not in by_name:
        return None, (f"hybrid_repairer={repairer!r} is not a roster model "
                      f"({', '.join(by_name)}); running as one model")
    if order == "ar-first":
        first = pick(writer, "chat")
        second = pick(repairer, "canvas", avoid=first)
    else:
        first = pick(writer, "canvas")
        second = pick(repairer, "chat", avoid=first)
    if not first and not second:
        if unknown and mode == "auto":
            return None, ("the roster's models are of unknown kind (the "
                          "host cannot say which fill a canvas before "
                          "loading them); running as one model — name "
                          "hybrid_writer and hybrid_repairer to force it")
        return None, ("no roster model of the kind the order needs; "
                      "running as one model")
    if not second:
        # one kind only: the same model again, with the ladder's fresh
        # starts; still three passes and no swap
        first = first or loaded or slots[0].name
        return (Roles(order=order, first=first, second=first,
                      first_kind=kinds.get(first, ""),
                      second_kind=kinds.get(first, ""),
                      why=(f"the roster has no "
                           f"{'canvas' if order == 'ar-first' else 'chat'} "
                           f"model, so the later phases are the same model "
                           f"with fresh-start prompts")),
                "")
    if not first:
        first = loaded if loaded in by_name else [n for n in by_name
                                                  if n != second][0]
    why = ("the chat model writes whole files of any length from the full "
           "request; the canvas model's holes are fast and start from new "
           "seeds" if order == "ar-first" else
           "the canvas model writes first by request; the chat model judges "
           "and repairs whole files")
    return (Roles(order=order, first=first, second=second,
                  first_kind=kinds.get(first, ""),
                  second_kind=kinds.get(second, ""), why=why), "")
