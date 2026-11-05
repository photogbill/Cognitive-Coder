# SPDX-License-Identifier: Apache-2.0
"""Reasoning levels for the engine: how hard the model thinks, per phase.

WHY THIS EXISTS, 2026-11-05 (docs/NEXT_PLAN_20261105.md §2)
----------------------------------------------------------
Bill: *"We should add all of the reasoning options that chat has, for
cognitive coder. Because reasoning may prove useful when trying to design
complex software solutions. But we must also be careful that the thinking
is handled in a way that it doesn't bleed into the actual files."*

Until today the engine had one switch: thinking OFF for every call whose
output it parses (code, JSON, a file list) and ON, with a budget, for the
one prose call (the diagnosis). That was the right first rule — a model
that reasons into a file is a file that does not run — and it stays the
DEFAULT: ``SessionConfig.reasoning = "off"`` means every path is exactly
what it was. What is added:

* **Five levels, the same vocabulary as ATK's chat** — off · on (the
  model's own default depth) · low · medium · high · xhigh. The engine
  never decides HOW a level is said to a model (a template variable, a
  system-prompt line, ``/think``); that is the host's business, read off
  the loaded model's own template, and reaches the core only as
  ``ModelCapabilities.reasoning`` (C1: the core imports no host).
* **A default level per phase**, because the phases are not alike:
  planning and the interface skeleton decide what everything else pays
  for (high); writing a file whose stub is pinned is mostly transcription
  (low, medium when it is being retried); a diagnosis and a Debug & Repair
  pass are exactly the kind of thinking a small model does badly without
  room (high); the review is in between (medium). A grammar-constrained
  ask can never think (the grammar binds from the first token) and is off
  whatever the policy says.
* **The operator overrides any phase** (``SessionConfig.reasoning_phases``),
  or pins one level everywhere (``reasoning = "high"``), or turns it all
  off (the default).
* **The bleed guard** is not here — it is two walls elsewhere: wall 1 in
  `loop.Loop._complete` splits the reasoning off every reply and sends it
  to the journal only; wall 2 in `patcher.py` refuses a hunk that carries a
  thinking delimiter on a line of its own, whatever the model. This module
  only decides the level; the walls do not depend on it being right.

Pure: nothing here talks to a model, a file or a host.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: The operator's vocabulary, in order. "on" is "think, at the model's own
#: default depth"; the four efforts are for models that take a depth.
LEVELS: tuple[str, ...] = ("off", "on", "low", "medium", "high", "xhigh")
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

#: ``SessionConfig.reasoning``: "off" (the 1.0 behaviour — the switch is
#: thrown off for every parsed call, on for the diagnosis), "auto" (the
#: per-phase defaults below), or one level for every phase.
MODES: tuple[str, ...] = ("off", "auto", *LEVELS[1:])

#: The phases a call belongs to. Every `_complete` in the loop and every
#: module that asks the port names one; an unknown name is "generate".
PHASES: tuple[str, ...] = ("plan", "interfaces", "blueprint", "generate",
                           "retry", "continue", "narrow", "diagnose",
                           "repair", "debug", "review", "improve",
                           "bughunt", "redteam", "interview", "structured",
                           "adjudicate", "notes")

#: §2b — the defaults, operator-overridable per phase.
PHASE_DEFAULTS: dict[str, str] = {
    "plan": "high", "interfaces": "high", "blueprint": "high",
    "generate": "low", "retry": "medium", "continue": "off",
    "narrow": "low", "diagnose": "high", "repair": "high", "debug": "high",
    "review": "medium", "improve": "medium", "bughunt": "medium",
    "redteam": "medium", "interview": "medium",
    # a grammar binds from the first token: there is no room to think
    "structured": "off", "adjudicate": "off",
    # the canvas notes are a short prose ask; low is plenty
    "notes": "low",
}

_RANK = {"off": 0, "on": 1, "low": 1, "medium": 2, "high": 3, "xhigh": 4}


def normalise(level: Any) -> str:
    """Anything → one of LEVELS. True is "on", False/None/unknown "off"."""
    if level is True:
        return "on"
    if level is False or level is None:
        return "off"
    value = str(level).strip().lower()
    if value in ("true", "yes", "enabled"):
        return "on"
    return value if value in LEVELS else "off"


def is_on(level: Any) -> bool:
    return normalise(level) != "off"


def label(level: Any) -> str:
    return {"off": "Off", "on": "On (model's default)", "low": "Low",
            "medium": "Medium", "high": "High",
            "xhigh": "XHigh"}.get(normalise(level), "Off")


def clamp(level: str, levels: tuple[str, ...]) -> str:
    """The model's level for an operator level: xhigh is "the model's
    highest"; anything else the highest offered level not above it; ""
    when the model offers nothing at or below it (its default stands)."""
    level = normalise(level)
    if not levels or level in ("off", "on"):
        return level if level == "on" else ""
    offered = [lv for lv in levels if lv in _RANK]
    if not offered:
        return ""
    if level == "xhigh":
        return offered[-1]
    want = _RANK[level]
    below = [lv for lv in offered if _RANK[lv] <= want]
    return below[-1] if below else ""


@dataclass(frozen=True)
class Policy:
    """What level each phase gets, from the operator's settings.

    ``mode``: "off" | "auto" | a level. ``phases``: operator overrides,
    phase → level; an override wins over the mode, so "auto" with
    ``{"generate": "off"}`` is the plan's defaults except that files are
    written without thinking.
    """
    mode: str = "off"
    phases: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> Policy:
        mode = str(getattr(config, "reasoning", "off") or "off").strip().lower()
        if mode not in MODES:
            mode = "off"
        raw = getattr(config, "reasoning_phases", None) or {}
        phases = {}
        try:
            for k, v in dict(raw).items():
                key = str(k).strip().lower()
                if key:
                    phases[key] = normalise(v)
        except Exception:                                # noqa: BLE001
            phases = {}
        return cls(mode=mode, phases=phases)

    @property
    def enabled(self) -> bool:
        return self.mode != "off" or any(is_on(v) for v in self.phases.values())

    def level(self, phase: str, *, attempt: int = 1) -> bool | str:
        """The level for one call: ``False`` when nothing is to change
        (the 1.0 switch-off), else a level name.

        A generation past its first attempt is a RETRY (§2b: "low —
        medium on retry"); the operator's "generate" override, when given,
        is kept for retries too, because an override is an instruction.
        """
        key = str(phase or "generate").strip().lower()
        if key == "generate" and attempt > 1:
            if "generate" in self.phases:
                chosen = self.phases["generate"]
            elif "retry" in self.phases:
                chosen = self.phases["retry"]
            else:
                chosen = self._default_for("retry")
        elif key in self.phases:
            chosen = self.phases[key]
        else:
            chosen = self._default_for(key)
        # the grammar rule is absolute: a constrained ask cannot think
        if key in ("structured", "adjudicate"):
            chosen = "off"
        return chosen if is_on(chosen) else False

    def _default_for(self, phase: str) -> str:
        if self.mode == "off":
            return "off"
        if self.mode == "auto":
            return PHASE_DEFAULTS.get(phase, "low")
        return self.mode                                 # one level everywhere

    def describe(self) -> str:
        """One line for BUILD_LOG.txt."""
        if not self.enabled:
            return "reasoning: off (the model is told not to think into a file)"
        if self.mode == "auto":
            base = "reasoning: auto — plan/interfaces high · files low " \
                   "(medium on retry) · diagnosis/repair high · review medium"
        else:
            base = f"reasoning: {self.mode} for every phase"
        if self.phases:
            over = ", ".join(f"{k} {v}" for k, v in sorted(self.phases.items()))
            base += f"; operator overrides: {over}"
        return base


@dataclass(frozen=True)
class Resolved:
    """What one call actually sends, given what the model can do."""
    asked: str                 # the policy's level
    effective: str             # what reaches the model ("off" = nothing)
    note: str = ""             # one line when they differ, for the journal


def resolve(level: bool | str, support: Any) -> Resolved:
    """Reconcile the policy's level with `ModelCapabilities.reasoning`.

    A model that cannot think gets nothing sent and the journal says so;
    a switch-only model gets on/off; a levelled model gets the clamp; a
    model that always thinks is left alone (its reasoning still arrives
    delimited and is stripped by wall 1).
    """
    asked = normalise(level)
    kind = str(getattr(support, "kind", "none") or "none")
    levels = tuple(getattr(support, "levels", ()) or ())
    if asked == "off":
        return Resolved(asked, "off")
    if kind in ("none", ""):
        return Resolved(asked, "off",
                        f"{asked} asked, but this model has no thinking "
                        f"control the host can drive — nothing sent")
    if kind == "always":
        return Resolved(asked, "on",
                        f"{asked} asked; this model always thinks at its "
                        f"own depth and cannot be set")
    if kind == "switch":
        return Resolved(asked, "on",
                        "" if asked == "on" else
                        f"{asked} asked; this model only switches thinking "
                        f"on or off — on")
    # levels
    got = clamp(asked, levels)
    if not got:
        return Resolved(asked, "on",
                        f"{asked} asked; this model's levels are "
                        f"{'/'.join(levels) or 'unknown'} — its default is "
                        f"in force")
    return Resolved(asked, got,
                    "" if got == asked else
                    f"{asked} → {got} (this model's levels: "
                    f"{'/'.join(levels)})")


def prepare(llm: Any, profile: Any, policy: Any, phase: str,
            messages: Any, *, max_tokens: int, attempt: int = 1,
            thinking: bool | None = False) -> tuple[list, dict, int]:
    """(messages to send, extra kwargs, max_tokens) for one call outside
    the loop — the planner, the tabs — so every caller applies the policy
    the same way the loop does. With no policy, or one that is off, the
    bool ``thinking`` goes through exactly as before."""
    from . import profiles as _profiles  # local: profiles → types
    from .context import reasoning_tax
    level: Any = thinking
    if policy is not None and getattr(policy, "enabled", False):
        chosen = policy.level(phase, attempt=attempt)
        if chosen is not False:
            level = chosen
    extra = _profiles.call_kwargs(llm, profile, thinking=level)
    sent = _profiles.soft_switch(messages, profile, thinking=level)
    budget = int(max_tokens or 0)
    if isinstance(level, str) and budget:
        try:
            caps = llm.capabilities()
            total = int(getattr(caps, "budget_context_tokens", 0)
                        or getattr(caps, "context_tokens", 0) or 0)
        except Exception:                                # noqa: BLE001
            total = 0
        budget += reasoning_tax(level, total)
    return sent, extra, budget


def summarise(reasoning: str, limit: int = 160) -> str:
    """One line of the model's reasoning for progress.md — the first
    sentence that says something, never the whole thing."""
    text = " ".join((reasoning or "").split())
    if not text:
        return ""
    for sep in (". ", "; ", "\n"):
        head, found, _ = text.partition(sep)
        if found and 20 <= len(head) <= limit:
            return head + sep.strip()
    return text[:limit] + ("…" if len(text) > limit else "")
