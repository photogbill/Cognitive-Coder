# SPDX-License-Identifier: Apache-2.0
"""The provider registry — and the gate that keeps C3 true.

C3 is the single most important constraint in this project: **offline is the
default; the network is an explicit, visible choice.** One host (ATK) is an
air-gapped, zero-telemetry tool used where an outbound connection is a safety
problem rather than an inconvenience. A coding module that silently phoned
home would break that promise on its host's behalf.

So this module is deliberately more than a dictionary of constructors. It is
the place where "may we use the network at all" is decided, and it answers no
unless the answer was set for THIS session, by a person:

  1. **No env-var auto-detection.** Finding `ANTHROPIC_API_KEY` in the
     environment does not turn anything on (M42). A key is not consent.
  2. **`enable_remote(...)` is explicit and per-session.** It takes the
     provider name, so enabling one does not enable five.
  3. **`ApprovalPort.approve_remote` is called before the first remote call**,
     showing what is about to leave the machine.
  4. **A persistent REMOTE indicator** is emitted for the host to display, for
     as long as remote mode is on (M42.6).

**The remote providers now exist** (phase 8), and every one of them comes
through the gate below — which was built first, on purpose, so that adding
one could not accidentally route around it. Each also redacts every outbound
message including tool results (M43) and enforces a budget that HALTS rather
than warns (M42.4). See `remote.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import ConfigurationError
from .base import (  # noqa: F401
    ProviderBase,
    estimate_tokens,
    family_for,
    messages_to_openai,
    parse_tool_calls,
    repair_json,
    supports_fim,
)
from .local_llamacpp import LocalLlamaCpp
from .openai_compatible import OpenAICompatible, detect, is_local_url
from .remote import REMOTE_PROVIDERS, key_from

#: name → (constructor, is_remote, one-line description)
_REGISTRY: dict[str, tuple[Callable[..., Any], bool, str]] = {
    "openai_compatible": (
        OpenAICompatible, False,
        "Any OpenAI-compatible endpoint: llama.cpp server, Ollama, LM "
        "Studio, vLLM, LiteLLM. Local unless the URL says otherwise."),
    "local_llamacpp": (
        LocalLlamaCpp, False,
        "A GGUF loaded in-process through llama-cpp-python. Needs the "
        "[llamacpp] extra."),
}

for _name, (_ctor, _why) in REMOTE_PROVIDERS.items():
    _REGISTRY[_name] = (_ctor, True, _why)

#: Anything the spec calls for that is not in this build. Empty now that
#: phase 8 has landed, and kept because naming an absence beats offering a
#: name that fails at call time.
_NOT_BUILT: dict[str, str] = {}


class RemoteGate:
    """Per-session permission to talk to anything off this machine.

    Owned by the `Session`, one per session, never global and never
    persisted. A gate that survives a restart is a gate that turns itself on
    while nobody is looking.
    """

    def __init__(self, events: Any = None, approval: Any = None) -> None:
        self._enabled: set[str] = set()
        self._approved: set[str] = set()
        self._events = events
        self._approval = approval
        self.bytes_out = 0
        self.redactions = 0

    @property
    def active(self) -> bool:
        return bool(self._enabled)

    def enable(self, provider: str, *, reason: str = "") -> None:
        """Turn on ONE remote provider, for this session, deliberately."""
        self._enabled.add(provider)
        self._emit("remote",
                   f"REMOTE MODE IS ON for {provider} — data will leave this "
                   f"machine{f' ({reason})' if reason else ''}.",
                   {"provider": provider, "enabled": True,
                    "providers": sorted(self._enabled)})

    def disable(self, provider: str = "") -> None:
        if provider:
            self._enabled.discard(provider)
            self._approved.discard(provider)
        else:
            self._enabled.clear()
            self._approved.clear()
        self._emit("remote", "Remote mode is off; everything stays on this "
                             "machine.", {"enabled": False})

    def allowed(self, provider: str) -> bool:
        return provider in self._enabled

    def check(self, provider: str, *, bytes_out: int = 0,
              estimate: str = "") -> None:
        """Raise unless this provider was enabled AND approved (M42).

        Called by every remote provider before its first call. Deliberately
        raising rather than returning False: a caller that forgets to check a
        boolean is the exact failure mode C3 cannot survive.
        """
        if provider not in self._enabled:
            raise ConfigurationError(
                f"{provider} is a remote provider and remote mode is off, so "
                f"nothing was sent. Everything stays on this machine unless "
                f"you turn a remote provider on for this session.")
        if provider not in self._approved:
            ok = True
            if self._approval is not None:
                ok = bool(self._approval.approve_remote(
                    provider, bytes_out,
                    estimate or "the prompt and any file contents it "
                                "includes"))
            if not ok:
                raise ConfigurationError(
                    f"Sending data to {provider} was declined, so nothing "
                    f"left this machine.")
            self._approved.add(provider)
        self.bytes_out += max(0, bytes_out)
        self._emit("remote",
                   f"REMOTE — {self.bytes_out:,} bytes sent to {provider} so "
                   f"far this session.",
                   {"provider": provider, "bytes_out": self.bytes_out,
                    "redactions": self.redactions, "enabled": True})

    def banner(self) -> str:
        """The persistent indicator a host displays while remote mode is on."""
        if not self._enabled:
            return ""
        return (f"REMOTE MODE — data leaves this machine "
                f"({', '.join(sorted(self._enabled))})")

    def _emit(self, kind: str, message: str, data: dict) -> None:
        if self._events is None:
            return
        try:
            self._events.event(kind, message, data)
        except Exception:                                # noqa: BLE001
            pass


def available_providers() -> dict[str, dict]:
    """What can be used, what cannot, and honestly why not (C7)."""
    out: dict[str, dict] = {}
    for name, (_ctor, remote, why) in _REGISTRY.items():
        out[name] = {"built": True, "remote": remote, "description": why}
    for name, why in _NOT_BUILT.items():
        out[name] = {"built": False, "remote": True, "description": why}
    return out


def make_provider(name: str, *, gate: RemoteGate | None = None,
                  **kwargs: Any) -> Any:
    """Construct a provider by name, refusing anything remote without a gate.

    The gate check is here rather than in each provider so that adding a
    provider cannot forget it. That is the whole design: make the safe path
    the only path through the constructor.
    """
    key = (name or "").strip().lower()
    if key in _NOT_BUILT:
        raise ConfigurationError(
            f"The {key} provider is not part of this build: "
            f"{_NOT_BUILT[key]}. Use a local provider, or a llama.cpp / "
            f"Ollama endpoint through openai_compatible.")
    if key not in _REGISTRY:
        raise ConfigurationError(
            f"There is no provider called {name!r}. Available: "
            f"{', '.join(sorted(_REGISTRY))}.")
    ctor, declared_remote, _why = _REGISTRY[key]

    # A provider that is remote BY DESIGN checks the gate in its own
    # constructor, so it has to be handed one. Passing it only where the
    # constructor accepts it keeps the local providers' signatures clean —
    # `OpenAICompatible` has no business knowing what a gate is when it is
    # pointed at 127.0.0.1.
    if declared_remote and "gate" not in kwargs:
        kwargs["gate"] = gate

    try:
        provider = ctor(**kwargs)
    except TypeError as exc:
        raise ConfigurationError(
            f"The {key} provider could not be built with those settings: "
            f"{exc}.") from exc

    # `openai_compatible` is local or remote depending on its URL, so for
    # that one the real answer comes from the instance rather than the
    # registry row. This is the second check for a provider that already
    # checked itself, and it is deliberate: the two paths into remote mode
    # are a declared-remote provider and a local-looking provider pointed
    # somewhere public, and both have to be closed.
    is_remote = bool(getattr(provider, "is_remote", declared_remote))
    if is_remote and (gate is None or not gate.allowed(key)):
        raise ConfigurationError(
            f"{key} would send data off this machine (the endpoint is "
            f"not a local address), and remote mode is off. Nothing was "
            f"sent. Turn on remote mode for this session if that is what "
            f"you want.")
    return provider


def register(name: str, ctor: Callable[..., Any], *, remote: bool,
             description: str) -> None:
    """Add a provider at runtime — for hosts and for phase 8.

    Remote registrations still pass through `make_provider`'s gate; there is
    no registration path that skips it, and that is deliberate.
    """
    _REGISTRY[(name or "").strip().lower()] = (ctor, remote, description)
    _NOT_BUILT.pop((name or "").strip().lower(), None)


__all__ = ["OpenAICompatible", "LocalLlamaCpp", "ProviderBase", "RemoteGate",
           "make_provider", "available_providers", "register", "detect",
           "is_local_url", "repair_json", "parse_tool_calls", "family_for",
           "supports_fim", "estimate_tokens", "messages_to_openai",
           "REMOTE_PROVIDERS", "key_from"]
