# SPDX-License-Identifier: Apache-2.0
"""What THIS model gets right: a profile per model family, calibrated.

WHY THIS EXISTS, 2026-10-27
---------------------------
`models.py` decides whether a model is for code; `providers.base.
family_for` guesses its chat template. Everything else about a model was
one global default: the Engineer persona's 0.15, the ladder's rungs
(0.15 / 0.45 / 0.70), whole-file replies until a file is 40 lines long,
narrowed edits after. Devstral's own card says 0.15; Qwen3-Coder's says
0.7 with top_p 0.8, top_k 20 and a repetition penalty of 1.05, and the
engine ran it at Devstral's number. The ladder moved only temperature —
`complete()` could not pass `top_p`, `top_k`, `min_p` or `repeat_penalty`
at all, so every rung ran on the binding's unrecorded defaults — while
ATK's `chat_stream` has accepted all four since August. And the edit
format a model gets WELL-FORMED most often (whole file, named
definitions, a FIM hole) differs per model, as Aider's leaderboard
measures, and was one setting for all.

WHAT A PROFILE HOLDS
--------------------
* sampling — the model card's recommended temperature and samplers;
* `edit_format` — "auto" (the loop's rules), "whole", "symbols" (narrowed
  edits for every change), or "fim" (holes first, where the probe says
  the model fills them);
* FIM — the ORDER its tokens take ("psm": prefix, suffix, middle; "spm":
  suffix first, Mistral's) and the token texts, for a host that builds
  the raw prompt itself. Whether the model actually fills a hole is never
  taken from here: `fim.probe` decides that at the model check;
* thinking — "none", "switch" (the model thinks unless told not to; the
  engine turns it off for every call whose output it parses and on, with
  a budget, for a diagnosis), or "always"; and how the switch is thrown
  (`enable_thinking` through the chat template, a `/no_think` text
  switch, or both);
* `source` — "built-in" (the table below, from the model cards),
  "calibrated" (`ccoder bench --calibrate` measured it), or "operator".

HOW A PROFILE IS FOUND
----------------------
By the loaded model's name, as `models.py` does: a calibrated file beside
the engine (`bench/profiles/<stem>.json`) first, then the built-in table,
then a generic profile that changes nothing. A host may hand the session
one outright (`SessionConfig.profile`).

The table is data, not a model of the world: add to it freely, and prefer
a number from the model's own card to a number from memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import inspect
import json
from pathlib import Path
import re
from typing import Any

#: The knobs a profile may set on a call, in the names llama.cpp uses.
SAMPLING_KEYS = ("top_p", "top_k", "min_p", "repeat_penalty")

#: The engine's own base temperature: the Engineer persona's. A profile's
#: temperature replaces it only when the profile says so.
BASE_TEMPERATURE = 0.15


@dataclass(frozen=True)
class ModelProfile:
    family: str = ""
    label: str = ""
    #: The model card's temperature, or None for the persona's own.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    #: auto | whole | symbols | fim
    edit_format: str = "auto"
    #: psm | spm | none | unknown — the token order; usability is probed
    fim: str = "unknown"
    #: (prefix, suffix, middle) token texts; middle may be "" for spm
    fim_tokens: tuple[str, ...] = ()
    #: none | switch | always
    thinking: str = "none"
    #: how to turn thinking off: "template" (enable_thinking=False through
    #: the chat template), "text:<switch>" (a soft switch appended to the
    #: user turn), or "template,text:<switch>" for both
    think_off: str = ""
    source: str = "built-in"
    note: str = ""
    #: What `ccoder bench --calibrate` measured, when it did.
    measured: dict = field(default_factory=dict)

    def sampling(self) -> dict:
        """The non-None samplers, for `complete(sampling=…)`."""
        return {k: getattr(self, k) for k in SAMPLING_KEYS
                if getattr(self, k) is not None}

    @property
    def has_switch(self) -> bool:
        return self.thinking == "switch"

    @property
    def text_switch(self) -> str:
        for part in (self.think_off or "").split(","):
            if part.strip().startswith("text:"):
                return part.strip()[5:]
        return ""

    @property
    def template_switch(self) -> bool:
        return "template" in (self.think_off or "")

    def describe(self) -> str:
        """One line for BUILD_LOG.txt."""
        knobs = []
        if self.temperature is not None:
            knobs.append(f"temperature {self.temperature:g}")
        knobs += [f"{k.replace('_', ' ')} {v:g}"
                  for k, v in self.sampling().items()]
        bits = [f"profile: {self.label or self.family or 'generic'} "
                f"({self.source})"]
        if knobs:
            bits.append(", ".join(knobs))
        bits.append(f"edit format {self.edit_format}")
        if self.fim != "unknown":
            bits.append(f"fim {self.fim}")
        if self.thinking != "none":
            bits.append(f"thinking {self.thinking}"
                        + (f" ({self.think_off})" if self.think_off else ""))
        if self.note:
            bits.append(self.note)
        return " · ".join(bits)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelProfile:
        known = set(cls.__dataclass_fields__)
        kept = {k: v for k, v in dict(data).items() if k in known}
        if "fim_tokens" in kept:
            kept["fim_tokens"] = tuple(str(t) for t in kept["fim_tokens"])
        if "measured" in kept and not isinstance(kept["measured"], dict):
            kept["measured"] = {}
        return cls(**kept)


GENERIC = ModelProfile(family="", label="generic", note="no card-specific "
                       "settings; the engine's defaults")

#: (name markers, profile). First match wins; order matters where one
#: name contains another (`qwen3-coder` before `qwen3`).
_QWEN_FIM = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
BUILTIN: tuple[tuple[tuple[str, ...], ModelProfile], ...] = (
    (("qwen3-coder", "qwen3_coder", "qwen3coder"), ModelProfile(
        family="qwen3-coder", label="Qwen3-Coder",
        temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05,
        fim="psm", fim_tokens=_QWEN_FIM, thinking="none",
        note="the card's sampling; FIM in every version")),
    (("qwen2.5-coder", "qwen2_5-coder", "qwen2.5coder"), ModelProfile(
        family="qwen2.5-coder", label="Qwen2.5-Coder",
        temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.05,
        fim="psm", fim_tokens=_QWEN_FIM, thinking="none")),
    (("qwen3", "qwen-3"), ModelProfile(
        family="qwen3", label="Qwen3",
        temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
        thinking="switch", think_off="template,text:/no_think",
        note="thinks unless told not to; the card's non-thinking sampling")),
    (("devstral",), ModelProfile(
        family="devstral", label="Devstral", temperature=0.15,
        thinking="none",
        note="FIM is documented for Codestral, not Devstral — probed")),
    (("codestral",), ModelProfile(
        family="codestral", label="Codestral", temperature=0.2,
        fim="spm", fim_tokens=("[PREFIX]", "[SUFFIX]", ""),
        thinking="none")),
    (("magistral",), ModelProfile(
        family="magistral", label="Magistral", temperature=0.7, top_p=0.95,
        thinking="always", note="a reasoning model; its thinking is "
                                "stripped before any reply is used")),
    (("mistral-small", "mistral_small", "mixtral", "mistral"), ModelProfile(
        family="mistral", label="Mistral", temperature=0.15,
        thinking="none")),
    (("deepseek-coder", "deepseek_coder", "deepseekcoder"), ModelProfile(
        family="deepseek-coder", label="DeepSeek-Coder", fim="psm",
        fim_tokens=("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>"),
        thinking="none")),
    (("deepseek-r1", "deepseek_r1"), ModelProfile(
        family="deepseek-r1", label="DeepSeek-R1", temperature=0.6,
        top_p=0.95, thinking="always")),
    (("starcoder",), ModelProfile(
        family="starcoder", label="StarCoder", fim="psm",
        fim_tokens=("<fim_prefix>", "<fim_suffix>", "<fim_middle>"))),
    (("codegemma",), ModelProfile(
        family="codegemma", label="CodeGemma", fim="psm",
        fim_tokens=_QWEN_FIM)),
    (("granite-code", "granite_code", "granitecode"), ModelProfile(
        family="granite-code", label="Granite Code", fim="psm",
        fim_tokens=("<fim_prefix>", "<fim_suffix>", "<fim_middle>"))),
    (("codellama", "code-llama", "code_llama"), ModelProfile(
        family="codellama", label="Code Llama", fim="psm",
        fim_tokens=("<PRE>", "<SUF>", "<MID>"),
        note="infilling in the 7B and 13B; not the 34B")),
    (("gemma-3", "gemma3", "gemma_3"), ModelProfile(
        family="gemma3", label="Gemma 3", temperature=1.0, top_p=0.95,
        top_k=64, thinking="none")),
    (("glm-4", "glm4", "glm_4"), ModelProfile(
        family="glm4", label="GLM-4", thinking="switch",
        think_off="template")),
    (("gpt-oss", "gpt_oss"), ModelProfile(
        family="gpt-oss", label="gpt-oss", thinking="always",
        note="reasoning effort is set in its system prompt, not here")),
    (("diffusiongemma", "diffusion-gemma", "diffusion_gemma"), ModelProfile(
        family="diffusiongemma", label="DiffusionGemma", fim="none",
        note="a canvas model: diffusion mode, not FIM")),
)


def _norm(name: str) -> str:
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.sub(r"\.(gguf|bin|safetensors)$", "", base)


def _stem(name: str) -> str:
    """The calibrated file's name for this model: the short name
    `models._short` would print, made file-safe."""
    from .models import _short
    short = _short(name) or _norm(name) or "model"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", short).strip("_") or "model"


def builtin_for(model_name: str) -> ModelProfile:
    """The table's profile for this name, or GENERIC."""
    name = _norm(model_name)
    if not name:
        return GENERIC
    # the LONGEST marker wins, not the first: `DeepSeek-R1-0528-Qwen3-8B`
    # is a DeepSeek-R1 distill (always thinking, its card's 0.6 / 0.95),
    # not a Qwen3 with a switch — "qwen3" is in its name too, earlier in
    # the table (the 2026-10-27 review)
    best: tuple[int, ModelProfile] | None = None
    for markers, profile in BUILTIN:
        hit = max((len(m) for m in markers if m in name), default=0)
        if hit and (best is None or hit > best[0]):
            best = (hit, profile)
    if best is None:
        return GENERIC
    profile = best[1]
    if profile.family == "qwen3":
        # the 2507 split: Instruct never thinks, Thinking always
        if "thinking" in name:
            return replace(profile, thinking="always", think_off="",
                           temperature=0.6, top_p=0.95, min_p=0.0,
                           note="a thinking-only Qwen3; its thinking is "
                                "stripped before use")
        if "instruct" in name:
            return replace(profile, thinking="none", think_off="",
                           note="a non-thinking Qwen3 (2507)")
    return profile


def profiles_dir() -> Path:
    """Where calibrated profiles live: beside the engine, with the bench's
    runs — never under a user profile."""
    return Path(__file__).resolve().parent.parent / "bench" / "profiles"


def calibrated_for(model_name: str, directory: Path | str | None = None
                   ) -> ModelProfile | None:
    """The calibrated profile for this model, or None."""
    base = Path(directory) if directory else profiles_dir()
    path = base / f"{_stem(model_name)}.json"
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return None
    try:
        return replace(ModelProfile.from_dict(data), source="calibrated")
    except Exception:                                    # noqa: BLE001
        return None


def save_calibrated(model_name: str, profile: ModelProfile,
                    directory: Path | str | None = None) -> Path:
    base = Path(directory) if directory else profiles_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{_stem(model_name)}.json"
    path.write_text(replace(profile, source="calibrated").to_json() + "\n",
                    encoding="utf-8")
    return path


def profile_for(model_name: str, *, given: Mapping[str, Any] | ModelProfile
                | None = None, directory: Path | str | None = None
                ) -> ModelProfile:
    """The profile a session runs on: the host's, else the calibrated
    file's, else the table's, else GENERIC — the first that exists."""
    if isinstance(given, ModelProfile):
        return replace(given, source=given.source or "operator")
    # A dict that names NO field of a profile is not a profile — it is a
    # host handing over the wrong settings entry (ATK's `ccoder.profile`
    # is the operator's skill level) — and must not blank the table: a
    # blank "operator" profile would silently cost Devstral its 0.15 and
    # Qwen its FIM tokens. Only a dict that says something is taken.
    if given and any(k in ModelProfile.__dataclass_fields__
                     for k in dict(given)):
        try:
            return replace(ModelProfile.from_dict(given), source="operator")
        except Exception:                                # noqa: BLE001
            pass
    found = calibrated_for(model_name, directory)
    if found is not None:
        return found
    return builtin_for(model_name)


# --------------------------------------------------------------------------
# the ladder, relative to the profile's temperature
# --------------------------------------------------------------------------

#: The engine's rungs above its base: 0.15 → 0.45 → 0.70.
RUNG_STEPS = (0.0, 0.30, 0.55)


def rungs_for(profile: ModelProfile,
              default: Sequence[float] = (0.15, 0.45, 0.70)
              ) -> tuple[float, ...]:
    """The escalation ladder for this model: the configured rungs when the
    profile names no temperature, else the same steps up from its own —
    a model whose card says 0.7 is not sent to 0.15 and back."""
    if profile.temperature is None:
        return tuple(default)
    base = float(profile.temperature)
    return tuple(round(min(1.0, base + step), 2) for step in RUNG_STEPS)


# --------------------------------------------------------------------------
# calling a port with what it accepts
# --------------------------------------------------------------------------

_ACCEPTS_ATTR = "_cc_accepts"


def accepts(port: Any, name: str) -> bool:
    """Does this port's `complete` take keyword ``name`` (or **kw)?

    Cached ON the port object (never by `id()`, which the interpreter
    reuses for a new object the moment an old one is collected). A host
    written to the 1.0 Port, whose `complete` has no `sampling` or
    `thinking` argument, is never sent one — a TypeError from a host is a
    crash the loop cannot act on."""
    cache = getattr(port, _ACCEPTS_ATTR, None)
    if isinstance(cache, dict) and name in cache:
        return cache[name]
    fn = getattr(port, "complete", None)
    ok = False
    try:
        params = inspect.signature(fn).parameters
        ok = name in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        ok = False
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(port, _ACCEPTS_ATTR, cache)
        except Exception:                                # noqa: BLE001
            return ok                      # a port that cannot be written to
    cache[name] = ok
    return ok


def call_kwargs(port: Any, profile: ModelProfile | None, *,
                thinking: bool | None = None,
                sampling: Mapping[str, Any] | None = None) -> dict:
    """The extra keyword arguments for one `complete()` call: the
    profile's samplers (overridden by ``sampling``) and the thinking
    switch — each only when the port takes it."""
    out: dict[str, Any] = {}
    knobs = dict(profile.sampling()) if profile is not None else {}
    if sampling:
        knobs.update({k: v for k, v in sampling.items() if v is not None})
    if knobs and accepts(port, "sampling"):
        out["sampling"] = knobs
    if (thinking is not None and profile is not None and profile.has_switch
            and accepts(port, "thinking")):
        out["thinking"] = bool(thinking)
    return out


def soft_switch(messages: Sequence[Any], profile: ModelProfile | None, *,
                thinking: bool | None) -> list:
    """The messages with the profile's TEXT switch on the last user turn,
    when thinking is being turned off and the profile has one. The
    template switch is the port's business; this one is the prompt's."""
    if thinking is not False or profile is None or not profile.text_switch:
        return list(messages)
    switch = profile.text_switch
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if getattr(m, "role", "") == "user":
            content = m.content or ""
            if content.rstrip().endswith(switch):
                return out
            out[i] = replace(m, content=f"{content.rstrip()}\n{switch}")
            return out
    return out
