# SPDX-License-Identifier: Apache-2.0
"""Is the loaded model one that can build software? Say so before building.

The engine uses whatever model the host has loaded. On Aug 8, 2026 that was
`Slimaki-Tavern-24B` — a roleplay merge — because it happened to be loaded
for something else, and the build log recorded the name in a header nobody
read. Nineteen minutes later every file had failed. A tool meant for people
who cannot read code cannot rely on them reading a header; the check belongs
in the engine, and its result in a sentence.

Three verdicts, by name alone (a name is all a GGUF reliably offers):

  * ``coding``  — a known coding family, or a name that says so.
  * ``not_coding`` — a name that says what it is for, and it is not code:
    roleplay, companion, fiction, "uncensored" merges. Refused by default.
  * ``unknown`` — a general chat model or a name that says nothing. Built
    with, under a warning: general models of 14B+ do write code, worse.

The lists are data, not a model of the world: add to them freely.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

#: Substrings of a lower-cased model name that mean "trained for code".
CODING_MARKERS: tuple[str, ...] = (
    "devstral", "codestral", "coder", "code-", "-code", "_code", "codellama",
    "codegemma", "starcoder", "deepseek-coder", "granite-code", "wizardcoder",
    "magicoder", "opencoder", "stable-code", "yi-coder", "seed-coder",
    "codeqwen", "qwen2.5-coder", "qwen3-coder", "glm-4-code", "codegeex",
    "replit-code", "santacoder", "phi-4", "nxcode", "autocoder",
)

#: Substrings that mean "trained for something else entirely".
NOT_CODING_MARKERS: tuple[str, ...] = (
    "tavern", "roleplay", "role-play", "rp-", "-rp", "_rp", "erp", "nsfw",
    "uncensored", "waifu", "companion", "girlfriend", "boyfriend", "storywriter",
    "story-writer", "novelist", "fiction", "lewd", "spicy", "chronos",
    "mythomax", "pygmalion", "slimaki", "magnum", "lumimaid", "euryale",
    "stheno", "midnight-miqu", "cydonia", "rocinante", "nemomix",
    "celeste", "dolphin", "hermes-roleplay",
)

#: Below this many tokens of context the engine cannot show a model the
#: interfaces it needs AND leave room for an answer. 16k is the floor the
#: Build Specification's budget maths assumes; 32k is comfortable.
MIN_CONTEXT_TOKENS = 16384


@dataclass(frozen=True)
class ModelVerdict:
    verdict: str          # coding | not_coding | unknown
    reason: str           # one plain sentence
    context_ok: bool
    context_note: str     # "" when fine

    @property
    def refuse(self) -> bool:
        return self.verdict == "not_coding"

    @property
    def warn(self) -> bool:
        return self.verdict == "unknown" or not self.context_ok


def classify(model_name: str) -> str:
    """``coding`` | ``not_coding`` | ``unknown``, by name."""
    name = _norm(model_name)
    if not name:
        return "unknown"
    if any(m in name for m in NOT_CODING_MARKERS):
        # A coding marker inside a roleplay merge's name does not redeem it:
        # "X-coder-tavern" is a tavern model.
        return "not_coding"
    if any(m in name for m in CODING_MARKERS):
        return "coding"
    return "unknown"


def judge(model_name: str, context_tokens: int = 0) -> ModelVerdict:
    """The verdict on the loaded model, with the sentences a host shows."""
    kind = classify(model_name)
    short = _short(model_name) or "the loaded model"
    if kind == "coding":
        reason = f"{short} is a coding model."
    elif kind == "not_coding":
        reason = (f"{short} is not a coding model — by its name it was made "
                  f"for conversation or fiction. Builds with it mostly fail; "
                  f"load a coding model (Devstral, Codestral, Qwen-Coder, "
                  f"DeepSeek-Coder) or allow it on purpose with "
                  f"`allow_any_model`.")
    else:
        reason = (f"{short} is not a model this engine recognises as a "
                  f"coding model. A general model can write code, usually "
                  f"worse; expect more attempts.")
    ctx_ok = context_tokens <= 0 or context_tokens >= MIN_CONTEXT_TOKENS
    note = "" if ctx_ok else (
        f"the model is running with {context_tokens:,} tokens of context; "
        f"builds need at least {MIN_CONTEXT_TOKENS:,} to show a file the "
        f"interfaces it depends on. Raise the server's context size.")
    return ModelVerdict(verdict=kind, reason=reason, context_ok=ctx_ok,
                        context_note=note)


def _norm(name: str) -> str:
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return re.sub(r"\.(gguf|bin|safetensors)$", "", base)


def _short(name: str) -> str:
    """`mistralai_Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf` →
    `Devstral-Small-2-24B-Instruct-2512`: the part a person recognises."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"\.(gguf|bin|safetensors)$", "", base, flags=re.I)
    base = re.sub(r"[-_.]?(Q\d(_[A-Z0-9]+)*|IQ\d\w*|F16|BF16|fp16|int8|"
                  r"4bit|8bit)$", "", base, flags=re.I)
    if "_" in base and base.split("_", 1)[0].islower():
        base = base.split("_", 1)[1]          # drop the `mistralai_` prefix
    return base
