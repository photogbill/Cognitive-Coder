# SPDX-License-Identifier: Apache-2.0
"""What every provider shares: JSON repair, message shaping, honest counting.

A provider is just an `LLMPort` implementation that this package ships rather
than the host. Everything here is the part that would otherwise be written
five times, slightly differently, with the bug fixed in only three of them.

**The JSON repair parser is the piece that matters** (D9). Small models emit
JSON that is nearly JSON: trailing commas, single quotes, a sentence of prose
before the object, `//` comments, a fence around it. The repair is worth
doing — but the repair is also a *signal*, so `ToolCall.repaired` is set every
time one is needed. A model that needs its arguments fixed on every call is
telling you something (usually that grammar-constrained decoding is available
and switched off), and silently patching over it destroys the message.

Grammar-constrained decoding, where the provider supports it, is strictly
better than repair and is used first (§4.2, ATK already does this with GBNF).
Repair is the fallback, not the plan.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import json
import re
from typing import Any

from ..types import Completion, Message, ModelCapabilities, ToolCall, ToolSpec

# A prose preamble before the object is the most common malformation, and the
# most harmless to strip. `{` … `}` balance-matching beats a regex here
# because JSON nests.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def repair_json(text: str) -> tuple[dict, bool]:
    """Near-JSON → a dict, and whether repair was needed (D9).

    Returns ``({}, False)`` for input that was never going to be JSON, so a
    caller can tell "the model didn't answer with arguments" apart from "the
    model answered badly and we fixed it". Those are different problems.
    """
    raw = (text or "").strip()
    if not raw:
        return {}, False
    try:
        value = json.loads(raw)
        return (value, False) if isinstance(value, dict) else ({}, False)
    except (ValueError, TypeError):
        pass

    repaired = raw
    m = _FENCE.search(repaired)
    if m:
        repaired = m.group(1).strip()

    # Take the outermost balanced object, discarding prose either side.
    start = repaired.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(repaired[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    repaired = repaired[start:i + 1]
                    break

    # Line comments, then trailing commas, then single-quoted keys/values.
    repaired = re.sub(r"//[^\n\"]*$", "", repaired, flags=re.M)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    for attempt in (repaired, _single_to_double(repaired)):
        try:
            value = json.loads(attempt)
            if isinstance(value, dict):
                return value, True
        except (ValueError, TypeError):
            continue
    return {}, False


def _single_to_double(text: str) -> str:
    """Single-quoted JSON → double-quoted, leaving apostrophes in strings.

    Deliberately conservative: it only rewrites quotes that look like
    delimiters (preceded by `{`, `[`, `,` or `:`). Rewriting every `'` turns
    "it's" into a syntax error, which is a worse outcome than not repairing.
    """
    return re.sub(r"(?<=[\{\[,:\s])'([^'\n]*)'", r'"\1"', text)


def parse_tool_calls(raw_calls: Sequence[dict]) -> tuple[ToolCall, ...]:
    """OpenAI-shaped tool calls → our `ToolCall`s, with arguments PARSED.

    The Port contract says arguments arrive parsed (§5.3), so every provider
    does this rather than every call site.
    """
    out: list[ToolCall] = []
    for i, call in enumerate(raw_calls or ()):
        fn = call.get("function", call) or {}
        args_raw = fn.get("arguments", call.get("arguments", ""))
        repaired = False
        if isinstance(args_raw, dict):
            args = args_raw
        else:
            args, repaired = repair_json(str(args_raw or ""))
        out.append(ToolCall(id=str(call.get("id") or f"call_{i}"),
                            name=str(fn.get("name") or call.get("name") or ""),
                            arguments=args, repaired=repaired))
    return tuple(out)


def messages_to_openai(messages: Sequence[Message]) -> list[dict]:
    """Our messages → the wire shape every OpenAI-compatible endpoint wants.

    Vision content is emitted in the content-parts form; a server that does
    not support it ignores the parts it doesn't know, and `capabilities()`
    told the core not to send images in the first place.
    """
    out: list[dict] = []
    for m in messages:
        row: dict[str, Any] = {"role": m.role}
        if m.images:
            parts: list[dict] = []
            if m.content:
                parts.append({"type": "text", "text": m.content})
            for blob, media in m.images:
                import base64
                b64 = base64.b64encode(blob).decode("ascii")
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{media};base64,{b64}"}})
            row["content"] = parts
        else:
            row["content"] = m.content
        if m.tool_calls:
            row["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name,
                              "arguments": json.dumps(c.arguments)}}
                for c in m.tool_calls]
        if m.tool_call_id:
            row["tool_call_id"] = m.tool_call_id
        out.append(row)
    return out


def estimate_tokens(text: str) -> int:
    """The stated fallback: ~4 characters per token.

    Used only when a provider has no tokenizer. Whenever it is used,
    `token_count_is_estimate` is True and the core declares the assumption in
    the prompt (M14). An undeclared estimate is how a context overflows.
    """
    return max(1, len(text or "") // 4)


class ProviderBase:
    """Shared behaviour. Providers are structural `LLMPort`s, not subclasses.

    This is a mixin of conveniences, not a base class anyone must inherit —
    a host may implement `LLMPort` with nothing from this package at all
    (C2), and the conformance kit is what checks it.
    """

    name = "base"
    is_remote = False

    def stream(self, messages: Sequence[Message], **kw) -> Iterator[str]:
        """A host without streaming may yield one chunk — this is that host.

        Defined rather than absent: a UI that calls `stream()` should get
        text, not an AttributeError, even from a provider that generates in
        one blocking shot.
        """
        yield self.complete(messages, **kw).text          # type: ignore[attr-defined]

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)

    def capabilities(self) -> ModelCapabilities:          # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def tools_payload(tools: Sequence[ToolSpec]) -> list[dict]:
        return [t.to_openai() for t in tools]

    @staticmethod
    def cancelled(model: str = "") -> Completion:
        return Completion(text="", finish_reason="cancelled", model=model)


def family_for(model_name: str) -> str:
    """Guess the chat-template family from a model name.

    Only used when the endpoint does not say. Guessing is honest here because
    the consequence of guessing wrong is a slightly worse prompt template,
    not a wrong answer — and the alternative is asking the operator a question
    they should not have to answer.
    """
    name = (model_name or "").lower()
    for key, family in (("devstral", "mistral"), ("magistral", "mistral"),
                        ("mistral", "mistral"), ("mixtral", "mistral"),
                        ("codestral", "mistral"), ("qwen", "qwen"),
                        ("llama", "llama"), ("deepseek", "deepseek"),
                        ("gemma", "gemma"), ("phi", "phi"),
                        ("granite", "granite"), ("command", "cohere")):
        if key in name:
            return family
    return "unknown"


def supports_fim(model_name: str) -> bool:
    """Whether fill-in-the-middle is worth attempting (G.4).

    FIM is structurally incapable of touching code outside the hole, which is
    the clean answer to D6 — a model "helpfully" rewriting code it was not
    asked to touch. Where the model has it, edits should prefer it.
    """
    name = (model_name or "").lower()
    return any(k in name for k in ("codestral", "devstral", "deepseek-coder",
                                   "qwen2.5-coder", "qwen3-coder",
                                   "starcoder", "codegemma", "codellama"))
