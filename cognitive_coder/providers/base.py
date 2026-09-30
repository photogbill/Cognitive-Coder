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

from collections.abc import Iterable, Iterator, Sequence
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
    spans = _balanced_objects(repaired)
    if spans:
        repaired = repaired[spans[0][0]:spans[0][1]]

    # Comments and trailing commas, OUTSIDE strings only, then single-quoted
    # keys/values. These were two regexes over the whole text, so
    # `{"old": "xs = [1, 2,]"}` came back with `old == "xs = [1, 2]"` — a
    # silently corrupted `apply_patch` argument that then matched nothing.
    repaired = _strip_outside_strings(repaired)
    for attempt in (repaired, _single_to_double(repaired)):
        try:
            value = json.loads(attempt)
            if isinstance(value, dict):
                return value, True
        except (ValueError, TypeError):
            continue
    return {}, False


def _balanced_objects(text: str) -> list[tuple[int, int]]:
    """(start, end) of every top-level balanced `{…}`, string-aware.

    Braces inside double-quoted strings do not count, and neither does an
    escaped quote, so `{"pattern": "a { b }"}` is one object.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            # Only inside an object: a quote in the prose around it (an
            # apostrophe-like stray, an unmatched quotation) is not JSON.
            in_str = depth > 0
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
    return spans


def _strip_outside_strings(text: str) -> str:
    """Drop `//` and `/* */` comments and trailing commas — outside strings.

    Walks the text once, copying every string literal (double- OR single-
    quoted, escapes honoured) verbatim, so nothing a model put inside a
    string value is ever rewritten.
    """
    out: list[str] = []
    i, n = 0, len(text)
    quote = ""
    esc = False
    while i < n:
        ch = text[i]
        if quote:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def find_json_object(text: str, keys: Sequence[str] = ()) -> dict:
    """The model's ANSWER object from a reply, or `{}`.

    Not the first balanced object: a reasoning model restates the schema it
    was shown before answering, so the first object is the example — the
    review pass took it, and reported findings titled "one line" with
    severity "high|medium|low" while the real finding was dropped. So every
    top-level balanced object is considered (string-aware), each is parsed
    with the same repair as `repair_json`, and the winner is the LAST one
    that carries any of `keys` — or is `{}`, which is a complete answer
    ("nothing") and must not lose to the echoed schema before it. Objects
    with none of the keys (a trailing `{"note": …}`) win only when nothing
    better is present.
    """
    raw = text or ""
    best: dict = {}
    best_rank = (-1, -1)
    for index, (a, b) in enumerate(_balanced_objects(raw)):
        value, _ = repair_json(raw[a:b])
        empty = bool(re.fullmatch(r"\{\s*\}", raw[a:b]))
        if not value and not empty:
            continue                                     # not JSON at all
        fits = empty or not keys or any(k in value for k in keys)
        rank = (int(fits), index)
        if rank >= best_rank:
            best, best_rank = value, rank
    return best


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
                url = f"data:{media};base64,{b64}"
                parts.append({"type": "image_url",
                              "image_url": {"url": url}})
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


def first_choice(data: Any) -> dict:
    """`data["choices"][0]` when every step is the shape it should be; else
    `{}`.

    Written out because `(data.get("choices") or [{}])[0]` assumed a dict
    at each step, and servers do not: bodies of `[]`, `null`,
    `{"choices": [null]}` and `{"choices": [{"message": "hi"}]}` each raised
    AttributeError straight out of `complete()` — M11's neighbour, an
    exception where a Completion was promised.
    """
    if not isinstance(data, dict):
        return {}
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    return choice if isinstance(choice, dict) else {}


#: How much of a server's own error message reaches a sentence.
ERROR_CHARS = 200


def error_detail(raw: Any) -> str:
    """A server's error message, redacted and cut to `ERROR_CHARS`, or "".

    Only a MESSAGE field of a JSON error body is quoted — `error.message`,
    `error` as a string, or a top-level `message`. Anything else (an HTML
    page from a proxy, a body that echoes the request) is not quoted at all,
    because an error body can carry the request and the request can carry
    file contents. What is quoted is scrubbed like any outbound text first.
    """
    from .. import redact

    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            return ""
    msg: Any = ""
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or ""
        elif isinstance(err, str):
            msg = err
        if not msg and isinstance(data.get("message"), str):
            msg = data["message"]
    msg = " ".join(str(msg or "").split())
    if not msg:
        return ""
    clean, _report = redact.redact_text(msg)
    if len(clean) > ERROR_CHARS:
        clean = clean[:ERROR_CHARS].rstrip() + "…"
    return clean


def http_error_detail(exc: Any) -> str:
    """`error_detail` of an HTTPError's body, read with a bound."""
    try:
        return error_detail(exc.read(16384))
    except Exception:                                    # noqa: BLE001
        return ""


def as_dict(value: Any) -> dict:
    """`value` if it is a dict, else `{}` — for one more `.get()` down."""
    return value if isinstance(value, dict) else {}


def sse_data(lines: Iterable[bytes]) -> Iterator[str]:
    """Each server-sent event's `data`, assembled per the SSE spec.

    The old reader parsed every `data:` LINE as JSON on its own. The spec
    says an event is the `data:` lines up to a blank line, joined with a
    newline — and a server that splits one JSON object over two `data:`
    lines is legal, so its chunk was silently dropped. Lines are decoded
    whole: `\\n` never occurs inside a UTF-8 sequence, so a character that
    arrived split across two TCP writes is reassembled by the line reader
    before it is decoded.

    One leniency, for servers that omit the blank line between events: a
    new `data:` line after a buffer that is ALREADY a complete JSON value
    dispatches the buffer first. Nothing legal is lost by that, because a
    complete JSON value followed by more data would not parse either.
    """
    buf: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if buf:
                yield "\n".join(buf)
                buf = []
            continue
        if line.startswith(":"):
            continue                                     # a comment
        name, _, value = line.partition(":")
        if name != "data":
            continue                                     # event/id/retry
        if value.startswith(" "):
            value = value[1:]
        if buf and _is_json("\n".join(buf)):
            yield "\n".join(buf)
            buf = []
        buf.append(value)
    if buf:
        yield "\n".join(buf)


def _is_json(text: str) -> bool:
    if text.strip() == "[DONE]":
        return True
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


def _as_ms(value: Any) -> int | None:
    """A server-reported duration in ms, or None if it did not report one.

    Kept at least 1 when reported: a fully cached prefix really does take
    0.6 ms, and `int(0.6)` is 0, which the journal reads as "no timing" and
    which the old code then replaced with the whole call's wall time — the
    exact opposite of what the cache had done.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return max(1, round(value))


class StreamAccumulator:
    """OpenAI-shaped stream chunks → one `Completion`, with honest timings.

    Streaming is how prefill and decode become separable without the
    server's help: the time to the FIRST generated token is prompt
    processing plus one token, and everything after it is decode. Measured
    that way, a prefix-cache miss shows up as a jump in `prompt_ms` instead
    of vanishing into a whole-call figure that tracks output length.
    llama.cpp's own `timings` (in the last chunk) override the clock when
    present, because they are exact.
    """

    def __init__(self, t0: float, model: str = "") -> None:
        self.t0 = t0
        self.model = model
        self.parts: list[str] = []
        self.calls: dict[int, dict[str, str]] = {}
        self.finish = ""
        self.usage: dict = {}
        self.timings: dict = {}
        self.error: Any = None
        self.first: float | None = None
        self.events = 0

    def feed(self, chunk: Any, now: float) -> None:
        if not isinstance(chunk, dict):
            return
        self.events += 1
        if chunk.get("error"):
            self.error = chunk["error"]
        if chunk.get("model"):
            self.model = str(chunk["model"])
        if isinstance(chunk.get("usage"), dict):
            self.usage = chunk["usage"]
        if isinstance(chunk.get("timings"), dict):
            self.timings = chunk["timings"]
        choice = first_choice(chunk)
        delta = as_dict(choice.get("delta")) or as_dict(choice.get("message"))
        produced = False
        content = delta.get("content")
        if isinstance(content, str) and content:
            self.parts.append(content)
            produced = True
        # A reasoning model's thinking is generation too: it is decode time,
        # not prefill, even though it is not part of the answer text.
        if delta.get("reasoning_content") or delta.get("reasoning"):
            produced = True
        for i, raw in enumerate(delta.get("tool_calls") or ()):
            if not isinstance(raw, dict):
                continue
            index = raw.get("index", i)
            slot = self.calls.setdefault(
                index if isinstance(index, int) else i,
                {"id": "", "name": "", "arguments": ""})
            if raw.get("id") and not slot["id"]:
                slot["id"] = str(raw["id"])
            fn = as_dict(raw.get("function"))
            # The name arrives once; arguments arrive in pieces. Appending
            # the name as well doubles it on servers that repeat it.
            if fn.get("name") and not slot["name"]:
                slot["name"] = str(fn["name"])
            args = fn.get("arguments")
            if isinstance(args, dict):
                slot["arguments"] = json.dumps(args)
            elif args:
                slot["arguments"] += str(args)
            produced = True
        if produced and self.first is None:
            self.first = now
        if choice.get("finish_reason"):
            self.finish = str(choice["finish_reason"])

    def completion(self, now: float) -> Completion:
        first = self.first if self.first is not None else now
        prompt_ms = _as_ms(self.timings.get("prompt_ms"))
        decode_ms = _as_ms(self.timings.get("predicted_ms"))
        if prompt_ms is None:
            prompt_ms = max(1, int((first - self.t0) * 1000))
        if decode_ms is None:
            decode_ms = max(0, int((now - first) * 1000))
        calls = parse_tool_calls([
            {"id": slot["id"] or f"call_{i}",
             "function": {"name": slot["name"],
                          "arguments": slot["arguments"]}}
            for i, slot in sorted(self.calls.items())])
        finish = map_finish(self.finish, bool(calls))
        return Completion(
            text="".join(self.parts), tool_calls=calls,
            finish_reason=finish,
            tokens_in=_as_int(self.usage.get("prompt_tokens")),
            tokens_out=_as_int(self.usage.get("completion_tokens")),
            model=self.model, prompt_ms=prompt_ms, decode_ms=decode_ms,
            prompt_processed=processed_tokens(self.timings, self.usage))


def processed_tokens(timings: dict, usage: dict) -> int:
    """Prompt tokens the server processed rather than served from cache.

    llama-server says so directly (`timings.prompt_n`); an OpenAI-shaped
    server says how many were CACHED, and the rest were processed. 0 when
    neither is reported — which `Journal.cache_health` reads as "unknown".
    """
    n = _as_int((timings or {}).get("prompt_n"))
    if n > 0:
        return n
    details = (usage or {}).get("prompt_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in details:
        total = _as_int((usage or {}).get("prompt_tokens"))
        return max(1, total - _as_int(details.get("cached_tokens")))
    return 0


def map_finish(reason: str, has_calls: bool) -> str:
    """A server's finish reason → one of ours (FINISH_REASONS)."""
    finish = {"tool_calls": "tool_calls", "function_call": "tool_calls",
              "length": "length", "max_tokens": "length"}.get(
                  str(reason or ""), "stop")
    if has_calls and finish != "length":
        finish = "tool_calls"
    return finish


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


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
        yield self.complete(                              # type: ignore
            messages, **kw).text

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

    @staticmethod
    def _patience(seconds: float | None) -> float | None:
        """Seconds to wait, where 0 (or None) means "as long as it takes".

        `urlopen` wants `None` for no limit. It must never be handed a literal
        `0`, which is not "no timeout" but a NON-BLOCKING socket — every
        request would fail instantly, and the failure would look like the
        server being down rather than like a configuration mistake. (It was:
        `remote.py` passed `timeout=0` straight through.)
        """
        return seconds if seconds and seconds > 0 else None

    def failed(self, sentence: str, *, model: str = "", prompt_ms: int = 0,
               status: int | None = None) -> Completion:
        """An error Completion carrying its sentence — journaled and emitted.

        M11: a provider hands back something the loop can reason about,
        never an exception. The sentence goes to `Completion.error`, the
        journal (as an `error` event) and the EventPort when the provider
        was given them; `text` stays empty so it is never mistaken for code.
        """
        journal = getattr(self, "_journal", None)
        if journal is not None:
            try:
                journal.log("error", sentence=sentence, provider=self.name,
                            status=status)
            except Exception:                            # noqa: BLE001
                pass
        events = getattr(self, "_events", None)
        if events is not None:
            try:
                events.event("error", sentence,
                             {"provider": self.name, "status": status})
            except Exception:                            # noqa: BLE001
                pass
        return Completion(text="", finish_reason="error", model=model,
                          prompt_ms=prompt_ms, error=sentence)


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
