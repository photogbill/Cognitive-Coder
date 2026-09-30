# SPDX-License-Identifier: Apache-2.0
"""Outbound scrubbing — the last thing that runs before bytes leave (M43).

**Every outbound message passes through here.** The initial payload, every
subsequent turn, and — the one people forget — **every tool-result message in
a tool round-trip.** A file slice returned to a remote model is outbound
context exactly like any other, and a redactor that only covers the first
message covers the least interesting part of the conversation.

WHAT THIS IS AND IS NOT

It is a scrubber for the things that are *recognisable* — 18 secret shapes
(`PATTERNS`, counted by a test so this number stays true): keys with known
shapes, private-key blocks, `.env` secrets, connection strings. It is not a
guarantee that nothing sensitive leaves, and no honest document should say it
is — a variable called `x` holding a patient's name is invisible to every
pattern here. The guarantee this project actually makes is different and
stronger: **nothing leaves at all unless a person turned remote mode on for
this session** (C3). Redaction is the second line, for the case where they
did.

Two design decisions that follow from that:

  * **The redaction COUNT is reported**, per message and per session. A
    number that is climbing is the operator learning that their prompts
    contain more than they thought — which is information they want before
    the fifth call, not after the fiftieth.
  * **Replacements are shaped, not blanked.** `[REDACTED:aws_key]` tells the
    model that something was there and what kind of thing it was, so it can
    reason about the structure without seeing the value. A blank produces
    code that references a variable that now appears not to exist.

And one that is easy to get wrong: **the same secret gets the same
placeholder within one payload** — and different secrets different ones,
across every message of it. If a key appears in three places and becomes
three different tokens, the model sees three unrelated values and writes
code that treats them as different. A caller that wants the numbering to
hold across a whole SESSION passes one `RedactionState` to every call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from .types import Message

# (kind, pattern). Ordered: the most specific first, because a connection
# string contains a password and should be reported as the connection string
# it is rather than as a loose credential.
#
# Where a pattern has a `secret` group, ONLY that group is replaced and only
# that group is judged harmless or not. Both used to apply to the whole
# match: `postgres://user:RealPassw0rd@localhost/db` leaked because the
# match contained `localhost`, and a label like `aws_secret_access_key = `
# vanished along with the value, so the model saw code with a hole in it.
_KEYWORDS = (r"(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?key|"
             r"access[_-]?token|auth[_-]?token|client[_-]?secret|"
             r"private[_-]?key)")
# An unquoted value is only a secret if it looks like one: 8+ characters
# with a letter AND a digit, ending at whitespace or a separator. That keeps
# `password=password`, `token = get_token()` and `request.form[...]` out.
_BARE = r"[^\s'\"#,;()\[\]{}<>]"
PATTERNS: tuple[tuple[str, str], ...] = (
    ("private_key",
     r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
     r".*?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE "
     r"KEY-----"),
    ("connection_string",
     r"\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp|mssql)"
     r"://[^\s'\"<>/:@]*:(?P<secret>[^\s'\"<>]+)@[^\s'\"<>@]+"),
    # AKIA is a long-lived key id; ASIA the temporary (STS) one.
    ("aws_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("aws_secret",
     r"(?i)aws_?secret_?access_?key['\"]?\s*[:=]\s*['\"]?"
     r"(?P<secret>[0-9a-zA-Z/+]{40})"),
    # Anthropic BEFORE OpenAI, and OpenAI's pattern excludes `ant-`:
    # `sk-[A-Za-z0-9_-]{20,}` happily swallows `sk-ant-…`, and whichever
    # runs first wins. The value is scrubbed either way, but the KIND is
    # what the operator reads in the count — being told an Anthropic key is
    # an OpenAI key sends them to revoke the wrong credential.
    ("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai_key", r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}"),
    ("google_key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("github_token",
     r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}"),
    ("slack_token",
     r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b|\bxapp-[A-Za-z0-9-]{10,}"),
    ("stripe_key", r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("basic_auth",
     r"(?i)\bauthorization\s*[:=]\s*['\"]?basic\s+"
     r"(?P<secret>[A-Za-z0-9+/=]{8,})"),
    # A separator or `bearer ` is required: `authorization_header_value_…`
    # is an identifier, and scrubbing it broke the code it was in.
    ("bearer_token",
     r"(?i)(?:\bauthorization\s*[:=]\s*['\"]?(?:bearer\s+)?|\bbearer\s+)"
     r"['\"]?(?P<secret>[A-Za-z0-9._~+/=-]{24,})"),
    ("api_key_header",
     r"(?i)(?<![\w-])(?:x-api-key|x-auth-token|api-key)\s*:\s*['\"]?"
     r"(?P<secret>[A-Za-z0-9._~+/=-]{16,})"),
    ("credential_assignment",
     r"(?i)\b[\w-]*?" + _KEYWORDS + r"[\w-]*['\"]?\s*[:=]\s*"
     r"(?P<q>['\"])(?P<secret>[^'\"\n]{6,})(?P=q)"),
    # `.env` syntax — `NAME=value`, no spaces around `=` — for a NAME that
    # says it holds a secret. The name is kept; only the value goes. It used
    # to match ANY `UPPER_CASE = <8+ chars>` line and replace the whole
    # line, so a settings module lost LOG_FORMAT, DATA_DIR and DATABASE_URL
    # and the model was shown code using variables that did not exist.
    ("env_line",
     r"(?m)^[ \t]*(?:export[ \t]+)?[A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|"
     r"PASSWORD|PASSWD|PWD|PASS|CREDENTIALS?|AUTH|PRIVATE|SALT|DSN|"
     r"SESSION|COOKIE)[A-Z0-9_]*=(?P<q>['\"]?)(?P<secret>[^\s'\"#]{6,})"
     r"(?P=q)[ \t]*(?:#[^\n]*)?$"),
    # The same credential names, UNQUOTED: `db_password=…`, YAML
    # `password: …`, TOML `token = …`.
    ("credential_assignment",
     r"(?i)\b[\w-]*?(?:" + _KEYWORDS[3:-1] + r"|token)[\w-]*[ \t]*[:=][ \t]*"
     r"(?P<secret>(?=" + _BARE + r"*\d)(?=" + _BARE + r"*[A-Za-z])"
     + _BARE + r"{8,})(?=[\s#,;]|$)"),
    ("private_ip", r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))"
                   r"\.\d{1,3}\.\d{1,3}\b"),
)

# Values that look like secrets and are not. Redacting these produces a
# prompt full of `[REDACTED:…]` where the model needed to see the shape, and
# it teaches the operator that the count means nothing. Judged on the SECRET
# alone (see above), never on the text around it.
_HARMLESS = re.compile(
    r"(?i)\b(changeme|example|placeholder|your[-_]?(?:key|token|secret)|"
    r"xxx+|todo|dummy|fake|sample|redacted|none|null|true|false|"
    r"localhost|127\.0\.0\.1|0\.0\.0\.0)\b|\*{3,}|<[^<>\n]+>")

#: Kinds that are advisory rather than secret. On by default because C3's
#: host is air-gapped and a LAN topology is information; a host that does not
#: care can drop them. (`env_line` used to be here; now that it only fires on
#: a name that says "secret", it is a secret, and the journal scrubber —
#: which skips soft kinds — must not skip it.)
SOFT_KINDS = ("private_ip",)

_COMPILED = [(kind, re.compile(pattern, re.S))
             for kind, pattern in PATTERNS]


@dataclass
class RedactionState:
    """Placeholder numbering, carried across calls when a caller wants it.

    Without one, numbering restarts per call, and a key that was
    `[REDACTED:aws_key_id_2]` in one message is `[REDACTED:aws_key_id]` in
    the next — the model then treats one credential as two. Holds the
    secrets themselves in memory, as keys, for as long as the caller keeps
    it; it is never serialised.
    """
    seen: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)


@dataclass
class Redaction:
    """What was removed, by kind. Never holds the value itself."""
    kind: str
    count: int = 0
    placeholder: str = ""


@dataclass
class RedactionReport:
    """The count, per kind, for the event and the journal."""
    redactions: list[Redaction] = field(default_factory=list)
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def total(self) -> int:
        return sum(r.count for r in self.redactions)

    def merge(self, other: RedactionReport) -> None:
        by_kind = {r.kind: r for r in self.redactions}
        for r in other.redactions:
            if r.kind in by_kind:
                by_kind[r.kind].count += r.count
            else:
                self.redactions.append(Redaction(r.kind, r.count,
                                                 r.placeholder))
        self.bytes_before += other.bytes_before
        self.bytes_after += other.bytes_after

    def summary(self) -> str:
        """Counts DISTINCT secrets, not occurrences.

        One key appearing four times is one thing to revoke, not four, and
        the number the operator wants is "how many credentials were in that
        prompt".
        """
        if not self.total:
            return "nothing needed redacting"
        parts = ", ".join(f"{r.count}× {r.kind}"
                          for r in sorted(self.redactions,
                                          key=lambda r: -r.count) if r.count)
        return f"{self.total} secret(s) redacted: {parts}"

    def as_dict(self) -> dict:
        return {"total": self.total,
                "kinds": {r.kind: r.count for r in self.redactions
                          if r.count},
                "bytes_out": self.bytes_after}


def redact_text(text: str, *, soft: bool = True,
                extra: Sequence[tuple[str, str]] = (),
                state: RedactionState | None = None
                ) -> tuple[str, RedactionReport]:
    """Scrub one string. Returns (clean, report).

    ``extra`` is the host's own configured patterns — a project name, an
    internal hostname, a client identifier. The host knows things about what
    is sensitive that no general pattern can. A host pattern may name a
    `secret` group to have only that part replaced.

    ``state`` keeps placeholder numbering stable across calls. Without it
    each call stands alone, as before. The report counts secrets NEW to the
    state, so a session total is a count of distinct secrets.
    """
    report = RedactionReport(bytes_before=len(text.encode("utf-8")))
    if not text:
        report.bytes_after = 0
        return text, report

    out = text
    # The same value gets the same placeholder, so a key appearing three
    # times does not become three apparently-different values the model
    # then treats as unrelated.
    st = state if state is not None else RedactionState()
    new: dict[str, int] = {}

    patterns = list(_COMPILED)
    patterns += [(kind, re.compile(pattern, re.S)) for kind, pattern in extra]

    for kind, pattern in patterns:
        if not soft and kind in SOFT_KINDS:
            continue
        grouped = "secret" in pattern.groupindex

        def replace(match: re.Match, kind: str = kind,
                    grouped: bool = grouped) -> str:
            whole = match.group(0)
            if grouped and match.group("secret") is not None:
                value = match.group("secret")
                lo = match.start("secret") - match.start()
                hi = match.end("secret") - match.start()
            else:
                value, lo, hi = whole, 0, len(whole)
            if _HARMLESS.search(value):
                return whole
            if value not in st.seen:
                index = st.counts.get(kind, 0) + 1
                st.counts[kind] = index
                suffix = "" if index == 1 else f"_{index}"
                st.seen[value] = f"[REDACTED:{kind}{suffix}]"
                new[kind] = new.get(kind, 0) + 1
            return whole[:lo] + st.seen[value] + whole[hi:]

        out = pattern.sub(replace, out)

    report.redactions = [Redaction(kind=kind, count=n,
                                   placeholder=f"[REDACTED:{kind}]")
                         for kind, n in new.items()]
    report.bytes_after = len(out.encode("utf-8"))
    return out, report


def redact_messages(messages: Sequence[Message], *, soft: bool = True,
                    extra: Sequence[tuple[str, str]] = (),
                    state: RedactionState | None = None
                    ) -> tuple[list[Message], RedactionReport]:
    """Scrub EVERY message, of every role (M43).

    Including `role="tool"`. That is the clause people skip, and it is where
    the file contents live: a `read_slice` result handed back to a remote
    model is a chunk of somebody's source going over the wire, and it did not
    pass through the redactor on the way in because it never came from the
    model.

    One numbering for the whole payload (or for ``state``'s whole session):
    each message used to restart it, so two DIFFERENT keys in two messages
    both became `[REDACTED:aws_key_id]`.
    """
    import dataclasses

    out: list[Message] = []
    total = RedactionReport()
    st = state if state is not None else RedactionState()
    for message in messages:
        clean, report = redact_text(message.content, soft=soft, extra=extra,
                                    state=st)
        total.merge(report)
        if message.tool_calls:
            # Arguments travel too — an `apply_patch` call carries source.
            import json
            calls = []
            for call in message.tool_calls:
                raw = json.dumps(call.arguments)
                scrubbed, sub = redact_text(raw, soft=soft, extra=extra,
                                            state=st)
                total.merge(sub)
                try:
                    args = json.loads(scrubbed)
                except (ValueError, TypeError):
                    args = call.arguments
                calls.append(dataclasses.replace(call, arguments=args))
            out.append(dataclasses.replace(message, content=clean,
                                           tool_calls=tuple(calls)))
        else:
            out.append(dataclasses.replace(message, content=clean))
    return out, total


def outbound_bytes(messages: Sequence[Message]) -> int:
    """What is about to leave, for `approve_remote` to show a human.

    "About 40 kilobytes" is a number somebody can react to. "Some context"
    is not, and it is the difference between informed consent and a dialog
    people click through.
    """
    total = 0
    for m in messages:
        total += len(m.content.encode("utf-8"))
        for blob, _media in m.images:
            total += len(blob)
    return total


def describe(messages: Sequence[Message], report: RedactionReport) -> str:
    """One sentence for the approval prompt."""
    size = outbound_bytes(messages)
    unit = f"{size / 1024:.0f} KB" if size >= 1024 else f"{size} bytes"
    files = sum(1 for m in messages if m.role == "tool")
    bits = [f"about {unit} of prompt"]
    if files:
        bits.append(f"{files} file excerpt(s)")
    if report.total:
        bits.append(report.summary())
    else:
        bits.append("nothing matched a secret pattern, which is not the same "
                    "as nothing sensitive being present")
    return "; ".join(bits)


# --------------------------------------------------------------------------
# budgets (M42.4)
# --------------------------------------------------------------------------

@dataclass
class Budget:
    """Token and spend ceilings that STOP rather than warn.

    §6.12 rule 4 is explicit: when exceeded, stop and report — never silently
    continue. A budget that logs a warning and carries on is not a budget, it
    is a diary of the money you spent.
    """
    max_tokens: int = 0          # 0 ⇒ no ceiling
    max_spend: float = 0.0       # in whatever currency the host prices in
    tokens_in: int = 0
    tokens_out: int = 0
    spend: float = 0.0
    calls: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def record(self, *, tokens_in: int = 0, tokens_out: int = 0,
               cost: float = 0.0) -> None:
        self.tokens_in += max(0, tokens_in)
        self.tokens_out += max(0, tokens_out)
        self.spend += max(0.0, cost)
        self.calls += 1

    def exceeded(self) -> str:
        """The reason to stop, or "". Checked BEFORE each call, not after."""
        if self.max_tokens and self.tokens >= self.max_tokens:
            return (f"the session token budget of {self.max_tokens:,} is "
                    f"used up ({self.tokens:,} across {self.calls} call(s))")
        if self.max_spend and self.spend >= self.max_spend:
            return (f"the session spend budget of {self.max_spend:.2f} is "
                    f"used up ({self.spend:.2f} across {self.calls} call(s))")
        return ""

    def remaining(self) -> str:
        bits = []
        if self.max_tokens:
            bits.append(f"{max(0, self.max_tokens - self.tokens):,} tokens")
        if self.max_spend:
            bits.append(f"{max(0.0, self.max_spend - self.spend):.2f} spend")
        return " · ".join(bits) or "no ceiling set"

    def as_dict(self) -> dict:
        return {"tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "spend": round(self.spend, 4), "calls": self.calls,
                "max_tokens": self.max_tokens, "max_spend": self.max_spend}


def scrub_for_log(value: Any) -> Any:
    """Belt and braces for M44: nothing key-shaped reaches a log or journal.

    The journal is written by code that never sees a key, so this should
    never fire. It exists because "should never" and "does never" are
    different, and the cost of being wrong here is a key in a file the
    operator will later paste into a bug report.
    """
    if isinstance(value, str):
        return redact_text(value, soft=False)[0]
    if isinstance(value, dict):
        return {k: scrub_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_for_log(v) for v in value]
    return value
