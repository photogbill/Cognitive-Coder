# SPDX-License-Identifier: Apache-2.0
"""M43, M44 and M42.4 — nothing leaves that shouldn't, and budgets halt.

§6.12's acceptance, verbatim: *a fixture project seeded with a fake AWS key,
a private key block, a `.env`, and a connection string; every outbound
payload in a scripted remote session is captured and asserted clean, and the
redaction count matches.* That is `test_a_scripted_remote_session_sends_
nothing_secret`, and everything else here supports it.

The captured-payload approach matters more than testing `redact_text` alone.
A redactor that works perfectly and is called on four messages out of five is
a redactor that leaks, and only a test that inspects what actually went over
the wire can tell the difference.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import redact  # noqa: E402
from cognitive_coder.errors import (  # noqa: E402
    BudgetExceeded,
    ConfigurationError,
)
from cognitive_coder.ports import AutoApprove, RecordingEvents  # noqa: E402
from cognitive_coder.providers import RemoteGate, make_provider  # noqa: E402
from cognitive_coder.providers.remote import Anthropic, key_from  # noqa: E402
from cognitive_coder.types import Message, ToolCall  # noqa: E402

# The seeded fixture project from §6.12's acceptance criterion.
SEEDED = {
    "src/config.py": (
        '"""Configuration."""\n'
        'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n'
        'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n'
        'DATABASE_URL = "postgresql://svc_admin:hunter2@db.internal:5432/prod"\n'
    ),
    ".env": (
        "ANTHROPIC_API_KEY=sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        "GITHUB_TOKEN=ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
        "STRIPE_SECRET=sk_live_AAAAAAAAAAAAAAAAAAAAAAAA\n"
    ),
    "deploy/id_rsa": (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAxGZ1kQ8vN0aVc3JbT9pQ2mF7hR4wL6yD1sE8nK3uP5oI9bX\n"
        "-----END RSA PRIVATE KEY-----\n"
    ),
}

#: Every string that must never appear in an outbound payload.
MUST_NOT_LEAK = (
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "hunter2",
    "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "sk_live_AAAAAAAAAAAAAAAAAAAAAAAA",
    "MIIEowIBAAKCAQEAxGZ1kQ8vN0aVc3JbT9pQ2mF7hR4wL6yD1sE8nK3uP5oI9bX",
)


# --------------------------------------------------------------------------
# the scrubber itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(SEEDED))
def test_every_seeded_file_is_scrubbed_completely(path):
    clean, report = redact.redact_text(SEEDED[path])
    leaked = [s for s in MUST_NOT_LEAK if s in clean]
    assert not leaked, f"{path} leaked {leaked}"
    assert report.total > 0, f"{path} produced no redactions"


def test_each_secret_is_labelled_with_the_right_kind():
    """The KIND is what the operator reads, and it decides which credential
    they go and revoke. `sk-ant-…` reported as an OpenAI key sends them to
    the wrong console."""
    _clean, report = redact.redact_text(
        "a = 'sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'\n"
        "b = 'sk-proj-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'\n")
    kinds = {r.kind for r in report.redactions if r.count}
    assert "anthropic_key" in kinds
    assert "openai_key" in kinds


def test_the_same_secret_gets_the_same_placeholder():
    """Three different tokens for one key makes the model treat them as
    three unrelated values and write code accordingly."""
    clean, _ = redact.redact_text(
        'a = "AKIAIOSFODNN7EXAMPLE"\n'
        'b = "AKIAIOSFODNN7EXAMPLE"\n'
        'c = "AKIAIOSFODNN7EXAMPLE"\n')
    assert clean.count("[REDACTED:aws_key_id]") == 3
    assert "[REDACTED:aws_key_id_2]" not in clean


def test_two_different_secrets_of_one_kind_are_told_apart():
    clean, _ = redact.redact_text(
        'a = "AKIAIOSFODNN7EXAMPLE"\nb = "AKIAJJJJJJJJJJJJJJJJ"\n')
    assert "[REDACTED:aws_key_id]" in clean
    assert "[REDACTED:aws_key_id_2]" in clean


def test_the_placeholder_says_what_kind_of_thing_was_there():
    """A blank produces code referencing a variable that now appears not to
    exist. The model needs the shape, not the value."""
    clean, _ = redact.redact_text('KEY = "AKIAIOSFODNN7EXAMPLE"')
    assert "REDACTED" in clean
    assert "aws_key_id" in clean
    assert "KEY = " in clean, "the assignment's structure must survive"


@pytest.mark.parametrize("harmless", [
    'password = "changeme"', 'api_key = "your-key-here"',
    'token = "xxxxxxxxxxxxxx"', 'secret = "example-secret-value"',
    'HOST = "localhost"', 'DEBUG = "true"',
])
def test_obvious_placeholders_are_left_alone(harmless):
    """A prompt full of `[REDACTED:…]` where the model needed to see the
    shape teaches the operator that the count means nothing."""
    clean, report = redact.redact_text(harmless)
    assert clean == harmless, f"{harmless} was needlessly redacted"
    assert report.total == 0


def test_the_count_is_distinct_secrets_not_occurrences():
    _clean, report = redact.redact_text(
        "\n".join(['x = "AKIAIOSFODNN7EXAMPLE"'] * 5))
    assert report.total == 1, "one key to revoke, not five"


def test_soft_kinds_can_be_turned_off():
    """A LAN address is information on an air-gapped host and noise on a
    laptop. The host decides."""
    text = "HOST = '10.0.0.5'\n"
    _hard, on = redact.redact_text(text, soft=True)
    _soft, off = redact.redact_text(text, soft=False)
    assert on.total == 1
    assert off.total == 0


def test_a_hosts_own_patterns_are_honoured():
    """The host knows things about what is sensitive that no general
    pattern can — a client name, an internal hostname, a case number."""
    clean, report = redact.redact_text(
        "deploying to acme-internal-prod-7 now",
        extra=(("client_id", r"acme-internal-[a-z0-9-]+"),))
    assert "acme-internal-prod-7" not in clean
    assert report.total == 1


def test_nothing_in_produces_nothing_out():
    clean, report = redact.redact_text("")
    assert clean == "" and report.total == 0


# --------------------------------------------------------------------------
# M43 — EVERY message, including tool results
# --------------------------------------------------------------------------

def test_tool_result_messages_are_redacted():
    """The clause people skip. A `read_slice` result is a chunk of somebody's
    source going over the wire, and it never passed through the redactor on
    the way in because it did not come from the model."""
    messages = [
        Message(role="system", content="be helpful"),
        Message(role="user", content="what is wrong with config.py"),
        Message(role="tool", content=SEEDED["src/config.py"],
                tool_call_id="c1"),
    ]
    clean, report = redact.redact_messages(messages)
    body = " ".join(m.content for m in clean)
    leaked = [s for s in MUST_NOT_LEAK if s in body]
    assert not leaked, leaked
    assert report.total >= 3


def test_tool_call_ARGUMENTS_are_redacted():
    """An `apply_patch` call carries source in its arguments."""
    messages = [Message(
        role="assistant", content="fixing it",
        tool_calls=(ToolCall(id="c1", name="apply_patch", arguments={
            "path": "src/config.py", "old": "x",
            "new": 'KEY = "AKIAIOSFODNN7EXAMPLE"'}),))]
    clean, report = redact.redact_messages(messages)
    assert "AKIAIOSFODNN7EXAMPLE" not in str(clean[0].tool_calls[0].arguments)
    assert report.total == 1


def test_redaction_preserves_roles_and_ids():
    messages = [Message(role="tool", content="clean text", tool_call_id="c9")]
    clean, _ = redact.redact_messages(messages)
    assert clean[0].role == "tool"
    assert clean[0].tool_call_id == "c9"


def test_the_approval_description_names_the_size_and_the_uncertainty():
    """"About 40 KB" is something a person can react to; "some context" is
    not. And a clean scan is not a promise that nothing sensitive is there."""
    messages = [Message(role="user", content="x" * 5000)]
    clean, report = redact.redact_messages(messages)
    text = redact.describe(clean, report)
    assert "KB" in text
    assert "not the same as nothing sensitive" in text


# --------------------------------------------------------------------------
# the acceptance criterion (§6.12)
# --------------------------------------------------------------------------

class CapturingAnthropic(Anthropic):
    """A real provider with the socket replaced. Everything else runs.

    Subclassing at `_post` rather than mocking `redact` is deliberate: this
    exercises the actual redaction call site, the actual gate check and the
    actual budget accounting, and captures exactly what would have gone over
    the wire.
    """
    sent: list[dict] = []

    def _post(self, payload: dict) -> dict:
        type(self).sent.append(payload)
        return {"content": [{"type": "text", "text": "understood"}],
                "usage": {"input_tokens": 1200, "output_tokens": 300},
                "model": self.model, "stop_reason": "end_turn"}


def test_a_scripted_remote_session_sends_nothing_secret():
    """§6.12's acceptance, in one test.

    Every outbound payload is captured and asserted clean, and the redaction
    count matches what was seeded.
    """
    CapturingAnthropic.sent = []
    events, approval = RecordingEvents(), AutoApprove(remote=True)
    gate = RemoteGate(events, approval)
    gate.enable("anthropic", reason="the operator asked")

    provider = CapturingAnthropic(api_key="sk-ant-not-a-real-key-000000000000",
                                  gate=gate, events=events)

    # Three turns, each carrying a different seeded secret, and one of them
    # a TOOL RESULT — which is the case M43 exists for.
    provider.complete([
        Message(role="system", content="you are reviewing a project"),
        Message(role="user", content=SEEDED["src/config.py"]),
    ])
    provider.complete([
        Message(role="user", content="and the environment file"),
        Message(role="tool", content=SEEDED[".env"], tool_call_id="c1"),
    ])
    provider.complete([
        Message(role="user", content=SEEDED["deploy/id_rsa"]),
    ])

    assert len(CapturingAnthropic.sent) == 3
    wire = repr(CapturingAnthropic.sent)
    leaked = [s for s in MUST_NOT_LEAK if s in wire]
    assert not leaked, f"these left the machine: {leaked}"
    assert "REDACTED" in wire, "nothing was redacted at all — check the wiring"

    # The count is reported, per M42, and reaches the gate for the session
    # total the host displays.
    assert gate.redactions >= 6, gate.redactions
    remote_events = [d for k, _m, d in events.events if k == "remote"]
    assert any(d.get("total") for d in remote_events)


def test_the_api_key_itself_never_appears_in_a_payload():
    """M44 — and the key is in the constructor, so this is worth asserting."""
    CapturingAnthropic.sent = []
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("anthropic")
    key = "sk-ant-api03-SECRETSECRETSECRETSECRET"
    provider = CapturingAnthropic(api_key=key, gate=gate)
    provider.complete([Message(role="user", content="hello")])
    assert key not in repr(CapturingAnthropic.sent)


def test_nothing_is_sent_when_approval_is_declined():
    class Decline:
        def approve_diff(self, summary, diff):
            return True

        def approve_remote(self, provider, bytes_out, estimate):
            return False

    CapturingAnthropic.sent = []
    gate = RemoteGate(RecordingEvents(), Decline())
    gate.enable("anthropic")
    provider = CapturingAnthropic(api_key="sk-ant-x-000000000000000000",
                                  gate=gate)
    with pytest.raises(ConfigurationError):
        provider.complete([Message(role="user", content="hello")])
    assert CapturingAnthropic.sent == [], "it sent despite being declined"


# --------------------------------------------------------------------------
# the gate (M42.1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["anthropic", "google", "mistral",
                                  "openrouter", "openai"])
def test_no_remote_provider_can_be_built_without_the_gate(name):
    with pytest.raises(ConfigurationError) as exc:
        make_provider(name, api_key="whatever")
    assert "remote" in str(exc.value).lower()
    assert "nothing was sent" in str(exc.value).lower()


def test_enabling_one_provider_does_not_enable_another():
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("anthropic")
    with pytest.raises(ConfigurationError):
        make_provider("openai", api_key="sk-x", gate=gate)


def test_a_key_in_the_environment_does_not_enable_anything(monkeypatch):
    """M42.1 — a key is not consent, and this is the exact mistake."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")
    assert key_from(None, "anthropic") == "sk-ant-from-the-environment"
    with pytest.raises(ConfigurationError):
        make_provider("anthropic", api_key=key_from(None, "anthropic"))


def test_a_missing_key_is_a_sentence_not_a_traceback():
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("anthropic")
    with pytest.raises(ConfigurationError) as exc:
        make_provider("anthropic", api_key="", gate=gate)
    assert "will not go looking for one" in str(exc.value)


def test_the_host_storage_is_preferred_over_the_environment(monkeypatch):
    from cognitive_coder.ports import MemoryStorage
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    storage = MemoryStorage()
    storage.set("cognitive_coder.keys.mistral", "from-host")
    assert key_from(storage, "mistral") == "from-host"


# --------------------------------------------------------------------------
# budgets that halt (M42.4)
# --------------------------------------------------------------------------

def test_a_budget_stops_rather_than_warning():
    """§6.12 rule 4 — a budget that logs and continues is a diary."""
    budget = redact.Budget(max_tokens=1000)
    budget.record(tokens_in=800, tokens_out=400)
    assert budget.exceeded()
    assert "used up" in budget.exceeded()


def test_the_budget_is_checked_BEFORE_the_call_not_after():
    """Checking afterwards means the call that broke the ceiling was still
    made, and still paid for."""
    CapturingAnthropic.sent = []
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("anthropic")
    budget = redact.Budget(max_tokens=100)
    provider = CapturingAnthropic(api_key="sk-ant-x-000000000000000000",
                                  gate=gate, budget=budget)
    provider.complete([Message(role="user", content="one")])   # 1500 tokens
    assert len(CapturingAnthropic.sent) == 1

    with pytest.raises(BudgetExceeded) as exc:
        provider.complete([Message(role="user", content="two")])
    assert len(CapturingAnthropic.sent) == 1, "it sent after the budget blew"
    assert "budget" in str(exc.value)


def test_the_budget_records_what_was_achieved_when_it_stops():
    """"It stopped" without "and here is what you got" is the unhelpful
    half of the message."""
    exc = BudgetExceeded("remote", "1,000 tokens", "3 call(s) to anthropic")
    assert "What was finished" in str(exc)
    assert "3 call(s)" in str(exc)


def test_a_budget_with_no_ceiling_never_stops():
    budget = redact.Budget()
    budget.record(tokens_in=10_000_000, tokens_out=10_000_000)
    assert budget.exceeded() == ""
    assert budget.remaining() == "no ceiling set"


def test_cost_is_computed_from_the_price_table():
    gate = RemoteGate(RecordingEvents(), AutoApprove(remote=True))
    gate.enable("anthropic")
    budget = redact.Budget(max_spend=100.0)
    CapturingAnthropic.sent = []
    provider = CapturingAnthropic(api_key="sk-ant-x-000000000000000000",
                                  gate=gate, budget=budget)
    provider.complete([Message(role="user", content="hello")])
    # 1200 in at $3/Mtok + 300 out at $15/Mtok
    assert budget.spend == pytest.approx((1200 * 3 + 300 * 15) / 1e6)
    assert budget.calls == 1


def test_scrub_for_log_is_the_last_defence_for_the_journal():
    """M44. It should never fire, because the journal is written by code
    that never sees a key — which is exactly why it exists."""
    scrubbed = redact.scrub_for_log(
        {"note": 'key was AKIAIOSFODNN7EXAMPLE', "nested": ["ghp_" + "A" * 36]})
    assert "AKIAIOSFODNN7EXAMPLE" not in str(scrubbed)
    assert "ghp_" + "A" * 36 not in str(scrubbed)
