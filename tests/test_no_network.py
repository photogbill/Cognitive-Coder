# SPDX-License-Identifier: Apache-2.0
"""C3's enforcement — the test that fails if any socket opens (M51).

This is the most important test in the suite, because C3 is the most
important constraint in the document. One host is an air-gapped,
zero-telemetry tool used where an outbound connection is a safety problem
rather than an inconvenience. A coding module that silently phoned home would
break that promise on its host's behalf.

The mechanism matters, so that the test actually catches something: **every
socket entry point is monkeypatched to raise**, and then a FULL scripted
session is driven — plan, skeleton, generate, verify, patch, journal. A test
that only asserts "we didn't call requests" proves nothing; a test that makes
the operating system's networking unusable and then runs the whole engine
proves the thing C3 claims.
"""

from __future__ import annotations

from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
)
from cognitive_coder.errors import ConfigurationError  # noqa: E402
from cognitive_coder.providers import RemoteGate, make_provider  # noqa: E402


class NetworkWasOpened(AssertionError):
    """Raised the instant anything tries to open a socket."""


@pytest.fixture
def no_network(monkeypatch):
    """Make networking impossible, at every entry point that matters."""
    def forbidden(*args, **kwargs):
        raise NetworkWasOpened(
            "Something tried to open a network connection during a "
            "local-only run. That is C3, and it is not negotiable.")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    if hasattr(socket, "create_server"):
        monkeypatch.setattr(socket, "create_server", forbidden)
    return forbidden


def _session(tmp_path, replies):
    host = Host(
        llm=ScriptedLLM(replies, supports_tools=False),
        fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
        storage=MemoryStorage(str(tmp_path / ".state")),
        events=RecordingEvents(), approval=AutoApprove())
    return host, Session(host, config=SessionConfig(attempts=2,
                                                    skeleton_first=True))


def test_a_full_session_runs_with_networking_disabled(no_network, tmp_path):
    """M51 — the whole engine, start to finish, with sockets poisoned."""
    replies = [
        "src/parser.py — parse a line into fields\n"
        "src/report.py — format the parsed fields\n",
        '```python\ndef parse(line):\n    """Split a line into fields."""\n'
        '    return [p.strip() for p in line.split(",")]\n```',
        '```python\nfrom src.parser import parse\n\n'
        'def report(line):\n    """Format the parsed fields."""\n'
        '    return " | ".join(parse(line))\n```',
    ]
    host, session = _session(tmp_path, replies)
    session.run("parse and report comma-separated lines")

    assert [o.ok for o in session.outcomes] == [True, True]
    assert (tmp_path / "src" / "parser.py").exists()
    # And the journal agrees that nothing left the machine.
    assert session.journal.stats()["remote_calls"] == 0
    assert "no network calls" in session.journal.summary()


def test_the_guard_itself_would_catch_a_socket_in_generated_code():
    """Belt and braces: generated code that opens a socket is refused."""
    from cognitive_coder import guard
    findings = guard.scan("import socket\ns = socket.socket()\n", "python")
    assert guard.blocked(findings)
    assert "network" in guard.blocked(findings)


def test_a_remote_provider_cannot_be_built_without_explicit_enablement():
    """M42 — a key is not consent, and neither is an environment variable."""
    with pytest.raises(ConfigurationError) as exc:
        make_provider("openai_compatible",
                      base_url="https://api.example.com/v1")
    assert "remote mode is off" in str(exc.value).lower()
    assert "nothing was sent" in str(exc.value).lower()


def test_enabling_remote_is_per_provider_and_per_session():
    """Enabling one provider must not enable five."""
    gate = RemoteGate()
    gate.enable("openai_compatible")
    assert gate.allowed("openai_compatible")
    assert not gate.allowed("anthropic")
    gate.disable()
    assert not gate.allowed("openai_compatible")


def test_the_remote_banner_is_produced_whenever_remote_is_active():
    """M42.6 — a persistent indicator, for the host to keep on screen."""
    events = RecordingEvents()
    gate = RemoteGate(events)
    assert gate.banner() == ""
    gate.enable("openai_compatible", reason="the operator asked")
    assert "REMOTE MODE" in gate.banner()
    assert any(kind == "remote" for kind, _m, _d in events.events)


def test_a_declined_remote_approval_sends_nothing():
    """The approval gate is a real gate, not a notification."""
    class Decline:
        def approve_diff(self, summary, diff):
            return True

        def approve_remote(self, provider, bytes_out, estimate):
            return False

    gate = RemoteGate(RecordingEvents(), Decline())
    gate.enable("openai_compatible")
    with pytest.raises(ConfigurationError) as exc:
        gate.check("openai_compatible", bytes_out=1024)
    assert "nothing left this machine" in str(exc.value).lower()


def test_a_local_endpoint_is_not_treated_as_remote():
    """C3 is about leaving the MACHINE, not about using a socket.

    `http://127.0.0.1:8080` is a llama.cpp server on the same box. Treating
    it as remote would make the warning meaningless through overuse, which is
    how a safety indicator stops being read.
    """
    from cognitive_coder.providers import is_local_url
    assert is_local_url("http://127.0.0.1:8080")
    assert is_local_url("http://localhost:11434")
    assert is_local_url("http://192.168.1.40:8080")
    assert is_local_url("http://10.0.0.5:1234")
    assert not is_local_url("https://api.anthropic.com")


def test_proxy_variables_are_scrubbed_from_the_child_environment():
    """A proxy variable is a network path, so C3 has no exception for it."""
    from cognitive_coder.runner import scrubbed_env
    env = scrubbed_env("/tmp/x")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "ALL_PROXY"):
        assert env.get(key, "") == "", f"{key} leaked into the child"
    assert env.get("CARGO_NET_OFFLINE") == "true"
    assert env.get("npm_config_offline") == "true"


def test_no_api_key_shaped_variable_reaches_a_child_process(monkeypatch):
    """M44's neighbour: keys never reach a build of generated code."""
    from cognitive_coder.runner import scrubbed_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak-either")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_nope")
    env = scrubbed_env("/tmp/x")
    leaked = [k for k, v in env.items() if "should-not-leak" in str(v)
              or "ghp_nope" in str(v)]
    assert not leaked, leaked


def test_a_full_session_never_touches_the_remote_gate(no_network, tmp_path):
    """Phase 8 landed; C3 must be exactly as true as it was before.

    A session that was never told to go remote must not construct a remote
    provider, must not raise the banner, and must report nothing sent — with
    every socket poisoned to prove it.
    """
    replies = ["src/one.py — a thing\n",
               '```python\ndef one():\n    """One."""\n    return 1\n```',
               '{"security": [], "performance": [], "overall": "fine"}']
    host, session = _session(tmp_path, replies)
    session.run("one small module")

    assert not session.gate.active
    assert session.gate.banner() == ""
    assert session.gate.bytes_out == 0
    assert session.journal.stats()["remote_calls"] == 0
    assert not [k for k, _m, _d in host.events.events if k == "remote"]


def test_importing_the_remote_providers_opens_no_socket(no_network):
    """Import-time side effects are how a careful design leaks anyway."""
    import importlib

    from cognitive_coder.providers import remote
    importlib.reload(remote)
    assert remote.REMOTE_PROVIDERS


def test_redaction_needs_no_network_of_its_own(no_network):
    from cognitive_coder import redact
    clean, report = redact.redact_text('K = "AKIAIOSFODNN7EXAMPLE"')
    assert "AKIA" not in clean and report.total == 1
