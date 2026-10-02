# SPDX-License-Identifier: Apache-2.0
"""Packages a request names are known before any code is written — and a
host that can install them is asked to, before the first file.

Oct 1, 2026: "using pygame" on the spec's first line; the install question
at minute 13; thirteen more minutes until someone answered it.
"""

from __future__ import annotations

from cognitive_coder import (
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
from cognitive_coder import packages

SPEC = ("Build a retro, pseudo-3D racing game using pygame. The road is "
        "rendered scanline by scanline like classic 16-bit arcade racers. "
        "Use json for the high-score file. Click the Start button to begin. "
        "Tests use unittest and MUST NOT open a window.")


def test_the_request_s_packages_are_read_from_its_text():
    assert packages.packages_in(SPEC) == [("pygame", "pygame")]
    # ordinary words that are also package names are not read from prose
    assert packages.packages_in("click the rich arcade-style button") == []
    # the standard library is never a package to install
    assert packages.packages_in("use json, math and sqlite3") == []
    # import name and pip name both count, once, in first-mention order
    assert packages.packages_in("sprites with Pillow, maths with numpy, "
                                "pytest tests") == [
        ("PIL", "pillow"), ("numpy", "numpy"), ("pytest", "pytest")]
    assert packages.packages_in("a pygame-based racer") == [
        ("pygame", "pygame")]


def test_known_and_mentioned_decide_what_needs_no_question():
    assert packages.is_known("pygame")
    assert packages.is_known("scikit-learn") and packages.is_known("sklearn")
    assert not packages.is_known("reqeusts")           # the typo case
    assert not packages.is_known("")
    assert packages.mentioned_in(SPEC, "pygame")
    assert packages.mentioned_in("needs Pillow for sprites", "PIL")
    assert not packages.mentioned_in(SPEC, "numpy")
    assert packages.pip_name("cv2") == "opencv-python"
    assert packages.pip_name("unlisted") == "unlisted"


class _InstallingExec(SubprocessExec):
    """A host that can install: records what it was asked to ensure."""

    def __init__(self) -> None:
        super().__init__()
        self.ensured: list[list[tuple[str, str]]] = []

    def ensure_packages(self, wanted):
        self.ensured.append(list(wanted))
        return f"ensured {', '.join(p for _i, p in wanted)}"


def _host(tmp_path, replies, ex):
    return Host(llm=ScriptedLLM(replies, supports_tools=False,
                                name="Devstral-Small-2-24B",
                                context_tokens=16384),
                fs=LocalFileSystem(str(tmp_path)), exec=ex,
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


PLAN = "src/alpha.py — the first thing\n"
ALPHA = '```python\ndef alpha():\n    """First."""\n    return 1\n```'


def test_a_host_that_can_install_is_asked_before_the_first_file(tmp_path):
    ex = _InstallingExec()
    host = _host(tmp_path, [PLAN, ALPHA], ex)
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("a tiny pygame thing with numpy maths")
    assert ex.ensured == [[("pygame", "pygame"), ("numpy", "numpy")]]
    statuses = [m for _k, m, _d in host.events.of("status")]
    assert any("the request names pygame, numpy" in m for m in statuses), \
        statuses
    # it ran BEFORE any file was generated: only the plan was asked for
    assert len(host.llm.prompts) == 1


def test_a_host_without_an_installer_is_left_alone(tmp_path):
    host = _host(tmp_path, [PLAN, ALPHA], SubprocessExec())
    session = Session(host, config=SessionConfig(attempts=1))
    session.start("a tiny pygame thing")
    assert not any("the request names" in m
                   for _k, m, _d in host.events.of("status"))


def test_an_installer_that_fails_does_not_stop_the_build(tmp_path):
    class _Broken(_InstallingExec):
        def ensure_packages(self, wanted):
            raise OSError("no network")
    host = _host(tmp_path, [PLAN, ALPHA], _Broken())
    session = Session(host, config=SessionConfig(attempts=1))
    outcomes = session.run("a tiny pygame thing")
    assert outcomes and outcomes[0].ok
    warnings = [m for _k, m, _d in host.events.of("warning")]
    assert any("could not pre-install pygame" in m for m in warnings)
