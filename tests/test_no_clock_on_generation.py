# SPDX-License-Identifier: Apache-2.0
"""Nothing may cut a model off, and a running game is not a broken game.

Bill: *"if there are any time limits on generation they should be removed. I
might at some point use a 70B GGUF Q4_K_M since I can CPU offload heavily, and
accept the lengthy generations. The whole thing is that I don't want to lose
final result quality for speed reasons. And I keep seeing some 'this exceeded
15s' in the metadata."*

Two requests, and they turned out to be about **different clocks**.

THE ONE HE ASKED ABOUT was a 900-second HTTP timeout on the llama-server
provider. Fifteen minutes reads as generous until you price the machine it has
to serve: a 70B Q4_K_M with most layers on the CPU decodes at roughly 1–2
tok/s, so a 4096-token file is 35 to 70 minutes. It would have been cut off
mid-file — and a truncated generation costs the wait AND the result, then hands
the repair loop a syntax error no rewrite can fix. Gone.

THE ONE HE WAS SEEING was `run_timeout=15.0` — a clock on the *program the
model wrote*, not on the model. Which matters more than the mix-up suggests,
because for the racing game it was firing every single time and being recorded
as a failure. A pygame program opens a window and enters `while running:`. It
CANNOT exit. Fifteen seconds later it was killed, marked `ok=False`, and the
timeout text was fed back to the model as a diagnostic — so it spent its
remaining attempts fixing code that was already correct.

Third instance of the same defect in this project, in the opposite direction
from the other two: not success claimed for work not done, but failure claimed
for work done properly.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import langs, runner                      # noqa: E402
from cognitive_coder.ports import LocalFileSystem, SubprocessExec   # noqa: E402
from cognitive_coder.providers.openai_compatible import (      # noqa: E402
    DEFAULT_TIMEOUT, OpenAICompatible)
from cognitive_coder.types import PhaseResult, ProcResult, Timeouts  # noqa: E402


# ==========================================================================
# [1] generation waits as long as it takes
# ==========================================================================

def test_the_local_provider_has_no_ceiling():
    assert DEFAULT_TIMEOUT == 0.0, (
        "a 70B Q4_K_M on CPU can decode for an hour; any ceiling here "
        "truncates a good answer")


def test_zero_reaches_urlopen_as_none_not_as_zero():
    """The trap that makes "no timeout" mean "instant failure".

    `urlopen(timeout=0)` is not unlimited — it is a NON-BLOCKING socket, and
    every request fails at once. The failure then looks like the server being
    down, which sends you debugging llama-server instead of this line.
    """
    p = OpenAICompatible()
    assert p._patience(0) is None
    assert p._patience(0.0) is None
    assert p._patience(None) is None
    assert p._patience(-1) is None
    assert p._patience(30.0) == 30.0


def test_a_remote_call_keeps_its_guard():
    """The one clock that stays, and only because the risk is inverted.

    A local model that runs an hour costs an hour. A hung METERED call bills
    for a socket nobody is reading.
    """
    from cognitive_coder.providers import remote
    assert remote.DEFAULT_TIMEOUT >= 1800.0


def test_no_wall_clock_by_default():
    """F11's budget is opt-in. It has to stay that way."""
    from cognitive_coder.loop import LoopConfig
    from cognitive_coder.session import SessionConfig
    assert LoopConfig().wall_clock_s == 0.0
    assert SessionConfig().wall_clock_s == 0.0
    assert SessionConfig().per_task_s == 0.0


# ==========================================================================
# [2] zero means WAIT, everywhere it can be set
# ==========================================================================

def test_resolve_keeps_a_deliberate_zero():
    """`chosen or default` silently discards 0 — the exact bug this change
    is about, reintroduced by an idiom."""
    assert Timeouts.resolve(None, 60.0) == 60.0
    assert Timeouts.resolve(0, 60.0) == 0.0, "0 is a choice, not an absence"
    assert Timeouts.resolve(0.0, 60.0) == 0.0
    assert Timeouts.resolve(5, 60.0) == 5.0


def test_the_exec_port_waits_when_told_to():
    """A real subprocess, because this is where 0 has to survive."""
    ex = SubprocessExec()
    res = ex.run([sys.executable, "-c",
                  "import time; time.sleep(1.5); print('finished')"],
                 cwd=str(Path.cwd()), timeout=0)
    assert not res.timed_out, "0 must not be read as 'give up immediately'"
    assert "finished" in res.stdout


def test_a_real_timeout_still_kills_the_tree():
    """Removing the ceiling must not remove the guarantee (M16)."""
    ex = SubprocessExec()
    res = ex.run([sys.executable, "-c", "import time; time.sleep(30)"],
                 cwd=str(Path.cwd()), timeout=1.0)
    assert res.timed_out
    assert res.exit_code == -9
    assert "still running after" in res.stderr


def test_the_session_passes_its_clocks_down_to_the_loop():
    src = (Path(__file__).resolve().parent.parent
           / "cognitive_coder" / "session.py").read_text(encoding="utf-8")
    flat = " ".join(src.split())
    assert "timeouts=Timeouts(build=self.config.build_timeout," in flat
    assert "run=self.config.run_timeout," in flat
    assert "test=self.config.test_timeout" in flat


# ==========================================================================
# [3] a program that is SUPPOSED to keep running
# ==========================================================================

_PYGAME_GAME = """\
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    clock.tick(60)
"""


def _timed_out_phase(output: str = "") -> PhaseResult:
    return PhaseResult(
        name="run", argv=("python", "main.py"),
        proc=ProcResult(exit_code=-9, stdout=output, duration_s=15.0,
                        timed_out=True),
        ok=False)


def test_the_racing_game_is_recognised_as_long_running():
    assert runner.has_main_loop(_PYGAME_GAME, "python")


def test_so_are_the_other_things_that_do_not_exit():
    for code, lang in (
            ("import tkinter as tk\nroot = tk.Tk()\nroot.mainloop()", "python"),
            ("from flask import Flask\napp = Flask(__name__)", "python"),
            ("import uvicorn", "python"),
            ("from PySide6.QtWidgets import QApplication", "python"),
            ("const express = require('express')", "javascript"),
            ("http.ListenAndServe(\":8080\", nil)", "go"),
    ):
        assert runner.has_main_loop(code, lang), code[:40]


def test_an_ordinary_script_is_not_excused():
    """The excuse must be earned. A script with no loop that hangs is a
    deadlock, and calling it healthy would hide a real defect."""
    ordinary = "import json\nprint(json.dumps({'a': 1}))\n"
    assert not runner.has_main_loop(ordinary, "python")
    assert not runner._still_running(_timed_out_phase(), ordinary, "python")


def test_a_game_still_playing_when_the_clock_ran_out_passed():
    assert runner._still_running(
        _timed_out_phase("pygame 2.5.2 (SDL 2.28.3)"),
        _PYGAME_GAME, "python")


def test_but_a_game_that_crashed_first_did_not():
    """Two signals, not one — the whole reason this is safe."""
    crashed = _timed_out_phase(
        "Traceback (most recent call last):\n"
        "  File \"main.py\", line 4\n"
        "ModuleNotFoundError: No module named 'pygame'")
    assert not runner._still_running(crashed, _PYGAME_GAME, "python")


def test_a_program_that_exited_cleanly_is_not_touched():
    fine = PhaseResult(name="run", proc=ProcResult(exit_code=0), ok=True)
    assert not runner._still_running(fine, _PYGAME_GAME, "python")


def test_end_to_end_a_main_loop_program_passes_and_says_why(tmp_path):
    """The whole path, with a real process that genuinely never ends.

    A real filesystem and a real subprocess, because the thing under test is
    a verdict about a process that had to actually be launched and killed.
    `pygame` is faked into `sys.modules` so the test needs no game library —
    what matters is the import NAME, which is the marker.
    """
    fs = LocalFileSystem(str(tmp_path))
    forever = ("import sys, types, time\n"
               "sys.modules['pygame'] = types.ModuleType('pygame')\n"
               "import pygame\n"
               "print('window open')\n"
               "while True:\n"
               "    time.sleep(0.05)\n")
    result = runner.build_and_run(
        forever, "python", fs=fs, ex=SubprocessExec(), stem="main",
        workdir=str(tmp_path), path="main.py", skip_guard=True,
        timeout=2.0)

    assert result.ok, "a program doing exactly what it should is not a failure"
    run_phase = [p for p in result.phases if p.name == "run"][-1]
    assert run_phase.ok
    assert "still running" in run_phase.note
    #: And it must not overclaim. Startup was verified; nothing else was.
    joined = " ".join(result.caveats)
    assert "NOT a failure" in joined
    assert "only startup was verified" in joined
    assert "only a test can do that" in joined


def test_the_timeout_message_names_the_right_clock():
    """"exceeded 15s" gave an operator no way to tell whether the limit was
    on the model or on the program, and the two have opposite remedies."""
    phase = runner._phase(
        SubprocessExec(), "run",
        [sys.executable, "-c", "import time; time.sleep(20)"],
        cwd=str(Path.cwd()), timeout=1.0)
    assert not phase.ok
    assert "not on generation" in phase.note
    assert "Setup" in phase.note, "say which knob raises it"


# ==========================================================================
# [4] the defaults suit a project, not a scratch file
# ==========================================================================

def test_the_run_limit_is_no_longer_shorter_than_a_program_starting():
    py = langs.get("python")
    assert py.run_timeout >= 60.0
    assert py.build_timeout >= 300.0
    assert py.test_timeout >= 900.0, (
        "a real suite takes minutes; 120s failed suites for being thorough")


def test_every_language_gets_at_least_that():
    for lang_id in langs.ids():
        lang = langs.get(lang_id)
        assert lang.run_timeout >= 60.0, lang_id
        assert lang.test_timeout >= 900.0, lang_id
