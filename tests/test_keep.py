# SPDX-License-Identifier: Apache-2.0
"""What a change to an existing file must keep (keep.py).

The failure this guards is silent: a reply that leaves `draw_hud` out, or
writes `# ... rest unchanged` where a method was, parses, imports and — on
a project with no tests — verifies. Each test here fails on the code
before keep.py, where such a reply was written to disk.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_loop import _host, _loop  # noqa: E402

from cognitive_coder import keep  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.types import Task  # noqa: E402

GAME = '''"""A tiny game."""

SPEED = 3


def move(x):
    return x + SPEED


def draw_hud(score):
    return f"score: {score}"


class Player:
    def __init__(self):
        self.x = 0

    def step(self):
        self.x = move(self.x)
        return self.x


if __name__ == "__main__":
    print(draw_hud(Player().step()))
'''


def test_a_reply_that_leaves_a_function_out_is_refused():
    reply = GAME.replace('def draw_hud(score):\n    return f"score: {score}"'
                         '\n\n\n', "")
    f = keep.check(GAME, reply, "python", path="game.py",
                   request="make the player faster")
    assert f.refused
    assert f.dropped == ("draw_hud",)
    d = f.diagnostic()
    assert d.code == "dropped-code" and "`draw_hud`" in d.message


def test_a_placeholder_comment_is_refused_wherever_it_is():
    reply = GAME.replace("    def step(self):\n        self.x = move(self.x)"
                         "\n        return self.x\n",
                         "    # ... rest of the methods unchanged\n")
    f = keep.check(GAME, reply, "python", path="game.py",
                   request="rewrite it from scratch")
    assert f.refused, "no request makes a placeholder right"
    assert f.placeholders and "rest of the methods" in f.placeholders[0][1]


def test_a_body_turned_into_ellipsis_is_refused():
    reply = GAME.replace("        self.x = move(self.x)\n        return self.x"
                         "\n", "        ...\n")
    f = keep.check(GAME, reply, "python", path="game.py", request="faster")
    assert f.refused and f.stubbed == ("Player.step",)


def test_a_comment_that_merely_mentions_code_is_not_a_placeholder():
    reply = GAME.replace("def move(x):",
                         "# the existing code moved here from utils.py\n"
                         "def move(x):").replace(
        "return x + SPEED", "return x + SPEED  # ... and more")
    f = keep.check(GAME, reply, "python", path="game.py", request="x")
    # "existing code moved here" is a sentence about history, not "the code
    # goes here": the pattern needs a word that says it is NOT here
    assert not f.placeholders, f.placeholders


def test_what_the_request_names_may_go():
    reply = GAME.replace('def draw_hud(score):\n    return f"score: {score}"'
                         '\n\n\n', "").replace(
        "print(draw_hud(Player().step()))", "print(Player().step())")
    f = keep.check(GAME, reply, "python", path="game.py",
                   request="Take draw_hud out, the HUD is gone")
    assert not f.refused


def test_a_rename_nothing_still_uses_is_allowed():
    reply = GAME.replace("def move(x):", "def advance(x):").replace(
        "self.x = move(self.x)", "self.x = advance(self.x)")
    f = keep.check(GAME, reply, "python", path="game.py", request="tidy",
                   used_elsewhere=lambda name: "")
    assert not f.refused and f.renamed == (("move", "advance"),)


def test_a_rename_another_file_still_uses_is_refused():
    reply = GAME.replace("def move(x):", "def advance(x):").replace(
        "self.x = move(self.x)", "self.x = advance(self.x)")
    f = keep.check(GAME, reply, "python", path="game.py", request="faster",
                   used_elsewhere=lambda name: "ai.py" if name == "move"
                   else "")
    assert f.refused and f.still_used == (("move", "advance", "ai.py"),)


def test_a_rewrite_request_turns_a_missing_definition_into_a_warning():
    reply = GAME.replace('def draw_hud(score):\n    return f"score: {score}"'
                         '\n\n\n', "")
    f = keep.check(GAME, reply, "python", path="game.py",
                   request="refactor the game loop")
    assert f.dropped and not f.refused and f.warn_only


def test_the_main_guard_counts_as_a_definition():
    reply = GAME[:GAME.index('if __name__')]
    f = keep.check(GAME, reply, "python", path="game.py", request="faster")
    assert f.refused and "__main__" in f.dropped


# --------------------------------------------------------------------------
# in the loop: refused BEFORE writing, the file still whole on disk
# --------------------------------------------------------------------------

def _fence(code: str) -> str:
    return f"```python\n{code}```"


def test_the_loop_refuses_a_dropped_function_before_writing_it(tmp_path):
    (tmp_path / "game.py").write_text(GAME)
    lazy = GAME.replace('def draw_hud(score):\n    return f"score: {score}"'
                        '\n\n\n', "").replace("SPEED = 3", "SPEED = 5")
    good = GAME.replace("SPEED = 3", "SPEED = 5")
    host = _host(tmp_path, [_fence(lazy), _fence(good)])
    loop = _loop(host, attempts=3, narrow="off")
    loop.journal = Journal(host.fs, "t")
    out = loop.run_task(Task(id="t1", path="game.py", purpose="the game",
                             lang="python"), request="make the player faster")
    assert out.ok, out.stopped_because
    assert "refused before writing" in out.attempts[0].note
    on_disk = (tmp_path / "game.py").read_text()
    assert "def draw_hud" in on_disk and "SPEED = 5" in on_disk
    # the next attempt was told what it left out, and was shown the file
    # as it stood — whole, since the refused reply never reached the disk
    tail = [m.content for m in host.llm.prompts[1] if m.role == "user"][0]
    assert "left out `draw_hud`" in tail
    assert 'def draw_hud(score):' in tail
    events = [e for e in loop.journal.events() if e.get("event") == "keep"]
    assert events and events[0]["data"]["refused"] is True


def test_after_a_refusal_the_next_attempt_is_narrowed(tmp_path):
    """The cure for a reply that leaves code out: a reply that holds only
    what it changes."""
    (tmp_path / "game.py").write_text(GAME)
    lazy = GAME.replace("    def step(self):\n        self.x = move(self.x)\n"
                        "        return self.x\n", "    # ... unchanged\n")
    host = _host(tmp_path, [_fence(lazy), _fence("SPEED = 5\n")])
    loop = _loop(host, attempts=3)
    out = loop.run_task(Task(id="t1", path="game.py", purpose="the game",
                             lang="python"), request="make SPEED 5")
    assert out.ok, out.stopped_because
    second = "\n".join(m.content for m in host.llm.prompts[1])
    assert "Return only the definitions you change" in second
    on_disk = (tmp_path / "game.py").read_text()
    assert "SPEED = 5" in on_disk and "def step(self):" in on_disk


def test_a_fresh_start_of_the_operators_file_is_shown_that_file(tmp_path):
    """The ladder's third rung hides the BROKEN attempts (D11). The
    operator's own file is not a broken attempt; hiding it regenerated
    their file blind."""
    (tmp_path / "game.py").write_text(GAME)
    broken = GAME.replace("return x + SPEED", "return x + SPEEED")
    replies = [_fence(broken), _fence(broken.replace("SPEEED", "SPEEEED")),
               _fence(GAME.replace("SPEED = 3", "SPEED = 4"))]
    host = _host(tmp_path, replies)
    loop = _loop(host, attempts=3, narrow="off", narrow_on_stagnation=False)
    loop.run_task(Task(id="t1", path="game.py", purpose="the game",
                       lang="python"), request="make SPEED 4")
    third = [m.content for m in host.llm.prompts[2] if m.role == "user"][0]
    assert "was written before and did not work" in third   # fresh start
    assert "THE FILE AS IT STOOD BEFORE THIS TASK" in third
    assert 'return f"score: {score}"' in third


PLACEHOLDERS = [
    "# ... rest unchanged", "    # ... rest of the methods unchanged",
    "# rest of the code", "# The rest of the code remains the same",
    "# (rest of file unchanged)", "# existing code here",
    "// ... existing code ...", "# keep the rest of the file",
    "# ... (other methods)", "# remaining methods unchanged",
    "# code omitted for brevity", "    # same as before",
    "/* rest of the function */", "# your existing code here"]
NOT_PLACEHOLDERS = [
    "# the existing code moved here from utils.py", "x = 1  # ... and more",
    "# ... and then we compute the speed", "# Compute the rest of the velocity",
    "# previous frame time", "# rest of the frame budget goes to physics",
    "# other players", "# TODO: handle the rest of the cases"]


def test_placeholder_lines_are_told_from_comments_about_code():
    for line in PLACEHOLDERS:
        assert keep.placeholders("", line), line
    for line in NOT_PLACEHOLDERS:
        assert not keep.placeholders("", line), line
    # a placeholder the file ALREADY had is the file's, not the reply's
    assert not keep.placeholders("# rest of the code\n", "# rest of the code")
