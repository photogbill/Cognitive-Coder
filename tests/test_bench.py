# SPDX-License-Identifier: Apache-2.0
"""The regression suite (bench.py) — Aider's benchmark design, OpenHands'
separate grading: hidden tests the model never sees, a run compared with
the one before task by task, and replay at no inference cost."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import bench  # noqa: E402
from cognitive_coder.ports import ScriptedLLM  # noqa: E402

GOOD = ["answer.py — the answer\n",
        "```python\ndef answer():\n    return 42\n```"]
BAD = ["answer.py — the answer\n",
       "```python\ndef answer():\n    return 41\n```"]


def _suite(tmp_path: Path) -> Path:
    suite = tmp_path / "suite"
    task = suite / "answer"
    (task / "hidden").mkdir(parents=True, exist_ok=True)
    (task / "task.md").write_text("Write answer.py with a function answer() "
                                  "that returns 42.")
    (task / "bench.json").write_text('{"kind": "build"}')
    (task / "hidden" / "test_answer.py").write_text(
        "import unittest\nfrom answer import answer\n\n\n"
        "class T(unittest.TestCase):\n    def test_it(self):\n"
        "        self.assertEqual(answer(), 42)\n")
    return suite


def _run(tmp_path, replies, label=""):
    def make(_task, folder):
        return bench.RecordingLLM(ScriptedLLM(list(replies),
                                              supports_tools=False),
                                  folder / bench.REPLIES)
    return bench.run_suite(make, suite=_suite(tmp_path),
                           out=tmp_path / "runs", label=label,
                           config={"pair_tests": False},
                           say=lambda _s: None)


def test_the_starter_suite_is_sound(tmp_path):
    """Every task's hidden tests pass on its reference solution and — for
    change and debug tasks — fail on its start files: each task measures
    something, and can be passed."""
    assert bench.check_suite(scratch=tmp_path / "check") == []
    tasks, problems = bench.load_suite()
    assert not problems
    assert {t.kind for t in tasks} == {"build", "change", "debug"}


def test_a_run_is_graded_by_tests_the_model_never_saw(tmp_path):
    run = _run(tmp_path, GOOD)
    row = run.rows[0]
    assert row["ok"] and row["hidden"]["passed"] == 1
    assert row["calls"] == 2
    assert (run.path / "results.jsonl").exists()
    assert "1 of 1 tasks passed" in (run.path / "SUMMARY.md").read_text()
    # the model's prompts never contained the hidden test
    replies = (run.path / "answer" / bench.REPLIES).read_text()
    assert "assertEqual(answer(), 42)" not in replies


def test_a_later_run_says_what_got_worse(tmp_path):
    _run(tmp_path, GOOD, label="before")
    run = _run(tmp_path, BAD, label="after")
    assert not run.rows[0]["ok"]
    assert run.previous is not None and "before" in run.previous.name
    assert "- worse: answer" in run.summary
    assert "Why the failures failed" in run.summary


def test_a_replay_answers_with_the_recorded_replies(tmp_path):
    first = _run(tmp_path, GOOD)
    recorded = first.path / "answer" / bench.REPLIES

    def make(_task, _folder):
        return bench.ReplayLLM(recorded)
    run = bench.run_suite(make, suite=tmp_path / "suite",
                          out=tmp_path / "runs", label="replay",
                          config={"pair_tests": False}, say=lambda _s: None)
    row = run.rows[0]
    assert row["ok"]
    assert row["replay"] == {"diverged_at": 0, "exhausted": False}


def test_a_replay_that_runs_out_or_diverges_says_so(tmp_path):
    first = _run(tmp_path, GOOD)
    recorded = first.path / "answer" / bench.REPLIES
    rows = [json.loads(x) for x in recorded.read_text().splitlines()]
    short = tmp_path / "short.jsonl"
    short.write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    llm = bench.ReplayLLM(short)
    from cognitive_coder.types import Message
    llm.complete([Message(role="user", content="not the recorded prompt")])
    assert llm.diverged_at == 1
    out = llm.complete([Message(role="user", content="more")])
    assert llm.exhausted and out.finish_reason == "error"


def test_a_suite_with_a_broken_task_says_which(tmp_path):
    suite = _suite(tmp_path)
    (suite / "nothing").mkdir()
    (suite / "badkind").mkdir()
    (suite / "badkind" / "bench.json").write_text('{"kind": "dance"}')
    _tasks, problems = bench.load_suite(suite)
    assert any(p.startswith("nothing: no bench.json") for p in problems)
    assert any("badkind" in p and "dance" in p for p in problems)


def test_the_cli_checks_the_suite():
    from cognitive_coder.cli import main
    assert main(["bench", "--check", "--only", "build-roman"]) == 0
