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


# --------------------------------------------------------------------------
# calibration (profiles.py, 2026-10-27)
# --------------------------------------------------------------------------

def test_calibration_measures_a_run_and_chooses_by_what_passed():
    from cognitive_coder import bench
    rows = [{"ok": True, "hidden": {"passed": 3, "total": 3},
             "generations": 4, "narrow_unplaced": 1, "keep_refused": 0,
             "fim_tried": 2, "fim_filled": 1, "tokens_out_total": 400,
             "decode_ms_total": 20000, "seconds": 30},
            {"ok": False, "hidden": {"passed": 1, "total": 3},
             "generations": 6, "narrow_unplaced": 0, "keep_refused": 2,
             "seconds": 40}]
    m = bench.measure(rows)
    assert m["passed"] == 1 and m["hidden_passed"] == 4
    assert m["well_formed"] == 0.7 and m["tok_s"] == 20.0
    assert m["fim_filled"] == 1 and m["seconds"] == 70
    measured = {"whole": {"passed": 2, "hidden_passed": 5, "well_formed": 0.9},
                "symbols": {"passed": 2, "hidden_passed": 6,
                            "well_formed": 0.8},
                "fim": {"passed": 2, "hidden_passed": 6, "well_formed": 0.95,
                        "fim_filled": 0}}
    # fim filled nothing, so it is not offered; symbols beats whole on
    # hidden tests
    assert bench.choose_format(measured) == "symbols"
    measured["fim"]["fim_filled"] = 3
    assert bench.choose_format(measured) == "fim"
    assert bench.choose_format({}) == "auto"


def test_calibration_writes_the_profile_from_its_runs(tmp_path, monkeypatch):
    from cognitive_coder import bench, profiles

    def fake_run_suite(make_llm, *, suite, out, only, label, config, say,
                       events, ex):
        fmt = label.split("-", 1)[1]
        ok = fmt == "symbols"
        rows = [{"task": "t", "ok": ok, "generations": 2,
                 "hidden": {"passed": 2 if ok else 0, "total": 2},
                 "fim_tried": 1 if fmt == "fim" else 0,
                 "fim_filled": 1 if fmt == "fim" else 0}]
        run_dir = tmp_path / "runs" / label
        run_dir.mkdir(parents=True)
        return bench.BenchRun(path=run_dir, rows=rows, problems=[])
    monkeypatch.setattr(bench, "run_suite", fake_run_suite)
    report = bench.calibrate(lambda t, f: None, "Qwen3-Coder-30B-Q4.gguf",
                             suite=tmp_path, out=tmp_path / "runs",
                             say=lambda _s: None,
                             profile_dir=tmp_path / "profiles")
    assert report["chosen"] == "symbols"
    assert "| symbols | 1/1 |" in report["text"]
    found = profiles.profile_for("Qwen3-Coder-30B-Q4.gguf",
                                 directory=tmp_path / "profiles")
    assert found.source == "calibrated" and found.edit_format == "symbols"
    assert found.measured["formats"]["fim"]["fim_filled"] == 1
    assert found.sampling() == profiles.builtin_for("qwen3-coder").sampling()
    assert (tmp_path / "runs" / "CALIBRATION.md").exists()
