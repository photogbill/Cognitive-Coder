# SPDX-License-Identifier: Apache-2.0
"""`ccoder audit`: look at an existing project, report, and plan the fix.

The owner's ask: "show it the folder and ask it to improve it further, or
to analyze and tell me what improvements could be made in the first
place." These tests hold the promises that make that safe to do on a real
project: nothing but `.cc_state/` is written, the program is never run,
nothing is approved, a model reply is read as its ANSWER not its echo of
the schema, and the plan is something `ccoder build --spec` can use.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import audit, spec  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    NullLLM,
    RecordingEvents,
    ScriptedLLM,
    SubprocessExec,
)

GAME = ('API_KEY = "AKIAIOSFODNN7EXAMPLE"\n\n\n'
        'def speed(level):\n    return level * 2\n\n\n'
        'def old_helper(x):\n    return x + 1\n\n\n'
        'def spawn(n):\n    return [enemy_factory(i) for i in range(n)]\n')
SCORE = "def total(scores):\n    return sum(scores)\n"
TESTS = ("import unittest\nfrom src.game import speed\n\n\n"
         "class T(unittest.TestCase):\n"
         "    def test_speed(self):\n        self.assertEqual(speed(2), 4)\n\n"
         "    def test_wrong(self):\n        self.assertEqual(speed(1), 3)\n")


def _project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / "src" / "game.py").write_text(GAME)
    (tmp_path / "src" / "score.py").write_text(SCORE)
    (tmp_path / "tests" / "test_game.py").write_text(TESTS)
    return tmp_path


class CountingApproval(AutoApprove):
    """Records every question. An audit must never ask one."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def approve_diff(self, summary, unified_diff):
        self.asked.append(summary)
        return True


def _host(root: Path, llm=None, approval=None) -> Host:
    return Host(llm=llm or NullLLM(), fs=LocalFileSystem(str(root)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(root / ".cc_state")),
                events=RecordingEvents(), approval=approval or AutoApprove())


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file() and ".cc_state" not in p.parts}


def _titles(report) -> list[str]:
    return [f.title for f in report.findings]


# --------------------------------------------------------------------------
# the tools
# --------------------------------------------------------------------------

def test_the_tools_find_what_was_seeded_and_rank_it(tmp_path):
    report = audit.audit_project(_host(_project(tmp_path)),
                                 config=audit.AuditConfig(model_pass=False))
    titles = " | ".join(_titles(report))
    assert "AWS access key" in titles
    assert "enemy_factory" in titles
    assert "`old_helper` is possibly unused" in titles
    assert "the Python test suite fails" in titles
    assert any(f.path == "src/score.py" and "no test file" in f.title
               for f in report.findings)
    # High first in the document.
    doc = report.document()
    assert doc.index("AWS access key") < doc.index("possibly unused")


def test_a_failing_test_is_reported_with_a_project_relative_path(tmp_path):
    """An absolute path is not a project file to spec.py, and a bare
    `test_game.py:10:` in the message was read as a required test at the
    ROOT — which build would then have created."""
    report = audit.audit_project(_host(_project(tmp_path)),
                                 config=audit.AuditConfig(model_pass=False))
    failing = [f for f in report.findings if "suite fails" in f.title]
    assert failing and failing[0].path == "tests/test_game.py"
    plan = spec.from_text(report.to_spec())
    assert plan.required_tests == ("tests/test_game.py",)
    assert "src/game.py" in plan.mentioned_paths


def test_a_function_used_only_by_name_is_not_called_unused(tmp_path):
    """The call graph cannot see a callback passed by name; the text can."""
    root = _project(tmp_path)
    (root / "src" / "hooks.py").write_text(
        "def on_hit(e):\n    return e\n\n\nHANDLERS = [on_hit]\n")
    report = audit.audit_project(_host(root),
                                 config=audit.AuditConfig(model_pass=False))
    assert "`on_hit` is possibly unused" not in _titles(report)


def test_one_fact_is_said_once(tmp_path):
    """A function that is possibly unused, in a module no test mentions,
    also collected "is public and is not mentioned in the tests"."""
    report = audit.audit_project(_host(_project(tmp_path)),
                                 config=audit.AuditConfig(model_pass=False))
    assert not [t for t in _titles(report)
                if t.startswith("`total` is public")]


def test_the_audit_writes_only_its_own_state_and_asks_nothing(tmp_path):
    root = _project(tmp_path)
    before = _snapshot(root)
    approval = CountingApproval()
    audit.audit_project(_host(root, approval=approval),
                        config=audit.AuditConfig(model_pass=False))
    assert _snapshot(root) == before
    assert approval.asked == []
    assert (root / ".cc_state" / "audit" / "report.md").exists()
    assert (root / ".cc_state" / "audit" / "plan.md").exists()


def test_the_program_itself_is_never_run(tmp_path):
    """Tests run; `main` does not. A game, a server or a deletion script
    must not be started to find out what it does."""
    root = _project(tmp_path)
    marker = root / "RAN"
    (root / "main.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n")
    audit.audit_project(_host(root),
                        config=audit.AuditConfig(model_pass=False))
    assert not marker.exists()


def test_with_no_model_it_says_so_and_still_reports(tmp_path):
    report = audit.audit_project(_host(_project(tmp_path), llm=NullLLM()))
    assert "no model is loaded" in report.model_skipped
    assert "no model read the code" in report.document()
    assert report.findings


def test_a_project_with_no_tests_says_so(tmp_path):
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    report = audit.audit_project(_host(tmp_path),
                                 config=audit.AuditConfig(model_pass=False))
    assert "the project has no tests" in _titles(report)
    assert report.tests_ran is False
    assert any("No test files" in s for s in report.verification)


def test_other_languages_are_inventoried_by_language(tmp_path):
    (tmp_path / "ui.js").write_text("export function f() { return 1; }\n")
    (tmp_path / "core.py").write_text("X = 1\n")
    report = audit.audit_project(_host(tmp_path),
                                 config=audit.AuditConfig(model_pass=False))
    assert dict(report.files) == {"core.py": "python", "ui.js": "javascript"}


def test_an_empty_folder_is_a_sentence_not_a_crash(tmp_path):
    report = audit.audit_project(_host(tmp_path))
    assert not report.files and "nothing to audit" in report.document()


# --------------------------------------------------------------------------
# the model pass
# --------------------------------------------------------------------------

ECHO_THEN_ANSWER = (
    "I will answer in this shape: " + audit.AUDIT_CONTRACT + "\n\nNow:\n"
    '{"findings": [{"line": "12-13", "severity": "CRITICAL", '
    '"kind": "bug", "title": "spawn calls an undefined factory", '
    '"detail": "NameError on the first spawn", '
    '"change": "import or define enemy_factory"}]}')


def _one_file(tmp_path: Path) -> Path:
    (tmp_path / "game.py").write_text(GAME)
    return tmp_path


def test_the_answer_is_read_not_the_echoed_schema(tmp_path):
    """A reasoning model restates the schema before answering; the first
    object is the example. The last object carrying `findings` is read,
    its line range coerced, and an unknown severity fails CLOSED."""
    llm = ScriptedLLM([ECHO_THEN_ANSWER])
    report = audit.audit_project(
        _host(_one_file(tmp_path), llm=llm),
        config=audit.AuditConfig(run_tests=False))
    mine = [f for f in report.findings if f.by_model]
    assert [f.title for f in mine] == ["spawn calls an undefined factory"]
    assert mine[0].line == 12 and mine[0].severity == "high"
    assert "one line" not in _titles(report)
    doc = report.document()
    assert "not two independent confirmations" in doc
    assert doc.index("independent") < doc.index("## Found by the tools")


def test_nothing_found_and_no_answer_are_different_sentences(tmp_path):
    root = _one_file(tmp_path)
    (root / "b.py").write_text("Y = 2\n")
    llm = ScriptedLLM(['{"findings": []}', "I think it is fine overall."])
    report = audit.audit_project(
        _host(root, llm=llm), config=audit.AuditConfig(run_tests=False))
    notes = " | ".join(report.model_notes)
    assert "found nothing it would change" in notes
    assert "NOT reviewed by it" in notes


def test_the_cached_prefix_is_byte_identical_across_the_audit(tmp_path):
    root = _one_file(tmp_path)
    (root / "b.py").write_text("Y = 2\n")
    (root / "c.py").write_text("Z = 3\n")
    llm = ScriptedLLM(['{"findings": []}'] * 3)
    audit.audit_project(_host(root, llm=llm),
                        config=audit.AuditConfig(run_tests=False))
    assert len(llm.prompts) == 3
    prefixes = {tuple(m.content for m in p[:-1]) for p in llm.prompts}
    assert len(prefixes) == 1, "the prefix changed between calls (M52)"
    tails = {p[-1].content for p in llm.prompts}
    assert len(tails) == 3


def test_the_model_reads_the_files_most_worth_reading(tmp_path):
    root = _one_file(tmp_path)
    (root / "quiet.py").write_text("Q = 1\n")
    llm = ScriptedLLM(['{"findings": []}'])
    report = audit.audit_project(
        _host(root, llm=llm),
        config=audit.AuditConfig(run_tests=False, max_model_files=1))
    assert report.model_files == ("game.py",)


def test_the_focus_reaches_the_model_and_the_plan(tmp_path):
    llm = ScriptedLLM(['{"findings": []}'])
    report = audit.audit_project(
        _host(_one_file(tmp_path), llm=llm),
        config=audit.AuditConfig(run_tests=False,
                                 focus="the spawning code"))
    assert "the spawning code" in llm.prompts[0][-1].content
    assert "the spawning code" in report.to_spec()


# --------------------------------------------------------------------------
# iterations
# --------------------------------------------------------------------------

def test_a_second_audit_says_what_was_fixed(tmp_path):
    root = _project(tmp_path)
    cfg = audit.AuditConfig(model_pass=False)
    first = audit.audit_project(_host(root), config=cfg)
    assert first.since_last == ""
    # Fix two things, break one.
    (root / "src" / "game.py").write_text(
        GAME.replace('API_KEY = "AKIAIOSFODNN7EXAMPLE"', "import os\n"
                     'API_KEY = os.environ.get("API_KEY", "")')
        .replace("enemy_factory(i)", "i"))
    (root / "src" / "new.py").write_text("def g():\n    return nope()\n")
    second = audit.audit_project(_host(root), config=cfg)
    assert "gone" in second.since_last and "new" in second.since_last
    gone = int(second.since_last.split(": ")[1].split(" of")[0])
    assert gone >= 2, second.since_last
    assert second.document().index(second.since_last) < \
        second.document().index("## Does it work")


def test_the_plan_leaves_low_findings_out_and_says_so(tmp_path):
    report = audit.audit_project(_host(_project(tmp_path)),
                                 config=audit.AuditConfig(model_pass=False))
    plan = report.to_spec()
    assert "possibly unused" not in plan
    assert "low-severity finding(s)" in plan
    assert "Change ONLY the files named below" in plan


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------

def test_ccoder_audit_runs_on_a_folder_and_exits_zero(tmp_path, capsys):
    from cognitive_coder import cli
    root = _project(tmp_path)
    code = cli.main(["audit", str(root), "--no-model"])
    out = capsys.readouterr().out
    assert code == 0
    assert "# Audit" in out and "AWS access key" in out
    assert "ccoder build -p" in out and "--spec" in out


def test_ccoder_audit_on_a_missing_folder_is_a_sentence(tmp_path, capsys):
    from cognitive_coder import cli
    code = cli.main(["audit", str(tmp_path / "nope"), "--no-model"])
    assert code == 2
    assert "There is no folder" in capsys.readouterr().err
