# SPDX-License-Identifier: Apache-2.0
"""§6.10's acceptance — and M41, the honesty requirement.

The spec's phase-7 criterion is exactly two things: **find a planted secret
and a planted O(n²)**. Both are here, with the rest of the deterministic
surface alongside them.

The other half of this file is M41, which matters more than it looks. Two
personas that are the same local model are not an adversarial system, and a
document that presents self-review as independent scrutiny is worse than one
that skips the review — it manufactures confidence. So the non-independence
line is asserted to be present, and asserted to be near the top.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import review  # noqa: E402
from cognitive_coder.personas import PromptBuilder  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    LocalFileSystem,
    ScriptedLLM,
    SubprocessExec,
)

PLANTED = '''"""A report builder with things wrong with it."""
import os

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
DB_URL = "postgresql://admin:hunter2@db.internal:5432/prod"


def build_report(rows, request_path):
    """Build a report."""
    out = ""
    for row in rows:
        out += str(row) + "\\n"
    path = os.path.join("/data", request_path)
    try:
        open(path).read()
    except Exception:
        pass
    return out


def run(expr):
    return eval(expr)
'''


@pytest.fixture
def result():
    return review.review(PLANTED, "src/report.py", lang_id="python",
                         use_model=False)


# --------------------------------------------------------------------------
# the phase-7 acceptance criterion
# --------------------------------------------------------------------------

def test_it_finds_a_planted_secret(result):
    """§11, phase 7. Half of the criterion, stated verbatim."""
    hits = [f for f in result.findings if "AWS access key" in f.title]
    assert hits
    assert hits[0].severity == "high"
    assert hits[0].category == "security"
    assert hits[0].line == 4


def test_it_finds_a_planted_quadratic(result):
    """The other half."""
    hits = [f for f in result.findings
            if "repeated concatenation" in f.title]
    assert hits
    assert hits[0].category == "performance"
    assert "square" in hits[0].detail


def test_a_secret_is_masked_in_the_report():
    """M44's spirit: a report gets pasted into tickets and chat windows.

    Reproducing the key in the document that warns about the key would be a
    fine joke and a real leak.
    """
    result = review.review('KEY = "AKIAIOSFODNN7EXAMPLE"\n', "m.py",
                           use_model=False)
    text = " ".join(f.detail + f.title for f in result.findings)
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "AKIAIO" in text, "it must still be findable in the file"


def test_a_placeholder_is_not_reported_as_a_leak():
    """`password = "changeme"` is not a credential, and flagging it is how a
    scanner teaches people to ignore it."""
    for code in ('password = "changeme"\n', 'api_key = "your-key-here"\n',
                 'token = "xxxxxxxxxxxx"\n', 'secret = "example-secret"\n'):
        result = review.review(code, "m.py", use_model=False)
        assert not [f for f in result.findings if f.category == "security"], (
            f"{code.strip()} was reported as a leaked credential")


# --------------------------------------------------------------------------
# the rest of the deterministic surface (§6.10)
# --------------------------------------------------------------------------

def test_eval_on_a_non_literal_is_high_severity(result):
    hits = [f for f in result.findings if "eval() on" in f.title]
    assert hits and hits[0].severity == "high"


def test_eval_on_a_literal_is_not_reported():
    """`eval("2 + 2")` cannot execute anything the author did not write."""
    result = review.review('x = eval("2 + 2")\n', "m.py", use_model=False)
    assert not [f for f in result.findings if "eval()" in f.title]


def test_an_empty_exception_handler_is_reported(result):
    assert [f for f in result.findings
            if "does nothing" in f.title or "swallows" in f.title]


def test_a_bare_except_is_worse_than_a_typed_one():
    bare = review.review("try:\n    x = 1\nexcept:\n    pass\n", "m.py",
                         use_model=False)
    typed = review.review("try:\n    x = 1\nexcept ValueError:\n    pass\n",
                          "m.py", use_model=False)
    assert bare.findings[0].severity == "medium"
    assert typed.findings[0].severity == "low"


def test_a_path_built_from_input_is_reported(result):
    assert [f for f in result.findings if "containment check" in f.title]


def test_a_long_function_is_reported_as_a_fact_not_an_opinion():
    body = "\n".join(f"    x{n} = {n}" for n in range(60))
    result = review.review(f"def big():\n{body}\n", "m.py", use_model=False)
    hits = [f for f in result.findings if "lines long" in f.title]
    assert hits and hits[0].category == "quality"
    assert hits[0].severity == "low", "length is a smell, not a defect"


def test_a_todo_in_finished_code_is_a_note():
    result = review.review("# TODO: handle the empty case\nx = 1\n", "m.py",
                           use_model=False)
    hits = [f for f in result.findings if "TODO" in f.title]
    assert hits and hits[0].severity == "note"
    assert "handle the empty case" in hits[0].detail


def test_a_public_function_missing_from_the_tests_is_reported():
    code = ("def exported():\n    return 1\n\n\n"
            "def _internal():\n    return 2\n")
    tests = "from m import exported\n\n\ndef test_it():\n    exported()\n"
    result = review.review(code, "m.py", test_source=tests, use_model=False)
    untested = [f for f in result.findings if "not mentioned" in f.title]
    assert not untested, "exported() IS mentioned in the tests"

    result2 = review.review(code, "m.py", test_source="def test_nothing():\n"
                                                      "    pass\n",
                            use_model=False)
    untested2 = [f.title for f in result2.findings if "not mentioned" in f.title]
    assert any("exported" in t for t in untested2)
    assert not any("_internal" in t for t in untested2), "privates are not API"


def test_a_membership_test_against_a_list_in_a_loop_is_reported():
    code = ("def f(rows):\n    for r in rows:\n"
            "        if r in ['a', 'b']:\n            pass\n")
    result = review.review(code, "m.py", use_model=False)
    assert [f for f in result.findings if "membership test" in f.title]


def test_clean_code_produces_nothing():
    """A review that always finds something is a review nobody believes."""
    clean = ('"""Add two numbers."""\n\n\n'
             "def add(a: int, b: int) -> int:\n"
             '    """Return the sum."""\n    return a + b\n')
    result = review.review(clean, "m.py", use_model=False)
    assert result.findings == []
    assert "nothing was flagged" in result.summary()


# --------------------------------------------------------------------------
# M41 — the honesty requirement
# --------------------------------------------------------------------------

def _model_result(overall="looks fine"):
    llm = ScriptedLLM([
        '{"security": [{"severity": "medium", "title": '
        '"input reaches the filesystem", "detail": "d", '
        '"line": 2, "fix": "check containment"}], '
        f'"performance": [], "overall": "{overall}"}}'])
    return review.review("def f(p):\n    return open(p).read()\n", "src/x.py",
                         llm=llm, prompts=PromptBuilder())


def test_a_same_model_review_is_labelled_non_independent():
    """M41 — the confident-wrongness failure mode, refused."""
    doc = review.recommendation_document(_model_result(), files=["src/x.py"])
    assert "not independent" in doc
    assert "one reviewer's opinion expressed twice" in doc


def test_the_non_independence_line_is_near_the_top():
    """"Near the top" is the requirement, not "somewhere in the document".

    A caveat below the findings is a caveat read after the reader has already
    decided what to believe.
    """
    doc = review.recommendation_document(_model_result(), files=["src/x.py"])
    position = doc.index("not independent") / len(doc)
    assert position < 0.25, f"the caveat sits {position:.0%} of the way down"
    assert doc.index("not independent") < doc.index("## Vulnerabilities")


def test_a_review_with_no_model_carries_no_independence_claim():
    """Nothing to disclaim when nothing claimed to be a second opinion."""
    result = review.review("x = 1\n", "m.py", use_model=False)
    doc = review.recommendation_document(result)
    assert "not independent" not in doc


def test_a_model_that_returns_unusable_json_says_so_rather_than_pretending():
    llm = ScriptedLLM(["I'd rather describe the code than review it."])
    result = review.review("x = 1\n", "m.py", llm=llm, prompts=PromptBuilder())
    assert not result.model_reviewed
    assert any("did not produce a usable answer" in n for n in result.notes)


def test_a_repaired_model_answer_reports_that_it_was_repaired():
    """D9 in the review stage: a model needing repair every time is telling
    you grammar-constrained decoding is available and switched off."""
    llm = ScriptedLLM(["Sure! {'security': [], 'performance': [], "
                       "'overall': 'fine',}"])
    result = review.review("x = 1\n", "m.py", llm=llm, prompts=PromptBuilder())
    assert result.model_reviewed
    assert any("needed repairing" in n for n in result.notes)


# --------------------------------------------------------------------------
# the Recommendation Document (FORGE §5.3)
# --------------------------------------------------------------------------

def test_the_document_has_all_four_sections():
    doc = review.recommendation_document(
        review.review(PLANTED, "src/report.py", use_model=False),
        request="build a report", files=["src/report.py"])
    for heading in ("## Executive summary", "## Quality assessment",
                    "## Vulnerabilities and fixes", "## Deployment guide"):
        assert heading in doc, heading


@pytest.mark.parametrize("skill,marker", [
    ("novice", "Back up what you have"),
    ("intermediate", "blast radius"),
    ("senior", "Profile before acting"),
])
def test_the_deployment_guide_is_pitched_at_the_readers_level(skill, marker):
    """A guide that assumes knowledge the reader lacks cannot be followed;
    one that explains what they know is one they stop reading."""
    doc = review.recommendation_document(
        review.review("x = 1\n", "m.py", use_model=False),
        skill_level=skill)
    assert marker in doc


def test_an_unknown_skill_level_falls_back_rather_than_failing():
    doc = review.recommendation_document(
        review.review("x = 1\n", "m.py", use_model=False),
        skill_level="wizard")
    assert "## Deployment guide" in doc


def test_a_high_severity_security_finding_says_do_not_deploy():
    doc = review.recommendation_document(
        review.review(PLANTED, "src/report.py", use_model=False))
    assert "Do not deploy this yet" in doc


def test_clean_code_does_not_say_do_not_deploy():
    doc = review.recommendation_document(
        review.review("def add(a, b):\n    return a + b\n", "m.py",
                      use_model=False))
    assert "Do not deploy" not in doc
    assert "Nothing was found that should stop this" in doc


def test_the_document_says_which_scanners_did_not_run():
    """C7 — "nothing was flagged" means something different depending on
    what was installed, and the reader deserves to know which sentence they
    are reading."""
    result = review.review(
        "x = 1\n", "m.py", lang_id="python",
        fs=LocalFileSystem(tempfile.mkdtemp()), ex=SubprocessExec(),
        use_model=False)
    doc = review.recommendation_document(result)
    if result.scanners_absent:
        assert "Not checked, because these are not installed" in doc
    else:
        assert "Scanners run:" in doc


def test_verification_caveats_reach_the_document():
    """A suite of zero tests must not be laundered into confidence by the
    review stage."""
    result = review.review("x = 1\n", "m.py", use_model=False)
    doc = review.recommendation_document(
        result, caveats=("the test command ran ZERO tests",))
    assert "ZERO tests" in doc
    assert "Caveats on the evidence" in doc


def test_the_document_never_claims_completeness():
    doc = review.recommendation_document(
        review.review("x = 1\n", "m.py", use_model=False))
    assert "not a guarantee that nothing else is wrong" in doc


def test_findings_convert_to_diagnostics_for_a_host_to_render():
    diags = review.as_diagnostics(
        review.review(PLANTED, "src/report.py", use_model=False))
    assert diags
    assert diags[0].severity == "error"        # high maps to error
    assert diags[0].tool in ("built-in", "model")


# --------------------------------------------------------------------------
# the session wiring (§4.3 — after, never instead)
# --------------------------------------------------------------------------

def test_the_review_refuses_to_run_when_nothing_verified(tmp_path):
    """Reviewing code that does not build spends tokens on a moot point."""
    from cognitive_coder import (
        AutoApprove,
        Host,
        MemoryStorage,
        RecordingEvents,
        Session,
        SessionConfig,
    )
    host = Host(llm=ScriptedLLM([]), fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".s")),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(attempts=1))
    assert session.review() == ""
    assert any("nothing to review" in m.lower()
               for _k, m, _d in host.events.events)


def test_a_full_session_writes_the_recommendation_document(tmp_path):
    from cognitive_coder import (
        AutoApprove,
        Host,
        MemoryStorage,
        RecordingEvents,
        Session,
        SessionConfig,
    )
    replies = [
        "src/report.py — build a report\n",
        '```python\nAWS_KEY = "AKIAIOSFODNN7EXAMPLE"\n\n\n'
        'def build(rows):\n    """Build."""\n    return len(rows)\n```',
        '{"security": [], "performance": [], "overall": "reads fine to me"}',
    ]
    host = Host(llm=ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".s")),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(attempts=1,
                                                 skeleton_first=False))
    session.run("a report builder", {"skill_level": "senior"})

    doc = (tmp_path / "Recommendation.md").read_text(encoding="utf-8")
    # The point of C5, in one document: the model said it read fine; the
    # deterministic scan found the key it was sitting on.
    assert "reads fine to me" in doc
    assert "AWS access key" in doc
    assert "Do not deploy this yet" in doc
    assert any(r.get("event") == "review" for r in session.journal.events())

