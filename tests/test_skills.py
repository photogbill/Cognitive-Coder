# SPDX-License-Identifier: Apache-2.0
"""Deployed skills (F3) — discovery, scoping, budgets, and the prefix.

The load-bearing assertions are the boring-looking ones: sorted order,
byte-identical blocks, and SKIPPED-never-truncated. Each guards a failure
that would not raise anything — a reordered prefix just gets 20× slower,
and a truncated rule just ships half a sentence of policy.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import cli  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryFileSystem,
    ScriptedLLM,
)
from cognitive_coder.session import Session, SessionConfig  # noqa: E402
from cognitive_coder.skills import (  # noqa: E402
    SKILLS_DIR,
    STARTER_SKILLS,
    load_skills,
    parse_skill,
)


def _fs(files: dict[str, str]) -> MemoryFileSystem:
    return MemoryFileSystem({k: v.encode() for k, v in files.items()})


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_a_header_is_parsed_and_kept_out_of_the_body():
    s = parse_skill(f"{SKILLS_DIR}/10-style.md",
                    "---\nname: style\ndescription: how we write\n"
                    "lang: python, rust\n---\nComment the why.\n")
    assert s.name == "style"
    assert s.description == "how we write"
    assert s.langs == ("python", "rust")
    assert s.body == "Comment the why."
    assert "---" not in s.body


def test_a_file_with_no_header_is_all_body_named_for_its_file():
    s = parse_skill(f"{SKILLS_DIR}/20-tests.md", "Test the contract.\n")
    assert s.name == "20-tests"
    assert s.body == "Test the contract."
    assert s.applies_to("python") and s.applies_to("")


def test_an_unclosed_header_degrades_to_body_not_to_nothing():
    """Guidance with a broken header should stay guidance."""
    text = "---\nname: broken\nNever swallow exceptions.\n"
    s = parse_skill(f"{SKILLS_DIR}/x.md", text)
    assert "Never swallow exceptions." in s.body
    assert s.name == "x"


# --------------------------------------------------------------------------
# discovery, order, scoping, budgets
# --------------------------------------------------------------------------

def test_discovery_is_sorted_and_the_block_is_byte_identical():
    files = {f"{SKILLS_DIR}/20-b.md": "Rule B.",
             f"{SKILLS_DIR}/10-a.md": "Rule A.",
             "src/real_code.py": "x = 1\n"}
    first = load_skills(_fs(files))
    second = load_skills(_fs(files))
    assert [s.path for s in first.skills] == [
        f"{SKILLS_DIR}/10-a.md", f"{SKILLS_DIR}/20-b.md"]
    assert first.block() == second.block()
    assert "Rule A." in first.block() and "Rule B." in first.block()
    assert first.block().index("Rule A.") < first.block().index("Rule B.")


def test_a_project_without_skills_is_the_ordinary_case_not_an_error():
    load = load_skills(_fs({"src/app.py": "x = 1\n"}))
    assert load.skills == () and load.skipped == ()
    assert load.block() == ""


def test_language_scoping_filters_before_the_budget_is_spent():
    files = {f"{SKILLS_DIR}/10-rust.md": "---\nlang: rust\n---\n" + "R" * 90,
             f"{SKILLS_DIR}/20-any.md": "A" * 90}
    load = load_skills(_fs(files), lang="python", max_total_chars=100)
    # The rust skill neither loads nor consumes the python session's budget.
    assert [s.name for s in load.skills] == ["20-any"]
    assert any("scoped to rust" in reason for _, reason in load.skipped)


def test_an_oversized_skill_is_skipped_by_name_never_truncated():
    body = "Never do the thing. " * 40
    load = load_skills(_fs({f"{SKILLS_DIR}/big.md": body}),
                       max_skill_chars=100)
    assert load.skills == ()
    path, reason = load.skipped[0]
    assert path.endswith("big.md") and "too large" in reason
    # Nothing partial leaked into the prompt.
    assert load.block() == ""


def test_the_total_budget_keeps_earlier_sorting_files():
    files = {f"{SKILLS_DIR}/10-first.md": "F" * 60,
             f"{SKILLS_DIR}/20-second.md": "S" * 60}
    load = load_skills(_fs(files), max_total_chars=100)
    assert [s.name for s in load.skills] == ["10-first"]
    assert any("total budget" in reason for _, reason in load.skipped)


def test_the_starter_pack_parses_scopes_everywhere_and_fits_the_budget():
    total = 0
    for filename, content in STARTER_SKILLS.items():
        s = parse_skill(f"{SKILLS_DIR}/{filename}", content)
        assert s.body and s.description and s.langs == ()
        total += len(s.body)
    from cognitive_coder.skills import MAX_TOTAL_CHARS
    assert total <= MAX_TOTAL_CHARS


# --------------------------------------------------------------------------
# the session actually uses them — and can be told not to
# --------------------------------------------------------------------------

def _host(files: dict[str, str]) -> Host:
    return Host(llm=ScriptedLLM(["ok"]), fs=_fs(files),
                approval=AutoApprove())


def test_skills_reach_the_cached_prefix_with_provenance():
    host = _host({f"{SKILLS_DIR}/10-style.md":
                  "---\nname: style\n---\nComment the why."})
    session = Session(host)
    assert "Comment the why." in session.prompts.conventions
    prov = session.skill_load.provenance()
    assert prov and prov[0]["name"] == "style" and prov[0]["sha256"]
    # M52: the same construction yields byte-identical conventions.
    again = Session(_host({f"{SKILLS_DIR}/10-style.md":
                           "---\nname: style\n---\nComment the why."}))
    assert again.prompts.conventions == session.prompts.conventions


def test_config_conventions_and_skills_layer_rather_than_replace():
    host = _host({f"{SKILLS_DIR}/10-a.md": "Rule from disk."})
    session = Session(host, config=SessionConfig(
        conventions="Rule from config."))
    assert "Rule from config." in session.prompts.conventions
    assert "Rule from disk." in session.prompts.conventions
    assert session.prompts.conventions.index("Rule from config.") \
        < session.prompts.conventions.index("Rule from disk.")


def test_use_skills_false_leaves_the_prompt_untouched():
    host = _host({f"{SKILLS_DIR}/10-a.md": "Rule from disk."})
    session = Session(host, config=SessionConfig(use_skills=False))
    assert "Rule from disk." not in session.prompts.conventions
    assert session.skill_load.skills == ()


# --------------------------------------------------------------------------
# the CLI: deploy seeds, and never overwrites
# --------------------------------------------------------------------------

def test_deploy_seeds_once_and_respects_edits(tmp_path):
    project = str(tmp_path)
    assert cli.main(["skills", "deploy", "--project", project]) == 0
    fs = LocalFileSystem(project)
    deployed = fs.list(f"{SKILLS_DIR}/*.md")
    assert len(deployed) == len(STARTER_SKILLS)

    # The user edits one file; a second deploy must not undo the edit.
    edited = f"{SKILLS_DIR}/10-house-style.md"
    fs.write(edited, "---\nname: house-style\n---\nTabs, always.\n")
    assert cli.main(["skills", "deploy", "--project", project]) == 0
    assert fs.read(edited).strip().endswith("Tabs, always.")


def test_the_list_command_reports_active_and_skipped(tmp_path, capsys):
    project = str(tmp_path)
    fs = LocalFileSystem(project)
    fs.write(f"{SKILLS_DIR}/10-any.md", "For everyone.")
    fs.write(f"{SKILLS_DIR}/20-rust.md", "---\nlang: rust\n---\nRust only.")
    assert cli.main(["skills", "list", "--project", project,
                     "--lang", "python"]) == 0
    out = capsys.readouterr().out
    assert "10-any" in out
    assert "scoped to rust" in out
