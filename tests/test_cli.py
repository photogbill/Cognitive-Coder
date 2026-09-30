# SPDX-License-Identifier: Apache-2.0
"""`ccoder` driven end to end, with a scripted model in place of a server.

The CLI was the least-covered module in the tree (31%) for a boring reason:
every path through `build` starts by asking an HTTP endpoint what is loaded,
and CI has no model server. The seam is `cli.make_provider` — the one call
that turns `--url` into an LLMPort — so each test swaps it for a function
returning a `ScriptedLLM`, and everything downstream of it (planning,
approval, writing, verifying, the report, the exit code) is the real thing.

Tests call `cli.main([...])` rather than spawning `ccoder`: the argument
parsing is what is under test, and a subprocess per test would triple the
run time for no extra evidence. The one test that needs a SECOND PROCESS —
history surviving the end of the one that wrote it — spawns one.
"""

from __future__ import annotations

import builtins
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import cli, providers  # noqa: E402
from cognitive_coder.errors import ConfigurationError  # noqa: E402
from cognitive_coder.ports import NullLLM, ScriptedLLM  # noqa: E402
from cognitive_coder.session import Session  # noqa: E402
from cognitive_coder.skills import STARTER_SKILLS  # noqa: E402
from cognitive_coder.types import ModelCapabilities, TaskOutcome  # noqa: E402

PLAN = "src/alpha.py — the first thing\nsrc/beta.py — the second thing\n"
PLAN3 = PLAN + "src/gamma.py — the third thing\n"
ALPHA = '```python\ndef alpha():\n    """First."""\n    return 1\n```'
BETA = '```python\ndef beta():\n    """Second."""\n    return 2\n```'


class Scripted:
    """Installs a ScriptedLLM behind `cli.make_provider`, and remembers."""

    def __init__(self, monkeypatch) -> None:
        self._mp = monkeypatch
        self.llm: ScriptedLLM | None = None
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, replies, llm=None):
        self.llm = llm or ScriptedLLM(list(replies), supports_tools=False)

        def factory(*args, **kwargs):
            self.calls.append((args, kwargs))
            return self.llm

        self._mp.setattr(cli, "make_provider", factory)
        return self.llm


@pytest.fixture
def scripted(monkeypatch):
    # `doctor` probes five local ports; a CI box answers none of them, but
    # the probe is still a network call the suite has no business making.
    monkeypatch.setattr(cli, "detect", lambda *a, **k: [])
    return Scripted(monkeypatch)


@pytest.fixture
def no_input(monkeypatch):
    """Fail loudly if anything prompts: a test must never hang on input()."""
    def refuse(prompt=""):
        raise AssertionError(f"unexpected prompt: {prompt!r}")
    monkeypatch.setattr(builtins, "input", refuse)


def _sessions(project: Path) -> list[str]:
    return sorted(p.stem for p in (project / ".cc_journal").glob("*.jsonl"))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def test_build_with_yes_writes_every_file_and_exits_zero(
        tmp_path, scripted, no_input, capsys):
    scripted([PLAN, ALPHA, BETA])
    rc = cli.main(["build", "two small modules", "--yes", "-p",
                   str(tmp_path), "--attempts", "1"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "return 1" in (tmp_path / "src" / "alpha.py").read_text()
    assert "return 2" in (tmp_path / "src" / "beta.py").read_text()
    for marker in ("Model: scripted", "[plan]", "[build 2/2]", "committed"):
        assert marker in out


def test_build_passes_url_and_model_to_the_provider(tmp_path, scripted):
    scripted([PLAN])
    cli.main(["build", "x", "--preview", "-p", str(tmp_path),
              "--url", "http://127.0.0.1:9999", "--model", "devstral"])
    _args, kwargs = scripted.calls[0]
    assert kwargs["base_url"] == "http://127.0.0.1:9999"
    assert kwargs["model"] == "devstral"


def test_build_without_yes_asks_and_a_no_writes_nothing(
        tmp_path, scripted, monkeypatch, capsys):
    asked: list[str] = []
    monkeypatch.setattr(builtins, "input",
                        lambda prompt="": asked.append(prompt) or "n")
    scripted([PLAN, ALPHA, BETA])
    rc = cli.main(["build", "two", "-p", str(tmp_path), "--attempts", "1"])
    out = capsys.readouterr().out
    assert rc == 1, "a build whose every change was declined did not work"
    # Three questions: the skeleton's stubs (one transaction), then each
    # of the two files. The skeleton asks too — a refused build must leave
    # the project exactly as it found it, stubs included.
    assert len(asked) == 3 and "Apply this change?" in asked[0]
    assert "+    return 1" in out, "the diff is shown before the question"
    assert not (tmp_path / "src" / "alpha.py").exists()
    assert not (tmp_path / "src" / "beta.py").exists()


def test_build_with_eof_at_the_prompt_is_a_no(tmp_path, scripted,
                                              monkeypatch):
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr(builtins, "input", eof)
    scripted([PLAN, ALPHA, BETA])
    assert cli.main(["build", "two", "-p", str(tmp_path),
                     "--attempts", "1"]) == 1


def test_dry_run_refuses_every_write_without_asking(tmp_path, scripted,
                                                    no_input):
    scripted([PLAN, ALPHA, BETA])
    rc = cli.main(["build", "two", "-p", str(tmp_path), "--attempts", "1",
                   "--dry-run", "--yes"])
    assert rc == 1
    # Not even a stub: the skeleton is a transaction like any other write.
    assert not (tmp_path / "src" / "alpha.py").exists()


def test_build_reads_the_request_from_a_spec_file(tmp_path, scripted,
                                                  no_input):
    spec = tmp_path / "plan.md"
    spec.write_text("# Two modules\n\nBuild src/alpha.py and src/beta.py.\n",
                    encoding="utf-8")
    llm = scripted([PLAN, ALPHA, BETA])
    rc = cli.main(["build", "--spec", str(spec), "--yes", "-p",
                   str(tmp_path), "--attempts", "1"])
    assert rc == 0
    planning_prompt = "\n".join(m.content for m in llm.prompts[0])
    assert "Build src/alpha.py and src/beta.py." in planning_prompt


def test_a_request_and_a_spec_together_are_refused(tmp_path, scripted,
                                                   capsys):
    spec = tmp_path / "plan.md"
    spec.write_text("# x\n", encoding="utf-8")
    rc = cli.main(["build", "a sentence", "--spec", str(spec), "-p",
                   str(tmp_path)])
    assert rc == 2
    assert "not both" in capsys.readouterr().err
    assert scripted.calls == [], "nothing is asked of a model"


def test_a_missing_spec_file_is_a_sentence(tmp_path, scripted, capsys):
    rc = cli.main(["build", "--spec", str(tmp_path / "nope.md"), "-p",
                   str(tmp_path)])
    assert rc == 2
    assert "Cannot read that specification" in capsys.readouterr().err


def test_no_request_at_all_is_a_sentence(tmp_path, scripted, capsys):
    assert cli.main(["build", "-p", str(tmp_path)]) == 2
    assert "Nothing to build" in capsys.readouterr().err


def test_preview_plans_prints_and_writes_nothing(tmp_path, scripted,
                                                 no_input, capsys):
    scripted([PLAN])
    rc = cli.main(["build", "two", "--preview", "-p", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Build order (2 file(s))" in out
    assert "1. src/alpha.py" in out and "2. src/beta.py" in out
    assert "Tests: 0 named in the request, 0 in the plan" in out
    assert "Nothing was written" in out
    assert not (tmp_path / "src").exists()


def test_max_files_caps_the_plan(tmp_path, scripted, capsys):
    scripted([PLAN3])
    rc = cli.main(["build", "three", "--preview", "--max-files", "2", "-p",
                   str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Build order (2 file(s))" in captured.out
    assert "gamma" not in captured.out
    assert "truncated at 2 files" in captured.err


def _house_style_line() -> str:
    body = STARTER_SKILLS["10-house-style.md"].split("---", 2)[-1]
    return next(line for line in body.splitlines() if len(line) > 30)


def test_deployed_skills_reach_the_prompt_and_no_skills_keeps_them_out(
        tmp_path, scripted, capsys):
    assert cli.main(["skills", "deploy", "-p", str(tmp_path)]) == 0
    line = _house_style_line()

    with_skills = scripted([PLAN])
    cli.main(["build", "two", "--preview", "-p", str(tmp_path)])
    assert line in with_skills.prompts[0][0].content

    without = scripted([PLAN])
    cli.main(["build", "two", "--preview", "--no-skills", "-p",
              str(tmp_path)])
    assert line not in without.prompts[0][0].content


def test_no_model_loaded_is_exit_3_and_a_sentence(tmp_path, scripted,
                                                  capsys):
    scripted([], llm=NullLLM())
    rc = cli.main(["build", "two", "-p", str(tmp_path)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "No model is loaded" in err and "ccoder doctor" in err


def test_a_provider_that_cannot_be_built_is_exit_2(tmp_path, monkeypatch,
                                                   capsys):
    def refuse(*a, **k):
        raise ConfigurationError("That endpoint cannot be used.")
    monkeypatch.setattr(cli, "make_provider", refuse)
    assert cli.main(["build", "two", "-p", str(tmp_path)]) == 2
    assert "That endpoint cannot be used." in capsys.readouterr().err


class _Vanishing(ScriptedLLM):
    """Loaded for planning, then gone — the operator unloaded it mid-run."""

    def __init__(self, replies, loaded_for: int) -> None:
        super().__init__(replies, supports_tools=False)
        self._left = loaded_for

    def capabilities(self) -> ModelCapabilities:
        caps = super().capabilities()
        self._left -= 1
        if self._left >= 0:
            return caps
        return ModelCapabilities(name="", family="none", context_tokens=0)


def test_an_engine_error_mid_build_is_exit_4_and_a_sentence(
        tmp_path, scripted, no_input, capsys):
    # One call for the CLI's own check, one for the session's start: the
    # task boundary then sees nothing loaded.
    scripted([], llm=_Vanishing([PLAN], loaded_for=2))
    rc = cli.main(["build", "two", "--yes", "-p", str(tmp_path)])
    assert rc == 4
    assert "Traceback" not in capsys.readouterr().err


class _Interrupted(ScriptedLLM):
    """Ctrl-C arrives while the first file is being generated."""

    def complete(self, messages, **kw):
        if len(self.prompts) >= 1:
            raise KeyboardInterrupt
        return super().complete(messages, **kw)


def test_ctrl_c_stops_cleanly_and_still_reports(tmp_path, scripted,
                                                no_input, capsys):
    scripted([], llm=_Interrupted([PLAN], supports_tools=False))
    rc = cli.main(["build", "two", "--yes", "-p", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 1, "a stop before any file was built is not a success"
    assert "Stopping at the next safe point" in captured.err
    assert "[plan]" in captured.out, "the report is printed after a stop"


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def test_doctor_prints_the_summary_and_exits_zero(scripted, capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    for marker in ("installation summary", "core engine", "Python",
                   "interpreter", "Languages usable right now",
                   "No local model endpoint is answering"):
        assert marker in out, marker


def test_no_subcommand_is_doctor(scripted, capsys):
    assert cli.main([]) == 0
    assert "installation summary" in capsys.readouterr().out


def test_doctor_names_an_answering_endpoint(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect",
                        lambda *a, **k: ["http://127.0.0.1:8080"])
    cli.main(["doctor"])
    assert "answering right now: http://127.0.0.1:8080" in (
        capsys.readouterr().out)


# --------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------

def test_skills_list_deploy_new_round_trip(tmp_path, capsys):
    root = str(tmp_path)
    assert cli.main(["skills", "-p", root]) == 0
    assert "No skills deployed" in capsys.readouterr().out

    assert cli.main(["skills", "deploy", "-p", root]) == 0
    out = capsys.readouterr().out
    assert out.count("deployed  ") == len(STARTER_SKILLS)

    # Deploy never overwrites: the files belong to the project now.
    edited = tmp_path / ".ccoder" / "skills" / "10-house-style.md"
    edited.write_text(edited.read_text() + "\nOur own rule.\n")
    assert cli.main(["skills", "deploy", "-p", root]) == 0
    assert "kept      " in capsys.readouterr().out
    assert "Our own rule." in edited.read_text()

    assert cli.main(["skills", "list", "-p", root]) == 0
    assert "Active:" in capsys.readouterr().out

    assert cli.main(["skills", "new", "API Conventions", "-p", root]) == 0
    made = tmp_path / ".ccoder" / "skills" / "api-conventions.md"
    assert made.exists() and "name: api-conventions" in made.read_text()
    capsys.readouterr()

    assert cli.main(["skills", "new", "api-conventions", "-p", root]) == 2
    assert "already exists" in capsys.readouterr().err

    assert cli.main(["skills", "list", "--lang", "python", "-p",
                     root]) == 0
    assert "for a python session" in capsys.readouterr().out


def test_skills_new_needs_a_name(tmp_path, capsys):
    assert cli.main(["skills", "new", "-p", str(tmp_path)]) == 2
    assert "A name is needed" in capsys.readouterr().err


def test_skills_list_shows_what_did_not_load(tmp_path, capsys):
    bad = tmp_path / ".ccoder" / "skills" / "broken.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("---\nname: \n---\n", encoding="utf-8")
    assert cli.main(["skills", "-p", str(tmp_path)]) == 0
    assert "Not loaded:" in capsys.readouterr().out


# --------------------------------------------------------------------------
# history and resume
# --------------------------------------------------------------------------

def test_history_on_an_untouched_project(tmp_path, capsys):
    assert cli.main(["history", "-p", str(tmp_path)]) == 0
    assert "Nothing has been changed" in capsys.readouterr().out


def test_resume_with_no_sessions_says_so(tmp_path, scripted, capsys):
    assert cli.main(["resume", "-p", str(tmp_path)]) == 1
    assert "no previous sessions" in capsys.readouterr().out


def test_resume_lists_sessions_then_resumes_one(tmp_path, scripted,
                                                no_input, capsys):
    scripted([PLAN, ALPHA, BETA])
    assert cli.main(["build", "two", "--yes", "-p", str(tmp_path),
                     "--attempts", "1"]) == 0
    capsys.readouterr()
    (session_id,) = _sessions(tmp_path)

    assert cli.main(["resume", "-p", str(tmp_path)]) == 0
    assert session_id in capsys.readouterr().out

    # Everything verified the first time: resuming finds nothing to do and
    # asks the model for nothing.
    scripted([])
    rc = cli.main(["resume", session_id, "--yes", "-p", str(tmp_path)])
    assert rc == 0
    assert "2 file(s) already verified, 0 to go" in capsys.readouterr().out


# --------------------------------------------------------------------------
# defects, each pinned by the observation that found it
# --------------------------------------------------------------------------

AUTO_APPLY = "auto-apply is ON: every diff will be written without asking"


def test_yes_says_that_auto_apply_is_on(tmp_path, scripted, no_input,
                                        capsys):
    """§6.5: a host that auto-applies must SAY so. `--yes` said nothing."""
    scripted([PLAN, ALPHA, BETA])
    cli.main(["build", "two", "--yes", "-p", str(tmp_path),
              "--attempts", "1"])
    assert AUTO_APPLY in capsys.readouterr().out


def test_without_yes_or_when_nothing_is_written_it_is_not_claimed(
        tmp_path, scripted, capsys):
    scripted([PLAN])
    cli.main(["build", "two", "--preview", "--yes", "-p", str(tmp_path)])
    scripted([PLAN, ALPHA, BETA])
    cli.main(["build", "two", "--dry-run", "--yes", "-p", str(tmp_path),
              "--attempts", "1"])
    assert AUTO_APPLY not in capsys.readouterr().out


def test_a_build_that_built_nothing_is_not_a_success(tmp_path, scripted,
                                                     no_input, capsys):
    """`--budget -1` once printed "Stopped… nothing" and exited 0."""
    scripted([PLAN])
    rc = cli.main(["build", "two", "--yes", "-p", str(tmp_path),
                   "--budget", "0.00001"])
    assert rc == 1
    assert "Nothing was built" in capsys.readouterr().err


def test_the_exit_code_judges_the_last_outcome_per_file():
    """A module repaired against its test has two outcomes; the later one
    is the verdict, and a stale failure must not fail the run."""
    first = TaskOutcome(task_id="t1", path="src/a.py", ok=False)
    fixed = TaskOutcome(task_id="t1-fix", path="src/a.py", ok=True)
    other = TaskOutcome(task_id="t2", path="src/b.py", ok=True)
    assert cli._verdicts([first, other, fixed]) == [fixed, other]
    assert cli._exit_code([first, other, fixed]) == 0
    assert cli._exit_code([fixed, first]) == 1
    assert cli._exit_code([]) == 1


@pytest.mark.parametrize("flag,value,words", [
    ("--max-files", "-3", "1 or more"),
    ("--max-files", "0", "1 or more"),
    ("--attempts", "0", "1 or more"),
    ("--max-tokens", "0", "1 or more"),
    ("--budget", "-1", "0 or more"),
    ("--temperature", "-0.5", "0 or more"),
    ("--max-files", "twelve", "whole number"),
])
def test_numbers_that_cannot_mean_anything_are_refused(
        tmp_path, scripted, capsys, flag, value, words):
    """`--max-files -3` once printed "only the first -3 were kept"."""
    with pytest.raises(SystemExit) as stop:
        cli.main(["build", "two", "-p", str(tmp_path), flag, value])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert flag in err and words in err
    assert scripted.calls == []


def test_skills_new_with_a_blank_name_is_refused(tmp_path, capsys):
    """`skills new "  "` once created `.ccoder/skills/.md`."""
    assert cli.main(["skills", "new", "  ", "-p", str(tmp_path)]) == 2
    assert "no letters or digits" in capsys.readouterr().err
    assert not (tmp_path / ".ccoder" / "skills" / ".md").exists()
    assert cli.main(["skills", "new", "!!!", "-p", str(tmp_path)]) == 2


@pytest.mark.parametrize("argv", [
    ["build", "two"], ["build", "two", "--preview"], ["history"],
    ["resume"], ["skills"], ["skills", "deploy"],
])
def test_a_missing_project_folder_is_refused_not_created(
        tmp_path, scripted, capsys, argv):
    """A typo in --project silently created a whole new tree."""
    typo = tmp_path / "porject"
    rc = cli.main([*argv, "-p", str(typo)])
    assert rc == 2
    assert not typo.exists()
    err = capsys.readouterr().err
    assert "There is no folder at" in err and "--create" in err
    assert scripted.calls == []


def test_create_makes_the_project_folder_and_says_so(tmp_path, scripted,
                                                     capsys):
    fresh = tmp_path / "new-project"
    scripted([PLAN])
    rc = cli.main(["build", "two", "--preview", "--create", "-p",
                   str(fresh)])
    assert rc == 0 and fresh.is_dir()
    assert f"Created a new project folder at {fresh}" in (
        capsys.readouterr().out)


def test_a_file_where_the_project_should_be_is_refused(tmp_path, capsys):
    target = tmp_path / "a-file"
    target.write_text("x")
    assert cli.main(["history", "-p", str(target), "--create"]) == 2
    assert "is a file, not a folder" in capsys.readouterr().err


@pytest.mark.parametrize("url,words", [
    ("ftp://127.0.0.1:8080", "http:// or https://"),
    ("127.0.0.1:8080", "has no scheme"),
    ("localhost:8080", "has no scheme"),
    ("http://127.0.0.1:8080/v1/", "/v1"),
    ("http://127.0.0.1:8080/v1", "/v1"),
    ("http://:8080", "names no host"),
    ("http://127.0.0.1:eighty", "port"),
])
def test_a_malformed_url_is_named_before_any_model_is_asked(
        tmp_path, scripted, capsys, url, words):
    """All of these used to become "No model is loaded at that endpoint"."""
    rc = cli.main(["build", "two", "-p", str(tmp_path), "--url", url])
    assert rc == 2
    err = capsys.readouterr().err
    assert words in err and "No model is loaded" not in err
    assert scripted.calls == []


def test_the_v1_suffix_refusal_names_the_url_to_use(tmp_path, scripted,
                                                    capsys):
    cli.main(["build", "two", "-p", str(tmp_path),
              "--url", "http://127.0.0.1:8080/v1/"])
    assert "--url http://127.0.0.1:8080" in capsys.readouterr().err


def test_a_url_with_a_path_prefix_is_still_accepted(tmp_path, scripted):
    scripted([PLAN])
    assert cli.main(["build", "two", "--preview", "-p", str(tmp_path),
                     "--url", "https://gpu-box.lan:8443/llm/"]) == 0


def test_a_remote_url_without_remote_names_the_flag(tmp_path, monkeypatch,
                                                    capsys):
    def refuse(*a, **k):
        raise ConfigurationError(
            "openai_compatible would send data off this machine, and "
            "remote mode is off. Nothing was sent.")
    monkeypatch.setattr(cli, "make_provider", refuse)
    rc = cli.main(["build", "two", "-p", str(tmp_path),
                   "--url", "https://api.example.com"])
    assert rc == 2
    assert "--remote openai_compatible" in capsys.readouterr().err


# -- --remote --------------------------------------------------------------

class _RemoteScripted(ScriptedLLM):
    def capabilities(self) -> ModelCapabilities:
        caps = super().capabilities()
        return ModelCapabilities(
            name=caps.name, family=caps.family,
            context_tokens=caps.context_tokens, supports_tools=False,
            is_remote=True)


@pytest.fixture
def fake_remote(monkeypatch):
    """A remote provider registered the way a real one is, recording what
    it was built with. Its presence in the registry is scoped to one test.
    """
    class Fake:
        built: list[dict] = []
        replies: list[str] = [PLAN, ALPHA, BETA]

    def ctor(**kwargs):
        if not kwargs["gate"].allowed("fakeremote"):
            raise ConfigurationError("fakeremote was not enabled.")
        Fake.built.append(kwargs)
        return _RemoteScripted(list(Fake.replies), supports_tools=False)

    monkeypatch.setitem(providers._REGISTRY, "fakeremote",
                        (ctor, True, "a remote provider for the tests"))
    return Fake


def test_remote_turns_the_gate_on_for_this_run_and_shows_the_banner(
        tmp_path, fake_remote, no_input, monkeypatch, capsys):
    """`--remote` was documented and did not exist, so a remote URL could
    only ever produce "Turn on remote mode for this session"."""
    monkeypatch.setattr(cli, "detect", lambda *a, **k: [])
    rc = cli.main(["build", "two", "--yes", "--remote", "fakeremote",
                   "--model", "big-one", "-p", str(tmp_path),
                   "--attempts", "1"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "REMOTE MODE IS ON for fakeremote" in err
    (kwargs,) = fake_remote.built
    assert kwargs["model"] == "big-one"
    # Built through the SESSION, so the budget and journal are this run's.
    for bound in ("gate", "budget", "events", "journal"):
        assert kwargs[bound] is not None, bound


def test_remote_openai_compatible_is_built_by_the_session(
        tmp_path, monkeypatch, no_input, capsys):
    seen: list[tuple] = []

    def remote_provider(self, name, **kwargs):
        seen.append((name, kwargs, self.gate.allowed(name)))
        return ScriptedLLM([PLAN], supports_tools=False)

    monkeypatch.setattr(Session, "remote_provider", remote_provider)
    monkeypatch.setattr(cli, "make_provider", lambda *a, **k: pytest.fail(
        "the ungated constructor must not be used with --remote"))
    rc = cli.main(["build", "two", "--preview", "--remote",
                   "openai_compatible", "--url", "https://api.example.com",
                   "--model", "m", "-p", str(tmp_path)])
    assert rc == 0
    ((name, kwargs, enabled),) = seen
    assert name == "openai_compatible" and enabled
    assert kwargs == {"base_url": "https://api.example.com", "model": "m"}


@pytest.mark.parametrize("name,words", [
    ("nosuch", "There is no provider called"),
    ("local_llamacpp", "runs on this machine"),
])
def test_remote_refuses_a_name_that_is_not_a_remote_provider(
        tmp_path, scripted, capsys, name, words):
    rc = cli.main(["build", "two", "--remote", name, "-p", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2 and words in err
    assert "REMOTE MODE IS ON" not in err, "refused BEFORE the gate opens"


def test_remote_without_a_key_is_a_sentence(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = cli.main(["build", "two", "--remote", "anthropic", "-p",
                   str(tmp_path)])
    assert rc == 2
    assert "No API key was given for anthropic" in capsys.readouterr().err


# -- resume ----------------------------------------------------------------

def _stopped_before_building(tmp_path, scripted) -> str:
    scripted([PLAN])
    cli.main(["build", "two", "--yes", "-p", str(tmp_path),
              "--budget", "0.00001"])
    (session_id,) = _sessions(tmp_path)
    return session_id


def test_resume_of_an_unknown_session_is_a_sentence(tmp_path, scripted,
                                                    capsys):
    """A bogus id was a FileNotFoundError traceback."""
    rc = cli.main(["resume", "cc-nope", "-p", str(tmp_path)])
    assert rc == 2
    assert "no session called cc-nope" in capsys.readouterr().err
    assert scripted.calls == [], "no model is asked about a missing session"


def test_resume_with_no_model_loaded_is_exit_3(tmp_path, scripted,
                                               no_input, capsys):
    """A down model was a NoModelLoadedError traceback."""
    session_id = _stopped_before_building(tmp_path, scripted)
    capsys.readouterr()
    scripted([], llm=NullLLM())
    assert cli.main(["resume", session_id, "-p", str(tmp_path)]) == 3
    assert "No model is loaded" in capsys.readouterr().err


def test_resume_builds_what_was_left_and_exits_zero(tmp_path, scripted,
                                                   no_input, capsys):
    session_id = _stopped_before_building(tmp_path, scripted)
    scripted([ALPHA, BETA])
    rc = cli.main(["resume", session_id, "--yes", "-p", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert AUTO_APPLY in out
    assert "return 2" in (tmp_path / "src" / "beta.py").read_text()


def test_resume_that_fails_says_so_in_the_exit_code(tmp_path, scripted,
                                                   monkeypatch, capsys):
    """Resume returned 0 whatever the resumed build did."""
    session_id = _stopped_before_building(tmp_path, scripted)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
    scripted([ALPHA, BETA])
    assert cli.main(["resume", session_id, "-p", str(tmp_path)]) == 1


def test_resume_that_hits_an_engine_error_is_exit_4(tmp_path, scripted,
                                                   no_input, capsys):
    session_id = _stopped_before_building(tmp_path, scripted)
    scripted([], llm=_Vanishing([ALPHA], loaded_for=1))
    rc = cli.main(["resume", session_id, "--yes", "-p", str(tmp_path)])
    assert rc == 4
    assert "Traceback" not in capsys.readouterr().err


def test_resume_with_remote_uses_the_resumed_sessions_gate(
        tmp_path, scripted, fake_remote, no_input, capsys):
    session_id = _stopped_before_building(tmp_path, scripted)
    fake_remote.replies = [ALPHA, BETA]      # a resumed session plans nothing
    rc = cli.main(["resume", session_id, "--yes", "--remote", "fakeremote",
                   "-p", str(tmp_path)])
    assert rc == 0
    assert "REMOTE MODE IS ON for fakeremote" in capsys.readouterr().err
    (kwargs,) = fake_remote.built
    assert kwargs["journal"] is not None and kwargs["gate"].active


# -- doctor ----------------------------------------------------------------

def test_doctor_lists_remote_providers_as_remote(scripted, capsys):
    """Five REMOTE providers were printed under "local providers"."""
    cli.main(["doctor"])
    out = capsys.readouterr().out
    def names(heading: str) -> set[str]:
        line = next(ln for ln in out.splitlines() if heading in ln)
        listed = line.split(heading, 1)[1].split("—", 1)[0]
        return {n.strip() for n in listed.split(",")}

    remote = {"anthropic", "google", "mistral", "openai", "openrouter"}
    assert names("local providers") == {"openai_compatible",
                                        "local_llamacpp"}
    assert names("remote providers") == remote
    assert "--remote" in out


def _interpreter(path: Path, target: Path | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if target is None:
        path.write_text("")
    else:
        path.symlink_to(target)
    return str(path)


def test_provenance_of_a_copied_venv_inside_the_clone(tmp_path):
    """Any interpreter inside the clone was "fetched into this clone
    (.python/)" — including a plain `.venv` made with --copies."""
    clone = tmp_path / "clone"
    exe = _interpreter(clone / ".venv" / "bin" / "python")
    _exe, label = cli.interpreter_provenance(exe, str(clone))
    assert label.startswith("the clone's .venv")
    assert ".python/" not in label


def test_provenance_of_a_venv_over_the_fetched_python(tmp_path):
    clone = tmp_path / "clone"
    real = Path(_interpreter(clone / ".python" / "cpython" / "bin" / "py3"))
    exe = _interpreter(clone / ".venv" / "bin" / "python", real)
    _exe, label = cli.interpreter_provenance(exe, str(clone))
    assert label.startswith("the clone's .venv")
    assert "fetched by the installer into .python/" in label


def test_provenance_of_a_venv_over_the_system_python(tmp_path):
    system = Path(_interpreter(tmp_path / "usr" / "bin" / "python3.11"))
    clone = tmp_path / "clone"
    exe = _interpreter(clone / ".venv" / "bin" / "python", system)
    _exe, label = cli.interpreter_provenance(exe, str(clone))
    assert "the system Python" in label and str(system) in label


@pytest.mark.parametrize("parts,words", [
    ((".python", "bin", "python3"), "fetched by the installer into .python/"),
    ((".tools", "bin", "python3"), "fetched by uv into .tools/"),
    (("vendor", "python3"), "inside this clone"),
])
def test_provenance_outside_a_venv(tmp_path, parts, words):
    clone = tmp_path / "clone"
    exe = _interpreter(clone.joinpath(*parts))
    assert words in cli.interpreter_provenance(exe, str(clone))[1]


def test_provenance_of_the_system_python(tmp_path):
    exe = _interpreter(tmp_path / "usr" / "bin" / "python3")
    label = cli.interpreter_provenance(exe, str(tmp_path / "clone"))[1]
    assert label == "the system Python"


# --------------------------------------------------------------------------
# state that outlives the process
# --------------------------------------------------------------------------

def test_history_in_a_second_process_shows_what_a_build_did(
        tmp_path, scripted, no_input):
    """`ccoder history` always said "Nothing has been changed" while
    .cc_snapshots/0001-t1 sat on disk: the log lived in MemoryStorage and
    died with the build. Asked from a genuinely separate process here."""
    import os
    import subprocess

    scripted([PLAN, ALPHA, BETA])
    assert cli.main(["build", "two", "--yes", "-p", str(tmp_path),
                     "--attempts", "1"]) == 0
    repo = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=repo)
    later = subprocess.run(
        [sys.executable, "-m", "cognitive_coder.cli", "history", "-p",
         str(tmp_path)], capture_output=True, text=True, env=env,
        timeout=60)
    assert later.returncode == 0, later.stderr
    rows = [ln.split() for ln in later.stdout.splitlines() if ln.strip()]
    seqs = [r[0] for r in rows if r[0].isdigit()]
    assert seqs[:2] == ["1", "2"], later.stdout
    assert "src/alpha.py" in later.stdout and "SEALED" in later.stdout


def test_a_second_build_continues_the_sequence(tmp_path, scripted,
                                               no_input, capsys):
    """Per-process storage restarted the counter at 1 on every run, so the
    log's numbering no longer proved it was linear (M25)."""
    scripted([PLAN, ALPHA, BETA])
    cli.main(["build", "two", "--yes", "-p", str(tmp_path),
              "--attempts", "1"])
    scripted(["src/gamma.py — the third thing\n",
              '```python\ndef gamma():\n    """Third."""\n    return 3\n```'])
    cli.main(["build", "one more", "--yes", "-p", str(tmp_path),
              "--attempts", "1"])
    capsys.readouterr()
    cli.main(["history", "-p", str(tmp_path)])
    seqs = [int(ln.split()[0]) for ln in capsys.readouterr().out.splitlines()
            if ln.strip() and ln.split()[0].isdigit()]
    assert seqs == sorted(set(seqs)) and seqs[-1] >= 3, seqs


def test_a_corrupt_history_log_is_a_sentence(tmp_path, capsys):
    state = tmp_path / ".cc_state"
    state.mkdir()
    (state / "cognitive_coder.patcher.log.json").write_text("[{half")
    assert cli.main(["history", "-p", str(tmp_path)]) == 4
    err = capsys.readouterr().err
    assert "not valid JSON" in err and "Traceback" not in err


# --------------------------------------------------------------------------
# the embedding examples the docs print — the first code most readers run
# --------------------------------------------------------------------------

def _readme_example() -> str:
    text = (Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8")
    section = text.split("## Ten lines of embedding", 1)[1]
    return section.split("```python\n", 1)[1].split("\n```", 1)[0]


def _docstring_example() -> str:
    import textwrap

    import cognitive_coder

    doc = cognitive_coder.__doc__ or ""
    body = doc.split("Ten-line embedding example", 1)[1]
    body = body.split("\n\n", 1)[1].split("\nThe meta-lesson", 1)[0]
    return textwrap.dedent(body)


@pytest.mark.parametrize("source", [_readme_example, _docstring_example],
                         ids=["README", "package docstring"])
def test_the_embedding_example_runs_as_printed(source, tmp_path,
                                               monkeypatch, capsys):
    """Both printed examples scripted ONE reply, which the planner took;
    the build then died with "ScriptedLLM ran out of replies"."""
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    code = source()
    assert "ScriptedLLM" in code and len(code.splitlines()) <= 14
    exec(compile(code, "<embedding example>", "exec"), {})   # noqa: S102
    out = capsys.readouterr().out
    assert "[build 1/1] greet.py" in out and "→ committed" in out
