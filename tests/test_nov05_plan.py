# SPDX-License-Identifier: Apache-2.0
"""The 2026-11-05 build (docs/NEXT_PLAN_20261105.md): what is pinned.

  §8 H1  the context budget is the engine's loaded n_ctx, never the model
         card's — a header that says 1,048,576 with an engine at 8,192
         budgets 8,192 and says so in the journal;
  §2c    the bleed guard — a planted `<think>` block (lower-case, spaced,
         mid-line, bracketed) never reaches disk, the refusal is a
         sentence in the journal, and a file that MENTIONS the tag in a
         string is still writable; wall 1 strips the reasoning off every
         reply and journals it (sized, hashed, one line);
  §2a/b  reasoning levels — off is byte-for-byte the 1.0 behaviour; auto
         gives plan high / file low / retry medium / diagnosis high; a
         port that takes `reasoning=` gets the level, a 1.0 port gets the
         bool switch; the budget taxes the level and caps the tax;
  §1     the plan files — task_plan.md validates, the phases follow the
         dependency order, the Status line is computed, the header is
         bounded, findings are restored from the journal on Continue;
  §3     retry this file — queued from another thread, applied at the
         boundary, undo-first by default, the diagnostics seeded, the
         model swap asked of the host, journaled as `retry`;
  §8 H3  a module that built and passed proposes a skill under
         proposed/, which is never loaded until kept.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
    context,
    planfile,
    profiles,
    reasoning,
    skills,
)
from cognitive_coder import journal as journal_mod
from cognitive_coder import patcher as patcher_mod
from cognitive_coder.types import (  # noqa: E402
    Edit,
    Message,
    ModelCapabilities,
    Plan,
    ReasoningSupport,
    Task,
)

PLAN = ("src/alpha.py — the first thing\nsrc/beta.py — the second thing\n")
ALPHA = '```python\ndef alpha():\n    """First."""\n    return 1\n```'
BETA = '```python\ndef beta():\n    """Second."""\n    return 2\n```'
SKELETON = ('```python\n# file: src/alpha.py\ndef alpha() -> int:\n'
            '    """First."""\n    raise NotImplementedError\n```\n'
            '```python\n# file: src/beta.py\ndef beta() -> int:\n'
            '    """Second."""\n    raise NotImplementedError\n```')


def _host(tmp_path, replies, llm=None):
    return Host(llm=llm or ScriptedLLM(replies, supports_tools=False),
                fs=LocalFileSystem(str(tmp_path)), exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=RecordingEvents(), approval=AutoApprove())


def _rows(tmp_path, sid) -> list[dict]:
    return list(journal_mod.read_jsonl(LocalFileSystem(str(tmp_path)),
                                       f".cc_journal/{sid}.jsonl"))


# --------------------------------------------------------------------------
# §8 H1 — the context budget is the engine's, never the model card's
# --------------------------------------------------------------------------

class _HeaderLiesLLM(ScriptedLLM):
    """A model whose card says 1M while the engine loaded 8,192."""

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        import dataclasses
        return dataclasses.replace(base, context_tokens=8192,
                                   trained_context_tokens=1_048_576)


class _CardSmallerLLM(ScriptedLLM):
    """The other way round: the engine loaded 32k, the card says 4k."""

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        import dataclasses
        return dataclasses.replace(base, context_tokens=32768,
                                   trained_context_tokens=4096)


def test_h1_budget_is_the_engines_loaded_context_not_the_cards():
    llm = _HeaderLiesLLM([])
    budget = context.measure_budget(llm)
    assert budget.context_tokens == 8192
    assert "not budgeted" in budget.note and "1,048,576" in budget.note
    caps = llm.capabilities()
    assert caps.budget_context_tokens == 8192


def test_h1_the_smaller_of_the_two_wins_when_the_card_is_smaller():
    llm = _CardSmallerLLM([])
    budget = context.measure_budget(llm)
    assert budget.context_tokens == 4096
    assert "smaller" in budget.note


def test_h1_the_journal_says_which_number_was_budgeted(tmp_path):
    llm = _HeaderLiesLLM([PLAN, SKELETON, ALPHA, BETA], supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1))
    session.run("two small modules")
    rows = [r for r in _rows(tmp_path, session.id)
            if r.get("event") == "budget"
            and (r.get("data") or {}).get("what") == "context_rule"]
    assert rows, "the context rule was not written to the journal"
    data = rows[0]["data"]
    assert data["loaded"] == 8192 and data["trained"] == 1_048_576
    assert data["budgeted"] == 8192
    assert "engine's loaded context" in data["sentence"]


# --------------------------------------------------------------------------
# §2c — the bleed guard, wall 2 (the patcher) and wall 1 (the loop)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("planted", [
    "<think>\nlet me see\n</think>\n",
    "<THINK>\nlet me see\n</THINK>\n",
    "  < think >\nlet me see\n</ think >\n",
    "<think>Let me see, the user wants a function\n",
    "[THINK]\nlet me see\n[/THINK]\n",
    "<|channel|>analysis<|message|>let me see\n",
    "<reasoning>\nhmm\n</reasoning>\n",
])
def test_wall_2_refuses_a_hunk_carrying_a_thinking_delimiter(planted):
    fs = MemoryFileSystem()
    p = patcher_mod.Patcher(fs, MemoryStorage(), AutoApprove())
    tx = p.begin("t1")
    code = planted + "def f():\n    return 1\n"
    [res] = tx.apply([Edit(path="src/f.py", kind="create", new=code)],
                     approve=False)
    assert not res.ok
    assert "thinking delimiter" in res.reason and "Nothing was written" in res.reason
    assert not fs.exists("src/f.py")
    assert p.bled and p.bled[0]["path"] == "src/f.py"
    tx.commit()


def test_wall_2_lets_code_that_mentions_the_tag_through():
    fs = MemoryFileSystem()
    p = patcher_mod.Patcher(fs, MemoryStorage(), AutoApprove())
    tx = p.begin("t1")
    code = ('import re\n'
            'PAT = re.compile(r"<(think|thinking)>.*?</\\1>")\n'
            's = "<think>" + "x"\n'
            'print("</think>")\n')
    [res] = tx.apply([Edit(path="src/g.py", kind="create", new=code)],
                     approve=False)
    assert res.ok, res.reason
    assert fs.exists("src/g.py")
    tx.commit()


def test_wall_2_judges_the_hosts_own_delimiters_too():
    fs = MemoryFileSystem()
    p = patcher_mod.Patcher(fs, MemoryStorage(), AutoApprove())
    p.bleed_delimiters = ("<|begin_of_thought|>", "<|end_of_thought|>")
    tx = p.begin("t1")
    [res] = tx.apply([Edit(path="src/h.py", kind="create",
                           new="<|begin_of_thought|>\nhmm\n"
                               "<|end_of_thought|>\ndef h(): pass\n")],
                     approve=False)
    assert not res.ok and not fs.exists("src/h.py")
    tx.commit()


def test_wall_2_does_not_blame_an_edit_for_a_marker_already_in_the_file():
    """A file that already held a delimiter (the operator's own fixture,
    say) is not refused for keeping it — only the NEW text is judged."""
    fs = MemoryFileSystem({"src/fx.py": b"<think>\nold\n</think>\nx = 1\n"})
    p = patcher_mod.Patcher(fs, MemoryStorage(), AutoApprove())
    tx = p.begin("t1")
    [res] = tx.apply([Edit(path="src/fx.py", kind="replace", old="x = 1",
                           new="x = 2")], approve=False)
    assert res.ok, res.reason
    tx.commit()


def test_wall_1_strips_the_reasoning_and_journals_it(tmp_path):
    """A reply with a `<think>` block: the code is written, the reasoning
    is a `reasoning` journal event (size, hash, one line), never on disk."""
    thinking_alpha = ("<think>\nThe stub asks for alpha() returning an int. "
                      "I will return 1.\n</think>\n" + ALPHA)
    llm = ScriptedLLM([PLAN, SKELETON, thinking_alpha, BETA],
                      supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1))
    session.run("two small modules")
    text = (tmp_path / "src" / "alpha.py").read_text(encoding="utf-8")
    assert "<think" not in text.lower()
    rows = [r for r in _rows(tmp_path, session.id)
            if r.get("event") == "reasoning"]
    assert rows, "the reasoning was not journaled"
    data = rows[0]["data"]
    assert data["chars"] > 20 and len(data["sha256"]) == 64
    assert "stub asks for alpha" in data["summary"]
    assert rows[0]["task"] == "src/alpha.py"
    # the journal holds a summary and a hash — never the reasoning itself
    assert "I will return 1" not in json.dumps(rows[0])


def test_a_host_that_splits_the_reasoning_itself_is_journaled_the_same(tmp_path):
    class _Splitting(ScriptedLLM):
        def complete(self, messages, **kw):
            c = super().complete(messages, **kw)
            if c.text.startswith("SPLIT:"):
                import dataclasses
                return dataclasses.replace(
                    c, text=c.text[len("SPLIT:"):],
                    reasoning="I thought about alpha for a while.")
            return c
    llm = _Splitting([PLAN, SKELETON, "SPLIT:" + ALPHA, BETA],
                     supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1))
    session.run("two small modules")
    rows = [r for r in _rows(tmp_path, session.id)
            if r.get("event") == "reasoning"]
    assert rows and "thought about alpha" in rows[0]["data"]["summary"]


# --------------------------------------------------------------------------
# §2a/b — reasoning levels
# --------------------------------------------------------------------------

def test_policy_off_is_the_1_0_behaviour():
    p = reasoning.Policy.from_config(SessionConfig())
    assert not p.enabled
    assert p.level("generate") is False
    assert p.level("diagnose") is False
    assert "off" in p.describe()


def test_policy_auto_gives_the_phase_defaults_and_medium_on_retry():
    p = reasoning.Policy.from_config(SessionConfig(reasoning="auto"))
    assert p.level("plan") == "high"
    assert p.level("interfaces") == "high"
    assert p.level("generate") == "low"
    assert p.level("generate", attempt=2) == "medium"
    assert p.level("diagnose") == "high"
    assert p.level("review") == "medium"
    # a grammar binds from the first token: never
    assert p.level("structured") is False
    assert p.level("adjudicate") is False


def test_operator_overrides_win_and_one_level_everywhere_works():
    p = reasoning.Policy.from_config(SessionConfig(
        reasoning="auto", reasoning_phases={"generate": "off",
                                            "diagnose": "xhigh"}))
    assert p.level("generate") is False
    assert p.level("generate", attempt=3) is False      # an instruction
    assert p.level("diagnose") == "xhigh"
    assert p.level("plan") == "high"
    q = reasoning.Policy.from_config(SessionConfig(reasoning="high"))
    assert q.level("generate") == "high" and q.level("review") == "high"
    assert q.level("structured") is False


def test_resolve_against_what_the_model_can_do():
    none = ReasoningSupport()
    assert reasoning.resolve("high", none).effective == "off"
    assert "no thinking control" in reasoning.resolve("high", none).note
    switch = ReasoningSupport(kind="switch")
    assert reasoning.resolve("high", switch).effective == "on"
    levels = ReasoningSupport(kind="levels", levels=("low", "medium", "high"))
    assert reasoning.resolve("xhigh", levels).effective == "high"
    assert reasoning.resolve("medium", levels).effective == "medium"
    always = ReasoningSupport(kind="always", delimiters=("<think>", "</think>"))
    assert reasoning.resolve("low", always).effective == "on"
    assert reasoning.resolve("off", levels).effective == "off"


class _LevelPort(ScriptedLLM):
    """A port written to the new contract: `complete` takes `reasoning`."""

    def complete(self, messages, *, reasoning=None, thinking=None, **kw):
        kw["reasoning"] = reasoning
        kw["thinking"] = thinking
        return super().complete(messages, **kw)

    def capabilities(self) -> ModelCapabilities:
        import dataclasses
        return dataclasses.replace(
            super().capabilities(),
            reasoning=ReasoningSupport(kind="levels",
                                       levels=("low", "medium", "high"),
                                       delimiters=("<think>", "</think>")))


class _SwitchPort(ScriptedLLM):
    """A port written to the 2026-10-27 contract: a bool `thinking`, the
    samplers, and NO **kw — so `reasoning=` must never be sent to it."""

    def complete(self, messages, *, tools=(), temperature=0.15,
                 max_tokens=2048, stop=None, grammar=None, seed=None,
                 cancel=None, sampling=None, thinking=None):
        return super().complete(messages, tools=tools, temperature=temperature,
                                max_tokens=max_tokens, seed=seed,
                                sampling=sampling, thinking=thinking)


class _PlainPort(ScriptedLLM):
    """A port written to the 1.0 signature: neither switch nor level."""

    def complete(self, messages, *, tools=(), temperature=0.15,
                 max_tokens=2048, stop=None, grammar=None, seed=None,
                 cancel=None):
        return super().complete(messages, tools=tools, temperature=temperature,
                                max_tokens=max_tokens, seed=seed)


def test_call_kwargs_sends_a_level_where_the_port_takes_one():
    prof = profiles.builtin_for("qwen3-8b")           # has a switch
    port = _LevelPort([])
    out = profiles.call_kwargs(port, prof, thinking="high")
    assert out["reasoning"] == "high" and out["thinking"] is True
    out = profiles.call_kwargs(port, prof, thinking="off")
    assert out.get("thinking") is False and "reasoning" not in out
    # the 1.0 port: the level becomes the switch
    legacy = _SwitchPort([])
    out = profiles.call_kwargs(legacy, prof, thinking="medium")
    assert out["thinking"] is True and "reasoning" not in out
    # a port that takes neither sees nothing but the samplers
    plain = _PlainPort([])
    assert "reasoning" not in profiles.call_kwargs(plain, prof, thinking="high")
    assert "thinking" not in profiles.call_kwargs(plain, prof, thinking="high")
    # a level never appends the text off-switch; "off" does
    msgs = [Message(role="user", content="write it")]
    assert profiles.soft_switch(msgs, prof, thinking="high")[0].content == "write it"
    assert profiles.soft_switch(msgs, prof, thinking="off")[0].content.endswith("/no_think")


def test_the_levels_reach_the_port_per_phase(tmp_path):
    llm = _LevelPort([PLAN, SKELETON, ALPHA, BETA], supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1, reasoning="auto",
                                           review_after_build=False,
                                           mutation_check=False))
    session.run("two small modules")
    levels = [c.get("reasoning") for c in llm.calls]
    # the plan and the interfaces think high; the files low
    assert levels[0] == "high" and levels[1] == "high"
    assert "low" in levels[2:]
    assert all(lv in ("high", "low", "medium", None) for lv in levels)
    rows = [r for r in _rows(tmp_path, session.id)
            if r.get("event") == "profile"
            and (r.get("data") or {}).get("what") == "reasoning"]
    assert rows and rows[0]["data"]["policy"] == "auto"
    assert rows[0]["data"]["port_takes_levels"] is True


def test_policy_off_sends_exactly_what_it_did_before(tmp_path):
    llm = _LevelPort([PLAN, SKELETON, ALPHA, BETA], supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1,
                                           review_after_build=False,
                                           mutation_check=False))
    session.run("two small modules")
    assert all(c.get("reasoning") is None for c in llm.calls)


def test_the_reply_budget_grows_with_the_level_within_the_context():
    assert context.reasoning_tax(False, 16384) == 0
    assert context.reasoning_tax("off", 16384) == 0
    assert context.reasoning_tax(True, 16384) == 1024
    assert context.reasoning_tax("high", 16384) == 4096
    assert context.reasoning_tax("xhigh", 16384) == 4096     # a quarter
    assert context.reasoning_tax("xhigh", 4096) == 1024      # still a quarter
    b = context.measure_budget(ScriptedLLM([]), reasoning="medium")
    assert "2048 for reasoning (medium)" in b.note


# --------------------------------------------------------------------------
# §1 — working memory on disk
# --------------------------------------------------------------------------

def test_task_plan_validates_and_phases_follow_the_dependency_order(tmp_path):
    fs = LocalFileSystem(str(tmp_path))
    pf = planfile.PlanFiles(fs)
    plan = Plan(request="a game with a board and a renderer", tasks=(
        Task(id="t1", path="src/board.py", purpose="the board"),
        Task(id="t2", path="src/render.py", purpose="draws it",
             depends_on=("t1",)),
        Task(id="t3", path="src/main.py", purpose="runs it",
             depends_on=("t2",), status="pending"),
    ))
    plan = plan.replace(plan.tasks[0].with_status("done", attempts=1))
    plan = plan.replace(plan.tasks[1].with_status("active"))
    text = pf.render(plan)
    assert planfile.validate(text) == []
    assert "### Phase 1" in text and "### Phase 3" in text
    assert text.index("### Phase 1") < text.index("### Phase 2") < text.index("### Phase 3")
    assert "**Status:** complete" in text.split("### Phase 2")[0]
    assert "**Status:** in_progress" in text.split("### Phase 2")[1].split("### Phase 3")[0]
    assert "**Status:** pending" in text.split("### Phase 3")[1]
    assert "**Next step:** src/render.py" in text
    assert "**Current phase:** Phase 2" in text
    assert (tmp_path / ".ccoder" / "plan" / "task_plan.md").exists()


def test_a_malformed_task_plan_is_named_not_trusted():
    assert planfile.validate("") == ["the file is empty"]
    bad = ("# Task plan\n**Goal:** x\n**Next step:** y\n"
           "**Current phase:** z\n### Phase 1 — a\n**Status:** complete\n"
           "**Status:** pending\n- [ ] a.py — a\n### Phase 3 — b\n"
           "**Status:** maybe\n")
    problems = planfile.validate(bad)
    assert any("2 `**Status:**`" in p for p in problems)
    assert any("numbered" in p for p in problems)
    assert any("'maybe'" in p for p in problems)
    assert any("lists no files" in p for p in problems)


def test_the_header_is_bounded_and_keeps_the_newest_findings(tmp_path):
    pf = planfile.PlanFiles(LocalFileSystem(str(tmp_path)))
    pf.render(Plan(request="goal " * 50, tasks=(
        Task(id="t1", path="a.py", purpose="a"),)))
    for i in range(12):
        pf.note(f"finding number {i} " + "detail " * 10, source="engine")
    big = pf.header(budget_tokens=600)
    small = pf.header(budget_tokens=120)
    assert big.startswith("[WHERE YOU ARE]")
    assert "finding number 11" in big
    assert len(small) // 4 <= 130
    assert "finding number 0" not in big          # only the last few ride
    assert (tmp_path / ".ccoder" / "plan" / "findings.md").read_text(
        encoding="utf-8").count("finding number") == 12


def test_the_plan_files_are_written_during_a_build_and_the_header_is_in_the_prompt(tmp_path):
    llm = ScriptedLLM([PLAN, SKELETON, ALPHA, BETA], supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1,
                                           review_after_build=False,
                                           mutation_check=False))
    session.run("two small modules")
    plan_dir = tmp_path / ".ccoder" / "plan"
    text = (plan_dir / "task_plan.md").read_text(encoding="utf-8")
    assert planfile.validate(text) == []
    assert "[x] src/alpha.py" in text and "[x] src/beta.py" in text
    assert "finished" in text or "nothing left" in text
    progress = (plan_dir / "progress.md").read_text(encoding="utf-8")
    assert "src/alpha.py" in progress and "src/beta.py" in progress
    # the generation prompts carried the header in their VOLATILE tail
    gen_prompts = list(llm.prompts[2:])
    assert any("[WHERE YOU ARE]" in m.content and m.role == "user"
               for p in gen_prompts for m in p)
    assert not any("[WHERE YOU ARE]" in m.content and m.role == "system"
                   for p in gen_prompts for m in p)
    # the `note` events are in the journal, and the file is a view of them
    notes = [r for r in _rows(tmp_path, session.id) if r.get("event") == "note"]
    assert isinstance(notes, list)


def test_findings_come_back_from_the_journal_on_continue(tmp_path):
    session = Session(_host(tmp_path, [PLAN, SKELETON, ALPHA]),
                      config=SessionConfig(attempts=1))
    session.stop_after_current()
    session.run("two small modules")
    session.note("the board is zero-indexed, the spec says so", task="src/alpha.py")
    sid = session.id
    del session
    # the files are regenerated from the record, not trusted as found
    (tmp_path / ".ccoder" / "plan" / "findings.md").write_text("garbage")
    resumed = Session.resume(_host(tmp_path, [BETA]), sid,
                             config=SessionConfig(attempts=1,
                                                  review_after_build=False,
                                                  mutation_check=False))
    resumed.run("")
    findings = (tmp_path / ".ccoder" / "plan" / "findings.md").read_text(
        encoding="utf-8")
    assert "zero-indexed" in findings and "garbage" not in findings


def test_a_hand_edit_between_stop_and_continue_is_named_and_reindexed(tmp_path):
    session = Session(_host(tmp_path, [PLAN, SKELETON, ALPHA]),
                      config=SessionConfig(attempts=1))
    session.stop_after_current()
    session.run("two small modules")
    sid = session.id
    del session
    alpha = tmp_path / "src" / "alpha.py"
    alpha.write_text(alpha.read_text(encoding="utf-8")
                     + "\n\ndef alpha_extra():\n    return 3\n",
                     encoding="utf-8")
    resumed = Session.resume(_host(tmp_path, [BETA]), sid,
                             config=SessionConfig(attempts=1,
                                                  review_after_build=False,
                                                  mutation_check=False))
    resumed.run("")
    findings = (tmp_path / ".ccoder" / "plan" / "findings.md").read_text(
        encoding="utf-8")
    assert "changed since the build stopped" in findings
    assert "src/alpha.py" in findings
    assert resumed.codemap.resolves("alpha_extra")


# --------------------------------------------------------------------------
# §3 — retry this file
# --------------------------------------------------------------------------

class _RosterPort(ScriptedLLM):
    """A port that can load a second model, and says what it was asked."""

    def __init__(self, replies, **kw):
        super().__init__(replies, **kw)
        self.loaded = "small"
        self.loads: list[str] = []

    def roster(self):
        from cognitive_coder.types import ModelSlot
        return [ModelSlot(name="small", path="small.gguf", kind="chat",
                          loaded=self.loaded == "small"),
                ModelSlot(name="big", path="big.gguf", kind="chat",
                          loaded=self.loaded == "big")]

    def load_model(self, name, *, cancel=None):
        self.loads.append(name)
        self.loaded = name
        self._name = name
        return ""

    def capabilities(self):
        import dataclasses
        return dataclasses.replace(super().capabilities(), name=self.loaded)


def test_retry_after_a_build_rewrites_the_file_undo_first_with_the_asked_model(tmp_path):
    bad_beta = '```python\ndef beta():\n    """Second."""\n    return 2 +\n```'
    llm = _RosterPort([PLAN, SKELETON, ALPHA, bad_beta], supports_tools=False)
    # hybrid mode would re-open the failure with the other model on its
    # own; off, so the swap seen below is the retry's and nothing else's
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1, hybrid="off",
                                           review_after_build=False,
                                           mutation_check=False))
    session.run("two small modules")
    beta = tmp_path / "src" / "beta.py"
    assert not any(o.ok for o in session.outcomes if o.path == "src/beta.py")
    before_retry = beta.read_text(encoding="utf-8")
    # the operator, from another thread: retry beta with the big model
    err: list[str] = []
    t = threading.Thread(target=lambda: err.append(
        session.retry("src/beta.py", model="big", reasoning="high")))
    t.start()
    t.join()
    assert err == [""]
    assert session.pending_retries()[0]["path"] == "src/beta.py"
    llm._replies.extend([BETA])
    session._finished = False
    session.run("")                                   # drains the queue
    after = beta.read_text(encoding="utf-8")
    assert "return 2\n" in after and after != before_retry
    assert llm.loads == ["big"]                       # swapped at the boundary
    rows = [r for r in _rows(tmp_path, session.id) if r.get("event") == "retry"]
    assert rows and rows[0]["task"] == "src/beta.py"
    assert rows[0]["model"] == "big"          # a head field of the journal
    assert rows[0]["data"]["reasoning"] == "high"
    assert rows[0]["data"]["undo_first"] is True
    assert "rolled back" in rows[0]["data"]["rolled_back"]
    assert rows[0]["data"]["seeded"] is True
    final = [o for o in session.outcomes if o.path == "src/beta.py"][-1]
    assert final.ok


def test_retry_keep_current_does_not_roll_back(tmp_path):
    llm = ScriptedLLM([PLAN, SKELETON, ALPHA, BETA], supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1,
                                           review_after_build=False,
                                           mutation_check=False))
    session.run("two small modules")
    assert session.retry("src/beta.py", undo_first=False,
                         include_diagnostics=False) == ""
    assert session.retry("src/nothing.py") == "src/nothing.py is not in this build's plan"
    llm._replies.extend([BETA])
    session._finished = False
    session.run("")
    rows = [r for r in _rows(tmp_path, session.id) if r.get("event") == "retry"]
    assert rows and rows[0]["data"]["rolled_back"] == ""
    assert rows[0]["data"]["seeded"] is False


def test_retry_queued_mid_build_is_taken_at_the_next_boundary(tmp_path):
    class _Hooked(ScriptedLLM):
        def __init__(self, replies, **kw):
            super().__init__(replies, **kw)
            self.session = None
            self.fired = False

        def complete(self, messages, **kw):
            c = super().complete(messages, **kw)
            # once alpha has been written, ask for it again — from inside
            # the model call, which is as mid-build as it gets
            if not self.fired and c.text == ALPHA and self.session is not None:
                self.fired = True
                self.session.retry("src/alpha.py", undo_first=False,
                                   reason="asked again mid-build")
            return c
    llm = _Hooked([PLAN, SKELETON, ALPHA, BETA, ALPHA], supports_tools=False)
    session = Session(_host(tmp_path, [], llm=llm),
                      config=SessionConfig(attempts=1,
                                           review_after_build=False,
                                           mutation_check=False))
    llm.session = session
    session.run("two small modules")
    paths = [o.path for o in session.outcomes]
    assert paths.count("src/alpha.py") == 2
    rows = [r for r in _rows(tmp_path, session.id) if r.get("event") == "retry"]
    assert rows and rows[0]["data"]["reason"] == "asked again mid-build"


# --------------------------------------------------------------------------
# §8 H3 — a skill proposed from a module that built and passed
# --------------------------------------------------------------------------

def test_a_verified_module_proposes_a_skill_that_is_not_loaded_until_kept(tmp_path):
    fs = LocalFileSystem(str(tmp_path))
    fs.write("src/mod.py", "import json\nimport requests\n\n\ndef mod():\n"
             "    return json.dumps({})\n")
    from cognitive_coder.types import PhaseResult, RunResult, TaskOutcome
    task = Task(id="t1", path="src/mod.py", purpose="a module")
    outcome = TaskOutcome(task_id="t1", path="src/mod.py", ok=True,
                          verified=True,
                          result=RunResult(ok=True, phases=(
                              PhaseResult(name="test",
                                          argv=("python", "-m", "pytest"),
                                          ok=True),)))
    path = skills.propose_learned(fs, task=task, outcome=outcome,
                                  findings=["requests needs a timeout"])
    assert path == ".ccoder/skills/proposed/90-learned-mod.md"
    text = fs.read(path)
    assert "requests" in text and "json" not in text.split("## Libraries")[1].split("##")[0]
    assert "python -m pytest" in text and "`mod`" in text
    assert "requests needs a timeout" in text
    # not loaded from proposed/
    load = skills.load_skills(fs)
    assert not any("learned" in s.name for s in load.skills)
    assert skills.proposed(fs) == [path]
    kept = skills.keep_proposed(fs, path)
    assert kept == ".ccoder/skills/90-learned-mod.md"
    load = skills.load_skills(fs, lang="python")
    # it names a trigger (the module's stem), so it is HELD until a request
    # names it — a learned skill never rides into every prompt unasked
    assert any(s.name == "learned-mod" for s in load.held)
    assert any(s.name == "learned-mod"
               for s in load.for_request("extend mod with a cache").skills)
    assert skills.proposed(fs) == []
