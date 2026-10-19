# SPDX-License-Identifier: Apache-2.0
"""Hybrid mode: one model writes, another repairs, the first finishes.

Three phases, two swaps, roles by capability (docs/HYBRID.md, hybrid.py).
The host's side is the `ModelRoster` port; the double below plays two
models — a chat model and a canvas model — and remembers every load.
"""

from __future__ import annotations

from cognitive_coder import (
    AutoApprove,
    CanvasResult,
    Completion,
    Host,
    LocalFileSystem,
    MemoryStorage,
    ModelCapabilities,
    ModelSlot,
    RecordingEvents,
    Session,
    SessionConfig,
    SubprocessExec,
    hybrid,
)

# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------

CHAT = ModelSlot(name="devstral.gguf", kind="chat")
CANVAS = ModelSlot(name="diffusion-gemma.gguf", kind="canvas")
MYSTERY = ModelSlot(name="something.gguf")


def test_auto_picks_the_chat_model_to_write_and_the_canvas_one_to_repair():
    roles, reason = hybrid.plan_roles("auto", [CANVAS, CHAT], "devstral.gguf",
                                      loaded_kind="chat")
    assert roles is not None and not reason
    assert roles.order == "ar-first"
    assert roles.first == "devstral.gguf" and roles.second == \
        "diffusion-gemma.gguf"
    names = [(p.number, p.name, p.model) for p in roles.phases()]
    assert names == [(1, "write", "devstral.gguf"),
                     (2, "repair", "diffusion-gemma.gguf"),
                     (3, "finish", "devstral.gguf")]
    assert roles.swaps and "two swaps" in roles.describe()


def test_diffusion_first_runs_the_phases_the_other_way_round():
    roles, _ = hybrid.plan_roles("diffusion-first", [CHAT, CANVAS],
                                 "devstral.gguf", loaded_kind="chat")
    assert roles.first == "diffusion-gemma.gguf"
    assert roles.second == "devstral.gguf"
    assert [p.name for p in roles.phases()] == ["write", "repair", "finish"]


def test_off_and_no_roster_mean_one_model():
    assert hybrid.plan_roles("off", [CHAT, CANVAS], "devstral.gguf")[0] is None
    roles, reason = hybrid.plan_roles("auto", [], "devstral.gguf")
    assert roles is None and "no roster" in reason


def test_one_kind_only_is_three_passes_without_a_swap():
    roles, _ = hybrid.plan_roles("auto", [CHAT], "devstral.gguf",
                                 loaded_kind="chat")
    assert roles is not None and not roles.swaps
    assert "no swap" in roles.describe()


def test_unknown_kinds_are_not_guessed_in_auto():
    roles, reason = hybrid.plan_roles("auto", [MYSTERY, CHAT], "devstral.gguf",
                                      loaded_kind="chat")
    # the chat model is known; the mystery one is not a canvas model
    assert roles is not None and not roles.swaps
    roles, reason = hybrid.plan_roles("auto", [MYSTERY], "")
    assert roles is None and "unknown kind" in reason


def test_named_roles_win_over_kinds():
    roles, _ = hybrid.plan_roles("auto", [CHAT, CANVAS, MYSTERY], "devstral.gguf",
                                 loaded_kind="chat", repairer="something.gguf")
    assert roles.second == "something.gguf"
    roles, reason = hybrid.plan_roles("auto", [CHAT], "devstral.gguf",
                                      writer="nope.gguf")
    assert roles is None and "hybrid_writer" in reason


# --------------------------------------------------------------------------
# the double: two models behind one port
# --------------------------------------------------------------------------

class TwoModels:
    """A `LLMPort` + `CanvasLLM` + `ModelRoster` with two models in it.

    `chat` answers `complete` while the chat model is loaded; the canvas
    model answers `canvas_fill` with the next scripted fill and `complete`
    with the next scripted chat reply of ITS own list (notes, whole-file
    fallbacks). Every load is remembered.
    """

    name = "two-models"

    def __init__(self, chat_replies, canvas_fills, canvas_chat=(),
                 loaded="devstral.gguf", fail_load=""):
        self.chat_replies = list(chat_replies)
        self.canvas_fills = list(canvas_fills)
        self.canvas_chat = list(canvas_chat)
        self.loaded = loaded
        self.loads: list[str] = []
        self.fail_load = fail_load
        self.prompts: list = []
        self.requests: list = []

    # -- roster --------------------------------------------------------
    def roster(self):
        return [ModelSlot(name="devstral.gguf", kind="chat",
                          loaded=self.loaded == "devstral.gguf"),
                ModelSlot(name="diffusion-gemma.gguf", kind="canvas",
                          loaded=self.loaded == "diffusion-gemma.gguf")]

    def load_model(self, name, *, cancel=None):
        self.loads.append(name)
        if name == self.fail_load:
            return f"{name} would not load (no room)"
        self.loaded = name
        return ""

    # -- the model -----------------------------------------------------
    @property
    def canvas(self) -> bool:
        return self.loaded == "diffusion-gemma.gguf"

    def complete(self, messages, **kw):
        self.prompts.append((self.loaded, tuple(messages)))
        queue = self.canvas_chat if self.canvas else self.chat_replies
        if not queue:
            raise AssertionError(f"{self.loaded} ran out of replies")
        reply = queue.pop(0)
        if isinstance(reply, Completion):
            return reply
        return Completion(text=str(reply), finish_reason="stop",
                          model=self.loaded, tokens_in=10, tokens_out=10,
                          prompt_ms=5, decode_ms=50)

    def stream(self, messages, **kw):
        yield self.complete(messages, **kw).text

    def canvas_fill(self, request, *, cancel=None):
        self.requests.append(request)
        if not self.canvas_fills:
            return CanvasResult(finish_reason="error", error="no fills left",
                                model=self.loaded)
        fill = self.canvas_fills.pop(0)
        lines = fill.count("\n") + 1
        return CanvasResult(text=fill, finish_reason="stop", model=self.loaded,
                            line_confidence=(0.9,) * lines,
                            line_flips=(0,) * lines, steps=6, seed=1)

    def capabilities(self):
        return ModelCapabilities(
            name=self.loaded, family="gemma" if self.canvas else "mistral",
            context_tokens=32768, supports_tools=False,
            supports_canvas=self.canvas, token_count_is_estimate=True)

    def count_tokens(self, text):
        return max(1, len(text or "") // 4)


PLAN = "src/calc.py — add two numbers\n"

#: attempt 1 by the chat model: a NameError at line 3, raised when the
#: module runs (the call at the bottom), so `[run]` fails with a located
#: error inside `add` — the hole a canvas repair cuts out
CALC_WRONG = ('```python\ndef add(a: int, b: int) -> int:\n'
              '    """The sum."""\n    total = a + c\n    return total\n'
              '\n\nprint(add(1, 2))\n```')
#: the same again on later attempts: the chat model is stuck
STUCK = CALC_WRONG
FIXED = ('def add(a: int, b: int) -> int:\n    """The sum."""\n'
         '    return a + b\n\n\nprint(add(1, 2))\n')


def _run(tmp_path, llm, **cfg):
    events = RecordingEvents()
    host = Host(llm=llm, fs=LocalFileSystem(str(tmp_path)),
                exec=SubprocessExec(),
                storage=MemoryStorage(str(tmp_path / ".state")),
                events=events, approval=AutoApprove())
    cfg.setdefault("attempts", 2)
    cfg.setdefault("pin_interfaces", False)
    cfg.setdefault("review_after_build", False)
    cfg.setdefault("mutation_check", False)
    cfg.setdefault("speed_probe", False)
    session = Session(host, config=SessionConfig(**cfg))
    session.run("add two numbers in src/calc.py")
    return host, session


def test_the_canvas_model_repairs_what_the_chat_model_left_failing(tmp_path):
    """Phase 1: the chat model writes calc.py wrong twice. One swap: the
    canvas model rewrites the hole and the file verifies. Phase 3 is
    skipped — nothing left — so the second swap never happens."""
    llm = TwoModels(chat_replies=[PLAN, CALC_WRONG, STUCK],
                    canvas_fills=["    return a + b\n"])
    host, session = _run(tmp_path, llm)
    assert llm.loads == ["diffusion-gemma.gguf"], llm.loads
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].ok, final["src/calc.py"].stopped_because
    assert len(llm.requests) >= 1, "the repair was a hole"
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "hybrid (ar-first): devstral.gguf writes, diffusion-gemma.gguf " \
           "repairs" in log
    assert "PHASE 2 — repair: diffusion-gemma.gguf" in log
    assert "loading diffusion-gemma.gguf — phase 2 (repair)" in log
    assert "PHASE 3 (finish) — nothing left to do — skipped" in log
    journal = "\n".join(p.read_text() for p in
                        (tmp_path / ".cc_journal").rglob("*.jsonl"))
    assert '"event": "swap"' in journal and '"event": "phase"' in journal
    kinds = [k for k, _m, _d in host.events.events]
    assert "phase" in kinds


def test_phase_three_brings_the_chat_model_back_for_what_holes_missed(
        tmp_path):
    """The canvas model's holes never come back usable; phase 3 swaps the
    chat model back in and its fresh-start repair fixes the file."""
    llm = TwoModels(chat_replies=[PLAN, CALC_WRONG, STUCK,
                                  f"```python\n{FIXED}```"],
                    canvas_fills=[], canvas_chat=[STUCK, STUCK])
    host, session = _run(tmp_path, llm, attempts=2)
    assert llm.loads == ["diffusion-gemma.gguf", "devstral.gguf"], llm.loads
    assert (tmp_path / "src" / "calc.py").read_text() == FIXED
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].ok
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "PHASE 3 — finish: devstral.gguf" in log
    assert "PHASE 2 (repair) ended" in log and "still failing: src/calc.py" \
        in log
    assert "PHASE 3 (finish) ended" in log and "nothing left failing" in log


def test_a_first_phase_whose_model_is_not_loaded_swaps_before_planning(
        tmp_path):
    llm = TwoModels(chat_replies=[PLAN, f"```python\n{FIXED}```"],
                    canvas_fills=[], loaded="diffusion-gemma.gguf")
    host, session = _run(tmp_path, llm)
    assert llm.loads[0] == "devstral.gguf"
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].ok
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "phase 1 (write) is devstral.gguf's" in log


def test_a_swap_the_host_refuses_is_said_and_the_build_goes_on(tmp_path):
    llm = TwoModels(chat_replies=[PLAN, CALC_WRONG, STUCK,
                                  f"```python\n{FIXED}```"],
                    canvas_fills=[], fail_load="diffusion-gemma.gguf")
    host, session = _run(tmp_path, llm)
    assert llm.loads[0] == "diffusion-gemma.gguf"
    warnings = [m for k, m, _d in host.events.events if k == "warning"]
    assert any("could not load diffusion-gemma.gguf" in m for m in warnings)
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "was not loaded" in log and "would not load (no room)" in log
    # phase 2 ran on the chat model that was still loaded, and fixed it
    final = {o.path: o for o in session._final_outcomes()}
    assert final["src/calc.py"].ok


def test_hybrid_off_never_touches_the_roster(tmp_path):
    llm = TwoModels(chat_replies=[PLAN, CALC_WRONG, STUCK], canvas_fills=[])
    host, session = _run(tmp_path, llm, hybrid="off")
    assert llm.loads == []
    log = (tmp_path / "BUILD_LOG.txt").read_text(encoding="utf-8")
    assert "hybrid    hybrid is off" in log
    assert "PHASE 2" not in log
