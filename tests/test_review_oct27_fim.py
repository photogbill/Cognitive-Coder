# SPDX-License-Identifier: Apache-2.0
"""Review of the 2026-10-27 FIM / profiles change — reproducers.

Every test here FAILS against the code as it stands. Each docstring says
what is wrong and what the correct behaviour is. Model-side half only:
fim.py, profiles.py, the two providers' `fim_fill`, the loop's FIM path,
the session's probe.
"""

from __future__ import annotations

import http.client
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_fim import (  # noqa: E402
    ERR,
    GOOD_FILL,
    PHYSICS,
    FimScripted,
    _host,
    _same,
    _task,
    _write,
)
from test_loop import _loop  # noqa: E402

from cognitive_coder import fim, profiles  # noqa: E402
from cognitive_coder.journal import Journal  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    Host,
    MemoryFileSystem,
    MemoryStorage,
    RecordingEvents,
)
from cognitive_coder.providers import openai_compatible as oc  # noqa: E402
from cognitive_coder.session import Session, SessionConfig  # noqa: E402
from cognitive_coder.types import (  # noqa: E402
    Diagnostic,
    FimRequest,
    FimResult,
    ModelCapabilities,
    Task,
)


def _llama_server(post):
    """An OpenAICompatible pointed at a fake llama-server: `/props` was
    answered (grammar=True), so `fim_fill` takes the `/infill` path."""
    p = oc.OpenAICompatible("http://127.0.0.1:1", model="m",
                            supports_tools=False)
    p.capabilities = lambda: ModelCapabilities(          # type: ignore
        name="m", family="qwen", context_tokens=8192, supports_grammar=True)
    p._post = post                                        # type: ignore
    return p


# --------------------------------------------------------------------------
# 1. /infill: tokens_in double-counts the cached prompt tokens
# --------------------------------------------------------------------------

def test_infill_tokens_in_double_counts_the_cached_tokens():
    """openai_compatible.fim_fill (the llama-server branch) reports
    ``tokens_in = tokens_evaluated + timings.cache_n``. llama-server's
    `tokens_evaluated` is the WHOLE prompt (`n_prompt_tokens`, the same
    number its OpenAI-shaped `usage.prompt_tokens` carries — see
    providers.base.processed_tokens, which treats `prompt_tokens` as the
    total and `timings.prompt_n` as the processed part); `cache_n` is the
    part of that total served from the cache. Adding them counts every
    cached token twice: a 100-token hole with 60 from the cache is
    journaled as 160 tokens in, and the loop's `prompt_processed`
    (tokens_in - cached_tokens) becomes the full 100 — "nothing came from
    the cache" — for a call that was 60 % cached.

    Correct: tokens_in == tokens_evaluated (100); cached_tokens == 60."""
    def post(path, payload, timeout=None):
        assert path == "/infill"
        return {"content": "    return a + b\n", "tokens_predicted": 6,
                "tokens_evaluated": 100, "stop_type": "eos",
                "truncated": False,
                "timings": {"prompt_n": 40, "prompt_ms": 50,
                            "predicted_n": 6, "predicted_ms": 120,
                            "cache_n": 60}}
    result = _llama_server(post).fim_fill(FimRequest(prefix="a", suffix="b"))
    assert result.ok
    assert result.cached_tokens == 60
    assert result.tokens_in == 100, result.tokens_in


# --------------------------------------------------------------------------
# 2. the FIM generate event claims an exact cache count the host never gave
# --------------------------------------------------------------------------

def test_fim_generate_asserts_a_cache_count_the_host_never_reported(tmp_path):
    """`Loop._fim_generate` builds the attempt's Completion with
    ``prompt_processed = tokens_in - cached_tokens``. `FimResult.
    cached_tokens` defaults to 0 and neither the in-process provider
    (`LocalLlamaCpp.fim_fill`) nor a scripted host ever sets it, so a
    host that reported NOTHING about its cache is journaled as having
    processed every prompt token from scratch. `Completion.prompt_processed`
    is documented as "0 means not reported" and `Journal.cache_health`
    treats a non-zero value as the server's exact count — so two FIM
    repairs through such a host (900-token prompts prefilled in 40 ms,
    plainly cached) make the session report say the prefix cache "was
    invalidated when it should not have been".

    Correct: leave prompt_processed at 0 (unknown) when the host gave no
    cache count — as the canvas path does — so the cache verdict is not
    drawn from a number nobody measured."""
    _write(tmp_path)
    (tmp_path / "src" / "other.py").write_text(PHYSICS)

    def fill():
        return FimResult(text=GOOD_FILL, model="m", tokens_in=900,
                         tokens_out=27, prompt_ms=40, decode_ms=300)
    host = _host(tmp_path, [], [fill(), fill()])
    loop = _loop(host, attempts=2)
    loop.fim_ok = True
    loop.journal = Journal(host.fs, "t")
    assert loop.run_task(_task(), request="ship physics", seed=ERR).ok
    err2 = (Diagnostic(file="src/other.py", line=12, severity="error",
                       message="AssertionError: 45.0 != 90.0"),)
    assert loop.run_task(Task(id="t2", path="src/other.py",
                              purpose="physics", lang="python"),
                         request="ship physics", seed=err2).ok
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    assert len(gens) == 2 and all(g["data"].get("via") == "fim" for g in gens)
    verdict = loop.journal.cache_health()
    assert "invalidated" not in verdict, verdict
    assert all(g["data"].get("prompt_processed", 0) == 0 for g in gens), \
        [g["data"].get("prompt_processed") for g in gens]


# --------------------------------------------------------------------------
# 3. fim_fill raises on an HTTP-layer failure
# --------------------------------------------------------------------------

def test_fim_fill_raises_on_an_http_exception_from_the_server():
    """`FimLLM.fim_fill` MUST NOT raise (ports.py, guarantee 1). The
    openai_compatible implementation catches HTTPError, URLError, OSError,
    TimeoutError and ValueError — but `http.client.HTTPException` and its
    subclasses (`IncompleteRead`, `BadStatusLine`, `LineTooLong`: a server
    that dies mid-reply or answers garbage) derive from Exception only,
    and escape. In a repair that ends the task with a traceback instead
    of "this attempt takes another path". (`complete()` has the same
    gap; the new code copied it.)

    Correct: a FimResult with finish_reason "error" and a sentence."""
    def post(path, payload, timeout=None):
        raise http.client.IncompleteRead(b"")
    provider = _llama_server(post)
    try:
        result = provider.fim_fill(FimRequest(prefix="a", suffix="b"))
    except Exception as exc:                                # noqa: BLE001
        raise AssertionError(f"fim_fill raised {type(exc).__name__}") from exc
    assert result.finish_reason == "error" and result.error


# --------------------------------------------------------------------------
# 4. the generate event records the attempt's sampling, not the fill's
# --------------------------------------------------------------------------

def test_the_generate_event_records_the_sampling_the_fill_did_not_use(
        tmp_path):
    """`Loop._sampling` is documented as "what the current attempt
    actually samples at, so the journal and BUILD_LOG.txt print the
    temperature and seed that were USED". A FIM attempt's `generate`
    event takes `temperature` and `seed` from that Sampling — but
    `fim.requests_for` never sends them: the first candidate's seed is
    ``seed + 7919`` and the second is 0.25 warmer with another seed. When
    the first fill is unusable and the second is kept, the journal says
    the code came from temperature 0.15 / seed 3 while the fill that
    produced it ran at 0.40 / seed 120570. The `fim` event has the real
    numbers; the `generate` event, which `stats()` and the replay read,
    does not.

    Correct: the generate event carries the kept candidate's temperature
    and seed."""
    _write(tmp_path)
    host = _host(tmp_path, [], ["    return (\n", GOOD_FILL])
    loop = _loop(host, attempts=2, seed=3)
    loop.fim_ok = True
    loop.journal = Journal(host.fs, "t")
    assert loop.run_task(_task(), request="ship physics", seed=ERR).ok
    assert len(host.llm.fim_requests) == 2
    kept = host.llm.fim_requests[1]
    gens = [e for e in loop.journal.events() if e.get("event") == "generate"]
    assert gens[-1]["data"].get("via") == "fim"
    assert gens[-1]["seed"] == kept.seed, (gens[-1]["seed"], kept.seed)
    assert abs(float(gens[-1]["temperature"]) - kept.temperature) < 1e-9, \
        (gens[-1]["temperature"], kept.temperature)


# --------------------------------------------------------------------------
# 5. the probe is a model call the journal records without its tokens
# --------------------------------------------------------------------------

PLAN = "greet.py — say hello\n"
FILE = "```python\ndef greet(name):\n    return f'hi {name}'\n```"


def _session(fills, name="Qwen3-Coder-30B", mode="on"):
    llm = FimScripted([PLAN, FILE], fills, supports_tools=False, name=name)
    host = Host(llm=llm, fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(
        attempts=1, review_after_build=False, mutation_check=False,
        adjudicate=False, fim=mode))
    return session, llm, host


def test_the_probe_is_journaled_without_its_token_counts():
    """Every model call is journaled with provenance. The FIM probe is
    one (a prompt of ~60 tokens, up to 48 out, per session and per
    swap), and `Session._probe_fim` logs it as a `fim` event with only
    `ok`, `why` and the first 200 characters of the text: no tokens_in,
    tokens_out, prompt_ms or decode_ms, although the host handed all of
    them back on the FimResult. `Journal.stats()` sums tokens over
    `generate` events only, so the probe's cost appears nowhere.

    Correct: the probe event carries the FimResult's token counts (and
    timings), as the `fim` repair event does for every candidate."""
    session, llm, _host_ = _session([FimResult(
        text="    return a + b\n", model="m", tokens_in=61, tokens_out=6,
        prompt_ms=30, decode_ms=90)])
    session.run("a greet function")
    assert session.loop.fim_ok
    probe = [e for e in session.journal.events()
             if e["event"] == "fim" and e["data"].get("kind") == "probe"]
    assert probe, "the probe was not journaled at all"
    row = probe[0]
    tokens_in = row.get("tokens_in") or row["data"].get("tokens_in")
    tokens_out = row.get("tokens_out") or row["data"].get("tokens_out")
    assert tokens_in == 61 and tokens_out == 6, (tokens_in, tokens_out)


# --------------------------------------------------------------------------
# 6. the probe passes a port that never reads the suffix
# --------------------------------------------------------------------------

def test_the_probe_passes_a_port_that_never_reads_the_suffix():
    """The probe hole's answer (`return a + b` under the docstring
    "Return the sum of a and b") is fully determined by the PREFIX. A
    server that ignores `suffix` — `/v1/completions` on LM Studio, or any
    plain-completion endpoint — continues the prefix, writes the sum and
    runs on into a `def sub` of its own; the dedent cut in
    `fim.probe_verdict` trims that, and the verdict is "the model filled
    a known hole correctly". FIM mode then turns on for a model that was
    never shown the far side of a hole: every repair is a prefix
    completion cut at the dedent, journaled `via: fim`, and the session
    says "this model fills in the middle".

    Correct: the probe should not pass a fill produced without the
    suffix — a hole whose right answer depends on what follows it (for
    example prefix ``    result = `` / suffix ``\\n    return result``)
    would tell the two apart."""
    class PrefixOnly:
        """What a server that ignores `suffix` sends back for the probe."""
        def fim_fill(self, request, *, cancel=None):
            assert request.suffix       # it was given one — and ignores it
            return FimResult(text="    return a + b\n\n\ndef sub(a, b):\n"
                                  "    \"\"\"Return a minus b.\"\"\"\n"
                                  "    return a - b\n\n\ndef mul(a, b):\n"
                                  "    return a * b\n",
                             finish_reason="length")
    ok, why = fim.probe_verdict(PrefixOnly().fim_fill(fim.probe_request()))
    assert not ok, why


# --------------------------------------------------------------------------
# 7. a refused swap leaves the previous model's profile and probe in force
# --------------------------------------------------------------------------

class _Swappable(FimScripted):
    """A FimScripted with ModelRoster's `load_model`: the name changes,
    the script does not."""

    def load_model(self, name, *, cancel=None):
        self._name = name
        return ""


def test_a_refused_swap_keeps_the_old_models_profile_and_fim_verdict():
    """`Session._check_model` raises NotACodingModelError for a model the
    name table refuses BEFORE it reaches `_apply_profile` and
    `_probe_fim`. `_swap_to` catches that exception and carries the build
    on with the new model — but `loop.profile` is still the OLD model's
    (Qwen3-Coder's temperature 0.7, its samplers, its FIM token texts)
    and `loop.fim_ok` is still True from the old model's probe, so the
    next repair sends FIM requests, with Qwen's tokens, to a model that
    was never probed. The docs promise the profile and the probe are
    re-decided "after every swap".

    Correct: after the swap the profile describes the loaded model and
    fim_ok is False until that model's own probe says otherwise."""
    llm = _Swappable([PLAN, FILE], ["    return a + b\n"],
                     supports_tools=False, name="Qwen3-Coder-30B")
    host = Host(llm=llm, fs=MemoryFileSystem(), storage=MemoryStorage(),
                events=RecordingEvents(), approval=AutoApprove())
    session = Session(host, config=SessionConfig(
        attempts=1, review_after_build=False, mutation_check=False,
        adjudicate=False, fim="on"))
    session.run("a greet function")
    assert session.loop.fim_ok and session.loop.profile.family == "qwen3-coder"
    assert session._swap_to("dolphin-2.9-llama3-8b", why="the chat phase")
    assert session.host.llm.capabilities().name == "dolphin-2.9-llama3-8b"
    assert not session.loop.fim_ok, session.loop.fim_why
    assert session.loop.profile.family != "qwen3-coder"


# --------------------------------------------------------------------------
# 8. the "same hole twice" rule compares the raw failure message
# --------------------------------------------------------------------------

def test_the_same_hole_rule_is_defeated_by_an_address_in_the_message(
        tmp_path):
    """`Loop._fim_generate` keys a filled hole on
    ``(function, the failure's message)`` and skips the hole when the key
    comes back: "the same function with the same failure's words says the
    fault is elsewhere". The message is compared raw. A failure whose
    words carry anything that varies per run — an object address
    (`<Vec object at 0x7f...>`), a temp path, a timing — never matches,
    and the same body is filled again on the next attempt. The loop's
    own `_recall_key` blanks addresses and numbers for exactly this
    reason; the FIM key does not use it.

    Correct: one fill of `bearing_between` for this failure, as
    test_fim's `..._not_filled_a_third_time` asserts for a stable
    message; the second attempt goes to another path."""
    _write(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_p.py").write_text(
        "import unittest\nfrom src.physics import bearing_between\n\n\n"
        "class T(unittest.TestCase):\n    def test_north(self):\n"
        "        self.assertEqual(bearing_between(0, 0, 1, 1), object())\n")
    still_wrong = ("    dx, dy = x2 - x1, y2 - y1\n"
                   "    heading = math.degrees(math.atan2(dy, dx)) + 0.0\n"
                   "    return normalize(heading)\n")
    whole = "```python\n" + PHYSICS + "```"
    host = _host(tmp_path, [whole, whole],
                 [still_wrong, still_wrong.replace("+ 0.0", "- 0.0")])
    loop = _loop(host, attempts=2)
    loop.fim_ok = True
    loop.journal = Journal(host.fs, "t")
    task = Task(id="t1", path="src/physics.py", purpose="physics",
                lang="python", test_path="tests/test_p.py")
    seed = (Diagnostic(file="src/physics.py", line=12, severity="error",
                       message="AssertionError: 45.0 != <object object at "
                               "0x7f0000000001>"),)
    loop.run_task(task, request="ship physics", seed=seed)
    assert len(host.llm.fim_requests) == 1, len(host.llm.fim_requests)


# --------------------------------------------------------------------------
# 9. an echoed hint lands in the file
# --------------------------------------------------------------------------

def test_an_echoed_hint_comment_is_spliced_into_the_file():
    """fim.py promises "The comment is in the prompt only; the file on
    disk never has it." The hint is two or three `#` lines at the hole's
    indentation, right before the code the model writes — the position
    a model is likeliest to echo. `fim.judge` strips echoed FIM tokens
    and `canvas.clean_fill` drops echoed `>>>`/`<<<` markers, but an
    echoed hint is an ordinary comment at body indentation: not cut
    (comments are exempt from the dedent cut), parses, differs from the
    old lines — so `# FIX: the body of `bearing_between` failed:
    AssertionError: 45.0 != 90.0` is written into the repaired file as
    code.

    Correct: lines of the fill that repeat the hint are dropped, as the
    canvas's marker echoes are."""
    hole = fim.hole_for_repair(PHYSICS, "python", "src/physics.py", ERR,
                               _same)
    hint = fim.hint(hole, ERR, "src/physics.py", _same)
    cand = fim.judge(PHYSICS, hole, FimResult(text=hint + GOOD_FILL),
                     "python")
    assert cand.usable, cand.problem
    assert "FIX:" not in cand.code and "signature above" not in cand.code


# --------------------------------------------------------------------------
# 10. a DeepSeek-R1 distill of Qwen3 is matched as a Qwen3 with a switch
# --------------------------------------------------------------------------

def test_a_deepseek_r1_distill_of_qwen3_is_not_a_qwen3_with_a_switch():
    """`profiles.BUILTIN` is matched first-marker-wins, and "qwen3" sits
    above "deepseek-r1". `DeepSeek-R1-0528-Qwen3-8B` (a real release: R1
    reasoning distilled onto a Qwen3 base, always thinks, card says
    temperature 0.6 / top_p 0.95) matches "qwen3" first and gets the
    Qwen3 profile: thinking "switch", `/no_think` appended to every
    prompt, `enable_thinking=False` sent through the template, 0.7 /
    top_p 0.8. The model has no switch; the engine believes it turned
    thinking off and the journal's `profile` event says so.

    Correct: the deepseek-r1 row (thinking "always", its card's sampling)
    — the more specific marker must win, as "qwen3-coder" does over
    "qwen3"."""
    p = profiles.builtin_for("DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf")
    assert p.thinking == "always", (p.family, p.thinking)
    assert p.family == "deepseek-r1"
