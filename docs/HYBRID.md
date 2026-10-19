# Hybrid mode — one model writes, another repairs, and the harness never lies

*Designed 2026-10-16 from the first two real builds of the Star Trek spec
("Begin 4: A Deterministic Tactical Starship Simulation", 15,070
characters, 7 source files, 3 test files). One build ran DiffusionGemma
alone, one ran Devstral-Small-24B alone. Both failed. This document says
why — from the journals, not from memory — and what was built so the next
build does not fail the same way.*

Modules: `cognitive_coder/specslice.py`, `specvectors.py`, `adjudicate.py`,
`hybrid.py`; the loop's escalation ladder in `loop.py`; the roster port in
`ports.py`; the session's phases in `session.py`. ATK's side:
`atk/core/ccoder_host.py` (`ATKLLM.roster` / `load_model`).

## The two builds, in numbers

| | DiffusionGemma (canvas) | Devstral 24B Q4_K_M |
|---|---|---|
| wall time | 14.5 min, then **crashed** | 2 h 33 min, then stopped by hand |
| files built | 2 of 7 (neither verified) | 1 of 7 verified-ish; 6 failed |
| model speed | 2–5 s per short hole, ~3 s/step | **3–5 tok/s** — spilled into system RAM |
| model time of wall time | — | 89 % (6,464 s decode + 1,707 s prefill) |
| test failures, by whose fault | — | **10 wrong tests, 4 real code bugs, 2 ambiguous spec** |

Neither failure was mostly the model's.

### What actually went wrong, ranked by the minutes it cost

1. **Every diffusion prompt cut the request at 2,000 characters**
   (`canvas.stub_text`, `repair_text`), and so did every whole-file repair
   (`personas.REPAIR_REQUEST_CHARS`). The spec's formulas — `atan2(x2-x1,
   -(y2-y1))` at character 7,363, the `+30°` sector offset, `overflow *
   0.25`, `warp * 25`, the subsystem keys, the skirmish positions — all sat
   past the cut. **DiffusionGemma never saw the spec it was implementing,
   and Devstral lost it on every repair.** Devstral's *first* attempts, which
   carried the whole request, followed the spec line for line.
2. **`[run]` executed package modules as scripts** (`python src/physics.py`),
   so any `from .models import …` failed with "attempted relative import
   with no known parent package" — a non-error that cost DiffusionGemma two
   of four physics attempts, whose "fix" (`from models import …`) then broke
   the package for the test runner.
3. **The host tried to `pip install physics`** when that import failed,
   then crashed assigning to a frozen `ProcResult`. The crash reached the
   journal as an `error` event; `BUILD_LOG.txt` never mentioned it, and
   `session_end` said `ok: true`.
4. **The test-vs-code rule had no judge.** "The test is never rewritten;
   the code must change" is right against a model bending a test to match
   its bug — and wrong 10 times out of 16 here, where the *tests* assumed a
   y-up world (`bearing_between(0,0,1,1) == 45`) against a spec that says
   north is `(0, -1)`. Devstral refused to break correct code (returned the
   file unchanged — the right call) and the build had no path forward but
   "fix by hand".
5. **The skeleton was parsed, never imported.** The spec's `Ship` and
   `WorldState` put defaulted fields before required ones — invalid as a
   dataclass. `ast.parse` passed; the `TypeError` surfaced at models.py's
   first run and cost two attempts (~20 min).
6. **Hard-requirement extraction stripped underscores from identifiers**
   (`test_physics.py` → `testphysics.py`, `SUBTICKS_PER_TURN` →
   `SUBTICKSPERTURN`) — and those mangled names were "pinned in every
   prompt".
7. **Sampling never varied, or varied in secret.** The engine logged
   `temperature=0.15 seed=None` on every attempt. ATK's host raised the
   temperature behind the engine's back on repeated prompts, so the log
   cannot say what Devstral actually sampled at. Either way, attempt 2 of a
   repair was attempt 1 with the same input, and the stagnation check
   stopped the file.
8. Diffusion-specific: a `tail` canvas echoed the `# <<< END OF THE LINES TO
   REWRITE` marker and a second `def normalize_angle` that *shadowed the
   correct one* with `raise NotImplementedError` — chosen over a 98 %-sure
   `open` canvas because it was 100 % sure. Four of five rejected canvases
   had finished with `length` and were discarded before the run-on cut
   could salvage them; for the one-line `normalize_angle`, both held the
   correct `return deg % 360.0` as their first line.

## What the hybrid is for — and what it is not

The question that started this was: *Gemma writes, Devstral evaluates,
Gemma fixes, Devstral fixes, Gemma revisits?* The useful idea in it is a
**different prior**: two models are wrong about different things, so the
second one can escape a loop the first is stuck in. The idea it rests on —
that Devstral varies more — is backwards on the evidence: at 0.15 Devstral
returned byte-identical files on repair; DiffusionGemma's seeded canvases
came back different every time.

So the hybrid is built around phases, not ping-pong:

* **a swap is expensive on one 16 GB card** — an unload, a load and a cold
  prompt cache — so the build swaps **twice**, not per file;
* **what a model is good at decides its role**, by capability
  (`ModelCapabilities.supports_canvas`), not by name;
* **the harness faults above are closed first**, because alternating models
  on a wrong test means two models take turns breaking correct code.

### The phases

```
PHASE 1  write      the WRITER (an ordinary chat model) plans, pins the
                    skeleton, writes every file and every test with the
                    full per-file spec slice; the loop's escalation ladder
                    runs within each file; test-vs-code disagreements are
                    adjudicated (below)
         ── one swap ──
PHASE 2  repair     the REPAIRER (the canvas model) rewrites only the hole
                    each remaining failure is in — located by the error
                    line, else by the names the failing test calls — with
                    the judge's verdicts as its implementation notes
         ── one swap ──
PHASE 3  finish     the WRITER again: whole-file repairs of what holes
                    could not fix, fresh-start prompts (no broken file in
                    view), temperature up
```

Each phase repairs in **dependency order** (`models` before `physics` before
`ai`), re-verifies the test of each repaired module, and only then attempts
the tasks that were blocked on it. A phase ends when nothing it can act on
is left.

**Default order: the chat model writes first.** The reasons, from the
evidence: a canvas body is at most `OPEN_CANVASES_STUB` (3) canvases, about
600 tokens, and `apply_orders` in this spec is ~900 — DiffusionGemma leaves
the long functions as stubs *structurally*; a whole file written by one
model is coherent across its functions in a way ten separately-filled holes
are not; and Devstral wrote the spec's physics line for line once it could
see the spec. `hybrid = "diffusion-first"` runs the phases the other way
round (canvas writes → chat model judges and fixes → canvas revisits); it is
one setting, kept for the day the measurement says otherwise.

### Roles, by capability

| role | picked by | default |
|---|---|---|
| writer | a roster model that does **not** fill a canvas | the loaded model, if it is one; else the first such slot |
| repairer | a roster model that **does** fill a canvas | the first such slot |
| judge | the writer (no swap) | `hybrid_judge` names another slot |

With no roster port, no second slot, or `hybrid = "off"`, nothing changes:
one model, the ordinary loop, exactly as before. With a roster but only one
kind of model, the phases still run (a second pass with fresh-start prompts
is still worth a swap-free retry) — said in the log.

## The roster port

A host that can load more than one model implements, beside `LLMPort`:

```python
class ModelRoster(Protocol):
    def roster(self) -> Sequence[ModelSlot]: ...          # what could be loaded
    def load_model(self, name: str, *, cancel=None) -> str: ...  # "" or a sentence
```

`ModelSlot` (types.py): `name`, `path`, `kind` (`"canvas"` | `"chat"` |
`""`), `loaded`, `note`. After `load_model` returns "", `capabilities()`
reports the new model; the session re-reads it and bumps the epoch as it
does for any swap. The core still contains no loading code — it asks, the
host decides, and a host that answers with a sentence ("it would not
load") gets the build carried on with whatever is loaded, said in the log.

ATK's `ATKLLM` builds the roster from `settings["ccoder"]["models"]` (the
slots the Coder tab already has, with their own `n_ctx` / `n_gpu_layers`),
marks a slot `canvas` when `dgemma.is_diffusion_gguf` says so, and
`load_model` is the panel's existing `swap_model`.

## The judge: test-vs-code, ruled by a quotation

When a test fails against the module it tests (`_test_disagrees_with_code`),
the loop used to blame the module unconditionally. Now it asks ONE short
question first (`adjudicate.py`):

> Which does the request support — the module, or the test? Quote the
> request's words that decide it.

The reply is JSON: `verdict` (`code` | `test` | `unclear`), `quote`, `why`.
**The quote is checked**: it must appear in the request (whitespace
normalised). No quote, or a quote that is not there, and the verdict is
downgraded to `unclear`. Then:

* `code` → the module is repaired against the test, as before;
* `test` → the **test** is repaired, with the quote as its diagnostic
  ("the request says *… north is (0, -1) …*; this assertion contradicts
  it; change the assertion, not the module");
* `unclear` → the conservative rule, as before, and the operator is told it
  was unclear.

One question per (test file, failure signature). Every verdict, quote and
"why" goes to `BUILD_LOG.txt` and the journal (`adjudicate` events). A model
returning the file unchanged on a repair ("I am certain the test is
wrong") now triggers the same question instead of stopping the file.

## Spec vectors: the arbiter nobody wrote

`specvectors.py` lifts, by rule, the examples a specification states
outright and turns them into tests no model can rewrite:

* **`` `literal` -> `literal` `` pairs** in a sentence that names a function
  (`` Implement `parse_bank_spec(spec: str) …` supporting `"1-3"` -> `(1, 2, 3)` ``)
  become `self.assertEqual(parse_bank_spec("1-3"), (1, 2, 3))`;
* **UPPER_CASE constants with literal values** (`` `SUBTICKS_PER_TURN: int = 10` ``)
  become `self.assertEqual(models.SUBTICKS_PER_TURN, 10)`.

Each vector is attributed to a file by the nearest heading or sentence that
names one, written to `tests/test_spec_vectors_<stem>.py` by the harness
(marked as such), and run as part of that module's verification — a vector
failing is the module's failure, with the spec's own words as the message.
A model-written test that contradicts a vector is wrong by definition; the
judge is told the vectors passed.

Not a replacement for the model's tests: for the Star Trek spec this yields
seven assertions. They are the seven that cannot be wrong.

## The escalation ladder, in the engine, in the log

Attempt by attempt, per file (`loop.py`):

| attempt | what changes | temperature |
|---|---|---|
| 1 | the first try | the persona's (0.15) |
| 2 | repair: the diagnostics, enriched with the real definitions they name | 0.15 |
| 3 | **fresh start**: the broken file is *not* shown; the pinned interface, the notes, the test and the diagnostics are | 0.45, seeded |
| 4+ | whole-file repair | 0.70, seeded |

Stagnation ("identical code twice", "same errors twice") no longer stops a
file while a rung is left: it moves to the next rung and says so. The
temperature and seed actually used are what `generate` lines print. ATK's
host no longer varies sampling on its own when the engine has this ladder.

## Honest logs

* `BUILD_LOG.txt` gets a `STOPPED UNEXPECTEDLY` event with the sentence and
  the traceback; `session_end` carries `error` and is never `ok` after one.
* A `SUMMARY` closes every build log (it closed only debug sessions).
* A **speed probe** after model load and after every swap: one short
  generation, its tok/s and prefill written down. Under 8 tok/s the log
  says the model is probably spilling into system RAM and what to change.
* Every phase has a header with its model, its swap's duration, and a
  summary of what it fixed; every adjudication its verdict and quote; every
  spec slice the sections it kept and the ones it left out.

## Smaller rules, each from one failure

* `constraints.py` strips `*emphasis*` only at word boundaries, so
  `test_physics.py` stays `test_physics.py`.
* The skeleton is **imported**, not only parsed; a dataclass whose
  defaulted fields precede required ones is made valid with
  `field(default=…, kw_only=True)` — order and names untouched — and the
  log says so.
* Hard-requirement sentences that name a pinned module, or one of its
  declared names, are copied into that module's docstring under
  "Conventions from the request", word for word.
* `[run]` executes `python -m src.physics` from the project root for any
  module inside a package.
* The host does not `pip install` a name that is a module of the project;
  it tells the model which import it meant.
* Canvas: marker echoes are stripped; a fill that defines a name the file
  already defines at top level is rejected; a canvas that ended with
  `length` is salvaged where the run-on cut finds a parsing prefix; a
  failure that names no line of the file is located through the names the
  failing test calls before falling back to doubt. `open` runs before
  `tail` in the schedule.

## Settings

| SessionConfig | default | |
|---|---|---|
| `hybrid` | `"auto"` | `"off"`, `"ar-first"`, `"diffusion-first"` |
| `hybrid_writer` / `hybrid_repairer` / `hybrid_judge` | `""` | roster names; `""` picks by kind |
| `adjudicate` | `True` | ask the judge before blaming a module |
| `spec_vectors` | `True` | write and run the harness-owned vector tests |
| `escalate` | `True` | the ladder above; off restores the old stop-on-stagnation |
| `speed_probe` | `None` | on for a real model port; the scripted test doubles are never probed |
| `request_chars_first` / `request_chars_repair` | 16,000 / 8,000 | the per-file spec slice budgets |

## What is still not known

* Which order wins on a *full* build — `ar-first` is the evidence-based
  guess; the journals of the next builds decide.
* Whether DiffusionGemma judges well. The quote check makes a bad judge
  harmless (its verdicts become `unclear`), not useful.
* Devstral's real speed when it fits in VRAM. The probe will say; a
  Q3-class quantization with a 16 K context and a quantized KV cache is the
  first thing to try.
