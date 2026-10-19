# Diffusion mode — writing and repairing by holes

*Built 2026-10-10. Module: `cognitive_coder/canvas.py`; the loop's side is
`Loop._canvas_*` in `loop.py`; tests: `tests/test_canvas.py`.*

## What it is, in one paragraph

When the loaded model is a **diffusion** model — DiffusionGemma, behind a
host that says `ModelCapabilities.supports_canvas=True` — Cognitive Coder
stops asking it for whole files. A repair rewrites only the **hole** the
error is in: the lines on both sides are *held* on the model's canvas, the
old lines are scrambled in place, and the model fills the gap seeing the
code before and after it. A new file whose interface was pinned is written
**one body at a time**, every signature, field and docstring held. Every
other model runs exactly as before.

## Why

The engine's own catalogue of small-model failures (build spec, Appendix D)
put most of the damage in the repair loop, and the loop asked for the
complete corrected file every time:

| Failure | What happened | What a hole does |
|---|---|---|
| **D6** helpful rewriting | asked for one fix, got the function rewritten "while it was in there" | code outside the hole is never written by the model at all |
| **D3** the same fix forever | Oct 2: `render.py` "REPAIRED" byte for byte | the old lines are noise on the canvas — there is nothing to copy back; each seed starts somewhere new |
| **D11** context poisoning | the model pattern-matches its own broken code | the broken lines are scrambled; the diagnostics carry forward |
| **D2 / D5** commentary, fences | prose around the code; unpaired fences | the reply is code at known positions — there is nothing to extract |
| whole-file size refusal | a file too big for the reply budget could not be repaired | the reply is one canvas, whatever the file's size |
| pinned-interface drift | a body that renamed or dropped a pinned name | signatures are held; a body cannot touch them |

Appendix G.4 already called fill-in-the-middle "PROVEN, and underused".
Every provider reported `supports_fim`; nothing read it. A diffusion canvas
is that missing piece, and more: the model sees the code *after* the hole
**inside** the canvas (bidirectional attention), not only in the prompt, and
it reports how sure it was of every token it wrote.

## What it does not fix

Be clear about this, because it was the first question asked:

* **Cross-file mistakes.** DiffusionGemma sees both directions only inside
  one 256-token canvas — roughly 20 lines. A `TrackSegment` that is a tuple
  in one file and read as `seg.x` in another is still found by the pinned
  interfaces, the codemap and `enrich.py`, exactly as before.
* **D4 invented APIs.** Holding lines does not make a name exist. The
  codemap's checks are still the defence.
* **Whether the code works.** The tests decide, as always. The model's
  confidence ranks candidates and points at doubt; it never accepts code.

## How a repair works

1. **Where.** `canvas.repair_hole` takes the first real error that names a
   line of *this* file. In Python, the hole is the body of the innermost
   function containing it (`kind="body"`). A body longer than
   `BODY_MAX_LINES` (36), or a file that does not parse, gets a **window**
   of lines around the error instead, shrunk until it fits the canvas
   budget by the host's own token count. No located line → no hole (see
   *doubt-guided*, below), and the whole-file path runs.
2. **The prompt** is the same context a whole-file repair gets — the
   interfaces, the pinned contract, the failing test, the caller, the
   enriched errors — with the file shown **marked** (`# >>> REWRITE LINES
   3-4 … # <<< END`), a `[THE HOLE]` block that shows the lines the reply
   starts with, and a canvas output contract. The `[TASK]` text is the same
   for every hole in a file, so a prompt cache keeps everything before the
   part that changes.
3. **Candidates** (`REPAIR_SCHEDULE`, `diffusion_candidates`, default 2):
   * `("tail", 0.5)` — the code after the hole is held on the canvas; the
     old lines start the free region with about half of them scrambled, and
     the denoising starts halfway along its path. The model's own
     uncertainty decides what changes. The conservative fix.
   * `("open", 1.0)` — the hole ends where the model ends its reply; the
     code after it is shown only in the prompt. From noise: a rewrite.
   * `("tail", 1.0)`, `("open", 1.0)` with a third and fourth candidate.
4. **Judging.** Each fill is cleaned (`clean_fill`: padding and a run-on
   cut, a copied tail dropped), spliced (`splice`: everything outside the
   region byte for byte unchanged), and screened (`prescreen`: it must
   parse; `judge`: it must not be the old lines). The usable candidate the
   model was surest of goes to the loop — which runs the tests on it like
   any other attempt. No usable candidate → that attempt is a whole-file
   repair, said in a warning.
5. **The same hole twice is enough.** If the same function (or error line)
   fails with the same error after its hole was rewritten, the fault is not
   in those lines — a caller, a definition elsewhere — and the next attempt
   is a whole-file repair.
6. **Each attempt is one hole.** The next attempt gets the new diagnostics
   and a new hole — so best-of-N is no longer blind: it is iterative, with
   the test output between tries.

### Doubt-guided repairs

A failure that names no line of the file — `AssertionError: 7 != 6`, raised
in the test — cannot be located. In diffusion mode the loop remembers how
sure the model was of every body it wrote (`Loop.doubt`), and repairs the
body it was **least sure of**, then the next one. It says so: *"the failure
names no line of this file, and the model was least sure of this body when
it wrote it"*.

## How a pinned file is written

When the skeleton pinned a file's interface (`interfaces.py`), its first
attempt in diffusion mode:

1. makes **one ordinary call for implementation notes** — a canvas starts
   with code, so a body cannot think first; the notes are the thinking,
   written down where every body in the file can read them
   (`diffusion_notes`, default on);
2. fills each `raise NotImplementedError` body as an **open** hole, in file
   order, with the def line and docstring held and the file so far in view
   (up to `OPEN_CANVASES_STUB` = 3 canvases a body), a second seed when the
   first does not parse;
3. leaves a body that would not come back usable as `raise
   NotImplementedError` — the tests then fail **at that line**, and the next
   attempt repairs exactly that hole. If more than half fail, the attempt
   writes the whole file instead.

## Settings

| SessionConfig | LoopConfig | Default | |
|---|---|---|---|
| `diffusion` | `canvas` | `"auto"` | `"off"`: whole files always |
| `diffusion_candidates` | `canvas_candidates` | `2` | canvases per repair attempt |
| `diffusion_notes` | `canvas_notes` | `True` | the planning call before a pinned file's bodies |

## The host's side

A host opts in by setting `supports_canvas` and implementing one method on
its LLM port — `CanvasLLM.canvas_fill(request, *, cancel) -> CanvasResult`
(`ports.py`). It never raises; errors, a hole that does not fit, and Stop
come back as results. Tokens are the host's business (M14): the core sends
text (`head`, `seed_text`, `tail`, `noise`, `slack`, `max_canvases`) and
receives the free region's text plus a per-line confidence (exp(−entropy),
1.0 = certain) and per-line "flips" (how often a token changed its mind).

ATK implements it in `atk/core/ccoder_host.py` → `LLMEngine.canvas_fill` →
`dgemma.DiffusionGemmaLlama.canvas_fill`, on the DiffusionGemma server built
by `get_diffusion_gemma.bat` with **ATK patch 2** (the canvas layout, held
positions, the warm start, the readout, a prompt cache).

## What the operator sees

* At start: *"diffusion mode: this model fills a canvas, so a repair
  rewrites only the lines its error is in…"*
* Per hole: *"src/calc.py: attempt 2 rewrites lines 3-4 (`add`) as a hole —
  its error is there; 2 canvas(es)"* and *"… filled — tail canvas, 6
  step(s), the model 85% sure on average"*.
* The event's data carries the filled lines, their confidence and flips,
  so a host can draw a **doubt map**. ATK's Coder panel does: Engineer mode
  shows the lines tinted green / amber / red with ⟳ marks; Guided mode shows
  one coloured square per line and says where to look.
* The journal records a `canvas` event per hole: the region, why it was
  chosen, every candidate (style, noise, seed, confidence, steps, prefill,
  cache) and which one was kept — and a `canvas` event with `kind: "skip"`
  every time diffusion mode stepped aside, with the reason.
* **BUILD_LOG.txt** says all of it in words: the hole and why, each canvas
  and why it was rejected, the lines cut out, the kept lines with a
  confidence column (`?` under 50 %, `~` under 85 %, `--` not read back),
  the notes in full, and every fallback to a whole file. Search the log
  for `diffusion ` to follow one file's holes; search for ` ? |` to find
  the lines the model was least sure of.

## What is not known yet — read the journal after the first real builds

* **Which style wins.** `tail` (the far side held, padding needed) or
  `open` (no padding question). The `canvas` events record which candidate
  was kept; a few builds will say.
* **Whether the model pads.** A held tail makes the free region a fixed
  size. Blank lines are legal Python and the cleaner strips them; whether
  DiffusionGemma pads or rambles is an empirical question.
* **Speed on a 16 GB card.** With experts in system RAM, a step costs what
  it costs. The prompt cache makes the 2nd+ candidate's prefill nearly
  free; the server log's `prompt eval time … N reused` line shows it.

## What the first real build taught (2026-10-16)

The Star Trek spec, DiffusionGemma alone: 14.5 minutes, two files built,
then a host crash. Most of it was the harness, not the model —
docs/HYBRID.md has the whole account. What changed here:

* every hole prompt now carries the FILE'S SLICE of the request
  (`specslice.py`), not its first 2,000 characters — the formulas the
  bodies were judged against sat past that cut;
* the notes call sees the same slice (it saw no request at all);
* `clean_fill` drops the prompt's own marker lines a canvas copies back,
  and cuts at a definition repeated inside one fill; `judge` rejects a
  fill that makes the file define a name twice (the second definition
  silently wins), and SALVAGES a reply that finished with `length` where
  the run-on cut finds the point it left the region — four of the five
  canvases the build threw away held the right body in their first lines;
* a failure that names no line of the file is located through the names
  the FAILING TEST calls (the asserting line's own call first) before the
  doubt-guided fallback;
* `open` runs before `tail` in `REPAIR_SCHEDULE`: the one kept `tail`
  canvas echoed the markers and re-opened the function;
* the escalation ladder (`loop.Sampling`) reaches a canvas model as a new
  seed and no warm start — never a temperature, which does not reach it;
  the `generate` line says so instead of printing one.

## Hybrid — built (docs/HYBRID.md)

A chat model writes every file, ONE swap, DiffusionGemma rewrites only the
hole each remaining failure is in, one swap back for whole-file repairs of
what is left. `SessionConfig.hybrid` ("auto" / "off" / "ar-first" /
"diffusion-first"); the host side is `ports.ModelRoster`.

## Not built yet, on purpose
* **Native tool calls for DiffusionGemma.** It uses the text-marker
  fallback (D10). Its chat template has tool calls (`<|tool_call>`); the
  server would need to return special tokens for ATK to parse them.
* **Several seeds in one denoising step.** If the step is dominated by
  moving expert weights over PCIe, batching candidates is nearly free.
  Measure ms/step first.
