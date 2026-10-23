# Narrowed edits and the dropped-code guard

Two changes that together decide how an EXISTING file is changed, built
2026-10-19 from a comparison with Aider, OpenHands and Devika (ideas
re-implemented here, no code taken). Code: `cognitive_coder/symedit.py`,
`cognitive_coder/keep.py`, their wiring in `cognitive_coder/loop.py`.

## The problem

Every repair, and every change to a file that already had real code in it,
asked an ordinary (non-diffusion) model for the **complete file**. On a small
local model that costs three ways:

| | |
|---|---|
| **Decode time** | A whole-file reply runs to `max_tokens` (2,048) before continuation. At the 3–5 tok/s Devstral ran at on 2026-10-16 that is 7–11 minutes an attempt. One function is a few hundred tokens. |
| **Collateral damage (D6)** | A model rewriting 400 lines to fix one touches others — or leaves them out: `# ... rest of the methods unchanged`. On a project with no tests nothing failed, and the code was gone. |
| **Large files** | A file too large to show whole was refused before any model call. |

F5 of the build specification ("progressive narrowing: whole file → class →
function → line range") said what to do and was never built:
`LoopConfig.narrow_on_stagnation` existed and nothing read it.

## The dropped-code guard (`keep.py`)

Runs **before the write**, in the guard's position, for any file that had real
code when its task began (not a stub, not a pinned skeleton). A reply is
**refused** — never written, the operator's file still whole on disk for the
next attempt — when it:

- leaves out a function, class, method, UPPER_CASE constant or the
  `if __name__ == "__main__":` block that the request does not name;
- renames one while another file (or the reply itself) still uses the old
  name — a rename nothing uses any more is allowed;
- turns a real body into `...` / `pass` / `raise NotImplementedError`;
- has a placeholder comment where code was (`# ... rest unchanged`,
  `// existing code here`, `# (rest of file unchanged)`). A placeholder the
  file already had is the file's, not the reply's.

The refusal is a located error the next attempt sees ("Your reply left out
`draw_hud`…"), and the next attempt is a **narrowed edit**: the cure for a
reply that leaves code out is a reply that holds only what it changes.

What the operator asked for may go — but the asking has to be specific,
because a long specification uses every verb somewhere:

- a definition is ALLOWED to go when one sentence of the request names it
  with an intent word ("Remove `summary`", "rename `load` to `read`", "turn
  these functions into a class"); it is logged as a warning. A name merely
  mentioned ("add a guard" does not let `add()` go), or named only in a
  traceback, exempts nothing;
- a definition MOVED is not lost: the reply imports it (it went to another
  file), or defines it under a class (a function made a method) and nothing
  still calls it bare;
- a rewrite of the whole file has to be asked for in a sentence that names
  the file ("rewrite `game.py`", "start over"); then a missing definition is
  a warning.

Placeholders are refused whatever the request says. Python is checked
through its AST (a property's setter counts as its own definition; a file
with a byte-order mark is read past it); other languages through the
codemap's parsers (pattern-matched, so one missing name there is not enough
to refuse a reply).

`SessionConfig.keep_existing` (default on); `ccoder build --no-keep`.
BUILD_LOG.txt: a `keep` decision per attempt, naming what was left out.
Journal: `keep`.

## Narrowed edits (`symedit.py`)

A narrowed reply holds **only the definitions it changes** — each function,
method or class complete — and they are put back **by name**. Aider's
leaderboard is the reason for the shape: whole-file replies are what small
models get right (Qwen2.5-Coder-32B: 99.6 % well-formed whole, 71.6 % with
search/replace blocks), and its unified-diff study's advice is to rewrite
whole functions, not lines. A narrowed reply is whole-file output at
function size.

**When** (`SessionConfig.narrow`, default `"auto"`):

| | |
|---|---|
| a Python repair of a file over `narrow_min_lines` (40; was 150 until 2026-10-22 — see HYBRID.md, the second-attempt finding) lines | the functions the error is in |
| the same attempt came back twice (F5), or a reply was refused for leaving code out | for the rest of that task |
| any repair or change when the whole file does not fit the context | the targets in full, every other definition as its header (`condensed_view`) |
| `"always"` | every repair and change of an existing Python file |
| `"off"` | never — the whole file, as before |

Shorter files keep the whole-file path they were tuned on. Diffusion models
keep their canvas holes. Other languages keep the whole file (said in the
log). The ladder's fresh start (attempt 3) is always a fresh start — a whole
file, the earlier attempts not shown — and narrowing, once on, resumes on
the attempt after it; for a file too large to show whole, the fresh start
is narrowed too.

**How a reply is placed** — by rule, the model decides nothing:

- a function, constant or main guard the file has is replaced where it is;
- a method may come inside its `class` line (only the methods given are
  replaced; new ones are added at the end of the class) or bare at column 0
  (resolved through the error's location, or the one class that has it);
- new definitions go before the main guard; new imports join the leading
  import block (a `from __future__` import goes first; a shebang stays line
  one);
- a name defined twice (a property and its setter) is never replaced by
  name; a body that is only a placeholder is never spliced over real code;
- a top-level statement that is not a definition, a partial class that
  changes class-level lines, a result that does not parse or defines a name
  twice — refuse the **whole** reply. A half-applied change verifies as
  something nobody asked for.

A reply that cannot be placed falls back to the whole file in the same
attempt; for a file too large for that, the attempt is refused with the
reason, and the next one is told it. A reply that returns every definition
unchanged is the file returned unchanged — the model's "the test is wrong" —
and goes to the same rule and the same judge as a whole file would.

Every placement keeps the reply's order (a helper before the main guard
that calls it, a base class before its subclass); a new constant that uses
a definition goes after the definitions. A method reply is placed as the
method even when the file has a top-level function of the same name; a
partial class that changes its class line, decorators or docstring is
refused; a nested class in a partial reply is merged member by member; a
name defined inside a top-level `if`/`try` is never shadowed by a new one.

BUILD_LOG.txt: a `narrow` decision per attempt — why, what was asked for,
whether the file was shown whole or condensed, what was replaced/added, or
why the reply could not be placed (with the reply). Journal: `narrow`; the
`generate` event carries `via: narrow`.

## Also in this change

- **Where the errors are** (`diagnostics.where_block`, Aider's lint
  context): every repair prompt carries each located error's line marked
  `█`, inside the def/class lines that hold it, with line numbers — in its
  own block, never in the file the model copies.
- **A fresh start of a file that was here before** shows the file as it
  stood when the task began. The ladder hides the broken ATTEMPTS (D11); the
  operator's own file is not one of them.
- **Regression memory is read back** (F10). A fix is recorded by what
  changed (`changed `Car.update` in src/car.py; added 1 import`), keyed by
  the error with numbers blanked, and a later repair of the same error is
  told how it was fixed before — a hint. It used to write "model repair" for
  every fix, and nothing read the table.
- **Describe first** (opt-in, `diagnose_first`, `--diagnose-first`): one
  short call says what is wrong before each repair; the repair is shown it.
  Aider's architect/editor split with one model; one more call per repair.
