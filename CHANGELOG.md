# Changelog

Kept from the first commit, per §10.4. Without PyPI, **the git tag is the
release** — so this file and the tags are the whole record.

The format follows [Keep a Changelog](https://keepachangelog.com/); versions
follow semver, where the public API is `cognitive_coder/__init__.py` only —
the Ports of `ports.py` and the shared types of `types.py`. Breaking either
is a major version.

## [Unreleased]

### Added — Continue: a stopped build picked up where it left off (2026-10-23)

Bill: "What if an operator stops the coding, but wants to go back to where
he left off?" `Session.resume` existed (§6.13) but was not something an
operator could lean on — it rebuilt no task board, wrote no build-log
header, forgot the pinned constraints, the spec vectors, the hybrid phases
and the test baseline, and trusted the journal over the disk. Now it is the
engine half of a Continue button.

- `resumable.py` — what a project's journal says about every build in it,
  as plain data a UI can show without running anything: how it ended
  (`stopped` / `budget` / `error` / `crashed` / `finished` / `abandoned` /
  `preview` / `empty`, judged on the LATEST run so a continued build is not
  still "stopped"), parts done and left, last file, model, a title from the
  request, and one sentence in Guided or Engineer register. `scan(root)`
  finds the unfinished builds under a projects folder read-only;
  `abandon()` sets one aside (journal event `abandoned`, files untouched).
- `Session.resume` restores everything a run had: the baseline
  (`known_failing`), the language, constraints, spec vectors, hybrid
  phases; emits a `plan` event (`continued: true`, per-task `verified`) so
  the board shows green work green; writes `SESSION … — CONTINUED` into
  `BUILD_LOG.txt`; journals `continue`.
- **Re-verify before writing anything.** `session_end` now records a hash
  of every plan file. On Continue, if nothing changed and the last run
  ended cleanly, nothing is re-run (`reverify` journalled `skipped`);
  otherwise every done file and every edited pending file is checked
  again. A done file that still passes is kept; a hand fix to a pending
  file that passes is KEPT, not overwritten; a file broken during the
  pause goes back to pending and is repaired from what is there. A crash
  (no `session_end`) is treated as "unknown — check everything".
  `SessionConfig.reverify_on_continue` (default on).
- `resume_state` is last-verdict-wins: a file that failed and was then
  fixed is done, and one that was done and failed a re-check is not —
  the old append-only reading kept both.
- **The graceful stop.** `Session.stop_after_current()` — thread-safe like
  `cancel()` — finishes and checks the part being written, then stops
  before the next one: nothing in progress is thrown away or rolled back.
  `Cancelled(after=path)`; the `cancel` journal event and the status
  carry `after` and `graceful`.
- CLI: `ccoder resume` lists each build with how it ended and
  "← can be continued"; `ccoder resume --last` continues the newest
  unfinished build; `ccoder resume --discard ID` sets one aside.
- `.cc_tabs/` — where ATK now keeps each chat tab's conversation per
  project — joins `SKIP_DIRS` and `scope.SKIP`: a transcript is never
  indexed, scoped or offered to the model as source.
- `tests/test_continue.py` (25) and 4 CLI tests: 1520 passed, 13 skipped.

### Added — the four tabs built, and a grammar-backed structured ask (2026-10-22)

Bill asked to finish the tabs ("the most bang for buck per reinstall") and
then to make the tool "the best coding tool for small models possible"
(24B–40B), chat-first because "being able to chat with the model for
brainstorming helps — each tab brings a tool to the table."

**Four engine modules, each chat-first where a conversation helps, each
constraining the edit shape and nothing else:**

- `scope.py` — one definition of a working set: the whole project, one
  file, one folder, or an explicit list. Skips `.venv`, `.cc_*`; source is
  `*.py`. Every tab's file/folder picker resolves through it.
- `bughunt.py` — the Evaluate & Bug Hunt engine. Ruff, format check,
  codemap unresolved, `keep` scan and pyright/mypy run first; the model's
  pass runs LAST and every bug it names is a real line or it is set aside
  (`could_not_locate`). `votes>1` asks the model repeatedly and keeps only
  what two passes place within ±2 lines of each other; the rest are listed
  `unconfirmed`, never counted. `to_debug_reports` hands located errors to
  Debug & Repair.
- `improve.py` — converse (consultant / expert / mad-scientist / structurer
  personas) → `structure` the agreed conversation into a checked change
  list → `apply` each checked item through the repair loop, one at a time,
  with an `IMPROVEMENTS.md` ledger each later item is shown.
- `redteam.py` — scan (guard + whatever scanners are installed) →
  `threat_model` → `attack`: one reproducer per threat, a unittest under
  `tests/redteam/` that FAILS against the code as it is (and is itself
  guard-checked, so an attack that leaves the sandbox is refused, not run).
  A finding without a reproducer that fails now is a guess and marked one.
  `converse` lets the attacker withdraw by id (`WITHDRAW t1: why`);
  `to_repairs` carries the reproducer's command so Debug & Repair confirms
  the fix by the reproducer going green. `to_json`/`from_json` so a UI can
  hold the state between turns.
- `blueprint.py` — a system of modules. `converse` (systems-architect
  persona) plans it FIRST — Bill's point that the back-and-forth belongs
  before the design, not after. `decompose` turns the agreed plan + spec
  into modules with public contracts; `validate` refuses a blueprint whose
  contracts lack a single owner, whose files are double-owned, or whose
  dependencies cycle. `order` is dependencies-first; `write_stubs` gives a
  module the contract stubs of dependencies not built yet; `run_module`
  builds one module through its own session with a `BLUEPRINT.md` ledger.

**`structured.py` — one way to ask a model for a JSON object, and get one.**
Every tab (and Debug & Repair's locate) asks through it: a GBNF grammar
when the host enforces one (`supports_grammar`), and ONE retry that quotes
the parse problem back with the first 300 characters the model sent. A
24B–40B model returns usable JSON nine times in ten; this makes the tenth a
second ask instead of a silent `{}`. `improve.structure`,
`blueprint.decompose`, `redteam.threat_model` and `redteam.attack` and the
bug-hunt model pass were moved onto it.

Journal events added: `structured`, `bughunt`, `improve`,
`improve_structure`, `improve_apply`, `redteam_threats`, `redteam_attack`,
`blueprint`, `blueprint_plan`, `blueprint_module`. Six personas added to
`personas.py`: consultant, expert, mad_scientist, structurer, attacker,
systems_architect. 20 new tests in `tests/test_tabs_engine.py`; full suite 1498
green on Python 3.13.


### Added — the harness fixes from five failed builds; `ccoder postmortem`; the next tabs designed (2026-10-22)

Five journals (10-18 ×2, 10-20 ×3) and 26 Debug & Repair sessions were read
from `.cc_journal/*.jsonl`; the findings and their fixes are the new last
section of `docs/HYBRID.md`. Most of the minutes went to the harness, not
the models. Every fix below is pinned by a test that reproduces the
journal's own shape.

**The second attempt** (`narrow_min_lines` 150 → 40, `loop.py`,
`session.py`). Shown its whole file plus the error, Devstral returned the
file byte-identical on attempt 2 in every phase of the 10-18 build — ten
minutes per file-task at 3–5 tok/s for nothing. A repair of any file over 40
lines is a narrowed edit from attempt 2; whole-file stays for the first
attempt of a new file, and for files too small to narrow.

**The deterministic floor** (`LoopConfig.format_before_write`,
`LoopConfig.lint_gate`; `tests/test_deterministic_floor.py`). `format_code`
and `lint_code` were defined, tested, exported to ATK — and called by
nothing. Now the reply is formatted before it is written (ruff/black when
installed; one warning per session when neither is), and a lint finding
that is a certain bug — F821 undefined name, F811 redefinition, F823,
F631/F632, F702/F706, E999 — joins the verify verdict: a run that passed
with an undefined name the tests never reached is not verified, and a run
that failed gets the linter's exact line beside the traceback's. Both are
`SessionConfig` fields; `lint_gate=()` turns the gate off (the canvas tests
do, because their fixtures plant the very error the gate would catch first).

**`keep.py`: a planted placeholder is never protected and never accepted.**
A debug session planted `def run_headless_demo_fixed(): pass` beside the
real function plus the model's own narration as comments; the next two
sessions were refused for dropping the stub. Now a NEW function with an
empty body is refused on arrival (`Finding.planted`); a new name that is an
existing name plus `_fixed`/`_new`/`_v2`/`_2`/`_corrected`/`_updated`/`_old`
is a copy, refused (`Finding.duplicated`); a definition the file already
holds with a placeholder body is not a loss when dropped; a comment that is
the model narrating its reply ("I will provide the clean, corrected
version", "Note: the user …", "Re-defining …") is a placeholder.

**`nonerror.py`: tracebacks that are not the code's fault.** Four of 26
debug sessions were `attempted relative import with no known parent
package` / `No module named 'src'` — the program run from inside `src\`.
Debug & Repair now answers these, and a missing third-party package, with a
sentence and the command (`python -m src.main`) before any model call; a
report that NAMES a file still repairs.

**`speclint.py`: what a spec says that a build will faithfully get wrong.**
`Reserve_energy` (capital R) was pinned as written, then half-corrected by a
model. Before the plan, the request is linted for a capitalised or camelCase
field in a snake_case spec, one name in two spellings, and one constant
with two values; each is a warning with the line and the probable spelling,
journaled as `speclint`, never applied.

**The lying green.** Two re-runs over one folder ended `ok: true` with
every file untested: the folder's own tests were all already failing, got
baselined away, and nothing checked what was built. The baseline now says
when EVERY test module is excluded (`baseline.all_excluded`, a warning), a
session with tests present and none run ends `ok: false` with
`session_end.why_not_ok`, and the report's first line is `[UNTESTED]`. A
project that never had tests and whose plan wrote none is still "untested,
ok" — the 10-19 decision stands.

**`ccoder postmortem`** (`postmortem.py`). The journals read back and ranked
by the minutes they cost — same-size second attempts, canvas skips with
their reasons, located-not-repaired, not-a-code-error, no-tests-ran, a
model under 8 tok/s — each with its harness fix named. Run on the 10-18 and
10-20 journals it reproduces every finding of the hand reading and adds one
(16 of 23 canvas repairs skipped in the 10-20 build).

**`docs/NEXT_TABS.md`** — the design of record for Blueprint (a system of
modules of files: contracts between modules, one session per module in
dependency order, a cross-module ledger, stubs from contracts), Improve (a
conversation that ends in a checkboxed change list applied by narrowed
edits, with a persona picker and an evolutionary mode), Red Team (an
attacker that must prove every finding with a reproducer, scanners first,
threat model shown before any attack, findings become Repair tasks with
the reproducer as the acceptance test), Evaluate & Bug Hunt (the floor made
visible, a type-checker lane, test adjudication shown, properties from the
spec), and the per-file right-click. Engine operations before tabs.

**Two stale needles** in `test_repair_acceptance` and
`test_interfaces_pinned` expected `Tuple[...]` where the enricher now
renders `tuple[...]`; widened to accept the shape in either spelling.


### Added — narrowed edits, the dropped-code guard, a ranked repo map, the regression suite; what Aider, OpenHands and Devika had worth taking (2026-10-19)

Bill asked which features of Aider, OpenHands and Devika would improve this
engine — to replicate, not to use. The three were read at source (Aider
0.86.3.dev; OpenHands' current agent SDK and its 0.62.0 tag; Devika's last
code commit, 2024-09-19) and compared with this engine's code. What follows
is what was worth building, re-implemented here — no code was taken. Two
gaps turned up in this engine's own code on the way and are fixed below:
F5 was specified and never wired (`narrow_on_stagnation` was read by
nothing), and F10's regression memory was written and never read. Design:
`docs/NARROWED_EDITS.md`, `docs/BENCH.md`.

**The dropped-code guard** (`keep.py`, `LoopConfig/SessionConfig.
keep_existing`, on; `--no-keep`). A change to a file that already had real
code asked for "the complete updated file", and only a file pinned THIS
session was held to its names. A reply that left out `draw_hud`, or wrote
`# ... rest of the methods unchanged`, was written to disk; with no tests,
nothing failed. Now such a reply is refused BEFORE it is written — the
operator's file is still whole for the next attempt — when it drops a
function, class, method, UPPER_CASE constant or main guard the request does
not name, renames one something still uses, turns a real body into `...`,
or has a placeholder comment where code was (Aider counts these "lazy
comments"). A rewrite request ("refactor", "remove", "rename", …) makes a
missing definition a warning; a placeholder is always refused. The refusal
is the next attempt's located error, and that attempt is a narrowed edit.

**Narrowed edits** (`symedit.py`, F5; `SessionConfig.narrow` auto /
always / off, `narrow_min_lines` 150; `--narrow`). A repair returns only the
definitions it changes — each whole — and they are put back BY NAME:
whole-file output at function size, the format Aider's leaderboard shows
small models get right (Qwen2.5-Coder-32B: 99.6 % well-formed whole, 71.6 %
with search/replace). "auto": a Python repair of a file over 150 lines; the
rest of a task after the same attempt twice or a dropped-code refusal; and
any repair or change of a file too large to show whole — which used to be
refused before any model call, and is now shown condensed (targets in full,
every other definition as its header). A bare method finds its class; new
definitions go before the main guard, new imports into the leading block,
`from __future__` first, a shebang stays line one; an ambiguous name, a
placeholder body, a stray top-level statement, a partial class that changes
class-level lines, a result that does not parse or defines a name twice
refuse the whole reply. Unplaceable: the whole file in the same attempt, or,
for a file too large for that, a refused attempt with the reason. New
persona `editor`; contract `CONTRACT_SYMBOLS`; journal event `narrow`;
`generate` events carry `via: narrow`.

**Where the errors are** (`diagnostics.where_block`, `mark_errors`, on).
Aider's lint context: every repair prompt carries each located error's line
marked `█` inside the def/class lines that hold it, numbered, in its own
block — never in the file the model copies.

**A fresh start of a file that was here before** is shown the file as it
stood when the task began. The ladder's third rung hid the broken attempts
(D11) — and with them the operator's own file, which a fresh start then
regenerated blind.

**A forgiving anchor** (`patcher.forgiving_replace`). `CC-EDIT` blocks and
`apply_patch` (and ATK's chat edits, through `ccoder_compat.apply`) matched
exactly or not at all. Now, after the exact whole-word match: trailing
spaces, an indentation shifted the same on every line (the replacement
re-indented with it), blank edge lines, and `...` lines standing for the
code between pieces — every fallback on WHOLE lines and unique, so the
whole-word rule holds. A refusal names every match's line when there are
several, the lines it probably meant when there are none ("Did you mean
these lines, from line 12?"), and "already applied" when it is. Aider
measured nine times more failed edits with its flexibility off.

**A ranked repo map** (`codemap/rank.py`, `Store.file_graph`,
`CodeMap.ranked`). Aider's PageRank over the reference graph the codemap
already holds. The architecture block listed the first 60 files
ALPHABETICALLY; past 60 it now keeps the most referenced (still printed in
path order, still snapshotted per epoch, so byte-stable). Debug & Repair's
locate step offers files most likely first — toward the names and paths the
error and the description mention — instead of in path order under a cap;
the audit reads the files the project depends on most before the biggest.

**A parser's syntax check when there is no compiler**
(`parse_treesitter.syntax_errors`, `runner.treesitter_syntax`). Aider lints
every language it has a grammar for. With no toolchain, a C file was "not
checked" by the audit and BLOCKED by the build, a stray brace in it. Now the
brace is a located error the model can fix; a clean parse is still "not
checked". TypeScript is left out, as Aider leaves it out.

**Skills with triggers** (`skills.py`). OpenHands' keyword-triggered
knowledge: `triggers: pygame, sprite` in a skill's header loads it only for a
request that names one (whole words). Matched once, at session start,
before the first prompt — the prefix still never changes mid-session.
`ccoder skills` lists them apart.

**Regression memory, read back** (F10, `recall_fixes`, on). A fix is
recorded by what changed between the failing attempt and the passing one
(`changed `Car.update` in src/car.py; added 1 import`), keyed by the error
with numbers blanked; a later repair of the same error, in any file, is told
how it was fixed before, as a hint. It used to record "model repair" for
every model fix, and nothing read the table. Journal event `recall`.

**Describe first** (opt-in: `diagnose_first`, `--diagnose-first`). Aider's
architect/editor split with one model: one short call says what is wrong
and what to change before each repair, and the repair is shown it. Aider
measured gpt-4o-mini 55.6 % → 60.2 % with the same model in both roles. One
more call per repair, so off by default.

**The regression suite** (`bench.py`, `ccoder bench`, `bench/tasks/`). Aider's
benchmark design and OpenHands' separate grading: tasks run headless, graded
by hidden tests copied in after the run, each run compared with the one
before task by task ("better: …, worse: …"), every reply recorded, and
`--replay` answering from a recorded run with no model — saying where the
prompts diverged, so a diverged replay is not mistaken for a result.
`--check` proves each task's hidden tests pass on its reference and fail on
its start files. Eight starter tasks (build, change, debug; two of them
files over 150 lines). Runs go to `bench/runs/` beside the code
(`.gitignore`d). For the engine, not for comparing models.

**Not taken, and why.** Git auto-commits (the guard deliberately blocks git;
snapshots and undo already exist); conversation summarising (this loop never
accumulates history — M33); web research and browsers (offline by design);
Docker sandboxes; a model-reported "security risk" field; a "think" tool;
OpenHands' 6–7k-token prompts; Devika's pipeline as a whole (abandoned since
2024; its search results never reached its coder). Grammar-constrained JSON
for the judge and the audit is worth doing and is left for a build that can
be tested against the real llama.cpp server.

Public: `keep`, `symedit`, `bench` modules; `SessionConfig.keep_existing`,
`narrow`, `narrow_min_lines`, `mark_errors`, `recall_fixes`,
`diagnose_first`; `patcher.forgiving_replace`; `CodeMap.ranked`; journal
events `narrow`, `keep`, `recall`; persona `editor`. All additive.

**Fixed before it shipped — an independent review of the change found
twenty-three defects, each reproduced, each now a test**
(`test_review_oct19` and the module suites):
- a method reply (`def run(self)`, the error in `App.run`) overwrote a
  top-level function of the same name — silently, and the task "passed";
- new definitions placed at one spot came out in REVERSE (a helper after
  the main guard that calls it, a subclass before its base), and a new
  constant that uses a function went above it — NameError at import;
- the guard was switched off by ordinary wording: "remove", "split",
  "merge" anywhere in a request made every loss a warning, and any name the
  request or a traceback merely MENTIONED could be dropped unannounced —
  the `change-todo` task built to test the guard did not trigger it. Intent
  now has to sit in one sentence with the name; tracebacks exempt nothing;
  a rewrite has to name the file;
- the guard refused legitimate refactors forever: functions turned into a
  class's methods, functions moved to another file and imported. Those are
  now MOVES. "# Other functions", "# Keep the original implementation…" and
  a docstring bullet were taken for placeholders;
- refusals piled up — five in a row carried five copies of the refusal and
  pushed the real NameError out of the prompt — and pointed at line 1 or at
  a line of the reply, which the narrowed edit then took as the error's
  function;
- "did you mean" compared every window of the file character by character:
  32 s for a 40-line anchor on a 1,000-line file, minutes beyond. Now
  candidates are chosen by exact line overlap first (milliseconds);
- an unplaceable narrowed reply could end the task as "the model returned
  nothing" when the whole file did not fit either; it is now the attempt's
  refusal, with the reason;
- a fresh start of a file too large to show had its original CUT to fit
  while being asked for the whole file; Debug & Repair round 2 regenerated
  the operator's file blind; once narrowing was on it took over the
  ladder's fresh start, while the log still said "starts fresh";
- a partial class reply dropped a new base class, decorator or docstring
  while reporting success; a nested class in a partial reply replaced the
  whole nested class;
- on Python 3.12+ a multi-line f-string inside a moved method had its value
  changed by the re-indent (3.12 tokenizes f-strings in pieces);
- a file saved with a byte-order mark switched the guard and narrowing off
  (and, already before this change, failed the syntax check every time);
- a dropped property setter passed the guard, and a setter could not be
  repaired by name (a property and its setter are now `v` and `v@setter`);
- the tree-sitter check reported valid C with `#ifdef` or macros as broken;
- a debug task's replay always "diverged" at call one (the traceback named
  the run's own, time-stamped folder; tasks now run in a fixed folder);
- an unchanged narrowed reply lost the "return it unchanged if the test is
  wrong" escape hatch; the "enriched" status went silent; the WHERE block
  counted lines with `str.splitlines`; a fallback call was not journaled; a
  path inside another file's name ranked as a mention; an anchor matching
  twice after spacing was reported as missing; the suite's pytest totals
  read the first "N passed" in the output rather than the summary.

Tests: `test_symedit` (31), `test_keep` (13), `test_narrow` (9),
`test_review_oct19` (21), `test_forgiving_edits` (9), `test_rank` (7),
`test_treesitter_syntax` (5), `test_bench` (7), two more in `test_skills`.
Whole suite (Python 3.11, Linux): 1451 pass, 8 skipped, and the same
2 failures the untouched code has here (both assert how a `Tuple` alias
is printed in an interface block; neither involves this change).

### Added — hybrid mode, the judge, spec vectors, the ladder; what two real builds taught (2026-10-16)

The first two builds of a real specification (the Star Trek spec: 15,070
characters, seven source files, three test files) — one on DiffusionGemma,
one on Devstral-24B — both failed, and the journals put most of the damage
on the harness. `docs/HYBRID.md` is the account; this is the change list.

**Hybrid mode** (`hybrid.py`, `Session._hybrid_phases`, `ports.ModelRoster`,
`types.ModelSlot`; `SessionConfig.hybrid` = auto / off / ar-first /
diffusion-first, `hybrid_writer`, `hybrid_repairer`). With a host that
offers a roster of models, a chat model writes every file, ONE swap and the
canvas model rewrites only the hole each remaining failure is in, one swap
back and the chat model repairs what holes could not fix with fresh-start
prompts. Three phases, two swaps per build — never one per attempt. Roles
by capability, never by name; each phase in dependency order; a phase with
nothing left is skipped; a swap the host refuses is said and the build goes
on. M10 amended: the core asks the host to load a model at a phase boundary
and still loads nothing itself. BUILD_LOG.txt gets a header per phase, each
swap with its duration, and a summary of what each phase fixed.

**The judge** (`adjudicate.py`, `SessionConfig.adjudicate`, on). When a
model's test disagrees with the module it tests, one short question is
asked — which does the REQUEST support — and the answer's quote is checked
against the request. `code` → the module is repaired, as before; `test`
(with a real quote) → the TEST is repaired, with the quote as its
diagnostic; `unclear` → the old conservative rule, said to be unclear. A
module repair that returns its file unchanged ("I am certain the test is
wrong") now asks the same question instead of giving up. Ten of the Star
Trek build's sixteen failing tests were the test's fault.

**Spec vectors** (`specvectors.py`, `SessionConfig.spec_vectors`, on). The
examples a request states outright — `` `"1-3"` -> `(1, 2, 3)` `` beside a
named function, `` `SUBTICKS_PER_TURN: int = 10` `` — lifted by rule into a
test file per module that the harness owns
(`tests/test_spec_vectors_<stem>.py`), run by the runner BEFORE the
module's own tests; a failing vector is the module's failure, with the
request's sentence as the message. Seven for the Star Trek spec: the seven
that cannot be wrong.

**The escalation ladder** (`loop.Sampling`, `LoopConfig.escalate`,
`temperatures`). Attempt 1 at the persona's temperature; 2 a repair with
the enriched diagnostics; 3 a FRESH START (the broken file is not shown;
`personas.fresh_start_task`) at 0.45 with a seed; 4+ whole-file repairs at
0.70 with new seeds. Stagnation moves to the next rung instead of stopping
the file while a rung is left. The `generate` line and the journal record
the temperature and seed USED; a canvas model is told a seed and the line
says a temperature does not reach it. ATK's host no longer varies sampling
on its own when the engine has this.

**The request, sliced per file** (`specslice.py`; `SessionConfig.
request_chars_first` 16,000 / `request_chars_repair` 8,000). Every
diffusion prompt and every whole-file repair repeated the request's first
2,000 characters — past which every formula of the Star Trek spec sat.
Now: the file's own section, the hard requirements, the manifest lines
naming it, the opening; what is left out is named in the prompt and in the
log. The notes call sees it too (it saw no request at all).

**Honest logs.** A crash reaches BUILD_LOG.txt as `STOPPED UNEXPECTEDLY`
with the traceback, and `session_end` is never `ok` after one (it said
`ok: true` from the two files built before the crash). Every build log now
closes with a SUMMARY (only Debug & Repair's did). A **speed probe** after
the model check and after every swap writes the model's tok/s and prefill
to the log and warns under 8 tok/s that the model is spilling into system
RAM (Devstral ran the whole build at 3–5 tok/s unnoticed;
`SessionConfig.speed_probe`, never for the scripted test doubles).

**Smaller, each from one failure.** `constraints.py` strips emphasis only
at word boundaries (`test_physics.py` was pinned as `testphysics.py`). The
skeleton is IMPORTED, not only parsed (`Planner._import_skeleton`); a
dataclass with defaulted fields before required ones is made valid with
`field(default=…, kw_only=True)`, order and names kept
(`interfaces.fix_dataclass_order`); hard-requirement sentences naming a
pinned module or a name it declares go into its docstring word for word
(`interfaces.with_conventions`); the architect is asked to put the
request's conventions in docstrings. `[run]` executes `python -m
src.physics` from the project root for a module inside a package
(`langs.module_run_cmd`, `langs.module_name`). Canvas: marker echoes
dropped, a definition repeated inside a fill is a run-on cut, a fill that
makes the file define a name twice is rejected, a `length` reply is
salvaged where the run-on cut finds the exit from the region, holes are
located through the names the failing test calls before the doubt-guided
fallback (`canvas.test_named_functions`), `open` before `tail`.

Public: `ModelSlot`, `ModelRoster`. Journal events: `adjudicate`, `phase`,
`swap`. New persona: `judge`.

### Added — Debug & Repair, the mutation check, the pinned requirements (2026-10-10, night)

**Debug & Repair** (`debugrepair.py`, `Session.debug`). Bill: *"if I can
try running it myself on my own and then feed the errors back into
cognitive coder … a codebase debug / repair tab"*. A `DebugReport` carries
any of: the error text as printed, the command that was run, a description
in words, the file(s). The session LOCATES the fault (the parser every
build uses; the deepest frame in THIS project, made relative; otherwise ONE
model call naming at most three files from the project's list — an
invented name is dropped — with `confirm_targets` stopping there with the
proposal), REPAIRS each file through the ordinary `Loop.run_task(seed=…,
baseline=…)` (same attempts, verification, transaction, snapshots and
approval as a build; a no-change "repair" stops at once), CONFIRMS by
running the operator's command again (a program still running at the
timeout with no error counts as clean, and says so; the same error back is
a failed round, a different one drives the next, up to `rounds`), and
RECORDS every round: BUILD_LOG.txt in words, `debug` journal events,
`.cc_state/debug/<session>.md` with each file's diff, and
`.cc_state/debug/history.jsonl`. States: fixed, repaired (unconfirmed — no
command), still-failing, no-change, nothing-to-repair, located, not-located,
stopped. A debug session also pins the requirements of the project's
BUILD_SPEC.md (below). Public: `DebugReport`, `DebugResult`, `DebugRound`,
`debug_history`.

**The mutation check** (`mutation.py`, `SessionConfig.mutation_check`, on).
"Verified" means the tests passed, not that they would fail on wrong code —
and a small model writing both is where they might not. After a build, up
to `mutation_max` deliberate bugs (`<`→`>=`, `==`→`!=`, `+`→`-`,
`and`→`or`, a returned constant changed) go into a COPY of each verified
Python module under `.cc_state/mutants/`, its own test file runs against
each, and a bug the tests do not notice is a WARNING with the line and the
change ("may not check" — an equivalent mutant is possible). The control
run must pass in the copy first, or the check says it cannot judge. Never
changes the project or the verdict; bounded by `mutation_budget_s`; `[mutation]`
lines in the report, `mutation` journal events, every mutant in the log.

**The pinned requirements** (`constraints.py`, `SessionConfig.pin_constraints`,
on). The request's MUST / NEVER / DO NOT sentences are taken out ONCE, by
rule, word for word (never paraphrased; code blocks skipped; at most 15,
the rest counted), and pinned in the cached prefix of every prompt for the
session, after the conventions (`PromptBuilder.constraints`) — so the sixth
file's repair still sees "do not add dependencies". Logged as HARD
REQUIREMENTS; `constraints` journal event (the golden trace gains it,
read and re-written).

**Fixed before it shipped — an independent review of Debug & Repair found
ten defects, five serious, each now a test:** the locate step listed
`**/*`, which misses every top-level file (`main.py`) and filled its cap
with the project's `.venv` (now `*`, with venv / node_modules / build
output left out BEFORE the cap; the mutation check's copy had the same
bug); a run that was already clean BEFORE the repair was reported "fixed"
afterwards (now a confirming run counts only when a run here showed the
problem first; otherwise "repaired — running it here cannot confirm");
a command that could not start (`main.py` alone, `npm` on Windows, `cd … &&`)
was treated as the program's error and files were rewritten for it (now
`command-failed`, with the reason, and shell features are refused before
anything runs); an assertion failing inside a TEST file made the test the
target and the model bent it (now a failing test points at the module it
tests — by name and by its imports — and is never rewritten unless named);
a program waiting for keyboard input, or still running at the timeout with
nothing built to keep running, was judged fixed (now "needs input" /
"may hang" — unconfirmed; a game or GUI anywhere in the project still counts
as running cleanly); a question or a stop in a later round lost the record
of earlier rounds (one result throughout; diffs computed on every way out);
`Session.debug` raised on a refused model or an exhausted budget (now a
result, journaled and recorded); locating used the build's language box
(now every parser); a failed repair left the model's broken attempt in the
operator's file (debug tasks are atomic — put back as it was); "the same
error" compared line numbers and object addresses. Windows' own argument
splitting (CommandLineToArgvW) for the command.

Tests: `test_debugrepair` (16), `test_mutation_constraints` (10), the
golden trace re-written for the new `constraints` event. Whole suite on the
Linux VM: 1241 pass, 20 skipped, the same 3 baseline failures.

### Added — every diffusion decision in BUILD_LOG.txt (2026-10-10, evening)

Bill: comprehensive logging *"really should be a rule for anything during
development"*. Diffusion mode reached BUILD_LOG.txt only as the finished
file, under a `generate` line naming a temperature it never sampled at;
what it decided went to the JSONL as numbers, or nowhere. Now the readable
log says, per hole:

- **which lines and why** — `diffusion src/calc.py — attempt 1: rewrite
  lines 3-4 (`add`) as a hole — kept canvas 1 of 2`, then `why this hole:
  its error is at line 3` (or: the model was least sure of this body; or:
  a pinned stub, signature held);
- **every canvas** — style, noise, seed, steps, prefill (and how many prompt
  tokens came from the server's cache), decode, and *usable: 85% sure on
  average, least sure line 61%* or *REJECTED:* with the exact reason;
- **the lines cut out**, **the hole as kept** with a confidence column per
  line (`?` under 50 %, `~` under 85 %, `--` not read back — searchable,
  since a text file cannot be coloured), and each rejected canvas as it came
  back;
- **every time diffusion mode steps aside** — no line of the file named,
  every doubted body tried, the hole or its prompt does not fit, the same
  hole failing the same way twice, most bodies unusable — as `no hole this
  attempt — <why>; the whole file is written instead` (also a `canvas`
  journal event, `kind: "skip"`);
- **the implementation notes** written before a pinned file's bodies, in
  full, or that the planning call failed;
- the `generate` line of an attempt made by canvases says `via diffusion
  canvases`, and the session's config line says `diffusion=on (2
  canvas(es) per repair, notes on)` / `off` / `unavailable (the model fills
  no canvas)`.

`SessionLog.diffusion` (new), `SessionLog.generation(via=)`,
`canvas.Candidate.describe`, `canvas.doubt_rows`, `canvas.SURE/UNSURE`
(ATK's doubt-map bands).

An independent review of the change found four things, each now fixed and
a test: a doubted body that did not fit a canvas was logged as "every
doubted body has been tried" (it names the body now — the next attempt
tries the next one); the config line's candidate count could stop a
session from starting on a bad value and claimed more canvases than the
schedule asks (clamped, and never raises — nor does the loop on a `None`
count); a stub body that could not be found again was skipped without a
word (it is named, and counted as not written); a usable canvas that was
not chosen was headed "(rejected)".

Tests: 9 more in `test_canvas` (40). Whole suite on the Linux VM: 1215
pass, 20 skipped, the same 3 baseline failures.

### Added — diffusion mode: holes instead of whole files (2026-10-10)

Bill: *"most of the problems it runs into could be fixed with a
bidirectional model"* — and *"go ahead and build those"*. When the loaded
model fills a CANVAS (DiffusionGemma behind a host that says so), a repair
rewrites only the hole its error is in, and a pinned file is written body
by body. Design and limits: `docs/DIFFUSION_MODE.md`.

- **`canvas.py` (new)** decides, by rule: WHERE a hole goes (the body of
  the function the located error is in; a window when the body is long or
  the file does not parse; nothing when no error names a line here), what
  the canvas holds (head, tail, the old lines for a warm start, the noise),
  and how a fill comes back (padding and a run-on cut, the file outside the
  region byte for byte unchanged, the old lines and a fill that does not
  parse refused). Pure, so every rule is a test.
- **The loop** asks for a hole before it asks for a file
  (`Loop._canvas_generate`): `diffusion_candidates` canvases per repair
  attempt (default 2: a warm start with the far side held, and an open
  rewrite from noise), the usable one the model was surest of goes to the
  tests. No usable hole → that attempt is the whole-file repair, said in a
  warning. A pinned file: one notes call (`diffusion_notes`), then each
  `raise NotImplementedError` body as an open hole, signatures held; a body
  that will not come back is left as the stub, so its test fails AT that
  line and the next attempt fills it.
- **Doubt-guided repair**: a failure that names no line of the file (an
  assertion in its test) is repaired in the body the model was least sure
  of when it wrote it (`Loop.doubt`), then the next.
- **The same hole twice is not tried a third time**: the same function (or
  error line) failing with the same error after its hole was rewritten
  says the fault is elsewhere, and that attempt rewrites the whole file.
- **The contract**: `ModelCapabilities.supports_canvas` (defaulted),
  `CanvasRequest` / `CanvasResult` (types.py), the optional
  `ports.CanvasLLM.canvas_fill` — never raises, like `complete`. Journal
  event `canvas` (every candidate, which was kept); host events carry the
  filled lines with their per-line confidence and flips, for a doubt map.
  All minor-version additions; a host without the capability sees nothing.
- `SessionConfig.diffusion` ("auto" / "off"), `diffusion_candidates`,
  `diffusion_notes`; `LoopConfig.canvas`, `canvas_candidates`,
  `canvas_notes`. The session says once, at the model check, whether
  diffusion mode is on.
- `models.py` recognises DiffusionGemma by name ("diffusiongemma"), and the
  model check does not warn "unknown model" about a model that fills a
  canvas whatever its file is called (Bill's stock file is
  `Dg_Rc0P1_Patched`).

**Fixed before it shipped — an independent review of the change found
eleven defects, five of them silent** (the spliced file parsed, so nothing
downstream would have caught them), each now a test:
- the run-on cut compared stripped text, so an inner `}` or `return None`
  truncated a hole — it now needs the next lines WITH their indentation, in
  order, and a cut the parser accepts; outside Python there is no run-on
  cut at all and every candidate holds the far side;
- a `# comment` at column 0 inside a body ended the body (the function then
  returned None) — comments and lines inside strings are never cut points;
  and no line, whitespace or blank line inside a fill is changed any more;
- lines were split with `str.splitlines`, which counts form feeds and
  Unicode separators the parser does not — a form feed before a `def` made
  the splice delete it; lines now end where the parser says;
- a one-line `def f(): return x` made a hole of its signature; it gets a
  window, and a one-line stub is not a body hole;
- a model that wrote only the FIRST lines of the held tail kept them twice;
- lines past the first canvas (never read out) counted as certain;
- a hole was counted as "tried" before any candidate came back, so a
  server hiccup ended diffusion mode for that error; a located error whose
  hole did not fit was sent to the doubt rule; the context check was one
  canvas looser than the server; the notes call ran before the size check.

Also proved against the REAL patched server (a CPU build with a tiny
random-weight model): a repair asks three canvases, the 2nd and 3rd reuse
1,096 of 1,163 prompt tokens from the server's cache, every candidate is
judged, and the attempt falls back to the whole file as designed.

Tests: `test_canvas` (31). Whole suite on the Linux VM: 1206 pass, 20
skipped, 3 fail exactly as before this change (Python 3.10 there: the
doctor's 3.11 floor ×2, the SQLite-threading test).


### Fixed — the skeleton pins the interfaces; tests sit beside their modules; a repair answers for its result (2026-10-03)

The racing spec on Qwen3-Coder-30B-A3B (Q4_K_M, 32k), Oct 2, 23:23: 998 s,
three files failed, two built with no test. Its generations were replayed
through a scripted session to see the prompts each file was given; three
of the defects it showed were the harness's, not the model's.

- **The skeleton carries the interfaces (§4.2 step 2).** Every stub was a
  docstring and `def main(): raise NotImplementedError`, so each file was
  written blind to the others: `track.py` made `TrackSegment =
  Tuple[float, float, int]` and `render.py` read `seg.x` from it —
  `'tuple' object has no attribute 'x'`, main.py dead on its first frame.
  Root cause, in two parts. (1) The Phase 0 "interfaces" fix is the
  codemap's `# INTERFACES YOU MAY CALL` block, and it did fire for
  render.py (with math3d, physics and track as built siblings) — but the
  codemap indexed functions and classes only, so `TrackSegment` did not
  exist for it: render.py was shown `build_track() -> List[TrackSegment]`
  and nothing of what a TrackSegment is. (2) The skeleton itself never
  carried signatures: `Planner.stub_for` writes the rule stub, and the
  richer skeleton was deferred "to its own pass" (`_role_order`'s
  docstring). For the first three files there was nothing to show at all.
  Now:
  * `interfaces.py` (new) + `Planner._pin_interfaces`: ONE model call
    (persona `architect`, contract `CONTRACT_FILES`) writes a stub for
    every planned Python module — shared data types defined once and
    imported by name, every public signature. The reply is reduced BY RULE
    (`interfaces.sanitise`): bodies become `raise NotImplementedError`
    (an `__init__` keeps its `self.x = …` lines), anything that would run
    at import is dropped, an entry point keeps `if __name__ ==
    "__main__": raise SystemExit(main())`, and the result must parse. A
    pinned stub carries `interfaces.PINNED_TEXT` (still the `cc-stub:`
    sentinel). Asked only for two or more Python modules, or a module
    whose test is written first; a file whose block is missing or broken
    keeps the rule stub, and the session says which and why. The skeleton
    check now also catches `from src.track import Lane` when the track
    stub has no `Lane`. Order is then learned from the stubs' imports.
  * The loop shows each file `[THE INTERFACE PINNED FOR THIS FILE]` —
    keep every name, signature, field, alias and constant — and holds the
    body to it (D12): a body that no longer defines a pinned name fails
    with a located `pinned-interface` diagnostic ("src/track.py does not
    define `Segment`…") and is sent back with the interface in view,
    instead of passing its own check and breaking the files built on it.
  * The codemap indexes module-level **type aliases** and **constants**
    (`alias`/`constant` symbols whose signature is the source statement,
    its comment as the docstring), a class's constants and an Enum's
    **members**; signatures keep their **defaults** (`segment_count: int =
    300` used to read as required); a field's default is shown as written
    (`field(default_factory=list)`); a ctor decorator keeps its arguments
    (`@dataclass(frozen=True)`). The interface block puts an alias before
    what uses it, and says of a `Tuple`/`dict`/`list` alias that it is a
    plain container with no named attributes. `enrich` answers `'tuple'
    object has no attribute 'x'` with the project's tuple aliases, exactly,
    and a `NameError` with an alias or constant's definition.
- **Tests are built beside their modules (F2).** PLAN order was math3d,
  physics, track, render, main, test_math3d, test_physics: a test depends
  only on its module, so the topological sort left every test last and
  four modules were "verified" against zero tests. `planner._place_tests`
  puts each test right AFTER its module — the module is checked by its own
  test, and repaired against it, before the next module is built on it.
  When the request asks for tests first ("The engine must implement tests
  before writing the module bodies", "test-first", "TDD") or
  `SessionConfig.test_first=True`, each test whose module's interface was
  pinned is written right BEFORE its module, against the pinned stub
  (`[THE MODULE UNDER TEST — …, interface only]`); failing against the
  stub is accepted as what such a test must do (its own faults — a syntax
  error, a name the stub lacks — are still repaired); a test that PASSES
  against the stub is said to test nothing; the module's first attempt is
  shown the test (`[THE TEST THIS FILE MUST PASS]`), and when the module
  verifies, so does the test. Test-first without a pinned interface falls
  back to test-after, as a plan caveat. The build spec's F2 makes
  test-first the design, but Phase 0 deliberately shows a tester the
  module it tests, so test-after stays the default for requests that do
  not ask; `test_first=True` makes test-first the default.
- **A cross-file repair is accepted only when its cause is gone.** At
  358.9 s the log said `REPAIRED src/render.py`; snapshot 0007-t4-fix is
  empty — the model had returned render.py byte for byte, and it passed
  its own check (an import, zero tests). `Loop.run_task` takes `baseline`
  (the file when the repair began) and `accept` (the task whose failure
  caused it). A reply identical after normalising (the AST, so a comment
  is no change) stops the repair at once: NOT REPAIRED, "the model
  returned the same code", nothing run. Once the repaired file's own check
  passes, the caller's runs too; while it still fails inside the repaired
  file, the attempt is not accepted and the caller's error drives the
  next one. That acceptance run is the re-check `reverify` reports, so a
  game is not run twice. A module repaired against its test gets the
  unchanged-file rule too (its test was always its acceptance).
- New: `SessionConfig.pin_interfaces` (default on), `SessionConfig.
  test_first` (default None: follow the request); `ccoder build
  --no-interfaces`, `--tests-first`, `--tests-after`; journal event
  `interfaces` (the call's prompt hash, model, tokens, what was pinned and
  what fell back) — a minor-version addition to `JOURNAL_EVENTS`; the
  BUILD_LOG shows the interface reply and the build order. **The interface
  call is one more model call**: a host suite that scripts replies for a
  plan of two or more Python modules needs one more reply after the plan
  (this repo's suites were updated), or `pin_interfaces=False`.
- Unchanged, deliberately: a test is still never rewritten to agree with
  the code.

Tests: `test_interfaces_pinned`, `test_test_order`,
`test_repair_acceptance` (new; each fails on the previous code), one more
in `test_repair_routing_oct2`; the scripted suites gained the skeleton
reply; the golden trace gained one event (`interfaces`). 1214 pass, 20
skipped, 3 fail on the Linux VM's Python 3.10 exactly as before this
change (doctor's Python 3.11 floor ×2, one SQLite-threading test).

### Fixed — extraction and repair routing (2026-10-02)

The racing spec again, on Devstral-Small-2-24B (Q4_K_M, 32k): 23 minutes,
three files built-not-verified, four failed — the same score as Qwen3 on
Oct 1. Two of the four failures were the engine's, not the model's.

- **A reply with an unmatched fence line no longer reaches the disk with
  it.** Devstral wrote both test files with no opening fence and then a
  closing one. `_FENCE` needs a pair, the whole reply failed to parse, and
  `extract_code` fell back to the whole reply: each test was written with
  a trailing "```" and could not be imported. `_unpaired_fence_candidates`
  now offers the reply with the odd fence line removed (before a bare
  closer, after a tagged opener), validated like every other candidate. An
  even number of fence lines is left alone, so a Markdown file that ends in
  a code block keeps its closing fence.
- **A test file that does not parse is blamed on the test.**
  `runner.syntax_check` recorded `file=exc.filename or src_path`, and
  `ast.parse` without a filename reports `"<unknown>"` — truthy, so the
  path was never used. The printed line was re-parsed as a gcc message
  (`tool='gcc'`, `file='<unknown>'`), and `_test_disagrees_with_code` could
  not see that the error was in the test: it reported "It is src/math3d.py
  that has to change". The parse is now given the path, the diagnostic
  object travels with the phase (`PhaseResult.diagnostics`), and the runner
  uses it instead of re-reading text. A syntax-phase diagnostic in the test
  file counts as the test's own fault.
- **A module is not rewritten for its test file's syntax error.**
  `src/physics.py` was regenerated twice — identical both times — because
  `tests/test_physics.py` did not parse. `_culprit_elsewhere` skipped test
  files entirely; it now names a test file as the culprit when the error
  is a parse error located in it (SyntaxError, IndentationError, TabError).
  "cannot import name X", raised in a test, still belongs to the module.
- **A failure that moves to a third file is followed.** main.py died in
  math3d.py; math3d.py was repaired; main.py then died in render.py, and
  `reverify` reported "it still fails" and stopped. It now names the new
  culprit, and the session queues that repair like the first
  (`Session._queue_repair`). Each file still gets one repair per session,
  which bounds the chain; a file blamed again after its repair is
  reported, with a warning that says so.

Tests: `test_extract_unpaired_fence` (built on the Oct 2 reply, verbatim),
`test_repair_routing_oct2` (new). 1151 pass, 20 skipped.

### Fixed — the harness tells the truth (Phase 0, 2026-10-01)

Three builds of one specification — a pseudo-3D racing game, seven files —
failed the same way on Aug 8 and again on Oct 1, on a clean folder. The log
of the Oct 1 run was read line by line against the code; every item below
is a defect that run exhibited, with the line that proved it.

- **Interfaces now carry constructors, fields and instance attributes.**
  `interface()` and the codemap's `# INTERFACES YOU MAY CALL` block
  filtered every name starting with `_` — which is `__init__`. A caller
  saw `class CarPhysics` and `def update(…)` and nothing of how the thing
  was made or what it held, so three builds running invented
  `CarState(speed=…)` against a constructor that takes nothing, and
  `ProjectedSegment(z=…, curve=…)` against a NamedTuple whose fields are
  `x, y, width`. The Python parser now emits `field` symbols (annotated
  class-level names: NamedTuple, dataclass, TypedDict) and `attribute`
  symbols (`self.x = …`, typed from the annotation, the parameter it
  copies, or the literal), and one renderer (`context.interface_lines`)
  serves both callers: members indented under their class, fields on one
  line, attributes on one line, a `# construct:` line for any class with
  no explicit constructor, dunders that define use (`__init__`,
  `__call__`, `__iter__`, …) kept, single-underscore names still private.
- **Planned dependencies reach the first attempt.** `Task.depends_on`
  holds task ids (`t3`); the codemap matched them against file paths, so
  the planned-dependency injection — the whole reason `planned=` exists —
  never fired on any build. The session now resolves ids to paths
  (`Session._planned_paths`) and a test task always gets the module it
  covers. Measured on the Oct 1 prompts: `main.py`, which depends on four
  modules, had 56 more tokens of context than `math3d.py`, which depends
  on none.
- **Zero collected tests is never "done".** A test file that the runner
  collects nothing from now FAILS, with a diagnostic the model can act on
  (`no-tests-collected`: name the methods `test_*` on a `TestCase`;
  return the test, not the module). It used to pass — once with a file
  that was not a test at all. `TaskOutcome.verified` is the new, stronger
  claim: the file's own tests ran, at least one, and passed; `ok` means
  only that nothing contradicted it. Log lines read `VERIFIED`, `BUILT
  (not verified)` or `FAILED`, never `DONE`.
- **A failure that does not name a file is not that file's failure.** A
  module whose own test does not exist yet was judged by the whole suite,
  and a stale `tests/test_physics.py` from an earlier session failed
  `math3d.py`, `physics.py` and `track.py` in turn — each regenerated
  correctly, each stopped by stagnation. `run_tests` now attributes
  (`_attribute_failures`): failures that name the module are its; failures
  that name other test modules are a caveat; an unattributed failure stays
  a failure. A test task is scoped to its own file. Only for languages
  whose tests are separate files — a Rust module's tests are inside it.
- **The session baselines the folder before building.** Tests already on
  disk run once first; the modules that fail are `LoopConfig.known_failing`
  and are never charged to the build. The operator is told, by name, in
  one sentence, and a folder whose `BUILD_LOG.txt` records earlier
  sessions is said to be one.
- **The right fence is written when the model sends several.** Aug 8:
  "Let me check math3d.py…" + an imagined class in a ```python fence, then
  "Here's the corrected file:" + the real test; the first fence was
  written as `tests/test_math3d.py`. Same-language fences are now ranked
  (`patcher._rank_fences`): a test path wants a fence with tests; "let me
  check the existing file" marks context, "here's the corrected file"
  marks the answer; longer beats shorter; document order last.
- **Repair prompts carry the facts the error named** (`enrich.py`, rung 1
  of the escalation ladder). `cannot import name 'X' from 'm'` → what `m`
  defines, with the nearest name; an unexpected keyword or a missing
  positional argument → the exact signature, and for a constructor the
  class's fields and attributes; `'C' object has no attribute 'a'` → the
  class and the nearest attribute; `NameError` → where the name lives and
  the import line; `No module named` → the project's modules. Same model,
  same temperature, different input — which is what a retry needs before
  it can produce anything but the same answer.
- **A model that is not for code is refused before planning.**
  `models.judge()` reads the loaded model's name: a known coding family
  builds; a roleplay or fiction merge (`-Tavern`, `-RP`, `Uncensored`, …)
  is refused in one sentence (`NotACodingModelError`;
  `SessionConfig.allow_any_model` turns it into a warning); an unknown
  general model builds under a warning. A context window under 16k is
  named as a problem. On Aug 8 a roleplay merge built the racing spec for
  nineteen minutes because it happened to be loaded.
- Python 3.12+ `unittest` exits 5 and prints `NO TESTS RAN` where 3.11
  exited 0; the test that pins the discovery fact now states both.
- Journal: two new events, `model_check` and `baseline`; the golden trace
  gained the one line.
- **Host event `plan`** (added to `EVENT_KINDS`; a minor-version addition):
  the plan as data — one dict per task with `id`, `path`, `purpose`,
  `test_path`, `persona`, `depends_on`, `status`, `attempts` — emitted
  after planning and again when a re-plan changes the set of files. The
  console line `plan: 7 file(s) proposed` was all a host had; a host that
  draws a task board needs the rows. Hosts that ignore unknown kinds are
  unaffected.

### Fixed — what the first Phase 0 build showed (2026-10-02)

The racing spec again, on the Phase 0 engine with Qwen3-Coder-30B at 32k:
28 minutes, three files built-not-verified, four failed. Read line by
line, as before. Every item is a defect that run exhibited.

- **A module is shown what already exists.** `render.py` was written
  with no interface of `physics`, `track` or `math3d` in front of it —
  the stubs import nothing, so the plan had no edge to follow and the
  Phase 0 interface block had nothing to attach to — and invented
  `player_state.x` and `segment.width` against a class with neither.
  `Session._planned_paths` now appends every module already built in
  the project, after the real dependencies: not tests, not stubs, not
  the entry point, at most ten (`SIBLINGS_SHOWN`); the interface block's
  budget serves real dependencies first. Visibility, not a dependency:
  no import is written and no order changes.
- **A failure is repaired where it is raised.** `main.py` ran and died
  at `render.py:61`; the loop asked for main.py again, got the identical
  file (correctly), and gave up with "the task is probably too large".
  When the first error's innermost project frame is another file this
  plan owns (`Loop._culprit_elsewhere`), the task stops at once and
  names it; the session repairs that file against the failure, with the
  caller shown as a caller (`[THE CALLER THAT FAILS INSIDE THIS FILE]`)
  and the real definitions under the error; then the caller runs again.
  Once per file per session. A culprit outside the plan is named, not
  touched.
- **pytest tests are run by pytest when it is there.** The model wrote
  `tests/test_physics.py` for pytest; the host installed pytest on
  request; the engine ran `unittest discover`, collected zero tests, and
  got the same file back. `runner.python_test_runner` picks pytest when
  the tests in play are pytest-style (`import pytest`, bare `test_*`
  functions, classes without TestCase) AND pytest is importable in the
  environment that runs them — probed through the host's ExecPort, so a
  per-project venv answers for itself; `--tb=native` keeps the
  diagnostics parser's tracebacks, `-rfE` keeps failure attribution's
  `FAILED path::test` lines. Without pytest, the zero-tests message says
  the file is written for pytest and what to write instead. And the
  test author is told the runner's rules before it writes
  (`Lang.test_note`, Python set).
- **A test written after its module sees the module.** The tester
  asserted `x = cx + x/z·(w/2)` to seven places against an FOV-scaled
  projection; it had the signature, not the formula. A test task whose
  covered module has a real body gets `[THE MODULE UNDER TEST]` with
  the rule: assert the requested behaviour and the docstrings' promises;
  take exact numbers from how the module computes them, never from a
  formula of your own; if the module contradicts the request, assert
  the request and let it fail.
- **Packages the request names install before the first file.**
  "using pygame" was the spec's first line; the engine learned it at
  minute 13, at an import, and the host's install question waited
  thirteen more for a person. `packages.py` (new): the well-known PyPI
  names, `packages_in(request)`, `is_known`, `mentioned_in`.
  `Session.start` hands what the request names to the host's optional
  `ExecPort.ensure_packages`; a host without one is unaffected, and a
  host may use `is_known`/`mentioned_in` to install those without
  asking. (ATK does, by default, into the project's own venv; an
  unknown name — a model's typo — still asks.)
- `planner.is_entry_point` is public (it was a closure in
  `derive_order`); `tests/test_repair_routing.py`,
  `tests/test_pytest_runner.py`, `tests/test_packages.py` are new.

### Changed — the prompt cache: two design decisions, measured

A local model re-reads every prompt token its prefix cache cannot reuse,
and the cache keeps only the longest run matching the previous prompt.
Measured on a scripted build that extends a fifteen-module project, the
reusable share of each prompt went from 37% to 69% — 46% less prompt
processing, identical results.

- **The persona goes last in the cached prefix**, after the conventions
  and the architecture. It came first and it is the part that changes
  most (planner, engineer, tester, repairer on every repair), so each
  switch threw away the whole prompt: about 1% survived, measured.
- **The architecture snapshot is no longer rebuilt after every file.**
  `maybe_bump_epoch(target=<the file just written>)` always fired. What
  the rebuild bought — the first attempt at a file seeing what was just
  written — now comes from the tail: the staleness note carries each
  changed file's CURRENT line in the summary's format. Rebuilds happen at
  G.7.2's threshold (five changed files) or when the plan gains or loses
  files. M31's forced every-write rebuild for tool-less models is met the
  same way, since snapshot plus note never lags.
- The review stage's prompt carries the architecture too, so it starts
  from the build's cache instead of from nothing.
- `Completion.prompt_processed` (new, defaulted): tokens the server really
  processed (`timings.prompt_n`, or OpenAI's `cached_tokens`). The session
  report now says "N% of prompt tokens came from the cache" when the
  server reports it. `cache_health()` counts planned full reads — it used
  to call a healthy run with a planned rebuild broken — and, without the
  server's count, reads time per prompt token, so a cache broken on most
  calls can no longer hide in the median.
- `tests/test_prefix_stability.py` measures the reuse directly: each of
  the new tests fails on the previous design.

### Added — `ccoder audit`: improve what already exists

`ccoder audit FOLDER` reviews an existing project — one this engine built
or one written by hand — and reports what is wrong and what could be
better, ranked, with file and line; it writes a plan `ccoder build --spec`
can carry out, and the next audit says what was fixed, what remains and
what is new. Tools first (the review scanners, undefined names,
possibly-unused code where the call graph and a text search agree,
untested modules, does it compile, do its tests run), then one bounded
model read per file with a byte-identical cached prefix. It never runs the
program, never writes a source file, never asks for approval.
`--focus`, `--no-model`, `--no-tests`, `--only`, `--max-files`,
`--max-model-files`. See `docs/AUDIT.md`. Public API: `audit_project`,
`AuditConfig`, `AuditReport`, `AuditFinding`.

### Fixed — the September review pass

A five-part review reproduced about eighty defects; every fix below has a
test that fails on the previous code. The ones that change what a build
does come first.

**Writes that went around approval, snapshots and undo**
- `runner.autofix`, `format_code` and `lint_code` wrote the model's
  candidate to `<root>/<stem>.py` before the guard and before approval —
  a refused build still left files behind, and a task named `main.py`
  overwrote the operator's `main.py`. They now work on a scratch copy.
- The skeleton wrote stubs with `fs.write`, outside any transaction: a
  plan naming an existing file replaced it with a stub, with no snapshot
  and no undo, and under `DenyAll` stubs were written anyway. The
  skeleton is now one approved, snapshotted transaction, and an existing
  file with real work in it is kept as found.
- `undo_to` read a pruned snapshot as "created by this transaction" and
  deleted the file. It now reads the manifest, deletes only files it
  created, and refuses with a sentence when old bytes are gone.
- **The first attempt on an existing file never showed it.** "Write the
  complete contents of `x`" regenerated a working file blind. The model is
  now shown the file and asked for a change; a file too large to show
  whole is refused before any model call.

**"Verified" now means verified**
- A module and its test are actually paired; a test that collected zero
  tests is not sealed as verified; a test that fails because the MODULE
  is wrong is never "repaired" into agreement — the module is repaired
  against the test.
- The test phase is scoped to the task's own test file, so one failing
  test elsewhere no longer blocks every later file.
- Rust tests were compiled and never run; JavaScript could never verify
  on Node 21+; rustc's "aborting due to 1 previous error" was the one line
  the model saw. All three fixed, checked against the real toolchains.

**The loop**
- Truncation detection used the tokenizer for Python and handles escapes
  and comments elsewhere; continuation seams handle a re-emitted partial
  line, a re-opened fence and a restart from the top.
- The commentary stripper removes whole lines only — it had turned
  `## Why this exists: …` in a shell script into a command.
- Stagnation is keyed on what changed, not on line numbers; a guard-blocked
  attempt counts; the cosmetic-churn rule works.
- Empty and tool-exhausted attempts are journaled; repairs keep the
  request; `prompt_sha256` hashes the messages actually sent.
- **The prompt is measured against the model's context.** Nothing did,
  and llama.cpp's context shift drops the system prompt first. Reference
  material is cut to fit and named; the file being repaired never is.
- The first file now sees the skeleton in its cached prefix; a context
  resize is an epoch boundary, as a model swap already was.

**Offline, and the environment**
- `openai_compatible` pointed at a remote URL skipped the gate, redaction
  and budget once enabled. It now goes through all three.
- Local endpoints bypass the system proxy; URLs are classified without a
  DNS lookup.
- `SubprocessExec` treated `env=None` as "inherit everything", so the
  review stage's scanners ran with API keys and proxy variables. It now
  scrubs, as the README always said.
- Output capture is bounded (a printing loop no longer exhausts memory
  before its timeout), never raises on bad bytes, and keeps the tail.
- Redaction judges only the secret itself, keeps one numbering per
  session, and knows 18 shapes (was 15).

**Measurement and provenance**
- `complete()` streams internally, so `prompt_ms` is prefill and
  `decode_ms` is decode — `cache_health()` can finally give a verdict.
- The architecture prefix is snapshotted per epoch rather than rendered
  live, so it is byte-identical within an epoch (M52).
- Journal writes append instead of rewriting the file; numeric zeros
  survive; unknown event names warn.

**Review and codemap**
- The review's model pass reads the last object carrying the answer, not
  the schema the model restated; unknown severities fail closed; a
  scanner that timed out is reported, not counted as clean; semgrep runs
  only with local rules.
- The codemap binds `self.x` to its own class, gives modules a symbol, no
  longer cries wolf on aliases, `from pathlib import Path` or other
  languages, and works from the worker thread a host runs the build on —
  **in ATK every codemap call from the build had been failing**.
- `read_slice` is capped and refuses `.env` and skipped directories.

**ATK adapter, CLI, installers, CI**
- The panel has an approval dialog (every diff was refused), builds no
  dialog on the worker thread, and `ATKExec` uses the core's capture;
  `timeout=0` waits; storage is per project.
- `ccoder history` persists (`JsonFileStorage`); `--remote` exists; bad
  flags and URLs are sentences; `resume` never tracebacks.
- Installers put Python in the clone as documented; `push.bat` retires
  its commit message after use.
- CI lint covers `adapters/` and is green; the coverage floor is 85 again
  (measured 86%); the tree-sitter extra lists grammars that load.

**Numbers:** 442 → 1,094 tests; branch coverage 76% → 86%.

**Open design questions, deliberately not changed:** the epoch still
bumps after every task (`maybe_bump_epoch(target=task.path)`), which
rebuilds the cached prefix per file; and the persona block sits first in
the prefix, so a switch to the repairer persona discards the cache.
Both are the specification's current rules.

### Added — deployed skills: guidance that travels with the code (F3)

`.ccoder/skills/*.md` now loads into every session's cached prompt prefix
as project conventions — the seat `SessionConfig.conventions` always
reserved and nothing on disk fed. Borrowed from deepseek-cowork's deployed
skill templates, and worth more here than there: a 7B-24B local model
needs the scaffolding, and files beside the code are versioned, diffed
and offline.

- `cognitive_coder/skills.py` — zero-dep discovery and a hand-parsed
  `---` header (`name`, `description`, `lang`). Sorted order is priority
  order; loaded once at Session construction (M52 — the prefix must not
  change mid-session); an oversized skill is **skipped by name, never
  truncated**; `lang:` scoping filters before the budget is spent.
- `ccoder skills list | deploy | new`, and `--no-skills` on `build`.
  Deploy seeds three editable starters and never overwrites.
- The journal gains a `skills` event: each active skill's path, name and
  content hash at session start, plus what was skipped and why (C8).
- Public API additions: `Skill`, `SkillLoad`, `load_skills`, `SKILLS_DIR`,
  `STARTER_SKILLS`; `SessionConfig.use_skills` / `.skills_dir`;
  `"skills"` in `JOURNAL_EVENTS`. All additive.


### Fixed — `prompt_ms` was measuring the wrong thing

`Completion` and `JournalEvent` gain `decode_ms`, and `prompt_ms` now means
what M55 says it means: **prompt processing only**.

A host adapter had been putting whole-call wall-time in `prompt_ms`. Across a
real eleven-generation session the figure tracked `tokens_out` almost
perfectly and `tokens_in` not at all — 423 tokens out took 36 s, 1,209 took
114 s, while the prompt stayed near 1,500 tokens throughout. It was measuring
decode. G.7.5's prefix-cache check then read that as *"prompt processing is
steady … the prefix cache looks healthy"*, which was a confident statement
about a number that did not mean what its name said.

The two are separable for nothing when the provider streams: everything before
the first token is prefill, everything after is decode.

- `cache_health()` now **withholds the verdict** when `decode_ms` is absent,
  saying the provider does not separate the two and nothing can be concluded,
  rather than drawing a conclusion from a number it cannot interpret. A wrong
  diagnosis is worse than a missing one, because it stops anyone looking.
- `stats()` gains `tokens_per_s_median`, so decode speed is a recorded figure
  rather than an impression. Milliseconds of decode mean nothing without the
  token count beside them — 120 s is fast for 1,200 tokens and catastrophic
  for 40.
- Both fields default to 0, so a provider that cannot separate them keeps
  working and simply gets the honest verdict.

### Added — the plan size limit is reachable

`SessionConfig.max_files` and `ccoder build --max-files N`, defaulting to the
previous constant of 12.

A cap is right: a model asked for "a web framework" will propose sixty files
and finish none of them. But 12 was a module-level constant no host could
reach, it suits a request typed in one sentence, and it is arbitrary for a
four-section design document — which is precisely the case `--spec` exists to
serve, so the two features had to arrive together or the second would be
capped by the first. Truncation now emits a warning as well as a caveat, and
says how many files were dropped.

### Added — a build request can be a file

`cc build --spec plan.md` reads the request from a `.md` or `.txt` file, and
`--preview` plans and prints what would be built without generating anything.

The CLI had always described the request as "what you want built, in a
sentence", and that framing was wrong for the work this engine is good at. A
sentence types fast and plans badly. Everything that decides whether a build
goes well — which modules exist, what each owns, what must not import what,
which tests must be written — is thinking done before the model is asked for
anything, and it does not fit in a shell argument or a one-line text box. The
specification that exposed the four fixes below was sixty lines in four
numbered sections, and it was pasted into a field showing one line at a time.

- `spec.py` — reads the file, strips YAML front matter, takes the document's
  own first heading as its title, and reports what can be seen in it: the
  source paths it names, the test files it requires, and its size in
  approximate tokens. It does **not** interpret the specification; the text
  goes through verbatim, because the planner and the persona prompts are
  where meaning is extracted and a second answer to that question would only
  disagree with the first. Path detection is deliberately strict — `Pseudo-3D`,
  `16-bit` and `version 1.1` are prose, and a preview that lists imaginary
  files reads as though the engine understood something it did not.

- `Session.preview()` — plans and stops. Returns the build order, the context
  cost, and `tests_required` beside `tests_planned`: the two numbers whose
  disagreement went unnoticed for an entire build. Planning costs one small
  completion; a build costs twenty minutes of a local model's time, and both
  questions worth asking are answerable in between. It writes nothing, and a
  test asserts that — a preview that scaffolds a project is a build with a
  misleading name.

- A file that cannot be used says why in a sentence rather than a traceback
  (C6). This runs at the very start of a long operation, where a stack trace
  for a mistyped filename costs the whole run. A specification larger than
  most local models' context is flagged and never silently truncated: the
  operator decides.

### Fixed — four faults found by the first real workload

The same pseudo-3D racing specification was built twice on 2026-08-07, with
Devstral-24B and with Ministral-24B, and both projects and console logs were
kept. Reading the logs and then *running the generated code* turned up four
separate faults. All four are the same shape: the engine was reporting
honestly and the report was still misleading, because the thing being reported
was three steps downstream of the thing that was wrong.

Regression tests for all four are in `tests/test_planning_regressions.py`,
written against the actual plans and filenames from those two runs.

- **The build order was never derived.** `Planner.derive_order` reads imports
  off disk, and ran only after `skeleton()` — but `stub_for` writes a stub's
  imports from `depends_on`, which `derive_order` is what populates. Empty in,
  empty out: it returned the model's proposed order untouched, on every
  project, since it was written. It failed silently for the best possible
  reason, in that a function returning *an* order looks like it worked.

  The cost was visible in the two runs. One model proposed
  `math3d → physics → track → render → main` and produced three usable files;
  the other proposed `main → math3d → …`, spent three attempts failing to
  import a class from a module not yet written, and gave up. The difference
  was luck, and this function was the mechanism meant to remove it.

  Ordering now runs *before* the skeleton, seeded by the one fact available
  before any code exists — an entry point is imported by nothing — and is
  re-derived from real imports after every completed file. Test files are
  ordered after the module they cover. A plan that was already in dependency
  order is left untouched.

- **Test files named in the request were dropped from the plan.** The
  specification had a section headed "Testing Requirements (Strict)" naming
  `tests/test_math3d.py` and `tests/test_physics.py`. Both models proposed
  five files, all under `src/`, no tests, and nothing checked. Every build
  step then reported "the test command succeeded but ran ZERO tests" —
  truthfully, about ten times, describing a symptom whose cause was in the
  plan. With no tests, verification degraded to "the file imports", which is
  how a physics module was committed green with an `update()` signature no
  caller satisfied; the program died on its first frame.

  Explicitly named test paths are now extracted from the request and added if
  the plan omits them, as a visible caveat and a warning rather than a silent
  correction. Only explicit paths: "please write tests" is a wish and is not
  interpreted, and `src/latest_data.py` is not mistaken for a test file.

- **`unresolved_in` cried wolf on almost every generated file.** It treated
  the head of every dotted call as a name the project should define, so
  `screen.fill` — where `screen` came from `pygame.display.set_mode()` two
  lines above — was reported as undefined. The damage was concealment rather
  than noise: those appeared in the same sentence, in the same format, as
  `CarState`, `generate_track`, `render_road` and `TrackSegment`, every one of
  which was genuinely missing and became an `ImportError` minutes later.

  New `parse_python.bound_names` collects names bound by executable statements
  — assignments, parameters, loop and comprehension targets, walrus, `with`
  and `except` bindings — and dotted names whose head is one are no longer
  reported. Imports are deliberately excluded from that set: they do bind
  names, but counting them here silenced `unresolved_in` about a symbol
  imported from a module that does not exist, which is among the most
  valuable things it catches. That regression was caught by
  `test_codemap.py::test_a_name_that_exists_nowhere_is_reported` within a
  minute of the change.

- **The Recommendation Document contradicted itself.** The reviewer is handed
  the files that were *committed*, so when a build collapsed the failed files
  were simply absent from what it read — the worse the run went, the less
  there was to criticise. A program that crashed on startup was summarised as
  "Nothing was found that should stop this being used", four lines above a
  Verification line reading "3 of 5 file(s) built".

  `recommendation_document` takes `unfinished` and leads with it, states that
  the review covers only the files that built, and says plainly that the
  absence of findings about a missing file is not evidence it is fine. The
  all-clear sentence is now something only a completed build can earn.

### Added

- `.gitignore` — coverage data, `__pycache__`, tool caches and screenshots
  were showing as tracked changes.

### Still empirical rather than structural

The journal records enough (context size, timings, first-attempt success rate)
to answer G.9's tuning question with a query instead of an argument, and that
answer needs a few real sessions on the target machine before anything is
changed on the strength of it. The two runs above are the first two.

One thing those runs exposed and this release does **not** fix: a module that
imports a symbol from a sibling is still ordered by role, not by that import,
because before it is written the fact exists nowhere — the stub imports
nothing and the purpose line is prose. Matching purpose text against sibling
module names was tried and rejected: it made a module described as "pure
projection logic for track segments" depend on `track`, the exact reverse of
the specification's requirement that it depend on nothing, and a heuristic
that inverts a stated architectural constraint is worse than none. The repair
is a skeleton whose stubs declare the symbols their module exports, so an
importer can be checked before anything is generated. That is its own change.

## [0.9.0] — 2026-08-07

Phases 6 through 9. The engine is feature-complete against the specification;
every one of Appendix H's 55 numbered obligations is met, and 48 of them have
a test that asserts it. See `docs/CONFORMANCE.md`.

### Review (phase 7)

- `review.py` — deterministic checks that need nothing installed (hardcoded
  credentials with the value **masked** in the report, `eval`/`exec` on
  non-literals, path traversal, empty exception handlers, TODOs left in
  "finished" code, over-long and deeply-nested functions, quadratic string
  building, list membership inside a loop, public functions absent from the
  tests), then `bandit`/`semgrep`/`cppcheck`/`gosec`/`shellcheck` where they
  exist and a named cost where they do not, then **one** structured model
  pass covering security and performance together.
- The Recommendation Document: executive summary, quality assessment,
  vulnerabilities and fixes, and a deployment guide pitched at the reader's
  stated skill level.
- **The non-independence line.** Where the two perspectives came from one
  model, the document says so in the first quarter of the page, because a
  caveat below the findings is a caveat read after the reader has decided
  what to believe.
- The review runs *after* everything builds and its tests pass, and refuses
  to run at all when nothing verified.

### Redaction, remote providers and budgets (phase 8)

- `redact.py` — fifteen secret shapes, the same secret always getting the
  same placeholder, obvious placeholders left alone, and the host's own
  patterns honoured. Counts DISTINCT secrets, because one key appearing four
  times is one thing to revoke.
- **Every outbound message is scrubbed, including tool results and tool-call
  arguments** — the clause that is easiest to skip and where the file
  contents actually live.
- Anthropic, Google, Mistral, OpenRouter and OpenAI, all on stdlib `urllib`,
  all behind the gate that was built two phases before they were.
- Budgets that **halt**, checked before the call rather than after, and
  reporting what was achieved when they stop.

### The ATK adapter (phase 9)

- Six Port implementations bound to ATK, deliberately Qt-free so they are
  testable without a QApplication.
- A workspace panel with detachable panes, a persistent REMOTE banner, and
  screenshot attachment — Devstral is multimodal, and a picture of a broken
  dialog beside the code that renders it catches what reading source cannot.
- `atk_compat.py`, and the reason it exists: **the migration is not a
  rename.** The engine takes Ports where ATK's modules took paths, so
  `Lang.available()`, `diagnostics.feedback()`, `Diagnostic.source`,
  `coderun.build_and_run()` and `patcher.apply()` all changed shape. A plain
  re-export shim would have passed a smoke test and then failed at runtime in
  whatever code path ran first.
- `migrate.py` — dry-run by default, backs up every file it replaces, and
  refuses to touch anything outside the six modules.
- `test_migration.py` — 56 checks on ATK's OLD call surface, in ATK's own
  test style, run against a throwaway copy. **The live checkout has not been
  modified**; that is a decision for the owner to make with ATK's suite green
  either side.

## [0.5.0] — 2026-08-06

The minimum viable engine: phases 0–5 of the build specification. Everything
below is implemented and tested; `ccoder doctor` reports which phases are
built rather than leaving anyone to find out.

### The contract (phase 0)

- `ports.py` — six Protocols (`LLMPort`, `FileSystemPort`, `ExecPort`,
  `StoragePort`, `EventPort`, `ApprovalPort`), each with a Null
  implementation so the engine runs hostless, and each method documenting
  what a host may assume and what it must guarantee.
- `types.py` — the frozen dataclasses the Ports carry.
- `tests/port_conformance.py` — a reusable kit a host runs against its **own**
  implementations, covering atomicity, process-tree kill, the project-root
  jail, and capabilities honesty.
- `examples/tiny_host.py` — a whole session end to end with no model, no
  network and no host application.

### Installers (phase 0b)

- `install.bat` and `install.sh`: non-interactive, idempotent, nothing
  global, every optional component degrading with a stated cost, an honest
  summary, and a meaningful exit code.
- Python 3.11 is probed by **version**, not by name, and fetched into the
  clone when the machine has nothing usable — the normal path on an older
  machine, not an error.
- `ccoder doctor` prints the same summary on demand, including **which
  interpreter is in use and where it came from**.

### The deterministic layer (phase 1)

- 17 languages with argv-list commands, runnable scaffolds and test hooks.
  **GDScript is first class**: `--check-only` syntax checking, GUT and
  gdUnit4 detection, Godot's error formats parsed, `res://` translated at the
  boundary, and a headless caveat on any test touching the scene tree,
  physics or rendering.
- Compiler-output parsing for gcc/clang, MSVC, rustc, javac, Go, TypeScript,
  Python, Node, cppcheck, unittest, pytest and Godot — with the source
  quoted around each error. Unrecognised output yields one diagnostic, never
  an empty list.
- The static screen, stated in its own docstring as a screen against
  **accidents and not a security boundary**, now also blocking
  version-control commands in generated code.
- Attributable build/run/test phases, a scrubbed environment, and per-phase
  timeouts with a real process-tree kill.

### Edits (phase 2)

- Explicit transactions with **sequence numbers rather than timestamps**,
  sealed commits that a later rollback cannot touch, an `undo_to` that states
  in plain words how much verified work it would discard, and a queryable
  linear history.
- Encoding, BOM and line-ending preservation, with the snapshot storing
  original **bytes** so undo is byte-identical by construction.
- Context assembly under a measured budget that always ends by naming what it
  left out.
- An append-only JSONL journal recording provider, model, prompt hash,
  attempt, verification outcome, timestamp and `prompt_ms`.

### Providers (phase 3)

- `openai_compatible` covering llama.cpp server, Ollama, LM Studio, vLLM and
  LiteLLM, on stdlib `urllib` alone, with local-versus-remote decided by the
  address rather than by configuration.
- `local_llamacpp` for an in-process GGUF, with the import inside the
  constructor so its absence is a sentence rather than a crash.
- The remote gate — per-session, per-provider, approval-checked, banner-
  raising — **built before the remote providers it will govern**, so that
  adding one later cannot route around it.

### CodeMap (phase 4)

- SQLite symbol index with a call graph, and an `unresolved` table so a graph
  that could not bind everything says so instead of looking complete.
- Blast radius: transitive callers, the files to refactor, and the tests to
  run first.
- Semantic zoom split at the same seam as the prompt cache: stable
  architecture in the cached prefix, high-resolution interfaces in the
  volatile tail, and the staleness note in the tail where it cannot
  invalidate the cache it describes.
- Five agentic tools with JSON schemas, plus the text-marker fallback with
  its three-lookup cap.
- Per-project regression memory, kept small, inspectable and clearable.

### The loop (phase 5)

- Truncation **continued rather than regenerated**, detected structurally.
- Failed attempts kept out of the context; the diagnostics carry forward, the
  broken code does not.
- Deterministic pre-fixes before the model sees an error, each one logged.
- Stagnation and cycle detection over both code and diagnostics, with a cycle
  set that catches the 3- and 4-cycles no pairwise comparison finds — and a
  give-up message that **names the cycle** rather than counting attempts.
- Prompts layered on the model's own system prompt, with an output contract
  and a commentary detector at every consuming call site.
- Skeleton-first planning, dependency order derived from the skeleton's
  imports rather than asserted by the model, and replanning after each file.
- Session orchestration with resume derived from the journal on disk, so it
  survives a crash rather than merely a pause.

### Notes

- The Appendix E worked session is committed as
  `tests/fixtures/worked_session.jsonl` and shape-diffed in CI, so the
  specification's example is executed rather than admired.
- A file that mixes line-ending styles cannot round-trip byte-for-byte. It is
  **declared** rather than silently normalised, and undo is unaffected
  because the snapshot holds the original bytes.
- Remote providers are named as absent by `available_providers()` rather than
  offered and then failing.
