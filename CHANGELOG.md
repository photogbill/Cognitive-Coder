# Changelog

Kept from the first commit, per §10.4. Without PyPI, **the git tag is the
release** — so this file and the tags are the whole record.

The format follows [Keep a Changelog](https://keepachangelog.com/); versions
follow semver, where the public API is `cognitive_coder/__init__.py` only —
the Ports of `ports.py` and the shared types of `types.py`. Breaking either
is a major version.

## [Unreleased]

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
