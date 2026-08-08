# Changelog

Kept from the first commit, per §10.4. Without PyPI, **the git tag is the
release** — so this file and the tags are the whole record.

The format follows [Keep a Changelog](https://keepachangelog.com/); versions
follow semver, where the public API is `cognitive_coder/__init__.py` only —
the Ports of `ports.py` and the shared types of `types.py`. Breaking either
is a major version.

## [Unreleased]

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
