# Changelog

Kept from the first commit, per §10.4. Without PyPI, **the git tag is the
release** — so this file and the tags are the whole record.

The format follows [Keep a Changelog](https://keepachangelog.com/); versions
follow semver, where the public API is `cognitive_coder/__init__.py` only —
the Ports of `ports.py` and the shared types of `types.py`. Breaking either
is a major version.

## [Unreleased]

Nothing outstanding from the build specification. The next work is empirical
rather than structural: the journal now records enough (context size, timings,
first-attempt success rate) to answer G.9's tuning question with a query
instead of an argument, and that answer needs a few real sessions on the
target machine before anything is changed on the strength of it.

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
