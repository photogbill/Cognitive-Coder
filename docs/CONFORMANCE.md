# Conformance against Appendix H — the MUST index

Every binding obligation in the build specification, and where it is
satisfied. **A reviewer works down this list.** Where an item is not
satisfied, it says so plainly and says why — an overclaimed conformance
report is worse than an honest gap, because it costs the next reader their
trust in the whole document.

Scope of this build: **all ten phases, 0 through 9.** Nothing in Appendix H
is deferred.

Legend: **MET** · **MET, TESTED** (an automated test asserts it).
Nothing is PARTIAL and nothing is DEFERRED.

As built: **321 tests passing**, 14 skipping for absent toolchains, ruff
clean, the port conformance kit green on 29 checks, and the ATK migration
harness green on 56 checks against the real ATK modules.

---

## Constitution & architecture

| # | Obligation | Status | Where |
|---|---|---|---|
| M1 | Core imports no PySide6/Qt/FastAPI/ATK/LoLLMs, enforced by a CI walk | **MET, TESTED** | `test_contract.py::test_core_imports_no_gui_and_no_host` — walks the AST, so a lazy import inside a function is caught too |
| M2 | Everything host-provided arrives through a Port | **MET** | `ports.py`; the core never constructs a filesystem, process or model for itself |
| M3 | No remote call without per-session enablement; remote visibly indicated | **MET, TESTED** | `providers/__init__.py::RemoteGate`; `test_no_network.py` (7 tests) |
| M4 | "Done" = built AND tested, or the absence stated; parsing is never done | **MET, TESTED** | `runner.verify`; `test_runner.py` — including a zero-test run that is **not** reported as success |
| M5 | Operator-facing errors are sentences; tracebacks go to the journal | **MET, TESTED** | `errors.py` — `str(exc)` returns the sentence, so the lazy call site is the correct one; `Journal.error` takes the detail |
| M6 | Missing optional dependencies degrade with a stated cost | **MET, TESTED** | `parse_treesitter.degraded_note`, `runner.lint_code`, `langs.missing_note`; `test_runner.py` |
| M7 | Every artefact journals provider, model, prompt hash, attempt, verification, timestamp | **MET, TESTED** | `journal.Journal.generation`; `test_golden_trace.py::test_provenance_is_complete_for_every_generation` |
| M8 | Ports **and** shared types frozen at 1.0 under semver | **MET** | `__init__.py` is the only public surface; `types.py` docstring states the rule; `CHANGELOG.md` from the first commit |
| M9 | No documentation calls guard+sandbox+jail a security boundary | **MET, TESTED** | `guard.py` says "not a security boundary" in its first paragraph; `test_contract.py` greps the docs |
| M10 | No swap logic in the core; `capabilities()` re-read at task boundaries; "no model" is normal | **MET, TESTED** | `session._capabilities`; `test_session.py` — including an AST check that no swap function exists |

## The contract (§5)

| # | Obligation | Status | Where |
|---|---|---|---|
| M11 | `complete()` never raises on refusal | **MET, TESTED** | documented in `ports.py`; conformance kit asserts it against any host |
| M12 | A host without tools ignores `tools` and reports `supports_tools=False` | **MET, TESTED** | conformance kit; the loop's text-marker fallback covers the other side |
| M13 | `capabilities()` reflects the currently loaded model | **MET, TESTED** | `test_session.py::test_a_model_change_is_treated_as_an_epoch_boundary` |
| M14 | `count_tokens` honesty; tokenizer deps never in the core | **MET, TESTED** | `context.Budget.declare()`; `test_codemap.py`; enforced by the stdlib-only test |
| M15 | `write_bytes` atomic, or PORTS.md says otherwise | **MET, TESTED** | `LocalFileSystem.write_bytes` writes a temp file in the same directory then renames; conformance kit checks a 2 MB payload |
| M16 | `ExecPort.run` kills the whole process tree on timeout | **MET, TESTED** | `SubprocessExec._kill_tree` — `setsid`+`killpg` on POSIX, `taskkill /T /F` on Windows; the conformance kit spawns a grandchild and checks it died |
| M17 | `StoragePort` values are JSON-serialisable | **MET, TESTED** | `MemoryStorage.set` round-trips through `json.dumps` to fail at the moment of the mistake; conformance kit |
| M18 | Every write, including model-initiated `apply_patch`, routes through approval and the transaction | **MET, TESTED** | `codemap.call_tool` hands `apply_patch` to a `patch_sink`; **the side door is not built**. `test_patcher.py`, `test_codemap.py` |
| M19 | Event kinds are the closed set of §5.4 | **MET, TESTED** | `types.EVENT_KINDS`; `test_contract.py` asserts the exact tuple |
| M20 | Every Port ships a Null implementation; the engine runs hostless | **MET, TESTED** | `ports.py`; `examples/tiny_host.py` runs a whole session |
| M21 | Cancellation at phase boundaries; resumable state; transactions rolled back | **MET, TESTED** | `loop._check_cancel`; `test_loop.py`, `test_session.py` |

## Engine behaviour

| # | Obligation | Status | Where |
|---|---|---|---|
| M22 | Build/run/test/guard failures attributed to their phase, distinguishably | **MET, TESTED** | `test_runner.py` — the spec's own discrimination test: a broken C file fails in `build`, a divide-by-zero fails in `run` |
| M23 | An anchor matching more than once is refused, never guessed | **MET, TESTED** | `patcher._unique_index`; `test_patcher.py` |
| M24 | No write outside the resolved real project root, in any mode | **MET, TESTED** | `patcher.safe_relpath` + `LocalFileSystem._resolve`; `test_patcher.py` parametrised over 7 escape shapes; conformance kit adds a symlink |
| M25 | Transactions: planner atomicity, sequence numbers, sealing, journaled rollback, queryable history | **MET, TESTED** | `patcher.Transaction`; 8 tests in `test_patcher.py` |
| M26 | Undo restores byte-identical content — encoding, BOM and EOL | **MET, TESTED** | snapshots store **bytes**; property test over an 8-file corpus plus 25 randomised trials |
| M27 | The engine never runs git; generated VCS commands blocked; `.git/` excluded | **MET, TESTED** | `guard.py` VCS rules; `test_session.py` walks the AST for a `git` command; `list()` excludes `.git/` |
| M28 | Injected context ends with the declared-omissions block | **MET, TESTED** | `context.build_context`, `zoom.generate_architecture_context` — present **even when nothing was dropped**, because a block that appears only sometimes is one the model learns to ignore |
| M29 | Unrecognised toolchain output yields one diagnostic, never `[]` | **MET, TESTED** | `diagnostics.parse`; `test_diagnostics.py` |
| M30 | The codemap query interface is never stale; only the injected summary may lag | **MET, TESTED** | `CodeMap.reindex_after_write` vs `zoom.staleness_note`; `test_codemap.py` |
| M31 | The text-marker fallback caps lookups at 3 and forces epoch-per-write | **MET, TESTED** | `CodeMap.answer_text_lookups` (cap, multi-syntax, correct-once) and `CodeMap.force_epoch_per_write`, set from `capabilities().supports_tools` at every task boundary; `test_codemap.py` asserts both halves |
| M32 | Truncation triggers continuation, not regeneration, and is journaled | **MET, TESTED** | `loop._is_truncated`, `_join_continuation`; `test_loop.py`, `test_golden_trace.py` |
| M33 | Failed attempts not accumulated; diagnostics carry forward, not broken code | **MET, TESTED** | `personas.repair_task`; `test_loop.py` asserts the first failure's code is absent from the third prompt |
| M34 | Stagnation hashes normalised code AND diagnostics, keeps a cycle set, reports the cycle in words | **MET, TESTED** | `loop._stagnation`, `_describe_cycle`; `test_loop.py` |
| M35 | Deterministic pre-fixes before the model sees an error; every auto-fix logged | **MET, TESTED** | `runner.autofix`, `loop._prefix_fixes`; `test_loop.py` |
| M36 | Machine-consumed prompts end with an output contract; every consuming site runs the detector | **MET, TESTED** | `personas.CONTRACT_*`, `detect_commentary` called in `loop._generate`; `test_loop.py` including false-positive cases |
| M37 | `<think>` blocks stripped before any use | **MET, TESTED** | `personas.strip_think`, called on every completion; `test_loop.py` including the unclosed-tag case |
| M38 | Model paths assigned by the planner; validated against layout for new files | **MET** | `planner.validate_path`, `observe_layout` |
| M39 | Every implementation task carries its test path | **MET** | `planner.test_path_for`, derived from the project's observed convention |
| M40 | Headless Godot passes on scene/physics/rendering tests carry the caveat | **MET, TESTED** | `langs.headless_caveat_for`; `test_runner.py` — and a pure test earns no caveat, because a caveat on everything is one nobody reads |
| M41 | Same-model review labelled non-independent, near the top | **MET, TESTED** | `review.recommendation_document` emits it above the findings; `test_review.py` asserts both its presence and that it sits in the first quarter of the document |

## Providers, network, distribution

| # | Obligation | Status | Where |
|---|---|---|---|
| M42 | Remote: session-explicit, no env-var auto-on, approval first, budgets, persistent indicator | **MET, TESTED** | `RemoteGate` + `redact.Budget`; `test_redact.py` — including that the budget is checked BEFORE the call, and that a key in the environment enables nothing |
| M43 | Every outbound message, including tool results, passes through `redact.py` | **MET, TESTED** | `redact.redact_messages` covers every role and tool-call arguments; `test_redact.py` captures the actual wire payloads of a scripted three-turn session and asserts them clean |
| M44 | Keys never touch the journal, codemap or any log | **MET, TESTED** | no key is ever passed to the journal; `test_no_network.py` asserts key-shaped variables are scrubbed from child environments |
| M45 | Installers: non-interactive, idempotent, nothing global, degrading, honest summary, meaningful exit code | **MET** | `install.sh`, `install.bat`; CI runs each twice to prove idempotence |
| M46 | The installer verifies interpreter version, fetches 3.11 when needed, reports which branch it took | **MET** | both installers probe by `--version`, fetch via `uv` into `.tools/`, and print the provenance |
| M47 | `requires-python = ">=3.11"`, no cap; CI on 3.11 for Windows and Linux | **MET** | `pyproject.toml`, `.github/workflows/ci.yml` |
| M48 | Zero required runtime deps; CI fails on a non-stdlib module-level import in the core | **MET, TESTED** | `test_contract.py::test_core_has_no_module_level_third_party_imports` |
| M49 | `ccoder doctor` prints the install summary and interpreter provenance | **MET** | `cli.doctor`, `interpreter_provenance` |
| M50 | Vendored use works: no import-time side effects, no metadata reads, version from `version.py` | **MET, TESTED** | three tests, one of which imports the package in an empty directory and asserts the directory is still empty |

## Testing (the load-bearing four)

| # | Obligation | Status | Where |
|---|---|---|---|
| M51 | The no-network test drives a full scripted session with sockets monkeypatched to raise | **MET, TESTED** | `test_no_network.py` — `socket.socket`, `create_connection`, `getaddrinfo` and `create_server` all poisoned, then a whole two-file session runs |
| M52 | Prefix stability: byte-identical prefixes per (persona, epoch, target) | **MET, TESTED** | `test_prefix_stability.py` — 8 tests, including one that scans the prefix for timestamps, uuids and session ids |
| M53 | The Appendix E golden trace is committed and shape-diffed in CI | **MET, TESTED** | `tests/fixtures/worked_session.jsonl`, `test_golden_trace.py` |
| M54 | The port conformance kit ships and covers atomicity, tree-kill, jail, capabilities honesty | **MET, TESTED** | `tests/port_conformance.py` — 29 checks, runnable standalone |
| M55 | The journal records `prompt_ms` per call | **MET, TESTED** | `Completion.prompt_ms` through to `Journal.generation`; `journal.cache_health()` turns it into a sentence |

---

## Known gaps, stated rather than buried

**The ATK migration is written and tested, not performed.** `migrate.py` is
dry-run by default, backs up every file it replaces, and refuses to touch
anything outside the six modules. It has been exercised end to end against a
throwaway copy of the real ATK tree — 56 checks on the OLD call surface — but
**the live checkout has not been modified.** That is a decision for the owner
to make with ATK's own suite green either side, which is what §7.3 requires
and what a script cannot judge.

A finding worth recording, because a plain re-export shim would have hidden
it: **the migration is not a rename.** The engine takes Ports where ATK's
modules took paths, so `Lang.available()`, `diagnostics.feedback()`,
`Diagnostic.source`, `coderun.build_and_run()` and `patcher.apply()` all
changed shape. `adapters/atk/atk_compat.py` supplies the old signatures on
top of the new engine; without it every one of those call sites would have
failed at runtime, in whatever code path happened to run first.

**Mixed line endings.** A file using more than one line-ending style cannot
round-trip byte-for-byte, because "restore the file's style" has no single
answer for it. The engine declares this in `TextFile.assumption` rather than
normalising quietly, and **undo is unaffected** — snapshots hold the original
bytes. M26 is met; the limitation is in writing new content into an already
mixed file.

**Coverage — 74% overall, short of §9's ≥85% target.** Stated as a number
rather than as a claim, because the shortfall is concentrated and worth
naming:

| Module | Cover | Why |
|---|---|---|
| `cli.py` | 0% | needs a terminal and a live endpoint |
| `providers/local_llamacpp.py` | 14% | needs `llama-cpp-python` installed |
| `codemap/parse_treesitter.py` | 21% | needs `tree-sitter` installed |

**Excluding those three, the core is 82%.** They are the modules whose whole
purpose is to bind to something that is not present on a bare machine, and
they are the ones an integration run exercises rather than a unit suite.

The modules §9 singles out — *"the parsers and the loop should be near 100%
because they are where wrongness hides"* — sit at: `redact` 94%, `personas`
91%, `types` 90%, `langs` 89%, `session` 88%, `zoom` 88%, `parse_python` 84%,
`loop` 82%, `codemap` 81%, `context` 80%, `review` 79%, `diagnostics` 79%,
`textio` 79%, `store` 79%. Near 100% they are not; the remaining gap in
`diagnostics` is mostly toolchain-specific parser branches that only fire on
real output from a compiler this machine does not have, and the gap in
`providers/remote` (63%) is the four providers whose wire format no test can
exercise without an account — `Anthropic` is covered end to end because the
capture harness subclasses it.

Eleven language tests skip by design on a machine without Rust, Go, Java,
Zig, Lua or Godot — §9 requires that they skip with a printed note rather
than fail, and they do.

---

## How to re-run this audit

```bash
pytest -q                        # the whole suite
python tests/port_conformance.py # the kit, against the Null ports
ruff check cognitive_coder tests examples
ccoder doctor                    # what this machine can actually do
```

CI names the five architecture-guarding checks individually rather than
burying them in "run pytest", because each guards something that would still
*work* if it broke.
