# The ATK adapter — built, tested without Qt, not yet installed in ATK

This directory holds ATK's PySide6 panel and its six Port implementations.
**It is deliberately outside `cognitive_coder/`**: the panel imports PySide6,
and keeping adapters out of the package is what enforces the no-GUI-in-the-
core rule mechanically rather than by good intentions. A test walks the core
and fails the build if it ever imports from here.

(This file used to say "specified and not yet built" and "will contain". The
code below exists; what has NOT happened is installing it into a live ATK
checkout — see the migration.)

## What is here

| File | What it is |
|---|---|
| `ccoder_host.py` | The six Ports, plus the panel's logic that does not need a widget. **Qt-free**, so it is tested without a QApplication. Destination: `atk/core/ccoder_host.py`. |
| `ccoder_panel.py` | The workspace tab — the ONLY file here that imports Qt. Destination: `atk/ui/ccoder_panel.py`. |
| `atk_compat.py` | ATK's OLD call signatures on top of the engine. Destination: `atk/core/ccoder_compat.py`. |
| `migrate.py` | Installs the three above and turns ATK's six coding modules into shims. Dry run by default. |
| `test_adapter.py` | pytest; the Ports against the engine's contract and the conformance kit. Runs in Cognitive Coder's own suite. |
| `test_migration.py` | ATK's test style (a script printing PASS/FAIL): does ATK's old call surface survive the migration? |

## The six Ports (§7.2)

| Port | Class | Binds to, and how |
|---|---|---|
| `LLMPort` | `ATKLLM` | `atk/core/llm_engine.py`'s loaded model — it never loads one; ATK's swap button stays ATK's. Drains `chat_stream`; the first chunk splits `prompt_ms` (prefill) from `decode_ms`; `split_think()` when ATK is present; exact counts through `mistral-common` for Mistral-family models, re-chosen when the model changes. |
| `FileSystemPort` | `ATKFileSystem` | The project root, jailed on resolved real paths. Atomic writes that keep the file's permission bits; the optional `append_bytes` for the journal. |
| `ExecPort` | `ATKExec` | A tree-killing subprocess runner — **not** `atk/core/sandbox.py`, which screens GENERATED code and stays for the legacy Developer Sandbox. No `env` means the scrubbed environment, never the inherited one; a timeout of 0 waits. |
| `StoragePort` | `ATKStorage` | `ctx.settings["ccoder"]["projects"][<key>]`, with SQLite in `<DATA_DIR>/ccoder/<key>/`, where `<key>` is 12 hex digits of SHA-256 of the project root — one bucket per PROJECT. **Never ATK's `state.db`**: a second database file is fine, a second schema in that file is not. The worker thread works on a private copy; the panel's `flush()` writes it back and saves on the GUI thread, and says so if the save fails. |
| `EventPort` | `ATKEvents` | Plain callables. The panel passes `_Bridge` signals, so every widget update is queued onto the GUI thread. |
| `ApprovalPort` | `ATKApproval` | Diffs: written without asking only if auto-apply is on (Setup → System & Resources → Advanced; off by default); otherwise a diff dialog, **Apply / Don't apply, default Don't**. Remote sends: **always** a question, default No (C3). Both dialogs are built on the GUI thread — `AskOnGuiThread` over `_Bridge.ask` with a blocking queued connection — never on the worker. |

The panel also refuses to build with no project folder set (it used to fall
back to the working directory, which is ATK's own source tree), keeps Build
disabled while a build runs, and cancels a running build when it is closed.
Its console, diff view and CodeMap tree register with `atk/ui/detach.py`
like every other ATK pane.

Tests: `pytest adapters/atk/test_adapter.py` (part of `pytest -q` from the
clone) and `python adapters/atk/test_migration.py`. Neither needs Qt; the
panel itself has no automated test, which is why every decision in it that
could be made without a widget was moved into `ccoder_host.py`.

## The migration (§7.3)

Six modules in `ATK/atk/core/` were the starting point for this engine and
carry bug fixes found the hard way:

`langs.py` · `diagnostics.py` · `codeguard.py` · `coderun.py` ·
`patcher.py` · `codectx.py`

All six have been ported into the core, with the hard-won details preserved
and the reasoning kept in the docstrings:

- rustc's message and location are on separate lines and must be paired **in
  order**;
- Python's deepest frame is **last**, JavaScript's is **first**;
- an ambiguous anchor is **refused, never guessed**;
- unparsed toolchain output never yields an empty list;
- commands are argv lists, never strings.

`migrate.py` writes **nine files** — the six become re-export shims (each
original kept as `<name>.py.pre-ccoder`), and `ccoder_compat.py`,
`ccoder_host.py` and `ui/ccoder_panel.py` are installed — and nothing else:

```
python adapters/atk/migrate.py --atk D:/Analyst_Toolkit/ATK            # look
python adapters/atk/migrate.py --atk D:/Analyst_Toolkit/ATK --dry-run  # same
python adapters/atk/migrate.py --atk D:/Analyst_Toolkit/ATK --apply    # do
```

It checks importability with ATK's own `.venv` interpreter (or `--python`),
and refuses the whole run if any target is a symlink or resolves outside the
tree. **ATK's full suite must be green before and after**; only then are
ATK's imports updated and the shims deleted — by a person, not the script.
The legacy `atk/core/sandbox.py` is not extended — `guard.py` and
`runner.py` supersede it.

## Two ATK constraints the core already respects

- **16 GB VRAM ceiling, one model at a time.** The core has no swap logic.
  It asks `capabilities()` what is loaded, treats a change as an epoch
  boundary, and journals the model per call. The swap button is ATK's, and
  the operator guidance — *at most once per session, at a phase boundary* —
  belongs in the panel text, not in code.
- **Zero telemetry.** Offline is the default and remote mode cannot be
  enabled by an environment variable, a config file, or an accident. See
  `docs/PROVIDERS.md`.
