# The ATK adapter — phase 9, specified and not yet built

This directory is where ATK's PySide6 panel and its six Port implementations
will live. **It is deliberately outside `cognitive_coder/`**: this adapter
imports PySide6, and keeping adapters out of the package is what enforces the
no-GUI-in-the-core rule mechanically rather than by good intentions. A test
walks the core and fails the build if it ever imports from here.

## What the adapter will contain (§7.2)

Six Port implementations, each a thin binding to something ATK already has:

| Port | Binds to |
|---|---|
| `LLMPort` | `atk/core/llm_engine.py` — GGUF loading, streaming, GBNF grammar-constrained JSON, `split_think()` |
| `FileSystemPort` | ATK's project root, with its own jail |
| `ExecPort` | a tree-killing subprocess runner — **not** `atk/core/sandbox.py`, which stays for the legacy Developer Sandbox until that panel is replaced |
| `StoragePort` | ATK's `state.db` and settings |
| `EventPort` | `atk/core/workers.py` — `QRunnable` + `submit()`, marshalled to the UI thread |
| `ApprovalPort` | a diff dialog, with auto-apply behind Setup → System & Resources → Advanced |

Plus the panel itself, registered with `atk/ui/detach.py` so its panes detach
like every other ATK pane.

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

When phase 9 lands, ATK's copies become thin re-exports or are deleted, and
**ATK's full suite must be green before and after.** The legacy
`atk/core/sandbox.py` is not extended — `guard.py` and `runner.py` supersede
it.

## Two ATK constraints the core already respects

- **16 GB VRAM ceiling, one model at a time.** The core has no swap logic.
  It asks `capabilities()` what is loaded, treats a change as an epoch
  boundary, and journals the model per call. The swap button is ATK's, and
  the operator guidance — *at most once per session, at a phase boundary* —
  belongs in the panel text, not in code.
- **Zero telemetry.** Offline is the default and remote mode cannot be
  enabled by an environment variable, a config file, or an accident. See
  `docs/PROVIDERS.md`.
