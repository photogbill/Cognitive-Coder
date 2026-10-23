# The next tabs — Blueprint, Improve, Red Team, Evaluate

> **BUILT 2026-10-22.** All four shipped as `cognitive_coder/{scope,bughunt,improve,redteam,blueprint}.py` with `structured.py` behind their JSON asks, and as ATK's `atk/core/ccoder_ops.py` +
> `atk/ui/ccoder_tabs.py` (four chat-first tabs, per-file right-click on the board). What follows is the design of record they were built against.

*Design of record, 2026-10-22, from Bill's notes of 10-20 and the journals of
five failed builds (HYBRID.md, "The next five builds"). This is the plan the
engine work below is built against; the ATK panel (four tabs, per-file
right-click) is built on top of these engine operations, not before them.*

The harness fixes shipped first — this document assumes them:
`narrow_min_lines=40`, the deterministic floor (format before write, the
lint gate in the verdict), `keep.py`'s planted/duplicated/narration rules,
`nonerror.py`, `speclint.py`, the untested-is-not-ok rule, `ccoder
postmortem`. A tab built on the old loop would be a hundred files of the
same defects.

One rule governs every tab: **the tab constrains the edit shape, never the
model.** Improve, Repair and Red Team never request a whole file once a file
has real content. A diffusion model answers with a hole, a chat model with
a named definition (`symedit`), and `keep.py` refuses both if code goes
missing. Whole-file generation is for the first attempt of a new file and
nothing else.

---

## 1. Blueprint — a system of modules of files

Bill: *"instead of writing 10 files, it might write 10 modules each with 10
files for one system."*

Why it is not "the planner, bigger": the two spec-truncation bugs and
"neither model saw the spec" (HYBRID.md) get catastrophically worse at a
hundred files. A system is a different object from a long file list. It
has internal contracts, a build order, and a reason for each boundary.

### The shape

```
system spec
  └─ decomposition  → modules, each with: purpose, owned files, PUBLIC
  │                   CONTRACT (the pinned interface, one level up: the
  │                   signatures other modules may call), dependencies
  ├─ build order    → the dependency graph, leaves first (codemap.rank
  │                   already holds the reference graph); a leaf can be
  │                   tested alone, which is the only reason to build it
  │                   first
  ├─ one SESSION per module, in order
  │     the other modules are present as STUBS that satisfy their
  │     contracts (the skeleton stage, one level up), so the module
  │     compiles and its tests run against contracts, not against code
  │     that does not exist yet
  │     each session has its own journal, snapshots, verify, postmortem
  │     — which is what makes a 100-file failure diagnosable
  ├─ the cross-module ledger (BLUEPRINT.md in the project) — every
  │     contract, every decision, every deviation a module session made,
  │     carried into the next module's prompt as band 0
  └─ integration  → the system-level tests last, written from the spec's
                    own examples (specvectors, one level up)
```

### Engine operations

* `blueprint.decompose(spec) -> Blueprint` — modules, contracts, order.
  Deterministic checks on the result: every contract name is defined by
  exactly one module; no dependency cycle; every file belongs to one module;
  speclint on the spec first.
* `blueprint.stubs(bp, module) -> files` — the other modules' contracts as
  stubs, the skeleton stage's own code.
* `Session.run_module(bp, module)` — a normal session with band 0 =
  the module brief + the ledger + the contracts it may call.
* `blueprint.ledger` — append-only, rewritten into the prompt as the
  "governing claims" band (CONTEXT_ARCHITECTURE: ~50 tokens a claim).
* The spec slice per module is the module's section plus every contract it
  depends on — never the whole system spec, which is the truncation bug
  waiting to happen again.

### What it does not try to do
Build modules in parallel (one model, one card), or guess a decomposition
the operator has not approved: the decomposition is shown, checkboxed, and
editable before the first session starts.

---

## 2. Improve — a conversation that ends in a change list

Bill: *"a chat tab specifically for going back and forth with a model
regarding possible improvements … it could then structure the instructions
optimally once they are agreed upon."* And: *"I should be able to type in
what I think the software needs, and … have the AI examine the software
code from the perspective of an expert in that particular kind of
software."*

The 26 debug sessions of 10-20 were this conversation held in a box built
for tracebacks. The tab is the box built for it.

### The shape

1. **Scope** — a file, a folder, or the project; the codemap's ranked map is
   the model's view of it.
2. **Perspective** — a persona from the registry (`personas.py`): the
   generalist, the domain expert ("a game-engine programmer", "a signals
   engineer"), the mad scientist (same registry, a different risk
   appetite: proposes what works and nobody tries). Bill's "thinking hat /
   mixture of experts" is this picker.
3. **The conversation** — free text both ways. The model may ask; the
   operator may redirect. Nothing is written to disk during it.
4. **The change list** — when the operator says "that's it", the model
   structures what was agreed into `ChangeItem`s: `title`, `files`, `what
   must still hold` (the contract), `how to verify` (a test to add, or an
   existing one that must still pass), `risk`. Each is a checkbox. The
   structuring prompt is the one place a small model is good at exactly
   this: given the shape, it fills it.
5. **Apply** — checked items run one at a time through the narrowed-edit
   loop (never whole-file), each verified, each a snapshot; the project
   notes file (`IMPROVEMENTS.md`, the Improve ledger) gets a row per
   item with its outcome, and the next item's prompt carries the rows.

### Engine operations
* `improve.converse(session, scope, persona, messages) -> reply`
* `improve.structure(session, transcript) -> list[ChangeItem]` (grammar-
  constrained: JSON_ARRAY_GBNF; a reasoning model's thinking stripped —
  `atk-reasoning-model-json`)
* `improve.apply(session, items) -> outcomes` — each item a narrowed task
  with `keep_existing` on and the contract pinned.

### Evolutionary mode
Bill: *"build simple, then improve it from that base into something more."*
This is Improve run in a loop with the mad-scientist persona proposing and
the generalist applying — the proposer never writes code; the applier never
proposes. The ledger is the genome.

---

## 3. Red Team — an attacker who has to prove it

Bill: *"otherwise they may use code that instantly exposes them."* Own tab,
chat interface, a file or a folder selectable. Creative control given.

### The rule that makes it honest
**A finding without a reproducer is a guess.** The attacker persona does
not opine; it produces, for every finding, one of:
* a **test** that fails against the current code and must pass after the
  fix (the mutation harness runs it to prove it actually bites), or
* a **crafted input** and the observed bad behaviour (a crash, a path
  outside the project, a shell command reached, a secret printed), run in
  the sandbox and captured.

A finding the reproducer does not confirm is reported as "could not be
demonstrated" — shown, never counted.

### Order of work, deterministic first
1. **Scanners** — `guard.scan` (already: secrets, dangerous calls), plus
   `bandit` and `semgrep` when installed (the same cost-of-absence note as
   every optional tool). Their findings are real before any model runs.
2. **The attacker's threat model** — the model reads the scope and writes
   the threat model as a structured list: *trust boundaries* (every input
   that comes from outside: argv, files, sockets, env, the user), *assets*
   (what the code can reach: files, network, subprocess, credentials),
   *what would expose the operator* (a path traversal that reads outside
   the project, an injection that reaches a shell, a secret in a log, a
   default that binds 0.0.0.0, an unsafe deserialization, an eval). The
   list is the attack plan; the operator sees it before any attack runs.
3. **Attacks** — one per threat, each a reproducer as above, run in the
   sandbox with the project's own runner (`runner.build_and_run`), never on
   the operator's files directly.
4. **The report** — `REDTEAM.md` in the project: scope, threat model,
   findings with severity and the reproducer, "could not demonstrate" set
   aside. Checkboxes. Checked findings become Repair tasks with the
   reproducer as the failing test — the repair is accepted only when the
   reproducer passes, which is the acceptance rule the loop already has.
5. **The conversation** — the operator can argue: "that input can never
   reach there, it comes from a config file we control". The attacker
   either demonstrates it can, or withdraws the finding and says so.

### Personas
`attacker` (writes reproducers; never fixes), `defender` (the repairer,
with the reproducer as its test; never argues). The same registry.

### What it refuses
Anything that is not against the operator's own code in the sandbox. The
tab is a reviewer of code the operator wrote or had written; it is not a
tool for attacking anything else, and the prompt says so.

---

## 4. Evaluate & Bug Hunt — the floor made visible

Bill: *"a software debugging tab specifically to hunt for software bugs …
a list of bugs, such as mismatched function names, indentation errors …
THEN I can feed the output into the code repair tab."*

Most of this tab is the deterministic floor, made visible and saved:

1. `ruff` (all rules, not only the gate), `ruff format --check`, the
   codemap's unresolved names, `keep.py`-style placeholder/narration scan
   over the whole tree, a **type checker** (pyright or mypy when installed —
   one `langs.py` entry, one `_parse_*` in `diagnostics.py` beside the
   fifteen there; pygame ships `.pyi` stubs and the 10-22 `blit` bug was a
   type error).
2. **Test adjudication**, shown: each failing test with the judge's verdict
   (code / test / spec) and the quoted spec line — the thing that was true
   10:4 in Devstral's run and invisible.
3. **Properties from the spec** — ask the model for invariants ("heading
   always in [0, 360)", "six shield entries") as Hypothesis-style tests;
   exact-value tests were the wrong ones.
4. The model's pass (the "fix sheet"): only after 1–3, with their findings
   in the prompt, so it names what the tools could not — and every item
   it names must be located to a line or it is "could not locate".

Output: `BUGHUNT.md`, checkboxes, each row a Repair task — the hand-off
Bill described.

---

## 5. Per-file right-click (every tab)

The spine of Repair and Improve. Three short submenus that compose, not a
flat list: **what** (diagnose & edit / rewrite from nothing / red team /
improve) × **who** (the roster: any loaded model, incl. a 70B for one
file) × **how much** (narrowed edit / whole file). The engine already
supports each axis (`debugrepair`, `roster`/`load_model`, `narrow`). The
70B per file is a narrowed-edit feature: a whole-file attempt at 1 tok/s is
a 40-minute attempt.

Open in Visual Studio / VS Code: `devenv /edit file` and `code -g
file:line` as menu entries; twenty lines, no repo. Editing *inside* VS is a
VSIX and its own project, as Bill said.

---

## Order of build

Engine, then panel; each tab's engine ops land with their tests before the
tab that calls them.

1. Evaluate & Bug Hunt engine ops (the floor is mostly built; the type
   checker lane and the properties prompt are new).
2. Improve: `converse` / `structure` / `apply` and the ledger.
3. Red Team: scanners, threat model, reproducer harness, report.
4. Blueprint: decompose, stubs, run_module, ledger, integration.
5. The four tabs and the right-click menu in ATK's `ccoder_panel.py`.
