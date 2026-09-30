# Auditing an existing project

`ccoder build` makes something from a request. `ccoder audit` is the other
half of the loop: point it at a project that already exists — one this
engine built, or one somebody wrote by hand — and it tells you what is
wrong and what could be better, ranked, with file and line. It then writes
a plan that `ccoder build --spec` can carry out, and the next audit says
what was fixed.

```
ccoder audit game/                               # report + plan
ccoder audit game/ --focus "collisions feel wrong"
ccoder build -p game/ --spec game/.cc_state/audit/plan.md
ccoder audit game/                               # what was fixed, what is new
```

## What it does, in order

The same order as everything else here: deterministic first, model second,
human last.

1. **Inventory.** Which files are source, in which language. What it left
   out, and why — the skip list, the per-file size cap, the file cap — is
   the first thing the report says, not the last.
2. **Tools.** The review stage's scanners on every file (credentials,
   `eval` on non-literals, empty handlers, and bandit/semgrep/… where they
   are installed), names the project uses but does not define, functions
   nothing reaches ("possibly unused" — only when the call graph AND a
   text search of the whole project agree), and modules no test mentions.
3. **Does it work as it stands?** Every file is parsed or compiled, and
   the project's tests are run once per language. Zero tests is reported
   as zero tests, not as a pass.
4. **The model.** One bounded completion per file for the files most worth
   reading — the ones the tools flagged, then the biggest — up to
   `--max-model-files`. Every call shares one byte-identical cached prefix
   (the project architecture and its `.ccoder/skills/`), so a local server
   processes it once. The reply is read as the LAST object carrying
   `findings`: a reasoning model restates the schema before answering. A
   severity the contract did not offer counts as high. "Looked and found
   nothing" and "gave no usable answer" are different sentences.
5. **Report and plan.** `.cc_state/audit/report.md` and
   `.cc_state/audit/plan.md`. The report keeps the tools' findings and the
   model's apart, and says near the top that the model was shown the
   tools' findings first — so agreement between them is not two
   independent confirmations.

## What it never does

- **It never runs the program.** Tests run; `main` does not. An audit must
  not start somebody's game, server or clean-up script to find out what it
  does.
- **It never writes a source file.** Everything it writes is under
  `.cc_state/`: the report, the plan, `last.json` and a journal under
  `.cc_state/audit/`, and the code index beside them.
- **It never asks for approval**, because it has nothing to approve. The
  CLI gives it `DenyAll` so that, if anything ever did ask, the answer
  would be no. Changing code is `build`'s job — through the transaction,
  the snapshot and the approval gate.

## The iteration

The plan is a build request. It lists the high and medium findings as work
items under the files they concern, tells the model to change only those
files and only what each item asks, and leaves the low-severity findings
in the report. Existing test files are named only when an item concerns
them: a named test file becomes a task, and the planner would otherwise
rewrite a test that should only be run.

When `build` works on a file that already exists, its first attempt shows
the model the file and asks for a change — it does not regenerate the file
from the request. A file too large to show whole is refused with a
sentence, before any model call, rather than rewritten from its first
half.

The next audit compares the tools' findings with the last one — by file,
kind and title, not by line — and leads with "N are gone, M remain, K are
new". The model's findings are not compared: two runs of the same model
word things differently, and a comparison of wordings would report churn
as progress.

## Options

| Option | Default | Effect |
|---|---|---|
| `FOLDER` | `--project`, or here | the project to audit |
| `--focus TEXT` | — | what to look at especially; reaches the model and the plan |
| `--no-model` | off | tools only |
| `--no-tests` | off | do not run the project's tests |
| `--max-files N` | 60 | files reviewed; the rest are named as not reviewed |
| `--max-model-files N` | 12 | files the model reads |
| `--only GLOB` | all | review only matching paths (repeatable) |
| `--url` | `http://127.0.0.1:8080` | the local model endpoint |

The audit runs on a local model only for now; `--remote` is refused with a
sentence.

## Embedding it

```python
from cognitive_coder import AuditConfig, audit_project

report = audit_project(host, config=AuditConfig(focus="the renderer"))
print(report.document())        # the report
plan = report.to_spec()         # the build request
```

A host provides the same Ports as for a build. `approval` is never called.
`cancel` (any object with `is_set()`) is checked between phases and
between model calls.
