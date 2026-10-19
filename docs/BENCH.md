# The regression suite — `ccoder bench`

Did a change to the ENGINE make builds better or worse? Every fix in the
CHANGELOG came from one real build read line by line; nothing then checked
that it did not break what an earlier fix had mended. This answers that, the
way Aider's benchmark and OpenHands' evaluation harness do: small tasks,
graded by tests the model never sees, run headless, compared run to run.

It is **not** for comparing models. Run it with the model you build with,
before and after a change to the engine.

```
ccoder bench --check                  # is the suite itself sound?
ccoder bench --url http://127.0.0.1:8080 --label before
  … change the engine …
ccoder bench --url http://127.0.0.1:8080 --label after
ccoder bench --replay 20261019-201500-before    # no model needed
```

## Where things are

| | |
|---|---|
| `bench/tasks/<name>/` | the suite (eight starter tasks, below) |
| `bench/runs/<when>[-label]/` | each run: `SUMMARY.md`, `results.jsonl`, and per task the project as the engine left it, `replies.jsonl`, `hidden_tests.txt` |

Both beside the engine, on the same drive — never under a user profile.
`bench/runs/` is in `.gitignore`.

## A task

```
task.md        the request, as an operator would type it
bench.json     {"kind": "build" | "change" | "debug", "lang": "python",
                "timeout_s": 900, "note": "...",
                "debug": {"error_text": "...", "command": "python main.py",
                          "description": "..."}}
start/         the project's files before the run (change and debug tasks)
hidden/        tests copied in AFTER the run; the model never sees them
reference/     a solution, used only by --check
```

A debug task's pasted traceback writes `{PROJECT}/` where the folder goes; the
run puts the real path in.

`--check` proves each task measures something: its hidden tests pass on
`start/` + `reference/`, and — for change and debug tasks — fail on `start/`
alone.

## What a run reports

Per task: PASS/FAIL (every hidden test passed), hidden tests passed/total,
files verified/built/failed, model calls, seconds, narrowed edits placed,
replies refused by the dropped-code guard, and why a failure failed. Then,
against the run before: **which tasks got better, which got worse** — by
name, never only the total.

## Replay

Every model reply of a live run is recorded (`replies.jsonl`, with each
prompt's hash). `--replay RUN` answers with those replies, in order, and no
model. When the engine asks the same questions, a change to extraction,
splicing or routing is tested at no inference cost. When it asks different
ones — a new call was added — the summary says at which call the prompts
diverged, so a diverged replay is not mistaken for a result. Diffusion
canvases are not recorded; replay is for chat models.

## The starter suite

| task | kind | what it exercises |
|---|---|---|
| build-roman | build | one module, exact behaviour, several error cases |
| build-inventory | build | three files that must agree on one data type: pinned interfaces, order, tests beside modules |
| build-wordfreq | build | ordering rules stated in words: spec slicing, the judge |
| build-test-first | build | a request that asks for tests first |
| change-todo | change | a change where everything not mentioned must survive |
| change-big-file | change | one function of a 176-line file changes |
| debug-traceback | debug | a pasted traceback and its command: locate, repair, confirm |
| debug-big-file | debug | a crash inside one function of a 167-line module: a narrowed repair |

Add your own: a folder with the four files above. Keep tasks small enough
that a 7–24B model can pass them; a suite that everything fails measures
nothing.
