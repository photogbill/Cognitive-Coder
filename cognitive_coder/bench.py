# SPDX-License-Identifier: Apache-2.0
"""The regression suite: did a change to the ENGINE make it better or worse?

WHY THIS EXISTS, 2026-10-19
---------------------------
Every fix in the CHANGELOG came from one real build — the racing spec, the
Star Trek spec — read line by line. Nothing then checked that the Oct 16
fixes did not break what the Oct 2 fixes had mended. Aider answers that
question with a benchmark of small exercises with hidden tests, run headless
(two tries: the second sees the test output), with replay mode to re-run
saved model replies through a changed parser at no inference cost.
OpenHands' harness adds the separation that keeps it honest: the agent
works, and a SEPARATE step grades with tests the agent never saw.

This is that, for this engine. It is not for comparing models — the
operator runs it with the model they use — but for comparing the ENGINE
before and after a change: "3 tasks better, 1 worse, here is which".

A TASK is a folder under `bench/tasks/`:

    task.md        the request, exactly as an operator would type it
    bench.json     {"kind": "build" | "change" | "debug", "lang": "python",
                    "timeout_s": 900, "debug": {"error_text", "command",
                    "description"}, "note": "..."}
    start/         files the project starts with (change and debug tasks)
    hidden/        tests copied in AFTER the run and never shown to the model
    reference/     a solution — only for `check_suite`, which proves the
                   hidden tests pass on it, and fail on start/ alone

A RUN is a folder under `bench/runs/<when>[-label]/` — beside the code, on
the same drive, never under a user profile — holding, per task, the project
as the engine left it, `replies.jsonl` (every model reply, for replay), and
for the run: `results.jsonl` and `SUMMARY.md`, which compares the run with
the one before it.

REPLAY feeds a run's recorded replies back, in order, to the CURRENT engine.
When the engine asks the same questions, the replies fit and a parser,
splice or routing change is tested with no model loaded; when it asks
different ones (a new call was added), the run says at which call the
prompts diverged, so a number from a diverged replay is not mistaken for a
real one. Diffusion canvases are not recorded: replay is for chat models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

from .journal import prompt_hash
from .types import Completion, ModelCapabilities, ToolCall

#: The engine's own checkout: `bench/` lives beside `cognitive_coder/`.
REPO = Path(__file__).resolve().parent.parent
SUITE_DIR = REPO / "bench" / "tasks"
RUNS_DIR = REPO / "bench" / "runs"
KINDS = ("build", "change", "debug")
HIDDEN_DIR = "bench_hidden_tests"
REPLIES = "replies.jsonl"
DEFAULT_TIMEOUT_S = 1800.0


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchTask:
    name: str
    path: Path
    kind: str
    request: str
    lang: str = "python"
    timeout_s: float = DEFAULT_TIMEOUT_S
    debug: dict = field(default_factory=dict)
    note: str = ""

    @property
    def start(self) -> Path:
        return self.path / "start"

    @property
    def hidden(self) -> Path:
        return self.path / "hidden"

    @property
    def reference(self) -> Path:
        return self.path / "reference"


def load_suite(suite: Path | str = SUITE_DIR, only: Sequence[str] = ()
               ) -> tuple[list[BenchTask], list[str]]:
    """(tasks, problems): every well-formed task folder, and a sentence for
    each folder that is not one. Sorted by name, so runs line up."""
    root = Path(suite)
    tasks: list[BenchTask] = []
    problems: list[str] = []
    if not root.is_dir():
        return [], [f"there is no suite at {root}"]
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and folder.name not in only:
            continue
        meta_path = folder / "bench.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            problems.append(f"{folder.name}: no bench.json")
            continue
        except (OSError, ValueError) as exc:
            problems.append(f"{folder.name}: bench.json is not readable "
                            f"JSON ({exc})")
            continue
        kind = str(meta.get("kind", "build")).lower()
        if kind not in KINDS:
            problems.append(f"{folder.name}: kind {kind!r} is not one of "
                            f"{', '.join(KINDS)}")
            continue
        try:
            request = (folder / "task.md").read_text(encoding="utf-8")
        except OSError:
            request = ""
        if kind != "debug" and not request.strip():
            problems.append(f"{folder.name}: task.md is missing or empty")
            continue
        if not (folder / "hidden").is_dir():
            problems.append(f"{folder.name}: no hidden/ tests to grade it by")
            continue
        tasks.append(BenchTask(
            name=folder.name, path=folder, kind=kind, request=request.strip(),
            lang=str(meta.get("lang", "python")),
            timeout_s=float(meta.get("timeout_s", DEFAULT_TIMEOUT_S)),
            debug=dict(meta.get("debug", {}) or {}),
            note=str(meta.get("note", ""))))
    for name in only:
        if name not in {t.name for t in tasks} and not any(
                p.startswith(f"{name}:") for p in problems):
            problems.append(f"{name}: no such task in {root}")
    return tasks, problems


# --------------------------------------------------------------------------
# recording and replaying a model
# --------------------------------------------------------------------------

def _completion_dict(c: Completion) -> dict:
    return {"text": c.text, "finish_reason": c.finish_reason,
            "tool_calls": [{"id": t.id, "name": t.name,
                            "arguments": t.arguments,
                            "repaired": t.repaired} for t in c.tool_calls],
            "tokens_in": c.tokens_in, "tokens_out": c.tokens_out,
            "model": c.model, "prompt_ms": c.prompt_ms,
            "decode_ms": c.decode_ms, "error": c.error}


def _completion_from(d: dict) -> Completion:
    return Completion(
        text=str(d.get("text", "")),
        finish_reason=str(d.get("finish_reason", "stop")),
        tool_calls=tuple(ToolCall(id=str(t.get("id", "")),
                                  name=str(t.get("name", "")),
                                  arguments=dict(t.get("arguments") or {}),
                                  repaired=bool(t.get("repaired")))
                         for t in d.get("tool_calls") or ()),
        tokens_in=int(d.get("tokens_in", 0) or 0),
        tokens_out=int(d.get("tokens_out", 0) or 0),
        model=str(d.get("model", "")),
        prompt_ms=int(d.get("prompt_ms", 0) or 0),
        decode_ms=int(d.get("decode_ms", 0) or 0),
        error=str(d.get("error", "")))


def _caps_dict(c: ModelCapabilities) -> dict:
    return {"name": c.name, "family": c.family,
            "context_tokens": c.context_tokens,
            "supports_tools": c.supports_tools,
            "supports_grammar": c.supports_grammar,
            "supports_vision": c.supports_vision,
            "supports_fim": c.supports_fim,
            "token_count_is_estimate": c.token_count_is_estimate}


class RecordingLLM:
    """An LLMPort that passes every call through and writes it down.

    Counted and timed here, so the suite's numbers do not depend on what
    the provider reports. Everything else is delegated (`__getattr__`), so
    a capability the wrapped port has — or lacks — is the wrapper's too.
    """

    def __init__(self, inner: Any, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            caps = _caps_dict(inner.capabilities())
        except Exception:                                # noqa: BLE001
            caps = {}
        self._write({"capabilities": caps})

    def _write(self, row: dict) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def complete(self, messages, **kw) -> Completion:
        t0 = time.monotonic()
        c = self._inner.complete(messages, **kw)
        dt = time.monotonic() - t0
        self.calls += 1
        self.tokens_in += int(c.tokens_in or 0)
        self.tokens_out += int(c.tokens_out or 0)
        self.seconds += dt
        self._write({"call": self.calls, "prompt_sha256":
                     prompt_hash(list(messages)), "seconds": round(dt, 3),
                     "completion": _completion_dict(c)})
        return c

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ReplayLLM:
    """An LLMPort that answers with a recorded run's replies, in order.

    Says where it stopped fitting: ``diverged_at`` is the first call whose
    prompt differs from the recorded one (0: none did), and ``exhausted``
    that the engine asked for more replies than the run recorded — then it
    answers with an error completion, which the engine stops on cleanly.
    """

    def __init__(self, path: Path) -> None:
        rows = [json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        caps = next((r["capabilities"] for r in rows if "capabilities" in r),
                    {}) or {}
        self._caps = ModelCapabilities(
            name=str(caps.get("name", "replay")),
            family=str(caps.get("family", "")),
            context_tokens=int(caps.get("context_tokens", 32768) or 32768),
            supports_tools=bool(caps.get("supports_tools", False)),
            supports_grammar=bool(caps.get("supports_grammar", False)),
            supports_vision=bool(caps.get("supports_vision", False)),
            supports_fim=bool(caps.get("supports_fim", False)),
            is_remote=False,
            token_count_is_estimate=bool(caps.get("token_count_is_estimate",
                                                  True)))
        self._rows = [r for r in rows if "completion" in r]
        self.calls = 0
        self.diverged_at = 0
        self.exhausted = False
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0

    def complete(self, messages, **kw) -> Completion:
        self.calls += 1
        if self.calls > len(self._rows):
            self.exhausted = True
            return Completion(text="", finish_reason="error",
                              model=self._caps.name,
                              error="the replay ran out of recorded replies "
                                    "— the engine asked for more than the "
                                    "recorded run did")
        row = self._rows[self.calls - 1]
        if not self.diverged_at and row.get("prompt_sha256") != prompt_hash(
                list(messages)):
            self.diverged_at = self.calls
        c = _completion_from(row["completion"])
        self.tokens_in += c.tokens_in
        self.tokens_out += c.tokens_out
        return c

    def stream(self, messages, **kw):
        yield self.complete(messages, **kw).text

    def capabilities(self) -> ModelCapabilities:
        return self._caps

    def count_tokens(self, text: str) -> int:
        return max(1, len(text or "") // 4)


# --------------------------------------------------------------------------
# grading: the hidden tests, never shown to the model
# --------------------------------------------------------------------------

@dataclass
class Graded:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    total: int = 0
    runner: str = ""
    output: str = ""
    problem: str = ""

    @property
    def ok(self) -> bool:
        return self.total > 0 and self.passed == self.total \
            and not self.problem


def _scrubbed_env() -> dict:
    keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP",
            "HOME", "USERPROFILE", "LANG", "LC_ALL", "PYTHONIOENCODING",
            "COMSPEC", "PATHEXT")
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def grade(project: Path, hidden: Path, ex: Any, *,
          timeout: float = 300.0) -> Graded:
    """Copy the hidden tests in and run them — pytest if it is there."""
    target = Path(project) / HIDDEN_DIR
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(hidden, target)
    (target / "__init__.py").touch(exist_ok=True)
    env = _scrubbed_env()
    probe = ex.run([sys.executable, "-c", "import pytest"], cwd=str(project),
                   timeout=60, env=env)
    if probe.ok:
        argv = [sys.executable, "-m", "pytest", "-q", "-p",
                "no:cacheprovider", "--tb=short", HIDDEN_DIR]
        runner = "pytest"
    else:
        argv = [sys.executable, "-m", "unittest", "discover", "-s",
                HIDDEN_DIR, "-t", "."]
        runner = "unittest"
    proc = ex.run(argv, cwd=str(project), timeout=timeout, env=env)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    g = Graded(runner=runner, output=out[-6000:])
    if proc.timed_out:
        g.problem = f"the hidden tests did not finish in {timeout:.0f}s"
        return g
    if runner == "pytest":
        # the LAST summary line: a test's own output or an assertion message
        # can say "3 passed" too
        summary = [ln for ln in out.splitlines()
                   if re.search(r"\d+ (?:passed|failed|errors?)\b", ln)]
        last = summary[-1] if summary else ""
        for key, attr in (("passed", "passed"), ("failed", "failed"),
                          ("error", "errors")):
            m = re.search(rf"(\d+) {key}", last)
            if m:
                setattr(g, attr, int(m.group(1)))
        g.total = g.passed + g.failed + g.errors
    else:
        m = re.search(r"Ran (\d+) tests?", out)
        g.total = int(m.group(1)) if m else 0
        f = re.search(r"failures=(\d+)", out)
        e = re.search(r"errors=(\d+)", out)
        g.failed = int(f.group(1)) if f else 0
        g.errors = int(e.group(1)) if e else 0
        g.passed = max(0, g.total - g.failed - g.errors)
    if g.total == 0:
        g.problem = "the hidden tests collected nothing — check the task"
    return g


# --------------------------------------------------------------------------
# one task
# --------------------------------------------------------------------------

def _copy_into(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def run_task(task: BenchTask, folder: Path, llm: Any, *,
             config: dict | None = None, events: Any = None,
             ex: Any = None, work: Path | None = None) -> dict:
    """Run one task headless; grade it; copy the project into ``folder``;
    return its row.

    The task runs in ``work`` — the SAME folder for that task on every run
    (`<out>/_work/<task>`) — so the paths a traceback, a log or a prompt
    names are the same from run to run, and a replay's prompts match the
    recording's. It used to run inside the run's own folder, whose name has
    the time in it: every debug task's replay "diverged" at its first call.
    """
    from .filestorage import JsonFileStorage
    from .ports import (
        AutoApprove,
        Host,
        LocalFileSystem,
        SilentEvents,
        SubprocessExec,
    )
    from .session import Session, SessionConfig
    project = Path(work) if work is not None else Path(folder) / "project"
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    _copy_into(task.start, project)
    ex = ex or SubprocessExec()
    host = Host(llm=llm, fs=LocalFileSystem(str(project)), exec=ex,
                storage=JsonFileStorage(str(project / ".cc_state")),
                events=events or SilentEvents(), approval=AutoApprove())
    settings = {"lang": task.lang, "review_after_build": False,
                "mutation_check": False, "speed_probe": False,
                "wall_clock_s": task.timeout_s}
    settings.update(config or {})
    fields = set(SessionConfig.__dataclass_fields__)
    session = Session(host, config=SessionConfig(
        **{k: v for k, v in settings.items() if k in fields}))
    row: dict = {"task": task.name, "kind": task.kind}
    t0 = time.monotonic()
    try:
        if task.kind == "debug":
            from .debugrepair import DebugReport
            d = task.debug
            # a pasted traceback names the files where they WERE; the
            # task stores `{PROJECT}/` for the folder this run uses
            pasted = str(d.get("error_text", "")).replace(
                "{PROJECT}/", str(project) + os.sep)
            result = session.debug(DebugReport(
                error_text=pasted,
                command=str(d.get("command", "")),
                description=str(d.get("description", "")) or task.request,
                files=tuple(d.get("files", ()) or ())))
            row["state"] = getattr(result, "state", "")
            row["verdict"] = str(getattr(result, "verdict", ""))[:300]
        else:
            session.run(task.request)
    except Exception as exc:                             # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    row["seconds"] = round(time.monotonic() - t0, 1)
    try:
        final = session._final_outcomes()
    except Exception:                                    # noqa: BLE001
        final = []
    row["files"] = {"verified": sum(1 for o in final if o.verified),
                    "built": sum(1 for o in final if o.ok and not o.verified),
                    "failed": sum(1 for o in final if not o.ok)}
    row["stopped"] = [f"{o.path}: {o.stopped_because}"[:200]
                      for o in final if o.stopped_because][:4]
    row.update(_journal_counts(session))
    for attr in ("calls", "tokens_in", "tokens_out"):
        row[attr] = int(getattr(llm, attr, 0) or 0)
    if isinstance(llm, ReplayLLM):
        row["replay"] = {"diverged_at": llm.diverged_at,
                         "exhausted": llm.exhausted}
    g = grade(project, task.hidden, ex, timeout=min(600.0, task.timeout_s))
    row["hidden"] = {"passed": g.passed, "failed": g.failed,
                     "errors": g.errors, "total": g.total,
                     "runner": g.runner, "problem": g.problem}
    row["ok"] = g.ok and "error" not in row
    (Path(folder) / "hidden_tests.txt").write_text(g.output,
                                                   encoding="utf-8")
    if work is not None:
        # the project as the engine left it, kept with the run
        shutil.copytree(project, Path(folder) / "project",
                        dirs_exist_ok=True)
    return row


def _journal_counts(session: Any) -> dict:
    out = {"narrow_placed": 0, "narrow_unplaced": 0, "keep_refused": 0,
           "generations": 0, "fim_filled": 0, "fim_tried": 0,
           "tokens_out_total": 0, "decode_ms_total": 0, "resamples": 0}
    try:
        events = session.journal.events()
    except Exception:                                    # noqa: BLE001
        return out
    for e in events:
        kind = e.get("event")
        data = e.get("data") or {}
        if kind == "narrow":
            out["narrow_placed" if data.get("placed")
                else "narrow_unplaced"] += 1
        elif kind == "keep" and data.get("refused"):
            out["keep_refused"] += 1
        elif kind == "generate":
            out["generations"] += 1
            out["tokens_out_total"] += int(e.get("tokens_out") or 0)
            out["decode_ms_total"] += int(e.get("decode_ms") or 0)
        elif kind == "fim" and data.get("kind") == "repair":
            out["fim_tried"] += 1
            if data.get("kept") is not None:
                out["fim_filled"] += 1
        elif kind == "resample":
            out["resamples"] += 1
    return out


# --------------------------------------------------------------------------
# a run
# --------------------------------------------------------------------------

@dataclass
class BenchRun:
    path: Path
    rows: list
    problems: list
    previous: Path | None = None
    summary: str = ""


def _previous_run(out: Path, this: Path) -> Path | None:
    """The run finished most recently before this one — by when its results
    were written, not by name: two runs in one second sort by their labels
    ("after" < "before"), which is not the order they ran in."""
    if not out.is_dir():
        return None
    runs = [p for p in out.iterdir()
            if p.is_dir() and p != this and not p.name.startswith("_")
            and (p / "results.jsonl").exists()]
    if not runs:
        return None
    return max(runs, key=lambda p: ((p / "results.jsonl").stat().st_mtime_ns,
                                    p.name))


def read_rows(run: Path) -> list[dict]:
    path = Path(run) / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def run_suite(make_llm: Callable[[BenchTask, Path], Any], *,
              suite: Path | str = SUITE_DIR, out: Path | str = RUNS_DIR,
              only: Sequence[str] = (), label: str = "",
              config: dict | None = None,
              say: Callable[[str], None] = print,
              events: Any = None, ex: Any = None) -> BenchRun:
    """Every task, one after the other, into a new run folder.

    ``make_llm(task, folder)`` returns the LLMPort for one task — a
    `RecordingLLM` around the real model, or a `ReplayLLM` of an earlier
    run's replies for that task.
    """
    tasks, problems = load_suite(suite, only)
    out = Path(out)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # short: Windows paths stop at 260 characters, and a run folder holds
    # the project and its `.cc_state` below it
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)[:40].strip("-")
    name = f"{stamp}-{tag}" if tag else stamp
    run_dir = out / name
    n = 1
    while run_dir.exists():
        n += 1
        run_dir = out / f"{name}-{n}"
    run_dir.mkdir(parents=True)
    for p in problems:
        say(f"  suite: {p}")
    rows: list[dict] = []
    for i, task in enumerate(tasks, 1):
        folder = run_dir / task.name
        folder.mkdir(parents=True, exist_ok=True)
        say(f"[{i}/{len(tasks)}] {task.name} ({task.kind}) …")
        try:
            llm = make_llm(task, folder)
        except Exception as exc:                         # noqa: BLE001
            row = {"task": task.name, "kind": task.kind, "ok": False,
                   "error": f"no model for this task: {exc}"}
        else:
            row = run_task(task, folder, llm, config=config, events=events,
                           ex=ex, work=out / "_work" / task.name)
        rows.append(row)
        with (run_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        h = row.get("hidden", {})
        say(f"      {'PASS' if row.get('ok') else 'FAIL'}  hidden tests "
            f"{h.get('passed', 0)}/{h.get('total', 0)}  "
            f"{row.get('seconds', 0)}s  {row.get('calls', 0)} calls"
            + (f"  — {row['error']}" if row.get("error") else ""))
    previous = _previous_run(out, run_dir)
    summary = summarise(rows, read_rows(previous) if previous else [],
                        run_dir.name, previous.name if previous else "",
                        problems)
    (run_dir / "SUMMARY.md").write_text(summary, encoding="utf-8")
    return BenchRun(path=run_dir, rows=rows, problems=problems,
                    previous=previous, summary=summary)


def summarise(rows: Sequence[dict], before: Sequence[dict], name: str,
              before_name: str = "", problems: Sequence[str] = ()) -> str:
    """SUMMARY.md: the totals, a row per task, and what changed since the
    run before — task by task, never only the total."""
    passed = sum(1 for r in rows if r.get("ok"))
    tests_p = sum(r.get("hidden", {}).get("passed", 0) for r in rows)
    tests_t = sum(r.get("hidden", {}).get("total", 0) for r in rows)
    calls = sum(r.get("calls", 0) for r in rows)
    secs = sum(r.get("seconds", 0) for r in rows)
    lines = [f"# Cognitive Coder regression suite — {name}", "",
             f"**{passed} of {len(rows)} tasks passed** · hidden tests "
             f"{tests_p}/{tests_t} · {calls} model calls · {secs:.0f}s", ""]
    replays = [r for r in rows if r.get("replay")]
    if replays:
        diverged = [r["task"] for r in replays
                    if r["replay"].get("diverged_at")]
        lines += [f"Replay: {len(replays)} task(s) from recorded replies; "
                  + (f"the engine asked different questions in "
                     f"{', '.join(diverged)} — those numbers are not "
                     f"comparable with a live run."
                     if diverged else "every prompt matched the recording."),
                  ""]
    lines += ["| task | kind | result | hidden tests | files v/b/f | calls "
              "| seconds | narrowed | refused |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        h = r.get("hidden", {})
        f = r.get("files", {})
        lines.append(
            f"| {r['task']} | {r.get('kind', '')} | "
            f"{'PASS' if r.get('ok') else 'FAIL'} | "
            f"{h.get('passed', 0)}/{h.get('total', 0)} | "
            f"{f.get('verified', 0)}/{f.get('built', 0)}/{f.get('failed', 0)}"
            f" | {r.get('calls', 0)} | {r.get('seconds', 0)} | "
            f"{r.get('narrow_placed', 0)} | {r.get('keep_refused', 0)} |")
    if before:
        old = {r["task"]: r for r in before}
        better = [r["task"] for r in rows if r.get("ok")
                  and r["task"] in old and not old[r["task"]].get("ok")]
        worse = [r["task"] for r in rows if not r.get("ok")
                 and r["task"] in old and old[r["task"]].get("ok")]
        lines += ["", f"## Compared with {before_name}", "",
                  f"- better: {', '.join(better) or 'none'}",
                  f"- worse: {', '.join(worse) or 'none'}",
                  f"- tasks passed: {sum(1 for r in before if r.get('ok'))}"
                  f" → {passed}"]
    failing = [r for r in rows if not r.get("ok")]
    if failing:
        lines += ["", "## Why the failures failed", ""]
        for r in failing:
            why = (r.get("error") or r.get("hidden", {}).get("problem")
                   or "; ".join(r.get("stopped", [])[:2])
                   or "the hidden tests failed (see hidden_tests.txt)")
            lines.append(f"- **{r['task']}**: {why}")
    if problems:
        lines += ["", "## Suite problems", ""] + [f"- {p}" for p in problems]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# calibration: which edit format THIS model gets right (profiles.py)
# --------------------------------------------------------------------------

#: The formats a calibration runs the suite under, and the settings each
#: one is: whole files always; named definitions for every change; holes
#: first (which the session's probe may turn off — then the format is not
#: offered).
FORMATS: tuple[tuple[str, dict], ...] = (
    ("whole", {"narrow": "off", "fim": "off"}),
    ("symbols", {"narrow": "always", "fim": "off"}),
    ("fim", {"narrow": "auto", "fim": "auto"}),
)


def measure(rows: Sequence[dict]) -> dict:
    """What one run of the suite says about a format: tasks passed, hidden
    tests passed, the well-formed rate (generations that were placed and
    kept, over all generations), and tok/s where the journal timed it."""
    gens = sum(r.get("generations", 0) for r in rows)
    refused = sum(r.get("narrow_unplaced", 0) + r.get("keep_refused", 0)
                  for r in rows)
    tried = sum(r.get("fim_tried", 0) for r in rows)
    filled = sum(r.get("fim_filled", 0) for r in rows)
    tokens = sum(r.get("tokens_out_total", 0) for r in rows)
    ms = sum(r.get("decode_ms_total", 0) for r in rows)
    return {
        "tasks": len(rows),
        "passed": sum(1 for r in rows if r.get("ok")),
        "hidden_passed": sum(r.get("hidden", {}).get("passed", 0)
                             for r in rows),
        "hidden_total": sum(r.get("hidden", {}).get("total", 0)
                            for r in rows),
        "generations": gens,
        "well_formed": round(1.0 - refused / gens, 3) if gens else 0.0,
        "fim_tried": tried, "fim_filled": filled,
        "tok_s": round(tokens / (ms / 1000.0), 1) if ms > 0 else 0.0,
        "seconds": round(sum(r.get("seconds", 0) for r in rows), 1),
    }


def choose_format(measured: Mapping[str, Mapping[str, Any]]) -> str:
    """The format to write into the profile: most tasks passed, then most
    hidden tests, then the best well-formed rate, then the narrowest
    (fim > symbols > whole) — a tie goes to the one that writes least."""
    order = {"fim": 0, "symbols": 1, "whole": 2}
    # a format that passed nothing and generated nothing is no evidence —
    # a dead server measures every format at zero, and "narrowest wins"
    # would then pin `symbols` for every later build (the 2026-10-27
    # review); a format needs at least one generation to be judged
    def evidence(v: Mapping[str, Any]) -> bool:
        return bool(v.get("generations", 0) or v.get("passed", 0)
                    or v.get("hidden_passed", 0))
    usable = {k: v for k, v in measured.items()
              if evidence(v) and (k != "fim" or v.get("fim_filled", 0) > 0)}
    if not usable or not any(v.get("passed", 0) > 0
                             or v.get("hidden_passed", 0) > 0
                             for v in usable.values()):
        return "auto"
    best = max(usable.items(), key=lambda kv: (
        kv[1].get("passed", 0), kv[1].get("hidden_passed", 0),
        kv[1].get("well_formed", 0.0), -order.get(kv[0], 9)))
    return best[0]


def calibrate(make_llm: Callable[[BenchTask, Path], Any],
              model_name: str, *, suite: Path | str = SUITE_DIR,
              out: Path | str = RUNS_DIR, only: Sequence[str] = (),
              formats: Sequence[str] = (), config: dict | None = None,
              say: Callable[[str], None] = print, events: Any = None,
              ex: Any = None, profile_dir: Path | str | None = None) -> dict:
    """Run the suite once per edit format and write the model's profile
    (profiles.py) with the format it got right most often and what was
    measured. Returns the report (also written as `CALIBRATION.md` in the
    runs folder)."""
    from . import profiles
    wanted = [f for f, _c in FORMATS if not formats or f in formats]
    measured: dict[str, dict] = {}
    runs: dict[str, str] = {}
    for name in wanted:
        settings = dict(config or {})
        settings.update(dict(FORMATS)[name])
        say(f"== calibrating: {name} ==")
        run = run_suite(make_llm, suite=suite, out=out, only=only,
                        label=f"calibrate-{name}", config=settings, say=say,
                        events=events, ex=ex)
        measured[name] = measure(run.rows)
        runs[name] = run.path.name
    chosen = choose_format(measured)
    base = profiles.profile_for(model_name, directory=profile_dir)
    if base.source == "calibrated":
        base = profiles.builtin_for(model_name)
    report = {"model": model_name, "chosen": chosen, "measured": measured,
              "runs": runs, "profile": ""}
    if chosen == "auto":
        # nothing was measured (a dead server, every task errored): no
        # profile is written, and a stale one from an earlier calibration
        # is left as it was — said, never pinned
        say("calibration measured no passing generation in any format; "
            "no profile was written")
        report["problem"] = ("no format passed a task or a hidden test — "
                             "the model server may have been down; the "
                             "profile is unchanged")
    else:
        profile = profiles.replace(
            base, edit_format=chosen,
            measured={"formats": measured, "runs": runs,
                      "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "model": model_name})
        path = profiles.save_calibrated(model_name, profile,
                                        directory=profile_dir)
        report["profile"] = str(path)
    text = calibration_text(report)
    try:
        (Path(out) / "CALIBRATION.md").write_text(text, encoding="utf-8")
    except OSError:
        pass
    report["text"] = text
    return report


def calibration_text(report: Mapping[str, Any]) -> str:
    lines = [f"# Calibration — {report.get('model', '')}", "",
             "| format | tasks passed | hidden tests | well-formed | "
             "fim filled/tried | tok/s | seconds | run |",
             "|---|---|---|---|---|---|---|---|"]
    for name, m in (report.get("measured") or {}).items():
        lines.append(
            f"| {name} | {m.get('passed', 0)}/{m.get('tasks', 0)} | "
            f"{m.get('hidden_passed', 0)}/{m.get('hidden_total', 0)} | "
            f"{m.get('well_formed', 0.0):.0%} | "
            f"{m.get('fim_filled', 0)}/{m.get('fim_tried', 0)} | "
            f"{m.get('tok_s', 0.0)} | {m.get('seconds', 0)} | "
            f"{(report.get('runs') or {}).get(name, '')} |")
    lines += ["", f"**Chosen edit format: `{report.get('chosen')}`** — "
                  f"written to `{report.get('profile', '')}`; the next "
                  f"session on this model reads it before the built-in "
                  f"table.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# is the suite itself sound?
# --------------------------------------------------------------------------

def check_suite(suite: Path | str = SUITE_DIR, scratch: Path | str = "",
                only: Sequence[str] = (), ex: Any = None) -> list[str]:
    """Sentences for every task that does not prove what it should: its
    hidden tests must PASS on its reference solution, and — for a change
    or debug task — FAIL on its start files alone (else there is nothing
    to do and the task measures nothing). Empty: the suite is sound."""
    from .ports import SubprocessExec
    ex = ex or SubprocessExec()
    tasks, problems = load_suite(suite, only)
    base = Path(scratch) if scratch else RUNS_DIR / "_check"
    base.parent.mkdir(parents=True, exist_ok=True)
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    out = list(problems)
    for task in tasks:
        if not task.reference.is_dir():
            out.append(f"{task.name}: no reference/ solution to check the "
                       f"hidden tests against")
            continue
        good = base / task.name / "reference"
        good.mkdir(parents=True)
        _copy_into(task.start, good)
        _copy_into(task.reference, good)
        g = grade(good, task.hidden, ex)
        if not g.ok:
            out.append(f"{task.name}: the hidden tests do not pass on the "
                       f"reference ({g.passed}/{g.total}"
                       f"{'; ' + g.problem if g.problem else ''})")
        if task.kind in ("change", "debug"):
            bare = base / task.name / "start"
            bare.mkdir(parents=True)
            _copy_into(task.start, bare)
            g0 = grade(bare, task.hidden, ex)
            if g0.ok:
                out.append(f"{task.name}: the hidden tests already pass on "
                           f"start/ — the task asks for nothing")
    shutil.rmtree(base, ignore_errors=True)
    return out
