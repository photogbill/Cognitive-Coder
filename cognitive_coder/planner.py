# SPDX-License-Identifier: Apache-2.0
"""Skeleton-first decomposition — the replacement for the DAG-of-files plan.

WHY NOT THE OBVIOUS THING (§4.2). The tempting design asks the model for "a
strict Directed Acyclic Graph of executable files" as JSON, up front, for the
whole project. With a frontier model that works. With a small local model it
is the most likely point of total failure — and the reason is not the JSON.

At 24B the *format* risk is modest: Devstral emits valid JSON. The structural
argument holds anyway: **a compiling skeleton catches ARCHITECTURAL error,
which valid JSON does not.** A plan that is syntactically perfect and
architecturally wrong poisons every downstream step, and nothing later in the
loop can recover from it.

So, four steps instead:

  1. Ask for a **file list with one-line purposes** — small, cheap,
     constrained, and the thing small models are actually good at.
  2. Generate **stubs only** — signatures, imports, docstrings,
     `raise NotImplementedError`. Verify the whole skeleton *imports and
     compiles*. This catches architectural nonsense in seconds, before any
     real work.
  3. Fill bodies **one file at a time**, verifying after each.
  4. **Re-plan** after each file if the codemap says the shape changed.

The DAG survives as a *data structure* — dependency order is genuinely useful
for choosing what to build next — but it is **derived from the imports in the
skeleton** (deterministic, C5) rather than asserted by the model.

TWO THINGS THAT ARE NOT DECORATION:

  * **Every implementation task names its test file** (M39). Test-first (F2)
    is only mechanisable if the planner pairs `src/stats.py` with
    `tests/test_stats.py` at planning time. Tests are planned artefacts, not
    afterthoughts.
  * **`atomic` flows from the plan into the patcher transaction** (M25 rule
    1). The planner is the only component that knows whether two files are
    one change; nothing downstream can work it out.

And one rule about paths (D8, M38): **the model never chooses the path for an
existing file.** The planner assigns it and the model is told. For new files
the proposed path is validated against the project's observed layout before
it is accepted — a model that writes to `src/main.py` in a project that uses
`app/main.py` is confidently, invisibly wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import posixpath
import re
from typing import Any

from . import langs
from .codemap import parse_python, parse_regex
from .personas import CONTRACT_LIST, PERSONAS, PromptBuilder, strip_think
from .types import Plan, Task

# A file list longer than this is a request that should have been split.
MAX_FILES = 12


@dataclass
class Planner:
    """Turns a request into a `Plan`, then into a compiling skeleton."""

    host: Any
    codemap: Any = None
    journal: Any = None
    prompts: PromptBuilder = field(default_factory=PromptBuilder)
    lang: str = "python"
    src_dir: str = ""
    test_dir: str = ""

    # ------------------------------------------------------------------
    def plan(self, request: str, profile: dict | None = None) -> Plan:
        """Ask for a file list. Small, cheap, and constrained (step 1)."""
        self.prompts.profile = dict(profile or self.prompts.profile)
        layout = self.observe_layout()
        persona = PERSONAS["planner"]
        task_text = _plan_prompt(request, self.lang, layout)
        prompt = self.prompts.build(persona, task_text,
                                    architecture=self._architecture(),
                                    contract=CONTRACT_LIST)
        completion = self.host.llm.complete(
            prompt.messages(), temperature=persona.temperature,
            max_tokens=700)
        rows = _parse_file_list(strip_think(completion.text))

        if not rows:
            # A model that returns nothing usable does not stop the session:
            # a single-file plan is a real plan, and it is far better than an
            # exception in front of the operator (C6).
            rows = [(self._default_path(request), request.strip()[:120]
                     or "the whole request in one file")]

        tasks = self._to_tasks(rows, layout)
        plan = Plan(request=request, tasks=tuple(tasks),
                    layout_note=layout.get("note", ""),
                    caveats=() if len(rows) <= MAX_FILES else (
                        f"the model proposed {len(rows)} files; only the "
                        f"first {MAX_FILES} were kept",))
        if self.journal is not None:
            self.journal.log("plan", request=request,
                             files=[t.path for t in tasks],
                             tests=[t.test_path for t in tasks if t.test_path])
        self.host.emit("status",
                       f"plan: {len(tasks)} file(s) proposed",
                       {"files": [t.path for t in tasks]})
        return plan

    # ------------------------------------------------------------------
    def _to_tasks(self, rows: Sequence[tuple[str, str]],
                  layout: dict) -> list[Task]:
        tasks: list[Task] = []
        for i, (path, purpose) in enumerate(rows[:MAX_FILES]):
            path = self.validate_path(path, layout)
            lang_id = langs.id_for_path(path) or self.lang
            is_test = _looks_like_test(path)
            tasks.append(Task(
                id=f"t{i + 1}", path=path, purpose=purpose,
                test_path="" if is_test else self.test_path_for(path, layout),
                persona="tester" if is_test else "engineer",
                lang=lang_id, atomic=False))
        return tasks

    def validate_path(self, path: str, layout: dict) -> str:
        """Accept the model's path only if it fits the project (D8, M38).

        For a file that already exists the answer is not negotiable — the
        existing path wins. For a new file, a proposal that ignores the
        project's observed layout is corrected, not honoured.
        """
        clean = str(path or "").strip().strip("`'\"").replace("\\", "/")
        clean = posixpath.normpath(clean).lstrip("./")
        if not clean or clean.startswith(".."):
            return self._default_path("file")
        if self.host.fs.exists(clean):
            return clean
        src = layout.get("src_dir", "")
        if src and "/" not in clean:
            # The project keeps sources in a directory and the model
            # forgot — that is the confident-wrong-path failure, corrected
            # here rather than discovered by a failing import.
            return f"{src}/{clean}"
        return clean

    def test_path_for(self, path: str, layout: dict | None = None) -> str:
        """The test file this implementation is paired with (M39).

        Derived from the project's OBSERVED convention where there is one,
        because a project with `tests/test_x.py` everywhere and one
        `src/x_test.py` has answered the question already.
        """
        layout = layout or self.observe_layout()
        lang = langs.get(langs.id_for_path(path) or self.lang)
        ext = lang.ext if lang else ".py"
        stem = path.replace("\\", "/").rsplit("/", 1)[-1]
        stem = stem[:-len(ext)] if stem.endswith(ext) else stem
        pattern = layout.get("test_pattern", "")
        tdir = layout.get("test_dir") or self.test_dir or "tests"
        if pattern == "suffix":
            folder = path.rsplit("/", 1)[0] if "/" in path else ""
            return f"{folder}/{stem}_test{ext}" if folder \
                else f"{stem}_test{ext}"
        return f"{tdir}/test_{stem}{ext}"

    def observe_layout(self) -> dict:
        """What this project's layout actually IS, not what it should be.

        Observation rather than configuration: the answer is sitting in the
        file list, and asking the operator to configure something the tool
        can see is a question that should not be asked.
        """
        try:
            paths = [p.replace("\\", "/") for p in self.host.fs.list("*")]
        except Exception:                                # noqa: BLE001
            paths = []
        code = [p for p in paths if langs.id_for_path(p)]
        tests = [p for p in code if _looks_like_test(p)]
        dirs: dict[str, int] = {}
        for p in code:
            if "/" in p and not _looks_like_test(p):
                dirs[p.split("/", 1)[0]] = dirs.get(p.split("/", 1)[0], 0) + 1
        src_dir = max(dirs, key=lambda k: dirs[k]) if dirs else \
            (self.src_dir or "")
        test_dirs: dict[str, int] = {}
        for p in tests:
            if "/" in p:
                test_dirs[p.split("/", 1)[0]] = \
                    test_dirs.get(p.split("/", 1)[0], 0) + 1
        test_dir = max(test_dirs, key=lambda k: test_dirs[k]) \
            if test_dirs else (self.test_dir or "")
        pattern = "suffix" if any(
            p.rsplit("/", 1)[-1].split(".")[0].endswith("_test")
            for p in tests) else "prefix"
        note = ""
        if src_dir or test_dir:
            note = (f"this project keeps sources in "
                    f"`{src_dir or 'the root'}` and tests in "
                    f"`{test_dir or 'tests'}`")
        return {"src_dir": src_dir, "test_dir": test_dir,
                "test_pattern": pattern, "files": code, "note": note}

    def _default_path(self, request: str) -> str:
        lang = langs.get(self.lang)
        ext = lang.ext if lang else ".py"
        stem = re.sub(r"\W+", "_", (request or "main").strip().lower())[:24]
        folder = (self.src_dir + "/") if self.src_dir else ""
        return f"{folder}{stem or 'main'}{ext}"

    def _architecture(self) -> str:
        if self.codemap is None:
            return ""
        return self.codemap.prefix_block()

    # ------------------------------------------------------------------
    # step 2: the skeleton
    # ------------------------------------------------------------------
    def skeleton(self, plan: Plan) -> dict:
        """Stubs only, then verify the whole thing compiles (step 2).

        This is the step that earns the whole design. Signatures, imports,
        docstrings, `raise NotImplementedError` — no bodies. If the skeleton
        does not import, the ARCHITECTURE is wrong, and finding that out in
        seconds beats finding it out after four files of real work.
        """
        written: list[str] = []
        for task in plan.tasks:
            if _looks_like_test(task.path):
                continue
            stub = self.stub_for(task, plan)
            self.host.fs.write(task.path, stub)
            written.append(task.path)
            if self.codemap is not None:
                self.codemap.reindex_after_write(task.path)

        ok, note = self.verify_skeleton(written)
        if self.journal is not None:
            self.journal.log("skeleton", files=written, ok=ok, note=note)
        self.host.emit("phase" if ok else "warning",
                       f"skeleton: {note}",
                       {"phase": "skeleton", "files": written, "ok": ok})
        return {"ok": ok, "files": written, "note": note}

    def stub_for(self, task: Task, plan: Plan) -> str:
        """A stub that compiles. Written deterministically where possible.

        For Python the stub is generated by rule rather than by the model:
        it is a mechanical transformation of the file's purpose, and asking
        a model to produce something a rule can produce is C5 backwards.
        """
        lang_id = task.lang or self.lang
        if lang_id == "python":
            imports = [f"from {_module(t.path)} import *"
                       for t in plan.tasks
                       if t.id in task.depends_on]
            body = [f'"""{task.purpose}"""', ""]
            body += imports + ([""] if imports else [])
            body += ["", "def main() -> int:",
                     f'    """{task.purpose}"""',
                     "    raise NotImplementedError(",
                     f'        "{_module(task.path)}.main is not written '
                     f'yet")', ""]
            return "\n".join(body)
        scaffold = langs.scaffold_for(lang_id, task.purpose[:40] or "module",
                                      _stem(task.path))
        return scaffold or f"{_comment(lang_id)} {task.purpose}\n"

    def verify_skeleton(self, paths: Sequence[str]) -> tuple[bool, str]:
        """Does the skeleton import/compile? Seconds, not minutes.

        Deliberately a SYNTAX-and-imports check, not a build: the point is to
        catch architectural nonsense — a file importing something no file
        provides — before real work starts. C4 is not in play here, because
        nothing is being claimed as done.
        """
        from . import runner
        broken: list[str] = []
        for path in paths:
            lang_id = langs.id_for_path(path) or self.lang
            try:
                text = self.host.fs.read(path)
            except Exception:                            # noqa: BLE001
                continue
            phase = runner.syntax_check(text, lang_id, ex=self.host.exec,
                                        cwd=self.host.fs.root(),
                                        src_path=path)
            if phase is not None and not phase.ok:
                broken.append(f"{path} ({phase.output.splitlines()[0][:70]})"
                              if phase.output else path)
        if broken:
            return False, ("the skeleton does not compile: "
                           + "; ".join(broken[:3]))
        unresolved = self._skeleton_imports(paths)
        if unresolved:
            return False, (f"the skeleton imports things nothing provides: "
                           f"{', '.join(unresolved[:4])} — the file split is "
                           f"probably wrong")
        return True, (f"stubs written, imports resolved, {len(paths)} file(s) "
                      f"compile")

    def _skeleton_imports(self, paths: Sequence[str]) -> list[str]:
        """Imports of PROJECT modules that no planned file provides.

        This is the check that makes step 2 worth doing. A skeleton where
        `cli.py` imports `from stats import summarise` and no file provides
        `stats` is architecturally wrong, and it is wrong NOW — in seconds,
        before any real generation — rather than after four files of work.

        Only project-shaped imports count. A missing third-party package is a
        real problem but a different one, and not the planner's to diagnose:
        the build will say so in its own words, with a better message than
        anything guessable from here.
        """
        provided: set[str] = set()
        for p in paths:
            module = _module(p)
            provided.add(module)
            provided.add(module.split(".")[-1])
            # A package directory provides its own name: `src/stats.py`
            # means `src` is importable too.
            head = p.replace("\\", "/").split("/")[0]
            if head and "." not in head:
                provided.add(head)

        missing: list[str] = []
        for path in paths:
            lang_id = langs.id_for_path(path) or self.lang
            try:
                text = self.host.fs.read(path)
            except Exception:                            # noqa: BLE001
                continue
            names = (parse_python.imports_of(text) if lang_id == "python"
                     else parse_regex.imports_of(text, lang_id))
            for raw in names:
                name = str(raw).lstrip(".")
                if not name:
                    continue
                head = name.split(".")[0]
                if head in provided or name in provided:
                    continue
                if head in _STDLIB_ISH or name in _STDLIB_ISH:
                    continue
                if self.codemap is not None and self.codemap.resolves(name):
                    continue
                # Anything not clearly ours is assumed to be a real package.
                # Being wrong in this direction costs nothing; being wrong in
                # the other direction blocks a legitimate plan.
                if not self._looks_local(name, paths):
                    continue
                if name not in missing:
                    missing.append(name)
        return missing

    @staticmethod
    def _looks_local(name: str, paths: Sequence[str]) -> bool:
        """Does this import name look like it means a file in this project?

        Heuristic and deliberately narrow: a dotted name whose head matches a
        directory in the plan, or a bare name matching a planned file's stem
        with different spelling. Everything else is assumed to be a package
        somebody installed.
        """
        head = name.split(".")[0]
        folders = {p.replace("\\", "/").split("/")[0] for p in paths
                   if "/" in p.replace("\\", "/")}
        return head in folders

    # ------------------------------------------------------------------
    # step 3 support: dependency order, DERIVED (C5)
    # ------------------------------------------------------------------
    def derive_order(self, plan: Plan) -> Plan:
        """Dependency order from the skeleton's IMPORTS, not the model's word.

        This is §4.2's compromise kept honestly: the DAG is useful, so it is
        computed — from what the files actually import, which is a fact,
        rather than from what the model asserted, which is a claim.
        """
        by_module = {_module(t.path): t.id for t in plan.tasks}
        depends: dict[str, set] = {t.id: set() for t in plan.tasks}
        for task in plan.tasks:
            try:
                text = self.host.fs.read(task.path)
            except Exception:                            # noqa: BLE001
                continue
            lang_id = task.lang or self.lang
            names = (parse_python.imports_of(text) if lang_id == "python"
                     else parse_regex.imports_of(text, lang_id))
            for name in names:
                target = by_module.get(_module(str(name)))
                if target and target != task.id:
                    depends[task.id].add(target)

        ordered = _topological(plan.tasks, depends)
        tasks = tuple(
            Task(id=t.id, path=t.path, purpose=t.purpose,
                 test_path=t.test_path, persona=t.persona,
                 depends_on=tuple(sorted(depends.get(t.id, ()))),
                 atomic=t.atomic, lang=t.lang, status=t.status,
                 attempts=t.attempts)
            for t in ordered)
        return Plan(request=plan.request, tasks=tasks,
                    layout_note=plan.layout_note, caveats=plan.caveats)

    # ------------------------------------------------------------------
    # step 4: replan
    # ------------------------------------------------------------------
    def replan(self, plan: Plan, *, reason: str = "") -> Plan:
        """Revise the remaining tasks when the shape has changed.

        **A plan that cannot change is a plan that will be wrong by file
        five.** A replan is an epoch boundary (G.7.2) — the caller bumps it,
        because the caller owns the codemap's cache lifecycle.
        """
        remaining = [t for t in plan.tasks if t.status == "pending"]
        if not remaining or self.codemap is None:
            return plan
        revised: list[Task] = []
        for task in plan.tasks:
            if task.status != "pending":
                revised.append(task)
                continue
            if self.host.fs.exists(task.path) and _has_body(
                    self.host.fs, task.path, task.lang or self.lang):
                # Somebody already wrote it — a real occurrence when one file
                # legitimately implements two planned responsibilities.
                revised.append(task.with_status("done"))
                continue
            revised.append(task)
        if self.journal is not None:
            self.journal.log("plan", replan=True, reason=reason,
                             remaining=[t.path for t in revised
                                        if t.status == "pending"])
        return Plan(request=plan.request, tasks=tuple(revised),
                    layout_note=plan.layout_note, caveats=plan.caveats)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_STDLIB_ISH = {
    "os", "sys", "re", "json", "csv", "math", "time", "typing", "pathlib",
    "dataclasses", "collections", "itertools", "functools", "argparse",
    "unittest", "sqlite3", "hashlib", "logging", "abc", "enum", "io",
    "subprocess", "textwrap", "datetime", "random", "statistics", "string",
    "std", "core", "alloc", "fmt", "iostream", "vector", "string.h",
    "stdio.h", "stdlib.h", "node", "assert", "pytest",
}

_LINE = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?"
    r"[`'\"]?(?P<path>[\w./\\-]+\.\w+)[`'\"]?"
    r"\s*(?:[-—:–]|\s)\s*(?P<purpose>.+?)\s*$", re.M)


def _parse_file_list(text: str) -> list[tuple[str, str]]:
    """`path — purpose` per line, forgiving about the separator (D5/D9).

    Permissive about form, strict about what it accepts as a path: a bullet,
    a number, a backtick or an em-dash instead of a hyphen are all fine; a
    "path" with no extension is not a path.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _LINE.finditer(text or ""):
        path = m.group("path").strip()
        purpose = m.group("purpose").strip(" -—:–\t")
        if not purpose or path in seen:
            continue
        if not langs.id_for_path(path):
            continue
        seen.add(path)
        rows.append((path, purpose[:200]))
    return rows


def _plan_prompt(request: str, lang: str, layout: dict) -> str:
    lang_obj = langs.get(lang)
    lines = [
        "Break this request into the smallest set of files that delivers it.",
        "",
        f"REQUEST: {request}",
        "",
        f"Language: {lang_obj.label if lang_obj else lang}",
    ]
    if layout.get("note"):
        lines.append(f"Layout: {layout['note']} — follow it.")
    if layout.get("files"):
        shown = ", ".join(layout["files"][:12])
        lines.append(f"Files that already exist: {shown}")
    lines += [
        "",
        "Rules: three to five files is usually right. Do not invent "
        "structure the request does not need. Do not list test files — they "
        "are paired automatically.",
    ]
    return "\n".join(lines)


def _looks_like_test(path: str) -> bool:
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1].lower()
    stem = name.split(".")[0]
    return (stem.startswith("test_") or stem.endswith("_test")
            or stem.endswith("test") and len(stem) > 4
            or "/test" in str(path).replace("\\", "/").lower())


def _module(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    for ext in (".py", ".pyw"):
        if p.endswith(ext):
            p = p[:-len(ext)]
    if p.endswith("/__init__"):
        p = p[:-len("/__init__")]
    return p.strip("/").replace("/", ".")


def _stem(path: str) -> str:
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _comment(lang_id: str) -> str:
    lang = langs.get(lang_id)
    return lang.comment if lang else "#"


def _has_body(fs: Any, path: str, lang_id: str) -> bool:
    """Is this a real implementation or still a stub?"""
    try:
        text = fs.read(path)
    except Exception:                                    # noqa: BLE001
        return False
    return bool(text.strip()) and "NotImplementedError" not in text


def _topological(tasks: Sequence[Task],
                 depends: dict[str, set]) -> list[Task]:
    """Dependency order, stable, and tolerant of a cycle.

    A cycle in a derived graph means two files import each other, which is a
    real thing that happens and is not the planner's business to refuse. The
    remaining tasks are appended in their original order rather than dropped.
    """
    by_id = {t.id: t for t in tasks}
    done: list[Task] = []
    placed: set[str] = set()
    changed = True
    while changed:
        changed = False
        for t in tasks:
            if t.id in placed:
                continue
            if depends.get(t.id, set()) <= placed:
                done.append(t)
                placed.add(t.id)
                changed = True
    for t in tasks:
        if t.id not in placed:
            done.append(by_id[t.id])
    return done
