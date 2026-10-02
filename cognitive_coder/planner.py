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
from .types import Edit, Plan, Task

# A file list longer than this is a request that should have been split.
MAX_FILES = 12

#: WRITTEN INTO EVERY STUB, in the language's own comment syntax, so "is this
#: still a stub?" has one answer in every language.
#:
#: It used to be answered by looking for `NotImplementedError`, which only
#: the Python stub contains. A JavaScript scaffold — a function with a
#: `console.log` in it — read as finished work, and after the first file
#: `replan` marked every remaining task done: a three-file plan built one
#: file and reported nothing wrong.
STUB_SENTINEL = "cc-stub:"
_SENTINEL_TEXT = (f"{STUB_SENTINEL} written by the skeleton; replaced when "
                  f"this file is built")


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
    #: HOW MANY FILES ONE PLAN MAY CONTAIN.
    #:
    #: A cap is right — a model asked for "a web framework" will happily
    #: propose sixty files and finish none of them — but 12 was a constant
    #: nobody could reach, and on a large written specification it is the
    #: binding limit long before tokens are. Twelve is a reasonable default
    #: for a request typed in a sentence and an arbitrary one for a
    #: four-section design document.
    #:
    #: Truncation is reported as a caveat and always was; what changes is
    #: that the operator can now raise it instead of rewriting the request
    #: to fit a number they were never told about.
    max_files: int = MAX_FILES
    #: The skeleton's writes go through this, as ONE transaction: snapshotted,
    #: approved once, undoable. None only for a planner used on its own (the
    #: tests do), which falls back to writing directly.
    patcher: Any = None
    #: Paths that already had real work in them when the skeleton ran. They
    #: were NOT stubbed, and `replan` must not mistake their body for this
    #: session's work and mark their task done.
    kept_as_found: set = field(default_factory=set)
    #: Add a tester task for every source file whose test file the plan
    #: lacks (M39: tests are planned artefacts). The planner prompt has always
    #: said "Do not list test files — they are paired automatically", and
    #: nothing paired them. Off by default for one reason, stated plainly: it
    #: adds a generation per source file, and the committed golden trace and
    #: several host test suites script their replies for the unpaired plan.
    #: With it off, a plan no longer PROMISES a test file nothing will write.
    pair_tests: bool = False

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
        rows, not_code = _parse_file_list_with_skipped(
            strip_think(completion.text))

        if not rows:
            # A model that returns nothing usable does not stop the session:
            # a single-file plan is a real plan, and it is far better than an
            # exception in front of the operator (C6).
            rows = [(self._default_path(request), request.strip()[:120]
                     or "the whole request in one file")]

        tasks = self._to_tasks(rows, layout)

        #: THE REQUEST NAMED TEST FILES. THE PLAN MUST CONTAIN THEM.
        #: See _required_tests for what this cost when it was missing.
        tasks, added = self._ensure_required_tests(request, tasks, layout)
        unpaired: list[str] = []
        if self.pair_tests:
            tasks, unpaired = self._pair_tests(tasks)
        tasks = self._unpromise_missing_tests(tasks)

        caveats: list[str] = []
        if unpaired:
            caveats.append(
                f"no room to pair {', '.join(unpaired)} with "
                f"{'a test' if len(unpaired) == 1 else 'tests'} within the "
                f"plan's limit of {self.max_files} files — "
                f"{'it' if len(unpaired) == 1 else 'they'} will be built and "
                f"run, but no test will check "
                f"{'it' if len(unpaired) == 1 else 'them'}")
        if len(rows) > self.max_files:
            caveats.append(f"the model proposed {len(rows)} files; only the "
                           f"first {self.max_files} were kept — raise the "
                           f"plan file limit if the request really needs "
                           f"that many")
            self.host.emit("warning",
                           f"the plan was truncated at {self.max_files} "
                           f"files; {len(rows) - self.max_files} more were "
                           f"proposed and dropped",
                           {"phase": "plan", "proposed": len(rows)})
        if added:
            caveats.append(
                f"the request asked for {len(added)} test file(s) that the "
                f"plan left out: {', '.join(added)} — they were added, "
                f"because a build that skips the tests it was told to write "
                f"cannot report whether it worked")
        planned = {t.path.replace("\\", "/").lower() for t in tasks}
        unplanned = [p for p in _required_tests(request)
                     if p.lower() not in planned]
        if unplanned:
            caveats.append(
                f"the request named {', '.join(unplanned)}, but the plan "
                f"was already at its limit of {self.max_files} files, so "
                f"{'it was' if len(unplanned) == 1 else 'they were'} NOT "
                f"added — nothing will check what "
                f"{'it' if len(unplanned) == 1 else 'they'} would have "
                f"checked. Raise the plan file limit to include "
                f"{'it' if len(unplanned) == 1 else 'them'}")
        if not_code:
            caveats.append(
                f"the model also proposed {', '.join(not_code)}, which no "
                f"language this engine builds owns — left out of the plan; "
                f"write {'it' if len(not_code) == 1 else 'them'} by hand if "
                f"the request needs {'it' if len(not_code) == 1 else 'them'}")
            self.host.emit("warning",
                           f"the plan left out {', '.join(not_code)}: this "
                           f"engine only writes source files",
                           {"phase": "plan", "not_code": not_code})
        plan = Plan(request=request, tasks=tuple(tasks),
                    layout_note=layout.get("note", ""),
                    caveats=tuple(caveats))
        if self.journal is not None:
            #: `tasks` is what resume rebuilds the plan FROM. The event used
            #: to carry paths only, and resume invented the rest: every task
            #: became "(resumed) part of: …", an engineer, in the session's
            #: language, paired with a test of its own — so a test lost its
            #: tester and `beta.js` became Python. `files` stays for readers
            #: of older journals and for `resume_state`.
            self.journal.log("plan", request=request,
                             files=[t.path for t in tasks],
                             tests=[t.test_path for t in tasks if t.test_path],
                             tasks=[{"path": t.path, "purpose": t.purpose,
                                     "persona": t.persona, "lang": t.lang,
                                     "test_path": t.test_path,
                                     "atomic": t.atomic} for t in tasks])
        self.host.emit("status",
                       f"plan: {len(tasks)} file(s) proposed",
                       {"files": [t.path for t in tasks]})
        return plan

    # ------------------------------------------------------------------
    def _ensure_required_tests(self, request: str, tasks: list[Task],
                               layout: dict) -> tuple[list[Task], list[str]]:
        """Add test files the REQUEST named and the plan left out.

        WHY THIS EXISTS, 2026-08-07
        ---------------------------
        A specification arrived with a section headed **"Testing Requirements
        (Strict)"**. It named ``tests/test_math3d.py`` and
        ``tests/test_physics.py``, and required that tests be written before
        the module bodies.

        Two models were given that specification. Both proposed five files,
        all under ``src/``. Neither proposed a single test. Nothing noticed.

        The consequence ran through the entire build. Every file reported::

            the test command succeeded but ran ZERO tests — that is not
            evidence the code works, only that nothing contradicted it

        which was true, and was printed about ten times, and was describing a
        symptom whose cause was three steps upstream. With no tests,
        "verifying" degraded to "the file imports". Three files were committed
        green. One of them defined ``update(self, dt, accel, brake, steer,
        curve)`` while the caller passed four arguments; the program died on
        its first frame. The test the specification asked for would have
        caught it in under a second.

        A planner is allowed to choose the shape of a solution. It is not
        allowed to silently drop a requirement it was handed. So: paths that
        look like test files are extracted from the request, and any the plan
        omitted are added — as a CAVEAT, visible in the plan, not as a silent
        correction.

        Only EXPLICIT paths are honoured. "Please write tests" is a wish and
        this does not try to interpret it; ``tests/test_math3d.py`` is an
        instruction, and this holds the plan to it.
        """
        wanted = _required_tests(request)
        if not wanted:
            return tasks, []
        have = {t.path.replace("\\", "/").lower() for t in tasks}
        added: list[str] = []
        no_room: list[str] = []
        out = list(tasks)
        for path in wanted:
            if path.lower() in have:
                continue
            if len(out) >= self.max_files:
                # Skipped silently, once — the exact omission this method
                # exists to prevent. Said out loud now; `plan()` also puts
                # it in the plan's caveats.
                no_room.append(path)
                continue
            lang_id = langs.id_for_path(path) or self.lang
            out.append(Task(
                id=f"t{len(out) + 1}", path=path,
                purpose=f"tests required by the request for "
                        f"{_stem(path).removeprefix('test_')}",
                test_path="", persona="tester", lang=lang_id, atomic=False))
            added.append(path)

        #: Reported HERE, beside the decision, rather than by the caller.
        #: A silent correction is still a correction the operator did not
        #: ask for and cannot audit — and this particular omission is one
        #: nobody noticed for a whole build, so it gets said out loud at the
        #: moment it is detected.
        if added:
            self.host.emit("warning",
                           f"the plan omitted {len(added)} test file(s) named "
                           f"in the request — added: {', '.join(added)}",
                           {"phase": "plan", "added_tests": added})
        if no_room:
            self.host.emit("warning",
                           f"the request named {', '.join(no_room)}, but the "
                           f"plan is at its limit of {self.max_files} files "
                           f"— NOT added",
                           {"phase": "plan", "tests_not_added": no_room})
        return out, added

    def _pair_tests(self, tasks: list[Task]) -> tuple[list[Task], list[str]]:
        """(tasks with a tester task per unpaired source file, left out).

        Only where the test file neither exists nor is planned; placed after
        its module by `derive_order`'s test-follows-its-module rule.
        """
        planned = {t.path.replace("\\", "/").lower() for t in tasks}
        out = list(tasks)
        unpaired: list[str] = []
        for task in tasks:
            if not task.test_path or task.persona == "tester":
                continue
            key = task.test_path.replace("\\", "/").lower()
            if key in planned or self.host.fs.exists(task.test_path):
                continue
            if len(out) >= self.max_files:
                unpaired.append(task.path)
                continue
            out.append(Task(
                id=f"t{len(out) + 1}", path=task.test_path,
                purpose=f"tests for {task.path}: {task.purpose}",
                test_path="", persona="tester",
                lang=langs.id_for_path(task.test_path) or task.lang,
                atomic=False))
            planned.add(key)
        return out, unpaired

    def _unpromise_missing_tests(self, tasks: list[Task]) -> list[Task]:
        """Clear a `test_path` that nothing will ever write.

        The engineer was told "Its tests live in tests/test_alpha.py and must
        pass" about a file that was not planned and did not exist — so the
        claim was false, and a file "verified" under it had been tested by
        nothing. A test path is kept only when it is planned or on disk.
        """
        planned = {t.path.replace("\\", "/").lower() for t in tasks}
        out = []
        for t in tasks:
            if t.test_path and t.test_path.replace("\\", "/").lower() \
                    not in planned and not self.host.fs.exists(t.test_path):
                t = Task(id=t.id, path=t.path, purpose=t.purpose,
                         test_path="", persona=t.persona,
                         depends_on=t.depends_on, atomic=t.atomic,
                         lang=t.lang, status=t.status, attempts=t.attempts)
            out.append(t)
        return out

    # ------------------------------------------------------------------
    def _to_tasks(self, rows: Sequence[tuple[str, str]],
                  layout: dict) -> list[Task]:
        tasks: list[Task] = []
        for i, (path, purpose) in enumerate(rows[:self.max_files]):
            path = self.validate_path(path, layout)
            lang_id = langs.id_for_path(path) or self.lang
            is_test = _looks_like_test(path)
            unpaired = is_test or _in_test_dir(path)
            tasks.append(Task(
                id=f"t{i + 1}", path=path, purpose=purpose,
                test_path="" if unpaired else self.test_path_for(path, layout),
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
            if "/" in p and not (_looks_like_test(p) or _in_test_dir(p)):
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

        IT IS A WRITE TO SOMEONE'S PROJECT, AND OBEYS THE RULES FOR ONE.
        This used to call `host.fs.write` for every non-test task,
        unconditionally. A plan that named an existing file — "extend
        src/util.py" — replaced the hand-written file with a stub, with no
        snapshot, no approval and no undo; and under the library default
        `DenyAll` the stubs and `src/__init__.py` were written anyway while
        the CLI said "nothing was written". So now:

          * a file that already has real work in it is KEPT AS FOUND: no
            stub over it, its path stays in the plan, and the caller says so;
          * every stub and package file goes through ONE patcher
            transaction, approved once with the whole diff in view,
            snapshotted, and undoable like any other change;
          * refused approval means nothing is written, and the result says
            that rather than reporting a skeleton that does not exist.
        """
        stubs: list[Edit] = []
        kept: list[str] = []
        for task in plan.tasks:
            if _looks_like_test(task.path):
                continue
            if self._has_real_work(task.path):
                kept.append(task.path)
                continue
            stubs.append(Edit(path=task.path, kind="whole",
                              new=self.stub_for(task, plan),
                              note="skeleton stub"))
        self.kept_as_found = set(kept)
        if kept:
            self.host.emit("warning",
                           f"skeleton: {', '.join(kept)} already "
                           f"{'has' if len(kept) == 1 else 'have'} real "
                           f"work in {'it' if len(kept) == 1 else 'them'}, "
                           f"so no stub was written over "
                           f"{'it' if len(kept) == 1 else 'them'}",
                           {"phase": "skeleton", "kept": kept})

        if self.patcher is None:
            written, approved = self._write_directly(stubs, plan)
        else:
            written, approved = self._write_in_transaction(
                stubs + self._package_edits(plan))

        if not approved:
            note = ("the skeleton was not approved, so nothing was written "
                    "— no stubs and no package files")
            if self.journal is not None:
                self.journal.log("skeleton", files=[], ok=True, note=note,
                                 approved=False, kept=kept)
            self.host.emit("warning", f"skeleton: {note}",
                           {"phase": "skeleton", "files": [], "ok": True,
                            "approved": False})
            return {"ok": True, "files": [], "note": note,
                    "approved": False, "kept": kept}

        stub_paths = [e.path for e in stubs]
        written = [p for p in written if p in stub_paths]
        if self.codemap is not None:
            for path in written:
                self.codemap.reindex_after_write(path)

        ok, note = self.verify_skeleton(written)
        if self.journal is not None:
            self.journal.log("skeleton", files=written, ok=ok, note=note,
                             kept=kept)
        self.host.emit("phase" if ok else "warning",
                       f"skeleton: {note}",
                       {"phase": "skeleton", "files": written, "ok": ok})
        return {"ok": ok, "files": written, "note": note, "approved": True,
                "kept": kept}

    def _has_real_work(self, path: str) -> bool:
        """Does this path already hold something a stub must not replace?

        Existing, non-empty, and not a stub this engine wrote. Deliberately
        NOT "has no NotImplementedError": an abstract base class raises it
        on purpose, and that is exactly the kind of hand-written file a stub
        would destroy.
        """
        try:
            if not self.host.fs.exists(path):
                return False
            text = self.host.fs.read(path)
        except Exception:                                # noqa: BLE001
            return False
        return bool(text.strip()) and not _is_our_stub(text)

    def _write_in_transaction(self, edits: Sequence[Edit]
                              ) -> tuple[list[str], bool]:
        """(paths written, approved?) — one transaction, one approval."""
        if not edits:
            return [], True
        parts = []
        for edit in edits:
            if not edit.new and not self.host.fs.exists(edit.path):
                # A preview has nothing to show for an empty new file, and
                # "no change" beside a file about to be created misleads the
                # person approving it.
                parts.append(f"# {edit.path}: new, empty — makes the folder "
                             f"an importable package")
            else:
                parts.append(self.patcher.preview([edit]))
        summary = (f"skeleton: {len(edits)} stub file"
                   f"{'s' * (len(edits) != 1)}")
        try:
            approved = bool(self.patcher.approval.approve_diff(
                summary, "\n".join(parts)))
        except Exception:                                # noqa: BLE001
            approved = False
        if not approved:
            return [], False
        tx = self.patcher.begin("skeleton", atomic=False)
        try:
            results = tx.apply(edits, approve=False, summary=summary)
        except Exception:
            tx.rollback("the skeleton could not be written")
            raise
        tx.commit(verified=False)        # a stub is written, never verified
        return [r.path for r in results
                if r.ok or "no change" in (r.reason or "")], True

    def _write_directly(self, stubs: Sequence[Edit],
                        plan: Plan) -> tuple[list[str], bool]:
        """The planner used on its own, with no patcher: the old path."""
        written = []
        for edit in stubs:
            self.host.fs.write(edit.path, edit.new)
            written.append(edit.path)
        self._make_packages(plan)
        return written, True

    def _package_edits(self, plan: Plan) -> list[Edit]:
        return [Edit(path=p, kind="create", new="",
                     note="make the folder a package")
                for p in self._missing_packages(plan)]

    def _make_packages(self, plan: Plan) -> None:
        """Give every Python subdirectory an `__init__.py`.

        THE REASON THIS EXISTS IS THE MOST EXPENSIVE BUG THIS ENGINE HAD.

        Every build, of every project, reported on every file::

            the test command succeeded but ran ZERO tests — that is not
            evidence the code works, only that nothing contradicted it

        That sentence was written as a warning about an edge case. It was in
        fact describing the normal state of affairs, every time, and it was
        exactly right — nothing was ever being verified.

        The cause is one line of `unittest` behaviour:

            discover will not descend into a directory that is not an
            importable package

        (kept on one line so a grep for it finds it). The
        generated `tests\\` folder had no `__init__.py`, so discovery from the
        project root walked past it, found nothing, ran nothing, and exited
        zero. Measured on a real project: **0 tests, "OK"**. Adding the file:
        **14 tests, 10 failing.** Ten real defects that a green build had been
        quietly stepping over.

        It is worth being precise about how this hid. The caveat was honest,
        prominent and repeated — and because it appeared on every single file
        it read as boilerplate rather than as a finding:

            a warning that is always on is a warning nobody sees

        Empty files, written once, never regenerated. `src\\` gets one too:
        a namespace package is enough for `import src.physics` to work, but
        being explicit costs nothing and removes the difference between a
        layout that happens to work and one that is meant to.
        """
        for init in self._missing_packages(plan):
            try:
                self.host.fs.write(init, "")
            except Exception:                                # noqa: BLE001
                continue          # a folder we cannot write is the write
                #: jail doing its job, and it is not this method's business
                #: to argue with it.

    def _missing_packages(self, plan: Plan) -> list[str]:
        """The `__init__.py` files the plan's Python folders still need."""
        if (self.lang or "python") != "python":
            return []
        folders: set[str] = set()
        for task in plan.tasks:
            path = str(task.path).replace("\\", "/")
            if "/" not in path or not path.endswith(".py"):
                continue
            folders.add(path.rsplit("/", 1)[0])
        missing = []
        for folder in sorted(folders):
            init = f"{folder}/__init__.py"
            try:
                if self.host.fs.exists(init):
                    continue
            except Exception:                                # noqa: BLE001
                continue
            missing.append(init)
        return missing

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
            body = [f"# {_SENTINEL_TEXT}", f'"""{task.purpose}"""', ""]
            body += imports + ([""] if imports else [])
            body += ["", "def main() -> int:",
                     f'    """{task.purpose}"""',
                     "    raise NotImplementedError(",
                     f'        "{_module(task.path)}.main is not written '
                     f'yet")', ""]
            return "\n".join(body)
        scaffold = langs.scaffold_for(lang_id, task.purpose[:40] or "module",
                                      _stem(task.path))
        return _mark_stub(scaffold or f"{_comment(lang_id)} {task.purpose}\n",
                          lang_id)

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

        THE DEADLOCK THIS USED TO SIT IN, 2026-08-07
        --------------------------------------------
        The principle was right and the method could never work. The order is
        derived by reading imports out of the files on disk. At the only point
        this was called, those files were the SKELETON — and ``stub_for``
        writes a stub's imports from ``task.depends_on``, which is populated
        by this method. So:

            plan()          -> every depends_on is empty
            skeleton()      -> stubs written with NO imports, because of that
            derive_order()  -> reads the stubs, finds no imports, learns
                               nothing, and returns the model's order intact

        It was a no-op on every project ever built, and silently: an ordering
        function that returns *an* order looks like it worked.

        Two real runs of the same racing-game spec showed exactly what that
        costs. One model happened to propose math3d → physics → track →
        render → main and produced three usable files. The other proposed
        main → math3d → physics → render → track, spent three attempts
        failing to import ``CarState`` from a module that had not been written
        yet, gave up, and left two files that still do not import. The
        difference was luck, and the mechanism meant to remove that luck was
        this function.

        THE FIX. Dependencies are learned from whatever real evidence exists,
        and when there is none yet, from the one fact available before any
        code is written: some files are entry points and cannot be imported
        by their siblings. That is a naming and role convention, not a claim
        by the model, and it is checkable. Ordering it last is right whenever
        it applies and harmless when it does not.

        Called before the skeleton, the role rule seeds the order. Stubs then
        get real imports, and every later call refines the order from those —
        facts replacing the heuristic as soon as facts exist.
        """
        by_module = {_module(t.path): t.id for t in plan.tasks}
        depends: dict[str, set] = {t.id: set() for t in plan.tasks}
        learned = False
        for task in plan.tasks:
            try:
                text = self.host.fs.read(task.path)
            except Exception:                            # noqa: BLE001
                continue
            lang_id = task.lang or self.lang
            names = (_python_import_targets(text, task.path)
                     if lang_id == "python"
                     else parse_regex.imports_of(text, lang_id))
            for name in names:
                target = by_module.get(_module(str(name)))
                if target and target != task.id:
                    depends[task.id].add(target)
                    learned = True

        #: Nothing on disk to learn from — the pre-skeleton call, or a set of
        #: files that genuinely do not import one another. Fall back to role.
        if not learned:
            depends = self._role_order(plan)

        #: A test depends on the module it covers, always. This holds no
        #: matter which branch produced `depends`, and without it a test can
        #: be scheduled before the thing it tests exists.
        for task in plan.tasks:
            if not _looks_like_test(task.path):
                continue
            stem = _stem(task.path)
            covered = stem[5:] if stem.startswith("test_") else stem
            for other in plan.tasks:
                if other.id != task.id and _stem(other.path) == covered:
                    depends[task.id].add(other.id)

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

    def _role_order(self, plan: Plan) -> dict[str, set]:
        """Ordering from role, for use before any code exists to read.

        The only claim made here: **an entry point is imported by nothing.**
        A module named ``main``/``app``/``cli``/``server``, or whose stated
        purpose is to be the event loop or the initialisation, sits at the top
        of the import graph. Everything else can precede it; it can precede
        nothing.

        That is weaker than reading real imports and it is meant to be. It is
        a floor, replaced by fact the moment a single file has a body. What it
        buys is that the *first* build is not a coin toss — which is precisely
        what went wrong when a planner put ``main.py`` first and the run was
        over before it started.

        Deliberately NOT inferred: any ordering among non-entry-point files.
        Guessing that ``physics`` precedes ``track`` from their names would be
        the model's-word problem wearing a different hat, and the topological
        sort is stable, so they keep the order they were proposed in.

        WHAT THIS STILL CANNOT KNOW, stated plainly so nobody assumes it can.
        In the run that prompted this, ``render`` imported ``TrackSegment``
        from ``track`` and was scheduled first, so it failed. Nothing here
        fixes that, because before ``render`` is written the fact does not
        exist anywhere: the stub imports nothing, and the purpose line said
        only "Pygame-specific rendering of projected track segments".

        Matching that sentence against sibling module names was tried and
        rejected. It would have made ``math3d`` — "pure projection logic for
        track segments" — depend on ``track``, which is the exact opposite of
        the specification's requirement that the projection module depend on
        nothing. A heuristic that reverses a stated architectural constraint
        is worse than no heuristic.

        The real repair for that case is a richer skeleton — stubs that
        declare the symbols their module is supposed to export, so an importer
        can be checked against them before anything is generated. That is a
        larger change and belongs in its own pass.
        """
        entries = [t.id for t in plan.tasks if is_entry_point(t)]
        #: A TEST IS NOT A DEPENDENCY OF THE PROGRAM, and saying otherwise
        #: cost a whole build.
        #:
        #: This first read "everything that is not an entry point", which
        #: swept in the test files. Two things followed, both wrong:
        #:
        #:   * `stub_for` writes a stub's imports from `depends_on`, so
        #:     `src/main.py` was scaffolded with `from tests.test_physics
        #:     import *` — an entry point importing its own test suite,
        #:     which is backwards and would ship that way;
        #:   * when those tests failed, `main.py` had unmet dependencies
        #:     forever. `next_ready()` never returned it, and the run ended
        #:     at six of seven files with no explanation offered.
        #:
        #: A test consumes the module; the module does not consume the test.
        #: The dependency runs one way and it is already expressed by the
        #: test-follows-its-module rule in `derive_order`.
        others = [t.id for t in plan.tasks
                  if t.id not in entries and not _looks_like_test(t.path)
                  and not _in_test_dir(t.path)]
        return {t.id: (set(others) if t.id in entries else set())
                for t in plan.tasks}

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
        kept = getattr(self, "kept_as_found", None) or set()
        for task in plan.tasks:
            if task.status != "pending":
                revised.append(task)
                continue
            if task.path in kept:
                # Its body was there BEFORE this session: the skeleton kept
                # it rather than stubbing over it. That is the operator's
                # work, not evidence the task is done — "extend src/util.py"
                # must still extend it.
                revised.append(task)
                continue
            if self.host.fs.exists(task.path) and _has_body(
                    self.host.fs, task.path, task.lang or self.lang):
                # Somebody already wrote it — a real occurrence when one file
                # legitimately implements two planned responsibilities.
                revised.append(task.with_status("done"))
                continue
            revised.append(task)

        #: RE-ORDER WHAT IS LEFT, now that a real file exists to read.
        #:
        #: Before the first build the order rests on a role heuristic, which
        #: is a floor rather than an answer. Every completed file replaces a
        #: little of that guess with fact: its imports are now on disk and can
        #: be parsed. Re-deriving here is what lets the plan discover, after
        #: math3d is written, that render imports track and must follow it —
        #: a relationship no amount of staring at filenames would reveal, and
        #: the exact one that left two files unimportable in a real run.
        #:
        #: Done work is never re-ordered. It is finished, its position is
        #: history, and moving it would make the journal a poor record of what
        #: actually happened.
        revised = self._reorder_pending(
            Plan(request=plan.request, tasks=tuple(revised),
                 layout_note=plan.layout_note, caveats=plan.caveats))

        if self.journal is not None:
            self.journal.log("plan", replan=True, reason=reason,
                             remaining=[t.path for t in revised
                                        if t.status == "pending"])
        return Plan(request=plan.request, tasks=tuple(revised),
                    layout_note=plan.layout_note, caveats=plan.caveats)

    def _reorder_pending(self, plan: Plan) -> list[Task]:
        """Re-sort only the pending tail, preserving finished work in place."""
        ordered = self.derive_order(plan).tasks
        settled = [t for t in plan.tasks if t.status != "pending"]
        settled_ids = {t.id for t in settled}
        pending = [t for t in ordered if t.id not in settled_ids]
        return settled + pending


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
    return _parse_file_list_with_skipped(text)[0]


def _parse_file_list_with_skipped(text: str
                                  ) -> tuple[list[tuple[str, str]],
                                             list[str]]:
    """(rows, paths left out because no language owns them).

    The second list exists because the left-out rows used to vanish:
    `README.md` and `pyproject.toml` were dropped without a word, and the
    operator who asked for them found out by their absence.
    """
    rows: list[tuple[str, str]] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for m in _LINE.finditer(text or ""):
        path = m.group("path").strip()
        purpose = m.group("purpose").strip(" -—:–\t")
        if not purpose or path in seen:
            continue
        seen.add(path)
        if not langs.id_for_path(path):
            skipped.append(path)
            continue
        rows.append((path, purpose[:200]))
    return rows, skipped


#: A path that is explicitly a test file. Anchored on the FILENAME beginning
#: `test_`/`spec_` or ending `_test`/`_spec`/`.test`/`.spec`, so `tests/
#: helpers.py` is not swept up and `src/latest_data.py` is not mistaken for
#: one. Directory part optional: a request may say `test_math3d.py` alone.
#: The CamelCase `FooTest.java` / `FooTests.cs` form is included because the
#: old `_looks_like_test` accepted it and dropping it would be a regression
#: for Java, Kotlin and C#. It needs a CAPITAL `Test`, so `Contest.java` and
#: `Attest.kt` stay modules.
_TEST_PATH = re.compile(
    r"(?<![\w/.\\])"
    r"(?P<path>(?:[\w.-]+[/\\])*"
    r"(?:(?:test|spec)_[\w-]+|[\w-]+(?:_(?:test|spec)|\.(?:test|spec))"
    r"|[A-Z][A-Za-z0-9]*(?:Tests?|Spec))"
    r"\.\w{1,4})"
    r"(?![\w])")


def _required_tests(request: str) -> list[str]:
    """Test files the request named outright, in the order they appeared.

    Used by :meth:`Planner._ensure_required_tests`. Kept as a free function so
    it can be tested against a real specification without constructing a host.

    Conservative on purpose. It answers "did the operator name this file?",
    not "should there be tests?" — the second question invites a planner to
    invent work nobody asked for, and the first is the one that was being
    ignored.
    """
    out: list[str] = []
    for m in _TEST_PATH.finditer(request or ""):
        path = m.group("path").replace("\\", "/").lstrip("./")
        if not langs.id_for_path(path):
            continue
        if path not in out:
            out.append(path)
    return out


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
    """Is this a test file? ONE definition, shared with `_required_tests`.

    This used to be looser than `_TEST_PATH`: any stem ending in "test", or
    any path containing "/test". So `src/contest.py`, `src/attest.py` and
    `src/testimonials.py` got the tester persona and no stub, while the
    same paths named in a request were (correctly) not required tests. Two
    definitions of one idea disagree eventually; now there is one.
    """
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return bool(_TEST_PATH.fullmatch(name)
                or _TEST_PATH.fullmatch(name.lower()))


_TEST_DIRS = {"test", "tests", "spec", "specs", "__tests__"}


def _in_test_dir(path: str) -> bool:
    """Support code that lives with the tests: fixtures, helpers, conftest.

    Not a test file — but not a module that should be paired with a test of
    its own, or counted when working out where the project keeps sources.
    """
    parts = str(path).replace("\\", "/").split("/")[:-1]
    return any(part.lower() in _TEST_DIRS for part in parts)


def _module(path: str) -> str:
    p = str(path or "").replace("\\", "/")
    for ext in (".py", ".pyw"):
        if p.endswith(ext):
            p = p[:-len(ext)]
    if p.endswith("/__init__"):
        p = p[:-len("/__init__")]
    return p.strip("/").replace("/", ".")


def _python_import_targets(text: str, importer: str) -> list[str]:
    """Every dotted module a Python file's imports could mean, resolved.

    `parse_python.imports_of` returns what an import statement NAMES, which
    is right for the codemap and not enough for ordering. The two idioms
    models use most defeated it, and the render-before-track failure in the
    CHANGELOG survived for exactly those two:

      * `from .track import T` names `.track`, which means nothing until it
        is resolved against the importer's own package — `src.track`;
      * `from src import track` names `src`, when the dependency is on
        `src.track`. So each imported name is also tried as a submodule.

    Candidates that are not project modules simply match nothing.
    """
    import ast
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    module = _module(importer)
    is_package = importer.replace("\\", "/").endswith("/__init__.py")
    package = module if is_package else (
        module.rsplit(".", 1)[0] if "." in module else "")
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                up = node.level - 1
                if up > len(parts):
                    continue                 # climbs out of the project
                base_parts = parts[:len(parts) - up] if up else parts
                base = ".".join(base_parts + (
                    [node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                out.append(base)
            out.extend(f"{base}.{alias.name}" if base else alias.name
                       for alias in node.names if alias.name != "*")
    return out


def _stem(path: str) -> str:
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] if "." in name else name


def _comment(lang_id: str) -> str:
    lang = langs.get(lang_id)
    return lang.comment if lang else "#"


def _has_body(fs: Any, path: str, lang_id: str) -> bool:
    """Is this a real implementation or still a stub?

    The sentinel answers it in every language. The `NotImplementedError`
    check stays as the fallback for Python stubs written before the
    sentinel existed.
    """
    try:
        text = fs.read(path)
    except Exception:                                    # noqa: BLE001
        return False
    if STUB_SENTINEL in text:
        return False
    return bool(text.strip()) and "NotImplementedError" not in text


#: Module names and purpose phrases that mark an ENTRY POINT — a file
#: nothing else imports. Used to seed the build order before any code
#: exists (`derive_order`) and to leave entry points out of the "what
#: already exists" interfaces shown to a module (`Session._built_siblings`).
ENTRY_NAMES = frozenset({"main", "app", "cli", "__main__", "index", "run",
                         "server", "start", "program", "game"})
ENTRY_WORDS = ("entry point", "event loop", "initialization",
               "initialisation", "main loop", "bootstrap", "game loop",
               "startup", "command line")


def is_entry_point(task: Task) -> bool:
    """Is this task's file an entry point — imported by nothing?"""
    if _looks_like_test(task.path):
        return False
    if _stem(task.path).lower() in ENTRY_NAMES:
        return True
    purpose = (task.purpose or "").lower()
    return any(w in purpose for w in ENTRY_WORDS)


def _is_our_stub(text: str) -> bool:
    """Did this engine's skeleton write this file?

    The sentinel, or the old Python stub's own marker line — so a stub left
    by an earlier run can be replaced while an abstract base class that
    raises NotImplementedError on purpose cannot.
    """
    return STUB_SENTINEL in text or (
        "NotImplementedError(" in text and "is not written yet" in text)


def _mark_stub(text: str, lang_id: str) -> str:
    """The stub with its sentinel comment, placed where it cannot hurt.

    After a shebang or `@echo off`, never before: a script whose first line
    is a comment is no longer a script. The block form is used where the
    language declares one, because C's scaffold avoids `//` for a reason.
    """
    lang = langs.get(lang_id)
    if lang and lang.block_comment:
        line = f"{lang.block_comment[0]} {_SENTINEL_TEXT} " \
               f"{lang.block_comment[1]}"
    else:
        line = f"{_comment(lang_id)} {_SENTINEL_TEXT}"
    lines = text.split("\n")
    first = lines[0].strip().lower() if lines else ""
    at = 1 if first.startswith(("#!", "@echo off", "<?php")) else 0
    return "\n".join(lines[:at] + [line] + lines[at:])


def strip_stub_sentinel(code: str) -> str:
    """Generated code, minus a copied stub marker.

    A model shown the stub may echo its marker line into the real file,
    which would then read as a stub forever: `replan` would treat its task
    as unwritten and a later skeleton as replaceable. Only a line carrying
    the full sentinel text is removed.
    """
    if _SENTINEL_TEXT not in code:
        return code
    return "\n".join(ln for ln in code.split("\n")
                     if _SENTINEL_TEXT not in ln)


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
