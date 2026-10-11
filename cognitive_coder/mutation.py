# SPDX-License-Identifier: Apache-2.0
"""The mutation check: do a file's tests notice when its code is wrong?

"Verified" means the tests passed. It does not mean the tests would FAIL if
the code were wrong — and a small model writing both the code and its tests
is exactly the case where they might not: `assert result is not None`,
`assertTrue(len(rows) >= 0)`, a test that calls the function and checks
nothing. A green build on such tests is a claim with nothing behind it.

So after a build, for each verified Python module whose own test file ran,
a few DELIBERATE BUGS (mutants) are put into a copy of the module, one at a
time, and its test file is run against each:

    `<`  → `>=`      `==` → `!=`      `+` → `-`      `and` → `or`
    `return True` → `return False`    a returned / compared number n → n+1

A mutant the tests catch is "killed". One that passes them all "survived":
the tests do not check that behaviour. Survivors are reported — as a
WARNING, with the line and the change — and nothing else happens. The check
never changes the project, never fails a build and never asks the model
anything; it is evidence about the tests, for the operator.

LIMITS, stated: at most `max_mutants` per file and `budget_s` seconds per
session (each mutant is a test run); Python only (it mutates the AST); the
copy holds the project's source files under `.cc_state/mutants/` — a file
that needs something the copy lacks (a large asset) makes the CONTROL run
fail, and the check then says it could not judge rather than guessing.
A survivor can be an equivalent mutant (a change that does not change
behaviour — `>=` for `>` where the values can never be equal); the warning
says "may not check", not "does not".
"""

from __future__ import annotations

import ast
import copy
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

MUTANT_DIR = ".cc_state/mutants"
#: The copy is skipped past this size (bytes of source copied).
COPY_LIMIT = 5_000_000
_SKIP_DIRS = (".cc_", ".git", "__pycache__", ".venv", "venv", "env",
              "node_modules", "site-packages", ".mypy_cache",
              ".pytest_cache", "build", "dist")

_COMPARE = {ast.Lt: (ast.GtE, ">="), ast.GtE: (ast.Lt, "<"),
            ast.Gt: (ast.LtE, "<="), ast.LtE: (ast.Gt, ">"),
            ast.Eq: (ast.NotEq, "!="), ast.NotEq: (ast.Eq, "==")}
_COMPARE_TEXT = {ast.Lt: "<", ast.GtE: ">=", ast.Gt: ">", ast.LtE: "<=",
                 ast.Eq: "==", ast.NotEq: "!="}
_BINOP = {ast.Add: (ast.Sub, "+", "-"), ast.Sub: (ast.Add, "-", "+"),
          ast.Mult: (ast.Add, "*", "+")}


@dataclass(frozen=True)
class Mutant:
    line: int
    what: str             # "`<` → `>=`"
    code: str             # the whole mutated module


@dataclass
class MutationReport:
    path: str
    test_path: str
    tried: list = field(default_factory=list)       # [(Mutant, killed)]
    note: str = ""        # why it was skipped or cut short

    @property
    def survivors(self) -> list[Mutant]:
        return [m for m, killed in self.tried if not killed]

    @property
    def killed(self) -> int:
        return sum(1 for _m, k in self.tried if k)

    def line(self) -> str:
        if not self.tried:
            return f"{self.path}: not checked — {self.note or 'no mutants'}"
        n, s = len(self.tried), len(self.survivors)
        head = (f"{self.path}: {self.test_path} caught {n - s} of {n} "
                f"deliberate bug{'' if n == 1 else 's'}")
        if s:
            head += " — it may not check: " + "; ".join(
                f"line {m.line} {m.what}" for m in self.survivors)
        if self.note:
            head += f" ({self.note})"
        return head


def _in_main_guard(tree: ast.Module) -> set:
    """ids of nodes under `if __name__ == "__main__":` — never mutated."""
    out = set()
    for node in tree.body:
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name)
                and node.test.left.id == "__name__"):
            for sub in ast.walk(node):
                out.add(id(sub))
    return out


def _sites(tree: ast.Module) -> list[tuple[int, str, Any]]:
    """Every place a mutant can go: (line, description, how-to-apply)."""
    skip = _in_main_guard(tree)
    sites: list[tuple[int, str, Any]] = []
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _COMPARE:
                    new, text = _COMPARE[type(op)]
                    sites.append((line, f"`{_COMPARE_TEXT[type(op)]}` → "
                                        f"`{text}`",
                                  ("compare", node, i, new)))
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOP:
            new, old_t, new_t = _BINOP[type(node.op)]
            # `"a" + "b"` → `"a" - "b"` is a TypeError, not a test of the
            # tests: leave string arithmetic alone
            if not any(isinstance(side, ast.Constant)
                       and isinstance(side.value, str)
                       for side in (node.left, node.right)) \
                    and not isinstance(node.left, (ast.JoinedStr,)):
                sites.append((line, f"`{old_t}` → `{new_t}`",
                              ("binop", node, new)))
        elif isinstance(node, ast.BoolOp):
            new = ast.Or if isinstance(node.op, ast.And) else ast.And
            old_t = "and" if isinstance(node.op, ast.And) else "or"
            sites.append((line, f"`{old_t}` → "
                                f"`{'or' if old_t == 'and' else 'and'}`",
                          ("boolop", node, new)))
        elif isinstance(node, ast.Return) and isinstance(node.value,
                                                         ast.Constant):
            v = node.value.value
            if isinstance(v, bool):
                sites.append((line, f"`return {v}` → `return {not v}`",
                              ("const", node.value, not v)))
            elif isinstance(v, int) and not isinstance(v, bool):
                sites.append((line, f"`return {v}` → `return {v + 1}`",
                              ("const", node.value, v + 1)))
    sites.sort(key=lambda s: s[0])
    return sites


def _spread(items: Sequence, n: int) -> list:
    """`n` items spread evenly across the list (deterministic)."""
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def mutants(code: str, limit: int = 5) -> list[Mutant]:
    """Up to `limit` mutants of a Python module, spread through the file.
    A module that does not parse has none."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    chosen = _spread(_sites(tree), limit)
    out = []
    for line, what, how in chosen:
        # mutate a fresh copy: find the same node by walking in step
        fresh = copy.deepcopy(tree)
        target = how[1]
        pairs = zip(ast.walk(tree), ast.walk(fresh))
        twin = next((b for a, b in pairs if a is target), None)
        if twin is None:
            continue
        kind = how[0]
        if kind == "compare":
            twin.ops[how[2]] = how[3]()
        elif kind == "binop":
            twin.op = how[2]()
        elif kind == "boolop":
            twin.op = how[2]()
        elif kind == "const":
            twin.value = how[2]
        try:
            text = ast.unparse(ast.fix_missing_locations(fresh))
        except Exception:                                # noqa: BLE001
            continue
        out.append(Mutant(line, what, text + "\n"))
    return out


# ---------------------------------------------------------------------------
# running them
# ---------------------------------------------------------------------------

def _copy_paths(fs: Any) -> tuple[list[str], str]:
    """The project's files to copy, or ([], why not)."""
    try:
        # "*", not "**/*": the ports' list matches the glob against the
        # relative path, and "**/*" needs a "/" — a top-level main.py
        # would never be copied
        every = fs.list("*")
    except Exception as exc:                             # noqa: BLE001
        return [], f"the project could not be listed ({exc})"
    keep, total = [], 0
    for path in every:
        p = path.replace("\\", "/")
        parts = p.split("/")
        if any(part.startswith(_SKIP_DIRS) for part in parts[:-1]) \
                or parts[-1].startswith(".cc_"):
            continue
        try:
            size = len(fs.read_bytes(p))
        except Exception:                                # noqa: BLE001
            continue
        if size > 1_000_000:
            continue
        total += size
        if total > COPY_LIMIT:
            return [], (f"the project is larger than "
                        f"{COPY_LIMIT // 1_000_000} MB of source")
        keep.append(p)
    return keep, ""


def check(*, fs: Any, ex: Any, make_fs: Any, run_tests: Any, path: str,
          test_path: str, code: str, max_mutants: int = 5,
          deadline: float = 0.0, timeout: float | None = None,
          cancel: Any = None) -> MutationReport:
    """Run the mutants of `path` against `test_path`, in a copy.

    ``make_fs(root)`` returns a FileSystemPort rooted at the copy;
    ``run_tests(fs=, ex=, workdir=, path=, test_path=, timeout=)`` is the
    runner's, bound to the language. ``deadline`` is a monotonic time after
    which no further mutant starts.
    """
    report = MutationReport(path, test_path)
    cands = mutants(code, max_mutants)
    if not cands:
        report.note = "nothing in it to mutate"
        return report
    files, why = _copy_paths(fs)
    if not files:
        report.note = why or "no files to copy"
        return report
    root = fs.root().rstrip("/\\")
    copy_rel = MUTANT_DIR
    copy_root = f"{root}/{copy_rel}".replace("\\", "/")
    written: list[str] = []
    try:
        for p in files:
            fs.write_bytes(f"{copy_rel}/{p}", fs.read_bytes(p))
            written.append(f"{copy_rel}/{p}")
        cfs = make_fs(copy_root)
        control = run_tests(fs=cfs, ex=ex, workdir=copy_root, path=path,
                            test_path=test_path, timeout=timeout)
        if not getattr(control, "ok", False):
            report.note = ("its tests do not pass in a copy of the project, "
                           "so mutants cannot be judged")
            return report
        for m in cands:
            if cancel is not None and cancel.is_set():
                report.note = "stopped"
                break
            if deadline and time.monotonic() > deadline:
                report.note = "the time for the check ran out"
                break
            cfs.write(path, m.code)
            res = run_tests(fs=cfs, ex=ex, workdir=copy_root, path=path,
                            test_path=test_path, timeout=timeout)
            report.tried.append((m, not getattr(res, "ok", False)))
        return report
    finally:
        for p in written:
            try:
                fs.delete(p)
            except Exception:                            # noqa: BLE001
                pass
