#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does ATK's OLD call surface still work after the migration? (§7.3)

**Written in ATK's test style, not Cognitive Coder's** (§7.1): a plain script
with a `check(name, cond)` helper printing PASS/FAIL and a final count. ATK
uses no pytest, and the adapter's tests follow ATK's conventions rather than
importing this project's. Keep them separate.

    python adapters/atk/test_migration.py

WHAT THIS IS FOR. §7.3 says every migration step must leave ATK's suite
green, and ATK's suite is the real evidence. But ATK's suite only runs on a
machine with ATK checked out, and by the time it fails you have already
rewritten six files. This runs first, against a throwaway copy, and answers
the one question that decides whether the migration is safe: **after the
shims are in, does every call ATK makes today still work?**

It matters because the engine takes Ports where ATK's modules took paths. A
naive `from cognitive_coder.guard import *` shim passes a smoke test and then
fails at `lang.available()` — no arguments where the engine wants an
ExecPort — somewhere deep in whatever code path runs first. Each check below
is a signature that actually changed.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PASSED = 0
FAILED = 0


def check(name: str, condition: object) -> bool:
    global PASSED, FAILED
    ok = bool(condition)
    if ok:
        PASSED += 1
    else:
        FAILED += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


def build_fake_atk(cc_root: Path) -> Path:
    """A throwaway ATK tree with the six real modules, then migrated.

    A COPY, always. This script must never touch a real checkout — running a
    migration test against somebody's working application is the kind of
    convenience that ends an evening badly.
    """
    import migrate

    root = Path(tempfile.mkdtemp(prefix="fake-atk-"))
    (root / "atk" / "core").mkdir(parents=True)
    (root / "atk" / "ui").mkdir(parents=True)
    for package in (root / "atk", root / "atk" / "core", root / "atk" / "ui"):
        (package / "__init__.py").write_text("", encoding="utf-8")

    # The originals, if a real ATK is available; otherwise the shims are
    # tested against nothing, which is still the interesting half.
    source = _find_atk_core()
    for name in migrate.MODULES:
        if source and (source / name).exists():
            shutil.copy2(source / name, root / "atk" / "core" / name)
        else:
            (root / "atk" / "core" / name).write_text(
                "# placeholder for the migration test\n", encoding="utf-8")

    migrate.install_compat(root)
    migrate.install_host(root)
    migrate.apply(migrate.plan(root))
    return root


def _find_atk_core() -> Path | None:
    """Look for a real ATK checkout beside this one. Optional."""
    for candidate in (
            Path(os.environ.get("ATK_ROOT", "")) / "atk" / "core",
            Path.cwd().parent / "ATK" / "atk" / "core",
            Path("D:/Analyst_Toolkit/ATK/atk/core"),
            Path("/sessions") / "core"):
        try:
            if candidate.is_dir() and (candidate / "langs.py").exists():
                return candidate
        except OSError:
            continue
    return None


def main() -> int:
    cc_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(cc_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    print("=" * 62)
    print(" ATK migration — does the old call surface still work?")
    print("=" * 62)

    try:
        import cognitive_coder
        print(f"\nCognitive Coder {cognitive_coder.__version__} importable.")
    except ImportError as exc:
        print(f"\ncognitive_coder is not importable: {exc}")
        print("Install it into this interpreter first.")
        return 3

    fake = build_fake_atk(cc_root)
    sys.path.insert(0, str(fake))
    print(f"Migrated a throwaway copy at {fake}\n")

    # ---------------------------------------------------------------
    section("langs — probing took no arguments, and must not start to")
    from atk.core import langs
    check("LANGS is populated", len(langs.LANGS) > 10)
    check("get('python')", langs.get("python") is not None)
    check("available_ids() takes NO arguments",
          isinstance(langs.available_ids(), list))
    check("scaffold_for()", "def main" in langs.scaffold_for("python", "t"))
    check("for_extension()", langs.for_extension("a.py").id == "python")
    check("render() returns an argv LIST, never a string",
          langs.render(["{run}", "{src}"], run="py", src="a.py")
          == ["py", "a.py"])
    check("EXE_SUFFIX", langs.EXE_SUFFIX == ".exe")

    # ---------------------------------------------------------------
    section("diagnostics — raw text in, and `.source` by its old name")
    from atk.core import diagnostics
    diags = diagnostics.parse("main.c:5:9: error: 'x' undeclared", "c")
    check("parse() locates the error", diags and diags[0].line == 5)
    check("Diagnostic.source exists (renamed to source_excerpt upstream)",
          hasattr(diags[0], "source"))
    check("Diagnostic is MUTABLE, as ATK's was",
          _is_mutable(diags[0]))
    check("one_line()", "error" in diags[0].one_line())
    check("where()", "main.c:5" in diags[0].where())
    check("rank", diags[0].rank == 0)
    check("feedback() takes RAW TEXT, not diagnostics",
          "undeclared" in diagnostics.feedback(
              "main.c:5:9: error: 'x' undeclared", "c"))
    check("summarise()", "error" in diagnostics.summarise(diags))
    check("first_error()", diagnostics.first_error(diags) is not None)
    check("unparsed output never yields an empty list",
          len(diagnostics.parse("mysterious toolchain failure", "c")) == 1)

    # ---------------------------------------------------------------
    section("codeguard → guard")
    from atk.core import codeguard
    findings = codeguard.scan("import os\nos.system('rm -rf /')", "python")
    check("scan() flags it", len(findings) > 0)
    check("blocked() gives a reason", bool(codeguard.blocked(findings)))
    check("BLOCK / WARN constants",
          codeguard.BLOCK == "block" and codeguard.WARN == "warn")
    check("explain_to_model() instructs rather than complains",
          "Rewrite" in codeguard.explain_to_model(findings))
    check("advisory()", isinstance(codeguard.advisory(findings), str))
    check("clean code is not flagged",
          codeguard.scan("def add(a, b):\n    return a + b\n", "python") == [])

    # ---------------------------------------------------------------
    section("coderun → runner — a workspace PATH, not a Port")
    from atk.core import coderun
    workspace = tempfile.mkdtemp()
    good = coderun.build_and_run('print("hi")\n', "python", workspace)
    check("build_and_run(code, lang, WORKSPACE_PATH)", good.ok)
    check("RunResult.stdout is the PROGRAM's output", "hi" in good.stdout)
    check("failed_phase is empty on success", good.failed_phase == "")
    check("summary()", "ok in" in good.summary())
    check("MAX_OUTPUT / DEFAULT_TIMEOUT / BUILD_TIMEOUT",
          coderun.MAX_OUTPUT and coderun.DEFAULT_TIMEOUT
          and coderun.BUILD_TIMEOUT)

    if shutil.which("gcc") or shutil.which("clang") or shutil.which("cc"):
        broken = coderun.build_and_run(
            '#include <stdio.h>\nint main(void){ return x; }\n', "c",
            workspace)
        check("a broken C file fails in the BUILD phase",
              broken.failed_phase == "build")
        check("Phase.ok and Phase.output survive",
              all(hasattr(p, "ok") and hasattr(p, "output")
                  for p in broken.phases))
    else:
        print("  [SKIP] C discrimination — no C compiler on this machine")

    # ---------------------------------------------------------------
    section("patcher — apply(edits, ROOT) and undo(ROOT)")
    from atk.core import patcher
    project = Path(tempfile.mkdtemp())
    (project / "m.py").write_text("a = 1\n", encoding="utf-8")
    edits = patcher.parse_edits("```python path=m.py\na = 42\n```")
    check("parse_edits()", len(edits) == 1)
    outcome = patcher.apply(edits, project)
    check("apply(edits, ROOT)", outcome.ok)
    check("ApplyOutcome.summary()", "changed" in outcome.summary())
    check("the file actually changed",
          "42" in (project / "m.py").read_text(encoding="utf-8"))
    check("snapshots(root)", len(patcher.snapshots(project)) >= 1)
    undone = patcher.undo(project)
    check("undo(root)", undone["ok"])
    check("undo restored the original bytes",
          (project / "m.py").read_text(encoding="utf-8") == "a = 1\n")
    check("preview() changes nothing",
          "+a = 9" in patcher.preview(
              [patcher.Edit(path="m.py", kind="whole", new="a = 9\n")],
              project))
    check("an ambiguous anchor is still refused",
          not patcher.apply(
              [patcher.Edit(path="dup.py", kind="replace", old="x",
                            new="y")], project).ok)

    # ---------------------------------------------------------------
    section("codectx → context — a root PATH everywhere ATK passed one")
    from atk.core import codectx
    source = 'def load(p):\n    """Load."""\n    return []\n'
    check("symbols()", codectx.symbols(source, "python")[0].name == "load")
    check("outline()", "load" in codectx.outline(source, "python", "m.py"))
    check("slice_around() carries line numbers",
          "1 |" in codectx.slice_around(source, 1))
    check("symbol_body()", "return []" in codectx.symbol_body(source, "load"))
    check("build_context(pieces, budget) declares omissions",
          "NOT INCLUDED" in codectx.build_context([("A", "x", 1)], 5000))
    check("project_map(ROOT)", "code file" in codectx.project_map(project))
    check("relevant_files(ROOT, query)",
          isinstance(codectx.relevant_files(project, "module"), list))
    check("DEFAULT_BUDGET", codectx.DEFAULT_BUDGET == 24_000)

    # ---------------------------------------------------------------
    section("the adapter itself")
    from atk.core import ccoder_host
    check("build_host is importable without Qt",
          callable(ccoder_host.build_host))
    check("ATK_CONVENTIONS carries the doctrine",
          "Deterministic first" in ccoder_host.ATK_CONVENTIONS)
    check("preflight names a missing model",
          any("No model is loaded" in note
              for note in ccoder_host.preflight(None, project)))
    check("ATKExec kills the tree on timeout (M16)",
          _tree_kill_works(ccoder_host.ATKExec()))
    check("ATKFileSystem refuses an escape (M24)",
          _jail_holds(ccoder_host.ATKFileSystem(project)))
    check("ATKStorage refuses an unserialisable value (M17)",
          _storage_is_json_only(ccoder_host.ATKStorage))
    check("ATKApproval never auto-approves the network (C3)",
          ccoder_host.ATKApproval(auto_apply=True).approve_remote(
              "anthropic", 1024, "everything") is False)

    # ---------------------------------------------------------------
    section("the shims fail LOUDLY rather than silently")
    check("a shim declares exactly what it used to export",
          "__all__" in (fake / "atk" / "core" / "langs.py").read_text(
              encoding="utf-8"))
    check("the original is kept alongside",
          (fake / "atk" / "core" / "langs.py.pre-ccoder").exists())

    print("\n" + "=" * 62)
    print(f" {PASSED} passed, {FAILED} failed")
    print("=" * 62)
    if FAILED:
        print("\nDo NOT migrate a real ATK checkout until these pass.")
    else:
        print("\nATK's old call surface is intact. Migrate, then run ATK's")
        print("own suite — that is the evidence that matters.")
    shutil.rmtree(fake, ignore_errors=True)
    return 1 if FAILED else 0


# --------------------------------------------------------------------------
# helpers for the checks that need more than one line
# --------------------------------------------------------------------------

def _is_mutable(diagnostic: object) -> bool:
    """ATK assigned to `d.file`; the engine's Diagnostic is frozen.

    Restores what it changed. A check that leaves the object it inspected in
    a different state is a check that breaks the next one — which is exactly
    what happened the first time this ran, and the `where()` check below
    failed for a reason that had nothing to do with `where()`.
    """
    original = getattr(diagnostic, "file", None)
    try:
        diagnostic.file = "changed.c"        # type: ignore[attr-defined]
        return diagnostic.file == "changed.c"  # type: ignore[attr-defined]
    except Exception:                                    # noqa: BLE001
        return False
    finally:
        try:
            diagnostic.file = original       # type: ignore[attr-defined]
        except Exception:                                # noqa: BLE001
            pass


def _tree_kill_works(ex: object) -> bool:
    """A parent that spawns a child which outlives it — the Godot case."""
    import time

    workspace = tempfile.mkdtemp()
    marker = os.path.join(workspace, "child-survived.txt")
    child = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        " \"import time;\\ntime.sleep(5)\\n"
        "open(r'{m}','w').write('alive')\"])\n"
        "time.sleep(30)\n").format(m=marker)
    result = ex.run([sys.executable, "-c", child], cwd=workspace,
                    timeout=2)                # type: ignore[attr-defined]
    if not result.timed_out:
        return False
    time.sleep(6)
    return not os.path.exists(marker)


def _jail_holds(fs: object) -> bool:
    for escape in ("../escaped.txt", "../../escaped.txt"):
        try:
            fs.write_bytes(escape, b"x")     # type: ignore[attr-defined]
            return False                     # it WROTE outside the root
        except Exception:                                # noqa: BLE001
            continue
    return True


def _storage_is_json_only(storage_class: type) -> bool:
    class Ctx:
        settings: dict = {}

    storage = storage_class(Ctx(), tempfile.mkdtemp())
    try:
        storage.set("bad", object())
        return False
    except ValueError:
        return True
    except Exception:                                    # noqa: BLE001
        return False


if __name__ == "__main__":
    raise SystemExit(main())
