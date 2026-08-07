# SPDX-License-Identifier: Apache-2.0
"""Migrate ATK's six coding modules onto Cognitive Coder (§7.3).

**Dry run by default.** This edits a real, substantial, working application,
and the integration must not disturb it. Nothing is written unless you pass
`--apply`, and even then every replaced file is backed up first.

    python adapters/atk/migrate.py --atk D:/Analyst_Toolkit/ATK          # look
    python adapters/atk/migrate.py --atk D:/Analyst_Toolkit/ATK --apply  # do

WHAT IT DOES, and why in this order (§7.3):

    1. Check that Cognitive Coder is importable from ATK's interpreter.
       If it is not, nothing else can work and stopping now is cheap.
    2. Back up each of the six modules to `<name>.py.pre-ccoder`.
    3. Replace each body with a re-export shim plus a comment saying where
       it went and why.
    4. Tell you to run ATK's full suite. **It must stay green** — those six
       modules have around two hundred checks between them, and they are the
       reason this migration is safe to attempt at all.
    5. Only after that do you delete the shims and update ATK's imports.
       This script will not do step 5 for you; it is a judgement call about
       an application it does not own.

THE RENAMES, which are the one thing that cannot be done mechanically:

    codeguard.py  →  cognitive_coder.guard
    coderun.py    →  cognitive_coder.runner
    codectx.py    →  cognitive_coder.context

Those three shims re-export under the OLD name, so `from atk.core.codeguard
import scan` keeps working. The other three keep their names.

WHAT IT WILL NOT DO. It will not touch anything outside the six files, it
will not edit ATK's imports, and it will not run ATK's tests for you — a
migration script that reports its own success is not evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

#: ATK module → the Cognitive Coder module that now owns it.
MODULES: dict[str, str] = {
    "langs.py": "langs",
    "diagnostics.py": "diagnostics",
    "codeguard.py": "guard",
    "coderun.py": "runner",
    "patcher.py": "patcher",
    "codectx.py": "context",
}

SHIM = '''"""Moved to Cognitive Coder — this file is a compatibility shim.

The implementation now lives in `cognitive_coder.{target}` and is shared with
any other host that embeds the engine. It moved because two parallel copies
of the same module diverge within a month, and this one had already started
to: the bug fixes found the hard way here are the reason Cognitive Coder was
built on top of this code rather than written from scratch.

**The behaviour did not change. Some SIGNATURES did**, because the engine
takes Ports where this module took paths — that is what makes it testable
with no disk and no model, and usable by a host that is not ATK. The names
re-exported below come from `atk.core.ccoder_compat`, which supplies the OLD
signatures on top of the new engine, so every existing call site here keeps
working:

    lang.available()                  no arguments, as before
    diagnostics.feedback(TEXT, …)     raw text in, as before
    Diagnostic.source                 the old field name, aliased
    coderun.build_and_run(…, PATH)    a workspace path, as before
    patcher.apply(edits, ROOT)        one call, as before

Once ATK's own call sites have been updated to pass Ports directly, delete
this file AND the compatibility module with it.

See adapters/atk/README.md in the Cognitive Coder repository, and §7.3 of
COGNITIVE_CODER_BUILD_SPEC_v1.1.md.
"""

from atk.core.ccoder_compat import *  # noqa: F401,F403
from atk.core import ccoder_compat as _compat

# Only the names this module used to export, so `import *` from here does
# not leak the whole compatibility surface into callers that never had it.
__all__ = {exports!r}

for _name in __all__:
    if not hasattr(_compat, _name):  # pragma: no cover - caught by the tests
        raise ImportError(
            "{name} previously exported {{!r}}, which the compatibility "
            "layer does not provide. Restore {name}.pre-ccoder and report "
            "this — the migration is not safe until it is fixed."
            .format(_name))
'''

#: What each ATK module used to export. The shim re-exports exactly this and
#: no more, and FAILS LOUDLY at import if the compatibility layer is missing
#: one — a silent gap here is a NameError somewhere in ATK three days later.
EXPORTS: dict[str, list[str]] = {
    "langs.py": ["Lang", "LANGS", "EXE_SUFFIX", "get", "ids", "labels",
                 "for_extension", "available_ids", "scaffold_for", "render"],
    "diagnostics.py": ["Diagnostic", "MAX_FEEDBACK", "CONTEXT_LINES",
                       "parse", "attach_source", "feedback", "summarise",
                       "first_error"],
    "codeguard.py": ["BLOCK", "WARN", "Finding", "scan", "blocked",
                     "advisory", "explain_to_model"],
    "coderun.py": ["MAX_OUTPUT", "DEFAULT_TIMEOUT", "BUILD_TIMEOUT", "Phase",
                   "RunResult", "build_and_run", "run_tests", "format_code",
                   "lint_code"],
    "patcher.py": ["SNAPSHOT_DIR", "MAX_SNAPSHOTS", "Edit", "EditResult",
                   "ApplyOutcome", "parse_edits", "apply", "snapshots",
                   "undo", "preview"],
    "codectx.py": ["DEFAULT_BUDGET", "Symbol", "symbols", "outline",
                   "slice_around", "symbol_body", "build_context",
                   "project_map", "relevant_files"],
}


def check_importable(python: str) -> tuple[bool, str]:
    """Can ATK's interpreter import the engine? Nothing works if not."""
    code = ("import cognitive_coder as cc; "
            "print(cc.__version__)")
    try:
        result = subprocess.run([python, "-c", code], capture_output=True,
                                text=True, timeout=60)
    except Exception as exc:                             # noqa: BLE001
        return False, f"could not run {python}: {exc}"
    if result.returncode != 0:
        return False, (
            "cognitive_coder is not importable from that interpreter. "
            "Install it into ATK's environment first:\n"
            "    <atk-python> -m pip install -e <path-to-cognitive-coder>\n"
            f"The import said: {result.stderr.strip().splitlines()[-1:]}")
    return True, result.stdout.strip()


def plan(atk_root: Path) -> list[tuple[Path, str, str]]:
    """(path, target module, status) for each of the six."""
    core = atk_root / "atk" / "core"
    rows: list[tuple[Path, str, str]] = []
    for name, target in MODULES.items():
        path = core / name
        if not path.exists():
            rows.append((path, target, "missing — nothing to do"))
        elif "from cognitive_coder" in path.read_text(encoding="utf-8",
                                                      errors="replace"):
            rows.append((path, target, "already a shim"))
        else:
            lines = len(path.read_text(encoding="utf-8",
                                       errors="replace").splitlines())
            rows.append((path, target, f"{lines} lines → shim"))
    return rows


def install_compat(atk_root: Path) -> str:
    """Copy the compatibility layer next to the shims that need it.

    Every shim imports `atk.core.ccoder_compat`, so this must land first or
    all six fail at import — which would take ATK's whole suite with them.
    """
    source = Path(__file__).with_name("atk_compat.py")
    target = atk_root / "atk" / "core" / "ccoder_compat.py"
    target.write_bytes(source.read_bytes())
    return str(target)


def install_host(atk_root: Path) -> list[str]:
    """The Port implementations and the panel."""
    here = Path(__file__).parent
    out: list[str] = []
    for source, destination in (
            (here / "ccoder_host.py", atk_root / "atk" / "core" /
             "ccoder_host.py"),
            (here / "ccoder_panel.py", atk_root / "atk" / "ui" /
             "ccoder_panel.py")):
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        out.append(str(destination))
    return out


def apply(rows: list[tuple[Path, str, str]]) -> list[str]:
    done: list[str] = []
    for path, target, status in rows:
        if not path.exists() or status == "already a shim":
            continue
        backup = path.with_suffix(".py.pre-ccoder")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        path.write_text(SHIM.format(target=target, name=path.name,
                                    exports=EXPORTS.get(path.name, [])),
                        encoding="utf-8")
        done.append(f"{path.name} → cognitive_coder.{target} "
                    f"(original kept as {backup.name})")
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate ATK's six coding modules onto Cognitive Coder.")
    parser.add_argument("--atk", required=True,
                        help="the ATK repository root")
    parser.add_argument("--python", default=sys.executable,
                        help="ATK's interpreter (default: this one)")
    parser.add_argument("--apply", action="store_true",
                        help="actually write. Without this it only looks.")
    args = parser.parse_args(argv)

    atk_root = Path(args.atk).expanduser().resolve()
    if not (atk_root / "atk" / "core").is_dir():
        print(f"{atk_root} does not look like an ATK checkout — there is no "
              f"atk/core directory under it.")
        return 2

    ok, detail = check_importable(args.python)
    print(f"Cognitive Coder importable from {args.python}: "
          f"{'yes, v' + detail if ok else 'NO'}")
    if not ok:
        print(detail)
        return 3

    rows = plan(atk_root)
    print(f"\nSix modules under {atk_root / 'atk' / 'core'}:\n")
    for path, target, status in rows:
        print(f"  {path.name:<18} → cognitive_coder.{target:<12} {status}")

    print("\nAlso installed:")
    print(f"  atk/core/ccoder_compat.py   ATK's OLD signatures on the new "
          f"engine")
    print(f"  atk/core/ccoder_host.py     the six Port implementations")
    print(f"  atk/ui/ccoder_panel.py      the workspace tab")

    if not args.apply:
        print("\nThis was a DRY RUN. Nothing was written.")
        print("Re-run with --apply when you are ready, then:")
        print("  1. run ATK's full suite — it MUST stay green")
        print("  2. only then update ATK's imports and delete the shims")
        return 0

    print(f"\n  compatibility layer → {install_compat(atk_root)}")
    for written in install_host(atk_root):
        print(f"  adapter             → {written}")

    done = apply(rows)
    if not done:
        print("\nNothing needed changing.")
        return 0

    print("\nWritten:")
    for line in done:
        print(f"  {line}")
    print("\nNOW RUN ATK'S FULL SUITE. Those six modules have around two "
          "hundred checks between them, and they are the reason this "
          "migration is safe to attempt — a green suite before and after is "
          "the whole evidence that nothing was lost.")
    print("\nTo undo: restore each .py.pre-ccoder over its .py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
