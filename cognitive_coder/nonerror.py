# SPDX-License-Identifier: Apache-2.0
"""Tracebacks that are not the code's fault.

WHY THIS EXISTS, 2026-10-22
---------------------------
Of 26 Debug & Repair sessions on one project, four were

    ImportError: attempted relative import with no known parent package
    ModuleNotFoundError: No module named 'src'

The program had been started from inside `src\\` (`python main.py`) or by
path from the root (`python src\\main.py`). The imports were RIGHT, the
command was wrong, and the loop spent model time "repairing" correct code —
the 2026-10-16 build had already lost two of four physics attempts to the
same non-error when the runner did it (HYBRID.md). The answer is a sentence
and a command, and it costs no model call.

`classify(text, fs)` looks at the pasted text BEFORE anything is located.
It returns a Verdict when the text is one of these, else None:

  * **relative import with no parent** — run as a module from the project
    root: `python -m src.main`.
  * **no module named <the project's own package>** — same cause from the
    other side: the root is not on the path.
  * **no module named <a third-party package>** — not a code error; the
    package is not installed in the interpreter that ran it. Named with the
    pip name when it is known.

Everything else is a real error and goes to `locate`. The rule for adding a
case: it must be recognisable from the traceback text alone, with no model
and no guess, and the fix must be a command the operator runs, not a change
to the code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from . import packages

_REL_IMPORT = re.compile(
    r"ImportError: attempted relative import with no known parent package")
_NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z_][\w.]*)'")
#: the file the traceback was running when it died — the LAST "File ..." line
_FILE = re.compile(r'File "([^"]+)", line \d+')
#: the shell prompt line a Windows paste starts with: `D:\...\src>python main.py`
_PROMPT = re.compile(r"^(?P<cwd>[A-Za-z]:\\[^>\n]*)>\s*(?P<cmd>.+)$", re.M)


@dataclass(frozen=True)
class Verdict:
    kind: str          # relative-import | not-on-path | missing-package
    sentence: str      # what to tell the operator
    command: str = ""  # the command that runs it right, when there is one
    package: str = ""  # the pip name, for missing-package


def _packages_of(fs: Any) -> list[str]:
    """Top-level directories of the project that are Python packages."""
    out = []
    try:
        inits = fs.list("*/__init__.py")
    except Exception:                                     # noqa: BLE001
        return out
    for p in inits:
        head = str(p).replace("\\", "/").split("/")[0]
        if head and head not in out and not head.startswith("."):
            out.append(head)
    return out


def _module_of(path: str, pkgs: list[str]) -> str:
    """`D:\\x\\proj\\src\\main.py` → `src.main` when `src` is a package."""
    # a paste from a JSON log carries doubled backslashes; both split
    parts = [p for p in re.split(r"[\\/]+", str(path)) if p]
    for i, part in enumerate(parts):
        if part in pkgs:
            rest = parts[i:]
            if rest and rest[-1].endswith(".py"):
                rest[-1] = rest[-1][:-3]
            if rest and rest[-1] == "__init__":
                rest = rest[:-1]
            return ".".join(rest)
    return ""


def classify(text: str, fs: Any) -> Verdict | None:
    """The non-error this text is, or None when it is a real error."""
    text = text or ""
    pkgs = _packages_of(fs)
    files = _FILE.findall(text)
    dying = files[-1] if files else ""
    module = _module_of(dying, pkgs) if dying else ""
    entry = module or (f"{pkgs[0]}.main" if pkgs else "main")
    prompt = _PROMPT.search(text)
    cmd = f"python -m {entry}"

    if _REL_IMPORT.search(text):
        where = ""
        if prompt:
            where = (f" It was started from `{prompt.group('cwd')}` with "
                     f"`{prompt.group('cmd').strip()}`.")
        return Verdict(
            "relative-import",
            f"This is not a code error. `from .x import y` is a package "
            f"import, and a file run as a script (`python main.py`, or "
            f"`python src\\main.py`) has no package, so Python refuses it."
            f"{where} Run it as a module from the project root instead: "
            f"`{cmd}`. Changing the imports to absolute ones would break "
            f"the test runner, which imports the package.",
            command=cmd)

    m = _NO_MODULE.search(text)
    if m:
        name = m.group(1).split(".")[0]
        if name in pkgs:
            return Verdict(
                "not-on-path",
                f"This is not a code error. `{name}` is this project's own "
                f"package, and it was not found because the program was "
                f"not run from the project root (the folder that holds "
                f"`{name}\\`). Run it from there as a module: `{cmd}`.",
                command=cmd)
        pip = packages.pip_name(name) if hasattr(packages, "pip_name") \
            else name
        known = packages.is_known(name) if hasattr(packages, "is_known") \
            else False
        return Verdict(
            "missing-package",
            f"This is not a code error. The package `{name}` is not "
            f"installed in the Python that ran the program"
            + (f" — install it with `pip install {pip}` into that "
               f"interpreter (the project's .venv when it has one) and "
               f"run again." if known else
               f" — install it into that interpreter and run again; if "
               f"`{name}` is a module of this project, the file that "
               f"defines it is missing."),
            command=f"pip install {pip}" if known else "",
            package=pip if known else name)
    return None
