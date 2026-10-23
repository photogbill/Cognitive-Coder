# SPDX-License-Identifier: Apache-2.0
"""What a LIBRARY actually exports: enrichment for the names a small model
invents on pygame, numpy, requests and the standard library.

`enrich.py` answers "what does THIS PROJECT define" from the codemap. It
has nothing to say when the error is

    AttributeError: module 'pygame.draw' has no attribute 'circle_outline'
    ImportError: cannot import name 'load_image' from 'pygame.image'
    AttributeError: module 'pygame' has no attribute 'gfxdraw'

— and these are the small model's signature failure: an API that sounds
right. Told only the error, it guesses another name that sounds right.
Told the real list, it picks from it.

The list is read from the project's own interpreter (the venv the code
runs in, through the exec Port), so it is the installed version's truth,
not the model's memory of some version. The lookup imports the library
module — the same import the program already performed when it hit the
error — and nothing else: no project code is executed.

A submodule that was never imported (`pygame.gfxdraw`) is the one case
where the fix is not a different name but an import line, and the fact
says so.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import re
from typing import Any

from . import langs
from .enrich import _nearest
from .types import Diagnostic

HEADER = "[WHAT THE LIBRARY ACTUALLY EXPORTS — use these names, do not invent others]"
_MODULE = re.compile(r"^[A-Za-z_][\w.]*$")
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("module_attr", re.compile(
        r"AttributeError: module '(?P<module>[\w.]+)' has no attribute "
        r"'(?P<attr>\w+)'")),
    ("import_name", re.compile(
        r"ImportError: cannot import name '(?P<attr>\w+)' from "
        r"'(?P<module>[\w.]+)'")),
)
MAX_FACTS = 2
MAX_NAMES = 40
_TIMEOUT = 25.0

#: Runs in the project's interpreter. argv[1] is the module, argv[2] the
#: attribute asked about. Prints one JSON object.
_SCRIPT = r'''
import importlib, importlib.util, inspect, json, sys
name, attr = sys.argv[1], sys.argv[2]
out = {"module": name, "attr": attr, "ok": False}
try:
    mod = importlib.import_module(name)
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out)); sys.exit(0)
out["ok"] = True
out["file"] = str(getattr(mod, "__file__", "") or "")
try:
    out["submodule"] = importlib.util.find_spec(f"{name}.{attr}") is not None
except Exception:
    out["submodule"] = False
names = []
public = getattr(mod, "__all__", None) or [n for n in dir(mod)
                                           if not n.startswith("_")]
for n in sorted(set(public)):
    try:
        obj = getattr(mod, n)
    except Exception:
        continue
    if inspect.ismodule(obj):
        kind = "module"
    elif inspect.isclass(obj):
        kind = "class"
    elif callable(obj):
        kind = "function"
    else:
        kind = "constant"
    sig = ""
    if kind in ("class", "function"):
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "(...)"
    doc = (inspect.getdoc(obj) or "").strip().splitlines()
    if len(sig) > 88:
        sig = sig[:85] + "…)"
    names.append([n, kind, sig, (doc[0] if doc else "")[:60]])
out["names"] = names
print(json.dumps(out))
'''

_cache: dict[tuple[str, str, str], dict] = {}


def _is_project_module(dotted: str, fs: Any) -> bool:
    if fs is None:
        return False
    for c in (dotted.replace(".", "/") + ".py",
              dotted.replace(".", "/") + "/__init__.py",
              dotted.split(".")[0] + ".py",
              dotted.split(".")[0] + "/__init__.py"):
        try:
            if fs.exists(c):
                return True
        except Exception:                                    # noqa: BLE001
            continue
    return False


def lookup(module: str, attr: str, *, ex: Any, root: str = "",
           python: str = "") -> dict:
    """The module's public names, from the project's interpreter."""
    if not _MODULE.match(module) or not re.match(r"^\w+$", attr):
        return {"ok": False, "error": "not a module name"}
    key = (python, module, attr)
    if key in _cache:
        return _cache[key]
    py = python or langs.get("python").which_run(ex)
    if not py or py == "-":
        return {"ok": False, "error": "no python"}
    try:
        proc = ex.run([py, "-c", _SCRIPT, module, attr], cwd=root or ".",
                      timeout=_TIMEOUT)
    except Exception as exc:                                 # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    text = (proc.stdout or "").strip().splitlines()
    data: dict = {"ok": False, "error": "no answer"}
    for line in reversed(text):
        if line.startswith("{"):
            try:
                data = json.loads(line)
                break
            except ValueError:
                continue
    _cache[key] = data
    return data


def fact_for(module: str, attr: str, *, ex: Any, fs: Any = None,
             root: str = "", python: str = "", kind: str = "module_attr"
             ) -> str:
    if _is_project_module(module, fs):
        return ""                      # enrich.py's job: the codemap knows
    data = lookup(module, attr, ex=ex, root=root, python=python)
    if not data.get("ok"):
        return ""
    if data.get("submodule"):
        return (f"`{module}.{attr}` is a SUBMODULE of `{module}`, not an "
                f"attribute: it exists but must be imported on its own — "
                f"write `import {module}.{attr}` at the top of the file.")
    rows = data.get("names") or []
    names = [r[0] for r in rows]
    near = _nearest(attr, names)
    hint = f" Did you mean `{near}`?" if near else ""
    verb = "has no" if kind == "module_attr" else "does not export"
    shown = []
    # the nearest names first, then the rest, capped
    order = sorted(rows, key=lambda r: (r[0] != near, r[1] == "module", r[0]))
    for n, k, sig, doc in order[:MAX_NAMES]:
        line = f"    {k} {n}{sig if k in ('class', 'function') else ''}"
        if doc:
            line += f"  # {doc}"
        shown.append(line[:150])
    more = len(rows) - len(shown)
    body = "\n".join(shown) or "    (nothing public)"
    if more > 0:
        body += f"\n    … and {more} more"
    return (f"`{module}` (installed here) {verb} `{attr}`.{hint} "
            f"It exports:\n{body}")


def facts_for(diags: Sequence[Diagnostic], *, ex: Any, fs: Any = None,
              lang: str = "python", root: str = "", python: str = "") -> str:
    """The library block for these diagnostics, or ""."""
    if lang != "python" or not diags or ex is None:
        return ""
    facts: list[str] = []
    seen: set[tuple[str, str]] = set()
    for d in diags:
        text = " ".join(x for x in (d.message, d.source_excerpt) if x)
        for kind, pattern in _PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            key = (m.group("module"), m.group("attr"))
            if key in seen:
                break
            seen.add(key)
            fact = fact_for(key[0], key[1], ex=ex, fs=fs, root=root,
                            python=python, kind=kind)
            if fact:
                facts.append(fact)
            break
        if len(facts) >= MAX_FACTS:
            break
    if not facts:
        return ""
    return HEADER + "\n" + "\n\n".join(facts)
