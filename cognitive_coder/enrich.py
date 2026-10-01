# SPDX-License-Identifier: Apache-2.0
"""Deterministic enrichment of a repair prompt: the facts behind the error.

A traceback says what went wrong. It does not say what is true. On Oct 1
``main.py`` failed with ``Renderer.render() missing 1 required positional
argument: 'car_physics'`` — and the model, told only that, guessed again,
because the sentence it needed was never in front of it: *render takes
(self, screen, track_segments, car_physics)*. On Aug 8 ``CarState(speed=…)``
became ``CarState(current_speed=…)`` for the same reason. In every such case
the harness already KNEW the answer; it was in the codemap.

This module reads the diagnostics of the last attempt, recognises the error
shapes that name a symbol, looks that symbol up in the codemap (and, where
the codemap is silent, in the file itself), and returns a block of plain
facts for the repair prompt:

    [WHAT ACTUALLY EXISTS — use these names, do not invent others]
    `src.physics` does not define `CarState`. It defines:
        class CarPhysics
            # instance attributes: speed: float = 0.0, max_speed: …
            def __init__(self)
            def update(self, dt: float, …) -> None

Changing the INPUT is the first rung of the escalation ladder. It costs no
model call, it is exact, and it is what a retry at temperature 0.15 needs
before it can produce anything but the same answer.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

from .context import interface_lines, is_public_symbol
from .types import Diagnostic

HEADER = "[WHAT ACTUALLY EXISTS — use these names, do not invent others]"

#: Error shapes that name something the codemap can look up. Each pattern's
#: groups are read by the handler named in `_HANDLERS`.
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("import_name", re.compile(
        r"ImportError: cannot import name '(?P<name>\w+)' from "
        r"'(?P<module>[\w.]+)'")),
    ("no_module", re.compile(
        r"ModuleNotFoundError: No module named '(?P<module>[\w.]+)'")),
    ("bad_kwarg", re.compile(
        r"TypeError: (?P<callable>[\w.]+)\(\) got an unexpected keyword "
        r"argument '(?P<arg>\w+)'")),
    ("no_args", re.compile(
        r"TypeError: (?P<callable>[\w.]+)\(\) takes no arguments")),
    ("arity", re.compile(
        r"TypeError: (?P<callable>[\w.]+)\(\) (?:missing \d+ required "
        r"positional argument|takes \d+ positional argument)")),
    ("object_attr", re.compile(
        r"AttributeError: '(?P<cls>\w+)' object has no attribute "
        r"'(?P<attr>\w+)'")),
    ("module_attr", re.compile(
        r"AttributeError: module '(?P<module>[\w.]+)' has no attribute "
        r"'(?P<attr>\w+)'")),
    ("type_attr", re.compile(
        r"AttributeError: type object '(?P<cls>\w+)' has no attribute "
        r"'(?P<attr>\w+)'")),
    ("name_error", re.compile(
        r"NameError: name '(?P<name>\w+)' is not defined")),
)

MAX_FACTS = 4
MAX_LINES_PER_FACT = 18


def facts_for(diags: Sequence[Diagnostic], *, codemap: Any = None,
              fs: Any = None, lang: str = "python") -> str:
    """The enrichment block for these diagnostics, or "" when there is
    nothing to say. Python only for now: the error shapes are Python's."""
    if lang != "python" or not diags:
        return ""
    store = getattr(codemap, "store", None)
    facts: list[str] = []
    seen: set[str] = set()
    for d in diags:
        text = " ".join(x for x in (d.message, d.source_excerpt) if x)
        for kind, pattern in _PATTERNS:
            m = pattern.search(text)
            if not m:
                continue
            fact = _HANDLERS[kind](m.groupdict(), store=store, fs=fs)
            if fact and fact not in seen:
                seen.add(fact)
                facts.append(_clip(fact))
            break
        if len(facts) >= MAX_FACTS:
            break
    if not facts:
        return ""
    return HEADER + "\n" + "\n\n".join(facts)


# ---------------------------------------------------------------------------
# handlers — each returns one fact, or ""
# ---------------------------------------------------------------------------

def _import_name(g: dict, *, store: Any, fs: Any) -> str:
    path, rows = _module_rows(g["module"], store=store, fs=fs)
    if not path:
        return ""
    lines, _ = interface_lines(rows)
    body = "\n".join("    " + ln for ln in lines) or "    (nothing public)"
    near = _nearest(g["name"], [r["name"] for r in rows])
    hint = f" Did you mean `{near}`?" if near else ""
    return (f"`{g['module']}` ({path}) does not define `{g['name']}`.{hint} "
            f"It defines:\n{body}")


def _no_module(g: dict, *, store: Any, fs: Any) -> str:
    wanted = g["module"]
    paths = _project_modules(store=store, fs=fs)
    if not paths:
        return ""
    stem = wanted.split(".")[-1]
    near = [p for p in paths if p.rsplit("/", 1)[-1].rsplit(".", 1)[0] == stem]
    shown = near or paths[:12]
    how = ""
    if near:
        mod = near[0].rsplit(".", 1)[0].replace("/", ".")
        how = f" Import it as `from {mod} import …`."
    return (f"There is no module `{wanted}`.{how} The project's modules "
            f"are: {', '.join(shown)}")


def _bad_kwarg(g: dict, *, store: Any, fs: Any) -> str:
    return _callable_fact(g["callable"], store=store, fs=fs,
                          lead=f"`{g['callable']}` has no parameter "
                               f"`{g['arg']}`.")


def _no_args(g: dict, *, store: Any, fs: Any) -> str:
    return _callable_fact(g["callable"], store=store, fs=fs,
                          lead=f"`{g['callable']}` takes no arguments.")


def _arity(g: dict, *, store: Any, fs: Any) -> str:
    return _callable_fact(g["callable"], store=store, fs=fs,
                          lead=f"The call to `{g['callable']}` does not match "
                               f"its signature.")


def _object_attr(g: dict, *, store: Any, fs: Any) -> str:
    rows = _class_rows(g["cls"], store=store)
    if not rows:
        return ""
    lines, _ = interface_lines(rows)
    body = "\n".join("    " + ln for ln in lines)
    attrs = [r["name"].split(".")[-1] for r in rows
             if r.get("kind") in ("attribute", "field")]
    near = _nearest(g["attr"], attrs)
    hint = f" Did you mean `{near}`?" if near else ""
    return (f"`{g['cls']}` has no attribute `{g['attr']}`.{hint} "
            f"Its interface is:\n{body}")


def _type_attr(g: dict, *, store: Any, fs: Any) -> str:
    return _object_attr(g, store=store, fs=fs)


def _module_attr(g: dict, *, store: Any, fs: Any) -> str:
    path, rows = _module_rows(g["module"], store=store, fs=fs)
    if not path:
        return ""
    lines, _ = interface_lines(rows)
    body = "\n".join("    " + ln for ln in lines) or "    (nothing public)"
    near = _nearest(g["attr"], [r["name"] for r in rows])
    hint = f" Did you mean `{near}`?" if near else ""
    return (f"`{g['module']}` ({path}) has no `{g['attr']}`.{hint} "
            f"It defines:\n{body}")


def _name_error(g: dict, *, store: Any, fs: Any) -> str:
    if store is None:
        return ""
    name = g["name"]
    try:
        rows = store.find(name)
    except Exception:                                    # noqa: BLE001
        return ""
    hits = [r for r in rows if r["name"].split(".")[-1] == name
            and r.get("kind") in ("class", "function")]
    if not hits:
        return (f"Nothing in this project defines `{name}`. Either it is "
                f"from the standard library and needs an import, or it "
                f"does not exist and must not be used.")
    r = hits[0]
    mod = r["path"].rsplit(".", 1)[0].replace("/", ".")
    return (f"`{name}` is defined in {r['path']} as `{r['signature']}`. "
            f"Import it: `from {mod} import {name}`.")


_HANDLERS = {
    "import_name": _import_name, "no_module": _no_module,
    "bad_kwarg": _bad_kwarg, "no_args": _no_args, "arity": _arity,
    "object_attr": _object_attr, "module_attr": _module_attr,
    "type_attr": _type_attr, "name_error": _name_error,
}


# ---------------------------------------------------------------------------
# lookups
# ---------------------------------------------------------------------------

def _callable_fact(dotted: str, *, store: Any, fs: Any, lead: str) -> str:
    """`CarPhysics.__init__`, `Renderer.render`, `build_track` → its
    signature (and, for a constructor, the class's fields/attributes)."""
    if store is None:
        return ""
    cls, _, member = dotted.rpartition(".")
    if member in ("__init__", "__new__") and cls:
        rows = _class_rows(cls, store=store)
        if not rows:
            return ""
        lines, _ = interface_lines(rows)
        return lead + " The class is:\n" + "\n".join("    " + ln
                                                     for ln in lines)
    try:
        rows = store.find(dotted)
    except Exception:                                    # noqa: BLE001
        return ""
    exact = [r for r in rows if r["name"] == dotted] or \
        [r for r in rows if r["name"].split(".")[-1] == member or
         r["name"] == member]
    if not exact:
        return ""
    r = exact[0]
    where = f" ({r['path']}:{r['line']})" if r.get("path") else ""
    return f"{lead} Its exact signature{where} is:\n    {r['signature']}"


def _class_rows(cls: str, *, store: Any) -> list[dict]:
    """The class row plus every member row, from whichever file holds it."""
    if store is None:
        return []
    try:
        hits = [r for r in store.find(cls)
                if r["name"] == cls or r["name"].endswith("." + cls)]
        hits = [r for r in hits if r.get("kind") in ("class", "struct")]
        if not hits:
            return []
        head = hits[0]
        rows = store.symbols_in(head["path"])
    except Exception:                                    # noqa: BLE001
        return []
    name = head["name"]
    return [r for r in rows
            if r["name"] == name or r["name"].startswith(name + ".")]


def _module_rows(dotted: str, *, store: Any, fs: Any
                 ) -> tuple[str, list[dict]]:
    """(path, public symbol rows) for `src.physics`, or ("", [])."""
    candidates = [dotted.replace(".", "/") + ".py",
                  dotted.split(".")[-1] + ".py",
                  dotted.replace(".", "/") + "/__init__.py"]
    path = ""
    for c in candidates:
        try:
            if fs is not None and fs.exists(c):
                path = c
                break
        except Exception:                                # noqa: BLE001
            continue
    if not path and store is not None:
        try:
            known = {row["path"] for row in store.files()}
        except Exception:                                # noqa: BLE001
            known = set()
        path = next((c for c in candidates if c in known), "")
    if not path:
        return "", []
    rows: list[dict] = []
    if store is not None:
        try:
            rows = store.symbols_in(path)
        except Exception:                                # noqa: BLE001
            rows = []
    if not rows and fs is not None:
        # Not indexed (yet): read the file itself. Exact beats absent.
        try:
            from .codemap import parse_python
            symbols, _e, _u = parse_python.parse(fs.read(path), path)
            rows = [{"name": s.name, "kind": s.kind, "line": s.line,
                     "end_line": s.end_line, "signature": s.signature,
                     "docstring": s.docstring, "approximate": s.approximate}
                    for s in symbols if s.kind != "module"]
        except Exception:                                # noqa: BLE001
            rows = []
    return path, [r for r in rows if is_public_symbol(r["name"])]


def _project_modules(*, store: Any, fs: Any) -> list[str]:
    paths: list[str] = []
    if store is not None:
        try:
            paths = [row["path"] for row in store.files()
                     if str(row["path"]).endswith(".py")]
        except Exception:                                # noqa: BLE001
            paths = []
    if not paths and fs is not None:
        try:
            paths = [str(p).replace("\\", "/") for p in fs.list("*.py")]
        except Exception:                                # noqa: BLE001
            paths = []
    return sorted(p for p in paths
                  if not any(part.startswith(".") for part in p.split("/")))


def _nearest(name: str, names: Sequence[str]) -> str:
    """The closest public short name, when it is close enough to be a typo
    or a rename (`CarState` → `CarPhysics`, `w` → `width`)."""
    import difflib
    shorts = sorted({n.split(".")[-1] for n in names
                     if is_public_symbol(n)})
    if not shorts:
        return ""
    close = difflib.get_close_matches(name, shorts, n=1, cutoff=0.6)
    if close:
        return close[0]
    # A prefix match catches `pos` → `position_x` style renames.
    pref = [s for s in shorts if s.startswith(name) or name.startswith(s)]
    if len(pref) == 1:
        return pref[0]
    # Two class names sharing a stem — `CarState` → `CarPhysics` — are a
    # rename, and the one with the longest shared prefix is the answer.
    if name[:1].isupper():
        def shared(s: str) -> int:
            n = 0
            for a, b in zip(name, s, strict=False):
                if a != b:
                    break
                n += 1
            return n
        best = max((s for s in shorts if s[:1].isupper()), key=shared,
                   default="")
        if best and shared(best) >= 3:
            return best
    return ""


def _clip(fact: str) -> str:
    lines = fact.splitlines()
    if len(lines) <= MAX_LINES_PER_FACT:
        return fact
    return "\n".join(lines[:MAX_LINES_PER_FACT]) + \
        f"\n    … ({len(lines) - MAX_LINES_PER_FACT} more lines)"
