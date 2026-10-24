# SPDX-License-Identifier: Apache-2.0
"""What "a file, a folder, or the project" means to the tabs."""

from __future__ import annotations

from typing import Any

#: Never part of a scope: the engine's own state, environments, caches.
SKIP = (".venv/", "venv/", ".cc_state/", ".cc_journal/", ".cc_snapshots/",
        ".cc_tabs/", "__pycache__/", ".git/", "node_modules/", "site-packages/",
        ".mypy_cache/", ".pytest_cache/", ".ruff_cache/")
SOURCE_SUFFIXES = (".py",)


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").strip("/")


def _kept(path: str) -> bool:
    p = _norm(path) + ("/" if not path.endswith("/") else "")
    return not any(f"/{s}" in f"/{p}" for s in SKIP)


def files(fs: Any, scope: str | list[str] | tuple = "") -> list[str]:
    """The source files a scope names, in path order.

    `scope` is "" (the project), a file, a folder, or a list of those.
    Tests are included: the tabs decide what to do with them.
    """
    wanted = [scope] if isinstance(scope, str) else list(scope or [""])
    out: list[str] = []
    try:
        everything = [_norm(p) for p in fs.list("**/*")]
    except Exception:                                     # noqa: BLE001
        everything = []
    sources = [p for p in everything
               if p.endswith(SOURCE_SUFFIXES) and _kept(p)]
    for item in wanted:
        item = _norm(item)
        if not item:
            out.extend(sources)
            continue
        if item in sources:
            out.append(item)
            continue
        prefix = item + "/"
        out.extend(p for p in sources if p.startswith(prefix))
    seen: set[str] = set()
    unique = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def is_test(path: str) -> bool:
    name = _norm(path).rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py") or \
        "/tests/" in f"/{_norm(path)}"
