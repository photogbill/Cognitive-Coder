# SPDX-License-Identifier: Apache-2.0
"""A StoragePort whose state outlives the process: one JSON file per key.

`MemoryStorage` is right for tests and wrong for a CLI. The patcher keeps
its sequence counter and its transaction log in the StoragePort, so under
`MemoryStorage` both died with each `ccoder build`: `ccoder history` then
answered "Nothing has been changed" with `.cc_snapshots/0001-t1/` sitting
beside it, and the next build numbered its first transaction 1 again —
which makes M25's claim, that the numbering proves the log is linear,
false across runs.

One file per key, not one file for everything, so a write replaces only
the key it changes, and so a person can read `.cc_state/` with an editor.
Writes are atomic (temp file in the same directory, then `os.replace`), and
values are JSON-checked at `set` exactly as `MemoryStorage` checks them
(M17). Stdlib only (M48); nothing happens at import (M50).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .errors import PortError

#: Names Windows will not create as files, whatever the extension.
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
             *(f"LPT{i}" for i in range(10))}


class JsonFileStorage:
    """A StoragePort rooted at one directory, e.g. `<project>/.cc_state/`."""

    def __init__(self, directory: str) -> None:
        self._dir = Path(directory).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        """A file name for `key` that stays inside the directory.

        Readable when the key allows it; a hash suffix when it does not, so
        `a/b` and `a_b` — which sanitise alike — never share a file.
        """
        key = str(key)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key).strip(".") or "_"
        if (safe != key or len(safe) > 100
                or safe.split(".")[0].upper() in _RESERVED):
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
            safe = f"{safe[:100]}-{digest}"
        return self._dir / f"{safe}.json"

    def get(self, key: str, default: Any = None) -> Any:
        path = self._path(key)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default
        try:
            return json.loads(text)
        except ValueError as exc:
            # Not the default: a patcher handed a fresh counter would
            # number its next transaction 1 again, silently.
            raise PortError(
                f"The stored value {path} is not valid JSON, so it was not "
                f"used. Restore it or delete it; deleting it forgets what it "
                f"recorded.", str(exc)) from exc

    def set(self, key: str, value: Any) -> None:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"StoragePort values must be JSON-serialisable; {key!r} is "
                f"not ({exc}).") from exc
        path = self._path(key)
        fd, tmp = tempfile.mkstemp(dir=str(self._dir), prefix=".cc-",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)          # atomic within one filesystem
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def sqlite_path(self, name: str) -> str:
        return str(self._dir / f"{name}.sqlite3")
