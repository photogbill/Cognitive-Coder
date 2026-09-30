# SPDX-License-Identifier: Apache-2.0
"""The SQLite registry behind the codemap — and the honesty of `unresolved`.

Schema (§6.7):

    files(id, path, lang, mtime, hash, indexed_at)
    symbols(id, file_id, name, kind, line, end_line, signature, docstring,
            parent_id)
    edges(src_symbol_id, dst_symbol_id, kind)   -- calls | imports | contains
    unresolved(src_symbol_id, name, kind)       -- calls we couldn't bind

**`unresolved` is the table that makes this trustworthy.** A call graph that
silently drops what it could not bind looks complete and isn't — and a model
told "nothing calls this function" when six things do will cheerfully delete
it. Everything that could not be resolved is kept, counted, and reported as a
resolution rate the operator can see.

Two design decisions worth stating:

  * **The path is through `StoragePort.sqlite_path`** (C2). The host decides
    where state lives; the core does not go looking for a home directory.
  * **The query interface is NEVER stale** (M30). This reads live SQLite
    every time, which is what makes it safe for the *injected text summary*
    to lag by an epoch (G.7): staleness in a cached hint costs at most one
    extra tool call, never a wrong answer. Re-index on every write; that is
    cheap because `hash` and `mtime` make a rescan incremental.

Regression memory (F10) lives here too, in `fixes`: when a repair succeeds,
the pair (normalised diagnostic signature → the shape of the fix that worked)
is recorded, per project. Over months the tool gets measurably better at THIS
codebase with no training, no network and no data leaving the machine — which
for an air-gapped tool is a genuinely distinctive property, and the only part
of this design that improves with use. It is kept small, inspectable and
clearable: a learned "fix" that is wrong must be as easy to delete as it was
to acquire.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import sqlite3
import threading
import time

from ..types import CodemapStats, Symbol

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    lang TEXT NOT NULL DEFAULT '',
    mtime REAL NOT NULL DEFAULT 0,
    hash TEXT NOT NULL DEFAULT '',
    approximate INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    line INTEGER NOT NULL DEFAULT 0,
    end_line INTEGER NOT NULL DEFAULT 0,
    signature TEXT NOT NULL DEFAULT '',
    docstring TEXT NOT NULL DEFAULT '',
    parent_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
    approximate INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    src_symbol_id INTEGER NOT NULL,
    dst_symbol_id INTEGER NOT NULL,
    kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS unresolved (
    src_symbol_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fixes (
    id INTEGER PRIMARY KEY,
    signature TEXT NOT NULL,
    shape TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 1,
    last_seen REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS ix_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src_symbol_id);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst_symbol_id);
CREATE INDEX IF NOT EXISTS ix_unresolved_name ON unresolved(name);
CREATE INDEX IF NOT EXISTS ix_fixes_sig ON fixes(signature);
"""


def _enclosing_class(src: str, kinds: dict) -> str:
    """The class `src` is a method of, from its dotted name."""
    parts = str(src).split(".")
    for i in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:i])
        if kinds.get(prefix) == "class":
            return prefix
    return ""


def _import_names(edges: Sequence[tuple], kinds: dict) -> dict[str, str]:
    """Local name → imported target, from this file's `imports` edges.

    `import pkg.util` makes `pkg.util` usable; `from pkg import util` makes
    `util` mean `pkg.util`; `from pkg.util import helper` makes `helper`
    mean `pkg.util.helper`. The edge does not say which form it came from,
    so both the full target and its last component are mapped — harmless,
    because a target outside the project binds to nothing either way.
    """
    out: dict[str, str] = {}
    for src, dst, kind in edges:
        if kind != "imports" or kinds.get(src) != "module":
            continue
        target = str(dst)
        out.setdefault(target, target)
        out.setdefault(target.rsplit(".", 1)[-1], target)
    return out


def _module_names(path: str) -> list[str]:
    """The names other files may use for this module: `a/b/c.py` →
    `a.b.c`, `b.c`, `c` (a src layout imports without its first part)."""
    p = str(path or "").replace("\\", "/").strip("/")
    for suffix in ("/__init__.py", ".py"):
        if p.endswith(suffix):
            p = p[: -len(suffix)]
            break
    else:
        return []
    parts = [x for x in p.split("/") if x]
    return [".".join(parts[i:]) for i in range(len(parts))]


#: Bumped when the on-disk shape changes. 1 was the spec's schema (no
#: `approximate` columns); 2 added them, module rows and the per-epoch
#: architecture snapshot. Recorded in `meta` so the next change can tell
#: what it is opening instead of crashing on it — an older database made
#: `put_file` raise "table files has no column named approximate".
SCHEMA_VERSION = 2

#: Columns added after a table first shipped: (table, column, DDL). Adding
#: a column to an existing SQLite table needs a default, which each has.
_ADDED_COLUMNS = (
    ("files", "lang", "TEXT NOT NULL DEFAULT ''"),
    ("files", "mtime", "REAL NOT NULL DEFAULT 0"),
    ("files", "hash", "TEXT NOT NULL DEFAULT ''"),
    ("files", "approximate", "INTEGER NOT NULL DEFAULT 0"),
    ("files", "indexed_at", "REAL NOT NULL DEFAULT 0"),
    ("symbols", "kind", "TEXT NOT NULL DEFAULT ''"),
    ("symbols", "line", "INTEGER NOT NULL DEFAULT 0"),
    ("symbols", "end_line", "INTEGER NOT NULL DEFAULT 0"),
    ("symbols", "signature", "TEXT NOT NULL DEFAULT ''"),
    ("symbols", "docstring", "TEXT NOT NULL DEFAULT ''"),
    ("symbols", "parent_id", "INTEGER"),
    ("symbols", "approximate", "INTEGER NOT NULL DEFAULT 0"),
    ("fixes", "hits", "INTEGER NOT NULL DEFAULT 1"),
    ("fixes", "last_seen", "REAL NOT NULL DEFAULT 0"),
)

#: Milliseconds a writer waits for another's lock before failing. Two
#: sessions on one project are a normal thing for a host to allow.
BUSY_TIMEOUT_MS = 5000


class _SharedConnection(sqlite3.Connection):
    """A connection any thread may use, one statement at a time.

    A host builds the Session where its UI lives and runs it on a worker —
    ATK's panel does exactly that. The default connection belongs to the
    thread that opened it, so every codemap call from the build raised
    `ProgrammingError: SQLite objects created in a thread can only be used
    in that same thread`: the tools answered "That tool call failed", and
    indexing silently stopped. With `check_same_thread=False` the module
    allows sharing when SQLite is serialized (`sqlite3.threadsafety == 3`),
    and the lock keeps one thread's statements and commits from
    interleaving with another's.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()

    def execute(self, *args, **kwargs):
        with self.lock:
            return super().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        with self.lock:
            return super().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        with self.lock:
            return super().executescript(*args, **kwargs)

    def commit(self) -> None:
        with self.lock:
            super().commit()

    def rollback(self) -> None:
        with self.lock:
            super().rollback()


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


class Store:
    """The codemap's database. One per project."""

    def __init__(self, path: str) -> None:
        self.path = path
        shared = sqlite3.threadsafety == 3
        self.db = sqlite3.connect(
            path, timeout=BUSY_TIMEOUT_MS / 1000,
            factory=_SharedConnection if shared else sqlite3.Connection,
            check_same_thread=not shared)
        self.db.row_factory = sqlite3.Row
        # WAL lets a reader proceed while another session writes, and
        # busy_timeout makes a writer WAIT for a lock rather than fail at
        # once. Both are best-effort: a database in a read-only or unusual
        # location may refuse WAL, and still works without it.
        for pragma in ("PRAGMA journal_mode=WAL",
                       f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}"):
            try:
                self.db.execute(pragma)
            except sqlite3.DatabaseError:
                pass
        statements = [x.strip() for x in SCHEMA.split(";") if x.strip()]
        tables = [x for x in statements if x.startswith("CREATE TABLE")]
        indexes = [x for x in statements if x.startswith("CREATE INDEX")]
        self.db.executescript(";\n".join(tables) + ";")
        self._migrate()
        self.db.executescript(";\n".join(indexes) + ";")
        self.db.commit()

    def _migrate(self) -> None:
        """Bring an older database up to `SCHEMA_VERSION`, additively."""
        for table, column, ddl in _ADDED_COLUMNS:
            have = {r[1] for r in self.db.execute(
                f"PRAGMA table_info({table})")}
            if have and column not in have:
                self.db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),))

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:                                # noqa: BLE001
            pass

    # -- meta / epochs ----------------------------------------------------
    def meta(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?",
                              (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)))
        self.db.commit()

    @property
    def epoch(self) -> int:
        return int(self.meta("epoch", "0") or 0)

    def bump_epoch(self, why: str = "") -> int:
        """A new epoch invalidates the cached prompt prefix (G.7.2).

        Deliberately explicit and rare: session start, a replan, N files
        changed, a blast-radius hit on the current target, a model change, or
        the operator asking. Bumping it on every write would mean never
        benefiting from the prefix cache at all.
        """
        n = self.epoch + 1
        self.set_meta("epoch", str(n))
        self.set_meta("epoch_reason", why)
        self.set_meta("epoch_at", str(time.time()))
        self.set_meta("changed_since_epoch", "")
        # The architecture block is SNAPSHOTTED here, and served from the
        # snapshot until the next bump. It used to be rendered live, so its
        # bytes changed whenever any file's symbol set changed — i.e. on
        # every patch — while `should_bump_epoch` said no: the "epoch-
        # scoped" part of the cached prefix was not epoch-scoped at all.
        from .zoom import architecture_prefix
        self.set_meta("architecture_prefix", architecture_prefix(self))
        self.set_meta("architecture_epoch", str(n))
        return n

    def architecture_snapshot(self) -> str | None:
        """This epoch's architecture block, or None if none was taken."""
        if self.meta("architecture_epoch") != str(self.epoch):
            return None
        return self.meta("architecture_prefix") or None

    def note_change(self, path: str) -> list[str]:
        """Record a file changed since the epoch snapshot was taken.

        This list is what the volatile tail declares (G.7.3): *"three files
        have changed since this snapshot; call `search_codemap` rather than
        trusting the summary above."* It goes in the TAIL, never the prefix —
        a note in the prefix would change the prefix bytes and invalidate the
        very cache it describes.
        """
        current = [p for p in
                   (self.meta("changed_since_epoch", "") or "").split("\n")
                   if p]
        if path not in current:
            current.append(path)
        self.set_meta("changed_since_epoch", "\n".join(current))
        return current

    def changed_since_epoch(self) -> list[str]:
        return [p for p in (self.meta("changed_since_epoch", "") or "")
                .split("\n") if p]

    # -- indexing ---------------------------------------------------------
    def needs_index(self, path: str, text: str) -> bool:
        row = self.db.execute("SELECT hash FROM files WHERE path=?",
                              (path,)).fetchone()
        return not row or row["hash"] != content_hash(text)

    def put_file(self, path: str, lang: str, text: str,
                 symbols: Sequence[Symbol], edges: Sequence[tuple],
                 unresolved: Sequence[tuple]) -> int:
        """Replace everything known about one file, atomically.

        Replace rather than merge: a symbol deleted from the source must
        disappear from the map, and a merge cannot tell a deletion from an
        absence.
        """
        approximate = int(any(s.approximate for s in symbols))
        cur = self.db.execute("SELECT id FROM files WHERE path=?", (path,))
        row = cur.fetchone()
        if row:
            file_id = row["id"]
            self.db.execute(
                "UPDATE files SET lang=?, hash=?, mtime=?, indexed_at=?, "
                "approximate=? WHERE id=?",
                (lang, content_hash(text), time.time(), time.time(),
                 approximate, file_id))
            ids = [r["id"] for r in self.db.execute(
                "SELECT id FROM symbols WHERE file_id=?", (file_id,))]
            if ids:
                marks = ",".join("?" * len(ids))
                self.db.execute(
                    f"DELETE FROM edges WHERE src_symbol_id IN ({marks})",
                    ids)
                self.db.execute(
                    f"DELETE FROM unresolved WHERE src_symbol_id IN ({marks})",
                    ids)
            self.db.execute("DELETE FROM symbols WHERE file_id=?", (file_id,))
        else:
            cur = self.db.execute(
                "INSERT INTO files(path,lang,mtime,hash,approximate,"
                "indexed_at) VALUES(?,?,?,?,?,?)",
                (path, lang, time.time(), content_hash(text), approximate,
                 time.time()))
            file_id = int(cur.lastrowid)

        name_to_id: dict[str, int] = {}
        for s in symbols:
            cur = self.db.execute(
                "INSERT INTO symbols(file_id,name,kind,line,end_line,"
                "signature,docstring,approximate) VALUES(?,?,?,?,?,?,?,?)",
                (file_id, s.name, s.kind, s.line, s.end_line, s.signature,
                 s.docstring, int(s.approximate)))
            name_to_id[s.name] = int(cur.lastrowid)
        for s in symbols:
            if s.parent and s.parent in name_to_id:
                self.db.execute("UPDATE symbols SET parent_id=? WHERE id=?",
                                (name_to_id[s.parent], name_to_id[s.name]))

        kinds = {s.name: s.kind for s in symbols}
        imports = _import_names(edges, kinds)
        # The SOURCE of every edge is in this file by construction, so it is
        # looked up here and nowhere else. It used to fall back to a project-
        # wide suffix match, which attributed `import csv` in src/stats.py to
        # a method `R.stats` in another file.
        for src, dst, kind in edges:
            src_id = name_to_id.get(src)
            if not src_id:
                continue
            if kind == "imports":
                dst_id = self._resolve_import(str(dst))
            else:
                dst_id = (name_to_id.get(dst)
                          or self._resolve(str(dst), src, file_id, kinds,
                                           imports))
            if dst_id and dst_id != src_id:
                self.db.execute(
                    "INSERT INTO edges(src_symbol_id,dst_symbol_id,kind) "
                    "VALUES(?,?,?)", (src_id, dst_id, kind))
            else:
                # It could not be bound. It is KEPT, not dropped (§6.7).
                self.db.execute(
                    "INSERT INTO unresolved(src_symbol_id,name,kind) "
                    "VALUES(?,?,?)", (src_id, str(dst), kind))
        for src, name, kind in unresolved:
            src_id = name_to_id.get(src)
            if not src_id:
                continue
            # Binding runs in BOTH directions, and forgetting this one is a
            # silent, plausible bug: `_rebind` below catches "a definition
            # arrived for a call we already knew about", but a call arriving
            # for a definition already indexed needs looking up now. Without
            # this, `callers_of` returns nothing for every cross-file call
            # whose target happened to be indexed first — which looks like a
            # project with no call graph rather than like a bug.
            dst_id = self._resolve(str(name), src, file_id, kinds, imports)
            if dst_id and dst_id != src_id:
                self.db.execute(
                    "INSERT INTO edges(src_symbol_id,dst_symbol_id,kind) "
                    "VALUES(?,?,?)", (src_id, dst_id, kind))
            else:
                self.db.execute(
                    "INSERT INTO unresolved(src_symbol_id,name,kind) "
                    "VALUES(?,?,?)", (src_id, str(name), kind))
        self.db.commit()
        self.note_change(path)
        self._rebind(path, symbols, name_to_id)
        return file_id

    # -- binding ------------------------------------------------------------
    #
    # WHY THIS IS STRICTER THAN IT WAS. Binding used to fall back to
    # `name LIKE '%.short'` everywhere, ordered by id — "the first symbol in
    # the database whose last component matches". Observed: `self.save()` in
    # `Doc` bound to `Db.save` in another file; `subprocess.run(...)` was
    # re-bound to a later `class Job: def run`, lifting the resolution rate
    # from 0% to 67% on an edge that does not exist and fabricating a blast
    # radius. A wrong edge is worse than an unresolved one: unresolved is
    # counted and reported, a wrong edge is believed.
    #
    # So, in order: `self.x`/`cls.x` against the enclosing class; the exact
    # name in the same file; for a plain name imported with `from m import
    # x`, `x` in module m's file (and nothing else — an import from outside
    # the project is not a licence to bind to a project symbol of the same
    # name); the exact name project-wide; and for `mod.x` where `mod` is an
    # imported PROJECT module, `x` in that module's file.

    def _resolve(self, name: str, src: str, file_id: int, kinds: dict,
                 imports: dict[str, str]) -> int | None:
        head, _, rest = name.partition(".")
        if head in ("self", "cls"):
            cls = _enclosing_class(src, kinds)
            if cls and rest:
                return self._symbol_in(file_id, f"{cls}.{rest}")
            return None
        local = self._symbol_in(file_id, name)
        if local:
            return local
        if not rest:
            target = imports.get(name)
            if target:
                mod, _, sym = target.rpartition(".")
                fid = self._file_for_module(mod) if mod else None
                return self._symbol_in(fid, sym) if fid else None
            return self._exact(name)
        exact = self._exact(name)
        if exact:
            return exact
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            target = imports.get(".".join(parts[:i]))
            if not target:
                continue
            fid = self._file_for_module(target)
            return self._symbol_in(fid, ".".join(parts[i:])) if fid else None
        return None

    def _resolve_import(self, target: str) -> int | None:
        """An import target → that module's row, or the symbol it names."""
        fid = self._file_for_module(target)
        if fid:
            return self._module_row(fid)
        mod, _, sym = target.rpartition(".")
        fid = self._file_for_module(mod) if mod else None
        if not fid:
            return None
        return self._symbol_in(fid, sym) or self._module_row(fid)

    def _symbol_in(self, file_id: int | None, name: str) -> int | None:
        if not file_id:
            return None
        row = self.db.execute(
            "SELECT id FROM symbols WHERE file_id=? AND name=? "
            "AND kind != 'module' ORDER BY id LIMIT 1",
            (file_id, name)).fetchone()
        return int(row["id"]) if row else None

    def _exact(self, name: str) -> int | None:
        row = self.db.execute(
            "SELECT id FROM symbols WHERE name=? AND kind != 'module' "
            "ORDER BY id LIMIT 1", (name,)).fetchone()
        return int(row["id"]) if row else None

    def _module_row(self, file_id: int) -> int | None:
        row = self.db.execute(
            "SELECT id FROM symbols WHERE file_id=? AND kind='module' "
            "LIMIT 1", (file_id,)).fetchone()
        return int(row["id"]) if row else None

    def _file_for_module(self, module: str) -> int | None:
        """The indexed file implementing `module` (`pkg.util` → pkg/util.py,
        pkg/util/__init__.py, or src/pkg/util.py), or a file by the literal
        name (a C `#include "util.h"`). None for anything outside the
        project, which is the point."""
        module = (module or "").strip()
        if not module or module.startswith("."):
            return None
        stem = module.replace(".", "/")
        exact = (f"{stem}.py", f"{stem}/__init__.py", module)
        row = self.db.execute(
            "SELECT id FROM files WHERE path IN (?,?,?) OR path LIKE ? "
            "OR path LIKE ? OR path LIKE ? ORDER BY length(path) LIMIT 1",
            (*exact, f"%/{stem}.py", f"%/{stem}/__init__.py",
             f"%/{module}")).fetchone()
        return int(row["id"]) if row else None

    def _rebind(self, path: str, symbols: Sequence[Symbol],
                name_to_id: dict[str, int]) -> None:
        """Late binding: an unresolved call that now HAS a target → an edge.

        This is what makes the graph improve as more of the project is
        indexed. File A calling `B.load` before B was indexed is unresolved;
        the moment B lands, it becomes a real edge — and the resolution rate
        going up is the visible sign the map is getting more complete.

        EXACT names only — the name itself, or the name qualified by this
        file's module (`util.helper`, `pkg.util.helper`) — plus `self.x` /
        `cls.x` for a method `x`, since an inherited method is defined in a
        file the caller never names. A bare suffix match is what rebound
        `subprocess.run` to `Job.run`.
        """
        if not name_to_id:
            return
        modules = _module_names(path)
        for s in symbols:
            sym_id = name_to_id.get(s.name)
            if not sym_id:
                continue
            if s.kind == "module":
                names = list(modules)
                where = "kind='imports'"
            else:
                names = [s.name] + [f"{m}.{s.name}" for m in modules]
                where = "1=1"
                if "." in s.name and s.kind == "method":
                    short = s.name.rsplit(".", 1)[-1]
                    names += [f"self.{short}", f"cls.{short}"]
            if not names:
                continue
            marks = ",".join("?" * len(names))
            rows = self.db.execute(
                f"SELECT rowid, src_symbol_id, kind FROM unresolved "
                f"WHERE name IN ({marks}) AND {where}", names).fetchall()
            for row in rows:
                if row["src_symbol_id"] == sym_id:
                    continue
                self.db.execute(
                    "INSERT INTO edges(src_symbol_id,dst_symbol_id,kind) "
                    "VALUES(?,?,?)",
                    (row["src_symbol_id"], sym_id, row["kind"]))
                self.db.execute("DELETE FROM unresolved WHERE rowid=?",
                                (row["rowid"],))
        self.db.commit()

    def forget(self, path: str) -> None:
        """Remove a file that no longer exists from the map.

        Its callers are not silently disconnected: an edge INTO the file
        becomes an unresolved call again, so the resolution rate drops and
        `callers_of` stops claiming a target that is gone.
        """
        row = self.db.execute("SELECT id FROM files WHERE path=?",
                              (path,)).fetchone()
        if not row:
            return
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM symbols WHERE file_id=?", (row["id"],))]
        if ids:
            marks = ",".join("?" * len(ids))
            self.db.execute(
                f"INSERT INTO unresolved(src_symbol_id,name,kind) "
                f"SELECT e.src_symbol_id, d.name, e.kind FROM edges e "
                f"JOIN symbols d ON d.id = e.dst_symbol_id "
                f"WHERE e.dst_symbol_id IN ({marks}) "
                f"AND e.src_symbol_id NOT IN ({marks}) "
                f"AND d.kind != 'module'", ids + ids)
            self.db.execute(
                f"DELETE FROM edges WHERE src_symbol_id IN ({marks}) "
                f"OR dst_symbol_id IN ({marks})", ids + ids)
            self.db.execute(
                f"DELETE FROM unresolved WHERE src_symbol_id IN ({marks})",
                ids)
        self.db.execute("DELETE FROM symbols WHERE file_id=?", (row["id"],))
        self.db.execute("DELETE FROM files WHERE id=?", (row["id"],))
        self.db.commit()
        self.note_change(path)

    # -- queries (live, never stale — M30) --------------------------------
    def find(self, name: str, limit: int = 12) -> list[dict]:
        short = str(name or "").split(".")[-1]
        rows = self.db.execute(
            "SELECT s.name, s.kind, s.line, s.end_line, s.signature, "
            "       s.docstring, s.approximate, f.path, f.lang "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE (s.name = ? OR s.name LIKE ? OR s.name = ?) "
            "AND s.kind != 'module' "
            "ORDER BY (s.name = ?) DESC, s.name LIMIT ?",
            (name, f"%.{short}", short, name, limit)).fetchall()
        return [dict(r) for r in rows]

    def symbols_in(self, path: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT s.name, s.kind, s.line, s.end_line, s.signature, "
            "       s.docstring, s.approximate "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE f.path = ? AND s.kind != 'module' ORDER BY s.line",
            (path,)).fetchall()
        return [dict(r) for r in rows]

    def files(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT path, lang, hash, approximate FROM files ORDER BY path")]

    def file_of(self, symbol: str) -> str:
        rows = self.find(symbol, limit=1)
        return rows[0]["path"] if rows else ""

    def callers_of(self, symbol: str, depth: int = 2) -> list[dict]:
        """Blast radius: who calls this, transitively, with a depth limit.

        On a signature change this answers two questions at once — which
        files need refactoring, and which tests to run FIRST. Depth-limited
        because an unbounded transitive closure on a real codebase returns
        "everything", which is true and useless.
        """
        seen: set[int] = set()
        # The exact name when it exists. A suffix match on top of it made
        # `callers_of("Db.save")` report the callers of `Doc.save` too.
        frontier = [int(r["id"]) for r in self.db.execute(
            "SELECT id FROM symbols WHERE name=? AND kind != 'module'",
            (symbol,))] or [int(r["id"]) for r in self.db.execute(
                "SELECT id FROM symbols WHERE name LIKE ? "
                "AND kind != 'module'",
                (f"%.{str(symbol).split('.')[-1]}",))]
        out: list[dict] = []
        for level in range(max(1, depth)):
            if not frontier:
                break
            marks = ",".join("?" * len(frontier))
            rows = self.db.execute(
                f"SELECT DISTINCT s.id, s.name, s.kind, s.line, f.path "
                f"FROM edges e "
                f"JOIN symbols s ON s.id = e.src_symbol_id "
                f"JOIN files f ON f.id = s.file_id "
                f"WHERE e.dst_symbol_id IN ({marks}) AND e.kind='calls'",
                frontier).fetchall()
            frontier = []
            for r in rows:
                if r["id"] in seen:
                    continue
                seen.add(int(r["id"]))
                out.append({"name": r["name"], "kind": r["kind"],
                            "path": r["path"], "line": r["line"],
                            "distance": level + 1})
                frontier.append(int(r["id"]))
        return out

    def blast_radius(self, symbol: str, depth: int = 2) -> dict:
        """Files to refactor and tests to run first, for a signature change."""
        callers = self.callers_of(symbol, depth)
        files = sorted({c["path"] for c in callers})
        tests = [p for p in files
                 if "test" in p.replace("\\", "/").lower()]
        others = [p for p in files if p not in tests]
        # Test files that COVER the callers, not just test files that call
        # the symbol directly — those are the ones that catch the breakage.
        for path in list(others):
            stem = path.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
            for candidate in self.db.execute(
                    "SELECT path FROM files WHERE path LIKE ?",
                    (f"%test%{stem}%",)):
                if candidate["path"] not in tests:
                    tests.append(candidate["path"])
        return {"symbol": symbol, "callers": callers, "files": others,
                "tests_first": sorted(set(tests)),
                "note": ("nothing calls this — or nothing that has been "
                         "indexed yet does" if not callers else "")}

    def unresolved_names(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT u.name, u.kind, s.name AS src, f.path "
            "FROM unresolved u "
            "JOIN symbols s ON s.id = u.src_symbol_id "
            "JOIN files f ON f.id = s.file_id "
            "ORDER BY u.name LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def resolves(self, name: str) -> bool:
        """Does this symbol exist anywhere in the project? The D4 check.

        Called after generation and before running anything: an invented
        import or API is cheaper to catch here than in a failed build, and
        far more precise — "there is no `parse_config` in `utils`" beats an
        ImportError traceback for a small model every time.
        """
        return bool(self.find(name, limit=1))

    def stats(self) -> CodemapStats:
        """Counts for the one-line summary.

        Unresolved IMPORTS are kept (they are how a project module indexed
        later gets bound) but not counted: `import csv` is a dependency
        outside the project, not a call the map failed to bind, and counting
        every stdlib import would make the resolution rate meaningless.
        Module rows are not counted as symbols for the same reason.
        """
        one = self.db.execute(
            "SELECT (SELECT COUNT(*) FROM files) AS files, "
            "       (SELECT COUNT(*) FROM symbols "
            "        WHERE kind != 'module') AS symbols, "
            "       (SELECT COUNT(*) FROM edges) AS edges, "
            "       (SELECT COUNT(*) FROM unresolved "
            "        WHERE kind != 'imports') AS unresolved"
        ).fetchone()
        return CodemapStats(files=one["files"], symbols=one["symbols"],
                            edges=one["edges"], unresolved=one["unresolved"],
                            epoch=self.epoch)

    # -- regression memory (F10) ------------------------------------------
    def remember_fix(self, signature: str, shape: str) -> None:
        """Record that this diagnostic shape was fixed this way."""
        row = self.db.execute(
            "SELECT id, hits FROM fixes WHERE signature=? AND shape=?",
            (signature, shape)).fetchone()
        if row:
            self.db.execute(
                "UPDATE fixes SET hits=?, last_seen=? WHERE id=?",
                (int(row["hits"]) + 1, time.time(), row["id"]))
        else:
            self.db.execute(
                "INSERT INTO fixes(signature,shape,hits,last_seen) "
                "VALUES(?,?,1,?)", (signature, shape, time.time()))
        self.db.commit()

    def recall_fix(self, signature: str) -> list[dict]:
        """What worked last time for this diagnostic, most-used first."""
        rows = self.db.execute(
            "SELECT shape, hits FROM fixes WHERE signature=? "
            "ORDER BY hits DESC LIMIT 3", (signature,)).fetchall()
        return [dict(r) for r in rows]

    def forget_fixes(self, signature: str = "") -> int:
        """Clearable, because a learned fix that is wrong must be deletable.

        As easy to delete as it was to acquire — that is the condition on
        which F10 is safe to ship at all.
        """
        if signature:
            cur = self.db.execute("DELETE FROM fixes WHERE signature=?",
                                  (signature,))
        else:
            cur = self.db.execute("DELETE FROM fixes")
        self.db.commit()
        return cur.rowcount
