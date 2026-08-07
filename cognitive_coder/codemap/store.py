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


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:32]


class Store:
    """The codemap's database. One per project."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

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
        return n

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

        for src, dst, kind in edges:
            src_id = name_to_id.get(src) or self._find_id(src)
            dst_id = self._find_id(dst) or name_to_id.get(dst)
            if src_id and dst_id:
                self.db.execute(
                    "INSERT INTO edges(src_symbol_id,dst_symbol_id,kind) "
                    "VALUES(?,?,?)", (src_id, dst_id, kind))
            elif src_id:
                # It could not be bound. It is KEPT, not dropped (§6.7).
                self.db.execute(
                    "INSERT INTO unresolved(src_symbol_id,name,kind) "
                    "VALUES(?,?,?)", (src_id, str(dst), kind))
        for src, name, kind in unresolved:
            src_id = name_to_id.get(src) or self._find_id(src)
            if not src_id:
                continue
            # Binding runs in BOTH directions, and forgetting this one is a
            # silent, plausible bug: `_rebind` below catches "a definition
            # arrived for a call we already knew about", but a call arriving
            # for a definition already indexed needs looking up now. Without
            # this, `callers_of` returns nothing for every cross-file call
            # whose target happened to be indexed first — which looks like a
            # project with no call graph rather than like a bug.
            dst_id = self._find_id(name)
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
        self._rebind(name_to_id)
        return file_id

    def _rebind(self, new_symbols: dict[str, int]) -> None:
        """Late binding: an unresolved call that now HAS a target becomes an edge.

        This is what makes the graph improve as more of the project is
        indexed. File A calling `B.load` before B was indexed is unresolved;
        the moment B lands, it becomes a real edge — and the resolution rate
        going up is the visible sign the map is getting more complete.
        """
        if not new_symbols:
            return
        for name, sym_id in new_symbols.items():
            short = name.split(".")[-1]
            rows = self.db.execute(
                "SELECT rowid, src_symbol_id, kind FROM unresolved "
                "WHERE name=? OR name=? OR name LIKE ?",
                (name, short, f"%.{short}")).fetchall()
            for row in rows:
                self.db.execute(
                    "INSERT INTO edges(src_symbol_id,dst_symbol_id,kind) "
                    "VALUES(?,?,?)",
                    (row["src_symbol_id"], sym_id, row["kind"]))
                self.db.execute("DELETE FROM unresolved WHERE rowid=?",
                                (row["rowid"],))
        self.db.commit()

    def _find_id(self, name: str) -> int | None:
        row = self.db.execute(
            "SELECT id FROM symbols WHERE name=? ORDER BY id LIMIT 1",
            (str(name),)).fetchone()
        if row:
            return int(row["id"])
        short = str(name).split(".")[-1]
        row = self.db.execute(
            "SELECT id FROM symbols WHERE name=? OR name LIKE ? "
            "ORDER BY id LIMIT 1", (short, f"%.{short}")).fetchone()
        return int(row["id"]) if row else None

    def forget(self, path: str) -> None:
        row = self.db.execute("SELECT id FROM files WHERE path=?",
                              (path,)).fetchone()
        if not row:
            return
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM symbols WHERE file_id=?", (row["id"],))]
        if ids:
            marks = ",".join("?" * len(ids))
            self.db.execute(
                f"DELETE FROM edges WHERE src_symbol_id IN ({marks}) "
                f"OR dst_symbol_id IN ({marks})", ids + ids)
            self.db.execute(
                f"DELETE FROM unresolved WHERE src_symbol_id IN ({marks})",
                ids)
        self.db.execute("DELETE FROM symbols WHERE file_id=?", (row["id"],))
        self.db.execute("DELETE FROM files WHERE id=?", (row["id"],))
        self.db.commit()

    # -- queries (live, never stale — M30) --------------------------------
    def find(self, name: str, limit: int = 12) -> list[dict]:
        short = str(name or "").split(".")[-1]
        rows = self.db.execute(
            "SELECT s.name, s.kind, s.line, s.end_line, s.signature, "
            "       s.docstring, s.approximate, f.path, f.lang "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE s.name = ? OR s.name LIKE ? OR s.name = ? "
            "ORDER BY (s.name = ?) DESC, s.name LIMIT ?",
            (name, f"%.{short}", short, name, limit)).fetchall()
        return [dict(r) for r in rows]

    def symbols_in(self, path: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT s.name, s.kind, s.line, s.end_line, s.signature, "
            "       s.docstring, s.approximate "
            "FROM symbols s JOIN files f ON f.id = s.file_id "
            "WHERE f.path = ? ORDER BY s.line", (path,)).fetchall()
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
        frontier = [int(r["id"]) for r in self.db.execute(
            "SELECT id FROM symbols WHERE name=? OR name LIKE ?",
            (symbol, f"%.{str(symbol).split('.')[-1]}"))]
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
        one = self.db.execute(
            "SELECT (SELECT COUNT(*) FROM files) AS files, "
            "       (SELECT COUNT(*) FROM symbols) AS symbols, "
            "       (SELECT COUNT(*) FROM edges) AS edges, "
            "       (SELECT COUNT(*) FROM unresolved) AS unresolved"
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
