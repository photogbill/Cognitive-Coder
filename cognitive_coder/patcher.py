# SPDX-License-Identifier: Apache-2.0
"""Applying model-proposed edits to real files — with a way back.

The failure mode with small models is not vandalism, it is **plausible
wrongness**: an edit that looks right, applies cleanly, and quietly breaks
something three files away. You cannot prevent that with a confirmation dialog
— you skim it and click yes. You prevent it with cheap undo and a readable
record, which is what this module is.

THREE EDIT FORMATS, in order of how much can go wrong:

  1. **Anchored replace** — exact old text → new text. Applies only if the
     anchor appears EXACTLY ONCE. **Ambiguity is refused, never guessed**
     (M23): an anchor that matches twice means the model didn't give enough
     context, and picking the first match is how the wrong function gets
     edited. This is the single rule that prevents the worst class of damage.
  2. **Unified diff** — parsed and applied with context verification.
  3. **Whole file** — the blunt one. Fine for new files, and for small files
     where a model rewriting the lot is more reliable than it patching.

TRANSACTIONS — EXPLICIT, NEVER INFERRED (§6.5)

"Undo restores an apply as one unit" is ambiguous the moment a loop edits file
A (verified good) and then file B (fails). Both readings are needed and which
one applies is **declared by the planner**, not guessed here:

  * A verified and committed, B fails → **A stands.** Reverting verified work
    because a later, separate task failed destroys progress the operator
    watched succeed.
  * A and B are two halves of one change — a signature and its caller →
    **both revert.** Leaving A applied is a tree that compiles nowhere.

    tx = patcher.begin(task_id="add-parser", atomic=True)
    tx.apply(edits_for_a)
    tx.apply(edits_for_b)
    tx.commit()          # or tx.rollback() → reverts EVERYTHING in this tx

Five rules make that model trustworthy rather than merely stated:

  1. Transactions are opened by the **planner**, per task, carrying `atomic`
     from the plan. A multi-file refactor is one atomic transaction; three
     independent files are three transactions.
  2. **Sequence numbers, not timestamps** (M25). Timestamps collide within a
     second, sort wrongly across a clock change, and are ambiguous over a DST
     boundary. The log is strictly linear and the numbering proves it.
  3. **A committed, verified transaction is SEALED.** `rollback()` on a later
     transaction can never touch it. Reaching further back requires an
     explicit `undo_to(seq)` from the operator, which states in plain words
     how many verified transactions it is about to discard.
  4. **Rollback is itself journaled**, with its own sequence number. The
     history is append-only; an undo is a new fact, not an erasure.
  5. `history()` returns the linear log. That is what the UI shows and what
     "what did it just do to my project" is answered from.

AND THE TWO THAT ARE EASY TO ERODE:

  * **Model-initiated edits are not special** (M18). When the model calls the
    `apply_patch` tool, the edit enters the SAME transaction and the SAME
    `ApprovalPort` gate as any other. Tool calling must never become a side
    door around the approval default.
  * **Nothing is written outside the project root, ever, in any mode** (M24)
    — and "outside" is judged on RESOLVED real paths: normalise, resolve
    symlinks, then check containment. A `..` component or a symlink pointing
    out of the tree is a refusal with a plain sentence, not a write.

Why bespoke snapshots and not git (§6.5b), stated so nobody "fixes" it: this
must work on non-repos, must not depend on a `git` binary, and must never
touch the operator's history, stash or index. A tool that quietly makes
commits in someone's repo is indistinguishable from a mess. **The engine never
runs git** (M27).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import difflib
import posixpath
import re
from typing import Any

from . import textio
from .errors import PathEscape, TransactionError
from .types import Edit, EditResult, TransactionRecord

SNAPSHOT_DIR = ".cc_snapshots"
# Keep this many transactions before the oldest snapshot is pruned. Snapshots
# are cheap (text files) but not free, and an unbounded pile is its own
# problem. The LOG is never pruned — only the restorable bytes.
MAX_SNAPSHOTS = 40

_SEQ_KEY = "cognitive_coder.patcher.seq"
_LOG_KEY = "cognitive_coder.patcher.log"


# ---------------------------------------------------------------------------
# parsing model output into edits
# ---------------------------------------------------------------------------

# ```lang path=src/main.c
_FENCE = re.compile(
    r"```(?P<lang>[\w+#.-]*)[ \t]*(?P<attrs>[^\n]*)\n(?P<body>.*?)```", re.S)
_PATH_ATTR = re.compile(r"(?:path|file)\s*=\s*[\"']?([^\"'\s]+)")
# The anchored-edit markers. Chosen to be unmistakable in a token stream and
# impossible to confuse with real code in any supported language.
_EDIT_BLOCK = re.compile(
    r"<<<+\s*CC-EDIT\s+(?P<path>[^\n>]+?)\s*>*\n"
    r"(?P<old>.*?)\n===+\n(?P<new>.*?)\n>>>+\s*CC-END", re.S)
# ATK's older marker, accepted so a project mid-migration keeps working.
_EDIT_BLOCK_ATK = re.compile(
    r"<<<+\s*ATK-EDIT\s+(?P<path>[^\n>]+?)\s*>*\n"
    r"(?P<old>.*?)\n===+\n(?P<new>.*?)\n>>>+\s*ATK-END", re.S)


def parse_edits(text: str, default_path: str = "") -> list[Edit]:
    """Pull edits out of a model reply, in whichever format it used.

    Deliberately **permissive about form and strict about application**: a
    small model will not reliably produce one exact format, so several are
    accepted here — and then `apply` refuses anything ambiguous. Better to
    understand the intent and verify it than to reject a whole reply because
    a fence said ```py instead of ```python (D5).
    """
    edits: list[Edit] = []
    for pattern in (_EDIT_BLOCK, _EDIT_BLOCK_ATK):
        for m in pattern.finditer(text or ""):
            edits.append(Edit(path=m.group("path").strip(), kind="replace",
                              old=m.group("old"), new=m.group("new")))
    for m in _FENCE.finditer(text or ""):
        attrs = m.group("attrs") or ""
        body = m.group("body")
        pm = _PATH_ATTR.search(attrs)
        path = pm.group(1) if pm else ""
        if not path and attrs.strip() and ("/" in attrs or "\\" in attrs):
            path = attrs.strip()
        path = path or default_path
        if not path:
            continue
        if body.lstrip().startswith(("--- ", "diff --git", "@@ ")):
            edits.append(Edit(path=path, kind="diff", new=body))
        else:
            edits.append(Edit(path=path, kind="whole", new=body))
    return edits


def extract_code(text: str, lang_id: str = "", validator=None) -> str:
    """The code from a model reply, when the whole reply should be one file.

    Fence confusion is D5: three backticks inside a docstring, a language tag
    that isn't a language, no fence at all, two fences with different content.
    So: prefer a fence whose tag matches the target language, then the longest
    fence, then the whole reply — and **validate by parsing** where a
    validator is supplied, trying the next candidate before giving up. Never
    assume the first fence.
    """
    body = text or ""
    fences = list(_FENCE.finditer(body))
    candidates: list[str] = []
    if lang_id:
        aliases = {lang_id, lang_id[:2], {"python": "py", "javascript": "js",
                                          "typescript": "ts", "csharp": "cs",
                                          "gdscript": "gd"}.get(lang_id, "")}
        candidates += [m.group("body") for m in fences
                       if (m.group("lang") or "").lower() in aliases]
    candidates += [m.group("body")
                   for m in sorted(fences, key=lambda m: -len(m.group("body")))]
    candidates.append(body)

    for cand in candidates:
        cand = cand.strip("\n")
        if not cand.strip():
            continue
        if validator is None or validator(cand):
            return cand
    return (candidates[0] if candidates else body).strip("\n")


# ---------------------------------------------------------------------------
# applying
# ---------------------------------------------------------------------------

def _unique_index(haystack: str, needle: str) -> int:
    """Index of ``needle`` if it appears exactly once; -1 missing, -2 ambiguous."""
    first = haystack.find(needle)
    if first < 0:
        return -1
    if haystack.find(needle, first + 1) >= 0:
        return -2
    return first


def _apply_diff(original: str, diff_text: str) -> tuple[str, str]:
    """Apply a unified diff by verifying context. Returns (text, reason)."""
    lines = original.splitlines(keepends=True)
    out = list(lines)
    hunks = re.findall(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@.*$",
                       diff_text, re.M)
    if not hunks:
        return original, "no hunks were found in the diff"
    body = diff_text.splitlines()
    idx = 0
    offset = 0
    for start, _count in hunks:
        while idx < len(body) and not body[idx].startswith("@@"):
            idx += 1
        idx += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        while idx < len(body) and not body[idx].startswith("@@"):
            ln = body[idx]
            if ln.startswith("-"):
                old_lines.append(ln[1:] + "\n")
            elif ln.startswith("+"):
                new_lines.append(ln[1:] + "\n")
            elif ln.startswith(" ") or ln == "":
                old_lines.append(ln[1:] + "\n" if ln else "\n")
                new_lines.append(ln[1:] + "\n" if ln else "\n")
            idx += 1
        at = int(start) - 1 + offset
        window = "".join(out[at:at + len(old_lines)])
        if window != "".join(old_lines):
            # The line numbers didn't match — search for the context instead
            # of writing at the wrong place. A MISAPPLIED hunk is worse than a
            # refused one, by a long way: it corrupts silently.
            found = _unique_index("".join(out), "".join(old_lines))
            if found < 0:
                return original, ("the diff's context does not match the file "
                                  "(it may be out of date, or it matched in "
                                  "more than one place)")
            prefix = "".join(out)[:found]
            at = prefix.count("\n")
            window = "".join(out[at:at + len(old_lines)])
            if window != "".join(old_lines):
                return original, "the diff's context does not match the file"
        out[at:at + len(old_lines)] = new_lines
        offset += len(new_lines) - len(old_lines)
    return "".join(out), ""


def unified(before: str, after: str, path: str, context: int = 2) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}", n=context))


def safe_relpath(fs: Any, rel: str) -> str:
    """A root-relative path, or a refusal (M24).

    Containment is judged on the RESOLVED real path, so `..` components and
    symlinks pointing out of the tree are both caught. Project mode relaxes
    what generated code may DO; it never relaxes where an edit may LAND.
    """
    raw = str(rel or "").replace("\\", "/")
    if raw.startswith(("res://", "user://")):
        raw = raw.split("://", 1)[1]          # Godot paths, §6.1a
    root = str(fs.root()).replace("\\", "/").rstrip("/")
    if root and raw.startswith(root + "/"):
        raw = raw[len(root) + 1:]
    if re.match(r"^(?:[A-Za-z]:|/|\\\\)", raw):
        raise PathEscape(rel, fs.root())
    norm = posixpath.normpath(raw)
    if norm.startswith("..") or norm == "." or posixpath.isabs(norm):
        raise PathEscape(rel, fs.root())
    if norm.startswith(".git/") or norm == ".git":
        raise PathEscape(rel, fs.root())      # M27: .git is never patched
    # The host's FileSystemPort resolves symlinks; LocalFileSystem raises on
    # an escape and MemoryFileSystem has no symlinks to follow. This check is
    # the core's own, deliberately duplicated — a jail with one door is not a
    # jail, and this one costs nothing.
    return norm


# ---------------------------------------------------------------------------
# the transaction
# ---------------------------------------------------------------------------

class Transaction:
    """One unit of change, atomic or not, as the planner declared it.

    Not created directly — `Patcher.begin()` opens one, so the sequence
    number and the log entry exist before a single byte is written.
    """

    def __init__(self, patcher: Patcher, seq: int, task_id: str,
                 atomic: bool) -> None:
        self._p = patcher
        self.seq = seq
        self.task_id = task_id
        self.atomic = atomic
        self.state = "open"
        self.verified = False
        self.files: list[str] = []
        self.results: list[EditResult] = []
        self._snapshots: dict[str, bytes | None] = {}   # None ⇒ did not exist
        self._diffs: list[str] = []

    # -- context manager: a lost transaction rolls back, never dangles ----
    def __enter__(self) -> Transaction:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.state == "open":
            if exc_type is None:
                self.commit()
            else:
                # An exception mid-transaction leaves a half-applied tree,
                # which is the state §5.2 promises never to leave behind.
                self.rollback(f"aborted: {exc}")
        return False

    # -- applying ---------------------------------------------------------
    def apply(self, edits: Sequence[Edit], *, approve: bool = True,
              summary: str = "") -> list[EditResult]:
        """Apply edits into this transaction. Snapshots first, then writes.

        Every write routes through `ApprovalPort.approve_diff` (M18) unless
        the caller has already approved this batch — and "already approved"
        means a human saw THIS diff, not that a flag was set somewhere.
        """
        if self.state != "open":
            raise TransactionError(
                f"Transaction {self.seq} is {self.state}; it cannot take more "
                f"edits. Open a new one.")
        results: list[EditResult] = []
        for edit in edits:
            results.append(self._apply_one(edit, approve, summary))
        self.results.extend(results)
        return results

    def _apply_one(self, edit: Edit, approve: bool,
                   summary: str) -> EditResult:
        fs = self._p.fs
        try:
            rel = safe_relpath(fs, edit.path)
        except PathEscape as exc:
            return EditResult(edit.path, False, exc.sentence)

        existed = fs.exists(rel)
        tf = None
        before = ""
        if existed:
            try:
                tf = textio.read(fs, rel)
                before = tf.text
            except Exception as exc:                     # noqa: BLE001
                return EditResult(edit.path, False,
                                  f"could not read it: {exc}")
        else:
            tf = textio.new_file(fs, "")

        # -- compute the new text -----------------------------------------
        if edit.kind == "replace":
            if not (edit.old or "").strip():
                return EditResult(edit.path, False,
                                  "the anchor is empty — there is nothing to "
                                  "match against")
            anchor = textio.normalise(edit.old)
            pos = _unique_index(before, anchor)
            if pos == -1:
                return EditResult(
                    edit.path, False,
                    "the text to replace is not in the file (it may have "
                    "changed since the model read it)")
            if pos == -2:
                return EditResult(
                    edit.path, False,
                    "the text to replace appears more than once — refusing "
                    "rather than guessing which one was meant")
            after = (before[:pos] + textio.normalise(edit.new)
                     + before[pos + len(anchor):])
        elif edit.kind == "diff":
            after, why = _apply_diff(before, textio.normalise(edit.new))
            if why:
                return EditResult(edit.path, False, why)
        else:                                            # whole | create
            after = textio.normalise(edit.new)
            if after and not after.endswith("\n"):
                after += "\n"

        if after == before and existed:
            return EditResult(edit.path, False,
                              "no change — the edit was already applied")

        diff = unified(before, after, rel)

        # -- the approval gate, for EVERY write (M18) ---------------------
        if approve:
            label = summary or (
                f"{'create' if not existed else 'edit'} {rel} "
                f"(task {self.task_id})")
            try:
                if not self._p.approval.approve_diff(label, diff):
                    return EditResult(edit.path, False,
                                      "not approved — nothing was written")
            except Exception as exc:                     # noqa: BLE001
                return EditResult(edit.path, False,
                                  f"the approval step failed: {exc}")

        # -- snapshot BEFORE writing --------------------------------------
        # An edit with no way back is not applied. This is the rule that
        # makes auto-apply survivable rather than reckless, and deleting it
        # to "simplify" would mean deleting auto-apply first (§6.5).
        if rel not in self._snapshots:
            try:
                self._snapshots[rel] = (fs.read_bytes(rel) if existed
                                        else None)
            except Exception as exc:                     # noqa: BLE001
                return EditResult(
                    edit.path, False,
                    f"refused: the original could not be snapshotted first "
                    f"({exc}) — an edit with no way back is not applied")

        try:
            textio.write(fs, rel, tf, after)
        except Exception as exc:                         # noqa: BLE001
            return EditResult(edit.path, False, f"the write failed: {exc}")

        if rel not in self.files:
            self.files.append(rel)
        self._diffs.append(diff)
        self._p.events("patch", f"{rel} updated (transaction {self.seq})",
                       {"seq": self.seq, "files": [rel],
                        "task": self.task_id})
        return EditResult(edit.path, True, "", diff)

    # -- finishing --------------------------------------------------------
    def commit(self, verified: bool = False) -> TransactionRecord:
        """Close the transaction. `verified=True` SEALS it (M25 rule 3)."""
        if self.state != "open":
            raise TransactionError(
                f"Transaction {self.seq} is already {self.state}.")
        self.state = "committed"
        self.verified = bool(verified)
        self._p._persist_snapshots(self)
        rec = self.record()
        # _replace_log, not _write_log: `begin()` already wrote this
        # transaction's row. One row per sequence number is what makes the
        # log readable as a history rather than as a stream of state changes
        # — and readability is the point of it (M25 rule 5).
        self._p._replace_log(rec)
        return rec

    def mark_verified(self) -> TransactionRecord:
        """Seal a committed transaction once verification has passed.

        Separate from `commit()` because the loop commits, THEN runs the
        build and tests — sealing before the evidence exists would be a lie
        about what "verified" means.
        """
        if self.state != "committed":
            raise TransactionError(
                f"Transaction {self.seq} is {self.state}; only a committed "
                f"transaction can be marked verified.")
        self.verified = True
        rec = self.record()
        self._p._replace_log(rec)
        return rec

    def rollback(self, note: str = "") -> TransactionRecord:
        """Undo everything in THIS transaction, and nothing else.

        Restores the snapshotted BYTES, so the result is byte-identical
        including encoding, BOM and line endings (M26) — that guarantee comes
        from storing bytes rather than reconstructing text, which is why the
        snapshot is taken at the bytes level.
        """
        fs = self._p.fs
        restored: list[str] = []
        for rel, raw in self._snapshots.items():
            try:
                if raw is None:
                    if fs.exists(rel):
                        fs.delete(rel)          # it did not exist before
                        restored.append(rel)
                else:
                    fs.write_bytes(rel, raw)
                    restored.append(rel)
            except Exception:                            # noqa: BLE001
                continue
        self.state = "rolled_back"
        rec = self.record(note=note or f"rolled back {len(restored)} file(s)")
        self._p._replace_log(rec)
        # Rule 4: the rollback is itself a fact in the log, with its own
        # sequence number. History is append-only; undo is never an erasure.
        undo_seq = self._p._next_seq()
        self._p._write_log(TransactionRecord(
            seq=undo_seq, task_id=self.task_id, atomic=self.atomic,
            files=tuple(restored), state="rollback_of",
            note=f"undid transaction {self.seq}"))
        self._p.events("patch",
                       f"transaction {self.seq} rolled back — "
                       f"{len(restored)} file(s) restored",
                       {"seq": undo_seq, "files": restored,
                        "undid": self.seq})
        return rec

    def record(self, note: str = "") -> TransactionRecord:
        return TransactionRecord(
            seq=self.seq, task_id=self.task_id, atomic=self.atomic,
            files=tuple(self.files), state=self.state, verified=self.verified,
            snapshot_dir=self._p._snapshot_dir(self.seq, self.task_id),
            diff="".join(self._diffs), note=note)

    @property
    def diff(self) -> str:
        return "".join(self._diffs)


# ---------------------------------------------------------------------------
# the patcher
# ---------------------------------------------------------------------------

class Patcher:
    """Transactions, snapshots, undo and the linear history.

    Takes the Ports it needs rather than reaching for them (C2). The
    ApprovalPort defaults to the host's, which defaults to approval-required
    — a new host must never silently write to someone's project.
    """

    def __init__(self, fs: Any, storage: Any, approval: Any,
                 events: Any = None) -> None:
        self.fs = fs
        self.storage = storage
        self.approval = approval
        self._events = events
        self._open: Transaction | None = None

    # -- events -----------------------------------------------------------
    def events(self, kind: str, message: str, data: dict | None = None):
        if self._events is None:
            return
        try:
            self._events.event(kind, message, data)
        except Exception:                                # noqa: BLE001
            pass

    # -- sequence and log -------------------------------------------------
    def _next_seq(self) -> int:
        seq = int(self.storage.get(_SEQ_KEY, 0)) + 1
        self.storage.set(_SEQ_KEY, seq)
        return seq

    def _log(self) -> list[dict]:
        return list(self.storage.get(_LOG_KEY, []) or [])

    def _write_log(self, rec: TransactionRecord) -> None:
        log = self._log()
        log.append(_rec_to_dict(rec))
        self.storage.set(_LOG_KEY, log)

    def _replace_log(self, rec: TransactionRecord) -> None:
        log = self._log()
        for i, row in enumerate(log):
            if row.get("seq") == rec.seq and row.get("state") != "rollback_of":
                log[i] = _rec_to_dict(rec)
                break
        else:
            log.append(_rec_to_dict(rec))
        self.storage.set(_LOG_KEY, log)

    def _snapshot_dir(self, seq: int, task_id: str) -> str:
        # NNNN-<task_id>/ with a monotonic counter (M25 rule 2).
        safe = re.sub(r"[^\w.-]+", "-", task_id or "task")[:40]
        return f"{SNAPSHOT_DIR}/{seq:04d}-{safe}"

    def _persist_snapshots(self, tx: Transaction) -> None:
        """Write the transaction's snapshot bytes into the project.

        In-memory snapshots survive a crash badly; on-disk ones survive it
        well, and resume (§6.13) is derived from durable state rather than
        from an object that died with the process.
        """
        base = self._snapshot_dir(tx.seq, tx.task_id)
        manifest: list[str] = []
        for rel, raw in tx._snapshots.items():
            if raw is None:
                manifest.append(f"NEW {rel}")
                continue
            try:
                self.fs.write_bytes(f"{base}/files/{rel}", raw)
                manifest.append(f"OLD {rel}")
            except Exception:                            # noqa: BLE001
                manifest.append(f"?? {rel} (could not be snapshotted)")
        try:
            self.fs.write(
                f"{base}/MANIFEST.txt",
                f"transaction {tx.seq}  task={tx.task_id}  "
                f"atomic={tx.atomic}\n"
                + "\n".join(manifest)
                + "\n\n--- what changed ---\n" + tx.diff)
        except Exception:                                # noqa: BLE001
            pass
        self._prune()

    def _prune(self) -> None:
        """Drop the oldest snapshot BYTES. The log itself is never pruned."""
        try:
            dirs = sorted({p.split("/")[1] for p in
                           self.fs.list(f"{SNAPSHOT_DIR}/*")
                           if p.startswith(SNAPSHOT_DIR + "/")
                           and "/" in p[len(SNAPSHOT_DIR) + 1:]})
        except Exception:                                # noqa: BLE001
            return
        for old in dirs[:-MAX_SNAPSHOTS]:
            for path in self.fs.list(f"{SNAPSHOT_DIR}/{old}/**"):
                try:
                    self.fs.delete(path)
                except Exception:                        # noqa: BLE001
                    pass

    # -- the API ----------------------------------------------------------
    def begin(self, task_id: str, atomic: bool = False) -> Transaction:
        """Open a transaction. Opened by the PLANNER, per task (M25 rule 1)."""
        if self._open is not None and self._open.state == "open":
            raise TransactionError(
                f"Transaction {self._open.seq} is still open. Commit or roll "
                f"it back before starting another.")
        tx = Transaction(self, self._next_seq(), task_id, atomic)
        self._open = tx
        self._write_log(tx.record(note="opened"))
        return tx

    def history(self) -> list[TransactionRecord]:
        """The linear log: sequence, task, files, verification, rollbacks.

        This is what the UI shows and what "what did it just do to my
        project" is answered from (M25 rule 5).
        """
        return [_dict_to_rec(row) for row in self._log()]

    def undo_to(self, seq: int, *, confirm=None) -> dict:
        """Reach back past sealed transactions — deliberately awkward (rule 3).

        States in plain words how many VERIFIED transactions it is about to
        discard, and asks. A sealed transaction is work the operator watched
        succeed; discarding it should never be a side effect of something
        else going wrong.
        """
        log = self.history()
        doomed = [r for r in log
                  if r.seq > seq and r.state == "committed"]
        sealed = [r for r in doomed if r.sealed]
        if not doomed:
            return {"ok": False,
                    "note": f"nothing has happened since transaction {seq}."}
        sentence = (
            f"This will discard {len(doomed)} transaction(s), "
            f"{len(sealed)} of which were verified — meaning they built and "
            f"their tests passed. Files affected: "
            f"{', '.join(sorted({f for r in doomed for f in r.files}))}.")
        if confirm is not None and not confirm(sentence):
            return {"ok": False, "note": "cancelled; nothing was changed."}

        restored: list[str] = []
        # Newest first, so an older snapshot overwrites a newer one and the
        # tree ends at the requested point rather than somewhere in between.
        for rec in sorted(doomed, key=lambda r: -r.seq):
            base = self._snapshot_dir(rec.seq, rec.task_id)
            for rel in rec.files:
                snap = f"{base}/files/{rel}"
                try:
                    if self.fs.exists(snap):
                        self.fs.write_bytes(rel, self.fs.read_bytes(snap))
                        restored.append(rel)
                    elif self.fs.exists(rel):
                        self.fs.delete(rel)      # it was created by this tx
                        restored.append(rel)
                except Exception:                        # noqa: BLE001
                    continue
            undo_seq = self._next_seq()
            self._write_log(TransactionRecord(
                seq=undo_seq, task_id=rec.task_id, atomic=rec.atomic,
                files=rec.files, state="rollback_of",
                note=f"undo_to({seq}) undid transaction {rec.seq}"))
        self.events("patch",
                    f"rolled back to transaction {seq} — "
                    f"{len(set(restored))} file(s) restored",
                    {"seq": seq, "files": sorted(set(restored))})
        return {"ok": True, "restored": sorted(set(restored)),
                "discarded": [r.seq for r in doomed], "note": sentence}

    def preview(self, edits: Iterable[Edit]) -> str:
        """What WOULD change, as a diff, without touching anything.

        Exists so a UI can show the change the moment it is proposed, and so
        a cautious operator can look before approving — which is the default.
        """
        out: list[str] = []
        for edit in edits:
            try:
                rel = safe_relpath(self.fs, edit.path)
            except PathEscape as exc:
                out.append(f"# {edit.path}: REFUSED — {exc.sentence}")
                continue
            before = ""
            if self.fs.exists(rel):
                try:
                    before = textio.read(self.fs, rel).text
                except Exception:                        # noqa: BLE001
                    pass
            if edit.kind == "replace":
                pos = _unique_index(before, textio.normalise(edit.old))
                if pos < 0:
                    out.append(f"# {rel}: the anchor "
                               + ("was not found" if pos == -1
                                  else "is ambiguous — it matches more than "
                                       "once, so this would be refused"))
                    continue
                after = (before[:pos] + textio.normalise(edit.new)
                         + before[pos + len(textio.normalise(edit.old)):])
            elif edit.kind == "diff":
                after, why = _apply_diff(before, textio.normalise(edit.new))
                if why:
                    out.append(f"# {rel}: {why}")
                    continue
            else:
                after = textio.normalise(edit.new)
            out.append(unified(before, after, rel) or f"# {rel}: no change")
        return "\n".join(out)


def _rec_to_dict(rec: TransactionRecord) -> dict:
    return {"seq": rec.seq, "task_id": rec.task_id, "atomic": rec.atomic,
            "files": list(rec.files), "state": rec.state,
            "verified": rec.verified, "snapshot_dir": rec.snapshot_dir,
            "note": rec.note}
    # `diff` is deliberately NOT stored in the key-value log: it belongs in
    # the snapshot MANIFEST, where it can be read without loading the whole
    # history into memory.


def _dict_to_rec(row: dict) -> TransactionRecord:
    return TransactionRecord(
        seq=int(row.get("seq", 0)), task_id=row.get("task_id", ""),
        atomic=bool(row.get("atomic")), files=tuple(row.get("files", ())),
        state=row.get("state", "open"), verified=bool(row.get("verified")),
        snapshot_dir=row.get("snapshot_dir", ""), note=row.get("note", ""))
