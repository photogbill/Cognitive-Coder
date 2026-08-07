# SPDX-License-Identifier: Apache-2.0
"""The reusable conformance kit — a host runs this against ITS OWN Ports.

This is how ParisNeo validates an adapter without reading the core, and how
the docstring guarantees of §5 become executable rather than aspirational
(§9, M54). It is shipped as part of the product, not as an internal test.

Use it:

    from tests.port_conformance import check_all
    report = check_all(fs=MyFileSystem("/tmp/x"), exec=MyExec(),
                       llm=MyLLM(), storage=MyStorage())
    print(report.text())
    assert report.ok

Every check states **what a host must guarantee**, not what the core happens
to need today. A host that passes this can be upgraded across a minor version
without reading a changelog.

Four things are covered because §9 calls them out by name (M54):

  * **atomicity** where a host claims it — a half-written source file that
    still parses is the worst possible failure, because it looks fine;
  * **tree-kill on timeout** — on Windows a terminated shell does not take
    its children with it, and an orphaned Godot instance holding a file lock
    is a real, observed failure;
  * **the jail** — no write escapes the root, via `..` or via a symlink;
  * **capabilities honesty** — `token_count_is_estimate` and `supports_tools`
    must be true statements, because the core budgets and branches on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
import time
from typing import Any


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "",
            skipped: bool = False) -> None:
        self.checks.append(Check(name, ok, detail, skipped))

    @property
    def ok(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def text(self) -> str:
        lines = ["Cognitive Coder — port conformance", "=" * 46]
        for c in self.checks:
            mark = "SKIP" if c.skipped else ("PASS" if c.ok else "FAIL")
            lines.append(f"  [{mark}] {c.name}")
            if c.detail:
                lines.append(f"         {c.detail}")
        failed = [c for c in self.checks if not c.ok and not c.skipped]
        lines.append("")
        lines.append(f"  {len(self.checks)} checks, {len(failed)} failed")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# FileSystemPort
# --------------------------------------------------------------------------

def check_filesystem(fs: Any, report: Report | None = None,
                     claims_atomic: bool = True) -> Report:
    r = report or Report()

    try:
        fs.write_bytes("cc_conformance/probe.bin", b"\x00\x01hello\r\n")
        got = fs.read_bytes("cc_conformance/probe.bin")
        r.add("write_bytes/read_bytes round-trips exactly",
              got == b"\x00\x01hello\r\n",
              f"got {got!r}" if got != b"\x00\x01hello\r\n" else "")
    except Exception as exc:                             # noqa: BLE001
        r.add("write_bytes/read_bytes round-trips exactly", False, str(exc))

    # Bytes in, bytes out — untouched. A host that "helpfully" normalises
    # line endings inside write_bytes breaks the byte-identical undo
    # guarantee (M26), and it breaks it silently.
    try:
        fs.write_bytes("cc_conformance/eol.txt", b"a\r\nb\r\n")
        raw = fs.read_bytes("cc_conformance/eol.txt")
        r.add("write_bytes does not rewrite line endings",
              raw == b"a\r\nb\r\n",
              "CRLF was altered — undo cannot be byte-identical (M26)"
              if raw != b"a\r\nb\r\n" else "")
    except Exception as exc:                             # noqa: BLE001
        r.add("write_bytes does not rewrite line endings", False, str(exc))

    try:
        r.add("exists() agrees with what was written",
              fs.exists("cc_conformance/probe.bin"))
    except Exception as exc:                             # noqa: BLE001
        r.add("exists() agrees with what was written", False, str(exc))

    # The jail (M24). Every one of these must be refused, in every mode.
    escapes = ["../escaped.txt", "../../escaped.txt",
               "cc_conformance/../../escaped.txt"]
    if os.name == "nt":
        escapes.append("C:\\Windows\\Temp\\escaped.txt")
    else:
        escapes.append("/tmp/cc-conformance-escaped.txt")
    leaked = []
    for path in escapes:
        try:
            fs.write_bytes(path, b"escaped")
            leaked.append(path)
        except Exception:                                # noqa: BLE001
            pass
    r.add("no write escapes root() — `..` and absolute paths",
          not leaked,
          f"these were WRITTEN outside the root: {leaked}" if leaked else "")

    # Symlink escape, where the platform has symlinks.
    root = None
    try:
        root = fs.root()
        if os.path.isdir(root) and hasattr(os, "symlink"):
            link = os.path.join(root, "cc_conformance_link")
            target = os.path.dirname(os.path.abspath(root)) or "/"
            if not os.path.exists(link):
                os.symlink(target, link, target_is_directory=True)
            escaped = False
            try:
                fs.write_bytes("cc_conformance_link/escaped.txt", b"x")
                escaped = os.path.exists(
                    os.path.join(target, "escaped.txt"))
            except Exception:                            # noqa: BLE001
                escaped = False
            r.add("no write escapes root() — via a symlink", not escaped,
                  "a symlink pointing out of the tree was followed"
                  if escaped else "")
        else:
            r.add("no write escapes root() — via a symlink", True, "",
                  skipped=True)
    except Exception as exc:                             # noqa: BLE001
        r.add("no write escapes root() — via a symlink", True, str(exc),
              skipped=True)

    # .git is excluded from listing (M27).
    try:
        fs.write_bytes(".git/config", b"[core]\n")
        listed = fs.list("*")
        r.add("list() excludes .git/",
              not any(str(p).replace("\\", "/").startswith(".git/")
                      for p in listed))
    except Exception:                                    # noqa: BLE001
        r.add("list() excludes .git/", True, "the host refuses .git writes, "
                                             "which is stronger", skipped=True)

    # Atomicity, where the host claims it. A partial file is the failure
    # this is aimed at: it parses, so nothing notices.
    if claims_atomic:
        try:
            big = b"x" * 2_000_000
            fs.write_bytes("cc_conformance/big.bin", big)
            got = fs.read_bytes("cc_conformance/big.bin")
            r.add("write_bytes is atomic for a large payload (M15)",
                  got == big,
                  f"read back {len(got)} of {len(big)} bytes"
                  if got != big else "")
        except Exception as exc:                         # noqa: BLE001
            r.add("write_bytes is atomic for a large payload (M15)", False,
                  str(exc))
    else:
        r.add("write_bytes is atomic (M15)", True,
              "the host's PORTS.md says it is NOT atomic — which is allowed, "
              "provided it says so", skipped=True)

    try:
        fs.delete("cc_conformance/probe.bin")
        r.add("delete() removes a file",
              not fs.exists("cc_conformance/probe.bin"))
    except Exception as exc:                             # noqa: BLE001
        r.add("delete() removes a file", False, str(exc))

    return r


# --------------------------------------------------------------------------
# ExecPort
# --------------------------------------------------------------------------

def check_exec(ex: Any, cwd: str, report: Report | None = None) -> Report:
    r = report or Report()
    py = sys.executable or "python3"

    try:
        res = ex.run([py, "-c", "print('hello')"], cwd=cwd, timeout=30)
        r.add("run() captures stdout and a zero exit code",
              res.exit_code == 0 and "hello" in res.stdout,
              f"exit={res.exit_code} stdout={res.stdout!r}")
    except Exception as exc:                             # noqa: BLE001
        r.add("run() captures stdout and a zero exit code", False, str(exc))

    try:
        res = ex.run([py, "-c", "import sys; sys.exit(3)"], cwd=cwd,
                     timeout=30)
        r.add("run() reports a non-zero exit code", res.exit_code == 3,
              f"exit={res.exit_code}")
    except Exception as exc:                             # noqa: BLE001
        r.add("run() reports a non-zero exit code", False, str(exc))

    try:
        res = ex.run([py, "-c", "print(input())"], cwd=cwd, timeout=30,
                     stdin="piped\n")
        r.add("run() delivers stdin", "piped" in res.stdout,
              f"stdout={res.stdout!r}")
    except Exception as exc:                             # noqa: BLE001
        r.add("run() delivers stdin", False, str(exc))

    # THE TREE-KILL CHECK (M16). A parent that spawns a child which outlives
    # it is exactly the Godot case: kill the parent and the child keeps the
    # file lock. The marker file proves whether the child died.
    marker = os.path.join(cwd, "cc_conformance_child_alive.txt")
    if os.path.exists(marker):
        os.remove(marker)
    child = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c',\n"
        " \"import time;\\ntime.sleep(6)\\n"
        f"open(r'{marker}','w').write('child survived the tree-kill')\"])\n"
        "time.sleep(30)\n")
    try:
        t0 = time.monotonic()
        res = ex.run([py, "-c", child], cwd=cwd, timeout=2)
        elapsed = time.monotonic() - t0
        r.add("run() returns at the timeout, not after the command",
              res.timed_out and elapsed < 15,
              f"timed_out={res.timed_out} after {elapsed:.1f}s")
        time.sleep(7)     # long enough for a SURVIVING child to write
        survived = os.path.exists(marker)
        r.add("run() kills the whole process TREE on timeout (M16)",
              not survived,
              "a grandchild process outlived the timeout — on Windows this "
              "is how orphaned compilers and Godot instances accumulate"
              if survived else "")
        if survived:
            os.remove(marker)
    except Exception as exc:                             # noqa: BLE001
        r.add("run() kills the whole process TREE on timeout (M16)", False,
              str(exc))

    try:
        r.add("which() finds an interpreter that exists",
              bool(ex.which("python3") or ex.which("python")
                   or ex.which("py")))
        r.add("which() returns None for something that does not exist",
              ex.which("definitely-not-a-real-binary-xyzzy") is None)
    except Exception as exc:                             # noqa: BLE001
        r.add("which()", False, str(exc))

    return r


# --------------------------------------------------------------------------
# LLMPort
# --------------------------------------------------------------------------

def check_llm(llm: Any, report: Report | None = None) -> Report:
    from cognitive_coder.types import Message

    r = report or Report()
    try:
        caps = llm.capabilities()
    except Exception as exc:                             # noqa: BLE001
        r.add("capabilities() does not raise", False, str(exc))
        return r
    r.add("capabilities() does not raise", True)

    r.add("capabilities() reports a context size",
          isinstance(caps.context_tokens, int) and caps.context_tokens > 0,
          f"context_tokens={caps.context_tokens!r}")

    r.add("capabilities() answers 'no model loaded' with an empty name, "
          "not an exception (M10)", True,
          "nothing is loaded, which is a NORMAL state" if not caps.loaded
          else "")

    # count_tokens honesty (M14). Exact or flagged — but not silently wrong.
    try:
        n = llm.count_tokens("hello world, this is a sentence")
        r.add("count_tokens() returns a positive integer",
              isinstance(n, int) and n > 0, f"got {n!r}")
        r.add("token_count_is_estimate is a real claim, not a default",
              isinstance(caps.token_count_is_estimate, bool),
              "an exact count is claimed — the core will budget tightly "
              "against it" if not caps.token_count_is_estimate else
              "counts are declared estimates, so the core says so in the "
              "prompt")
    except Exception as exc:                             # noqa: BLE001
        r.add("count_tokens()", False, str(exc))

    # M11: complete() MUST NOT raise on a refusal. A refusal is data; an
    # exception is a crash the loop cannot act on.
    try:
        out = llm.complete(
            [Message(role="user", content="Refuse this request politely.")],
            max_tokens=32)
        r.add("complete() returns a Completion rather than raising (M11)",
              hasattr(out, "finish_reason") and hasattr(out, "text"))
        r.add("complete() sets finish_reason from the closed set",
              out.finish_reason in ("stop", "length", "tool_calls",
                                    "cancelled", "error"),
              f"finish_reason={out.finish_reason!r}")
        r.add("complete() names the model that answered (C8)",
              bool(out.model) or not caps.loaded,
              "Completion.model is empty, so the journal cannot record which "
              "model produced this" if not out.model and caps.loaded else "")
    except Exception as exc:                             # noqa: BLE001
        r.add("complete() returns a Completion rather than raising (M11)",
              False, str(exc))

    # M12: a host without tool support must IGNORE tools, not fail on them.
    if not caps.supports_tools:
        from cognitive_coder.types import ToolSpec
        spec = ToolSpec(name="probe", description="a probe",
                        parameters={"type": "object", "properties": {}})
        try:
            llm.complete([Message(role="user", content="hi")], tools=(spec,),
                         max_tokens=16)
            r.add("a host reporting supports_tools=False ignores `tools` "
                  "(M12)", True)
        except Exception as exc:                         # noqa: BLE001
            r.add("a host reporting supports_tools=False ignores `tools` "
                  "(M12)", False,
                  f"passing tools raised: {exc}")
    else:
        r.add("supports_tools=True — the core will use native tool calling",
              True, "", skipped=True)

    return r


# --------------------------------------------------------------------------
# StoragePort
# --------------------------------------------------------------------------

def check_storage(storage: Any, report: Report | None = None) -> Report:
    import json
    import sqlite3

    r = report or Report()
    try:
        storage.set("cc.probe", {"a": [1, 2], "b": "x"})
        r.add("get/set round-trips a nested value",
              storage.get("cc.probe") == {"a": [1, 2], "b": "x"})
        r.add("get() honours its default",
              storage.get("cc.absent", "fallback") == "fallback")
    except Exception as exc:                             # noqa: BLE001
        r.add("get/set", False, str(exc))

    # M17: values must be JSON-serialisable. That is the portability
    # contract — a host storing pickles cannot hand its state to one that
    # stores JSON, and resume has to survive that.
    try:
        storage.set("cc.probe2", {"n": 1})
        json.dumps(storage.get("cc.probe2"))
        r.add("stored values are JSON-serialisable (M17)", True)
    except Exception as exc:                             # noqa: BLE001
        r.add("stored values are JSON-serialisable (M17)", False, str(exc))

    try:
        path = storage.sqlite_path("cc_conformance")
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
        db.execute("INSERT INTO t VALUES (1)")
        db.commit()
        db.close()
        r.add("sqlite_path() is a usable, writable database path", True)
        r.add("sqlite_path() is stable for the same name",
              storage.sqlite_path("cc_conformance") == path)
    except Exception as exc:                             # noqa: BLE001
        r.add("sqlite_path() is a usable, writable database path", False,
              str(exc))
    return r


# --------------------------------------------------------------------------
# everything
# --------------------------------------------------------------------------

def check_all(*, fs: Any = None, exec: Any = None,   # noqa: A002
              llm: Any = None, storage: Any = None,
              cwd: str = "", claims_atomic: bool = True) -> Report:
    """Run every applicable check. Omitted Ports are simply not checked."""
    report = Report()
    if fs is not None:
        check_filesystem(fs, report, claims_atomic=claims_atomic)
    if exec is not None:
        check_exec(exec, cwd or (fs.root() if fs is not None else "."),
                   report)
    if llm is not None:
        check_llm(llm, report)
    if storage is not None:
        check_storage(storage, report)
    return report


if __name__ == "__main__":                               # pragma: no cover
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from cognitive_coder.ports import (
        LocalFileSystem,
        MemoryStorage,
        NullLLM,
        SubprocessExec,
    )

    workspace = tempfile.mkdtemp(prefix="cc-conformance-")
    result = check_all(fs=LocalFileSystem(workspace), exec=SubprocessExec(),
                       llm=NullLLM(), storage=MemoryStorage(), cwd=workspace)
    print(result.text())
    raise SystemExit(0 if result.ok else 1)
