# SPDX-License-Identifier: Apache-2.0
"""The smallest possible host — and the proof that the embedding story works.

This is the first thing anyone embedding Cognitive Coder should read, and the
first thing ParisNeo is handed. It drives a whole session end to end on the
Null ports: **no model, no network, no host application** — a scripted model
and a temporary directory are the only inputs. If this runs, the
architecture's central claim is true: the core really does depend on nothing
but the Ports.

Run it:

    python examples/tiny_host.py

What it demonstrates, in order:

  1. A host is six objects. You can bring your own or use the ones in
     `ports.py`; nothing here inherits from anything of ours (C2).
  2. `ScriptedLLM` stands in for a model. The whole engine is drivable with
     canned replies, which is how you develop a UI without a 14 GB model
     loaded — and how the test suite works (§9).
  3. Events are the host's UI. Six lines of `print` is a complete host.
  4. The journal is provenance you can read afterwards (C8).

It is deliberately not clever. Every line here is a line a host author will
write, and anything clever would be a line they would have to unpick.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Vendoring path: insert the repo root and import. No install, no package
# metadata, no import-time side effects (§1.2, M50). If this line works, a
# git submodule works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import (  # noqa: E402
    AutoApprove,
    Host,
    LocalFileSystem,
    MemoryStorage,
    ScriptedLLM,
    Session,
    SessionConfig,
    SubprocessExec,
)


class PrintEvents:
    """A complete EventPort. This is the entire UI of this host.

    A real host renders these into a panel; `phase` drives a progress bar,
    `diagnostic` populates a problems list, `remote` lights a warning that
    stays lit. The vocabulary is closed (§5.4) precisely so a host can switch
    on it without worrying about tomorrow's addition.
    """

    def event(self, kind: str, message: str, data: dict | None = None) -> None:
        if kind == "remote":
            print(f"  *** {message} ***")
        elif kind in ("warning", "error"):
            print(f"  ! {message}")
        else:
            print(f"  [{kind}] {message}")


def main() -> int:
    # What the "model" will say, in order. A real host passes an LLMPort
    # wrapping whatever it already has loaded.
    replies = [
        # 1. the plan
        "greeter.py — say hello to a named person\n",
        # 2. the file
        '```python\ndef greet(name):\n    """Return a greeting."""\n'
        '    return f"hello, {name}"\n\n\n'
        'def main():\n    """Entry point."""\n    print(greet("world"))\n'
        '    return 0\n```',
    ]

    # A real directory, because C4 means the engine BUILDS and TESTS what it
    # writes, and a build needs somewhere to run. `MemoryFileSystem` works
    # perfectly for planning, editing and the codemap — swap it in below and
    # the engine will say, in one plain sentence, that it cannot verify. That
    # is C7 working as intended rather than a limitation being hidden, and it
    # is worth seeing once.
    import tempfile
    workspace = tempfile.mkdtemp(prefix="tiny-host-")

    host = Host(
        llm=ScriptedLLM(replies, supports_tools=False),
        fs=LocalFileSystem(workspace),  # or MemoryFileSystem() — see above
        exec=SubprocessExec(),
        storage=MemoryStorage(),
        events=PrintEvents(),
        approval=AutoApprove(),         # a real host asks; see §6.5
    )

    print("Cognitive Coder — tiny host")
    print(f"no model, no network, no host application\nworkspace: "
          f"{workspace}\n")

    session = Session(host, config=SessionConfig(skeleton_first=False,
                                                 attempts=1))
    session.run("a module with a greet function")

    print("\n--- report " + "-" * 48)
    print(session.report())

    print("\n--- what was written " + "-" * 38)
    for path in sorted(host.fs.list("*")):
        if path.startswith((".cc_journal", ".cc_snapshots", ".cc_state")):
            continue
        print(f"\n### {path}")
        print(host.fs.read(path))

    print("\n--- provenance " + "-" * 45)
    # C8: every artefact is traceable to a model, a prompt and an outcome.
    for row in session.journal.events():
        print(f"  {row.get('event'):<14} {row.get('task', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
