# SPDX-License-Identifier: Apache-2.0
"""`ccoder` — the terminal host, and `ccoder doctor`.

Two jobs, and the second is the one the spec calls a required deliverable
rather than a nicety (§10.2a):

  * **`ccoder build`** — drive a session from a terminal. It is a HOST, and a
    deliberately small one: it implements the Ports and renders the events,
    and every line of it is an example of how to embed the engine.
  * **`ccoder doctor`** — print the install summary on demand, including
    **which interpreter is in use and where it came from**. When something
    behaves oddly six months from now, that line is the first question
    answered, and "run one command" beats "reconstruct what the installer
    did in March".

The CLI is also where C3 becomes visible. `--remote` exists, it is off, and
turning it on prints a banner that stays up. There is no environment variable
that enables it and no configuration file that can. That is M42, and it is the
reason the flag is spelled out rather than inferred.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
import sys

from . import langs
from .codemap import parse_treesitter
from .errors import CognitiveCoderError
from .ports import (
    DenyAll,
    Host,
    LocalFileSystem,
    MemoryStorage,
    SubprocessExec,
)
from .providers import available_providers, detect, make_provider
from .session import Session, SessionConfig
from .skills import SKILLS_DIR, STARTER_SKILLS, load_skills
from .version import IMPLEMENTED_PHASES, __version__

# The toolchains the installer records and `doctor` re-probes. The record is
# informational only — §6.1 probes again at runtime, so a compiler installed
# the week after install day simply works.
TOOLCHAINS = (
    ("python", ("python3", "python", "py"), "Python targets"),
    ("C/C++", ("gcc", "clang", "cc", "cl"), "C and C++ targets"),
    ("Rust", ("rustc",), "Rust targets"),
    ("Java", ("javac",), "Java targets"),
    ("Go", ("go",), "Go targets"),
    ("Node", ("node",), "JavaScript and TypeScript targets"),
    (".NET", ("dotnet",), "C# targets"),
    ("Zig", ("zig",), "Zig targets"),
    ("Lua", ("lua", "luajit"), "Lua targets"),
    ("Ruby", ("ruby",), "Ruby targets"),
    ("SQLite", ("sqlite3",), "SQL targets"),
    ("Godot", ("godot", "godot4"),
     "GDScript syntax checking, running and tests — without it GDScript is "
     "outline-and-edit only"),
)


class ConsoleEvents:
    """An EventPort that prints. The whole of a terminal host's UI.

    Note what it does with `remote`: the banner is printed every time,
    because a persistent indicator is the requirement (M42.6) and a terminal
    has no status bar to put one in.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def event(self, kind: str, message: str, data: dict | None = None
              ) -> None:
        if kind == "token":
            sys.stdout.write(message)
            sys.stdout.flush()
            return
        if kind == "remote":
            print(f"\n*** {message} ***", file=sys.stderr)
            return
        if kind in ("error", "warning"):
            print(f"[{kind}] {message}", file=sys.stderr)
            return
        if kind == "phase" and not self.verbose:
            return
        print(f"[{kind}] {message}")


class ConsoleApproval:
    """Approval-required, at a prompt. The library default, made real.

    A host that auto-approves must say so (§6.5); this one asks, and
    `--yes` is how an operator opts out, having been told what that means.
    """

    def __init__(self, auto: bool = False) -> None:
        self.auto = auto

    def approve_diff(self, summary: str, unified_diff: str) -> bool:
        if self.auto:
            return True
        print(f"\n--- {summary} ---")
        print(unified_diff[:4000] or "(no diff)")
        try:
            answer = input("Apply this change? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")

    def approve_remote(self, provider: str, bytes_out: int,
                       estimate: str) -> bool:
        print(f"\n*** {provider} would send about {bytes_out:,} bytes off "
              f"this machine: {estimate}")
        try:
            answer = input("Allow it? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in ("y", "yes")


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------

def interpreter_provenance() -> tuple[str, str]:
    """Which interpreter, and where it came from (§10.2a).

    "System, fetched, or vendored" is the distinction that matters. A venv
    does NOT give a project its own Python — it isolates packages, and the
    interpreter is whatever created it. A 3.9 `python` produces a 3.9 venv,
    and Cognitive Coder then fails at the first `X | None` annotation with a
    syntax error that looks like broken code rather than a wrong interpreter.
    This function is how that misunderstanding gets diagnosed in one command
    rather than in an afternoon.
    """
    exe = sys.executable or "unknown"
    real = os.path.realpath(exe)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.commonpath([real, root]) == root if root in real else False:
        return exe, "fetched into this clone (.python/)"
    for marker, label in ((os.sep + ".python" + os.sep, "fetched by the "
                                                        "installer into "
                                                        ".python/"),
                          (os.sep + ".venv" + os.sep, "the clone's .venv"),
                          (os.sep + ".tools" + os.sep, "fetched by uv into "
                                                       ".tools/")):
        if marker in real or marker in exe:
            return exe, label
    return exe, "the system Python"


def doctor(argv: Sequence[str] | None = None) -> int:
    """The install summary, on demand. A required deliverable (§10.2a).

    Exit code follows the installer contract (§10.1 rule 7): **0 if the core
    engine is usable**, non-zero only if it is not. A missing optional
    toolchain is not a failure, and saying so in the exit code is what stops
    a CI pipeline treating "no Rust installed" as a broken install.
    """
    ex = SubprocessExec()
    exe, provenance = interpreter_provenance()
    version = ".".join(str(n) for n in sys.version_info[:3])
    ok_core = sys.version_info >= (3, 11)

    print("=" * 60)
    print(" Cognitive Coder — installation summary")
    print("=" * 60)
    mark = "OK" if ok_core else "!!"
    print(f"  [{mark}] core engine          v{__version__}, 0 required deps, "
          f"phases {min(IMPLEMENTED_PHASES)}–{max(IMPLEMENTED_PHASES)} built")
    print(f"  [{'OK' if ok_core else '!!'}] Python                {version} "
          f"— {provenance}")
    print(f"       interpreter          {exe}")
    if not ok_core:
        print("       ^^ Python 3.11 or later is required. A virtual "
              "environment does not change this: it isolates packages, not "
              "the interpreter.")

    print()
    for label, binaries, cost in TOOLCHAINS:
        found = next((b for b in binaries if ex.which(b)), "")
        if found:
            print(f"  [OK] {label:<20} {ex.which(found)}")
        else:
            print(f"  [--] {label:<20} not found — {cost}")

    ts = parse_treesitter.degraded_note("c")
    print(f"  [{'--' if ts else 'OK'}] tree-sitter          "
          f"{ts or 'installed — C/C++/Rust/JS outlines are parsed'}")

    providers = available_providers()
    built = [n for n, v in providers.items() if v["built"]]
    absent = [n for n, v in providers.items() if not v["built"]]
    print(f"  [OK] local providers      {', '.join(built)}")
    if absent:
        print(f"  [--] remote providers     not built in this version "
              f"({', '.join(absent)}) — offline by default either way")

    print()
    print("  [--] entries are OPTIONAL: each disables one feature, not the "
          "tool.")
    print("  Re-run the installer to retry only what is missing.")
    print()
    endpoints = detect()
    if endpoints:
        print(f"  Local model endpoints answering right now: "
              f"{', '.join(endpoints)}")
    else:
        print("  No local model endpoint is answering. Start llama.cpp "
              "server, Ollama or LM Studio, or point --url at one.")
    available = langs.available_ids(ex)
    print(f"  Languages usable right now ({len(available)} of "
          f"{len(langs.ids())}): {', '.join(available)}")
    return 0 if ok_core else 1


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def _request_from(args: argparse.Namespace) -> tuple[str, object] | None:
    """The build request, from --spec or from the positional argument.

    Returns (text, Spec) or None after printing why it cannot continue. The
    error goes to stderr as a sentence rather than a traceback (C6): this is
    the first thing that happens in a long operation, and a stack trace here
    costs the whole run for a mistyped filename.
    """
    from . import spec as spec_mod

    if args.spec and args.request:
        print("Pass a request or --spec, not both. They are two ways to say "
              "the same thing and I cannot know which you meant.",
              file=sys.stderr)
        return None
    if args.spec:
        try:
            loaded = spec_mod.load(args.spec)
        except spec_mod.SpecError as exc:
            print(f"Cannot read that specification: {exc}", file=sys.stderr)
            return None
        return loaded.text, loaded
    if not args.request:
        print("Nothing to build. Give a request in a sentence, or point at a "
              "file with --spec plan.md.", file=sys.stderr)
        return None
    return args.request, spec_mod.from_text(args.request)


def _print_preview(pv: dict) -> None:
    """What the engine understood, before it is asked to write anything."""
    print()
    if pv["title"]:
        print(f"  {pv['title']}")
    print(f"  model: {pv['model'] or 'unknown'} · "
          f"request ~{pv['approx_tokens']} tokens"
          + (f" of {pv['context_tokens']} context"
             if pv["context_tokens"] else ""))
    print()
    print(f"  Build order ({len(pv['files'])} file(s)):")
    for i, path in enumerate(pv["files"], 1):
        purpose = pv["purposes"].get(path, "")
        print(f"    {i}. {path}"
              + (f"  — {purpose[:60]}" if purpose else ""))

    #: The two numbers whose disagreement went unnoticed for a whole build.
    #: Printed together, always, including when they agree — a check you only
    #: see when it fails is one you do not know is running.
    req, planned = pv["tests_required"], pv["tests_planned"]
    print()
    print(f"  Tests: {len(req)} named in the request, "
          f"{len(planned)} in the plan")
    for path in planned:
        print(f"    + {path}"
              + ("  (added — the plan had left it out)"
                 if path in req and any("left out" in c
                                        for c in pv["caveats"]) else ""))
    if pv["tests_missing"]:
        print("    [!] named in the request but NOT planned: "
              + ", ".join(pv["tests_missing"]))
        print("        A build with no tests cannot report whether it worked.")
    elif not req and not planned:
        print("    none — nothing will verify the result beyond 'it imports'")

    for note in pv["warnings"] + pv["caveats"]:
        print(f"  note: {note}")
    print()


def build(args: argparse.Namespace) -> int:
    got = _request_from(args)
    if got is None:
        return 2
    request, _spec = got

    root = os.path.abspath(args.project or os.getcwd())
    events = ConsoleEvents(verbose=args.verbose)
    host = Host(
        fs=LocalFileSystem(root), exec=SubprocessExec(),
        storage=MemoryStorage(os.path.join(root, ".cc_state")),
        events=events,
        approval=(ConsoleApproval(auto=args.yes) if not args.dry_run
                  else DenyAll()))

    try:
        provider = make_provider(
            "openai_compatible", base_url=args.url, model=args.model)
    except CognitiveCoderError as exc:
        print(str(exc), file=sys.stderr)          # C6: a sentence, not a trace
        return 2
    host.llm = provider

    caps = provider.capabilities()
    if not caps.loaded:
        print("No model is loaded at that endpoint, so there is nothing to "
              "ask. Start a model server, or pass --url. `ccoder doctor` "
              "will show which endpoints are answering.", file=sys.stderr)
        return 3
    print(f"Model: {caps.name} · {caps.context_tokens} tokens of context · "
          f"tools {'yes' if caps.supports_tools else 'no'}")

    session = Session(host, config=SessionConfig(
        lang=args.lang, attempts=args.attempts,
        temperature=args.temperature, max_tokens=args.max_tokens,
        wall_clock_s=args.budget * 60 if args.budget else 0.0,
        max_files=args.max_files,
        skeleton_first=not args.no_skeleton,
        use_skills=not args.no_skills))
    if args.preview:
        try:
            _print_preview(session.preview(request))
        except CognitiveCoderError as exc:
            print(str(exc), file=sys.stderr)
            return 4
        print("  Nothing was written. Re-run without --preview to build.")
        return 0

    try:
        session.run(request)
    except KeyboardInterrupt:
        session.cancel()
        print("\nStopping at the next safe point…", file=sys.stderr)
        session.finish()
    except CognitiveCoderError as exc:
        print(str(exc), file=sys.stderr)
        session.journal.error(str(exc), getattr(exc, "detail", ""))
        return 4
    print()
    print(session.report())
    return 0 if all(o.ok for o in session.outcomes) else 1


def resume(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.project or os.getcwd())
    host = Host(fs=LocalFileSystem(root), exec=SubprocessExec(),
                storage=MemoryStorage(os.path.join(root, ".cc_state")),
                events=ConsoleEvents(verbose=args.verbose),
                approval=ConsoleApproval(auto=args.yes))
    sessions = Session.previous_sessions(host)
    if not args.session:
        if not sessions:
            print("There are no previous sessions in this project.")
            return 1
        print("Previous sessions:")
        for name in sessions:
            print(f"  {name}")
        return 0
    host.llm = make_provider("openai_compatible", base_url=args.url,
                             model=args.model)
    session = Session.resume(host, args.session)
    session.run()
    print(session.report())
    return 0


def history(args: argparse.Namespace) -> int:
    """What did it do to my project? Answered from the transaction log."""
    root = os.path.abspath(args.project or os.getcwd())
    host = Host(fs=LocalFileSystem(root),
                storage=MemoryStorage(os.path.join(root, ".cc_state")))
    from .patcher import Patcher
    rows = Patcher(host.fs, host.storage, host.approval).history()
    if not rows:
        print("Nothing has been changed in this project by Cognitive Coder.")
        return 0
    for rec in rows:
        seal = " SEALED" if rec.sealed else ""
        files = ", ".join(rec.files) or "—"
        print(f"  {rec.seq:>4}  {rec.state:<12}{seal:<8} {rec.task_id:<16} "
              f"{files}")
        if rec.note:
            print(f"        {rec.note}")
    return 0


def skills_cmd(args) -> int:
    """`ccoder skills list | deploy | new <name>` — deployed guidance (F3).

    `deploy` NEVER overwrites: the starter files are a seed, and once a file
    exists it belongs to the project, not to us. Re-running deploy after
    editing must be safe or nobody will dare run it twice.
    """
    root = os.path.abspath(args.project or os.getcwd())
    fs = LocalFileSystem(root)

    if args.action == "deploy":
        wrote, kept = [], []
        for filename, content in STARTER_SKILLS.items():
            path = f"{SKILLS_DIR}/{filename}"
            if fs.exists(path):
                kept.append(path)
            else:
                fs.write(path, content)
                wrote.append(path)
        for path in wrote:
            print(f"  deployed  {path}")
        for path in kept:
            print(f"  kept      {path}  (already exists; not touched)")
        if wrote:
            print("\nEdit them — they are starting points, not rules we "
                  "chose for you.\nThey load into every session's prompt "
                  "from now on; `--no-skills` turns that off per run.")
        return 0

    if args.action == "new":
        if not args.name:
            print("A name is needed: `ccoder skills new api-conventions`",
                  file=sys.stderr)
            return 2
        stem = "".join(c if c.isalnum() or c in "-_" else "-"
                       for c in args.name.strip().lower()).strip("-")
        path = f"{SKILLS_DIR}/{stem}.md"
        if fs.exists(path):
            print(f"{path} already exists; edit it instead.", file=sys.stderr)
            return 2
        fs.write(path, ("---\n"
                        f"name: {stem}\n"
                        "description: what this guidance covers, in a line\n"
                        "---\n"
                        "Write the rules here, one short paragraph each.\n"))
        print(f"  created  {path}")
        return 0

    # list — show exactly what a session would load, and what it would not.
    load = load_skills(fs, lang=args.lang)
    if not load.skills and not load.skipped:
        print(f"No skills deployed (nothing in {SKILLS_DIR}/).")
        print("`ccoder skills deploy` writes an editable starter pack.")
        return 0
    if load.skills:
        scope = f" for a {args.lang} session" if args.lang else ""
        print(f"Active{scope}:")
        for sk in load.skills:
            langs_note = ", ".join(sk.langs) if sk.langs else "all languages"
            desc = f"  — {sk.description}" if sk.description else ""
            print(f"  {sk.name:<20} {sk.path}  [{langs_note}, "
                  f"{len(sk.body)} chars, {sk.sha256[:12]}]{desc}")
    if load.skipped:
        print("Not loaded:")
        for path, reason in load.skipped:
            print(f"  {path}: {reason}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ccoder",
        description="Write, build, test and fix code with a local model.")
    parser.add_argument("--version", action="version",
                        version=f"cognitive-coder {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="print the install summary and exit")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", "-p", default="",
                        help="project root (default: the current directory)")
    common.add_argument("--url", default="http://127.0.0.1:8080",
                        help="an OpenAI-compatible endpoint (local by "
                             "default; a non-local URL needs remote mode)")
    common.add_argument("--model", default="", help="model name at that URL")
    common.add_argument("--verbose", "-v", action="store_true")
    common.add_argument("--yes", "-y", action="store_true",
                        help="apply changes without asking. Undo is then the "
                             "only safety net — snapshots are kept, but read "
                             "the history afterwards.")

    b = sub.add_parser("build", parents=[common],
                       help="plan and build from a request")
    #: Optional, because --spec replaces it. A sentence types fast and plans
    #: badly; anything worth twenty minutes of model time is worth writing in
    #: an editor and keeping next to the code it produced.
    b.add_argument("request", nargs="?", default="",
                   help="what you want built, in a sentence "
                        "(or use --spec for a file)")
    b.add_argument("--spec", "-f", default="", metavar="FILE",
                   help="read the build request from a .md or .txt file")
    b.add_argument("--preview", action="store_true",
                   help="plan and print what would be built, then stop "
                        "without generating anything")
    b.add_argument("--lang", default="python")
    b.add_argument("--attempts", type=int, default=4)
    b.add_argument("--temperature", type=float, default=0.15)
    b.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    b.add_argument("--budget", type=float, default=0.0,
                   help="wall-clock ceiling in minutes; it stops cleanly and "
                        "leaves the session resumable")
    b.add_argument("--max-files", dest="max_files", type=int, default=12,
                   help="how many files one plan may contain (default 12); "
                        "raise it for a large written specification")
    b.add_argument("--no-skeleton", action="store_true",
                   help="skip the compiling-skeleton step (not advised)")
    b.add_argument("--dry-run", action="store_true",
                   help="refuse every write, to see what it would do")
    b.add_argument("--no-skills", action="store_true",
                   help="build without the deployed skills in "
                        f"{SKILLS_DIR}/ (see `ccoder skills`)")

    r = sub.add_parser("resume", parents=[common],
                       help="resume a session, or list them")
    r.add_argument("session", nargs="?", default="")

    sub.add_parser("history", parents=[common],
                   help="what has been changed in this project")

    sk = sub.add_parser("skills", parents=[common],
                        help="project guidance files that load into every "
                             "session's prompt")
    sk.add_argument("action", nargs="?", default="list",
                    choices=("list", "deploy", "new"),
                    help="list what would load; deploy the starter pack; "
                         "new <name> scaffolds an empty skill")
    sk.add_argument("name", nargs="?", default="",
                    help="the new skill's name (for `new`)")
    sk.add_argument("--lang", default="",
                    help="list as a session in this language would see it")

    args = parser.parse_args(argv)
    if args.command == "doctor" or args.command is None:
        return doctor()
    if args.command == "build":
        return build(args)
    if args.command == "resume":
        return resume(args)
    if args.command == "history":
        return history(args)
    if args.command == "skills":
        return skills_cmd(args)
    parser.print_help()
    return 0


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(main())
