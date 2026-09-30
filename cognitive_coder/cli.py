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

The CLI is also where C3 becomes visible. `--remote PROVIDER` exists, it is
off, and turning it on prints a banner that stays up. There is no environment
variable that enables it and no configuration file that can. That is M42, and
it is the reason the flag is spelled out rather than inferred. (It was
documented here before it existed: a remote URL could then only ever produce
"turn on remote mode for this session", with no way to do so.)

Exit codes, the same for `build` and `resume`: 0 every file verified, 1 some
file did not (or nothing was built at all), 2 the command line or the project
was refused before any work, 3 no model is loaded, 4 the engine stopped on an
error it could name.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import os
import sys
import urllib.parse

from . import langs
from .codemap import parse_treesitter
from .errors import CognitiveCoderError
from .filestorage import JsonFileStorage
from .ports import DenyAll, Host, LocalFileSystem, SubprocessExec
from .providers import available_providers, detect, key_from, make_provider
from .session import Session, SessionConfig
from .skills import SKILLS_DIR, STARTER_SKILLS, load_skills
from .types import TaskOutcome
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

#: Where the CLI keeps engine state inside a project. On disk, in
#: `JsonFileStorage`: the transaction log lives here, and `ccoder history`
#: runs in a later process than the build that wrote it.
STATE_DIR = ".cc_state"

#: §6.5: a host that auto-applies must say so. Printed once, before any work.
AUTO_APPLY = "auto-apply is ON: every diff will be written without asking"

NO_MODEL = ("No model is loaded at that endpoint, so there is nothing to "
            "ask. Start a model server, or pass --url. `ccoder doctor` will "
            "show which endpoints are answering.")


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
        # `--yes` is NOT consulted here. It answers "may you write my
        # files"; sending them off this machine is a different question
        # and C3 does not let one answer stand in for the other.
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

def _has_dir(path: str, name: str) -> bool:
    """Whether `name` is a directory component of `path`."""
    parts = os.path.normcase(os.path.abspath(path)).replace("\\", "/")
    return f"/{os.path.normcase(name)}/" in parts


def _inside(path: str, root: str) -> bool:
    a = os.path.normcase(os.path.abspath(path))
    b = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:                  # different drives on Windows
        return False


def _origin(real: str, clone: str) -> str:
    """Where a REAL (symlink-resolved) interpreter came from, as a noun."""
    if _has_dir(real, ".python"):
        return "the Python fetched by the installer into .python/"
    if _has_dir(real, ".tools"):
        return "the Python fetched by uv into .tools/"
    if _inside(real, clone):
        return "a Python inside this clone"
    return "the system Python"


def _venv_base(exe: str) -> str:
    """The directory of the interpreter a venv was made from, or "".

    Read from `pyvenv.cfg`'s `home`, which is the only place a venv records
    it: a venv made with `--copies` (the Windows default) has no symlink to
    follow, so its own path says nothing about where its Python came from.
    """
    cfg = os.path.join(os.path.dirname(os.path.dirname(exe)), "pyvenv.cfg")
    try:
        with open(cfg, encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition("=")
                if key.strip().lower() == "home" and value.strip():
                    return value.strip()
    except OSError:
        pass
    return ""


def interpreter_provenance(exe: str | None = None, clone: str | None = None
                           ) -> tuple[str, str]:
    """Which interpreter, and where it came from (§10.2a).

    "System, fetched, or vendored" is the distinction that matters. A venv
    does NOT give a project its own Python — it isolates packages, and the
    interpreter is whatever created it. A 3.9 `python` produces a 3.9 venv,
    and Cognitive Coder then fails at the first `X | None` annotation with a
    syntax error that looks like broken code rather than a wrong interpreter.
    This function is how that misunderstanding gets diagnosed in one command
    rather than in an afternoon.

    So a venv is reported AS a venv, together with the interpreter under it.
    It used to be tested last, after "is it anywhere inside the clone?" —
    which every `.venv` in the clone is — so the clone's own venv was
    reported as "fetched into this clone (.python/)", the one answer that
    hides exactly the misunderstanding this exists to expose.

    `exe` and `clone` are parameters so the classification is testable
    without the interpreter under test being the one running the test.
    """
    exe = exe or sys.executable or "unknown"
    clone = clone or os.path.dirname(os.path.dirname(os.path.abspath(
        __file__)))
    real = os.path.realpath(exe)
    if not _has_dir(exe, ".venv"):
        return exe, _origin(real, clone)
    venv = os.path.dirname(os.path.dirname(exe))
    where = ("the clone's .venv" if _inside(exe, clone)
             else f"a .venv at {venv}")
    base = _venv_base(exe)
    if base:
        return exe, f"{where}, created from {_origin(base, clone)} ({base})"
    if os.path.normcase(real) != os.path.normcase(os.path.abspath(exe)):
        return exe, f"{where}, created from {_origin(real, clone)} ({real})"
    return exe, (f"{where} — its pyvenv.cfg does not say which Python it "
                 f"was made from")


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

    # Local and remote on separate lines. They were one line headed "local
    # providers", which listed five providers that send everything they are
    # given to someone else's computer — on the one screen whose job is to
    # say accurately what this install will do.
    providers = available_providers()
    local = [n for n, v in providers.items() if v["built"] and not v["remote"]]
    remote = [n for n, v in providers.items() if v["built"] and v["remote"]]
    absent = [n for n, v in providers.items() if not v["built"]]
    print(f"  [OK] local providers      {', '.join(local)}")
    if remote:
        print(f"  [OK] remote providers     {', '.join(remote)} — each off "
              f"unless `--remote NAME` turns it on for one run")
    if absent:
        print(f"  [--] not built            {', '.join(absent)}")

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
# refusing what cannot work, before any work
# --------------------------------------------------------------------------

def _project_root(args: argparse.Namespace) -> str | None:
    """The project folder, or None after saying why it will not be used.

    A missing folder is refused rather than created: `LocalFileSystem`
    creates its root, so a typo in `--project` used to start a whole new
    tree next to the real one — and every later command, pointed at the
    real one, then found nothing. `--create` is how a NEW project is asked
    for, and the creation is announced.
    """
    root = os.path.abspath(args.project or os.getcwd())
    if os.path.isdir(root):
        return root
    if os.path.exists(root):
        print(f"{root} is a file, not a folder, so it cannot be a project. "
              f"Nothing was done.", file=sys.stderr)
        return None
    if not getattr(args, "create", False):
        print(f"There is no folder at {root}, so nothing was done. Check the "
              f"spelling, or pass --create to start a new project there.",
              file=sys.stderr)
        return None
    os.makedirs(root)
    print(f"Created a new project folder at {root}.")
    return root


def url_problem(url: str) -> str:
    """Why `--url` cannot be a model endpoint, or "" if it can.

    Checked before anything is contacted. `ftp://…`, `127.0.0.1:8080` and
    `…/v1/` all used to reach the provider, fail its probe, and be reported
    as "No model is loaded at that endpoint" — true, and useless, because
    the fix is in the command line and not in the model server.
    """
    text = (url or "").strip()
    example = "for example --url http://127.0.0.1:8080"
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError as exc:
        return (f"--url {text!r} is not an address ({exc}), so nothing was "
                f"contacted. Give the whole URL, {example}.")
    if parts.scheme not in ("http", "https"):
        if "://" not in text:
            return (f"--url {text!r} has no scheme, so nothing was "
                    f"contacted. Give the whole URL, {example}.")
        return (f"--url {text!r} uses {parts.scheme}://, and a model "
                f"endpoint speaks http:// or https://. Nothing was "
                f"contacted.")
    if not parts.hostname:
        return (f"--url {text!r} names no host, so nothing was contacted. "
                f"Give one, {example}.")
    try:
        parts.port                                  # noqa: B018 — validates
    except ValueError:
        return (f"--url {text!r} has a port that is not a number, so "
                f"nothing was contacted.")
    if parts.query or parts.fragment:
        return (f"--url {text!r} carries a ?query or #fragment, which would "
                f"be glued onto every request path. Give the address alone.")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        base = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, path[:-3], "", ""))
        return (f"--url {text!r} ends in /v1, and the engine adds "
                f"/v1/chat/completions itself — every request would go to "
                f".../v1/v1/... and fail as if no model were loaded. Use "
                f"--url {base}.")
    return ""


def remote_problem(name: str) -> str:
    """Why `--remote NAME` cannot be turned on, or "" if it can.

    Checked BEFORE the gate opens, so a typo never prints "REMOTE MODE IS
    ON" for a provider that does not exist.
    """
    if not name:
        return ""
    known = available_providers()
    remote = sorted(n for n, v in known.items() if v["remote"])
    if name not in known:
        return (f"There is no provider called {name!r}, so remote mode was "
                f"not turned on. Remote providers: {', '.join(remote)} — or "
                f"openai_compatible for a --url that is not on this "
                f"machine.")
    if not known[name]["remote"] and name != "openai_compatible":
        return (f"{name} runs on this machine, so --remote has nothing to "
                f"turn on for it. Leave --remote off.")
    return ""


def _host(root: str, args: argparse.Namespace, approval) -> Host:
    return Host(fs=LocalFileSystem(root), exec=SubprocessExec(),
                storage=JsonFileStorage(os.path.join(root, STATE_DIR)),
                events=ConsoleEvents(verbose=args.verbose),
                approval=approval)


def _model(args: argparse.Namespace, host: Host,
           session: Session | None):
    """The LLMPort for this run: local by default, remote only by --remote.

    With `--remote`, the provider is built by the SESSION, after the session
    has turned the gate on — so it is bound to this run's gate, redaction
    count, budget and journal (M42), and the banner is up before anything
    could be sent. Without it, the ungated constructor is used, and it
    refuses any URL that is not on this machine.
    """
    if not args.remote:
        return make_provider("openai_compatible", base_url=args.url,
                             model=args.model)
    assert session is not None
    session.enable_remote(args.remote, reason="--remote on the command line")
    if args.remote == "openai_compatible":
        return session.remote_provider("openai_compatible",
                                       base_url=args.url, model=args.model)
    return session.remote_provider(
        args.remote, model=args.model,
        api_key=key_from(host.storage, args.remote))


def _refusal(exc: CognitiveCoderError, args: argparse.Namespace) -> str:
    """The provider's sentence, plus the flag that answers it, if one does."""
    text = str(exc)
    if not args.remote and "remote mode" in text:
        text += (" For this run only, that is --remote openai_compatible; "
                 "it still asks before anything is sent.")
    return text


def _loaded(host: Host) -> bool:
    """Print what answered, or why nothing did. False means exit 3."""
    caps = host.llm.capabilities()
    if not caps.loaded:
        print(NO_MODEL, file=sys.stderr)
        return False
    print(f"Model: {caps.name} · {caps.context_tokens} tokens of context · "
          f"tools {'yes' if caps.supports_tools else 'no'}")
    return True


def _verdicts(outcomes: Sequence[TaskOutcome]) -> list[TaskOutcome]:
    """The LAST outcome per file, in the order files were first built.

    A module repaired against its failing test has two outcomes, and the
    later one is the verdict. Judging all of them would fail a run whose
    every file ended verified.
    """
    last: dict[str, TaskOutcome] = {}
    for outcome in outcomes:
        last.pop(outcome.path, None)
        last[outcome.path] = outcome
    order = list(dict.fromkeys(o.path for o in outcomes))
    return [last[p] for p in order]


def _exit_code(outcomes: Sequence[TaskOutcome]) -> int:
    """0 only if something was built and every file ended verified.

    `all([])` is True, which is how a run that built NOTHING — out of
    budget before the first file, or stopped with Ctrl-C — exited 0.
    """
    verdicts = _verdicts(outcomes)
    return 0 if verdicts and all(o.ok for o in verdicts) else 1


def _run(session: Session, request: str = "") -> int | None:
    """Run to the end. None when it ended; 4 when it stopped on an error."""
    try:
        session.run(request)
    except KeyboardInterrupt:
        session.cancel()
        print("\nStopping at the next safe point…", file=sys.stderr)
        session.finish()
    except CognitiveCoderError as exc:
        print(str(exc), file=sys.stderr)          # C6: a sentence, not a trace
        session.journal.error(str(exc), getattr(exc, "detail", ""))
        return 4
    return None


def _finish(session: Session) -> int:
    print()
    print(session.report())
    if not session.outcomes:
        print(f"Nothing was built, so this run did not succeed. What was "
              f"planned is resumable: ccoder resume {session.id}",
              file=sys.stderr)
    return _exit_code(session.outcomes)


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
    root = _project_root(args)
    if root is None:
        return 2
    problem = url_problem(args.url) or remote_problem(args.remote)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.yes and not (args.preview or args.dry_run):
        print(AUTO_APPLY)

    host = _host(root, args, approval=(ConsoleApproval(auto=args.yes)
                                       if not args.dry_run else DenyAll()))
    config = SessionConfig(
        lang=args.lang, attempts=args.attempts,
        temperature=args.temperature, max_tokens=args.max_tokens,
        wall_clock_s=args.budget * 60 if args.budget else 0.0,
        max_files=args.max_files,
        skeleton_first=not args.no_skeleton,
        use_skills=not args.no_skills)
    # With --remote the session exists FIRST: its gate is what the provider
    # is bound to. Without it, nothing is built until a model is known to be
    # there, as before.
    session = Session(host, config=config) if args.remote else None
    try:
        host.llm = _model(args, host, session)
    except CognitiveCoderError as exc:
        print(_refusal(exc, args), file=sys.stderr)
        return 2
    if not _loaded(host):
        return 3
    session = session or Session(host, config=config)

    if args.preview:
        try:
            _print_preview(session.preview(request))
        except CognitiveCoderError as exc:
            print(str(exc), file=sys.stderr)
            return 4
        print("  Nothing was written. Re-run without --preview to build.")
        return 0

    stopped = _run(session, request)
    return stopped if stopped is not None else _finish(session)


# --------------------------------------------------------------------------
# resume, history, skills
# --------------------------------------------------------------------------

def resume(args: argparse.Namespace) -> int:
    """Finish a session from its journal — with build's refusals and codes.

    It had none of them: a mistyped id was a FileNotFoundError traceback, a
    stopped model server a NoModelLoadedError traceback, and a resumed
    build that failed every file still exited 0.
    """
    root = _project_root(args)
    if root is None:
        return 2
    host = _host(root, args, approval=ConsoleApproval(auto=args.yes))
    sessions = Session.previous_sessions(host)
    if not args.session:
        if not sessions:
            print("There are no previous sessions in this project.")
            return 1
        print("Previous sessions:")
        for name in sessions:
            print(f"  {name}")
        return 0
    if args.session not in sessions:
        known = (f" The ones here are: {', '.join(sessions)}." if sessions
                 else " This project has none.")
        print(f"There is no session called {args.session} in {root}, so "
              f"there is nothing to resume.{known}", file=sys.stderr)
        return 2
    problem = url_problem(args.url) or remote_problem(args.remote)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.yes:
        print(AUTO_APPLY)

    # Local: ask the model before touching the journal, so a stopped server
    # does not leave a session_start with no work after it. Remote: the
    # provider is bound to the resumed session's gate, so that comes first.
    if not args.remote:
        try:
            host.llm = _model(args, host, None)
        except CognitiveCoderError as exc:
            print(_refusal(exc, args), file=sys.stderr)
            return 2
        if not _loaded(host):
            return 3
    try:
        session = Session.resume(host, args.session)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CognitiveCoderError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    if args.remote:
        try:
            host.llm = _model(args, host, session)
        except CognitiveCoderError as exc:
            print(_refusal(exc, args), file=sys.stderr)
            return 2
        if not _loaded(host):
            return 3

    pending = [t for t in (session.plan.tasks if session.plan else ())
               if t.status == "pending"]
    if not pending:
        session.finish()
        print(f"Every file in {args.session} was already verified; there "
              f"is nothing left to do.")
        return 0
    stopped = _run(session)
    return stopped if stopped is not None else _finish(session)


def history(args: argparse.Namespace) -> int:
    """What did it do to my project? Answered from the transaction log."""
    root = _project_root(args)
    if root is None:
        return 2
    host = Host(fs=LocalFileSystem(root),
                storage=JsonFileStorage(os.path.join(root, STATE_DIR)))
    from .patcher import Patcher
    try:
        rows = Patcher(host.fs, host.storage, host.approval).history()
    except CognitiveCoderError as exc:
        print(str(exc), file=sys.stderr)
        return 4
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
    root = _project_root(args)
    if root is None:
        return 2
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
        if not stem.strip("_"):
            # "  " and "!!!" both reduced to nothing, and the file was then
            # written as `.ccoder/skills/.md` — a hidden file with no name.
            print(f"{args.name!r} has no letters or digits to make a file "
                  f"name from, so nothing was created. Try `ccoder skills "
                  f"new api-conventions`.", file=sys.stderr)
            return 2
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


# --------------------------------------------------------------------------
# the parser — one registration function per subcommand
# --------------------------------------------------------------------------

#: What a subcommand's registration returns: the function that runs it.
Handler = Callable[[argparse.Namespace], int]


def _at_least(minimum: float, kind: type = int) -> Callable[[str], float]:
    """An argparse `type` that refuses numbers which cannot mean anything.

    `--max-files -3` was accepted and printed "only the first -3 were kept";
    `--budget -1` stopped before the first file and exited 0.
    """
    noun = "a whole number" if kind is int else "a number"

    def parse(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{text!r} is not {noun}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"must be {minimum:g} or more, not {text}")
        return value

    return parse


def _common_options() -> argparse.ArgumentParser:
    """The options every project-scoped subcommand shares."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project", "-p", default="",
                        help="project root (default: the current directory)")
    common.add_argument("--create", action="store_true",
                        help="create the --project folder if it does not "
                             "exist (otherwise a missing folder is refused, "
                             "because it is usually a typo)")
    common.add_argument("--url", default="http://127.0.0.1:8080",
                        help="an OpenAI-compatible endpoint (local by "
                             "default; a non-local URL needs --remote)")
    common.add_argument("--model", default="", help="model name at that URL")
    common.add_argument("--verbose", "-v", action="store_true")
    common.add_argument("--yes", "-y", action="store_true",
                        help="apply changes without asking. Undo is then the "
                             "only safety net — snapshots are kept, but read "
                             "the history afterwards.")
    return common


def _add_remote_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--remote", default="", metavar="PROVIDER",
        type=lambda s: s.strip().lower(),
        help="turn remote mode on for THIS run only and use PROVIDER "
             "(anthropic, google, mistral, openai, openrouter — or "
             "openai_compatible for a --url that is not on this machine). "
             "A banner stays up, secrets are redacted, and it asks before "
             "the first byte leaves. Nothing else turns it on.")


def _add_doctor_parser(sub, common) -> Handler:
    sub.add_parser("doctor", help="print the install summary and exit")
    return lambda _args: doctor()


def _add_build_parser(sub, common) -> Handler:
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
    b.add_argument("--attempts", type=_at_least(1), default=4)
    b.add_argument("--temperature", type=_at_least(0, float), default=0.15)
    b.add_argument("--max-tokens", dest="max_tokens", type=_at_least(1),
                   default=2048)
    b.add_argument("--budget", type=_at_least(0, float), default=0.0,
                   help="wall-clock ceiling in minutes (0, the default, is "
                        "none); it stops cleanly and leaves the session "
                        "resumable")
    b.add_argument("--max-files", dest="max_files", type=_at_least(1),
                   default=12,
                   help="how many files one plan may contain (default 12); "
                        "raise it for a large written specification")
    b.add_argument("--no-skeleton", action="store_true",
                   help="skip the compiling-skeleton step (not advised)")
    b.add_argument("--dry-run", action="store_true",
                   help="refuse every write, to see what it would do")
    b.add_argument("--no-skills", action="store_true",
                   help="build without the deployed skills in "
                        f"{SKILLS_DIR}/ (see `ccoder skills`)")
    _add_remote_option(b)
    return build


def _add_resume_parser(sub, common) -> Handler:
    r = sub.add_parser("resume", parents=[common],
                       help="resume a session, or list them")
    r.add_argument("session", nargs="?", default="")
    _add_remote_option(r)
    return resume


def _add_history_parser(sub, common) -> Handler:
    sub.add_parser("history", parents=[common],
                   help="what has been changed in this project")
    return history


def _add_skills_parser(sub, common) -> Handler:
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
    return skills_cmd


def _add_audit_parser(sub, common) -> Handler | None:
    """`ccoder audit`, registered by `cognitive_coder/audit.py` itself.

    That module owns its own parser (`add_cli(subparsers)`) and its own
    handler (`run_cli(args) -> int`); this is the whole of the CLI's side
    of the contract. Absent module, absent subcommand — but ONLY when the
    module itself is what is missing: a broken import inside audit.py must
    surface as the error it is, not as a subcommand that quietly vanished.
    """
    import importlib

    name = f"{__package__}.audit"
    try:
        # import_module, not `from . import audit`: the latter reports a
        # missing submodule as a bare ImportError with no `.name`, which
        # cannot be told apart from audit.py failing on its own import.
        audit = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        return None
    audit.add_cli(sub, common)
    return audit.run_cli


#: The subcommands, in `--help` order. A new one is a function above and a
#: row here; `main()` itself does not change.
SUBCOMMANDS: tuple[tuple[str, Callable[..., Handler | None]], ...] = (
    ("doctor", _add_doctor_parser),
    ("build", _add_build_parser),
    ("resume", _add_resume_parser),
    ("history", _add_history_parser),
    ("skills", _add_skills_parser),
    ("audit", _add_audit_parser),
)


def make_parser() -> tuple[argparse.ArgumentParser, dict[str, Handler]]:
    """The parser, and which function runs each subcommand."""
    parser = argparse.ArgumentParser(
        prog="ccoder",
        description="Write, build, test and fix code with a local model.")
    parser.add_argument("--version", action="version",
                        version=f"cognitive-coder {__version__}")
    sub = parser.add_subparsers(dest="command")
    common = _common_options()
    dispatch: dict[str, Handler] = {}
    for name, register in SUBCOMMANDS:
        handler = register(sub, common)
        if handler is not None:
            dispatch[name] = handler
    return parser, dispatch


def main(argv: Sequence[str] | None = None) -> int:
    parser, dispatch = make_parser()
    args = parser.parse_args(argv)
    # No subcommand is `doctor`: the first thing anyone types after an
    # install is the bare command, and the summary is the useful answer.
    return dispatch[args.command or "doctor"](args)


if __name__ == "__main__":                               # pragma: no cover
    raise SystemExit(main())
