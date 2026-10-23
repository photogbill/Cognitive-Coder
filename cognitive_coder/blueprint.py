# SPDX-License-Identifier: Apache-2.0
"""Blueprint — a system of modules of files, built one module at a time.

Bill (2026-10-22): *"instead of writing 10 files, it might write 10 modules
each with 10 files for one system."*

Why this is not "the planner, bigger": the two spec-truncation bugs and
"neither model saw the spec" (HYBRID.md) get catastrophically worse at a
hundred files. A system is a different object: it has internal CONTRACTS,
a BUILD ORDER, and a reason for each boundary.

The shape:

    plan        → a conversation FIRST (systems_architect persona): what
                  the system is, what is in and out, which boundaries
                  matter, what it must never do. Bill (2026-10-22): the
                  back-and-forth before handing over for the design. A
                  one-shot decomposition of a system nobody talked through
                  is the truncated-spec failure again, one level up.
    system spec + the agreed conversation
      └─ decompose  → modules: purpose, owned files, the PUBLIC CONTRACT
      │               (the pinned interface one level up: the exact
      │               signatures other modules may call), dependencies
      ├─ validate   → every contract name defined by exactly one module,
      │               every dependency exists, no cycle, every file owned
      │               by one module — refused otherwise, with the reason
      ├─ order      → dependencies first (a leaf can be tested alone,
      │               which is the only reason to build it first)
      ├─ stubs      → before a module's session, the contracts it depends
      │               on exist as stubs (raise NotImplementedError) unless
      │               the real module is already built — so the module
      │               compiles and its tests run against contracts
      ├─ brief      → the module's request: its purpose, its files, the
      │               contract it MUST provide (exact signatures), the
      │               contracts it MAY call, the ledger rows so far. Never
      │               the whole system spec.
      └─ ledger     → BLUEPRINT.md: every module's contract, its outcome,
                      the deviations its session recorded — band 0 of
                      every later module's prompt

One SESSION per module (the host constructs it: its own journal,
snapshots, verify, postmortem), run in order. That is what makes a
hundred-file failure diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from . import speclint, structured
from .personas import PERSONAS, strip_think
from .types import Message

LEDGER = "BLUEPRINT.md"
_SCHEMA = """{"system": "one line", "modules": [{"id": "core", "name": "Core models", "purpose": "two sentences", "files": ["src/core/models.py", "tests/test_core_models.py"], "contract": [{"name": "Ship", "kind": "class|function|constant", "signature": "class Ship(hull: float, shields: list[float])", "doc": "one line"}], "depends": ["other_id"]}]}"""
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class Contract:
    name: str
    kind: str               # class | function | constant
    signature: str
    doc: str = ""


@dataclass
class Module:
    id: str
    name: str
    purpose: str
    files: list[str]
    contract: list[Contract] = field(default_factory=list)
    depends: list[str] = field(default_factory=list)
    status: str = "planned"  # planned | built | failed | skipped
    note: str = ""


@dataclass
class Blueprint:
    system: str
    modules: list[Module]
    problems: list[str] = field(default_factory=list)
    spec_findings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def module(self, mid: str) -> Module | None:
        return next((m for m in self.modules if m.id == mid), None)

    def owner_of(self, name: str) -> Module | None:
        for m in self.modules:
            if any(c.name == name for c in m.contract):
                return m
        return None


# --------------------------------------------------------------------------
# validate and order — deterministic
# --------------------------------------------------------------------------

def validate(bp: Blueprint) -> list[str]:
    problems: list[str] = []
    ids = [m.id for m in bp.modules]
    if not ids:
        return ["no modules"]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        problems.append(f"module id used twice: {', '.join(sorted(dup))}")
    owners: dict[str, list[str]] = {}
    files: dict[str, list[str]] = {}
    for m in bp.modules:
        if not m.files:
            problems.append(f"module `{m.id}` owns no files")
        for c in m.contract:
            owners.setdefault(c.name, []).append(m.id)
        for f in m.files:
            files.setdefault(f, []).append(m.id)
        for d in m.depends:
            if d not in ids:
                problems.append(f"module `{m.id}` depends on `{d}`, which "
                                f"does not exist")
            if d == m.id:
                problems.append(f"module `{m.id}` depends on itself")
    for name, who in owners.items():
        if len(who) > 1:
            problems.append(f"contract `{name}` is provided by "
                            f"{', '.join(who)} — it must have one owner")
    for f, who in files.items():
        if len(who) > 1:
            problems.append(f"file `{f}` is owned by {', '.join(who)} — "
                            f"one owner")
    if _cycle(bp):
        problems.append(f"dependency cycle: {' → '.join(_cycle(bp))}")
    return problems


def _cycle(bp: Blueprint) -> list[str]:
    graph = {m.id: list(m.depends) for m in bp.modules}
    state: dict[str, int] = {}
    path: list[str] = []

    def visit(n: str) -> list[str]:
        if state.get(n) == 1:
            return path[path.index(n):] + [n]
        if state.get(n) == 2:
            return []
        state[n] = 1
        path.append(n)
        for d in graph.get(n, []):
            if d in graph:
                found = visit(d)
                if found:
                    return found
        path.pop()
        state[n] = 2
        return []

    for n in graph:
        found = visit(n)
        if found:
            return found
    return []


def order(bp: Blueprint) -> list[str]:
    """Dependencies first; among the free, the declared order."""
    done: list[str] = []
    remaining = [m.id for m in bp.modules]
    deps = {m.id: [d for d in m.depends if d in remaining]
            for m in bp.modules}
    while remaining:
        free = [m for m in remaining if all(d in done for d in deps[m])]
        if not free:
            done.extend(remaining)       # a cycle: validate() reports it
            break
        done.append(free[0])
        remaining.remove(free[0])
    return done


# --------------------------------------------------------------------------
# plan — the conversation before the design
# --------------------------------------------------------------------------

@dataclass
class Plan:
    """The planning conversation: the spec as it stands, and the turns."""
    spec: str = ""
    messages: list[Message] = field(default_factory=list)

    def transcript(self) -> str:
        return "\n\n".join(
            f"{'Operator' if m.role == 'user' else 'Architect'}: {m.content}"
            for m in self.messages)


def converse(session: Any, plan: Plan, text: str, *,
             max_tokens: int = 1200) -> str:
    """One planning turn with the systems architect. It asks about the
    boundaries before it proposes any; it never writes code or the
    decomposition itself — that is `decompose`, from the agreed plan."""
    plan.messages.append(Message(role="user", content=text))
    history = plan.messages[:-1][-14:]
    task = "\n".join([
        "PLAN a system with the operator before it is decomposed into "
        "modules. Talk about: what the system is for and for whom; what "
        "is in and what is out; the boundaries that matter (where "
        "untrusted input enters, what must be swappable, what must never "
        "happen); the data that flows between parts; how it will be "
        "tested. Propose module boundaries only when the goals are clear, "
        "and say WHY each boundary exists. Ask when something is "
        "undecided. Do not write code and do not output the decomposition "
        "— that is structured afterwards from what was agreed.",
        "",
        "[THE SPECIFICATION AS IT STANDS]",
        plan.spec.strip() or "(none written yet — ask for it)",
        "",
        "[THE CONVERSATION SO FAR]",
        "\n\n".join(f"{'Operator' if m.role == 'user' else 'You'}: "
                    f"{m.content}" for m in history) or "(this is the start)",
        "",
        f"Operator: {text}",
    ])
    persona = PERSONAS["systems_architect"]
    prompt = session.prompts.build(persona, task, contract="")
    completion = session.host.llm.complete(
        prompt.messages(), temperature=max(persona.temperature, 0.4),
        max_tokens=max_tokens)
    reply = strip_think(completion.text).strip()
    plan.messages.append(Message(role="assistant", content=reply))
    try:
        session.journal.log("blueprint_plan", turn=len(plan.messages) // 2,
                            chars=len(reply))
    except Exception:                                        # noqa: BLE001
        pass
    return reply


# --------------------------------------------------------------------------
# decompose — the model, then the checks
# --------------------------------------------------------------------------

def from_dict(data: dict) -> Blueprint:
    modules: list[Module] = []
    for row in data.get("modules") or []:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("id", "")).strip().lower().replace(" ", "_")
        if not mid:
            continue
        contract = []
        for c in row.get("contract") or []:
            if isinstance(c, dict) and str(c.get("name", "")).strip():
                contract.append(Contract(
                    name=str(c["name"]).strip(),
                    kind=str(c.get("kind", "function")).strip().lower(),
                    signature=str(c.get("signature", "")).strip(),
                    doc=str(c.get("doc", "")).strip()))
        modules.append(Module(
            id=mid, name=str(row.get("name", mid)).strip(),
            purpose=str(row.get("purpose", "")).strip(),
            files=[str(f).replace("\\", "/").strip()
                   for f in (row.get("files") or []) if str(f).strip()],
            contract=contract,
            depends=[str(d).strip().lower().replace(" ", "_")
                     for d in (row.get("depends") or []) if str(d).strip()]))
    bp = Blueprint(system=str(data.get("system", "")).strip(), modules=modules)
    bp.problems = validate(bp)
    return bp


def decompose(session: Any, spec: str, *, plan: Plan | None = None,
              max_modules: int = 12, lang: str = "python") -> Blueprint:
    """The system spec — and the agreed planning conversation, when there
    was one — as modules with contracts. Spec-linted first; the result
    validated; problems are on the Blueprint, never silent."""
    findings = speclint.lint(spec)
    agreed = plan.transcript() if plan and plan.messages else ""
    task = "\n".join([
        f"DECOMPOSE this system into at most {max_modules} modules, each "
        f"small enough for one build session (two to ten {lang} files). "
        "For each: a short id, a name, its purpose, the files it owns "
        "(source AND its tests), the PUBLIC CONTRACT it provides — every "
        "class, function and constant another module may use, with the "
        "exact signature — and the ids of the modules it depends on. A "
        "contract name must be provided by exactly one module. No "
        "dependency cycles. Leaves (modules depending on nothing) first.",
        "",
        "[THE SYSTEM SPECIFICATION]", spec,
    ] + ([
        "",
        "[WHAT WAS AGREED IN PLANNING — the boundaries and exclusions "
        "decided here override the specification where they differ]",
        agreed,
    ] if agreed else []))
    got = structured.ask(session, PERSONAS["systems_architect"], task,
                         keys=("modules",), example=_SCHEMA,
                         temperature=0.2, max_tokens=3500)
    if not got.ok:
        bp = Blueprint(system="", modules=[],
                       problems=["the model did not return a decomposition "
                                 f"in the shape asked for ({got.problem})"])
    else:
        bp = from_dict(got.data)
    bp.spec_findings = findings
    try:
        session.journal.log("blueprint", modules=len(bp.modules),
                            problems=list(bp.problems),
                            spec_findings=len(findings),
                            planned_turns=len(plan.messages) // 2
                            if plan else 0)
    except Exception:                                        # noqa: BLE001
        pass
    return bp


# --------------------------------------------------------------------------
# stubs and briefs
# --------------------------------------------------------------------------

def _stub_for(c: Contract) -> str:
    sig = c.signature.strip()
    doc = f'    """{c.doc or "contract stub: the real module is not built yet"}"""\n'
    if c.kind == "class":
        head = sig if sig.startswith("class ") else f"class {c.name}"
        if "(" in head and not head.rstrip().endswith(":"):
            # `class Ship(hull: float, shields: list[float])` is the
            # contract's shorthand for a constructor, not a base list
            m = re.match(r"class\s+(\w+)\s*\((.*)\)\s*$", head)
            if m:
                return (f"class {m.group(1)}:\n{doc}"
                        f"    def __init__(self, {m.group(2)}):\n"
                        f"        raise NotImplementedError("
                        f"'contract stub: {c.name}')\n")
        return (f"{head.rstrip(':')}:\n{doc}"
                f"    def __init__(self, *args, **kwargs):\n"
                f"        raise NotImplementedError('contract stub: "
                f"{c.name}')\n")
    if c.kind == "constant":
        value = sig.split("=", 1)[1].strip() if "=" in sig else "None"
        return f"{c.name} = {value}  # contract stub\n"
    head = sig if sig.startswith("def ") else f"def {c.name}(*args, **kwargs)"
    head = head.rstrip(":")
    return (f"{head}:\n{doc}"
            f"    raise NotImplementedError('contract stub: {c.name}')\n")


def stub_files(bp: Blueprint, module: Module) -> dict[str, str]:
    """The contract stubs a module's session needs: one file per
    dependency module's FIRST source file, holding that module's contract.
    (The session's own skeleton stage writes the module's own files.)"""
    out: dict[str, str] = {}
    for dep_id in module.depends:
        dep = bp.module(dep_id)
        if dep is None or not dep.contract:
            continue
        target = next((f for f in dep.files if not _looks_like_test(f)),
                      None)
        if not target:
            continue
        body = [f'"""CONTRACT STUB for module `{dep.id}` ({dep.name}) — '
                f'written by Blueprint so `{module.id}` can be built and '
                f'tested against the contract before `{dep.id}` exists. '
                f'Replaced by the real module."""', "",
                "from __future__ import annotations", ""]
        body += [_stub_for(c) for c in dep.contract]
        out[target] = "\n".join(body) + "\n"
    return out


def _looks_like_test(path: str) -> bool:
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return name.startswith("test_") or "/tests/" in f"/{path}"


def write_stubs(session: Any, bp: Blueprint, module: Module) -> list[str]:
    """Stubs for dependencies that are NOT built yet; the built ones are
    real code and are left alone. Returns the paths written."""
    written = []
    for path, code in stub_files(bp, module).items():
        owner = next((m for m in bp.modules if path in m.files), None)
        if owner is not None and owner.status == "built":
            continue
        if session.host.fs.exists(path) and owner is not None \
                and owner.status == "built":
            continue
        session.host.fs.write(path, code)
        written.append(path)
    return written


def brief(bp: Blueprint, module: Module, ledger_rows: str = "") -> str:
    """The module's request. Its own section of the system, never the
    whole system: the whole system is the truncation bug waiting."""
    parts = [f"BLUEPRINT — module `{module.id}` ({module.name}) of the "
             f"system: {bp.system}", "", module.purpose, "",
             "FILES THIS MODULE OWNS (write exactly these, no others):"]
    parts += [f"  - {f}" for f in module.files]
    if module.contract:
        parts += ["", "THE PUBLIC CONTRACT THIS MODULE MUST PROVIDE — exact "
                      "names and signatures; other modules are built "
                      "against them:"]
        parts += [f"  - {c.signature or c.name}"
                  + (f"  — {c.doc}" if c.doc else "") for c in module.contract]
    deps = [bp.module(d) for d in module.depends]
    deps = [d for d in deps if d is not None]
    if deps:
        parts += ["", "CONTRACTS THIS MODULE MAY CALL (they exist as the "
                      "real module or as stubs; use these names, never "
                      "invent others):"]
        for d in deps:
            where = next((f for f in d.files if not _looks_like_test(f)), "")
            parts.append(f"  from `{d.id}` ({where}):")
            parts += [f"    - {c.signature or c.name}" for c in d.contract]
    if ledger_rows:
        parts += ["", "WHAT EARLIER MODULES RECORDED (keep consistent with "
                      "it):", ledger_rows]
    parts += ["", "Tests for this module import only its own files and the "
                  "contracts above."]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# the ledger and the run
# --------------------------------------------------------------------------

def ledger_text(bp: Blueprint) -> str:
    lines = [f"# Blueprint — {bp.system}", "",
             f"{len(bp.modules)} module(s), built in this order: "
             + " → ".join(order(bp)), ""]
    if bp.spec_findings:
        lines.append("**Specification lint:**")
        lines += [f"- line {f.line}: {f.sentence}" for f in bp.spec_findings]
        lines.append("")
    if bp.problems:
        lines.append("**Problems (the blueprint is not buildable until "
                     "these are fixed):**")
        lines += [f"- {p}" for p in bp.problems]
        lines.append("")
    for m in bp.modules:
        lines.append(f"## {m.id} — {m.name}  [{m.status}]")
        lines.append(m.purpose)
        lines.append("")
        lines.append("files: " + ", ".join(m.files))
        if m.depends:
            lines.append("depends: " + ", ".join(m.depends))
        if m.contract:
            lines.append("contract:")
            lines += [f"- `{c.signature or c.name}`"
                      + (f" — {c.doc}" if c.doc else "") for c in m.contract]
        if m.note:
            lines.append(f"- note: {m.note}")
        lines.append("")
    return "\n".join(lines)


def _ledger_rows(bp: Blueprint) -> str:
    rows = []
    for m in bp.modules:
        if m.status in ("built", "failed"):
            rows.append(f"- {m.id}: {m.status}" + (f" — {m.note}" if m.note
                                                    else ""))
    return "\n".join(rows)


def run_module(session: Any, bp: Blueprint, module: Module, *,
               ledger: str = LEDGER) -> list:
    """One module through one session: stubs for unbuilt dependencies,
    the brief as the request, the outcome into the ledger."""
    written = write_stubs(session, bp, module)
    request = brief(bp, module, _ledger_rows(bp))
    try:
        outcomes = session.run(request)
    except Exception as exc:                                 # noqa: BLE001
        module.status = "failed"
        module.note = f"the session stopped: {type(exc).__name__}: {exc}"
        session.host.fs.write(ledger, ledger_text(bp))
        raise
    ok = bool(outcomes) and all(getattr(o, "ok", False) for o in outcomes)
    module.status = "built" if ok else "failed"
    failed = [o.path for o in outcomes if not getattr(o, "ok", False)]
    module.note = ("" if ok else f"not finished: {', '.join(failed)}") + (
        f" (stubs written: {', '.join(written)})" if written else "")
    session.host.fs.write(ledger, ledger_text(bp))
    try:
        session.journal.log("blueprint_module", module=module.id,
                            status=module.status, stubs=written,
                            files=[o.path for o in outcomes])
    except Exception:                                        # noqa: BLE001
        pass
    return outcomes


def to_json(bp: Blueprint) -> str:
    return json.dumps({
        "system": bp.system, "problems": bp.problems,
        "modules": [{
            "id": m.id, "name": m.name, "purpose": m.purpose,
            "files": m.files, "depends": m.depends, "status": m.status,
            "note": m.note,
            "contract": [c.__dict__ for c in m.contract]} for m in bp.modules]
    }, indent=1)


def from_json(text: str) -> Blueprint:
    data = json.loads(text)
    bp = from_dict(data)
    for row in data.get("modules") or []:
        m = bp.module(str(row.get("id", "")).strip().lower().replace(" ", "_"))
        if m is not None:
            m.status = str(row.get("status", m.status))
            m.note = str(row.get("note", ""))
    return bp
