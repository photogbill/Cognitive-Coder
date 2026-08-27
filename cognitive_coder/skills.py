# SPDX-License-Identifier: Apache-2.0
"""Deployed skills — project guidance that travels with the code (F3).

THE BORROWED IDEA. deepseek-cowork deploys "skill" template files into the
work directory and lets the model pick them up as steering context. That is
the right shape for THIS engine too, for a reason that project never had:
a 7B-24B local model needs scaffolding, and guidance that lives beside the
code is versioned with the code, diffed in review like the code, and present
on the disconnected laptop where a settings screen is not.

A skill is a markdown file in `.ccoder/skills/`. Its whole body goes into
the CACHED PROMPT PREFIX as project conventions (§6.11 layer 2), which is
exactly where F3 already reserved the seat — `SessionConfig.conventions`
had the slot and nothing on disk fed it until now.

THE RULES, AND WHY EACH ONE EXISTS:

  * **Deterministic order.** Files are taken sorted by path. The prefix must
    be byte-identical between calls (M52) or the KV cache silently dies;
    "whatever order the OS returned" is how that happens. Sorted order is
    also the priority order — name files `10-style.md`, `20-tests.md` to
    choose what survives the budget.
  * **Loaded once, at Session construction.** Editing a skill mid-session
    would change the prefix mid-session, which is the cache-killer again.
    A changed skill takes effect on the next session, like an epoch.
  * **An oversized skill is SKIPPED, never truncated.** Truncation cuts a
    rule in half and ships the half that says the opposite of what the
    author meant. A skipped file is reported by name; a mangled rule is
    invisible.
  * **Language scoping filters before budgeting.** A `lang: rust` skill on
    a Python session neither enters the prompt nor spends the budget.
  * **Zero dependencies.** The header is a hand-parsed `key: value` block
    between `---` fences, not YAML. PyYAML for three keys is a bad trade
    in a package whose core promise is "no required runtime dependencies".
  * **Provenance.** The journal records each active skill's name, path and
    content hash at session start (C8) — a generated line influenced by a
    skill is traceable to the exact revision of the skill that did it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import posixpath

from .ports import EventPort, FileSystemPort

#: Where deployed skills live, relative to the project root. Hidden, beside
#: `.cc_journal` and `.cc_state`, so the codemap and the planner never
#: mistake guidance for source.
SKILLS_DIR = ".ccoder/skills"

#: Budget defaults, in characters (≈ chars/4 tokens). Generous enough for
#: real house rules, small enough that a skills folder cannot quietly eat
#: the context window a small model needs for the actual code (G.8 spirit:
#: a starting point to measure from, not a recommendation to hardcode).
MAX_SKILL_CHARS = 4_000
MAX_TOTAL_CHARS = 12_000


@dataclass(frozen=True)
class Skill:
    """One deployed guidance file, parsed."""
    path: str                       # project-relative, /-separated
    name: str                       # header `name:` or the filename stem
    description: str = ""
    langs: tuple[str, ...] = ()     # header `lang:`; empty = every language
    body: str = ""

    def applies_to(self, lang: str) -> bool:
        """Empty scope applies everywhere; otherwise exact id match."""
        if not self.langs:
            return True
        return bool(lang) and lang.lower() in self.langs

    @property
    def sha256(self) -> str:
        """Hash of the raw body — the provenance identity of this revision."""
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillLoad:
    """What discovery found: the active set, and what was left out by name.

    `skipped` is part of the result, not a side channel — a host that wants
    to render "2 skills active, 1 skipped (too large)" gets it from here,
    and the CLI's `skills list` does exactly that.
    """
    skills: tuple[Skill, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()   # (path, reason)

    def block(self) -> str:
        """The prompt text, for the CACHED prefix. Empty when no skills.

        The framing line says what the files ARE without naming machinery
        we do not want echoed back (§4.4's vocabulary rule): no "skill
        system", no "deployed templates" — just rules to follow.
        """
        if not self.skills:
            return ""
        parts = ["These are standing project rules. Follow them as if they "
                 "were part of every task."]
        for s in self.skills:
            parts.append(f"[RULE-SET {s.name}]\n{s.body.strip()}")
        return "\n\n".join(parts)

    def provenance(self) -> list[dict]:
        """One dict per active skill, for the journal (C8)."""
        return [{"path": s.path, "name": s.name, "sha256": s.sha256[:12],
                 "chars": len(s.body)} for s in self.skills]


def parse_skill(path: str, text: str) -> Skill:
    """Parse one file. Header optional; a malformed header is BODY.

    The header is a `---` fence pair at the very top holding `key: value`
    lines. If the closing fence never comes, the whole file is treated as
    body — guidance with a broken header should degrade to guidance, not
    vanish because of a missing three characters.
    """
    stem = posixpath.basename(path)
    stem = stem[:-3] if stem.lower().endswith(".md") else stem
    name, description, langs = stem, "", ()
    body = text
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, min(len(lines), 30)):
            if lines[i].strip() == "---":
                for raw in lines[1:i]:
                    if ":" not in raw:
                        continue
                    key, _, val = raw.partition(":")
                    key, val = key.strip().lower(), val.strip()
                    if key == "name" and val:
                        name = val
                    elif key == "description":
                        description = val
                    elif key == "lang" and val:
                        langs = tuple(sorted(
                            t.strip().lower()
                            for t in val.split(",") if t.strip()))
                body = "\n".join(lines[i + 1:])
                break
    return Skill(path=path, name=name, description=description,
                 langs=langs, body=body.strip("\n"))


def load_skills(fs: FileSystemPort, *, lang: str = "",
                directory: str = SKILLS_DIR,
                max_skill_chars: int = MAX_SKILL_CHARS,
                max_total_chars: int = MAX_TOTAL_CHARS,
                events: EventPort | None = None) -> SkillLoad:
    """Discover, parse, scope and budget the project's deployed skills.

    Never raises: a project without a skills directory — which is most
    projects — is the ordinary case, not an error. Every exclusion is
    reported in `skipped` and, when an EventPort is given, as a `warning`
    event, because silently ignoring a file someone wrote to steer the
    model is the worst available behaviour.
    """
    try:
        paths = sorted(p for p in fs.list(f"{directory}/*.md")
                       if p.replace("\\", "/").startswith(directory + "/"))
    except Exception:                                    # noqa: BLE001
        paths = []
    active: list[Skill] = []
    skipped: list[tuple[str, str]] = []
    total = 0
    for path in paths:
        try:
            text = fs.read(path)
        except Exception as exc:                         # noqa: BLE001
            skipped.append((path, f"unreadable: {exc}"))
            continue
        skill = parse_skill(path, text)
        if not skill.body:
            skipped.append((path, "empty"))
            continue
        if not skill.applies_to(lang):
            scope = ", ".join(skill.langs)
            skipped.append((path, f"scoped to {scope}, session is "
                                  f"{lang or 'unset'}"))
            continue
        if len(skill.body) > max_skill_chars:
            skipped.append((path, f"too large ({len(skill.body)} chars; "
                                  f"limit {max_skill_chars}). Split it — "
                                  "it will not be truncated."))
            continue
        if total + len(skill.body) > max_total_chars:
            skipped.append((path, f"total budget of {max_total_chars} chars "
                                  "reached; earlier-sorting files won"))
            continue
        total += len(skill.body)
        active.append(skill)
    if events is not None:
        for path, reason in skipped:
            if not reason.startswith("scoped to"):
                events.event("warning", f"skill {path} not loaded: {reason}")
    return SkillLoad(skills=tuple(active), skipped=tuple(skipped))


# --------------------------------------------------------------------------
# the starter pack — what `ccoder skills deploy` writes
# --------------------------------------------------------------------------
#
# Written to be EDITED, and short on purpose: every character here is spent
# from a small model's context on every single call. The numbers prefix the
# filenames because sorted order is priority order (see module docstring).
# Deploy never overwrites: once a file exists it belongs to the project.

STARTER_SKILLS: dict[str, str] = {
    "10-house-style.md": """\
---
name: house-style
description: How code in this project is written. Edit me to match yours.
---
Match the style of the surrounding file before any other preference.

Name things for what they are, not for their type: `retries`, not
`retry_count_int`. No abbreviations the project does not already use.

Keep functions short enough to read without scrolling. If a function needs
a section comment, it wants to be two functions.

Comment the WHY, never the what. Delete a comment that restates its line.
""",
    "20-testing.md": """\
---
name: testing
description: What a test must do to count as one. Edit me to match yours.
---
Every behaviour change ships with a test that FAILS before the change and
passes after it.

Test the contract, not the implementation: call the public surface, assert
on observable results, never on private state.

One behaviour per test, named for the behaviour: `test_rejects_expired_token`,
not `test_token_2`.

Include the edge the description implies but does not state: empty input,
the boundary value, the error path.
""",
    "30-forbidden.md": """\
---
name: forbidden
description: Things never to do here, each with its reason. Edit me.
---
Never swallow an exception without recording it. An empty `except` block
hides the failure you will spend a day finding.

Never hardcode a path, port, credential or URL that belongs in
configuration.

Never add a dependency to solve a problem the standard library already
solves. Every dependency is a supply chain and an install step.

Never delete or weaken a failing test to make a build pass. The test is
the specification.
""",
}
