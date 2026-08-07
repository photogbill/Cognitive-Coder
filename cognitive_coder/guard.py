# SPDX-License-Identifier: Apache-2.0
"""Static screen for generated code, in every language the engine runs.

WHAT THIS IS, STATED PLAINLY AND FIRST, BECAUSE M9 REQUIRES IT AND BECAUSE IT
IS TRUE: **this is a screen against ACCIDENTS. It is not a security boundary.**

A determined adversary defeats a regex in about a minute. A language model
that has misunderstood the task and reached for `system("rm -rf /")` does not.
The threat model is the second one, and only the second one. Nobody who reads
this file should leave thinking otherwise, and no documentation this project
ships may describe `guard` + the host's sandbox + the project-root jail as a
defence against a hostile model (C10, M9). Say what it is: a screen against
mistakes, operated by a human who stays in charge.

The layers that DO exist, in order, none sufficient alone:

    guard.py's static screen  (this file — accidents, not adversaries)
    the host's ExecPort sandboxing policy (the host decides what that means)
    the scrubbed environment (§6.4)
    the project-root jail (§6.5, M24)
    the approval gate (ApprovalPort, M18)

TWO SEVERITIES, and the distinction matters:

  * **block** — refuse to run. Destructive or exfiltrating.
  * **warn**  — run, but say so. Legitimate in real code, worth a second look
    in generated code: raw pointers, `unsafe`, reflection, an empty catch.

Project mode relaxes the file-path rules, because editing a real project is
the point of project mode — but the destructive and network patterns still
apply, since neither becomes acceptable just because the folder is real.

One addition over the ATK original: **version-control commands are blocked**
(M27). The engine has its own snapshot story (§6.5b) and generated code has no
business touching the operator's history, stash or index.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from .types import GuardFinding

BLOCK = "block"
WARN = "warn"

# (pattern, severity, reason, languages or () for all)
_RULES: list[tuple[str, str, str, tuple]] = [
    # --- destructive filesystem ------------------------------------------
    (r"\brm\s+-[rf]{1,2}\b", BLOCK, "recursive delete", ()),
    (r"\bdel\s+/[sqf]\b|\brmdir\s+/s\b", BLOCK, "recursive delete", ()),
    (r"\bformat\s+[a-z]:", BLOCK, "drive format", ()),
    (r"\bmkfs\b|\bdiskpart\b", BLOCK, "filesystem/partition tool", ()),
    (r"\bshutil\s*\.\s*rmtree\b", BLOCK, "recursive tree deletion",
     ("python",)),
    (r"\bstd::fs::remove_dir_all\b", BLOCK, "recursive tree deletion",
     ("rust",)),
    (r"\bos\.RemoveAll\b", BLOCK, "recursive tree deletion", ("go",)),
    (r"\bfs\.rmSync\b[^)]*recursive", BLOCK, "recursive tree deletion",
     ("javascript", "typescript")),
    (r"\bFiles\.walkFileTree\b.*delete", BLOCK, "recursive deletion",
     ("java",)),
    (r"\bDirAccess\.remove_absolute\b|\bOS\.move_to_trash\b", BLOCK,
     "filesystem deletion", ("gdscript",)),

    # --- version control (M27) -------------------------------------------
    # The engine never runs git, and generated code that does is blocked.
    # A tool that quietly makes commits in someone's repo is
    # indistinguishable from a mess, and undoing a rewritten history is not
    # something this project's snapshots can help with.
    (r"\bgit\s+(commit|push|reset|rebase|filter-branch|checkout|clean|"
     r"stash|tag|remote|config)\b", BLOCK,
     "version-control command — this engine never touches your git history",
     ()),
    (r"\bgit\s+(add|rm|mv)\b", BLOCK,
     "staging changes into your git index", ()),
    (r"\bhg\s+(commit|push)\b|\bsvn\s+(commit|delete)\b", BLOCK,
     "version-control command", ()),

    # --- shelling out -----------------------------------------------------
    # A build that shells out escapes every constraint above it. In generated
    # code it is almost never necessary and almost always a mistake.
    (r"\bsystem\s*\(", BLOCK, "shell execution", ("c", "cpp")),
    (r"\bpopen\s*\(|\bexecv?[ple]?\s*\(|\bfork\s*\(", BLOCK,
     "process spawning", ("c", "cpp")),
    (r"\bstd::process::Command\b", BLOCK, "process spawning", ("rust",)),
    (r"\bRuntime\.getRuntime\(\)\.exec\b|\bProcessBuilder\b", BLOCK,
     "process spawning", ("java",)),
    (r"\bos/exec\b|\bexec\.Command\b", BLOCK, "process spawning", ("go",)),
    (r"\bchild_process\b|\bexecSync\b|\bspawnSync\b", BLOCK,
     "process spawning", ("javascript", "typescript")),
    (r"\bsubprocess\b|\bos\s*\.\s*system\b|\bos\s*\.\s*popen\b", BLOCK,
     "process spawning", ("python",)),
    (r"\bProcess\.Start\b", BLOCK, "process spawning", ("csharp",)),
    (r"\bOS\.execute\b|\bOS\.create_process\b", BLOCK, "process spawning",
     ("gdscript",)),

    # --- network ----------------------------------------------------------
    # C3 is the single most important constraint in this project: one host is
    # an air-gapped tool where an outbound connection is a safety problem,
    # not an inconvenience. Generated code does not get to open one.
    (r"\b(socket|WSAStartup|getaddrinfo|connect)\s*\(", BLOCK,
     "network access (this engine runs offline by default)", ("c", "cpp")),
    (r"\bstd::net::\b|\breqwest\b|\bhyper::\b", BLOCK, "network access",
     ("rust",)),
    (r"\bjava\.net\.|\bHttpClient\b|\bURLConnection\b", BLOCK,
     "network access", ("java",)),
    (r"\bnet/http\b|\bhttp\.Get\b", BLOCK, "network access", ("go",)),
    (r"\bfetch\s*\(|\brequire\(['\"](https?|net|dgram)['\"]\)|"
     r"\bfrom\s+['\"]node:(https?|net)['\"]", BLOCK, "network access",
     ("javascript", "typescript")),
    (r"\b(socket|urllib|requests|httpx|http\.client|ftplib|smtplib)\b", BLOCK,
     "network access", ("python",)),
    (r"\b(curl|wget|Invoke-WebRequest|Invoke-RestMethod)\b", BLOCK,
     "network access", ("bash", "powershell", "batch")),
    (r"\bHttpClient\b|\bWebClient\b", BLOCK, "network access", ("csharp",)),
    (r"\bHTTPRequest\b|\bHTTPClient\b|\bStreamPeerTCP\b|\bWebSocketPeer\b",
     BLOCK, "network access", ("gdscript",)),

    # --- registry / system state ------------------------------------------
    (r"\bwinreg\b|\bregedit\b|\bRegOpenKey\b|\bSet-ItemProperty\s+HK",
     BLOCK, "Windows registry access", ()),
    (r"\bshutdown\b\s+/|\bRestart-Computer\b|\bStop-Computer\b", BLOCK,
     "shutdown / reboot", ()),
    (r"\bTaskKill\b|\bStop-Process\b|\bkill\s+-9\b", BLOCK,
     "killing other processes", ()),

    # --- dynamic evaluation ----------------------------------------------
    (r"\beval\s*\(|\bnew\s+Function\s*\(", BLOCK, "dynamic evaluation",
     ("javascript", "typescript")),
    (r"\beval\s*\(|\bexec\s*\(|\b__import__\b", BLOCK,
     "dynamic evaluation", ("python",)),
    (r"\bInvoke-Expression\b|\biex\b", BLOCK, "dynamic evaluation",
     ("powershell",)),
    (r"\bExpression\.new\b|\.parse\s*\(.*\)\s*;?\s*.*\.execute\b", WARN,
     "dynamic expression evaluation", ("gdscript",)),
    (r"\bctypes\b|\bcffi\b", BLOCK, "native FFI escape", ("python",)),

    # --- warn only: legitimate, but worth a look in generated code -------
    (r"\bunsafe\b", WARN, "unsafe block — memory safety is off here",
     ("rust",)),
    (r"\bgets\s*\(", BLOCK, "gets() cannot be used safely at any size",
     ("c", "cpp")),
    (r"\b(strcpy|strcat|sprintf)\s*\(", WARN,
     "unbounded string copy — prefer the n-variants", ("c", "cpp")),
    (r"\bmalloc\s*\(|\bfree\s*\(", WARN,
     "manual memory management — check every path frees exactly once",
     ("c", "cpp")),
    (r"\bcatch\s*\(\s*(Exception|Throwable)\s+\w+\s*\)\s*\{\s*\}", WARN,
     "empty catch swallows the error", ("java",)),
    (r"\bexcept\s*:\s*\n\s*pass\b", WARN,
     "bare except: pass swallows the error", ("python",)),
    (r"\.unwrap\(\)", WARN, "unwrap() panics on error — is that intended?",
     ("rust",)),
    (r"\bimport\s+reflect\b|\breflect\.", WARN, "reflection", ("go", "java")),
]

_COMPILED = [(re.compile(p, re.I | re.M), sev, reason, langs)
             for p, sev, reason, langs in _RULES]

# Absolute paths that leave the workspace. Relaxed in project mode, where
# editing a real tree is the entire point.
_ABS_PATH = re.compile(
    r"""['"]\s*(?:[a-zA-Z]:[\\/]|[\\/]{1,2}(?:etc|windows|system32|users|"""
    r"""bin|boot|dev|proc|sys|root|home)\b)""", re.I)


def scan(code: str, lang_id: str = "",
         project_mode: bool = False) -> list[GuardFinding]:
    """Every finding, blocks first. An empty list means nothing was flagged.

    "Nothing was flagged" means the regexes found nothing — read the module
    docstring before concluding anything stronger.
    """
    if not code:
        return []
    lang = (lang_id or "").lower()
    out: list[GuardFinding] = []
    for pattern, sev, reason, langs_for in _COMPILED:
        if langs_for and lang not in langs_for:
            continue
        m = pattern.search(code)
        if m:
            out.append(GuardFinding(
                severity=sev, reason=reason,
                match=m.group(0).strip()[:60],
                line=code[:m.start()].count("\n") + 1))
    if not project_mode:
        m = _ABS_PATH.search(code)
        if m:
            out.append(GuardFinding(
                severity=BLOCK,
                reason="absolute path leaves the workspace (use relative "
                       "paths, or switch to project mode)",
                match=m.group(0).strip()[:60],
                line=code[:m.start()].count("\n") + 1))
    out.sort(key=lambda f: 0 if f.severity == BLOCK else 1)
    return out


def blocked(findings: Sequence[GuardFinding]) -> str:
    """The reason to refuse, or "" to proceed."""
    hard = [f for f in findings if f.severity == BLOCK]
    if not hard:
        return ""
    return "; ".join(f.reason for f in hard[:3])


def advisory(findings: Sequence[GuardFinding]) -> str:
    warns = [f for f in findings if f.severity == WARN]
    return "; ".join(f.one_line() for f in warns[:4])


def explain_to_model(findings: Sequence[GuardFinding]) -> str:
    """What to tell the model so its NEXT attempt is different.

    Phrased as an INSTRUCTION rather than a complaint. A small model handed
    "that was blocked" tends to reword the same code; one handed "use X
    instead of Y" changes its approach. This is the same principle as
    `diagnostics.feedback` — specificity is what makes a small model useful.
    """
    hard = [f for f in findings if f.severity == BLOCK]
    if not hard:
        return ""
    lines = ["Your code was refused before it ran. Rewrite it so that none of "
             "the following is needed:"]
    for f in hard[:4]:
        lines.append(f"  - {f.reason} (you wrote `{f.match}`)")
    lines.append("Use only the language's standard library, keep all file "
                 "paths relative to the project, do not start other "
                 "processes, do not open network connections, and do not run "
                 "version-control commands.")
    return "\n".join(lines)
