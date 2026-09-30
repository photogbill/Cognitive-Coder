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
    in generated code: raw pointers, `unsafe`, reflection, an empty catch,
    deleting a file the program itself made.

Project mode relaxes the file-path rules, because editing a real project is
the point of project mode — but the destructive and network patterns still
apply, since neither becomes acceptable just because the folder is real. A
delete, move or write aimed at an absolute or `../` path is refused in every
mode.

WHAT IS SCANNED. Comments are never code, so they are blanked before any
rule runs: `# never run rm -rf here` used to be refused. String contents are
blanked too for most rules — a docstring saying "never uses requests" is not
a network call — while the rules that are ABOUT a string (a module name in
`require('net')` or Go's `import "net"`, a path handed to a delete) see the
strings with only the comments gone. In the shells (bash, PowerShell,
batch) strings are commands, so they are always scanned. This makes the
screen quieter for accidents, and it is equally easy for anyone TRYING to
hide something — `getattr(os, "sys" + "tem")` walks straight past it. See
the first paragraph.

Case: rules for the shells, batch and SQL ignore case, because those
languages do; rules for C, Rust, Go, Python and the rest do not, so a C
function named `System` is not "shell execution".

One addition over the ATK original: **version-control commands are blocked**
(M27). The engine has its own snapshot story (§6.5b) and generated code has no
business touching the operator's history, stash or index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from . import langs as _langs
from .types import GuardFinding

BLOCK = "block"
WARN = "warn"

#: Languages whose strings are commands, and which ignore case.
_SHELLS = ("bash", "powershell", "batch")
_NOCASE = ("powershell", "batch", "sql")

# A string literal holding a path outside the project: absolute, home-
# relative, or climbing out with `..`.
_OUTSIDE = r"""['"](?:/|[A-Za-z]:[\\/]|~|\.\.[\\/])"""

# (pattern, severity, reason, languages or () for all, view, nocase)
#
#   view    "bare": comments AND string contents blanked (the default)
#           "text": comments blanked, strings kept — for rules about a
#                   string's contents
#   nocase  None: ignore case iff the language does (_NOCASE); else as given
_Rule = tuple[str, str, str, tuple, str, "bool | None"]


def _r(pattern: str, sev: str, reason: str, langs_for: tuple = (),
       view: str = "bare", nocase: bool | None = None) -> _Rule:
    return (pattern, sev, reason, langs_for, view, nocase)


_PY = ("python",)
_C = ("c", "cpp")
_JS = ("javascript", "typescript")

_RULES: list[_Rule] = [
    # --- destructive filesystem, any language ----------------------------
    _r(r"\brm\s+(?:-\w*[rRf]\w*|--recursive|--force)\b", BLOCK,
       "recursive delete"),
    _r(r"\b(?:del|erase)\s+(?:/\w\s+)*/[sqf]\b|\b(?:rmdir|rd)\s+(?:/\w\s+)*"
       r"/s\b", BLOCK, "recursive delete"),
    _r(r"\bformat\s+[a-zA-Z]:", BLOCK, "drive format"),
    _r(r"\bmkfs\b|\bdiskpart\b", BLOCK, "filesystem/partition tool"),
    _r(r"\bdd\b[^\n]*\bof=/dev/", BLOCK, "raw write to a device"),

    # --- version control (M27), any language ------------------------------
    # The engine never runs git, and generated code that does is blocked.
    # A tool that quietly makes commits in someone's repo is
    # indistinguishable from a mess, and undoing a rewritten history is not
    # something this project's snapshots can help with.
    _r(r"\bgit\s+(?:commit|push|reset|rebase|filter-branch|checkout|clean|"
       r"stash|tag|remote|config)\b", BLOCK,
       "version-control command — this engine never touches your git "
       "history"),
    _r(r"\bgit\s+(?:add|rm|mv)\b", BLOCK,
       "staging changes into your git index"),
    _r(r"\bhg\s+(?:commit|push)\b|\bsvn\s+(?:commit|delete)\b", BLOCK,
       "version-control command"),

    # --- registry / system state, any language ---------------------------
    _r(r"\bwinreg\b|\bregedit\b|\bRegOpenKey\w*\b|\bSet-ItemProperty\s+HK",
       BLOCK, "Windows registry access"),
    _r(r"\bshutdown\b\s+/|\bRestart-Computer\b|\bStop-Computer\b", BLOCK,
       "shutdown / reboot"),
    _r(r"\bTaskKill\b|\bStop-Process\b|\bkill\s+-9\b", BLOCK,
       "killing other processes"),

    # --- Python -----------------------------------------------------------
    _r(r"\bshutil\s*\.\s*rmtree\b", BLOCK, "recursive tree deletion", _PY),
    # Process spawning. A build that shells out escapes every constraint
    # above it; in generated code it is almost never necessary.
    _r(r"\bsubprocess\b|\bos\s*\.\s*(?:system|popen|exec\w*|spawn\w*|"
       r"posix_spawn\w*|startfile)\b|\bpty\s*\.\s*spawn\b|"
       r"^[ \t]*from[ \t]+os[ \t]+import[^\n]*\b(?:system|popen|exec\w*|"
       r"spawn\w*)\b", BLOCK, "process spawning", _PY),
    # Network — by IMPORT, since nothing reaches the network without one,
    # and a variable called `requests` is not a request. C3 is the single
    # most important constraint here: one host is air-gapped.
    _r(r"^[ \t]*(?:from[ \t]+|import[ \t]+(?:[\w.]+[ \t]*,[ \t]*)*)"
       r"(?:socket|ssl|requests|httpx|aiohttp|websockets?|urllib[23]|"
       r"urllib(?!\.parse\b)|http\.client|ftplib|smtplib|telnetlib|"
       r"paramiko)\b", BLOCK,
       "network access (this engine runs offline by default)", _PY),
    _r(r"\basyncio\s*\.\s*open_connection\b", BLOCK, "network access", _PY),
    _r(r"(?<![.\w])eval\s*\(|(?<![.\w])exec\s*\(|\b__import__\b|"
       r"\bimportlib\b", BLOCK, "dynamic evaluation", _PY),
    _r(r"\bctypes\b|\bcffi\b", BLOCK, "native FFI escape", _PY),
    _r(r"\bos\s*\.\s*(?:remove|unlink|rmdir|removedirs|rename|replace)\s*\(|"
       r"\bshutil\s*\.\s*move\b|\.\s*(?:unlink|rmdir)\s*\(", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", _PY),
    _r(r"(?:\bos\s*\.\s*(?:remove|unlink|rmdir|removedirs|rename|replace)|"
       r"\bshutil\s*\.\s*(?:move|copy\w*))\s*\([^\n]*" + _OUTSIDE + "|"
       r"\bPath\s*\(\s*" + _OUTSIDE + r"[^'\"\n]*['\"]\s*\)\s*\.\s*"
       r"(?:unlink|rmdir|rename|replace|write_\w+|touch|mkdir)\b|"
       r"\bopen\s*\(\s*" + _OUTSIDE + r"[^'\"\n]*['\"]\s*,\s*['\"][wax]|"
       r"\bopen\s*\([^\n]*\bexpanduser\s*\(\s*['\"]~[^\n]*,\s*['\"][wax]",
       BLOCK, "deletes, moves or writes a file outside the project", _PY,
       view="text"),
    _r(r"\bexcept\s*:\s*\n\s*pass\b", WARN,
       "bare except: pass swallows the error", _PY),

    # --- C / C++ ----------------------------------------------------------
    _r(r"\bsystem\s*\(", BLOCK, "shell execution", _C),
    _r(r"\bpopen\s*\(|\bexecv?[ple]{0,2}\s*\(|\bfork\s*\(|"
       r"\b(?:CreateProcess\w*|ShellExecute\w*|WinExec)\s*\(", BLOCK,
       "process spawning", _C),
    _r(r"\b(?:socket|WSAStartup|getaddrinfo|connect)\s*\(", BLOCK,
       "network access (this engine runs offline by default)", _C),
    _r(r"\bstd::filesystem::remove_all\b|\bSHFileOperation\w*\b", BLOCK,
       "recursive tree deletion", _C),
    _r(r"\b(?:remove|unlink|rmdir|_unlink|DeleteFile\w*|"
       r"std::filesystem::remove)\s*\(", WARN,
       "deletes files — check it only touches files this program created",
       _C),
    _r(r"\b(?:remove|unlink|rmdir|_unlink|DeleteFile\w*|rename|"
       r"std::filesystem::(?:remove|rename))\s*\(\s*\"(?:/|[A-Za-z]:|~|"
       r"\.\.[\\/])",
       BLOCK, "deletes or moves a file outside the project", _C,
       view="text"),
    _r(r"\bgets\s*\(", BLOCK, "gets() cannot be used safely at any size",
       _C),
    _r(r"\b(?:strcpy|strcat|sprintf)\s*\(", WARN,
       "unbounded string copy — prefer the n-variants", _C),
    _r(r"\bmalloc\s*\(|\bfree\s*\(", WARN,
       "manual memory management — check every path frees exactly once",
       _C),

    # --- Rust -------------------------------------------------------------
    # `std::process` with no `::Command` after it, so the brace form —
    # `use std::process::{Command, Stdio}` — is caught too; it was not.
    _r(r"\bstd::process\b(?!::(?:exit|abort|id)\b)|\bCommand::new\b", BLOCK,
       "process spawning", ("rust",)),
    _r(r"\bstd::net\b|\bTcpStream\b|\bUdpSocket\b|\bTcpListener\b|"
       r"\breqwest\b|\bhyper::", BLOCK, "network access", ("rust",)),
    _r(r"\bremove_dir_all\b", BLOCK, "recursive tree deletion", ("rust",)),
    _r(r"\bremove_(?:file|dir)\s*\(|\bfs::rename\s*\(", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("rust",)),
    _r(r"\b(?:remove_(?:file|dir|dir_all)|rename)\s*\(\s*\"(?:/|[A-Za-z]:|"
       r"~|\.\.[\\/])", BLOCK, "deletes or moves a file outside the project",
       ("rust",), view="text"),
    _r(r"\bunsafe\b", WARN, "unsafe block — memory safety is off here",
       ("rust",)),
    _r(r"\.unwrap\(\)", WARN, "unwrap() panics on error — is that intended?",
       ("rust",)),

    # --- Go ---------------------------------------------------------------
    _r(r"\bos\.RemoveAll\b", BLOCK, "recursive tree deletion", ("go",)),
    _r(r"\bexec\.Command\w*\b|\bos\.StartProcess\b|"
       r"\bsyscall\.(?:Exec|ForkExec|StartProcess)\b", BLOCK,
       "process spawning", ("go",)),
    _r(r"\"os/exec\"", BLOCK, "process spawning", ("go",), view="text"),
    # Imported under any alias — `import n "net"` — so by the import path.
    _r(r"\"net(?:/[\w/]+)?\"", BLOCK, "network access", ("go",),
       view="text"),
    _r(r"\bnet\.(?:Dial|Listen)\w*\s*\(|\bhttp\.(?:Get|Post|Head)\b", BLOCK,
       "network access", ("go",)),
    _r(r"\bos\.(?:Remove|Rename)\s*\(", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("go",)),
    _r(r"\bos\.(?:Remove|RemoveAll|Rename)\s*\(\s*\"(?:/|[A-Za-z]:|~|"
       r"\.\.[\\/])", BLOCK, "deletes or moves a file outside the project",
       ("go",), view="text"),
    _r(r"\bimport\s+\"reflect\"|\breflect\.", WARN, "reflection", ("go",),
       view="text"),

    # --- JavaScript / TypeScript -------------------------------------------
    # Across lines: `fs.rmSync(dir, {\n recursive: true\n})` slipped past a
    # one-line pattern.
    _r(r"\bfs(?:\.promises)?\s*\.\s*(?:rm|rmSync|rmdir|rmdirSync)\s*\("
       r"[^)]*\brecursive\s*:\s*true", BLOCK, "recursive tree deletion",
       _JS),
    _r(r"\bfs(?:\.promises)?\s*\.\s*(?:rm|rmSync|rmdir|rmdirSync|unlink|"
       r"unlinkSync|rename|renameSync)\s*\(", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", _JS),
    _r(r"\bchild_process\b", BLOCK, "process spawning", _JS, view="text"),
    _r(r"\bexecSync\b|\bspawnSync\b|\bexecFileSync\b", BLOCK,
       "process spawning", _JS),
    _r(r"\brequire\s*\(\s*['\"](?:node:)?(?:https?|http2|net|dgram|tls|"
       r"axios|undici|node-fetch|ws)['\"]\s*\)|\bfrom\s+['\"](?:node:)?"
       r"(?:https?|http2|net|dgram|tls|axios|undici|node-fetch|ws)['\"]",
       BLOCK, "network access", _JS, view="text"),
    _r(r"(?<![.\w])fetch\s*\(|\bnew\s+WebSocket\s*\(|\bXMLHttpRequest\b",
       BLOCK, "network access", _JS),
    _r(r"(?<![.\w])eval\s*\(|\bnew\s+Function\s*\(|\bvm\s*\.\s*run\w*\b|"
       r"\bprocess\.binding\b", BLOCK, "dynamic evaluation", _JS),
    _r(r"\b(?:require\s*\(\s*|from\s+)['\"](?:node:)?vm['\"]", BLOCK,
       "dynamic evaluation", _JS, view="text"),

    # --- Java ---------------------------------------------------------------
    _r(r"\bRuntime\.getRuntime\(\)\.exec\b|\bProcessBuilder\b", BLOCK,
       "process spawning", ("java",)),
    _r(r"\bjava\.net\.|\bHttpClient\b|\bURLConnection\b|"
       r"\bnew\s+(?:Server)?Socket\s*\(", BLOCK, "network access",
       ("java",)),
    _r(r"\bFiles\.walkFileTree\b[^;]*delete", BLOCK, "recursive deletion",
       ("java",)),
    _r(r"\bFiles\.(?:delete\w*|move)\s*\(|\.delete\s*\(\s*\)", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("java",)),
    _r(r"\bcatch\s*\(\s*(?:Exception|Throwable)\s+\w+\s*\)\s*\{\s*\}", WARN,
       "empty catch swallows the error", ("java",)),
    _r(r"\bimport\s+java\.lang\.reflect\b|\breflect\.", WARN, "reflection",
       ("java",)),

    # --- C# -----------------------------------------------------------------
    _r(r"\bProcess\.Start\b", BLOCK, "process spawning", ("csharp",)),
    _r(r"\bHttpClient\b|\bWebClient\b|\bWebRequest\b|\bTcpClient\b|"
       r"\bSystem\.Net\.Sockets\b", BLOCK, "network access", ("csharp",)),
    _r(r"\bDirectory\.Delete\s*\([^)]*,\s*true\s*\)", BLOCK,
       "recursive tree deletion", ("csharp",)),
    _r(r"\b(?:File|Directory)\.(?:Delete|Move)\s*\(", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("csharp",)),

    # --- GDScript -----------------------------------------------------------
    _r(r"\bDirAccess\.remove_absolute\b|\bOS\.move_to_trash\b", BLOCK,
       "filesystem deletion", ("gdscript",)),
    _r(r"\bOS\.(?:execute|create_process|create_instance|shell_open)\b",
       BLOCK, "process spawning", ("gdscript",)),
    _r(r"\bHTTPRequest\b|\bHTTPClient\b|\bStreamPeerTCP\b|\bWebSocketPeer\b|"
       r"\bPacketPeerUDP\b", BLOCK, "network access", ("gdscript",)),
    _r(r"\bExpression\.new\b|\.parse\s*\(.*\)\s*;?\s*.*\.execute\b", WARN,
       "dynamic expression evaluation", ("gdscript",)),

    # --- bash ---------------------------------------------------------------
    _r(r"\bfind\b[^\n]*\s-delete\b|\bfind\b[^\n]*-exec\s+rm\b", BLOCK,
       "recursive delete", ("bash",)),
    _r(r"\b(?:curl|wget|nc|ncat|netcat|ssh|scp|sftp|rsync|telnet|ftp)\b",
       BLOCK, "network access", ("bash",)),
    _r(r">>?\s*/(?:etc|bin|sbin|boot|usr|lib|sys|proc|root)/|"
       r">>?\s*/dev/(?!null\b|stdout\b|stderr\b|tty\b)", BLOCK,
       "writes a system file outside the project", ("bash",)),

    # --- PowerShell ---------------------------------------------------------
    _r(r"\b(?:Remove-Item|ri|rm|rmdir|del|erase)\b[^\n]*\s-r(?:ecurse)?\b",
       BLOCK, "recursive delete", ("powershell",)),
    _r(r"\bInvoke-WebRequest\b|\bInvoke-RestMethod\b|\biwr\b|\birm\b|"
       r"\bcurl\b|\bwget\b|\bNet\.WebClient\b|\bNet\.Sockets\b|"
       r"\bStart-BitsTransfer\b", BLOCK, "network access", ("powershell",)),
    _r(r"\bStart-Process\b|\bInvoke-Item\b|\bStart-Job\b", BLOCK,
       "process spawning", ("powershell",)),
    _r(r"\bInvoke-Expression\b|\biex\b", BLOCK, "dynamic evaluation",
       ("powershell",)),

    # --- batch --------------------------------------------------------------
    _r(r"\b(?:curl|wget|bitsadmin|ftp)\b|\bcertutil\b[^\n]*-urlcache",
       BLOCK, "network access", ("batch",)),

    # --- Ruby ---------------------------------------------------------------
    _r(r"(?<![.\w])(?:system|exec|spawn|fork)\b\s*[(\s'\"]|`[^`\n]*`|"
       r"%x[({\[]|\bOpen3\b|\bIO\.popen\b|\bPTY\.spawn\b|"
       r"\bKernel\.(?:system|exec|spawn)\b", BLOCK, "process spawning",
       ("ruby",)),
    _r(r"\bFileUtils\.(?:rm_rf|rm_r|remove_dir|remove_entry\w*|rmtree)\b",
       BLOCK, "recursive tree deletion", ("ruby",)),
    _r(r"\bFile\.(?:delete|unlink|rename)\b|\bFileUtils\.(?:rm|mv)\b", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("ruby",)),
    _r(r"\bNet::(?:HTTP|FTP|SMTP|Telnet)\b|\b(?:TCP|UDP)Socket\b|"
       r"\bSocket\.new\b|\bURI\.open\b", BLOCK, "network access", ("ruby",)),
    _r(r"\brequire\s*\(?\s*['\"](?:open-uri|net/\w+|socket|httparty|"
       r"faraday)['\"]", BLOCK, "network access", ("ruby",), view="text"),
    _r(r"(?<![.\w])eval\s*[(\s]", BLOCK, "dynamic evaluation", ("ruby",)),

    # --- Lua ----------------------------------------------------------------
    _r(r"\bos\.execute\b|\bio\.popen\b", BLOCK, "process spawning",
       ("lua",)),
    _r(r"\brequire\s*\(?\s*['\"](?:socket|ssl|http|luasocket)[\w.]*['\"]",
       BLOCK, "network access", ("lua",), view="text"),
    _r(r"\bos\.(?:remove|rename)\b", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("lua",)),
    _r(r"\bloadstring\s*\(|\bdofile\s*\(", BLOCK, "dynamic evaluation",
       ("lua",)),

    # --- Zig ----------------------------------------------------------------
    _r(r"\bstd\.process\.Child\b|\bstd\.ChildProcess\b|"
       r"\bstd\.process\.exec\w*\b|\bstd\.os\.exec\w*\b", BLOCK,
       "process spawning", ("zig",)),
    _r(r"\bstd\.net\b|\bstd\.http\b", BLOCK, "network access", ("zig",)),
    _r(r"\bdeleteTree(?:Absolute)?\b", BLOCK, "recursive tree deletion",
       ("zig",)),
    _r(r"\b(?:deleteFile|deleteDir|rename)(?:Absolute)?\s*\(", WARN,
       "deletes or moves files — check it only touches files this program "
       "created", ("zig",)),

    # --- SQL (SQLite) ---------------------------------------------------
    # The runner points sqlite3 at a scratch database; these reach past it.
    _r(r"\bATTACH\b\s+(?:DATABASE\b\s+)?", BLOCK,
       "attaches another database file — only the scratch database may be "
       "used", ("sql",)),
    _r(r"^[ \t]*\.(?:shell|system|load)\b|\bload_extension\s*\(", BLOCK,
       "runs a program or loads native code from SQL", ("sql",)),
    _r(r"^[ \t]*\.(?:output|once|import|save|restore)\b|"
       r"\bVACUUM\s+INTO\b", WARN, "reads or writes a file outside the "
       "scratch database", ("sql",)),
]


def _compile(rule: _Rule) -> tuple:
    pattern, sev, reason, langs_for, view, nocase = rule
    flags = re.M
    return (re.compile(pattern, flags), re.compile(pattern, flags | re.I),
            sev, reason, langs_for, view, nocase)


_COMPILED = [_compile(rule) for rule in _RULES]

# Absolute paths that leave the workspace. Relaxed in project mode, where
# editing a real tree is the entire point. Scanned in the "text" view: it is
# ABOUT string literals.
_ABS_PATH = re.compile(
    r"""['"]\s*(?:[a-zA-Z]:[\\/]|[\\/]{1,2}(?:etc|windows|system32|users|"""
    r"""bin|boot|dev|proc|sys|root|home)\b)""", re.I)


def languages_with_rules() -> set[str]:
    """Every language with at least one rule of its own. A language missing
    here is screened only by the universal shell-command rules."""
    out: set[str] = set()
    for _p, _s, _r2, langs_for, _v, _n in _RULES:
        out |= set(langs_for)
    return out


# ---------------------------------------------------------------------------
# comments and strings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Syntax:
    line: str = "#"                  # line-comment marker
    block: tuple = ()                # (open, close)
    quotes: str = "\"'"              # string delimiters
    char_quote: bool = False         # `'` only as a short char literal
    triple: bool = False             # Python/GDScript """…"""
    multiline: str = ""              # quotes that may span lines
    long_bracket: bool = False       # Lua [[…]]


_STRINGS: dict[str, dict] = {
    "python": {"triple": True}, "gdscript": {"triple": True},
    "c": {"char_quote": True}, "cpp": {"char_quote": True},
    "java": {"char_quote": True}, "csharp": {"char_quote": True},
    "zig": {"char_quote": True},
    "rust": {"char_quote": True, "multiline": "\""},
    "go": {"char_quote": True, "quotes": "\"'`", "multiline": "`"},
    "javascript": {"quotes": "\"'`", "multiline": "`"},
    "typescript": {"quotes": "\"'`", "multiline": "`"},
    "ruby": {"multiline": "\"'"}, "bash": {"multiline": "\"'"},
    "powershell": {"multiline": "\"'"}, "batch": {"quotes": "\""},
    "lua": {"long_bracket": True}, "sql": {"quotes": "'", "multiline": "'"},
}

_CHAR_LITERAL = re.compile(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'")
_LONG_OPEN = re.compile(r"\[(=*)\[")


def _syntax(lang_id: str) -> _Syntax:
    lang = _langs.get(lang_id)
    kw = dict(_STRINGS.get(lang_id, {}))
    return _Syntax(line=(lang.comment if lang else "#"),
                   block=tuple(lang.block_comment) if lang else (), **kw)


def _comment_at(code: str, i: int, syn: _Syntax, lang_id: str) -> bool:
    if lang_id == "batch":
        head = code[code.rfind("\n", 0, i) + 1:i]
        if head.strip(" \t@"):
            return False
        rest = code[i:i + 4]
        return rest[:2] == "::" or (
            rest[:3].upper() == "REM"
            and (len(rest) < 4 or rest[3] in " \t\r\n"))
    if not code.startswith(syn.line, i):
        return False
    if syn.line == "#" and lang_id in _SHELLS:
        # `#` opens a comment only at the start of a word: not `$#`,
        # `${#arr}` or `a#b`.
        return i == 0 or code[i - 1] in " \t\n;|&("
    return True


def _string_end(code: str, i: int, syn: _Syntax) -> int:
    """End (exclusive) of a string literal starting at ``i``, else 0."""
    ch = code[i]
    if syn.triple and code.startswith(('"""', "'''"), i):
        close = code.find(code[i:i + 3], i + 3)
        return len(code) if close < 0 else close + 3
    if syn.long_bracket and ch == "[":
        m = _LONG_OPEN.match(code, i)
        if m:
            close = code.find("]" + m.group(1) + "]", m.end())
            return len(code) if close < 0 else close + len(m.group(1)) + 2
        return 0
    if ch not in syn.quotes:
        return 0
    if ch == "'" and syn.char_quote:
        m = _CHAR_LITERAL.match(code, i)
        return m.end() if m else 0            # a Rust lifetime, not a string
    j = i + 1
    raw = ch == "`"                           # Go raw / JS template
    while j < len(code):
        c = code[j]
        if c == "\\" and not raw:
            j += 2
            continue
        if c == ch:
            return j + 1
        if c == "\n" and ch not in syn.multiline:
            return j                          # unterminated: stop at EOL
        j += 1
    return len(code)


def _views(code: str, lang_id: str) -> tuple[str, str]:
    """(text, bare): comments blanked; comments AND string contents blanked.

    Blanking keeps every newline and every offset, so a match in either
    view is at the same place — and on the same line — as in the source.
    In the shells strings are commands, so `bare` keeps them.
    """
    syn = _syntax(lang_id)
    text = list(code)
    bare = list(code)
    shell = lang_id in _SHELLS

    def blank(buf: list, a: int, b: int) -> None:
        for k in range(a, b):
            if buf[k] != "\n":
                buf[k] = " "

    i, n = 0, len(code)
    while i < n:
        if syn.block and code.startswith(syn.block[0], i):
            close = code.find(syn.block[1], i + len(syn.block[0]))
            end = n if close < 0 else close + len(syn.block[1])
            blank(text, i, end)
            blank(bare, i, end)
            i = end
            continue
        if _comment_at(code, i, syn, lang_id):
            end = code.find("\n", i)
            end = n if end < 0 else end
            blank(text, i, end)
            blank(bare, i, end)
            i = end
            continue
        end = _string_end(code, i, syn)
        if end:
            if not shell:
                blank(bare, i, end)
            i = end
            continue
        i += 1
    return "".join(text), "".join(bare)


def views(code: str, lang_id: str) -> tuple[str, str]:
    """(text, bare) views of ``code`` — see `_views`. Public so the runner
    can ask "is this a main-loop program?" of CODE, not of its comments."""
    return _views(code or "", (lang_id or "").lower())


# ---------------------------------------------------------------------------
# the screen
# ---------------------------------------------------------------------------

def scan(code: str, lang_id: str = "",
         project_mode: bool = False) -> list[GuardFinding]:
    """Every finding, blocks first. An empty list means nothing was flagged.

    "Nothing was flagged" means the regexes found nothing — read the module
    docstring before concluding anything stronger.
    """
    if not code:
        return []
    lang = (lang_id or "").lower()
    text, bare = _views(code, lang)
    out: list[GuardFinding] = []
    for cs, ci, sev, reason, langs_for, view, nocase in _COMPILED:
        if langs_for and lang not in langs_for:
            continue
        pattern = ci if (lang in _NOCASE if nocase is None else nocase) \
            else cs
        m = pattern.search(text if view == "text" else bare)
        if m:
            out.append(GuardFinding(
                severity=sev, reason=reason,
                match=" ".join(code[m.start():m.end()].split())[:60],
                line=code[:m.start()].count("\n") + 1))
    if not project_mode:
        m = _ABS_PATH.search(text)
        if m:
            out.append(GuardFinding(
                severity=BLOCK,
                reason="absolute path leaves the workspace (use relative "
                       "paths, or switch to project mode)",
                match=code[m.start():m.end()].strip()[:60],
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
