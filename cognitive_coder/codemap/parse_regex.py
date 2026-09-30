# SPDX-License-Identifier: Apache-2.0
"""A ctags-style signature extractor for everything Python's `ast` can't do.

Zero dependencies, and **honestly approximate**. It looks for declarations at
the start of a line; it does not parse. Every symbol it produces carries
`approximate=True`, and that flag survives all the way to the prompt, where
the outline says *"pattern-matched, not parsed — it can miss unusual
declarations."* An outline that is 95% right is far more useful than no
outline; an outline that CLAIMS to be 100% right and isn't is worse than
either.

`parse_treesitter.py` is strictly better for C/C++/Rust/JS and strictly
optional (C7). When `tree_sitter` is importable this module is the fallback;
when it isn't, this is the whole story, and the engine says which mode it is
in rather than leaving the operator to wonder.

One anchoring rule, learned by getting it wrong: **patterns anchor with
`^[ \\t]*`, never `^\\s*`.** In multiline mode `\\s` matches newlines, so
`^\\s*func` happily starts matching on the blank line above and every line
number comes out one too low — a bug that is invisible until someone tries to
open the file at the reported line.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from ..types import Symbol

_KEYWORDS = {"if", "for", "while", "switch", "return", "catch", "else", "do",
             "try", "match", "elif", "with", "case", "defer", "go", "select",
             # Block-introducing words that look like `name (…) {` to a
             # pattern: `if constexpr (…) {` was indexed as a C++ function
             # called `constexpr`, `synchronized (lock) {` as a Java method.
             "constexpr", "synchronized", "sizeof", "alignof", "decltype",
             "static_assert", "foreach", "lock", "using", "fixed",
             "checked", "unchecked", "unsafe", "new", "delete", "throw",
             "await", "yield", "typeof", "instanceof", "func", "fn",
             "function", "loop", "until", "unless", "when", "assert"}

# Pieces shared by the brace languages. A PARAMETER LIST allows one level of
# nested parentheses (`map<U>(f: (t: T) => U)`, `@Named("x") int n`) and may
# span lines, but never crosses `;`, `{` or `}` — it cannot run into the
# next declaration. The PREFIX before a name never crosses a newline: the
# old `[\w*\s]+?` did, so `static int\nhelper(` was reported at line 1 with
# the signature `static int`.
_PARAMS = r"\((?:[^;{}()]|\([^;{}()]*\))*\)"
_PREFIX = r"(?:[A-Za-z_][\w:<>,*&\[\] \t]*?[ \t*&])?"

# Group names: `name*` → function, `cls*` → a type, `method` with `recv`
# → a Go method named `Recv.method`, `impl` → a Rust impl block, `signal`
# → a GDScript signal.
_DECL: dict[str, str] = {
    "c": (rf"^[ \t]*{_PREFIX}(?P<name>[A-Za-z_]\w*)[ \t]*{_PARAMS}"
          r"[ \t\n]*\{|"
          r"^[ \t]*(?:typedef[ \t]+)?(?:struct|union|enum)[ \t]+"
          r"(?P<cls>[A-Za-z_]\w*)[ \t\n]*\{|"
          r"^[ \t]*typedef[ \t]+(?:struct|union|enum)\b[^;{]*\{[^}]*\}"
          r"[ \t]*(?P<cls2>[A-Za-z_]\w*)[ \t]*;|"
          r"^[ \t]*typedef[ \t]+[\w \t*]+?[ \t*](?P<cls3>[A-Za-z_]\w*)"
          r"[ \t]*;"),
    "cpp": (rf"^[ \t]*{_PREFIX}(?P<name>~?[A-Za-z_][\w:~]*)[ \t]*{_PARAMS}"
            r"(?:[ \t\n]*(?:const|override|final|noexcept|mutable)\b)*"
            r"(?:[ \t\n]*->[ \t]*[\w:<>*& ]+)?"
            r"(?:[ \t\n]*:[^;{}]*)?[ \t\n]*\{|"
            r"^[ \t]*(?:template[ \t]*<[^>]*>[ \t]*)?(?:class|struct|union|"
            r"enum(?:[ \t]+class)?)[ \t]+(?P<cls>[A-Za-z_]\w*)[^;{]*\{|"
            r"^[ \t]*(?:typedef|using)[ \t]+[^;{}=]*?\b(?P<cls2>[A-Za-z_]\w*)"
            r"[ \t]*(?:=[^;{}]*)?;"),
    "rust": (r"^[ \t]*(?:pub(?:\([\w: ]+\))?[ \t]+)?"
             r"(?:(?:const|async|unsafe|default|"
             r"extern(?:[ \t]+\"[^\"\n]*\")?)[ \t]+)*"
             r"fn[ \t]+(?P<name>[A-Za-z_]\w*)|"
             r"^[ \t]*(?:pub(?:\([\w: ]+\))?[ \t]+)?(?:unsafe[ \t]+)?"
             r"(?:struct|enum|trait|union|type)[ \t]+(?P<cls>[A-Za-z_]\w*)|"
             r"^[ \t]*(?:unsafe[ \t]+)?impl(?:[ \t]*<[^>{\n]*>)?[ \t]+"
             r"(?:[^{;\n]*?[ \t]+for[ \t]+)?(?:&?[\w:]*::)?"
             r"(?P<impl>[A-Za-z_]\w*)"),
    "java": (r"^[ \t]*(?:@[\w.]+(?:\([^)\n]*\))?[ \t]+)*"
             r"(?:(?:public|private|protected|static|final|abstract|"
             r"synchronized|native|default|strictfp)[ \t]+)*"
             r"(?:<[^>\n]*>[ \t]+)?(?:[\w.<>\[\],?]+[ \t]+)?"
             rf"(?P<name>[A-Za-z_]\w*)[ \t]*{_PARAMS}"
             r"(?:[ \t\n]*throws[ \t\n]+[\w., \t\n]+)?[ \t\n]*\{|"
             r"^[ \t]*(?:(?:public|private|protected|static|final|abstract|"
             r"sealed)[ \t]+)*(?:class|interface|enum|record|@interface)"
             r"[ \t]+(?P<cls>[A-Za-z_]\w*)"),
    "go": (r"^func[ \t]*\([ \t]*\w*[ \t]*\*?[ \t]*(?P<recv>[A-Za-z_]\w*)"
           r"[^)\n]*\)[ \t]*(?P<method>[A-Za-z_]\w*)|"
           r"^func[ \t]+(?P<name>[A-Za-z_]\w*)|"
           r"^type[ \t]+(?P<cls>[A-Za-z_]\w*)"),
    "javascript": (
        r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?"
        r"function[ \t]*\*?[ \t]*(?P<name>[A-Za-z_$][\w$]*)|"
        r"^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:abstract[ \t]+)?"
        r"class[ \t]+(?P<cls>[A-Za-z_$][\w$]*)|"
        r"^[ \t]*(?:export[ \t]+)?(?:const|let|var)[ \t]+"
        r"(?P<name2>[A-Za-z_$][\w$]*)[ \t]*(?::[^=\n]+)?=[ \t]*"
        r"(?:async[ \t]*)?(?:function\b|\([^)\n]*\)[ \t]*(?::[^=\n]+)?=>|"
        r"[A-Za-z_$][\w$]*[ \t]*=>)|"
        r"^[ \t]*(?:module\.)?exports\.(?P<name3>[A-Za-z_$][\w$]*)[ \t]*="
        r"[ \t]*(?:async[ \t]*)?(?:function\b|\([^)\n]*\)[ \t]*=>|"
        r"[A-Za-z_$][\w$]*[ \t]*=>)|"
        r"^[ \t]+(?:(?:public|private|protected|static|async|get|set|"
        r"readonly|override|abstract)[ \t]+)*\*?[ \t]*"
        r"(?P<name4>[A-Za-z_$][\w$]*)[ \t]*(?:<[^>\n]*>)?[ \t]*"
        rf"{_PARAMS}[ \t]*(?::[^;{{}}=\n]+)?[ \t]*\{{|"
        r"^[ \t]*(?:export[ \t]+)?(?:declare[ \t]+)?"
        r"(?:interface|type|enum|const[ \t]+enum|namespace)[ \t]+"
        r"(?P<cls2>[A-Za-z_$][\w$]*)"),
    "csharp": (r"^[ \t]*(?:\[[^\]\n]*\][ \t]*)*"
               r"(?:(?:public|private|protected|internal|static|async|"
               r"virtual|override|sealed|abstract|extern|partial|new|"
               r"unsafe)[ \t]+)*(?:[\w.<>\[\],?]+[ \t]+)?"
               rf"(?P<name>[A-Za-z_]\w*)[ \t]*(?:<[^>\n]*>)?[ \t]*{_PARAMS}"
               r"[^;{}]*\{|"
               r"^[ \t]*(?:(?:public|private|protected|internal|static|"
               r"sealed|abstract|partial)[ \t]+)*"
               r"(?:class|struct|interface|record|enum)[ \t]+"
               r"(?P<cls>[A-Za-z_]\w*)"),
    "zig": (r"^[ \t]*(?:pub[ \t]+)?(?:export[ \t]+)?(?:inline[ \t]+)?"
            r"fn[ \t]+(?P<name>\w+)|"
            r"^[ \t]*(?:pub[ \t]+)?const[ \t]+(?P<cls>\w+)[ \t]*=[ \t]*"
            r"(?:extern[ \t]+|packed[ \t]+)?(?:struct|enum|union)"),
    "ruby": r"^[ \t]*def[ \t]+(?P<name>[\w.?!]+)|"
            r"^[ \t]*(?:class|module)[ \t]+(?P<cls>\w+)",
    "lua": r"^[ \t]*(?:local[ \t]+)?function[ \t]+(?P<name>[\w.:]+)",
    "bash": r"^[ \t]*(?:function[ \t]+)?(?P<name>\w+)[ \t]*\(\)[ \t]*\{",
    "powershell": r"^[ \t]*function[ \t]+(?P<name>[\w-]+)",
    "sql": r"^[ \t]*CREATE[ \t]+(?:TABLE|VIEW|INDEX|TRIGGER)[ \t]+"
           r"(?:IF[ \t]+NOT[ \t]+EXISTS[ \t]+)?(?P<cls>\w+)",
    # GDScript (§6.1a). Indentation-scoped like Python, so the end of a body
    # is DERIVABLE rather than guessed — which is why it gets its own path
    # below rather than sharing the brace-counting one.
    "gdscript": r"^[ \t]*(?:@\w+(?:\([^)]*\))?[ \t]+)*(?:static[ \t]+)?"
                r"func[ \t]+(?P<name>\w+)|"
                r"^[ \t]*class_name[ \t]+(?P<cls>\w+)|"
                r"^[ \t]*class[ \t]+(?P<cls2>\w+)|"
                r"^[ \t]*signal[ \t]+(?P<signal>\w+)",
}
_DECL["typescript"] = _DECL["javascript"]
_DECL["python"] = (r"^[ \t]*(?:async[ \t]+)?def[ \t]+(?P<name>\w+)|"
                   r"^[ \t]*class[ \t]+(?P<cls>\w+)")

_INDENT_SCOPED = {"python", "gdscript"}

# Call sites, for the (approximate) call graph. Deliberately crude: a bare
# `name(` at a plausible position. It over-reports slightly, which for blast
# radius is the safe direction — a false caller costs one extra file read; a
# missed caller costs a broken build nobody predicted.
#
# DOTTED, so that `document.getElementById(` records the object it is called
# on. It used to capture only the last identifier, and `getElementById`,
# `Println`, `push` were then reported by the D4 check as names the project
# does not define — on nearly every JS, Go and Rust file.
#
# Not after `@`: a Java/C#/TS annotation `@Named("x")` is not a call.
_CALL = re.compile(
    r"(?<![\w$.@])(?P<name>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)"
    r"\s*\(")

# -- comments, strings and dead preprocessor blocks ---------------------------
#
# Matched against a BLANKED copy of the text: comments and string contents
# become spaces (newlines kept, so every offset and line number is the
# same). Without it, a commented-out function or one inside `#if 0` was
# indexed as live, `printf("call foo(")` recorded a call to `foo`, and a
# `"{"` in a string stretched a function's body over the next one.
_LINE_COMMENT: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(("c", "cpp", "java", "csharp", "javascript",
                     "typescript", "rust", "go", "zig"), ("//",)),
    **dict.fromkeys(("python", "ruby", "bash", "gdscript", "powershell"),
                    ("#",)),
    "lua": ("--",), "sql": ("--",),
}
_BLOCK_COMMENT = {"c", "cpp", "java", "csharp", "javascript", "typescript",
                  "rust", "go", "sql"}
#: Languages where `'…'` is a whole string rather than a character literal.
#: In Rust `'a` is a lifetime, and treating it as a string opener would
#: blank the rest of the file.
_SINGLE_QUOTE_STRINGS = {"javascript", "typescript", "python", "ruby",
                         "bash", "lua", "sql", "powershell", "gdscript"}
_CHAR_LITERAL = re.compile(r"'(?:\\.[^'\n]{0,8}|[^\\'\n])'")
_IF0 = re.compile(r"^[ \t]*#[ \t]*if[ \t]+0\b")
_IF_ANY = re.compile(r"^[ \t]*#[ \t]*if")
_ENDIF = re.compile(r"^[ \t]*#[ \t]*endif\b")
_ELSE = re.compile(r"^[ \t]*#[ \t]*(?:else|elif)\b")


def blank_noise(text: str, lang: str) -> str:
    """`text` with comments and string contents replaced by spaces."""
    out = list(text)
    n = len(text)
    line_comments = _LINE_COMMENT.get(lang, ())
    block = lang in _BLOCK_COMMENT
    full_single = lang in _SINGLE_QUOTE_STRINGS
    backtick = lang in ("javascript", "typescript", "go")
    triple = lang in ("python", "gdscript")

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] not in "\r\n":
                out[k] = " "

    i = 0
    while i < n:
        ch = text[i]
        if block and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
            continue
        hit = next((c for c in line_comments if text.startswith(c, i)), "")
        if hit and (hit != "#" or i == 0 or text[i - 1] in " \t\n;"):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if triple and text.startswith(('"""', "'''"), i):
            j = text.find(text[i:i + 3], i + 3)
            j = n if j < 0 else j + 3
            blank(i + 3, j - 3)
            i = j
            continue
        if ch == '"' or (ch == "`" and backtick) or (ch == "'"
                                                    and full_single):
            j = i + 1
            while j < n and text[j] != ch:
                if text[j] == "\\" and not (ch == "`" and lang == "go"):
                    j += 2
                    continue
                if text[j] == "\n" and ch != "`":
                    break                                # unterminated
                j += 1
            blank(i + 1, j)
            i = j + 1
            continue
        if ch == "'":
            m = _CHAR_LITERAL.match(text, i)
            if m:
                blank(i + 1, m.end() - 1)
                i = m.end()
                continue
        i += 1
    blanked = "".join(out)
    if lang in ("c", "cpp"):
        blanked = _blank_if0(blanked)
    return blanked


def _blank_if0(text: str) -> str:
    """Blank the body of `#if 0 … #endif` (up to an `#else`), nesting-aware."""
    lines = text.split("\n")
    depth = 0
    for k, line in enumerate(lines):
        if depth == 0:
            if _IF0.match(line):
                depth = 1
            continue
        if _IF_ANY.match(line):
            depth += 1
        elif _ENDIF.match(line):
            depth -= 1
        elif depth == 1 and _ELSE.match(line):
            depth = 0
        if depth > 0:
            lines[k] = " " * len(line)
    return "\n".join(lines)

_IMPORT: dict[str, str] = {
    "c": r'^[ \t]*#\s*include\s*[<"](?P<name>[^">]+)[">]',
    "cpp": r'^[ \t]*#\s*include\s*[<"](?P<name>[^">]+)[">]',
    "rust": r"^[ \t]*(?:pub\s+)?use\s+(?P<name>[\w:]+)",
    "java": r"^[ \t]*import\s+(?:static\s+)?(?P<name>[\w.]+)",
    "go": r'^[ \t]*(?:import\s+)?"(?P<name>[\w./-]+)"',
    "javascript": r"^[ \t]*import\s+.*?from\s+['\"](?P<name>[^'\"]+)['\"]|"
                  r"require\(['\"](?P<name2>[^'\"]+)['\"]\)",
    "csharp": r"^[ \t]*using\s+(?P<name>[\w.]+)\s*;",
    "python": r"^[ \t]*(?:from\s+(?P<name>[\w.]+)\s+import|"
              r"import\s+(?P<name2>[\w.]+))",
    "gdscript": r"^[ \t]*(?:extends[ \t]+(?P<name>[\w.\"'/:]+)|"
                r"(?:const|var)\s+\w+\s*(?::=|=)\s*(?:preload|load)"
                r"\(['\"](?P<name2>[^'\"]+)['\"]\))",
    "lua": r"require\s*\(?['\"](?P<name>[\w.]+)['\"]",
    "ruby": r"^[ \t]*require(?:_relative)?\s+['\"](?P<name>[^'\"]+)['\"]",
}
_IMPORT["typescript"] = _IMPORT["javascript"]


def parse(text: str, path: str = "",
          lang_id: str = "") -> tuple[list[Symbol], list[tuple], list[tuple]]:
    """(symbols, edges, unresolved) — the same contract as `parse_python`."""
    lang = (lang_id or "").lower()
    pattern = _DECL.get(lang)
    module = _module_of(path)
    if not pattern:
        return [], [], []
    if module == "<file>":
        module = "<module>"

    lines = text.splitlines()
    blanked = blank_noise(text, lang)
    blines = blanked.splitlines()
    symbols: list[Symbol] = []
    declared_at: set[int] = set()
    for m in re.finditer(pattern, blanked, re.M):
        groups = {k: v for k, v in m.groupdict().items() if v}
        if "method" in groups and "recv" in groups:
            key, kind = "method", "method"
            name = f"{groups['recv']}.{groups['method']}"
        else:
            key = next((k for k in groups if k != "recv"), "")
            if not key:
                continue
            name = groups[key]
            kind = ("impl" if key == "impl" else "signal" if key == "signal"
                    else "class" if key.startswith("cls") else "function")
        bare = re.split(r"[.:]+", name)[-1]
        if not bare or bare in _KEYWORDS:
            continue
        line = blanked.count("\n", 0, m.start(key)) + 1
        declared_at.add(m.start(key))
        end = (_indent_end(lines, line) if lang in _INDENT_SCOPED
               else _brace_end(blines, line))
        symbols.append(Symbol(
            name=name, kind=kind, line=line, end_line=end,
            signature=_signature(text, m, key, lines, line, lang),
            path=path, approximate=True))

    edges: list[tuple] = []
    for name in imports_of(text, lang):
        edges.append((module, name, "imports"))

    # Attribute calls to the enclosing symbol by line span, so blast radius
    # points at a function rather than at a file. Calls are found in the
    # BLANKED text, so a string or a comment is never a call site.
    own = {s.name for s in symbols}
    by_short = {s.name.rsplit(".", 1)[-1]: s.name for s in symbols}
    unresolved: list[tuple] = []
    for m in _CALL.finditer(blanked):
        if m.start() in declared_at:
            continue                     # the declaration's own name
        called = m.group("name")
        head, short = called.split(".")[0], called.rsplit(".", 1)[-1]
        if head in ("this", "self") and "." in called:
            called = short                   # a method on this object
        if (head in _KEYWORDS or short in _KEYWORDS
                or called in ("print", "printf", "return")):
            continue
        line = blanked.count("\n", 0, m.start()) + 1
        src = _enclosing(symbols, line) or module
        if called == src:
            continue
        if called in own:
            edges.append((src, called, "calls"))
        elif short in by_short and by_short[short] != src:
            # `obj.method(` with `method` declared in this file: the object's
            # type is unknowable to a pattern, and the local method is the
            # likeliest target — the safe direction for blast radius.
            edges.append((src, by_short[short], "calls"))
        else:
            unresolved.append((src, called, "calls"))
    # The module's own row, so module-level calls and the imports above have
    # a source the store can bind (see parse_python.parse).
    symbols.insert(0, Symbol(
        name=module, kind="module", line=1, end_line=max(1, len(lines)),
        path=path, approximate=True))
    return symbols, edges, unresolved


#: A line holding only a return type or a template header, which a GNU-style
#: C declaration or a C++ template puts ABOVE the name.
_TYPE_LINE = re.compile(r"^[ \t]*(?:template[ \t]*<.*>|"
                        r"[A-Za-z_][\w:<>,*& \t]*)[ \t]*$")


def _signature(text: str, m: re.Match, key: str, lines: list[str],
               line: int, lang: str) -> str:
    """The declaration as written, whitespace collapsed, one line.

    Built from the MATCH when the pattern runs to the opening brace — so a
    parameter list over several lines is kept whole — and from the line
    otherwise. A C/C++ return type or template header on the line above is
    prepended; the old signature for `static int\nhelper(int a,\n int b)`
    was just `static int`.
    """
    start = text.rfind("\n", 0, m.start(key)) + 1
    chunk = text[start:m.end()]
    if chunk.rstrip().endswith("{"):
        sig = " ".join(chunk.rstrip().rstrip("{").split())
    else:
        sig = (lines[line - 1].strip().rstrip("{").strip()
               if line - 1 < len(lines) else m.group(key))
        sig = " ".join(sig.split())
    if lang in ("c", "cpp") and line >= 2:
        above = lines[line - 2].strip()
        head = sig.split("(")[0].strip()
        template = above.startswith("template")
        # Not an access label (`public:`) or anything that ends a statement.
        if (above and _TYPE_LINE.match(above)
                and not above.endswith((";", ":", "}", "{", ")"))
                and (template or " " not in head)):
            sig = f"{' '.join(above.split())} {sig}"
    return sig[:160]


def imports_of(text: str, lang_id: str = "") -> list[str]:
    pattern = _IMPORT.get((lang_id or "").lower())
    if not pattern:
        return []
    out: list[str] = []
    for m in re.finditer(pattern, text, re.M):
        name = m.groupdict().get("name") or m.groupdict().get("name2")
        if name:
            out.append(name)
    return out


#: Names BOUND in a file, per language — locals, parameters, imports — so
#: the D4 check can recognise `el.addEventListener` as an attribute on a
#: local object rather than an invented name (the Python equivalent is
#: `parse_python.bound_names`). Deliberately generous: every identifier in a
#: parameter list or destructuring counts, types included. Over-binding can
#: only make the check quieter about a dotted call on a declared name; under-
#: binding makes it cry wolf, which is how a check gets switched off.
_BOUND: dict[str, tuple[str, ...]] = {
    "javascript": (
        r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)",
        r"\b(?:const|let|var)\s*[\[{]([^\]}=]*)[\]}]",
        r"\bimport\s+([A-Za-z_$][\w$]*)",
        r"\bimport\s*\*\s*as\s+([A-Za-z_$][\w$]*)",
        r"\bimport\s*(?:[A-Za-z_$][\w$]*\s*,\s*)?\{([^}]*)\}",
        r"\bfunction\b[^(]*\(([^)]*)\)",
        r"\(([^()]*)\)\s*=>",
        r"\b([A-Za-z_$][\w$]*)\s*=>",
        r"\bcatch\s*\(\s*([A-Za-z_$][\w$]*)",
    ),
    "go": (
        r"\b((?:[A-Za-z_]\w*\s*,\s*)*[A-Za-z_]\w*)\s*:=",
        r"\bvar\s+([A-Za-z_]\w*)",
        r"\bfunc\s*(?:\([^)]*\)\s*)?\w*\s*\(([^)]*)\)",
        r"\bfunc\s*\(([^)]*)\)",
    ),
    "rust": (
        r"\blet\s+(?:mut\s+)?([A-Za-z_]\w*)",
        r"\blet\s+(?:mut\s+)?\(([^)]*)\)",
        r"\bfor\s+([A-Za-z_]\w*)\s+in\b",
        r"\|([^|]*)\|",
        r"\bfn\s+\w+\s*(?:<[^>]*>)?\s*\(([^)]*)\)",
        r"\b(?:if|while)\s+let\s+\w+\s*\(([^)]*)\)",
    ),
    "c": (
        r"\b[A-Za-z_][\w:<>,]*[\s*&]+([A-Za-z_]\w*)\s*(?:=|;|,|\[|\)|:)",
    ),
    "java": (
        r"\b[A-Za-z_][\w<>,.\[\]]*\s+([A-Za-z_]\w*)\s*(?:=|;|,|\)|:)",
        r"\bvar\s+([A-Za-z_]\w*)",
    ),
}
_BOUND["typescript"] = _BOUND["javascript"]
_BOUND["cpp"] = _BOUND["c"]
_BOUND["csharp"] = _BOUND["java"]
_BOUND_ANY = (
    r"\b(?:local|var|let|const|my|our)\s+([A-Za-z_]\w*)",
    r"(?m)^[ \t]*([A-Za-z_]\w*)\s*(?::=|=)(?!=)",
)
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


def bound_names(text: str, lang_id: str = "") -> set[str]:
    """Names declared anywhere in `text` (see `_BOUND`)."""
    out: set[str] = set()
    for pattern in _BOUND.get((lang_id or "").lower(), ()) + _BOUND_ANY:
        for m in re.finditer(pattern, text or ""):
            out.update(_IDENT.findall(m.group(1) or ""))
    return out


def _enclosing(symbols: Iterable[Symbol], line: int) -> str:
    best = ""
    best_line = -1
    for s in symbols:
        if s.line <= line <= (s.end_line or s.line) and s.line > best_line:
            best, best_line = s.name, s.line
    return best


def _indent_end(lines: list[str], line: int) -> int:
    """End of an indentation-scoped body. Derivable, so derive it."""
    if line - 1 >= len(lines):
        return line
    head = lines[line - 1]
    indent = len(head) - len(head.lstrip())
    end = line
    for n in range(line, len(lines)):
        body = lines[n]
        if body.strip() and (len(body) - len(body.lstrip())) <= indent:
            break
        if body.strip():
            end = n + 1
    return end


def _brace_end(lines: list[str], line: int) -> int:
    """End of a brace-delimited body. Approximate, and BOUNDED.

    Given the BLANKED lines (see `blank_noise`), so a brace inside a string
    or a comment does not move the end.

    Bounded on purpose: an unbalanced brace in a file mid-edit would
    otherwise walk this to the end of the file and report one symbol
    swallowing everything.
    """
    depth = 0
    seen = False
    for n in range(line - 1, min(len(lines), line + 400)):
        row = lines[n]
        if not seen and ";" in row and ("{" not in row
                                        or row.index(";") < row.index("{")):
            # A declaration that ends before any body opens — `type Pair =
            # [T, T];`, `const add = (a, b) => a + b;`. It used to run on to
            # the NEXT declaration's closing brace.
            return n + 1
        depth += row.count("{") - row.count("}")
        if "{" in row:
            seen = True
        if seen and depth <= 0:
            return n + 1
    return min(len(lines), line + 60)


def _module_of(path: str) -> str:
    p = str(path or "").replace("\\", "/").strip("/")
    return p or "<file>"
