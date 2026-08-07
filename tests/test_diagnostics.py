# SPDX-License-Identifier: Apache-2.0
"""Golden diagnostics — real captured toolchain output, asserted (§9).

Table-driven, with output copied from actual runs. Each case asserts the file,
the line and the message, because those three are what make a diagnostic
FIXABLE rather than merely present.

The last test in this file is the most important one in the module: **parsing
output nobody recognised must never return an empty list** (M29). Returning
`[]` on a failed build is how a loop reports success on broken code, and it is
the worst bug this module could have.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import diagnostics as dx  # noqa: E402

# (label, lang, raw output, expected file, expected line, text in the message)
GOLDEN = [
    ("gcc", "c",
     "main.c: In function 'main':\n"
     "main.c:5:9: error: 'x' undeclared (first use in this function)\n"
     "    5 |     x = 1;\n      |     ^\n",
     "main.c", 5, "undeclared"),

    ("gcc warning", "c",
     "main.c:7:12: warning: unused variable 'y' [-Wunused-variable]\n",
     "main.c", 7, "unused variable"),

    ("clang", "cpp",
     "src/app.cpp:12:5: error: use of undeclared identifier 'foo'\n",
     "src/app.cpp", 12, "undeclared identifier"),

    ("MSVC", "c",
     "C:\\proj\\main.c(14,9): error C2065: 'x': undeclared identifier\n",
     "C:\\proj\\main.c", 14, "undeclared identifier"),

    ("rustc", "rust",
     "error[E0425]: cannot find value `q` in this scope\n"
     " --> src/main.rs:3:13\n"
     "  |\n3 |     let z = q + 1;\n  |             ^ not found in this scope\n",
     "src/main.rs", 3, "cannot find value"),

    ("javac", "java",
     "Main.java:6: error: ';' expected\n        int x = 1\n"
     "                 ^\n1 error\n",
     "Main.java", 6, "expected"),

    ("go", "go",
     "./main.go:9:2: undefined: fmt.Printl\n",
     "./main.go", 9, "undefined"),

    ("TypeScript", "typescript",
     "src/app.ts(4,17): error TS2345: Argument of type 'string' is not "
     "assignable to parameter of type 'number'.\n",
     "src/app.ts", 4, "not assignable"),

    ("cppcheck", "c",
     "main.c:22:5: error: Memory leak: buf [memleak]\n",
     "main.c", 22, "Memory leak"),

    ("Godot script error", "gdscript",
     "SCRIPT ERROR: Invalid get index 'speed' on base: 'Nil'.\n"
     "   at: _ready (res://player.gd:14)\n",
     "player.gd", 14, "Invalid get index"),

    ("Godot parse error", "gdscript",
     "SCRIPT ERROR: Parse Error: Expected end of statement after "
     "expression, found ':' instead.\n"
     "          at: GDScript::reload (res://main.gd:7)\n",
     "main.gd", 7, "Parse Error"),
]


@pytest.mark.parametrize("label,lang,text,path,line,fragment", GOLDEN,
                         ids=[g[0] for g in GOLDEN])
def test_golden_toolchain_output(label, lang, text, path, line, fragment):
    diags = dx.parse(text, lang)
    assert diags, f"{label}: nothing was parsed"
    located = [d for d in diags if d.file]
    assert located, f"{label}: parsed, but with no location"
    best = located[0]
    assert best.file == path, f"{label}: file was {best.file!r}"
    assert best.line == line, f"{label}: line was {best.line}"
    assert fragment.lower() in best.message.lower(), (
        f"{label}: message was {best.message!r}")


def test_a_python_traceback_takes_the_DEEPEST_frame():
    """The deepest frame is where it broke; the rest is how it got there."""
    text = ('Traceback (most recent call last):\n'
            '  File "cli.py", line 12, in <module>\n    main()\n'
            '  File "app.py", line 40, in main\n    load()\n'
            '  File "io.py", line 88, in load\n    return 1 / 0\n'
            'ZeroDivisionError: division by zero\n')
    diags = dx.parse(text, "python")
    assert diags[0].file == "io.py"
    assert diags[0].line == 88
    assert "ZeroDivisionError" in diags[0].message


def test_a_javascript_stack_takes_the_FIRST_frame():
    """The opposite of Python. Getting it backwards points at the entry point."""
    text = ("TypeError: rows.map is not a function\n"
            "    at summarise (/app/src/stats.js:14:18)\n"
            "    at main (/app/src/cli.js:7:3)\n"
            "    at Object.<anonymous> (/app/src/index.js:1:1)\n")
    diags = dx.parse(text, "javascript")
    assert diags[0].file == "/app/src/stats.js"
    assert diags[0].line == 14


def test_unrecognised_output_yields_exactly_one_diagnostic_never_zero():
    """M29 — the worst bug this module could have.

    Returning `[]` on a failed build makes the loop report success on broken
    code. So anything unparseable comes back as one diagnostic holding the
    last meaningful lines.
    """
    text = ("linking...\nsome proprietary toolchain said something odd\n"
            "BUILD ABORTED, reason 0x8007\n")
    diags = dx.parse(text, "c")
    assert len(diags) == 1
    assert diags[0].code == "unparsed"
    assert "BUILD ABORTED" in diags[0].message


def test_empty_output_yields_nothing():
    """Nothing failed, so there is nothing to report. Not the same as M29."""
    assert dx.parse("", "c") == []
    assert dx.parse("   \n\n", "python") == []


def test_errors_sort_ahead_of_warnings():
    text = ("main.c:2:1: warning: unused variable 'a'\n"
            "main.c:9:5: error: 'b' undeclared\n")
    diags = dx.parse(text, "c")
    assert diags[0].severity == "error"
    assert diags[1].severity == "warning"


def test_the_located_duplicate_wins_over_the_unlocated_one():
    """Two parsers matching one line: keep the one that can be acted on."""
    text = ('  File "x.py", line 3, in f\n    1/0\n'
            'ZeroDivisionError: division by zero\n')
    diags = dx.parse(text, "python")
    same = [d for d in diags if "ZeroDivisionError" in d.message]
    assert len(same) == 1
    assert same[0].file == "x.py"


def test_source_is_quoted_around_the_error():
    """This is what turns a citation into something a small model can fix."""
    source = "a = 1\nb = 2\nc = undefined_name\nd = 4\ne = 5\n"
    diags = dx.parse("m.py:3:5: error: undefined name\n", "python")
    attached = dx.attach_source(diags, sources={"m.py": source})
    assert ">>    3 | c = undefined_name" in attached[0].source_excerpt
    assert "b = 2" in attached[0].source_excerpt      # context before
    assert "d = 4" in attached[0].source_excerpt      # context after


def test_feedback_is_capped_and_says_how_much_it_held_back():
    text = "\n".join(f"m.c:{n}:1: error: problem {n}" for n in range(1, 9))
    diags = dx.parse(text, "c")
    out = dx.feedback(diags, max_errors=3)
    assert out.count("error: problem") == 3
    assert "5 more" in out


def test_cascade_languages_get_one_error_and_a_cascade_explanation():
    """F7 — in C++ the fortieth error is a consequence of the first."""
    text = "\n".join(f"a.cpp:{n}:1: error: expected ';'" for n in range(1, 12))
    out = dx.feedback_for(text, "cpp")
    assert out.count("error: expected") == 1
    assert "cascade" in out.lower()


def test_non_cascade_languages_keep_the_cap_of_three():
    text = "\n".join(f"m.py:{n}:1: error: problem {n}" for n in range(1, 9))
    out = dx.feedback_for(text, "python")
    assert out.count("error: problem") == 3


def test_the_diagnostic_signature_is_order_independent():
    """The stagnation detector hashes this; ordering is not a change (M34)."""
    a = dx.parse("m.c:1:1: error: one\nm.c:2:1: error: two\n", "c")
    b = dx.parse("m.c:2:1: error: two\nm.c:1:1: error: one\n", "c")
    assert dx.signature(a) == dx.signature(b)
