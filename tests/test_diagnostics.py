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

# Captured from a real rustc 1.95.0 run with NO_COLOR=1 TERM=dumb, the
# environment the runner gives it. Not hand-written: the trailer lines are
# the point.
RUSTC_195_ONE = (
    "error[E0425]: cannot find value `q` in this scope\n"
    " --> src/main.rs:2:13\n  |\n2 |     let z = q + 1;\n"
    "  |             ^ not found in this scope\n\n"
    "error: aborting due to 1 previous error\n\n"
    "For more information about this error, try `rustc --explain E0425`.\n")
RUSTC_195_TWO = (
    "error[E0425]: cannot find value `q` in this scope\n"
    " --> src/two.rs:3:13\n  |\n3 |     let z = q + 1;\n"
    "  |             ^ not found in this scope\n\n"
    "error[E0308]: mismatched types\n --> src/two.rs:4:21\n  |\n"
    "4 |     let s: String = 5;\n"
    "  |            ------   ^ expected `String`, found integer\n"
    "  |            |\n  |            expected due to this\n  |\n"
    "help: try using a conversion method\n  |\n"
    "4 |     let s: String = 5.to_string();\n"
    "  |                      ++++++++++++\n\n"
    "error: aborting due to 2 previous errors\n\n"
    "Some errors have detailed explanations: E0308, E0425.\n"
    "For more information about an error, try `rustc --explain E0308`.\n")


# Real output, captured from the installed toolchains (Python 3.12,
# Node 22.22, pytest 8, Ruby 3.3.6, Go 1.24, ruff 0.15) with NO_COLOR=1
# TERM=dumb, the environment the runner gives them. Only the scratch
# folder's path was replaced with /home/dev/proj; long lines are split
# into adjacent literals, byte for byte.

PY312_LIBRARY_RAISE = (
    'Traceback (most recent call last):\n'
    '  File "/home/dev/proj/lib_err.py", line 8, in <module>\n'
    "    load('{bad')\n"
    '  File "/home/dev/proj/lib_err.py", line 5, in load\n'
    '    return json.loads(text)\n'
    '           ^^^^^^^^^^^^^^^^\n'
    '  File "/usr/lib/python3.12/json/__init__.py", line 346, in loads\n'
    '    return _default_decoder.decode(s)\n'
    '           ^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
    '  File "/usr/lib/python3.12/json/decoder.py", line 337, in decode\n'
    '    obj, end = self.raw_decode(s, idx=_w(s, 0).end())\n'
    '               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
    '  File "/usr/lib/python3.12/json/decoder.py", line 353, in raw_decode\n'
    '    obj, end = self.scan_once(s, idx)\n'
    '               ^^^^^^^^^^^^^^^^^^^^^^\n'
    'json.decoder.JSONDecodeError: Expecting property name enclos'
    'ed in double quotes: line 1 column 2 (char 1)\n')

NODE22_CJS_SYNTAX = (
    '/home/dev/proj/syn.js:2\n'
    '  return 1;\n'
    '         ^\n'
    '\n'
    'SyntaxError: Unexpected number\n'
    '    at wrapSafe (node:internal/modules/cjs/loader:1637:18)\n'
    '    at Module._compile (node:internal/modules/cjs/loader:1679:20)\n'
    '    at Object..js (node:internal/modules/cjs/loader:1838:10)\n'
    '    at Module.load (node:internal/modules/cjs/loader:1441:32)\n'
    '    at Function._load (node:internal/modules/cjs/loader:1263:12)\n'
    '    at TracingChannel.traceSync (node:diagnostics_channel:328:14)\n'
    '    at wrapModuleLoad (node:internal/modules/cjs/loader:237:24)\n'
    '    at Function.executeUserEntryPoint [as runMain] (node:int'
    'ernal/modules/run_main:171:5)\n'
    '    at node:internal/main/run_main_module:36:49\n'
    '\n'
    'Node.js v22.22.2\n')

NODE22_ESM_SYNTAX = (
    'file:///home/dev/proj/bad.mjs:1\n'
    'import { nope } from "./app.mjs";\n'
    '         ^^^^\n'
    "SyntaxError: The requested module './app.mjs' does not provi"
    "de an export named 'nope'\n"
    '    at ModuleJob._instantiate (node:internal/modules/esm/mod'
    'ule_job:226:21)\n'
    '    at async ModuleJob.run (node:internal/modules/esm/module'
    '_job:335:5)\n'
    '    at async onImport.tracePromise.__proto__ (node:internal/'
    'modules/esm/loader:665:26)\n'
    '    at async asyncRunEntryPointWithESMLoader (node:internal/'
    'modules/run_main:117:5)\n'
    '\n'
    'Node.js v22.22.2\n')

NODE22_TAP_FAILURE = (
    'TAP version 13\n'
    '# Subtest: adds\n'
    'not ok 1 - adds\n'
    '  ---\n'
    '  duration_ms: 1.293519\n'
    "  type: 'test'\n"
    "  location: '/home/dev/proj/t/a.test.mjs:4:1'\n"
    "  failureType: 'testCodeFailure'\n"
    '  error: |-\n'
    '    Expected values to be strictly equal:\n'
    '    \n'
    '    2 !== 3\n'
    '    \n'
    "  code: 'ERR_ASSERTION'\n"
    "  name: 'AssertionError'\n"
    '  expected: 3\n'
    '  actual: 2\n'
    "  operator: 'strictEqual'\n"
    '  stack: |-\n'
    '    TestContext.<anonymous> (file:///home/dev/proj/t/a.test.mjs:5:12)\n'
    '    Test.runInAsyncScope (node:async_hooks:214:14)\n'
    '    Test.run (node:internal/test_runner/test:1047:25)\n'
    '    Test.start (node:internal/test_runner/test:944:17)\n'
    '    startSubtestAfterBootstrap (node:internal/test_runner/ha'
    'rness:296:17)\n'
    '  ...\n'
    '1..1\n'
    '# tests 1\n'
    '# suites 0\n'
    '# pass 0\n'
    '# fail 1\n'
    '# cancelled 0\n'
    '# skipped 0\n'
    '# todo 0\n'
    '# duration_ms 65.010691\n')

PYTEST8_LONG = (
    'FF                                                          '
    '             [100%]\n'
    '=================================== FAILURES ==============='
    '====================\n'
    '____________________________________ test_a ________________'
    '____________________\n'
    '\n'
    '    def test_a():\n'
    '>       assert 3 == 4\n'
    'E       assert 3 == 4\n'
    '\n'
    'test_x.py:2: AssertionError\n'
    '____________________________________ test_b ________________'
    '____________________\n'
    '\n'
    '    def test_b():\n'
    '>       raise KeyError("k")\n'
    "E       KeyError: 'k'\n"
    '\n'
    'test_x.py:6: KeyError\n'
    '=========================== short test summary info ========'
    '====================\n'
    'FAILED test_x.py::test_a - assert 3 == 4\n'
    "FAILED test_x.py::test_b - KeyError: 'k'\n"
    '2 failed in 0.02s\n')

RUBY33_SYNTAX = (
    "ruby: r.rb:3: syntax error, unexpected `end' (SyntaxError)\n")

RUBY33_RUNTIME = (
    "r2.rb:2:in `f': undefined method `foo' for nil (NoMethodError)\n"
    '\n'
    '  nil.foo\n'
    '     ^^^^\n'
    "\tfrom r2.rb:4:in `<main>'\n")

GO124_VET_TYPE = (
    '# command-line-arguments\n'
    '# [command-line-arguments]\n'
    'vet: ./vt.go:5:19: undefined: fmt.Printl\n')

GO124_VET_CHECK = (
    '# command-line-arguments\n'
    '# [command-line-arguments]\n'
    './vet.go:6:2: fmt.Printf format %d has arg "x" of wrong type string\n')

RUFF015_FULL = (
    'F401 [*] `os` imported but unused\n'
    ' --> lint_me.py:1:8\n'
    '  |\n'
    '1 | import os\n'
    '  |        ^^\n'
    '  |\n'
    'help: Remove unused import: `os`\n'
    '\n'
    'F821 Undefined name `undefined_name`\n'
    ' --> lint_me.py:5:12\n'
    '  |\n'
    '4 | def f():\n'
    '5 |     return undefined_name\n'
    '  |            ^^^^^^^^^^^^^^\n'
    '  |\n'
    '\n'
    'Found 2 errors.\n'
    '[*] 1 fixable with the `--fix` option.\n')

RUFF015_CONCISE = (
    'lint_me.py:1:8: F401 [*] `os` imported but unused\n'
    'lint_me.py:5:12: F821 Undefined name `undefined_name`\n'
    'Found 2 errors.\n'
    '[*] 1 fixable with the `--fix` option.\n')


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

    ("rustc 1.95, real, with its trailer", "rust", RUSTC_195_ONE,
     "src/main.rs", 2, "cannot find value"),

    ("Python 3.12, raised inside the stdlib", "python", PY312_LIBRARY_RAISE,
     "/home/dev/proj/lib_err.py", 5, "JSONDecodeError"),

    ("Node 22 CommonJS SyntaxError", "javascript", NODE22_CJS_SYNTAX,
     "/home/dev/proj/syn.js", 2, "Unexpected number"),

    ("Node 22 ESM SyntaxError", "javascript", NODE22_ESM_SYNTAX,
     "/home/dev/proj/bad.mjs", 1, "does not provide an export"),

    ("node --test TAP failure", "javascript", NODE22_TAP_FAILURE,
     "/home/dev/proj/t/a.test.mjs", 5, "2 !== 3"),

    ("pytest 8 long traceback", "python", PYTEST8_LONG,
     "test_x.py", 2, "assert 3 == 4"),

    ("ruby -c", "ruby", RUBY33_SYNTAX, "r.rb", 3, "syntax error"),

    ("Ruby runtime error", "ruby", RUBY33_RUNTIME, "r2.rb", 2,
     "NoMethodError: undefined method"),

    ("go vet, type error", "go", GO124_VET_TYPE, "./vt.go", 5, "undefined"),

    ("go vet, a vet check", "go", GO124_VET_CHECK, "./vet.go", 6,
     "wrong type"),

    ("ruff, full format", "python", RUFF015_FULL, "lint_me.py", 5,
     "Undefined name"),

    ("ruff, concise", "python", RUFF015_CONCISE, "lint_me.py", 5,
     "Undefined name"),

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


def test_rustc_feedback_names_the_real_error_not_the_trailer():
    """Observed: `error: aborting due to 1 previous error` is unlocated, so
    it sorted FIRST — and Rust cascades, so the feedback cap is one. The
    model was handed the trailer and nothing else."""
    out = dx.feedback_for(RUSTC_195_ONE, "rust")
    assert "E0425" in out and "cannot find value" in out, out
    assert "aborting" not in out
    assert "further error" not in out, "the trailer was counted as an error"


def test_rustc_two_errors_are_both_counted_and_the_trailer_is_not():
    diags = dx.parse(RUSTC_195_TWO, "rust")
    assert [d.code for d in diags] == ["E0425", "E0308"], diags
    out = dx.feedback_for(RUSTC_195_TWO, "rust")
    assert "E0425" in out and "(1 further error" in out, out


@pytest.mark.parametrize("trailer", [
    "error: aborting due to 3 previous errors; 1 warning emitted",
    "warning: 2 warnings emitted",
    "error: could not compile `app` (bin \"app\") due to 1 previous error",
])
def test_rustc_and_cargo_trailers_are_not_diagnostics(trailer):
    text = ("warning: unused variable: `x`\n --> src/w.rs:2:9\n  |\n"
            "2 |     let x = 3;\n  |         ^ help: prefix it: `_x`\n\n"
            + trailer + "\n")
    diags = dx.parse(text, "rust")
    assert [d.message for d in diags] == ["unused variable: `x`"], diags


def test_unlocated_diagnostics_sort_after_located_ones_of_equal_rank():
    """A diagnostic with no file is the same information minus the part
    that makes it fixable; it must not displace one that has it."""
    text = ("error: linker `cc` not found\n\n"
            "error[E0425]: cannot find value `q` in this scope\n"
            " --> src/main.rs:2:13\n")
    diags = dx.parse(text, "rust")
    assert len(diags) == 2, diags
    assert diags[0].file == "src/main.rs", diags


def test_a_library_raise_is_located_in_the_projects_own_frame():
    """Observed: the deepest frame was /usr/lib/python3.12/json/decoder.py,
    so the model was pointed at the standard library — and attach_source,
    which reads through the project's FileSystemPort, quoted nothing."""
    for root in ("/home/dev/proj", ""):
        d = dx.parse(PY312_LIBRARY_RAISE, "python", root=root)[0]
        assert (d.file, d.line) == ("/home/dev/proj/lib_err.py", 5), root
        assert d.code == "in load"


def test_a_frame_outside_the_given_root_is_not_the_projects():
    text = ('Traceback (most recent call last):\n'
            '  File "/srv/app/main.py", line 3, in <module>\n    run()\n'
            '  File "/opt/vendored/lib.py", line 9, in run\n    1 / 0\n'
            'ZeroDivisionError: division by zero\n')
    d = dx.parse(text, "python", root="/srv/app")[0]
    assert (d.file, d.line) == ("/srv/app/main.py", 3)
    # With no root, the vendored path is not recognisably a library.
    assert dx.parse(text, "python")[0].file == "/opt/vendored/lib.py"


def test_node_file_urls_become_paths_attach_source_can_read():
    text = ("TypeError: rows.map is not a function\n"
            "    at file:///home/dev/proj/app.mjs:4:6\n"
            "    at ModuleJob.run "
            "(node:internal/modules/esm/module_job:271:25)\n")
    d = dx.parse(text, "javascript")[0]
    assert (d.file, d.line) == ("/home/dev/proj/app.mjs", 4)


def test_pytest_failures_carry_their_line_and_each_test_is_one_failure():
    diags = [d for d in dx.parse(PYTEST8_LONG, "python") if d.file]
    got = {(d.code, d.line) for d in diags if d.tool == "pytest"}
    assert got == {("test_a", 2), ("test_b", 6)}, diags


def test_ruff_lint_findings_keep_their_codes_and_severities():
    diags = dx.parse(RUFF015_FULL, "python")
    by_code = {d.code: d for d in diags}
    assert by_code["F401"].severity == "warning" and by_code["F401"].line == 1
    assert by_code["F821"].severity == "error"
    assert diags[0].code == "F821", "the error sorts ahead of the warning"


def test_feedback_for_a_python_library_raise_quotes_the_users_line():
    src = "import json\n\n\ndef load(text):\n    return json.loads(text)\n"
    out = dx.feedback_for(PY312_LIBRARY_RAISE, "python",
                          sources={"/home/dev/proj/lib_err.py": src})
    assert ">>    5 |     return json.loads(text)" in out, out


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
