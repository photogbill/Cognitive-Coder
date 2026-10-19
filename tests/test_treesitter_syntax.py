# SPDX-License-Identifier: Apache-2.0
"""A parser's syntax check when there is no compiler (Aider's lint idea).

Before this, a C file on a machine with no compiler was "not checked" by
the audit and BLOCKED by the build — with a stray brace in it that any
parser can see. Now the brace is a located error the model can fix; a
clean parse still says "not checked", because a grammar is not a compiler.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import runner  # noqa: E402
from cognitive_coder.codemap import parse_treesitter  # noqa: E402

needs_c = pytest.mark.skipif(not parse_treesitter.available("c"),
                             reason="the tree-sitter C grammar is not "
                                    "installed (an optional extra)")

BROKEN_C = "int main(void) {\n    int x = 1\n    return x;\n}\n"
GOOD_C = "int main(void) {\n    int x = 1;\n    return x;\n}\n"


@needs_c
def test_a_missing_semicolon_is_found_and_located():
    found = parse_treesitter.syntax_errors(BROKEN_C, "c")
    assert found, "a parser sees the missing semicolon"
    line, _col, what = found[0]
    assert line in (2, 3)
    assert "missing" in what or "does not parse" in what


@needs_c
def test_a_clean_file_has_no_errors_and_a_clean_parse_is_not_a_pass():
    assert parse_treesitter.syntax_errors(GOOD_C, "c") == []
    # syntax_check with no compiler: None — "not checked", never "fine"
    assert runner.syntax_check(GOOD_C, "c", ex=None, src_path="m.c") is None


@needs_c
def test_with_no_compiler_the_syntax_check_reports_the_error():
    phase = runner.syntax_check(BROKEN_C, "c", ex=None, src_path="m.c")
    assert phase is not None and not phase.ok
    d = phase.diagnostics[0]
    assert d.file == "m.c" and d.tool == "tree-sitter" and d.line in (2, 3)
    assert "no compiler" in d.message


def test_a_language_with_no_grammar_is_not_checked():
    assert parse_treesitter.syntax_errors("anything at all {", "lua") is None
    # TypeScript is left out on purpose: its grammar cries wolf
    assert parse_treesitter.syntax_errors("let x = 1;", "typescript") is None


class _NoCompiler:
    """An ExecPort with nothing on PATH: every toolchain is missing."""

    def which(self, name):
        return None

    def run(self, *a, **kw):                    # pragma: no cover
        raise AssertionError("nothing should run without a compiler")


@needs_c
def test_a_build_with_no_compiler_reports_the_syntax_error_not_blocked(
        tmp_path):
    from cognitive_coder.ports import LocalFileSystem
    fs = LocalFileSystem(str(tmp_path))
    broken = runner.build_and_run(BROKEN_C, "c", fs=fs, ex=_NoCompiler(),
                                  stem="m", path="m.c")
    assert not broken.ok and not broken.blocked
    assert broken.diagnostics and broken.diagnostics[0].tool == "tree-sitter"
    clean = runner.build_and_run(GOOD_C, "c", fs=fs, ex=_NoCompiler(),
                                 stem="m", path="m.c")
    assert clean.blocked, "a clean parse is still not a build"
