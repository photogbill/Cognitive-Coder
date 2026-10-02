# SPDX-License-Identifier: Apache-2.0
"""pytest-style tests are run by pytest when it is there — and named when
it is not.

Oct 1, 2026, 21:22: the model wrote `tests/test_physics.py` for pytest —
`import pytest`, a bare class, bare asserts. The host installed pytest on
request (eight minutes of dialog), then the engine ran `unittest discover`,
which collected zero tests, told the model "for unittest, methods named
test_* on a class that subclasses unittest.TestCase", and got the identical
file back. Twelve good tests, never run.

Two things fixed. When the tests in play are pytest-style and pytest is
importable in the environment that runs them, pytest runs them (it runs
unittest tests too). When pytest is NOT there, the zero-tests message says
the file is written for pytest and what to write instead.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import LocalFileSystem, SubprocessExec  # noqa: E402
from cognitive_coder import runner  # noqa: E402

PYTEST_STYLE = textwrap.dedent('''\
    import pytest
    from src.calc import add


    class TestAdd:
        def test_add(self):
            assert add(1, 2) == 3

        def test_add_negative(self):
            assert add(-1, 1) == 0
    ''')

BARE_FUNCTIONS = textwrap.dedent('''\
    from src.calc import add


    def test_add():
        assert add(1, 2) == 3
    ''')

UNITTEST_STYLE = textwrap.dedent('''\
    import unittest
    from src.calc import add


    class T(unittest.TestCase):
        def test_add(self):
            self.assertEqual(add(1, 2), 3)


    if __name__ == "__main__":
        unittest.main()
    ''')

FAILING_PYTEST = textwrap.dedent('''\
    from src.calc import add


    def test_add_wrong():
        assert add(1, 2) == 4
    ''')


def _project(tmp_path, test_source, name="tests/test_calc.py"):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("")
    (tmp_path / name).write_text(test_source)
    return LocalFileSystem(str(tmp_path))


class _NoPytest(SubprocessExec):
    """An environment where `import pytest` fails, whatever is installed
    here."""

    def run(self, argv, **kw):
        if len(argv) >= 3 and argv[1] == "-c" and "import pytest" in argv[2]:
            from cognitive_coder.types import ProcResult
            return ProcResult(exit_code=1, stderr="No module named pytest")
        return super().run(argv, **kw)


def test_style_detection():
    assert runner.is_pytest_style(PYTEST_STYLE)
    assert runner.is_pytest_style(BARE_FUNCTIONS)
    assert not runner.is_pytest_style(UNITTEST_STYLE)
    # TestCase wins even next to an `import pytest`: unittest collects it
    assert not runner.is_pytest_style("import pytest\n" + UNITTEST_STYLE)
    assert not runner.is_pytest_style("")


def test_pytest_style_tests_run_under_pytest_when_it_is_importable(tmp_path):
    import pytest as _pytest  # noqa: F401 — this interpreter has it
    fs = _project(tmp_path, PYTEST_STYLE)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              stem="calc", path="tests/test_calc.py",
                              timeout=60)
    assert result.ok, result.summary()
    assert any("pytest" in c for c in result.caveats), result.caveats
    assert not any("ZERO" in c for c in result.caveats), result.caveats
    assert all(d.code != "no-tests-collected" for d in result.diagnostics)


def test_a_failing_pytest_test_is_a_failure_with_a_location(tmp_path):
    fs = _project(tmp_path, FAILING_PYTEST)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              stem="calc", path="tests/test_calc.py",
                              timeout=60)
    assert not result.ok
    errors = [d for d in result.diagnostics if d.is_error]
    assert errors, result.summary()
    assert any("assert" in (d.message or "").lower() for d in errors)
    # `--tb=native` gives the parser an ordinary traceback with the file
    assert any(d.file and d.file.replace("\\", "/").endswith(
        "tests/test_calc.py") for d in errors), [d.file for d in errors]


def test_unittest_style_tests_keep_the_unittest_runner(tmp_path):
    fs = _project(tmp_path, UNITTEST_STYLE)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              stem="calc", path="tests/test_calc.py",
                              timeout=60)
    assert result.ok, result.summary()
    assert not any("pytest" in c for c in result.caveats), result.caveats


def test_without_pytest_the_zero_tests_message_names_the_cause(tmp_path):
    fs = _project(tmp_path, PYTEST_STYLE)
    result = runner.run_tests("python", fs=fs, ex=_NoPytest(),
                              stem="calc", path="tests/test_calc.py",
                              timeout=60)
    assert not result.ok
    diag = next(d for d in result.diagnostics
                if d.code == "no-tests-collected")
    assert "written for pytest" in diag.message
    assert "pytest is not installed" in diag.message
    assert "unittest.TestCase" in diag.message


def test_a_module_s_own_test_path_is_run_with_pytest_too(tmp_path):
    """The module task's verify (scoped to its test_path) must pick the
    same runner as the test task's, or a module is "built, not verified"
    while its pytest tests pass."""
    fs = _project(tmp_path, PYTEST_STYLE)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              stem="calc", path="src/calc.py",
                              test_path="tests/test_calc.py", timeout=60)
    assert result.ok, result.summary()
    assert any("pytest" in c for c in result.caveats), result.caveats
    assert not any("ZERO" in c for c in result.caveats), result.caveats


def test_whole_suite_uses_pytest_when_any_test_file_needs_it(tmp_path):
    fs = _project(tmp_path, PYTEST_STYLE)
    (tmp_path / "tests" / "test_more.py").write_text(UNITTEST_STYLE)
    result = runner.run_tests("python", fs=fs, ex=SubprocessExec(),
                              stem="calc", whole_suite=True, timeout=60)
    assert result.ok, result.summary()
    assert any("pytest" in c for c in result.caveats), result.caveats
