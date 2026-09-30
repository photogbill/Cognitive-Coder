# SPDX-License-Identifier: Apache-2.0
"""SubprocessExec output capture — bounded, decodable, and keeping the end.

The engine RUNS generated code. A generated program is exactly the kind of
program that loops printing, prints bytes that are not UTF-8, or prints
twelve thousand lines of verbose test progress before the one line that
says what failed. Each of those was observed to hurt the ENGINE, not the
program:

  * a four-line C `for(;;) puts(...)` OOM-killed the engine (exit 137 /
    MemoryError after 1.1 s) — `communicate()` buffered everything and the
    200 KB cap ran afterwards, on output that no longer fitted in memory;
  * a child printing byte 0xFF raised UnicodeDecodeError out of `run()` as
    a traceback, because the pipe was decoded strictly in the locale codec;
  * a verbose unittest run kept its first 200 KB — `test_11109 ... ok` —
    and dropped `FAIL:` and `Ran N tests`, so the model was handed a
    passing line as the diagnostic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cognitive_coder import diagnostics  # noqa: E402
from cognitive_coder.ports import SubprocessExec  # noqa: E402

PY = sys.executable


def _need(*binaries):
    for b in binaries:
        if SubprocessExec().which(b):
            return
    pytest.skip(f"none of {binaries} is installed on this machine")


# --------------------------------------------------------------------------
# a runaway printer cannot take the engine down
# --------------------------------------------------------------------------

_HARNESS = textwrap.dedent("""
    import json, resource, sys, time
    sys.path.insert(0, {root!r})
    # The engine process gets a ceiling it would blow through if it
    # buffered a runaway child's output, so the failure is a clean
    # MemoryError in THIS process rather than the OOM-killer elsewhere.
    resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
    from cognitive_coder.ports import SubprocessExec
    t0 = time.monotonic()
    res = SubprocessExec().run({argv!r}, cwd={cwd!r}, timeout=0)
    print(json.dumps({{"seconds": time.monotonic() - t0,
                      "exit": res.exit_code, "truncated": res.truncated,
                      "timed_out": res.timed_out,
                      "stdout": len(res.stdout), "stderr": res.stderr}}))
""")


def _run_in_bounded_engine(argv, cwd):
    """Run SubprocessExec in a child Python with a hard memory ceiling."""
    script = _HARNESS.format(root=str(ROOT), argv=list(argv), cwd=str(cwd))
    try:
        done = subprocess.run([PY, "-c", script], capture_output=True,
                              text=True, timeout=90)
    except subprocess.TimeoutExpired:
        pytest.fail("the engine never returned from a runaway printer with "
                    "timeout=0 — output capture has no ceiling")
    assert done.returncode == 0, (
        f"the engine process died (exit {done.returncode}): "
        f"{done.stderr[-400:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.toolchain
def test_an_endless_c_printer_with_no_timeout_returns_bounded(tmp_path):
    _need("gcc", "cc", "clang")
    cc = next(b for b in ("gcc", "cc", "clang") if SubprocessExec().which(b))
    src = tmp_path / "spam.c"
    src.write_text('#include <stdio.h>\nint main(void){ for(;;) '
                   'puts("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"); }\n')
    subprocess.run([cc, str(src), "-o", str(tmp_path / "spam")], check=True)
    out = _run_in_bounded_engine([str(tmp_path / "spam")], tmp_path)
    assert out["seconds"] < 20, out
    assert out["truncated"], out
    assert out["stdout"] < 400_000, "the kept output is not bounded"
    assert "bytes omitted" in out["stderr"] or out["stdout"], out
    # Said as a sentence, naming what happened and why it was stopped (C6).
    assert "more than" in out["stderr"] and "stopped" in out["stderr"]


def test_an_endless_python_printer_with_no_timeout_returns_bounded(
        tmp_path):
    out = _run_in_bounded_engine(
        [PY, "-c", "import sys\nw = sys.stdout.write\n"
                   "while True: w('y' * 4096 + '\\n')"], tmp_path)
    assert out["seconds"] < 30, out
    assert out["truncated"] and out["stdout"] < 400_000, out


# --------------------------------------------------------------------------
# bytes that are not text
# --------------------------------------------------------------------------

def test_a_child_printing_a_non_utf8_byte_does_not_raise(tmp_path):
    res = SubprocessExec().run(
        [PY, "-c", "import sys; sys.stdout.buffer.write(b'\\xff done\\n'); "
                   "sys.stderr.buffer.write(b'err \\xfe\\n')"],
        cwd=str(tmp_path), timeout=30)
    assert res.exit_code == 0
    assert "done" in res.stdout and "err" in res.stderr


@pytest.mark.toolchain
def test_a_c_program_printing_0xff_is_a_result_not_a_traceback(tmp_path):
    _need("gcc", "cc", "clang")
    from cognitive_coder import runner
    from cognitive_coder.ports import LocalFileSystem
    code = ('#include <stdio.h>\nint main(void){ putchar(0xff); '
            'puts(" done"); return 0; }\n')
    result = runner.build_and_run(code, "c", fs=LocalFileSystem(
        str(tmp_path)), ex=SubprocessExec(), stem="raw")
    assert result.ok, result.summary()
    assert "done" in result.phases[-1].proc.stdout


# --------------------------------------------------------------------------
# the END of a long log is where the verdict is
# --------------------------------------------------------------------------

def test_a_12k_line_unittest_run_keeps_its_FAIL_line(tmp_path):
    (tmp_path / "test_big.py").write_text(textwrap.dedent("""
        import unittest

        class T(unittest.TestCase):
            def test_zz_the_real_failure(self):
                self.assertEqual(2 + 2, 5)

        for i in range(12000):
            setattr(T, f"test_{i:05d}", lambda self: None)
    """))
    res = SubprocessExec().run(
        [PY, "-m", "unittest", "discover", "-s", str(tmp_path), "-p",
         "test_*.py", "-v"], cwd=str(tmp_path), timeout=300)
    assert res.exit_code != 0
    assert res.truncated, "12k lines should exceed the kept window"
    assert "FAIL: test_zz_the_real_failure" in res.stderr
    assert "Ran 12001 tests" in res.stderr
    assert "bytes omitted" in res.stderr
    diags = diagnostics.parse(res.output, "python")
    assert any("test_zz_the_real_failure" in d.message
               or "AssertionError" in d.message for d in diags), diags[:3]


# --------------------------------------------------------------------------
# the rest of the ExecPort contract still holds (M16)
# --------------------------------------------------------------------------

def test_the_conformance_kit_exec_checks_pass(tmp_path):
    from tests.port_conformance import check_exec
    report = check_exec(SubprocessExec(), str(tmp_path))
    assert report.ok, report.text()


def test_a_grandchild_in_its_own_session_does_not_cost_the_output(
        tmp_path):
    """Best effort, and documented as such: a grandchild that called
    `start_new_session` escapes the process-group kill and keeps the pipe
    open. `run()` must still return promptly WITH what was captured, rather
    than waiting on the pipe and then throwing the output away."""
    inner = "import time; time.sleep(8)"
    child = ("import subprocess, sys, time\n"
             "print('captured before the hang', flush=True)\n"
             f"subprocess.Popen([sys.executable, '-c', {inner!r}], "
             "start_new_session=True)\n"
             "time.sleep(60)\n")
    t0 = time.monotonic()
    res = SubprocessExec().run([PY, "-c", child], cwd=str(tmp_path),
                               timeout=1.5)
    elapsed = time.monotonic() - t0
    assert res.timed_out
    assert elapsed < 5, f"returned after {elapsed:.1f}s"
    assert "captured before the hang" in res.stdout
    subprocess.run(["pkill", "-f", inner], capture_output=True) \
        if os.name == "posix" else None


def test_no_env_is_the_scrubbed_env_not_the_inherited_one(tmp_path,
                                                         monkeypatch):
    """`scrub_env` was stored and never read, so `env=None` was Popen's
    "inherit everything": the review stage's scanners, which passed no
    env, ran with the operator's API keys and proxy variables."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    res = SubprocessExec().run(
        [PY, "-c", "import os; print(os.environ.get('ANTHROPIC_API_KEY')); "
                   "print(bool(os.environ.get('HTTPS_PROXY')))"],
        cwd=str(tmp_path), timeout=30)
    assert res.stdout.split() == ["None", "False"], res.stdout


def test_a_host_that_wants_inheritance_can_still_ask_for_it(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("CC_INHERITED", "yes")
    res = SubprocessExec(scrub_env=False).run(
        [PY, "-c", "import os; print(os.environ.get('CC_INHERITED'))"],
        cwd=str(tmp_path), timeout=30)
    assert res.stdout.strip() == "yes"
