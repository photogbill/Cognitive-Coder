# SPDX-License-Identifier: Apache-2.0
"""The ATK adapter's Ports, tested without Qt and without ATK.

pytest style, unlike `test_migration.py` beside it: that one is written the
way ATK's own suite is (a script with PASS/FAIL lines, §7.1) because it
proves ATK's call surface; this one proves the Ports against the engine's
contract, and runs in Cognitive Coder's CI with the rest of its suite.

Every test here pins a defect a review reproduced — each docstring says
what was observed. None imports `ccoder_panel` (PySide6 is not a test
dependency); the panel's logic that could be wrong without Qt was moved
into `ccoder_host` so that it could be tested here.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
import types

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))              # the CC clone
sys.path.insert(0, str(HERE))                            # this adapter
sys.path.insert(0, str(HERE.parent.parent / "tests"))    # the kit

import ccoder_host  # noqa: E402
from ccoder_host import (  # noqa: E402
    ATKLLM,
    AskOnGuiThread,
    ATKApproval,
    ATKExec,
    ATKStorage,
    build_host,
    change_log,
    failure_line,
    project_key,
    project_root_for,
)
import port_conformance  # noqa: E402

from cognitive_coder import Message  # noqa: E402
from cognitive_coder.patcher import Patcher  # noqa: E402
from cognitive_coder.ports import (  # noqa: E402
    AutoApprove,
    MemoryFileSystem,
    MemoryStorage,
)
from cognitive_coder.types import Edit  # noqa: E402

USER = [Message(role="user", content="x")]


# ==========================================================================
# FileSystemPort
# ==========================================================================

def test_filesystem_passes_the_conformance_kit(tmp_path):
    report = port_conformance.check_filesystem(
        ccoder_host.ATKFileSystem(tmp_path))
    assert report.ok, report.text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_an_edit_keeps_the_files_permission_bits(tmp_path):
    """mkstemp's 0600 used to replace the target's mode on every write."""
    script = tmp_path / "build.sh"
    script.write_text("echo old\n")
    script.chmod(0o755)
    fs = ccoder_host.ATKFileSystem(tmp_path)
    fs.write_bytes("build.sh", b"echo new\n")
    assert script.stat().st_mode & 0o777 == 0o755
    fs.write_bytes("fresh.txt", b"x")
    assert (tmp_path / "fresh.txt").stat().st_mode & 0o777 != 0o600


def test_append_bytes_appends_and_stays_in_the_jail(tmp_path):
    fs = ccoder_host.ATKFileSystem(tmp_path / "proj")
    fs.append_bytes(".cc_journal/s.jsonl", b'{"a": 1}\n')
    fs.append_bytes(".cc_journal/s.jsonl", b'{"b": 2}\n')
    assert fs.read(".cc_journal/s.jsonl") == '{"a": 1}\n{"b": 2}\n'
    with pytest.raises(ValueError):
        fs.append_bytes("../outside.txt", b"x")
    assert not (tmp_path / "outside.txt").exists()


# ==========================================================================
# ExecPort
# ==========================================================================

def test_exec_passes_the_conformance_kit_including_the_tree_kill(tmp_path):
    report = port_conformance.check_exec(ATKExec(), str(tmp_path))
    assert report.ok, report.text()


def test_timeout_zero_waits_instead_of_killing_at_once(tmp_path):
    """Observed: exit -9, "this exceeded 0s", for `print('done')`."""
    result = ATKExec().run([sys.executable, "-c",
                            "import time; time.sleep(0.2); print('done')"],
                           cwd=str(tmp_path), timeout=0)
    assert (result.exit_code, result.timed_out) == (0, False), result.stderr
    assert result.stdout.strip() == "done"


@pytest.mark.parametrize("env", [None, {}])
def test_no_env_means_the_scrubbed_env_not_the_inherited_one(
        tmp_path, monkeypatch, env):
    """Observed: `env={}` let a child read CC_REVIEW_SECRET=sk-leak."""
    monkeypatch.setenv("CC_REVIEW_SECRET", "sk-leak")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    result = ATKExec().run(
        [sys.executable, "-c",
         "import os; print(os.environ.get('CC_REVIEW_SECRET')); "
         "print(bool(os.environ.get('HTTPS_PROXY')))"],
        cwd=str(tmp_path), timeout=30, env=env)
    # The proxy is blanked rather than absent (runner forces it empty);
    # either way the child has no network path through it.
    assert result.stdout.split() == ["None", "False"], result.stdout


def test_an_explicit_env_is_used_as_given(tmp_path):
    env = {"PATH": os.environ.get("PATH", ""), "CC_GIVEN": "yes"}
    result = ATKExec().run(
        [sys.executable, "-c",
         "import os; print(os.environ.get('CC_GIVEN'))"],
        cwd=str(tmp_path), timeout=30, env=env)
    assert result.stdout.strip() == "yes"


# ==========================================================================
# LLMPort
# ==========================================================================

class Engine:
    """The reviewer's stub: ~50 ms of prefill, then ~50 ms per chunk."""

    is_loaded = True

    def __init__(self, chunks=("a ", "b ", "c "), prefill=0.05, per=0.05,
                 model="devstral-x.gguf") -> None:
        self.metadata = {"model_file": model, "n_ctx": 8192}
        self.chunks = list(chunks)
        self.prefill = prefill
        self.per = per
        self.payloads: list = []

    def chat_stream(self, payload, temperature, max_tokens):
        self.payloads.append(payload)
        time.sleep(self.prefill)
        for tok in self.chunks:
            time.sleep(self.per)
            yield tok


def test_prefill_and_decode_are_timed_apart():
    """Observed: prompt_ms was the whole call and decode_ms was 0.

    The boundary is the first chunk, so prompt_ms is time-to-first-token.
    Two engines, one prefill-heavy and one decode-heavy, show that the two
    numbers move independently — which a single wall-clock figure cannot.
    """
    slow_prompt = ATKLLM(Engine(chunks=["x"] * 5, prefill=0.2,
                                per=0.01)).complete(USER)
    slow_decode = ATKLLM(Engine(chunks=["x"] * 5, prefill=0.0,
                                per=0.05)).complete(USER)
    assert slow_prompt.prompt_ms >= 180, slow_prompt
    assert slow_prompt.decode_ms < 150, slow_prompt
    assert slow_decode.prompt_ms < 120, slow_decode
    assert slow_decode.decode_ms >= 180, slow_decode


def test_an_error_mid_stream_is_data_with_both_timings():
    class Broken(Engine):
        def chat_stream(self, payload, temperature, max_tokens):
            time.sleep(0.03)
            yield "partial "
            raise RuntimeError("CUDA out of memory")

    said: list[tuple[str, str]] = []
    c = ATKLLM(Broken(), events=lambda k, m: said.append((k, m))).complete(
        USER)
    assert c.finish_reason == "error" and c.text == "partial "
    assert c.prompt_ms >= 20
    assert any("CUDA out of memory" in m for _, m in said)


def test_a_complete_file_under_the_limit_is_not_called_truncated():
    """Observed: 7,200 characters at max_tokens=2048 → finish "length",
    while the same Completion said tokens_out=1,800."""
    engine = Engine(chunks=["abcd"] * 1800, prefill=0, per=0)
    c = ATKLLM(engine).complete(USER, max_tokens=2048)
    assert c.tokens_out == 1800
    assert c.finish_reason == "stop"


def test_a_reply_that_reaches_the_limit_is_length():
    engine = Engine(chunks=["abcd"] * 2048, prefill=0, per=0)
    c = ATKLLM(engine).complete(USER, max_tokens=2048)
    assert c.finish_reason == "length"


def test_nothing_loaded_is_error_and_a_sentence_once():
    class Unloaded:
        is_loaded = False
        metadata: dict = {}

    said: list[str] = []
    llm = ATKLLM(Unloaded(), events=lambda k, m: said.append(m))
    for _ in range(3):
        c = llm.complete(USER)
        assert (c.finish_reason, c.model) == ("error", "")
    assert len(said) == 1 and "No model is loaded" in said[0]


def test_a_tool_result_stays_where_it_was_in_the_conversation():
    engine = Engine(prefill=0, per=0)
    ATKLLM(engine).complete([
        Message(role="user", content="q1"),
        Message(role="tool", content="answer to q1", tool_call_id="c1"),
        Message(role="user", content="q2")])
    contents = [m["content"] for m in engine.payloads[0]]
    assert contents == ["q1", "[tool result]\nanswer to q1", "q2"]


def _fake_mistral_common(monkeypatch, *, encode_raises=False):
    class Tok:
        def encode(self, text):
            if encode_raises:
                raise RuntimeError("tokenizer exploded")
            return list(range(len(text.split())))

    class MistralTokenizer:
        @staticmethod
        def v3():
            return types.SimpleNamespace(instruct_tokenizer=Tok())

    for name in ("mistral_common", "mistral_common.tokens",
                 "mistral_common.tokens.tokenizers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    leaf = types.ModuleType("mistral_common.tokens.tokenizers.mistral")
    leaf.MistralTokenizer = MistralTokenizer
    monkeypatch.setitem(sys.modules, leaf.__name__, leaf)


def test_the_tokenizer_follows_a_model_swap(monkeypatch):
    """Observed: the choice was frozen at first use, so after a swap to
    Qwen the Mistral vocabulary kept counting — and claimed exactness."""
    _fake_mistral_common(monkeypatch)
    engine = Engine(model="devstral-small.gguf")
    llm = ATKLLM(engine)
    assert llm.count_tokens("one two three") == 3
    assert llm.capabilities().token_count_is_estimate is False

    engine.metadata = {"model_file": "qwen2.5-coder.gguf", "n_ctx": 8192}
    assert llm.count_tokens("one two three") == len("one two three") // 4
    assert llm.capabilities().token_count_is_estimate is True


def test_a_failing_tokenizer_falls_back_and_says_so_once(monkeypatch):
    _fake_mistral_common(monkeypatch, encode_raises=True)
    said: list[str] = []
    llm = ATKLLM(Engine(), events=lambda k, m: said.append(m))
    llm.count_tokens("a b")
    llm.count_tokens("c d")
    assert len(said) == 1 and "tokenizer failed" in said[0]


def test_a_failing_split_think_is_reported_and_the_raw_reply_kept(
        monkeypatch):
    for name in ("atk", "atk.core"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    engine_mod = types.ModuleType("atk.core.llm_engine")

    def split_think(raw):
        raise ValueError("unbalanced [THINK]")

    engine_mod.split_think = split_think
    monkeypatch.setitem(sys.modules, "atk.core.llm_engine", engine_mod)
    said: list[str] = []
    c = ATKLLM(Engine(prefill=0, per=0),
               events=lambda k, m: said.append(m)).complete(USER)
    assert c.text == "a b c "
    assert any("split_think failed" in m for m in said)


def test_the_llm_passes_the_llm_conformance_kit():
    report = port_conformance.check_llm(ATKLLM(Engine(prefill=0, per=0)))
    assert report.ok, report.text()


# ==========================================================================
# StoragePort
# ==========================================================================

class Ctx:
    def __init__(self) -> None:
        self.settings: dict = {}
        self.saves = 0
        self.fail = False

    def save_settings(self) -> None:
        if self.fail:
            raise OSError("disk full")
        self.saves += 1


def test_storage_passes_the_conformance_kit(tmp_path):
    storage = ATKStorage(Ctx(), tmp_path / "data", tmp_path / "proj")
    assert port_conformance.check_storage(storage).ok


def test_two_projects_do_not_share_state_or_databases(tmp_path):
    """Observed: project B saw project A's codemap and transactions."""
    ctx = Ctx()
    a = ATKStorage(ctx, tmp_path / "data", tmp_path / "A")
    b = ATKStorage(ctx, tmp_path / "data", tmp_path / "B")
    a.set("cognitive_coder.patcher.seq", 7)
    assert b.get("cognitive_coder.patcher.seq") is None
    assert a.sqlite_path("codemap") != b.sqlite_path("codemap")
    assert a.key == project_key(tmp_path / "A") and len(a.key) == 12
    assert Path(a.sqlite_path("codemap")).parent.name == a.key


def test_state_survives_into_a_new_storage_for_the_same_project(tmp_path):
    ctx = Ctx()
    first = ATKStorage(ctx, tmp_path / "data", tmp_path / "A")
    first.set("k", {"n": 1})
    first.flush()
    again = ATKStorage(ctx, tmp_path / "data", tmp_path / "A")
    assert again.get("k") == {"n": 1}
    entry = ctx.settings["ccoder"]["projects"][first.key]
    assert entry["root"] == str((tmp_path / "A").resolve())


def test_set_never_saves_settings_flush_saves_once(tmp_path):
    """Observed: save_settings() on EVERY set, from the worker thread."""
    ctx = Ctx()
    storage = ATKStorage(ctx, tmp_path / "data", tmp_path / "A")
    for i in range(5):
        storage.set(f"k{i}", i)
    assert ctx.saves == 0 and "projects" not in ctx.settings.get(
        "ccoder", {}), "the worker thread never touches ctx.settings"
    assert storage.flush() is True and ctx.saves == 1
    assert storage.flush() is True and ctx.saves == 1, "nothing new to save"


def test_a_failed_save_is_reported_and_retried(tmp_path):
    """Observed: save failures were swallowed with `except: pass`."""
    ctx = Ctx()
    said: list[str] = []
    storage = ATKStorage(ctx, tmp_path / "data", tmp_path / "A",
                         report=said.append)
    storage.set("k", 1)
    ctx.fail = True
    assert storage.flush() is False
    assert said and "disk full" in said[0] and "could not save" in said[0]
    ctx.fail = False
    assert storage.flush() is True and ctx.saves == 1


def test_storage_refuses_what_json_cannot_hold(tmp_path):
    storage = ATKStorage(Ctx(), tmp_path / "data", tmp_path / "A")
    with pytest.raises(ValueError, match="JSON-serialisable"):
        storage.set("bad", object())


def test_a_returned_value_is_a_copy(tmp_path):
    storage = ATKStorage(Ctx(), tmp_path / "data", tmp_path / "A")
    storage.set("log", [1])
    storage.get("log").append(2)
    assert storage.get("log") == [1]


# ==========================================================================
# ApprovalPort, and the thread the question is asked on
# ==========================================================================

class FakeGuiThread:
    """A thread that runs posted jobs, standing in for Qt's GUI thread.

    `post` blocks until the job has run — BlockingQueuedConnection's
    contract — and, like it, deadlocks if called from the GUI thread
    itself; here that is an assertion rather than a hang.
    """

    def __init__(self) -> None:
        import queue

        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            job, done = self._jobs.get()
            if job is None:
                return
            job()
            done.set()

    @property
    def ident(self) -> int | None:
        return self._thread.ident

    def is_current(self) -> bool:
        return threading.get_ident() == self._thread.ident

    def post(self, job) -> None:
        assert not self.is_current(), "a blocking post to one's own thread"
        done = threading.Event()
        self._jobs.put((job, done))
        assert done.wait(5), "the GUI thread never ran the job"

    def stop(self) -> None:
        self._jobs.put((None, None))


@pytest.fixture
def gui():
    thread = FakeGuiThread()
    yield thread
    thread.stop()


def _in_worker(fn):
    out: list = []
    worker = threading.Thread(target=lambda: out.append(fn()))
    worker.start()
    worker.join(5)
    return out[0]


def test_both_questions_are_asked_on_the_gui_thread(gui):
    """Observed: the remote QMessageBox was built on the WORKER thread."""
    seen: list[tuple[str, int]] = []
    ask = AskOnGuiThread(on_gui_thread=gui.is_current, post=gui.post)
    approval = ATKApproval(
        ask_diff=ask.for_diff(
            lambda s, d: seen.append(("diff", threading.get_ident()))
            or True),
        ask_remote=ask.for_remote(
            lambda p, n, e: seen.append(("remote", threading.get_ident()))
            or False))
    assert _in_worker(lambda: approval.approve_diff("s", "d")) is True
    assert _in_worker(lambda: approval.approve_remote("x", 1, "e")) is False
    assert seen == [("diff", gui.ident), ("remote", gui.ident)]


def test_a_question_from_the_gui_thread_is_asked_directly(gui):
    """A blocking queued call to one's own thread deadlocks."""
    answers: list[bool] = []
    ask = AskOnGuiThread(on_gui_thread=gui.is_current, post=gui.post)
    approval = ATKApproval(ask_diff=ask.for_diff(lambda s, d: True))
    gui.post(lambda: answers.append(approval.approve_diff("s", "d")))
    assert answers == [True]


def test_a_dialog_that_raises_is_a_no_and_a_sentence(gui):
    said: list[str] = []

    def broken(summary, diff):
        raise RuntimeError("no display")

    ask = AskOnGuiThread(on_gui_thread=gui.is_current, post=gui.post,
                         report=said.append)
    approval = ATKApproval(ask_diff=ask.for_diff(broken))
    assert _in_worker(lambda: approval.approve_diff("s", "d")) is False
    assert said and "no display" in said[0]


def test_approval_defaults():
    assert ATKApproval().approve_diff("s", "d") is False
    assert ATKApproval(auto_apply=True).approve_diff("s", "d") is True
    assert ATKApproval(auto_apply=True).approve_remote(
        "anthropic", 1024, "everything") is False, "C3: never auto-approved"


# ==========================================================================
# the panel's Qt-free pieces
# ==========================================================================

def test_change_log_reads_committed_diffs_and_tolerates_an_open_one():
    """Observed: iterating `patcher._open` raised TypeError on EVERY patch
    event, so the codemap and recommendation never refreshed."""
    fs = MemoryFileSystem()
    patcher = Patcher(fs, MemoryStorage(), AutoApprove())
    session = types.SimpleNamespace(history=patcher.history,
                                    host=types.SimpleNamespace(fs=fs))
    tx = patcher.begin("t1")
    tx.apply([Edit(path="a.py", kind="whole", new="a = 1\n")])
    assert change_log(session) == "", "open: no manifest yet, no crash"
    tx.commit(verified=True)
    open_tx = patcher.begin("t2")
    open_tx.apply([Edit(path="b.py", kind="whole", new="b = 2\n")])
    text = change_log(session)
    assert "+a = 1" in text and "transaction 1" in text
    assert "b = 2" not in text


def test_no_project_is_refused_not_guessed():
    """Observed: the fallback was Path.cwd() — ATK's own source tree."""
    root, why = project_root_for(types.SimpleNamespace(project_root=None))
    assert root is None and "No project folder is set" in why
    root, why = project_root_for(object())
    assert root is None


def test_a_project_holding_atk_itself_is_refused(tmp_path, monkeypatch):
    checkout = tmp_path / "ATK"
    (checkout / "atk").mkdir(parents=True)
    (checkout / "atk" / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(checkout))
    monkeypatch.delitem(sys.modules, "atk", raising=False)
    import importlib
    importlib.invalidate_caches()
    for bad in (checkout, checkout / "atk"):
        root, why = project_root_for(types.SimpleNamespace(project_root=bad))
        assert root is None and "ATK's own source" in why, bad
    fine = tmp_path / "work"
    fine.mkdir()
    root, why = project_root_for(types.SimpleNamespace(project_root=fine))
    assert root == fine.resolve() and why == ""


@pytest.mark.parametrize("detail,expected", [
    ("", "no detail was given — see ATK's log"),
    ("   \n  ", "no detail was given — see ATK's log"),
    ("Traceback…\n  File x\nValueError: bad\n\n", "ValueError: bad"),
])
def test_failure_line_never_raises(detail, expected):
    """Observed: `splitlines()[-1]` → IndexError on an empty detail."""
    assert failure_line(detail) == expected


def test_a_raising_panel_callback_is_logged_not_raised(caplog):
    def broken(kind, text):
        raise RuntimeError("widget deleted")

    events = ccoder_host.ATKEvents(console=broken)
    events.event("status", "hello")
    assert "widget deleted" in caplog.text


def test_build_host_keys_storage_by_project_and_wires_llm_events(tmp_path):
    said: list[str] = []
    host = build_host(Ctx(), types.SimpleNamespace(is_loaded=False,
                                                   metadata={}),
                      tmp_path / "proj", data_dir=tmp_path / "data",
                      status=said.append)
    assert host.storage.key == project_key(tmp_path / "proj")
    host.llm.complete(USER)
    assert any("No model is loaded" in m for m in said)


def test_a_child_that_floods_output_is_bounded_not_buffered(tmp_path):
    """ATKExec kept its own copy of the capture code after the core was
    fixed: `communicate()` buffered every byte, so a generated program
    printing in a loop ran ATK out of memory before its timeout. The
    ceiling kills it even with timeout 0."""
    t0 = time.monotonic()
    result = ATKExec().run(
        [sys.executable, "-c", "import sys\nw = sys.stdout.write\n"
                               "while True: w('y' * 4096 + '\\n')"],
        cwd=str(tmp_path), timeout=0)
    assert time.monotonic() - t0 < 30
    assert result.truncated and len(result.stdout) < 400_000


def test_a_non_utf8_byte_is_a_result_not_a_traceback(tmp_path):
    """The locale codec raised UnicodeDecodeError out of run()."""
    result = ATKExec().run(
        [sys.executable, "-c",
         "import sys; sys.stdout.buffer.write(b'\\xff done\\n')"],
        cwd=str(tmp_path), timeout=30)
    assert result.exit_code == 0 and "done" in result.stdout
