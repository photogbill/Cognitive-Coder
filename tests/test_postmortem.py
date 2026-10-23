# SPDX-License-Identifier: Apache-2.0
"""`ccoder postmortem` — the journals read back, ranked by cost."""

import json

from cognitive_coder import postmortem
from cognitive_coder.ports import MemoryFileSystem


def _ev(event, t, **kw):
    row = {"event": event, "session": "cc-x", "t": t}
    row.update(kw)
    return json.dumps(row)


def _fs(rows, debug_rows=()):
    fs = MemoryFileSystem()
    fs.write(".cc_journal/cc-x.jsonl", "\n".join(rows) + "\n")
    if debug_rows:
        fs.write(".cc_state/debug/history.jsonl",
                 "\n".join(json.dumps(r) for r in debug_rows) + "\n")
    return fs


def test_a_same_size_second_attempt_is_found_and_costed():
    rows = [
        _ev("session_start", "2026-10-22T10:00:00Z", model="Devstral", data={}),
        _ev("generate", "2026-10-22T10:00:10Z", task="src/a.py", attempt="1",
            tokens_out=2696, decode_ms=600000, model="Devstral"),
        _ev("generate", "2026-10-22T10:10:10Z", task="src/a.py", attempt="2",
            tokens_out=2696, decode_ms=600000, model="Devstral"),
        _ev("verify", "2026-10-22T10:20:00Z", task="src/a.py",
            verify={"ok": False}),
        _ev("session_end", "2026-10-22T10:20:10Z",
            data={"ok": False, "failed": ["src/a.py"], "untested": []}),
    ]
    rep = postmortem.analyse(_fs(rows))
    kinds = {f.kind: f for f in rep.findings}
    assert "same-size second attempt" in kinds
    assert kinds["same-size second attempt"].minutes == 10.0
    assert "narrowed edit" in kinds["same-size second attempt"].fix
    assert "slow model" in kinds            # 2696 tokens / 600 s = 4.5 tok/s
    assert rep.sessions[0].slowest_tok_s == 4.5
    text = postmortem.render(rep)
    assert "FINDINGS" in text and "[same-size second attempt] 10.0 min" in text


def test_ended_ok_with_nothing_tested_is_named():
    rows = [
        _ev("session_start", "2026-10-22T10:00:00Z", model="M", data={}),
        _ev("verify", "2026-10-22T10:01:00Z", task="src/a.py",
            verify={"ok": True}),
        _ev("session_end", "2026-10-22T10:02:00Z",
            data={"ok": True, "failed": [], "untested": ["src/a.py"]}),
    ]
    rep = postmortem.analyse(_fs(rows))
    assert any(f.kind == "no tests ran" for f in rep.findings)
    assert "ENDED OK WITH NOTHING TESTED" in postmortem.render(rep)


def test_debug_sessions_located_and_non_errors():
    rows = [
        _ev("session_start", "2026-10-22T10:00:00Z", model="M",
            data={"error_text": "ImportError: attempted relative import "
                                "with no known parent package"}),
        _ev("debug", "2026-10-22T10:00:30Z",
            data={"state": "located", "rounds": 0,
                  "targets": ["src/main.py"]}),
        _ev("session_end", "2026-10-22T10:00:40Z", data={"ok": False}),
    ]
    rep = postmortem.analyse(_fs(rows, debug_rows=[{"state": "located"},
                                                    {"state": "repaired"}]))
    kinds = {f.kind for f in rep.findings}
    assert {"located, not repaired", "not a code error"} <= kinds
    assert rep.debug_states == {"located": 1, "repaired": 1}
    assert rep.sessions[0].mode == "debug"


def test_a_diffusion_session_is_not_called_a_slow_model():
    rows = [
        _ev("session_start", "2026-10-22T10:00:00Z", model="DGemma", data={}),
        _ev("canvas", "2026-10-22T10:00:01Z", task="src/a.py", attempt="1",
            data={"kind": "skip", "reason": "the failure names no line"}),
        _ev("generate", "2026-10-22T10:00:10Z", task="src/a.py", attempt="1",
            tokens_out=1000, decode_ms=200000, model="DGemma"),
        _ev("session_end", "2026-10-22T10:05:00Z", data={"ok": False}),
    ]
    rep = postmortem.analyse(_fs(rows))
    kinds = {f.kind for f in rep.findings}
    assert "slow model" not in kinds
    assert "canvas skipped" in kinds


def test_no_journals_is_an_empty_report():
    rep = postmortem.analyse(MemoryFileSystem())
    assert rep.sessions == [] and rep.findings == []
