# SPDX-License-Identifier: Apache-2.0
"""Which files matter most (codemap/rank.py) — Aider's repo-map ranking.

The architecture block showed the first 60 files ALPHABETICALLY; Debug &
Repair offered the model 80 files in path order; the audit read the
biggest files first. Each test here fails on that code.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import audit, debugrepair  # noqa: E402
from cognitive_coder.codemap import CodeMap, rank, zoom  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem, MemoryStorage  # noqa: E402


def test_pagerank_favours_what_everything_points_at():
    edges = {("a", "core"): 1.0, ("b", "core"): 1.0, ("c", "core"): 1.0,
             ("core", "util"): 1.0}
    scores = rank.pagerank(["a", "b", "c", "core", "util", "lonely"], edges)
    order = rank.ordered(scores)
    assert order[0] in ("core", "util")
    assert order.index("core") < order.index("a")
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_personalization_pulls_toward_what_the_text_names():
    files = ["src/a.py", "src/physics.py", "src/z.py"]
    defined = {"src/a.py": ["helper"], "src/physics.py": ["apply_drag"],
               "src/z.py": ["other"]}
    pers = rank.personalization(files, defined,
                                "AttributeError in apply_drag")
    scores = rank.pagerank(files, {}, pers)
    assert rank.ordered(scores)[0] == "src/physics.py"


def test_the_order_is_the_same_every_time():
    edges = {("x", "y"): 1.0}
    one = rank.ordered(rank.pagerank(["x", "y", "w", "v"], edges))
    two = rank.ordered(rank.pagerank(["v", "w", "y", "x"], edges))
    assert one == two


def _project(n_leaves: int) -> dict:
    """`zcore.py` is what every leaf imports; it sorts LAST by path."""
    files = {"zcore.py": "def shared(x):\n    return x\n"}
    for i in range(n_leaves):
        files[f"a{i:03d}.py"] = (f"from zcore import shared\n\n\n"
                                 f"def leaf_{i}():\n    return shared({i})\n")
    return {k: v.encode() for k, v in files.items()}


def test_the_architecture_block_keeps_the_most_referenced_files():
    cm = CodeMap(MemoryFileSystem(_project(70)), MemoryStorage())
    cm.index_project()
    block = zoom.architecture_prefix(cm.store, max_files=60)
    assert "zcore.py" in block, "the file everything uses was cut"
    assert "the least referenced" in block
    # printed in path order still, and the same bytes every time
    lines = [ln for ln in block.splitlines() if ln.startswith("  a") or
             ln.startswith("  z")]
    assert lines == sorted(lines)
    assert block == zoom.architecture_prefix(cm.store, max_files=60)


def test_a_small_project_is_listed_exactly_as_before():
    cm = CodeMap(MemoryFileSystem(_project(5)), MemoryStorage())
    cm.index_project()
    block = zoom.architecture_prefix(cm.store, max_files=60)
    assert "more files" not in block


def test_debug_and_repair_offers_the_likeliest_file_first():
    files = _project(5)
    files["zz_physics.py"] = b"def apply_drag(v):\n    return v * 0.9\n"
    cm = CodeMap(MemoryFileSystem(files), MemoryStorage())
    cm.index_project()

    class _S:
        codemap = cm
    order = debugrepair._likely_order(
        _S(), sorted(files), "TypeError: apply_drag() takes 1 argument")
    assert order[0] == "zz_physics.py"


def test_the_audit_reads_the_file_everything_depends_on_before_a_big_one():
    files = [("big.py", "python", "x = 1\n" * 900),
             ("zcore.py", "python", "def shared(x):\n    return x\n")]
    chosen = audit._worth_reading(files, [], 1, ranked=["zcore.py",
                                                         "big.py"])
    assert chosen[0][0] == "zcore.py"
    # with no ranking, size decides as before
    assert audit._worth_reading(files, [], 1)[0][0] == "big.py"
