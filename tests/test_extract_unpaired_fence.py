# SPDX-License-Identifier: Apache-2.0
"""A reply with ONE fence line — no partner — must not reach the disk.

Oct 2, 2026, Devstral-Small-2-24B (Q4_K_M), the racing spec, both test
files. The model wrote each file with no opening fence and then closed one.
`_FENCE` needs a pair and found none, the whole reply failed to parse, and
the fallback returned the whole reply. tests/test_math3d.py and
tests/test_physics.py were written with a trailing "```" on their last line,
and neither could be imported. (What made that fatal — the syntax error
being blamed on the module under test — is test_repair_routing's business.)

REPLY below is that model's reply for tests/test_math3d.py, verbatim.
"""

from __future__ import annotations

import ast

from cognitive_coder import patcher
from cognitive_coder.loop import _extract

REPLY = '''import unittest
from src.math3d import project_segment, get_visible_segments

class TestMath3D(unittest.TestCase):
    def test_project_segment(self):
        # Test projection of a segment at known distance
        segment_x, segment_y, segment_z = 0.0, 0.0, 10.0
        camera_x, camera_y, camera_z = 0.0, 0.0, 0.0
        screen_width, screen_height = 800, 600

        # Expected scale is 1/10 = 0.1
        # Expected Y coordinate should be 0 (since segment_y is 0)
        X, Y, W = project_segment(segment_x, segment_y, segment_z,
                                 camera_x, camera_y, camera_z,
                                 screen_width, screen_height)
        self.assertAlmostEqual(W, 0.1, places=7)
        self.assertAlmostEqual(Y, 300.0, places=7)  # (0 * 0.1) * (600/2) + 300 = 300

    def test_get_visible_segments(self):
        # Test that only segments within max_distance are returned
        segments = [
            (0.0, 0.0, 5.0),    # Within distance
            (0.0, 0.0, 150.0),  # Within distance
            (0.0, 0.0, 350.0),  # Beyond distance
            (0.0, 0.0, 400.0)   # Beyond distance
        ]
        camera_x, camera_y, camera_z = 0.0, 0.0, 0.0
        max_distance = 300.0

        visible = get_visible_segments(segments, camera_x, camera_y, camera_z, max_distance)
        self.assertEqual(len(visible), 2)
        self.assertEqual(visible, segments[:2])

    def test_project_segment_zero_depth(self):
        # Test that division by zero is avoided when segment is at same depth as camera
        segment_x, segment_y, segment_z = 0.0, 0.0, 0.0
        camera_x, camera_y, camera_z = 0.0, 0.0, 0.0
        screen_width, screen_height = 800, 600

        X, Y, W = project_segment(segment_x, segment_y, segment_z,
                                 camera_x, camera_y, camera_z,
                                 screen_width, screen_height)
        self.assertEqual(W, 10.0)  # 1/0.1 = 10 (since max(0, 0.1) is 0.1)

if __name__ == '__main__':
    unittest.main()
```
'''


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def test_the_oct_2_reply_is_written_without_its_closing_fence():
    code = _extract(REPLY, "python", "tests/test_math3d.py")
    assert "```" not in code
    assert _parses(code)
    assert code.startswith("import unittest")
    assert code.rstrip().endswith("unittest.main()")


def test_without_a_validator_the_code_before_a_bare_closer_wins():
    # Languages with no in-process parser get a non-empty check only, so the
    # ORDER of the candidates is what decides. Code, then a bare closer.
    reply = "fn main() {\n    println!(\"hi\");\n}\n```\n"
    code = patcher.extract_code(reply, "rust", path="src/main.rs")
    assert code == "fn main() {\n    println!(\"hi\");\n}"


def test_a_tagged_opener_with_no_closer_keeps_what_follows_it():
    reply = "Here is the file:\n```python\ndef f():\n    return 1\n"
    code = _extract(reply, "python", "src/f.py")
    assert code == "def f():\n    return 1"


def test_the_file_then_the_file_again_in_a_fence_is_one_copy():
    # main.py, same run: the file unfenced, then a full ```python fence
    # holding it again. The pair rule must win and the file appear once.
    body = "import os\n\n\ndef main():\n    return os.sep\n"
    reply = body + "```python\n" + body + "```\n"
    code = _extract(reply, "python", "src/main.py")
    assert code == body.strip("\n")
    assert code.count("def main") == 1


def test_a_balanced_markdown_file_keeps_its_last_fence():
    # An EVEN number of fence lines is not "unpaired": a README that ends in
    # a code block must keep the fence that closes it.
    readme = "# Tool\n\nRun it:\n\n```\nccoder build\n```\n"
    assert patcher._unpaired_fence_candidates(readme) == []


def test_a_lone_trailing_fence_after_several_is_dropped():
    reply = ("x = 1\n```python\ny = 2\n```\nz = 3\n```\n")
    cands = patcher._unpaired_fence_candidates(reply)
    assert cands and not cands[0].rstrip().endswith("```")
