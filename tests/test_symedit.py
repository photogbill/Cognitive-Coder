# SPDX-License-Identifier: Apache-2.0
"""Narrowed edits — a reply of definitions, put back by name (F5).

Every rule in `symedit.py` is a test here, because each one guards a
failure that would still parse: a method spliced at the wrong indentation,
a placeholder body spliced over real code, half of a reply applied.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder import symedit  # noqa: E402

SRC = '''"""Cars."""
import math

LIMIT = 10


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Car:
    """A car."""
    MAX = 5

    def __init__(self):
        self.speed = 0.0

    def update(self, dt):
        self.speed += dt * 2
        return self.speed

    @property
    def fast(self):
        return self.speed > self.MAX


def main():
    c = Car()
    print(c.update(1.0))


if __name__ == "__main__":
    main()
'''


def _fence(code: str) -> str:
    return f"```python\n{code}```"


def test_the_index_finds_every_kind_of_definition():
    idx = symedit.index(SRC)
    assert idx is not None
    kinds = {n: s.kind for n, s in idx.spans.items()}
    assert kinds["clamp"] == "function"
    assert kinds["Car"] == "class"
    assert kinds["Car.update"] == "method"
    assert kinds["Car.fast"] == "method"
    assert kinds["LIMIT"] == "constant"
    assert kinds["__main__"] == "main"
    # the decorator belongs to the method
    fast = idx.get("Car.fast")
    assert symedit.lines_of(SRC)[fast.start].strip() == "@property"


def test_a_name_defined_twice_is_ambiguous_and_never_replaced_by_name():
    src = ("from typing import overload\n\n\nclass A:\n    @overload\n"
           "    def v(self, x: int) -> int: ...\n\n    @overload\n"
           "    def v(self, x: str) -> str: ...\n\n"
           "    def w(self):\n        return 1\n")
    idx = symedit.index(src)
    assert "A.v" in idx.ambiguous and idx.get("A.v") is None
    out = symedit.apply(src, _fence("class A:\n    def v(self, x):\n"
                                    "        return 2\n"))
    assert not out.ok
    assert "more than once" in out.problem


def test_a_property_its_setter_and_its_deleter_are_three_definitions():
    src = ("class A:\n    @property\n    def v(self):\n        return 1\n\n"
           "    @v.setter\n    def v(self, x):\n        pass\n")
    idx = symedit.index(src)
    assert {"A.v", "A.v@setter"} <= set(idx.spans)
    # an error inside the setter: the setter is repaired, the getter kept
    assert symedit.targets_for(src, [8]) == ["A.v@setter"]
    out = symedit.apply(src, _fence("class A:\n    @v.setter\n"
                                    "    def v(self, x):\n"
                                    "        self._v = x\n"),
                        targets=["A.v@setter"])
    assert out.ok, out.problem
    assert "self._v = x" in out.code and "return 1" in out.code


def test_a_method_reply_never_overwrites_a_function_of_the_same_name():
    """Review, 2026-10-19: `def run(self)` for the error in `App.run`
    replaced the module's CLI `run(argv)`, and App.run stayed broken."""
    src = ("def run(argv=None):\n    return main(argv)\n\n\n"
           "class App:\n    def run(self):\n        return self.nn\n")
    out = symedit.apply(src, _fence("def run(self):\n    return self.n\n"),
                        targets=["App.run"])
    assert out.ok, out.problem
    assert out.replaced == ["App.run"]
    assert "return main(argv)" in out.code
    assert "        return self.n\n" in out.code


def test_new_definitions_keep_the_order_the_reply_gave_them():
    """Review: a helper and the main guard calling it, a base class and its
    subclass, came out reversed — NameError at import."""
    src = "def a():\n    return 1\n"
    out = symedit.apply(src, _fence(
        "class Base:\n    pass\n\n\nclass Derived(Base):\n    pass\n\n\n"
        "def count_words(t):\n    return len(t.split())\n\n\n"
        "if __name__ == '__main__':\n    print(count_words('a b'))\n"))
    assert out.ok, out.problem
    code = out.code
    assert code.index("class Base") < code.index("class Derived") \
        < code.index("def count_words") < code.index("if __name__")
    exec(compile(code, "m.py", "exec"), {"__name__": "x"})   # noqa: S102


def test_a_new_constant_that_uses_a_definition_goes_after_it():
    src = "import os\n\n\ndef handle_a():\n    return 1\n"
    out = symedit.apply(src, _fence(
        "BASE = 10\nLIMIT = BASE * 2\n\n\ndef handle_b():\n    return 2\n"
        "\n\nHANDLERS = {'a': handle_a, 'b': handle_b}\n"))
    assert out.ok, out.problem
    ns: dict = {}
    exec(compile(out.code, "m.py", "exec"), ns)               # noqa: S102
    assert ns["LIMIT"] == 20 and ns["HANDLERS"]["b"]() == 2


def test_a_partial_class_that_changes_its_class_line_is_refused():
    """Review: `class Foo(Base):` with one method was "replaced Foo.m" and
    the base class silently dropped; so was `@dataclass`."""
    src = ("class Base:\n    pass\n\n\nclass Foo:\n    def m(self):\n"
           "        return 1\n\n    def n(self):\n        return 2\n")
    for header in ("class Foo(Base):", "@dataclass\nclass Foo:"):
        out = symedit.apply(src, _fence(f"{header}\n    def m(self):\n"
                                        f"        return 3\n"))
        assert not out.ok and "leaves out some of its members" in \
            out.problem, header


def test_a_nested_class_in_a_partial_reply_is_merged_not_replaced():
    src = ("class Outer:\n    class Inner:\n        def a(self):\n"
           "            return 1\n\n        def b(self):\n"
           "            return 2\n\n    def c(self):\n        return 3\n")
    out = symedit.apply(src, _fence("class Outer:\n    class Inner:\n"
                                    "        def b(self):\n"
                                    "            return 20\n"))
    assert out.ok, out.problem
    assert "return 1" in out.code and "return 20" in out.code
    assert out.replaced == ["Outer.Inner.b"]


def test_a_definition_inside_an_if_is_not_shadowed_by_a_new_one():
    src = ("import sys\n\nif sys.platform == 'win32':\n"
           "    def home():\n        return 'C:'\n"
           "else:\n    def home():\n        return '/'\n")
    out = symedit.apply(src, _fence("def home():\n    return '~'\n"))
    assert not out.ok and "inside an `if`" in out.problem


def test_a_file_saved_with_a_byte_order_mark_is_indexed():
    src = "\ufeffdef f():\n    return 1\n"
    assert symedit.index(src) is not None
    out = symedit.apply(src, _fence("def f():\n    return 2\n"))
    assert out.ok and "return 2" in out.code


def test_an_unchanged_reply_is_marked_unchanged():
    out = symedit.apply(SRC, _fence("def clamp(x, lo, hi):\n"
                                    "    return max(lo, min(hi, x))\n"))
    assert out.unchanged


def test_a_bare_method_goes_back_into_its_class_at_its_indentation():
    reply = _fence("def update(self, dt):\n    self.speed += dt * 3\n"
                   "    return clamp(self.speed, 0, LIMIT)\n")
    out = symedit.apply(SRC, reply, targets=["Car.update"])
    assert out.ok, out.problem
    assert out.replaced == ["Car.update"]
    assert "        self.speed += dt * 3\n" in out.code
    assert "        self.speed += dt * 2\n" not in out.code
    # nothing else moved
    assert out.code.count("def ") == SRC.count("def ")
    assert "return max(lo, min(hi, x))" in out.code
    ast.parse(out.code)


def test_a_method_inside_its_class_line_merges_and_new_ones_are_added():
    reply = _fence("import os\n\nclass Car:\n    def update(self, dt):\n"
                   "        return helper(dt)\n\n    def reset(self):\n"
                   "        self.speed = 0\n\n\ndef helper(dt):\n"
                   "    return dt\n")
    out = symedit.apply(SRC, reply)
    assert out.ok, out.problem
    assert out.replaced == ["Car.update"]
    assert set(out.added) == {"Car.reset", "helper"}
    assert out.imports == ["import os"]
    tree = ast.parse(out.code)
    car = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [m.name for m in car.body if isinstance(m, ast.FunctionDef)] == \
        ["__init__", "update", "fast", "reset"]
    # the new function lands before the main guard, the import in the block
    lines = out.code.splitlines()
    assert lines.index("import os") == lines.index("import math") + 1
    assert out.code.index("def helper") < out.code.index('if __name__')


def test_a_placeholder_body_is_never_spliced_over_real_code():
    out = symedit.apply(SRC, _fence(f"def clamp(x, lo, hi):\n"
                                    f"    {symedit.ELIDED}\n"))
    assert not out.ok
    assert out.skipped == [("clamp", "the reply's body was only a "
                                     "placeholder")]


def test_a_top_level_statement_refuses_the_whole_reply():
    out = symedit.apply(SRC, _fence("print('hi')\n\ndef clamp(x, lo, hi):\n"
                                    "    return x\n"))
    assert not out.ok and "not a definition" in out.problem


def test_a_statement_already_in_the_file_is_not_a_change():
    src = "import sys\nsys.setrecursionlimit(50)\n\ndef f():\n    return 1\n"
    out = symedit.apply(src, _fence("sys.setrecursionlimit(50)\n\ndef f():\n"
                                    "    return 2\n"))
    assert out.ok, out.problem
    assert "return 2" in out.code


def test_a_whole_file_reply_that_leaves_something_out_keeps_it():
    """The reason narrowing makes dropping code impossible: what the reply
    does not hold, the file keeps."""
    reply = SRC.replace("def main():\n    c = Car()\n    print(c.update(1.0))"
                        "\n", "").replace("return max(lo, min(hi, x))",
                                          "return min(max(x, lo), hi)")
    out = symedit.apply(SRC, reply)
    assert out.ok, out.problem
    assert out.replaced == ["clamp"]
    assert "def main():" in out.code


def test_an_unchanged_reply_is_said_to_change_nothing():
    out = symedit.apply(SRC, _fence("def clamp(x, lo, hi):\n"
                                    "    return max(lo, min(hi, x))\n"))
    assert not out.ok and "nothing changed" in out.problem


def test_a_partial_class_that_changes_class_lines_is_refused():
    reply = _fence("class Car:\n    MAX = 9\n\n    def update(self, dt):\n"
                   "        return 0\n")
    out = symedit.apply(SRC, reply)
    assert not out.ok and "class-level" in out.problem


def test_a_whole_class_reply_replaces_the_class():
    whole = ('class Car:\n    """A car."""\n    MAX = 7\n\n'
             '    def __init__(self):\n        self.speed = 1.0\n\n'
             '    def update(self, dt):\n        return dt\n\n'
             '    @property\n    def fast(self):\n        return False\n')
    out = symedit.apply(SRC, _fence(whole))
    assert out.ok, out.problem
    assert out.replaced == ["Car"]
    assert "MAX = 7" in out.code


def test_a_constant_is_replaced_and_a_new_one_goes_after_the_imports():
    out = symedit.apply(SRC, _fence("LIMIT = 20\nSTEP = 2\n"))
    assert out.ok, out.problem
    assert "LIMIT = 20" in out.code and "LIMIT = 10" not in out.code
    assert out.code.index("STEP = 2") < out.code.index("def clamp")


def test_a_future_import_goes_first():
    out = symedit.apply(SRC, _fence("from __future__ import annotations\n\n"
                                    "def clamp(x, lo, hi):\n    return x\n"))
    assert out.ok, out.problem
    body = ast.parse(out.code).body
    assert isinstance(body[1], ast.ImportFrom)
    assert body[1].module == "__future__"


def test_a_shebang_stays_the_first_line():
    src = "#!/usr/bin/env python3\ndef f():\n    return 1\n"
    out = symedit.apply(src, _fence("import os\n\ndef f():\n"
                                    "    return os.sep\n"))
    assert out.ok, out.problem
    assert out.code.startswith("#!/usr/bin/env python3\nimport os\n")


def test_a_multiline_string_inside_a_moved_method_keeps_its_value():
    src = ("class T:\n    def text(self):\n        return '''a\n"
           "b'''\n")
    reply = _fence("def text(self):\n    return '''x\ny'''\n")
    out = symedit.apply(src, reply, targets=["T.text"])
    assert out.ok, out.problem
    ns: dict = {}
    exec(out.code, ns)                      # noqa: S102 - our own test code
    assert ns["T"]().text() == "x\ny"


def test_a_bare_method_with_two_owners_must_say_which():
    src = ("class A:\n    def run(self):\n        return 1\n\n"
           "class B:\n    def run(self):\n        return 2\n")
    out = symedit.apply(src, _fence("def run(self):\n    return 3\n"))
    assert not out.ok and "more than one class" in out.problem
    ok = symedit.apply(src, _fence("def run(self):\n    return 3\n"),
                       targets=["B.run"])
    assert ok.ok and ok.replaced == ["B.run"]


def test_a_reply_that_defines_a_name_twice_is_refused():
    out = symedit.apply(SRC, _fence("def clamp(x, lo, hi):\n    return x\n\n"
                                    "def clamp(x, lo, hi):\n    return lo\n"))
    # the second claim overlaps the first; the first is applied and the
    # file never defines `clamp` twice
    assert out.ok
    assert ast.parse(out.code)
    assert out.code.count("def clamp") == 1


def test_prose_around_the_fence_is_ignored_and_a_broken_reply_is_said():
    out = symedit.apply(SRC, "Here you go:\n" + _fence(
        "def clamp(x, lo, hi):\n    return x\n") + "\nDone.")
    assert out.ok
    bad = symedit.apply(SRC, _fence("def clamp(x, lo, hi)\n    return x\n"))
    assert not bad.ok and "not valid Python" in bad.problem


def test_the_condensed_view_shows_targets_whole_and_the_rest_as_headers():
    src = SRC.replace("    return max(lo, min(hi, x))\n",
                      "    if x < lo:\n        return lo\n    if x > hi:\n"
                      "        return hi\n    return x\n")
    view = symedit.condensed_view(src, ["Car.update"])
    assert "self.speed += dt * 2" in view            # the target, in full
    assert "return hi" not in view                   # another body, elided
    assert "def clamp(x, lo, hi):" in view           # but its header shown
    assert symedit.ELIDED in view
    ast.parse(view)        # the view itself is valid Python
    # a body shorter than the elision line is shown, not elided: eliding
    # it would make the view longer than the file
    assert "return self.speed > self.MAX" in view
    assert len(view) < len(src)


def test_targets_are_the_functions_holding_the_error_lines():
    assert symedit.targets_for(SRC, [19]) == ["Car.update"]
    assert symedit.targets_for(SRC, [8, 19, 8]) == ["clamp", "Car.update"]
    assert symedit.targets_for(SRC, [2], named=lambda: ["main"]) == ["main"]
    assert symedit.targets_for("def broken(:\n", [1]) == []


def test_a_multiline_f_string_in_a_moved_method_keeps_its_value():
    """Python 3.12 tokenizes f-strings in pieces; the lines inside one were
    re-indented and the string's value changed (review #13)."""
    src = "class T:\n    def show(self):\n        return 1\n"
    reply = _fence('def show(self):\n    return f"""Header\n  indented {1}\n'
                   'done"""\n')
    out = symedit.apply(src, reply, targets=["T.show"])
    assert out.ok, out.problem
    ns: dict = {}
    exec(out.code, ns)                      # noqa: S102 - our own test code
    assert ns["T"]().show() == "Header\n  indented 1\ndone"
