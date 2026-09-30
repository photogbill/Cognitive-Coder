# SPDX-License-Identifier: Apache-2.0
"""The pattern-matched outline (item 16): right names, right lines.

Every expectation here is a case the reviewers ran against the old
patterns: a multi-line C signature reported at line 1 as `static int`
(the `[\\w*\\s]+?` prefix spanned newlines — the very mistake the module
docstring says it learned from); commented-out and `#if 0` functions
indexed as live; no C structs; a C++ constructor swallowing the next
method; Rust `impl … for`, `const fn` and `extern "C" fn` missed; JS
arrows and class methods missed; `synchronized (lock)` reported as a Java
method; and all of it printed under "exact signatures".
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.codemap import CodeMap, zoom  # noqa: E402
from cognitive_coder.codemap import parse_regex as rx  # noqa: E402
from cognitive_coder.ports import MemoryFileSystem, MemoryStorage  # noqa: E402


def _outline(lang, text):
    syms, edges, unresolved = rx.parse(text, f"f.{lang}", lang)
    syms = [s for s in syms if s.kind != "module"]
    return {s.name: s for s in syms}, edges, unresolved


def test_a_multi_line_c_signature_is_reported_where_the_name_is():
    syms, _e, _u = _outline(
        "c", "static int\nhelper(int a,\n       int b)\n{\n"
             "    return a + b;\n}\n")
    assert list(syms) == ["helper"]
    assert syms["helper"].line == 2
    assert "helper(int a, int b)" in syms["helper"].signature
    assert syms["helper"].signature.startswith("static int")


def test_c_structs_unions_enums_and_typedefs():
    syms, _e, _u = _outline(
        "c", "struct point {\n    int x;\n};\n"
             "typedef struct { int y; } pt_t;\nenum color { RED };\n")
    assert {"point", "pt_t", "color"} <= set(syms)


def test_commented_out_and_if0_code_is_not_live():
    for text in ("/*\nint old_fn(int a) {\n    return a;\n}\n*/\n"
                 "int live(void) { return 1; }\n",
                 "#if 0\nint dead(int a) { return a; }\n#endif\n"
                 "int live(void) { return 1; }\n",
                 "// int gone(void) { return 0; }\n"
                 "int live(void) { return 1; }\n"):
        syms, _e, _u = _outline("c", text)
        assert list(syms) == ["live"], text


def test_strings_and_comments_hold_no_declarations_or_calls():
    syms, _e, _u = _outline(
        "c", 'int main(void) {\n    const char *s = "function foo(";\n'
             "    if (s) {\n    }\n    return 0;\n}\n")
    assert list(syms) == ["main"]
    _s, _e, unresolved = _outline(
        "c", 'int main(void) {\n    printf("call foo(\\n");\n'
             "    /* bar(1); */\n    return 0;\n}\n")
    assert not {u[1] for u in unresolved} & {"foo", "bar"}


def test_a_brace_in_a_string_does_not_stretch_a_body():
    syms, edges, _u = _outline(
        "c", 'int one(void) {\n    const char *s = "{";\n    return 1;\n}\n'
             "int two(void) {\n    return one();\n}\n")
    assert (syms["one"].line, syms["one"].end_line) == (1, 4)
    assert (syms["two"].line, syms["two"].end_line) == (5, 7)
    assert ("two", "one", "calls") in edges


def test_cpp_template_constructor_and_methods():
    syms, _e, _u = _outline(
        "cpp", "template <typename T>\nT max_of(T a, T b) {\n"
               "    if constexpr (std::is_integral_v<T>) {\n"
               "        return a > b ? a : b;\n    }\n    return a;\n}\n")
    assert list(syms) == ["max_of"]
    assert syms["max_of"].line == 2
    syms, _e, _u = _outline(
        "cpp", "class Foo : public Bar {\npublic:\n"
               "    Foo(int a) : x(a) {}\n    void run() const {\n"
               "        helper();\n    }\n};\n")
    assert syms["run"].line == 4
    ctor = [s for s in rx.parse(
        "class Foo : public Bar {\npublic:\n    Foo(int a) : x(a) {}\n"
        "    void run() const {\n        helper();\n    }\n};\n",
        "f.cpp", "cpp")[0] if s.name == "Foo" and s.kind == "function"]
    assert ctor and ctor[0].line == 3 and ctor[0].end_line == 3


def test_rust_impl_const_extern_fns():
    syms, _e, _u = _outline(
        "rust", "impl<T: Clone> Trait for Foo<T> {\n"
                "    pub async fn run(&self) {}\n"
                "    pub const fn size() -> usize { 0 }\n"
                "    pub(crate) unsafe fn raw() {}\n}\n"
                'extern "C" fn cfn() {}\nimpl Display for Foo {}\n')
    assert {"Foo", "run", "size", "raw", "cfn"} <= set(syms)
    assert "Display" not in syms


def test_javascript_arrows_exports_and_class_methods():
    syms, _e, _u = _outline(
        "javascript",
        "const add = (a, b) => a + b;\nconst inc = a => a + 1;\n"
        "export const fetchAll = async (u) => {};\n"
        "module.exports.helper = function (x) { return x; };\n"
        "class Box {\n  constructor(v) { this.v = v; }\n"
        "  get() { return this.v; }\n  static of(v) { return new Box(v); }\n"
        "}\n")
    assert {"add", "inc", "fetchAll", "helper", "Box", "constructor", "get",
            "of"} <= set(syms)


def test_typescript_types_and_generic_methods():
    syms, _e, _u = _outline(
        "typescript",
        "export interface Shape { area(): number }\n"
        "export type Pair<T> = [T, T];\nexport class Box<T> {\n"
        "  private v: T;\n"
        "  map<U>(f: (t: T) => U): Box<U> { return new Box(f(this.v)); }\n"
        "}\nexport function id<T>(x: T): T { return x; }\n"
        "export enum Color { Red }\n")
    assert {"Shape", "Pair", "Box", "map", "id", "Color"} <= set(syms)


def test_go_methods_carry_their_receiver():
    syms, edges, _u = _outline(
        "go", "package main\n\nfunc (s *Server) Start() error {\n"
              "\treturn s.helper()\n}\n\n"
              "func (s *Server) helper() error { return nil }\n\n"
              "type Server struct{}\ntype Handler interface{ Serve() }\n")
    assert {"Server.Start", "Server.helper", "Server", "Handler"} <= set(syms)
    assert ("Server.Start", "Server.helper", "calls") in edges


def test_java_blocks_and_annotations_are_not_methods_or_calls():
    syms, _e, unresolved = _outline(
        "java", "public class Svc {\n"
                "    public <T> List<T> map(Function<T> f, "
                "@Named(\"x\") int n) {\n"
                "        synchronized (lock) {\n            return null;\n"
                "        }\n    }\n    @Override\n"
                "    public void run() {}\n}\n")
    assert {"Svc", "map", "run"} <= set(syms)
    assert "synchronized" not in syms
    assert "Named" not in {u[1] for u in unresolved}


def test_pattern_matched_interfaces_are_not_called_exact():
    """`dependency_interfaces` printed regex rows under "exact signatures —
    use these, do not guess" with no marker at all."""
    cm = CodeMap(MemoryFileSystem({
        "lib/util.c": b"static int\nhelper(int a,\n       int b)\n{\n"
                      b"    return a + b;\n}\n",
        "lib/util.h": b"int helper(int a, int b);\n",
        "src/app.c": b'#include "util.h"\n#include "../lib/util.c"\n'
                     b"int main(void) {\n    return helper(1, 2);\n}\n"}),
        MemoryStorage(), use_treesitter=False)
    cm.index_project()
    text = zoom.dependency_interfaces(cm.store, "src/app.c")
    assert "~approx" in text
    assert "exact signatures" not in text.splitlines()[0]
