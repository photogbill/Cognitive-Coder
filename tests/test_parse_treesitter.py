# SPDX-License-Identifier: Apache-2.0
"""tree-sitter: an upgrade when it works, honest when it does not (item 18).

Observed with today's packages: `pip install cognitive-coder[treesitter]`
installs tree-sitter 0.26 and tree-sitter-languages 1.10, whose
`get_parser` raises TypeError against that tree-sitter; the fallback needs
`tree_sitter_<lang>` packages the extra does not list; so `available("c")`
was False and `degraded_note` said "tree-sitter is not installed" on a
machine where it was. These tests need no tree-sitter at all: they stand
in fake modules to reproduce each state.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cognitive_coder.codemap import parse_treesitter as ts  # noqa: E402


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(ts, "_cache", {})
    monkeypatch.setattr(ts, "_why", {}, raising=False)


def _fake_modules(monkeypatch, *, tree_sitter=True):
    languages = types.ModuleType("tree_sitter_languages")

    def get_parser(name):
        raise TypeError("__init__() takes exactly 1 argument (2 given)")

    languages.get_parser = get_parser
    monkeypatch.setitem(sys.modules, "tree_sitter_languages", languages)
    if tree_sitter:
        core = types.ModuleType("tree_sitter")
        core.Language = lambda ptr: object()
        core.Parser = lambda language: object()
        monkeypatch.setitem(sys.modules, "tree_sitter", core)
    else:
        monkeypatch.setitem(sys.modules, "tree_sitter", None)
    monkeypatch.setitem(sys.modules, "tree_sitter_c", None)


def test_an_incompatible_install_falls_back_and_says_so(monkeypatch,
                                                        fresh_cache):
    _fake_modules(monkeypatch, tree_sitter=True)
    assert not ts.available("c")
    note = ts.degraded_note("c")
    assert "not installed" not in note
    assert "installed" in note and "tree-sitter-c" in note
    syms, _e, _u = ts.parse("int f(void) { return 0; }\n", "f.c", "c")
    assert [s.name for s in syms if s.kind != "module"] == ["f"]
    assert all(s.approximate for s in syms)


def test_a_missing_install_says_not_installed(monkeypatch, fresh_cache):
    _fake_modules(monkeypatch, tree_sitter=False)
    assert not ts.available("c")
    assert "not installed" in ts.degraded_note("c")


def test_a_walk_that_breaks_falls_back_to_the_regex_outline(monkeypatch,
                                                            fresh_cache):
    class Broken:
        def parse(self, data):
            class Tree:
                @property
                def root_node(self):
                    raise AttributeError("API changed")
            return Tree()

    ts._cache["c"] = Broken()
    syms, _e, _u = ts.parse("int f(void) { return 0; }\n", "f.c", "c")
    assert [s.name for s in syms if s.kind != "module"] == ["f"]


_REAL = all(importlib.util.find_spec(m) is not None
            for m in ("tree_sitter", "tree_sitter_rust", "tree_sitter_go",
                      "tree_sitter_typescript", "tree_sitter_java"))


@pytest.mark.skipif(not _REAL, reason="tree-sitter grammars not installed")
def test_real_grammars_find_impls_types_and_calls(fresh_cache):
    syms, _e, _u = ts.parse(
        "impl<T: Clone> Trait for Foo<T> {\n    pub fn run(&self) {}\n}\n",
        "f.rs", "rust")
    assert {"Foo", "Foo.run"} <= {s.name for s in syms}
    syms, edges, _u = ts.parse(
        "package main\nfunc (s *Server) Start() error { return s.helper() }"
        "\nfunc (s *Server) helper() error { return nil }\n"
        "type Server struct{}\n", "f.go", "go")
    assert {"Server", "Server.Start", "Server.helper"} <= {
        s.name for s in syms}
    assert ts.available("typescript")
    _s, edges, unresolved = ts.parse(
        "public class Svc {\n  void run() { obj.helper(); }\n}\n",
        "f.java", "java")
    assert "helper" in {u[1].split(".")[-1] for u in unresolved}
    assert "obj" not in {u[1] for u in unresolved}


def test_the_summary_note_is_silent_when_every_grammar_loads(monkeypatch,
                                                             fresh_cache):
    """`ccoder doctor` asks with no language named. It used to answer
    "incompatible" whenever tree-sitter imported, without trying a single
    grammar — so a fully working install was reported as broken."""
    monkeypatch.setattr(ts, "_load", lambda key: (object(), ""))
    assert ts.degraded_note() == ""


def test_the_summary_note_names_only_the_languages_that_fall_back(
        monkeypatch, fresh_cache):
    def load(key):
        if key in ("go", "java"):
            return None, "incompatible: tree_sitter_go: ImportError"
        return object(), ""

    monkeypatch.setattr(ts, "_load", load)
    note = ts.degraded_note()
    assert "Go" in note and "Java" in note
    assert "Rust" not in note and "C++" not in note
    assert "no language was named" not in note


def test_the_summary_note_says_not_installed_when_nothing_is(monkeypatch,
                                                             fresh_cache):
    monkeypatch.setattr(ts, "_load", lambda key: (None, "missing"))
    assert "not installed" in ts.degraded_note()
