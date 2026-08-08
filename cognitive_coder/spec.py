# SPDX-License-Identifier: Apache-2.0
"""Read a build request out of a file, and say what is in it.

    cc build --spec plan.md

WHY A FILE AND NOT A SENTENCE
-----------------------------
The CLI has always taken the request as one positional argument, described as
"what you want built, in a sentence". That framing is wrong for the work this
engine is actually good at.

A sentence types fast and plans badly. Everything that makes a build go well —
which modules exist, what each one owns, what must not import what, which
tests must be written — is thinking done *before* the model is asked for
anything, and it does not fit in a shell argument or a one-line text box. The
first real specification put through this engine was sixty lines with four
numbered sections, and the person writing it had to paste it into a field that
showed one line of it at a time.

So: write the plan in an editor, take as long as it needs, keep it in version
control next to the code it produced, and hand the file over.

WHAT THIS DOES NOT DO
---------------------
It does not interpret the specification. The text is passed through verbatim
as the request, because the planner and the persona prompts are where meaning
is extracted and duplicating that here would create two answers to the same
question. What this module adds is only what a *file* makes possible and a
string does not: a name, a size, and a description of the contents that can be
shown to somebody before they spend twenty minutes of model time on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from .planner import _required_tests

#: Front matter delimiters. A spec written in an editor with a YAML header is
#: common enough to handle, and passing `---\ntitle: ...\n---` to a model as
#: though it were part of the requirements is noise at best.
_FRONT_MATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.S)

#: Roughly four characters per token for English prose. Deliberately crude:
#: the real tokenizer lives behind the LLM port and is not reachable from
#: here, and this number exists to answer "is this obviously too big?" rather
#: than to be precise. `Session` re-measures with the real tokenizer.
_CHARS_PER_TOKEN = 4

SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".text")

#: A specification is prose. Something far larger than this is a novel, a log
#: file, or a mistake — and reading a 40 MB file into memory to find that out
#: is avoidable.
MAX_BYTES = 2 * 1024 * 1024


class SpecError(Exception):
    """The file cannot be used as a build request, with the reason why."""


@dataclass
class Spec:
    """A build request that came from a file, and what can be seen in it."""

    text: str
    path: Path | None = None
    title: str = ""
    #: Test files the spec names outright. The planner extracts these too and
    #: adds any the model omits; surfacing them HERE is what lets a preview
    #: say "your spec names 2 test files" before a single token is spent.
    required_tests: tuple[str, ...] = ()
    #: Paths the spec mentions that look like source files. A hint for the
    #: reader, never a constraint on the planner — the planner is entitled to
    #: choose a different shape, and this only makes the difference visible.
    mentioned_paths: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def approx_tokens(self) -> int:
        return max(1, len(self.text) // _CHARS_PER_TOKEN)

    def summary(self) -> str:
        """One line, for a status bar or a log."""
        where = self.path.name if self.path else "typed"
        return (f"{where}: ~{self.approx_tokens} tokens, "
                f"{len(self.mentioned_paths)} file(s) named, "
                f"{len(self.required_tests)} test file(s) required")


def _title_of(text: str) -> str:
    """The spec's own name for itself: first heading, or first real line."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()
        #: An underlined setext heading, and the plain-text case: the first
        #: non-empty line is what a person would call the document.
        return line[:120]
    return ""


#: A path-shaped token: at least one directory OR a known source extension.
#: Bare words with dots in them (`version 1.1`, `Pole Position`) must not
#: qualify, because a preview that lists imaginary files is worse than one
#: that lists none.
_PATH = re.compile(
    r"(?<![\w/.\\-])"
    r"((?:[\w.-]+[/\\])+[\w.-]+\.\w{1,4}|[\w-]+\.(?:py|pyi|js|ts|tsx|jsx|"
    r"rs|go|java|c|h|cpp|hpp|cs|rb|lua|sh|ps1|sql|zig|bat|json|toml|yaml|yml))"
    r"(?![\w])")


def _paths_in(text: str) -> tuple[str, ...]:
    out: list[str] = []
    for m in _PATH.finditer(text):
        path = m.group(1).replace("\\", "/").lstrip("./")
        if path not in out:
            out.append(path)
    return tuple(out)


def from_text(text: str, *, path: Path | None = None) -> Spec:
    """Describe a request that is already in hand. No I/O."""
    warnings: list[str] = []
    stripped = _FRONT_MATTER.sub("", text)
    if stripped != text:
        warnings.append("YAML front matter was removed before sending")
    stripped = stripped.strip()
    if not stripped:
        raise SpecError("there is nothing in it but whitespace")

    tests = tuple(_required_tests(stripped))
    paths = _paths_in(stripped)
    #: Source files only in `mentioned_paths`; the tests are reported
    #: separately and listing them twice makes the preview read as though the
    #: spec asked for more than it did.
    sources = tuple(p for p in paths if p not in tests)

    if len(stripped) > 40_000:
        warnings.append(
            f"this is {len(stripped):,} characters (~{len(stripped) // 4:,} "
            f"tokens) — larger than most local models' context, so the "
            f"planner may see only part of it")
    return Spec(text=stripped, path=path, title=_title_of(stripped),
                required_tests=tests, mentioned_paths=sources,
                warnings=tuple(warnings))


def load(path: str | Path) -> Spec:
    """Read a build request from a file.

    Errors name the file and say what to do, because this runs at the very
    start of a long operation and a bare traceback here costs the whole run.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise SpecError(f"{p} does not exist")
    if p.is_dir():
        raise SpecError(f"{p} is a folder, not a specification file")
    if p.suffix.lower() not in SUFFIXES:
        raise SpecError(
            f"{p.name} is not a text specification "
            f"({', '.join(SUFFIXES)}). If it really is plain text, rename it "
            f"to .md or .txt.")
    size = p.stat().st_size
    if size == 0:
        raise SpecError(f"{p.name} is empty")
    if size > MAX_BYTES:
        raise SpecError(
            f"{p.name} is {size / 1_048_576:.1f} MB. A build specification "
            f"is prose; this is large enough that it is probably the wrong "
            f"file.")
    #: UTF-8 with a lenient fallback. A spec pasted out of Word arrives as
    #: cp1252 often enough that failing on it would be a poor trade, and a
    #: mangled quotation mark costs nothing here.
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = p.read_text(encoding="utf-8", errors="replace")
    spec = from_text(text, path=p)
    if "�" in spec.text:
        spec = Spec(text=spec.text, path=spec.path, title=spec.title,
                    required_tests=spec.required_tests,
                    mentioned_paths=spec.mentioned_paths,
                    warnings=spec.warnings + (
                        "some characters could not be decoded and were "
                        "replaced — save the file as UTF-8 if that matters",))
    return spec
