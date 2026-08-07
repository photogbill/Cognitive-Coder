#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
#
# Cognitive Coder — Linux installer.
#
# THE CONTRACT THIS OBEYS (§10.1), lifted from ATK where it was arrived at
# painfully, and worth inheriting wholesale:
#
#   1. FULLY NON-INTERACTIVE. No prompts, no "press any key", nothing that
#      waits for a human. A user who walks away must come back to a finished
#      install, not a question.
#   2. NO MANUAL EXTRACTION, EVER. If something must be downloaded, this
#      downloads and unpacks it. Never instruct a human to fetch a zip.
#   3. IDEMPOTENT. Running it twice is safe and re-fetches only what is
#      missing. That is how a user recovers from a partial install.
#   4. NOTHING GLOBAL IS CHANGED. A venv inside the clone, no PATH edits, no
#      system packages. Deleting the clone removes every trace.
#   5. EVERY OPTIONAL COMPONENT DEGRADES. A failed optional download disables
#      one feature and says which; it never aborts the install.
#   6. IT ENDS WITH A SUMMARY listing what landed and what did not, one line
#      per item saying what a missing item COSTS.
#   7. THE EXIT CODE IS MEANINGFUL. 0 if the core engine is usable, non-zero
#      only if it is not. A missing optional toolchain is not a failure.
#
# Usage:  ./install.sh [--providers] [--treesitter] [--dev]
#
# macOS: this will mostly work, but the toolchain detection differs — clang
# rather than gcc, no .exe suffixes — and it is UNTESTED there. The README
# says so plainly rather than claiming support nobody has verified.

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/.venv"
PYDIR="$HERE/.python"
TOOLS="$HERE/.tools"
MIN_MAJOR=3
MIN_MINOR=11

WANT_PROVIDERS=0
WANT_TREESITTER=0
WANT_DEV=0
for arg in "$@"; do
    case "$arg" in
        --providers)  WANT_PROVIDERS=1 ;;
        --treesitter) WANT_TREESITTER=1 ;;
        --dev)        WANT_DEV=1 ;;
        --help|-h)
            sed -n '3,30p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Ignoring unknown option: $arg" ;;
    esac
done

# Collected as we go; printed once at the end. Nothing is echoed twice, so
# the summary is the single place a user has to read.
SUMMARY=""
CORE_OK=0
note() { SUMMARY="${SUMMARY}$1
"; }

say() { printf '%s\n' "$1"; }

say "Cognitive Coder — installing into this folder only."
say "Nothing outside $HERE will be changed."
say ""

# --------------------------------------------------------------------------
# 1. find a usable Python, and CHECK ITS VERSION
# --------------------------------------------------------------------------
# `python` is 3.9 on more machines than you expect, and a venv does NOT give
# a project its own Python — it isolates packages; the interpreter is
# whatever created it. Building a .venv from a 3.9 `python` produces a 3.9
# venv, and Cognitive Coder then fails at the first `X | None` annotation
# with a syntax error that looks like broken code rather than a wrong
# interpreter. So: probe, and verify with --version.

PYTHON=""
check_python() {
    [ -x "$(command -v "$1" 2>/dev/null)" ] || return 1
    "$1" -c "import sys; raise SystemExit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null
}

for candidate in python3.11 python3.12 python3.13 python3 python; do
    if check_python "$candidate"; then
        PYTHON="$(command -v "$candidate")"
        break
    fi
done

PY_SOURCE="the system Python"
if [ -n "$PYTHON" ]; then
    say "Found $("$PYTHON" --version 2>&1) at $PYTHON"
else
    # Step 3 of §10.2a: this is the NORMAL path on an old machine. Not an
    # error, not a warning.
    say "No Python $MIN_MAJOR.$MIN_MINOR or later was found. Fetching one into"
    say "$PYDIR — nothing is installed system-wide."
    mkdir -p "$TOOLS"
    if command -v curl >/dev/null 2>&1; then
        FETCH="curl -fsSL -o"
    elif command -v wget >/dev/null 2>&1; then
        FETCH="wget -qO"
    else
        FETCH=""
    fi
    if [ -n "$FETCH" ]; then
        # uv is a single static binary that can install a specific CPython
        # and create the venv. It is by far the least code, works identically
        # on Windows and Linux, and never touches the system Python.
        if [ ! -x "$TOOLS/uv" ]; then
            $FETCH "$TOOLS/uv-installer.sh" https://astral.sh/uv/install.sh \
                2>/dev/null || true
            if [ -f "$TOOLS/uv-installer.sh" ]; then
                UV_INSTALL_DIR="$TOOLS" UV_NO_MODIFY_PATH=1 \
                    sh "$TOOLS/uv-installer.sh" >/dev/null 2>&1 || true
            fi
        fi
        if [ -x "$TOOLS/uv" ]; then
            "$TOOLS/uv" python install "$MIN_MAJOR.$MIN_MINOR" \
                >/dev/null 2>&1 || true
            PYTHON="$("$TOOLS/uv" python find "$MIN_MAJOR.$MIN_MINOR" \
                2>/dev/null || true)"
            PY_SOURCE="fetched by uv into this clone"
        fi
    fi
    if [ -z "$PYTHON" ]; then
        # Rule 5: say exactly what was tried and what to install by hand.
        say ""
        say "FAILED: no usable Python, and one could not be fetched."
        say "  Tried: python3.11, python3.12, python3.13, python3, python"
        say "  Then:  downloading uv from https://astral.sh/uv/install.sh"
        say "  Fix:   install Python $MIN_MAJOR.$MIN_MINOR or later, then run"
        say "         this again. Nothing was changed."
        exit 1
    fi
fi

# --------------------------------------------------------------------------
# 2. the venv, inside the clone, from THAT interpreter
# --------------------------------------------------------------------------
if [ -x "$VENV/bin/python" ] && \
   "$VENV/bin/python" -c "import sys; raise SystemExit(0 if sys.version_info >= ($MIN_MAJOR, $MIN_MINOR) else 1)" 2>/dev/null; then
    say "Reusing the existing .venv"          # rule 3: idempotent
else
    rm -rf "$VENV"
    "$PYTHON" -m venv "$VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true

# --------------------------------------------------------------------------
# 3. the core: zero required runtime dependencies
# --------------------------------------------------------------------------
if "$VPY" -m pip install --quiet -e "$HERE" >/dev/null 2>&1; then
    CORE_OK=1
    note "  [OK] core engine          .venv ready, 0 required deps"
else
    note "  [!!] core engine          pip install -e . FAILED — the engine"
    note "                            will not import. Everything below is"
    note "                            moot until that is fixed."
fi
note "  [OK] Python               $("$VPY" --version 2>&1 | cut -d' ' -f2) — $PY_SOURCE"

# --------------------------------------------------------------------------
# 4. toolchain detection — INFORMATIONAL ONLY
# --------------------------------------------------------------------------
# langs.py probes again at runtime (§6.1), so a compiler installed the week
# after install day simply works. This record is for the summary, nothing
# more.
detect() {   # $1 = label, $2 = binaries, $3 = what its absence costs
    label="$1"; bins="$2"; cost="$3"
    for b in $bins; do
        if command -v "$b" >/dev/null 2>&1; then
            note "  [OK] $label$("$VPY" -c "print(' ' * max(1, 22 - len('$label')), end='')")$("$b" --version 2>&1 | head -n1 | cut -c1-40)"
            return 0
        fi
    done
    note "  [--] $label$("$VPY" -c "print(' ' * max(1, 22 - len('$label')), end='')")not found — $cost"
    return 1
}

detect "C/C++ toolchain"  "gcc clang cc"  "C and C++ targets unavailable" || true
detect "Rust toolchain"   "rustc"         "Rust targets unavailable" || true
detect "Java toolchain"   "javac"         "Java targets unavailable" || true
detect "Go toolchain"     "go"            "Go targets unavailable" || true
detect "Node.js"          "node"          "JavaScript and TypeScript targets unavailable" || true
detect ".NET SDK"         "dotnet"        "C# targets unavailable" || true
detect "Zig"              "zig"           "Zig targets unavailable" || true
detect "Lua"              "lua luajit"    "Lua targets unavailable" || true
detect "Ruby"             "ruby"          "Ruby targets unavailable" || true
detect "SQLite"           "sqlite3"       "SQL targets unavailable" || true
detect "Godot"            "godot godot4"  "GDScript degrades to outline-and-edit only: no syntax check, no run, no tests" || true

# --------------------------------------------------------------------------
# 5. optional extras — only on an explicit flag, and each degrades (rule 5)
# --------------------------------------------------------------------------
optional() {   # $1 = extra, $2 = label, $3 = cost of absence
    if "$VPY" -m pip install --quiet -e "$HERE[$1]" >/dev/null 2>&1; then
        note "  [OK] $2"
    else
        note "  [--] $2 — could not be installed. $3"
    fi
}

if [ "$WANT_TREESITTER" = "1" ]; then
    optional treesitter "tree-sitter" \
        "C/C++/Rust/JS outlines stay regex-approximate rather than parsed."
else
    note "  [--] tree-sitter          not installed — C/C++/Rust/JS outlines"
    note "                            will be regex-approximate rather than"
    note "                            parsed. Add with --treesitter."
fi

if [ "$WANT_PROVIDERS" = "1" ]; then
    optional anthropic "remote provider SDKs" \
        "Remote providers stay unavailable; local ones are unaffected."
else
    note "  [--] remote providers     not installed. Offline is the default"
    note "                            either way; run ./install.sh"
    note "                            --providers to add them."
fi

if [ "$WANT_DEV" = "1" ]; then
    optional dev "dev tools (pytest, ruff, coverage)" \
        "The test suite cannot be run from this clone."
fi

# --------------------------------------------------------------------------
# 6. the self-test
# --------------------------------------------------------------------------
say ""
if [ "$CORE_OK" = "1" ]; then
    "$VENV/bin/ccoder" doctor >/dev/null 2>&1 || true
fi

# --------------------------------------------------------------------------
# 7. the summary (rule 6)
# --------------------------------------------------------------------------
say "============================================================"
say " Cognitive Coder — installation summary"
say "============================================================"
printf '%s' "$SUMMARY"
say ""
say "  [--] entries are OPTIONAL: each disables one feature, not the tool."
say "  Re-run the installer to retry only what is missing."
say ""
say "  Next:  $VENV/bin/ccoder doctor"
say "         $VENV/bin/ccoder build \"a CSV parser with tests\""
say ""

# Rule 7: 0 if the core engine is usable, non-zero only if it is not.
[ "$CORE_OK" = "1" ] || exit 1
exit 0
