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

# uv, PINNED, and its installer script verified before it runs. It was
# `curl https://astral.sh/uv/install.sh | sh`: whatever that URL served on
# the day, unverified, executed. Now a fixed release's installer is fetched
# to a file and its SHA-256 compared with the one below; a mismatch, or no
# sha256 tool to check with, means it is NOT run. What remains trusted:
# that installer downloads the uv archive itself, over HTTPS from the same
# GitHub release, and 0.8.22's installer carries no checksum for it.
# To move to a newer uv: change both lines, from the release's own asset.
UV_VERSION="0.8.22"
UV_INSTALLER_SHA256="f1ac30b1849f90b17ca93a9c2d40f74d7ad79cf57ad5a179bc5c9948848a70cd"
UV_INSTALLER_URL="https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-installer.sh"

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

# pip's stderr, kept so a failure can SAY something. It went to /dev/null,
# and a failed core install was then the single line "[!!] FAILED" — with
# the one sentence that explained it thrown away.
PIP_ERR="$(mktemp 2>/dev/null || echo "${TMPDIR:-/tmp}/cc-pip-err.$$")"
# pip's download cache, too, stays in the clone (rule 4); it was the one
# thing a run left in $HOME.
export PIP_CACHE_DIR="$TOOLS/pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
trap 'rm -f "$PIP_ERR" "$PIP_ERR.tail"' EXIT

# pip install, offline-first. `pip install -e .` builds in an ISOLATED env
# by default, and that env fetches setuptools from PyPI even when every
# wheel the project needs is already present — so an offline machine with
# a perfectly good venv failed here. --no-build-isolation uses the venv's
# own setuptools; only if that cannot build does the isolated way run.
pip_install() {
    "$VPY" -m pip install --quiet --no-build-isolation "$@" \
        >/dev/null 2>"$PIP_ERR" && return 0
    "$VPY" -m pip install --quiet "$@" >/dev/null 2>"$PIP_ERR"
}

# The last five lines pip wrote to stderr, indented, into the summary.
# Read from a file, not a pipe: `tail | while` runs the loop in a subshell
# in POSIX sh, and every `note` it made would vanish with it.
pip_tail() {
    tail -n 5 "$PIP_ERR" > "$PIP_ERR.tail" 2>/dev/null || return 0
    while IFS= read -r line; do
        note "        | $line"
    done < "$PIP_ERR.tail"
    rm -f "$PIP_ERR.tail"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

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
    # INTO THE CLONE, all of it. Without these, uv puts the interpreter in
    # its global data directory, its cache in ~/.cache and (from 0.8) a
    # python3.x shim in ~/.local/bin — while this script, the README and
    # CONFORMANCE M46 all said "into the clone", and PYDIR was only ever
    # echoed. Deleting the clone must remove every trace (rule 4).
    export UV_PYTHON_INSTALL_DIR="$PYDIR"
    export UV_PYTHON_BIN_DIR="$TOOLS/bin"
    export UV_CACHE_DIR="$TOOLS/uv-cache"
    export UV_PYTHON_PREFERENCE="only-managed"
    UV_WHY=""
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
            rm -f "$TOOLS/uv-installer.sh"
            $FETCH "$TOOLS/uv-installer.sh" "$UV_INSTALLER_URL" \
                2>/dev/null || true
            GOT="$(sha256_of "$TOOLS/uv-installer.sh" 2>/dev/null || true)"
            if [ ! -f "$TOOLS/uv-installer.sh" ]; then
                UV_WHY="the download of $UV_INSTALLER_URL failed"
            elif [ -z "$GOT" ]; then
                UV_WHY="there is no sha256sum or shasum to verify it with,"
                UV_WHY="$UV_WHY so it was not run"
            elif [ "$GOT" != "$UV_INSTALLER_SHA256" ]; then
                UV_WHY="its SHA-256 was $GOT, not the pinned"
                UV_WHY="$UV_WHY $UV_INSTALLER_SHA256, so it was not run"
            else
                # UV_UNMANAGED_INSTALL: into this folder, no receipt, no
                # PATH edit, no self-update — uv's documented CI mode.
                UV_UNMANAGED_INSTALL="$TOOLS" UV_NO_MODIFY_PATH=1 \
                    sh "$TOOLS/uv-installer.sh" >/dev/null 2>&1 || true
                [ -x "$TOOLS/uv" ] || UV_WHY="its installer did not produce $TOOLS/uv"
            fi
            [ -z "$UV_WHY" ] || rm -f "$TOOLS/uv-installer.sh"
        fi
        if [ -x "$TOOLS/uv" ]; then
            "$TOOLS/uv" python install "$MIN_MAJOR.$MIN_MINOR" \
                >/dev/null 2>&1 || true
            PYTHON="$("$TOOLS/uv" python find "$MIN_MAJOR.$MIN_MINOR" \
                2>/dev/null || true)"
            PY_SOURCE="fetched by uv into this clone's .python/"
        fi
    else
        UV_WHY="there is neither curl nor wget to download it with"
    fi
    if [ -z "$PYTHON" ]; then
        # Rule 5: say exactly what was tried and what to install by hand.
        say ""
        say "FAILED: no usable Python, and one could not be fetched."
        say "  Tried: python3.11, python3.12, python3.13, python3, python"
        say "  Then:  uv $UV_VERSION from $UV_INSTALLER_URL"
        [ -z "$UV_WHY" ] || say "         — $UV_WHY."
        say "  Fix:   install Python $MIN_MAJOR.$MIN_MINOR or later, then run"
        say "         this again. Nothing outside .tools/ was changed."
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
if pip_install -e "$HERE"; then
    CORE_OK=1
    note "  [OK] core engine          .venv ready, 0 required deps"
else
    note "  [!!] core engine          pip install -e . FAILED — the engine"
    note "                            will not import. Everything below is"
    note "                            moot until that is fixed. pip said:"
    pip_tail
fi
note "  [OK] Python               $("$VPY" --version 2>&1 | cut -d' ' -f2) — $PY_SOURCE"

# --------------------------------------------------------------------------
# 4. toolchain detection — INFORMATIONAL ONLY
# --------------------------------------------------------------------------
# langs.py probes again at runtime (§6.1), so a compiler installed the week
# after install day simply works. This record is for the summary, nothing
# more.
# Not every tool spells it --version: `go --version` prints "flag provided
# but not defined", and that sentence used to BE the Go line of the summary.
version_of() {
    case "$1" in
        go|zig)     "$1" version ;;
        lua|luajit) "$1" -v ;;
        javac)      "$1" -version ;;
        *)          "$1" --version ;;
    esac 2>&1 | sed '/^Picked up /d' | head -n1 | cut -c1-40
    # (a JVM with JAVA_TOOL_OPTIONS set announces it before the version)
}

detect() {   # $1 = label, $2 = binaries, $3 = what its absence costs
    label="$1"; bins="$2"; cost="$3"
    for b in $bins; do
        if command -v "$b" >/dev/null 2>&1; then
            note "  [OK] $label$("$VPY" -c "print(' ' * max(1, 22 - len('$label')), end='')")$(version_of "$b")"
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
    if pip_install -e "$HERE[$1]"; then
        note "  [OK] $2"
    else
        note "  [--] $2 — could not be installed. $3 pip said:"
        pip_tail
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
