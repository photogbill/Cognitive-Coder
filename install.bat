@echo off
REM SPDX-License-Identifier: Apache-2.0
REM
REM Cognitive Coder - Windows installer.
REM
REM THE CONTRACT THIS OBEYS - section 10.1:
REM   1. FULLY NON-INTERACTIVE. No prompts, no "press any key". A user who
REM      walks away comes back to a finished install, not a question. This
REM      rule exists because a third-party installer's hidden prompt once
REM      stalled an entire unattended run.
REM   2. NO MANUAL EXTRACTION, EVER. If something must be downloaded, this
REM      downloads and unpacks it.
REM   3. IDEMPOTENT. Running it twice is safe and re-fetches only what is
REM      missing. That is how a user recovers from a partial install.
REM   4. NOTHING GLOBAL IS CHANGED. A .venv inside the clone, no PATH edits,
REM      no registry writes. Deleting the clone removes every trace.
REM   5. EVERY OPTIONAL COMPONENT DEGRADES. A failed optional download
REM      disables one feature and says which; it never aborts the install.
REM   6. IT ENDS WITH A SUMMARY saying what a missing item COSTS.
REM   7. THE EXIT CODE IS MEANINGFUL. 0 if the core engine is usable,
REM      non-zero only if it is not.
REM
REM   8. THE BATCH GOTCHA, stated because it has bitten this author before:
REM      inside a parenthesised block, "echo (text)" terminates the block
REM      early. There are therefore NO round brackets in any echoed text in
REM      this file. Where one is unavoidable it is escaped as ^( and ^).
REM      Read that twice before editing anything below.
REM
REM Usage:  install.bat [/providers] [/treesitter] [/dev]

setlocal EnableDelayedExpansion

set "HERE=%~dp0"
if "%HERE:~-1%"=="\" set "HERE=%HERE:~0,-1%"
set "VENV=%HERE%\.venv"
set "PYDIR=%HERE%\.python"
set "TOOLS=%HERE%\.tools"
set "SUMMARY=%TEMP%\cc-install-summary-%RANDOM%.txt"
set "CORE_OK=0"
set "WANT_PROVIDERS=0"
set "WANT_TREESITTER=0"
set "WANT_DEV=0"

if exist "%SUMMARY%" del /q "%SUMMARY%" >nul 2>&1
type nul > "%SUMMARY%"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="/providers"  set "WANT_PROVIDERS=1"
if /i "%~1"=="/treesitter" set "WANT_TREESITTER=1"
if /i "%~1"=="/dev"        set "WANT_DEV=1"
shift
goto parse
:parsed

echo Cognitive Coder - installing into this folder only.
echo Nothing outside %HERE% will be changed.
echo.

REM ----------------------------------------------------------------------
REM 1. find a usable Python, and CHECK ITS VERSION
REM ----------------------------------------------------------------------
REM `python` is 3.9 on more machines than you expect. And a virtual
REM environment does NOT give a project its own Python - it isolates
REM packages; the interpreter is whatever created it. A .venv built from a
REM 3.9 `python` is a 3.9 venv, and Cognitive Coder then fails at the first
REM union-type annotation with a syntax error that looks like broken code
REM rather than a wrong interpreter. So: probe, then verify with --version.

set "PYTHON="
set "PY_SOURCE=the system Python"

call :try_python "py -3.11"
if defined PYTHON goto have_python
call :try_python "py -3.12"
if defined PYTHON goto have_python
call :try_python "python3.11"
if defined PYTHON goto have_python
call :try_python "python"
if defined PYTHON goto have_python
call :try_python "py -3"
if defined PYTHON goto have_python

REM Step 3 of section 10.2a: fetching is the NORMAL path on an old machine.
REM Not an error, not a warning.
echo No Python 3.11 or later was found. Fetching one into
echo %PYDIR% - nothing is installed system-wide.
if not exist "%TOOLS%" mkdir "%TOOLS%" >nul 2>&1
if not exist "%TOOLS%\uv.exe" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ErrorActionPreference='SilentlyContinue'; $env:UV_INSTALL_DIR='%TOOLS%'; $env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 ^| iex" >nul 2>&1
)
if exist "%TOOLS%\uv.exe" (
    "%TOOLS%\uv.exe" python install 3.11 >nul 2>&1
    for /f "usebackq delims=" %%P in (`"%TOOLS%\uv.exe" python find 3.11 2^>nul`) do set "PYTHON=%%P"
    set "PY_SOURCE=fetched by uv into this clone"
)
if not defined PYTHON (
    echo.
    echo FAILED: no usable Python, and one could not be fetched.
    echo   Tried: py -3.11, py -3.12, python3.11, python, py -3
    echo   Then:  downloading uv from https://astral.sh/uv/install.ps1
    echo   Fix:   install Python 3.11 or later, then run this again.
    echo          Nothing was changed.
    exit /b 1
)

:have_python
for /f "delims=" %%V in ('"%PYTHON%" --version 2^>^&1') do set "PYVER=%%V"
echo Found %PYVER% at %PYTHON%

REM ----------------------------------------------------------------------
REM 2. the venv, inside the clone, from THAT interpreter
REM ----------------------------------------------------------------------
set "REUSE=0"
if exist "%VENV%\Scripts\python.exe" (
    "%VENV%\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "REUSE=1"
)
if "%REUSE%"=="1" (
    echo Reusing the existing .venv
) else (
    if exist "%VENV%" rmdir /s /q "%VENV%" >nul 2>&1
    "%PYTHON%" -m venv "%VENV%"
)
set "VPY=%VENV%\Scripts\python.exe"
"%VPY%" -m pip install --quiet --upgrade pip >nul 2>&1

REM ----------------------------------------------------------------------
REM 3. the core: zero required runtime dependencies
REM ----------------------------------------------------------------------
"%VPY%" -m pip install --quiet -e "%HERE%" >nul 2>&1
if errorlevel 1 (
    call :note "  [!!] core engine          pip install -e . FAILED - the"
    call :note "                            engine will not import."
) else (
    set "CORE_OK=1"
    call :note "  [OK] core engine          .venv ready, 0 required deps"
)
for /f "delims=" %%V in ('"%VPY%" --version 2^>^&1') do set "VPYVER=%%V"
call :note "  [OK] Python               %VPYVER% - %PY_SOURCE%"

REM ----------------------------------------------------------------------
REM 4. toolchain detection - INFORMATIONAL ONLY
REM ----------------------------------------------------------------------
REM langs.py probes again at runtime, so a compiler installed the week after
REM install day simply works. This record is for the summary, nothing more.
call :detect "C/C++ toolchain " "gcc clang cl"   "C and C++ targets unavailable"
call :detect "Rust toolchain  " "rustc"          "Rust targets unavailable"
call :detect "Java toolchain  " "javac"          "Java targets unavailable"
call :detect "Go toolchain    " "go"             "Go targets unavailable"
call :detect "Node.js         " "node"           "JavaScript and TypeScript targets unavailable"
call :detect ".NET SDK        " "dotnet"         "C# targets unavailable"
call :detect "Zig             " "zig"            "Zig targets unavailable"
call :detect "Lua             " "lua luajit"     "Lua targets unavailable"
call :detect "Ruby            " "ruby"           "Ruby targets unavailable"
call :detect "SQLite          " "sqlite3"        "SQL targets unavailable"
call :detect "Godot           " "godot godot4"   "GDScript degrades to outline-and-edit only - no syntax check, no run, no tests"

REM ----------------------------------------------------------------------
REM 5. optional extras - only on an explicit flag, and each degrades
REM ----------------------------------------------------------------------
if "%WANT_TREESITTER%"=="1" (
    call :optional treesitter "tree-sitter            " "C/C++/Rust/JS outlines stay regex-approximate."
) else (
    call :note "  [--] tree-sitter          not installed - C/C++/Rust/JS"
    call :note "                            outlines will be regex-approximate"
    call :note "                            rather than parsed. Add with"
    call :note "                            install.bat /treesitter"
)

if "%WANT_PROVIDERS%"=="1" (
    call :optional anthropic "remote provider SDKs   " "Remote providers stay unavailable; local ones are unaffected."
) else (
    call :note "  [--] remote providers     not installed. Offline is the"
    call :note "                            default either way; run"
    call :note "                            install.bat /providers to add them."
)

if "%WANT_DEV%"=="1" (
    call :optional dev "dev tools              " "The test suite cannot be run from this clone."
)

REM ----------------------------------------------------------------------
REM 6. the self-test
REM ----------------------------------------------------------------------
if "%CORE_OK%"=="1" "%VENV%\Scripts\ccoder.exe" doctor >nul 2>&1

REM ----------------------------------------------------------------------
REM 7. the summary
REM ----------------------------------------------------------------------
echo.
echo ============================================================
echo  Cognitive Coder - installation summary
echo ============================================================
type "%SUMMARY%"
echo.
echo   [--] entries are OPTIONAL: each disables one feature, not the tool.
echo   Re-run the installer to retry only what is missing.
echo.
echo   Next:  %VENV%\Scripts\ccoder doctor
echo          %VENV%\Scripts\ccoder build "a CSV parser with tests"
echo.
del /q "%SUMMARY%" >nul 2>&1

if "%CORE_OK%"=="1" exit /b 0
exit /b 1

REM ======================================================================
REM subroutines
REM ======================================================================

:try_python
REM %~1 is a command line, which may be "py -3.11". Verify the VERSION.
for /f "delims=" %%E in ('%~1 -c "import sys; print(sys.executable if sys.version_info >= (3,11) else '''''')" 2^>nul') do (
    if not "%%E"=="" set "PYTHON=%%E"
)
goto :eof

:detect
REM %~1 label, %~2 space-separated binaries, %~3 what its absence costs
set "FOUND="
for %%B in (%~2) do (
    if not defined FOUND (
        for /f "delims=" %%W in ('where %%B 2^>nul') do (
            if not defined FOUND set "FOUND=%%W"
        )
    )
)
if defined FOUND (
    call :note "  [OK] %~1     !FOUND!"
) else (
    call :note "  [--] %~1     not found - %~3"
)
goto :eof

:optional
REM %~1 extra name, %~2 label, %~3 cost of absence
"%VPY%" -m pip install --quiet -e "%HERE%[%~1]" >nul 2>&1
if errorlevel 1 (
    call :note "  [--] %~2 could not be installed. %~3"
) else (
    call :note "  [OK] %~2 installed"
)
goto :eof

:note
REM Appending through a file rather than a variable: a summary built in an
REM environment variable hits the 8191-character limit on a machine with
REM many toolchains, and truncates the very report that was supposed to be
REM honest about what is missing.
echo %~1>> "%SUMMARY%"
goto :eof
