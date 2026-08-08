@echo off
REM ============================================================================
REM  Cognitive Coder — stage, commit and push this session's work.
REM
REM  Usage:  push.bat          show what would happen, change nothing
REM          push.bat /go      do it
REM
REM  Written because two of the steps below are easy to get wrong and one of
REM  them is invisible until it bites:
REM
REM    * a stale .git\index.lock makes every git command fail with "Another
REM      git process seems to be running", which is alarming and almost never
REM      true. It is left behind when a git process is killed or, in this
REM      case, when one runs against a mount that cannot unlink the file;
REM    * .gitignore does NOT apply retroactively. 49 files were committed
REM      before it existed — coverage data, __pycache__, tool caches — and
REM      they stay tracked, producing diff noise on every machine, until they
REM      are explicitly untracked. `git rm -r --cached .` followed by
REM      `git add -A` is the idiom: it unstages everything, then re-adds only
REM      what the ignore rules now allow. Nothing is deleted from disk.
REM ============================================================================
setlocal EnableDelayedExpansion
set "SCRIPT_VERSION=2026-08-07.1"
cd /d "%~dp0"

set "GO="
if /i "%~1"=="/go" set "GO=1"

echo(
echo === Cognitive Coder push %SCRIPT_VERSION% ===
if not defined GO echo   DRY RUN - nothing will be changed. Use /go to commit.
echo(

where git >nul 2>&1
if errorlevel 1 (
    echo [ERR] git is not on PATH.
    endlocal
    exit /b 1
)
if not exist ".git" (
    echo [ERR] This is not a git repository. Run this from the Cognitive
    echo       Coder folder.
    endlocal
    exit /b 1
)
if not exist ".git-commit-message.txt" (
    echo [ERR] .git-commit-message.txt is missing - there is no message to
    echo       commit with. Write one, or use: git commit -m "..."
    endlocal
    exit /b 1
)

REM ---- 1. the stale lock -----------------------------------------------
if exist ".git\index.lock" (
    echo [1/5] removing a stale .git\index.lock
    echo       No git process is running; this is left-over state.
    if defined GO del /q ".git\index.lock"
) else (
    echo [1/5] no stale lock - nothing to clear
)

REM ---- 2. engine debris -------------------------------------------------
REM  Written when the engine is pointed at its own repository as a project.
if exist ".s" (
    echo [2/5] removing .s\ - codemap state from a test run
    if defined GO rmdir /s /q ".s" 2>nul
) else (
    echo [2/5] no .s\ debris
)

REM ---- 3. apply .gitignore retroactively --------------------------------
for /f %%N in ('git ls-files ^| findstr /r "__pycache__ \.pyc$ ^^\.coverage \.writetest" ^| find /c /v ""') do set "STALE=%%N"
if not defined STALE set "STALE=0"
echo [3/5] %STALE% tracked file(s) the .gitignore now covers
if "%STALE%"=="0" goto :staged
echo       Untracking them. They stay on disk; only the index changes.
if defined GO git rm -r --cached . >nul 2>&1

:staged
REM ---- 4. stage and show -------------------------------------------------
echo [4/5] staging
if defined GO git add -A
echo(
echo   --- what will be committed ---
if defined GO (
    git status --short
) else (
    git status --short --untracked-files=all
)
echo(

REM ---- 5. commit and push ------------------------------------------------
if not defined GO (
    echo [5/5] DRY RUN - stopping here.
    echo       Re-run as:  push.bat /go
    echo(
    endlocal
    exit /b 0
)

echo [5/5] committing
git commit -F ".git-commit-message.txt"
if errorlevel 1 (
    echo(
    echo [ERR] The commit failed. Nothing was pushed. The most likely cause
    echo       is that there was nothing staged - check the list above.
    endlocal
    exit /b 1
)
echo(
echo   pushing to origin/main ...
git push
if errorlevel 1 (
    echo(
    echo [ERR] The push failed, but THE COMMIT SUCCEEDED - your work is
    echo       safe in local history. Usual causes: no network, or GitHub
    echo       credentials have expired. Fix it and run: git push
    endlocal
    exit /b 1
)
echo(
echo   Done. %COMPUTERNAME% is in sync with origin/main.
git log --oneline -3
echo(
endlocal
exit /b 0
