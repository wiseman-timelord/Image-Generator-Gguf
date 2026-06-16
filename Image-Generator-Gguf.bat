@echo off
REM Image-Generator-Gguf.bat
REM Batch menu for Image Generator GGUF.
REM No parentheses in logic per project constraint.

mode con cols=84 lines=30
powershell -noprofile -command "& { $w = $Host.UI.RawUI; $b = $w.BufferSize; $b.Height = 6000; $w.BufferSize = $b; }"

REM ==== Ensure System32 is on PATH - ping, timeout, where, netsh ====
REM Prepend System32 without discarding the existing PATH so installed
REM tools, the venv Scripts folder and Python itself remain reachable.
set "PATH=%SystemRoot%\System32;%SystemRoot%\System32\Wbem;%PATH%"

REM ==== DP0 TO SCRIPT BLOCK ====
set "ScriptDirectory=%~dp0"
set "ScriptDirectory=%ScriptDirectory:~0,-1%"
cd /d "%ScriptDirectory%"
echo Dp0ing to: %ScriptDirectory%

REM ==== Admin Check ====
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo Error: Admin Required!
    timeout /t 3 /nobreak >nul
    echo Right Click, Run As Administrator.
    timeout /t 3 /nobreak >nul
    goto :end_of_script_console
)
echo Status: Administrator
timeout /t 2 /nobreak >nul

REM Ensure base directories exist
if not exist "data"    mkdir data
if not exist "output"  mkdir output
if not exist "models"  mkdir models
if not exist "scripts" mkdir scripts

REM Locate system Python - used only to bootstrap installer / venv
set "SYSPYTHON="

py --version >nul 2>&1
if not errorlevel 1 set "SYSPYTHON=py"
if not "%SYSPYTHON%"=="" goto :found_python

python3.13 --version >nul 2>&1
if not errorlevel 1 set "SYSPYTHON=python3.13"
if not "%SYSPYTHON%"=="" goto :found_python

python3.12 --version >nul 2>&1
if not errorlevel 1 set "SYSPYTHON=python3.12"
if not "%SYSPYTHON%"=="" goto :found_python

python3.11 --version >nul 2>&1
if not errorlevel 1 set "SYSPYTHON=python3.11"
if not "%SYSPYTHON%"=="" goto :found_python

python --version >nul 2>&1
if not errorlevel 1 set "SYSPYTHON=python"
if not "%SYSPYTHON%"=="" goto :found_python

echo.
echo   ERROR: Python 3.11+ not found on PATH.
echo   Install Python from https://python.org then re-run.
echo.
pause
exit /b 1

:found_python

REM venv python path - used to run launcher directly (no activate needed)
set "VENVPY=venv\Scripts\python.exe"

:menu
cls
echo ================================================================================
echo       Image-Generator-Gguf: Batch Menu
echo ================================================================================
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo       1. Run Main Program
echo.
echo       2. Run Installation
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo.
echo ================================================================================
set /p "CHOICE=   Selection; Menu Options = 1-2, Exit Batch = X: "

if /i "%CHOICE%"=="1" goto :run_program
if /i "%CHOICE%"=="2" goto :run_install
if /i "%CHOICE%"=="x" goto :exit_batch
if /i "%CHOICE%"=="X" goto :exit_batch

REM Invalid input - loop back
goto :menu


REM -------------------------------------------------------------------------
:run_program
REM Run launcher.py directly through the venv python executable.
REM Using the venv python directly avoids activate.bat PATH issues while
REM still ensuring all venv packages are available.
if not exist "%VENVPY%" goto :no_venv

echo.
echo   Starting application at http://127.0.0.1:7860
echo.
"%VENVPY%" launcher.py

echo.
echo   Application exited. Press any key to return to menu.
pause >nul
goto :menu


:no_venv
echo.
echo   Virtual environment not found at .\venv\
echo   Please run option 2 Installation first.
echo.
pause
goto :menu


REM -------------------------------------------------------------------------
:run_install
echo.
echo   Starting installer...
echo.
%SYSPYTHON% installer.py

echo.
echo   Press any key to return to menu.
pause >nul
goto :menu


REM -------------------------------------------------------------------------
:exit_batch
echo.
echo   Goodbye.
echo.
exit /b 0


:end_of_script_console
pause
exit /b 1