@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ----------------------------------------
REM Change to the directory of this script
REM ----------------------------------------
cd /d "%~dp0"

REM ----------------------------------------
REM Virtual environment directory name
REM ----------------------------------------
set VENV_DIR=.venv

echo ========================================
echo   CSV Explorer - Launcher
echo ========================================
echo.

REM ----------------------------------------
REM Detect Python command (prefer py -3, fallback to python)
REM ----------------------------------------
set "PY_CMD="

py -3 --version >nul 2>&1
if %ERRORLEVEL%==0 (
    set "PY_CMD=py -3"
)

if not defined PY_CMD (
    python --version >nul 2>&1
    if %ERRORLEVEL%==0 (
        set "PY_CMD=python"
    )
)

if not defined PY_CMD (
    echo [ERROR] No Python installation found via "py -3" or "python".
    echo Make sure Python 3.11 (recommended) is installed.
    echo You can also run troubleshoot_python.bat for more help.
    echo.
    pause
    exit /b 1
)

REM ----------------------------------------
REM Check Python version (require 3.9–3.12, recommend 3.11)
REM ----------------------------------------
for /f "tokens=2" %%v in ('%PY_CMD% --version') do set PY_VER=%%v

for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

echo [INFO] Detected Python version: !PY_VER!

if NOT "!PY_MAJOR!"=="3" (
    echo [ERROR] Unsupported Python major version: !PY_MAJOR!
    echo Please install Python 3.11 (recommended) and try again.
    pause
    exit /b 1
)

REM Block too-new versions (e.g. 3.13, where pandas wheels may be missing)
if !PY_MINOR! GEQ 13 (
    echo [ERROR] Python 3.!PY_MINOR! is currently too new for this app.
    echo Some dependencies (like pandas) do not provide ready-made wheels and
    echo would require a full C++ build toolchain.
    echo.
    echo Please install Python 3.11 (recommended) or 3.12 instead, then run this script again.
    pause
    exit /b 1
)

echo [INFO] Python version is acceptable.
echo.

REM ----------------------------------------
REM Create venv if it does not exist
REM ----------------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Please check your Python installation or run troubleshoot_python.bat
        pause
        exit /b 1
    )
) else (
    echo [INFO] Using existing virtual environment in %VENV_DIR%.
)

REM ----------------------------------------
REM Activate the virtual environment
REM ----------------------------------------
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM ----------------------------------------
REM Upgrade pip (quietly) and install dependencies
REM ----------------------------------------
echo [INFO] Preparing Python packages...
echo     (this may take a few minutes on the very first run)
echo.

REM Upgrade pip silently
python -m pip install --upgrade pip --quiet >nul 2>&1

REM Install requirements:
REM  - --no-input avoids any interactive prompts
REM  - --quiet keeps the console clean
REM  - output is logged to pip_install.log so we can inspect on error
echo [INFO] Checking / installing required packages...
python -m pip install --no-input --quiet -r requirements.txt >pip_install.log 2>&1

if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    echo Possible causes:
    echo   - Unsupported Python version
    echo   - Network issues
    echo   - Broken pip cache
    echo.
    echo See details below (from pip_install.log):
    echo ----------------------------------------
    type pip_install.log
    echo ----------------------------------------
    echo.
    echo You can also run troubleshoot_python.bat for additional checks.
    pause
    exit /b 1
) else (
    del pip_install.log >nul 2>&1
    echo [INFO] All required packages are installed.
)

REM ----------------------------------------
REM Run the Streamlit app
REM ----------------------------------------
echo.
echo [INFO] Starting CSV Explorer app...
echo     If your browser does not open automatically, go to: http://localhost:8501
echo.
streamlit run app.py

REM ----------------------------------------
REM Keep window open after Streamlit exits
REM ----------------------------------------
echo.
echo [INFO] Streamlit app has stopped.
pause
endlocal
