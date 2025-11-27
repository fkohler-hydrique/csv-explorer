@echo off
setlocal

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
REM Try to detect a suitable Python command
REM Priority: py -3.13, then py -3.12, then py -3.11, then python
REM ----------------------------------------
set "PY_CMD="

py -3.13 --version >nul 2>&1 && set "PY_CMD=py -3.13"
if not defined PY_CMD py -3.12 --version >nul 2>&1 && set "PY_CMD=py -3.12"
if not defined PY_CMD py -3.11 --version >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD python --version >nul 2>&1 && set "PY_CMD=python"

if not defined PY_CMD (
    echo [ERROR] No Python installation found via:
    echo         py -3.13, py -3.12, py -3.11 or python
    echo.
    echo This usually means Python is installed but not on PATH, or
    echo the "py" launcher was not installed.
    echo.
    echo Please install Python 3.11+ (3.11, 3.12 or 3.13 recommended)
    echo from https://www.python.org/downloads/windows/
    echo and make sure:
    echo   - "Install launcher for all users" is checked
    echo   - (optionally) "Add python.exe to PATH" is checked
    echo.
    echo You can also run troubleshoot_python.bat for more help.
    echo.
    pause
    exit /b 1
)

echo [INFO] Using Python command: %PY_CMD%
%PY_CMD% --version
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
REM  - --no-input avoids interactive prompts
REM  - --quiet keeps the console clean
REM  - output is logged to pip_install.log for debugging if needed
echo [INFO] Checking / installing required packages...
python -m pip install --no-input --quiet -r requirements.txt >pip_install.log 2>&1

if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    echo Possible causes:
    echo   - Unsupported / too old Python version (prefer 3.11–3.13)
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
