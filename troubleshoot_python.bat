@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   CSV Explorer - Python Troubleshooter
echo ========================================
echo.

REM ----------------------------------------
REM Detect Python (prefer py -3, fallback to python)
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
    echo [ERROR] No Python installation found via "py" or "python".
    echo.
    echo Please install Python 3 from:
    echo   https://www.python.org/downloads/windows/
    echo and check the options:
    echo   - "Use admin privileges when installing py.exe"
    echo   - "Add python.exe to PATH"
    echo.
    pause
    exit /b 1
)

echo [OK] Python command detected as: %PY_CMD%
echo.

REM Show Python version
echo [INFO] Python version:
call %PY_CMD% --version
echo.

REM ----------------------------------------
REM Check pip
REM ----------------------------------------
echo [CHECK] Checking pip...
call %PY_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [WARN] pip does not seem to be installed. Attempting to install pip...
    call %PY_CMD% -m ensurepip --default-pip
    if errorlevel 1 (
        echo [ERROR] Could not install pip automatically.
        echo Please reinstall or repair your Python installation.
        pause
        exit /b 1
    )
) else (
    echo [OK] pip is installed:
    call %PY_CMD% -m pip --version
)

echo.
echo [INFO] Upgrading pip to the latest version...
call %PY_CMD% -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARN] Failed to upgrade pip. You may need to run this as Administrator.
) else (
    echo [OK] pip successfully upgraded.
)

echo.
echo [INFO] Basic Python/pip setup looks OK.
echo You can now try running: run_app.bat
echo.
pause
endlocal
