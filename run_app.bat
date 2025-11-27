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

REM Marker file to know if deps are already installed
set DEPS_FLAG=.deps_installed

echo ========================================
echo   CSV Explorer - Launcher
echo ========================================
echo.

REM ----------------------------------------
REM Detect Python command
REM ----------------------------------------
set "PY_CMD=py -3"
%PY_CMD% --version >nul 2>&1
if errorlevel 1 (
    set "PY_CMD=python"
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No Python installation found via "py -3" or "python".
        echo.
        echo Please install Python 3.11+ from:
        echo   https://www.python.org/downloads/windows/
        echo and make sure either:
        echo   - the "py" launcher is installed, or
        echo   - "python.exe" is added to PATH.
        echo.
        echo You can also run troubleshoot_python.bat for more help.
        echo.
        pause
        exit /b 1
    )
)

echo [INFO] Using Python command:
%PY_CMD% --version
echo.

REM ----------------------------------------
REM Python version + architecture checks
REM (Require: 3.11 <= Python <= 3.13, 64-bit)
REM ----------------------------------------
set "PY_MAJOR="
set "PY_MINOR="
set "PY_PATCH="
set "PY_ARCH="
set "PY_BITS="

for /f %%A in ('%PY_CMD% -c "import sys; print(sys.version_info[0])"') do set PY_MAJOR=%%A
for /f %%A in ('%PY_CMD% -c "import sys; print(sys.version_info[1])"') do set PY_MINOR=%%A
for /f %%A in ('%PY_CMD% -c "import sys; print(sys.version_info[2])"') do set PY_PATCH=%%A

if not defined PY_MAJOR goto PY_INFO_ERROR
if not defined PY_MINOR goto PY_INFO_ERROR
if not defined PY_PATCH goto PY_INFO_ERROR

for /f %%A in ('%PY_CMD% -c "import platform; print(platform.architecture()[0])"') do set PY_ARCH=%%A
if not defined PY_ARCH goto PY_INFO_ERROR

set "PY_BITS=unknown"
if /I "%PY_ARCH%"=="64bit" set "PY_BITS=64"
if /I "%PY_ARCH%"=="32bit" set "PY_BITS=32"
if "%PY_BITS%"=="unknown" goto PY_INFO_ERROR

echo [INFO] Detected Python version: %PY_MAJOR%.%PY_MINOR%.%PY_PATCH% (%PY_BITS%-bit)
echo.

REM Require Python 3.x
if not "%PY_MAJOR%"=="3" goto PY_VERSION_TOO_OLD

REM Too old if < 3.11
if %PY_MINOR% LSS 11 goto PY_VERSION_TOO_OLD

REM Too high if > 3.13 (we allow 3.11, 3.12, 3.13)
if %PY_MINOR% GTR 13 goto PY_VERSION_TOO_HIGH

REM 32-bit not supported
if "%PY_BITS%"=="32" goto PY_32BIT

REM ----------------------------------------
REM Step 1/3: Create venv if needed
REM ----------------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [SETUP] Step 1/3: Creating virtual environment in %VENV_DIR% ...
    echo          This can take a little while the first time.
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python is correctly installed.
        echo You can also run troubleshoot_python.bat for more help.
        echo.
        pause
        exit /b 1
    )
    echo [SETUP] Step 1/3: Virtual environment created successfully.
    echo.
) else (
    echo [INFO] Using existing virtual environment in %VENV_DIR%.
    echo.
)

REM ----------------------------------------
REM Activate the virtual environment
REM ----------------------------------------
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to activate virtual environment.
    echo.
    pause
    exit /b 1
)
echo.

REM ----------------------------------------
REM Step 2/3 & 3/3: Upgrade pip + install deps (first run only)
REM ----------------------------------------
if not exist "%DEPS_FLAG%" (
    echo [SETUP] Step 2/3: Upgrading pip...
    echo          (Details are written to pip_upgrade.log)
    python -m pip install --upgrade pip --disable-pip-version-check >pip_upgrade.log 2>&1
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to upgrade pip.
        echo   See pip_upgrade.log for details.
        echo.
        pause
        exit /b 1
    )
    echo [SETUP] Step 2/3: Pip upgraded successfully.
    echo.

    echo [SETUP] Step 3/3: Installing required packages from requirements.txt...
    echo          (Details are written to pip_install.log)
    echo.

    REM Clean any old temp file for exit code
    del pip_exit_code.tmp >nul 2>&1

    REM Run pip in a background cmd with delayed expansion for !ERRORLEVEL!
    start "" /B cmd /V:ON /C "pip install -r requirements.txt --disable-pip-version-check -q >pip_install.log 2>&1 & echo !ERRORLEVEL!>pip_exit_code.tmp"

    REM ---------- Spinner while pip is running (single-line) ----------
    REM Get a backspace character
    for /F %%A in ('"prompt $H & for %%B in (1) do rem"') do set "BS=%%A"

    setlocal EnableDelayedExpansion
    set "SPINNER=-\|/"
    set "IDX=0"

    REM Initial line
    <nul set /p "=Installing packages, please wait... -"

:wait_for_pip
    if exist pip_exit_code.tmp goto pip_done

    set /a IDX=(IDX+1)%%4
    set "CH=!SPINNER:~%IDX%,1!"
    REM Move one char back and print new spinner symbol on same line
    <nul set /p "=%BS%!CH!"

    REM small delay (~1 second)
    ping -n 2 127.0.0.1 >nul
    goto wait_for_pip

:pip_done
    endlocal
    echo.
    REM ---------- End spinner section ----------

    REM Read pip exit code from file
    set "EXIT_CODE=1"
    for /f "usebackq" %%E in ("pip_exit_code.tmp") do set "EXIT_CODE=%%E"
    del pip_exit_code.tmp >nul 2>&1

    if not "%EXIT_CODE%"=="0" (
        echo.
        echo [ERROR] Failed to install required packages.
        echo   See pip_install.log for details.
        echo.
        pause
        exit /b 1
    )

    echo [SETUP] Step 3/3: All dependencies installed successfully.
    echo.

    REM Create marker so we don't install again next time
    type nul > "%DEPS_FLAG%"
) else (
    echo [INFO] Python dependencies already installed. Skipping pip install.
    echo.
)

REM ----------------------------------------
REM Clean screen before starting the app
REM ----------------------------------------
cls
echo ========================================
echo   CSV Explorer - Launcher
echo ========================================
echo.
echo [INFO] Setup complete. Starting CSV Explorer app...
echo     If your browser does not open automatically, go to: http://localhost:8501
echo.

REM ----------------------------------------
REM Run the Streamlit app
REM ----------------------------------------
streamlit run app.py

REM ----------------------------------------
REM Keep window open after Streamlit exits
REM ----------------------------------------
echo.
echo [INFO] Streamlit app has stopped.
pause
endlocal
exit /b 0

REM ----------------------------------------
REM Error handlers for version / arch
REM ----------------------------------------

:PY_INFO_ERROR
echo.
echo [ERROR] Could not determine Python version or architecture.
echo       Something looks wrong with the Python installation.
echo.
echo Please try:
echo   - Running troubleshoot_python.bat
echo   - Or reinstalling Python 3.11–3.13 (64-bit) from python.org
echo.
pause
exit /b 1

:PY_VERSION_TOO_OLD
echo.
echo [ERROR] Python version is too old: %PY_MAJOR%.%PY_MINOR%.%PY_PATCH%
echo       This app requires Python 3.11 or newer (64-bit).
echo.
echo Please install Python 3.11–3.13 (64-bit) from:
echo   https://www.python.org/downloads/windows/
echo.
pause
exit /b 1

:PY_VERSION_TOO_HIGH
echo.
echo [ERROR] Python version is newer than tested: %PY_MAJOR%.%PY_MINOR%.%PY_PATCH%
echo       This app is tested with Python 3.11–3.13 (64-bit).
echo       Newer versions may cause dependency issues.
echo.
echo Please install Python 3.11–3.13 (64-bit) from python.org
echo and use that interpreter for this app.
echo.
pause
exit /b 1

:PY_32BIT
echo.
echo [ERROR] 32-bit Python detected.
echo       32-bit Python is not supported for this app.
echo       Libraries like pandas expect 64-bit Python on Windows.
echo.
echo Please:
echo   1) Uninstall 32-bit Python
echo   2) Install Python 3.11–3.13 (64-bit) from python.org
echo   3) Run this launcher again
echo.
pause
exit /b 1
