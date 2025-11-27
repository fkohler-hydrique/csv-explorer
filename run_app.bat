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

REM ----------------------------------------
REM Create venv if it does not exist
REM ----------------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    py -3 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python 3 is installed and "py" command works.
        pause
        exit /b 1
    )
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
REM Upgrade pip and install dependencies
REM ----------------------------------------
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

echo [INFO] Installing required packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    pause
    exit /b 1
)

REM ----------------------------------------
REM Run the Streamlit app
REM ----------------------------------------
echo [INFO] Starting CSV Explorer app...
streamlit run app.py

REM ----------------------------------------
REM Keep window open after Streamlit exits
REM ----------------------------------------
echo.
echo [INFO] Streamlit app has stopped.
pause
endlocal
