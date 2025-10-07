@echo off
echo ======================================
echo SURADATA Backend Startup Script
echo ======================================

:: Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed
    pause
    exit /b 1
)

echo Python version:
python --version

:: Create virtual environment if not exists
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

:: Verify installations
echo.
echo Verifying critical packages...
python -c "import fastapi; print('✓ FastAPI:', fastapi.__version__)" 2>nul || echo ✗ FastAPI not installed
python -c "import uvicorn; print('✓ Uvicorn installed')" 2>nul || echo ✗ Uvicorn not installed
python -c "from config import Config; print('✓ Config validated')" 2>nul || echo ✗ Config error

:: Check if port is in use
netstat -ano | findstr :8000 >nul
if %ERRORLEVEL% EQU 0 (
    echo.
    echo WARNING: Port 8000 is already in use
    echo Kill the process or use a different port
    pause
    exit /b 1
)

:: Start server
echo.
echo ======================================
echo Starting SURADATA API Server...
echo ======================================
echo API will be available at: http://localhost:8000
echo Documentation: http://localhost:8000/docs
echo Press CTRL+C to stop the server
echo ======================================
echo.

uvicorn api:app --reload --host 0.0.0.0 --port 8000