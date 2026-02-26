@echo off
REM Aviation RAG Web Interface - Startup Script (Windows)

echo 🛩️  Aviation RAG Web Interface - Starting...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check if .env exists
if not exist ".env" (
    echo ⚠️  Warning: .env file not found!
    echo Creating .env from .env.example...
    copy env.example .env
    echo.
    echo ⚠️  IMPORTANT: Please edit .env file and set your API_KEY!
    echo.
)

REM Start the application
echo Starting application...
echo Access the web interface at: http://localhost:8001
echo.
python main.py

pause
