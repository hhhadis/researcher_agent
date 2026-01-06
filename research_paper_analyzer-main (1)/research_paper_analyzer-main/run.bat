@echo off
echo ========================================
echo Research Logic Graph Extractor
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo.
echo Starting Streamlit application...
streamlit run app.py

pause

