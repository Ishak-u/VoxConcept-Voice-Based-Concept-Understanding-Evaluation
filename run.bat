@echo off
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -3.11 -m venv venv
  if errorlevel 1 (
    echo Failed to create the virtual environment. Install Python 3.11 and try again.
    pause
    exit /b 1
  )
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run "Source Code\app.py"
