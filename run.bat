@echo off
REM One-click start for Windows: creates .venv on first run, then serves http://127.0.0.1:8000
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv || goto :error
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
if not exist .env (
  copy .env.example .env >nul
  echo.
  echo Created .env. Paste your free Groq API key after LLM_API_KEY= ^(https://console.groq.com/keys^),
  echo or switch to the offline Ollama option described in the same file. Save it, then press a key here.
  start "" notepad .env
  pause
)
echo Starting DocChat on http://127.0.0.1:8000
start "" http://127.0.0.1:8000
.venv\Scripts\python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
goto :eof
:error
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
