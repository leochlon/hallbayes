@echo off
cd /d "%~dp0"
"%SystemRoot%\System32\where.exe" python >nul 2>&1
IF ERRORLEVEL 1 (
  echo Python is not on PATH. Install Python 3 and retry.
  pause
  exit /b 1
)
python "%~dp0\..\app\launcher\entry.py"
pause

