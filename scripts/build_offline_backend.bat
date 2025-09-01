@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set PY=%PYTHON_BIN%
if "%PY%"=="" set PY=python

echo Using Python: %PY%

%PY% -m pip install --upgrade pip wheel setuptools pyinstaller || goto :error
%PY% -m pip install -r requirements.txt || goto :error

pyinstaller --noconfirm --onefile scripts\backend_entry.py ^
  --add-data app;app ^
  --add-data scripts;scripts || goto :error

if exist dist\backend_entry.exe (
  if not exist bin mkdir bin
  move /Y dist\backend_entry.exe bin\hallucination-backend.exe >nul
  echo Built offline backend: bin\hallucination-backend.exe
) else (
  echo Build failed: dist\backend_entry.exe not found
  goto :error
)

exit /b 0

:error
echo Build failed.
exit /b 1
