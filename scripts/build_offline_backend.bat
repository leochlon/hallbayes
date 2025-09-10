@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

set PY=%PYTHON_BIN%
if "%PY%"=="" set PY=python
set VENV_DIR=%BUILD_VENV_DIR%
if "%VENV_DIR%"=="" set VENV_DIR=.build-venv

echo Using bootstrap Python: %PY%
echo Build venv: %VENV_DIR%

if not exist "%VENV_DIR%" (
  %PY% -m venv "%VENV_DIR%" || goto :error
)

set VENV_PY="%VENV_DIR%\Scripts\python.exe"
echo Using venv Python: %VENV_PY%

%VENV_PY% -m pip install --upgrade pip wheel setuptools || goto :error
%VENV_PY% -m pip install pyinstaller || goto :error
%VENV_PY% -m pip install -r requirements.txt || goto :error

%VENV_PY% -m PyInstaller --noconfirm --onefile scripts\backend_entry.py ^
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
