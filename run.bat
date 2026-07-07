@echo off
REM Convenience wrapper: `run scan 76307.pdf --set 76307` instead of the long
REM `.venv\Scripts\python.exe -m legopartlocator.cli ...` path.
REM Works from any directory; %~dp0 is this script's own folder.

set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [run] Virtual environment not found at "%PY%".
  echo [run] Create it first:
  echo         py -m venv .venv
  echo         .venv\Scripts\python.exe -m pip install -e ".[dev]"
  exit /b 1
)

"%PY%" -m legopartlocator.cli %*
