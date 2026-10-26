@echo off
REM ============================================================
REM  MEW Jet Annotator launcher
REM  Double-click this file to open the app (no console window).
REM  Keep it in the project root (same folder as jet_annotator_app.py
REM  and the .venv folder).
REM ============================================================

REM cd to this file's own folder, so relative paths (checkpoints, .venv) resolve
cd /d "%~dp0"

REM Prefer pythonw.exe (windowed, no console). Fall back to python.exe.
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "jet_annotator_app.py" %*
) else (
    start "" ".venv\Scripts\python.exe" "jet_annotator_app.py" %*
)
