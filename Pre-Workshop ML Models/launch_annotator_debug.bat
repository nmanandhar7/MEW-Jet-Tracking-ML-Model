@echo off
REM ============================================================
REM  MEW Jet Annotator launcher (DEBUG)
REM  Use this if the normal launcher does nothing when clicked.
REM  It keeps a console window open so you can read any error.
REM ============================================================

cd /d "%~dp0"

echo Launching Jet Annotator (debug mode)...
echo.
".venv\Scripts\python.exe" "jet_annotator_app.py" %*

echo.
echo ------------------------------------------------------------
echo App closed. If there was an error, it is shown above.
echo ------------------------------------------------------------
pause
