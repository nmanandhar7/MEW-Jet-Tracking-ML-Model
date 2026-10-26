# ============================================================
#  Creates a Desktop shortcut "MEW Jet Annotator" that opens
#  the app directly (no console window) with the custom icon.
#
#  Run it once. Right-click this file -> "Run with PowerShell",
#  or in a terminal:
#      powershell -ExecutionPolicy Bypass -File make_shortcut.ps1
#
#  Keep this script in the project root (next to jet_annotator_app.py).
# ============================================================

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Prefer windowed python (no console); fall back to python.exe
$pythonw = Join-Path $here ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) {
    $pythonw = Join-Path $here ".venv\Scripts\python.exe"
}
if (-not (Test-Path $pythonw)) {
    Write-Host "Could not find python in .venv\Scripts. Is the virtual environment set up?" -ForegroundColor Red
    exit 1
}

$script = Join-Path $here "jet_annotator_app.py"
$icon   = Join-Path $here "jet_annotator.ico"

$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "MEW Jet Annotator.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($lnkPath)
$sc.TargetPath       = $pythonw
$sc.Arguments        = '"jet_annotator_app.py"'
$sc.WorkingDirectory = $here
$sc.WindowStyle      = 1
$sc.Description       = "MEW Jet Annotator"
if (Test-Path $icon) {
    $sc.IconLocation = $icon
}
$sc.Save()

Write-Host "Created shortcut:" -ForegroundColor Green
Write-Host "    $lnkPath"
Write-Host ""
Write-Host "Double-click it to open the annotator. You can also drag a video"
Write-Host "file onto the shortcut to open the app with that video pre-loaded."
Write-Host ""
Write-Host "Tip: right-click the shortcut -> Pin to Taskbar / Pin to Start."
