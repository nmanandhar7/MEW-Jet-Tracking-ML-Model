param(
    [string]$PythonVersion = "3.11"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment with Python $PythonVersion..."
& py "-$PythonVersion" -m venv .venv

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Failed to create .venv with Python $PythonVersion."
    Write-Host "Install Python $PythonVersion and retry, or pass a different version:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts/setup_mew_annotator.ps1 -PythonVersion 3.10"
    exit 1
}

Write-Host "Activating environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the annotator with:"
Write-Host "python \"Pre-Workshop ML Models\\jet_annotator_app.py\""
Write-Host ""
Write-Host "If no model is present, place a checkpoint in checkpoints/ or train one via the scripts in Pre-Workshop ML Models/."
