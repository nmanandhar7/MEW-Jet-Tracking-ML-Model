param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment..."
& $Python -m venv .venv

Write-Host "Activating environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Run the annotator with:"
Write-Host "python \"Pre-Workshop ML Models\\jet_annotator_app.py\""
Write-Host ""
Write-Host "If no model is present, place a checkpoint in checkpoints/ or train one via the scripts in Pre-Workshop ML Models/."
