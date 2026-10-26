# MEW-Jet-Tracking-ML-Model

ML tools for MEW (melt electrowriting) jet detection, annotation, and metric extraction.

This repository includes:
- a GUI annotator app for video inference,
- model inference and metrics utilities,
- training and data preparation scripts,
- a publication archive reference pipeline.

## Quick Start (PI Setup)

Use this section if you only want to run the annotator app.

### 1) Prerequisites

- Windows 10 or 11
- Python 3.11 installed
- Git installed

### 2) Install Python 3.11 (if not already installed)

Install Python 3.11 from one of these options:
- Python.org installer: https://www.python.org/downloads/release/python-3119/
- Windows package manager:

```powershell
winget install -e --id Python.Python.3.11
```

Verify the Python launcher can see 3.11:

```powershell
py -0p
```

The output should include a Python 3.11 entry.

### 3) Clone the repository

```powershell
git clone https://github.com/nmanandhar7/MEW-Jet-Tracking-ML-Model.git
cd MEW-Jet-Tracking-ML-Model
```

### 4) Create and activate environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `py -3.11` fails, install Python 3.11 first, then rerun the command.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 5) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6) Run the annotator app

```powershell
python "Pre-Workshop ML Models\jet_annotator_app.py"
```

In the app:
1. Choose an input video.
2. Choose or confirm model path.
3. Choose output folder.
4. Click Process Video.

Outputs:
- annotated video
- CSV metrics file

## One-command setup option

You can also run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_mew_annotator.ps1
```

Then launch:

```powershell
python "Pre-Workshop ML Models\jet_annotator_app.py"
```

## Model file expectations

The app expects a segmentation model path, commonly under the checkpoints folder, for example:

```text
checkpoints/unet_mobilenet.onnx
```

If no model file is present, copy a valid trained model into the checkpoints folder and select it in the app.

## Repository layout

- Pre-Workshop ML Models: main scripts and annotator app
- scripts: setup helpers
- checkpoints: local model storage
- data: local datasets and annotations
- inference: local generated outputs

## Notes about large files

Large datasets, videos, checkpoints, and generated outputs are intentionally excluded from Git tracking for lightweight sharing. Add those assets locally as needed for inference/training.

## Troubleshooting

### ModuleNotFoundError when launching app

Ensure the virtual environment is active and dependencies are installed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### NumPy build/compiler error on Windows

If you see an error like `Unknown compiler(s)` while installing `numpy`, your environment is likely using an unsupported Python version (for example Python 3.14), which forces a source build.

Fix:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then verify:

```powershell
python -c "import cv2, numpy, pandas, torch, onnxruntime, PIL, skimage; print('imports ok')"
```

### App opens but processing fails

- Verify model path is correct and file exists.
- Test with a short video first.
- Confirm output folder is writable.

### Verify install quickly

```powershell
python -c "import cv2, numpy, pandas, torch, onnxruntime, PIL, skimage; print('imports ok')"
```

If this prints imports ok, core dependencies are available.
