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
- Python 3.10 or 3.11 installed
- Git installed

### 2) Clone the repository

```powershell
git clone https://github.com/nmanandhar7/MEW-Jet-Tracking-ML-Model.git
cd MEW-Jet-Tracking-ML-Model
```

### 3) Create and activate environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 4) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5) Run the annotator app

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

### App opens but processing fails

- Verify model path is correct and file exists.
- Test with a short video first.
- Confirm output folder is writable.

### Verify install quickly

```powershell
python -c "import cv2, numpy, pandas, torch, onnxruntime, PIL, skimage; print('imports ok')"
```

If this prints imports ok, core dependencies are available.
