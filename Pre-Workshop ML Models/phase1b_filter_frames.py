"""
Phase 1b — Filter blurry / occluded frames from raw_frames/.

Strategy:
  1. Laplacian variance  -> measures sharpness (low = blurry)
  2. Jet presence check  -> ensures a dark vertical structure exists in ROI
  3. Brightness check    -> rejects over/under-exposed frames

Frames passing all checks are COPIED to data/filtered_frames/.
A CSV report is written so you can review borderline cases.
"""

import cv2
import numpy as np
import os
import shutil
import csv
from pathlib import Path
from typing import Optional


# ── tuneable thresholds ────────────────────────────────────────────────────────
BLUR_THRESHOLD      = 50.0   # Laplacian variance below this → blurry (lower = stricter)
MIN_BRIGHTNESS      = 20.0   # Mean pixel value below this → too dark
MAX_BRIGHTNESS      = 240.0  # Mean pixel value above this → overexposed
JET_MIN_DARK_PIXELS = 50     # Minimum dark pixels in jet ROI to confirm jet is present
JET_DARK_THRESHOLD  = 60     # Pixel value below this counts as "dark" (jet material)

# Jet ROI — region where the jet is expected (fraction of frame, adjust to your setup)
# Format: (x_start_frac, y_start_frac, x_end_frac, y_end_frac)
# Default: central vertical strip, middle 60% of height
JET_ROI_FRAC = (0.40, 0.45, 0.60, 0.90)
# ──────────────────────────────────────────────────────────────────────────────


def laplacian_variance(gray: np.ndarray) -> float:
    """Higher = sharper. Below ~80 is typically blurry for microscopy footage."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def jet_present(gray: np.ndarray, roi_frac: tuple, dark_thresh: int, min_pixels: int) -> bool:
    """Check that a minimum number of dark pixels exist in the expected jet region."""
    h, w = gray.shape
    x0 = int(roi_frac[0] * w)
    y0 = int(roi_frac[1] * h)
    x1 = int(roi_frac[2] * w)
    y1 = int(roi_frac[3] * h)
    roi = gray[y0:y1, x0:x1]
    dark_pixels = int(np.sum(roi < dark_thresh))
    return dark_pixels >= min_pixels


def filter_frames(
    input_dir:  str = "data/raw_frames",
    output_dir: str = "data/filtered_frames",
    blur_thresh: float = BLUR_THRESHOLD,
    min_bright:  float = MIN_BRIGHTNESS,
    max_bright:  float = MAX_BRIGHTNESS,
    jet_roi:     tuple = JET_ROI_FRAC,
    jet_dark_thresh: int = JET_DARK_THRESHOLD,
    jet_min_px:  int  = JET_MIN_DARK_PIXELS,
    report_csv:  str  = "data/filter_report.csv",
    max_frames:  int  = None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(report_csv).parent.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(Path(input_dir).glob("*.png"))
    if max_frames:
        frame_files = frame_files[:max_frames]
    if not frame_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    print(f"Filtering {len(frame_files)} frames from '{input_dir}' ...")
    print(f"  Blur threshold      : >{blur_thresh}")
    print(f"  Brightness range    : {min_bright} – {max_bright}")
    print(f"  Jet dark pixels min : {jet_min_px} (thresh={jet_dark_thresh})")
    print()

    passed = 0
    rejected_blur = 0
    rejected_bright = 0
    rejected_jet = 0

    rows = []

    for fpath in frame_files:
        img  = cv2.imread(str(fpath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lap     = laplacian_variance(gray)
        bright  = float(gray.mean())
        has_jet = jet_present(gray, jet_roi, jet_dark_thresh, jet_min_px)

        fail_reason = None
        if lap < blur_thresh:
            fail_reason = "blurry"
            rejected_blur += 1
        elif bright < min_bright:
            fail_reason = "too_dark"
            rejected_bright += 1
        elif bright > max_bright:
            fail_reason = "overexposed"
            rejected_bright += 1
        elif not has_jet:
            fail_reason = "no_jet_detected"
            rejected_jet += 1

        status = "pass" if fail_reason is None else "fail"
        rows.append({
            "filename":    fpath.name,
            "status":      status,
            "reason":      fail_reason or "",
            "sharpness":   f"{lap:.1f}",
            "brightness":  f"{bright:.1f}",
            "jet_present": str(has_jet),
        })

        if status == "pass":
            shutil.copy(str(fpath), os.path.join(output_dir, fpath.name))
            passed += 1

    # Write CSV report
    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = len(frame_files)
    print(f"Results:")
    print(f"  Passed          : {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"  Rejected (blur) : {rejected_blur}")
    print(f"  Rejected (light): {rejected_bright}")
    print(f"  Rejected (no jet): {rejected_jet}")
    print(f"\nFiltered frames → '{output_dir}/'")
    print(f"Report CSV       → '{report_csv}'")
    return passed


def visualize_thresholds(input_dir: str = "data/raw_frames", n_samples: int = 6):
    """
    Show a grid of random frames with their sharpness / brightness scores.
    Useful for calibrating the thresholds before committing to a filter run.
    """
    import random
    files = list(Path(input_dir).glob("*.png"))
    if not files:
        print("No frames found.")
        return
    samples = random.sample(files, min(n_samples, len(files)))

    for fpath in samples:
        img  = cv2.imread(str(fpath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap    = laplacian_variance(gray)
        bright = float(gray.mean())
        label  = f"sharp={lap:.0f}  bright={bright:.0f}"
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow(fpath.name, img)

    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter blurry/occluded frames.")
    parser.add_argument("--input",  default="data/raw_frames",      help="Raw frames directory")
    parser.add_argument("--output", default="data/filtered_frames", help="Output directory")
    parser.add_argument("--blur",   type=float, default=BLUR_THRESHOLD, help="Min sharpness score")
    parser.add_argument("--max",    type=int, default=None, help="Max frames to process")
    parser.add_argument("--preview", type=int, nargs='?', const=6, help="Preview N sample frames with scores (default 6)")
    args = parser.parse_args()

    if args.preview is not None:
        visualize_thresholds(args.input, n_samples=args.preview)
    else:
        filter_frames(input_dir=args.input, output_dir=args.output, blur_thresh=args.blur, max_frames=args.max)
