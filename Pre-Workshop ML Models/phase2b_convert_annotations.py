"""
Phase 2b — Convert LabelMe annotation JSONs -> single CSV manifest.
 
For each annotated frame, extracts:
  - jet_start (x1, y1)
  - jet_end   (x2, y2)
  - angle_deg  (degrees from vertical, computed from the two points)
  - split      (train / val / test, inferred from directory)
 
Output: data/annotations.csv
Errors: data/annotation_errors.csv  (frames with missing/bad labels)
 
Run after finishing all annotation in LabelMe:
    python phase2b_convert_annotations.py
"""
 
import sys
sys.stdout.reconfigure(encoding='utf-8')
 
import json
import csv
import os
import numpy as np
from pathlib import Path
 
 
LABELED_DIR  = "data/labeled"
OUTPUT_CSV   = "data/annotations.csv"
ERRORS_CSV   = "data/annotation_errors.csv"
SPLITS       = ["train", "val", "test"]
 
# Sanity check bounds — flag if points are suspiciously close or outside frame
MIN_JET_LENGTH_PX = 10   # jet shorter than this is probably a misclick
MAX_ANGLE_DEG     = 45   # jet angle beyond this from vertical is suspicious
 
 
def load_labelme_json(json_path: Path):
    """
    Parse a LabelMe JSON and return (start_xy, end_xy) or raise ValueError.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
 
    img_w = data.get("imageWidth",  None)
    img_h = data.get("imageHeight", None)
 
    points = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "").strip().lower()
        if label in ("jet_start", "jet_end"):
            pts = shape["points"]
            if len(pts) == 1:
                points[label] = (float(pts[0][0]), float(pts[0][1]))
            elif len(pts) >= 1:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                points[label] = (float(np.mean(xs)), float(np.mean(ys)))
 
    if "jet_start" not in points:
        raise ValueError("Missing label: jet_start")
    if "jet_end" not in points:
        raise ValueError("Missing label: jet_end")
 
    return points["jet_start"], points["jet_end"], img_w, img_h
 
 
def compute_angle(x1, y1, x2, y2):
    """
    Angle of the jet from the vertical axis, in degrees.
    0 deg = perfectly vertical, positive = tilted right, negative = tilted left.
    """
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = np.arctan2(dx, dy)
    return float(np.degrees(angle_rad))
 
 
def convert(
    labeled_dir: str = LABELED_DIR,
    output_csv:  str = OUTPUT_CSV,
    errors_csv:  str = ERRORS_CSV,
):
    rows   = []
    errors = []
 
    for split in SPLITS:
        split_dir = Path(labeled_dir) / split
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue
 
        json_files = sorted(split_dir.glob("*.json"))
        print(f"  {split:6s}: {len(json_files)} JSON files found")
 
        for jpath in json_files:
            img_name = jpath.stem + ".png"
 
            try:
                (x1, y1), (x2, y2), img_w, img_h = load_labelme_json(jpath)
 
                jet_len = np.hypot(x2 - x1, y2 - y1)
                angle   = compute_angle(x1, y1, x2, y2)
 
                warnings = []
                if jet_len < MIN_JET_LENGTH_PX:
                    warnings.append(f"jet too short ({jet_len:.1f}px)")
                if abs(angle) > MAX_ANGLE_DEG:
                    warnings.append(f"angle suspicious ({angle:.1f} deg)")
                if img_w and img_h:
                    for name, px, py in [("jet_start", x1, y1), ("jet_end", x2, y2)]:
                        if not (0 <= px <= img_w and 0 <= py <= img_h):
                            warnings.append(f"{name} outside frame bounds")
 
                rows.append({
                    "filename":   img_name,
                    "split":      split,
                    "x1":         f"{x1:.2f}",
                    "y1":         f"{y1:.2f}",
                    "x2":         f"{x2:.2f}",
                    "y2":         f"{y2:.2f}",
                    "angle_deg":  f"{angle:.4f}",
                    "jet_len_px": f"{jet_len:.1f}",
                    "img_w":      img_w or "",
                    "img_h":      img_h or "",
                    "warnings":   "; ".join(warnings),
                })
 
            except Exception as e:
                errors.append({
                    "filename": img_name,
                    "split":    split,
                    "json":     str(jpath),
                    "error":    str(e),
                })
 
    # Write main CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
 
    # Write errors CSV
    if errors:
        with open(errors_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=errors[0].keys())
            writer.writeheader()
            writer.writerows(errors)
 
    # Summary
    warned = [r for r in rows if r["warnings"]]
    print(f"\nResults:")
    print(f"  Annotated frames : {len(rows)}")
    print(f"  With warnings    : {len(warned)}")
    print(f"  Errors / skipped : {len(errors)}")
    print(f"\n  annotations.csv  -> '{output_csv}'")
    if errors:
        print(f"  errors.csv       -> '{errors_csv}'  <- review these")
    if warned:
        print(f"\n  Frames with warnings (review recommended):")
        for r in warned[:10]:
            print(f"    {r['filename']:30s}  {r['warnings']}")
        if len(warned) > 10:
            print(f"    ... and {len(warned)-10} more (see CSV)")
 
    if rows:
        angles = [float(r["angle_deg"]) for r in rows]
        print(f"\n  Angle distribution (degrees from vertical):")
        print(f"    Mean   : {np.mean(angles):.2f} deg")
        print(f"    Std    : {np.std(angles):.2f} deg")
        print(f"    Min    : {np.min(angles):.2f} deg")
        print(f"    Max    : {np.max(angles):.2f} deg")
 
    return rows, errors
 
 
def spot_check(n: int = 10, labeled_dir: str = LABELED_DIR, annotations_csv: str = OUTPUT_CSV):
    """
    Visually verify N random annotations. Press SPACE to advance, Q to quit.
    """
    import cv2
    import random
 
    with open(annotations_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
 
    sample = random.sample(all_rows, min(n, len(all_rows)))
    print(f"Spot-checking {len(sample)} frames. SPACE=next  Q=quit")
 
    for row in sample:
        split    = row["split"]
        img_path = Path(labeled_dir) / split / row["filename"]
        if not img_path.exists():
            print(f"  [missing] {img_path}")
            continue
 
        img = cv2.imread(str(img_path))
        x1, y1 = int(float(row["x1"])), int(float(row["y1"]))
        x2, y2 = int(float(row["x2"])), int(float(row["y2"]))
        angle   = float(row["angle_deg"])
 
        cv2.line(img,   (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (x1, y1), 6, (255, 100, 0), -1)
        cv2.circle(img, (x2, y2), 6, (0,   100, 255), -1)
        cv2.putText(img, f"{row['filename']}  angle={angle:.1f}deg",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
        if row["warnings"]:
            cv2.putText(img, f"WARN: {row['warnings']}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 1)
 
        cv2.imshow("Spot check — SPACE=next  Q=quit", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
 
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert LabelMe JSONs to annotations CSV.")
    parser.add_argument("--labeled",   default=LABELED_DIR, help="data/labeled/ directory")
    parser.add_argument("--output",    default=OUTPUT_CSV,  help="Output CSV path")
    parser.add_argument("--errors",    default=ERRORS_CSV,  help="Errors CSV path")
    parser.add_argument("--spotcheck", type=int, default=0, metavar="N",
                        help="After converting, visually verify N random frames")
    args = parser.parse_args()
 
    print("Converting annotations...")
    rows, errors = convert(args.labeled, args.output, args.errors)
 
    if args.spotcheck > 0 and rows:
        spot_check(args.spotcheck, args.labeled, args.output)