"""
Jet Skeleton Extractor
----------------------
Extracts the jet centerline from each annotated frame using:
  1. ROI crop around the jet (using jet_start / jet_end from annotations.csv)
  2. Adaptive thresholding to isolate dark jet pixels
  3. Skeletonization to reduce jet to 1px centerline
  4. Ordered point extraction top-to-bottom

Outputs:
  - LabelMe-compatible JSON per frame (linestrip label "jet_skeleton")
  - jet_metrics.csv with per-frame shape parameters:
      arc_length, chord_length, straightness_ratio,
      launch_angle, contact_angle, max_deflection, bending_energy

Run from your project root:
    python jet_skeleton_extractor.py

Optional flags:
    --split train          (only process one split)
    --preview              (show visualization window per frame, press any key)
    --no-json              (skip saving JSONs, only compute metrics CSV)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize

# ── Config ────────────────────────────────────────────────────────────────────
ANNOTATIONS  = "data/annotations.csv"
LABELED_DIR  = "data/labeled"
METRICS_CSV  = "data/jet_metrics.csv"
SPLITS       = ["train", "val", "test"]

ROI_PAD_X    = 35    # horizontal padding around jet bounding box (px)
ROI_PAD_Y    = 5     # vertical padding above jet_start (px)
TARGET_PTS   = 30    # approx number of skeleton points to save per frame
MIN_SKEL_PTS = 8     # reject frame if fewer skeleton points than this
# ──────────────────────────────────────────────────────────────────────────────


# ── Geometry helpers ──────────────────────────────────────────────────────────

def segment_angle(p1, p2):
    """Angle of segment p1->p2 from vertical, in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dx, dy)))


def point_to_line_dist(pt, line_start, line_end):
    """Perpendicular distance from pt to infinite line through line_start/line_end."""
    x0, y0 = pt
    x1, y1 = line_start
    x2, y2 = line_end
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = np.hypot(y2-y1, x2-x1)
    return float(num / den) if den > 0 else 0.0


def compute_metrics(points):
    """
    Compute jet shape metrics from ordered list of (x,y) points.
    Returns a dict of metric name -> value.
    """
    if len(points) < 3:
        return {}

    pts = np.array(points, dtype=float)

    # Segment lengths
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:,0], diffs[:,1])

    # Arc length and chord length
    arc_length   = float(seg_lengths.sum())
    chord_length = float(np.hypot(*(pts[-1] - pts[0])))
    straightness = arc_length / chord_length if chord_length > 0 else 1.0

    # Arc position (cumulative length at each point)
    arc_pos = np.concatenate([[0], np.cumsum(seg_lengths)])

    # Launch and contact angles (from first/last segment)
    launch_angle  = segment_angle(pts[0], pts[1])
    contact_angle = segment_angle(pts[-2], pts[-1])
    total_angle_change = float(sum(
        abs(segment_angle(pts[i], pts[i+1]) - segment_angle(pts[i-1], pts[i]))
        for i in range(1, len(pts)-1)
    ))

    # Max deflection (perpendicular distance from chord)
    max_deflection = float(max(
        point_to_line_dist(pts[i], pts[0], pts[-1])
        for i in range(1, len(pts)-1)
    )) if len(pts) > 2 else 0.0

    # Curvature and bending energy
    curvatures = []
    for i in range(1, len(pts)-1):
        v1 = pts[i]   - pts[i-1]
        v2 = pts[i+1] - pts[i]
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) ** 2
        curvatures.append(cross / denom if denom > 0 else 0.0)

    bending_energy = float(sum(k**2 * ds for k, ds in zip(curvatures, seg_lengths[1:])))

    return {
        "arc_length":         round(arc_length, 2),
        "chord_length":       round(chord_length, 2),
        "straightness_ratio": round(straightness, 4),
        "launch_angle_deg":   round(launch_angle, 3),
        "contact_angle_deg":  round(contact_angle, 3),
        "total_angle_change": round(total_angle_change, 3),
        "max_deflection_px":  round(max_deflection, 2),
        "bending_energy":     round(bending_energy, 6),
        "n_skeleton_pts":     len(points),
    }


# ── Skeleton extraction ───────────────────────────────────────────────────────

def extract_skeleton(gray, x1, y1, x2, y2, img_w, img_h):
    """
    Extract ordered jet centerline from grayscale frame.

    Args:
        gray      : full grayscale image
        x1,y1     : jet_start pixel coordinates
        x2,y2     : jet_end pixel coordinates
        img_w/h   : full frame dimensions

    Returns:
        List of (x,y) points in full image coordinates, top to bottom.
        None if extraction failed.
    """
    # Build ROI — bounding box around jet with padding
    roi_x1 = max(0,       int(min(x1, x2)) - ROI_PAD_X)
    roi_x2 = min(img_w,   int(max(x1, x2)) + ROI_PAD_X)
    roi_y1 = max(0,       int(y1) - ROI_PAD_Y)
    roi_y2 = min(img_h - 5, int(y2))   # stop above substrate

    if roi_x2 - roi_x1 < 5 or roi_y2 - roi_y1 < 5:
        return None

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    # Adaptive threshold — handles varying contrast across frames
    thresh = cv2.adaptiveThreshold(
        roi, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=3
    )

    # Remove tiny noise blobs
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Skeletonize
    skel = skeletonize(thresh > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)

    if len(ys) < MIN_SKEL_PTS:
        return None

    # Sort top to bottom
    order = ys.argsort()
    ys, xs = ys[order], xs[order]

    # Convert from ROI coords to full image coords
    full_xs = xs + roi_x1
    full_ys = ys + roi_y1

    # Downsample to ~TARGET_PTS evenly spaced points
    step = max(1, len(full_xs) // TARGET_PTS)
    full_xs = full_xs[::step]
    full_ys = full_ys[::step]

    return [(int(x), int(y)) for x, y in zip(full_xs, full_ys)]


# ── LabelMe JSON ─────────────────────────────────────────────────────────────

def save_labelme_json(json_path, img_path, img_w, img_h, points, label="jet_skeleton"):
    """Save skeleton as a LabelMe linestrip JSON."""
    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": [[float(x), float(y)] for x, y in points],
                "group_id": None,
                "description": "",
                "shape_type": "linestrip",
                "flags": {},
                "mask": None
            }
        ],
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth":  img_w,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_skeleton(img, points, jet_start, jet_end):
    """Draw skeleton overlay on image."""
    out = img.copy()
    for i in range(len(points) - 1):
        cv2.line(out, points[i], points[i+1], (0, 220, 0), 2)
    for p in points:
        cv2.circle(out, p, 2, (0, 180, 0), -1)
    cv2.circle(out, (int(jet_start[0]), int(jet_start[1])), 6, (255, 100, 0), -1)
    cv2.circle(out, (int(jet_end[0]),   int(jet_end[1])),   6, (0,  60, 255), -1)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def run(splits_to_process, annotations_csv, labeled_dir,
        save_json=True, preview=False):

    with open(annotations_csv, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if r["split"] in splits_to_process]
    print(f"Processing {len(rows)} frames across splits: {splits_to_process}\n")

    metrics_rows = []
    success  = 0
    failed   = 0
    skipped  = 0

    for i, row in enumerate(rows):
        split     = row["split"]
        filename  = row["filename"]
        img_path  = Path(labeled_dir) / split / filename
        json_path = img_path.with_suffix(".json")

        # If JSON already has a skeleton, read it and compute metrics
        if save_json and json_path.exists():
            try:
                existing = json.loads(json_path.read_text(encoding="utf-8"))
                shapes = existing.get("shapes", [])
                skel_shape = next(
                    (s for s in shapes if s.get("label") == "jet_skeleton"), None
                )
                if skel_shape:
                    pts = [(int(p[0]), int(p[1])) for p in skel_shape["points"]]
                    metrics = compute_metrics(pts)
                    metrics_rows.append({
                        "filename": filename,
                        "split":    split,
                        "status":   "ok",
                        **metrics,
                    })
                    skipped += 1
                    continue
            except Exception:
                pass

        if not img_path.exists():
            print(f"  [missing] {img_path}")
            failed += 1
            continue

        # Load image
        img  = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape

        x1, y1 = float(row["x1"]), float(row["y1"])
        x2, y2 = float(row["x2"]), float(row["y2"])

        # Extract skeleton
        points = extract_skeleton(gray, x1, y1, x2, y2, img_w, img_h)

        if points is None or len(points) < MIN_SKEL_PTS:
            print(f"  [failed]  {filename} — too few skeleton points")
            failed += 1
            # Still record angle from annotations
            metrics_rows.append({
                "filename": filename,
                "split": split,
                "status": "failed",
                **{k: "" for k in [
                    "arc_length","chord_length","straightness_ratio",
                    "launch_angle_deg","contact_angle_deg",
                    "total_angle_change","max_deflection_px",
                    "bending_energy","n_skeleton_pts"
                ]}
            })
            continue

        # Compute metrics
        metrics = compute_metrics(points)

        metrics_rows.append({
            "filename": filename,
            "split":    split,
            "status":   "ok",
            **metrics,
        })

        # Save LabelMe JSON
        if save_json:
            save_labelme_json(json_path, img_path, img_w, img_h, points)

        success += 1

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(rows)}] {filename} "
                  f"— {len(points)} pts  "
                  f"straight={metrics.get('straightness_ratio', 0):.3f}  "
                  f"contact={metrics.get('contact_angle_deg', 0):.1f} deg")

        # Preview window
        if preview:
            vis = draw_skeleton(img, points, (x1, y1), (x2, y2))
            cv2.putText(vis,
                f"{filename}  straight={metrics.get('straightness_ratio',0):.3f}  "
                f"contact={metrics.get('contact_angle_deg',0):.1f}deg",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
            cv2.imshow("Skeleton preview — any key = next, Q = quit", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    if preview:
        cv2.destroyAllWindows()

    # Write metrics CSV — append if file exists, create fresh if not
    Path(METRICS_CSV).parent.mkdir(parents=True, exist_ok=True)
    if metrics_rows:
        fieldnames = list(metrics_rows[0].keys())
        csv_path = Path(METRICS_CSV)
        file_exists = csv_path.exists()
        mode = "a" if file_exists else "w"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(metrics_rows)
        action = "Appended to" if file_exists else "Created"
        print(f"\n  {action} metrics CSV -> '{METRICS_CSV}'")

    print(f"\nDone.")
    print(f"  Success : {success}")
    print(f"  Skipped : {skipped} (already had skeleton)")
    print(f"  Failed  : {failed}")

    # Print summary stats
    ok_rows = [r for r in metrics_rows if r["status"] == "ok"]
    if ok_rows:
        straights = [float(r["straightness_ratio"]) for r in ok_rows]
        contacts  = [float(r["contact_angle_deg"])  for r in ok_rows]
        deflects  = [float(r["max_deflection_px"])  for r in ok_rows]
        print(f"\nMetric summary across {len(ok_rows)} frames:")
        print(f"  Straightness ratio : mean={np.mean(straights):.3f}  "
              f"std={np.std(straights):.3f}  "
              f"range=[{np.min(straights):.3f}, {np.max(straights):.3f}]")
        print(f"  Contact angle (deg): mean={np.mean(contacts):.1f}  "
              f"std={np.std(contacts):.1f}  "
              f"range=[{np.min(contacts):.1f}, {np.max(contacts):.1f}]")
        print(f"  Max deflection (px): mean={np.mean(deflects):.1f}  "
              f"std={np.std(deflects):.1f}  "
              f"range=[{np.min(deflects):.1f}, {np.max(deflects):.1f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract jet skeleton centerline and metrics from annotated frames."
    )
    parser.add_argument("--split",   choices=["train","val","test","all"],
                        default="all", help="Which split to process")
    parser.add_argument("--annotations", default=ANNOTATIONS)
    parser.add_argument("--labeled",     default=LABELED_DIR)
    parser.add_argument("--preview",     action="store_true",
                        help="Show preview window per frame")
    parser.add_argument("--no-json",     action="store_true",
                        help="Skip saving JSONs, only compute metrics CSV")
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    run(
        splits_to_process = splits,
        annotations_csv   = args.annotations,
        labeled_dir       = args.labeled,
        save_json         = not args.no_json,
        preview           = args.preview,
    )
