"""
phase3_cv_to_json.py — Convert CV pipeline outputs to LabelMe JSON format.

Each JSON contains three shapes matching the annotated PNG 1-to-1:
  jet_left_edge   — smoothed left jet edge contour (linestrip)
  jet_right_edge  — smoothed right jet edge contour (linestrip)
  jet_skeleton    — smoothed centerline / midline (linestrip)
  jet_start       — nozzle tip point
  jet_end         — jet tip point

Source images come from data/annotated/ (the CV-annotated PNGs).
Coordinate data comes from data/cv_shapes/*.xz.
Output JSONs go to data/BarothJSONData/{train|val|test}/.

Run:
    python phase3_cv_to_json.py [--overwrite]
"""

import argparse
import json
import lzma
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


def load_shape(xz_path: Path) -> dict:
    with lzma.open(xz_path, "rb") as f:
        return pickle.load(f)


def contour_to_points(contour) -> list:
    """Convert (y_array, x_array) contour tuple to [[x, y], ...] LabelMe points."""
    ys = np.asarray(contour[0])
    xs = np.asarray(contour[1])
    return [[float(x), float(y)] for y, x in zip(ys, xs)]


def diameter_lines(mid, left, right, pcts=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
    """
    For each pct, find the midline point at that fraction of total arc length,
    then return a horizontal line [[x_left, y], [x_right, y]] at that y-row.
    """
    mid_ys = np.asarray(mid[0], dtype=float)
    mid_xs = np.asarray(mid[1], dtype=float)
    left_ys  = np.asarray(left[0],  dtype=float)
    left_xs  = np.asarray(left[1],  dtype=float)
    right_ys = np.asarray(right[0], dtype=float)
    right_xs = np.asarray(right[1], dtype=float)

    segs = np.hypot(np.diff(mid_ys), np.diff(mid_xs))
    arc  = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]

    lines = []
    for pct in pcts:
        idx = int(np.searchsorted(arc, (pct / 100.0) * total))
        idx = min(idx, len(mid_ys) - 1)
        y = float(mid_ys[idx])
        x_left  = float(left_xs [np.argmin(np.abs(left_ys  - y))])
        x_right = float(right_xs[np.argmin(np.abs(right_ys - y))])
        lines.append((pct, x_left, y, x_right))

    return lines


def make_shape(label: str, points: list, shape_type: str = "linestrip") -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
        "mask": None,
    }


def to_labelme(img_name: str, img_w: int, img_h: int,
               left_pts, right_pts, skeleton_pts,
               start_xy, end_xy, dia_lines) -> dict:
    shapes = [
        make_shape("jet_start",      [[float(start_xy[0]), float(start_xy[1])]], "point"),
        make_shape("jet_left_edge",  left_pts),
        make_shape("jet_right_edge", right_pts),
        make_shape("jet_skeleton",   skeleton_pts),
        make_shape("jet_end",        [[float(end_xy[0]),   float(end_xy[1])]],   "point"),
    ]
    for pct, x_left, y, x_right in dia_lines:
        shapes.append(make_shape(f"diameter_{pct}pct",
                                 [[x_left, y], [x_right, y]],
                                 "line"))
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def main():
    ap = argparse.ArgumentParser(description="Convert CV contours/midline to LabelMe JSONs.")
    ap.add_argument("--cv-shapes", default="data/cv_shapes")
    ap.add_argument("--output",    default="data/BarothJSONData")
    ap.add_argument("--annotated", default="data/annotated",      help="Folder to scan for successfully annotated frames")
    ap.add_argument("--source",    default="data/filtered_frames", help="Clean frames to copy alongside JSONs")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite JSONs that already exist in the output folder")
    args = ap.parse_args()

    shapes_dir    = Path(args.cv_shapes)
    output_dir    = Path(args.output)
    source_dir    = Path(args.source)
    annotated_dir = Path(args.annotated)

    # --- Collect frames from the annotated folder (successfully CV-processed) ---
    ok_frames = sorted(
        p.name for p in annotated_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg")
    )
    print(f"Frames found (via {args.annotated}): {len(ok_frames)}")

    # --- Find frames already in output (existing JSON) ---
    existing = {}  # frame_name -> split
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        if split_dir.exists():
            for jpath in split_dir.glob("*.json"):
                existing[jpath.stem + ".png"] = split

    print(f"Existing JSONs in output    : {len(existing)}")

    # --- Assign splits for new frames (70/15/15, seed=42) ---
    new_frames = [f for f in ok_frames if f not in existing]
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(new_frames))
    n_train = int(len(new_frames) * 0.70)
    n_val   = int(len(new_frames) * 0.15)
    split_map = {}
    for i, idx in enumerate(perm):
        name = new_frames[idx]
        if i < n_train:
            split_map[name] = "train"
        elif i < n_train + n_val:
            split_map[name] = "val"
        else:
            split_map[name] = "test"

    # --- Process ---
    n_written = n_skipped = n_missing = n_error = 0

    import cv2

    for frame_name in ok_frames:
        stem    = os.path.splitext(frame_name)[0]
        xz_path = shapes_dir / (stem + ".xz")
        src_img = source_dir / frame_name

        # Determine split
        if frame_name in existing:
            if not args.overwrite:
                n_skipped += 1
                continue
            split = existing[frame_name]
        else:
            split = split_map.get(frame_name, "train")

        if not xz_path.exists():
            n_missing += 1
            continue

        dst_dir  = output_dir / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        json_out = dst_dir / (stem + ".json")
        img_out  = dst_dir / frame_name

        try:
            shape = load_shape(xz_path)

            left  = shape.get("left_contour_smoothed")  or shape["left_contour"]
            right = shape.get("right_contour_smoothed") or shape["right_contour"]
            mid   = shape.get("midline_coordinates_smoothed") or shape["midline_coordinates"]

            left_pts     = contour_to_points(left)
            right_pts    = contour_to_points(right)
            skeleton_pts = contour_to_points(mid)
            dia          = diameter_lines(mid, left, right)

            # Image dimensions from source
            img = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE) if src_img.exists() else None
            img_h, img_w = img.shape if img is not None else (228, 1262)

            mid_ys = np.asarray(mid[0])
            mid_xs = np.asarray(mid[1])
            start_xy = (float(mid_xs[0]),  float(mid_ys[0]))
            end_xy   = (float(mid_xs[-1]), float(mid_ys[-1]))

            doc = to_labelme(frame_name, img_w, img_h,
                             left_pts, right_pts, skeleton_pts,
                             start_xy, end_xy, dia)

            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)

            if src_img.exists() and (not img_out.exists() or args.overwrite):
                shutil.copy2(src_img, img_out)

            n_written += 1

        except Exception as exc:
            print(f"  [error] {frame_name}: {exc}")
            n_error += 1

    print(f"\nDone.")
    print(f"  Written  : {n_written}")
    print(f"  Skipped  : {n_skipped}  (already exist; use --overwrite to replace)")
    print(f"  Missing  : {n_missing}  (.xz not found)")
    print(f"  Errors   : {n_error}")


if __name__ == "__main__":
    main()
