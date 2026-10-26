"""
phase3b_draw_annotations.py — Draw LabelMe annotations onto clean jet frames.

Reads JSONs from data/BarothJSONData/{train|val|test}/, loads the matching
clean frame from data/filtered_frames/, draws all annotation shapes, and
saves to data/BarothAnnotated/{split}/.

Shape colour key:
  jet_left_edge   — green
  jet_right_edge  — cyan
  jet_skeleton    — magenta
  diameter_Xpct   — orange line + label
  jet_start       — blue circle
  jet_end         — red circle

Run:
    python phase3b_draw_annotations.py
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# BGR colours
C_LEFT    = (0,   200,   0)    # green
C_RIGHT   = (200, 200,   0)    # cyan
C_SKEL    = (200,   0, 200)    # magenta
C_DIA     = (0,   140, 255)    # orange
C_START   = (255,  80,   0)    # blue
C_END     = (0,    80, 255)    # red
C_TEXT    = (255, 255, 255)    # white text


def draw_json(img_bgr, data: dict):
    for shape in data.get("shapes", []):
        label      = shape["label"]
        pts        = shape["points"]
        shape_type = shape["shape_type"]

        if shape_type == "linestrip" and len(pts) >= 2:
            arr = np.array(pts, dtype=np.int32)  # [[x, y], ...]
            if label == "jet_left_edge":
                cv2.polylines(img_bgr, [arr], False, C_LEFT,  1, cv2.LINE_AA)
            elif label == "jet_right_edge":
                cv2.polylines(img_bgr, [arr], False, C_RIGHT, 1, cv2.LINE_AA)
            elif label == "jet_skeleton":
                cv2.polylines(img_bgr, [arr], False, C_SKEL,  1, cv2.LINE_AA)

        elif shape_type == "line" and len(pts) == 2:
            p1 = (int(pts[0][0]), int(pts[0][1]))
            p2 = (int(pts[1][0]), int(pts[1][1]))
            cv2.line(img_bgr, p1, p2, C_DIA, 1, cv2.LINE_AA)
            # Label at midpoint
            mx = (p1[0] + p2[0]) // 2 + 2
            my = (p1[1] + p2[1]) // 2 - 2
            pct_str = label.replace("diameter_", "").replace("pct", "%")
            cv2.putText(img_bgr, pct_str, (mx, my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, C_TEXT, 1, cv2.LINE_AA)

        elif shape_type == "point" and len(pts) == 1:
            cx, cy = int(pts[0][0]), int(pts[0][1])
            colour = C_START if label == "jet_start" else C_END
            cv2.circle(img_bgr, (cx, cy), 3, colour, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Draw LabelMe annotations on clean frames.")
    ap.add_argument("--json-dir",  default="data/BarothJSONData",  help="LabelMe JSON root")
    ap.add_argument("--frames",    default="data/filtered_frames", help="Clean source frames")
    ap.add_argument("--output",    default="data/BarothAnnotated", help="Output root")
    args = ap.parse_args()

    json_root   = Path(args.json_dir)
    frames_dir  = Path(args.frames)
    output_root = Path(args.output)

    n_ok = n_missing = n_error = 0

    for split in ("train", "val", "test"):
        src_split = json_root / split
        if not src_split.exists():
            continue
        dst_split = output_root / split
        dst_split.mkdir(parents=True, exist_ok=True)

        json_files = sorted(src_split.glob("*.json"))
        for jpath in json_files:
            frame_name = jpath.stem + ".png"
            clean_path = frames_dir / frame_name

            if not clean_path.exists():
                n_missing += 1
                continue

            try:
                data = json.loads(jpath.read_text(encoding="utf-8"))

                img = cv2.imread(str(clean_path))
                if img is None:
                    n_missing += 1
                    continue

                draw_json(img, data)
                cv2.imwrite(str(dst_split / frame_name), img)
                n_ok += 1

            except Exception as exc:
                print(f"  [error] {frame_name}: {exc}")
                n_error += 1

    print(f"\nDone.")
    print(f"  Annotated : {n_ok}")
    print(f"  Missing   : {n_missing}  (clean frame not found in {args.frames})")
    print(f"  Errors    : {n_error}")
    print(f"  Saved to  : {args.output}/")


if __name__ == "__main__":
    main()
