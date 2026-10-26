"""
Phase 3b — Generate binary jet masks from skeleton JSONs.

Reads the jet_skeleton linestrip from each LabelMe JSON, draws it as a
thick line onto a blank mask, and saves the mask as a PNG.

These masks are the training labels for the U-Net segmentation model.

Output: data/masks/<split>/<filename>.png  (white jet on black background)

Run from your project root:
    python phase3b_make_masks.py

Optional flags:
    --thickness 5     (line thickness in px, default 5)
    --split train     (only one split)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import argparse
import numpy as np
import cv2
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
LABELED_DIR = "data/labeled"
MASKS_DIR   = "data/masks"
SPLITS      = ["train", "val", "test"]
THICKNESS   = 5    # how thick to draw the jet skeleton (px)
# ──────────────────────────────────────────────────────────────────────────────


def make_mask_from_json(json_path, thickness):
    """Read skeleton linestrip from JSON, return binary mask as numpy array."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_h = data["imageHeight"]
    img_w = data["imageWidth"]

    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Find the skeleton shape
    skel = next(
        (s for s in data.get("shapes", []) if s.get("label") == "jet_skeleton"),
        None
    )
    if skel is None:
        return None

    pts = np.array(skel["points"], dtype=np.int32)
    if len(pts) < 2:
        return None

    # Draw the skeleton as a connected thick polyline
    cv2.polylines(mask, [pts], isClosed=False, color=255,
                  thickness=thickness, lineType=cv2.LINE_AA)

    return mask


def run(splits_to_process, labeled_dir, masks_dir, thickness):
    success = 0
    failed  = 0

    for split in splits_to_process:
        src_dir = Path(labeled_dir) / split
        dst_dir = Path(masks_dir) / split
        dst_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(src_dir.glob("*.json"))
        print(f"  {split:6s}: {len(json_files)} JSONs")

        for jpath in json_files:
            mask = make_mask_from_json(jpath, thickness)
            if mask is None:
                failed += 1
                continue

            out_path = dst_dir / (jpath.stem + ".png")
            cv2.imwrite(str(out_path), mask)
            success += 1

    print(f"\nDone.")
    print(f"  Masks created : {success}")
    print(f"  Failed        : {failed}")
    print(f"  Saved to      : {masks_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary jet masks from skeleton JSONs.")
    parser.add_argument("--split",     choices=["train","val","test","all"], default="all")
    parser.add_argument("--labeled",   default=LABELED_DIR)
    parser.add_argument("--masks",     default=MASKS_DIR)
    parser.add_argument("--thickness", type=int, default=THICKNESS)
    args = parser.parse_args()

    splits = SPLITS if args.split == "all" else [args.split]
    print(f"Generating masks (thickness={args.thickness}px)...")
    run(splits, args.labeled, args.masks, args.thickness)
