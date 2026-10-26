"""
Fix JSON splits without a manifest.

Looks at which PNG files exist in labeled/train/, labeled/val/, labeled/test/
and moves the matching JSON from train/ to the correct folder.

Run from your project root:
    python fix_json_splits.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import shutil
from pathlib import Path

LABELED_DIR = Path("data/labeled")

def fix_splits():
    train_dir = LABELED_DIR / "train"
    val_dir   = LABELED_DIR / "val"
    test_dir  = LABELED_DIR / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            print(f"ERROR: folder not found: {d}")
            return

    # Get PNG stems in each split folder
    val_frames  = {f.stem for f in val_dir.glob("*.png")}
    test_frames = {f.stem for f in test_dir.glob("*.png")}

    print(f"PNGs found in val/  : {len(val_frames)}")
    print(f"PNGs found in test/ : {len(test_frames)}")

    moved_val  = 0
    moved_test = 0
    missing    = 0

    # Move val JSONs
    for stem in val_frames:
        src = train_dir / (stem + ".json")
        dst = val_dir   / (stem + ".json")
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_val += 1
        elif not dst.exists():
            missing += 1

    # Move test JSONs
    for stem in test_frames:
        src = train_dir / (stem + ".json")
        dst = test_dir  / (stem + ".json")
        if src.exists():
            shutil.move(str(src), str(dst))
            moved_test += 1
        elif not dst.exists():
            missing += 1

    print(f"\nDone.")
    print(f"  Moved to val/  : {moved_val}")
    print(f"  Moved to test/ : {moved_test}")
    print(f"  Missing JSONs  : {missing} (frames not yet annotated)")

    # Final counts
    print(f"\nJSON counts after fix:")
    for split in ["train", "val", "test"]:
        jsons = len(list((LABELED_DIR / split).glob("*.json")))
        pngs  = len(list((LABELED_DIR / split).glob("*.png")))
        print(f"  {split:6s}: {jsons} JSONs / {pngs} PNGs")

if __name__ == "__main__":
    fix_splits()