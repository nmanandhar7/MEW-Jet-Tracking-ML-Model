"""
Phase 1c — Organize filtered frames into train/val/test splits.

Output structure after this script:
  data/
    labeled/        <- frames you will annotate (to be split after annotation)
      train/
      val/
      test/
    unlabeled/      <- held-out frames, not annotated (for semi-supervised later)

By default, 80% of filtered frames go to 'labeled/' and 20% to 'unlabeled/'.
Within the labeled pool, frames are pre-split 70/15/15 (train/val/test)
so annotators know which split each frame belongs to from the start.

A manifest CSV is written listing every frame and its assigned split.
"""

import os
import shutil
import random
import csv
from pathlib import Path


# ── tuneable splits ────────────────────────────────────────────────────────────
LABELED_FRACTION   = 0.80   # fraction of filtered frames to annotate
TRAIN_FRACTION     = 0.70   # of labeled pool
VAL_FRACTION       = 0.15   # of labeled pool
TEST_FRACTION      = 0.15   # of labeled pool  (must sum to 1.0 with above two)
RANDOM_SEED        = 42
# ──────────────────────────────────────────────────────────────────────────────


def organize(
    input_dir:        str   = "data/filtered_frames",
    labeled_dir:      str   = "data/labeled",
    unlabeled_dir:    str   = "data/unlabeled",
    labeled_frac:     float = LABELED_FRACTION,
    train_frac:       float = TRAIN_FRACTION,
    val_frac:         float = VAL_FRACTION,
    seed:             int   = RANDOM_SEED,
    manifest_csv:     str   = "data/manifest.csv",
    copy:             bool  = True,   # False = move files instead of copying
):
    assert abs(train_frac + val_frac + (1 - train_frac - val_frac) - 1.0) < 1e-6

    frames = sorted(Path(input_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    random.seed(seed)
    random.shuffle(frames)

    n_labeled   = int(len(frames) * labeled_frac)
    labeled     = frames[:n_labeled]
    unlabeled   = frames[n_labeled:]

    n_train = int(len(labeled) * train_frac)
    n_val   = int(len(labeled) * val_frac)
    train   = labeled[:n_train]
    val     = labeled[n_train:n_train + n_val]
    test    = labeled[n_train + n_val:]

    splits = {
        "train":      (train,     Path(labeled_dir) / "train"),
        "val":        (val,       Path(labeled_dir) / "val"),
        "test":       (test,      Path(labeled_dir) / "test"),
        "unlabeled":  (unlabeled, Path(unlabeled_dir)),
    }

    # Create directories
    for _, (_, dest) in splits.items():
        dest.mkdir(parents=True, exist_ok=True)

    rows = []
    op = shutil.copy2 if copy else shutil.move
    op_name = "Copied" if copy else "Moved"

    for split_name, (file_list, dest_dir) in splits.items():
        for fpath in file_list:
            dest_path = dest_dir / fpath.name
            op(str(fpath), str(dest_path))
            rows.append({
                "filename": fpath.name,
                "split":    split_name,
                "src":      str(fpath),
                "dst":      str(dest_path),
            })

    # Write manifest
    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "src", "dst"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    total = len(frames)
    print(f"Organized {total} frames from '{input_dir}':")
    print(f"  train      : {len(train):>5}  ({100*len(train)/total:.1f}%)")
    print(f"  val        : {len(val):>5}  ({100*len(val)/total:.1f}%)")
    print(f"  test       : {len(test):>5}  ({100*len(test)/total:.1f}%)")
    print(f"  unlabeled  : {len(unlabeled):>5}  ({100*len(unlabeled)/total:.1f}%)")
    print(f"\nManifest -> '{manifest_csv}'")
    print(f"Structure:")
    print(f"  {labeled_dir}/train/   <- annotate these")
    print(f"  {labeled_dir}/val/")
    print(f"  {labeled_dir}/test/")
    print(f"  {unlabeled_dir}/       <- held out")


def print_structure(base: str = "data"):
    """Print a quick tree of the data directory."""
    for root, dirs, files in os.walk(base):
        dirs.sort()
        depth = root.replace(base, "").count(os.sep)
        indent = "  " * depth
        folder = os.path.basename(root)
        print(f"{indent}{folder}/  ({len(files)} files)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organize frames into train/val/test splits.")
    parser.add_argument("--input",     default="data/filtered_frames", help="Filtered frames dir")
    parser.add_argument("--labeled",   default="data/labeled",         help="Labeled output dir")
    parser.add_argument("--unlabeled", default="data/unlabeled",       help="Unlabeled output dir")
    parser.add_argument("--labeled-frac", type=float, default=LABELED_FRACTION)
    parser.add_argument("--train-frac",   type=float, default=TRAIN_FRACTION)
    parser.add_argument("--val-frac",     type=float, default=VAL_FRACTION)
    parser.add_argument("--move",  action="store_true", help="Move files instead of copying")
    parser.add_argument("--tree",  action="store_true", help="Print data/ directory tree and exit")
    args = parser.parse_args()

    if args.tree:
        print_structure()
    else:
        organize(
            input_dir=args.input,
            labeled_dir=args.labeled,
            unlabeled_dir=args.unlabeled,
            labeled_frac=args.labeled_frac,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            copy=not args.move,
        )
