import os
import csv
import random
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser(description="Split CV-annotated frames into train/val/test ML dataset")
    ap.add_argument("--data", default="data", help="Base data directory")
    ap.add_argument("--train", type=float, default=0.70, help="Train fraction (default 0.70)")
    ap.add_argument("--val", type=float, default=0.15, help="Val fraction (default 0.15)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    images_dir = os.path.join(args.data, "filtered_frames")
    masks_dir = os.path.join(args.data, "masks")
    metrics_csv = os.path.join(args.data, "cv_metrics.csv")
    out_dir = os.path.join(args.data, "ml_dataset")

    if not os.path.exists(metrics_csv):
        print(f"ERROR: {metrics_csv} not found. Run phase2_cv_annotate.py first.")
        return

    # Only include frames that were successfully annotated
    annotated = {}
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                annotated[row["frame"]] = row

    # Require both source image and generated mask to exist
    valid_frames = [
        fname for fname in sorted(annotated)
        if os.path.exists(os.path.join(images_dir, fname))
        and os.path.exists(os.path.join(masks_dir, fname))
    ]

    if not valid_frames:
        print("No valid annotated frames found (need image in filtered_frames/ and mask in masks/).")
        print("Run phase2_cv_annotate.py first.")
        return

    # Reproducible split
    rng = random.Random(args.seed)
    shuffled = valid_frames[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train: n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    # Create directories and copy files
    for split, fnames in splits.items():
        for sub in ("images", "masks"):
            os.makedirs(os.path.join(out_dir, split, sub), exist_ok=True)
        for fname in fnames:
            shutil.copy2(
                os.path.join(images_dir, fname),
                os.path.join(out_dir, split, "images", fname),
            )
            shutil.copy2(
                os.path.join(masks_dir, fname),
                os.path.join(out_dir, split, "masks", fname),
            )

    # Write manifest with measurements for each frame
    dia_cols = [f"dia_{p}pct_px" for p in [10, 25, 50, 75, 99]]
    fieldnames = ["frame", "split", "image_path", "mask_path", "angle_deg", "total_length_px"] + dia_cols
    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split, fnames in splits.items():
            for fname in fnames:
                row = annotated[fname]
                writer.writerow({
                    "frame": fname,
                    "split": split,
                    "image_path": os.path.join(split, "images", fname),
                    "mask_path": os.path.join(split, "masks", fname),
                    "angle_deg": row["angle_deg"],
                    "total_length_px": row["total_length_px"],
                    **{col: row[col] for col in dia_cols},
                })

    print(f"Dataset split ({n} frames, seed={args.seed}):")
    for split, fnames in splits.items():
        print(f"  {split:5s}: {len(fnames):4d} frames")
    print(f"\nManifest -> {manifest_path}")
    print(f"Dataset  -> {out_dir}/")
    print()
    print("Structure:")
    print(f"  {out_dir}/")
    for split in splits:
        print(f"    {split}/images/   <- original grayscale frames")
        print(f"    {split}/masks/    <- segmentation masks (0=bg, 1=left edge, 2=right edge, 3=centerline)")


if __name__ == "__main__":
    main()
