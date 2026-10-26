"""
Phase 1a — Extract frames from polymer jet video.

Output structure:
  data/
    raw_frames/       <- all extracted frames (PNG)
    filtered_frames/  <- sharp frames only (created in phase1b)
    labeled/          <- (populated later during annotation)
    unlabeled/        <- (populated later during annotation)
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(
    video_path: str,
    output_dir: str = "data/raw_frames",
    every_n_frames: int = 1,
    max_frames: int = None,
    roi: tuple = None,  # (x, y, w, h) to crop each frame, or None for full frame
):
    """
    Extract frames from a video file.

    Args:
        video_path:     Path to .mp4 / .avi / etc.
        output_dir:     Where to save PNGs.
        every_n_frames: Sample every N frames (1 = every frame, 5 = every 5th, etc.)
        max_frames:     Stop after this many saved frames (None = no limit).
        roi:            Optional (x, y, w, h) crop applied to every frame.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Sampling every {every_n_frames} frame(s)")
    if roi:
        print(f"  ROI crop   : x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print()

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(filename, frame)
            saved += 1

            if saved % 100 == 0:
                print(f"  Saved {saved} frames (frame index {frame_idx}/{total_frames})...")

            if max_frames and saved >= max_frames:
                print(f"  Reached max_frames={max_frames}, stopping.")
                break

        frame_idx += 1

    cap.release()
    print(f"\nDone. Saved {saved} frames to '{output_dir}/'")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from polymer jet video.")
    parser.add_argument("--output", default="data/raw_frames", help="Output directory")
    parser.add_argument("--every", type=int, default=1, help="Save every N-th frame")
    parser.add_argument("--max", type=int, default=None, help="Max frames to save")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                        default=None, help="Crop ROI: X Y W H (pixels)")
    args = parser.parse_args()

    extract_frames(
        video_path=r"C:\Users\nmana\Documents\VSCode\OpenCV MEW Jet Analysis\MEW-dalton.mp4",
        output_dir=args.output,
        every_n_frames=args.every,
        max_frames=args.max,
        roi=tuple(args.roi) if args.roi else None,
    )
