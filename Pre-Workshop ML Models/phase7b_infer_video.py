"""
Phase 7b — U-Net video inference and annotation.

Runs the trained model on each frame, skeletonizes the predicted jet mask,
computes the full metric suite, and writes:
  - Annotated MP4 (skeleton + 3 key metrics burned on each frame)
  - metrics.csv (full per-frame metric suite)

Inference goes through jet_inference.JetSegmenter, which auto-detects the model:
  .onnx  -> ONNX Runtime (fast — recommended; build it with quantize_onnx.py)
  .pth   -> PyTorch checkpoint (ResNet-50 or MobileNetV2 U-Net)

Metrics + overlay drawing live in jet_metrics_lib.py (single source of truth).

Run from your project root:
    python phase7b_infer_video.py <video> [--model checkpoints/unet_mobilenet.onnx]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import csv
import argparse
from pathlib import Path

from jet_inference import JetSegmenter, extract_skeleton_from_mask
from jet_metrics_lib import compute_all_metrics, draw_annotations, METRIC_SCHEMA

# ── Config ────────────────────────────────────────────────────────────────────
# Default to the fast ONNX model; falls back to PyTorch if you pass a .pth.
CHECKPOINT = "checkpoints/unet_mobilenet.onnx"
CONFIDENCE = 0.5     # threshold for predicted mask (0-1); the app overrides this
# ──────────────────────────────────────────────────────────────────────────────


# ── Backward-compatible API (used by jet_annotator_app.py) ──────────────────
def load_model(checkpoint_path, device=None, scale=1.0):
    """Return a JetSegmenter. `device` is accepted for compatibility (unused)."""
    return JetSegmenter(checkpoint_path, confidence=CONFIDENCE, scale=scale)


def predict_mask(segmenter, frame, device=None):
    """Predict a binary mask, honoring the current module-level CONFIDENCE."""
    segmenter.confidence = CONFIDENCE
    return segmenter.predict_mask(frame)
# ─────────────────────────────────────────────────────────────────────────────


def process_video(video_path, segmenter, output_dir, save_csv=True):
    """Process a video; yields (annotated_frame, metrics_row) per frame."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo info:")
    print(f"  Frames: {total_frames}")
    print(f"  FPS:    {fps:.2f}")
    print(f"  Size:   {w}x{h}")
    print(f"  Backend: {segmenter.backend}")

    frame_idx = 0
    success_count = 0
    metrics_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = segmenter.predict_mask(frame)
        skeleton = extract_skeleton_from_mask(mask)

        if skeleton:
            metrics = compute_all_metrics(skeleton)
            status = "ok"
            success_count += 1
        else:
            metrics = {}
            status = "failed"

        frame_annotated = draw_annotations(frame, skeleton, metrics)

        row = {"frame_idx": frame_idx, "status": status, **metrics}
        metrics_rows.append(row)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx+1:4d}/{total_frames}  {status}")

        yield frame_annotated, row
        frame_idx += 1

    cap.release()

    if save_csv and metrics_rows:
        csv_path = Path(output_dir) / "metrics.csv"
        fieldnames = ["frame_idx", "status"] + METRIC_SCHEMA
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"  Metrics CSV -> '{csv_path}'")

    print(f"\nProcessed {total_frames} frames ({success_count} with valid skeletons)")


def main():
    parser = argparse.ArgumentParser(
        description="Run the jet model on a video; output annotated MP4 + metrics CSV.")
    parser.add_argument("video",   type=str, help="Path to input video")
    parser.add_argument("--model", type=str, default=CHECKPOINT,
                        help="Model file (.onnx or .pth)")
    parser.add_argument("--output", type=str, default="inference", help="Output directory")
    parser.add_argument("--conf",  type=float, default=CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--scale", type=float, default=1.0, help="Downscale frames (0.5 = half resolution)")
    parser.add_argument("--no-mp4", action="store_true", help="Skip MP4 output")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: {video_path} not found")
        return
    if not Path(args.model).exists():
        print(f"Error: model '{args.model}' not found.")
        print("Export an ONNX model with quantize_onnx.py, or pass --model <your.pth>.")
        return

    print(f"Loading model '{args.model}'...")
    segmenter = JetSegmenter(args.model, confidence=args.conf, scale=args.scale)
    print(f"  Backend: {segmenter.backend}")
    if args.scale != 1.0:
        print(f"  Scale: {args.scale} (downscaling input frames)")

    print(f"Processing video '{video_path}'...")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    mp4_path = None
    if not args.no_mp4:
        mp4_path = output_dir / (video_path.stem + "_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))

    for frame_annotated, row in process_video(str(video_path), segmenter, output_dir):
        if writer:
            writer.write(frame_annotated)

    if writer:
        writer.release()
        print(f"\nAnnotated video -> '{mp4_path}'")

    print(f"Done. Check '{output_dir}' for outputs.")


if __name__ == "__main__":
    main()
