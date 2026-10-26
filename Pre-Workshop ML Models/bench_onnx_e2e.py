"""
bench_onnx_e2e.py — Measure real end-to-end FPS with the ONNX model.

Runs the full inference pipeline (preprocess → ONNX → skeleton → metrics → draw)
on a video, averaging over many frames so you see steady-state performance.

Run from project root:
    python bench_onnx_e2e.py MEW-dalton.mp4 --n-frames 100

Will print frame-by-frame timing and a summary showing your actual throughput.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import onnxruntime as ort
from skimage.morphology import skeletonize

from jet_metrics_lib import compute_all_metrics, draw_annotations

# ── Config (must match phase7b / quantize_onnx.py) ──────────────────────────
ONNX_FP32    = "checkpoints/unet_mobilenet.onnx"
RESIZE_H, RESIZE_W = 224, 416
CONFIDENCE = 0.5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SKEL_DOWNSAMPLE = 30
# ──────────────────────────────────────────────────────────────────────────


def preprocess(frame_bgr):
    """BGR frame -> (1,3,H,W) float32."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None]   # (1,3,H,W)
    return np.ascontiguousarray(arr)


def extract_skeleton_from_mask(mask):
    """Skeletonize a binary mask, return ordered (x,y) points or None."""
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(ys) < 5:
        return None
    order = ys.argsort()
    ys, xs = ys[order], xs[order]
    step = max(1, len(xs) // SKEL_DOWNSAMPLE)
    xs = xs[::step]
    ys = ys[::step]
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Input video")
    ap.add_argument("--n-frames", type=int, default=100, help="Frames to process")
    ap.add_argument("--skip-draw", action="store_true", help="Skip draw step (for pure timing)")
    ap.add_argument("--scale", type=float, default=1.0, help="Scale factor (0.5 = half resolution)")
    args = ap.parse_args()

    if not Path(ONNX_FP32).exists():
        print(f"Error: {ONNX_FP32} not found. Run quantize_onnx.py first.")
        return

    print(f"Loading ONNX model from '{ONNX_FP32}'...")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(ONNX_FP32, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    print("Loaded.\n")

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_orig = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {w}x{h}, {total} frames @ {fps_orig:.1f} FPS")
    print(f"Processing {args.n_frames} frames...\n")

    frame_times = []
    success_count = 0
    warmup_frames = 5

    for frame_idx in range(args.n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally downscale the input frame
        if args.scale != 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * args.scale), int(w * args.scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        t0 = time.time()

        # Preprocess
        frame_t = preprocess(frame)

        # Inference (ONNX)
        logits = sess.run(None, {in_name: frame_t})[0]

        # Threshold + resize back
        orig_h, orig_w = frame.shape[:2]
        pred_s = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        mask = (pred_s[0, 0] > CONFIDENCE).astype(np.uint8)
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Skeleton
        skeleton = extract_skeleton_from_mask(mask)

        # Metrics
        if skeleton:
            metrics = compute_all_metrics(skeleton)
            status = "ok"
            success_count += 1
        else:
            metrics = {}
            status = "fail"

        # Draw (optional)
        if not args.skip_draw:
            annotated = draw_annotations(frame, skeleton, metrics)

        dt = (time.time() - t0) * 1000  # ms

        if frame_idx >= warmup_frames:
            frame_times.append(dt)

        if (frame_idx + 1) % 20 == 0:
            fps = 1000.0 / dt if dt > 0 else 0
            print(f"  Frame {frame_idx+1:4d}/{args.n_frames}  {dt:6.1f} ms  {fps:5.1f} FPS  ({status})")

    cap.release()

    if frame_times:
        mean_ms = np.mean(frame_times)
        std_ms = np.std(frame_times)
        mean_fps = 1000.0 / mean_ms
        p95_ms = np.percentile(frame_times, 95)
        p95_fps = 1000.0 / p95_ms

        print("\n" + "="*60)
        print("END-TO-END BENCHMARK (after warmup)")
        print("="*60)
        print(f"  Frames processed (after warmup): {len(frame_times)}")
        print(f"  Mean latency:     {mean_ms:6.1f} ms/frame")
        print(f"  Std dev:          {std_ms:6.1f} ms")
        print(f"  p95 latency:      {p95_ms:6.1f} ms/frame")
        print(f"  Mean FPS:         {mean_fps:6.1f}")
        print(f"  p95 FPS:          {p95_fps:6.1f}")
        print(f"  Successful detections: {success_count}/{len(frame_times)}")
        print("="*60)

        if args.skip_draw:
            print("(Draw step skipped. Real FPS includes drawing.)")

        print(f"\nTarget for PID control: 30 FPS (33 ms/frame)")
        print(f"Your current: {mean_fps:.1f} FPS ({mean_ms:.1f} ms/frame)")
        if mean_fps >= 30:
            print("✓ GOOD — you can hit real-time on CPU")
        elif mean_fps >= 15:
            print("~ ACCEPTABLE — usable with frame dropping (discard old frames)")
        else:
            print("✗ Need GPU or model optimization")


if __name__ == "__main__":
    main()
