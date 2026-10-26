"""
Timing breakdown for the inference pipeline.

Runs your model on one video frame and shows where time is spent:
  - Frame read
  - Preprocessing (resize, normalize)
  - Model inference
  - Mask thresholding
  - Skeletonization
  - Metrics computation
  - Video frame drawing

Run from project root:
    python profile_pipeline.py MEW-dalton.mp4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import cv2
import numpy as np
import time
from pathlib import Path
from skimage.morphology import skeletonize

from phase4c_unet_mobilenet import JetUNetMobile
from jet_metrics_lib import compute_all_metrics, draw_annotations

CHECKPOINT = "checkpoints/unet_mobilenet_best.pth"
RESIZE_H, RESIZE_W = 224, 416
CONFIDENCE = 0.5

def load_model(checkpoint_path, device):
    model = JetUNetMobile(freeze_encoder=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def extract_skeleton_from_mask(mask):
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(ys) < 5:
        return None
    order = ys.argsort()
    ys, xs = ys[order], xs[order]
    step = max(1, len(xs) // 30)
    xs = xs[::step]
    ys = ys[::step]
    return [(int(x), int(y)) for x, y in zip(xs, ys)]

def time_stage(label, func):
    """Time a function and print result."""
    t0 = time.time()
    result = func()
    dt = (time.time() - t0) * 1000  # ms
    print(f"  {label:40s}: {dt:6.1f} ms")
    return result, dt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--n-frames", type=int, default=10, help="Frames to profile")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    cap = cv2.VideoCapture(args.video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h}\n")

    print(f"Loading model from '{CHECKPOINT}'...")
    model = load_model(CHECKPOINT, device)
    model.eval()
    print(f"Model loaded.\n")

    timings = {
        "read": [], "preprocess": [], "inference": [],
        "threshold": [], "skeleton": [], "metrics": [], "draw": []
    }

    for frame_idx in range(args.n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Frame {frame_idx + 1}/{args.n_frames}:")

        # Read (already done)
        print(f"  Frame size: {frame.shape}")

        # Preprocess
        def preproc():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rs = cv2.resize(frame_rgb, (RESIZE_W, RESIZE_H))
            frame_t = torch.from_numpy(frame_rs).permute(2, 0, 1).float() / 255.0
            frame_t[0] = (frame_t[0] - 0.485) / 0.229
            frame_t[1] = (frame_t[1] - 0.456) / 0.224
            frame_t[2] = (frame_t[2] - 0.406) / 0.225
            return frame_t.unsqueeze(0).to(device)
        
        frame_t, dt = time_stage("Preprocess (resize + normalize)", preproc)
        timings["preprocess"].append(dt)

        # Inference
        def infer():
            with torch.no_grad():
                pred = model(frame_t)
            return pred
        
        pred, dt = time_stage("Model inference", infer)
        timings["inference"].append(dt)

        # Threshold
        def thresh():
            pred_s = torch.sigmoid(pred)
            mask = (pred_s[0, 0].cpu().numpy() > CONFIDENCE).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            return mask
        
        mask, dt = time_stage("Threshold + resize back", thresh)
        timings["threshold"].append(dt)

        # Skeletonize
        def skel():
            return extract_skeleton_from_mask(mask)
        
        skeleton, dt = time_stage("Skeletonize", skel)
        timings["skeleton"].append(dt)

        # Metrics
        def mets():
            if skeleton:
                return compute_all_metrics(skeleton)
            return {}
        
        metrics, dt = time_stage("Compute 17 metrics", mets)
        timings["metrics"].append(dt)

        # Draw
        def drw():
            return draw_annotations(frame, skeleton, metrics)
        
        annotated, dt = time_stage("Draw overlay", drw)
        timings["draw"].append(dt)

        total = sum([timings[k][-1] for k in timings])
        fps = 1000 / total if total > 0 else 0
        print(f"  TOTAL: {total:6.1f} ms  ({fps:5.1f} FPS)\n")

    cap.release()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (average across all frames):")
    print("="*60)
    grand_total = 0
    for stage in ["preprocess", "inference", "threshold", "skeleton", "metrics", "draw"]:
        avg = np.mean(timings[stage]) if timings[stage] else 0
        pct = 100 * avg / sum(np.mean(timings[s]) for s in timings if timings[s])
        grand_total += avg
        print(f"  {stage:15s}: {avg:6.1f} ms  ({pct:5.1f}%)")
    
    fps = 1000 / grand_total if grand_total > 0 else 0
    print("="*60)
    print(f"  TOTAL            : {grand_total:6.1f} ms  ({fps:5.1f} FPS)")
    print("="*60)
    print(f"\nTo hit 30 FPS, you need to get below 33 ms/frame.")
    print(f"Currently spending {grand_total:.1f} ms — need to save {max(0, grand_total - 33):.1f} ms.")

if __name__ == "__main__":
    main()
