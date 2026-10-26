"""
quantize_onnx.py — Speed up the MobileNetV2 U-Net on CPU via ONNX Runtime.

Two levers, measured separately so you can see what each buys you:
  1. ONNX Runtime (fp32): graph optimization, no accuracy change.
  2. ONNX Runtime (INT8): static quantization calibrated on real video frames.

What it produces (next to your checkpoint, in checkpoints/):
  - unet_mobilenet.onnx        (fp32 ONNX export)
  - unet_mobilenet_int8.onnx   (quantized INT8 model)

And it prints a benchmark: PyTorch vs ORT-fp32 vs ORT-INT8 (FPS),
plus how much INT8 disagrees with fp32 on the actual jet mask.

Run from your project root:
    pip install onnx onnxruntime
    python quantize_onnx.py MEW-dalton.mp4

If export complains about 'onnxscript', either:
    pip install onnxscript          (then rerun)
  or it will already have tried the legacy exporter automatically.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch

from phase4c_unet_mobilenet import JetUNetMobile

# ── Config (must match phase7b / phase5c) ───────────────────────────────────
CHECKPOINT = "checkpoints/unet_mobilenet_best.pth"
ONNX_FP32  = "checkpoints/unet_mobilenet.onnx"
ONNX_INT8  = "checkpoints/unet_mobilenet_int8.onnx"
RESIZE_H, RESIZE_W = 224, 416
CONFIDENCE = 0.5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(frame_bgr):
    """BGR frame -> (1,3,H,W) float32, exactly as the model was trained/served."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None]   # (1,3,H,W)
    return np.ascontiguousarray(arr)


def grab_frames(video_path, n, stride=1):
    """Grab up to n frames (every `stride`-th) as preprocessed arrays."""
    cap = cv2.VideoCapture(video_path)
    frames, i = [], 0
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            frames.append(preprocess(frame))
        i += 1
    cap.release()
    return frames


def load_torch_model(device):
    model = JetUNetMobile(freeze_encoder=False).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def export_onnx(model):
    """Export to fp32 ONNX. Try legacy exporter first (no onnxscript dep)."""
    dummy = torch.randn(1, 3, RESIZE_H, RESIZE_W)
    Path(ONNX_FP32).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model, dummy, ONNX_FP32,
            input_names=["input"], output_names=["mask"],
            opset_version=17, dynamo=False,
        )
    except TypeError:
        # Older torch without the dynamo kwarg
        torch.onnx.export(
            model, dummy, ONNX_FP32,
            input_names=["input"], output_names=["mask"],
            opset_version=17,
        )
    print(f"  Exported fp32 ONNX -> {ONNX_FP32}")


def make_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def quantize_int8(calib_frames):
    """Static INT8 quantization, calibrated on real preprocessed frames."""
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, QuantType, QuantFormat)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre_path = ONNX_FP32.replace(".onnx", "_pre.onnx")
    quant_pre_process(ONNX_FP32, pre_path)

    class Reader(CalibrationDataReader):
        def __init__(self, frames):
            self.it = iter({"input": f} for f in frames)
        def get_next(self):
            return next(self.it, None)

    quantize_static(
        pre_path, ONNX_INT8, Reader(calib_frames),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    )
    print(f"  Quantized INT8 ONNX -> {ONNX_INT8}")


def bench(fn, frames, warmup=5):
    """Warm up, then time fn over all frames. Returns (ms_per_frame, fps)."""
    for f in frames[:warmup]:
        fn(f)
    t0 = time.time()
    for f in frames:
        fn(f)
    dt = (time.time() - t0) / len(frames)
    return dt * 1000, 1.0 / dt


def to_mask(logits):
    """Sigmoid + threshold -> binary mask (numpy)."""
    p = 1.0 / (1.0 + np.exp(-logits))
    return (p[0, 0] > CONFIDENCE).astype(np.uint8)


def dice(a, b, eps=1e-6):
    inter = np.logical_and(a, b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Video used for calibration + benchmarking")
    ap.add_argument("--calib", type=int, default=64, help="Calibration frames")
    ap.add_argument("--bench", type=int, default=60, help="Benchmark frames")
    args = ap.parse_args()

    if not Path(CHECKPOINT).exists():
        print(f"Error: {CHECKPOINT} not found. Train phase5c first.")
        return

    device = torch.device("cpu")
    print("Loading PyTorch model...")
    model = load_torch_model(device)

    print(f"Grabbing frames from {args.video}...")
    calib_frames = grab_frames(args.video, args.calib, stride=5)
    bench_frames = grab_frames(args.video, args.bench, stride=1)
    print(f"  {len(calib_frames)} calibration / {len(bench_frames)} benchmark frames")
    if not bench_frames:
        print("No frames read — check the video path.")
        return

    print("\nExporting to ONNX...")
    export_onnx(model)

    print("Quantizing to INT8 (calibrating on real frames)...")
    quantize_int8(calib_frames)

    # Inference callables
    def torch_infer(f):
        with torch.no_grad():
            return model(torch.from_numpy(f)).numpy()

    sess_fp32 = make_session(ONNX_FP32)
    sess_int8 = make_session(ONNX_INT8)
    in_name = sess_fp32.get_inputs()[0].name
    def ort_fp32(f): return sess_fp32.run(None, {in_name: f})[0]
    def ort_int8(f): return sess_int8.run(None, {in_name: f})[0]

    print("\nBenchmarking (with warmup)...")
    t_ms, t_fps = bench(torch_infer, bench_frames)
    f_ms, f_fps = bench(ort_fp32,    bench_frames)
    i_ms, i_fps = bench(ort_int8,    bench_frames)

    # Accuracy: how much does INT8 disagree with fp32 on the mask?
    dices = []
    for f in bench_frames[:20]:
        dices.append(dice(to_mask(ort_fp32(f)), to_mask(ort_int8(f))))
    mean_agree = float(np.mean(dices))

    print("\n" + "=" * 56)
    print("INFERENCE SPEED  (model forward pass only)")
    print("=" * 56)
    print(f"  PyTorch (eager)   : {t_ms:6.1f} ms/frame   {t_fps:5.1f} FPS")
    print(f"  ONNX Runtime fp32 : {f_ms:6.1f} ms/frame   {f_fps:5.1f} FPS   ({t_ms/f_ms:.2f}x)")
    print(f"  ONNX Runtime INT8 : {i_ms:6.1f} ms/frame   {i_fps:5.1f} FPS   ({t_ms/i_ms:.2f}x)")
    print("=" * 56)
    print(f"  INT8 vs fp32 mask agreement (Dice): {mean_agree:.4f}")
    print("=" * 56)
    print("\nNote: this times the model only. Your end-to-end FPS also includes")
    print("preprocess + skeleton + metrics + draw (~15-20 ms total from the profiler).")
    print("Add that to the ms/frame above to estimate real throughput.")


if __name__ == "__main__":
    main()
