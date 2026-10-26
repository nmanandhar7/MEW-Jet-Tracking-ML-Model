"""
jet_inference.py — single inference backend for the jet pipeline.

Auto-detects the model type from the file extension:
  .onnx   -> ONNX Runtime  (fast; produced by quantize_onnx.py — recommended)
  .pth/.pt -> PyTorch checkpoint (ResNet-50 or MobileNetV2 U-Net)

Everything that needs to run the model — phase7b_infer_video.py, the GUI app,
and your future live-camera loop — should go through JetSegmenter so the
preprocessing and thresholding stay identical everywhere.

Torch is imported lazily, so ONNX-only use does not require PyTorch installed.
"""

import numpy as np
import cv2
from pathlib import Path
from skimage.morphology import skeletonize

# ── Must match training (phase5b/phase5c) and quantize_onnx.py ──────────────
RESIZE_H, RESIZE_W = 224, 416
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_DOWNSAMPLE = 30
# ─────────────────────────────────────────────────────────────────────────────


def preprocess(frame_bgr):
    """BGR frame -> (1,3,H,W) float32 tensor, normalized exactly as in training."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None]      # (1,3,H,W)
    return np.ascontiguousarray(arr)


def extract_skeleton_from_mask(mask, downsample=DEFAULT_DOWNSAMPLE):
    """Skeletonize a binary mask -> ordered (x,y) points top->bottom, or None."""
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(ys) < 5:
        return None
    order = ys.argsort()
    ys, xs = ys[order], xs[order]
    step = max(1, len(xs) // downsample)
    xs = xs[::step]
    ys = ys[::step]
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def _load_torch_model(path, device):
    """Load a torch U-Net, auto-picking ResNet-50 vs MobileNetV2 architecture."""
    import torch
    from phase4b_unet_model import JetUNet
    try:
        from phase4c_unet_mobilenet import JetUNetMobile
        have_mobile = True
    except Exception:
        JetUNetMobile = None
        have_mobile = False

    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

    name = Path(path).name.lower()
    # Order candidate architectures by filename hint, then try the other as fallback.
    if "mobilenet" in name:
        if not have_mobile:
            raise RuntimeError(
                "Checkpoint looks like MobileNet but phase4c_unet_mobilenet.py is missing.")
        candidates = [JetUNetMobile, JetUNet]
    else:
        candidates = [JetUNet] + ([JetUNetMobile] if have_mobile else [])

    last_err = None
    for cls in candidates:
        try:
            model = cls(freeze_encoder=False).to(device)
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as e:
            last_err = e
    raise last_err


class JetSegmenter:
    """
    Unified jet segmenter. Construct once, call predict_mask() per frame.

        seg = JetSegmenter("checkpoints/unet_mobilenet.onnx", confidence=0.5, scale=1.0)
        mask = seg.predict_mask(frame_bgr)          # binary mask, original size
        pts  = seg.extract_skeleton(mask)           # centerline points or None
    
    Parameters:
        model_path (str): path to .onnx or .pth model
        confidence (float): threshold for jet pixels (0-1)
        scale (float): downscale input frames (0.5 = half resolution, faster)
    """

    def __init__(self, model_path, confidence=0.5, scale=1.0):
        self.path = str(model_path)
        self.confidence = float(confidence)
        self.scale = float(scale)
        self.kind = "onnx" if self.path.lower().endswith(".onnx") else "torch"

        if self.kind == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(
                self.path, so, providers=["CPUExecutionProvider"])
            self._in = self._sess.get_inputs()[0].name
        else:
            import torch
            self._torch = torch
            self._device = torch.device("cpu")
            self._model = _load_torch_model(self.path, self._device)

    @property
    def backend(self):
        return "ONNX Runtime" if self.kind == "onnx" else "PyTorch"

    def _logits(self, x):
        if self.kind == "onnx":
            return self._sess.run(None, {self._in: x})[0]
        t = self._torch.from_numpy(x)
        with self._torch.no_grad():
            out = self._model(t)
        return out.cpu().numpy()

    def predict_mask(self, frame_bgr):
        """Run the model on a BGR frame -> binary mask at the frame's resolution."""
        orig_h, orig_w = frame_bgr.shape[:2]
        
        # Optionally downscale for speed
        if self.scale != 1.0:
            new_h = int(orig_h * self.scale)
            new_w = int(orig_w * self.scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        x = preprocess(frame_bgr)
        logits = self._logits(x)
        prob = 1.0 / (1.0 + np.exp(-logits))                 # sigmoid
        mask = (prob[0, 0] > self.confidence).astype(np.uint8)
        
        # Resize mask back to *original* frame resolution (not scaled)
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def extract_skeleton(self, mask):
        return extract_skeleton_from_mask(mask)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("jet_inference.py self-check")
    # Pick whichever model exists, prefer ONNX
    for cand in ["checkpoints/unet_mobilenet.onnx",
                 "checkpoints/unet_mobilenet_best.pth",
                 "checkpoints/unet_best.pth"]:
        if Path(cand).exists():
            seg = JetSegmenter(cand)
            print(f"  Loaded {cand}  ->  backend: {seg.backend}")
            break
    else:
        print("  No model found in checkpoints/. Train or export one first.")
