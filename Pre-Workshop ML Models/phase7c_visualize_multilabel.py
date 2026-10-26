"""
Phase 7c — Visualize the 15-channel multilabel U-Net predictions vs ground truth.

For a handful of validation frames, renders a 3-up row:
    [ original | ground-truth overlay | predicted overlay ]
and stacks the rows into a single montage PNG so you can eyeball, per label,
how close the predictions are.

Each of the 15 channels gets a distinct color. A legend is drawn at the top.

Run from project root:
    python "Pre-Workshop ML Models/phase7c_visualize_multilabel.py"

Optional:
    --n 8                 number of val frames to render (default 6)
    --split val           which split to sample from
    --thresh 0.5          sigmoid threshold for predicted masks
    --ckpt <path>         checkpoint to load
"""

import sys, os, argparse
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from pathlib import Path

from phase4c_unet_mobilenet import JetUNetMobile
from jet_multimask import build_masks, load_json, CHANNELS, NUM_CLASSES

DATA_DIR   = "data/barrothandendpoints"
CKPT       = "checkpoints/unet_mobilenet_multilabel_best.pth"
OUT_PATH   = "checkpoints/unet_mobilenet_multilabel_predictions.png"
RESIZE_H   = 224
RESIZE_W   = 416

# 15 visually distinct BGR colors
PALETTE = [
    (0, 255, 0),     (255, 128, 0),   (255, 0, 255),    # skeleton, L-edge, R-edge
    (0, 255, 255),   (0, 0, 255),                        # start, end
    (0, 165, 255),   (0, 215, 255),   (0, 255, 128),     # diam 10,20,30
    (255, 255, 0),   (255, 0, 0),     (255, 0, 128),     # diam 40,50,60
    (128, 0, 255),   (203, 192, 255), (42, 42, 165),     # diam 70,80,90
    (180, 180, 180),                                     # diam 100
]


def overlay_masks(base_img, masks, alpha=0.6):
    """masks: (15, H, W) binary float. Draw each channel in its palette color."""
    out = base_img.copy()
    H, W = base_img.shape[:2]
    color_layer = np.zeros_like(base_img)
    for ch in range(NUM_CLASSES):
        m = masks[ch]
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        color_layer[m > 0.5] = PALETTE[ch]
    mask_any = (color_layer.sum(axis=2) > 0)
    out[mask_any] = cv2.addWeighted(base_img, 1 - alpha, color_layer, alpha, 0)[mask_any]
    return out


def label_bar(width, text):
    bar = np.full((22, width, 3), 30, dtype=np.uint8)
    cv2.putText(bar, text, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def make_legend(width):
    rows = (NUM_CLASSES + 2) // 3
    h = rows * 20 + 8
    leg = np.full((h, width, 3), 20, dtype=np.uint8)
    col_w = width // 3
    for i, name in enumerate(CHANNELS):
        r, c = i % rows, i // rows
        x = c * col_w + 6
        y = r * 20 + 16
        cv2.rectangle(leg, (x, y - 9), (x + 12, y + 1), PALETTE[i], -1)
        cv2.putText(leg, name, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
    return leg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--split", default="val")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = JetUNetMobile(freeze_encoder=False, out_channels=NUM_CLASSES).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded {args.ckpt}  (val mean Dice = {ckpt.get('val_dice', '?')})")

    split_dir = Path(DATA_DIR) / args.split
    jsons = sorted(split_dir.glob("*.json"))
    # Evenly sample n frames across the split
    idxs = np.linspace(0, len(jsons) - 1, args.n).astype(int)
    picks = [jsons[i] for i in idxs]

    rows = []
    legend_added_width = None
    for jp in picks:
        img_path = jp.with_suffix(".png")
        bgr = cv2.imread(str(img_path))
        H0, W0 = bgr.shape[:2]

        # Ground-truth masks at native res
        gt = build_masks(load_json(jp)).astype(np.float32) / 255.0   # (15,H,W)

        # Model prediction
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.resize(rgb, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0
        t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t)[0]                                     # (15, 224, 416)
        pred = (torch.sigmoid(logits) > args.thresh).float().cpu().numpy()

        gt_vis   = overlay_masks(bgr, gt)
        pred_vis = overlay_masks(bgr, pred)

        # stack the 3 panels horizontally, each with a caption bar
        panels = []
        for img, cap in [(bgr, f"{jp.stem}  ORIGINAL"),
                         (gt_vis, "GROUND TRUTH"),
                         (pred_vis, "PREDICTION")]:
            panels.append(np.vstack([label_bar(W0, cap), img]))
        row = np.hstack(panels)
        rows.append(row)
        legend_added_width = row.shape[1]

    montage = np.vstack(rows)
    legend = make_legend(legend_added_width)
    montage = np.vstack([legend, montage])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, montage)
    print(f"Saved montage ({montage.shape[1]}x{montage.shape[0]}) -> {args.out}")


if __name__ == "__main__":
    main()
