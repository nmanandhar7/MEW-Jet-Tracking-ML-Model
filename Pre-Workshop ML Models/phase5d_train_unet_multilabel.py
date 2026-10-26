"""
Phase 5d — Train the MobileNetV2 U-Net to predict ALL 15 labeled features.

Unlike phase5c (single jet_skeleton mask), this trains a 15-channel model:
one sigmoid output channel per labeled geometry —

    jet_skeleton, jet_left_edge, jet_right_edge,
    jet_start, jet_end,
    diameter_10pct ... diameter_100pct

See jet_multimask.py for the channel definitions / rasterization.

Masks are built ON THE FLY from the LabelMe JSONs in data/barrothandendpoints/,
so there's no separate mask-generation step and no multi-GB mask folder to store.

Run from your project root (the folder that contains data/):
    python "Pre-Workshop ML Models/phase5d_train_unet_multilabel.py"
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from phase4c_unet_mobilenet import JetUNetMobile
from jet_multimask import build_masks, load_json, CHANNELS, NUM_CLASSES, CHANNEL_INDEX

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/barrothandendpoints"          # holds train/ val/ test/ (json + png)
CHECKPOINT  = "checkpoints/unet_mobilenet_multilabel_best.pth"
PLOT_PATH   = "checkpoints/unet_mobilenet_multilabel_training.png"

RESIZE_H    = 224
RESIZE_W    = 416

BATCH_SIZE    = 8
LR_DECODER    = 1e-3
LR_FINETUNE   = 1e-4
STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 30
PATIENCE      = 8

# Indices that must swap when an image is horizontally flipped
# (a flipped left edge becomes the right edge, and vice versa).
HFLIP_SWAP = [(CHANNEL_INDEX["jet_left_edge"], CHANNEL_INDEX["jet_right_edge"])]
# ──────────────────────────────────────────────────────────────────────────────


class JetMultiDataset(Dataset):
    """Loads image + builds 15-channel target mask from the paired JSON."""
    def __init__(self, data_dir, split, augment=False):
        self.dir = Path(data_dir) / split
        self.augment = augment
        self.samples = []
        for json_path in sorted(self.dir.glob("*.json")):
            img_path = json_path.with_suffix(".png")
            if img_path.exists():
                self.samples.append((img_path, json_path))
        print(f"  {split:6s}: {len(self.samples)} image/json pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data  = load_json(json_path)
        masks = build_masks(data)                       # (15, H, W) uint8 0/255

        # Resize image (linear) and masks (linear, then re-binarize).
        img = cv2.resize(img, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
        masks = np.transpose(masks, (1, 2, 0))          # H, W, 15
        masks = cv2.resize(masks, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
        if masks.ndim == 2:                             # safety (shouldn't happen w/ 15 ch)
            masks = masks[:, :, None]
        masks = np.transpose(masks, (2, 0, 1))          # 15, H, W
        masks = (masks > 127).astype(np.float32)

        # Horizontal-flip augmentation (swap left/right edge channels).
        if self.augment and np.random.rand() > 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            masks = np.ascontiguousarray(masks[:, :, ::-1])
            for a, b in HFLIP_SWAP:
                masks[[a, b]] = masks[[b, a]]

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        masks = torch.from_numpy(masks).float()
        return img, masks


def per_channel_dice(pred, target, eps=1e-6):
    """Return a (C,) tensor of mean Dice per channel over the batch."""
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))             # (B, C)
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    # Channels empty in BOTH pred and target score a perfect 1.0.
    dice = (2 * inter + eps) / (union + eps)            # (B, C)
    return dice.mean(dim=0)                             # (C,)


class DiceBCELoss(nn.Module):
    """BCE + soft-Dice, averaged across channels (each label weighted equally)."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, eps=1e-6):
        bce_loss = self.bce(pred, target)
        pred_s = torch.sigmoid(pred)
        inter  = (pred_s * target).sum(dim=(2, 3))      # (B, C)
        union  = pred_s.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1 - ((2 * inter + eps) / (union + eps)).mean()
        return bce_loss + dice_loss


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    dice_sum = torch.zeros(NUM_CLASSES)
    n = 0
    with torch.set_grad_enabled(train):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = criterion(preds, masks)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = len(imgs)
            total_loss += loss.item() * bs
            dice_sum += per_channel_dice(preds, masks).detach().cpu() * bs
            n += bs
    return total_loss / n, (dice_sum / n)               # mean loss, (C,) dice


def train_stage(stage, model, train_loader, val_loader, device, lr, epochs, patience):
    print(f"\n{'='*54}")
    print(f"Stage {stage}  |  LR={lr}  |  Max epochs={epochs}")
    print(f"{'='*54}")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = DiceBCELoss()
    Path(CHECKPOINT).parent.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_loss, vl_dice = run_epoch(model, val_loader,   criterion, optimizer, device, False)
        scheduler.step()

        tr_mean = tr_dice.mean().item()
        vl_mean = vl_dice.mean().item()
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_mean); history["val_dice"].append(vl_mean)

        if vl_mean > best_dice:
            best_dice = vl_mean
            torch.save({"epoch": epoch, "stage": stage,
                        "model_state": model.state_dict(),
                        "val_dice": vl_mean,
                        "val_dice_per_channel": vl_dice.tolist(),
                        "channels": CHANNELS}, CHECKPOINT)
            patience_counter = 0
            tag = " <- best"
        else:
            patience_counter += 1
            tag = f" (patience {patience_counter}/{patience})"

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"mDice(tr)={tr_mean:.4f}  mDice(val)={vl_mean:.4f}{tag}")

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    return history, best_dice


def save_plot(h1, h2, path):
    tl = h1["train_loss"] + h2["train_loss"]
    vl = h1["val_loss"]   + h2["val_loss"]
    td = h1["train_dice"] + h2["train_dice"]
    vd = h1["val_dice"]   + h2["val_dice"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(tl, label="Train"); ax1.plot(vl, label="Val")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(td, label="Train"); ax2.plot(vd, label="Val")
    ax2.set_title("Mean Dice (all channels)"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()
    print(f"  Training plot -> {path}")


def report_per_channel(model, val_loader, device):
    print("\nPer-channel validation Dice (best model):")
    model.eval()
    dice_sum = torch.zeros(NUM_CLASSES)
    n = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            bs = len(imgs)
            dice_sum += per_channel_dice(preds, masks).cpu() * bs
            n += bs
    dice = dice_sum / n
    for i, name in enumerate(CHANNELS):
        print(f"  ch{i:2d} {name:16s}: {dice[i]:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Output channels: {NUM_CLASSES}")

    print("\nLoading datasets...")
    train_ds = JetMultiDataset(DATA_DIR, "train", augment=True)
    val_ds   = JetMultiDataset(DATA_DIR, "val",   augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nBuilding MobileNetV2 U-Net (15-channel head)...")
    model = JetUNetMobile(freeze_encoder=True, out_channels=NUM_CLASSES).to(device)

    h1, best1 = train_stage(1, model, train_loader, val_loader, device,
                            LR_DECODER, STAGE1_EPOCHS, PATIENCE)
    print(f"\n  Stage 1 best mean Dice: {best1:.4f}")

    for p in model.parameters():
        p.requires_grad = True
    print("Encoder unfrozen — full fine-tune")
    h2, best2 = train_stage(2, model, train_loader, val_loader, device,
                            LR_FINETUNE, STAGE2_EPOCHS, PATIENCE)
    print(f"\n  Stage 2 best mean Dice: {best2:.4f}")

    save_plot(h1, h2, PLOT_PATH)

    # Reload best checkpoint and report per-channel performance.
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    report_per_channel(model, val_loader, device)

    print(f"\nTraining complete.")
    print(f"  Best model -> '{CHECKPOINT}'  (mean Dice = {max(best1, best2):.4f})")


if __name__ == "__main__":
    main()
