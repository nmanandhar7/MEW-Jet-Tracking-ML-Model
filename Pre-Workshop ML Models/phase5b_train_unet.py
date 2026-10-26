"""
Phase 5b — Train the U-Net to segment polymer jets.

Dataset: pairs of (frame image, binary jet mask).
Loss:    combination of Dice loss + BCE — standard for thin-structure segmentation.
Metric:  Dice coefficient (overlap between predicted and true mask, 0-1).

Two-stage training:
  Stage 1: frozen encoder, train decoder only (fast warmup)
  Stage 2: unfreeze encoder, fine-tune everything

Run from your project root:
    python phase5b_train_unet.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from phase4b_unet_model import JetUNet

#Config
LABELED_DIR   = "data/labeled"
MASKS_DIR     = "data/masks"
CHECKPOINT    = "checkpoints/unet_best.pth"
PLOT_PATH     = "checkpoints/unet_training.png"

# Resize frames to this (H, W) for training — keeps aspect ratio of ~1262x228
RESIZE_H      = 224
RESIZE_W      = 416

BATCH_SIZE    = 8
LR_DECODER    = 1e-3
LR_FINETUNE   = 1e-4
STAGE1_EPOCHS = 12
STAGE2_EPOCHS = 25
PATIENCE      = 8



class JetSegDataset(Dataset):
    def __init__(self, labeled_dir, masks_dir, split, augment=False):
        self.img_dir  = Path(labeled_dir) / split
        self.mask_dir = Path(masks_dir) / split
        self.augment  = augment

        # Only include frames that have a corresponding mask
        self.samples = []
        for mask_path in sorted(self.mask_dir.glob("*.png")):
            img_path = self.img_dir / mask_path.name
            if img_path.exists():
                self.samples.append((img_path, mask_path))

        print(f"  {split:6s}: {len(self.samples)} image/mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img  = cv2.imread(str(img_path))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize both
        img  = cv2.resize(img,  (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_NEAREST)

        # Augment — horizontal flip
        if self.augment and np.random.rand() > 0.5:
            img  = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # To tensors
        img  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet stats
        img  = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0  # 0 or 1

        return img, mask


def dice_coeff(pred, target, eps=1e-6):
    """Dice overlap between predicted probabilities and target mask."""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice  = (2*inter + eps) / (union + eps)
    return dice.mean().item()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss — robust for thin/sparse masks."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, eps=1e-6):
        bce_loss = self.bce(pred, target)
        pred_s = torch.sigmoid(pred)
        inter  = (pred_s * target).sum(dim=(1,2,3))
        union  = pred_s.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice_loss = 1 - ((2*inter + eps) / (union + eps)).mean()
        return bce_loss + dice_loss


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_dice = 0.0
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

            total_loss += loss.item() * len(imgs)
            total_dice += dice_coeff(preds, masks) * len(imgs)
            n += len(imgs)

    return total_loss / n, total_dice / n


def train_stage(stage, model, train_loader, val_loader, device, lr, epochs, patience):
    print(f"\n{'='*52}")
    print(f"Stage {stage}  |  LR={lr}  |  Max epochs={epochs}")
    print(f"{'='*52}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
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

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice); history["val_dice"].append(vl_dice)

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save({
                "epoch": epoch, "stage": stage,
                "model_state": model.state_dict(),
                "val_dice": vl_dice,
            }, CHECKPOINT)
            patience_counter = 0
            tag = " <- best"
        else:
            patience_counter += 1
            tag = f" (patience {patience_counter}/{patience})"

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
              f"train_dice={tr_dice:.4f}  val_dice={vl_dice:.4f}{tag}")

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
    ax2.set_title("Dice coefficient"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()
    print(f"  Training plot -> {path}")


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    print("\nLoading datasets...")
    train_ds = JetSegDataset(LABELED_DIR, MASKS_DIR, "train", augment=True)
    val_ds   = JetSegDataset(LABELED_DIR, MASKS_DIR, "val",   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\nBuilding U-Net...")
    model = JetUNet(freeze_encoder=True).to(device)

    # Stage 1 — decoder only
    h1, best1 = train_stage(1, model, train_loader, val_loader, device,
                            LR_DECODER, STAGE1_EPOCHS, PATIENCE)
    print(f"\n  Stage 1 best Dice: {best1:.4f}")

    # Stage 2 — full fine-tune
    for p in model.parameters():
        p.requires_grad = True
    print("Encoder unfrozen — full fine-tune")
    h2, best2 = train_stage(2, model, train_loader, val_loader, device,
                            LR_FINETUNE, STAGE2_EPOCHS, PATIENCE)
    print(f"\n  Stage 2 best Dice: {best2:.4f}")

    save_plot(h1, h2, PLOT_PATH)

    print(f"\nTraining complete.")
    print(f"  Best model -> '{CHECKPOINT}'  (Dice = {max(best1, best2):.4f})")
    print(f"\nNext: run phase6b_evaluate_unet.py to test and visualize.")


if __name__ == "__main__":
    main()
