"""
Phase 4c — Lightweight U-Net (MobileNetV2 encoder) for real-time jet segmentation.

Same encoder-decoder idea as the ResNet-50 version (phase4b), but the backbone is
MobileNetV2 — ~10x fewer parameters and far cheaper convolutions. The point is
speed: this is the model you'd run live for the PID loop.

Drop-in compatible with the rest of the pipeline:
  - same input (3-channel image), same output (1-channel mask logits)
  - trains on the same masks from phase3b_make_masks.py
  - works with phase7b_infer_video.py and jet_annotator_app.py (just point the
    checkpoint at this model — see note at bottom)

MobileNetV2 feature taps used for skip connections (input 3xHxW):
    block 1  ->  16 ch  @ /2
    block 3  ->  24 ch  @ /4
    block 6  ->  32 ch  @ /8
    block 13 ->  96 ch  @ /16
    block 18 -> 1280 ch @ /32   (deepest features)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# Indices in mobilenet.features where resolution has just dropped.
# We grab the output AFTER each of these blocks as a skip connection.
TAP_POINTS = [1, 3, 6, 13, 18]
TAP_CHANNELS = [16, 24, 32, 96, 1280]


class DecoderBlock(nn.Module):
    """Upsample + concat skip + two depthwise-ish conv layers."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class JetUNetMobile(nn.Module):
    def __init__(self, freeze_encoder=False, out_channels=1):
        super().__init__()

        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features  # 19 blocks

        if freeze_encoder:
            for p in self.features.parameters():
                p.requires_grad = False
            print("  Encoder frozen")

        # Decoder: mirror the taps from deepest -> shallowest.
        # channels: 1280 -> 96 -> 32 -> 24 -> 16
        c5, c4, c3, c2, c1 = TAP_CHANNELS[4], TAP_CHANNELS[3], TAP_CHANNELS[2], \
                             TAP_CHANNELS[1], TAP_CHANNELS[0]
        self.dec4 = DecoderBlock(c5,  c4, 96)    # combine deepest (1280) + /16 skip (96)
        self.dec3 = DecoderBlock(96,  c3, 32)    # + /8 skip (32)
        self.dec2 = DecoderBlock(32,  c2, 24)    # + /4 skip (24)
        self.dec1 = DecoderBlock(24,  c1, 16)    # + /2 skip (16)

        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Run through the backbone, capturing the tap outputs
        skips = {}
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i in TAP_POINTS:
                skips[i] = out

        e1, e2, e3, e4, e5 = (skips[1], skips[3], skips[6], skips[13], skips[18])

        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        out = self.final(out)   # (B, 1, H, W) raw logits
        return out


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("Building MobileNetV2 U-Net...")
    model = JetUNetMobile(freeze_encoder=False)

    total, trainable = count_parameters(model)
    print(f"\nModel summary:")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  (ResNet-50 U-Net for comparison was ~32M+ params)")

    dummy = torch.randn(2, 3, 224, 416)
    out = model(dummy)
    print(f"\nForward pass test:")
    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(out.shape)}  (expect (2, 1, 224, 416))")

    # Rough single-frame CPU timing
    import time
    model.eval()
    one = torch.randn(1, 3, 224, 416)
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(one)
        t0 = time.time()
        for _ in range(10):
            model(one)
        dt = (time.time() - t0) / 10
    print(f"\nApprox CPU latency: {dt*1000:.0f} ms/frame  (~{1/dt:.0f} FPS)")
    print("Note: your machine will differ; this sandbox CPU is not representative.")
