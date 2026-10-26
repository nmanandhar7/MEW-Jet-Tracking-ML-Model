"""
Phase 4b — U-Net segmentation model for jet detection.

Architecture:
  - Encoder: ResNet-34 pretrained on ImageNet (downsamples, extracts features)
  - Decoder: upsampling path with skip connections (rebuilds spatial detail)
  - Output:  single-channel mask, same size as input (jet probability per pixel)

This is the classic U-Net design adapted to use a pretrained backbone,
which makes it train fast and accurately even on a small dataset.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    """One decoder step: upsample + concat skip connection + two convs."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample x to match skip's spatial size
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class JetUNet(nn.Module):
    def __init__(self, freeze_encoder=False):
        super().__init__()

        # ── Encoder: ResNet-34 ──────────────────────────────────────────────
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        resnet  = models.resnet34(weights=weights)

        # Break ResNet into stages so we can grab skip connections
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64ch,  /2
        self.pool = resnet.maxpool                                         #        /4
        self.enc1 = resnet.layer1   # 64 channels   /4
        self.enc2 = resnet.layer2   # 128 channels  /8
        self.enc3 = resnet.layer3   # 256 channels  /16
        self.enc4 = resnet.layer4   # 512 channels  /32

        if freeze_encoder:
            for m in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
                for p in m.parameters():
                    p.requires_grad = False
            print("  Encoder frozen")

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(512, 256, 256)   # combine enc4 + enc3
        self.dec3 = DecoderBlock(256, 128, 128)   # + enc2
        self.dec2 = DecoderBlock(128,  64,  64)   # + enc1
        self.dec1 = DecoderBlock( 64,  64,  32)   # + enc0

        # Final output: 1 channel (jet probability)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder with saved skips
        e0 = self.enc0(x)          # /2,  64ch
        e1 = self.enc1(self.pool(e0))  # /4,  64ch
        e2 = self.enc2(e1)         # /8,  128ch
        e3 = self.enc3(e2)         # /16, 256ch
        e4 = self.enc4(e3)         # /32, 512ch

        # Decoder with skip connections
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)

        # Upsample back to full input resolution
        out = F.interpolate(d1, size=input_size, mode='bilinear', align_corners=False)
        out = self.final(out)      # (B, 1, H, W) — raw logits
        return out


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("Building U-Net...")
    model = JetUNet(freeze_encoder=False)

    total, trainable = count_parameters(model)
    print(f"\nModel summary:")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")

    # Test forward pass with your frame dimensions
    dummy = torch.randn(2, 3, 224, 416)  # H, W (resized frame)
    out = model(dummy)
    print(f"\nForward pass test:")
    print(f"  Input shape  : {dummy.shape}")
    print(f"  Output shape : {out.shape}  (should match input H,W with 1 channel)")
    print("\nU-Net looks good!")
