"""
Custom encoder–decoder segmentation CNN with skip connections.
Not U-Net: distinct fusion blocks and channel schedule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Conv -> ReLU -> BatchNorm; return pre-pool features as skip, then MaxPool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        skip = x
        x = self.pool(x)
        return x, skip


class FuseThenUpsample(nn.Module):
    """Concatenate skip at current resolution, refine, then 2x upsample."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        fused = in_ch + skip_ch
        self.refine = nn.Sequential(
            nn.Conv2d(fused, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
        self.up = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.refine(x)
        return self.up(x)


class SkipBridgedEncoderDecoder(nn.Module):
    """
    Encoder: 3→64→128→256 with MaxPool after each block.
    Skips: 64 @ H/2, 128 @ H/4, 256 @ H/8.
    Decoder: fuse + upsample from bottleneck (H/8) to full resolution.
    """

    def __init__(self, num_classes: int, in_ch: int = 3) -> None:
        super().__init__()
        self.enc1 = EncoderBlock(in_ch, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
        )

        self.dec2 = FuseThenUpsample(256, 256, 128)
        self.dec1 = FuseThenUpsample(128, 128, 64)
        self.dec0 = FuseThenUpsample(64, 64, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_hw = x.shape[2:]
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x = self.bottleneck(x)
        x = self.dec2(x, s3)
        x = self.dec1(x, s2)
        x = self.dec0(x, s1)
        if x.shape[2:] != in_hw:
            x = F.interpolate(x, size=in_hw, mode="bilinear", align_corners=False)
        return self.head(x)
