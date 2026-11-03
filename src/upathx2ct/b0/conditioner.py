from __future__ import annotations


import torch
from torch import nn
import torch.nn.functional as F



def _groups(channels: int) -> int:
    for groups in range(min(16, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual










class LatentUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = ResBlock3D(in_channels, base_channels, dropout)
        self.down1 = nn.Sequential(nn.Conv3d(base_channels, base_channels * 2, 3, 2, 1), ResBlock3D(base_channels * 2, base_channels * 2, dropout))
        self.down2 = nn.Sequential(nn.Conv3d(base_channels * 2, base_channels * 4, 3, 2, 1), ResBlock3D(base_channels * 4, base_channels * 4, dropout))
        self.mid = nn.Sequential(ResBlock3D(base_channels * 4, base_channels * 4, dropout), ResBlock3D(base_channels * 4, base_channels * 4, dropout))
        self.up1 = ResBlock3D(base_channels * 4 + base_channels * 2, base_channels * 2, dropout)
        self.up2 = ResBlock3D(base_channels * 2 + base_channels, base_channels, dropout)
        self.out = nn.Sequential(nn.GroupNorm(_groups(base_channels), base_channels), nn.SiLU(), nn.Conv3d(base_channels, out_channels, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.down1(s0)
        x = self.mid(self.down2(s1))
        x = F.interpolate(x, size=s1.shape[-3:], mode="trilinear", align_corners=False)
        x = self.up1(torch.cat([x, s1], dim=1))
        x = F.interpolate(x, size=s0.shape[-3:], mode="trilinear", align_corners=False)
        x = self.up2(torch.cat([x, s0], dim=1))
        return self.out(x)
