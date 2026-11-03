from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


class FourierPositionEncoding3D(nn.Module):
    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_channels = 3 + 2 * 3 * num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode normalized xyz-like coordinates shaped [1, 3, D, H, W]."""
        enc = [coords]
        for freq in self.freqs.to(dtype=coords.dtype, device=coords.device):
            scaled = coords * freq
            enc.append(torch.sin(torch.pi * scaled))
            enc.append(torch.cos(torch.pi * scaled))
        return torch.cat(enc, dim=1)
