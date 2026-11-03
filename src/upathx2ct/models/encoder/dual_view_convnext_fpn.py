from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 4, layer_scale_init: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, mlp_ratio * channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * channels, channels)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(channels)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtFPNBackbone(nn.Module):
    """ConvNeXt-style backbone that exposes P2-P5-ready feature maps.

    For a 512x512 input, c2/c3/c4/c5 are 128/64/32/16 spatial maps.
    """

    VARIANTS = {
        "nano": {"dims": (48, 96, 192, 384), "depths": (2, 2, 4, 2)},
        "tiny": {"dims": (96, 192, 384, 768), "depths": (3, 3, 9, 3)},
        "small": {"dims": (96, 192, 384, 768), "depths": (3, 3, 27, 3)},
    }

    def __init__(self, in_channels: int = 1, variant: str = "tiny", drop_path_rate: float = 0.1):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown ConvNeXt variant {variant!r}; choose one of {sorted(self.VARIANTS)}.")
        spec = self.VARIANTS[variant]
        dims = spec["dims"]
        depths = spec["depths"]
        self.out_channels = {
            "c2": dims[0],
            "c3": dims[1],
            "c4": dims[2],
            "c5": dims[3],
        }

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsamples = nn.ModuleList([
            nn.Sequential(LayerNorm2d(dims[i]), nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            for i in range(3)
        ])

        total_depth = sum(depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_depth).tolist()
        idx = 0
        stages = []
        for stage_idx, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(dims[stage_idx], drop_path=drop_rates[idx]))
                idx += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        c2 = self.stages[0](self.stem(x))
        c3 = self.stages[1](self.downsamples[0](c2))
        c4 = self.stages[2](self.downsamples[1](c3))
        c5 = self.stages[3](self.downsamples[2](c4))
        return OrderedDict([("c2", c2), ("c3", c3), ("c4", c4), ("c5", c5)])


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels: dict[str, int], out_channels: dict[str, int]):
        super().__init__()
        self.levels = ("p2", "p3", "p4", "p5")
        in_keys = ("c2", "c3", "c4", "c5")
        self.lateral = nn.ModuleDict({
            p: nn.Conv2d(in_channels[c], out_channels[p], kernel_size=1)
            for p, c in zip(self.levels, in_keys)
        })
        self.output = nn.ModuleDict({
            p: nn.Sequential(
                nn.Conv2d(out_channels[p], out_channels[p], kernel_size=3, padding=1),
                LayerNorm2d(out_channels[p]),
                nn.GELU(),
            )
            for p in self.levels
        })
        self.proj_topdown = nn.ModuleDict({
            "p5_to_p4": nn.Conv2d(out_channels["p5"], out_channels["p4"], kernel_size=1),
            "p4_to_p3": nn.Conv2d(out_channels["p4"], out_channels["p3"], kernel_size=1),
            "p3_to_p2": nn.Conv2d(out_channels["p3"], out_channels["p2"], kernel_size=1),
        })

    def forward(self, feats: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        p2 = self.lateral["p2"](feats["c2"])
        p3 = self.lateral["p3"](feats["c3"])
        p4 = self.lateral["p4"](feats["c4"])
        p5 = self.lateral["p5"](feats["c5"])

        p4 = p4 + F.interpolate(self.proj_topdown["p5_to_p4"](p5), size=p4.shape[-2:], mode="nearest")
        p3 = p3 + F.interpolate(self.proj_topdown["p4_to_p3"](p4), size=p3.shape[-2:], mode="nearest")
        p2 = p2 + F.interpolate(self.proj_topdown["p3_to_p2"](p3), size=p2.shape[-2:], mode="nearest")

        return OrderedDict([
            ("p2", self.output["p2"](p2)),
            ("p3", self.output["p3"](p3)),
            ("p4", self.output["p4"](p4)),
            ("p5", self.output["p5"](p5)),
        ])


class CrossViewAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        heads = max(1, min(num_heads, channels // 32))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.ap_norm = nn.LayerNorm(channels)
        self.lat_norm = nn.LayerNorm(channels)
        self.ap_to_lat = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.lat_to_ap = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ap_scale = nn.Parameter(torch.zeros(1))
        self.lat_scale = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), (h, w)

    @staticmethod
    def _unflatten(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        h, w = hw
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], h, w)

    def forward(self, ap: torch.Tensor, lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ap_tokens, hw = self._flatten(ap)
        lat_tokens, _ = self._flatten(lat)
        ap_attn, _ = self.ap_to_lat(self.ap_norm(ap_tokens), self.lat_norm(lat_tokens), self.lat_norm(lat_tokens), need_weights=False)
        lat_attn, _ = self.lat_to_ap(self.lat_norm(lat_tokens), self.ap_norm(ap_tokens), self.ap_norm(ap_tokens), need_weights=False)
        ap = ap + self.ap_scale * self._unflatten(ap_attn, hw)
        lat = lat + self.lat_scale * self._unflatten(lat_attn, hw)
        return ap, lat


class DualViewConvNeXtFPNEncoder(nn.Module):
    """Shared ConvNeXt-FPN encoder for AP/LAT radiograph-style views."""

    def __init__(
        self,
        in_channels: int = 1,
        variant: str = "tiny",
        fpn_channels: tuple[int, int, int, int] = (128, 192, 256, 384),
        global_dim: int = 256,
        drop_path_rate: float = 0.1,
        cross_view_levels: tuple[str, ...] = ("p4", "p5"),
    ):
        super().__init__()
        self.levels = ("p2", "p3", "p4", "p5")
        self.backbone = ConvNeXtFPNBackbone(in_channels=in_channels, variant=variant, drop_path_rate=drop_path_rate)
        out_channels = dict(zip(self.levels, fpn_channels))
        self.out_channels = out_channels
        self.fpn = FeaturePyramid(self.backbone.out_channels, out_channels)
        self.view_embeddings = nn.ParameterDict({
            f"ap_{level}": nn.Parameter(torch.zeros(1, out_channels[level], 1, 1))
            for level in self.levels
        })
        self.view_embeddings.update({
            f"lat_{level}": nn.Parameter(torch.zeros(1, out_channels[level], 1, 1))
            for level in self.levels
        })
        self.cross_view = nn.ModuleDict({
            level: CrossViewAttention(out_channels[level])
            for level in cross_view_levels
        })
        pooled_dim = sum(out_channels.values())
        self.global_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, global_dim),
            nn.GELU(),
            nn.Linear(global_dim, global_dim),
        )
        self.global_dim = global_dim
        self.variant = variant

    def _encode_one(self, x: torch.Tensor, view: str) -> OrderedDict[str, torch.Tensor]:
        feats = self.fpn(self.backbone(x))
        return OrderedDict([
            (level, feat + self.view_embeddings[f"{view}_{level}"])
            for level, feat in feats.items()
        ])

    def _global(self, feats: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
        pooled = [feat.mean(dim=(-2, -1)) for feat in feats.values()]
        return self.global_proj(torch.cat(pooled, dim=1))

    def forward(self, x_ap: torch.Tensor, x_lat: torch.Tensor) -> dict[str, object]:
        ap_feats = self._encode_one(x_ap, "ap")
        lat_feats = self._encode_one(x_lat, "lat")
        for level, attention in self.cross_view.items():
            ap_feats[level], lat_feats[level] = attention(ap_feats[level], lat_feats[level])
        return {
            "ap_feats": ap_feats,
            "lat_feats": lat_feats,
            "ap_global": self._global(ap_feats),
            "lat_global": self._global(lat_feats),
            "feature_levels": list(self.levels),
            "channels": dict(self.out_channels),
        }
