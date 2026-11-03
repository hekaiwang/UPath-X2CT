from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from upathx2ct.b0.ema import resolve_checkpoint_model_state


@dataclass(frozen=True)
class AutoencoderConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    num_res_blocks: tuple[int, ...] = (2, 2, 2)
    attention_levels: tuple[bool, ...] = (False, False, True)
    latent_channels: int = 8
    norm_num_groups: int = 32
    with_encoder_nonlocal_attention: bool = True
    with_decoder_nonlocal_attention: bool = True
    use_flash_attention: bool = False
    use_checkpointing: bool = True


class HighFidelityAutoencoderKL(nn.Module):
    """Continuous 3D latent autoencoder trained on the target CT domain."""

    def __init__(self, config: AutoencoderConfig = AutoencoderConfig()) -> None:
        super().__init__()
        try:
            from generative.networks.nets import AutoencoderKL
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise ImportError("Install MONAI GenerativeModels to use B0") from exc
        self.config = config
        self.net = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=config.num_res_blocks,
            num_channels=config.channels,
            attention_levels=config.attention_levels,
            latent_channels=config.latent_channels,
            norm_num_groups=config.norm_num_groups,
            with_encoder_nonlocal_attn=config.with_encoder_nonlocal_attention,
            with_decoder_nonlocal_attn=config.with_decoder_nonlocal_attention,
            use_flash_attention=config.use_flash_attention,
            use_checkpointing=config.use_checkpointing,
            use_convtranspose=False,
        )
        self.register_buffer("latent_mean", torch.zeros(1, config.latent_channels, 1, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, config.latent_channels, 1, 1, 1))

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.config.channels) - 1)

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        mean = mean.reshape(1, -1, 1, 1, 1).to(self.latent_mean)
        std = std.reshape(1, -1, 1, 1, 1).to(self.latent_std).clamp_min(1e-6)
        if mean.shape != self.latent_mean.shape:
            raise ValueError(f"Latent mean shape mismatch: {mean.shape} vs {self.latent_mean.shape}")
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def set_decoder_only_trainable(self) -> None:
        """Freeze the latent coordinate system while adapting CT reconstruction."""

        self.requires_grad_(False)
        self.net.post_quant_conv.requires_grad_(True)
        self.net.decoder.requires_grad_(True)

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.latent_mean) / self.latent_std

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.latent_std + self.latent_mean

    def encode_distribution(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net.encode(volume)

    def encode(self, volume: torch.Tensor, sample: bool = False, normalize: bool = True) -> torch.Tensor:
        mean, sigma = self.encode_distribution(volume)
        latent = self.net.sampling(mean, sigma) if sample else mean
        return self.normalize_latent(latent) if normalize else latent

    def decode(self, latent: torch.Tensor, normalized: bool = True, clamp: bool = False) -> torch.Tensor:
        if normalized:
            latent = self.denormalize_latent(latent)
        reconstruction = self.net.decode(latent)
        return reconstruction.clamp(-1.0, 1.0) if clamp else reconstruction

    def forward(self, volume: torch.Tensor) -> dict[str, torch.Tensor]:
        reconstruction, mean, sigma = self.net(volume)
        return {
            "reconstruction": reconstruction,
            "mean": mean,
            "sigma": sigma,
            "latent": self.normalize_latent(mean),
        }


def load_autoencoder_checkpoint(path: str, device: torch.device | str = "cpu") -> HighFidelityAutoencoderKL:
    checkpoint = torch.load(path, map_location=device)
    return autoencoder_from_checkpoint(checkpoint)


def autoencoder_from_checkpoint(checkpoint: dict) -> HighFidelityAutoencoderKL:
    cfg = checkpoint.get("autoencoder_config", {})
    config = AutoencoderConfig(
        channels=tuple(cfg.get("channels", AutoencoderConfig.channels)),
        num_res_blocks=tuple(cfg.get("num_res_blocks", AutoencoderConfig.num_res_blocks)),
        attention_levels=tuple(cfg.get("attention_levels", AutoencoderConfig.attention_levels)),
        latent_channels=int(cfg.get("latent_channels", AutoencoderConfig.latent_channels)),
        norm_num_groups=int(cfg.get("norm_num_groups", AutoencoderConfig.norm_num_groups)),
        with_encoder_nonlocal_attention=bool(
            cfg.get("with_encoder_nonlocal_attention", True)
        ),
        with_decoder_nonlocal_attention=bool(
            cfg.get("with_decoder_nonlocal_attention", True)
        ),
        use_flash_attention=bool(cfg.get("use_flash_attention", False)),
        use_checkpointing=bool(cfg.get("use_checkpointing", False)),
    )
    model = HighFidelityAutoencoderKL(config)
    state = resolve_checkpoint_model_state(checkpoint)
    model.load_state_dict(state, strict=True)
    return model
