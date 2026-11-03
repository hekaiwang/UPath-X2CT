from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F



def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps + offset) / (1.0 + offset)) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


@dataclass(frozen=True)
class DiffusionConfig:
    latent_channels: int = 8
    condition_channels: int = 96
    model_channels: tuple[int, int, int] = (96, 192, 384)
    num_res_blocks: tuple[int, int, int] = (2, 2, 2)
    attention_levels: tuple[bool, bool, bool] = (False, True, True)
    train_timesteps: int = 1000
    residual_scale: float = 1.0
    condition_dropout: float = 0.10
    min_snr_gamma: float = 5.0


class ResidualLatentDiffusion(nn.Module):
    """Diffusion over residual latents around a deterministic reconstruction."""

    def __init__(self, config: DiffusionConfig = DiffusionConfig()) -> None:
        super().__init__()
        try:
            from generative.networks.nets import DiffusionModelUNet
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install MONAI GenerativeModels to use B0 diffusion") from exc
        self.config = config
        in_channels = config.latent_channels * 2 + config.condition_channels
        self.denoiser = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=config.latent_channels,
            num_res_blocks=config.num_res_blocks,
            num_channels=config.model_channels,
            attention_levels=config.attention_levels,
            norm_num_groups=32,
            num_head_channels=32,
            with_conditioning=False,
            resblock_updown=True,
        )
        betas = cosine_beta_schedule(config.train_timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("residual_scale", torch.full((1, config.latent_channels, 1, 1, 1), config.residual_scale))

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)

    def set_residual_scale(self, scale: torch.Tensor | float) -> None:
        value = torch.as_tensor(scale, dtype=self.residual_scale.dtype, device=self.residual_scale.device)
        if value.numel() == 1:
            value = value.expand(self.config.latent_channels)
        value = value.reshape(1, -1, 1, 1, 1).clamp_min(1e-4)
        if value.shape != self.residual_scale.shape:
            raise ValueError(f"Residual scale shape mismatch: {value.shape}")
        self.residual_scale.copy_(value)

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        output = values.gather(0, timesteps)
        return output.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, clean.shape) * clean
            + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, clean.shape) * noise
        )

    def predict_clean(self, noisy: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, noisy.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, noisy.shape)
        return (noisy - sqrt_one_minus * noise) / sqrt_alpha.clamp_min(1e-6)

    def _condition_input(
        self,
        noisy_residual: torch.Tensor,
        coarse_latent: torch.Tensor,
        condition: torch.Tensor,
        drop_condition: bool,
    ) -> torch.Tensor:
        if drop_condition and self.training and self.config.condition_dropout > 0:
            keep = (
                torch.rand(noisy_residual.shape[0], 1, 1, 1, 1, device=noisy_residual.device)
                >= self.config.condition_dropout
            ).to(noisy_residual.dtype)
            condition = condition * keep
        return torch.cat([noisy_residual, coarse_latent, condition], dim=1)

    def denoise(
        self,
        noisy_residual: torch.Tensor,
        timesteps: torch.Tensor,
        coarse_latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        model_input = self._condition_input(noisy_residual, coarse_latent, condition, drop_condition=False)
        return self.denoiser(model_input, timesteps=timesteps)

    def training_loss(
        self,
        target_latent: torch.Tensor,
        coarse_latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        residual = self.latent_to_target(target_latent, coarse_latent)
        noise = torch.randn_like(residual)
        timesteps = torch.randint(
            0,
            self.config.train_timesteps,
            (residual.shape[0],),
            device=residual.device,
            dtype=torch.long,
        )
        noisy = self.add_noise(residual, noise, timesteps)
        model_input = self._condition_input(noisy, coarse_latent, condition, drop_condition=True)
        predicted_noise = self.denoiser(model_input, timesteps=timesteps)
        per_sample_mse = (predicted_noise - noise).square().flatten(1).mean(1)
        alpha = self._extract(self.alphas_cumprod, timesteps, residual.shape).flatten()
        snr = alpha / (1.0 - alpha).clamp_min(1e-6)
        weights = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma)) / snr.clamp_min(1e-6)
        noise_loss = (weights * per_sample_mse).mean()
        predicted_residual = self.predict_clean(noisy, predicted_noise, timesteps)
        clean_loss = F.smooth_l1_loss(predicted_residual, residual)
        predicted_latent = self.target_to_latent(predicted_residual, coarse_latent)
        return {
            "loss_noise": noise_loss,
            "loss_clean": clean_loss,
            "predicted_latent": predicted_latent,
            "predicted_residual": predicted_residual,
            "timesteps": timesteps,
        }

    def forward(
        self,
        target_latent: torch.Tensor,
        coarse_latent: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.training_loss(target_latent, coarse_latent, condition)

    @torch.no_grad()
    def sample(
        self,
        coarse_latent: torch.Tensor,
        condition: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if steps < 2:
            raise ValueError("At least two DDIM steps are required")
        residual = torch.randn(
            coarse_latent.shape,
            device=coarse_latent.device,
            dtype=coarse_latent.dtype,
            generator=generator,
        )
        schedule = torch.linspace(
            self.config.train_timesteps - 1,
            0,
            steps,
            device=coarse_latent.device,
        ).round().long()
        schedule = torch.unique_consecutive(schedule)
        for index, timestep_value in enumerate(schedule):
            timestep = torch.full(
                (residual.shape[0],),
                int(timestep_value.item()),
                device=residual.device,
                dtype=torch.long,
            )
            predicted_noise = self.denoise(residual, timestep, coarse_latent, condition)
            predicted_clean = self.predict_clean(residual, predicted_noise, timestep).clamp(-5.0, 5.0)
            if index == len(schedule) - 1:
                residual = predicted_clean
                break
            previous_value = int(schedule[index + 1].item())
            alpha_t = self.alphas_cumprod[int(timestep_value.item())]
            alpha_previous = self.alphas_cumprod[previous_value]
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha_t)).clamp_min(0.0)
                * (1.0 - alpha_t / alpha_previous).clamp_min(0.0)
            )
            direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0)) * predicted_noise
            random_noise = torch.randn(
                residual.shape,
                device=residual.device,
                dtype=residual.dtype,
                generator=generator,
            )
            residual = torch.sqrt(alpha_previous) * predicted_clean + direction + sigma * random_noise
        return self.target_to_latent(residual, coarse_latent)

    def latent_to_target(self, latent: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return (latent - coarse) / self.residual_scale

    def target_to_latent(self, target: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return coarse + target * self.residual_scale
