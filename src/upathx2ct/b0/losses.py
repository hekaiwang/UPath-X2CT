from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from upathx2ct.b0.projection import ProjectionConfig, projection_consistency_loss


def normalized_to_hu(volume: torch.Tensor, hu_min: float, hu_max: float) -> torch.Tensor:
    return (volume + 1.0) * 0.5 * (hu_max - hu_min) + hu_min


def window_to_unit(volume_hu: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return torch.clamp((volume_hu - low) / (high - low), 0.0, 1.0)


def multi_window_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    hu_min: float = -1024.0,
    hu_max: float = 1500.0,
) -> torch.Tensor:
    pred_hu = normalized_to_hu(prediction, hu_min, hu_max)
    target_hu = normalized_to_hu(target, hu_min, hu_max)
    windows = ((-1024.0, 300.0), (-1000.0, -300.0), (-160.0, 240.0), (-500.0, 1500.0))
    return torch.stack(
        [
            F.l1_loss(window_to_unit(pred_hu, low, high), window_to_unit(target_hu, low, high))
            for low, high in windows
        ]
    ).mean()


def gradient_l1_3d(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    loss = prediction.new_zeros(())
    for axis in (-3, -2, -1):
        pred_grad = torch.diff(prediction, dim=axis)
        target_grad = torch.diff(target, dim=axis)
        loss = loss + F.l1_loss(pred_grad, target_grad)
    return loss / 3.0


def gradient_energy_calibration_loss_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Calibrate directional gradient energy while edge losses constrain texture."""

    losses = []
    for axis in (-3, -2, -1):
        prediction_energy = torch.diff(prediction, dim=axis).abs().flatten(1).mean(1)
        target_energy = torch.diff(target, dim=axis).abs().flatten(1).mean(1)
        log_ratio = torch.log(prediction_energy + epsilon) - torch.log(target_energy + epsilon)
        losses.append(F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=0.1))
    return torch.stack(losses).mean()


def high_frequency_l1_3d(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match local 3D high-pass residuals without rewarding hallucinated texture."""

    prediction_high = prediction - F.avg_pool3d(prediction, kernel_size=3, stride=1, padding=1)
    target_high = target - F.avg_pool3d(target, kernel_size=3, stride=1, padding=1)
    return F.l1_loss(prediction_high, target_high)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).to(prediction.dtype)
    return (torch.abs(prediction - target) * mask).sum() / mask.sum().clamp_min(1.0)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target = target.to(probability.dtype)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype)) + soft_dice_loss(logits, target)


def labeled_lesion_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    has_label: torch.Tensor,
) -> torch.Tensor:
    labeled = has_label.reshape(-1) > 0.5
    if not bool(labeled.any()):
        return logits.sum() * 0.0
    logits = logits[labeled]
    target = target[labeled].to(logits.dtype)
    positives = target.sum()
    negatives = target.numel() - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 50.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
    return bce + soft_dice_loss(logits, target)


def labeled_organ_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[1] == 0:
        return logits.sum() * 0.0
    if target.shape[1] != logits.shape[1] or valid.shape[1] != logits.shape[1]:
        raise ValueError("Organ supervision channel count does not match the model")
    target = F.adaptive_max_pool3d(target.to(logits.dtype), logits.shape[-3:])
    labeled = valid > 0.5
    if not bool(labeled.any()):
        return logits.sum() * 0.0
    selected_logits = logits[labeled].unsqueeze(1)
    selected_target = target[labeled].unsqueeze(1)
    return segmentation_loss(selected_logits, selected_target)


def gaussian_nll(error: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.exp(-log_variance) * error.square() + log_variance).mean()


@dataclass(frozen=True)
class AutoencoderLossWeights:
    l1: float = 1.0
    window: float = 0.75
    gradient: float = 0.25
    gradient_energy: float = 0.0
    high_frequency: float = 0.35
    ssim: float = 0.20
    kl: float = 1e-6
    body: float = 0.25
    lung: float = 0.25
    lesion: float = 0.0
    lesion_high_frequency: float = 0.0


class AutoencoderObjective(nn.Module):
    def __init__(
        self,
        weights: AutoencoderLossWeights = AutoencoderLossWeights(),
        hu_min: float = -1024.0,
        hu_max: float = 1500.0,
    ) -> None:
        super().__init__()
        from monai.losses import SSIMLoss

        self.weights = weights
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.ssim = SSIMLoss(spatial_dims=3, data_range=2.0, win_size=7, kernel_sigma=1.5)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        sigma: torch.Tensor,
        body_mask: torch.Tensor,
        lung_mask: torch.Tensor,
        lesion_mask: torch.Tensor | None = None,
        has_lesion_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l1 = F.l1_loss(reconstruction, target)
        window = multi_window_l1(reconstruction, target, self.hu_min, self.hu_max)
        gradient = gradient_l1_3d(reconstruction, target)
        gradient_energy = gradient_energy_calibration_loss_3d(reconstruction, target)
        high_frequency = high_frequency_l1_3d(reconstruction, target)
        ssim = self.ssim(reconstruction, target)
        variance = sigma.square().clamp_min(1e-8)
        kl = 0.5 * (mean.square() + variance - variance.log() - 1.0).mean()
        body = masked_l1(reconstruction, target, body_mask)
        lung = masked_l1(reconstruction, target, lung_mask)
        lesion = reconstruction.new_zeros(())
        lesion_high_frequency = reconstruction.new_zeros(())
        if lesion_mask is not None and has_lesion_label is not None:
            label_weight = has_lesion_label.reshape(-1, 1, 1, 1, 1).to(reconstruction.dtype)
            labeled_mask = lesion_mask * label_weight
            if bool(label_weight.sum() > 0) and bool(labeled_mask.any()):
                lesion = masked_l1(reconstruction, target, labeled_mask)
                local_mask = F.max_pool3d(labeled_mask, kernel_size=5, stride=1, padding=2)
                reconstruction_high = reconstruction - F.avg_pool3d(
                    reconstruction, kernel_size=3, stride=1, padding=1
                )
                target_high = target - F.avg_pool3d(target, kernel_size=3, stride=1, padding=1)
                lesion_high_frequency = masked_l1(
                    reconstruction_high,
                    target_high,
                    local_mask,
                )
        total = (
            self.weights.l1 * l1
            + self.weights.window * window
            + self.weights.gradient * gradient
            + self.weights.gradient_energy * gradient_energy
            + self.weights.high_frequency * high_frequency
            + self.weights.ssim * ssim
            + self.weights.kl * kl
            + self.weights.body * body
            + self.weights.lung * lung
            + self.weights.lesion * lesion
            + self.weights.lesion_high_frequency * lesion_high_frequency
        )
        return {
            "loss": total,
            "loss_l1": l1,
            "loss_window": window,
            "loss_gradient": gradient,
            "loss_gradient_energy": gradient_energy,
            "loss_high_frequency": high_frequency,
            "loss_ssim": ssim,
            "loss_kl": kl,
            "loss_body": body,
            "loss_lung": lung,
            "loss_lesion": lesion,
            "loss_lesion_high_frequency": lesion_high_frequency,
        }


@dataclass(frozen=True)
class CoarseLossWeights:
    latent: float = 1.0
    ct: float = 1.0
    window: float = 0.75
    gradient: float = 0.25
    high_frequency: float = 0.35
    projection: float = 0.5
    body_segmentation: float = 0.15
    lung_segmentation: float = 0.25
    lesion_segmentation: float = 1.0
    lesion_ct: float = 1.25
    organ_segmentation: float = 0.35
    uncertainty: float = 0.05


class CoarseObjective(nn.Module):
    def __init__(
        self,
        weights: CoarseLossWeights = CoarseLossWeights(),
        projection: ProjectionConfig = ProjectionConfig(),
    ) -> None:
        super().__init__()
        self.weights = weights
        self.projection = projection

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        decoded: torch.Tensor,
        target_latent: torch.Tensor,
        target_ct: torch.Tensor,
        target_ap: torch.Tensor,
        target_lat: torch.Tensor,
        body_mask: torch.Tensor,
        lung_mask: torch.Tensor,
        lesion_mask: torch.Tensor,
        has_lesion_label: torch.Tensor,
        organ_masks: torch.Tensor,
        organ_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = F.smooth_l1_loss(outputs["coarse_latent"], target_latent)
        ct = F.l1_loss(decoded, target_ct)
        window = multi_window_l1(decoded, target_ct, self.projection.hu_min, self.projection.hu_max)
        gradient = gradient_l1_3d(decoded, target_ct)
        high_frequency = high_frequency_l1_3d(decoded, target_ct)
        projection = projection_consistency_loss(decoded, target_ap, target_lat, self.projection)
        anatomy_size = outputs["body_logits"].shape[-3:]
        body_small = F.interpolate(body_mask, size=anatomy_size, mode="nearest")
        lung_small = F.interpolate(lung_mask, size=anatomy_size, mode="nearest")
        lesion_size = outputs["lesion_logits"].shape[-3:]
        lesion_small = F.adaptive_max_pool3d(lesion_mask, output_size=lesion_size)
        body_seg = segmentation_loss(outputs["body_logits"], body_small)
        lung_seg = segmentation_loss(outputs["lung_logits"], lung_small)
        label_weight = has_lesion_label.reshape(-1, 1, 1, 1, 1).to(decoded.dtype)
        lesion_seg = labeled_lesion_segmentation_loss(
            outputs["lesion_logits"], lesion_small, has_lesion_label
        )
        organ_seg = labeled_organ_segmentation_loss(
            outputs["organ_logits"], organ_masks, organ_valid
        )
        if bool(label_weight.sum() > 0) and bool(lesion_mask.any()):
            lesion_region = F.max_pool3d(lesion_mask, kernel_size=7, stride=1, padding=3)
            lesion_ct = masked_l1(decoded, target_ct, lesion_region * label_weight)
        else:
            lesion_ct = decoded.new_zeros(())
        latent_error = (outputs["coarse_latent"] - target_latent).abs().mean(dim=1, keepdim=True)
        uncertainty = gaussian_nll(latent_error, outputs["logvar"])
        total = (
            self.weights.latent * latent
            + self.weights.ct * ct
            + self.weights.window * window
            + self.weights.gradient * gradient
            + self.weights.high_frequency * high_frequency
            + self.weights.projection * projection
            + self.weights.body_segmentation * body_seg
            + self.weights.lung_segmentation * lung_seg
            + self.weights.lesion_segmentation * lesion_seg
            + self.weights.lesion_ct * lesion_ct
            + self.weights.organ_segmentation * organ_seg
            + self.weights.uncertainty * uncertainty
        )
        return {
            "loss": total,
            "loss_latent": latent,
            "loss_ct": ct,
            "loss_window": window,
            "loss_gradient": gradient,
            "loss_high_frequency": high_frequency,
            "loss_projection": projection,
            "loss_body_seg": body_seg,
            "loss_lung_seg": lung_seg,
            "loss_lesion_seg": lesion_seg,
            "loss_lesion_ct": lesion_ct,
            "loss_organ_seg": organ_seg,
            "loss_uncertainty": uncertainty,
        }
