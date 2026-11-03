"""Frozen B0 objectives using the project cone-beam projector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from upathx2ct.sota.cone_projection import DifferentiableConeProjector
from upathx2ct.b0.losses import (
    gaussian_nll,
    gradient_l1_3d,
    high_frequency_l1_3d,
    multi_window_l1,
    segmentation_loss,
)


@dataclass(frozen=True)
class B0LossWeights:
    latent: float = 1.0
    ct: float = 1.0
    window: float = 0.75
    gradient: float = 0.25
    high_frequency: float = 0.25
    ssim: float = 0.20
    projection: float = 0.35
    body_segmentation: float = 0.10
    lung_segmentation: float = 0.20
    uncertainty: float = 0.05


@dataclass(frozen=True)
class B0DiffusionLossWeights:
    noise: float = 1.0
    clean: float = 0.25
    decoded: float = 0.20
    ct: float = 1.0
    window: float = 0.50
    gradient: float = 0.20
    high_frequency: float = 0.25
    ssim: float = 0.15
    projection: float = 0.25
    lesion_ct: float = 0.0
    lesion_gradient: float = 0.0
    lesion_contrast: float = 0.0


class B0Objective(nn.Module):
    def __init__(
        self,
        projector: DifferentiableConeProjector,
        calibration: dict[str, Any],
        weights: B0LossWeights = B0LossWeights(),
    ) -> None:
        super().__init__()
        from monai.losses import SSIMLoss

        if calibration.get("status") != "pass":
            raise ValueError("Cone-projector calibration has not passed")
        self.projector = projector
        self.weights = weights
        self.ssim = SSIMLoss(
            spatial_dims=3,
            data_range=2.0,
            win_size=7,
            kernel_sigma=1.5,
        )
        values = []
        for view in ("frontal_ap", "left_lateral"):
            record = calibration["views"][view]
            values.append(
                (
                    float(record["gain"]),
                    float(record["bias"]),
                    float(record["target_log1p_mean"]),
                    float(record["target_log1p_std"]),
                )
            )
        tensor = torch.tensor(values, dtype=torch.float32)
        if (
            not torch.isfinite(tensor).all()
            or bool((tensor[:, 0] <= 0).any())
            or bool((tensor[:, 3] <= 0).any())
        ):
            raise ValueError("Invalid projector intensity calibration")
        self.register_buffer("projection_calibration", tensor)

    @staticmethod
    def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return image[..., 1:, :] - image[..., :-1, :], image[..., :, 1:] - image[..., :, :-1]

    def projection_loss(
        self,
        decoded: torch.Tensor,
        target_ap_raw: torch.Tensor,
        target_lat_raw: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self.projector(decoded)
        targets = (target_ap_raw, target_lat_raw)
        output_height, output_width = self.projector.config.detector_shape
        losses = []
        for view_index, (prediction, target) in enumerate(zip(predicted, targets)):
            scale_y = target.shape[-2] // output_height
            scale_x = target.shape[-1] // output_width
            if (
                scale_y * output_height != target.shape[-2]
                or scale_x * output_width != target.shape[-1]
            ):
                raise ValueError("Target DRR cannot be pooled to projector resolution")
            target = F.avg_pool2d(
                target,
                kernel_size=(scale_y, scale_x),
                stride=(scale_y, scale_x),
            )
            gain, bias, mean, std = self.projection_calibration[view_index]
            prediction = torch.clamp(gain * prediction + bias, min=0.0)
            prediction = (torch.log1p(prediction) - mean) / std
            target = (torch.log1p(torch.clamp(target, min=0.0)) - mean) / std
            loss = F.smooth_l1_loss(prediction, target)
            pred_dy, pred_dx = self._image_gradients(prediction)
            target_dy, target_dx = self._image_gradients(target)
            loss = loss + 0.25 * (
                F.l1_loss(pred_dy, target_dy)
                + F.l1_loss(pred_dx, target_dx)
            )
            losses.append(loss)
        return torch.stack(losses).mean()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        decoded: torch.Tensor,
        target_latent: torch.Tensor,
        target_ct: torch.Tensor,
        target_ap_raw: torch.Tensor,
        target_lat_raw: torch.Tensor,
        body_mask: torch.Tensor,
        lung_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = F.smooth_l1_loss(outputs["coarse_latent"], target_latent)
        ct = F.l1_loss(decoded, target_ct)
        window = multi_window_l1(decoded, target_ct, -1024.0, 1500.0)
        zero = decoded.new_zeros(())
        gradient = (
            gradient_l1_3d(decoded, target_ct)
            if self.weights.gradient > 0
            else zero
        )
        high_frequency = (
            high_frequency_l1_3d(decoded, target_ct)
            if self.weights.high_frequency > 0
            else zero
        )
        ssim = self.ssim(decoded, target_ct) if self.weights.ssim > 0 else zero
        projection = (
            self.projection_loss(
                decoded,
                target_ap_raw,
                target_lat_raw,
            )
            if self.weights.projection > 0
            else zero
        )
        if self.weights.body_segmentation > 0 or self.weights.lung_segmentation > 0:
            anatomy_size = outputs["body_logits"].shape[-3:]
            body_small = F.adaptive_max_pool3d(body_mask, anatomy_size)
            lung_small = F.adaptive_max_pool3d(lung_mask, anatomy_size)
            body_segmentation = (
                segmentation_loss(outputs["body_logits"], body_small)
                if self.weights.body_segmentation > 0
                else zero
            )
            lung_segmentation = (
                segmentation_loss(outputs["lung_logits"], lung_small)
                if self.weights.lung_segmentation > 0
                else zero
            )
        else:
            body_segmentation = lung_segmentation = zero
        if self.weights.uncertainty > 0:
            latent_error = (
                outputs["coarse_latent"] - target_latent
            ).abs().mean(dim=1, keepdim=True)
            uncertainty = gaussian_nll(latent_error, outputs["logvar"])
        else:
            uncertainty = zero
        total = (
            self.weights.latent * latent
            + self.weights.ct * ct
            + self.weights.window * window
            + self.weights.gradient * gradient
            + self.weights.high_frequency * high_frequency
            + self.weights.ssim * ssim
            + self.weights.projection * projection
            + self.weights.body_segmentation * body_segmentation
            + self.weights.lung_segmentation * lung_segmentation
            + self.weights.uncertainty * uncertainty
        )
        return {
            "loss": total,
            "loss_latent": latent,
            "loss_ct": ct,
            "loss_window": window,
            "loss_gradient": gradient,
            "loss_high_frequency": high_frequency,
            "loss_ssim": ssim,
            "loss_projection": projection,
            "loss_body_segmentation": body_segmentation,
            "loss_lung_segmentation": lung_segmentation,
            "loss_uncertainty": uncertainty,
        }


class B0DiffusionObjective(nn.Module):
    """Residual-diffusion fidelity with optional XML lesion supervision."""

    def __init__(
        self,
        projector: DifferentiableConeProjector,
        calibration: dict[str, Any],
        train_timesteps: int,
        weights: B0DiffusionLossWeights = B0DiffusionLossWeights(),
    ) -> None:
        super().__init__()
        if train_timesteps <= 0:
            raise ValueError("Diffusion timesteps must be positive")
        self.train_timesteps = int(train_timesteps)
        self.weights = weights
        self.fidelity = B0Objective(projector, calibration)
        self.use_lesion_objective = any(
            value > 0
            for value in (
                weights.lesion_ct,
                weights.lesion_gradient,
                weights.lesion_contrast,
            )
        )

    def forward(
        self,
        result: dict[str, torch.Tensor],
        autoencoder: nn.Module,
        target_ct: torch.Tensor,
        target_ap_raw: torch.Tensor,
        target_lat_raw: torch.Tensor,
        lesion_probability: torch.Tensor | None = None,
        lesion_consensus_mask: torch.Tensor | None = None,
        has_xml_contour_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        total = (
            self.weights.noise * result["loss_noise"]
            + self.weights.clean * result["loss_clean"]
        )
        zero = total.new_zeros(())
        ct = window = gradient = high_frequency = ssim = projection = zero
        lesion_ct = lesion_gradient = lesion_contrast = zero
        xml_positive_cases = zero
        low_noise = result["timesteps"] < self.train_timesteps // 2
        if self.weights.decoded > 0 and bool(low_noise.any()):
            prediction = autoencoder.decode(
                result["predicted_latent"][low_noise],
                normalized=True,
                clamp=False,
            ).float()
            target = target_ct[low_noise].float()
            ct = F.l1_loss(prediction, target)
            window = multi_window_l1(prediction, target, -1024.0, 1500.0)
            gradient = gradient_l1_3d(prediction, target)
            high_frequency = high_frequency_l1_3d(prediction, target)
            ssim = self.fidelity.ssim(prediction, target)
            projection = self.fidelity.projection_loss(
                prediction,
                target_ap_raw[low_noise].float(),
                target_lat_raw[low_noise].float(),
            )
            decoded = (
                self.weights.ct * ct
                + self.weights.window * window
                + self.weights.gradient * gradient
                + self.weights.high_frequency * high_frequency
                + self.weights.ssim * ssim
                + self.weights.projection * projection
            )
            total = total + self.weights.decoded * decoded
            if self.use_lesion_objective:
                if (
                    lesion_probability is None
                    or lesion_consensus_mask is None
                    or has_xml_contour_target is None
                ):
                    raise ValueError(
                        "Pathology diffusion requires XML lesion tensors"
                    )
                lesion = lesion_reconstruction_losses(
                    prediction,
                    target,
                    lesion_probability[low_noise],
                    lesion_consensus_mask[low_noise],
                    has_xml_contour_target[low_noise],
                )
                lesion_ct = lesion["loss_lesion_ct"]
                lesion_gradient = lesion["loss_lesion_gradient"]
                lesion_contrast = lesion["loss_lesion_contrast"]
                xml_positive_cases = lesion["xml_positive_cases"]
                total = (
                    total
                    + self.weights.lesion_ct * lesion_ct
                    + self.weights.lesion_gradient * lesion_gradient
                    + self.weights.lesion_contrast * lesion_contrast
                )
        return {
            "loss": total,
            "loss_noise": result["loss_noise"],
            "loss_clean": result["loss_clean"],
            "loss_ct": ct,
            "loss_window": window,
            "loss_gradient": gradient,
            "loss_high_frequency": high_frequency,
            "loss_ssim": ssim,
            "loss_projection": projection,
            "loss_lesion_ct": lesion_ct,
            "loss_lesion_gradient": lesion_gradient,
            "loss_lesion_contrast": lesion_contrast,
            "xml_positive_cases": xml_positive_cases,
        }


def _weighted_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    denominator = weight.sum().clamp_min(1.0)
    return ((prediction - target).abs() * weight).sum() / denominator


def _weighted_gradient_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for axis in (2, 3, 4):
        prediction_gradient = prediction.diff(dim=axis)
        target_gradient = target.diff(dim=axis)
        lower = [slice(None)] * weight.ndim
        upper = [slice(None)] * weight.ndim
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        gradient_weight = torch.maximum(
            weight[tuple(lower)],
            weight[tuple(upper)],
        )
        losses.append(
            _weighted_l1(
                prediction_gradient,
                target_gradient,
                gradient_weight,
            )
        )
    return torch.stack(losses).mean()


def lesion_reconstruction_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    probability: torch.Tensor,
    consensus: torch.Tensor,
    has_contour_target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return contour-local CT losses without declaring unlabeled cases negative."""

    if prediction.shape != target.shape:
        raise ValueError("Lesion loss prediction and target shapes differ")
    if probability.shape != target.shape or consensus.shape != target.shape:
        raise ValueError("Lesion supervision does not match the CT grid")
    positive = has_contour_target.flatten() > 0.5
    zero = prediction.sum() * 0.0
    if not bool(positive.any()):
        return {
            "loss_lesion_ct": zero,
            "loss_lesion_gradient": zero,
            "loss_lesion_contrast": zero,
            "xml_positive_cases": zero,
        }

    prediction = prediction[positive]
    target = target[positive]
    probability = probability[positive].float().clamp(0.0, 1.0)
    consensus = (consensus[positive] > 0.5).float()
    dilated_probability = F.max_pool3d(
        probability,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    dilated_consensus = F.max_pool3d(
        consensus,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    local_weight = torch.maximum(
        dilated_probability,
        0.5 * dilated_consensus,
    )
    lesion_ct = _weighted_l1(prediction, target, local_weight)
    lesion_gradient = _weighted_gradient_l1(
        prediction,
        target,
        local_weight,
    )

    outer = F.max_pool3d(
        consensus,
        kernel_size=5,
        stride=1,
        padding=2,
    )
    shell = (outer - consensus).clamp(0.0, 1.0)
    lesion_count = consensus.flatten(1).sum(dim=1).clamp_min(1.0)
    shell_count = shell.flatten(1).sum(dim=1).clamp_min(1.0)
    prediction_lesion = (
        (prediction * consensus).flatten(1).sum(dim=1) / lesion_count
    )
    target_lesion = (
        (target * consensus).flatten(1).sum(dim=1) / lesion_count
    )
    prediction_shell = (
        (prediction * shell).flatten(1).sum(dim=1) / shell_count
    )
    target_shell = (
        (target * shell).flatten(1).sum(dim=1) / shell_count
    )
    lesion_contrast = F.smooth_l1_loss(
        prediction_lesion - prediction_shell,
        target_lesion - target_shell,
    )

    return {
        "loss_lesion_ct": lesion_ct,
        "loss_lesion_gradient": lesion_gradient,
        "loss_lesion_contrast": lesion_contrast,
        "xml_positive_cases": positive.sum().to(dtype=prediction.dtype),
    }
