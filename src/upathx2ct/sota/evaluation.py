"""Protocol metrics for signed-normalized UPath-LIDC reconstructions."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from upathx2ct.sota.metrics import psnr_3d, psnr_from_mse

HU_MIN = -1024.0
HU_MAX = 1500.0
HU_RANGE = HU_MAX - HU_MIN
HU_PER_SIGNED_UNIT = HU_RANGE / 2.0
AXIS_NAMES = ("axial", "coronal", "sagittal")


def signed_to_unit(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Volume contains non-finite values")
    return np.clip((values + 1.0) * 0.5, 0.0, 1.0)


def signed_to_hu(volume: np.ndarray) -> np.ndarray:
    return signed_to_unit(volume) * HU_RANGE + HU_MIN


def _validate_pair(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            f"Expected same-shape 3D volumes, got {prediction.shape} and "
            f"{target.shape}"
        )
    return signed_to_unit(prediction), signed_to_unit(target)


def _masked_mae_hu(
    prediction_hu: np.ndarray,
    target_hu: np.ndarray,
    mask: np.ndarray,
) -> float:
    selected = np.asarray(mask) > 0.5
    if not selected.any():
        return math.nan
    return float(np.mean(np.abs(prediction_hu[selected] - target_hu[selected])))


def _gradient_energy(volume: np.ndarray) -> float:
    values = [
        float(np.mean(np.abs(np.diff(volume, axis=axis))))
        for axis in range(3)
    ]
    return float(np.mean(values))


def _slice_mean_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    axis: int,
) -> tuple[float, float]:
    prediction_slices = np.moveaxis(prediction, axis, 0)
    target_slices = np.moveaxis(target, axis, 0)
    psnr_values = []
    ssim_values = []
    for predicted_slice, target_slice in zip(
        prediction_slices,
        target_slices,
    ):
        mse = float(np.mean(np.square(predicted_slice - target_slice)))
        psnr_values.append(psnr_from_mse(mse))
        ssim_values.append(
            float(
                structural_similarity(
                    target_slice,
                    predicted_slice,
                    data_range=1.0,
                )
            )
        )
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    body_mask: np.ndarray,
    lung_mask: np.ndarray,
    lesion_mask: np.ndarray | None = None,
) -> dict[str, float]:
    prediction_unit, target_unit = _validate_pair(prediction, target)
    prediction_hu = prediction_unit * HU_RANGE + HU_MIN
    target_hu = target_unit * HU_RANGE + HU_MIN
    error_hu = prediction_hu - target_hu
    metrics = {
        "mae_hu": float(np.mean(np.abs(error_hu))),
        "rmse_hu": float(np.sqrt(np.mean(np.square(error_hu)))),
        "bias_hu": float(np.mean(error_hu)),
        "psnr_3d": psnr_3d(prediction_unit, target_unit),
        "mae_hu_body": _masked_mae_hu(
            prediction_hu,
            target_hu,
            body_mask,
        ),
        "mae_hu_lung": _masked_mae_hu(
            prediction_hu,
            target_hu,
            lung_mask,
        ),
    }
    slice_ssim = []
    slice_psnr = []
    for axis, name in enumerate(AXIS_NAMES):
        psnr_value, ssim_value = _slice_mean_metrics(
            prediction_unit,
            target_unit,
            axis,
        )
        metrics[f"psnr_2d_{name}"] = psnr_value
        metrics[f"ssim_2d_{name}"] = ssim_value
        slice_psnr.append(psnr_value)
        slice_ssim.append(ssim_value)
    metrics["psnr_2d_orthogonal_mean"] = float(np.mean(slice_psnr))
    metrics["ssim_2d_orthogonal_mean"] = float(np.mean(slice_ssim))
    target_gradient = _gradient_energy(target_hu)
    metrics["gradient_energy_ratio"] = float(
        _gradient_energy(prediction_hu) / max(target_gradient, 1e-8)
    )

    if lesion_mask is not None and bool((np.asarray(lesion_mask) > 0.5).any()):
        lesion = np.asarray(lesion_mask) > 0.5
        shell = ndimage.binary_dilation(lesion, iterations=2) & ~lesion
        metrics["mae_hu_lesion_proxy"] = _masked_mae_hu(
            prediction_hu,
            target_hu,
            lesion,
        )
        if shell.any():
            target_contrast = float(
                target_hu[lesion].mean() - target_hu[shell].mean()
            )
            prediction_contrast = float(
                prediction_hu[lesion].mean() - prediction_hu[shell].mean()
            )
            metrics["lesion_proxy_target_contrast_hu"] = target_contrast
            metrics["lesion_proxy_prediction_contrast_hu"] = prediction_contrast
            metrics["lesion_proxy_contrast_error_hu"] = abs(
                prediction_contrast - target_contrast
            )
            metrics["lesion_proxy_contrast_ratio"] = float(
                prediction_contrast
                / (
                    math.copysign(
                        max(abs(target_contrast), 1e-6),
                        target_contrast,
                    )
                )
            )
    return metrics


def uncertainty_metrics(
    prediction_mean: np.ndarray,
    prediction_variance: np.ndarray,
    target: np.ndarray,
    body_mask: np.ndarray,
    *,
    temperature: float = 1.0,
) -> dict[str, float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Uncertainty temperature must be finite and positive")
    mean = np.asarray(prediction_mean, dtype=np.float64)
    variance = np.asarray(prediction_variance, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(body_mask) > 0.5
    if mean.shape != target.shape or variance.shape != target.shape:
        raise ValueError("Uncertainty arrays must share a 3D shape")
    if not mask.any():
        mask = np.ones_like(target, dtype=bool)
    error = np.abs(mean[mask] - target[mask])
    squared_error = np.square(mean[mask] - target[mask])
    raw_variance = np.maximum(variance[mask], 1e-8)
    calibrated_variance = raw_variance * temperature
    standard_deviation = np.sqrt(calibrated_variance)
    pearson = (
        float(np.corrcoef(error, standard_deviation)[0, 1])
        if np.std(error) > 0 and np.std(standard_deviation) > 0
        else math.nan
    )
    spearman = (
        float(spearmanr(error, standard_deviation).statistic)
        if np.std(error) > 0 and np.std(standard_deviation) > 0
        else math.nan
    )
    order = np.argsort(standard_deviation, kind="stable")
    ordered_error_hu = error[order] * HU_PER_SIGNED_UNIT
    cumulative_risk = np.cumsum(ordered_error_hu) / np.arange(
        1,
        ordered_error_hu.size + 1,
    )
    oracle_error_hu = np.sort(error * HU_PER_SIGNED_UNIT)
    oracle_risk = np.cumsum(oracle_error_hu) / np.arange(
        1,
        oracle_error_hu.size + 1,
    )
    model_aurc = float(np.mean(cumulative_risk))
    oracle_aurc = float(np.mean(oracle_risk))
    random_aurc = float(np.mean(error) * HU_PER_SIGNED_UNIT)
    denominator = max(random_aurc - oracle_aurc, 1e-8)
    high_error = error >= np.quantile(error, 0.90)
    high_uncertainty = standard_deviation >= np.quantile(
        standard_deviation,
        0.90,
    )
    return {
        "uncertainty_temperature": float(temperature),
        "uncertainty_error_pearson": pearson,
        "uncertainty_error_spearman": spearman,
        "uncertainty_gaussian_nll": float(
            0.5
            * np.mean(
                squared_error / calibrated_variance
                + np.log(calibrated_variance)
            )
        ),
        "uncertainty_coverage_95": float(
            np.mean(error <= 1.96 * standard_deviation)
        ),
        "uncertainty_mean_std_hu": float(
            np.mean(standard_deviation) * HU_PER_SIGNED_UNIT
        ),
        "uncertainty_aurc_hu": model_aurc,
        "uncertainty_oracle_aurc_hu": oracle_aurc,
        "uncertainty_aurc_skill": float(
            (random_aurc - model_aurc) / denominator
        ),
        "uncertainty_top10_error_recall": float(
            np.logical_and(high_error, high_uncertainty).sum()
            / max(high_error.sum(), 1)
        ),
        "uncertainty_temperature_numerator": float(
            np.sum(squared_error / raw_variance)
        ),
        "uncertainty_temperature_denominator": int(error.size),
    }


def fit_uncertainty_temperature(
    rows: list[dict[str, float | str]],
) -> float:
    numerator = sum(
        float(row["uncertainty_temperature_numerator"])
        for row in rows
    )
    denominator = sum(
        int(row["uncertainty_temperature_denominator"])
        for row in rows
    )
    if denominator <= 0 or not math.isfinite(numerator):
        raise ValueError("No valid uncertainty calibration statistics")
    return float(np.clip(numerator / denominator, 1e-3, 1e3))
